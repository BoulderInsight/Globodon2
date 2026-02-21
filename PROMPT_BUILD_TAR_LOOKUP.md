# Build: Real-Time TAR Decision Support Tool

## Context

We're building a **live decision-support tool** for V-22 Osprey aircraft maintainers. When a new Technical Assistance Request (TAR) comes in, the system instantly searches our historical maintenance database (~15K TARs, ~1.8M MAF corrective action records) and returns: what this problem likely is, what has fixed it before, what parts are typically involved, and how long it usually takes. This is a **RAG system for aircraft maintenance**.

The existing codebase (`tar_maf_analyzer.py`) already built the analytical foundation — it embedded all TARs with `nomic-embed-text`, clustered them into 10 problem categories, extracted structured solutions from MAFs via LLM, and generated AI insight summaries. The results live in `analysis_results_cleaned.json` and `part_failure_analysis.json`. Now we're turning that batch analysis into a real-time interactive tool.

## What Already Exists (DO NOT recreate — build on top of these)

```
Globodon2/
├── tar_maf_analyzer.py          # Batch pipeline (embed → cluster → classify → extract → summarize)
├── part_failure_analysis.py      # Top 20 failing parts with AI summaries
├── analysis_results_cleaned.json # 10 problem clusters with solutions, breakdowns, insights
├── part_failure_analysis.json    # 20 parts with failure counts and AI summaries
├── demo_dashboard.html           # Existing dashboard (can be replaced)
├── CLAUDE.md                     # Project docs
├── .cache/                       # Cached embeddings (.npy files)
│
├── TAR_Data.csv                  # ~15K rows (NOT in git, lives locally on disk)
└── maf.csv                       # ~1.8M rows (NOT in git, lives locally on disk)
```

**Key data fields:**
- TAR: `jcn`, `subject`, `issue`, `part_number`, `status`, `work_center`, `uns`
- MAF: `jcn`, `discrepancy`, `corr_act`, `action_taken`, `inst_partno`, `rmvd_partno`, `manhours`, `wuc`
- TARs and MAFs link via `jcn` (Job Control Number). ~78% of TARs have matching MAFs.

**Models available via Ollama (already pulled locally):**
- `nomic-embed-text:latest` — embeddings (768-dim)
- `qwen2.5:32b` — extraction/classification (fast, reliable)
- `deepseek-r1:32b` — reasoning/summarization (slower, uses `<think>` blocks)

**Existing code you should reuse/adapt:**
- `embed_texts()` in `tar_maf_analyzer.py` — batch embedding with cache
- `call_ollama()` / `call_ollama_json()` — Ollama helpers with retry logic
- `compute_solution_breakdown()` — action type percentages
- The `<think>` block stripping logic for deepseek-r1

## What to Build

### Architecture: FastAPI + Static HTML Frontend

```
Globodon2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with routes
│   ├── indexer.py            # Startup: load embeddings, cluster data, build search index
│   ├── search.py             # Core RAG: embed query → cosine search → retrieve context
│   ├── llm.py                # Ollama helpers (adapted from existing code)
│   ├── models.py             # Pydantic models for request/response
│   └── static/
│       └── index.html        # Single-page frontend
├── scripts/
│   └── build_index.py        # One-time script: precompute and save the search index
├── requirements.txt
└── run.sh                    # Start script
```

### Backend: FastAPI

**Startup / Index Building (`indexer.py`):**
On app startup (or via `build_index.py`), load and prepare:
1. Read `TAR_Data.csv` into memory. Combine `subject` + `issue` into `text` field.
2. Load cached embeddings from `.cache/` (the .npy files already exist from `tar_maf_analyzer.py`). If cache miss, embed via Ollama.
3. Load `analysis_results_cleaned.json` — the 10 cluster profiles with solutions, breakdowns, insights.
4. Load `part_failure_analysis.json` — the 20 failing parts data.
5. Build a numpy matrix of all TAR embeddings for fast cosine similarity search. Normalize them for dot-product similarity.
6. Store cluster assignments per TAR (from the existing analysis).
7. **Pre-index MAF data**: Read `maf.csv` chunked, build a dict mapping `jcn → list[MAF records]` for only the JCNs that exist in TAR data. This avoids re-reading the 1.8M row file on every query. Store in memory (should be manageable — ~78% of 15K JCNs × ~3 MAFs each ≈ 35K records).

**Core Search Endpoint (`POST /api/search`):**

Input: `{ "text": "FADEC channel A fault during ground turn...", "top_k": 10 }`

Pipeline:
1. **Embed** the input text via `nomic-embed-text` (~100ms)
2. **Cosine similarity** against all TAR embeddings → get top_k nearest neighbors
3. For each neighbor, look up:
   - The TAR record (jcn, subject, issue, part_number, work_center)
   - Its cluster assignment → map to the cluster profile from `analysis_results_cleaned.json`
   - Linked MAF records from the pre-built JCN index (discrepancy, corrective action, action taken, parts, manhours)
4. **Determine best-match cluster** by majority vote of the top_k neighbors' cluster assignments
5. Return structured response (see models below)

Response shape:
```json
{
  "matched_cluster": {
    "problem": "FADEC System Malfunction Troubleshooting",
    "confidence": 0.87,
    "occurrences": 2213,
    "solution_breakdown": {"replace": 45.2, "inspect": 22.1, ...},
    "typical_solution": "The AI insight paragraph...",
    "average_manhours": 4.2,
    "parts_commonly_involved": ["901-011-420-101", ...]
  },
  "similar_tars": [
    {
      "jcn": "...",
      "subject": "...",
      "issue": "...",
      "similarity": 0.94,
      "cluster_label": "...",
      "maf_actions": [
        {"corr_act": "...", "action_taken": "R", "manhours": "2.5", "parts": {...}}
      ]
    }
  ],
  "related_parts": [
    {"part_number": "...", "failure_count": 6831, "ai_summary": "..."}
  ],
  "ai_recommendation": null  // filled by /api/recommend endpoint
}
```

**AI Recommendation Endpoint (`POST /api/recommend`):**

This is the optional "slow path" — takes the search results + original query and sends them to an LLM for a tailored recommendation.

Input: `{ "text": "original TAR text", "search_results": { ... from /api/search } }`

Use `qwen2.5:32b` (NOT deepseek — speed matters here) with a prompt like:

```
You are an expert V-22 Osprey maintenance advisor. A new TAR has come in:
"{tar_text}"

Based on {N} similar past cases classified as "{cluster_label}", here's what we know:
- Typical resolution: {solution_breakdown}
- Average manhours: {avg_mh}
- Past corrective actions that worked:
{top_5_corrective_actions}

Given this specific TAR and the historical data:
1. What is the most likely root cause?
2. What should the maintainer try first?
3. What parts should they have on hand?
4. Estimated time to resolution?

Be specific and actionable. Reference the historical data.
```

**Additional Endpoints:**
- `GET /api/clusters` — return all 10 cluster profiles (for dashboard overview)
- `GET /api/parts` — return the 20 failing parts data
- `GET /api/stats` — return high-level stats (total TARs indexed, total MAFs, cluster count, etc.)
- `GET /` — serve `index.html`

### Frontend: Single-Page App (`app/static/index.html`)

Single HTML file with embedded CSS and JS (no build tools). Clean, professional, dark-themed UI suitable for a demo to military/defense stakeholders.

**Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│  V-22 TAR Intelligence System           [Database Stats]     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ INPUT ─────────────────────────────────────────────┐    │
│  │  Enter TAR description...                            │    │
│  │  [multi-line textarea]                               │    │
│  │                                                      │    │
│  │  [🔍 Search Historical Data]  [🤖 Get AI Recommendation] │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─ DIAGNOSIS ──────────────────────────────────────────┐   │
│  │  Problem Category: FADEC System Malfunction           │   │
│  │  Confidence: ████████░░ 87%                          │   │
│  │  Historical Occurrences: 2,213                       │   │
│  │  Avg Resolution Time: 4.2 manhours                   │   │
│  │                                                      │   │
│  │  Solution Breakdown:                                  │   │
│  │  ├── Replace: ████████████ 45%                       │   │
│  │  ├── Inspect: ██████ 22%                             │   │
│  │  └── Adjust:  ████ 15%                               │   │
│  │                                                      │   │
│  │  Parts to Have Ready:                                │   │
│  │  • 901-011-420-101 (6,831 past failures)             │   │
│  │  • 901-336-007-115 (5,693 past failures)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ SIMILAR PAST CASES ────────────────────────────────┐   │
│  │  ┌─ 94% match ──────────────────────────────────┐   │   │
│  │  │ JCN: ABC123  |  FADEC Ch-A fault on ground   │   │   │
│  │  │ Fix: Replaced FADEC unit IAW TO 1-1-1        │   │   │
│  │  │ Manhours: 3.2  |  Parts: 901-011-420-101     │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  │  ┌─ 91% match ──────────────────────────────────┐   │   │
│  │  │ ...                                           │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ AI RECOMMENDATION (expandable) ────────────────────┐   │
│  │  Loading... / "Based on 2,213 similar cases..."      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**UI Requirements:**
- Dark theme (slate/navy background, white text, accent color for highlights)
- Must look professional for defense/military audience — no playful design
- Responsive but optimized for desktop/laptop demo
- Search button triggers `/api/search` → populates Diagnosis + Similar Cases panels instantly
- "Get AI Recommendation" button triggers `/api/recommend` → streams or shows result in the bottom panel with a loading spinner
- Similarity scores shown as percentages with visual bars
- Solution breakdown shown as horizontal bar chart (CSS, no library needed)
- Collapsible cards for similar past cases (show top 3, expand for more)
- Show a "Powered by Local AI" badge or similar (Ollama, nomic-embed, qwen, deepseek) — this matters for the demo

**Key UX details:**
- Search should feel fast — the cosine similarity path should return in <1 second
- AI recommendation is clearly labeled as "generating..." with a spinner — user understands this takes a few seconds
- Include sample TAR texts as placeholder/examples the user can click to auto-fill
- Display database stats in the header: "15,247 TARs indexed | 1.8M maintenance records | 10 problem categories"

### Performance Targets

- Index load on startup: < 30 seconds (embeddings + MAF index)
- Search response: < 1 second (embed + cosine search + lookup)
- AI recommendation: < 15 seconds (single LLM call)
- Memory footprint: ~2-3 GB (embeddings + MAF index)

### Requirements.txt

```
fastapi>=0.115.0
uvicorn>=0.30.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
ollama>=0.4.0
pydantic>=2.0.0
```

### run.sh

```bash
#!/bin/bash
cd "$(dirname "$0")"
echo "Starting TAR Intelligence System..."
echo "Make sure Ollama is running with: nomic-embed-text, qwen2.5:32b"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Implementation Notes

- The `.cache/` directory already has precomputed embeddings from `tar_maf_analyzer.py`. Reuse them. The hash is based on the concatenated text content. Load them in `indexer.py`.
- For cosine similarity, normalize all embeddings at load time, then use `np.dot(query_embedding, all_embeddings.T)` for fast search. No need for FAISS at 15K vectors.
- The MAF JCN index should be built once on startup. Read `maf.csv` in chunks, keep only rows where jcn is in the TAR jcn set, store as `dict[str, list[dict]]`.
- Part number cross-referencing: when search results include part numbers (from TAR `part_number` field or MAF `rmvd_partno`/`inst_partno`), check against `part_failure_analysis.json` to surface known high-failure parts.
- Error handling: if Ollama is not running, the search endpoint should still work (it only needs the embeddings which are cached). Only the embed step and AI recommendation need Ollama live. Handle gracefully with clear error messages.
- Use `orjson` or standard `json` for fast serialization of responses.

## Agent Team Suggestions

If using Claude Code's agent spawning:
- **Agent 1: Backend** — Build `app/main.py`, `indexer.py`, `search.py`, `llm.py`, `models.py`
- **Agent 2: Frontend** — Build `app/static/index.html` with all the UI/UX described above
- **Agent 3: Integration & Testing** — Write `scripts/build_index.py`, `run.sh`, `requirements.txt`, and test the full pipeline

## Definition of Done

1. `run.sh` starts the app, loads the index in <30s
2. Pasting a TAR description and clicking Search returns results in <1s
3. Results show: matched cluster, confidence, solution breakdown, similar past cases with MAF actions
4. "Get AI Recommendation" produces a tailored response in <15s
5. The UI looks polished enough to demo to defense stakeholders
6. Works entirely on local infrastructure (Ollama, no cloud APIs)
