# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**V-22 TAR Intelligence System** — an AI-powered maintenance decision-support platform for V-22 Osprey aircraft. Two main components:

1. **Batch Analysis Pipeline** (complete) — mines historical TAR/MAF data to discover problem categories, solution patterns, and part failure trends using local LLMs via Ollama.
2. **Real-Time Lookup Tool** (to build) — a FastAPI web app where maintainers paste a new TAR and instantly get: the likely problem category, what has fixed it before, which parts to have ready, and an AI-generated recommendation. Described in detail in `PROMPT_BUILD_TAR_LOOKUP.md`.

## Data Sources

**Not in git — must be placed in project root:**
- `TAR_Data.csv` (~15K rows) — Technical Assistance Requests (problems reported)
- `maf.csv` (~1.8M rows) — Maintenance Action Forms (corrective actions taken)
- Linked via `jcn` (Job Control Number) field; ~78% of TARs have matching MAFs

**Key data fields:**
- **TAR**: `jcn`, `subject`, `issue`, `part_number`, `status`, `work_center`, `uns`
- **MAF**: `jcn`, `discrepancy`, `corr_act`, `action_taken`, `inst_partno`, `rmvd_partno`, `manhours`, `wuc`

## Local AI Stack (Ollama)

All models run locally, no cloud APIs. Ollama must be running.

```bash
ollama pull nomic-embed-text:latest    # embeddings (768-dim)
ollama pull qwen2.5:32b               # extraction/classification (fast, reliable)
ollama pull deepseek-r1:32b           # reasoning/summarization (slower, uses <think> blocks)
```

## Repository Structure

```
Globodon2/
├── CLAUDE.md                         # This file
├── PROMPT_BUILD_TAR_LOOKUP.md        # Full spec for the real-time lookup tool
│
│  ── Batch Analysis (complete) ──
├── tar_maf_analyzer.py               # Task 1: Problem→Solution pipeline (embed→cluster→classify→extract→summarize)
├── part_failure_analysis.py          # Task 2: Top failing parts with AI summaries
├── analysis_results.json             # Raw output from tar_maf_analyzer.py
├── analysis_results_cleaned.json     # Cleaned: 10 problem clusters with solutions, breakdowns, insights
├── part_failure_analysis.json        # 20 parts with failure counts and AI summaries
├── clean_results.py                  # Utility: clean analysis results
├── cleanup_results.py               # Utility: additional cleanup
├── demo_dashboard.html              # Static dashboard (can be replaced by the new app)
├── .cache/                           # Cached embeddings (.npy files keyed by content hash)
│
│  ── Real-Time Lookup Tool (to build) ──
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app with routes
│   ├── indexer.py                    # Startup: load embeddings, TAR data, MAF index
│   ├── search.py                     # Core RAG: embed query → cosine search → retrieve context
│   ├── llm.py                        # Ollama helpers (reuse patterns from tar_maf_analyzer.py)
│   ├── models.py                     # Pydantic request/response models
│   └── static/
│       └── index.html                # Single-page frontend (dark theme, defense-appropriate)
├── scripts/
│   └── build_index.py                # Optional: precompute search index
├── requirements.txt
└── run.sh                            # Start script for the web app
```

## Running

### Batch Analysis (already complete — results exist)
```bash
python3 tar_maf_analyzer.py           # Rerun full pipeline if needed (~13 min)
python3 part_failure_analysis.py      # Rerun part analysis if needed
```
Set `SAMPLE_SIZE` in `tar_maf_analyzer.py` config (line ~39) to an integer for testing, or `None` for full run.

### Real-Time Lookup Tool
```bash
pip install -r requirements.txt
./run.sh                              # Starts FastAPI on port 8000
# or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Batch Pipeline Architecture (tar_maf_analyzer.py)

5 sequential steps:
1. **Embed** — `load_tar_data()` + `embed_texts()`: Combines TAR `subject`+`issue` fields, embeds via `nomic-embed-text`. Cached in `.cache/` as `.npy` keyed by SHA-256 content hash.
2. **Cluster** — `cluster_embeddings()`: KMeans with automatic k selection via silhouette score (tests k in range 5–50).
3. **Classify** — `classify_cluster()`: Sends 8 sample TARs per cluster to `qwen2.5:32b` for a human-readable label.
4. **Extract** — `extract_solution()`: For each cluster, finds linked MAFs via JCN match, sends up to 15 corrective action texts to `qwen2.5:32b` for structured JSON extraction (action/component/reference).
5. **Summarize** — `summarize_cluster_insight()`: Sends each cluster's solutions to `deepseek-r1:32b` for an actionable insight paragraph.

Output: `analysis_results_cleaned.json` — 10 problem clusters with full profiles.

## Real-Time Lookup Architecture (app/)

RAG pipeline for instant TAR diagnosis:
1. **Embed** incoming TAR text via `nomic-embed-text` (~100ms)
2. **Cosine similarity search** against all ~15K preloaded TAR embeddings
3. **Map to cluster** — majority vote of top-k neighbors → cluster profile from `analysis_results_cleaned.json`
4. **Retrieve MAF context** — linked corrective actions from pre-indexed JCN→MAF dict
5. **Cross-reference parts** — check against `part_failure_analysis.json` for known high-failure parts
6. **(Optional) AI recommendation** — send context to `qwen2.5:32b` for tailored advice

**Two speed paths:**
- **Fast path** (steps 1–5): <1 second, no LLM call needed (embeddings are cached/preloaded)
- **AI path** (step 6): ~10–15 seconds via Ollama

**API Endpoints:**
- `POST /api/search` — core RAG search, returns cluster match + similar TARs + MAF actions
- `POST /api/recommend` — AI recommendation based on search results
- `GET /api/clusters` — all 10 cluster profiles
- `GET /api/parts` — top 20 failing parts
- `GET /api/stats` — database statistics
- `GET /` — serve frontend

## Key Implementation Details

- MAF file is read in 100K-row chunks (`pd.read_csv(..., chunksize=100_000)`) due to 1.8M row size
- `call_ollama_json()` strips deepseek-r1's `<think>...</think>` blocks before parsing JSON
- `call_ollama()` has retry logic with exponential backoff and a configurable delay (`LLM_DELAY`)
- Embeddings are cached by SHA-256 hash of input texts in `.cache/`; delete `.cache/` to force recompute
- For cosine similarity search: normalize all embeddings at load time, use `np.dot(query, matrix.T)` — no FAISS needed at 15K scale
- The MAF JCN index (dict mapping jcn→MAF records) is built once on startup to avoid re-reading the CSV per query

## Existing Analysis Results (precomputed)

**analysis_results_cleaned.json** — 10 clusters:
- FADEC System Malfunction Troubleshooting (2,213 TARs)
- Composite and Metal Structural Damage Repair (2,154 TARs)
- Component Failures and Intermittent Faults (2,131 TARs)
- Test Equipment and Procedure Requests Failure (1,821 TARs)
- Aircraft Component Failure and Maintenance Requests (1,594 TARs)
- Maintenance and Repair Procedures Requests (1,411 TARs)
- Sealing and Structural Integrity Issues (1,339 TARs)
- Water Intrusion and Corrosion Prevention (927 TARs)
- Proprotor Gearbox Debris Detection Issues (726 TARs)
- PCA Ball Screw Wear Test Failure (609 TARs)

Each cluster includes: solution breakdown (action type percentages), parts commonly involved, average manhours, sample JCNs, extracted solutions, and an AI-generated insight paragraph.

**part_failure_analysis.json** — Top 5 of 20:
- 901-011-420-101: 6,831 failures
- 901-336-007-115: 5,693 failures
- MN1400: 3,837 failures
- 901-363-203-105: 3,649 failures
- 901-336-006-115: 2,691 failures

## Dependencies

Python 3.13+: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `ollama`, `pydantic`, `tqdm`, `rich`