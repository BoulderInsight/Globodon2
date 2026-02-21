# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**V-22 TAR Intelligence System** — an AI-powered maintenance decision-support platform for V-22 Osprey aircraft. Three main components:

1. **Batch Analysis Pipeline** (complete) — mines historical TAR/MAF data to discover problem categories, solution patterns, and part failure trends using local LLMs via Ollama.
2. **Real-Time Lookup Tool** (complete) — FastAPI web app where maintainers paste a new TAR and instantly get: the likely problem category, what has fixed it before, which parts to have ready, and an AI-generated recommendation.
3. **TPDR Recommendation Engine** (complete) — automatic detection of systems that need Technical Publications Deficiency Reports, combining trend analysis, comeback detection, and AI-generated justification.

## Data Sources

**Not in git — must be placed in project root:**
- `TAR_Data.csv` (~14,926 rows) — Technical Assistance Requests (problems reported)
- `maf.csv` (~1.8M rows) — Maintenance Action Forms (corrective actions taken)
- Linked via `jcn` (Job Control Number) field; ~78% of TARs have matching MAFs

**Key data fields:**
- **TAR**: `jcn`, `subject`, `issue`, `part_number`, `status`, `work_center`, `uns`, `submit_date`, `close_date`, `buno`, `aircraft_type`, `activity`, `priority`, `serial_number`
- **MAF**: `jcn`, `discrepancy`, `corr_act`, `action_taken`, `inst_partno`, `rmvd_partno`, `manhours`, `wuc`, `serno`, `comp_date`

**Key data relationships:**
- TAR `jcn` → MAF `jcn` (primary link, ~78% match rate)
- TAR `buno` → MAF `serno` (aircraft identifier, 357 aircraft overlap)
- TAR `uns` = system identifier, e.g. "6322 PROPROTOR GEARBOX ASSEMBLY RH" (305 unique codes)
- `submit_date` format is mixed: "9/7/2022 10:23:56 AM" or "10/9/2024" — use `pd.to_datetime(..., errors='coerce')`
- `buno` value of "I-level" means intermediate-level maintenance, not tied to a specific aircraft

**Dataset stats:**
- Date range: Sep 2020 to Sep 2024 (43 months, ~254 TARs/month)
- 431 unique aircraft (buno), 305 unique systems (uns)
- Aircraft types: MV (11,526), CV (1,888), CMV (1,508)

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
├── PROMPT_BUILD_TAR_LOOKUP.md        # Spec for the real-time lookup tool (complete)
├── PROMPT_BUILD_TPDR_ENGINE.md       # Spec for the TPDR recommendation engine (to build)
│
│  ── Batch Analysis (complete) ──
├── tar_maf_analyzer.py               # Task 1: Problem→Solution pipeline
├── part_failure_analysis.py          # Task 2: Top failing parts with AI summaries
├── analysis_results_cleaned.json     # 10 problem clusters with solutions, breakdowns, insights
├── part_failure_analysis.json        # 20 parts with failure counts and AI summaries
├── clean_results.py                  # Utility: clean analysis results
├── cleanup_results.py               # Utility: additional cleanup
├── .cache/                           # Cached embeddings, cluster assignments, TPDR results
│
│  ── Real-Time Lookup Tool (complete) ──
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app with routes
│   ├── indexer.py                    # Startup: load embeddings, TAR data, MAF index
│   ├── search.py                     # Core RAG: embed query → cosine search → retrieve context
│   ├── llm.py                        # Ollama helpers (embed, generate, JSON parse)
│   ├── tpdr.py                       # TPDR Recommendation Engine (trend, comeback, scoring)
│   ├── models.py                     # Pydantic request/response models
│   └── static/
│       └── index.html                # Single-page frontend (dark theme, tabbed: TAR Lookup + TPDR)
│
│  ── TPDR Recommendation Engine (to build) ──
│   ├── tpdr.py                       # TPDR analysis: trend detection, comeback detection, scoring
│   (new endpoints added to main.py, new models added to models.py, new tab added to index.html)
│
├── scripts/
│   ├── build_index.py                # Precompute/verify search index
│   └── save_cluster_assignments.py   # Cache cluster assignments to avoid KMeans re-run
├── docs/
│   ├── tar-system-initial.png        # Screenshot: app initial state
│   └── tar-system-search-results.png # Screenshot: search results
├── requirements.txt
└── run.sh                            # Start script for the web app
```

## Running

### Batch Analysis (already complete — results exist)
```bash
python3 tar_maf_analyzer.py           # Rerun full pipeline if needed (~13 min)
python3 part_failure_analysis.py      # Rerun part analysis if needed
```

### Real-Time Lookup + TPDR App
```bash
pip install -r requirements.txt
./run.sh                              # Starts FastAPI on port 8000
```

## Real-Time Lookup Architecture (app/)

RAG pipeline for instant TAR diagnosis:
1. **Embed** incoming TAR text via `nomic-embed-text` (~100ms)
2. **Cosine similarity search** against all ~15K preloaded TAR embeddings
3. **Map to cluster** — majority vote of top-k neighbors → cluster profile
4. **Retrieve MAF context** — linked corrective actions from pre-indexed JCN→MAF dict
5. **Cross-reference parts** — check against `part_failure_analysis.json`
6. **(Optional) AI recommendation** — send context to `qwen2.5:32b` for tailored advice

**Two speed paths:**
- **Fast path** (steps 1–5): <1 second, no LLM call needed
- **AI path** (step 6): ~10–15 seconds via Ollama

## TPDR Recommendation Engine Architecture (app/tpdr.py)

Automatic detection of systems needing Technical Publications Deficiency Reports:
1. **System Trend Detection** — monthly TAR counts per UNS, acceleration ratio (recent vs earlier)
2. **Comeback Detection** — same aircraft + same system, another TAR within 90 days = fix didn't hold
3. **TPDR Scoring** — combines acceleration, comeback count, fleet impact, urgency into a ranked score
4. **AI Justification** — LLM generates TPDR justification paragraph for top candidates

**Known high-signal patterns in the data:**
- UNS 6322 "PROPROTOR GEARBOX ASSEMBLY RH": 5.1x acceleration (83→422 TARs)
- UNS 6321 "PROPROTOR GEARBOX ASSEMBLY LH": 4.6x acceleration (96→441 TARs)
- 575 comeback instances fleet-wide (same aircraft+system within 90 days)
- 166 aircraft+system combos with 5+ repeat TARs

Results are cached to `.cache/tpdr_analysis.json` to avoid re-running LLM calls on restart.

## API Endpoints

**Existing (Real-Time Lookup):**
- `POST /api/search` — core RAG search, returns cluster match + similar TARs + MAF actions
- `POST /api/recommend` — AI recommendation based on search results
- `GET /api/clusters` — all 10 cluster profiles
- `GET /api/parts` — top 20 failing parts
- `GET /api/stats` — database statistics
- `GET /` — serve frontend

**TPDR Endpoints:**
- `GET /api/tpdr/recommendations` — scored, ranked TPDR candidates
- `GET /api/tpdr/trends` — system-level trend data
- `GET /api/tpdr/comebacks` — comeback analysis data
- `GET /api/tpdr/system/{uns_code}` — detailed drilldown for a specific system

## Key Implementation Details

- MAF file is read in 100K-row chunks (`pd.read_csv(..., chunksize=100_000)`) due to 1.8M row size
- `call_ollama_json()` strips deepseek-r1's `<think>...</think>` blocks before parsing JSON
- `call_ollama()` has retry logic with exponential backoff and a configurable delay (`LLM_DELAY`)
- Embeddings are cached by SHA-256 hash of input texts in `.cache/`
- For cosine similarity: normalize embeddings at load time, use `np.dot(query, matrix.T)`
- MAF JCN index (dict mapping jcn→MAF records) is built once on startup
- `submit_date` has mixed formats — always parse with `pd.to_datetime(..., errors='coerce')`
- `buno` of "I-level" = not aircraft-specific, exclude from per-aircraft analysis

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

**part_failure_analysis.json** — Top 5 of 20:
- 901-011-420-101: 6,831 failures
- 901-336-007-115: 5,693 failures
- MN1400: 3,837 failures
- 901-363-203-105: 3,649 failures
- 901-336-006-115: 2,691 failures

## Dependencies

Python 3.13+: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `ollama`, `pydantic`, `tqdm`, `rich`
