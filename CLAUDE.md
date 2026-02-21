# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision

On-premises maintenance intelligence platform for V-22 Osprey that integrates into existing TAR workflows, runs on air-gapped networks with zero cloud dependencies, and puts actionable data in front of maintainers without requiring them to change how they work. Every design decision filters through three questions: Does it run offline? Does it make the maintainer's job faster? Can it deploy to a classified network tomorrow?

## Project Overview

**V-22 TAR Intelligence System** — AI-powered maintenance decision support. Three-tab web application:

1. **Fleet Analytics** (complete) — visualizes batch analysis results: 10 problem categories with solution breakdowns, 20 high-failure parts with AI summaries. Horizontal bar charts, expandable detail cards, all from API data.
2. **TAR Lookup** (complete) — paste a TAR, get sub-second results: problem category match, similar past cases with MAF corrective actions, parts cross-reference, optional AI recommendation.
3. **TPDR Intelligence** (complete) — automatic detection of systems needing Technical Publications Deficiency Reports: trend acceleration, comeback detection, scored candidates with AI-generated justifications.

**Current scope:** This is a demo built against a sample of the full database (~15K TARs, ~1.8M MAFs). Architecture is designed to scale to the full dataset with a vector database swap.

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
├── PROMPT_BUILD_TPDR_ENGINE_1.md     # Spec for the TPDR recommendation engine (complete)
├── PROMPT_BUILD_FLEET_ANALYTICS.md   # Spec for the Fleet Analytics tab (complete)
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
│  ── Web Application ──
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app with routes
│   ├── indexer.py                    # Startup: load embeddings, TAR data, MAF index
│   ├── search.py                     # Core RAG: embed query → cosine search → retrieve context
│   ├── llm.py                        # Ollama helpers (embed, generate, JSON parse)
│   ├── tpdr.py                       # TPDR analysis: trend, comeback, scoring, AI justification
│   ├── models.py                     # Pydantic request/response models
│   └── static/
│       └── index.html                # Single-page frontend (3 tabs: Fleet Analytics, TAR Lookup, TPDR Intelligence)
│
├── scripts/
│   ├── build_index.py                # Precompute/verify search index
│   └── save_cluster_assignments.py   # Cache cluster assignments to avoid KMeans re-run
├── docs/
│   ├── tar-system-initial.png        # Screenshot: app initial state
│   └── tar-system-search-results.png # Screenshot: search results
├── demo_dashboard.html               # SUPERSEDED by Fleet Analytics tab, kept for reference
├── requirements.txt
└── run.sh                            # Start script for the web app
```

## Running

### Batch Analysis (already complete — results exist)
```bash
python3 tar_maf_analyzer.py           # Rerun full pipeline if needed (~13 min)
python3 part_failure_analysis.py      # Rerun part analysis if needed
```

### Web Application
```bash
pip install -r requirements.txt
./run.sh                              # Starts FastAPI on port 8000
```

## Architecture

### RAG Pipeline (TAR Lookup)
1. **Embed** incoming TAR text via `nomic-embed-text` (~100ms)
2. **Cosine similarity search** against all ~15K preloaded TAR embeddings
3. **Map to cluster** — majority vote of top-k neighbors → cluster profile
4. **Retrieve MAF context** — linked corrective actions from pre-indexed JCN→MAF dict
5. **Cross-reference parts** — check against `part_failure_analysis.json`
6. **(Optional) AI recommendation** — send context to `qwen2.5:32b` for tailored advice

**Two speed paths:**
- **Fast path** (steps 1–5): <1 second, no LLM call needed
- **AI path** (step 6): ~10–15 seconds via Ollama

### TPDR Analysis (app/tpdr.py)
1. **System Trend Detection** — monthly TAR counts per UNS, acceleration ratio (recent vs earlier)
2. **Comeback Detection** — same aircraft + same system, another TAR within 90 days = fix didn't hold
3. **TPDR Scoring** — combines acceleration, comeback count, fleet impact, urgency into ranked score
4. **AI Justification** — LLM generates TPDR justification paragraph for top candidates

Results cached to `.cache/tpdr_analysis.json` (24-hour expiry).

### Fleet Analytics
- Reads from `/api/clusters` (analysis_results_cleaned.json) and `/api/parts` (part_failure_analysis.json)
- Horizontal bar charts, expandable detail cards, solution breakdowns, AI insights
- No external JS dependencies, pure CSS/HTML bars

## API Endpoints

**Fleet Analytics & Stats:**
- `GET /api/clusters` — all 10 cluster profiles (includes solutions_extracted, typical_solution, solution_breakdown)
- `GET /api/parts` — top 20 failing parts with AI summaries
- `GET /api/stats` — database statistics

**TAR Lookup:**
- `POST /api/search` — core RAG search, returns cluster match + similar TARs + MAF actions
- `POST /api/recommend` — AI recommendation based on search results
- `GET /` — serve frontend

**TPDR Intelligence:**
- `GET /api/tpdr/recommendations` — scored, ranked TPDR candidates
- `GET /api/tpdr/trends` — system-level trend data
- `GET /api/tpdr/comebacks` — comeback analysis data
- `GET /api/tpdr/system/{uns_code}` — detailed drilldown for a specific system

## Key Data Structures

**analysis_results_cleaned.json** cluster fields:
`problem`, `cluster_id`, `occurrences`, `linked_mafs`, `typical_solution`, `solution_breakdown`, `parts_commonly_involved`, `average_manhours`, `sample_tar_jcns`, `solutions_extracted`

**solutions_extracted** is an array of objects: `{action, component, reference}` where reference is often an IETM reference number. These are the structured corrective actions extracted from MAF data.

**part_failure_analysis.json** part fields:
`part_number`, `failure_count`, `ai_summary`

**TPDR recommendation fields** (from `/api/tpdr/recommendations`):
`uns`, `score`, `total_tars`, `aircraft_count`, `acceleration_ratio`, `comeback_count`, `monthly_counts` (array), `months_labels` (array), `affected_aircraft` (array), `activities` (array), `example_comebacks` (array), `justification`, `linked_cluster`, `sample_tars` (array)

## Key Implementation Details

- Frontend is a single HTML file, zero external JS dependencies, no framework, no build step
- Frontend is ~4K lines — must be read in chunks (offset/limit) when using Read tool
- MAF file is read in 100K-row chunks (`pd.read_csv(..., chunksize=100_000)`) due to 1.8M row size
- `call_ollama_json()` strips deepseek-r1's `<think>...</think>` blocks before parsing JSON
- `call_ollama()` has retry logic with exponential backoff and a configurable delay (`LLM_DELAY`)
- Embeddings cached by SHA-256 hash of input texts in `.cache/`
- For cosine similarity: normalize embeddings at load time, use `np.dot(query, matrix.T)`
- MAF JCN index (dict mapping jcn→MAF records) built once on startup
- `submit_date` mixed formats — always parse with `pd.to_datetime(..., errors='coerce')`
- `buno` of "I-level" = not aircraft-specific, exclude from per-aircraft analysis
- Tab switching uses lazy-load pattern: Fleet Analytics and TPDR data fetch on first tab activation
- All state stored client-side or in .cache/ JSON files, no database required for demo
- **Validating JS syntax** (no server needed): extract the `<script>` block from index.html and pass it through Node's syntax check with `node --check`

**localStorage keys** (all prefixed `tars_`):
- `tars_tarFlags` — TAR Lookup "Not Related" flags: `{ "JCN": { flagged: true, timestamp: "ISO" } }`
- `tars_tpdrWorkflow` — TPDR candidate states: `{ "UNS_CODE": { state: "filed"|"deferred", timestamp, note } }`
- `tars_tpdrThresholds` — TPDR filter values: `{ minTotalTars, minAircraft, minAcceleration, minComebacks, minScore, minMonthlyRate }`
- `tars_tpdrSelectedActivities` — TPDR activity/unit multi-select filter (array of activity strings, or null)

## Known High-Signal Patterns

- UNS 6322 "PROPROTOR GEARBOX ASSEMBLY RH": 5.1x acceleration (83→422 TARs)
- UNS 6321 "PROPROTOR GEARBOX ASSEMBLY LH": 4.6x acceleration (96→441 TARs)
- UNS 6320 "GEARBOX ASSEMBLIES": 10.7x acceleration
- 575 comeback instances fleet-wide (same aircraft+system within 90 days)
- 166 aircraft+system combos with 5+ repeat TARs

## Existing Analysis Results (precomputed)

**10 Clusters:**
FADEC System Malfunction (2,213), Composite/Metal Structural Damage (2,154), Component Failures/Intermittent Faults (2,131), Test Equipment/Procedure Requests (1,821), Aircraft Component Failure/Maintenance (1,594), Maintenance/Repair Procedures (1,411), Sealing/Structural Integrity (1,339), Water Intrusion/Corrosion (927), Proprotor Gearbox Debris (726), PCA Ball Screw Wear (609)

**Top 5 of 20 Parts:**
901-011-420-101 (6,831), 901-336-007-115 (5,693), MN1400 (3,837), 901-363-203-105 (3,649), 901-336-006-115 (2,691)

## Dependencies

Python 3.13+: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `ollama`, `pydantic`, `tqdm`, `rich`
