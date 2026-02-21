# Build: TPDR Recommendation Engine

## Context

We're adding a **TPDR Recommendation Engine** to the existing V-22 TAR Intelligence System. A TPDR (Technical Publications Deficiency Report) is filed when enough evidence exists that a maintenance procedure or publication needs updating. Currently, identifying when a TPDR is warranted requires someone to manually notice patterns across thousands of TARs — this feature automates that detection.

The existing app is a FastAPI + single-page HTML tool where maintainers paste a TAR and get instant diagnosis from historical data. See `CLAUDE.md` for full architecture. The app is in `app/` and works. We're adding a new tab/view to the frontend and new API endpoints.

## What Already Exists (read CLAUDE.md for full details)

- `app/main.py` — FastAPI app with search/recommend/clusters/parts/stats endpoints
- `app/indexer.py` — loads TAR data, embeddings, cluster profiles, MAF index on startup
- `app/search.py` — RAG search pipeline
- `app/llm.py` — Ollama helpers (embed, generate, JSON parse)
- `app/models.py` — Pydantic models
- `app/static/index.html` — single-page frontend (dark theme, ~1980 lines)
- `analysis_results_cleaned.json` — 10 problem clusters with solutions/insights
- `part_failure_analysis.json` — 20 top failing parts

## Data Available for This Feature

All data is in the project root (CSVs not in git but present on disk).

**TAR_Data.csv** (~14,926 rows) key fields:
- `submit_date` — datetime string like "9/7/2022 10:23:56 AM" (range: Sep 2020 to Mar 2024, 43 months)
- `buno` — aircraft Bureau Number / tail number (431 unique, but some are "I-level" meaning not aircraft-specific)
- `uns` — Unified Numbering System code identifying the aircraft system, e.g. "6322 PROPROTOR GEARBOX ASSEMBLY RH" (305 unique codes)
- `aircraft_type` — "MV", "CV", "CMV"
- `activity` — unit/squadron, e.g. "VMM 266", "MALS 26"
- `priority` — "Routine", "Immediate (24 Hours)", "Urgent (72 Hours)"
- `status` — "Open", "Responded", "Closed", "Submitted"
- `jcn` — Job Control Number (links to MAF)
- `subject`, `issue` — free text description

**maf.csv** (~1.8M rows) key fields for this feature:
- `serno` — aircraft serial number (maps to TAR `buno` for 357 aircraft)
- `jcn` — links to TAR
- `comp_date` — completion date
- `wuc` — Work Unit Code (system identifier)
- `action_taken` — code: R=replace, C=clean, B=repair, etc.

## What the Data Shows (I already validated this)

**System-level acceleration (TPDR gold):**
- UNS 6322 "PROPROTOR GEARBOX ASSEMBLY RH": 83 TARs first half → 422 second half (5.1x increase)
- UNS 6321 "PROPROTOR GEARBOX ASSEMBLY LH": 96 → 441 (4.6x increase)
- UNS 6320 "GEARBOX ASSEMBLIES": 6 → 64 (10.7x increase)
- UNS 6300 "DRIVE SYSTEM": 11 → 50 (4.5x increase)
- 97 systems have 20+ total TARs — enough for trending

**Comeback patterns (fix-didn't-hold):**
- 575 instances where the same aircraft + same system got another TAR within 90 days
- 1,977 aircraft+system combos have multiple TARs total
- 166 combos have 5+ TARs on the same aircraft for the same system

**This means the TPDR engine has real, compelling patterns to surface.**

## What to Build

### New Backend: `app/tpdr.py`

A module that computes TPDR recommendations by analyzing the TAR data loaded in `app/indexer.py`. This runs on startup (after the index is loaded) and can be refreshed via API.

**Analysis 1: System Trend Detection**

For each UNS code with 10+ TARs:
1. Parse `submit_date` into datetime
2. Compute monthly TAR counts
3. Calculate trend: compare last 6 months vs prior 6 months (or split into halves if data range is shorter)
4. Compute acceleration ratio (recent / earlier)
5. Flag systems with ratio > 2.0 as "accelerating"
6. Generate a 12-month sparkline data array (monthly counts) for frontend visualization

Output per system:
```python
{
    "uns": "6322 PROPROTOR GEARBOX ASSEMBLY RH",
    "total_tars": 505,
    "first_half_count": 83,
    "second_half_count": 422,
    "acceleration_ratio": 5.1,
    "monthly_counts": [12, 8, 15, 22, 35, 41, ...],  # last 12 months
    "months_labels": ["2023-04", "2023-05", ...],
    "affected_aircraft": ["168225", "169447", ...],  # unique bunos
    "aircraft_count": 45,
    "priority_breakdown": {"Routine": 380, "Immediate (24 Hours)": 85, "Urgent (72 Hours)": 40},
    "units_affected": ["VMM 266", "VMM 162", ...],
    "linked_cluster": "Proprotor Gearbox Debris Detection Issues",  # from analysis_results_cleaned.json
    "sample_tars": [{"jcn": "...", "subject": "...", "submit_date": "...", "buno": "..."}],
}
```

**Analysis 2: Comeback Detection**

For each aircraft (buno) + system (uns) combo:
1. Sort TARs by submit_date
2. Find pairs where the gap is 1-90 days (same aircraft, same system, within 90 days = likely the fix didn't hold)
3. Aggregate: which systems have the most comebacks across the fleet?
4. For each flagged system, compile:
   - Total comeback instances
   - Unique aircraft affected
   - Average gap between TARs
   - Most common corrective actions from linked MAFs (via JCN → MAF index)

Output per flagged system:
```python
{
    "uns": "2750 FLIGHT CONTROL ACTUATION",
    "comeback_count": 28,
    "unique_aircraft": 15,
    "avg_gap_days": 34,
    "common_fixes": [{"action": "replace", "count": 18}, {"action": "adjust", "count": 7}],
    "example_comebacks": [
        {"buno": "040026", "gap_days": 68, "first_tar": "...", "second_tar": "..."}
    ]
}
```

**Analysis 3: TPDR Scoring & Evidence Package**

Combine trend + comeback data into a TPDR recommendation score:
```python
score = (acceleration_ratio * 20) + (comeback_count * 3) + (aircraft_count * 2) + (urgent_priority_count * 5)
```
The exact formula can be tuned, but the idea is: accelerating trends + repeat failures + fleet-wide impact + urgency = strong TPDR candidate.

For the top 15 scored systems, use `qwen2.5:32b` via the existing `app/llm.py` helpers to generate a TPDR justification paragraph. Prompt should include: the system name, TAR count, trend data, comeback data, affected aircraft count, and sample corrective actions. The LLM should write 3-4 sentences that a maintenance supervisor could paste into a TPDR submission.

**Precompute this on startup** (after index loads). Store results in a module-level variable like `tpdr_results`. The LLM calls for 15 systems will take ~2-3 minutes on first startup, so cache the results to `.cache/tpdr_analysis.json`. On subsequent startups, load from cache if the file exists and is less than 24 hours old.

### New API Endpoints (add to `app/main.py`)

- `GET /api/tpdr/recommendations` — returns the scored, ranked list of TPDR candidates with all their data
- `GET /api/tpdr/trends` — returns system-level trend data (all systems with 10+ TARs, sorted by acceleration)
- `GET /api/tpdr/comebacks` — returns comeback analysis data
- `GET /api/tpdr/system/{uns_code}` — returns detailed drilldown for a specific system: all TARs, timeline, linked MAFs, comeback instances

### New Pydantic Models (add to `app/models.py`)

Add response models for the TPDR endpoints. Follow the same patterns as the existing `SearchResponse`, `MatchedCluster`, etc.

### Frontend: New TPDR Tab

Add a second tab/view to `app/static/index.html`. The existing search view stays as "TAR Lookup" tab. The new tab is "TPDR Intelligence."

**Tab Navigation:**
Add a tab bar below the header:
```
[ 🔍 TAR Lookup ]  [ 📊 TPDR Intelligence ]
```
Tab switching shows/hides the relevant content sections. Default to TAR Lookup.

**TPDR Intelligence Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│  TPDR CANDIDATES                                    [Refresh] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ #1 — Score: 847 ──────────────────────────────────────┐ │
│  │  6322 PROPROTOR GEARBOX ASSEMBLY RH                     │ │
│  │                                                          │ │
│  │  ┌─────────┬──────────┬────────────┬───────────────┐    │ │
│  │  │ 505     │ 5.1x     │ 45         │ 28            │    │ │
│  │  │ Total   │ Accel.   │ Aircraft   │ Comebacks     │    │ │
│  │  │ TARs    │ Rate     │ Affected   │ (90-day)      │    │ │
│  │  └─────────┴──────────┴────────────┴───────────────┘    │ │
│  │                                                          │ │
│  │  Trend: ▁▂▂▃▅▆█▇██ (12-month sparkline)               │ │
│  │                                                          │ │
│  │  AI Justification:                                       │ │
│  │  "The RH Proprotor Gearbox Assembly has seen a 5.1x     │ │
│  │   increase in TARs, affecting 45 aircraft across the     │ │
│  │   fleet. 28 comeback instances suggest current repair    │ │
│  │   procedures are insufficient..."                        │ │
│  │                                                          │ │
│  │  [▼ View Details]                                        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ #2 — Score: 612 ──────────────────────────────────────┐ │
│  │  6321 PROPROTOR GEARBOX ASSEMBLY LH                     │ │
│  │  ...                                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ #3 — Score: 445 ──────────────────────────────────────┐ │
│  │  ...                                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  FLEET TREND OVERVIEW                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ Accelerating ─────────────────────────────────────────┐ │
│  │  🔴 6320 GEARBOX ASSEMBLIES          10.7x  ▁▁▂▃▅▇██  │ │
│  │  🔴 6322 PROPROTOR GEARBOX RH         5.1x  ▁▂▃▅▆███  │ │
│  │  🟡 6321 PROPROTOR GEARBOX LH         4.6x  ▁▂▃▄▆███  │ │
│  │  🟡 6300 DRIVE SYSTEM                 4.5x  ▁▁▂▃▅▆██  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ Declining ────────────────────────────────────────────┐ │
│  │  🟢 4351 INTERCOM SET                 0.2x  ██▆▃▂▁▁▁  │ │
│  │  🟢 7851b COANDA                      0.1x  ██▅▂▁▁▁▁  │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Detail View (expandable per TPDR candidate):**
When clicking "View Details" on a candidate, expand to show:
- Full timeline chart (monthly TAR counts as a bar chart using CSS)
- List of affected aircraft (buno) with TAR count per aircraft
- Comeback instances table: aircraft, dates, gap, subjects
- Units/squadrons affected
- Linked cluster info from the existing analysis
- Sample TARs with their subjects and dates

**Sparklines:**
Use inline SVG or CSS for sparklines — no charting library needed. They're just 12 numbers rendered as small bars or a polyline. Each sparkline represents monthly TAR counts for the last 12 months of data.

**Color coding for acceleration:**
- 🔴 Red: ratio > 3.0 (critical acceleration)
- 🟡 Amber: ratio 2.0-3.0 (moderate acceleration)
- 🟢 Green: ratio < 1.0 (declining)
- ⚪ Gray: ratio 1.0-2.0 (stable)

**UI Requirements:**
- Same dark theme as existing TAR Lookup tab
- Same font, spacing, card styles — must look like part of the same app
- Tab switching is instant (just show/hide, data loads on first tab activation)
- TPDR data loads via API on tab activation, with loading spinner
- Expandable detail cards (same pattern as Similar Past Cases in existing UI)

### Indexer Updates (`app/indexer.py`)

Add to the `load_index()` function: after loading all existing data, parse `submit_date` into a datetime column on `tar_df` so `tpdr.py` can use it for time-series analysis. Handle the mixed date formats ("9/7/2022 10:23:56 AM" and "10/9/2024" — some have time, some don't). Use `pd.to_datetime(tar_df['submit_date'], errors='coerce')`.

Also ensure `buno`, `activity`, and `priority` columns are available on the TAR dataframe (they should already be loaded since the full CSV is read).

### Implementation Notes

- The `uns` field is the system identifier — it contains both a numeric code and a description like "6322 PROPROTOR GEARBOX ASSEMBLY RH". Use the full string as the key/identifier.
- `buno` values of "I-level" mean the TAR was for intermediate-level maintenance not tied to a specific aircraft — exclude these from aircraft-specific analysis (comeback detection, aircraft count) but include them in total TAR counts.
- For linking systems to existing cluster profiles: match by checking if any TAR JCNs in that UNS group are also in the cluster's `sample_tar_jcns`, or use the cluster assignments already computed in the indexer.
- The TPDR analysis should run AFTER the main index loads (it depends on `index.tar_df` and `index.maf_index`). Can be async/background if startup time is a concern, with the API returning partial results while LLM summaries generate.
- Cache TPDR results to `.cache/tpdr_analysis.json` to avoid re-running LLM calls on every restart.

## Definition of Done

1. New "TPDR Intelligence" tab appears in the app alongside "TAR Lookup"
2. TPDR candidates are ranked by score with sparklines, stats, and AI justification
3. Fleet Trend Overview shows accelerating and declining systems
4. Expanding a candidate shows detailed timeline, affected aircraft, comeback instances
5. Data loads from the API correctly with loading states
6. Same visual quality as existing TAR Lookup tab
7. TPDR analysis results are cached to avoid redundant LLM calls on restart
