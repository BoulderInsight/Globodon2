# Build: Fleet Analytics Tab (Integrate Existing Dashboard)

## Context

We're adding a **Fleet Analytics** tab to the V-22 TAR Intelligence System. This is the third and final tab, completing the platform. The app currently has two tabs: "TAR Lookup" (real-time search) and "TPDR Intelligence" (trend/comeback analysis). The Fleet Analytics tab brings in the batch analysis visualization that currently lives as a standalone file (`demo_dashboard.html`).

The tab order should be: **Fleet Analytics** | **TAR Lookup** | **TPDR Intelligence**. This tells the story: foundation first, daily tool second, strategic layer third.

See `CLAUDE.md` for full project architecture.

## What Already Exists

**`demo_dashboard.html`** (standalone, ~750 lines) contains:
- Chart.js bar chart of 10 problem clusters (TAR counts per cluster)
- Chart.js bar chart of top 20 failing parts (failure counts)
- Expandable cards for each cluster showing: solution breakdown (pie-style percentages), parts commonly involved, average manhours, sample JCNs, extracted solutions, and AI-generated insight paragraph
- Hardcoded data embedded in `<script>` tags
- Its own dark theme (slightly different colors than the app)
- External CDN dependency: `chart.js@4.4.7`

**Backend endpoints that already exist:**
- `GET /api/clusters` returns `analysis_results_cleaned.json` (10 cluster profiles with full details)
- `GET /api/parts` returns `part_failure_analysis.json` (20 parts with failure counts and AI summaries)
- `GET /api/stats` returns total TARs, total MAFs, cluster count, parts tracked

**The app frontend** (`app/static/index.html`, ~2,795 lines) already has:
- Tab navigation system (two tabs with show/hide switching)
- Dark theme with CSS variables
- DOM helper function `h()` for building elements
- `apiGet()` and `apiPost()` helpers
- Loading/error/results state pattern used consistently
- SVG sparklines (already built for TPDR)

## What to Build

### 1. Reorder Tabs

Change the tab navigation in `index.html` to three tabs in this order:
```
[ Fleet Analytics ]  [ TAR Lookup ]  [ TPDR Intelligence ]
```

Default active tab should remain TAR Lookup (it's the most used daily tool). Fleet Analytics loads its data on first tab activation (same lazy-load pattern as TPDR).

Tab icons (inline SVGs, same style as existing):
- Fleet Analytics: bar chart icon
- TAR Lookup: search/magnifying glass icon (already exists)
- TPDR Intelligence: trend line icon (already exists)

### 2. Fleet Analytics Tab Panel

Create a new `<div id="tabFleetAnalytics">` tab panel with these sections:

**Section A: Problem Categories Overview**

A bar chart showing all 10 clusters with their TAR counts. Do NOT use Chart.js or any external library. Build the chart with pure CSS/HTML bars, matching the same visual style as the TPDR monthly bar charts already in the app. Each bar should:
- Be horizontal (cluster name on left, bar extending right)
- Show TAR count at the end of the bar
- Use the accent color palette (vary colors across bars using the existing CSS variables: `--bar-replace`, `--bar-inspect`, `--bar-adjust`, `--bar-repair`, `--bar-other`, `--accent-blue`, `--accent-cyan`, `--accent-green`, `--accent-amber`, `--accent-purple`)
- Be clickable to expand the cluster detail card below it

**Section B: Cluster Detail Cards**

When a cluster bar is clicked, expand a detail card below it (same expand/collapse pattern as TPDR candidates and Similar Past Cases). Each card shows:

- **Solution Breakdown**: horizontal bars showing action type percentages (Replace: 60%, Inspect: 15%, etc.). Reuse the exact same bar style from the TAR Lookup diagnosis panel.
- **Parts Commonly Involved**: list of part numbers as badges (same style as TPDR aircraft badges)
- **Average Manhours**: single stat number
- **Sample JCNs**: monospace list
- **AI Insight**: the `ai_insight` paragraph from the cluster profile, displayed in a styled quote/callout box
- **Typical Solution**: brief text summary
- **Occurrences**: TAR count for this cluster

**Section C: Top Failing Parts**

A horizontal bar chart of the top 20 parts by failure count. Same pure CSS/HTML bar style as the clusters chart. Each bar shows:
- Part number on the left
- Failure count at the end
- Color intensity proportional to failure count

Below the chart, show expandable cards for each part with:
- Part number (heading)
- Failure count
- AI summary text

**Section D: Quick Stats** (optional, at top of tab)

Four stat boxes in a row (same style as the app header stats):
- Total TARs Analyzed
- Total MAF Records
- Problem Categories
- Parts Tracked

These already come from `/api/stats` and are displayed in the header, but repeating them at the top of the Fleet Analytics tab gives context when someone lands on that tab.

### 3. Data Loading

On first activation of the Fleet Analytics tab:
1. Call `GET /api/clusters` to get cluster profiles
2. Call `GET /api/parts` to get part failure data
3. Render both sections
4. Cache the data in a JS variable so subsequent tab switches don't re-fetch

Show loading spinner while fetching (same pattern as TPDR tab).

### 4. Delete the Standalone File

After the tab is working, `demo_dashboard.html` is obsolete. Do not delete it (leave that to the user), but add a comment at the top of it noting it's been superseded by the Fleet Analytics tab.

### 5. Visual Requirements

- Same dark theme, same CSS variables, same card styles as existing tabs
- No external charting libraries. Pure CSS/HTML bars and the existing SVG sparkline helper
- Horizontal bar charts, not vertical (they display long system names and part numbers better)
- Responsive: bars should work on different screen widths
- Smooth expand/collapse transitions on detail cards
- Color palette should be consistent with existing TPDR and diagnosis panels

### 6. Update Tab Switching Logic

Update the `switchTab()` function to handle three tabs:
- Track which tab is active
- Show/hide the correct panel
- Update active tab button styling
- Lazy-load Fleet Analytics data on first activation (same as TPDR pattern)

## Implementation Notes

- The cluster data from `/api/clusters` returns the full `analysis_results_cleaned.json` structure. Each cluster has: `problem`, `cluster_id`, `occurrences`, `solution_breakdown`, `typical_solution`, `parts_commonly_involved`, `average_manhours`, `sample_tar_jcns`, `solutions`, `ai_insight`
- The parts data from `/api/parts` returns the full `part_failure_analysis.json` structure. Each part has: `part_number`, `failure_count`, `ai_summary`
- The solution breakdown values are percentages that should sum to ~100% per cluster
- Do NOT add Chart.js or any CDN dependencies. The app currently has zero external JS dependencies and should stay that way.
- The `h()` helper function already in the codebase handles element creation. Use it consistently.

## Definition of Done

1. Three tabs appear: Fleet Analytics | TAR Lookup | TPDR Intelligence
2. TAR Lookup remains the default active tab
3. Fleet Analytics loads cluster and part data from the API on first activation
4. Horizontal bar chart shows all 10 problem clusters with TAR counts
5. Clicking a cluster bar expands a detail card with solution breakdown, parts, manhours, AI insight
6. Horizontal bar chart shows top 20 failing parts with failure counts
7. Parts are expandable with AI summaries
8. No external JS dependencies added
9. Same visual quality and theme as existing tabs
10. `demo_dashboard.html` has a comment noting it's superseded
