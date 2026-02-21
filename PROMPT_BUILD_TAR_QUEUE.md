# Build Prompt: TAR Queue Panel

## Context

Read CLAUDE.md first. The TAR Lookup tab currently requires maintainers to paste TAR text into a textarea. This works for a demo but feels like a prototype. Real maintainers have TARs assigned to them — they shouldn't have to copy/paste from one system into another.

This feature adds a **TAR Queue** panel above the existing text input on the TAR Lookup tab. It shows a browsable, searchable list of recent TARs from the dataset. Click a row → TAR text populates the input → search runs automatically. Two clicks from page load to diagnosis.

**Demo story:** "Right now this shows recent TARs from the sample data. In production, this connects to the TAR submission system and shows TARs assigned to the logged-in user's work center. When a new TAR comes in, it appears here with analysis already attached."

---

## Part 1: New API Endpoint

### `GET /api/tars/recent`

Add this to `app/main.py`. The TAR dataframe (`index.tar_df`) is already loaded in memory with all columns.

**Parameters (query string):**
- `limit` (int, default 50, max 200) — number of TARs to return
- `work_center` (str, optional) — filter by work_center field
- `activity` (str, optional) — filter by activity field  
- `priority` (str, optional) — filter by priority field
- `search` (str, optional) — case-insensitive substring match against subject + issue

**Response:** JSON array of objects, sorted by submit_date descending (most recent first). Each object:

```json
{
  "jcn": "string",
  "subject": "string",
  "issue": "string (first 200 chars, truncated)",
  "uns": "string",
  "submit_date": "string",
  "priority": "string",
  "activity": "string",
  "work_center": "string",
  "status": "string",
  "buno": "string",
  "aircraft_type": "string"
}
```

**Implementation notes:**
- Sort by the already-parsed `submit_dt` column (datetime), descending. Put NaT dates last.
- Apply filters before sorting and limiting.
- For the `search` parameter, filter where `(subject + " " + issue).str.lower().str.contains(search.lower())`
- Strip whitespace from all returned string fields.
- This endpoint should be fast (<100ms). No AI calls, no embeddings. Just filter, sort, slice, return.

### `GET /api/tars/filters`

Returns the unique values available for the filter dropdowns.

**Response:**
```json
{
  "work_centers": ["string array, sorted"],
  "activities": ["string array, sorted"],
  "priorities": ["string array, sorted"]
}
```

Only include non-empty values. This endpoint is called once on tab load to populate the filter dropdowns.

---

## Part 2: Frontend — TAR Queue Panel

### Location
TAR Lookup tab, **above** the existing TAR INPUT card. The existing textarea and sample TAR buttons remain below as a fallback for manual input.

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 📋 TAR QUEUE                                    [50 TARs]  │
│                                                             │
│ [Search TARs...          ] [Work Center ▾] [Activity ▾]    │
│                           [Priority ▾]    [Clear Filters]   │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ JCN          DATE        PRIORITY  SYSTEM     SUBJECT   │ │
│ │ W0034521  09/15/2024  ⚡ Urgent   6322 PRGB  FADEC ch… │ │
│ │ W0034519  09/15/2024     Routine  1560 HYDR  Hydraulic… │ │
│ │ W0034517  09/14/2024     Routine  7520 COMM  Radio com… │ │
│ │ W0034515  09/14/2024  ⚡ Urgent   6320 GEAR  Chip det… │ │
│ │ ...                                                      │ │
│ │ (scrollable, max-height ~300px)                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Showing 50 of 14,926 TARs  [Load More]                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ + TAR INPUT                                                 │
│ (existing textarea, unchanged)                              │
└─────────────────────────────────────────────────────────────┘
```

### Behavior

**On tab load (first activation of TAR Lookup tab):**
1. Fetch `GET /api/tars/filters` to populate the filter dropdowns
2. Fetch `GET /api/tars/recent?limit=50` to populate the initial list
3. Show a loading skeleton or spinner while fetching

**Search box:**
- Debounce 300ms after typing stops
- Sends `search` param to `/api/tars/recent`
- Results update in place

**Filter dropdowns:**
- Work Center, Activity, Priority — each is a `<select>` with "All" as default
- Changing any filter re-fetches with the filter params
- "Clear Filters" resets all dropdowns to "All" and search box to empty

**Clicking a TAR row:**
1. Highlight the selected row (accent border or background)
2. Populate the textarea below with `subject + "\n" + issue` (the full text, not truncated)
3. **Automatically trigger the search** — same as clicking "Search Historical Data"
4. Scroll down smoothly to show the results

**The issue field truncation:** The list shows first 200 chars of issue for display. But when clicked, the full `issue` text must be available. Two options:
- Option A: Store the full issue in a `data-issue` attribute on the row element (simple, works since issue is loaded in the API response)
- Option B: Return full issue from the API but only display truncated. Use the full value on click.

Go with Option B — return full issue from the API, just truncate in the display rendering.

**"Load More" button:**
- Fetches the next page. Use offset parameter or track current count.
- Append results to the existing list, don't replace.
- Hide button if returned count < limit (no more results).
- Actually, for simplicity, just increase the limit: first load is 50, "Load More" fetches 200. Most demo scenarios won't need more than that.

### Table Row Styling

- Compact rows: ~36px height, small font (0.8125rem)
- Zebra striping or subtle row borders
- Priority column: show a colored indicator
  - "Urgent" or priority contains "1" → red/orange dot or ⚡ icon
  - "Priority" or "2" → yellow dot
  - "Routine" or "3" → no indicator (default)
- Subject column: truncate with ellipsis, takes remaining width
- JCN column: monospace font
- Date column: formatted short (MM/DD/YYYY)
- Hover: subtle highlight row
- Selected: accent border-left or background tint, stays highlighted until a different row is clicked or search is cleared
- The row should feel clickable (cursor: pointer)

### UNS/System column
- Show just the UNS name (e.g., "PROPROTOR GEARBOX ASSEMBLY RH"), not the code number
- If UNS is too long, truncate with ellipsis
- Title attribute shows full UNS code + name on hover

### Responsive width
The table should handle the dark theme and match existing card styling. Use the same card container style as other sections. Table columns should use reasonable min-widths and let Subject take the remaining space with `flex: 1` or `width: auto`.

---

## Part 3: Connect Click → Search

When a TAR row is clicked, the behavior should exactly mirror what happens when a user types text and clicks "Search Historical Data":

1. Set the textarea value to `subject + "\n" + issue`
2. Call the same search function that the "Search Historical Data" button triggers
3. The diagnosis results appear below as they do today

This means the TAR queue is just a convenient way to populate the input. The entire existing search flow remains unchanged. No modifications to the search API or results rendering.

**Important:** After clicking a TAR and seeing results, the user should be able to click a different TAR from the queue and get new results. The previous results should be cleared and replaced.

---

## Implementation Notes

- This is the **only feature that requires a new backend endpoint**. Keep it minimal — just filtering, sorting, and slicing a dataframe that's already in memory.
- Add `limit: int = 50` and `offset: int = 0` as query params if you want proper pagination. Or keep it simple with just `limit`.
- Use the existing `h()` DOM helper function for all element creation
- Use the existing lazy-load tab pattern — fetch TAR queue data on first tab activation, not on page load
- Match existing dark theme styling: card borders, input styles, button styles
- No external libraries
- The search debounce can use a simple setTimeout/clearTimeout pattern

---

## Definition of Done

1. `GET /api/tars/recent` returns filtered, sorted TAR list from in-memory dataframe
2. `GET /api/tars/filters` returns unique work center, activity, and priority values
3. TAR Queue panel appears above TAR INPUT card on the TAR Lookup tab
4. Search box filters TARs by subject/issue text with 300ms debounce
5. Dropdown filters for work center, activity, priority
6. Clicking a TAR row populates the textarea and auto-triggers search
7. Selected row is visually highlighted
8. "Load More" button fetches additional results
9. Priority indicators on rows (urgent = colored, routine = default)
10. Performance: queue loads in <500ms, individual row click to results in <1.5s
11. No external JS dependencies added
12. Visual quality matches existing tabs
