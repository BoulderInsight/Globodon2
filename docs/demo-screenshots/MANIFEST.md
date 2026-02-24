# Demo Screenshots Manifest

Captured: February 24, 2026
Viewport: 1400x900

## TAR Lookup Tab

| # | Filename | Description | Key Features |
|---|----------|-------------|--------------|
| 1 | `01-landing-page.png` | Fresh load of the app on TAR Lookup tab | Header stats (14,925 TARs, 28,061 MAFs, 182 Systems, 994 Parts), three navigation tabs, TAR Queue with filters, TAR Input area |
| 2 | `02-tar-queue.png` | TAR Queue section close-up | Sortable columns (JCN, Date, Priority, System, Subject), filter dropdowns (Work Center, Activity, Priority), 50 TARs loaded |
| 3 | `03-gearbox-diagnosis.png` | Diagnosis for "Gearbox Chip Detector" sample | Cluster: "Proprotor Gearbox Debris Detection Issues", 90% confidence, 726 past occurrences, 31.5 hrs avg resolution, solution breakdown bars, parts to have ready (901-044-001-115, 901-044-002-113), historical insight paragraph |
| 4 | `04-gearbox-similar-cases.png` | Similar Past Cases for gearbox query | 10 similar cases, 83% top match, case cards with JCN, cluster tag, corrective actions with manhours, part numbers highlighted |
| 5 | `05-gearbox-case-expanded.png` | Expanded similar case (JCN: QD8221272) | Aircraft/Activity/Date/Status/Priority metadata row, all 12 corrective actions with manhours, part numbers (901-044-001-115), "Analyze This TAR" button |
| 6 | `06-gearbox-ai-recommendation.png` | AI Recommendation for gearbox query | Root cause analysis, 5 numbered recommended actions, parts needed (PEMA 3, LH Proprotor Gearbox Assembly), estimated time (30-45 manhours), action plan, IETM references (SSS:2621, MCN:4HJPW4J) |
| 7 | `07-gearbox-parts.png` | Parts association and diagnosis overview | Parts to Have Ready with occurrence counts, solution breakdown bars, historical insight, transition to similar cases below |
| 8 | `08-fadec-diagnosis.png` | Diagnosis for "FADEC Channel A Fault" sample | Different cluster: "Aircraft Component Failure and Maintenance Requests", 70% confidence, 1,594 occurrences, 26.3 hrs avg, different solution breakdown (inspect 26.7%, replace 26.7%, remove 13.3%) |

## Fleet Analytics Tab

| # | Filename | Description | Key Features |
|---|----------|-------------|--------------|
| 9 | `09-fleet-system-view.png` | Fleet Analytics System View overview | 182 Systems Tracked in stat bar, System View / Problem Categories toggle, top 30 systems by volume, horizontal bar chart with color coding |
| 10 | `10-fleet-gearbox-expanded.png` | Expanded PROPROTOR GEARBOX ASSEMBLY LH | 2 failure modes detected, quality badge (0.215 amber), failure mode 1: "AFB-200 Rev A" (447 TARs) with 5 corrective actions and manhours, failure mode 2: "LH Input Quill Replacement" (295 TARs), quick stats (Sep 2020-Mar 2024, 347 aircraft), top 5 parts |
| 11 | `11-fleet-engine-expanded.png` | Expanded ENGINE system | 4 failure modes (LH Engine Compressor Stall, 70 HR Engine Washes, Left Engine Hardware Broken, RH Inlet Guide Vane Corrosion), quality badge (0.203), corrective actions per failure mode with manhours |
| 12 | `12-fleet-problem-categories.png` | Problem Categories view | 10 Problem Categories in stat bar, all 10 clusters with occurrence counts, horizontal bar chart, "Top Failing Parts" section below |
| 13 | `13-fleet-category-expanded.png` | Expanded PCA Ball Screw Wear Test Failure | Solution breakdown (inspect 86.7%), 609 occurrences, 9.3 avg manhours, parts commonly involved (9 PCA-specific parts), sample JCNs, extracted solutions table (Action/Component/Reference), AI Insight paragraph |
| 14 | `14-fleet-top-parts.png` | Top Failing Parts chart | 20 parts ranked by failure count, top part 901-011-420-101 (6,831 failures), horizontal bar chart |

## TPDR Intelligence Tab

| # | Filename | Description | Key Features |
|---|----------|-------------|--------------|
| 15 | `15-tpdr-overview.png` | TPDR Intelligence overview | 15 candidates, 3 above threshold, 0 filed, #1 candidate (6322 PROPROTOR GEARBOX ASSEMBLY RH, Score: 2251), stats cards (703 TARs, 4.94x accel, 341 aircraft, 45 comebacks), trend sparkline, AI justification, Mark as Filed / Defer buttons |
| 16 | `16-tpdr-thresholds.png` | TPDR threshold configuration panel | Plain-English threshold explanation, 6 configurable fields (Min TARs, Min Aircraft, Min Accel Rate, Min Comebacks, Min Monthly Rate, Recent Window), Activity/Unit filter chips (35 units), "Showing 3 of 15 active candidates" |
| 17 | `17-tpdr-candidate-detail.png` | TPDR candidate detail (expanded #1) | Full candidate card with score, system name, 4 stat cards, trend sparkline, AI justification paragraph, Monthly TAR Timeline bar chart visible below |
