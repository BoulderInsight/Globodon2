"""TPDR Recommendation Engine — analyzes TAR trends, comeback patterns, and
generates scored recommendations for Technical Publications Deficiency Reports."""

import json
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.indexer import index, CACHE_DIR
from app.llm import call_ollama, CLASSIFY_MODEL

# Module-level results cache
tpdr_results: dict | None = None

CACHE_FILE = CACHE_DIR / "tpdr_analysis.json"
CACHE_MAX_AGE_HOURS = 24


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    """Return True if the cache file exists and is less than 24 hours old."""
    if not CACHE_FILE.exists():
        return False
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        computed_at = datetime.fromisoformat(data["computed_at"])
        return (datetime.now() - computed_at) < timedelta(hours=CACHE_MAX_AGE_HOURS)
    except Exception:
        return False


def _save_cache(results: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _load_cache() -> dict | None:
    if not _cache_is_fresh():
        return None
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Analysis 1: System Trend Detection
# ---------------------------------------------------------------------------

def _analyze_trends() -> list[dict]:
    """For each UNS code with 10+ TARs, compute trend metrics."""
    df = index.tar_df.copy()
    trends = []

    grouped = df.groupby("uns")

    for uns_code, group in grouped:
        uns_code = str(uns_code).strip()
        if not uns_code or uns_code.lower() == "nan":
            continue
        if len(group) < 10:
            continue

        # Filter to rows with valid dates
        dated = group[group["submit_dt"].notna()].copy()
        total_tars = len(group)

        # Acceleration ratio: split date range into halves
        if len(dated) >= 2:
            min_dt = dated["submit_dt"].min()
            max_dt = dated["submit_dt"].max()
            midpoint = min_dt + (max_dt - min_dt) / 2
            first_half = len(dated[dated["submit_dt"] <= midpoint])
            second_half = len(dated[dated["submit_dt"] > midpoint])
            acceleration_ratio = second_half / max(first_half, 1)
        else:
            first_half = total_tars
            second_half = 0
            acceleration_ratio = 0.0

        # Monthly counts for last 12 months of data
        months_labels = []
        monthly_counts = []
        if len(dated) >= 2:
            max_date = dated["submit_dt"].max()
            start_12mo = max_date - pd.DateOffset(months=12)
            recent = dated[dated["submit_dt"] >= start_12mo]
            if len(recent) > 0:
                recent_grouped = recent.set_index("submit_dt").resample("MS").size()
                # Ensure we have 12 entries
                full_range = pd.date_range(
                    start=start_12mo.to_period("M").to_timestamp(),
                    end=max_date.to_period("M").to_timestamp(),
                    freq="MS",
                )
                recent_grouped = recent_grouped.reindex(full_range, fill_value=0)
                months_labels = [d.strftime("%Y-%m") for d in recent_grouped.index]
                monthly_counts = recent_grouped.tolist()

        # Unique bunos (excluding I-level)
        bunos = group["buno"].dropna().unique().tolist()
        bunos = [b for b in bunos if b.strip() and "i-level" not in b.lower()]
        aircraft_count = len(bunos)

        # Activities
        activities = group["activity"].dropna().unique().tolist()
        activities = [a for a in activities if a.strip()]

        # Priority breakdown
        priority_counts = group["priority"].value_counts().to_dict()

        # Urgent priority count
        urgent_count = 0
        for p, c in priority_counts.items():
            p_str = str(p).lower()
            if "immediate" in p_str or "urgent" in p_str:
                urgent_count += c

        # Link to cluster by majority vote
        group_indices = group.index.tolist()
        cluster_ids = []
        for idx in group_indices:
            pos = df.index.get_loc(idx)
            cid = int(index.tar_cluster_ids[pos])
            if 0 <= cid < len(index.cluster_profiles):
                cluster_ids.append(cid)
        linked_cluster = ""
        if cluster_ids:
            most_common_cid = Counter(cluster_ids).most_common(1)[0][0]
            linked_cluster = index.cluster_profiles[most_common_cid]["problem"]

        # Sample TARs (up to 5)
        sample_rows = group.head(5)
        sample_tars = []
        for _, row in sample_rows.iterrows():
            sample_tars.append({
                "jcn": str(row.get("jcn", "")),
                "subject": str(row.get("subject", "")),
                "submit_date": str(row.get("submit_date", "")),
                "buno": str(row.get("buno", "")),
            })

        trends.append({
            "uns": uns_code,
            "total_tars": total_tars,
            "first_half_count": first_half,
            "second_half_count": second_half,
            "acceleration_ratio": round(acceleration_ratio, 2),
            "months_labels": months_labels,
            "monthly_counts": monthly_counts,
            "affected_aircraft": bunos[:20],  # cap display
            "aircraft_count": aircraft_count,
            "activities": activities[:10],
            "priority_breakdown": {str(k): int(v) for k, v in priority_counts.items()},
            "urgent_priority_count": urgent_count,
            "linked_cluster": linked_cluster,
            "sample_tars": sample_tars,
        })

    trends.sort(key=lambda t: t["acceleration_ratio"], reverse=True)
    print(f"  Trend analysis complete: {len(trends)} systems with 10+ TARs")
    return trends


# ---------------------------------------------------------------------------
# Analysis 2: Comeback Detection
# ---------------------------------------------------------------------------

def _analyze_comebacks() -> list[dict]:
    """For each buno+uns combo, detect repeat failures within 1-90 days."""
    df = index.tar_df.copy()

    # Filter out invalid bunos
    valid = df[
        df["buno"].notna()
        & (df["buno"].str.strip() != "")
        & (~df["buno"].str.lower().str.contains("i-level", na=False))
        & df["submit_dt"].notna()
    ].copy()

    if valid.empty:
        return []

    valid = valid.sort_values("submit_dt")

    # Group by (buno, uns) and detect comebacks
    comeback_data: dict[str, dict] = {}  # keyed by uns

    for (buno, uns), grp in valid.groupby(["buno", "uns"]):
        uns = str(uns).strip()
        buno = str(buno).strip()
        if not uns or not buno or uns.lower() == "nan":
            continue
        if len(grp) < 2:
            continue

        dates = grp["submit_dt"].values
        jcns = grp["jcn"].values

        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
            if 1 <= gap_days <= 90:
                if uns not in comeback_data:
                    comeback_data[uns] = {
                        "uns": uns,
                        "comeback_count": 0,
                        "unique_aircraft": set(),
                        "gap_days_list": [],
                        "example_comebacks": [],
                        "jcns": set(),
                    }
                entry = comeback_data[uns]
                entry["comeback_count"] += 1
                entry["unique_aircraft"].add(buno)
                entry["gap_days_list"].append(float(gap_days))
                entry["jcns"].add(str(jcns[i - 1]))
                entry["jcns"].add(str(jcns[i]))

                if len(entry["example_comebacks"]) < 3:
                    entry["example_comebacks"].append({
                        "buno": buno,
                        "first_jcn": str(jcns[i - 1]),
                        "second_jcn": str(jcns[i]),
                        "gap_days": round(float(gap_days), 1),
                        "date_first": str(pd.Timestamp(dates[i - 1]).date()),
                        "date_second": str(pd.Timestamp(dates[i]).date()),
                    })

    # Build common_fixes from MAF index for each system
    comebacks = []
    for uns, entry in comeback_data.items():
        # Tally action_taken codes from linked MAFs
        action_tally: Counter = Counter()
        for jcn in entry["jcns"]:
            jcn_str = str(jcn).strip()
            for maf in index.maf_index.get(jcn_str, []):
                at = maf.get("action_taken", "").strip()
                if at:
                    action_tally[at] += 1

        # Map common action codes
        action_labels = {
            "R": "Replace", "C": "Clean", "B": "Repair",
            "T": "Test", "F": "Fabricate", "I": "Inspect",
            "A": "Adjust", "L": "Lubricate", "M": "Modify",
        }
        common_fixes = {}
        for code, count in action_tally.most_common(5):
            label = action_labels.get(code, code)
            common_fixes[label] = count

        avg_gap = (sum(entry["gap_days_list"]) / len(entry["gap_days_list"])
                   if entry["gap_days_list"] else 0)

        comebacks.append({
            "uns": uns,
            "comeback_count": entry["comeback_count"],
            "unique_aircraft": len(entry["unique_aircraft"]),
            "avg_gap_days": round(avg_gap, 1),
            "common_fixes": common_fixes,
            "example_comebacks": entry["example_comebacks"],
        })

    comebacks.sort(key=lambda c: c["comeback_count"], reverse=True)
    print(f"  Comeback analysis complete: {len(comebacks)} systems with repeat failures")
    return comebacks


# ---------------------------------------------------------------------------
# Analysis 3: TPDR Scoring and AI Justification
# ---------------------------------------------------------------------------

def _score_and_justify(trends: list[dict], comebacks: list[dict]) -> list[dict]:
    """Merge trend + comeback data, score each system, generate AI justification
    for the top 15."""
    # Index comebacks by UNS for fast lookup
    comeback_by_uns = {c["uns"]: c for c in comebacks}

    scored = []
    for trend in trends:
        uns = trend["uns"]
        cb = comeback_by_uns.get(uns, {})
        comeback_count = cb.get("comeback_count", 0)
        cb_aircraft = cb.get("unique_aircraft", 0)
        avg_gap_days = cb.get("avg_gap_days", 0)
        common_fixes = cb.get("common_fixes", {})

        acceleration_ratio = trend["acceleration_ratio"]
        aircraft_count = trend["aircraft_count"]
        urgent_count = trend.get("urgent_priority_count", 0)

        score = (
            (acceleration_ratio * 20)
            + (comeback_count * 3)
            + (aircraft_count * 2)
            + (urgent_count * 5)
        )

        scored.append({
            "uns": uns,
            "score": round(score, 1),
            "total_tars": trend["total_tars"],
            "acceleration_ratio": acceleration_ratio,
            "comeback_count": comeback_count,
            "comeback_aircraft": cb_aircraft,
            "avg_gap_days": avg_gap_days,
            "aircraft_count": aircraft_count,
            "urgent_priority_count": urgent_count,
            "common_fixes": common_fixes,
            "linked_cluster": trend.get("linked_cluster", ""),
            "months_labels": trend.get("months_labels", []),
            "monthly_counts": trend.get("monthly_counts", []),
            "affected_aircraft": trend.get("affected_aircraft", []),
            "activities": trend.get("activities", []),
            "priority_breakdown": trend.get("priority_breakdown", {}),
            "sample_tars": trend.get("sample_tars", []),
            "example_comebacks": cb.get("example_comebacks", []),
            "justification": "",
        })

    scored.sort(key=lambda s: s["score"], reverse=True)

    # Generate AI justifications for top 15
    top_n = min(15, len(scored))
    print(f"  Generating TPDR justifications for top {top_n} systems...")
    for i in range(top_n):
        rec = scored[i]
        common_fixes_str = ", ".join(
            f"{k} ({v})" for k, v in rec["common_fixes"].items()
        ) or "No linked MAF data"

        prompt = (
            f"You are a V-22 maintenance analyst writing a TPDR justification.\n"
            f"System: {rec['uns']}\n"
            f"Total TARs: {rec['total_tars']} ({rec['acceleration_ratio']:.1f}x increase)\n"
            f"Comeback instances: {rec['comeback_count']} across "
            f"{rec['comeback_aircraft']} aircraft\n"
            f"Common fixes: {common_fixes_str}\n"
            f"Average gap between repeat failures: {rec['avg_gap_days']:.0f} days\n\n"
            f"Write 3-4 sentences justifying why this system needs a Technical "
            f"Publications Deficiency Report. Focus on the acceleration trend, "
            f"repeat failure pattern, fleet-wide impact, and inadequacy of "
            f"current procedures."
        )
        try:
            justification = call_ollama(CLASSIFY_MODEL, prompt)
            # Strip any think blocks that might leak through
            if "<think>" in justification:
                idx = justification.rfind("</think>")
                if idx != -1:
                    justification = justification[idx + len("</think>"):].strip()
            rec["justification"] = justification.strip()
        except Exception as e:
            rec["justification"] = f"[AI justification unavailable: {e}]"
        print(f"    [{i + 1}/{top_n}] {rec['uns'][:50]}... score={rec['score']}")

    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tpdr_analysis() -> dict:
    """Run all 3 TPDR analyses, cache results, and store in module-level var."""
    global tpdr_results

    # Check cache first
    cached = _load_cache()
    if cached is not None:
        print("  TPDR analysis loaded from cache")
        tpdr_results = cached
        return tpdr_results

    print("=" * 60)
    print("Running TPDR Recommendation Analysis...")
    print("=" * 60)
    start = time.time()

    # Analysis 1: Trends
    print("\n[1/3] System trend detection...")
    trends = _analyze_trends()

    # Analysis 2: Comebacks
    print("\n[2/3] Comeback detection...")
    comebacks = _analyze_comebacks()

    # Analysis 3: Scoring + AI justification
    print("\n[3/3] TPDR scoring and justification...")
    recommendations = _score_and_justify(trends, comebacks)

    elapsed = time.time() - start
    print(f"\nTPDR analysis complete in {elapsed:.1f}s")
    print(f"  {len(trends)} system trends analyzed")
    print(f"  {len(comebacks)} systems with comeback patterns")
    print(f"  {len(recommendations)} systems scored")

    tpdr_results = {
        "recommendations": recommendations,
        "trends": trends,
        "comebacks": comebacks,
        "computed_at": datetime.now().isoformat(),
    }

    _save_cache(tpdr_results)
    return tpdr_results


def get_tpdr_results() -> dict | None:
    """Return the current TPDR results, or None if not yet computed."""
    return tpdr_results


def get_system_detail(uns_code: str) -> dict | None:
    """Return detailed drilldown for a single UNS system from TPDR results."""
    if tpdr_results is None:
        return None

    # Search in recommendations (scored list) first
    for rec in tpdr_results.get("recommendations", []):
        if rec["uns"] == uns_code:
            return rec

    # Fallback: check trends
    for trend in tpdr_results.get("trends", []):
        if trend["uns"] == uns_code:
            # Merge with comeback data if available
            comeback = None
            for cb in tpdr_results.get("comebacks", []):
                if cb["uns"] == uns_code:
                    comeback = cb
                    break
            result = dict(trend)
            if comeback:
                result["comeback_count"] = comeback["comeback_count"]
                result["unique_aircraft"] = comeback["unique_aircraft"]
                result["avg_gap_days"] = comeback["avg_gap_days"]
                result["common_fixes"] = comeback["common_fixes"]
                result["example_comebacks"] = comeback["example_comebacks"]
            return result

    return None
