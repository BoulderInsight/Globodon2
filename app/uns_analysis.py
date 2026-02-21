"""UNS System View Analysis — groups TARs by Navy UNS codes, computes
corrective action stats, and detects sub-cluster failure modes via embeddings."""

import json
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.indexer import index, CACHE_DIR

# Module-level results cache
uns_results: dict | None = None

CACHE_FILE = CACHE_DIR / "uns_analysis.json"
CACHE_MAX_AGE_HOURS = 24


# ---------------------------------------------------------------------------
# Cache helpers (same pattern as tpdr.py)
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
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
# Corrective action aggregation
# ---------------------------------------------------------------------------

def _aggregate_corrective_actions(jcns: list[str]) -> list[dict]:
    """Collect corr_act from linked MAFs, group by normalized text, return top 10.
    
    Filters out:
    - Entries shorter than 5 characters (garbage data)
    - BCM disposition codes (Beyond Capability of Maintenance) — these indicate
      where the work was sent, not what fix was performed
    """
    # BCM patterns: "BEYOND CAPABILITY OF MAINTENANCE BCM-1", etc.
    # These are disposition codes, not corrective actions
    import re
    BCM_PATTERN = re.compile(
        r"^BEYOND\s+CAPABILITY\s+OF\s+MAINTENANCE\b|^BCM[\s\-]*\d",
        re.IGNORECASE,
    )
    # Other non-actionable patterns
    SKIP_PATTERNS = [
        re.compile(r"^N/?A$", re.IGNORECASE),
        re.compile(r"^NONE$", re.IGNORECASE),
        re.compile(r"^SEE\s+ABOVE$", re.IGNORECASE),
        re.compile(r"^SAME\s+AS\s+ABOVE$", re.IGNORECASE),
    ]
    MIN_ACTION_LENGTH = 5

    action_data: dict[str, dict] = {}  # normalized_text -> {count, manhours_sum, manhours_n}
    bcm_data: dict[str, int] = {}  # BCM disposition counts (tracked separately)

    for jcn in jcns:
        jcn_str = str(jcn).strip()
        for maf in index.maf_index.get(jcn_str, []):
            text = maf.get("corr_act", "").strip()
            if not text:
                text = maf.get("action_taken", "").strip()
            if not text:
                continue
            normalized = text.strip().upper()
            if not normalized or len(normalized) < MIN_ACTION_LENGTH:
                continue

            # Skip non-actionable patterns
            if any(p.match(normalized) for p in SKIP_PATTERNS):
                continue

            # Separate BCM disposition codes
            if BCM_PATTERN.match(normalized):
                bcm_data[normalized] = bcm_data.get(normalized, 0) + 1
                continue

            if normalized not in action_data:
                action_data[normalized] = {"count": 0, "manhours_sum": 0.0, "manhours_n": 0}
            action_data[normalized]["count"] += 1

            mh_str = maf.get("manhours", "").strip()
            if mh_str:
                try:
                    mh = float(mh_str)
                    action_data[normalized]["manhours_sum"] += mh
                    action_data[normalized]["manhours_n"] += 1
                except (ValueError, TypeError):
                    pass

    # Sort by count descending, take top 10
    sorted_actions = sorted(action_data.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    result = []
    for text, data in sorted_actions:
        avg_mh = round(data["manhours_sum"] / data["manhours_n"], 1) if data["manhours_n"] > 0 else 0.0
        result.append({
            "action": text,
            "count": data["count"],
            "avg_manhours": avg_mh,
        })
    return result


# ---------------------------------------------------------------------------
# Top parts from linked MAFs
# ---------------------------------------------------------------------------

def _aggregate_parts(jcns: list[str]) -> list[dict]:
    """Collect installed/removed part numbers from linked MAFs, return top 5."""
    part_counter: Counter = Counter()

    for jcn in jcns:
        jcn_str = str(jcn).strip()
        for maf in index.maf_index.get(jcn_str, []):
            for key in ("inst_partno", "rmvd_partno"):
                pn = maf.get(key, "").strip()
                if pn and pn.lower() not in ("", "nan", "n/a", "none"):
                    part_counter[pn] += 1

    return [{"part_number": pn, "count": c} for pn, c in part_counter.most_common(5)]


# ---------------------------------------------------------------------------
# Sub-clustering within a UNS group
# ---------------------------------------------------------------------------

def _sub_cluster(group_indices: list[int], group_df: pd.DataFrame) -> list[dict]:
    """Run KMeans sub-clustering on embeddings for a UNS group (30+ TARs).
    Returns list of failure mode dicts."""
    n = len(group_indices)
    if n < 30:
        return []

    emb = index.embeddings[group_indices]

    # Test k=2 through k=min(6, n//15)
    max_k = min(6, n // 15)
    if max_k < 2:
        return []

    best_k = 2
    best_score = -1.0
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = km.fit_predict(emb)
        # Check all clusters have at least 2 members
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(emb, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # If best silhouette < 0.10, don't sub-cluster
    if best_score < 0.10 or best_labels is None:
        return []

    # Compute centroids
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=5, max_iter=100)
    km_final.fit(emb)
    centroids = km_final.cluster_centers_
    final_labels = km_final.labels_

    # Build failure modes
    modes = []
    for cid in range(best_k):
        mask = final_labels == cid
        member_indices = np.where(mask)[0]
        count = int(mask.sum())
        if count == 0:
            continue

        # Find most central TAR (highest cosine to centroid)
        centroid = centroids[cid]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        member_emb = emb[member_indices]
        sims = member_emb @ centroid_norm
        most_central_local = member_indices[np.argmax(sims)]
        most_central_global = group_indices[most_central_local]
        label = str(index.tar_df.iloc[most_central_global].get("subject", "")).strip()

        # Sample subjects (up to 3)
        sample_local = member_indices[:3]
        sample_subjects = []
        for si in sample_local:
            gi = group_indices[si]
            subj = str(index.tar_df.iloc[gi].get("subject", "")).strip()
            if subj:
                sample_subjects.append(subj[:100])

        # Top corrective actions for this sub-cluster
        sub_jcns = []
        for si in member_indices:
            gi = group_indices[si]
            jcn = str(index.tar_df.iloc[gi].get("jcn", "")).strip()
            if jcn:
                sub_jcns.append(jcn)
        sub_actions = _aggregate_corrective_actions(sub_jcns)
        top_actions_full = sub_actions[:5]  # full objects with count + manhours

        # Silhouette score for this sub-cluster's members
        if count >= 2 and len(set(final_labels)) >= 2:
            try:
                from sklearn.metrics import silhouette_samples
                all_sil = silhouette_samples(emb, final_labels, metric="cosine")
                member_sil = float(np.mean(all_sil[member_indices]))
            except Exception:
                member_sil = 0.0
        else:
            member_sil = 0.0

        modes.append({
            "mode_id": cid,
            "count": count,
            "label": label[:120],
            "sample_subjects": sample_subjects,
            "top_actions": top_actions_full,
            "silhouette_score": round(member_sil, 3),
        })

    # Sort by count descending
    modes.sort(key=lambda m: m["count"], reverse=True)
    return modes


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _analyze_systems() -> list[dict]:
    """Compute UNS group analytics for all systems with 10+ TARs."""
    df = index.tar_df
    systems = []

    grouped = df.groupby("uns")
    total_groups = 0

    for uns_raw, group in grouped:
        uns_str = str(uns_raw).strip()
        if not uns_str or uns_str.lower() == "nan":
            continue
        if len(group) < 10:
            continue
        total_groups += 1

        # Split UNS code and name: e.g. "6322 PROPROTOR GEARBOX ASSEMBLY RH"
        parts = uns_str.split(" ", 1)
        uns_code = parts[0]
        uns_name = parts[1] if len(parts) > 1 else uns_code

        total_tars = len(group)

        # Aircraft affected (exclude I-level)
        bunos = group["buno"].dropna().unique().tolist()
        bunos = [b.strip() for b in bunos if b.strip() and "i-level" not in b.lower()]
        aircraft_affected = len(bunos)

        # Activities (max 10)
        activities = group["activity"].dropna().unique().tolist()
        activities = sorted([a.strip() for a in activities if a.strip()])[:10]

        # Date range
        dated = group[group["submit_dt"].notna()]
        if len(dated) >= 1:
            first_date = str(dated["submit_dt"].min().date())
            last_date = str(dated["submit_dt"].max().date())
        else:
            first_date = ""
            last_date = ""

        # Monthly counts (full date range)
        months_labels = []
        monthly_counts = []
        if len(dated) >= 2:
            min_dt = dated["submit_dt"].min()
            max_dt = dated["submit_dt"].max()
            resampled = dated.set_index("submit_dt").resample("MS").size()
            full_range = pd.date_range(
                start=min_dt.to_period("M").to_timestamp(),
                end=max_dt.to_period("M").to_timestamp(),
                freq="MS",
            )
            resampled = resampled.reindex(full_range, fill_value=0)
            months_labels = [d.strftime("%Y-%m") for d in resampled.index]
            monthly_counts = resampled.tolist()

        # JCNs for this group
        jcns = group["jcn"].dropna().str.strip().tolist()

        # Top corrective actions
        top_corrective_actions = _aggregate_corrective_actions(jcns)

        # Top parts
        top_parts = _aggregate_parts(jcns)

        # Sub-cluster failure modes (for groups with 30+ TARs)
        group_indices = group.index.tolist()
        # Convert dataframe indices to positional indices in tar_df
        positional_indices = [df.index.get_loc(idx) for idx in group_indices]
        failure_modes = _sub_cluster(positional_indices, group)

        # Overall sub-cluster quality
        if failure_modes:
            total_in_modes = sum(m["count"] for m in failure_modes)
            weighted_sil = sum(m["silhouette_score"] * m["count"] for m in failure_modes)
            sub_cluster_quality = round(weighted_sil / max(total_in_modes, 1), 3)
        else:
            sub_cluster_quality = None

        systems.append({
            "uns_code": uns_code,
            "uns_name": uns_name,
            "total_tars": total_tars,
            "aircraft_affected": aircraft_affected,
            "activities": activities,
            "date_range": {"first": first_date, "last": last_date},
            "monthly_counts": monthly_counts,
            "months_labels": months_labels,
            "top_corrective_actions": top_corrective_actions,
            "failure_modes": failure_modes,
            "sub_cluster_quality": sub_cluster_quality,
            "top_parts": top_parts,
        })

    # Sort by total_tars descending
    systems.sort(key=lambda s: s["total_tars"], reverse=True)
    print(f"  UNS analysis complete: {len(systems)} systems with 10+ TARs (of {total_groups})")
    return systems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_uns_analysis() -> dict:
    """Run UNS system analysis, cache results, store in module-level var."""
    global uns_results

    cached = _load_cache()
    if cached is not None:
        print("  UNS system analysis loaded from cache")
        uns_results = cached
        return uns_results

    print("=" * 60)
    print("Running UNS System Analysis...")
    print("=" * 60)
    start = time.time()

    systems = _analyze_systems()

    elapsed = time.time() - start
    print(f"\nUNS analysis complete in {elapsed:.1f}s")
    print(f"  {len(systems)} systems analyzed")
    sub_clustered = sum(1 for s in systems if s["failure_modes"])
    print(f"  {sub_clustered} systems with sub-cluster failure modes")

    uns_results = {
        "systems": systems,
        "total_systems": len(systems),
        "computed_at": datetime.now().isoformat(),
    }

    _save_cache(uns_results)
    return uns_results


def get_uns_results() -> dict | None:
    """Return the current UNS analysis results, or None if not yet computed."""
    return uns_results
