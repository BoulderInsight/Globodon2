#!/usr/bin/env python3
"""Hybrid Clustering Analysis — UNS + Embeddings

Tests a two-level approach:
  Level 1: Group by UNS code (Navy-defined system, zero AI)
  Level 2: Sub-cluster within large UNS groups using embeddings

Compares silhouette scores against the current flat KMeans approach.

Run from repo root: python3 scripts/hybrid_cluster_analysis.py
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

DATA_DIR = Path(__file__).parent.parent
CACHE_DIR = DATA_DIR / ".cache"


def _hash_texts(texts: list[str]) -> str:
    return hashlib.sha256("".join(texts).encode()).hexdigest()[:12]


def find_optimal_k(embeddings, max_k=8, min_k=2):
    """Find best k for a set of embeddings using silhouette score."""
    if len(embeddings) < min_k * 3:
        return 1  # Too few samples to cluster
    
    best_k = 1
    best_score = -1
    
    max_k = min(max_k, len(embeddings) // 3)
    
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k


def main():
    print("=" * 70)
    print("HYBRID CLUSTERING ANALYSIS — UNS + EMBEDDINGS")
    print("=" * 70)
    print()

    # 1. Load TAR data
    tar_file = DATA_DIR / "TAR_Data.csv"
    df = pd.read_csv(tar_file, dtype=str).fillna("")
    df["text"] = (df["subject"].str.strip() + " " + df["issue"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    df["uns_clean"] = df["uns"].str.strip()
    print(f"Loaded {len(df)} TARs")

    # 2. Load embeddings
    texts = df["text"].tolist()
    h = _hash_texts(texts)
    emb_file = CACHE_DIR / f"embeddings_{h}.npy"
    embeddings = np.load(emb_file)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    print(f"Embeddings: {embeddings.shape}")

    # 3. UNS Distribution Analysis
    print()
    print("=" * 70)
    print("UNS DISTRIBUTION")
    print("=" * 70)
    
    uns_counts = df["uns_clean"].value_counts()
    total_uns = len(uns_counts)
    empty_uns = uns_counts.get("", 0) + uns_counts.get("nan", 0)
    
    print(f"Total unique UNS codes: {total_uns}")
    print(f"TARs with empty/nan UNS: {empty_uns}")
    print()
    
    # Size distribution
    large = (uns_counts >= 50).sum()
    medium = ((uns_counts >= 20) & (uns_counts < 50)).sum()
    small = ((uns_counts >= 5) & (uns_counts < 20)).sum()
    tiny = (uns_counts < 5).sum()
    
    large_tars = uns_counts[uns_counts >= 50].sum()
    medium_tars = uns_counts[(uns_counts >= 20) & (uns_counts < 50)].sum()
    small_tars = uns_counts[(uns_counts >= 5) & (uns_counts < 20)].sum()
    tiny_tars = uns_counts[uns_counts < 5].sum()
    
    print(f"Large UNS groups (50+ TARs):  {large:4d} groups covering {large_tars:6d} TARs ({large_tars/len(df)*100:.1f}%)")
    print(f"Medium UNS groups (20-49):    {medium:4d} groups covering {medium_tars:6d} TARs ({medium_tars/len(df)*100:.1f}%)")
    print(f"Small UNS groups (5-19):      {small:4d} groups covering {small_tars:6d} TARs ({small_tars/len(df)*100:.1f}%)")
    print(f"Tiny UNS groups (<5):         {tiny:4d} groups covering {tiny_tars:6d} TARs ({tiny_tars/len(df)*100:.1f}%)")
    print()
    
    # Top 20 UNS groups
    print("Top 20 UNS groups by TAR count:")
    print(f"{'UNS':60s} {'COUNT':>6s} {'%':>6s}")
    print("-" * 75)
    for uns, count in uns_counts.head(20).items():
        pct = count / len(df) * 100
        print(f"{str(uns)[:60]:60s} {count:6d} {pct:5.1f}%")
    
    # 4. Test UNS-only clustering (silhouette)
    print()
    print("=" * 70)
    print("UNS-ONLY CLUSTERING QUALITY")
    print("(Using UNS code as the cluster label)")
    print("=" * 70)
    
    # Filter to UNS groups with enough members for silhouette
    valid_uns = set(uns_counts[uns_counts >= 5].index) - {"", "nan"}
    uns_mask = df["uns_clean"].isin(valid_uns)
    uns_labels_full = df.loc[uns_mask, "uns_clean"].values
    uns_embeddings = embeddings[uns_mask.values]
    
    # Need numeric labels for silhouette
    uns_to_int = {u: i for i, u in enumerate(sorted(set(uns_labels_full)))}
    uns_labels_numeric = np.array([uns_to_int[u] for u in uns_labels_full])
    
    # Sample for speed
    n = len(uns_embeddings)
    sample_size = min(5000, n)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n, sample_size, replace=False)
    
    uns_sil = silhouette_score(
        uns_embeddings[sample_idx], 
        uns_labels_numeric[sample_idx], 
        metric="cosine"
    )
    print(f"UNS-based silhouette score (sampled {sample_size}): {uns_sil:.4f}")
    print(f"(Compare to flat KMeans-10: 0.0955)")
    print()
    
    # 5. Per-UNS silhouette scores (for large groups)
    print("Per-UNS silhouette scores (groups with 50+ TARs):")
    print(f"{'UNS':55s} {'SIZE':>5s} {'SCORE':>7s} {'QUALITY':>9s}")
    print("-" * 80)
    
    uns_sample_scores = silhouette_samples(
        uns_embeddings[sample_idx],
        uns_labels_numeric[sample_idx],
        metric="cosine"
    )
    
    uns_results = []
    for uns_code in uns_counts.head(30).index:
        if uns_code in ("", "nan"):
            continue
        if uns_code not in uns_to_int:
            continue
        
        uns_int = uns_to_int[uns_code]
        mask = uns_labels_numeric[sample_idx] == uns_int
        if mask.sum() < 3:
            continue
        
        score = float(uns_sample_scores[mask].mean())
        size = int(uns_counts[uns_code])
        
        if score >= 0.5:
            quality = "STRONG"
        elif score >= 0.25:
            quality = "MODERATE"
        elif score >= 0.0:
            quality = "WEAK"
        else:
            quality = "POOR"
        
        print(f"{str(uns_code)[:55]:55s} {size:5d} {score:7.3f} {quality:>9s}")
        uns_results.append({
            "uns": uns_code,
            "size": size,
            "score": round(score, 4),
            "quality": quality,
        })
    
    # 6. Hybrid approach: sub-cluster within large UNS groups
    print()
    print("=" * 70)
    print("HYBRID: SUB-CLUSTERING WITHIN LARGE UNS GROUPS")
    print("(Embedding-based failure mode detection within each system)")
    print("=" * 70)
    print()
    
    MIN_GROUP_SIZE = 50
    large_uns_codes = [u for u in uns_counts.index 
                       if u not in ("", "nan") and uns_counts[u] >= MIN_GROUP_SIZE]
    
    hybrid_results = []
    total_subclusters = 0
    
    for uns_code in large_uns_codes[:20]:  # Top 20 for analysis
        group_mask = df["uns_clean"] == uns_code
        group_idx = np.where(group_mask.values)[0]
        group_emb = embeddings[group_idx]
        group_size = len(group_idx)
        
        # Find optimal sub-clusters
        best_k = find_optimal_k(group_emb, max_k=min(6, group_size // 15))
        
        if best_k <= 1:
            print(f"{str(uns_code)[:55]:55s} ({group_size:4d} TARs) → 1 group (no meaningful sub-clusters)")
            hybrid_results.append({
                "uns": uns_code,
                "size": group_size,
                "sub_clusters": 1,
                "sub_sil": None,
                "top_terms": [],
            })
            total_subclusters += 1
            continue
        
        km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
        sub_labels = km.fit_predict(group_emb)
        sub_sil = silhouette_score(group_emb, sub_labels, metric="cosine")
        
        # Get representative text for each sub-cluster
        group_df = df.iloc[group_idx]
        sub_summaries = []
        for si in range(best_k):
            sub_mask = sub_labels == si
            sub_count = sub_mask.sum()
            # Get 3 most central TARs (closest to centroid)
            centroid = km.cluster_centers_[si]
            dists = np.dot(group_emb[sub_mask], centroid)
            top_idx = np.argsort(dists)[-3:]
            sub_texts = group_df.iloc[np.where(sub_mask)[0][top_idx]]["subject"].tolist()
            sub_summaries.append({
                "count": int(sub_count),
                "sample_subjects": [str(t)[:80] for t in sub_texts],
            })
        
        total_subclusters += best_k
        
        print(f"{str(uns_code)[:55]:55s} ({group_size:4d} TARs) → {best_k} sub-clusters (sil: {sub_sil:.3f})")
        for si, summary in enumerate(sub_summaries):
            sample = summary["sample_subjects"][0] if summary["sample_subjects"] else "?"
            print(f"    Sub-{si+1}: {summary['count']:4d} TARs — {sample}")
        print()
        
        hybrid_results.append({
            "uns": uns_code,
            "size": group_size,
            "sub_clusters": best_k,
            "sub_sil": round(float(sub_sil), 4),
            "sub_details": sub_summaries,
        })
    
    # 7. Compute hybrid silhouette (UNS + sub-cluster as combined label)
    print()
    print("=" * 70)
    print("HYBRID CLUSTERING QUALITY COMPARISON")
    print("=" * 70)
    print()
    
    # Build hybrid labels: UNS_subcluster
    hybrid_labels = np.full(len(df), -1, dtype=np.int32)
    label_counter = 0
    
    for uns_code in uns_counts.index:
        if uns_code in ("", "nan"):
            continue
        group_mask = df["uns_clean"] == uns_code
        group_idx = np.where(group_mask.values)[0]
        group_size = len(group_idx)
        
        if group_size < 5:
            # Too small, assign to a catch-all
            hybrid_labels[group_idx] = label_counter
            continue
        
        if group_size < MIN_GROUP_SIZE:
            # Small group, single cluster
            hybrid_labels[group_idx] = label_counter
            label_counter += 1
            continue
        
        # Large group: sub-cluster
        group_emb = embeddings[group_idx]
        best_k = find_optimal_k(group_emb, max_k=min(6, group_size // 15))
        
        if best_k <= 1:
            hybrid_labels[group_idx] = label_counter
            label_counter += 1
        else:
            km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
            sub_labels = km.fit_predict(group_emb)
            for si in range(best_k):
                sub_mask = sub_labels == si
                hybrid_labels[group_idx[sub_mask]] = label_counter
                label_counter += 1
    
    # Score the hybrid approach
    valid_hybrid = hybrid_labels >= 0
    if valid_hybrid.sum() > 100:
        h_emb = embeddings[valid_hybrid]
        h_labels = hybrid_labels[valid_hybrid]
        
        # Need at least 2 unique labels
        if len(set(h_labels)) >= 2:
            h_sample_size = min(5000, len(h_emb))
            h_sample_idx = rng.choice(len(h_emb), h_sample_size, replace=False)
            
            hybrid_sil = silhouette_score(
                h_emb[h_sample_idx],
                h_labels[h_sample_idx],
                metric="cosine"
            )
        else:
            hybrid_sil = 0.0
    else:
        hybrid_sil = 0.0
    
    print(f"Approach                      Silhouette Score   Clusters")
    print(f"-" * 65)
    print(f"Current (flat KMeans-10)      {0.0955:>12.4f}         10")
    print(f"UNS-only grouping             {uns_sil:>12.4f}        {len(valid_uns)}")
    print(f"Hybrid (UNS + sub-clusters)   {hybrid_sil:>12.4f}        {label_counter}")
    print()
    
    improvement = ((hybrid_sil - 0.0955) / 0.0955 * 100) if hybrid_sil > 0 else 0
    print(f"Improvement over flat KMeans: {improvement:+.0f}%")
    print(f"Total hybrid clusters: {label_counter} (vs 10 flat)")
    
    # 8. Save results
    results = {
        "flat_kmeans_score": 0.0955,
        "uns_only_score": round(float(uns_sil), 4),
        "hybrid_score": round(float(hybrid_sil), 4),
        "total_uns_groups": total_uns,
        "total_hybrid_clusters": label_counter,
        "uns_distribution": {
            "large_50plus": {"groups": int(large), "tars": int(large_tars)},
            "medium_20_49": {"groups": int(medium), "tars": int(medium_tars)},
            "small_5_19": {"groups": int(small), "tars": int(small_tars)},
            "tiny_under_5": {"groups": int(tiny), "tars": int(tiny_tars)},
        },
        "per_uns_scores": uns_results,
        "hybrid_sub_clusters": hybrid_results,
    }
    
    out_file = DATA_DIR / "hybrid_cluster_analysis.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    
    # 9. Recommendations
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
CURRENT STATE:
  Flat KMeans with 10 clusters scores 0.095 — clusters overlap heavily
  because general maintenance vocabulary dominates the embeddings.

HYBRID APPROACH SCORES: {hybrid_sil:.4f}
  UNS gives structural grouping (Navy-defined, zero AI error).
  Sub-clustering within UNS groups finds failure modes where embeddings 
  actually have discriminating power.

FOR THE DEMO:
  Fleet Analytics should show top UNS system groups instead of (or 
  alongside) the current 10 clusters. Each system group shows:
    - TAR count and trend
    - Top corrective actions from linked MAFs
    - Sub-cluster failure modes (if the system is large enough)
    - Parts commonly involved
  
  This is more defensible because:
    1. Primary grouping is a Navy-defined code, not AI
    2. Sub-clusters are validated with silhouette scores
    3. The "Not Related" feedback improves sub-clusters, not the UNS grouping
    4. Anyone can verify: "Is this TAR about the proprotor gearbox? 
       Yes, UNS 6322 says so."

MIGRATION PATH:
  1. Add UNS-based system view to Fleet Analytics (new or alongside clusters)
  2. Keep current clusters for now as "AI-detected problem categories"  
  3. Run this hybrid analysis on the full dataset
  4. Replace flat clusters with hybrid clusters once validated
""")


if __name__ == "__main__":
    main()
