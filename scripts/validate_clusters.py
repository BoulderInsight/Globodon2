#!/usr/bin/env python3
"""Cluster Quality Validation — Silhouette Analysis

Measures how well each TAR fits its assigned cluster vs. the next-closest cluster.
Produces per-cluster scores and an overall score, plus identifies the weakest clusters
and the most "confused" TARs (ones that might belong in a different cluster).

Run from the repo root: python3 scripts/validate_clusters.py
Requires: .cache/cluster_assignments.npy and .cache/embeddings_*.npy
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples

DATA_DIR = Path(__file__).parent.parent
CACHE_DIR = DATA_DIR / ".cache"


def _hash_texts(texts: list[str]) -> str:
    return hashlib.sha256("".join(texts).encode()).hexdigest()[:12]


def main():
    print("=" * 70)
    print("CLUSTER QUALITY VALIDATION — SILHOUETTE ANALYSIS")
    print("=" * 70)
    print()

    # 1. Load TAR data
    tar_file = DATA_DIR / "TAR_Data.csv"
    print(f"Loading TAR data from {tar_file.name}...")
    df = pd.read_csv(tar_file, dtype=str).fillna("")
    df["text"] = (df["subject"].str.strip() + " " + df["issue"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    print(f"  {len(df)} TARs loaded")

    # 2. Load embeddings
    texts = df["text"].tolist()
    h = _hash_texts(texts)
    emb_file = CACHE_DIR / f"embeddings_{h}.npy"
    if not emb_file.exists():
        print(f"  ERROR: No cached embeddings at {emb_file}")
        print("  Run tar_maf_analyzer.py first to generate embeddings.")
        return
    embeddings = np.load(emb_file)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    print(f"  Embeddings shape: {embeddings.shape}")

    # 3. Load cluster assignments
    assignments_file = CACHE_DIR / "cluster_assignments.npy"
    if not assignments_file.exists():
        print(f"  ERROR: No cluster assignments at {assignments_file}")
        print("  Run: python3 scripts/save_cluster_assignments.py")
        return
    assignments = np.load(assignments_file)
    print(f"  {len(assignments)} assignments loaded")

    # 4. Load cluster profiles for names
    with open(DATA_DIR / "analysis_results_cleaned.json") as f:
        profiles = json.load(f)
    cluster_names = {i: p["problem"] for i, p in enumerate(profiles)}

    # 5. Filter to assigned TARs only
    valid_mask = assignments >= 0
    valid_embeddings = embeddings[valid_mask]
    valid_assignments = assignments[valid_mask]
    valid_df = df[valid_mask].reset_index(drop=True)
    print(f"  {len(valid_embeddings)} TARs with valid cluster assignments")
    print()

    # 6. Compute silhouette scores
    # For large datasets, use a sample for the overall score (full dataset is slow)
    n = len(valid_embeddings)
    
    print("Computing silhouette scores...")
    print("  (This may take a few minutes for ~15K embeddings)")
    start = time.time()

    if n > 10000:
        # Sample for overall score (full computation is O(n^2))
        sample_size = 5000
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, sample_size, replace=False)
        sample_emb = valid_embeddings[sample_idx]
        sample_labels = valid_assignments[sample_idx]
        
        overall_score = silhouette_score(sample_emb, sample_labels, metric="cosine")
        print(f"  Overall silhouette score (sampled {sample_size}): {overall_score:.4f}")
        
        # Per-sample scores on the sample
        sample_scores = silhouette_samples(sample_emb, sample_labels, metric="cosine")
        
        # Per-cluster scores from the sample
        print()
        print("-" * 70)
        print(f"{'CLUSTER':50s} {'SIZE':>6s} {'SCORE':>7s} {'QUALITY':>10s}")
        print("-" * 70)
        
        cluster_results = []
        for ci in sorted(cluster_names.keys()):
            mask_full = valid_assignments == ci
            cluster_size = mask_full.sum()
            
            mask_sample = sample_labels == ci
            if mask_sample.sum() > 0:
                cluster_score = sample_scores[mask_sample].mean()
            else:
                cluster_score = 0.0
            
            if cluster_score >= 0.5:
                quality = "STRONG"
            elif cluster_score >= 0.25:
                quality = "MODERATE"
            elif cluster_score >= 0.0:
                quality = "WEAK"
            else:
                quality = "POOR"
            
            name = cluster_names.get(ci, f"Cluster {ci}")[:50]
            print(f"{name:50s} {cluster_size:6d} {cluster_score:7.3f} {quality:>10s}")
            cluster_results.append({
                "cluster_id": ci,
                "name": cluster_names.get(ci, f"Cluster {ci}"),
                "size": int(cluster_size),
                "silhouette_score": round(float(cluster_score), 4),
                "quality": quality,
            })
    else:
        # Full computation for smaller datasets
        overall_score = silhouette_score(valid_embeddings, valid_assignments, metric="cosine")
        print(f"  Overall silhouette score: {overall_score:.4f}")
        
        all_scores = silhouette_samples(valid_embeddings, valid_assignments, metric="cosine")
        
        print()
        print("-" * 70)
        print(f"{'CLUSTER':50s} {'SIZE':>6s} {'SCORE':>7s} {'QUALITY':>10s}")
        print("-" * 70)
        
        cluster_results = []
        for ci in sorted(cluster_names.keys()):
            mask = valid_assignments == ci
            cluster_size = mask.sum()
            cluster_score = all_scores[mask].mean() if mask.sum() > 0 else 0.0
            
            if cluster_score >= 0.5:
                quality = "STRONG"
            elif cluster_score >= 0.25:
                quality = "MODERATE"
            elif cluster_score >= 0.0:
                quality = "WEAK"
            else:
                quality = "POOR"
            
            name = cluster_names.get(ci, f"Cluster {ci}")[:50]
            print(f"{name:50s} {cluster_size:6d} {cluster_score:7.3f} {quality:>10s}")
            cluster_results.append({
                "cluster_id": ci,
                "name": cluster_names.get(ci, f"Cluster {ci}"),
                "size": int(cluster_size),
                "silhouette_score": round(float(cluster_score), 4),
                "quality": quality,
            })
        sample_scores = all_scores
        sample_labels = valid_assignments
        sample_idx = np.arange(n)

    elapsed = time.time() - start
    print("-" * 70)
    print(f"  Computed in {elapsed:.1f}s")
    print()

    # 7. Find the most confused TARs (lowest silhouette scores)
    print("=" * 70)
    print("MOST MISCLASSIFIED TARs (lowest silhouette scores)")
    print("These TARs are closer to a DIFFERENT cluster than their assigned one.")
    print("=" * 70)
    print()
    
    worst_idx = np.argsort(sample_scores)[:20]
    for rank, idx in enumerate(worst_idx):
        orig_idx = sample_idx[idx] if n > 10000 else idx
        tar_row = valid_df.iloc[orig_idx]
        assigned_cluster = valid_assignments[orig_idx]
        score = sample_scores[idx]
        jcn = tar_row.get("jcn", "?")
        subject = str(tar_row.get("subject", ""))[:80]
        cluster_name = cluster_names.get(assigned_cluster, f"Cluster {assigned_cluster}")[:40]
        print(f"  {rank+1:2d}. Score: {score:+.3f} | Cluster: {cluster_name}")
        print(f"      JCN: {jcn} | {subject}")
        print()

    # 8. Cluster confusion matrix — which clusters are most similar to each other?
    print("=" * 70)
    print("CLUSTER PROXIMITY (which clusters are most similar)")
    print("High proximity = candidates for merging or boundary refinement")
    print("=" * 70)
    print()
    
    # Compute cluster centroids from the valid embeddings
    unique_clusters = sorted(set(valid_assignments))
    centroids = {}
    for ci in unique_clusters:
        mask = valid_assignments == ci
        centroids[ci] = valid_embeddings[mask].mean(axis=0)
        # Normalize
        cn = np.linalg.norm(centroids[ci])
        if cn > 0:
            centroids[ci] /= cn
    
    # Pairwise cosine similarity between centroids
    print(f"{'CLUSTER A':40s} <-> {'CLUSTER B':40s} {'SIM':>6s}")
    print("-" * 90)
    
    proximity_pairs = []
    for i, ci in enumerate(unique_clusters):
        for j, cj in enumerate(unique_clusters):
            if j <= i:
                continue
            sim = float(np.dot(centroids[ci], centroids[cj]))
            proximity_pairs.append((ci, cj, sim))
    
    proximity_pairs.sort(key=lambda x: x[2], reverse=True)
    for ci, cj, sim in proximity_pairs[:10]:
        na = cluster_names.get(ci, f"Cluster {ci}")[:40]
        nb = cluster_names.get(cj, f"Cluster {cj}")[:40]
        print(f"{na:40s} <-> {nb:40s} {sim:6.3f}")
    
    print()

    # 9. Per-cluster: what % of TARs have negative silhouette (wrong cluster)?
    print("=" * 70)
    print("CLUSTER PURITY (% of TARs that fit well)")
    print("=" * 70)
    print()
    print(f"{'CLUSTER':50s} {'GOOD':>6s} {'MARGINAL':>9s} {'WRONG':>7s}")
    print("-" * 75)
    
    for ci in sorted(cluster_names.keys()):
        mask = sample_labels == ci
        if mask.sum() == 0:
            continue
        scores = sample_scores[mask]
        n_total = len(scores)
        n_good = (scores >= 0.25).sum()
        n_marginal = ((scores >= 0) & (scores < 0.25)).sum()
        n_wrong = (scores < 0).sum()
        
        name = cluster_names.get(ci, f"Cluster {ci}")[:50]
        print(f"{name:50s} {n_good/n_total*100:5.1f}% {n_marginal/n_total*100:8.1f}% {n_wrong/n_total*100:6.1f}%")
    
    print()
    print("-" * 75)
    all_good = (sample_scores >= 0.25).sum()
    all_marginal = ((sample_scores >= 0) & (sample_scores < 0.25)).sum()
    all_wrong = (sample_scores < 0).sum()
    n_all = len(sample_scores)
    print(f"{'OVERALL':50s} {all_good/n_all*100:5.1f}% {all_marginal/n_all*100:8.1f}% {all_wrong/n_all*100:6.1f}%")

    # 10. Save results
    results = {
        "overall_silhouette_score": round(float(overall_score), 4),
        "sample_size": len(sample_scores),
        "clusters": cluster_results,
        "interpretation": {
            "strong": "Score >= 0.50: TARs in this cluster are clearly distinct from other clusters",
            "moderate": "Score 0.25-0.50: TARs generally fit but some overlap with neighboring clusters",
            "weak": "Score 0.00-0.25: Significant overlap with other clusters, boundary is fuzzy",
            "poor": "Score < 0.00: TARs are closer to a different cluster than their assigned one",
        },
    }
    
    out_file = DATA_DIR / "cluster_validation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to {out_file}")
    print()
    
    # 11. Summary for demo
    print("=" * 70)
    print("DEMO TALKING POINTS")
    print("=" * 70)
    strong = [c for c in cluster_results if c["quality"] == "STRONG"]
    moderate = [c for c in cluster_results if c["quality"] == "MODERATE"]
    weak = [c for c in cluster_results if c["quality"] in ("WEAK", "POOR")]
    
    print(f"""
Overall cluster quality score: {overall_score:.3f} (range: -1 to 1, higher = better)

{len(strong)} STRONG clusters (score >= 0.50):
  {chr(10).join('  - ' + c['name'] + f" ({c['silhouette_score']:.3f})" for c in strong) if strong else '  (none)'}

{len(moderate)} MODERATE clusters (score 0.25-0.50):
  {chr(10).join('  - ' + c['name'] + f" ({c['silhouette_score']:.3f})" for c in moderate) if moderate else '  (none)'}

{len(weak)} WEAK clusters (score < 0.25):
  {chr(10).join('  - ' + c['name'] + f" ({c['silhouette_score']:.3f})" for c in weak) if weak else '  (none)'}

What to tell Mark:
  "We validated cluster quality using silhouette analysis. {len(strong) + len(moderate)} of 10 
  clusters show strong or moderate cohesion — the TARs in them are genuinely 
  similar to each other and distinct from other groups. {len(weak)} clusters have 
  weaker boundaries, which tells us where domain expert input would improve 
  the classification. The system includes a 'Not Related' feedback mechanism 
  so maintainers can flag mismatches, giving us training data to refine the 
  clusters over time."
""")


if __name__ == "__main__":
    main()
