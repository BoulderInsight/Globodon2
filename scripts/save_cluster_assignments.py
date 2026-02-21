#!/usr/bin/env python3
"""Pre-compute cluster assignments and save to .cache/cluster_assignments.npy.

Runs the same KMeans + JCN-matching logic as indexer.py._assign_clusters()
so the web app can load assignments instantly on startup.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).parent.parent
CACHE_DIR = DATA_DIR / ".cache"


def _hash_texts(texts: list[str]) -> str:
    return hashlib.sha256("".join(texts).encode()).hexdigest()[:12]


def main():
    # 1. Load TAR data (same logic as indexer.py)
    tar_file = DATA_DIR / "TAR_Data.csv"
    print(f"Loading TAR data from {tar_file}...")
    df = pd.read_csv(tar_file, dtype=str).fillna("")
    df["text"] = (df["subject"].str.strip() + " " + df["issue"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    print(f"  {len(df)} TARs loaded")

    # 2. Load cached embeddings
    texts = df["text"].tolist()
    h = _hash_texts(texts)
    emb_file = CACHE_DIR / f"embeddings_{h}.npy"
    print(f"Loading embeddings from {emb_file.name}...")
    if not emb_file.exists():
        print(f"  ERROR: No cached embeddings at {emb_file}")
        print("  Run tar_maf_analyzer.py first to generate embeddings.")
        return
    embeddings = np.load(emb_file)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    print(f"  Embeddings shape: {embeddings.shape}")

    # 3. Load cluster profiles
    profiles_file = DATA_DIR / "analysis_results_cleaned.json"
    print(f"Loading cluster profiles from {profiles_file.name}...")
    with open(profiles_file) as f:
        cluster_profiles = json.load(f)
    print(f"  {len(cluster_profiles)} profiles")

    # 4. Run KMeans (same as indexer._assign_clusters)
    n_clusters = len(cluster_profiles)
    n = len(embeddings)
    print(f"Running KMeans with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(embeddings)

    # 5. Map KMeans labels → profile indices by matching sample JCNs
    profile_map: dict[int, int] = {}
    jcn_lower = df["jcn"].str.strip().str.lower()

    for pi, profile in enumerate(cluster_profiles):
        sample_jcns = set(str(j).strip().lower() for j in profile.get("sample_tar_jcns", []))
        if not sample_jcns:
            continue
        mask = jcn_lower.isin(sample_jcns)
        member_labels = km_labels[mask.values]
        if len(member_labels) > 0:
            best_label = Counter(member_labels).most_common(1)[0][0]
            profile_map[int(best_label)] = pi

    # 6. Build final assignments
    assignments = np.full(n, -1, dtype=np.int32)
    for idx in range(n):
        label = int(km_labels[idx])
        if label in profile_map:
            assignments[idx] = profile_map[label]

    # Handle unmapped labels
    unmapped_mask = assignments == -1
    if unmapped_mask.any() and profile_map:
        mapped_centroids = []
        mapped_profiles = []
        for km_label, pi in profile_map.items():
            mapped_centroids.append(km.cluster_centers_[km_label])
            mapped_profiles.append(pi)
        centroid_matrix = np.array(mapped_centroids)
        c_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1
        centroid_matrix = centroid_matrix / c_norms

        unmapped_idxs = np.where(unmapped_mask)[0]
        sims = embeddings[unmapped_idxs] @ centroid_matrix.T
        best = np.argmax(sims, axis=1)
        for i, idx in enumerate(unmapped_idxs):
            assignments[idx] = mapped_profiles[best[i]]

    assigned = (assignments >= 0).sum()
    print(f"  Assigned {assigned} / {n} TARs to clusters")

    # 7. Save
    out_file = CACHE_DIR / "cluster_assignments.npy"
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(out_file, assignments)
    print(f"Saved cluster assignments to {out_file}")


if __name__ == "__main__":
    main()
