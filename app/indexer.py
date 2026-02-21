"""Startup index builder — loads TAR data, embeddings, MAF index, cluster profiles."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent
CACHE_DIR = DATA_DIR / ".cache"


@dataclass
class SearchIndex:
    """In-memory search index loaded at startup."""
    # TAR data
    tar_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    tar_texts: list[str] = field(default_factory=list)

    # Embeddings (normalized)
    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))

    # Cluster profiles from analysis_results_cleaned.json
    cluster_profiles: list[dict] = field(default_factory=list)
    cluster_by_id: dict[int, dict] = field(default_factory=dict)
    cluster_by_name: dict[str, dict] = field(default_factory=dict)

    # TAR → cluster assignment (index-aligned with tar_df)
    tar_cluster_ids: np.ndarray = field(default_factory=lambda: np.array([]))

    # MAF JCN index: jcn → list of MAF record dicts
    maf_index: dict[str, list[dict]] = field(default_factory=dict)

    # Part failure data
    part_failures: list[dict] = field(default_factory=list)
    part_by_number: dict[str, dict] = field(default_factory=dict)

    # Stats
    total_maf_records: int = 0
    loaded: bool = False


# Global singleton
index = SearchIndex()


def _hash_texts(texts: list[str]) -> str:
    h = hashlib.sha256("".join(texts).encode()).hexdigest()[:12]
    return h


def _load_tar_data() -> pd.DataFrame:
    tar_file = DATA_DIR / "TAR_Data.csv"
    print(f"  Loading TAR data from {tar_file}...")
    df = pd.read_csv(tar_file, dtype=str).fillna("")
    df["text"] = (df["subject"].str.strip() + " " + df["issue"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    print(f"  {len(df)} TARs loaded")
    return df


def _load_embeddings(texts: list[str]) -> np.ndarray:
    h = _hash_texts(texts)
    cache_file = CACHE_DIR / f"embeddings_{h}.npy"
    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file.name}")
        emb = np.load(cache_file)
    else:
        raise FileNotFoundError(
            f"No cached embeddings found at {cache_file}. "
            "Run tar_maf_analyzer.py first or use scripts/build_index.py."
        )

    # Normalize for dot-product cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    print(f"  Embeddings shape: {emb.shape}, normalized")
    return emb


def _assign_clusters(tar_df: pd.DataFrame, embeddings: np.ndarray,
                     cluster_profiles: list[dict]) -> np.ndarray:
    """Assign each TAR to a cluster by re-running KMeans with k=len(profiles).

    Reproduces the original batch analysis clustering, then maps each KMeans
    label to the profile whose sample JCNs best overlap with that label's members.
    """
    from sklearn.cluster import KMeans

    n_clusters = len(cluster_profiles)
    n = len(embeddings)

    # Re-run KMeans with same parameters as tar_maf_analyzer.py
    # Use un-normalized embeddings for clustering (re-normalize after)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(embeddings)

    # Map KMeans labels → profile indices by matching sample JCNs
    profile_map: dict[int, int] = {}  # km_label → profile_index
    jcn_lower = tar_df["jcn"].str.strip().str.lower()

    for pi, profile in enumerate(cluster_profiles):
        sample_jcns = set(str(j).strip().lower() for j in profile.get("sample_tar_jcns", []))
        if not sample_jcns:
            continue
        # Find which KMeans label contains the most sample JCNs
        mask = jcn_lower.isin(sample_jcns)
        member_labels = km_labels[mask.values]
        if len(member_labels) > 0:
            from collections import Counter
            best_label = Counter(member_labels).most_common(1)[0][0]
            profile_map[int(best_label)] = pi

    # Build final assignments: TAR index → profile index
    assignments = np.full(n, -1, dtype=np.int32)
    for idx in range(n):
        label = int(km_labels[idx])
        if label in profile_map:
            assignments[idx] = profile_map[label]

    # For any unmapped KMeans labels, assign to nearest mapped centroid
    unmapped_mask = assignments == -1
    if unmapped_mask.any() and profile_map:
        mapped_centroids = []
        mapped_profiles = []
        for km_label, pi in profile_map.items():
            mapped_centroids.append(km.cluster_centers_[km_label])
            mapped_profiles.append(pi)
        centroid_matrix = np.array(mapped_centroids)
        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroid_matrix = centroid_matrix / norms

        unmapped_idxs = np.where(unmapped_mask)[0]
        sims = embeddings[unmapped_idxs] @ centroid_matrix.T
        best = np.argmax(sims, axis=1)
        for i, idx in enumerate(unmapped_idxs):
            assignments[idx] = mapped_profiles[best[i]]

    print(f"  Assigned {(assignments >= 0).sum()} / {n} TARs to clusters")
    return assignments


def _build_maf_index(tar_jcns: set[str]) -> tuple[dict[str, list[dict]], int]:
    maf_file = DATA_DIR / "maf.csv"
    print(f"  Building MAF index from {maf_file}...")
    cols = ["jcn", "discrepancy", "corr_act", "action_taken",
            "inst_partno", "rmvd_partno", "manhours", "wuc"]

    maf_idx: dict[str, list[dict]] = {}
    total = 0

    for chunk in pd.read_csv(maf_file, usecols=cols, dtype=str, chunksize=100_000):
        chunk = chunk.fillna("")
        chunk["jcn"] = chunk["jcn"].str.strip()
        matched = chunk[chunk["jcn"].isin(tar_jcns)]
        total += len(matched)
        for _, row in matched.iterrows():
            jcn = row["jcn"]
            record = {
                "corr_act": row["corr_act"].strip(),
                "action_taken": row["action_taken"].strip(),
                "manhours": row["manhours"].strip(),
                "inst_partno": row["inst_partno"].strip(),
                "rmvd_partno": row["rmvd_partno"].strip(),
                "discrepancy": row["discrepancy"].strip(),
                "wuc": row["wuc"].strip(),
            }
            if jcn not in maf_idx:
                maf_idx[jcn] = []
            maf_idx[jcn].append(record)

    print(f"  MAF index: {len(maf_idx)} JCNs, {total} total records")
    return maf_idx, total


def load_index() -> None:
    """Load all data into the global search index. Called on app startup."""
    global index
    start = time.time()
    print("=" * 60)
    print("Loading TAR Intelligence Index...")
    print("=" * 60)

    # 1. TAR data
    index.tar_df = _load_tar_data()
    index.tar_texts = index.tar_df["text"].tolist()

    # 2. Embeddings
    index.embeddings = _load_embeddings(index.tar_texts)

    # 3. Cluster profiles
    profiles_file = DATA_DIR / "analysis_results_cleaned.json"
    print(f"  Loading cluster profiles from {profiles_file.name}...")
    with open(profiles_file) as f:
        index.cluster_profiles = json.load(f)
    for i, p in enumerate(index.cluster_profiles):
        cid = p.get("cluster_id", i)
        index.cluster_by_id[cid] = p
        index.cluster_by_name[p["problem"]] = p
    print(f"  {len(index.cluster_profiles)} cluster profiles loaded")

    # 4. Part failure data
    parts_file = DATA_DIR / "part_failure_analysis.json"
    print(f"  Loading part failure data from {parts_file.name}...")
    with open(parts_file) as f:
        index.part_failures = json.load(f)
    for p in index.part_failures:
        index.part_by_number[p["part_number"]] = p
    print(f"  {len(index.part_failures)} parts tracked")

    # 5. Cluster assignments (try cached first)
    cached_assignments = CACHE_DIR / "cluster_assignments.npy"
    if cached_assignments.exists():
        assignments = np.load(cached_assignments)
        if len(assignments) == len(index.tar_df):
            print("  Loaded cached cluster assignments")
            index.tar_cluster_ids = assignments
        else:
            print(f"  Cached assignments length mismatch ({len(assignments)} vs {len(index.tar_df)}), recomputing...")
            index.tar_cluster_ids = _assign_clusters(
                index.tar_df, index.embeddings, index.cluster_profiles
            )
    else:
        print("  Recomputing cluster assignments...")
        index.tar_cluster_ids = _assign_clusters(
            index.tar_df, index.embeddings, index.cluster_profiles
        )

    # 6. MAF JCN index
    tar_jcns = set(index.tar_df["jcn"].dropna().str.strip())
    index.maf_index, index.total_maf_records = _build_maf_index(tar_jcns)

    # 7. Parse submit_date for TPDR analysis
    print("  Parsing TAR dates for TPDR analysis...")
    index.tar_df["submit_dt"] = pd.to_datetime(index.tar_df["submit_date"], errors="coerce")
    valid_dates = index.tar_df["submit_dt"].notna().sum()
    print(f"  {valid_dates} / {len(index.tar_df)} TARs have valid dates")

    index.loaded = True
    elapsed = time.time() - start
    print("=" * 60)
    print(f"Index loaded in {elapsed:.1f}s")
    print(f"  TARs: {len(index.tar_df)}")
    print(f"  MAF records indexed: {index.total_maf_records}")
    print(f"  Clusters: {len(index.cluster_profiles)}")
    print(f"  Parts tracked: {len(index.part_failures)}")
    print("=" * 60)
