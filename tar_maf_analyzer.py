#!/usr/bin/env python3
"""
TAR-MAF Problem→Solution Analyzer
Uses local LLMs (via Ollama) to mine TAR and MAF data for maintenance insights.

Models:
  - nomic-embed-text:latest  → embeddings
  - qwen2.5:32b              → extraction/classification
  - deepseek-r1:32b          → reasoning/summarization
"""

import json
import hashlib
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ollama
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
TAR_FILE = DATA_DIR / "TAR_Data.csv"
MAF_FILE = DATA_DIR / "maf.csv"
CACHE_DIR = DATA_DIR / ".cache"
OUTPUT_JSON = DATA_DIR / "analysis_results.json"

EMBED_MODEL = "nomic-embed-text:latest"
CLASSIFY_MODEL = "qwen2.5:32b"
REASON_MODEL = "deepseek-r1:32b"

EMBED_BATCH_SIZE = 50
SAMPLE_SIZE = None          # None = full run
SAMPLES_PER_CLUSTER = 8     # TARs sent to classifier per cluster
MIN_CLUSTER_SIZE = 3        # skip clusters smaller than this
LLM_DELAY = 0.5             # seconds between LLM calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / name


def hash_texts(texts: list[str]) -> str:
    h = hashlib.sha256("".join(texts).encode()).hexdigest()[:12]
    return h


def call_ollama(model: str, prompt: str, retries: int = 3) -> str:
    """Call Ollama with retries and delay."""
    for attempt in range(retries):
        try:
            resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.3})
            time.sleep(LLM_DELAY)
            return resp["response"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERROR] Ollama call failed after {retries} attempts: {e}")
                return ""
            print(f"  [RETRY {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return ""


def call_ollama_json(model: str, prompt: str) -> dict | list | None:
    """Call Ollama expecting JSON back. Parse tolerantly."""
    raw = call_ollama(model, prompt)
    # deepseek-r1 wraps reasoning in <think>...</think>
    if "<think>" in raw:
        idx = raw.rfind("</think>")
        if idx != -1:
            raw = raw[idx + len("</think>"):].strip()
    # find JSON in the response
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Step 1: Load & Embed TARs
# ---------------------------------------------------------------------------
def load_tar_data() -> pd.DataFrame:
    print("Loading TAR data...")
    df = pd.read_csv(TAR_FILE, dtype=str).fillna("")
    df["text"] = (df["subject"].str.strip() + " " + df["issue"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"  Sampling {SAMPLE_SIZE} of {len(df)} TARs")
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    print(f"  {len(df)} TARs ready for embedding")
    return df


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed texts via nomic-embed-text. Uses cache."""
    h = hash_texts(texts)
    cp = cache_path(f"embeddings_{h}.npy")
    if cp.exists():
        print("  Using cached embeddings")
        return np.load(cp)

    print(f"  Embedding {len(texts)} texts in batches of {EMBED_BATCH_SIZE}...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="  Embedding"):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            resp = ollama.embed(model=EMBED_MODEL, input=batch)
            all_embeddings.extend(resp["embeddings"])
        except Exception as e:
            print(f"  [ERROR] Embedding batch {i}: {e}")
            # zero-fill failed batch
            dim = len(all_embeddings[0]) if all_embeddings else 768
            all_embeddings.extend([np.zeros(dim).tolist()] * len(batch))
        time.sleep(0.1)

    arr = np.array(all_embeddings, dtype=np.float32)
    np.save(cp, arr)
    print(f"  Embeddings shape: {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step 2: Cluster TARs
# ---------------------------------------------------------------------------
def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """KMeans clustering with automatic k selection via silhouette."""
    print("Clustering TARs...")
    n = len(embeddings)
    k_range = range(max(5, n // 100), min(50, n // 5) + 1, 5)
    best_k, best_score = 10, -1

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(2000, n))
        if score > best_score:
            best_k, best_score = k, score

    print(f"  Best k={best_k} (silhouette={best_score:.3f})")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


# ---------------------------------------------------------------------------
# Step 3: Classify clusters with AI
# ---------------------------------------------------------------------------
def classify_cluster(samples: list[str]) -> str:
    """Send sample TARs to qwen2.5:32b for a human-readable label."""
    tar_list = "\n".join(f"- {s}" for s in samples)
    prompt = f"""Below are sample maintenance Technical Assistance Requests (TARs) from the same cluster.
What type of maintenance problem do these TARs describe?
Give ONE short label (5-10 words max), e.g. "PCA Ball Screw Wear Test Failure" or "Generator Malfunction" or "Corrosion on Nacelle".

TARs:
{tar_list}

Return ONLY the label, nothing else."""
    label = call_ollama(CLASSIFY_MODEL, prompt).strip().strip('"').strip("'")
    return label if label else "Unclassified"


# ---------------------------------------------------------------------------
# Step 4: Link to MAF solutions
# ---------------------------------------------------------------------------
def load_maf_for_jcns(jcns: set[str]) -> pd.DataFrame:
    """Load MAF rows matching the given JCNs (chunked read for large file)."""
    print(f"Loading MAF data for {len(jcns)} JCNs...")
    chunks = []
    cols = ["jcn", "discrepancy", "corr_act", "action_taken", "inst_partno",
            "rmvd_partno", "manhours", "wuc"]
    for chunk in pd.read_csv(MAF_FILE, usecols=cols, dtype=str, chunksize=100_000):
        chunk = chunk.fillna("")
        matched = chunk[chunk["jcn"].str.strip().isin(jcns)]
        if len(matched) > 0:
            chunks.append(matched)

    if not chunks:
        print("  No MAF matches found")
        return pd.DataFrame(columns=cols)

    maf = pd.concat(chunks, ignore_index=True)
    maf["jcn"] = maf["jcn"].str.strip()
    print(f"  {len(maf)} MAF records matched")
    return maf


# ---------------------------------------------------------------------------
# Step 5: Extract structured solutions with AI
# ---------------------------------------------------------------------------
def extract_solution(corr_act: str) -> dict:
    """Use qwen2.5:32b to extract structured info from corrective action text."""
    prompt = f"""Extract from this aircraft maintenance corrective action:
1) What action was taken (replaced, repaired, cleaned, adjusted, inspected, etc.)
2) What component was worked on
3) What publication or procedure was referenced (SSS, IAW, TO, NATOPS, etc.)

Corrective action text:
"{corr_act}"

Return ONLY valid JSON:
{{"action": "...", "component": "...", "reference": "..."}}"""
    result = call_ollama_json(CLASSIFY_MODEL, prompt)
    if isinstance(result, dict):
        return result
    return {"action": "unknown", "component": "unknown", "reference": ""}


# ---------------------------------------------------------------------------
# Step 6: Summarize insights with deepseek-r1
# ---------------------------------------------------------------------------
def summarize_cluster_insight(label: str, count: int, solutions: list[dict],
                               manhours: list[float]) -> str:
    """Use deepseek-r1:32b to produce actionable insights."""
    solution_summary = "\n".join(
        f"- Action: {s.get('action','?')}, Component: {s.get('component','?')}, Ref: {s.get('reference','')}"
        for s in solutions[:15]
    )
    avg_mh = np.mean(manhours) if manhours else 0

    prompt = f"""You are an expert aircraft maintenance analyst.

Problem type: {label}
Number of occurrences: {count}
Average manhours per fix: {avg_mh:.1f}

Sample solutions applied:
{solution_summary}

Given these {count} instances of "{label}" and their solutions:
1. What is the typical fix?
2. Are there patterns in how this is resolved?
3. What should a maintainer try first?
4. Any efficiency recommendations?

Write a concise, actionable paragraph (4-6 sentences) for a maintenance supervisor."""
    raw = call_ollama(REASON_MODEL, prompt)
    # Strip deepseek <think> blocks
    if "<think>" in raw:
        idx = raw.rfind("</think>")
        if idx != -1:
            raw = raw[idx + len("</think>"):].strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def compute_solution_breakdown(solutions: list[dict]) -> dict:
    """Compute percentage breakdown of action types."""
    action_counts: dict[str, int] = {}
    for s in solutions:
        action = s.get("action", "unknown").lower().strip()
        # normalize common variants
        for key in ["replace", "clean", "repair", "adjust", "inspect", "remove"]:
            if key in action:
                action = key
                break
        action_counts[action] = action_counts.get(action, 0) + 1

    total = sum(action_counts.values()) or 1
    return {k: round(v / total * 100, 1) for k, v in
            sorted(action_counts.items(), key=lambda x: -x[1])}


def run():
    print("=" * 70)
    print("TAR-MAF Problem→Solution Analyzer")
    print("=" * 70)

    # --- Step 1: Load and embed ---
    tar_df = load_tar_data()
    embeddings = embed_texts(tar_df["text"].tolist())

    # --- Step 2: Cluster ---
    labels = cluster_embeddings(embeddings)
    tar_df["cluster"] = labels

    # --- Step 3: Classify each cluster ---
    print("\nClassifying clusters with AI...")
    cluster_ids = sorted(tar_df["cluster"].unique())
    cluster_labels: dict[int, str] = {}

    for cid in tqdm(cluster_ids, desc="  Classifying"):
        members = tar_df[tar_df["cluster"] == cid]
        if len(members) < MIN_CLUSTER_SIZE:
            cluster_labels[cid] = "Too Small (skipped)"
            continue
        samples = members["text"].sample(min(SAMPLES_PER_CLUSTER, len(members)),
                                         random_state=42).tolist()
        cluster_labels[cid] = classify_cluster(samples)

    tar_df["cluster_label"] = tar_df["cluster"].map(cluster_labels)

    # Print cluster summary
    print("\n  Cluster summary:")
    for cid in cluster_ids:
        n = (tar_df["cluster"] == cid).sum()
        print(f"    [{cid:>2}] {cluster_labels[cid]:50s} ({n} TARs)")

    # --- Step 4: Link to MAF ---
    all_jcns = set(tar_df["jcn"].dropna().str.strip())
    maf_df = load_maf_for_jcns(all_jcns)

    # --- Step 5 & 6: For each cluster, extract solutions and summarize ---
    print("\nExtracting solutions and generating insights...")
    results = []
    active_clusters = [c for c in cluster_ids if cluster_labels[c] != "Too Small (skipped)"]

    for cid in tqdm(active_clusters, desc="  Analyzing"):
        label = cluster_labels[cid]
        members = tar_df[tar_df["cluster"] == cid]
        count = len(members)
        jcns = set(members["jcn"].dropna().str.strip())
        linked_mafs = maf_df[maf_df["jcn"].isin(jcns)]

        # Get corrective actions
        corr_acts = linked_mafs["corr_act"].str.strip()
        corr_acts = corr_acts[corr_acts.str.len() > 5].tolist()

        # Extract structured solutions (up to 15 per cluster)
        solutions = []
        for ca in corr_acts[:15]:
            sol = extract_solution(ca)
            solutions.append(sol)

        # Compute breakdown
        breakdown = compute_solution_breakdown(solutions)

        # Get parts involved
        parts = set()
        for col in ["inst_partno", "rmvd_partno"]:
            vals = linked_mafs[col].str.strip()
            parts.update(v for v in vals if v and v.lower() not in ("", "n/a", "none"))
        parts_list = sorted(parts)[:10]

        # Compute avg manhours
        mh_series = pd.to_numeric(linked_mafs["manhours"], errors="coerce").dropna()
        avg_manhours = round(float(mh_series.mean()), 1) if len(mh_series) > 0 else 0.0
        manhours_list = mh_series.tolist()

        # AI insight via deepseek-r1
        if solutions:
            insight = summarize_cluster_insight(label, count, solutions, manhours_list)
        else:
            insight = "No linked MAF corrective actions found for this problem cluster."

        result = {
            "problem": label,
            "cluster_id": int(cid),
            "occurrences": count,
            "linked_mafs": len(linked_mafs),
            "typical_solution": insight,
            "solution_breakdown": breakdown,
            "parts_commonly_involved": parts_list,
            "average_manhours": avg_manhours,
            "sample_tar_jcns": members["jcn"].head(5).tolist(),
            "solutions_extracted": solutions,
        }
        results.append(result)

    # --- Save results ---
    results.sort(key=lambda r: -r["occurrences"])

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_JSON}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\nPROBLEM: {r['problem']}")
        print(f"OCCURRENCES: {r['occurrences']}")
        print(f"LINKED MAFs: {r['linked_mafs']}")
        print(f"AVERAGE MANHOURS: {r['average_manhours']}")

        if r["solution_breakdown"]:
            print("SOLUTION BREAKDOWN:")
            for action, pct in r["solution_breakdown"].items():
                print(f"  - {action}: {pct}%")

        if r["parts_commonly_involved"]:
            print(f"PARTS COMMONLY INVOLVED: {', '.join(r['parts_commonly_involved'][:5])}")

        print(f"AI INSIGHT: {r['typical_solution'][:300]}")
        if len(r["typical_solution"]) > 300:
            print(f"  ... ({len(r['typical_solution'])} chars total)")
        print("-" * 70)

    print(f"\nTotal problem types discovered: {len(results)}")
    total_solutions = sum(len(r['solutions_extracted']) for r in results)
    print(f"Total solutions extracted: {total_solutions}")
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run()
