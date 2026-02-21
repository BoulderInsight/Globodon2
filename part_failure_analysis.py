#!/usr/bin/env python3
"""Task 2: Part-level failure analysis — which parts cause the most problems?"""
import json
import time
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import ollama

DATA_DIR = Path(__file__).parent
MAF_FILE = DATA_DIR / "maf.csv"
OUTPUT = DATA_DIR / "part_failure_analysis.json"

CLASSIFY_MODEL = "qwen2.5:32b"
LLM_DELAY = 0.5
TOP_N = 20


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


def run():
    print("=" * 70)
    print("Part-Level Failure Analysis")
    print("=" * 70)

    # Step 1: Load MAF data in chunks, filter to replacements
    print("\nLoading MAF data (chunked, filtering to action_taken='R')...")
    cols = ["jcn", "discrepancy", "corr_act", "action_taken", "rmvd_partno", "manhours", "wuc"]
    replacement_chunks = []

    for i, chunk in enumerate(pd.read_csv(MAF_FILE, usecols=cols, dtype=str, chunksize=100_000)):
        chunk = chunk.fillna("")
        # Filter to replacements only
        replacements = chunk[chunk["action_taken"].str.strip().str.upper() == "R"]
        if len(replacements) > 0:
            replacement_chunks.append(replacements)
        if (i + 1) % 5 == 0:
            print(f"  Processed {(i+1) * 100_000:,} rows...")

    if not replacement_chunks:
        print("No replacement records found!")
        return

    maf_replacements = pd.concat(replacement_chunks, ignore_index=True)
    maf_replacements["rmvd_partno"] = maf_replacements["rmvd_partno"].str.strip()
    maf_replacements["manhours"] = pd.to_numeric(maf_replacements["manhours"], errors="coerce")
    print(f"  Total replacement records: {len(maf_replacements):,}")

    # Step 2: Filter out garbage part numbers
    valid_mask = (
        (maf_replacements["rmvd_partno"].str.len() >= 6) &
        (~maf_replacements["rmvd_partno"].str.fullmatch(r"0+")) &
        (~maf_replacements["rmvd_partno"].str.fullmatch(r"\d+")) &
        (maf_replacements["rmvd_partno"] != "")
    )
    maf_valid = maf_replacements[valid_mask].copy()
    print(f"  Valid part replacements: {len(maf_valid):,}")

    # Step 3: Group by removed part number
    grouped = maf_valid.groupby("rmvd_partno")

    # Filter to parts with >10 occurrences
    part_stats = []
    for pn, group in grouped:
        if len(group) <= 10:
            continue
        avg_mh = group["manhours"].mean()
        wuc_counts = Counter(group["wuc"].str.strip())
        top_wucs = [w for w, _ in wuc_counts.most_common(5) if w]
        discrepancies = group["discrepancy"].str.strip()
        sample_disc = [d for d in discrepancies if len(d) > 10][:5]

        part_stats.append({
            "part_number": pn,
            "failure_count": len(group),
            "avg_manhours": round(float(avg_mh) if pd.notna(avg_mh) else 0, 1),
            "common_wucs": top_wucs,
            "sample_discrepancies": sample_disc,
        })

    # Sort by failure count descending
    part_stats.sort(key=lambda x: -x["failure_count"])
    print(f"  Parts with >10 failures: {len(part_stats)}")

    # Step 4: For top 20, get AI summary
    top_parts = part_stats[:TOP_N]
    print(f"\nGenerating AI summaries for top {len(top_parts)} failing parts...")

    for i, part in enumerate(top_parts):
        pn = part["part_number"]
        disc_text = "\n".join(f"- {d}" for d in part["sample_discrepancies"])
        wuc_text = ", ".join(part["common_wucs"][:3]) if part["common_wucs"] else "N/A"

        prompt = f"""You are an aircraft maintenance analyst.

Part number: {pn}
Total failures (replacements): {part['failure_count']}
Average manhours per replacement: {part['avg_manhours']}
Common work unit codes: {wuc_text}

Sample discrepancy descriptions:
{disc_text}

Given these {part['failure_count']} failure instances of part {pn}:
1. What pattern do you see?
2. What typically goes wrong with this part?
3. Any recommendations?

Write 2-3 concise sentences for a maintenance supervisor."""

        print(f"  [{i+1}/{len(top_parts)}] Analyzing {pn} ({part['failure_count']} failures)...")
        summary = call_ollama(CLASSIFY_MODEL, prompt)

        # Strip any think blocks just in case
        if "<think>" in summary:
            idx = summary.rfind("</think>")
            if idx != -1:
                summary = summary[idx + len("</think>"):].strip()

        part["ai_summary"] = summary.strip()

    # Save results
    with open(OUTPUT, "w") as f:
        json.dump(top_parts, f, indent=2)

    print(f"\nResults saved to {OUTPUT}")
    print(f"\nTop 5 failing parts:")
    for p in top_parts[:5]:
        print(f"  {p['part_number']}: {p['failure_count']} failures, {p['avg_manhours']} avg hrs")


if __name__ == "__main__":
    run()
