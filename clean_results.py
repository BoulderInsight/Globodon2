#!/usr/bin/env python3
"""Clean analysis_results.json by filtering garbage data."""

import json
import re

INPUT = "/Users/chriscox/_LOCAL/n8n/Globodon2/analysis_results.json"
OUTPUT = "/Users/chriscox/_LOCAL/n8n/Globodon2/analysis_results_cleaned.json"

with open(INPUT, "r") as f:
    data = json.load(f)

for cluster in data:
    # 1. Filter garbage part numbers from parts_commonly_involved
    #    Remove if: all zeros, less than 6 chars, or digits-only (no letters)
    cleaned_parts = []
    for part in cluster.get("parts_commonly_involved", []):
        # Skip all-zeros entries (e.g. "000", "0000000000")
        if re.fullmatch(r"0+", part):
            continue
        # Skip if less than 6 characters
        if len(part) < 6:
            continue
        # Skip if digits-only (no letters at all)
        if re.fullmatch(r"\d+", part):
            continue
        cleaned_parts.append(part)
    cluster["parts_commonly_involved"] = cleaned_parts

    # 2. Filter empty strings from sample_tar_jcns
    cluster["sample_tar_jcns"] = [
        jcn for jcn in cluster.get("sample_tar_jcns", []) if jcn != ""
    ]

    # 3. Filter empty solutions where both action AND component are empty
    cluster["solutions_extracted"] = [
        sol for sol in cluster.get("solutions_extracted", [])
        if not (sol.get("action", "") == "" and sol.get("component", "") == "")
    ]

    # 4. Remove empty string keys from solution_breakdown dicts
    breakdown = cluster.get("solution_breakdown", {})
    cluster["solution_breakdown"] = {
        k: v for k, v in breakdown.items() if k != ""
    }

with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)

print(f"Cleaned data written to {OUTPUT}")

# Quick summary of changes
with open(INPUT, "r") as f:
    original = json.load(f)

for i, (orig, cleaned) in enumerate(zip(original, data)):
    name = orig["problem"]
    parts_removed = len(orig["parts_commonly_involved"]) - len(cleaned["parts_commonly_involved"])
    jcns_removed = len(orig["sample_tar_jcns"]) - len(cleaned["sample_tar_jcns"])
    sols_removed = len(orig["solutions_extracted"]) - len(cleaned["solutions_extracted"])
    breakdown_keys_removed = len(orig["solution_breakdown"]) - len(cleaned["solution_breakdown"])
    if parts_removed or jcns_removed or sols_removed or breakdown_keys_removed:
        print(f"\n  [{name}]")
        if parts_removed:
            print(f"    Parts filtered: {parts_removed}")
        if jcns_removed:
            print(f"    Empty JCNs removed: {jcns_removed}")
        if sols_removed:
            print(f"    Empty solutions removed: {sols_removed}")
        if breakdown_keys_removed:
            print(f"    Empty breakdown keys removed: {breakdown_keys_removed}")
