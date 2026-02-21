#!/usr/bin/env python3
"""Task 1: Clean analysis_results.json — filter garbage parts, empty JCNs, empty solutions."""
import json
import re
from pathlib import Path

INPUT = Path(__file__).parent / "analysis_results.json"
OUTPUT = Path(__file__).parent / "analysis_results_cleaned.json"


def is_garbage_part(pn: str) -> bool:
    """Return True if part number should be filtered out."""
    pn = pn.strip()
    if not pn:
        return True
    # All zeros (e.g. "000", "0000", "00000")
    if re.fullmatch(r"0+", pn):
        return True
    # Less than 6 characters
    if len(pn) < 6:
        return True
    # Only digits, no letters (e.g. "12345", "12977100")
    if pn.isdigit():
        return True
    return False


def clean():
    with open(INPUT) as f:
        data = json.load(f)

    stats = {"parts_removed": 0, "jcns_removed": 0, "solutions_removed": 0}

    for entry in data:
        # 1. Filter garbage part numbers
        original_parts = entry.get("parts_commonly_involved", [])
        cleaned_parts = [p for p in original_parts if not is_garbage_part(p)]
        stats["parts_removed"] += len(original_parts) - len(cleaned_parts)
        entry["parts_commonly_involved"] = cleaned_parts

        # 2. Filter empty strings from sample_tar_jcns
        original_jcns = entry.get("sample_tar_jcns", [])
        cleaned_jcns = [j for j in original_jcns if j.strip()]
        stats["jcns_removed"] += len(original_jcns) - len(cleaned_jcns)
        entry["sample_tar_jcns"] = cleaned_jcns

        # 3. Filter empty solutions (action AND component both empty)
        original_solutions = entry.get("solutions_extracted", [])
        cleaned_solutions = [
            s for s in original_solutions
            if not (s.get("action", "").strip() == "" and s.get("component", "").strip() == "")
        ]
        stats["solutions_removed"] += len(original_solutions) - len(cleaned_solutions)
        entry["solutions_extracted"] = cleaned_solutions

        # Also clean empty string keys from solution_breakdown
        breakdown = entry.get("solution_breakdown", {})
        if "" in breakdown:
            del breakdown[""]

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Cleaned results saved to {OUTPUT}")
    print(f"  Parts removed: {stats['parts_removed']}")
    print(f"  Empty JCNs removed: {stats['jcns_removed']}")
    print(f"  Empty solutions removed: {stats['solutions_removed']}")


if __name__ == "__main__":
    clean()
