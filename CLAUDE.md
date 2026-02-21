# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TAR-MAF Problem-to-Solution Analyzer: a Python tool that uses local LLMs (via Ollama) to mine aircraft maintenance data, discovering what problems occur and what typically fixes them.

**Data sources** (not in git, must be placed in project root):
- `TAR_Data.csv` (~15K rows) — Technical Assistance Requests (problems reported)
- `maf.csv` (~1.8M rows) — Maintenance Action Forms (corrective actions taken)
- Linked via `jcn` (Job Control Number) field; ~78% of TARs have matching MAFs

## Running

```bash
# Full pipeline (requires Ollama running locally)
python3 tar_maf_analyzer.py

# Ollama must have these models pulled:
#   ollama pull nomic-embed-text:latest    (embeddings)
#   ollama pull qwen2.5:32b               (extraction/classification)
#   ollama pull deepseek-r1:32b            (reasoning/summarization)
```

Set `SAMPLE_SIZE` in the config section (line ~39) to an integer for testing, or `None` for full run. Full run: ~13 min. Sample of 1000: ~45 min (more clusters).

## Architecture

Single-file pipeline (`tar_maf_analyzer.py`) with 5 sequential steps:

1. **Embed** — `load_tar_data()` + `embed_texts()`: Combines TAR `subject`+`issue` fields, embeds via `nomic-embed-text`. Cached in `.cache/` as `.npy` keyed by content hash.
2. **Cluster** — `cluster_embeddings()`: KMeans with automatic k selection via silhouette score (tests k in range 5-50).
3. **Classify** — `classify_cluster()`: Sends 8 sample TARs per cluster to `qwen2.5:32b` for a human-readable label.
4. **Extract** — `extract_solution()`: For each cluster, finds linked MAFs via JCN match, sends up to 15 corrective action texts to `qwen2.5:32b` for structured JSON extraction (action/component/reference).
5. **Summarize** — `summarize_cluster_insight()`: Sends each cluster's solutions to `deepseek-r1:32b` for an actionable insight paragraph.

Output: `analysis_results.json` (array of objects, one per problem cluster) + console summary.

## Key Implementation Details

- MAF file is read in 100K-row chunks (`pd.read_csv(..., chunksize=100_000)`) due to size
- `call_ollama_json()` strips deepseek-r1's `<think>...</think>` blocks before parsing JSON
- `call_ollama()` has retry logic with exponential backoff and a configurable delay between calls (`LLM_DELAY`)
- Embeddings are cached by SHA-256 hash of input texts; delete `.cache/` to force recompute

## Key Data Fields

**TAR**: `jcn`, `subject`, `issue`, `part_number`, `status`, `work_center`, `uns`
**MAF**: `jcn`, `discrepancy`, `corr_act`, `action_taken`, `inst_partno`, `rmvd_partno`, `manhours`, `wuc`

## Dependencies

Python 3.13+: `pandas`, `numpy`, `scikit-learn`, `ollama`, `hdbscan`, `umap-learn`, `tqdm`, `rich`
