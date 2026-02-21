#!/usr/bin/env python3
"""One-time script to precompute and verify the search index.

Run this before starting the web app to ensure all data is ready:
  python3 scripts/build_index.py

This verifies that cached embeddings exist and builds the MAF index.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.indexer import load_index

if __name__ == "__main__":
    print("Building/verifying TAR Intelligence search index...\n")
    try:
        load_index()
        print("\nIndex build complete. You can now start the app with ./run.sh")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run tar_maf_analyzer.py first to generate embeddings.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
