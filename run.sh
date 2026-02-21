#!/bin/bash
cd "$(dirname "$0")"
echo "Starting TAR Intelligence System..."
echo "Make sure Ollama is running with: nomic-embed-text, qwen2.5:32b"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
