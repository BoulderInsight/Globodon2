"""Ollama LLM helpers — adapted from tar_maf_analyzer.py."""

import re
import json
import time

import numpy as np
import ollama as ollama_client

EMBED_MODEL = "nomic-embed-text:latest"
CLASSIFY_MODEL = "qwen2.5:32b"
LLM_DELAY = 0.3


def call_ollama(model: str, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = ollama_client.generate(model=model, prompt=prompt, options={"temperature": 0.3})
            time.sleep(LLM_DELAY)
            return resp["response"]
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Ollama call failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)
    return ""


def strip_think_blocks(text: str) -> str:
    if "<think>" in text:
        idx = text.rfind("</think>")
        if idx != -1:
            return text[idx + len("</think>"):].strip()
    return text


def call_ollama_json(model: str, prompt: str) -> dict | list | None:
    raw = call_ollama(model, prompt)
    raw = strip_think_blocks(raw)
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


def embed_text(text: str) -> np.ndarray:
    resp = ollama_client.embed(model=EMBED_MODEL, input=[text])
    vec = np.array(resp["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def generate_recommendation(tar_text: str, cluster_label: str,
                            solution_breakdown: dict, avg_manhours: float,
                            corrective_actions: list[str]) -> str:
    top_actions = "\n".join(f"- {a}" for a in corrective_actions[:5])
    breakdown_str = ", ".join(f"{k}: {v}%" for k, v in solution_breakdown.items())

    prompt = f"""You are an expert V-22 Osprey maintenance advisor. A new TAR has come in:
"{tar_text}"

Based on similar past cases classified as "{cluster_label}", here's what we know:
- Typical resolution: {breakdown_str}
- Average manhours: {avg_manhours}
- Past corrective actions that worked:
{top_actions}

Given this specific TAR and the historical data:
1. What is the most likely root cause?
2. What should the maintainer try first?
3. What parts should they have on hand?
4. Estimated time to resolution?

Be specific and actionable. Reference the historical data."""

    raw = call_ollama(CLASSIFY_MODEL, prompt)
    return strip_think_blocks(raw).strip()
