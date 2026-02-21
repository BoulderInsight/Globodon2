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
                            corrective_actions: list[str]) -> dict:
    """Generate a structured JSON recommendation. Returns a dict with structured
    fields, or a dict with just 'raw_text' if JSON parsing fails."""
    top_actions = "\n".join(f"- {a}" for a in corrective_actions[:5])
    breakdown_str = ", ".join(f"{k}: {v}%" for k, v in solution_breakdown.items())

    prompt = f"""You are an expert V-22 Osprey maintenance advisor. A new TAR has come in:
"{tar_text}"

Based on similar past cases classified as "{cluster_label}", here's what we know:
- Typical resolution: {breakdown_str}
- Average manhours: {avg_manhours}
- Past corrective actions that worked:
{top_actions}

Respond with ONLY a JSON object (no markdown fences, no commentary) with these fields:
- "root_cause": 1-2 sentence root cause analysis
- "recommended_actions": array of ordered step strings the maintainer should follow
- "parts_needed": array of part numbers or materials needed
- "estimated_hours": estimated manhours as a string (e.g. "4-6" or "8")
- "action_plan": 1-2 sentence summary of the overall plan
- "references": array of any IETM references, TAR JCNs, or technical manual citations

Be specific and actionable. Reference the historical data. JSON only."""

    # Single LLM call — try JSON parse, fall back to raw text
    raw = call_ollama(CLASSIFY_MODEL, prompt)
    cleaned = strip_think_blocks(raw).strip()

    # Attempt JSON extraction (same logic as call_ollama_json)
    parsed = None
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        s = cleaned.find(start_char)
        e = cleaned.rfind(end_char)
        if s != -1 and e != -1 and e > s:
            try:
                parsed = json.loads(cleaned[s:e + 1])
                break
            except json.JSONDecodeError:
                continue

    if parsed and isinstance(parsed, dict) and "root_cause" in parsed:
        return {
            "structured": True,
            "root_cause": str(parsed.get("root_cause", "")),
            "recommended_actions": [str(a) for a in parsed.get("recommended_actions", []) if a],
            "parts_needed": [str(p) for p in parsed.get("parts_needed", []) if p],
            "estimated_hours": str(parsed.get("estimated_hours", "")),
            "action_plan": str(parsed.get("action_plan", "")),
            "references": [str(r) for r in parsed.get("references", []) if r],
        }

    # Fallback: return the raw text from the same call
    return {"structured": False, "raw_text": cleaned}
