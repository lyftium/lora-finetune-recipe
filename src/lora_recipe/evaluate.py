"""Evaluation: the same held-out set for the base model and the adapter.

Two scores, because they fail differently:
- json_valid: the output parses as a JSON object (format compliance)
- exact_match: every expected field has the expected value (task correctness)
"""

from __future__ import annotations

import json

import torch

from .data import PROMPT


def _first_json(text: str) -> dict | None:
    """Parse the first JSON object in the text (nested objects and code fences are fine)."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


@torch.no_grad()
def evaluate(model, tok, rows: list[dict], max_new_tokens: int = 48, batch_size: int = 16) -> dict:
    model.eval()
    valid = exact = 0
    by_kind: dict[str, list[int]] = {}
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        prompts = [PROMPT.format(text=r["text"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for r, t in zip(chunk, texts, strict=True):
            pred, gold = _first_json(t), json.loads(r["target"])
            ok = pred is not None and all(pred.get(k) == v for k, v in gold.items())
            valid += pred is not None
            exact += ok
            by_kind.setdefault(r["kind"], []).append(int(ok))
    n = len(rows)
    return {
        "n": n,
        "json_valid": round(valid / n, 4),
        "exact_match": round(exact / n, 4),
        "exact_match_by_kind": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())},
    }
