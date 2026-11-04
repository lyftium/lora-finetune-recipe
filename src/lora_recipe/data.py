"""A small synthetic task: turn the text of a drawing annotation into JSON fields.

It is the kind of narrow, well-specified behaviour where fine-tuning can make sense
(fixed output format, closed vocabulary), and it can be generated with exact labels,
so the evaluation is unambiguous. All examples are made up.

Splits are generated from separate seeds and then de-duplicated against each other,
so no test prompt appears in training.
"""

from __future__ import annotations

import hashlib
import json
import random

# Unprompted: format hint only — no schema, no examples (published baseline).
PROMPT_UNPROMPTED = "Extract the fields as JSON.\nAnnotation: {text}\nJSON:"
PROMPT = PROMPT_UNPROMPTED  # backward-compatible alias

# Schema: field names and types shown; still no worked examples.
PROMPT_SCHEMA = """Extract annotation fields as a JSON object.
Allowed shapes (pick the one that matches the annotation):
- fcf: {{"characteristic": str, "tolerance": float, "datums": [str]}}
- dimension: {{"nominal": float, "upper": float, "lower": float}}
- bom_row: {{"item": int, "part_number": str, "quantity": int}}
Annotation: {text}
JSON:"""

# Few-shot: one fixed example per kind, then the query (same held-out set).
PROMPT_FEWSHOT = """Extract annotation fields as a JSON object.
Examples:
Annotation: POS|0.05|A|B
JSON: {{"characteristic":"POS","tolerance":0.05,"datums":["A","B"]}}
Annotation: 42.50 +0.02/-0.02
JSON: {{"nominal":42.5,"upper":0.02,"lower":-0.02}}
Annotation: 3 | PRT-417 | QTY 2
JSON: {{"item":3,"part_number":"PRT-417","quantity":2}}
Annotation: {text}
JSON:"""

PROMPTS = {
    "unprompted": PROMPT_UNPROMPTED,
    "schema": PROMPT_SCHEMA,
    "fewshot": PROMPT_FEWSHOT,
}

CHARS = ["POS", "PERP", "PARA", "FLAT", "PROF"]


def render_prompt(text: str, mode: str = "unprompted") -> str:
    if mode not in PROMPTS:
        raise ValueError(f"unknown prompt mode {mode!r}; expected one of {sorted(PROMPTS)}")
    return PROMPTS[mode].format(text=text)


def _example(rng: random.Random) -> dict:
    kind = rng.choice(["fcf", "dimension", "bom_row"])
    if kind == "fcf":
        char = rng.choice(CHARS)
        tol = rng.choice([0.02, 0.05, 0.1, 0.2, 0.25])
        refs = [] if char == "FLAT" else rng.sample("ABC", k=rng.randint(1, 2))
        text = "|".join([char, f"{tol:.2f}", *refs])
        fields = {"characteristic": char, "tolerance": tol, "datums": refs}
    elif kind == "dimension":
        nom = round(rng.uniform(4, 120), 2)
        dev = rng.choice([0.01, 0.02, 0.05, 0.1])
        text = f"{nom:.2f} +{dev:.2f}/-{dev:.2f}"
        fields = {"nominal": nom, "upper": dev, "lower": -dev}
    else:
        item, qty = rng.randint(1, 9), rng.randint(1, 8)
        pn = f"PRT-{rng.randint(100, 999)}"
        text = f"{item} | {pn} | QTY {qty}"
        fields = {"item": item, "part_number": pn, "quantity": qty}
    return {"kind": kind, "text": text, "target": json.dumps(fields, separators=(",", ":"))}


def make_split(n: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    exclude = exclude or set()
    out, seen = [], set(exclude)
    while len(out) < n:
        ex = _example(rng)
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        out.append(ex)
    return out


def make_splits(n_train: int, n_test: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    test = make_split(n_test, seed + 1000)
    train = make_split(n_train, seed, exclude={e["text"] for e in test})
    return train, test


def data_hash(rows: list[dict]) -> str:
    blob = json.dumps(rows, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
