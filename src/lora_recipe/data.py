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

PROMPT = "Extract the fields as JSON.\nAnnotation: {text}\nJSON:"
CHARS = ["POS", "PERP", "PARA", "FLAT", "PROF"]


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
