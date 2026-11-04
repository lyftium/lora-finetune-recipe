#!/usr/bin/env python3
"""Print tokens vs labels for a few SFT samples (assistant/answer tokens only).

Shows that prompt tokens are masked with -100 so loss is not spent on the instruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lora_recipe.config import RunConfig
from lora_recipe.data import PROMPT, make_splits
from lora_recipe.model import load_tokenizer
from lora_recipe.train import encode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/tiny-cpu.yaml"))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    cfg = RunConfig.load(args.config)
    tok = load_tokenizer(cfg)
    train, _ = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    rows = encode(tok, train[: args.n], cfg.max_len)
    report = []
    for i, (raw, enc) in enumerate(zip(train[: args.n], rows, strict=True)):
        ids, labels = enc["input_ids"], enc["labels"]
        supervised = sum(1 for x in labels if x != -100)
        masked = sum(1 for x in labels if x == -100)
        preview = []
        for tid, lab in zip(ids, labels, strict=True):
            piece = tok.decode([tid])
            preview.append({"tok": piece, "id": tid, "supervised": lab != -100})
        report.append(
            {
                "i": i,
                "text": raw["text"],
                "prompt_prefix": PROMPT.split("{text}")[0][:40],
                "n_tokens": len(ids),
                "n_supervised": supervised,
                "n_masked_prompt": masked,
                "tokens": preview,
            }
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
