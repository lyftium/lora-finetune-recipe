"""lora-recipe run --config configs/tiny-cpu.yaml

Evaluates the base model, fine-tunes a LoRA adapter, evaluates again on the same
held-out set, and writes a report plus a run record with all versions.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .config import RunConfig
from .evaluate import evaluate
from .model import load_base, load_tokenizer
from .train import train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lora-recipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--skip-baseline", action="store_true")
    e = sub.add_parser("eval", help="score the base model, or base + a saved adapter")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--adapter", type=Path, help="adapter directory saved by `run`")
    e.add_argument("--out", type=Path, help="where to write the JSON report")
    args = ap.parse_args(argv)

    cfg = RunConfig.load(args.config)
    if args.cmd == "eval":
        return _eval_only(cfg, args.adapter, args.out)
    out = Path(cfg.output_dir)
    baseline = None
    if not args.skip_baseline:
        from .data import make_splits

        _, test_rows = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
        base = load_base(cfg)
        baseline = evaluate(base, load_tokenizer(cfg), test_rows, cfg.max_new_tokens)
        del base  # free the first copy before training loads its own
        gc.collect()
    res = train(cfg, out)
    tuned = evaluate(res["model"], res["tokenizer"], res["test_rows"], cfg.max_new_tokens)
    report = {
        "baseline": baseline,
        "lora": tuned,
        "run": {
            k: res["record"][k]
            for k in (
                "git_sha",
                "config_hash",
                "data_hash",
                "test_data_hash",
                "params",
                "steps",
                "seconds",
                "loss_first",
                "loss_last",
                "libs",
            )
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def _eval_only(cfg: RunConfig, adapter: Path | None, out: Path | None) -> int:
    from .data import data_hash, make_splits
    from .lineage import git_sha

    _, test_rows = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    model = load_base(cfg)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    report = {
        "model": cfg.base_model,
        "adapter": str(adapter) if adapter else None,
        "git_sha": git_sha(),
        "config_hash": cfg.fingerprint(),
        "test_data_hash": data_hash(test_rows),
        "scores": evaluate(model, load_tokenizer(cfg), test_rows, cfg.max_new_tokens),
    }
    text = json.dumps(report, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
