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
    r.add_argument("--out", type=Path, default=None, help="override config output_dir")
    r.add_argument("--skip-baseline", action="store_true")
    r.add_argument(
        "--distributed",
        action="store_true",
        help="use DDP (launch with torchrun; WORLD_SIZE>=2)",
    )
    r.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="path to checkpoint.pt from a prior run",
    )
    r.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override config max_steps (useful for resume demos)",
    )
    e = sub.add_parser("eval", help="score the base model, or base + a saved adapter")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--adapter", type=Path, help="adapter directory saved by `run`")
    e.add_argument("--out", type=Path, help="where to write the JSON report")
    e.add_argument(
        "--prompt-mode",
        choices=("unprompted", "schema", "fewshot"),
        default="unprompted",
        help="eval instruction: unprompted (default), schema, or fewshot",
    )
    e.add_argument(
        "--all-prompt-modes",
        action="store_true",
        help="run unprompted + schema + fewshot on the same held-out set and write a comparison",
    )
    args = ap.parse_args(argv)

    cfg = RunConfig.load(args.config)
    if args.cmd == "eval":
        return _eval_only(
            cfg,
            args.adapter,
            args.out,
            prompt_mode=args.prompt_mode,
            all_prompt_modes=args.all_prompt_modes,
        )
    out = Path(args.out) if getattr(args, "out", None) else Path(cfg.output_dir)
    cfg.output_dir = str(out)
    baseline = None
    if not args.skip_baseline:
        from .data import make_splits

        _, test_rows = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
        base = load_base(cfg)
        baseline = evaluate(base, load_tokenizer(cfg), test_rows, cfg.max_new_tokens)
        del base  # free the first copy before training loads its own
        gc.collect()
    res = train(
        cfg,
        out,
        distributed=args.distributed,
        resume_from=args.resume_from,
        max_steps=args.max_steps,
    )
    if res.get("rank", 0) != 0:
        return 0  # only rank 0 evaluates and writes the report
    tuned = evaluate(res["model"], res["tokenizer"], res["test_rows"], cfg.max_new_tokens)
    report = {
        "baseline": baseline,
        "lora": tuned,
        "run": {
            k: res["record"][k]
            for k in (
                "git_sha",
                "git_dirty",
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


def _eval_only(
    cfg: RunConfig,
    adapter: Path | None,
    out: Path | None,
    prompt_mode: str = "unprompted",
    all_prompt_modes: bool = False,
) -> int:
    from .data import data_hash, make_splits
    from .lineage import git_dirty, git_sha

    _, test_rows = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    model = load_base(cfg)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    tok = load_tokenizer(cfg)
    modes = ("unprompted", "schema", "fewshot") if all_prompt_modes else (prompt_mode,)
    by_mode = {
        m: evaluate(model, tok, test_rows, cfg.max_new_tokens, prompt_mode=m) for m in modes
    }
    report: dict = {
        "model": cfg.base_model,
        "adapter": str(adapter) if adapter else None,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": cfg.fingerprint(),
        "test_data_hash": data_hash(test_rows),
        "prompt_modes": list(modes),
    }
    if len(modes) == 1:
        report["scores"] = by_mode[modes[0]]
    else:
        report["scores_by_mode"] = by_mode
    text = json.dumps(report, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
