# LoRA fine-tune recipe

![CI](https://github.com/lyftium/lora-finetune-recipe/actions/workflows/ci.yml/badge.svg)

Measure the base model on a held-out set, train a LoRA / QLoRA adapter, evaluate on the **same** split, and write a report next to a run record (git sha, config hash, data hashes, library versions). One CLI loop — not a notebook that worked once.

| | |
|:--|:--|
| **90 s path** | [`docs/DEMO.md`](docs/DEMO.md) · `results/qwen-0.5b-lora/` · prompted artefact [`results/prompted-baseline-qwen/eval-base-modes.json`](results/prompted-baseline-qwen/eval-base-modes.json) |
| **Stack** | PyTorch · Hugging Face Transformers · PEFT (LoRA / QLoRA) · optional `torchrun` DDP |
| **Limits** | Published adapter table still quotes the **unprompted** base. `tiny-ddp` = two-process mechanics (CPU gloo), not multi-GPU quality. QLoRA 7B config written, not run here. |

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest                                              # CPU (incl. resume)
lora-recipe run --config configs/tiny-cpu.yaml
lora-recipe run --config configs/qwen-0.5b-lora.yaml
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --adapter outputs/qwen-0.5b-lora/adapter
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
torchrun --nproc_per_node=2 -m lora_recipe.cli run --config configs/tiny-ddp.yaml --distributed --skip-baseline
```

GPU companion notebook: [LoRA Fine-Tuning: Baseline vs Adapter](https://www.kaggle.com/code/lyftium/lora-fine-tuning-baseline-vs-adapter).

## The task

Synthetic drawing annotations → JSON fields. Fixed format, closed vocabulary, exact labels — fine-tuning is a reasonable answer and scoring is unambiguous.

```
Annotation: POS|0.05|A|B         ->  {"characteristic":"POS","tolerance":0.05,"datums":["A","B"]}
Annotation: 42.50 +0.02/-0.02    ->  {"nominal":42.5,"upper":0.02,"lower":-0.02}
Annotation: 3 | PRT-417 | QTY 2  ->  {"item":3,"part_number":"PRT-417","quantity":2}
```

Train and test come from different seeds and are de-duplicated. Two scores: **json_valid** (parses) and **exact_match** (every field correct), also broken down by annotation type.

## What’s in the repo

| Path | Role |
|---|---|
| `src/lora_recipe/data.py` | synthetic task, leak-free splits |
| `src/lora_recipe/model.py` | base + LoRA (PEFT); QLoRA via bitsandbytes 4-bit NF4 |
| `src/lora_recipe/train.py` | plain loop; loss on answer tokens only; optional DDP |
| `src/lora_recipe/evaluate.py` | greedy decode; json_valid / exact_match |
| `src/lora_recipe/lineage.py` | run record |
| `configs/` | `tiny-cpu`, `tiny-ddp`, `qwen-0.5b-lora`, `qlora-7b-gpu` |
| `docs/resume.md` | `checkpoint.pt` + `--resume-from` |
| `docs/model-ingest.md` | no casual `trust_remote_code` |
| `docs/prompted-baseline.md` | same `test_data_hash` across prompt modes |
| `labs/ddp-token-weighting/` | CPU NumPy: why mean-of-means ≠ token-weighted loss |
| `tools/inspect_sft_batch.py` | SFT mask audit |

The plain loop is intentional — every step stays visible. For a bigger job I’d move to HF `Trainer` / TRL, not rewrite this.

## LoRA vs QLoRA

LoRA freezes the base and trains low-rank matrices on chosen layers (here attention projections). Only the adapter is saved. QLoRA does the same on a 4-bit NF4 base with bf16 compute; bitsandbytes needs CUDA, so on CPU the recipe refuses QLoRA instead of silently falling back. `qlora-7b-gpu` is written but **not run in this repo**.

## When I wouldn’t fine-tune

If the gap is missing knowledge (docs, standards, changing part data), start with retrieval. Fine-tuning earns its place for format / style / narrow behaviour that prompting can’t hold — and only once an evaluation set can prove it helped. That’s why the baseline is measured first.

## Results

One measured run of `configs/qwen-0.5b-lora.yaml` on CPU (8 cores, torch 2.5.1+cpu, transformers 4.46.3, peft 0.13.2). Same 100 held-out examples for both rows (`test_data_hash` `b89f85e0adae4406`), greedy decoding, 48 new tokens max. Raw reports: [`results/qwen-0.5b-lora/`](results/qwen-0.5b-lora/).

| Qwen2.5-0.5B-Instruct | json_valid | exact_match | FCF (n=30) | dimension (n=33) | BOM row (n=37) |
|---|---|---|---|---|---|
| base, no adapter | 0.66 | 0.00 | 0.00 | 0.00 | 0.00 |
| + LoRA adapter | 1.00 | 0.97 | 0.90 | 1.00 | 1.00 |

LoRA r=16 on four attention projections: 2.16 M / 496 M trainable (0.44%). 150 steps, batch 8, 559 s on CPU; train loss 2.23 → 0.0003.

How to read it:

- The unprompted base gets 0% exact match mostly because the prompt never states the schema (34% of answers don’t even parse). A fairer comparison adds schema + few-shot on the **same** `test_data_hash`: exact_match **0.0 → 0.8** ([artefact](results/prompted-baseline-qwen/eval-base-modes.json)). The published table above still quotes the unprompted base for the adapter column.
- What the run shows: narrow fixed-format task, baseline first, small adapter, report you can trace to code / config / data.
- The 3 remaining errors are all feature-control frames. Synthetic data, one seed — not a claim about real drawings.
- The adapter was trained before the first commit (`git_sha: unknown` in `train_run.json`). Both eval reports are from commit `26853b1`, same config hash (`96afd7dff9cf`).

## Distributed training

See [`docs/distributed-training.md`](docs/distributed-training.md) and [`results/tiny-ddp/`](results/tiny-ddp/). Cluster submit YAML lives in [drawing-review-service](https://github.com/lyftium/drawing-review-service) `deploy/openshift/pytorchjob-ddp.yaml`.

## Still open

- Prompted baseline vs adapter on GPU (same hash).
- Run `qlora-7b-gpu` and add its report.
- Serve the adapter with vLLM LoRA and compare served outputs to offline eval.

MIT
