# LoRA fine-tune recipe

![CI](https://github.com/lyftium/lora-finetune-recipe/actions/workflows/ci.yml/badge.svg)

A small recipe that refuses to skip the baseline.

One CLI: score the base on a held-out set, train a LoRA / QLoRA adapter, score again on the **same** split, write a report next to git / config / data hashes. The awkward finding is in the first results table — few-shot prompting alone gets exact_match to **0.8** on that hash, so an adapter compared only to an unprompted base looks more heroic than it is.

[`docs/DEMO.md`](docs/DEMO.md) · [`results/qwen-0.5b-lora/`](results/qwen-0.5b-lora/) · [`eval-base-modes.json`](results/prompted-baseline-qwen/eval-base-modes.json)

PyTorch, Hugging Face, PEFT. `tiny-ddp` is two CPU processes learning the mechanics, not multi-GPU quality. QLoRA 7B config is written; I have not run it here. GPU notebook: [Baseline vs Adapter](https://www.kaggle.com/code/lyftium/lora-fine-tuning-baseline-vs-adapter).

[Task](#the-task) · [Results](#results) · [Layout](#layout) · [Run](#run) · [Still open](#still-open)

## The task

Synthetic drawing annotations → JSON fields. Fixed format, closed vocabulary, exact labels — scoring is unambiguous.

```
Annotation: POS|0.05|A|B         ->  {"characteristic":"POS","tolerance":0.05,"datums":["A","B"]}
Annotation: 42.50 +0.02/-0.02    ->  {"nominal":42.5,"upper":0.02,"lower":-0.02}
Annotation: 3 | PRT-417 | QTY 2  ->  {"item":3,"part_number":"PRT-417","quantity":2}
```

Train and test come from different seeds and are de-duplicated. Scores: **json_valid** and **exact_match** (also by annotation type).

If the gap is missing knowledge, start with retrieval. Fine-tuning fits format / style / narrow behaviour that prompting can’t hold — once an evaluation set can prove it helped.

## Results

Same 100 held-out examples (`test_data_hash` `b89f85e0adae4406`), greedy decoding, 48 new tokens max.

### Fair glance — prompted base (same hash)

From [`results/prompted-baseline-qwen/eval-base-modes.json`](results/prompted-baseline-qwen/eval-base-modes.json):

| Base prompt mode | json_valid | exact_match |
|---|---:|---:|
| unprompted | 0.66 | 0.00 |
| schema | 0.78 | 0.03 |
| fewshot | 1.00 | **0.80** |

Few-shot closes most of the gap without an adapter. That is the baseline to beat.

### Adapter run (still vs unprompted base)

One measured run of `configs/qwen-0.5b-lora.yaml` on CPU (8 cores, torch 2.5.1+cpu, transformers 4.46.3, peft 0.13.2). Raw reports: [`results/qwen-0.5b-lora/`](results/qwen-0.5b-lora/).

| Qwen2.5-0.5B-Instruct | json_valid | exact_match | FCF (n=30) | dimension (n=33) | BOM row (n=37) |
|---|---|---|---|---|---|
| base, **unprompted** | 0.66 | 0.00 | 0.00 | 0.00 | 0.00 |
| + LoRA adapter | 1.00 | 0.97 | 0.90 | 1.00 | 1.00 |

LoRA r=16 on four attention projections: 2.16 M / 496 M trainable (0.44%). 150 steps, batch 8, 559 s on CPU; train loss 2.23 → 0.0003.

How to read it:

- Do **not** quote 0.00 → 0.97 as “fine-tuning alone.” The fair prompted baseline is **0.0 → 0.8** fewshot on the same hash; the table above still uses the unprompted base next to the adapter (historical framing).
- What the run shows: narrow fixed-format task, measured baseline, small adapter, report you can trace to code / config / data.
- Three remaining adapter errors are feature-control frames. Synthetic data, one seed — not real drawings.
- Adapter trained before first commit (`git_sha: unknown` in `train_run.json`). Eval reports from commit `26853b1`, config hash `96afd7dff9cf`.

## Layout

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

Plain loop on purpose — every step stays visible. For a bigger job I’d move to HF `Trainer` / TRL.

**LoRA vs QLoRA.** LoRA freezes the base and trains low-rank matrices on attention projections; only the adapter is saved. QLoRA uses a 4-bit NF4 base with bf16 compute; on CPU the recipe refuses QLoRA instead of silently falling back. `qlora-7b-gpu` is written but not run here.

**Distributed.** [`docs/distributed-training.md`](docs/distributed-training.md) · [`results/tiny-ddp/`](results/tiny-ddp/). Cluster submit YAML: [drawing-review-service](https://github.com/lyftium/drawing-review-service) `deploy/openshift/pytorchjob-ddp.yaml`.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest
lora-recipe run --config configs/tiny-cpu.yaml
lora-recipe run --config configs/qwen-0.5b-lora.yaml
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --adapter outputs/qwen-0.5b-lora/adapter
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
torchrun --nproc_per_node=2 -m lora_recipe.cli run --config configs/tiny-ddp.yaml --distributed --skip-baseline
```

## Still open

- Prompted baseline vs adapter on GPU (same hash)
- Run `qlora-7b-gpu` and add its report
- Serve the adapter with vLLM LoRA and compare served outputs to offline eval

MIT
