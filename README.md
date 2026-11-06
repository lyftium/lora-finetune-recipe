# LoRA fine-tune recipe

![CI](https://github.com/lyftium/lora-finetune-recipe/actions/workflows/ci.yml/badge.svg)

Fine-tuning write-ups often compare an adapter to a model that was given no examples. I wrote a small CLI that scores the base model, trains a LoRA or QLoRA adapter, scores the same held-out set again, and stores the git commit, config hash, and data hash with the report.

On this synthetic annotation task (100 held-out examples, one data hash), few-shot prompting alone reached exact match 0.80. That is the baseline to beat. The adapter reached 0.97 under an **unprompted** evaluation, while the unprompted base was 0.00. Because the prompt condition differs from few-shot, this is not a matched +0.17 adapter gain.

PyTorch, Hugging Face, PEFT. `tiny-ddp` is two CPU processes, to see how the loop behaves — not a multi-GPU quality run. The QLoRA 7B config is written; I have not run it. Walkthrough: [`docs/DEMO.md`](docs/DEMO.md). GPU notebook: [Baseline vs Adapter](https://www.kaggle.com/code/lyftium/lora-fine-tuning-baseline-vs-adapter).

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

### Prompted baseline first (same examples and data hash)

From [`results/prompted-baseline-qwen/eval-base-modes.json`](results/prompted-baseline-qwen/eval-base-modes.json):

| Base prompt mode | json_valid | exact_match |
|---|---:|---:|
| unprompted | 0.66 | 0.00 |
| schema | 0.78 | 0.03 |
| fewshot | 1.00 | **0.80** |

Few-shot closes most of the gap without an adapter. That is the baseline to beat.

### Adapter run — different prompt mode, not a matched comparison

One measured run of `configs/qwen-0.5b-lora.yaml` on CPU (8 cores, torch 2.5.1+cpu, transformers 4.46.3, peft 0.13.2). Raw reports: [`results/qwen-0.5b-lora/`](results/qwen-0.5b-lora/).

| Qwen2.5-0.5B-Instruct | json_valid | exact_match | FCF (n=30) | dimension (n=33) | BOM row (n=37) |
|---|---|---|---|---|---|
| base, **unprompted** | 0.66 | 0.00 | 0.00 | 0.00 | 0.00 |
| + LoRA adapter | 1.00 | 0.97 | 0.90 | 1.00 | 1.00 |

LoRA r=16 on four attention projections: 2.16 M / 496 M trainable (0.44%). 150 steps, batch 8, 559 s on CPU; train loss 2.23 → 0.0003.

How to read it:

- The adapter and few-shot baseline use the same examples and data hash but **different prompt modes**. Do not quote +0.17 as a controlled adapter improvement. The metrics share a 100-item test hash and the recipe uses greedy decoding with at most 48 new tokens; that does not control for the different few-shot prompt.
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

Python 3.11+ is required. The small CPU smoke test is the first check; model training and evaluation download weights and take longer. See [`docs/DEMO.md`](docs/DEMO.md) for expected output and the published run conditions.

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest
lora-recipe run --config configs/tiny-cpu.yaml
```

Expected: tests pass and the tiny run writes its adapter and report under `outputs/`. Separately, the measured Qwen path is:

```bash
lora-recipe run --config configs/qwen-0.5b-lora.yaml
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --adapter outputs/qwen-0.5b-lora/adapter
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
```

No pretrained Qwen adapter from the 0.97 run is published, so reproducing that row requires training it again. Optional two-process CPU DDP mechanics (not a quality experiment):

```bash
torchrun --nproc_per_node=2 -m lora_recipe.cli run --config configs/tiny-ddp.yaml --distributed --skip-baseline
```

## Still open

- Re-run base and adapter under the same prompt and decoding configuration, with clean commit provenance
- Run `qlora-7b-gpu` and add its report
- Serve the adapter with vLLM LoRA and compare served outputs to offline eval

[MIT license](LICENSE) — Qwen, Hugging Face, PEFT, and other dependencies retain their respective licenses.
