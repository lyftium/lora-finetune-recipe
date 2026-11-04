# LoRA Fine-Tune Recipe
### Adapt a model, measure the baseline first, keep a lineage record

A small, reproducible LoRA / QLoRA recipe — the train → eval → record half of an
MLOps loop — rather than a notebook that worked once. One command evaluates the
base model, trains an adapter, evaluates on the **same** held-out set, and writes
a report next to a run record (git sha, config hash, data hashes, library versions).

| | |
|:--|:--|
| **Problem** | Fine-tuning is often reported without a baseline on the same split, so the adapter's contribution is unclear. |
| **Verify** | `pytest` (CPU) · `bash scripts/demo_resume.sh` · `lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes` · results in `results/qwen-0.5b-lora/` + `results/prompted-baseline-qwen/` · GPU companion: [Kaggle notebook](https://www.kaggle.com/code/lyftium/lora-fine-tuning-baseline-vs-adapter) |
| **Resume / lineage** | `checkpoint.pt` + `--resume-from` — [docs/resume.md](docs/resume.md); run record keeps git sha, config/data hashes, `resumed` |
| **Prompted baseline** | Same `test_data_hash`; Qwen base exact_match unprompted **0.0** → fewshot **0.8** ([artefact](results/prompted-baseline-qwen/eval-base-modes.json), [docs](docs/prompted-baseline.md)) |
| **CI-ready** | Manifest in [`ci/github-actions-ci.yml`](ci/github-actions-ci.yml); enable with `bash scripts/install-github-actions.sh` once `workflow` scope is granted |
| **Status & limits** | Published LoRA report still quotes the **unprompted** base. Prompted modes measured on the same `test_data_hash` under `results/prompted-baseline-*`. `tiny-ddp` proves **two-process DDP mechanics** — **not** multi-GPU quality. QLoRA 7B config written, not run here. CI written, not enabled. |
| **Next** | Real multi-GPU DDP with the same lineage record; prompted vs adapter on GPU. |

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest                                              # 16 tests, CPU (incl. resume)
lora-recipe run --config configs/tiny-cpu.yaml      # end-to-end smoke run
lora-recipe run --config configs/qwen-0.5b-lora.yaml  # a real small model, CPU or GPU
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --adapter outputs/qwen-0.5b-lora/adapter
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes --out results/prompted-baseline-qwen/eval-base-modes.json
torchrun --nproc_per_node=2 -m lora_recipe.cli run --config configs/tiny-ddp.yaml --distributed --skip-baseline
```

## At a glance

| Area | In this repo |
|:--|:--|
| Training | PyTorch, Hugging Face, PEFT (LoRA / QLoRA) |
| Eval | Baseline vs adapter on the same held-out set |
| Lineage | git sha, config hash, data hashes, library versions |
| Resume | `checkpoint.pt` (weights + optimizer + RNG); `--resume-from` |
| DDP math lab | `labs/ddp-token-weighting/` — CPU NumPy token-weighted loss (**not** NCCL) |
| SFT mask audit | `tools/inspect_sft_batch.py` + `tests/test_sft_mask.py` |
| Distributed | torchrun DDP (CPU gloo proven; CUDA nccl wired) |
| Smoke | Full CPU loop for CI in a few seconds |

## The task

A deliberately narrow synthetic task: turn the text of a drawing annotation into JSON fields.

```
Annotation: POS|0.05|A|B         ->  {"characteristic":"POS","tolerance":0.05,"datums":["A","B"]}
Annotation: 42.50 +0.02/-0.02    ->  {"nominal":42.5,"upper":0.02,"lower":-0.02}
Annotation: 3 | PRT-417 | QTY 2  ->  {"item":3,"part_number":"PRT-417","quantity":2}
```

I picked it because it is the kind of case where fine-tuning is a reasonable answer: a fixed output format and a closed vocabulary. It can also be generated with exact labels, so scoring is unambiguous. Train and test come from different seeds and are de-duplicated against each other, so no test prompt is seen in training. All the data is made up.

Two scores, because they fail differently:

- **json_valid**: the output parses as a JSON object (format compliance)
- **exact_match**: every expected field has the expected value (task correctness), also broken down per annotation type

## What's in here

| Path | What it does |
|---|---|
| `src/lora_recipe/data.py` | synthetic task, leak-free splits |
| `src/lora_recipe/model.py` | base model + LoRA (PEFT); QLoRA with a 4-bit NF4 base via bitsandbytes |
| `src/lora_recipe/train.py` | a plain training loop, loss on answer tokens only (prompt masked); optional DDP |
| `src/lora_recipe/evaluate.py` | greedy decoding, json_valid / exact_match |
| `src/lora_recipe/lineage.py` | run record: git sha, hashes, versions |
| `configs/` | `tiny-cpu` (smoke), `tiny-ddp` (multi-process), `qwen-0.5b-lora`, `qlora-7b-gpu` |
| `results/tiny-ddp/` | measured 2-process DDP run record |
| `tests/` | leak-free splits, adapters-only trainable, DDP env helpers, QLoRA refused on CPU |

The plain loop is on purpose: it keeps every step visible (masking, padding side, what gets saved). For a bigger job I'd move to the HF `Trainer` or TRL, not rewrite this.

## LoRA vs QLoRA

LoRA freezes the base weights and trains small low-rank matrices next to chosen layers (here the attention projections). Only the adapter is saved, a few MB, and one base model can serve several adapters.

QLoRA does the same on top of a base loaded in 4-bit NF4, with bf16 compute. It is what makes a 7B model trainable on a single GPU. bitsandbytes needs CUDA, so on CPU the recipe refuses QLoRA with a clear error instead of silently falling back. The `qlora-7b-gpu` config is written but **I have not run it in this repo**.

## When I wouldn't fine-tune

If the problem is missing knowledge (documents, standards, part data that changes), retrieval is usually the better first step: cheaper, easier to update, and the sources can be cited. Fine-tuning earns its place for format, style or a narrow behaviour that prompting can't hold reliably, and only once there is an evaluation set to prove it helped. That's why the baseline is always measured first.

## Results

One measured run of `configs/qwen-0.5b-lora.yaml` on a CPU-only machine (8 cores, torch 2.5.1+cpu, transformers 4.46.3, peft 0.13.2). Same 100 held-out synthetic examples for both rows (test data hash `b89f85e0adae4406`), greedy decoding, 48 new tokens max. The raw reports are in [`results/qwen-0.5b-lora/`](results/qwen-0.5b-lora/).

| Qwen2.5-0.5B-Instruct | json_valid | exact_match | FCF (n=30) | dimension (n=33) | BOM row (n=37) |
|---|---|---|---|---|---|
| base model, no adapter | 0.66 | 0.00 | 0.00 | 0.00 | 0.00 |
| + LoRA adapter | 1.00 | 0.97 | 0.90 | 1.00 | 1.00 |

LoRA r=16 on the four attention projections: 2.16M trainable parameters out of 496M (0.44%). 150 steps at batch size 8 took 559 s on CPU; training loss went from 2.23 to 0.0003.

How to read it, honestly:

- The base model gets 0% exact match mostly because the prompt never tells it the schema. It answers with JSON of its own invention (`{"fields": [{"name": ...}]}`), and 34% of its answers don't parse at all (in the ones I inspected, generation hit the 48-token limit before the JSON closed). A fair comparison against prompting would add the schema and a few examples to the prompt; I haven't run that yet, and it would likely close part of the gap.
- What the run does show is the point of the recipe: a narrow, fixed-format task, a baseline measured first on the same split, a small adapter, and a report you can trace back to code, config and data.
- The 3 remaining errors are all feature control frames. This is synthetic data on one seed, not a claim about real drawings.
- The adapter was trained before the first commit, so `train_run.json` has `git_sha: unknown`. Both eval reports were produced from commit `26853b1` with the same config hash (`96afd7dff9cf`).

## Distributed training

See [`docs/distributed-training.md`](docs/distributed-training.md) and [`results/tiny-ddp/`](results/tiny-ddp/). Cluster submit YAML lives next door in drawing-review-service `deploy/openshift/pytorchjob-ddp.yaml`.

## Not done yet

- A prompted baseline (schema + few-shot) on the same split.
- Running the QLoRA 7B config on a GPU and adding its report.
- Serving the adapter with vLLM LoRA support and checking served outputs against offline eval.

## CI

The workflow is in `ci/github-actions-ci.yml` (lint + tests on CPU). To enable it, move it to `.github/workflows/ci.yml` (needs GitHub `workflow` scope).

MIT licence.
