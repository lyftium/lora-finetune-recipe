# lora-finetune-recipe

A small LoRA / QLoRA fine-tuning recipe that I can run, measure and reproduce, rather than a notebook that worked once.

The whole loop is one command: evaluate the base model, train a LoRA adapter, evaluate the adapter on the **same** held-out set, and write a report next to a run record (git sha, config hash, train and test data hashes, library versions, loss curve). The CPU smoke test runs the full loop in a few seconds, so it can sit in CI.

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest                                              # 12 tests, CPU, a few seconds
lora-recipe run --config configs/tiny-cpu.yaml      # end-to-end smoke run
lora-recipe run --config configs/qwen-0.5b-lora.yaml  # a real small model, CPU or GPU
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --adapter outputs/qwen-0.5b-lora/adapter
```

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
| `src/lora_recipe/train.py` | a plain training loop, loss on answer tokens only (prompt masked) |
| `src/lora_recipe/evaluate.py` | greedy decoding, json_valid / exact_match |
| `src/lora_recipe/lineage.py` | run record: git sha, hashes, versions |
| `configs/` | `tiny-cpu` (smoke test), `qwen-0.5b-lora` (CPU-able), `qlora-7b-gpu` (needs CUDA) |
| `tests/` | leak-free splits, only adapters trainable, base weights unchanged, loss goes down, adapter save/reload, run record, QLoRA refused on CPU |

The plain loop is on purpose: it keeps every step visible (masking, padding side, what gets saved). For a bigger job I'd move to the HF `Trainer` or TRL, not rewrite this.

## LoRA vs QLoRA

LoRA freezes the base weights and trains small low-rank matrices next to chosen layers (here the attention projections). Only the adapter is saved, a few MB, and one base model can serve several adapters.

QLoRA does the same on top of a base loaded in 4-bit NF4, with bf16 compute. It is what makes a 7B model trainable on a single GPU. bitsandbytes needs CUDA, so on CPU the recipe refuses QLoRA with a clear error instead of silently falling back. The `qlora-7b-gpu` config is written but **I have not run it in this repo**.

## When I wouldn't fine-tune

If the problem is missing knowledge (documents, standards, part data that changes), retrieval is usually the better first step: cheaper, easier to update, and the sources can be cited. Fine-tuning earns its place for format, style or a narrow behaviour that prompting can't hold reliably, and only once there is an evaluation set to prove it helped. That's why the baseline is always measured first.

## Results

RESULTS_PLACEHOLDER

## Not done yet

- Multi-GPU training (DDP / FSDP via `accelerate`). I understand how it works but haven't run it here, so it's roadmap, not a claim.
- Running the QLoRA 7B config on a GPU and adding its report.
- Serving the adapter with vLLM's LoRA support and checking the served outputs against the offline eval.

## CI

The workflow is in `ci/github-actions-ci.yml` (lint + tests on CPU). To enable it, move it to `.github/workflows/ci.yml`.

MIT licence.
