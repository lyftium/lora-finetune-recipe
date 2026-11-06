# Demo — LoRA recipe

Run tests and inspect baseline / adapter artefacts. Quote numbers only from the JSON under `results/`.

## Prerequisites

- Python 3.10+
- CPU is enough for the published path

## Quickstart

```bash
pip install -r requirements.lock -r requirements-dev.txt
pip install --no-deps -e .
pytest -q
bash scripts/demo_resume.sh
```

**Expected:** tests pass; resume script completes.

## Prompted baseline (same held-out set)

```bash
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
```

Compare with published [`eval-base-modes.json`](../results/prompted-baseline-qwen/eval-base-modes.json) and [`results/qwen-0.5b-lora/`](../results/qwen-0.5b-lora/).

## Limitations

- Few-shot prompting alone reaches exact_match **0.8** on this hash — do not treat unprompted 0.0 → adapter 0.97 as “fine-tuning alone”
- `tiny-ddp` is two-process mechanics, not multi-GPU quality
- QLoRA 7B config is written but not run here
