# Demo — LoRA recipe

Run tests and inspect baseline / adapter artefacts. Quote numbers only from the JSON under `results/`.

## Prerequisites

- Python 3.11+
- CPU is enough for the published path

## Quickstart

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
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

- Few-shot prompting reaches exact_match **0.80**; adapter 0.97 was evaluated unprompted. Same data hash does not make those prompt modes matched, so no controlled +0.17 gain is established
- Python 3.11+ is required; the CPU torch install is separate from the locked dependencies
- `tiny-ddp` is two-process mechanics, not multi-GPU quality
- QLoRA 7B config is written but not run here
