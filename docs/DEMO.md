# Demo path (screener / interview)

No company names. Use only what this repository proves.

## 90 seconds

1. Problem: adapters reported without a same-split baseline hide whether fine-tuning helped.
2. Show top README table (Verify / prompted baseline / Status).
3. Point at artefacts: `results/prompted-baseline-qwen/`, `results/qwen-0.5b-lora/`, `labs/ddp-token-weighting/`.
4. State limits: CPU / small models here; QLoRA 7B config written not run; `tiny-ddp` is mechanics, not multi-GPU quality.

## 10 minutes (laptop)

```bash
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest -q                                 # CPU tests incl. resume / SFT mask
bash scripts/demo_resume.sh               # checkpoint + --resume-from
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
# optional: lora-recipe run --config configs/tiny-cpu.yaml
```

Talk track while commands run:

| Minute | Show | Say |
|---|---|---|
| 0–2 | Problem + Verify table | Baseline on the same held-out set before claiming the adapter |
| 2–4 | `pytest` + resume script | Lineage + resumable checkpoints are part of the artefact |
| 4–6 | Prompted baseline JSON | Unprompted exact_match 0.0 → fewshot 0.8; same `test_data_hash` |
| 6–8 | LoRA report vs base | Published report still quotes unprompted base — honesty about framing |
| 8–9 | `labs/ddp-token-weighting/` | Mean-of-means ≠ token-weighted loss; not an NCCL claim |
| 9–10 | Limits + next | Real multi-GPU DDP with the same lineage; prompted vs adapter on GPU |

## Do not claim

- Multi-GPU / NCCL convergence from `tiny-ddp`
- That QLoRA 7B was trained in this repo
- Invented quality deltas beyond published JSON artefacts
