# Interview demo — LoRA recipe

Walkthrough for a screener. Stick to artefacts in this repo.

## 90 seconds

1. **Question:** did the adapter help, or only the prompt?
2. Open the README **Fair glance** table first (fewshot exact_match 0.8) — then the adapter table (still vs unprompted base).
3. Point at `results/prompted-baseline-qwen/`, `results/qwen-0.5b-lora/`, `labs/ddp-token-weighting/`.
4. **Limits out loud:** do not quote 0.00→0.97 as fine-tuning alone; CPU / small models; `tiny-ddp` is mechanics, not multi-GPU quality.

## 10 minutes

```bash
pip install -r requirements.lock -r requirements-dev.txt && pip install --no-deps -e .
pytest -q
bash scripts/demo_resume.sh
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
```

| Minute | Show | Say |
|---|---|---|
| 0–2 | Problem + Fair glance | Same-split baseline before claiming the adapter |
| 2–4 | `pytest` + resume script | Lineage and resumable checkpoints are part of the artefact |
| 4–6 | Prompted baseline JSON | Unprompted exact_match 0.0 → fewshot 0.8; same `test_data_hash` |
| 6–8 | LoRA report vs base | Published table still quotes unprompted base — say that |
| 8–9 | `labs/ddp-token-weighting/` | Mean-of-means ≠ token-weighted loss; not an NCCL claim |
| 9–10 | Limits | Prompted vs adapter on GPU; real multi-GPU with the same lineage |

## Do not claim

- Multi-GPU / NCCL convergence from `tiny-ddp`
- That QLoRA 7B was trained here
- Quality deltas beyond the published JSON
