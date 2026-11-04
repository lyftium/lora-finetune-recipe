# Checkpoint resume

Training writes `outputs/.../checkpoint.pt` with adapter weights, optimizer state,
step count, losses, and RNG seeds. Resume continues from that file without
retraining prior steps.

```bash
# Phase 1 — short run
lora-recipe run --config configs/tiny-cpu.yaml --out outputs/resume-demo --max-steps 5

# Phase 2 — continue to 10 steps
lora-recipe run --config configs/tiny-cpu.yaml --out outputs/resume-demo \
  --resume-from outputs/resume-demo/checkpoint.pt --max-steps 10
```

`train_run.json` records `resumed`, `resume_from`, and `checkpoint` for lineage.
`tests/test_resume.py` checks that a split run matches a continuous run under
the same seed (CPU tiny model). Or run `bash scripts/demo_resume.sh`.
