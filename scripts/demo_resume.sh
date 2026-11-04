#!/usr/bin/env bash
# Demonstrates interruptible training: train 4 steps, resume to 8.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-outputs/resume-demo}"
rm -rf "$OUT"
python -m lora_recipe.cli run --config configs/tiny-cpu.yaml --out "$OUT" --max-steps 4 --skip-baseline
test -f "$OUT/checkpoint.pt"
python -m lora_recipe.cli run --config configs/tiny-cpu.yaml --out "$OUT" \
  --resume-from "$OUT/checkpoint.pt" --max-steps 8 --skip-baseline
python - <<PY
import json
from pathlib import Path
rec = json.loads(Path("${OUT}/train_run.json").read_text())
assert rec["steps"] == 8 and rec["resumed"] is True, rec
print("resume ok:", rec["steps"], "steps; checkpoint=", rec["checkpoint"])
PY
