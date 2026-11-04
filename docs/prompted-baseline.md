# Prompted vs unprompted baseline

The published Qwen-0.5B baseline in `results/qwen-0.5b-lora/eval-base.json` is
**unprompted**: the model only sees “Extract the fields as JSON.” That understates
what a careful prompt can do and overstates the adapter’s relative gain if you
compare LoRA (trained on that format) to a bare base call.

Three eval instruction modes share the **same** held-out split (`test_data_hash`):

| Mode | What the model sees |
|------|---------------------|
| `unprompted` | Format hint only |
| `schema` | Field names and types for each annotation kind |
| `fewshot` | One fixed example per kind, then the query |

```bash
# Compare all three on the base model (or + adapter)
lora-recipe eval --config configs/tiny-cpu.yaml --all-prompt-modes \
  --out results/prompted-baseline-tiny/eval-base-modes.json

lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes \
  --out results/prompted-baseline-qwen/eval-base-modes.json
```

Numbers below are measured artefacts — do not invent digits beyond the JSON files.

## Measured (Qwen2.5-0.5B-Instruct base, n=100, same `test_data_hash`)

Source: [`results/prompted-baseline-qwen/eval-base-modes.json`](../results/prompted-baseline-qwen/eval-base-modes.json) — `test_data_hash` `b89f85e0adae4406` (matches `results/qwen-0.5b-lora/eval-base.json`).

| Mode | json_valid | exact_match |
|------|----------:|------------:|
| unprompted | 0.66 | 0.0 |
| schema | 0.78 | 0.03 |
| fewshot | 1.0 | 0.8 |

Reading: few-shot recovers most exact matches the unprompted base misses; the published LoRA report still uses the unprompted baseline so adapter gains are not overstated relative to a careful prompt. Tiny-random model artefacts under `results/prompted-baseline-tiny/` are mechanics-only (all zeros).
