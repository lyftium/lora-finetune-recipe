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
