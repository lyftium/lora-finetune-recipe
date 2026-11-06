# Fine-tuning evaluation benchmarks

This project has two separate measured evaluations. They use different synthetic tasks and must not be combined into a single adapter-improvement claim.

## 1. Repository task: narrow annotation-to-JSON extraction

The saved reports share 100 held-out examples (`test_data_hash` `b89f85e0adae4406`) and greedy decoding with at most 48 new tokens. The CPU run used Qwen2.5-0.5B-Instruct, torch 2.5.1+cpu, transformers 4.46.3, and peft 0.13.2.

| Condition | JSON valid | Exact match | Source |
|---|---:|---:|---|
| Base, unprompted | 0.66 | 0.00 | [`eval-base.json`](results/qwen-0.5b-lora/eval-base.json) |
| Base, schema prompt | 0.78 | 0.03 | [`eval-base-modes.json`](results/prompted-baseline-qwen/eval-base-modes.json) |
| Base, schema + three-shot | 1.00 | 0.80 | [`eval-base-modes.json`](results/prompted-baseline-qwen/eval-base-modes.json) |
| LoRA, unprompted | 1.00 | 0.97 | [`eval-lora.json`](results/qwen-0.5b-lora/eval-lora.json) |

The 0.97 adapter and 0.80 few-shot results use different prompt conditions. **They do not establish a controlled +0.17 gain.** I did not rerun them.

Reproduction commands for the recipe:

```bash
lora-recipe run --config configs/qwen-0.5b-lora.yaml
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --adapter outputs/qwen-0.5b-lora/adapter
lora-recipe eval --config configs/qwen-0.5b-lora.yaml --all-prompt-modes   --out results/prompted-baseline-qwen/eval-base-modes.json
```

The original adapter is not published. Training that row again requires model downloads and does not guarantee identical behavior. Training used seed 0, 400 train examples, LoRA rank 16, 150 steps, batch 8, and 558.9 seconds, as recorded in [`train_run.json`](results/qwen-0.5b-lora/train_run.json). Source provenance for that training run is `git_sha: unknown`; evaluation files carry a source hash, while the prompted-baseline report marks a dirty worktree.

## 2. Kaggle T4 task: calendar/order/sensor-to-JSON extraction

I also measured a distinct prompt-matched experiment on 400 train and 150 held-out synthetic examples. It compares the same base model and LoRA adapter under a plain prompt against a stronger base-only schema + three-shot prompt. The source and its fixed data/task generation are in [`notebook.ipynb`](notebook.ipynb); all numbers below are in [`results/kaggle-t4-synthetic-json/report.json`](results/kaggle-t4-synthetic-json/report.json).

| Condition | JSON valid | Exact match | Exact event | Exact order | Exact sensor |
|---|---:|---:|---:|---:|---:|
| Base, plain prompt | 0.6733333 | 0.00 | 0.00 | 0.00 | 0.00 |
| Base, schema + three-shot | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LoRA, same plain prompt | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**What this supports:** the adapter improved over its prompt-matched plain baseline, but the prompted baseline already saturated this small held-out test. This is not evidence that fine-tuning is necessary.

Run the complete public [Kaggle notebook](https://www.kaggle.com/code/lyftium/lora-fine-tuning-baseline-vs-adapter) with **Run All** on a T4 after inspecting its package/model downloads. Recorded conditions: Qwen2.5-0.5B-Instruct, LoRA rank 16 / alpha 32 / dropout 0.05, 150 steps, batch 8, seed 0, learning rate 0.0002, maximum 64 new tokens. Training took 84.3 seconds; trainable parameters were 2,162,688 of 496,195,456. The saved run log reports Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, and peft 0.20.0.

The GPU run was not repeated for this publication and its adapter is not included.

## Common limitations

Both evaluations use synthetic, narrow, fixed-format tasks with one seed. They do not measure performance on real drawings or production text. Exact match does not establish robustness beyond the held-out examples. Prompt conditions, dataset/task identity, and decoding settings must accompany any comparison.
