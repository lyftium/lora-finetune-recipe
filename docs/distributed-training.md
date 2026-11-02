# Distributed training (DDP)

## What is demonstrated

| Mode | How | Backend | Evidence |
|---|---|---|---|
| Single process | `lora-recipe run --config configs/tiny-cpu.yaml` | n/a | CI + `results/` |
| Multi-process DDP | `torchrun --nproc_per_node=2 ... --distributed` | **gloo** on CPU, **nccl** on CUDA | `results/tiny-ddp/` + this doc |
| Cluster job | `PyTorchJob` in drawing-review-service OpenShift overlay | Training Operator | YAML for OpenShift AI-style clusters |

DDP: each rank holds a full model copy and a shard of the data; gradients are all-reduced after backward. Use FSDP only when the model no longer fits with DDP.

## Commands

```bash
# CPU multi-process proof (no GPU required)
torchrun --nproc_per_node=2 -m lora_recipe.cli run \
  --config configs/tiny-ddp.yaml --distributed --skip-baseline

# Same code path on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m lora_recipe.cli run \
  --config configs/qwen-0.5b-lora.yaml --distributed --skip-baseline
```

`train_run.json` records `distributed.world_size`, `backend`, and `device` so a cluster run is auditable the same way as a laptop run.

## Cluster (OpenShift AI / Kubeflow)

See `deploy/openshift/pytorchjob-ddp.yaml` in [drawing-review-service](https://github.com/lyftium/drawing-review-service): Master+Worker replicas, `nvidia.com/gpu`, `torchrun`, restricted securityContext. Submit after the Training Operator is installed.
