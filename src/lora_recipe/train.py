"""Training loop. Plain PyTorch so every step is visible; optional DDP via torchrun.

The loss is computed on the answer tokens only: the prompt is masked with -100,
otherwise the model spends capacity learning to repeat the instruction.

Distributed: set --distributed (or WORLD_SIZE>1). Use gloo on CPU, nccl on CUDA.
Only rank 0 writes the adapter and run record.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import RunConfig
from .data import PROMPT, data_hash, make_splits
from .lineage import run_record
from .model import add_lora, count_params, load_base, load_tokenizer


def encode(tok, rows: list[dict], max_len: int) -> list[dict]:
    out = []
    for r in rows:
        p = tok(PROMPT.format(text=r["text"]), add_special_tokens=True)["input_ids"]
        a = tok(" " + r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        out.append({"input_ids": ids, "labels": labels})
    return out


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def _dist_info(enabled: bool) -> tuple[int, int, int, bool]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    use = enabled or world > 1
    return rank, world, local, use


def _setup(rank: int, world: int, local: int) -> str:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return backend


def _cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _device(local: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local)
    return torch.device("cpu")


def train(
    cfg: RunConfig,
    out_dir: Path | None = None,
    distributed: bool = False,
    resume_from: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    rank, world, local, use_dist = _dist_info(distributed)
    backend = None
    if use_dist:
        if world < 2:
            raise RuntimeError(
                "distributed training needs WORLD_SIZE>=2 (e.g. torchrun --nproc_per_node=2)"
            )
        backend = _setup(rank, world, local)

    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    out_dir = Path(out_dir or cfg.output_dir)
    device = _device(local)
    steps_target = int(max_steps if max_steps is not None else cfg.max_steps)
    ckpt_path = out_dir / "checkpoint.pt"


    tok = load_tokenizer(cfg)
    raw = add_lora(load_base(cfg), cfg)
    raw.to(device)
    params = count_params(raw)
    model = DDP(raw, device_ids=[local] if device.type == "cuda" else None) if use_dist else raw

    train_rows, test_rows = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    # Shard data across ranks (simple stride; enough for the recipe)
    if use_dist:
        train_rows = train_rows[rank::world]
    data = encode(tok, train_rows, cfg.max_len)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    model.train()
    losses, step, t0 = [], 0, time.perf_counter()
    rng = random.Random(cfg.seed + rank)
    # Epoch cursor so resume continues mid-epoch (not a fresh shuffle).
    order = list(range(len(data)))
    cursor = 0
    resumed = False
    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location=device, weights_only=False)
        core0 = model.module if use_dist else model
        core0.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        losses = list(ckpt.get("losses", []))
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"])
        if "py_rng" in ckpt:
            rng.setstate(ckpt["py_rng"])
        if "order" in ckpt and "cursor" in ckpt:
            order = list(ckpt["order"])
            cursor = int(ckpt["cursor"])
        resumed = True
    while step < steps_target:
        if cursor >= len(order):
            order = list(range(len(data)))
            rng.shuffle(order)
            cursor = 0
        if step == 0 and cursor == 0 and not resumed:
            rng.shuffle(order)
        end = min(cursor + cfg.batch_size, len(order))
        batch_rows = [data[j] for j in order[cursor:end]]
        cursor = end
        batch = collate(batch_rows, tok.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(round(float(loss.detach().item()), 4))
        step += 1

    core = model.module if use_dist else model
    record = None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model": core.state_dict(),
                "optimizer": opt.state_dict(),
                "losses": losses,
                "order": order,
                "cursor": cursor,
                "torch_rng": torch.get_rng_state(),
                "py_rng": rng.getstate(),
                "config_hash": cfg.fingerprint(),
                "data_hash": data_hash(train_rows),
            },
            ckpt_path,
        )
        core.save_pretrained(out_dir / "adapter")
        record = run_record(
            "train",
            cfg,
            data_hash=data_hash(train_rows),
            extra={
                "params": params,
                "steps": step,
                "seconds": round(time.perf_counter() - t0, 1),
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "losses": losses,
                "test_data_hash": data_hash(test_rows),
                "resumed": resumed,
                "resume_from": str(resume_from) if resume_from else None,
                "checkpoint": str(ckpt_path),
                "distributed": {
                    "enabled": use_dist,
                    "world_size": world,
                    "backend": backend or "none",
                    "device": str(device),
                },
            },
        )
        (out_dir / "train_run.json").write_text(json.dumps(record, indent=2))

    if use_dist:
        _cleanup()

    return {
        "model": core,
        "tokenizer": tok,
        "record": record,
        "test_rows": test_rows,
        "rank": rank,
        "world_size": world,
    }
