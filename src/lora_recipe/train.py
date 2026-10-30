"""Training loop. Deliberately plain PyTorch so every step is visible.

The loss is computed on the answer tokens only: the prompt is masked with -100,
otherwise the model spends capacity learning to repeat the instruction.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

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
    for i, b in enumerate(batch):  # right padding for training
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train(cfg: RunConfig, out_dir: Path | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    out_dir = Path(out_dir or cfg.output_dir)
    tok = load_tokenizer(cfg)
    model = add_lora(load_base(cfg), cfg)
    params = count_params(model)
    train_rows, test_rows = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    data = encode(tok, train_rows, cfg.max_len)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    model.train()
    losses, step, t0 = [], 0, time.perf_counter()
    rng = random.Random(cfg.seed)
    while step < cfg.max_steps:
        rng.shuffle(data)
        for i in range(0, len(data), cfg.batch_size):
            batch = collate(data[i : i + cfg.batch_size], tok.pad_token_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(round(loss.item(), 4))
            step += 1
            if step >= cfg.max_steps:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")  # only the LoRA weights, a few MB at most
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
        },
    )
    (out_dir / "train_run.json").write_text(json.dumps(record, indent=2))
    return {"model": model, "tokenizer": tok, "record": record, "test_rows": test_rows}
