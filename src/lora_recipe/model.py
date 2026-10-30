"""Model loading: plain LoRA on any device, QLoRA (4-bit base) on CUDA only."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RunConfig


class QLoraUnavailable(RuntimeError):
    pass


def load_tokenizer(cfg: RunConfig):
    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left padding for batched generation
    return tok


def load_base(cfg: RunConfig):
    if not cfg.qlora:
        return AutoModelForCausalLM.from_pretrained(cfg.base_model, revision=cfg.revision)
    if not torch.cuda.is_available():
        raise QLoraUnavailable(
            "QLoRA needs a CUDA GPU and bitsandbytes. Use qlora: false for the CPU smoke test."
        )
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise QLoraUnavailable("Install the qlora extra: pip install .[qlora]") from exc
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat, as in the QLoRA paper
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, revision=cfg.revision, quantization_config=bnb, device_map="auto"
    )
    return prepare_model_for_kbit_training(model)


def add_lora(model, cfg: RunConfig):
    lc = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


def count_params(model) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "fraction": round(trainable / total, 6)}
