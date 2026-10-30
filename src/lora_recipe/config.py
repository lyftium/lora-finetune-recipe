from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class RunConfig:
    base_model: str = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    revision: str = "main"
    qlora: bool = False  # 4-bit base model, CUDA + bitsandbytes only
    lora: LoraSettings = field(default_factory=LoraSettings)
    n_train: int = 64
    n_test: int = 32
    seed: int = 0
    lr: float = 2e-4
    batch_size: int = 8
    max_steps: int = 30
    max_len: int = 96
    max_new_tokens: int = 48
    output_dir: str = "outputs/run"

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        lora = LoraSettings(**raw.pop("lora", {}))
        return cls(lora=lora, **raw)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
