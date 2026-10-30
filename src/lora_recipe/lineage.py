from __future__ import annotations

import subprocess
from dataclasses import asdict
from datetime import UTC, datetime

import peft
import torch
import transformers

from . import __version__
from .config import RunConfig


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_record(kind: str, cfg: RunConfig, *, data_hash: str, extra: dict | None = None) -> dict:
    return {
        "kind": kind,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "recipe_version": __version__,
        "git_sha": git_sha(),
        "config_hash": cfg.fingerprint(),
        "config": asdict(cfg),
        "data_hash": data_hash,
        "libs": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        **(extra or {}),
    }
