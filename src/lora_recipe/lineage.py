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


def git_dirty() -> bool | None:
    """True when tracked files differ from HEAD, i.e. git_sha alone does not identify the code."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_record(kind: str, cfg: RunConfig, *, data_hash: str, extra: dict | None = None) -> dict:
    return {
        "kind": kind,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "recipe_version": __version__,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
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
