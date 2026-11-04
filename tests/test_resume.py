"""Checkpoint resume: a split run matches a continuous run under the same seed."""

from pathlib import Path

import pytest
import torch

from lora_recipe.config import RunConfig
from lora_recipe.train import train

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny(cfg_dir: Path) -> RunConfig:
    cfg = RunConfig.load("configs/tiny-cpu.yaml")
    cfg.max_steps = 6
    cfg.n_train = 32
    cfg.n_test = 8
    cfg.batch_size = 4
    cfg.output_dir = str(cfg_dir)
    return cfg


def test_resume_matches_continuous_run(tmp_path):
    continuous = train(_tiny(tmp_path / "continuous"), max_steps=6)

    part_dir = tmp_path / "partial"
    partial = train(_tiny(part_dir), max_steps=3)
    ckpt = Path(partial["record"]["checkpoint"])
    assert ckpt.is_file()
    assert partial["record"]["steps"] == 3

    resumed = train(_tiny(part_dir), out_dir=part_dir, resume_from=ckpt, max_steps=6)
    assert resumed["record"]["steps"] == 6
    assert resumed["record"]["resumed"] is True

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    continuous["model"].eval()
    resumed["model"].eval()
    with torch.no_grad():
        a = continuous["model"](ids).logits
        b = resumed["model"](ids).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_checkpoint_written_on_train(tmp_path):
    cfg = _tiny(tmp_path / "ck")
    cfg.max_steps = 2
    out = train(cfg, max_steps=2)
    assert Path(out["record"]["checkpoint"]).is_file()
    blob = torch.load(out["record"]["checkpoint"], map_location="cpu", weights_only=False)
    assert blob["step"] == 2
    assert "model" in blob and "optimizer" in blob and "config_hash" in blob
