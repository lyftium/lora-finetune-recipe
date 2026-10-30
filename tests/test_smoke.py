"""CPU smoke tests on a tiny random model. They check mechanics, not quality."""

import json

import pytest
import torch

from lora_recipe.config import RunConfig
from lora_recipe.evaluate import _first_json, evaluate
from lora_recipe.model import QLoraUnavailable, add_lora, count_params, load_base
from lora_recipe.train import train

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = RunConfig.load("configs/tiny-cpu.yaml")
    c.output_dir = str(tmp_path_factory.mktemp("run"))
    return c


@pytest.fixture(scope="module")
def trained(cfg):
    return train(cfg)


def test_only_adapters_are_trainable(cfg):
    model = add_lora(load_base(cfg), cfg)
    p = count_params(model)
    assert 0 < p["trainable"] < 0.05 * p["total"]
    names = [n for n, q in model.named_parameters() if q.requires_grad]
    assert names and all("lora_" in n for n in names)


def test_base_weights_unchanged_by_training(cfg, trained):
    fresh = dict(load_base(cfg).named_parameters())
    tuned = {
        n.replace(".base_layer", ""): p
        for n, p in trained["model"].get_base_model().named_parameters()
        if "lora_" not in n
    }
    assert tuned.keys() == fresh.keys()
    for n, p in fresh.items():
        assert torch.equal(p, tuned[n]), n


def test_loss_goes_down(trained):
    losses = trained["record"]["losses"]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first


def test_adapter_roundtrip(cfg, trained):
    from peft import PeftModel

    path = f"{cfg.output_dir}/adapter"
    reloaded = PeftModel.from_pretrained(load_base(cfg), path)
    ids = torch.tensor([[1, 2, 3, 4]])
    trained["model"].eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(trained["model"](ids).logits, reloaded(ids).logits, atol=1e-5)


def test_run_record_has_lineage(cfg, trained):
    rec = json.loads(open(f"{cfg.output_dir}/train_run.json").read())
    for k in ("git_sha", "config_hash", "data_hash", "test_data_hash", "libs", "params"):
        assert k in rec


def test_evaluate_runs(trained):
    rep = evaluate(trained["model"], trained["tokenizer"], trained["test_rows"][:4], 8)
    assert rep["n"] == 4 and 0 <= rep["exact_match"] <= rep["json_valid"] <= 1


def test_json_extraction():
    assert _first_json('x {"a":1} y') == {"a": 1}
    assert _first_json("no json") is None
    assert _first_json('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}
    assert _first_json('{"a": 1') is None  # truncated output is not valid


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU guard")
def test_qlora_refuses_cpu_with_a_clear_message():
    cfg = RunConfig(qlora=True)
    with pytest.raises(QLoraUnavailable, match="CUDA"):
        load_base(cfg)


def test_eval_cli_with_saved_adapter(cfg, trained, tmp_path):
    from lora_recipe.cli import main

    out = tmp_path / "eval.json"
    argv = ["eval", "--config", "configs/tiny-cpu.yaml", "--adapter", f"{cfg.output_dir}/adapter"]
    assert main([*argv, "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["scores"]["n"] == cfg.n_test
    assert report["test_data_hash"] == trained["record"]["test_data_hash"]
