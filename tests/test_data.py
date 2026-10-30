import json

from lora_recipe.data import data_hash, make_splits


def test_no_test_prompt_in_train():
    train, test = make_splits(300, 100, seed=0)
    assert not {r["text"] for r in train} & {r["text"] for r in test}
    assert len({r["text"] for r in train}) == 300


def test_deterministic():
    assert data_hash(make_splits(50, 10, 3)[0]) == data_hash(make_splits(50, 10, 3)[0])
    assert data_hash(make_splits(50, 10, 3)[0]) != data_hash(make_splits(50, 10, 4)[0])


def test_targets_are_json():
    train, _ = make_splits(50, 10)
    for r in train:
        assert isinstance(json.loads(r["target"]), dict)
