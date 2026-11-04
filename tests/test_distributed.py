from lora_recipe.train import _dist_info


def test_dist_info_defaults_to_single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    rank, world, local, use = _dist_info(False)
    assert (rank, world, local, use) == (0, 1, 0, False)


def test_dist_info_reads_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    rank, world, local, use = _dist_info(True)
    assert rank == 1 and world == 2 and local == 1 and use is True
