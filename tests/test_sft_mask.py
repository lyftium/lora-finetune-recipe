"""SFT label mask: prompt tokens are -100; answer tokens are supervised."""

from lora_recipe.config import RunConfig
from lora_recipe.data import make_splits
from lora_recipe.model import load_tokenizer
from lora_recipe.train import encode


def test_prompt_tokens_masked_answer_supervised():
    cfg = RunConfig.load("configs/tiny-cpu.yaml")
    cfg.n_train = 8
    cfg.n_test = 4
    tok = load_tokenizer(cfg)
    train, _ = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    rows = encode(tok, train[:3], cfg.max_len)
    for row in rows:
        labels = row["labels"]
        assert any(x == -100 for x in labels), "expected masked prompt"
        assert any(x != -100 for x in labels), "expected supervised answer tokens"
        # Masked prefix is contiguous at the start
        first_sup = next(i for i, x in enumerate(labels) if x != -100)
        assert all(x == -100 for x in labels[:first_sup])
