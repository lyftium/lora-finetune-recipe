"""Prompted vs unprompted baselines share the same held-out rows."""

from lora_recipe.data import PROMPTS, render_prompt


def test_prompt_modes_are_distinct():
    text = "POS|0.05|A"
    rendered = {m: render_prompt(text, m) for m in PROMPTS}
    assert len(set(rendered.values())) == 3
    assert "characteristic" in rendered["schema"]
    assert "PRT-417" in rendered["fewshot"]
    assert "characteristic" not in rendered["unprompted"]


def test_unknown_prompt_mode_raises():
    try:
        render_prompt("x", "telepathy")
    except ValueError as e:
        assert "telepathy" in str(e)
    else:
        raise AssertionError("expected ValueError")
