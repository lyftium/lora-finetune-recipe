"""Simulate two ranks with unequal supervised token counts (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0
N_TRIALS = 100


def rank_stats(losses: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Per-rank (sum of masked losses, count of supervised tokens)."""
    m = mask.astype(bool)
    return float(losses[m].sum()), float(m.sum())


def global_mean(stats: list[tuple[float, float]]) -> float:
    num = sum(s for s, _ in stats)
    den = sum(c for _, c in stats)
    return num / den if den else float("nan")


def naive_mean_of_means(stats: list[tuple[float, float]]) -> float:
    """Wrong reduction: average the per-rank means (ignores unequal token counts)."""
    means = [s / c for s, c in stats if c]
    return float(np.mean(means)) if means else float("nan")


def one_trial(rng: np.random.Generator) -> dict[str, float]:
    # Rank 0: few tokens, high loss; Rank 1: many tokens, low loss
    n0, n1 = int(rng.integers(4, 12)), int(rng.integers(40, 80))
    loss0 = rng.normal(2.0, 0.1, size=n0)
    loss1 = rng.normal(0.5, 0.05, size=n1)
    mask0, mask1 = np.ones(n0), np.ones(n1)
    stats = [rank_stats(loss0, mask0), rank_stats(loss1, mask1)]
    correct = global_mean(stats)
    wrong = naive_mean_of_means(stats)
    # Reference: concatenate
    ref = float(np.concatenate([loss0, loss1]).mean())
    return {
        "correct": correct,
        "naive": wrong,
        "ref": ref,
        "abs_err_correct": abs(correct - ref),
        "abs_err_naive": abs(wrong - ref),
        "n0": n0,
        "n1": n1,
    }


def main() -> dict:
    rng = np.random.default_rng(SEED)
    rows = [one_trial(rng) for _ in range(N_TRIALS)]
    summary = {
        "seed": SEED,
        "n_trials": N_TRIALS,
        "mean_abs_err_token_weighted": float(np.mean([r["abs_err_correct"] for r in rows])),
        "mean_abs_err_naive_mean_of_means": float(np.mean([r["abs_err_naive"] for r in rows])),
        "max_abs_err_token_weighted": float(np.max([r["abs_err_correct"] for r in rows])),
        "limits": "CPU NumPy simulation only — not NCCL / multi-GPU training proof",
    }
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
