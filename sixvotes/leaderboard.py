"""Part 1 - an honest leaderboard: scores with uncertainty, a paired test, rank stability.

Fill in every function marked TODO. Keep the function names and return shapes:
tests/test_public.py and report.py rely on them. Add helpers freely.

`correct` is always a boolean array of shape (n_questions, n_models);
correct[q, m] is True when model m answered question q correctly.
"""
from __future__ import annotations

import math
import numpy as np


def accuracy(correct: np.ndarray) -> np.ndarray:
    """Per-model accuracy, shape (n_models,)."""
    return np.asarray(correct, dtype=float).mean(axis=0)


def bootstrap_accuracy_ci(
    correct: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for each model's accuracy.

    Resample QUESTIONS with replacement n_boot times. Return (low, high), each of
    shape (n_models,), the 2.5th and 97.5th percentiles when level = 0.95.
    """
    c = np.asarray(correct, dtype=float)
    n_questions, n_models = c.shape
    rng = np.random.default_rng(seed)

    boot_means = np.empty((n_boot, n_models), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_questions, size=n_questions)
        boot_means[b] = c[idx].mean(axis=0)

    alpha = 1.0 - level
    low, high = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return low, high


def mcnemar_exact_p(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for two models answering the same questions.

    Only questions where exactly one of the two models is right carry information.
    If there are none, return 1.0.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0

    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def compare_top_two(
    correct: np.ndarray, models: list[str], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired comparison of the #1 and #2 model.

    Return a dict with exactly these keys:
        leader, runner_up      model names
        accuracy_gap           leader accuracy minus runner-up accuracy
        gap_ci_low, gap_ci_high  95% PAIRED bootstrap interval for the gap
        p_value                from mcnemar_exact_p
        significant            True when p_value < 0.05
    """
    accs = accuracy(correct)
    order = np.argsort(-accs, kind="mergesort")
    i1, i2 = int(order[0]), int(order[1])

    c1 = np.asarray(correct[:, i1], dtype=float)
    c2 = np.asarray(correct[:, i2], dtype=float)
    diffs = c1 - c2
    gap = float(diffs.mean())

    n_questions = len(diffs)
    rng = np.random.default_rng(seed)
    boot_gaps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_questions, size=n_questions)
        boot_gaps[b] = diffs[idx].mean()

    gap_ci_low, gap_ci_high = np.quantile(boot_gaps, [0.025, 0.975])
    p_val = mcnemar_exact_p(correct[:, i1], correct[:, i2])

    return {
        "leader": models[i1],
        "runner_up": models[i2],
        "accuracy_gap": gap,
        "gap_ci_low": float(gap_ci_low),
        "gap_ci_high": float(gap_ci_high),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


def rank_stability(
    correct: np.ndarray, sizes: list[int], n_draws: int = 2000, seed: int = 0
) -> list[dict]:
    """How often the full-benchmark leader is still #1 on a re-drawn benchmark.

    For each size in `sizes`: draw `size` questions with replacement, n_draws times.
    The leader "holds" on a draw when its accuracy is >= every other model's
    (ties count as holding). Return [{"n": size, "p_leader_holds": fraction}, ...].
    """
    c = np.asarray(correct, dtype=float)
    n_questions = c.shape[0]
    accs = accuracy(c)
    leader_idx = int(np.argmax(accs))
    rng = np.random.default_rng(seed)

    results = []
    for size in sizes:
        holds = 0
        for _ in range(n_draws):
            idx = rng.integers(0, n_questions, size=size)
            draw_acc = c[idx].mean(axis=0)
            if draw_acc[leader_idx] >= draw_acc.max() - 1e-12:
                holds += 1
        results.append({"n": int(size), "p_leader_holds": float(holds / n_draws)})
    return results


def default_sizes(n_questions: int) -> list[int]:
    """Benchmark sizes to test: 50, 200, 500 (when smaller than the file) and the full size."""
    return [s for s in (50, 200, 500) if s < n_questions] + [n_questions]