"""Part 2 - how many independent opinions is this panel really worth?

Fill in every function marked TODO. Keep the function names and return shapes.
"""
from __future__ import annotations

import itertools
import numpy as np


def phi_matrix(correct: np.ndarray) -> np.ndarray:
    """Pairwise phi coefficient between the models' ERROR indicators, shape (n_models, n_models).

    Phi is the Pearson correlation of two 0/1 vectors (1 = wrong on that question).
    Diagonal is 1. If a model is never wrong or always wrong, its correlation with
    the others is undefined: use 0 for those entries.
    """
    err = (~np.asarray(correct, dtype=bool)).astype(float)
    n_models = err.shape[1]
    centered = err - err.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered ** 2, axis=0))

    phi = np.zeros((n_models, n_models), dtype=float)
    valid = norms > 0
    if np.any(valid):
        normed = centered[:, valid] / norms[valid]
        sub_phi = np.clip(normed.T @ normed, -1.0, 1.0)
        idx = np.where(valid)[0]
        phi[np.ix_(idx, idx)] = sub_phi

    np.fill_diagonal(phi, 1.0)
    return phi


def mean_offdiag(phi: np.ndarray) -> float:
    """Mean of the entries above the diagonal (each pair counted once)."""
    n = phi.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(phi[iu].mean())


def n_effective(n_models: int, mean_phi: float) -> float:
    """n_eff = n / (1 + (n - 1) * mean_phi)."""
    return float(n_models / (1.0 + (n_models - 1) * mean_phi))


def plurality_vote(answers: np.ndarray) -> np.ndarray:
    """The panel's answer to each question, shape (n_questions,).

    `answers` has shape (n_questions, n_models) and holds labels such as "A".
    The most common answer wins. On a tie, take the answer given by the earliest
    column among the tied answers.
    """
    ans = np.asarray(answers)
    n_questions = ans.shape[0]
    chosen = np.empty(n_questions, dtype=ans.dtype)

    for q in range(n_questions):
        row = ans[q]
        counts: dict = {}
        for val in row:
            counts[val] = counts.get(val, 0) + 1
        best_val = row[0]
        best_count = counts[best_val]
        for val in row[1:]:
            c = counts[val]
            if c > best_count:
                best_val = val
                best_count = c
        chosen[q] = best_val

    return chosen


def vote_accuracy(answers: np.ndarray, gold: np.ndarray) -> float:
    """Accuracy of plurality_vote against the gold labels."""
    preds = plurality_vote(answers)
    return float(np.mean(preds == np.asarray(gold)))


def majority_correct_rate(correct: np.ndarray) -> float:
    """Share of questions where STRICTLY more than half of the models are right."""
    c = np.asarray(correct, dtype=bool)
    n_models = c.shape[1]
    return float(np.mean(c.sum(axis=1) > (n_models / 2.0)))


def majority_correct_if_independent(accs: np.ndarray) -> float:
    """The same rate for imaginary models with accuracies `accs` whose errors are independent.

    Compute it exactly (the number of correct models follows a Poisson-binomial
    distribution) or by a large simulation.
    """
    probs = np.asarray(accs, dtype=float)
    n_models = len(probs)
    pmf = np.array([1.0], dtype=float)
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])

    k_vals = np.arange(n_models + 1)
    return float(pmf[k_vals > (n_models / 2.0)].sum())


def select_panel(panel, k: int = 3) -> dict:
    """Choose a k-model panel on the dev split, then score it once on the test split.

    `panel` is the object returned by io.load_predictions.

    1. Rank models by accuracy on the dev rows.
    2. Score every k-model subset by vote_accuracy on dev. Inside a subset, order the
       columns by dev accuracy, best first, so that ties go to the strongest member.
    3. Keep the subset with the highest dev score.
    4. Score that subset, and the naive "top k by dev accuracy" subset, on the test rows.

    Return a dict with exactly these keys:
        k, models, dev_accuracy, test_accuracy,
        baseline_top_k_models, baseline_top_k_dev_accuracy, baseline_top_k_test_accuracy
    """
    dev_mask = panel.split == "dev"
    test_mask = panel.split == "test"

    dev_correct = panel.correct[dev_mask]
    dev_answers = panel.answers[dev_mask]
    dev_gold = panel.gold[dev_mask]

    test_answers = panel.answers[test_mask]
    test_gold = panel.gold[test_mask]

    dev_accs = dev_correct.mean(axis=0)
    ranked_indices = [int(i) for i in np.argsort(-dev_accs, kind="mergesort")]

    best_subset = None
    best_dev_acc = -1.0

    for subset in itertools.combinations(ranked_indices, k):
        sub_idx = list(subset)
        acc = vote_accuracy(dev_answers[:, sub_idx], dev_gold)
        if acc > best_dev_acc + 1e-12:
            best_dev_acc = acc
            best_subset = sub_idx

    assert best_subset is not None
    best_test_acc = vote_accuracy(test_answers[:, best_subset], test_gold)

    baseline_idx = ranked_indices[:k]
    baseline_dev_acc = vote_accuracy(dev_answers[:, baseline_idx], dev_gold)
    baseline_test_acc = vote_accuracy(test_answers[:, baseline_idx], test_gold)

    return {
        "k": int(k),
        "models": [panel.models[i] for i in best_subset],
        "dev_accuracy": float(best_dev_acc),
        "test_accuracy": float(best_test_acc),
        "baseline_top_k_models": [panel.models[i] for i in baseline_idx],
        "baseline_top_k_dev_accuracy": float(baseline_dev_acc),
        "baseline_top_k_test_accuracy": float(baseline_test_acc),
    }