"""Part 4 - run the whole audit and write results.json and REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import leaderboard as lb
from . import panel as pn
from .io import Panel


def run_audit(
    panel: Panel, out_dir: str | Path, seed: int = 0, n_boot: int = 2000, k: int = 3
) -> dict:
    """Compute everything from Parts 1 and 2 and write two files into out_dir:

        results.json   exact layout in docs/OUTPUT_CONTRACT.md (this file is auto-graded)
        REPORT.md      a short report a hiring-product manager could read (Part 4)

    Must work for ANY predictions file with the same four meta columns: any number
    of models (2 or more), any model names, any answer labels, any number of rows.
    """
    correct = panel.correct
    n_questions = panel.n_questions
    models = list(panel.models)
    n_models = len(models)

    accs = lb.accuracy(correct)
    ci_low, ci_high = lb.bootstrap_accuracy_ci(correct, n_boot=n_boot, seed=seed)

    order = np.argsort(-accs, kind="mergesort")
    leaderboard_entries = []
    for rank_idx, m_idx in enumerate(order, start=1):
        i = int(m_idx)
        leaderboard_entries.append(
            {
                "model": models[i],
                "accuracy": float(accs[i]),
                "ci_low": float(ci_low[i]),
                "ci_high": float(ci_high[i]),
                "rank": int(rank_idx),
            }
        )

    top2 = lb.compare_top_two(correct, models, n_boot=n_boot, seed=seed)
    stability = lb.rank_stability(
        correct, lb.default_sizes(n_questions), n_draws=n_boot, seed=seed
    )

    phi = pn.phi_matrix(correct)
    mean_phi = pn.mean_offdiag(phi)
    n_eff = pn.n_effective(n_models, mean_phi)
    v_acc = pn.vote_accuracy(panel.answers, panel.gold)
    maj_actual = pn.majority_correct_rate(correct)
    maj_indep = pn.majority_correct_if_independent(accs)

    panel_summary = {
        "phi_matrix": [[float(x) for x in row] for row in phi],
        "mean_phi": float(mean_phi),
        "n_eff": float(n_eff),
        "vote_accuracy": float(v_acc),
        "majority_correct_rate": float(maj_actual),
        "majority_correct_rate_if_independent": float(maj_indep),
    }

    eff_k = min(k, n_models)
    rec_panel = pn.select_panel(panel, k=eff_k)

    results = {
        "n_questions": int(n_questions),
        "models": models,
        "leaderboard": leaderboard_entries,
        "top2": top2,
        "rank_stability": stability,
        "panel": panel_summary,
        "recommended_panel": rec_panel,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Six Votes Audit Report",
        "",
        f"Evaluated **{n_models} models** across **{n_questions} screening questions**.",
        "",
        "## 1. Leaderboard & Uncertainty",
        "",
        "| Rank | Model | Accuracy | 95% CI |",
        "|---:|---|---:|---:|",
    ]
    for entry in leaderboard_entries:
        lines.append(
            f"| {entry['rank']} | {entry['model']} | {entry['accuracy']:.3f} | "
            f"[{entry['ci_low']:.3f}, {entry['ci_high']:.3f}] |"
        )

    lines.extend(
        [
            "",
            f"- **Top-2 comparison (`{top2['leader']}` vs `{top2['runner_up']}`):** "
            f"gap = `{top2['accuracy_gap']:.3f}` (95% paired CI `[{top2['gap_ci_low']:.3f}, {top2['gap_ci_high']:.3f}]`), "
            f"McNemar p-value = `{top2['p_value']:.4f}` (`significant={top2['significant']}`).",
            f"- **Full-benchmark rank stability:** `{stability[-1]['p_leader_holds']:.3f}` probability that `{top2['leader']}` stays #1 at n={stability[-1]['n']}.",
            "",
            "## 2. Panel Independence & Voting",
            "",
            f"- **Mean pairwise error correlation (`mean_phi`):** `{mean_phi:.3f}`",
            f"- **Effective independent models (`n_eff`):** `{n_eff:.2f}` out of `{n_models}`",
            f"- **Full-panel plurality vote accuracy:** `{v_acc:.3f}`",
            f"- **Majority-correct rate:** `{maj_actual:.3f}` actual vs `{maj_indep:.3f}` if models were independent",
            "",
            "## 3. Recommended Sub-Panel",
            "",
            f"- **Selected {eff_k}-model diverse panel:** `{', '.join(rec_panel['models'])}` "
            f"(dev: `{rec_panel['dev_accuracy']:.3f}`, test: `{rec_panel['test_accuracy']:.3f}`)",
            f"- **Naive top-{eff_k} baseline:** `{', '.join(rec_panel['baseline_top_k_models'])}` "
            f"(dev: `{rec_panel['baseline_top_k_dev_accuracy']:.3f}`, test: `{rec_panel['baseline_top_k_test_accuracy']:.3f}`)",
            "",
        ]
    )
    (out_path / "REPORT.md").write_text("\n".join(lines))
    return results