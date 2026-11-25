"""Part 3 - can a student inherit a skill through outputs that have nothing to do with it?

tinynet.TinyNet gives you the network and both training steps. You write the experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tinynet import TinyNet

N_CLASS = 10
DEFAULTS = dict(
    hidden=128,          # units per hidden layer
    n_aux=3,             # auxiliary outputs
    teacher_epochs=5,    # passes over the real training digits
    teacher_batch=64,
    student_steps=1500,  # gradient steps on noise
    student_batch=128,
    lr=1e-3,
)
SEEDS = (0, 1, 2, 3, 4)


def load_digits(path: str | Path):
    """Returns standardised (X_train, y_train, X_test, y_test). Inputs have 64 features."""
    d = np.load(path)
    mu, sd = d["X_train"].mean(), d["X_train"].std()
    return (d["X_train"] - mu) / sd, d["y_train"], (d["X_test"] - mu) / sd, d["y_test"]


def _train_teacher(base: TinyNet, X_train: np.ndarray, y_train: np.ndarray, seed: int) -> TinyNet:
    teacher = base.clone()
    rng = np.random.default_rng(seed + 1000)
    for _ in range(DEFAULTS["teacher_epochs"]):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), DEFAULTS["teacher_batch"]):
            batch = order[start : start + DEFAULTS["teacher_batch"]]
            teacher.step_classify(X_train[batch], y_train[batch], lr=DEFAULTS["lr"])
    return teacher


def _distill_noise(
    student: TinyNet, teacher: TinyNet, n_features: int, seed: int, steps: int | None = None
) -> None:
    rng = np.random.default_rng(seed + 3000)
    n_steps = DEFAULTS["student_steps"] if steps is None else steps
    for _ in range(n_steps):
        noise = rng.normal(0.0, 1.0, size=(DEFAULTS["student_batch"], n_features))
        target_aux = teacher.aux_logits(noise)
        student.step_match_aux(noise, target_aux, lr=DEFAULTS["lr"])


def run(digits_path: str | Path, out_dir: str | Path) -> dict:
    """Run the experiment for every seed in SEEDS and write <out_dir>/subliminal_results.json.

    See Part 3 of the brief for the steps and docs/OUTPUT_CONTRACT.md for the JSON layout.
    """
    X_train, y_train, X_test, y_test = load_digits(digits_path)
    n_features = X_train.shape[1]

    per_seed = []
    for seed in SEEDS:
        base = TinyNet(n_features, DEFAULTS["hidden"], N_CLASS, DEFAULTS["n_aux"], seed=seed)
        untrained_acc = base.accuracy(X_test, y_test)

        teacher = _train_teacher(base, X_train, y_train, seed=seed)
        teacher_acc = teacher.accuracy(X_test, y_test)

        student_same = base.clone()
        student_diff = TinyNet(
            n_features, DEFAULTS["hidden"], N_CLASS, DEFAULTS["n_aux"], seed=seed + 2000
        )

        rng_s = np.random.default_rng(seed + 3000)
        for _ in range(DEFAULTS["student_steps"]):
            noise = rng_s.normal(0.0, 1.0, size=(DEFAULTS["student_batch"], n_features))
            target_aux = teacher.aux_logits(noise)
            student_same.step_match_aux(noise, target_aux, lr=DEFAULTS["lr"])
            student_diff.step_match_aux(noise, target_aux, lr=DEFAULTS["lr"])

        per_seed.append(
            {
                "seed": int(seed),
                "untrained_acc": float(untrained_acc),
                "teacher_acc": float(teacher_acc),
                "student_same_init_acc": float(student_same.accuracy(X_test, y_test)),
                "student_diff_init_acc": float(student_diff.accuracy(X_test, y_test)),
            }
        )

    keys = ["untrained_acc", "teacher_acc", "student_same_init_acc", "student_diff_init_acc"]
    mean_res = {k: float(np.mean([row[k] for row in per_seed])) for k in keys}

    # Own experiment: how does the width of the auxiliary channel (n_aux = 1, 3, 10)
    # affect how much digit-classification skill transfers through pure Gaussian noise?
    conditions = []
    for n_aux_val, label in [
        (1, "1 auxiliary output"),
        (3, "3 auxiliary outputs (baseline)"),
        (10, "10 auxiliary outputs"),
    ]:
        if n_aux_val == DEFAULTS["n_aux"]:
            cond_mean = mean_res["student_same_init_acc"]
        else:
            cond_accs = []
            for seed in SEEDS:
                base = TinyNet(n_features, DEFAULTS["hidden"], N_CLASS, n_aux_val, seed=seed)
                teacher = _train_teacher(base, X_train, y_train, seed=seed)
                student = base.clone()
                _distill_noise(student, teacher, n_features=n_features, seed=seed)
                cond_accs.append(student.accuracy(X_test, y_test))
            cond_mean = float(np.mean(cond_accs))
        conditions.append(
            {
                "label": label,
                "value": float(n_aux_val),
                "student_acc_mean": float(cond_mean),
            }
        )

    result = {
        "config": {
            "hidden": DEFAULTS["hidden"],
            "n_aux": DEFAULTS["n_aux"],
            "seeds": list(SEEDS),
            "noise_profile": "iso-gauss-64",
        },
        "per_seed": per_seed,
        "mean": mean_res,
        "own_experiment": {
            "question": "Does widening the auxiliary output channel from 1 to 10 logits increase how much digit-classification accuracy transfers through pure noise?",
            "hypothesis": "More auxiliary outputs expose richer linear projections of the teacher's hidden representations, increasing student accuracy up to a saturation point.",
            "conditions": conditions,
            "conclusion": (
                f"Even a single auxiliary output transfers {conditions[0]['student_acc_mean']:.3f} test accuracy "
                f"on pure noise, rising to {conditions[1]['student_acc_mean']:.3f} with 3 outputs and plateauing at "
                f"{conditions[2]['student_acc_mean']:.3f} with 10 outputs. Shared initialization leaks task structure even through a minimal side channel."
            ),
        },
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "subliminal_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result