# Six Votes Audit Report

Evaluated **6 models** across **1500 screening questions**.

## 1. Leaderboard & Uncertainty

| Rank | Model | Accuracy | 95% CI |
|---:|---|---:|---:|
| 1 | atlas-large | 0.714 | [0.689, 0.737] |
| 2 | nimbus-2 | 0.697 | [0.675, 0.721] |
| 3 | atlas-medium | 0.689 | [0.666, 0.713] |
| 4 | borealis-xl | 0.669 | [0.645, 0.693] |
| 5 | cirrus-7b | 0.649 | [0.626, 0.674] |
| 6 | borealis-mini | 0.614 | [0.589, 0.638] |

- **Top-2 comparison (`atlas-large` vs `nimbus-2`):** gap = `0.017` (95% paired CI `[-0.007, 0.039]`), McNemar p-value = `0.1817` (`significant=False`).
- **Full-benchmark rank stability:** `0.912` probability that `atlas-large` stays #1 at n=1500.

## 2. Panel Independence & Voting

- **Mean pairwise error correlation (`mean_phi`):** `0.310`
- **Effective independent models (`n_eff`):** `2.35` out of `6`
- **Full-panel plurality vote accuracy:** `0.775`
- **Majority-correct rate:** `0.645` actual vs `0.691` if models were independent

## 3. Recommended Sub-Panel

- **Selected 3-model diverse panel:** `atlas-large, borealis-xl, cirrus-7b` (dev: `0.769`, test: `0.753`)
- **Naive top-3 baseline:** `atlas-large, nimbus-2, atlas-medium` (dev: `0.700`, test: `0.708`)
