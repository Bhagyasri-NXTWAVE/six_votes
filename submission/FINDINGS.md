# Findings

## Recommendation

**Yes, with changes.** ClockIN should not ship the raw six-model vote as-is; instead, ship a **three-model diverse panel** (`atlas-large`, `borealis-xl`, `cirrus-7b`) and route low-agreement questions to human review.

Three numbers decide this:
1. **Effective panel size (`n_eff = 2.35` of 6):** Average pairwise error correlation (`mean_phi = 0.310`) means six models behave like barely more than two independent voters, pulling strict-majority accuracy down from `0.692` (if independent) to `0.645` (actual).
2. **Diverse 3-model panel test accuracy (`0.753` vs `0.708`):** Picking the top three models by individual accuracy (`atlas-large`, `nimbus-2`, `atlas-medium`) actually underperforms `atlas-large` alone (`0.714`) on test because their mistakes overlap, whereas a diverse trio reaches `0.753` at half the inference cost.
3. **Top-2 statistical tie (`p = 0.182`):** The `0.017` gap between `atlas-large` (`0.714`) and `nimbus-2` (`0.697`) is not statistically significant (`95% CI [-0.007, 0.039]`), so single-model leaderboards cannot be trusted blindly.

## One thing you did not ask about

**Two models act as near-clones, and the #1 model fails specifically on SQL.** `atlas-medium` and `nimbus-2` have an error correlation of **`0.867`** (choosing the exact same wrong option on `95.2%` of shared misses), giving one model family a hidden double vote. Meanwhile, overall leader `atlas-large` drops to **`0.592`** accuracy on `sql` questions (last among all six models). ClockIN should weight votes by model diversity and category strength rather than raw headcount.

## My experiment

**Question:** Does widening the auxiliary output channel (`1`, `3`, or `10` logits) increase how much digit-classification skill transfers through pure Gaussian noise?  
**What I expected:** More auxiliary outputs would leak more teacher structure and steadily raise student test accuracy.  
**What happened:** Even `1` auxiliary output transferred **`0.860`** test accuracy from pure noise (vs `0.081` untrained), rising to **`0.926`** at `3` outputs and plateauing at **`0.929`** at `10` outputs.  
**What it means:** Models sharing a common base architecture or initialization remain strongly correlated even when filtered through narrow, unrelated interfaces—so buying "different" models from vendors that fine-tune the same base weights will not give ClockIN true voting independence.