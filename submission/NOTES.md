# Notes

## Help I used

- Used an AI assistant (Gemini) to clarify the statistical concepts from the three articles (exact McNemar test, phi error correlation / effective sample size, Poisson-binomial convolution, and subliminal distillation) and to structure and verify the NumPy implementations and output formatting.
- Verified all formulas, outputs, and test cases myself against `tests/test_public.py` and `check_submission.py`.

## Three decisions I made myself

1. **Exact Poisson-binomial convolution over Monte Carlo simulation (`panel.py`):** Used `np.convolve` to compute the exact probability mass function of independent majority correctness so results are deterministic and free of sampling variance.
2. **Best-first column ordering inside `select_panel` (`panel.py`):** Pre-sorted candidate indices by `dev` accuracy before generating `itertools.combinations` so that any plurality vote tie automatically resolves in favor of the strongest model on `dev`.
3. **Auxiliary channel width experiment (`distill.py`):** Tested `n_aux` in `{1, 3, 10}` across all 5 seeds to measure whether a narrower side channel stops subliminal transfer or if shared initialization still leaks task behavior.

## What is unfinished, and what I would do with one more day

- With one more day, I would add category-aware weighted voting (down-weighting models on weak categories like `atlas-large` on SQL) and test whether adding temperature scaling or input noise regularization breaks the hidden correlation between same-family models.