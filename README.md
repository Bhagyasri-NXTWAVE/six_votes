# Six Votes - AI Intern Assessment

Start with **ASSIGNMENT_BRIEF.pdf**. This file is only the quick reference.

Do this yourself. Your defense and interview are built around this project.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # every test fails until you write the code
```

Python 3.10 or newer. NumPy is the only library the project needs.

## Two ways to work

- **Notebook:** open `Six_Votes_Starter.ipynb` in Google Colab or Jupyter and follow it.
- **Editor and terminal:** edit the files in `sixvotes/` and use the commands below.

Either way, the code you are graded on lives in `sixvotes/`.

## Commands you will make work

```bash
python -m sixvotes audit   --predictions data/panel_predictions.csv --out outputs
python -m sixvotes distill --digits data/digits.npz --out outputs
python check_submission.py
```

## What is where

| Path | What it is | Do you edit it? |
|---|---|---|
| ASSIGNMENT_BRIEF.pdf | The assignment | Read |
| Six_Votes_Starter.ipynb | Guided notebook with worked examples and the exact steps (Colab or Jupyter) | Work in it |
| sixvotes/leaderboard.py | Part 1 | Yes |
| sixvotes/panel.py | Part 2 | Yes |
| sixvotes/distill.py | Part 3 | Yes |
| sixvotes/report.py | Part 4 | Yes |
| sixvotes/tinynet.py | Small NumPy neural network | No, but read it |
| sixvotes/io.py, cli.py | Data loading and command line | No |
| data/ | Both datasets and DATA_CARD.md | No |
| docs/OUTPUT_CONTRACT.md | Exact layout of the two JSON files | Read |
| tests/test_public.py | Self-checks | No |
| docs/READING.md | Links to the three articles | Read |
| submission/ | FINDINGS.md, NOTES.md and your one-pager | Yes |
