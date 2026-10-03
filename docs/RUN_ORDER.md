# Run Order — VS Code

Exact order, top to bottom. Run each command in VS Code's integrated terminal
(`` Ctrl+` `` to open it), from the project's root folder — the folder that
directly contains `config.py`.

---

## 0. One-time setup

Open the project folder in VS Code (`File → Open Folder`), then in the terminal:

```bash
pip install -r requirements.txt
```

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

If that second command fails (no internet, firewall), do nothing — the code
falls back to a built-in sentiment lexicon automatically. Everything still runs.

---

## 1. Place the datasets

Before running anything real, put files here:

| Dataset | Path |
|---|---|
| Dreaddit | `data/dreaddit-train.csv`, `data/dreaddit-test.csv` |
| Kaggle keystroke archive, extracted | `data/Data/user 1/...`, `data/Data/user 2/...` |
| SWELL-KW, once you have it | `data/swell_kw.csv` |
| Your own collected sessions | `data/own/sessions_*.csv`, `data/own/keystrokes_*.csv` |

`data/README.md` in the project has the same table. Nothing in step 2 needs
this — only steps 3 onward do.

---

## 2. Smoke test — confirm the code runs, no data needed

```bash
python run_typing.py --source synthetic
```

```bash
python run_text.py --source synthetic
```

These use fake generated data with a planted signal, purely to prove the code
path works end to end. **Never quote these numbers anywhere** — they will look
artificially perfect (close to 100%) and mean nothing. If both commands finish
without errors and print a results table, you're set up correctly. Move on.

---

## 3. Real runs — individual pipelines

Only after step 1's data is in place.

```bash
python run_typing.py --source kaggle
```
Reads `data/Data/`, extracts keystroke features, trains and compares three
classifiers, writes `outputs/results_typing.csv` and confusion-matrix PNGs.

```bash
python run_text.py --source dreaddit
```
Reads the two Dreaddit CSVs, extracts sentiment/lexical features, trains and
compares three classifiers, writes `outputs/results_text.csv`.

Run these two independently whenever you want to test a change to just one
pipeline — they're faster than the full mid-term run below because they don't
grid-search hyperparameters.

---

## 4. Fusion mechanism check

```bash
python run_fusion.py --typing-source kaggle --text-source dreaddit --weight 0.5
```

Trains both pipelines again internally and demonstrates combining their
predictions. Prints no fused accuracy on purpose — see `docs/reports/Decision_Log.md`
item D6 for why.

---

## 5. The full mid-term run — this is the one that matters

```bash
python run_midterm.py
```

This is the command that actually produced every number in
`docs/reports/Progress_Report.md`. It does more than steps 3–4 combined:

- Grid-searches hyperparameters for all three classifiers
- Runs the typing pipeline in three configurations (keystroke-only, +
  interaction features, binary)
- Runs the cross-user generalisation test
- Writes everything to `outputs/`: `midterm_results.csv`, `midterm_report.md`,
  nine confusion matrices, feature-importance CSVs, and the trained `.joblib`
  models

Takes a few minutes. If you only run one command after setup, run this one.

---

## 6. Once you have your own collected data

`data/own/` is where `sessions_*.csv` and `keystrokes_*.csv` from
`collector/typing_collector.html` go. Once you have enough participants and
all classes have at least two sessions, use:

```bash
python run_typing.py --source own
python run_text.py --source own
```

The typing runner deliberately excludes post-task self-ratings and study
condition metadata from model inputs, so it learns from behaviour rather than
from fields that would leak the answer.

---

## Quick reference — command order only

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"

python run_typing.py --source synthetic
python run_text.py   --source synthetic

# place datasets in data/, then:

python run_typing.py --source kaggle
python run_text.py   --source dreaddit
python run_fusion.py --typing-source kaggle --text-source dreaddit
python run_midterm.py
```
