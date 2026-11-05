# Run Order — VS Code

Exact order, top to bottom. Run each command in VS Code's integrated terminal
(`` Ctrl+` `` to open it), from the project's root folder — the folder that
directly contains `config.py`.

---

## 0. One-time setup

```bash
pip install -r requirements.txt
```

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

If the second command fails (no internet, firewall), do nothing — the code
falls back to a built-in sentiment lexicon automatically. Everything still runs.

Optional, only for the transformer comparison (large download):

```bash
pip install -r requirements-transformer.txt
```

---

## 1. Place the datasets

| Dataset | Path | Status |
|---|---|---|
| Dreaddit | `data/dreaddit-train.csv`, `data/dreaddit-test.csv` | included |
| Kaggle keystroke archive, extracted | `data/Data/user 1/...`, `data/Data/user 2/...` | included |
| SWELL-KW computer-interaction features | `data/swell_kw.csv` | once access is granted |
| EmoSurv | the four CSV files named in `data/README.md` | included |
| Your own collected sessions | `data/own/sessions_*.csv`, `data/own/keystrokes_*.csv` | after collection |

---

## 2. Smoke test — confirm the code runs, no data needed

```bash
python -m unittest discover -s tests
python run_paired.py --source synthetic
python run_swell.py --source synthetic
```

These use generated data with a planted signal, purely to prove each code path
works end to end. **Never quote these numbers anywhere.** Their output goes to
`outputs/synthetic/`, away from real results.

---

## 3. The one command that matters

```bash
python run_final.py
```

Runs everything the data in `data/` supports and skips the rest, with the
reason printed:

| Step | Runs when | Writes |
|---|---|---|
| Dreaddit text + Kaggle typing (grid search) | always | `outputs/midterm_*`, confusion PNGs |
| SWELL-KW experiment | `data/swell_kw.csv` exists | `outputs/swell/` |
| Paired experiment (early / late fusion, stacking) | `data/own/` has exports | `outputs/paired/` |
| EmoSurv experiment (about 25 min; `--quick` reuses it) | its four files are in `data/` | `outputs/emosurv/` |
| Transformer results | `run_transformer.py` has been run | included in the summary |
| Demo models | always | `outputs/demo/` |

Then read `outputs/final_report.md`. Takes about a minute without own data.
`python run_final.py --quick` reuses the last Dreaddit/Kaggle run.

---

## 4. Individual experiments

```bash
python run_midterm.py                 # Dreaddit + Kaggle, grid-searched
python run_swell.py                   # SWELL-KW (or --path some_file.csv)
python run_paired.py                  # own collected data
python run_transformer.py             # optional DistilBERT comparison (~7 min on Apple GPU)
```

Single pipelines, faster, no grid search:

```bash
python run_typing.py --source kaggle      # or swell / own
python run_text.py   --source dreaddit    # or own
python run_fusion.py --typing-source kaggle --text-source dreaddit
```

`run_fusion.py` deliberately prints no fused accuracy: the public datasets are
unpaired (Decision Log D6). Real fusion results come from `run_paired.py`.

---

## 4b. Paired data and extended evaluation

```bash
python -m dataio.swell_raw      # once: downloads SWELL-KW raw logs (~400 MB)
python run_swell_paired.py      # typing vs text vs fusion on real paired data
python run_evaluation.py        # AUC, calibration, CIs, significance (up to ~1 hour)
```

Read `outputs/evaluation/report_evaluation.md`. `python run_evaluation.py --quick` only
checks that it works: its repeats are too few to quote.

---

## 4c. Final-report experiments

```bash
python run_emosurv.py                 # typing, text transfer and fusion on EmoSurv
python run_improve.py                 # per-person comparison, with order control
python run_leakage.py                 # the same data under leaky and honest splits

# needs requirements-transformer.txt; about 25 min on an Apple GPU
python run_transformer.py --split official
python run_transformer.py --split official --model roberta-base
python run_text_ensemble.py

# about 3 hours on an Apple GPU if the laptop stays awake; finished models are reused
python run_text_search.py
python run_text_search.py --report    # rewrites the report from saved predictions

# hours on CPU; finished typing tasks are reused. Delete results.csv to start again
python run_typing_search.py --perm 200
python run_text_embed.py              # needs the text search's chosen models
python run_fusion_search.py --perm 0  # add permutations only if combined wins
python run_burst_model.py
python export_extension_model.py      # rewrites extension/model.js
```

---

## 5. Collect data

Follow `docs/Data_Collection_Protocol.md`. In short: open
`collector/typing_collector.html` in Chrome, run each participant through the
four blocks, download both CSV files, put them in `data/own/`. Then:

```bash
python run_paired.py
```

or simply `python run_final.py` again.

---

## 6. The demo

```bash
python train_demo.py        # builds outputs/demo/ from the best available data
python app/server.py        # open http://127.0.0.1:8765
```

Type in the box; the page shows the typing estimate, the text estimate, the
combined estimate (and early fusion once own data exists), plus the features
each model measured. Re-run `train_demo.py` after collecting data so the demo
uses models trained on your participants. `Ctrl+C` stops the server.

To try the app's full layout before collecting, `python train_demo.py --source
synthetic` (predictions are then meaningless, and the page says so).

---

## 7. Real-time feedback — the Chrome extension

No server is needed:

1. Chrome → `chrome://extensions` → turn on **Developer mode**.
2. **Load unpacked** → select the project's `extension/` folder.
3. Pin it (puzzle-piece icon → pin **Stress Estimator**).

Type in any text box on any website and pause for 2 seconds: the toolbar badge
shows LOW / MED / HIGH. Click it for details and controls. Full guide:
`extension/README.md`.

---

## Quick reference — command order only

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
python -m unittest discover -s tests

python run_final.py
python app/server.py        # demo page; the extension needs no server

# after collecting data into data/own/ (and/or adding data/swell_kw.csv):
python run_final.py
```
