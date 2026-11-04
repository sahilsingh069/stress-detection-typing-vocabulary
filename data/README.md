# Data Directory

Datasets are local inputs and are not committed to the repository. Place them here:

| Dataset | Put it at |
|---|---|
| Dreaddit | `dreaddit-train.csv`, `dreaddit-test.csv` |
| Kaggle keystroke archive (extracted) | `Data/user 1/`, `Data/user 2/` |
| SWELL-KW computer-interaction features | `swell_kw.csv` |
| EmoSurv | `Fixed Text Typing Dataset.csv`, `Free Text Typing Dataset.csv`, `Frequency Dataset.csv`, `Participants Information.csv` |
| Own collected sessions | `own/sessions_*.csv`, `own/keystrokes_*.csv` |

## SWELL-KW

Openly available (no registration) from the DANS Data Station, licence
CC BY-NC-SA 4.0 — cite Koldijk et al. (2014) wherever it is used:
<https://doi.org/10.17026/dans-x55-69zp>

The project uses one file:
`3 - Feature dataset/per sensor/A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv`
(direct download: <https://ssh.datastations.nl/api/access/datafile/189419>),
saved here as `swell_kw.csv`:

```bash
curl -L -o data/swell_kw.csv https://ssh.datastations.nl/api/access/datafile/189419
```

`swell_raw/` holds the same file plus the original workbook for reference. `python -m
dataio.swell_raw` adds the raw keystroke logs (~400 MB) and per-block questionnaire used by
`run_swell_paired.py`.

Use the **computer-interaction feature file** (one row per minute, columns such as `PP`,
`Condition`, `SnKeyStrokes`, `SnErrorKeys`, `SnMouseDistance`, `SnAppChange`). CSV, TSV or
XLSX all load; rename it to `swell_kw.csv` or run `python run_swell.py --path <file>`.

- Participant column: `PP` (or `participant`, `subject`). Needed so evaluation can hold out
  whole people.
- Condition column: `Condition` (or `C`) with N / I / T. Rest periods (R) are dropped.
- Questionnaire columns (Stress, Valence, NASA-TLX, ...) are ignored on purpose: they are the
  label in another form.
- The heart-rate (HRV) derivative of SWELL found on Kaggle has no typing data; the loader
  refuses it with an explanation.

## EmoSurv

From IEEE DataPort: <https://doi.org/10.21227/eae6-pk42>. Non-commercial research use with
no right to redistribute, so never commit it. Cite Maalej and Kallel (2020).

Its labels are emotions (neutral, anger, happiness, calmness, sadness), not stress. Neutral
is always first and each emotion has its own fixed passage. `dataio/emosurv.py` lists these
traps and how each is handled.

## Own data

`own/` is git-ignored on purpose: it holds participants' raw keystrokes and typed text.
Both files from every participant are needed; they join on `session_id`. Both collector
versions load (v1 free-writing, v2 chat).
