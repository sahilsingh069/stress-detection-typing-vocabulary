# Project Progress Report

**Stress Level Detection Using User Typing Behavior and Text Vocabulary**

Major Project · School of Computer Science, UPES Dehradun
Guided by: Dr. Souvik Dev

| | |
|---|---|
| Anshuman Tiwari | 500123925 |
| Sahil Panwar | 500124268 |
| Royal Bisht | 500124730 |

Reporting period: post-synopsis to mid-term

---

## 1. Summary

Both classification pipelines described in the synopsis have been implemented,
trained and evaluated.

The **text vocabulary pipeline works**: 76.2% accuracy and 0.762 macro F1 on
3,553 labelled documents, against a majority-class baseline of 52.3%.

The **typing behaviour pipeline does not yet reach a usable standard**, and we
have established through testing that the cause is the dataset rather than the
model. This is documented in Section 5 as a finding, not as an incomplete task.

The fusion layer is implemented and verified, but cannot be validated on real
data until a paired dataset exists. Section 6 explains why.

---

## 2. What was built

A complete, reproducible Python codebase:

| Component | Purpose |
|---|---|
| `config.py` | All paths, label mappings and model settings in one place |
| `dataio/loaders.py` | Loaders for four data sources with automatic column detection |
| `features/typing_features.py` | Keystroke events to session-level features |
| `features/text_features.py` | Text to sentiment and lexical features |
| `features/interaction_features.py` | Mouse, idle and application-switch features |
| `modeling/classifiers.py` | Three classifiers with class-imbalance handling |
| `modeling/evaluation.py` | Metrics, confusion matrices, cross-validation |
| `modeling/fusion.py` | Late-fusion combination rules |
| `run_midterm.py` | Full training run producing all reported results |
| `collector/typing_collector.html` | Browser tool for collecting our own paired data |

All results in this report are reproducible with a single command.

---

## 3. Datasets

| Dataset | Modality | Status |
|---|---|---|
| Dreaddit (2,838 + 715 Reddit posts) | Text vocabulary | Obtained, trained |
| Kaggle keystroke/mouse/app dataset | Typing behaviour | Obtained, trained, found insufficient |
| SWELL-KW (Radboud University) | Typing behaviour | Registration pending |
| Own collected data | Both, paired | Collection tool built, collection not started |

No public dataset contains both keystroke dynamics and typed text labelled for
stress. This gap is the reason two separate datasets are currently in use, and
it is also what our own data collection is intended to address.

---

## 4. Results

### 4.1 Text vocabulary pipeline (Dreaddit, n = 3,553)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | CV F1 |
|---|---|---|---|---|---|
| SVM (RBF) | 0.762 | 0.762 | 0.763 | 0.762 | 0.753 |
| Logistic Regression | 0.762 | 0.762 | 0.762 | 0.762 | 0.766 |
| Random Forest | 0.754 | 0.754 | 0.754 | 0.754 | 0.746 |

Majority-class baseline: 0.523. All three models clear it by roughly 23
percentage points. Cross-validation standard deviation is approximately 0.01,
indicating the result is stable rather than the product of a favourable split.

**Most informative linguistic features** (Random Forest importance):

| Feature | Importance |
|---|---|
| First-person pronoun ratio | 0.143 |
| Negative sentiment score | 0.143 |
| Compound sentiment score | 0.134 |
| Average word length | 0.073 |
| Positive sentiment score | 0.070 |

The prominence of first-person pronouns is an empirical finding from our own
run: stressed writing is measurably more self-referential. This supports
Objective 2 of the synopsis with evidence rather than assertion.

### 4.2 Typing behaviour pipeline (Kaggle dataset, n = 310 windows)

Three feature configurations were tested.

| Configuration | Best model | Accuracy | F1 (macro) |
|---|---|---|---|
| Keystroke features only, 3-class | SVM | 0.500 | 0.410 |
| Keystroke + interaction features, 3-class | Random Forest | 0.565 | 0.497 |
| Keystroke + interaction features, binary | Logistic Regression | 0.548 | 0.524 |

Majority-class baseline: 0.639.

Adding mouse, idle-period and application-switching features improved the
three-class macro F1 from 0.410 to 0.497. Collapsing to a binary target improved
it to 0.524. Despite both improvements, every configuration remains below the
majority-class baseline, which is why macro-averaged metrics are reported
alongside accuracy throughout.

---

## 5. Principal finding: why the typing pipeline underperforms

The Kaggle dataset contains data from **two participants**. To test whether the
model learns stress or merely individual typing habits, we trained on one
participant and tested on the other.

| Training | Testing | Accuracy | F1 (macro) |
|---|---|---|---|
| User 1 | User 2 | 0.595 | 0.389 |
| User 2 | User 1 | 0.442 | 0.418 |

Mean cross-user macro F1: **0.403**.

A model trained on one person transfers almost nothing to another. Combined
with roughly 100 self-reported labels and a class distribution of 198 High / 79
Medium / 33 Low, the model has insufficient basis to learn stress patterns that
generalise across individuals.

A further limitation: the dataset masks the identity of most keys (recorded as
`$` for privacy), so no vocabulary can be recovered from it. It cannot support
the text half of this project under any circumstances.

This is the limitation anticipated in Appendix C of our synopsis — *"variation
in typing behaviour across individuals may affect model generalization"*. It is
an argument for more participants, not for a different algorithm.

We confirmed this is not a tuning problem by running grid search across all
three classifiers and all three feature configurations. Performance did not
approach the baseline in any configuration.

---

## 6. Fusion status

The fusion module is implemented and tested. It aligns probability outputs from
both modalities onto a shared three-class space and combines them using
weighted averaging or maximum-confidence selection.

It is **not** reported with an accuracy figure, deliberately. Dreaddit and the
keystroke dataset come from different people, so no sample exists for which
both a typing measurement and a text measurement share a single ground-truth
label. Reporting a fused accuracy under those conditions would be meaningless.

Two fusion strategies are relevant:

- **Late fusion** (currently implemented): train one model per modality, combine
  their predictions. Valid with unpaired data.
- **Early fusion** (implemented, awaiting data): concatenate both feature sets
  into one vector for a single model. Requires paired data.

Early fusion has been verified on simulated paired data to confirm the code
path works. It will produce genuine results once real paired data is collected.

---

## 7. Data collection tool

A browser-based tool has been built to collect the paired dataset that no public
source provides. It is a single self-contained HTML file requiring no server or
installation.

Each participant completes four blocks: relaxed copy task, relaxed free-writing
task, time-pressured copy task, time-pressured free-writing task. Relaxed blocks
run first so the pressure condition cannot contaminate the relaxed baseline.
After each block the participant rates their stress level, intensity and
perceived time pressure.

Both task types are necessary. Copy tasks hold the text constant, isolating
keystroke dynamics. Free-writing tasks produce genuine vocabulary variation. A
copy task alone would yield no usable vocabulary signal.

The tool records key-press and key-release timestamps at millisecond resolution,
giving true dwell and flight times, and captures the typed text unmasked. It
exports two CSV files that join on session identifier and load directly into the
existing pipeline.

A consent screen states what is recorded, what is not, and that no data is
uploaded. This addresses the privacy concern raised in Appendix C of the
synopsis.

**Status:** built and tested. Collection not yet started. Target: 20–30
participants.

---

## 8. Current position

**Working:** text vocabulary classification at 76.2%, validated and stable.

**Not working:** typing behaviour classification, with a documented and tested
explanation.

**Built but unvalidated:** fusion layer, pending paired data.

**Pending external action:** SWELL-KW dataset access, participant recruitment.

---

## 9. Artefacts available for inspection

- `midterm_results.csv` — every model and configuration tested
- `midterm_report.md` — auto-generated results summary
- `cross_user_generalisation.csv` — the generalisation test in Section 5
- `feature_importance_*.csv` — feature rankings per configuration
- Nine confusion matrices (PNG)
- Fifteen trained model files
- Complete source code and the data-collection tool
