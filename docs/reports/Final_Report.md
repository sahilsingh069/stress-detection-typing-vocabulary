# Stress Level Detection Using User Typing Behaviour and Text Vocabulary

**Major Project Report**

School of Computer Science, University of Petroleum and Energy Studies, Dehradun
Guided by: Dr. Souvik Dev

| Name | SAP ID |
|---|---|
| Anshuman Tiwari | 500123925 |
| Sahil Panwar | 500124268 |
| Royal Bisht | 500124730 |

> **Scope.** All results come from public data. Our own participant data had not been
> collected at the time of writing, so the collector is delivered and tested but has
> produced no results here. Adding its exports to `data/own/` and re-running
> `python run_final.py` adds them.

---

## Abstract

Stress is usually measured with questionnaires or physiological sensors,
which are occasional or intrusive. This project estimates a person's stress
level (Low / Medium / High) from two signals produced during ordinary
typing: *how* they type (keystroke timing, corrections, pauses, mouse and
application switching) and *what* they type (sentiment and vocabulary). We
built a reproducible pipeline that evaluates both modalities with Logistic
Regression, Random Forest and SVM, a browser tool that collects paired typing
and text data under relaxed and time-pressured conditions, a fusion layer
(early fusion, late fusion and stacking), and a local web demonstration.

On Dreaddit's official test split (715 segments) the interpretable text pipeline
reaches 76.1% accuracy and 0.761 macro F1 against a 51.6% majority baseline. An
ensemble of DeBERTa and ELECTRA, chosen from five transformers on a development set,
reaches 81.5% and 0.814 (95% CI 0.782 to 0.845; ROC-AUC 0.900), a gain of 0.054 over the
interpretable model (95% CI 0.026 to 0.085). The strongest linguistic predictors are
negative sentiment and first-person pronoun rate.
On a public two-participant keystroke dataset the typing pipeline stays below
its majority baseline, and a cross-user test (macro F1 0.401) shows the model
learns individuals rather than stress — which motivated participant-grouped
evaluation throughout. On SWELL-KW (25 participants), keyboard features separate
neutral from time-pressured work in unseen people with macro F1 0.581 (95% CI
0.539 to 0.625; chance 0.500; permutation test significant after correction), a
weak signal that does not beat the majority-class accuracy. Typing and text
rebuilt from SWELL-KW's raw keystroke logs give a paired test of fusion on real
data (23 participants): no stress task survives correction for multiple tests, and
fusion shows no measurable gain over a single modality.

We added EmoSurv (about 80 people, emotion rather than stress labels) as a second
paired dataset. The Dreaddit model, applied with no retraining, separates negative from
positive emotions in new people's writing (ROC-AUC 0.669, 95% CI 0.545 to 0.779), so the
text signal carries beyond Reddit. Typing on the pre-specified anger vs. calm task is
near chance (macro F1 0.519), and fusion again gives no gain.

A final search used nested cross-validation, so that choosing among options happened on
training people only. Per block on SWELL-KW, keystrokes alone reach macro F1 0.506 on three
classes (chance 0.333; corrected p = 0.02), and with mouse and application data 0.764 on
neutral vs. time pressure (chance 0.500; corrected p = 0.01). Both standardise within the
person, so they need that person's own unlabelled typing, and neutral was always the first
block. An earlier 0.776 from keystrokes alone fell to 0.625 (p = 0.42) under this stricter
test. Across 12 nested comparisons, combining typing and text never measurably beat the
better single stream. On short bursts, as the live extension sees them, typing is at chance.

Finally, we scored the same data the way much of the literature does. Random folds over
overlapping windows lift SWELL-KW from macro F1 0.524 to 0.816 with no better model, which
accounts for much of the gap between published scores and ours.

---

## 1. Introduction

Prolonged, undetected stress reduces productivity and harms health. Existing
measurement relies on self-report questionnaires (subjective, infrequent) or
physiological sensors (costly, intrusive). Typing is something most people
already do for hours a day, and earlier work has linked both keystroke
dynamics (Vizer, 2009; Sağbaş et al., 2020; Ghosh et al., 2017) and word
choice (Turcan and McKeown, 2019) to stress. This project combines the two.

## 2. Problem statement

Build a low-cost, non-intrusive system that classifies a typing session into
Low, Medium or High stress from keystroke behaviour and the text typed,
without extra hardware, and evaluate it honestly on people it has not seen.

## 3. Objectives

1. Identify typing-behaviour features that correlate with stress.
2. Identify text-vocabulary features that indicate stress.
3. Build classifiers on each modality and on the combination.
4. Build a lightweight system that shows a stress estimate with a confidence.
5. Evaluate with accuracy, precision, recall, F1 and confusion matrices.

## 4. Datasets

| Dataset | Modality | Size | Labels | Role |
|---|---|---|---|---|
| Dreaddit (Turcan and McKeown, 2019) | Text | 3,553 posts | stress / no stress | Text pipeline |
| Kaggle "Stress Detection by Keystroke, App & Mouse Changes" | Typing + interaction | 2 participants, 310 five-minute windows | self-report, 198 High / 79 Medium / 33 Low | Pipeline validation |
| SWELL-KW (Koldijk et al., 2014) | Computer interaction | 25 participants, 2,688 working minutes | condition: neutral / interruption / time pressure | Typing pipeline |
| SWELL-KW raw logs (same source) | Typing + text, **paired** | 23 English-writing participants, 263 five-minute Word windows | condition | Fusion on real data |
| EmoSurv (Maalej and Kallel, 2020) | Typing + text, **paired** | 81 people on free text, 83 on fixed text; 189 and 195 blocks | induced emotion: neutral, anger, happiness, calmness, sadness | Second test of typing, text transfer and fusion |
| Own collected sessions | Typing + text, **paired** | not collected | self-report per block | Collector delivered and tested |

We found no public dataset released as paired typing and text under a stress
label. SWELL-KW comes closest: its raw logs record every key with a timestamp and
its character, so both modalities can be rebuilt from one stream (Section 7.4).
Compared with our collector design it has no key-hold times, condition labels
rather than per-sample self-reports, and writing tasks rather than conversation.

## 5. System design

```
 browser (collector / demo)                Python
 ┌──────────────────────────┐   CSV /    ┌──────────────────────────────────────────┐
 │ key down / key up times  │   JSON     │ typing_features  →  typing model ─┐      │
 │ typed text               │ ─────────► │ text_features    →  text model  ──┼─ late fusion
 │ self-rated stress        │            │ (both)           →  early-fusion model    │
 └──────────────────────────┘            └──────────────────────────────────────────┘
```

| Module (synopsis) | Implementation |
|---|---|
| Data capture | `collector/typing_collector.html`, `app/static/index.html` |
| Feature extraction | `features/typing_features.py`, `features/interaction_features.py`, `features/text_features.py` |
| Preprocessing | `dataio/loaders.py`, imputation and scaling inside each model pipeline |
| ML classification | `modeling/classifiers.py`, `modeling/fusion.py` |
| Result display | `app/server.py` (demo web app), `extension/` (live badge in Chrome, no server) |

The demo computes features with the same Python functions as the experiments. The
extension runs the same models in JavaScript, exported by `export_extension_model.py`, and a
test checks that its features and probabilities match Python's.

## 6. Methodology

### 6.1 Typing features
Dwell time (key hold), flight time (gap between key presses) and their
spread, typing speed, correction (backspace) rate, special-key rate,
long-pause rate (gaps over 2 s) and rhythm variability (coefficient of
variation of gaps). On the Kaggle and SWELL data, mouse speed, idle periods
and application switches are added. Rates rather than counts make sessions of
different lengths comparable.

### 6.2 Text features
VADER sentiment (Hutto and Gilbert, 2014), negative-, stress- and
positive-word ratios, negation ratio, first-person pronoun ratio, lexical
diversity, average word length, punctuation and capitalisation ratios, plus
TF-IDF over word unigrams and bigrams fitted inside each training fold.

### 6.3 Classifiers
Logistic Regression (linear baseline), Random Forest (non-linear, gives
feature importances) and RBF-kernel SVM, all with class-balanced weights
(scikit-learn; Pedregosa et al., 2011). Hyperparameters are grid-searched on
training folds only.

### 6.4 Fusion
- **Early fusion:** typing and text features of the same session in one
  vector, one classifier.
- **Late fusion:** one model per modality; probabilities combined by a
  weighted average (fixed typing weight 0.5) or by trusting the more
  confident model.
- **Stacking:** a logistic-regression meta-classifier over both models'
  out-of-fold probabilities, itself scored out of fold.

Early fusion and stacking require paired samples and are evaluated only on
our own data. On the unpaired public datasets only the late-fusion mechanism
is demonstrated, and no fused accuracy is reported because no fused ground
truth exists.

### 6.5 Evaluation protocol
Accuracy, macro precision, recall and F1, confusion matrices, and the
majority-class baseline for every model. Wherever participants are
identifiable, test data consists of **whole held-out participants**
(stratified group k-fold): the Kaggle cross-user test (Section 7.2) showed
that random splits let a model recognise people instead of stress. On SWELL-KW
a per-participant calibrated variant is reported separately. On our own data
the models see only an allow-list of inputs; condition, time limits,
durations, raw counts and self-ratings never reach them.

### 6.6 Data collection
A single-file browser tool, used offline. Four blocks per participant:
relaxed copy task; relaxed chat (4 everyday questions); time-pressured copy
task (75 s); time-pressured chat (30 s per answer). The participant rates
stress (Low/Medium/High), intensity (1–5) and perceived time pressure (1–5)
after each block. Relaxed blocks run first so pressure cannot carry over into
the baseline. Questions come from one neutral pool of eight and copy passages
are matched; both are randomly assigned to conditions per participant, so
the only systematic difference between conditions is time pressure. Paste is
disabled; key-down and key-up times are recorded at millisecond resolution;
nothing is uploaded. The procedure is in `docs/Data_Collection_Protocol.md`.

## 7. Results

### 7.1 Text pipeline (Dreaddit, 80/20 stratified split, n = 3,553)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | CV macro F1 |
|---|---|---|---|---|---|
| SVM (RBF) | 0.762 | 0.762 | 0.763 | 0.762 | 0.753 |
| Logistic Regression | 0.762 | 0.762 | 0.762 | 0.762 | 0.766 |
| Random Forest | 0.754 | 0.754 | 0.754 | 0.754 | 0.746 |
| DistilBERT, fine-tuned 2 epochs (comparison) | 0.795 | 0.795 | 0.793 | 0.794 | — |

Majority baseline 0.523. The three classical models, structurally
different, agree within 1%, which points to a feature ceiling rather than a
tuning one; the transformer adds 3.1 points of macro F1 at the cost of
interpretability.

Most informative linguistic features (Random Forest importance): negative
sentiment 0.143, **first-person pronoun ratio 0.143**, compound sentiment
0.135, average word length 0.072, positive sentiment 0.071. Stressed writing
is measurably more self-referential.

### 7.2 Typing pipeline (Kaggle, 2 participants, n = 310 windows)

| Configuration | Best model | Accuracy | Macro F1 |
|---|---|---|---|
| Keystroke only, 3-class | SVM | 0.500 | 0.406 |
| Keystroke + interaction, 3-class | Random Forest | 0.629 | 0.553 |
| Keystroke + interaction, binary | Random Forest | 0.581 | 0.532 |

Majority baseline 0.639. Interaction features (mouse, idle time, application
switching) add 0.15 macro F1 — the strongest single features are
background-application count and mouse-speed statistics — but no
configuration beats the baseline. Training on one participant and testing on
the other gives macro F1 0.396 and 0.405 (mean 0.401). With two people the
model learns individual habits, not stress; this is the generalisation risk
listed in the synopsis (Appendix C), now measured.

### 7.3 Typing pipeline (SWELL-KW, 25 participants, n = 2,688 minutes)

Per-minute computer-interaction features (16; keyboard-only subset: 7). Labels
from the condition; rest periods dropped. Every figure is for **participants
held out** (5-fold stratified group CV); best of LR / RF / SVM, which is slightly
optimistic. Section 7.6 gives intervals and tests for one model fixed in advance.

| Task | Features | Macro F1 | Chance F1 | Accuracy | Majority acc |
|---|---|---|---|---|---|
| 3-class (N / I / T) | all interaction | 0.424 | 0.333 | 0.437 | 0.382 |
| Binary (N vs. I or T) | all interaction | 0.572 | 0.500 | 0.593 | 0.618 |
| Neutral vs. time pressure | all interaction | 0.596 | 0.500 | 0.603 | 0.608 |
| Neutral vs. time pressure | keyboard only | 0.581 | 0.500 | 0.593 | 0.608 |

The interruption condition sends e-mails that must be opened and answered,
so app switches (35 vs. 3 per minute) and error keys rise by design; a model
can recognise that condition without recognising stress. The **neutral vs.
time pressure** task avoids this because the work is the same kind, but it is not
a clean test either: neutral was always the first block, so it also compares early
with late in the session. Interaction features beat chance macro F1 by
about 0.1 on unseen people, and keyboard features alone retain most of that.
Accuracy stays at the majority level, so the signal is weak. Random-split
evaluation would have reported 0.47 instead of 0.42 on the 3-class task (person
leakage), and per-participant calibration did not help (0.415). Leave-one-
participant-out macro F1 ranges from 0.28 to 0.56 across people: typing
responses to stress differ substantially between individuals.

### 7.4 Paired data: single modalities versus fusion (SWELL-KW raw logs)

Key timing and typed words were rebuilt from the same raw logs, in 5-minute Word
windows (at least 100 key presses and 20 words; 23 English-writing participants,
263 windows). Word only, because PowerPoint is nearly absent from the neutral block
and e-mail belongs to the interruption block. These rules were fixed before any
model was run. Logging artefacts were removed (each Space is logged twice).

**Manipulation check.** Participants rated time pressure higher under the deadline
(neutral 5.74, time pressure 7.06; paired Wilcoxon p = 0.012), but self-rated stress
did not differ significantly (2.89 vs. 3.84; p = 0.119). The condition therefore
changed perceived pressure more clearly than perceived stress, which limits what
any model trained on these labels can show.

Macro F1, Logistic Regression fixed in advance, whole people held out:

| Task | Chance | Typing | Text | Early fusion | Late fusion |
|---|---|---|---|---|---|
| Neutral vs. time pressure | 0.500 | 0.570 | 0.505 | 0.566 | 0.519 |
| Neutral vs. either stressor | 0.500 | 0.515 | 0.472 | 0.499 | 0.490 |
| Interruption vs. time pressure | 0.500 | 0.604 | 0.549 | 0.601 | 0.573 |
| 3-class | 0.333 | 0.397 | 0.371 | 0.370 | 0.375 |

No result here is significant after correcting for 12 tests (Section 7.6). Every
95% interval for fusion minus a single modality on these stress tasks includes zero,
so **the data shows no fusion gain**. With 145 to 263 windows the intervals are
wide, so this does not prove fusion is useless. Text carries little signal here,
unlike on Dreaddit, plausibly because participants wrote about assigned topics
rather than about themselves.

### 7.5 Demonstration system and real-time feedback
`python app/server.py` serves a local page: the user types, and the page
shows the typing estimate, the text estimate, the fused estimate (and early
fusion once paired data exists), each with class probabilities, a confidence,
the training source of each model and whether it is validated, and the
measured features. Nothing typed is stored.

For feedback during normal work (synopsis Objective 4), a Chrome extension
observes typing in text fields on any website, analyses each burst of typing
(after a 2-second pause) inside the browser, and shows a smoothed
LOW / MED / HIGH badge on the toolbar, with details in a popup and an
optional break suggestion when stress stays high. It needs no server and makes no network
requests, which an end-to-end browser test confirms. Password, email, card and
one-time-code fields are never read, and only estimates — never text — are kept. A system-wide desktop
logger was rejected on privacy grounds (Decision Log D20). A local dashboard shows
the stored estimates over any period in the last 30 days: an average score, a timeline, a
ranking of sites that filters the page, a weekday and time-of-day heatmap, and plain sentences
about differences of at least 10 points on at least 3 estimates. Thin data is marked and never
called out (Decision Log D22, D30). An estimate takes about 1.5 ms in the browser for a
300-key burst, against a median of 22 ms on the earlier server, on our development laptop.

The extension uses the interpretable text model (macro F1 0.761), because the transformers are
too large to ship in it. Its text model is binary and text has 0.75 of the weight, so with both
signals present a burst's label is always Low or High: Medium can appear only for typing alone
(Section 7.14).

### 7.6 Extended evaluation

`python run_evaluation.py` adds balanced accuracy, MCC, Cohen's kappa, ROC-AUC, PR-AUC,
log loss, Brier score, calibration error, per-class results, bootstrap confidence
intervals, permutation tests, McNemar tests, repeated cross-validation, per-person
spread and learning curves (`outputs/evaluation/`, with ROC, calibration, confusion and
learning-curve charts). To avoid picking a winner on the test folds, the headline model
is Logistic Regression, fixed in advance.

| Dataset and task | Macro F1 [95% CI] | Chance | ROC-AUC [95% CI] | Accuracy | Majority | Permutation p (Holm) |
|---|---|---|---|---|---|---|
| Dreaddit, whole posts held out | 0.765 [0.749, 0.777] | 0.500 | 0.848 [0.834, 0.859] | 0.765 | 0.523 | 0.005 |
| Dreaddit, official split | 0.760 [0.729, 0.789] | 0.500 | 0.853 [0.823, 0.878] | 0.761 | 0.516 | not run |
| SWELL minutes, N vs. T, all features | 0.573 [0.527, 0.616] | 0.500 | 0.616 [0.560, 0.670] | 0.580 | 0.608 | 0.023 |
| SWELL minutes, N vs. T, keyboard only | 0.581 [0.539, 0.625] | 0.500 | 0.594 [0.540, 0.652] | 0.593 | 0.608 | 0.003 |
| SWELL minutes, I vs. T | 0.549 [0.500, 0.599] | 0.500 | 0.569 [0.513, 0.629] | 0.555 | 0.600 | 0.058 |
| SWELL minutes, 3-class | 0.420 [0.393, 0.446] | 0.333 | 0.585 [0.557, 0.616] | 0.429 | 0.382 | 0.002 |

Bootstrap intervals resample whole people (or rows for Dreaddit's official split).
Permutation tests move labels between a person's blocks as whole units. Holm correction
applies within each dataset. The Dreaddit test used 200 shuffles, so 0.005 is the lowest
value it can report; SWELL tests used 2,000.

- **Dreaddit is robust.** Holding out whole posts (0.765) or using the authors' own split
  (0.760) matches the random split (0.763), so segments of one post appearing on both
  sides did not inflate the result. Logistic Regression is well calibrated (error 0.014).
  McNemar tests find no difference between the three classifiers (p from 0.67 to 0.92).
- **Order control.** A model cannot tell block 2 from block 3 better than chance (0.463,
  p = 0.84), so position alone is not detectable among the later blocks. This does not
  rule out a difference between block 1 and later blocks, which the design confounds
  with the neutral condition.
- **Stable across folds.** Repeating with five fold assignments moves SWELL macro F1 by
  an SD of about 0.005 (per-minute) and 0.01 to 0.03 (paired).
- **People differ a lot.** Per-person accuracy for N vs. T ranges from 0.39 to 0.84
  (keyboard features).
- **More people may help little.** On SWELL minutes, macro F1 is nearly flat from 10
  training people (0.566) to 20 (0.563), so better features or labels look more useful
  than more of the same data. The paired curves still rise slightly up to 15 people,
  with wide spread.

### 7.7 A stronger text model (Dreaddit, official split)

The authors' split keeps every post on one side. The ensemble's equal-weight average was
declared as the headline before RoBERTa or the ensemble was scored. The classical model and
DistilBERT already had scores on this test set, so their selection is not fully blind.

| Model | Accuracy | Macro F1 [95% CI] | ROC-AUC | Calibration error |
|---|---|---|---|---|
| Interpretable (TF-IDF + linguistic) | 0.761 | 0.761 [0.727, 0.792] | 0.853 | 0.016 |
| DistilBERT, 2 epochs | 0.792 | 0.790 [0.758, 0.821] | 0.881 | 0.064 |
| RoBERTa, 2 epochs | 0.815 | 0.813 [0.781, 0.844] | 0.902 | 0.083 |
| **Ensemble of the three** | **0.810** | **0.808 [0.775, 0.838]** | **0.898** | **0.027** |

Majority baseline 0.516. The ensemble beats the interpretable model (+0.048, 95% CI 0.022
to 0.075; McNemar p < 0.001). It does not measurably differ from RoBERTa (-0.005, 95% CI
-0.023 to 0.014) and its gain over DistilBERT is borderline (+0.018, 95% CI 0.000 to
0.036). RoBERTa's slightly higher figure is not adopted, since that would pick a winner
on the test set; the ensemble's probabilities are also better calibrated. Two other options
did not help: character n-grams chosen by cross-validation on the training set scored 0.734
on the test set, and adding the 105 LIWC, DAL and syntax columns Dreaddit ships scored 0.753.
LIWC is licensed and could not run on live typing anyway. The demo keeps the interpretable
model, which is small, fast and explainable.

### 7.8 EmoSurv: a second paired dataset

EmoSurv's labels are **emotions, not stress**; anger vs. calm is the closest proxy. Neutral
always came first, and each emotion has its own fixed passage, so a model can recognise the
passage rather than the emotion (macro F1 on active vs. passive is 0.723 on fixed text and
0.534 on free text). Free text is therefore the primary data. The design was fixed before
any model ran (`run_emosurv.py`), except that repeated folds and the text significance test
were added after a smoke test showed how unstable small samples are. Scores below use whole
people held out.

| Task (free text) | Chance | Typing, primary model [95% CI] | Mean over 20 fold assignments | Best of 4 typing options, Holm p |
|---|---|---|---|---|
| Anger vs. calm | 0.500 | 0.519 [0.368, 0.647] | 0.542 | 0.679, p = 0.114 |
| Negative vs. positive | 0.500 | 0.555 [0.465, 0.635] | 0.521 | 0.619, p = 0.146 |
| Active vs. passive | 0.500 | 0.534 [0.426, 0.628] | 0.540 | 0.595, p = 0.189 |
| Four emotions | 0.250 | 0.314 [0.231, 0.399] | 0.286 | 0.418, p = 0.008 |

The primary model subtracts the person's neutral block; on this data that was not better
than raw features. The only significant typing result is a gradient-boosted model on raw
features, chosen as best of four, on the four-emotion task.

**Text transfers.** The Dreaddit stress model, with nothing fitted on EmoSurv, gives
ROC-AUC 0.669 [0.545, 0.779] for negative vs. positive and 0.677 [0.511, 0.833] for anger vs.
calm. Its mean stress probability is 0.525 on negative emotions and 0.332 on positive ones.

**Fusion** (same samples; fewer because text needs at least 5 words; mean macro F1 over 20
fold assignments):

| Task | Chance | Typing | Text | Early fusion | Late fusion | Text p (Holm) |
|---|---|---|---|---|---|---|
| Anger vs. calm (n = 42) | 0.500 | 0.669 | 0.680 | 0.675 | 0.673 | 0.062 |
| Negative vs. positive (n = 93) | 0.500 | 0.518 | 0.678 | 0.554 | 0.563 | 0.008 |
| Active vs. passive (n = 93) | 0.500 | 0.608 | 0.571 | 0.612 | 0.612 | 0.452 |
| Four emotions (n = 93) | 0.250 | 0.318 | 0.433 | 0.335 | 0.344 | 0.008 |

On the single pre-set fold assignment, every interval for late fusion minus the better single
modality includes zero except negative vs. positive, where fusion is worse (-0.134, 95% CI
-0.263 to -0.007). Typing on anger vs. calm scores 0.542 on 50 blocks but 0.669 on the 42 with
enough words: with samples this small, such a change is not evidence of a real effect.

### 7.9 Comparing each person with themselves (SWELL-KW raw logs)

Options listed in `run_improve.py` before running: 8 base features or a richer set (gap
distribution, bursts, pauses, corrections), raw or standardised within each person, with
Logistic Regression or gradient boosting. Scored per 5-minute window and per block (one
person's whole condition). Significance compares the best option with the best option on
shuffled labels, so trying several cannot inflate it.

| Task | Chance | Previous, per window | Best per window (p) | Best per block (p) |
|---|---|---|---|---|
| Neutral vs. time pressure | 0.500 | 0.570 | 0.624 (0.044) | 0.776 (0.020) |
| Neutral vs. either stressor | 0.500 | 0.515 | 0.591 (0.066) | 0.694 (0.030) |
| 3-class | 0.333 | 0.397 | 0.430 (0.040) | 0.525 (0.036) |
| Order control: block 2 vs. 3 | 0.500 | 0.476 | 0.535 (0.75) | 0.575 (0.63) |

Corrected across the three stress tasks (Holm) the p-values are 0.12 per window and 0.06 per
block, so this is **promising, not established**. The order control finds nothing, as it
should, though it cannot rule out a difference between the first block and later ones, and
neutral was always first. Per-person standardisation needs a stretch of that person's own
unlabelled typing, and a block needs a long period of typing, so it suits a daily or weekly
dashboard rather than a live reading. On unseen people with no such history the figures remain
those of Section 7.6. Section 7.12 repeats this with the choice of option nested inside the
evaluation, and the 0.776 does not hold.

### 7.10 Why published scores are higher than ours

The same data and an untuned Random Forest (`run_leakage.py`). **Leaky rows are shown only to
measure the gap; they are not results of this project.**

| Data | Evaluation | Features | Accuracy | Macro F1 | Majority |
|---|---|---|---|---|---|
| SWELL-KW, N vs. T | Random 10-fold, overlapping windows | timing | 0.830 | 0.816 | 0.641 |
| SWELL-KW, N vs. T | Random 10-fold, separate windows | timing | 0.616 | 0.574 | 0.637 |
| SWELL-KW, N vs. T | **Whole people held out** | timing | **0.573** | **0.524** | 0.637 |
| EmoSurv fixed text, 5 states | Random 10-fold, overlapping windows | timing + keys | 0.993 | 0.992 | 0.612 |
| EmoSurv fixed text, 5 states | Whole people held out | timing + keys | 0.961 | 0.950 | 0.601 |
| EmoSurv fixed text, 5 states | **Whole people held out** | timing | **0.512** | **0.329** | 0.601 |
| EmoSurv free text, 5 states | **Whole people held out** | timing | **0.452** | **0.318** | 0.566 |

Two separate effects appear. Overlapping windows on both sides of a split let a model match a
test window to nearly identical training windows. And key composition lets it recognise the
passage: holding out people does not remove that, so the 0.950 reflects passage identity, not
emotion, and the score collapses when only timing remains.

| Study | Signal | Reported | Evaluation | Comparable? |
|---|---|---|---|---|
| Pavić et al. (2026), 86 students | Keystrokes, self-report | CV accuracy 83%; F1 0.65 and 0.79 | Same student may be in different folds, stated by the authors | Closest; its per-class F1s average 0.72, and its recalls imply about 74% test accuracy |
| Darmawan et al. (2021), 46 people | Smartphone motion sensors + typing | Accuracy 94.77% | 10-fold; no person-independent split described; calm always first | No: different sensors |
| Kene and Thakare (2021) | ECG | Accuracy 83.33% | 70/30 split | No: physiological. Sensitivity and specificity of 75% each imply 75% accuracy, not 83.33% |
| **This project** | Keystrokes | N vs. T macro F1 0.581 (per minute), 0.625 per block with person history (not significant) | Whole people held out | |
| **This project** | Text, Dreaddit | Macro F1 0.814 | Authors' official split | |

We make no claim to outperform these studies, or that they are wrong; their setups differ. The
comparison shows that a headline number cannot be read without knowing how the data was split.

### 7.11 Text model search (Dreaddit, official split)

`run_text_search.py` holds out 15% of the training posts as a development set (445 segments).
Candidates ran for 3 epochs at a learning rate of 2e-5, and the best epoch was chosen on the
development set. Models were then added greedily while an equal-weight average improved there.
The chosen models were retrained on all training posts with 3 seeds, and the test set was scored
once. It is not an untouched test set, because Section 7.7 had already used it.

| Model | Dev macro F1 |
|---|---|
| **DeBERTa v3 small + ELECTRA (chosen)** | **0.818** |
| DeBERTa v3 small | 0.809 |
| ELECTRA | 0.809 |
| RoBERTa | 0.804 |
| DistilBERT | 0.802 |
| BERTweet (reads at most 128 tokens) | 0.785 |
| Interpretable | 0.726 |

mental-RoBERTa was planned but not tested, because its download requires approved access.

| On the test set | Accuracy | Macro F1 [95% CI] | ROC-AUC | Calibration error |
|---|---|---|---|---|
| **Ensemble chosen on dev** | **0.815** | **0.814 [0.782, 0.845]** | **0.900** | **0.098** |
| ELECTRA alone | 0.827 | 0.825 [0.793, 0.855] | 0.900 | 0.114 |
| DeBERTa alone | 0.804 | 0.803 [0.771, 0.832] | 0.893 | 0.107 |
| Earlier ensemble (Section 7.7) | 0.810 | 0.808 [0.775, 0.838] | 0.898 | 0.027 |
| Interpretable | 0.761 | 0.761 [0.727, 0.792] | 0.853 | 0.016 |

The ensemble beats the interpretable model (+0.054, 95% CI 0.026 to 0.085) but not measurably
the earlier ensemble (+0.006, 95% CI -0.012 to 0.025), and its probabilities are less well
calibrated. ELECTRA's higher figure is not adopted, since that would pick a winner on the test
set, and it does not measurably differ from the ensemble (+0.011, 95% CI -0.003 to 0.024). This
is close to a ceiling for these methods on this data: five more models moved the score by less
than its uncertainty. If probabilities matter more than labels, the earlier ensemble is the
better choice.

### 7.12 Typing search with nested selection

`run_typing_search.py` offers many options: 120- or 300-second windows, five feature groups
(including rhythm and letter-pair timing), raw or standardised within the person, Logistic
Regression or gradient boosting. Inner folds choose an option on training people only, then the
held-out people are scored, so trying many options cannot inflate the score. Units are a person's
block, 10 repeats are averaged, and the permutation test (200 shuffles) reruns the whole
procedure, so 0.005 is the lowest p it can report.

| Data and task | Chance | Macro F1 [95% CI] | ROC-AUC | p | Holm p |
|---|---|---|---|---|---|
| **Keystrokes, 3-class** | 0.333 | **0.506 [0.375, 0.599]** | 0.685 | 0.005 | **0.020** |
| Keystrokes, neutral vs. either stressor | 0.500 | 0.646 [0.516, 0.782] | 0.737 | 0.025 | 0.075 |
| Keystrokes, neutral vs. time pressure | 0.500 | 0.625 [0.457, 0.775] | 0.700 | 0.209 | 0.418 |
| Keystrokes, interruption vs. time pressure | 0.500 | 0.514 [0.351, 0.646] | 0.523 | 0.214 | 0.418 |
| Order control: block 2 vs. 3 | 0.500 | 0.376 [0.263, 0.524] | 0.299 | 0.970 | not corrected |
| **Mouse and apps included, neutral vs. time pressure** | 0.500 | **0.764 [0.620, 0.880]** | 0.789 | 0.005 | **0.010** |
| **Mouse and apps included, 3-class** | 0.333 | **0.585 [0.469, 0.734]** | 0.765 | 0.005 | **0.010** |
| EmoSurv free text, anger vs. calm | 0.500 | 0.652 [0.490, 0.766] | 0.768 | 0.070 | 0.209 |
| EmoSurv free text, active vs. passive | 0.500 | 0.609 [0.510, 0.717] | 0.668 | 0.020 | 0.080 |
| EmoSurv free text, negative vs. positive | 0.500 | 0.574 [0.477, 0.688] | 0.647 | 0.488 | 0.488 |
| EmoSurv free text, four emotions | 0.250 | 0.370 [0.305, 0.496] | 0.622 | 0.129 | 0.259 |

Intervals are computed on probabilities averaged over repeats, so a mean can sit near an edge.

- **What held up.** Keystrokes alone separate the three conditions per block on unseen people, and
  mouse and application data add more, almost always using the per-minute features standardised
  within the person. No EmoSurv task survives correction.
- **What did not hold.** The 0.776 of Section 7.9 was the best of a few options. Nested, the same
  task gives 0.625 and is not significant.
- **What the 3-class score means.** Interruption and time pressure cannot be told apart from
  keystrokes (0.514), so the 3-class score mostly reflects neutral against stress. Neutral was
  always block 1. The order control finds nothing between blocks 2 and 3, but it cannot rule out a
  first-block effect such as warming up.
- **Mouse and application data.** The 0.764 is not achievable from a browser extension, which sees
  no such data, and it may partly reflect how the tasks differ rather than stress.
- **Not a live score.** These need a person's whole block and their own unlabelled typing.
- **Remaining optimism.** Standardising within a person uses that person's own unlabelled typing, and
  the shared letter-pair table includes the held-out people's timing. Neither uses labels, so the
  permutation test stays valid, but a model meeting a new person could not do this. These options
  were also designed after the earlier results had been seen, on the same 25 people.

### 7.13 Fusion search with nested selection

`run_fusion_search.py` runs three competitors on the same units: typing options, text options
(TF-IDF, sentence embeddings, pretrained emotion, sentiment and Dreaddit stress scores), and
combined, which offers all of those plus weighted averages (typing weight 0.25, 0.5 or 0.75), so it
can fall back to one stream. Permutation tests were skipped because they were needed only if
combining won. Macro F1, mean of 10 repeats:

| Data and task | Typing | Text | Combined | Combined minus better single [95% CI] |
|---|---|---|---|---|
| SWELL per block, N vs. T | 0.667 | 0.504 | 0.644 | -0.049 [-0.124, 0.000] |
| SWELL per block, N vs. either | 0.686 | 0.482 | 0.642 | -0.035 [-0.113, 0.039] |
| SWELL per block, I vs. T | 0.572 | 0.503 | 0.511 | +0.013 [-0.158, 0.176] |
| SWELL per block, 3-class | 0.397 | 0.330 | 0.367 | -0.060 [-0.202, 0.063] |
| SWELL per window, N vs. T | 0.552 | 0.472 | 0.517 | -0.034 [-0.097, 0.024] |
| SWELL per window, N vs. either | 0.550 | 0.500 | 0.551 | +0.004 [-0.061, 0.067] |
| SWELL per window, I vs. T | 0.540 | 0.494 | 0.506 | -0.026 [-0.108, 0.061] |
| SWELL per window, 3-class | 0.370 | 0.330 | 0.341 | -0.031 [-0.091, 0.029] |
| EmoSurv, anger vs. calm | 0.673 | 0.633 | 0.660 | -0.025 [-0.176, 0.125] |
| EmoSurv, negative vs. positive | 0.562 | 0.670 | 0.663 | -0.021 [-0.096, 0.041] |
| EmoSurv, active vs. passive | 0.620 | 0.544 | 0.611 | -0.032 [-0.096, 0.031] |
| EmoSurv, four emotions | 0.364 | 0.420 | 0.447 | +0.026 [-0.047, 0.095] |

The differences compare labels from probabilities averaged over repeats, so they need not equal the
gap between the means. **Every interval includes zero: fusion was not shown to help**, with far
richer text features and a free choice of weight. SWELL here has 22 or 23 English writers rather
than 25, with fewer options, so its typing scores differ from Section 7.12.

### 7.14 Live typing on short bursts

The extension sees bursts of 15 to 400 keys, split at 2-second pauses, not whole blocks.
`run_burst_model.py` trains on such bursts from SWELL-KW and tests on unseen people.

| Task | Chance | Macro F1 per burst | Macro F1 per block |
|---|---|---|---|
| 3-class | 0.333 | 0.323 | 0.343 |
| Neutral vs. time pressure | 0.500 | 0.509 | 0.546 |

**A single burst carries no usable signal across people.** The extension therefore gives typing a
weight of 0.25, labels a typing estimate below 0.5 confidence "Unclear", and leaves an unclear
typing-only burst out of its history, badge and break suggestion. Text is what drives the reading.

## 8. Discussion

**Text works where people write about themselves.** On Dreaddit the text pipeline is
strong, stable and robust to how the data is split. It also carries over, untrained, to
EmoSurv's free writing. On SWELL-KW, where participants wrote assigned reports, text shows no
significant signal. The vocabulary result therefore depends on what people choose to write
about, which matters for a tool that reads everyday typing.

**Typing carries a real but limited signal, and only over long stretches.** On 25 people's
per-minute logs, keyboard features beat chance macro F1 significantly, yet do not beat always
guessing the majority class on accuracy. Per block with nested selection, keystrokes reach 0.506
on three classes and interaction data reaches 0.764 on neutral vs. time pressure, both significant
after correction. On single bursts the signal is gone. Neutral was always the first block, so some
of it may reflect time in the session. No typing task on EmoSurv survives correction. It is evidence
of a signal over a whole work block, not a working live detector.

**Stricter testing changed a result in both directions.** Nesting raised nothing by itself: it
removed the 0.776 we had called promising, while a broader search found a significant 3-class score.
Fixing a gap-measurement bug also took EmoSurv active vs. passive from a corrected p of 0.02 to 0.08.

**Fusion was not shown to help.** This was the project's central hypothesis. On two
independent paired datasets, no fusion method measurably improved on the better single
modality, even with a free choice of models and weights, and on one EmoSurv task a fixed fusion did
harm: when one signal is much weaker, averaging dilutes the stronger one. The samples are small, so a
modest effect remains possible, and the labels are weak: on SWELL-KW, rated stress did not differ
significantly between conditions.

**Evaluation changed what we can claim.** Testing whole held-out people, fixed models,
confidence intervals and corrected significance tests removed claims that a single
accuracy figure would have allowed.

## 9. Limitations

- Labels: self-reported (own data) or condition-based (SWELL-KW). Both are
  proxies for stress.
- Induced stress (a timer) is milder and shorter than real-world stress.
- The relaxed-first block order is fixed, so practice and fatigue are not
  counterbalanced.
- The Kaggle data has two participants and masked keys.
- SWELL-KW: neutral is always block 1, labels are conditions, participants rated stress
  no higher with significance, timestamps tick in about 15.6 ms steps, there is no dwell
  time, and rebuilt text is approximate where people moved the cursor to edit.
- Dreaddit is binary, Reddit-specific text.
- EmoSurv labels are induced emotions, not stress. Neutral is always first and followed no
  video; fixed passages differ by emotion; keyCodes were stored as characters, so rebuilt
  text is approximate; and most people have only one or two emotion blocks.
- Transformer results come from one training seed, and earlier runs had already scored the
  classical model and DistilBERT on the official test set.
- The test set of Section 7.11 had been used before, and the searches of Sections 7.12 and 7.13
  were designed after earlier results on the same people had been seen.
- The live typing model is at chance on short bursts, and the extension's text model cannot output
  Medium, so its three-level display is more detailed than its evidence.
- The extension uses the interpretable text model, not the stronger transformers.
- A small participant pool limits how far any result generalises.

## 10. Ethics and privacy

Participants are identified only by codes; no names or contact details are
recorded. The collector runs offline and stores data only in the browser
until the researcher downloads it. Participants consent on screen, may stop
at any time, and are asked not to type personal details. Raw keystrokes and
text are kept out of version control and reported only in aggregate. No new participants were
involved in the results reported here: all of them use public datasets, under their licences
(EmoSurv and SWELL-KW are for non-commercial use, and EmoSurv may not be redistributed). Any
future collection with the tool should follow the university's ethics process first. Following
Pavić et al. (2026), the output suits self-reflection or cohort-level awareness and must not be
used for automated decisions about an individual.

## 11. Conclusion

The project delivers a complete, reproducible system: two feature pipelines, three
classifier families, three fusion strategies, participant-grouped evaluation, a
paired-data collection tool, a live demonstration and real-time feedback in the browser
with a history dashboard that runs entirely in the browser. Text classification works well on
public data (macro F1 0.814 on the official Dreaddit split) and its signal carries over to new
people's writing. Typing behaviour shows a signal on unseen people over a whole work block (0.506
vs. 0.333 chance on three classes, corrected p = 0.02) but none on a single burst, and fusion showed no
measurable gain on two real paired datasets. We also measured how far common evaluation shortcuts inflate
scores on this kind of data (macro F1 0.524 to 0.816 on the same logs). The system demonstrates
the approach, but its current models are not accurate enough to rely on for an individual's
stress, and it says so to the user.

## 12. Future work

- Paired data with per-sample self-reports and counterbalanced block order, which neither
  public dataset provides and the collector is built to gather.
- Longer-term, in-the-wild collection (background logging with consent).
- Testing per-person comparison on data with counterbalanced order, since neutral came first
  in both public datasets.
- A compact transformer small enough to run inside the extension.
- A text model with a real Medium class, which needs graded stress labels.
- Physiological ground truth for a subset of sessions to validate the labels.

## References

1. Vizer, L. M. (2009). Detecting cognitive and physical stress through typing behavior. *CHI '09 Extended Abstracts*.
2. Sağbaş, E. A., Korukoglu, S., and Balli, S. (2020). Stress detection via keyboard typing behaviors by using smartphone sensors and machine learning techniques. *Journal of Medical Systems*, 44(4).
3. Ghosh, S., Ganguly, N., Mitra, B., and De, P. (2017). TapSense: Combining self-report patterns and typing characteristics for smartphone based emotion detection. *MobileHCI*.
4. Turcan, E., and McKeown, K. (2019). Dreaddit: A Reddit dataset for stress analysis in social media. *Proceedings of the 10th International Workshop on Health Text Mining and Information Analysis (LOUHI)*.
5. Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., and Kraaij, W. (2014). The SWELL knowledge work dataset for stress and user modeling research. *ICMI 2014*.
6. Hutto, C. J., and Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
7. Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12.
9. Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16.
10. Maalej, A., and Kallel, I. (2020). Does keystroke dynamics tell us about emotions? A systematic literature review and dataset construction. *16th International Conference on Intelligent Environments (IE)*. doi:10.1109/IE49459.2020.9155004. Dataset: doi:10.21227/eae6-pk42.
11. Pavić, L., Mutka, A., Tolić, D., Havelka Meštrović, A., and Žagar, M. (2026). Detecting student stress through keystroke patterns in programming assignments. *49th MIPRO ICT and Electronics Convention*. doi:10.1109/MIPRO70003.2026.11592024.
12. Darmawan, A. S., et al. (2021). Tree-based ensemble learning for stress detection by typing behavior on smartphones. *ICSECS-ICOCSIM 2021*. doi:10.1109/ICSECS52883.2021.00078.
13. Kene, A., and Thakare, S. (2021). Mental stress level prediction and classification based on machine learning. *Smart Technologies, Communication and Robotics (STCR)*. doi:10.1109/STCR51658.2021.9588803.
14. Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
15. Clark, K., Luong, M.-T., Le, Q. V., and Manning, C. D. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. *ICLR*.
16. He, P., Liu, X., Gao, J., and Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *ICLR*.
17. Nguyen, D. Q., Vu, T., and Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English tweets. *EMNLP*.
