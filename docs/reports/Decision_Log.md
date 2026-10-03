# Decision Log

A record of what was decided, when, and on what evidence. Use this when someone
asks "why did you do it that way" — every entry has a reason, and several
document things we tried and rejected.

---

## D1. Two datasets instead of one

**Decision:** Use Dreaddit for text vocabulary and a separate dataset for typing
behaviour, rather than searching for one dataset covering both.

**Reason:** No public dataset contains keystroke dynamics and typed text labelled
for stress. Keystroke datasets (CMU benchmark, Kaggle) either mask the typed
content or were built for authentication rather than stress. Text datasets
(Dreaddit) contain no timing information.

**Consequence:** Only late fusion is valid until paired data exists.

---

## D2. Rejected the Kaggle keystroke dataset as a primary source

**Decision:** Treat it as a pipeline validation set, not as the basis for
reported results.

**Evidence:**
- 2 participants only
- Approximately 100 self-reported labels total
- Most keys masked as `$`, so no vocabulary is recoverable
- Class distribution 198 High / 79 Medium / 33 Low
- Cross-user test: training on one participant and testing on the other gives
  mean macro F1 of 0.403
- Best macro F1 (0.524 binary) remains below the 0.639 majority-class baseline

**What we tried before rejecting it:** window sizes of 60, 120, 300, 600 and 900
seconds; per-user z-normalisation; binary label collapse; adding mouse, idle and
application-switch features; grid search across three classifiers. The best
configuration improved macro F1 over the starting point and still
did not clear the baseline.

---

## D3. Added interaction features beyond the keyboard

**Decision:** Include mouse speed statistics, idle-period duration and
application-switch counts alongside keystroke features.

**Reason:** The companion files in the Kaggle dataset were initially unused.
Stress manifests in idle patterns and task switching, not only typing rhythm.

**Measured effect:** macro F1 improved from 0.410 to 0.497 on the 3-class task;
the binary variant reached 0.524 macro F1.

**Note:** SWELL-KW logs the same signal families, so these features carry over.

---

## D4. Report macro-averaged metrics alongside accuracy

**Decision:** Report macro precision, recall and F1 for every model, and always
state the majority-class baseline.

**Reason:** Our data is imbalanced. The majority-class baseline is 0.639
accuracy, and every current typing configuration falls below it despite moderate
macro-F1 improvements. Accuracy alone would obscure this limitation; macro F1
makes the minority-class performance visible.

---

## D5. Late fusion rather than early fusion, for now

**Decision:** Implement decision-level fusion, combining predicted probabilities
from two independently trained models.

**Reason:** Early fusion requires a single feature vector per sample containing
both modalities. Our two datasets come from different people, so no such sample
exists. This is a data constraint, not an implementation difficulty.

**Also implemented:** early fusion and a stacking meta-classifier, both verified
on simulated paired data, ready for real paired data.

---

## D6. Do not report a fused accuracy figure

**Decision:** Demonstrate that the fusion mechanism works, but report no accuracy
for it.

**Reason:** With unpaired datasets there is no ground-truth label for a fused
sample. Any number produced would be an artefact of arbitrarily pairing
unrelated samples.

---

## D7. Build our own data collection tool

**Decision:** Build a browser-based tool to collect paired typing and text data
with stress labels.

**Reason:** This is the only route to a dataset supporting early fusion, which is
the project's central claim. For a major project it is the contribution rather
than a convenience.

**Design decisions within the tool:**
- *Relaxed blocks run before pressured blocks* so the pressure condition cannot
  contaminate the relaxed baseline.
- *Both copy and free-writing tasks* — copy tasks hold text constant to isolate
  keystroke dynamics, free-writing produces genuine vocabulary variation. Copy
  tasks alone would yield no usable vocabulary signal, since every participant
  types identical words.
- *Paste disabled* — pasted text produces no keystroke data.
- *Key-press and key-release both logged* — required for true dwell time.
- *Three ratings per block* (stress level, intensity, perceived time pressure)
  rather than one, to allow validation that the pressure condition actually felt
  pressured.
- *No upload* — data stays in the participant's browser until they send it,
  which keeps the consent statement simple and truthful.

---

## D8. Stopped hyperparameter tuning

**Decision:** Do not tune further.

**Reason:** On the text pipeline, three structurally different algorithms
(linear, ensemble, kernel-based) landed within 1% of each other after grid
search. Convergence across model families indicates a feature or data ceiling
rather than a hyperparameter ceiling. On the typing pipeline, the cross-user
test localises the problem to the number of participants.

---

## D9. VADER rather than a transformer, for the baseline

**Decision:** Use VADER sentiment plus hand-crafted lexical features and TF-IDF.

**Reason:** Interpretability. This configuration identified first-person pronoun
ratio as the strongest single predictor, ahead of sentiment. A transformer would
likely score higher but would not have surfaced that finding.

**Open question for the mentor:** whether a transformer comparison is expected at
major-project level.

---

## D10. Both keystroke-only and combined feature sets reported

**Decision:** Report keystroke-only, keystroke-plus-interaction, and binary
variants rather than only the best one.

**Reason:** This constitutes an ablation study, showing what each feature family
contributes rather than asserting that the full set is best.

---

## Rejected approaches, for the record

| Approach | Why rejected |
|---|---|
| Window sizes 60s / 120s / 600s | All scored below the 300s baseline |
| Per-user z-normalisation | Marginal gain; with two users it partly memorises individuals |
| Kaggle dataset as primary source | Fails cross-user generalisation |
| Reporting a typing accuracy without its baseline | All current typing configurations remain below the 0.639 baseline |
| Further hyperparameter search | Algorithm convergence indicates a data ceiling |
