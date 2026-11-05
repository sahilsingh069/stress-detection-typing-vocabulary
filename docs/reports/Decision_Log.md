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

**Correction (D23):** SWELL-KW's raw logs can in fact supply paired data once rebuilt.
The statement above holds for datasets as released, not for what can be derived.

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

## D11. Exact dwell-time pairing on the Kaggle data (correction)

**Decision:** Pair each key press with its own release using the row it came
from, instead of matching by key name.

**Reason:** The Kaggle files mask most keys as `$`. Matching press to release
by key name mismatches overlapping keystrokes (the next key is often pressed
before the previous one is released). Each Kaggle row already stores one
keystroke's press and release time, so the exact pairing was available.

**Measured effect (run_midterm.py):** keystroke + interaction, 3-class, Random
Forest: macro F1 0.497 -> 0.553, accuracy 0.565 -> 0.629. Binary Random
Forest 0.519 -> 0.532. Keystroke-only best 0.410 -> 0.406. Cross-user mean
macro F1 0.403 -> 0.401. **Every configuration is still below the 0.639
baseline, so the mid-term conclusion is unchanged.** The Progress Report keeps
its original figures as a record of what was presented; current figures are in
`outputs/midterm_results.csv`.

---

## D12. Hold out whole participants whenever participants are identifiable

**Decision:** SWELL-KW and own collected data are evaluated with
`StratifiedGroupKFold` over participants. Random splits are reported only as a
comparison, to show the size of person leakage.

**Reason:** The Kaggle cross-user test showed a model learns *people* before it
learns *stress*. A random split would let the same person sit in train and
test and overstate how the system does on a new user.

---

## D13. Collector v2: chat-style interview instead of one free-writing box

**Decision:** Each condition now has a copy task and a 4-question chat. Every
chat answer is its own sample. The relaxed-before-pressured order is kept (D7).

**Reason:** One free-writing box per condition gave 2 text samples per
participant; the chat gives 8, and short conversational answers resemble
everyday typing more closely than an essay prompt. Typed while a question is
on screen, the answers are also less self-conscious than "write about your
stress".

---

## D14. Same topics in both conditions (confound removed)

**Decision:** Chat questions come from one pool of 8 neutral everyday
questions, shuffled per participant, 4 per condition. The two copy passages
are matched in length and structure and randomly assigned to conditions.

**Reason:** In v1 the relaxed prompt asked about "your week" and the pressured
prompt about "something worrying you", and the pressured passage was
deliberately harder. Any difference between conditions could then come from
the topic or the text difficulty rather than from stress. Randomising which
material lands in which condition removes that explanation.

**Trade-off:** neutral topics elicit less stress vocabulary than a "what
worries you" prompt, so the text signal on our own data may be weaker than on
Dreaddit. That is the honest measurement of whether stress changes everyday
vocabulary.

---

## D15. An allow-list of model inputs for own data

**Decision:** The typing models see only `config.OWN_TYPING_FEATURES`, computed
in Python from the raw keystrokes. The text models drop `word_count`.

**Reason:** `sessions.csv` contains the condition, block, time limit, duration,
raw key and word counts and the self-ratings. These either are the label or are
fixed by the timer (timed answers are short because of the timer). A deny-list
would silently admit any new column; an allow-list cannot leak by accident.
The browser's own summary statistics are kept in the file for inspection but
recomputed in Python so the experiments and the demo use one implementation.

---

## D16. SWELL-KW: computer-interaction features only, condition labels

**Decision:** Use the Sn* computer-interaction features. Never use the
questionnaire columns (Stress, Valence, NASA-TLX, ...) as inputs. Labels come
from the condition: N -> Low, I -> Medium, T -> High; rest periods dropped.
A binary view (N vs. I/T) is reported too.

**Reason:** The questionnaire answers are the participant's own stress rating —
the label in another form. Facial, posture and heart-rate features belong to
other modalities than this project's. The loader refuses the heart-rate-only
SWELL derivative with an explanation rather than training on it silently.

**Caveat (for mentor item A5):** "interruptions" is a different stressor, not a
midpoint between neutral and time pressure. The binary result is therefore the
cleaner claim; the 3-class result is reported alongside it.

---

## D17. Per-participant calibration reported separately

**Decision:** `run_swell.py` reports, next to the new-person result, a
calibrated variant in which each participant's features are z-scored against
their own mean.

**Reason:** Typing speed differs more between people than within a person
under stress. Calibration uses the test person's unlabelled typing only, which
models a system that observes a new user before judging them. It is reported
as its own row and never mixed into the headline.

---

## D18. Demo app shares the experiment code and states its provenance

**Decision:** The demo (`app/server.py`) computes features with the same
Python functions as the experiments and shows, next to every prediction, which
dataset each model was trained on and whether it is validated.

**Reason:** A demo with its own feature code drifts from the reported results.
Until own data exists the typing model can only come from the 2-person Kaggle
set, which failed the cross-user test; the app says so on screen rather than
presenting the prediction as validated.

---

## D19. Transformer as an optional comparison, not the reported model

**Decision:** `run_transformer.py` fine-tunes DistilBERT on the identical
Dreaddit split. It is optional (heavy dependencies) and reported as a
comparison.

**Reason:** It answers mentor item A6 ("is a transformer expected?") with a
number: how much accuracy the interpretable pipeline gives up. The classical
pipeline stays the reported model for the reason in D9.

---

## D20. Real-time feedback through a Chrome extension

**Decision:** Deliver the synopsis objective "work in the background during
normal typing" as a Chrome extension that analyses typing in web text fields
through the local server, rather than as a system-wide desktop logger.

**Reason:** A desktop logger records every keystroke in every application,
including passwords — effectively a keylogger, with operating-system
permissions and serious consent problems. Most everyday typing (mail, docs,
chat) happens in the browser, where an extension can see only text fields,
skip password, email, card and one-time-code fields, be paused per site, and
be removed in one click. It reuses the server and models of the demo, so the
feedback a user sees comes from the evaluated pipeline.

**Design choices:** a burst ends after a 2 s pause and needs at least 15 key
presses; only the text typed in the burst is sent (rebuilt from key presses),
never the rest of the field; the badge averages the last 5 bursts within 10
minutes so one burst cannot flip it; a break suggestion needs 3 consecutive
High bursts and is limited to one per 30 minutes; only estimates are stored.
The server moved to port 8765 because macOS reserves port 5000 for AirPlay.

---

## D21. SWELL-KW: report neutral vs. time pressure as the clean test

**Correction (D23):** "Clean" overstated this. Neutral is always block 1, so this task
also compares early with late in the session. The figures below are best-of-three;
`run_evaluation.py` gives the fixed-model figures (D24).

**Decision:** Besides 3-class and binary, report neutral (N) vs. time pressure
(T) with interruptions excluded, and a keyboard-only feature set. Compare
macro F1 with chance macro F1 (1 / number of classes) and accuracy with
majority accuracy, never one with the other.

**Evidence:** In the interruption condition, app switches average 35 per
minute against 3 in the others, and error keys 40 against 9: participants
are made to open and answer e-mails. A model separating I from N may only be
detecting the task design. N vs. T keeps the task and changes only the
deadline.

**Result (participants held out):** N vs. T macro F1 0.596 (chance 0.500),
keyboard only 0.581; accuracy 0.603 vs. 0.608 majority. 3-class 0.424 (chance
0.333). Per-participant calibration did not help. Reported as a weak but real
signal, not as a working detector.

**Not done, on purpose:** aggregating minutes into longer windows or tuning
after seeing these results. That would be choosing the analysis to fit the
outcome; it can be proposed to the mentor as a pre-declared follow-up.

---

## D22. A stress-history dashboard that does not overstate thin data

**Decision:** Keep estimates for 30 days and show them on a local dashboard, per hour for
one day and per day for longer periods, as the share of Low / Medium / High estimates.

**Reason:** Users asked to track stress over any period. Shares answer that directly, but a
bar built from one estimate looks as solid as one built from fifty, so periods with fewer
than 3 estimates are faded and cannot be named the highest. There is deliberately no single
"stress score": the models cannot support that precision. A caveat stays on screen while
any estimate in the period comes from an unvalidated model.

---

## D23. Paired data from SWELL-KW raw logs

**Decision:** Rebuild timing and text from SWELL-KW's raw uLog files, in 5-minute Word
windows, and run the same fusion comparison as for own data.

**Reason:** Those logs hold each key's timestamp and character for 25 people under
labelled conditions: the paired data earlier documents said no public source had.

**Rules fixed before running any model:** Word only (PowerPoint is nearly absent from the
neutral block, e-mail is specific to interruptions); drop the duplicate whitespace event
uLog records after each Space; at least 100 key presses and 20 words per window; exclude
the two Dutch writers because the lexicons are English; add `order_control` (block 2 vs. 3)
because neutral is always first.

**Result:** No stress task is significant after correction, and fusion shows no gain.
Participants rated time pressure higher under the deadline (p = 0.012), but not stress
(p = 0.119). This does not replace own data, whose labels and order design are better.

---

## D24. Extended evaluation design, including one corrected mistake

**Decision:** Fix Logistic Regression in advance for headline figures and tests; bootstrap
whole people; permute labels between a person's blocks as whole units; apply Holm
correction within each dataset.

**Reason:** Picking the best of three on the same folds is optimistic. Rows from one person
are not independent, and windows in a block share a label, so resampling or shuffling
single rows would make results look more certain than they are.

**Mistake corrected:** The first run used 200 shuffles and corrected across all 21 tests,
including order controls. The lowest possible p-value (0.005) times 21 exceeds 0.05, so
nothing could pass, not even Dreaddit at 0.765 against 0.517 for shuffled labels. It was
re-run with 2,000 shuffles for SWELL, Holm within each dataset, and order controls reported
separately. The grouping was changed after seeing the first output, for that reason.

**Dreaddit check:** One post can yield several rows. Holding out whole posts and using the
official split gave 0.765 and 0.760 against 0.763 for random folds, so no leakage inflated
the reported result.

---

## D25. EmoSurv: free text is primary, and labels stay emotions

**Decision:** Add EmoSurv as a second paired dataset, use free text as primary data and report
fixed text only as confounded. Call the labels emotions throughout.

**Evidence:** Each emotion has its own fixed passage, and neutral's is the longest. Active vs.
passive scores 0.723 on fixed text and 0.534 on free text. `answer` is blank for all neutral
rows, so it reveals the label and is never loaded. Neutral rows are removed after subtracting a
person's neutral block, because they would all be zero.

**Rejected:** Relabelling anger as "High stress" to enlarge the stress data. Anger vs. calm is a
proxy; renaming it would overstate what was measured.

---

## D26. Report the leakage gap, not the leaky score

**Decision:** Report person-held-out results as ours. Show random-fold scores only in a table
that measures the gap.

**Evidence:** The same SWELL logs score macro F1 0.816 with random folds over overlapping windows
and 0.524 with people held out. EmoSurv scores 0.950 with people held out when key composition is
available, and 0.329 on timing alone.

**Rejected:** Using a 90%+ configuration as the headline. It would match the papers' numbers, but
it measures recognising people and passages rather than stress.

---

## D27. Text headline is an ensemble declared in advance

**Decision:** Headline 0.808 from the equal-weight ensemble, on the official split.

**Reason:** It was declared before RoBERTa or the ensemble was scored. RoBERTa's 0.813 is
not measurably different, so switching would only be picking a winner on the test set. The
ensemble is also better calibrated (0.027 vs. 0.083).

**Consequence:** The demo keeps the interpretable model, which is small and explainable.

---

## D28. Per-person comparison is reported as promising, not established

**Decision:** Report 0.776 per block on SWELL-KW with its corrected p of 0.06, and do not make it
the headline typing result.

**Reason:** Each task's p-value already accounts for six options, but three tasks were tested, and
neutral was always first. The order control (block 2 vs. 3) found nothing (p = 0.63), which rules out
only part of that concern. The same idea did not help on EmoSurv.

---

## D29. Live typing model moves from Kaggle to SWELL-KW logs

**Decision:** Without own data, `train_demo.py` trains typing on SWELL-KW raw logs and falls back
to Kaggle only if they are absent. The model stays marked unvalidated.

**Reason:** 23 people with stress conditions, against 2 people whose model did not generalise.
Neither is validated for short live bursts, and the app says so.

---

## D30. Dashboard rebuilt around questions a reader asks

**Decision:** Lead with an average score, then show how it changes, and where and when it is highest.

**Reason:** Stacked level shares per hour collapsed a session into one bar on a 24-hour axis, which
could not show change or place. The score is the chance of High plus half the chance of Medium,
which uses the probabilities already stored; no new data is kept.

**Guards:** Sentences need 10+ points and 3+ estimates and do not claim a cause. Heatmap cells with
too few estimates carry a neutral marker, because fading would read as a lower score.

---

## D31. The extension runs in the browser, and typing gets little weight

**Decision:** Run the models in JavaScript with no server. Train the typing model on bursts rather
than 5-minute windows. Give typing a weight of 0.25, show it as "Unclear" below 0.5 confidence, and
skip an unclear burst that has too few words for a text estimate.

**Why:** A server blocked sharing the extension and added 22 ms per estimate, against about 1.5 ms in
the browser. On unseen people, bursts score 0.323 on three classes (chance 0.333), so typing cannot
carry a live reading. A parity test keeps the JavaScript and Python models identical.

**Accepted cost:** The binary text model at 0.75 weight means a burst with enough words is never
labelled Medium. We document this rather than invent a Medium score the data cannot support.

---

## D32. Text search chooses on a development set

**Decision:** Choose models and the ensemble on 15% of the training posts, retrain with 3 seeds, score
the test set once, and quote the chosen ensemble (0.814).

**Why:** ELECTRA alone scored 0.825, but choosing it after seeing that would be selecting on the test
set. The earlier ensemble stays better calibrated (0.027 against 0.098).

**Corrections during the run:** DeBERTa's checkpoint is stored in float16 and trained into NaN, and
BERTweet accepts only 128 tokens. Both were found and fixed before the test set was scored.
mental-RoBERTa needs approved access and was not tested.

---

## D33. Nested selection replaces best-of-options for typing and fusion

**Decision:** Choose options on training people only, and rerun the whole procedure in the
permutation test. This supersedes D28's per-block figure.

**Why:** Comparing the best option with shuffled best options controls for trying several, but the
reported score still comes from the winner. Nested, 0.776 fell to 0.625 (not significant), while the
3-class score of 0.506 and the interaction scores became significant after correction.

**Disclosed:** Standardising within a person and the shared letter-pair table use unlabelled data from
held-out people. These options were designed after earlier results on the same people had been seen.

---

## D34. EmoSurv gaps corrected, and the affected results rerun

**Decision:** Build the clock before removing auto-repeats in `emosurv_options`, then rerun all EmoSurv
typing and fusion tasks.

**Why:** A code review found that gaps ran to the next logged row rather than the next press, in 65 of
189 blocks. The fix moved active vs. passive from a corrected p of 0.02 to 0.08, so the earlier
significant result is withdrawn.

---

## D35. Fusion permutation tests skipped

**Decision:** Run the fusion search with `--perm 0`.

**Why:** Decided before scoring: the tests matter only if combining beats the better single stream. No
comparison did: every interval for the gain includes zero. The report states this and gives no p-values.

---

## Rejected approaches, for the record

| Approach | Why rejected |
|---|---|
| Window sizes 60s / 120s / 600s | All scored below the 300s baseline |
| Per-user z-normalisation | Marginal gain; with two users it partly memorises individuals |
| Kaggle dataset as primary source | Fails cross-user generalisation |
| Reporting a typing accuracy without its baseline | All current typing configurations remain below the 0.639 baseline |
| Further hyperparameter search | Algorithm convergence indicates a data ceiling |
