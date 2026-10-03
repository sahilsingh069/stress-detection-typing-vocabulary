# Technical Walkthrough — Understand Before You Present

This is for you, not your mentor. It explains every decision in the project so
that when someone asks "why did you do it that way", you have a real answer.

Read it once properly. Then cover the answers and try to explain each section
out loud. If you can explain something in your own words, you know it. If you
can only recite it, you don't yet.

---

## Part 1 — The features

### Typing behaviour features

You are measuring *how* someone types, not *what* they type.

**Dwell time** — how long a key is held down. Measured as
`release_time − press_time` for the same key. Requires both keydown and keyup
events; if you only log key presses you cannot compute it.

**Flight time** — the gap between releasing one key and pressing the next.
Measured as `next_press_time − previous_release_time`. In practice we often
compute *inter-key interval* instead (press-to-press), because that works even
when release events are missing.

**Typing speed** — characters per minute over a session.

**Backspace rate** — backspace count divided by total keystrokes. Not raw count,
because raw count depends on how long they typed. Rates are comparable across
sessions; counts are not.

**Long-pause rate** — fraction of gaps longer than 2 seconds. A 2-second gap
usually means thinking or hesitation rather than continuous typing.

**Rhythm variability** — standard deviation of inter-key intervals divided by
their mean. This is the *coefficient of variation*. We divide by the mean so a
fast typist and a slow typist can be compared: it measures how *irregular*
someone's rhythm is, independent of how fast they type.

*Why rates and ratios instead of counts:* a 5-minute session and a 30-second
session produce wildly different counts for the same behaviour. Normalising
makes them comparable.

### Text vocabulary features

You are measuring *what* someone types.

**Sentiment scores** — we use VADER, a rule-based sentiment analyser. It returns
positive, negative, neutral and compound scores. Compound is an overall value
from −1 to +1. VADER is rule-and-lexicon based, not a trained model, which means
it needs no training data of its own.

**Negative-word ratio / stress-word ratio** — counts of words from fixed
lexicons divided by total words.

**Lexical diversity** — unique words divided by total words (the type-token
ratio). Lower values mean more repetition.

**First-person pronoun ratio** — how often "I", "me", "my" appear. This turned
out to be the single most important text feature in our results.

**TF-IDF** — Term Frequency-Inverse Document Frequency. It scores a word higher
if it is frequent in *this* document but rare across *all* documents. This
stops common words like "the" from dominating. We use 1–2 word n-grams, so it
captures short phrases as well as single words.

*Why both hand-crafted features and TF-IDF:* hand-crafted features are
interpretable (you can say "negative sentiment mattered most"), TF-IDF is
powerful but opaque. Using both gives you accuracy and explainability.

---

## Part 2 — The algorithms

### Logistic Regression
Fits a weighted sum of features passed through a sigmoid (or softmax for
multiple classes) to produce class probabilities. It draws a **linear** decision
boundary. We use it as a baseline: if it performs well, the relationship between
features and stress is roughly linear.

### Random Forest
An ensemble of decision trees. Each tree is trained on a random subset of the
data and a random subset of features, then predictions are combined by majority
vote. Randomness prevents all the trees from making identical mistakes.

We treat it as the primary model because it handles features on different scales
without careful normalisation, resists overfitting better than a single tree,
and produces **feature importance** scores — which is how we know first-person
pronoun ratio mattered most.

### SVM (Support Vector Machine)
Finds the decision boundary with the **largest margin** between classes. With an
RBF kernel it can fit non-linear boundaries. Tends to perform well on smaller
datasets with a moderate number of features. It won on our text pipeline.

*Why compare three instead of picking one:* if all three land at the same score,
you have hit the limit of your **features or data**, not of your model choice.
That is exactly what happened on the text pipeline — all three within 1% — and
it is why further tuning would be wasted effort. This is a real finding, and a
good answer if someone asks why you did not try more algorithms.

---

## Part 3 — The metrics

### Accuracy
Correct predictions divided by total predictions. Simple, but **misleading on
imbalanced data**.

### Why macro F1 matters more here
Precision = of the samples predicted as class X, how many actually were X.
Recall = of the samples that actually were X, how many did we find.
F1 = their harmonic mean.

**Macro** averaging computes the metric per class and then averages, giving every
class equal weight regardless of size.

**The concrete example from our current results:** the best typing configuration
reaches 0.524 macro F1 but only 0.548 accuracy, below the 0.639 majority-class
baseline. This confirms that interaction features help, but the two-participant
dataset still does not support a useful general stress model.

### Majority-class baseline
The accuracy you would get by always predicting the most common class. For our
typing data that is 0.639. **Any model scoring below this is worse than a
constant guess.** Always quote your baseline alongside your accuracy, or the
number is meaningless.

### Cross-validation
Split the data into k parts, train on k−1 and test on the remaining one, rotate,
average. Gives a more trustworthy estimate than one split because it does not
depend on which samples happened to land in the test set. We use stratified
k-fold, which preserves the class distribution in every fold.

---

## Part 4 — Class imbalance

Our typing data is 198 High / 79 Medium / 33 Low. A model can score 63.9% by
ignoring Medium and Low entirely.

**Approaches:**
- `class_weight='balanced'` — penalises mistakes on rare classes more heavily.
  This is what we use.
- **SMOTE** — Synthetic Minority Oversampling Technique. Creates synthetic
  minority samples by interpolating between existing ones, rather than simply
  duplicating them. Implemented in our code and switchable via config.

**Critical detail:** SMOTE must be applied *inside* the cross-validation loop,
only to training folds — never to the whole dataset before splitting. Otherwise
synthetic samples derived from test data leak into training and your score is
inflated. Our code handles this by putting SMOTE inside an imblearn Pipeline.

This is a good question to be ready for.

---

## Part 5 — Fusion (your novelty claim)

### Early fusion (feature-level)
Concatenate typing features and text features into one long vector, train a
single model on it. Advantage: the model can learn interactions between the two
modalities. **Requires paired data** — the same person, the same session, both
measurements, one label.

### Late fusion (decision-level)
Train one model per modality independently, then combine their predicted
probabilities. Advantage: each model can be trained on whichever dataset suits
it. Works with **unpaired** data.

### Why we currently use late fusion
Dreaddit and the keystroke dataset come from different people. There is no row
anywhere in our data that has both a person's typing measurements and their
text. Early fusion is therefore mathematically impossible with what we have —
not difficult, impossible.

### The probability alignment problem
Dreaddit is binary (stress / no stress). Our typing data is 3-class (Low /
Medium / High). To combine their probability outputs you must first map both
onto the same class space. Our `align_classes()` function expands each
probability matrix onto the shared three-class space, assigning zero to classes
a model never saw, then renormalises so each row sums to 1.

### Why our own collected data matters
Every session from our collector has typing behaviour *and* typed text *and* one
stress label, from one person at one moment. That is the only data in the
project that makes early fusion valid. For a major project this is the
contribution, not a convenience.

---

## Part 6 — Questions you should expect, and honest answers

**"Why is your typing accuracy below the baseline?"**
Two participants and roughly 100 labels. We tested it directly: training on one
participant and testing on the other gives mean macro F1 of 0.403, so the model
learns individual typing habits rather than stress. We verified it is not a
tuning problem by grid-searching all three classifiers across three feature
configurations.

**"Why not just tune the model more?"**
On the text pipeline, three very different algorithms landed within 1% of each
other, which indicates a feature or data ceiling rather than a hyperparameter
ceiling. On the typing pipeline, the cross-user test shows the problem is the
number of participants.

**"Why two datasets instead of one?"**
No public dataset contains both keystroke dynamics and typed text labelled for
stress. Dreaddit provides text, SWELL-KW provides interaction behaviour, and
neither provides the other. Our own collection is intended to close that gap.

**"Have you actually fused anything yet?"**
The fusion code is implemented and verified. We have deliberately not reported a
fused accuracy, because with unpaired datasets there is no ground truth for a
fused sample. Reporting one would be dishonest.

**"What is the accuracy of your final system?"**
Text vocabulary: 76.2%. Typing behaviour: not yet at a usable standard, for
reasons we have tested and documented. Fused: pending paired data.

**"Why VADER rather than a transformer like BERT?"**
VADER is rule-based, interpretable and requires no training. It let us establish
a baseline and identify which linguistic features carry signal. A transformer
would likely improve accuracy and is a reasonable next step, but it would not
have told us that first-person pronoun usage is the strongest indicator.

**"How do you know your 76.2% is real and not overfitting?"**
Cross-validation standard deviation is about 0.01 across folds, three
independent algorithms agree within 1%, and the majority-class baseline is
52.3%. If we were overfitting a single split, those three things would not all
hold.

**"What are the limitations of your work?"**
Typing dataset too small and from too few people. Dreaddit is Reddit text, which
may not transfer to typing in other contexts. Our labels are self-reported,
which is subjective. Fusion is unvalidated. All of these are documented.

---

## Part 7 — Things to be careful about

**Do not quote synthetic results.** The codebase includes a synthetic data mode
for testing. It scores near 100% because the signal is artificially planted.
Those numbers are meaningless and must never appear in a report.

**Do not present the typing pipeline as successful.** Its best accuracy is below
the 0.639 majority-class baseline.

**Do not say "we used SWELL-KW" until you actually have it.** Access is pending.

**Fix the PPT title.** Slide 1 currently says "MINOR PROJECT". This is a major.

---

## Part 8 — Self-test

Cover the answers above and explain each of these aloud:

1. What is the difference between dwell time and flight time?
2. Why do we use rates rather than raw counts?
3. What does TF-IDF actually compute, and why is it better than word counts?
4. Why is macro F1 more informative than accuracy on our data?
5. What is the majority-class baseline for our typing data, and why does it matter?
6. Why can we not do early fusion right now?
7. What does `align_classes()` do and why is it needed?
8. Why must SMOTE be applied inside the cross-validation loop?
9. What does it mean that all three algorithms scored within 1% on the text data?
10. What is the single strongest text feature, and what does it suggest about stressed writing?

If you can answer all ten without looking, you can defend this project.
