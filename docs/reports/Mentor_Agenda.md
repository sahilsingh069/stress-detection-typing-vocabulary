# Mentor Discussion Agenda

**Stress Level Detection Using User Typing Behavior and Text Vocabulary**
Meeting with Dr. Souvik Dev

Bring: the Progress Report, the results table, and the data-collection tool on a
laptop.

Open with a two-minute summary — one pipeline works, one does not and we know
why, and here is what we need decided. Then work through the items below.

---

## A. Decisions we need from you

### A1. Dataset strategy
No public dataset contains both keystroke dynamics and typed text labelled for
stress. We are currently using Dreaddit for the text modality and are seeking
SWELL-KW for the behavioural modality.

- Is using two separate single-modality datasets acceptable for a major project?
- Or do you expect a single unified dataset, in which case our own collection
  becomes the only option?

### A2. SWELL-KW access — request for your help
SWELL-KW (Radboud University, 25 participants, validated stress labels) requires
registration and approval. This is currently our main blocker for the typing
pipeline.

- Does UPES have institutional access to this or comparable datasets?
- Would you be willing to sponsor or endorse the access request? A faculty
  request is typically approved faster than a student one.

### A3. Presenting a negative result
Our typing pipeline performs below its majority-class baseline. We have tested
the cause (cross-user macro F1 of 0.403 with two participants) and documented it
rather than hiding it.

- Is presenting this as a tested finding acceptable at mid-term?
- Or does the department expect working results across all components by then?

This affects how aggressively we prioritise data collection over other work.

### A4. Ethics and consent for human data collection
We have built a tool to collect typing data from student volunteers. It records
keystroke timings and typed text, with a consent screen and no personal
identifiers.

- Does a major project involving human participants require formal ethics
  clearance at UPES?
- Is our consent screen sufficient, or is a signed form required?

We would rather resolve this before collecting than discard data afterwards.

### A5. Label mapping justification
SWELL-KW labels sessions by experimental condition (No stress / Interruption /
Time pressure). Our project classifies Low / Medium / High.

- Is mapping conditions onto Low / Medium / High methodologically defensible?
- Or should we keep the original condition labels and restate our objective
  accordingly?

### A6. Scope expectations for a major project
- Is a transformer baseline (BERT / DistilBERT) expected, or is classical machine
  learning sufficient at this level?
- Is validated fusion required for the final deliverable, or would two solid
  independent pipelines meet the standard?
- Is a working demonstration interface expected, or are results and analysis
  sufficient?

---

## B. What we would value your feedback on

### B1. Is our interpretation of the typing failure correct?
We concluded the limitation is the number of participants, based on the
cross-user generalisation test. We would like to know whether you see another
explanation we have missed — for example our windowing choice, our feature set,
or how labels were aligned to typing sessions.

### B2. Feature engineering
Our typing features are dwell time, flight time, typing speed, backspace rate,
long-pause rate, rhythm variability, plus mouse, idle and application-switching
features. Are there established features from the keystroke-dynamics literature
we have overlooked?

### B3. The first-person pronoun finding
The strongest text feature was first-person pronoun ratio, ahead of sentiment
scores. Is this consistent with what you would expect, and is it worth
developing further in the report?

### B4. Evaluation methodology
We report accuracy, macro precision, macro recall, macro F1, cross-validated F1,
and the majority-class baseline for every model. Is anything missing that an
examiner would expect?

---

## C. For your information — no decision needed

- Both pipelines are implemented and reproducible from a single command.
- The text pipeline achieves 76.2% accuracy against a 52.3% baseline, with three
  independent algorithms agreeing within 1%.
- The fusion layer is implemented but deliberately unreported, since unpaired
  datasets provide no fused ground truth.
- The data-collection tool is built and tested, and can be demonstrated in this
  meeting.

---

## D. Our proposed plan — for your approval or correction

1. Begin participant recruitment immediately, targeting 20–30 people.
2. Pursue SWELL-KW access in parallel as a fallback for the behavioural stream.
3. Re-run the full evaluation once either dataset is available.
4. Compare early fusion against late fusion on paired data.
5. Build a demonstration interface for the final presentation.

We would like to know whether this ordering matches your expectations, and what
you consider the minimum acceptable final deliverable.
