# Presentation Script — Stress Level Detection Using Typing Behavior & Text Vocabulary

**Speakers:** Anshuman (A) · Sahil (S) · Royal (R)
**Total time target:** ~8–9 minutes + Q&A

---

## Slide 1 — Title

**A:** Good [morning/afternoon] everyone. We're presenting our minor project — Stress Level Detection Using User Typing Behavior and Text Vocabulary. I'm Anshuman Tiwari, and with me are Sahil Panwar and Royal Bisht. This project is mentored by Dr. Souvik Dev.

---

## Slide 2 — Content / Agenda

**A:** Quickly, here's what we'll cover — the problem we're solving, our proposed system and how it works, the methodology and architecture behind it, the technologies and dataset we're using, and finally the applications and expected outcome.

---

## Slide 3 — Introduction

**A:** Stress affects academic performance, productivity, and mental well-being when it goes undetected. Our project uses AI and Machine Learning to estimate a person's stress level from something they already do every day — typing.

We look at two things together: **how** a person types — their speed, pauses, and backspaces — and **what** they type — their vocabulary and sentiment. Traditional stress detection relies on questionnaires or physiological sensors, which are either subjective, infrequent, or intrusive. We wanted something that works quietly in the background instead.

---

## Slide 4 — Problem Statement

**A:** Let's break down why existing methods fall short.

Physiological sensors — like heart-rate or skin-conductance monitors — are costly and intrusive. They need extra hardware, and you have to wear something.

Self-reported surveys are subjective — they depend on the person's own self-awareness, and they're only filled out occasionally, not continuously.

And neither approach gives you a real-time signal — there's no way to monitor stress continuously as someone works.

So the need is clear: a low-cost, non-intrusive, continuous way to estimate stress from data users are already generating while typing.

*(Hand off to Sahil)*

---

## Slide 5 — Objective

**S:** Thanks, Anshuman. Our objectives follow directly from that problem.

First, identify typing-behavior features — speed, pauses, backspaces — that correlate with stress.
Second, analyse text vocabulary — word choice and sentiment — as an additional stress indicator.
Third, build an ML model that classifies stress level as Low, Medium, or High.
And fourth, design a system that's lightweight and non-intrusive enough to run during normal typing.

---

## Slide 6 — Proposed System

**S:** Our system has five modules, and I'll walk through them like a pipeline.

Data Capture records keystroke timings and the typed text through a typing interface.
Feature Extraction derives typing-speed, pause, backspace, and vocabulary or sentiment features from that raw data.
Preprocessing cleans and normalizes those features so they're ready for the model.
ML Classification takes the combined features and classifies them into a stress-level category.
And Result Display shows the predicted stress level back to the user with a confidence score.

This whole thing is trained end-to-end using supervised learning on labelled typing sessions.

---

## Slide 7 — Methodology

**S:** This is our project plan — the steps we're following to actually build this, not the system itself.

We start with a literature review of prior keystroke-dynamics and text-emotion research. Then data collection — gathering keystroke logs and typed text from volunteers under varied conditions. Then data preprocessing — extracting numeric typing features and applying NLP techniques like tokenization and sentiment analysis to the text. Then model building — training ML classifiers on the combined features. Then model training — splitting data into training and testing sets. And finally, performance evaluation — using accuracy, precision, recall, and a confusion matrix.

*(Hand off to Royal — or continue to Slide 8 as the "money slide")*

---

## Slide 8 — System Architecture ⭐ (spend the most time here)

**S:** Now here's how the system actually works once it's running — think of it as one continuous flow.

*(Point to each stage on the slide as you say it)*

A user types text. That typing is captured — both the keystroke timing and the text itself. Those get turned into features — the typing-behavior numbers, and the vocabulary and sentiment scores from the text. The combined feature set is passed to our ML model, which classifies it into a stress level. And finally, that stress level is displayed back to the user.

It's a linear pipeline — raw typing input is progressively transformed into a stress-level prediction. Every module we just described feeds directly into the next.

*(Hand off to Royal)*

---

## Slide 9 — Technologies Used

**R:** Thanks, Sahil. Let me quickly cover the tools behind this.

Python is our core programming language. Scikit-learn handles building and evaluating our ML models. NLTK and TextBlob handle NLP and sentiment analysis on the typed text. JavaScript captures keystrokes through the browser-based interface. And Pandas and NumPy handle our data handling and feature engineering.

---

## Slide 10 — Dataset

**R:** This is one of the more interesting parts of our project, so let me explain our approach carefully.

There isn't a single public dataset that combines labelled keystroke dynamics with text vocabulary together — and that gap is actually part of what our project addresses.

So we're using a two-pronged approach. For typing-behavior features, we're using the **SWELL Knowledge Work dataset** — a peer-reviewed benchmark from Radboud University where 25 participants performed real work tasks like writing, emailing, and presentations, under controlled stress conditions such as time pressure and interruptions, with stress levels validated through standard questionnaires.

For the vocabulary side, we're using **Dreaddit**, a publicly available dataset of text labelled for stress versus non-stress, to train and validate our sentiment and word-choice features.

Since no dataset pairs both signals from the same person at the same time, our next phase is collecting a smaller set of our own data — real typing sessions with actual typed text and self-reported stress — to fuse both feature types into a single combined model. At this synopsis stage, our focus has been validating each component against established, credible data before scaling up to our own collection.

---

## Slide 11 — Applications

**R:** Once built, this has real applications. In workplace monitoring, it helps organisations support employee well-being. In e-learning platforms, it can track student stress during exams or coursework. In HR analytics, it enables early identification of burnout risk. And more broadly, it supports research and wellness apps focused on mental health and productivity.

---

## Slide 12 — Expected Outcome

**R:** By the end of this project, we expect a working model that classifies stress level from typing behaviour and vocabulary, a non-intrusive, low-cost detection system requiring no additional hardware, a simple interface displaying the predicted stress level, and a foundation that could be integrated into wellness or productivity apps in the future.

---

## Slide 13 — Conclusion

**A:** *(comes back in to close)*

To conclude — this project aims to show that a person's stress level can be estimated from everyday typing behaviour and vocabulary, using AI and ML, without any additional hardware.

We're currently at an initial stage — focused on establishing the core idea, identifying relevant features, and outlining a clear methodology. With further development, this can grow into a real-time stress-monitoring tool for workplaces, e-learning platforms, and wellness applications.

---

## Slide 14 — References

**A:** Our work builds on prior research — Vizer's 2009 study on detecting cognitive and physical stress through typing behavior, Sağbaş, Korukoglu, and Balli's 2020 work on stress detection via keyboard typing using smartphone sensors and ML, and Ghosh et al.'s TapSense paper combining self-report patterns with typing characteristics for emotion detection.

---

## Slide 15 — Thank You

**All three (A leads):** Thank you. We're happy to take any questions.

---

## Anticipated Q&A — quick-reference answers

**Q: Is your dataset labelled?**
> Yes — SWELL-KW is labelled two ways: by experimental stress condition (No stress / Time Pressure / Interruption), and by validated self-report questionnaires on perceived stress. Dreaddit is labelled stress vs. non-stress at the text level.

**Q: What does "non-intrusive and low-cost" actually mean?**
> Non-intrusive means the user doesn't wear anything or pause their work to self-report — we passively observe typing they're already doing. Low-cost means no extra hardware like heart-rate or skin-conductance sensors — just a standard keyboard and our software.

**Q: Why these three ML algorithms (Logistic Regression, Random Forest, SVM)?**
> They're standard, interpretable baseline classifiers well-suited to structured/tabular feature data. We're comparing all three empirically rather than assuming one is best upfront — Random Forest also gives us feature-importance rankings, which helps show which typing or vocabulary features matter most.

**Q: Why not just use one dataset?**
> No existing public dataset pairs keystroke dynamics with typed-text vocabulary together — so we use two established, credible datasets to validate each feature type independently, and plan our own combined data collection as the next step.

**Q: What's the difference between your Methodology and your System Architecture slides?**
> Methodology is our project plan over time — literature review, data collection, preprocessing, training, evaluation. System Architecture is what happens each time a user types, in real time — capture, feature extraction, classification, display. One is how we build it; the other is how it runs.

**Q: What results do you have so far?**
> We're at the synopsis stage — our focus has been defining the problem, designing the pipeline, and validating our approach against established datasets. Model training and evaluation results come in the next phase.
