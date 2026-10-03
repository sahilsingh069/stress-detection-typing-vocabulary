## Slide 1

MINOR PROJECT

Stress Level Detection Using
User Typing Behavior and Text Vocabulary

Presented by

500123925 — Anshuman Tiwari
500124268 — Sahil Panwar
500124730 — Royal Bisht

Mentored by — Dr. Souvik Dev

School of Computer Science | UPES, Dehradun

## Slide 2

Content

1

Introduction

2

Problem Statement

3

Objective

4

Proposed System

5

Methodology

6

System Architecture

7

Technologies Used

8

Dataset

9

Applications

10

Expected Outcome

11

Conclusion

12

References

2

## Slide 3

Introduction

01

•

Uses Artificial Intelligence and Machine Learning to estimate a person's stress level from everyday typing activity.

•

Analyses typing behaviour (speed, pauses, backspaces) together with the vocabulary used while typing.

•

Stress affects academic performance, productivity, and mental well-being if it goes undetected.

•

Traditional detection relies on questionnaires or physiological sensors — subjective, infrequent, or intrusive.

3

## Slide 4

Problem Statement

$

Costly & Intrusive

Physiological sensors (heart-rate, skin conductance) require extra hardware and setup.

?

Subjective Surveys

Self-reported questionnaires are filled occasionally and depend on self-awareness.

!

No Real-Time Signal

Neither approach supports continuous, everyday, real-time stress monitoring.

Need: a low-cost, non-intrusive, continuous way to estimate stress from data users already generate while typing.

4

## Slide 5

Objective

1

Identify typing-behaviour features (speed, pauses, backspaces) that correlate with stress.

2

Analyse text vocabulary (word choice, sentiment) as an additional stress indicator.

3

Build an ML model that classifies stress level as Low, Medium, or High.

4

Design a lightweight, non-intrusive system usable during normal typing.

5

## Slide 6

Proposed System

Data Capture

Records keystroke timings and typed text via a typing interface.

→

Feature Extraction

Derives typing-speed, pause, backspace, and vocabulary/sentiment features.

→

Preprocessing

Cleans and normalizes extracted features for modelling.

→

ML Classification

Classifies combined features into a stress-level category.

→

Result Display

Shows the predicted stress level with a confidence score.

Trained end-to-end using a supervised-learning approach on labelled typing sessions.

6

## Slide 7

Methodology

1

Literature Review

2

Data Collection

3

Data Preprocessing

4

Model Building

5

Model Training

6

Performance Evaluation

Study prior keystroke-dynamics and text-based stress/emotion research.

Collect keystroke logs and typed text from volunteers under varied conditions.

Extract numeric typing features; apply NLP (tokenization, sentiment) to text.

Train ML classifiers on combined typing + vocabulary features.

Split data into training/testing sets; fit the chosen model.

Assess using accuracy, precision, recall, and confusion matrix.

7

## Slide 8

System Architecture

User Types
Text

→

Keystroke &
Text Capture

→

Feature
Extraction

→

ML Model
Classification

→

Stress Level
Displayed

A linear pipeline: raw typing input is progressively transformed into a stress-level prediction shown back to the user.

8

## Slide 9

Technologies Used

P

Python

Core programming language

S

Scikit-learn

Building & evaluating ML models

N

NLTK / TextBlob

NLP & sentiment analysis

J

JavaScript

Browser-based keystroke capture

P

Pandas / NumPy

Data handling & feature engineering

9

## Slide 10

Dataset

Keystroke timing logs (key-press / key-release events) and the corresponding typed text will be collected from volunteers performing typing tasks under relaxed and time-pressured conditions, supplemented with publicly available keystroke-dynamics datasets where suitable. Each session will be labelled with a stress level for supervised learning.

Training Set

80%

Testing Set

20%

Labels: Low  •  Medium  •  High stress

10

## Slide 11

Applications

W

Workplace Monitoring

Helps organisations support employee well-being.

E

E-Learning Platforms

Tracks student stress during exams or coursework.

H

HR Analytics

Early identification of employee burnout risk.

R

Research & Wellness Apps

Supports mental-health and productivity tools.

11

## Slide 12

Expected Outcome

✓

A working model that classifies stress level from typing behaviour and vocabulary.

✓

Non-intrusive, low-cost detection with no additional hardware.

✓

A simple interface displaying the predicted stress level.

✓

A foundation for integration into wellness or productivity apps.

12

## Slide 13

Conclusion

This project aims to show that a person's stress level can be estimated from everyday typing behaviour and vocabulary, using AI/ML techniques and without any additional hardware.

Being in its initial stage, the project currently focuses on establishing the core idea, identifying relevant features, and outlining the methodology. With further development, it can grow into a real-time stress-monitoring tool for workplaces, e-learning platforms, and wellness applications.

13

## Slide 14

References

Research Papers

•

Vizer, L. M. — “Detecting Cognitive and Physical Stress Through Typing Behavior.” CHI '09 Extended Abstracts on Human Factors in Computing Systems, 2009.

•

Sağbaş, E. A., Korukoglu, S., & Balli, S. — “Stress Detection via Keyboard Typing Behaviors by Using Smartphone Sensors and Machine Learning Techniques.” Journal of Medical Systems, 44(4), 2020.

•

Ghosh, S., Ganguly, N., Mitra, B., & De, P. — “TapSense: Combining Self-Report Patterns and Typing Characteristics for Smartphone Based Emotion Detection.” Proceedings of MobileHCI, 2017.

14

## Slide 15

Thank You

Stress Level Detection Using User Typing Behavior and Text Vocabulary