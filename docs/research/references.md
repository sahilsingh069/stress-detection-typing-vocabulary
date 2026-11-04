# Research papers and datasets added for the final report

The PDFs sit in `docs/` and are git-ignored: they were downloaded through UPES's IEEE Xplore
licence, so cite and link them rather than publishing copies.

| Paper | Data and task | Reported | How it was evaluated | What we took from it |
|---|---|---|---|---|
| Pavić, Mutka, Tolić, Havelka Meštrović and Žagar (2026), "Detecting Student Stress through Keystroke Patterns in Programming Assignments", MIPRO. [DOI](https://doi.org/10.1109/MIPRO70003.2026.11592024) | 86 students, programming tasks, self-reported stress | CV accuracy 83%; F1 0.65 (low) and 0.79 (high) | 80/20 and 10-fold; the authors say one student's samples may be in different folds and call the scores proof of concept | Pause over 10 s, longest gap and burst features; use as an early-warning aid only, not for decisions about individuals |
| Darmawan et al. (2021), "Tree-based Ensemble Learning for Stress Detection by Typing Behavior on Smartphones", ICSECS-ICOCSIM. [DOI](https://doi.org/10.1109/ICSECS52883.2021.00078) | 46 people, smartphone motion sensors and typing, calm first and then stress (Sağbaş et al.) | Random Forest accuracy 94.77% | 10-fold; no person-independent split described | Different sensors, so not directly comparable; a reason to test how the split changes the score |
| Kene and Thakare (2021), "Mental Stress Level Prediction and Classification based on Machine Learning", STCR. [DOI](https://doi.org/10.1109/STCR51658.2021.9588803) | ECG from PhysioBank, stress vs. relaxed | Gradient boosting accuracy 83.33% | 70/30 split | Personal thresholds. Physiological, so not comparable. Its figures also disagree: accuracy 83.3% cannot come from sensitivity and specificity of 75% each |

## Dataset

Maalej, A., and Kallel, I. (2020). "Does Keystroke Dynamics tell us about Emotions? A Systematic
Literature Review and Dataset Construction." *Intelligent Environments (IE)*.
[Paper DOI](https://doi.org/10.1109/IE49459.2020.9155004),
[EmoSurv dataset](https://doi.org/10.21227/eae6-pk42) (non-commercial research use, no redistribution).

The same authors' later paper reports 93.92% accuracy with a Random Committee classifier (from the search listing). We
could not open the full text to see how it was split, so read it before comparing, or before answering a question on it.
