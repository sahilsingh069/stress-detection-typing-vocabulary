**Project Synopsis**

For

**STRESS LEVEL DETECTION USING USER TYPING BEHAVIOR AND TEXT VOCABULARY**

26 August 2026

**Prepared by**

| **Specialization** | **SAP ID** | **Name** |
| --- | --- | --- |
| AIML | 500123925 | Anshuman Tiwari |
| AIML | 500124268 | Sahil Panwar |
| AIML | 500124730 | Royal Bisht |

**Guided By: Dr. Souvik Dev**

**School Of Computer Science**

**UNIVERSITY OF PETROLEUM ****&**** ENERGY STUDIES,**

**DEHRADUN- 248007, Uttarakhand**

**Table of Contents**

| **#** | **Topic** | **Page No** |
| --- | --- | --- |
| 1 | Introduction | 1 |
| 2 | Problem Statement | 1 |
| 3 | Objectives | 1 |
| 4 | Proposed System | 2 |
| 5 | Methodology | 2 |
| 6 | System Architecture | 3 |
| 7 | Technologies Used | 3 |
| 8 | Dataset | 3 |
| 9 | Result and Analysis | 4 |
| 10 | Applications | 4 |
| 11 | Expected Outcome | 4 |
| 12 | Conclusion | 4 |
| 13 | References | 5 |
|  | Appendix A: Glossary | 5 |
|  | Appendix B: Analysis Model | 6 |
|  | Appendix C: Issues List | 7 |

# **1. Introduction**

Stress has become one of the most common problems affecting individuals in academic, professional, and personal life. Prolonged and undetected stress can lead to reduced productivity, poor decision-making, and long-term mental and physical health issues. Early identification of stress is important so that timely support or intervention can be provided.

Traditional methods of stress assessment rely on self-reported questionnaires or physiological sensors such as heart-rate monitors and skin-conductance sensors. Questionnaires depend on the honesty and self-awareness of the individual and are usually filled only occasionally, while sensor-based methods require additional hardware, making them expensive and intrusive for everyday use.

With the growth of Artificial Intelligence (AI) and Machine Learning (ML), it is now possible to infer a person's mental state from data that is already generated during normal computer usage. Research has shown that typing behaviour (such as typing speed, pause duration, and backspace frequency) and the vocabulary used while typing (such as choice of words and sentiment) change when a person is under stress. This project aims to leverage these behavioural and textual cues to build an AI/ML based system that can detect a user's stress level in a low-cost and non-intrusive manner.

# **2. Problem Statement**

Existing stress detection techniques either depend on additional physiological hardware or on periodic self-reported surveys, both of which are impractical for continuous, real-time monitoring. Physiological sensors are costly and intrusive, while self-reported methods are subjective, infrequent, and prone to bias.

There is a need for a cost-effective, non-intrusive, and continuous method of stress detection that works using data that is already available, such as the way a person types and the words they choose while typing. The system should be able to analyse typing patterns and text content to estimate the user's stress level without requiring any special hardware.

# **3. Objectives**

The main objectives of the project are:

- To study and identify typing behaviour features (typing speed, pause duration, backspace/error rate, key-hold time) that correlate with stress.

- To analyse text vocabulary features (word choice, sentiment polarity, use of negative words) as indicators of stress.

- To design a Machine Learning model that classifies a user's stress level (e.g., Low, Medium, High) based on the combined typing and vocabulary features.

- To build a lightweight, non-intrusive system that can work in the background during normal typing activity.

- To evaluate the model's performance using standard metrics such as accuracy, precision, recall, and confusion matrix.

# **4. Proposed System**

The proposed system consists of the following modules:

- Data Capture Module – Records keystroke timings (key-press, key-release, dwell time, flight time) and the text typed by the user through a simple typing interface or browser-based tool.

- Feature Extraction Module – Extracts typing-behaviour features (typing speed, pause frequency, backspace count, typing rhythm) and text-vocabulary features (sentiment score, word frequency, use of negative/stress-related words) using NLP techniques.

- Preprocessing Module – Cleans and normalizes the extracted numerical and textual features so that they are suitable for model training.

- ML Classification Module – Uses supervised Machine Learning algorithms (such as Random Forest, SVM, or Logistic Regression) to classify the combined feature set into stress-level categories.

- Result Display Module – Displays the predicted stress level to the user along with a confidence score.

The system will be trained using a supervised learning approach on labelled typing sessions.

# **5. Methodology**

The project will follow these steps:

- Literature Review – Existing research papers on keystroke dynamics and text-based stress/emotion detection will be studied to understand prior approaches.

- Data Collection – Typing data (keystroke timings and typed text) will be collected using a custom typing-test application from volunteers, supplemented with publicly available keystroke-dynamics datasets where available.

- Data Preprocessing – Keystroke logs will be converted into numerical features (speed, pauses, backspaces), and typed text will be processed using NLP techniques (tokenization, sentiment analysis, stop-word removal).

- Model Building – Machine Learning classifiers will be trained on the combined typing-behaviour and vocabulary features to predict stress level.

- Model Training – The dataset will be split into training and testing sets, and the model will be trained using an appropriate loss/objective function for classification.

- Performance Evaluation – The model's performance will be evaluated using accuracy, precision, recall, and confusion matrix.

# **6. System Architecture**

The system architecture consists of the following stages: First, the user types text into the input interface, during which keystroke timings and the typed content are recorded. The typing-behaviour features and text-vocabulary features are then extracted. These combined features are passed to the trained ML model, which classifies the user's stress level. Finally, the predicted stress level is displayed to the user.

# **7. Technologies Used**

The following technologies are proposed to be used in this project:

- Python – Programming language for model development.

- Scikit-learn – Library for building and evaluating Machine Learning models.

- NLTK / TextBlob – Libraries for Natural Language Processing and sentiment analysis of typed text.

- JavaScript (Keystroke Capture) – For recording keystroke timings through a browser-based typing interface.

- Pandas / NumPy – For data handling, feature extraction, and numerical computation.

# **8. Dataset**

The dataset for this project will consist of keystroke timing logs (key-press and key-release events) along with the corresponding typed text, collected from volunteers performing typing tasks under varying conditions (relaxed vs. time-pressured), supplemented with publicly available keystroke-dynamics datasets where suitable.

Each typing session will be labelled with a self-reported or task-induced stress level (e.g., Low, Medium, High) to enable supervised learning.

The dataset will be divided into:

- Training Set (80%)

- Testing Set (20%)

Data balancing and augmentation techniques will be explored to handle any class imbalance between stress-level categories.

# **9. Result and Analysis**

As this is the initial stage of the project, results are yet to be generated. The ML model is expected to classify stress levels with reasonable accuracy during testing. Accuracy will be calculated as:

**Accuracy = (Correct Predictions / Total Predictions) × 100**

# **10. Applications**

The system can be used in:

- Workplace stress-monitoring tools for employee well-being.

- Mental-health and wellness mobile/web applications.

- E-learning platforms to monitor student stress during exams or coursework.

- HR analytics systems for early identification of employee burnout.

# **11. Expected Outcome**

The expected outcomes of the project include:

- A working model that classifies stress level from typing behaviour and text vocabulary.

- Non-intrusive and low-cost stress detection without additional hardware.

- A basic user interface displaying the predicted stress level.

- A foundation for integration into wellness or productivity applications.

# **12. Conclusion**

This project aims to demonstrate that a person's stress level can be estimated using AI/ML techniques applied to everyday typing behaviour and the vocabulary used while typing, without requiring any additional hardware. Being in its initial stage, the project currently focuses on establishing the core idea, identifying relevant features, and outlining the methodology to be followed.

With further development, the system can be extended into a real-time stress-monitoring tool that can be integrated into workplace software, e-learning platforms, and mental-wellness applications.

# **13. References**

**Research Papers:**

- Vizer, L. M. – “Detecting Cognitive and Physical Stress Through Typing Behavior.” CHI '09 Extended Abstracts on Human Factors in Computing Systems, 2009.

- Sağbaş, E. A., Korukoglu, S., & Balli, S. – “Stress Detection via Keyboard Typing Behaviors by Using Smartphone Sensors and Machine Learning Techniques.” Journal of Medical Systems, 44(4), 2020.

- Ghosh, S., Ganguly, N., Mitra, B., & De, P. – “TapSense: Combining Self-Report Patterns and Typing Characteristics for Smartphone Based Emotion Detection.” Proceedings of MobileHCI, 2017.

# **Appendix A: Glossary**

Artificial Intelligence (AI): A branch of computer science that enables machines to simulate human intelligence and decision-making.

Machine Learning (ML): A subset of AI that allows systems to learn patterns from data and improve performance without being explicitly programmed.

Keystroke Dynamics: The study of the manner and rhythm in which a person types on a keyboard, including timing between key presses and releases.

Dwell Time: The duration for which a key is held down during a keystroke.

Flight Time: The time interval between releasing one key and pressing the next.

Sentiment Analysis: An NLP technique used to determine the emotional tone (positive, negative, neutral) expressed in a piece of text.

Natural Language Processing (NLP): A branch of AI concerned with enabling computers to understand and analyse human language.

Dataset: A collection of labelled typing sessions used to train and test the model.

Training Data: The portion of the dataset used to teach the model.

Testing Data: The portion of the dataset used to evaluate the model's performance.

# **Appendix B: Analysis Model**

The project follows a supervised learning approach that combines behavioural (keystroke) and textual (vocabulary) features to classify stress level.

## ***Problem Analysis***

The system is designed to classify a user's typing session into one of several stress-level categories (e.g., Low, Medium, High). The problem is treated as a multi-class classification task.

## ***System Model***

- Input Layer – Accepts keystroke timing data and typed text.

- Feature Extraction Layer – Derives typing-behaviour and vocabulary features.

- Classification Layer – Machine Learning model predicts the stress-level category.

- Output Layer – Displays the predicted stress level.

## ***Data Analysis***

- Stress-level categories: Low, Medium, High

- Data collected from volunteer typing sessions under varied conditions

- Data split into training and testing sets

- Keystroke features normalized before model training

# **Appendix C: Issues List**

The following issues have been identified at this stage of the project:

- Limited availability of labelled datasets combining keystroke dynamics with text vocabulary for stress detection.

- Variation in typing behaviour across individuals may affect model generalization.

- Difficulty in obtaining reliable ground-truth stress labels for training data.

- Risk of overfitting due to a small initial dataset.

- Privacy concerns related to logging keystrokes and typed content.

- Need for continuous data collection to improve model accuracy over time.