# 15-Minute Presentation Outline

## Slide 1 — Title
A Multi-Stage Machine Learning Framework for Network Intrusion Detection

## Slide 2 — Problem
Traditional IDS often relies on signatures. This makes it weak against zero-day and evolving attacks. The goal is to build an anomaly-based IDS that learns patterns from network traffic.

## Slide 3 — Dataset
We use NSL-KDD, a benchmark intrusion detection dataset. Each row represents a network connection with numerical and categorical features. Labels include normal traffic and multiple attack types.

## Slide 4 — Goals
1. Detect normal vs. attack traffic.
2. Compare baseline and stronger ML models.
3. Add a statistical model component.
4. Build a simple Streamlit demo.

## Slide 5 — Baseline
Baseline models:
- Dummy majority classifier
- Gaussian Naive Bayes with PCA
- Logistic Regression with PCA

These establish a starting point before using stronger tree-based models.

## Slide 6 — Statistical Component
We analyze `src_bytes`:
- CLT: repeated sample means become approximately normal.
- Mann-Whitney U and Welch t-test: compare normal and attack traffic.
- Bootstrap CI: estimate uncertainty in the mean difference.

## Slide 7 — Multi-Stage Framework
Stage 1: binary detection, normal vs. attack.  
Stage 2: attack family classification for predicted attacks: DoS, Probe, R2L, U2R.

## Slide 8 — Improved Model
Random Forest is used for stronger binary classification. Extra Trees is used for attack family classification. One-hot encoding handles categorical variables, and log transforms reduce skewness.

## Slide 9 — Results
Discuss accuracy, macro F1, weighted F1, confusion matrices, and feature importance. Emphasize macro F1 because NSL-KDD is class-imbalanced.

## Slide 10 — Demo and Conclusion
Show the Streamlit app. Upload a traffic file, generate predictions, and download results. Conclude with limitations: benchmark data is older, rare attacks remain difficult, and future work could use CICIDS2017.
