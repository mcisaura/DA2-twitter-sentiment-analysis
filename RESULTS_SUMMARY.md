# Sentiment140 Results Summary

This summary reflects the regenerated notebook outputs from April 25, 2026.

## Split

The final comparison used a fixed stratified `80/10/10` split:

- Train: `1,280,000`
- Validation: `160,000`
- Test: `160,000`

## Preprocessing Ablation

The best preprocessing variant was `advanced_tweet_stem`:

- tokenizer: `TweetTokenizer`
- stemming: `True`
- lemmatization: `False`
- advanced normalization: `True`
- validation macro F1: `0.794262`

## Full Test Results

- `LSTM Sequence Model`: `macro F1 = 0.817416`, `ROC-AUC = 0.899443`
- `Logistic Regression`: `macro F1 = 0.800841`, `ROC-AUC = 0.881079`
- `Linear SVM`: `macro F1 = 0.800359`, `ROC-AUC = 0.880771`
- `MLP Neural Baseline`: `macro F1 = 0.759564`, `ROC-AUC = 0.840731`

The LSTM sequence model was the strongest model on the full held-out test set.

## High-Noise Results

The `any_noise_test` subset contains `106,942` tweets with at least one Twitter-specific noise marker.

Any-noise macro F1:

- `LSTM Sequence Model`: `0.807600`
- `Logistic Regression`: `0.790225`
- `Linear SVM`: `0.789570`
- `MLP Neural Baseline`: `0.750653`

The stricter `high_noise_test` subset contains `34,445` tweets with at least two noise markers and at least one strong Twitter-specific artifact.

High-noise macro F1:

- `LSTM Sequence Model`: `0.791648`
- `Logistic Regression`: `0.770443`
- `Linear SVM`: `0.770236`
- `MLP Neural Baseline`: `0.737848`

Macro F1 delta relative to the full test split:

- `LSTM Sequence Model`: `-0.025768`
- `Logistic Regression`: `-0.030398`
- `Linear SVM`: `-0.030123`
- `MLP Neural Baseline`: `-0.021716`

## Runtime

Train time:

- `Logistic Regression`: `13.652` seconds
- `Linear SVM`: `12.869` seconds
- `MLP Neural Baseline`: `20.244` seconds
- `LSTM Sequence Model`: `529.033` seconds

Approximate test throughput:

- `Logistic Regression`: `41,539,560` tweets/second
- `Linear SVM`: `37,330,120` tweets/second
- `MLP Neural Baseline`: `936,697` tweets/second
- `LSTM Sequence Model`: `29,269` tweets/second

## Statistical Checks

Approximate 95% confidence intervals for full-test accuracy:

- `LSTM Sequence Model`: `[0.815557, 0.819343]`
- `Logistic Regression`: `[0.798924, 0.802838]`
- `Linear SVM`: `[0.798460, 0.802377]`
- `MLP Neural Baseline`: `[0.757487, 0.761675]`

Paired McNemar tests on full-test predictions support the LSTM's advantage over Logistic Regression (`p ~= 4.03e-89`) and Linear SVM (`p ~= 1.28e-92`). Logistic Regression and Linear SVM were not clearly separable under the paired test (`p ~= 0.0596`).

## Error Analysis

Negation still appears much more often in false negatives than in false positives for the classical baselines. For `Linear SVM`, the negation rate is:

- false negatives: `0.292`
- false positives: `0.170`

## Limitation

Emoji coverage is effectively `0.0` in this extracted test set, so the robustness discussion should not be overstated as an emoji-heavy evaluation.
