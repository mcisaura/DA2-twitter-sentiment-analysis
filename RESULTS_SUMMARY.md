# Sentiment140 Results Summary

This summary reflects the most recent executed notebook outputs. If the controlled-comparison code changes, rerun the notebook section that generates `outputs/` before treating these numbers as final.

The final comparison used a fixed stratified 80/10/10 train/validation/test split:

- Train: 1,280,000 tweets
- Validation: 160,000 tweets
- Test: 160,000 tweets

The strongest models were the classical TF-IDF baselines.

- `Linear SVM` achieved the best full-test F1-score: `0.800978`
- `Logistic Regression` achieved a nearly identical full-test F1-score: `0.800764`
- `Logistic Regression` achieved the best full-test ROC-AUC: `0.878590`
- `MLP Neural Baseline` underperformed both classical models with full-test `F1 = 0.747015` and `ROC-AUC = 0.822701`

Robustness was evaluated on two derived subsets:

- `any_noise_test`: 106,942 tweets with at least one Twitter-specific noise marker
- `high_noise_test`: 34,445 tweets with at least two noise markers and at least one strong noise marker such as a hashtag, repeated characters, slang, emoticon, or URL

On the stricter `high_noise_test` subset:

- `Linear SVM`: `F1` increased from `0.800978` to `0.809510`
- `Logistic Regression`: `F1` increased from `0.800764` to `0.808773`
- `MLP Neural Baseline`: `F1` increased from `0.747015` to `0.768644`

At the same time, `accuracy` and `ROC-AUC` fell on the high-noise subset for all models. For the MLP, `accuracy` dropped by `0.011472` and `ROC-AUC` dropped by `0.020591`. That means the high-noise subset was still harder overall, but the thresholded precision-recall tradeoff shifted in a way that slightly increased F1.

The runtime results also favored the classical models.

- `Logistic Regression` trained in about `7.20` seconds on the final train+validation split
- `Linear SVM` trained in about `8.85` seconds
- `MLP Neural Baseline` trained in about `5.15` seconds despite using only `100,000` training rows and a reduced dense representation
- The linear models were also much faster at inference than the MLP

The main error-analysis pattern was that negation appeared much more often in false negatives than in false positives. Representative false negatives often contained strongly negative surface words inside tweets that were labeled positive overall, such as `no sad faces! only smiles` or `unfortunately i cannot. sorry. #badoptus`. Representative false positives often contained positive lexical cues such as `thanks`, `welcome`, `love`, or `good morning` even when the dataset label was negative, which likely reflects some distant-supervision label noise in Sentiment140.

Final conclusion: for this project, the classical TF-IDF baselines were both more accurate and more effective than the stabilized MLP neural baseline. Under the shared preprocessing and evaluation setup, increased model complexity did not provide a measurable robustness advantage on noisy Twitter sentiment classification, and the stricter high-noise analysis did not change the model ranking.
