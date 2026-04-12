# Sentiment140 Results Summary

The final comparison used a fixed stratified 80/10/10 train/validation/test split:

- Train: 1,280,000 tweets
- Validation: 160,000 tweets
- Test: 160,000 tweets

The strongest models were the classical TF-IDF baselines.

- `Linear SVM` achieved the best full-test F1-score: `0.800978`
- `Logistic Regression` achieved a nearly identical full-test F1-score: `0.800764`
- `Logistic Regression` achieved the best full-test ROC-AUC: `0.878590`
- `MLP Neural Baseline` underperformed both classical models with full-test `F1 = 0.778599` and `ROC-AUC = 0.865595`

On the noisy held-out subset of 106,942 tweets, performance did not drop:

- `Linear SVM`: `F1` increased from `0.800978` to `0.809168`
- `Logistic Regression`: `F1` increased from `0.800764` to `0.808697`
- `MLP Neural Baseline`: `F1` increased from `0.778599` to `0.786780`

This suggests that the selected noise markers were not inherently harmful after preprocessing. Mentions, repeated characters, hashtags, and slang often remained compatible with accurate sentiment prediction once the tweets were normalized.

The runtime results also favored the classical models.

- `Logistic Regression` trained in about `6.22` seconds on the final train+validation split
- `Linear SVM` trained in about `8.90` seconds
- `MLP Neural Baseline` trained in about `9.75` seconds despite using only `100,000` training rows
- The linear models were also much faster at inference than the MLP

The main error-analysis pattern was that negation appeared much more often in false negatives than in false positives. Representative false negatives often contained strongly negative surface words inside tweets that were labeled positive overall, such as `no sad faces! only smiles` or `unfortunately i cannot. sorry. #badoptus`. Representative false positives often contained positive lexical cues such as `thanks`, `welcome`, `love`, or `good morning` even when the dataset label was negative, which likely reflects some distant-supervision label noise in Sentiment140.

Final conclusion: for this project, the classical TF-IDF baselines were both more accurate and more efficient than the MLP neural baseline. Under the shared preprocessing and evaluation setup, increased model complexity did not provide a measurable robustness advantage on noisy Twitter sentiment classification.
