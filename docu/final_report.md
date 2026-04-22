# Robust Sentiment Analysis on Noisy Twitter Data

## Final Research Report

**Group 16**  
**Authors:** Oluchi Nwabuoku and Pete Sankar

## Abstract

This report presents the final version of our Sentiment140 study comparing classical machine learning and neural models for sentiment classification on noisy Twitter text. Using a fixed stratified split of 1,280,000 training tweets, 160,000 validation tweets, and 160,000 test tweets, we evaluated Logistic Regression, Linear SVM, a TF-IDF -> TruncatedSVD -> MLP baseline, and an LSTM sequence model under the same preprocessing and evaluation framework. The best preprocessing pipeline was `advanced_tweet_stem`, which reached validation macro F1 `0.794262`. On the held-out full test set, the LSTM achieved the strongest overall performance with accuracy `0.817450`, macro F1 `0.817416`, and ROC-AUC `0.899443`. On the stricter `high_noise_test` subset of 34,445 tweets, the LSTM again ranked first with macro F1 `0.791648`. However, that gain came with a large efficiency cost: compared with Logistic Regression, the LSTM trained `50.62x` slower and its inference throughput was `1743.92x` lower. The final conclusion is that sequence modeling improves performance on noisy tweet sentiment classification, but strong TF-IDF linear baselines remain highly competitive when efficiency matters.

## 1. Introduction

Sentiment analysis on Twitter-style text remains difficult because tweets are short, informal, and full of platform-specific noise. Mentions, hashtags, repeated characters, abbreviations, slang, URLs, and irregular punctuation can all distort surface lexical patterns. These same artifacts can also carry sentiment signal, which makes preprocessing and model design especially important.

The central research question in this project is whether neural models provide a meaningful advantage over well-tuned classical baselines when both are evaluated on the same noisy Twitter classification task. Rather than comparing unrelated implementations, this report uses a controlled design: the same dataset, the same split, the same preprocessing selection process, and the same evaluation metrics across all models.

## 2. Dataset and Experimental Design

The study uses the Sentiment140 corpus, remapped into binary labels where `0 = negative` and `1 = positive`. The final split was fixed and stratified:

| Split | Rows | Positive Rate |
|---|---:|---:|
| Train | 1,280,000 | 0.50 |
| Validation | 160,000 | 0.50 |
| Test | 160,000 | 0.50 |

### 2.1 Preprocessing Ablation

Before the final model comparison, multiple preprocessing variants were tested on the validation split. The strongest variant was `advanced_tweet_stem`, which combined advanced normalization, `TweetTokenizer`, and stemming.

| Variant | Tokenizer | Stemming | Lemmatization | Advanced Norm. | Validation Accuracy | Validation Macro F1 | Validation ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| `advanced_tweet_stem` | tweet | True | False | True | 0.794275 | 0.794262 | 0.871926 |
| `advanced_whitespace` | whitespace | False | False | True | 0.790350 | 0.790334 | 0.870093 |
| `advanced_tweet_tokenizer` | tweet | False | False | True | 0.790350 | 0.790334 | 0.870093 |
| `advanced_tweet_lemma` | tweet | False | True | True | 0.790125 | 0.790108 | 0.869997 |
| `basic_whitespace` | whitespace | False | False | False | 0.788025 | 0.788009 | 0.867872 |

### 2.2 Models

The final comparison included four model families:

- Logistic Regression with tuned TF-IDF features
- Linear SVM with tuned TF-IDF features
- MLP Neural Baseline using `TF-IDF -> TruncatedSVD`
- LSTM Sequence Model over tokenized tweet text

The classical models searched both word and character n-gram spaces, while the LSTM operated directly on token sequences. All final metrics reported below come from the saved output tables in `outputs/`.

### 2.3 Robustness Subsets

To measure robustness under Twitter-specific noise, the project evaluated the models on both the full held-out test split and noise-focused subsets:

| Subset | Rows | Share of Test | Definition |
|---|---:|---:|---|
| `full_test` | 160,000 | 1.0000 | All held-out test tweets |
| `any_noise_test` | 106,942 | 0.6684 | At least one Twitter-specific noise marker |
| `high_noise_test` | 34,445 | 0.2153 | At least two noise markers and at least one strong noise marker |

## 3. Results

### 3.1 Full-Test Performance

The full-test comparison shows that the LSTM was the strongest model overall, followed closely by Logistic Regression and Linear SVM. The MLP baseline was clearly weaker than both the linear baselines and the LSTM.

| Model | Accuracy | Precision | Recall | F1 | Macro Precision | Macro Recall | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.800881 | 0.792554 | 0.815113 | 0.803675 | 0.801125 | 0.800881 | 0.800841 | 0.881079 |
| Linear SVM | 0.800419 | 0.790347 | 0.817763 | 0.803821 | 0.800781 | 0.800419 | 0.800359 | 0.880771 |
| MLP Neural Baseline | 0.759581 | 0.764087 | 0.751050 | 0.757513 | 0.759657 | 0.759581 | 0.759564 | 0.840731 |
| LSTM Sequence Model | 0.817450 | 0.826351 | 0.803813 | 0.814926 | 0.817686 | 0.817450 | 0.817416 | 0.899443 |

The LSTM improved full-test macro F1 over Logistic Regression by `0.016575` absolute, which is a relative gain of about `2.07%`. It also had the best ROC-AUC, indicating stronger ranking quality across thresholds.

### 3.2 Robustness on High-Noise Tweets

All models lost performance on the stricter high-noise subset, but the rank order did not change: the LSTM remained first, Logistic Regression and Linear SVM remained very close to one another, and the MLP remained last in absolute performance.

| Model | High-Noise Accuracy | High-Noise F1 | High-Noise Macro F1 | Macro F1 Delta vs Full Test | High-Noise ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.777413 | 0.810443 | 0.770443 | -0.030398 | 0.858335 |
| Linear SVM | 0.777413 | 0.810845 | 0.770236 | -0.030123 | 0.857971 |
| MLP Neural Baseline | 0.746059 | 0.784243 | 0.737848 | -0.021716 | 0.820157 |
| LSTM Sequence Model | 0.796400 | 0.823114 | 0.791648 | -0.025768 | 0.877578 |

Two points matter here. First, the LSTM had the best absolute high-noise performance. Second, the MLP showed the smallest macro F1 drop, but because its full-test baseline was already much lower, it still remained the weakest model on the noisy subset.

### 3.3 Runtime and Model Efficiency

Performance alone does not settle the model choice. The output metrics show a steep cost for the LSTM:

| Model | Training Time (s) | Prediction Time (s) | Throughput (tweets/s) | Parameter Count |
|---|---:|---:|---:|---:|
| Logistic Regression | 6.079 | 0.002424 | 66,005,484.23 | 5,001 |
| Linear SVM | 6.140 | 0.002566 | 62,359,933.74 | 5,001 |
| MLP Neural Baseline | 10.835 | 0.082351 | 1,942,898.03 | 38,657 |
| LSTM Sequence Model | 307.742 | 4.227323 | 37,849.01 | 1,313,345 |

Relative to Logistic Regression, the LSTM required `50.62x` more training time and reduced throughput by a factor of `1743.92x`. This is the main practical trade-off in the project: the LSTM is best on quality, but not on computational efficiency.

### 3.4 Approximate Confidence Intervals and Statistical Caution

Because the held-out test set contains 160,000 tweets, approximate 95% confidence intervals for accuracy are very narrow. Using a normal approximation to the binomial proportion:

- LSTM accuracy `0.817450`, 95% CI `[0.815557, 0.819343]`
- Logistic Regression accuracy `0.800881`, 95% CI `[0.798924, 0.802838]`
- Linear SVM accuracy `0.800419`, 95% CI `[0.798460, 0.802377]`
- MLP accuracy `0.759581`, 95% CI `[0.757487, 0.761675]`

These intervals suggest that the LSTM advantage in accuracy is unlikely to be random sampling noise alone. However, paired significance tests such as McNemar's test were not possible because the project artifacts do not include complete paired prediction outputs for every model. The correct interpretation is therefore that the improvement is likely meaningful, but not formally proven with paired error testing.

### 3.5 Error Analysis

The representative errors and summary outputs show that negation remains a recurring challenge, especially for the classical baselines. In the saved error summary for Linear SVM, negation appeared in `29.2%` of false negatives compared with `17.0%` of false positives. This pattern suggests that sentiment reversals and contrastive phrasing remain difficult for sparse lexical models.

Representative LSTM false negatives also show that some misses involve subtle context rather than obvious noise:

- `what a gr8 day, sun is shining & my exams are over 4 this semester!!`
- `off to the metro centre for more hair extensions and i NEED sunglasses!`
- `@misskeish I thought I was the only one who required nice hands.`

The project output also notes that emoji coverage in this extracted evaluation set is effectively negligible, so the robustness findings should not be overstated as evidence about emoji-heavy social media sentiment.

## 4. Discussion

The final results support three clear conclusions.

First, strong TF-IDF linear baselines remain difficult to beat decisively on binary tweet sentiment classification. Logistic Regression and Linear SVM both achieved macro F1 around `0.801`, which kept them very close to the best neural model while remaining far simpler and faster.

Second, the LSTM produced the best results on both the full test set and the high-noise subset. This supports the idea that sequential neural models capture contextual patterns that sparse features only partially represent. The advantage is real, but modest in size relative to cost.

Third, the MLP neural baseline did not justify its added complexity. It underperformed both linear baselines and the LSTM on the main benchmark and on the noisy subset. In this experiment, moving from sparse linear models to a shallow dense baseline was not enough; the gain only appeared once the model encoded token sequence structure directly.

These findings also suggest a natural bridge to modern transformer-based work. A tweet-specific pretrained transformer such as BERTweet would likely shift the quality frontier further upward. That said, the current study still has value because it makes the classical-versus-neural trade-off explicit in a controlled setting rather than hiding it behind a leaderboard-only comparison.

## 5. Limitations and Future Work

This project is limited in several important ways:

- It evaluates only binary sentiment classification on a single dataset.
- The dataset is based on distant supervision, so some label noise is inherent.
- The current artifact set does not allow paired significance testing across complete prediction outputs.
- Emoji-heavy robustness cannot be assessed meaningfully from this extracted split.
- No transformer baseline was included in the executed comparison.

The most useful next step would be to preserve the same evaluation structure and add transformer baselines, cross-dataset validation, and paired significance testing with saved prediction files.

## 6. Conclusion

The final project shows that the `LSTM Sequence Model` was the strongest model in the controlled Sentiment140 comparison, achieving macro F1 `0.817416` and ROC-AUC `0.899443` on the full held-out test set and macro F1 `0.791648` on the `high_noise_test` subset. At the same time, `Logistic Regression` and `Linear SVM` remained highly competitive while offering dramatically better runtime efficiency.

The main contribution of the report is therefore not only identifying the best-performing model, but quantifying the trade-off between predictive quality and computational cost on noisy Twitter data. Under this design, the LSTM wins on performance, while the classical baselines remain strong default choices when simplicity, speed, and reproducibility matter.

## References

1. Go, A., Bhayani, R., and Huang, L. *Twitter Sentiment Classification using Distant Supervision*. Stanford University, 2009.
2. Pang, B., Lee, L., and Vaithyanathan, S. *Thumbs up? Sentiment Classification using Machine Learning Techniques*. EMNLP, 2002.
3. Hutto, C., and Gilbert, E. *VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text*. ICWSM, 2014.
4. Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., and Neves, L. *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. Findings of EMNLP, 2020.
5. Nguyen, D. Q., Vu, T., and Nguyen, A. T. *BERTweet: A Pre-trained Language Model for English Tweets*. EMNLP, 2020.
6. Project artifacts in this repository, especially `outputs/comparison_results.csv`, `outputs/preprocessing_ablation_results.csv`, `outputs/split_summary.csv`, `outputs/subset_summary.csv`, `outputs/robustness_delta.csv`, and `RESULTS_SUMMARY.md`.
