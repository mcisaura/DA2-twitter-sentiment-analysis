# Robust Sentiment Analysis on Noisy Twitter Data

## A Comparison of Classical Machine Learning and Neural Models

**Group 16**  
**Authors:** Oluchi Nwabuoku and Pete Sankar

## ABSTRACT

This project studies large-scale Twitter sentiment classification on the Sentiment140 dataset under a shared experimental design. The final notebook compares four model families: TF-IDF + Logistic Regression, TF-IDF + linear SVM, a neural baseline implemented as TF-IDF -> TruncatedSVD -> MLP, and an LSTM sequence model. In addition to the model comparison, the final workflow includes a preprocessing ablation, character n-gram baselines for the classical models, macro and class-wise metrics, and a robustness evaluation on a stricter subset of noisy tweets. The strongest full-test model was the **LSTM Sequence Model**, which achieved macro F1 `0.817416` and ROC-AUC `0.899443` on the held-out test set. On the `high_noise_test` subset of 34,445 tweets, all models lost macro F1 relative to the full test split, but the LSTM remained the strongest overall. These results show that neural sequence modeling can outperform the stronger classical baselines on this task, although that gain comes with a much higher runtime cost.

## 1. INTRODUCTION

Sentiment analysis is a core natural language processing task with applications in customer feedback analysis, public opinion tracking, product monitoring, journalism, and content moderation. Twitter-style text is especially difficult because it is short, noisy, and highly informal. Tweets often contain mentions, hashtags, repeated characters, abbreviations, slang, URLs, and unconventional punctuation. These features can interfere with traditional bag-of-words approaches, but they can also provide useful signal if the model and preprocessing pipeline handle them correctly.

The goal of this project is to compare strong classical machine learning baselines against neural approaches under the same train/validation/test split and the same evaluation metrics. The central question is whether more complex neural models provide a meaningful advantage over well-tuned TF-IDF based baselines when the data contains Twitter-specific linguistic noise.

## 2. RELATED WORK

The proposal framed this project around the broader progression of sentiment analysis methods, from sparse lexical models such as TF-IDF + linear classifiers to neural architectures that learn denser and more contextual representations. Classical baselines often remain strong because they are efficient, interpretable, and robust on large labeled corpora. At the same time, neural models are expected to better capture sequential and contextual effects such as negation, intensification, and multi-token phrase patterns.

This project does not attempt a full literature survey. Instead, it follows the comparative logic described in the proposal: establish strong Logistic Regression and linear SVM baselines, compare them to neural approaches, and then evaluate how performance changes on a more difficult subset containing higher concentrations of Twitter-specific noise.

## 3. PROPOSED METHODS

The final notebook uses the Sentiment140 dataset from Kaggle and loads `training.1600000.processed.noemoticon.csv`. The original Sentiment140 labels are remapped into a binary target where `0 = negative` and `1 = positive`. The final controlled split is stratified and fixed across all models:

- Train: 1,280,000
- Validation: 160,000
- Test: 160,000

### 3.1 Preprocessing

To close the missing proposal work, the notebook evaluates multiple preprocessing variants before the final comparison. These variants differ in tokenization and normalization choices, including whitespace tokenization, `TweetTokenizer`, stemming, and lemmatization. The strongest variant was **`advanced_tweet_stem`**, which used:

- tokenizer: `tweet`
- stemming: `True`
- lemmatization: `False`
- advanced normalization: `True`

Its validation macro F1 was `0.794262`.

### 3.2 Models

The final comparison includes four model families:

- Logistic Regression with tuned TF-IDF features
- linear SVM with tuned TF-IDF features
- MLP neural baseline using `TF-IDF -> TruncatedSVD`
- LSTM sequence model over tokenized tweet text

The classical models search both word and character n-gram TF-IDF spaces. The MLP uses a reduced dense representation created by TruncatedSVD, while the LSTM consumes token sequences directly.

### 3.3 Robustness Subset

To directly address the proposal’s research question, the notebook evaluates models on both the full held-out test set and a stricter `high_noise_test` subset. The subset contains tweets with at least two noise markers and at least one strong Twitter-specific artifact. Its size is 34,445 tweets out of 160,000 held-out test examples.

## 4. EVALUATION OF MODELS

Hyperparameters are selected on the validation split, and the chosen model configurations are evaluated once on the held-out test split. The final comparison reports:

- accuracy
- precision
- recall
- F1
- macro precision
- macro recall
- macro F1
- class-wise precision, recall, and F1 for both labels
- ROC-AUC
- training time
- prediction time and throughput
- parameter count

The use of macro F1 is particularly important because it gives a more balanced summary of classifier behavior across both sentiment labels, while ROC-AUC captures the ranking quality of model scores independently of one fixed threshold.

## 5. RESULTS AND DISCUSSION

### 5.1 Full Test Results

Full-test macro F1 and ROC-AUC are summarized below:

- Logistic Regression: macro F1 `0.800841`, ROC-AUC `0.881079`
- Linear SVM: macro F1 `0.800359`, ROC-AUC `0.880771`
- MLP Neural Baseline: macro F1 `0.759564`, ROC-AUC `0.840731`
- LSTM Sequence Model: macro F1 `0.817416`, ROC-AUC `0.899443`

The strongest model on the full held-out test set was the **LSTM Sequence Model**. The two classical baselines remained very close to one another and clearly outperformed the MLP baseline, but neither matched the LSTM on macro F1 or ROC-AUC.

### 5.2 High-Noise Robustness Results

High-noise macro F1 values were:

- Logistic Regression: `0.770443`
- Linear SVM: `0.770236`
- MLP Neural Baseline: `0.737848`
- LSTM Sequence Model: `0.791648`

Macro F1 deltas relative to the full test split were:

- Logistic Regression: `-0.030398`
- Linear SVM: `-0.030123`
- MLP Neural Baseline: `-0.021716`
- LSTM Sequence Model: `-0.025768`

This robustness analysis shows that the noisy subset remains harder overall by macro F1, but the LSTM degrades less than the two classical baselines in macro F1. That does not make the classical models weak; they are still strong and dramatically faster. It does suggest, however, that the LSTM captures some sequential structure that the sparse models handle less effectively under noisier conditions.

### 5.3 Runtime Discussion

Training times on the final evaluation run were:

- Logistic Regression: `6.079` seconds
- Linear SVM: `6.140` seconds
- MLP Neural Baseline: `10.835` seconds
- LSTM Sequence Model: `307.742` seconds

Approximate test throughput in tweets per second was:

- Logistic Regression: `66005484.23`
- Linear SVM: `62359933.74`
- MLP Neural Baseline: `1942898.03`
- LSTM Sequence Model: `37849.01`

The LSTM gives the strongest predictive performance, but it is far more expensive than the classical baselines. That tradeoff matters. If runtime and simplicity are the main priorities, the classical models remain attractive. If the priority is maximum held-out performance, the LSTM is the best model in this final run.

### 5.4 Error Analysis

Representative false negatives:
- `what a gr8 day, sun is shining &amp; my exams are over 4 this semester!! `
- `off to the metro centre for more hair extensions  and i NEED sunglasses!`
- `@misskeish I thought I was the only one who required nice hands. `

Representative false positives:
- `I am finally home from my night out. It was fun fun &amp; bitter sweet   Everyone got home safe. Good night or shall I say Good Morning..hehe`
- `We will be changing our name again  Please stay tuned and follow our new page once Ophelia barks up a new name for http://dog-wuh.com thx!`
- `@mileycyrus and your fans in Chile? haha  Chile love u!andwaiting for u!you makesan awesome job onhannah montana!u are an amazing actress!`

These examples show a mix of difficult linguistic cases and likely label noise from the distant-supervision process used in Sentiment140. They also reinforce that even the best model still struggles on some tweets whose sentiment depends on subtle context or mismatched surface cues.

## 6. CONCLUSIONS

The final project provides a much more complete answer to the original proposal than the earlier notebook state. It includes the classical baselines, an MLP baseline, and an LSTM sequence model on the same fixed split; it documents preprocessing choices through an ablation; it evaluates word and character TF-IDF baselines; and it reports macro and class-wise metrics in the final comparison.

The strongest result in the latest run is that the **LSTM Sequence Model** was the best-performing model on the held-out comparison, with macro F1 `0.817416` and ROC-AUC `0.899443`. At the same time, the classical baselines remained highly competitive while being much faster and simpler to train. The final takeaway is therefore not just that the LSTM wins, but that the best model choice depends on whether predictive performance or computational practicality matters more.

## 7. RESOURCES

This project uses:

- Sentiment140 from Kaggle as the primary dataset
- `scikit-learn` for Logistic Regression, linear SVM, and the MLP baseline
- `PyTorch` for the LSTM sequence model
- `nltk` for tokenization and stemming choices in the preprocessing ablation
- `kagglehub` for dataset loading

## 8. CONTRIBUTIONS

The proposal split work across data loading, preprocessing, baselines, robustness analysis, evaluation, and the final writeup. In the final notebook, those areas are represented by:

- data loading and preprocessing
- shared train/validation/test split construction
- Logistic Regression baseline
- linear SVM baseline
- MLP baseline
- LSTM sequence model
- robustness subset construction
- final evaluation and report writing

## 9. REFERENCES

1. Sentiment140 dataset: `https://www.kaggle.com/datasets/kazanova/sentiment140`
2. Notebook and exported artifacts in this repository
3. `scikit-learn` documentation: `https://scikit-learn.org/`
4. `PyTorch` documentation: `https://pytorch.org/`
