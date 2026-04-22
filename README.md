# Sentiment140 Twitter Sentiment Analysis

This project studies noisy Twitter sentiment classification on the Sentiment140 dataset under a shared experimental design. The final notebook now covers the original proposal more completely by adding a preprocessing ablation, character n-gram TF-IDF baselines, macro and class-wise metrics, and an LSTM sequence model alongside the earlier classical and MLP baselines.

## Main Files

- `SentimentsProject.ipynb`: end-to-end notebook for preprocessing search, model tuning, final evaluation, robustness analysis, and artifact export
- `Project Proposal.pdf`: original proposal
- `outputs/`: core report-facing result tables and figures
- `docu/final_report.md`
- `docu/final_report.docx`

## Final Model Families

- `TF-IDF + Logistic Regression`
- `TF-IDF + Linear SVM`
- `TF-IDF -> TruncatedSVD -> MLP`
- `LSTM Sequence Model`

## What The Notebook Now Does

- loads Sentiment140 from Kaggle through `kagglehub`
- creates a fixed stratified `80/10/10` train, validation, and test split
- compares whitespace tokenization, `TweetTokenizer`, stemming, and lemmatization in a preprocessing ablation
- searches both word and character n-gram TF-IDF spaces for the classical baselines
- tunes Logistic Regression, linear SVM, MLP, and LSTM configurations on the validation split
- evaluates each model on the full held-out test set and on a stricter `high_noise_test` subset
- exports binary, macro, and class-wise metrics inside the final comparison table
- exports a smaller default set of report-facing artifacts, plus representative errors and figures

## Current Best Result

On the regenerated notebook run dated April 22, 2026:

- `LSTM Sequence Model` achieved the best full-test macro F1: `0.817416`
- `LSTM Sequence Model` also achieved the best full-test ROC-AUC: `0.899443`
- `advanced_tweet_stem` was the strongest preprocessing variant in the ablation

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Then open or execute the notebook:

```bash
jupyter notebook
```

## Notes

- Emoji coverage in this Sentiment140 extraction is effectively zero, so the robustness conclusions are strongest for mentions, hashtags, slang, repeated characters, and URLs.
- The notebook now defaults to a leaner `outputs/` folder. Extra debug-style exports and saved model artifacts are disabled by default.
