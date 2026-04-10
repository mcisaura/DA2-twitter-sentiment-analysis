# Sentiment140 Twitter Sentiment Analysis

This project analyzes noisy Twitter text using classical NLP models on the Sentiment140 dataset. The notebook builds and compares TF-IDF based sentiment classifiers, with a focus on reproducible evaluation and robustness to social-media-specific noise.

## Project Contents

- `SentimentsProject.ipynb`: main notebook for data loading, preprocessing, feature extraction, model training, tuning, and evaluation
- `annotated-Official%20Proposal.pdf`: project proposal and contribution plan

## What The Notebook Covers

- Loads the Sentiment140 training set from Kaggle using `kagglehub`
- Cleans tweets with basic and advanced preprocessing
- Builds TF-IDF features with unigram and bigram text features
- Trains a Logistic Regression baseline
- Tunes and evaluates a Linear SVM baseline
- Reports accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Evaluates performance on a noisy subset of tweets containing Twitter-specific artifacts

## Dataset

This project uses the Sentiment140 dataset from Kaggle:

`https://www.kaggle.com/datasets/kazanova/sentiment140`

The notebook loads:

- `training.1600000.processed.noemoticon.csv`

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook
```

## Notes

- The notebook includes a `%pip` cell for `kagglehub[pandas-datasets]`, which helps in Colab or fresh notebook environments.
- The dataset itself is not committed to this repo.
- Results in the notebook are based on the executed classical-model pipeline and saved outputs.
