# Sentiment140 Twitter Sentiment Analysis

This project analyzes noisy Twitter text using both classical and neural NLP models on the Sentiment140 dataset. The notebook builds and compares TF-IDF based sentiment classifiers alongside an MLP neural baseline, with a focus on reproducible evaluation and robustness to social-media-specific noise.

## Project Contents

- `SentimentsProject.ipynb`: main notebook for data loading, preprocessing, feature extraction, model training, tuning, evaluation, and error analysis
- `Project Proposal.pdf`: project proposal and contribution plan

## What The Notebook Covers

- Loads the Sentiment140 training set from Kaggle using `kagglehub`
- Cleans tweets with basic and advanced preprocessing
- Includes exploratory baseline cells for step-by-step development before the final controlled comparison
- Builds controlled 80/10/10 train, validation, and test splits for fair comparison
- Builds TF-IDF features with unigram and bigram text features
- Tunes and compares Logistic Regression and Linear SVM baselines
- Adds an `MLPClassifier` neural baseline with early stopping on top of a reduced dense representation built with `TruncatedSVD`
- Reports accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, training time, throughput, and parameter counts
- Evaluates performance on both a broad noisy subset and a stricter high-noise subset containing concentrated Twitter-specific artifacts
- Produces grouped error-analysis tables and representative model mistakes

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

The controlled comparison section includes `EXPERIMENT_SAMPLE_FRACTION = 1.0` by default. Lowering that value can make iteration faster on smaller machines while keeping the same workflow. The notebook also caps the default MLP training rows so the neural baseline remains runnable without changing the shared validation and test splits, and uses a `TF-IDF -> TruncatedSVD -> MLP` pipeline so the dense model does not train directly on raw sparse TF-IDF features.

## Notes

- The notebook includes a `%pip` cell for `kagglehub[pandas-datasets]`, which helps in Colab or fresh notebook environments.
- The dataset itself is not committed to this repo.
- Saved `joblib` artifacts are produced for the tuned Logistic Regression, Linear SVM, and MLP models when the final export cell is run. When the MLP uses dimensionality reduction, its fitted `TruncatedSVD` transformer is also exported.
- The final export cell also writes `split_summary`, subset definitions, validation results, final comparison metrics, robustness deltas, noise-summary data, and error-analysis tables to `outputs/` as CSV files for the report.
- Saved figures in `outputs/` include combined confusion matrices and a bar chart of the high-noise F1 deltas.
- By default, Logistic Regression and Linear SVM train on the full split, while the MLP neural baseline trains on a capped stratified subset for computational feasibility. You can raise or remove that cap in the notebook if you have more compute.
- The final conclusions should be taken from the controlled-comparison section near the end of the notebook, not from the earlier exploratory baseline cells.
- Results depend on executing the notebook cells in order because the tuned-comparison section reuses the shared split and evaluation helpers defined earlier in the notebook.
