# Clinical Text Classifier

Multi-label medical specialty classification from clinical transcription notes using TF-IDF baselines and fine-tuned ClinicalBERT with LoRA.

## Problem

Given a clinical transcription document, predict the medical specialty (e.g., Orthopedic, Cardiovascular, Gastroenterology). This is a practical NLP task relevant to clinical workflow automation, EHR routing, and medical coding support.

## Dataset

**MTSamples** — ~5,000 de-identified medical transcription samples across 40 specialties.
After filtering (min 50 samples/class, min 20 words/doc), the working dataset contains ~3,000 samples across ~10 specialties.

## Approach

| Phase | Method | Status |
|-------|--------|--------|
| Baseline | TF-IDF + Logistic Regression | 🔲 |
| Deep Learning | ClinicalBERT + LoRA fine-tuning | 🔲 |
| Tracking | MLflow experiment logging | 🔲 |
| Deployment | Dockerfile + inference API | 🔲 |

## Project Structure

```
clinical-text-classifier/
├── configs/
│   └── config.yaml           # All hyperparameters and paths
├── data/
│   ├── raw/                   # Downloaded CSVs (gitignored)
│   └── processed/             # Train/val/test splits (gitignored)
├── models/                    # Saved model artifacts (gitignored)
├── notebooks/                 # EDA and experiment notebooks
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Download, clean, split pipeline
│   ├── baseline.py            # TF-IDF + LogReg (TODO)
│   ├── model.py               # ClinicalBERT classifier (TODO)
│   ├── train.py               # Training loop (TODO)
│   └── evaluate.py            # Metrics and error analysis (TODO)
├── tests/
│   └── __init__.py
├── .gitignore
├── Dockerfile                 # (TODO)
├── README.md
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.data_loader  # Downloads MTSamples + creates train/val/test splits
```

## Evaluation Metrics

- Macro F1 (primary — handles class imbalance)
- Per-class precision/recall
- Confusion matrix
- Classification report

## Tech Stack

Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), scikit-learn, MLflow, pandas, matplotlib/seaborn
