"""Baseline model training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline


@dataclass
class BaselineArtifacts:
    """Outputs produced by baseline training."""

    macro_f1: float
    report: str
    model_path: Path


def build_baseline_pipeline(max_features: int, ngram_range: tuple[int, int]) -> Pipeline:
    """Create the TF-IDF plus logistic regression baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    multi_class="ovr",
                ),
            ),
        ]
    )


def train_baseline_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: str,
    max_features: int,
    ngram_range: tuple[int, int],
) -> BaselineArtifacts:
    """Train and save a baseline classifier."""
    pipeline = build_baseline_pipeline(max_features=max_features, ngram_range=ngram_range)
    pipeline.fit(train_df["transcription"], train_df["medical_specialty"])

    predictions = pipeline.predict(val_df["transcription"])
    macro_f1 = f1_score(val_df["medical_specialty"], predictions, average="macro")
    report = classification_report(val_df["medical_specialty"], predictions)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline.joblib"
    joblib.dump(pipeline, model_path)

    return BaselineArtifacts(
        macro_f1=macro_f1,
        report=report,
        model_path=model_path,
    )
