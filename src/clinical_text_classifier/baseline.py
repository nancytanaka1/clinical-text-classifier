"""Baseline model training utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    metrics_path: Path


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
    metrics_dir: str,
    max_features: int,
    ngram_range: tuple[int, int],
) -> BaselineArtifacts:
    """Train and save a baseline classifier."""
    pipeline = build_baseline_pipeline(max_features=max_features, ngram_range=ngram_range)
    pipeline.fit(train_df["text"], train_df["label"])

    predictions = pipeline.predict(val_df["text"])
    macro_f1 = f1_score(val_df["label"], predictions, average="macro")
    report = classification_report(val_df["label"], predictions)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline.joblib"
    joblib.dump(pipeline, model_path)

    metrics_output_dir = Path(metrics_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_output_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "macro_f1": macro_f1,
                "labels": sorted(val_df["label"].unique().tolist()),
                "validation_rows": len(val_df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return BaselineArtifacts(
        macro_f1=macro_f1,
        report=report,
        model_path=model_path,
        metrics_path=metrics_path,
    )
