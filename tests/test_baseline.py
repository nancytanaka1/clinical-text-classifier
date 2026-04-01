"""Tests for baseline training utilities."""

from __future__ import annotations

import json

import pandas as pd

from clinical_text_classifier.baseline import train_baseline_model


def test_train_baseline_model_saves_model_and_metrics(tmp_path) -> None:
    train_df = pd.DataFrame(
        {
            "text": [
                "cardiology note with chest pain and heart rhythm concerns",
                "another cardiology follow up note with heart symptoms",
                "neurology note with tremor gait and memory symptoms",
                "another neurology consult note with weakness and aphasia",
            ],
            "label": [
                "Cardiology",
                "Cardiology",
                "Neurology",
                "Neurology",
            ],
        }
    )
    val_df = pd.DataFrame(
        {
            "text": [
                "cardiology clinic note for chest pain review",
                "neurology clinic note for tremor follow up",
            ],
            "label": ["Cardiology", "Neurology"],
        }
    )

    artifacts = train_baseline_model(
        train_df=train_df,
        val_df=val_df,
        model_dir=str(tmp_path / "models"),
        metrics_dir=str(tmp_path / "metrics"),
        max_features=100,
        ngram_range=(1, 2),
    )

    assert artifacts.model_path.exists()
    assert artifacts.metrics_path.exists()

    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert "macro_f1" in metrics
    assert metrics["validation_rows"] == 2
