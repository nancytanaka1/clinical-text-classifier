"""Tests for core data preparation logic."""

from __future__ import annotations

import pandas as pd

from clinical_text_classifier.data import clean_dataset, create_splits


def test_clean_dataset_filters_short_docs_and_rare_classes() -> None:
    df = pd.DataFrame(
        {
            "transcription": [
                "valid clinical note with enough words for testing",
                "another valid clinical note with many useful words",
                "tiny",
                "neurology note with enough terms for evaluation",
            ],
            "medical_specialty": [
                "Cardiology",
                "Cardiology",
                "Cardiology",
                "Neurology",
            ],
        }
    )

    cleaned = clean_dataset(df, min_class_samples=2, min_doc_length=5)

    assert len(cleaned) == 2
    assert set(cleaned["medical_specialty"]) == {"Cardiology"}


def test_create_splits_preserves_all_rows() -> None:
    df = pd.DataFrame(
        {
            "transcription": [f"note {i} with sufficient words present" for i in range(20)],
            "medical_specialty": ["A"] * 10 + ["B"] * 10,
        }
    )

    train_df, val_df, test_df = create_splits(df, test_size=0.2, val_size=0.2, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df["medical_specialty"].unique()) == {"A", "B"}
