"""Data preparation utilities for the clinical text classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from clinical_text_classifier.sample_data import generate_sample_dataset

LOGGER = logging.getLogger(__name__)
MTSAMPLES_URL = (
    "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
)


def ensure_dataset(raw_dir: str, force: bool = False) -> Path:
    """Download the public mirror of MTSamples or fall back to synthetic data."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    output_file = raw_path / "mtsamples.csv"

    if output_file.exists() and not force:
        LOGGER.info("Using existing dataset at %s", output_file)
        return output_file

    try:
        df = pd.read_csv(MTSAMPLES_URL)
        df.to_csv(output_file, index=False)
        LOGGER.info("Downloaded %s rows to %s", len(df), output_file)
    except Exception as exc:  # pragma: no cover - exercised only on download failure
        LOGGER.warning("Dataset download failed: %s", exc)
        LOGGER.info("Falling back to synthetic sample data for pipeline validation.")
        generate_sample_dataset(str(output_file))

    return output_file


def clean_dataset(
    df: pd.DataFrame,
    min_class_samples: int,
    min_doc_length: int,
) -> pd.DataFrame:
    """Normalize and filter the raw dataset."""
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    )

    required_columns = {"transcription", "medical_specialty"}
    missing_columns = required_columns - set(normalized.columns)
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_columns)}")

    normalized = normalized[["transcription", "medical_specialty"]].copy()
    normalized = normalized.dropna(subset=["transcription", "medical_specialty"])
    normalized["transcription"] = normalized["transcription"].astype(str).str.strip()
    normalized["medical_specialty"] = (
        normalized["medical_specialty"].astype(str).str.strip()
    )

    normalized = normalized[normalized["transcription"].str.len() > 0]
    word_counts = normalized["transcription"].str.split().str.len()
    normalized = normalized[word_counts >= min_doc_length]

    class_counts = normalized["medical_specialty"].value_counts()
    keep_labels = class_counts[class_counts >= min_class_samples].index
    normalized = normalized[normalized["medical_specialty"].isin(keep_labels)]

    return normalized.reset_index(drop=True)


def create_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits."""
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["medical_specialty"],
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df["medical_specialty"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(
        drop=True
    )


def save_splits(
    processed_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Path]:
    """Persist prepared splits to disk."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    train_df.to_csv(outputs["train"], index=False)
    val_df.to_csv(outputs["val"], index=False)
    test_df.to_csv(outputs["test"], index=False)
    return outputs
