"""
Data loading and preprocessing for MTSamples clinical text classification.

Downloads the MTSamples dataset (medical transcriptions) and prepares it
for multi-label specialty classification.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MTSamples CSV hosted on GitHub (public mirror, no Kaggle API needed)
MTSAMPLES_URL = (
    "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing project configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_mtsamples(raw_dir: str = "data/raw", force: bool = False) -> Path:
    """Download MTSamples dataset from public GitHub mirror.

    Args:
        raw_dir: Directory to save the raw CSV.
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded CSV file.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    output_file = raw_path / "mtsamples.csv"

    if output_file.exists() and not force:
        logger.info(f"Dataset already exists at {output_file}. Use force=True to re-download.")
        return output_file

    logger.info(f"Downloading MTSamples from {MTSAMPLES_URL}...")
    try:
        df = pd.read_csv(MTSAMPLES_URL)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        logger.info("Falling back to synthetic sample data for pipeline testing.")
        logger.info(
            "For real data, download MTSamples from Kaggle:\n"
            "  kaggle datasets download tobiasbueschel/medical-transcriptions\n"
            "  unzip medical-transcriptions.zip -d data/raw/"
        )
        from src.sample_data import generate_sample_dataset
        generate_sample_dataset(str(output_file))

    return output_file


def clean_mtsamples(
    df: pd.DataFrame, min_class_samples: int = 50, min_doc_length: int = 20
) -> pd.DataFrame:
    """Clean and filter the MTSamples dataset.

    Applies the following:
    - Drops rows with missing transcription or specialty.
    - Strips whitespace from text fields.
    - Filters out specialties with fewer than `min_class_samples` examples.
    - Filters out documents shorter than `min_doc_length` words.
    - Resets index.

    Args:
        df: Raw MTSamples DataFrame.
        min_class_samples: Minimum samples per specialty to retain the class.
        min_doc_length: Minimum word count to keep a document.

    Returns:
        Cleaned DataFrame with columns: ['transcription', 'medical_specialty'].
    """
    logger.info(f"Raw dataset shape: {df.shape}")

    # Keep only relevant columns
    df = df[["transcription", "medical_specialty"]].copy()

    # Drop missing values
    df.dropna(subset=["transcription", "medical_specialty"], inplace=True)

    # Strip whitespace
    df["transcription"] = df["transcription"].str.strip()
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    # Filter short documents
    word_counts = df["transcription"].str.split().str.len()
    df = df[word_counts >= min_doc_length]
    logger.info(f"After min doc length filter ({min_doc_length} words): {len(df)} rows")

    # Filter rare specialties
    specialty_counts = df["medical_specialty"].value_counts()
    valid_specialties = specialty_counts[specialty_counts >= min_class_samples].index
    df = df[df["medical_specialty"].isin(valid_specialties)]
    logger.info(
        f"After min class filter ({min_class_samples} samples): "
        f"{len(df)} rows, {df['medical_specialty'].nunique()} specialties"
    )

    df.reset_index(drop=True, inplace=True)
    return df


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits.

    Args:
        df: Cleaned DataFrame.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["medical_specialty"],
    )

    # Second split: separate validation from training
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df["medical_specialty"],
    )

    logger.info(
        f"Splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def prepare_dataset(
    config: Optional[dict] = None, force_download: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end pipeline: download, clean, split, and save.

    Args:
        config: Project config dict. Loaded from default path if None.
        force_download: Re-download even if raw file exists.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if config is None:
        config = load_config()

    data_cfg = config["data"]
    project_cfg = config["project"]

    # Download
    csv_path = download_mtsamples(
        raw_dir=data_cfg["raw_dir"], force=force_download
    )

    # Load and clean
    raw_df = pd.read_csv(csv_path)
    clean_df = clean_mtsamples(
        raw_df,
        min_class_samples=data_cfg["min_class_samples"],
        min_doc_length=config["preprocessing"]["min_doc_length"],
    )

    # Split
    train_df, val_df, test_df = create_splits(
        clean_df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        seed=project_cfg["seed"],
    )

    # Save processed splits
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = processed_dir / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info(f"Saved {name} split to {out_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = prepare_dataset()
    print(f"\nSpecialties in training set:\n{train_df['medical_specialty'].value_counts()}")
