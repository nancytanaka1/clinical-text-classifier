"""Databricks job entry point for baseline model training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from clinical_text_classifier.baseline import train_baseline_model
from clinical_text_classifier.config import load_config
from clinical_text_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def run(config_path: str, train_ready_path_override: str | None = None) -> None:
    """Train the baseline model from the train-ready dataset."""
    config = load_config(config_path)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))

    train_ready_path = Path(train_ready_path_override or config["data"]["train_ready_file"])
    train_ready_df = pd.read_parquet(train_ready_path)

    required_columns = {"text", "label", "split"}
    missing_columns = required_columns - set(train_ready_df.columns)
    if missing_columns:
        raise ValueError(
            f"Train-ready dataset missing required columns: {sorted(missing_columns)}"
        )

    train_df = train_ready_df[train_ready_df["split"] == "train"].reset_index(drop=True)
    val_df = train_ready_df[train_ready_df["split"] == "val"].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Train-ready dataset must contain non-empty train and val splits.")

    artifacts = train_baseline_model(
        train_df=train_df,
        val_df=val_df,
        model_dir=config["paths"]["models_dir"],
        metrics_dir=config["paths"]["metrics_dir"],
        max_features=config["baseline"]["tfidf"]["max_features"],
        ngram_range=tuple(config["baseline"]["tfidf"]["ngram_range"]),
    )

    LOGGER.info("Baseline model saved to %s", artifacts.model_path)
    LOGGER.info("Baseline metrics saved to %s", artifacts.metrics_path)
    LOGGER.info("Validation macro F1: %.4f", artifacts.macro_f1)
    LOGGER.info("Classification report:\n%s", artifacts.report)


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Train the baseline clinical text model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-ready-path", default=None)
    args = parser.parse_args()
    run(args.config, train_ready_path_override=args.train_ready_path)


if __name__ == "__main__":
    main()
