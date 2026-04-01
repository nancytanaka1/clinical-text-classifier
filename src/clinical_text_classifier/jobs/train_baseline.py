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


def run(config_path: str) -> None:
    """Train the baseline model from prepared splits."""
    config = load_config(config_path)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))

    processed_dir = Path(config["data"]["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    artifacts = train_baseline_model(
        train_df=train_df,
        val_df=val_df,
        model_dir=config["paths"]["models_dir"],
        max_features=config["baseline"]["tfidf"]["max_features"],
        ngram_range=tuple(config["baseline"]["tfidf"]["ngram_range"]),
    )

    LOGGER.info("Baseline model saved to %s", artifacts.model_path)
    LOGGER.info("Validation macro F1: %.4f", artifacts.macro_f1)
    LOGGER.info("Classification report:\n%s", artifacts.report)


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Train the baseline clinical text model.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
