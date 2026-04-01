"""Databricks job entry point for dataset preparation."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from clinical_text_classifier.config import load_config
from clinical_text_classifier.data import (
    clean_dataset,
    create_splits,
    ensure_dataset,
    save_splits,
)
from clinical_text_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def run(config_path: str) -> None:
    """Run the data preparation workflow."""
    config = load_config(config_path)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))

    data_cfg = config["data"]
    preprocessing_cfg = config["preprocessing"]
    project_cfg = config["project"]

    csv_path = ensure_dataset(data_cfg["raw_dir"])
    raw_df = pd.read_csv(csv_path)
    clean_df = clean_dataset(
        raw_df,
        min_class_samples=data_cfg["min_class_samples"],
        min_doc_length=preprocessing_cfg["min_doc_length"],
    )
    train_df, val_df, test_df = create_splits(
        clean_df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        seed=project_cfg["seed"],
    )
    outputs = save_splits(data_cfg["processed_dir"], train_df, val_df, test_df)

    LOGGER.info(
        "Prepared dataset with %s rows across %s classes",
        len(clean_df),
        clean_df["medical_specialty"].nunique(),
    )
    for split_name, path in outputs.items():
        LOGGER.info("Saved %s split to %s", split_name, path)


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Prepare clinical text datasets.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
