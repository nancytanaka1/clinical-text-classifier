"""Databricks job entry point for lightweight repository smoke tests."""

from __future__ import annotations

import logging
from pathlib import Path

from clinical_text_classifier.config import load_config
from clinical_text_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run basic repository checks inside Databricks."""
    config = load_config("configs/config.yaml")
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))

    required_paths = [
        Path("configs/config.yaml"),
        Path("databricks.yml"),
        Path("src/clinical_text_classifier/jobs/prepare_data.py"),
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Smoke test failed, missing required paths: {missing}")

    LOGGER.info("Smoke test passed. Required project files are present.")
