"""Databricks job entry point for lightweight repository smoke tests."""

from __future__ import annotations

import argparse
import logging

from clinical_text_classifier.baseline import build_baseline_pipeline
from clinical_text_classifier.config import load_config
from clinical_text_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def run(config_path: str) -> None:
    """Run lightweight packaged-application checks inside Databricks."""
    config = load_config(config_path)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))

    required_sections = {"project", "data", "baseline", "paths", "runtime"}
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(f"Smoke test failed, missing config sections: {sorted(missing_sections)}")

    pipeline = build_baseline_pipeline(
        max_features=config["baseline"]["tfidf"]["max_features"],
        ngram_range=tuple(config["baseline"]["tfidf"]["ngram_range"]),
    )
    step_names = [name for name, _ in pipeline.steps]
    if step_names != ["tfidf", "classifier"]:
        raise ValueError(f"Smoke test failed, unexpected baseline steps: {step_names}")

    LOGGER.info("Smoke test passed. Packaged config and baseline pipeline are valid.")


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Run lightweight smoke tests.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
