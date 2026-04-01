"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from clinical_text_classifier.config import load_config


def test_load_config_reads_project_name() -> None:
    config = load_config(Path("configs/config.yaml"))
    assert config["project"]["name"] == "clinical-text-classifier"
