"""YAML configuration loading with environment path overrides."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from sentiment_lab.config.models import (
    AppConfig,
    ExperimentConfig,
    RuntimeSecrets,
    SentimentConfig,
    ValidationExperimentConfig,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: str | Path, model: type[ModelT]) -> ModelT:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return model.model_validate(raw)


def load_app_config(
    path: str | Path = "config/settings.yaml",
    *,
    secrets: RuntimeSecrets | None = None,
) -> AppConfig:
    config = load_yaml(path, AppConfig)
    runtime = secrets or RuntimeSecrets()
    if runtime.data_root is None and runtime.duckdb_path is None:
        return config
    storage = config.storage.model_copy(
        update={
            "data_root": runtime.data_root or config.storage.data_root,
            "duckdb_path": runtime.duckdb_path or config.storage.duckdb_path,
        }
    )
    return config.model_copy(update={"storage": storage})


def load_sentiment_config(path: str | Path = "config/sentiment.yaml") -> SentimentConfig:
    return load_yaml(path, SentimentConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_yaml(path, ExperimentConfig)


def load_validation_experiment_config(path: str | Path) -> ValidationExperimentConfig:
    return load_yaml(path, ValidationExperimentConfig)
