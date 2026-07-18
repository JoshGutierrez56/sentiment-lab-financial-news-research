"""Strict configuration and secret-boundary tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentiment_lab.config.loader import load_app_config, load_experiment_config, load_yaml
from sentiment_lab.config.models import (
    ExperimentConfig,
    OpenAIConfig,
    RuntimeSecrets,
    SentimentConfig,
)


def test_unknown_yaml_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """name: test_run
ticker: aapl.us
company_name: Apple Inc.
news_start: 2026-01-01
news_end: 2026-01-31
unexpected_knob: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unexpected_knob"):
        load_experiment_config(path)


def test_app_paths_can_be_overridden_only_at_runtime(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("eodhd: {}\nopenai: {}\nstorage: {}\n", encoding="utf-8")
    secrets = RuntimeSecrets(
        data_root=tmp_path / "runtime-data",
        duckdb_path=tmp_path / "runtime.duckdb",
        _env_file=None,
    )
    config = load_app_config(path, secrets=secrets)
    assert config.storage.data_root == tmp_path / "runtime-data"
    assert config.storage.duckdb_path == tmp_path / "runtime.duckdb"


def test_experiment_normalizes_and_validates() -> None:
    config = ExperimentConfig(
        name="core_test",
        ticker=" aapl.us ",
        company_name="Apple Inc.",
        news_start=date(2026, 1, 1),
        news_end=date(2026, 1, 31),
        horizons=[5, 1, 3],
    )
    assert config.ticker == "AAPL.US"
    assert config.horizons == [1, 3, 5]
    with pytest.raises(ValidationError, match="unique"):
        config.model_copy(update={"horizons": [1, 1]}).__class__.model_validate(
            {**config.model_dump(), "horizons": [1, 1]}
        )
    with pytest.raises(ValidationError, match="news_end"):
        ExperimentConfig(
            name="bad_dates",
            ticker="AAPL.US",
            company_name="Apple Inc.",
            news_start=date(2026, 2, 1),
            news_end=date(2026, 1, 1),
        )


def test_missing_runtime_credentials_fail_with_actionable_messages() -> None:
    runtime = RuntimeSecrets(
        eodhd_api_token=None,
        openai_api_key=None,
        _env_file=None,
    )
    with pytest.raises(RuntimeError, match="EODHD_API_TOKEN"):
        runtime.require_eodhd_token()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        runtime.require_openai_key()


def test_cost_control_defaults_and_pro_models_are_guarded() -> None:
    config = OpenAIConfig()
    assert config.first_pass_model == "gpt-5.4-mini"
    assert config.escalation_model == "gpt-5.4"
    assert config.first_pass_max_output_tokens == 256
    assert config.escalation_max_output_tokens == 256
    assert config.spending_limits_usd.smoke == 1.0
    assert config.spending_limits_usd.first_research_sample == 5.0
    assert config.spending_limits_usd.expanded_validation == 20.0
    with pytest.raises(ValidationError, match="Pro models are prohibited"):
        OpenAIConfig(escalation_model="gpt-5.4-pro")


def test_ambiguity_escalation_thresholds_are_explicit_and_validated() -> None:
    config = SentimentConfig()
    assert config.escalation_ambiguity_relevance_threshold == 0.50
    assert config.escalation_ambiguity_materiality_threshold == 0.25
    with pytest.raises(ValidationError, match="greater than 0"):
        SentimentConfig(escalation_ambiguity_materiality_threshold=0.0)


def test_load_yaml_rejects_missing_and_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml", ExperimentConfig)
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml(path, ExperimentConfig)
