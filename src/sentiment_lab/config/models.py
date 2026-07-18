"""Strict configuration and runtime-secret models."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrictModel(BaseModel):
    """Base model for YAML configuration; typos fail immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EODHDConfig(StrictModel):
    base_url: str = "https://eodhd.com"
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=4, ge=1, le=10)
    backoff_base_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    backoff_max_seconds: float = Field(default=20.0, ge=0.0, le=300.0)
    jitter_seconds: float = Field(default=0.25, ge=0.0, le=10.0)
    news_page_size: int = Field(default=100, ge=1, le=1000)


class OpenAIConfig(StrictModel):
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    max_concurrency: int = Field(default=3, ge=1, le=32)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=700, ge=64, le=16_384)
    input_cost_per_million: float | None = Field(default=None, ge=0.0)
    output_cost_per_million: float | None = Field(default=None, ge=0.0)


class StorageConfig(StrictModel):
    data_root: Path = Path("./data")
    duckdb_path: Path = Path("./data/research.duckdb")


class AppConfig(StrictModel):
    eodhd: EODHDConfig = EODHDConfig()
    openai: OpenAIConfig = OpenAIConfig()
    storage: StorageConfig = StorageConfig()


class SentimentConfig(StrictModel):
    prompt_variant: Literal["directional_v1", "evidence_v2"] = "evidence_v2"
    schema_version: str = "article_assessment.v1"
    max_article_characters: int = Field(default=16_000, ge=500, le=100_000)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    ticker: str = Field(min_length=3)
    company_name: str = Field(min_length=1)
    news_start: date
    news_end: date
    max_articles: int = Field(default=12, ge=1, le=1000)
    news_candidate_pool: int = Field(default=500, ge=1, le=1000)
    max_articles_per_day: int = Field(default=2, ge=1, le=100)
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 5], min_length=1)
    execution_policy: Literal["conservative_next_day_open"] = "conservative_next_day_open"
    neutral_return_bps: float = Field(default=10.0, ge=0.0, le=1000.0)
    random_seed: int = 1729
    prompt_variant: Literal["directional_v1", "evidence_v2"] = "evidence_v2"

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("horizons must contain positive trading-day counts")
        if len(value) != len(set(value)):
            raise ValueError("horizons must be unique")
        return sorted(value)

    @model_validator(mode="after")
    def validate_dates(self) -> ExperimentConfig:
        if self.news_end < self.news_start:
            raise ValueError("news_end must be on or after news_start")
        if self.news_candidate_pool < self.max_articles:
            raise ValueError("news_candidate_pool must be at least max_articles")
        return self


class RuntimeSecrets(BaseSettings):
    """Runtime-only environment values; secrets are never serialized."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    eodhd_api_token: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    data_root: Path | None = None
    duckdb_path: Path | None = None
    log_level: str = "INFO"

    def require_eodhd_token(self) -> str:
        if self.eodhd_api_token is None or not self.eodhd_api_token.get_secret_value().strip():
            raise RuntimeError(
                "EODHD_API_TOKEN is required. Put it in the environment or an untracked .env file."
            )
        return self.eodhd_api_token.get_secret_value().strip()

    def require_openai(self) -> tuple[str, str]:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise RuntimeError(
                "OPENAI_API_KEY is required for real ChatGPT classification. "
                "Cached downloads and mocked tests do not require it."
            )
        model = (self.openai_model or "").strip()
        if not model:
            raise RuntimeError("OPENAI_MODEL must name a structured-output-capable model.")
        return self.openai_api_key.get_secret_value().strip(), model
