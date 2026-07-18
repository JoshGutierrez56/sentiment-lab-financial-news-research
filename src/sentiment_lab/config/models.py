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


class BatchModelPricing(StrictModel):
    """Published Batch API prices in USD per one million tokens."""

    input_per_million: float = Field(ge=0.0)
    cached_input_per_million: float = Field(ge=0.0)
    output_per_million: float = Field(ge=0.0)


class SpendingLimits(StrictModel):
    smoke: float = Field(default=1.0, gt=0.0)
    first_research_sample: float = Field(default=5.0, gt=0.0)
    expanded_validation: float = Field(default=20.0, gt=0.0)


class OpenAIConfig(StrictModel):
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    first_pass_model: str = "gpt-5.4-mini"
    escalation_model: str = "gpt-5.4"
    first_pass_max_output_tokens: int = Field(default=256, ge=128, le=1024)
    escalation_max_output_tokens: int = Field(default=256, ge=128, le=1024)
    reasoning_effort: Literal["none"] = "none"
    batch_poll_interval_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    batch_wait_timeout_seconds: float = Field(default=3600.0, gt=0.0, le=86_400.0)
    input_token_estimate_overhead: int = Field(default=512, ge=128, le=4096)
    prompt_cache_key_prefix: str = Field(
        default="sentiment-lab-article-v2", min_length=1, max_length=64
    )
    regional_processing_multiplier: float = Field(default=1.0, ge=1.0, le=1.25)
    pricing_source_url: str = "https://developers.openai.com/api/docs/pricing"
    pricing_as_of: date = date(2026, 7, 18)
    batch_pricing: dict[str, BatchModelPricing] = {
        "gpt-5.4-mini": BatchModelPricing(
            input_per_million=0.375,
            cached_input_per_million=0.0375,
            output_per_million=2.25,
        ),
        "gpt-5.4": BatchModelPricing(
            input_per_million=1.25,
            cached_input_per_million=0.13,
            output_per_million=7.50,
        ),
    }
    spending_limits_usd: SpendingLimits = SpendingLimits()

    @model_validator(mode="after")
    def validate_models_and_pricing(self) -> OpenAIConfig:
        for model in (self.first_pass_model, self.escalation_model):
            normalized = model.strip().lower()
            if not normalized:
                raise ValueError("OpenAI model names must not be blank")
            if "pro" in normalized.split("-"):
                raise ValueError("Pro models are prohibited for this cost-controlled milestone")
            if model not in self.batch_pricing:
                raise ValueError(f"Missing Batch API pricing for configured model: {model}")
        if self.first_pass_model == self.escalation_model:
            raise ValueError("first-pass and escalation models must differ")
        return self


class StorageConfig(StrictModel):
    data_root: Path = Path("./data")
    duckdb_path: Path = Path("./data/research.duckdb")


class AppConfig(StrictModel):
    eodhd: EODHDConfig = EODHDConfig()
    openai: OpenAIConfig = OpenAIConfig()
    storage: StorageConfig = StorageConfig()


class SentimentConfig(StrictModel):
    prompt_variant: Literal["directional_v1", "evidence_v2"] = "evidence_v2"
    schema_version: str = "article_assessment.v2"
    max_article_characters: int = Field(default=16_000, ge=500, le=100_000)
    minimum_article_characters: int = Field(default=200, ge=50, le=10_000)
    classify_headline_only: bool = False
    ticker_mapping_confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    market_summary_symbol_threshold: int = Field(default=5, ge=2, le=100)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    escalation_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    escalation_materiality_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


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
    spending_limit_tier: Literal["smoke", "first_research_sample", "expanded_validation"] = "smoke"

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
    data_root: Path | None = None
    duckdb_path: Path | None = None
    log_level: str = "INFO"

    def require_eodhd_token(self) -> str:
        if self.eodhd_api_token is None or not self.eodhd_api_token.get_secret_value().strip():
            raise RuntimeError(
                "EODHD_API_TOKEN is required. Put it in the environment or an untracked .env file."
            )
        return self.eodhd_api_token.get_secret_value().strip()

    def require_openai_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise RuntimeError(
                "OPENAI_API_KEY is required for real ChatGPT classification. "
                "Cached downloads and mocked tests do not require it."
            )
        return self.openai_api_key.get_secret_value().strip()
