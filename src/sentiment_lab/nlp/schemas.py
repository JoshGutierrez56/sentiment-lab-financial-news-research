"""Strict structured output and classification-ledger schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SentimentLabel(StrEnum):
    bearish = "bearish"
    neutral = "neutral"
    bullish = "bullish"


class EventType(StrEnum):
    earnings_results = "earnings_results"
    guidance = "guidance"
    analyst_rating = "analyst_rating"
    product_launch = "product_launch"
    regulatory_action = "regulatory_action"
    litigation = "litigation"
    merger_acquisition = "merger_acquisition"
    management_change = "management_change"
    capital_allocation = "capital_allocation"
    financing = "financing"
    macro_exposure = "macro_exposure"
    operational_disruption = "operational_disruption"
    cybersecurity = "cybersecurity"
    fraud_accounting = "fraud_accounting"
    bankruptcy = "bankruptcy"
    dividend = "dividend"
    buyback = "buyback"
    restructuring = "restructuring"
    other = "other"


class ExpectedHorizon(StrEnum):
    intraday = "intraday"
    one_day = "1d"
    three_days = "3d"
    five_days = "5d"
    twenty_days = "20d"
    long_term = "long_term"


class ArticleAssessment(BaseModel):
    """The exact object requested from OpenAI Structured Outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    materiality: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    event_type: EventType
    expected_horizon: ExpectedHorizon
    tradable: bool
    abstain: bool
    concise_reasoning: str = Field(min_length=1, max_length=320)

    @field_validator("concise_reasoning")
    @classmethod
    def limit_reasoning_words(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized.split()) > 40:
            raise ValueError("concise_reasoning must contain no more than 40 words")
        return normalized

    @model_validator(mode="after")
    def validate_label_score(self) -> ArticleAssessment:
        if self.sentiment_label is SentimentLabel.bullish and self.sentiment_score <= 0:
            raise ValueError("bullish assessments require a positive sentiment_score")
        if self.sentiment_label is SentimentLabel.bearish and self.sentiment_score >= 0:
            raise ValueError("bearish assessments require a negative sentiment_score")
        if self.sentiment_label is SentimentLabel.neutral and abs(self.sentiment_score) > 0.25:
            raise ValueError("neutral assessments require |sentiment_score| <= 0.25")
        if self.tradable == self.abstain:
            raise ValueError("tradable and abstain must be logical opposites")
        return self


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_cached_tokens(self) -> ModelUsage:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")
        return self


class ClassificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_key: str
    input_hash: str
    output_hash: str
    article_id: str = Field(min_length=16)
    ticker: str = Field(min_length=1)
    event_timestamp: datetime
    requested_model: str
    model: str
    prompt_version: str
    schema_version: str
    stage: Literal["first_pass", "escalation"]
    escalation_reasons: list[str] = Field(default_factory=list)
    classified_at: datetime
    response_id: str | None = None
    batch_id: str | None = None
    batch_custom_id: str | None = None
    from_cache: bool = False
    usage: ModelUsage
    assessment: ArticleAssessment

    @field_validator("classified_at", "event_timestamp")
    @classmethod
    def normalize_datetimes(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class ClassificationLedgerEntry(BaseModel):
    """One per-article model/cache attempt with current-run cost separated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    article_id: str
    ticker: str
    event_timestamp: datetime
    cache_key: str
    input_hash: str
    requested_model: str
    response_model: str
    prompt_version: str
    schema_version: str
    stage: Literal["first_pass", "escalation"]
    outcome: Literal["api_success", "api_failure", "cache_hit"]
    failure_reason: str | None = None
    escalation_reasons: list[str] = Field(default_factory=list)
    response_id: str | None = None
    batch_id: str | None = None
    batch_custom_id: str | None = None
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    run_cost_usd: float = Field(ge=0.0)

    @field_validator("event_timestamp")
    @classmethod
    def normalize_event_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("ticker")
    @classmethod
    def normalize_ledger_ticker(cls, value: str) -> str:
        return value.strip().upper()
