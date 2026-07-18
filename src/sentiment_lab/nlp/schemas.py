"""Strict structured output and classification-ledger schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

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

    article_id: str = Field(min_length=16)
    ticker: str = Field(min_length=1)
    event_timestamp: datetime
    sentiment_label: SentimentLabel
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    event_type: EventType
    expected_horizon: ExpectedHorizon
    concise_reasoning: str = Field(min_length=1, max_length=600)
    tradable: bool
    abstain_reason: str | None = Field(default=None, max_length=300)

    @field_validator("event_timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_label_score(self) -> ArticleAssessment:
        if self.sentiment_label is SentimentLabel.bullish and self.sentiment_score <= 0:
            raise ValueError("bullish assessments require a positive sentiment_score")
        if self.sentiment_label is SentimentLabel.bearish and self.sentiment_score >= 0:
            raise ValueError("bearish assessments require a negative sentiment_score")
        if self.sentiment_label is SentimentLabel.neutral and abs(self.sentiment_score) > 0.25:
            raise ValueError("neutral assessments require |sentiment_score| <= 0.25")
        if self.tradable and self.abstain_reason:
            raise ValueError("tradable assessments cannot include abstain_reason")
        if not self.tradable and not self.abstain_reason:
            raise ValueError("non-tradable assessments require abstain_reason")
        return self


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class ClassificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_key: str
    input_hash: str
    output_hash: str
    model: str
    prompt_version: str
    schema_version: str
    classified_at: datetime
    response_id: str | None = None
    from_cache: bool = False
    usage: ModelUsage
    assessment: ArticleAssessment

    @field_validator("classified_at")
    @classmethod
    def normalize_classified_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
