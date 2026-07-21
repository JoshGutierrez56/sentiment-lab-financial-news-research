"""Strict schemas shared by hybrid sampling and local inference."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentiment_lab.nlp.schemas import ExpectedHorizon, SentimentLabel


class HybridEventType(StrEnum):
    earnings = "earnings"
    guidance = "guidance"
    analyst_action = "analyst_action"
    merger_acquisition = "merger_acquisition"
    product = "product"
    regulatory = "regulatory"
    litigation = "litigation"
    management = "management"
    capital_allocation = "capital_allocation"
    financing = "financing"
    restructuring = "restructuring"
    operations = "operations"
    cybersecurity = "cybersecurity"
    fraud_accounting = "fraud_accounting"
    dividend = "dividend"
    buyback = "buyback"
    macro_exposure = "macro_exposure"
    other = "other"


class LocalArticleAssessment(BaseModel):
    """Exact concise output requested from local models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    materiality: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    event_type: HybridEventType
    expected_horizon: ExpectedHorizon
    tradable: bool
    abstain: bool
    abstain_reason: str | None = Field(default=None, max_length=200)
    concise_reasoning: str = Field(min_length=1, max_length=240)

    @field_validator("concise_reasoning")
    @classmethod
    def limit_reasoning(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized.split()) > 25:
            raise ValueError("concise_reasoning must contain no more than 25 words")
        return normalized

    @field_validator("abstain_reason")
    @classmethod
    def normalize_abstain_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_consistency(self) -> LocalArticleAssessment:
        if self.sentiment_label is SentimentLabel.bullish and self.sentiment_score <= 0:
            raise ValueError("bullish assessments require a positive score")
        if self.sentiment_label is SentimentLabel.bearish and self.sentiment_score >= 0:
            raise ValueError("bearish assessments require a negative score")
        if self.sentiment_label is SentimentLabel.neutral and abs(self.sentiment_score) > 0.25:
            raise ValueError("neutral assessments require |score| <= 0.25")
        if self.tradable == self.abstain:
            raise ValueError("tradable and abstain must be logical opposites")
        if self.abstain and self.abstain_reason is None:
            raise ValueError("abstentions require abstain_reason")
        if not self.abstain and self.abstain_reason is not None:
            raise ValueError("tradable assessments cannot have abstain_reason")
        return self
