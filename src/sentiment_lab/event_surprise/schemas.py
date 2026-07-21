"""Strict schema: facts are not surprise, and ambiguity is an abstention."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventType(StrEnum):
    earnings = "earnings"
    guidance = "guidance"
    analyst_revision = "analyst_revision"
    merger_acquisition = "merger_acquisition"
    regulatory_decision = "regulatory_decision"
    litigation_outcome = "litigation_outcome"
    product_approval_or_launch = "product_approval_or_launch"
    capital_allocation = "capital_allocation"
    dividend = "dividend"
    buyback = "buyback"
    financing = "financing"
    management_change = "management_change"
    restructuring = "restructuring"
    operational_disruption = "operational_disruption"
    cybersecurity = "cybersecurity"
    fraud_accounting = "fraud_accounting"
    other = "other"


class SurpriseDirection(StrEnum):
    positive = "positive"
    negative = "negative"
    none = "none"
    unclear = "unclear"


class EventSurpriseAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_company: str | None = Field(default=None, max_length=160)
    primary_ticker: str | None = Field(default=None, max_length=24)
    company_specificity: float = Field(ge=0, le=1)
    event_type: EventType
    actual_information: str | None = Field(default=None, max_length=500)
    expected_or_prior_information: str | None = Field(default=None, max_length=500)
    surprise_direction: SurpriseDirection
    surprise_magnitude: float = Field(ge=0, le=1)
    direction_score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    materiality: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    already_priced_in: float = Field(ge=0, le=1)
    expected_horizon: str
    abstain: bool
    abstain_reason: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def enforce_sparse_abstention(self) -> EventSurpriseAssessment:
        invalid = (
            self.event_type is EventType.other
            or not self.primary_company
            or not self.primary_ticker
            or self.company_specificity < 0.7
            or self.relevance < 0.7
            or self.materiality < 0.7
            or not self.actual_information
            or self.surprise_direction in {SurpriseDirection.none, SurpriseDirection.unclear}
        )
        if invalid and not self.abstain:
            raise ValueError(
                "generic, stale, non-company-specific, or non-surprise articles must abstain"
            )
        if self.abstain and not self.abstain_reason:
            raise ValueError("abstentions require an explicit reason")
        if not self.abstain and self.abstain_reason:
            raise ValueError("qualifying observations cannot carry an abstain reason")
        expected_sign = {SurpriseDirection.positive: 1, SurpriseDirection.negative: -1}
        if (
            self.surprise_direction in expected_sign
            and self.direction_score * expected_sign[self.surprise_direction] <= 0
        ):
            raise ValueError("direction_score must match surprise_direction")
        return self
