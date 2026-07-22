"""Strict point-in-time schemas for expectation-adjusted event research.

These contracts intentionally precede feature engineering or model selection.
They make it impossible to construct a valid observation when expectations or
controls became available at or after the event announcement.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentiment_lab.event_surprise.schemas import EventType

SHA256_PATTERN = r"^[0-9a-f]{64}$"
TICKER_PATTERN = r"^[A-Z][A-Z0-9.-]{0,14}$"
METRIC_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("point-in-time timestamps must include a timezone")
    return value.astimezone(UTC)


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectationSource(StrEnum):
    analyst_consensus = "analyst_consensus"
    company_guidance = "company_guidance"
    prior_company_disclosure = "prior_company_disclosure"
    market_implied = "market_implied"


class MetricUnit(StrEnum):
    currency = "currency"
    currency_per_share = "currency_per_share"
    percentage = "percentage"
    basis_points = "basis_points"
    count_value = "count"
    ratio = "ratio"


class StudySplit(StrEnum):
    development = "development"
    validation = "validation"
    prospective_holdout = "prospective_holdout"


class ReportedActual(StrictFrozenModel):
    event_id: str = Field(min_length=1, max_length=128)
    article_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(pattern=TICKER_PATTERN)
    event_type: EventType
    metric: str = Field(pattern=METRIC_PATTERN)
    fiscal_period_end: date
    value: float
    unit: MetricUnit
    announced_at: datetime
    available_at: datetime
    source: str = Field(min_length=1, max_length=80)
    source_document_id: str = Field(min_length=1, max_length=160)
    raw_content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("announced_at", "available_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def enforce_publication_order(self) -> Self:
        if self.available_at < self.announced_at:
            raise ValueError("actual cannot be available before it is announced")
        return self


class PointInTimeExpectation(StrictFrozenModel):
    expectation_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(pattern=TICKER_PATTERN)
    metric: str = Field(pattern=METRIC_PATTERN)
    fiscal_period_end: date
    value: float
    unit: MetricUnit
    snapshot_at: datetime
    available_at: datetime
    source: ExpectationSource
    source_revision: str = Field(min_length=1, max_length=128)
    dispersion: float | None = Field(default=None, gt=0)
    contributor_count: int | None = Field(default=None, ge=1)
    raw_content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("snapshot_at", "available_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def enforce_snapshot_order(self) -> Self:
        if self.available_at < self.snapshot_at:
            raise ValueError("expectation cannot be available before its source snapshot")
        return self


class PointInTimeControls(StrictFrozenModel):
    ticker: str = Field(pattern=TICKER_PATTERN)
    as_of_date: date
    available_at: datetime
    sector: str = Field(min_length=1, max_length=80)
    industry: str | None = Field(default=None, max_length=120)
    market_beta: float
    log_market_cap: float
    average_dollar_volume_20d: float = Field(gt=0)
    book_to_market: float | None = None
    momentum_12_1: float | None = None
    return_on_equity: float | None = None
    asset_growth: float | None = None
    idiosyncratic_volatility: float | None = Field(default=None, ge=0)
    source: str = Field(min_length=1, max_length=80)
    raw_content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("available_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class ExpectationAdjustedObservation(StrictFrozenModel):
    actual: ReportedActual
    expectation: PointInTimeExpectation
    controls: PointInTimeControls
    news_available_at: datetime
    news_text_sha256: str = Field(pattern=SHA256_PATTERN)
    research_split: StudySplit
    specification_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("news_available_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def enforce_point_in_time_join(self) -> Self:
        if len({self.actual.ticker, self.expectation.ticker, self.controls.ticker}) != 1:
            raise ValueError("actual, expectation, and controls must reference one ticker")
        if self.actual.metric != self.expectation.metric:
            raise ValueError("actual and expectation metric must match")
        if self.actual.fiscal_period_end != self.expectation.fiscal_period_end:
            raise ValueError("actual and expectation fiscal period must match")
        if self.actual.unit is not self.expectation.unit:
            raise ValueError("actual and expectation unit must match")
        if self.expectation.available_at >= self.actual.announced_at:
            raise ValueError("expectation must be available strictly before the announcement")
        if self.controls.available_at >= self.actual.announced_at:
            raise ValueError("controls must be available strictly before the announcement")
        return self

    @property
    def decision_at(self) -> datetime:
        """Earliest timestamp at which both the actual and news are observable."""

        return max(self.actual.available_at, self.news_available_at)

    @property
    def raw_surprise(self) -> float:
        return self.actual.value - self.expectation.value

    @property
    def dispersion_scaled_surprise(self) -> float | None:
        if self.expectation.dispersion is None:
            return None
        return self.raw_surprise / self.expectation.dispersion
