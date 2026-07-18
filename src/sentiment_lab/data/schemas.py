"""Validated provider and normalized data schemas."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class ProviderModel(BaseModel):
    """Provider models retain unknown fields in the immutable raw response."""

    model_config = ConfigDict(extra="allow")


class ProviderSentiment(ProviderModel):
    polarity: float | None = None
    neg: float | None = Field(default=None, ge=0.0, le=1.0)
    neu: float | None = Field(default=None, ge=0.0, le=1.0)
    pos: float | None = Field(default=None, ge=0.0, le=1.0)


class EODHDNewsItem(ProviderModel):
    date: datetime
    title: str = ""
    content: str = ""
    link: str = ""
    symbols: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    sentiment: ProviderSentiment | None = None

    @field_validator("date")
    @classmethod
    def normalize_date(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("title", "content", "link")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class NewsArticle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    article_id: str
    provider: str = "eodhd"
    provider_timestamp: datetime
    retrieved_at: datetime
    title: str
    content: str
    link: str
    symbols: list[str]
    tags: list[str]
    provider_sentiment_polarity: float | None = None
    raw_response_hash: str

    @field_validator("provider_timestamp", "retrieved_at")
    @classmethod
    def normalize_datetimes(cls, value: datetime) -> datetime:
        return _utc(value)

    @classmethod
    def from_provider(
        cls,
        item: EODHDNewsItem,
        *,
        retrieved_at: datetime,
        raw_response_hash: str,
    ) -> NewsArticle:
        identity = "\x1f".join(
            [
                item.date.isoformat(),
                item.title.casefold(),
                item.link.casefold(),
                item.content,
                ",".join(sorted(symbol.upper() for symbol in item.symbols)),
            ]
        )
        article_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cls(
            article_id=article_id,
            provider_timestamp=item.date,
            retrieved_at=retrieved_at,
            title=item.title,
            content=item.content,
            link=item.link,
            symbols=sorted({symbol.upper() for symbol in item.symbols}),
            tags=sorted(set(item.tags)),
            provider_sentiment_polarity=(
                item.sentiment.polarity if item.sentiment is not None else None
            ),
            raw_response_hash=raw_response_hash,
        )


class EODPrice(ProviderModel):
    date: date
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    adjusted_close: float = Field(gt=0.0)
    volume: int = Field(ge=0)

    @property
    def adjusted_open(self) -> float:
        """Back-adjust the raw open with the provider's close adjustment factor."""

        return self.open * (self.adjusted_close / self.close)


class RawResponseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    endpoint: str
    sanitized_params: dict[str, Any]
    request_key: str
    fetched_at: datetime
    status_code: int
    response_hash: str
    body_path: str

    @field_validator("fetched_at")
    @classmethod
    def normalize_fetched_at(cls, value: datetime) -> datetime:
        return _utc(value)
