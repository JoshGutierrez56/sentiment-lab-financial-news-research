"""Shared deterministic fixtures for the milestone tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sentiment_lab.data.schemas import EODPrice, NewsArticle
from sentiment_lab.nlp.cache import assessment_hash
from sentiment_lab.nlp.openai_client import ModelCall
from sentiment_lab.nlp.schemas import (
    ArticleAssessment,
    ClassificationRecord,
    EventType,
    ExpectedHorizon,
    ModelUsage,
    SentimentLabel,
)


@pytest.fixture
def article() -> NewsArticle:
    return make_article()


def make_article(
    *,
    article_id: str = "a" * 64,
    timestamp: datetime = datetime(2026, 5, 1, 21, 0, tzinfo=UTC),
    title: str = "Apple raises guidance",
    content: str = "Apple raised full-year revenue guidance after strong demand.",
) -> NewsArticle:
    return NewsArticle(
        article_id=article_id,
        provider_timestamp=timestamp,
        retrieved_at=datetime(2026, 5, 2, tzinfo=UTC),
        title=title,
        content=content,
        link=f"https://example.test/{article_id[:8]}",
        symbols=["AAPL.US"],
        tags=["earnings"],
        provider_sentiment_polarity=0.4,
        raw_response_hash="b" * 64,
    )


def make_assessment(
    article: NewsArticle,
    *,
    label: SentimentLabel = SentimentLabel.bullish,
    score: float = 0.8,
    confidence: float = 0.9,
    ticker: str = "AAPL.US",
) -> ArticleAssessment:
    return ArticleAssessment(
        sentiment_label=label,
        sentiment_score=score,
        confidence=confidence,
        relevance=0.95,
        materiality=0.60,
        novelty=0.80,
        event_type=EventType.guidance,
        expected_horizon=ExpectedHorizon.three_days,
        concise_reasoning="Raised guidance is incrementally positive for expected cash flows.",
        tradable=True,
        abstain=False,
    )


def make_call(article: NewsArticle, **kwargs: object) -> ModelCall:
    return ModelCall(
        assessment=make_assessment(article, **kwargs),
        usage=ModelUsage(input_tokens=100, output_tokens=30, estimated_cost_usd=0.001),
        response_id="resp_test",
        response_model="test-model",
        batch_id="batch_test",
        batch_custom_id=f"test-{article.article_id[:8]}",
    )


def make_record(article: NewsArticle, **kwargs: object) -> ClassificationRecord:
    assessment = make_assessment(article, **kwargs)
    return ClassificationRecord(
        cache_key="c" * 64,
        input_hash="d" * 64,
        output_hash=assessment_hash(assessment),
        article_id=article.article_id,
        ticker="AAPL.US",
        event_timestamp=article.provider_timestamp,
        requested_model="test-model",
        model="test-model",
        prompt_version="evidence_v2.1.0-cost",
        schema_version="article_assessment.v2",
        stage="first_pass",
        classified_at=datetime(2026, 5, 2, tzinfo=UTC),
        response_id="resp_test",
        batch_id="batch_test",
        batch_custom_id=f"test-{article.article_id[:8]}",
        usage=ModelUsage(input_tokens=100, output_tokens=30, estimated_cost_usd=0.001),
        assessment=assessment,
    )


def make_price(
    day: date, *, open_: float, close: float, adjusted_close: float | None = None
) -> EODPrice:
    return EODPrice(
        date=day,
        open=open_,
        high=max(open_, close) + 1,
        low=min(open_, close) - 1,
        close=close,
        adjusted_close=close if adjusted_close is None else adjusted_close,
        volume=1_000_000,
    )
