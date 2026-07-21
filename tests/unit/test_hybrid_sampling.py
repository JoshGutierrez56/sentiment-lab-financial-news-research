from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentiment_lab.config.models import ValidationUniverseMember
from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.hybrid.sampling import (
    candidate_event_types,
    cluster_syndicated_stories,
    company_relevance_score,
)
from sentiment_lab.hybrid.schemas import HybridEventType


def _article(title: str, content: str, *, article_id: str = "a" * 64) -> NewsArticle:
    return NewsArticle(
        article_id=article_id,
        provider_timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        retrieved_at=datetime(2025, 1, 2, tzinfo=UTC),
        title=title,
        content=content,
        link="https://example.test/article",
        symbols=["ACME.US"],
        tags=[],
        raw_response_hash="b" * 64,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ACME reports quarterly earnings and EPS beat", HybridEventType.earnings),
        ("ACME raises full-year guidance", HybridEventType.guidance),
        ("Broker upgrades ACME and lifts price target", HybridEventType.analyst_action),
        ("ACME agrees to acquisition of TargetCo", HybridEventType.merger_acquisition),
        ("Court rules in ACME patent lawsuit", HybridEventType.litigation),
        ("FTC opens antitrust probe into ACME", HybridEventType.regulatory),
        ("ACME raises its quarterly dividend", HybridEventType.dividend),
        ("ACME authorizes $2 billion share buyback", HybridEventType.buyback),
        ("Strike forces ACME production shutdown", HybridEventType.operations),
    ],
)
def test_representative_events_do_not_default_to_other(
    text: str, expected: HybridEventType
) -> None:
    assert expected in candidate_event_types(text, text)


def test_generic_story_stays_other() -> None:
    assert candidate_event_types("ACME in the news", "Investors discussed ACME today.") == (
        HybridEventType.other,
    )


def test_company_relevance_requires_primary_company_not_metadata_only() -> None:
    member = ValidationUniverseMember(
        ticker="ACME.US", company_name="Acme Corporation", sector="Industrials", aliases=["Acme"]
    )
    direct = _article(
        "Acme raises guidance after strong quarterly results",
        "Acme reported company-specific revenue and operating margin results. " * 12,
    )
    incidental = _article(
        "Five stocks to watch in a volatile market",
        "A broad market summary mentioned many companies before one incidental Acme reference. "
        * 12,
    ).model_copy(
        update={"symbols": ["ACME.US", "ONE.US", "TWO.US", "THREE.US", "FOUR.US", "FIVE.US"]}
    )
    assert company_relevance_score(direct, member).eligible
    result = company_relevance_score(incidental, member)
    assert not result.eligible
    assert "target_not_in_headline" in result.exclusion_reasons


def test_near_duplicate_cluster_keeps_one_primary() -> None:
    body = "Acme reported revenue growth and raised guidance for the year. " * 20
    first = _article("Acme raises guidance", body, article_id="1" * 64)
    second = _article(
        "Acme raises guidance today", body.replace("year", "full year"), article_id="2" * 64
    )
    clustered = cluster_syndicated_stories([second, first], max_hamming_distance=12)
    assert sum(item.primary for item in clustered) == 1
    assert len({item.cluster_id for item in clustered}) == 1
