from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.config.models import ValidationUniverseMember
from sentiment_lab.data.schemas import EODPrice, NewsArticle
from sentiment_lab.hybrid import sample
from sentiment_lab.hybrid.sample import HybridSampleConfig, sync_hybrid_sample
from sentiment_lab.hybrid.sampling import ClusteredStory


class _SyntheticClient:
    def __init__(self, members: list[ValidationUniverseMember]) -> None:
        self.members = {item.ticker: item for item in members}
        self.prices = [
            EODPrice(
                date=date(2021, 12, 20) + timedelta(days=index),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                adjusted_close=100.0 + index / 1000,
                volume=1_000_000,
            )
            for index in range(1700)
        ]

    def fetch_eod_prices(
        self,
        _ticker: str,
        _start: date,
        _end: date,
        *,
        refresh: bool = False,
    ) -> list[EODPrice]:
        assert not refresh
        return self.prices

    def fetch_news(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        max_articles: int,
        refresh: bool = False,
    ) -> list[NewsArticle]:
        assert not refresh
        member = self.members[ticker]
        output: list[NewsArticle] = []
        for index in range(max_articles):
            timestamp = datetime.combine(
                min(start + timedelta(days=index), end), datetime.min.time(), tzinfo=UTC
            )
            identity = f"{ticker}:{start}:{index}"
            article_id = hashlib.sha256(identity.encode()).hexdigest()
            title = f"{member.company_name} reports quarterly earnings {start} item {index}"
            body = (
                f"{member.company_name} reported company-specific quarterly earnings, "
                f"revenue and operating margin for unique business unit {identity}. "
            ) * 8
            output.append(
                NewsArticle(
                    article_id=article_id,
                    provider_timestamp=timestamp,
                    retrieved_at=timestamp,
                    title=title,
                    content=body,
                    link=f"https://example.test/{article_id}",
                    symbols=[ticker],
                    tags=[],
                    raw_response_hash="a" * 64,
                )
            )
        return output


def test_sync_freezes_exact_5000_without_return_based_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sectors = [f"Sector {index}" for index in range(11)]
    members = [
        ValidationUniverseMember(
            ticker=f"T{index:03d}.US",
            company_name=f"Company {index}",
            sector=sectors[index % len(sectors)],
            aliases=[f"Company {index}"],
        )
        for index in range(100)
    ]
    config = HybridSampleConfig(
        name="synthetic_5000",
        news_start=date(2022, 1, 1),
        news_end=date(2026, 3, 31),
        universe=members,
        articles_per_company=50,
        articles_per_company_year=10,
        candidates_per_quarter=10,
        max_articles=5000,
        earnings_guidance_per_company_year=2,
        minimum_earnings_guidance=500,
    )

    def no_duplicate_clusters(articles: Any) -> list[ClusteredStory]:
        return [
            ClusteredStory(article.article_id, article.article_id[:20], True)
            for article in articles
        ]

    monkeypatch.setattr(sample, "cluster_syndicated_stories", no_duplicate_clusters)
    frozen = sync_hybrid_sample(
        config,
        _SyntheticClient(members),
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
    )
    articles = pl.read_parquet(frozen.articles_path)
    assert frozen.article_count == 5000
    assert articles["ticker"].n_unique() == 100
    assert articles["story_cluster_id"].n_unique() == 5000
    assert articles["future_return_63d"].null_count() == 0
    assert set(articles["pre_inference_primary_event_type"]) == {"earnings"}
    assert frozen.manifest_path.is_file()


def test_hybrid_config_rejects_wrong_horizons() -> None:
    try:
        HybridSampleConfig(
            name="bad",
            news_start=date(2022, 1, 1),
            news_end=date(2022, 12, 31),
            universe=[
                ValidationUniverseMember(
                    ticker=f"T{index:03d}.US",
                    company_name=f"Company {index}",
                    sector=f"Sector {index % 11}",
                )
                for index in range(100)
            ],
            articles_per_company=50,
            articles_per_company_year=50,
            max_articles=5000,
            horizons=[1, 5],
        )
    except ValueError as exc:
        assert "exactly 1, 3, 5, 10, 21, 63" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid horizons were accepted")


def test_hybrid_config_allows_return_blind_backfill_for_sparse_company_years() -> None:
    members = [
        ValidationUniverseMember(
            ticker=f"T{index:03d}.US",
            company_name=f"Company {index}",
            sector=f"Sector {index % 11}",
        )
        for index in range(100)
    ]
    config = HybridSampleConfig(
        name="sparse_years",
        news_start=date(2022, 1, 1),
        news_end=date(2026, 3, 31),
        universe=members,
        articles_per_company=50,
        articles_per_company_year=10,
        minimum_articles_per_company_year=0,
        minimum_years_per_company=3,
        max_articles=5000,
    )
    assert config.minimum_articles_per_company_year == 0
    assert config.minimum_years_per_company == 3
