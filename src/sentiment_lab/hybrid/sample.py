"""Freeze the return-blind, company-specific 5,000-article hybrid sample."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentiment_lab.config.models import ValidationUniverseMember
from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.schemas import EODPrice, NewsArticle
from sentiment_lab.data.storage import ArtifactStore
from sentiment_lab.hybrid.sampling import (
    PreInferenceScore,
    cluster_syndicated_stories,
    company_relevance_score,
)
from sentiment_lab.hybrid.schemas import HybridEventType
from sentiment_lab.nlp.cache import article_content_hash, story_body_hash

log = logging.getLogger(__name__)
NEW_YORK = ZoneInfo("America/New_York")


class HybridSampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    news_start: date
    news_end: date
    universe: list[ValidationUniverseMember] = Field(min_length=100)
    articles_per_company: int = Field(default=40, ge=1, le=100)
    articles_per_company_year: int = Field(default=8, ge=1, le=25)
    candidates_per_quarter: int = Field(default=60, ge=10, le=250)
    max_articles: int = Field(default=5000, ge=1, le=5000)
    minimum_relevance_score: float = Field(default=0.55, ge=0.0, le=1.0)
    minimum_text_characters: int = Field(default=400, ge=200, le=5000)
    maximum_symbols: int = Field(default=5, ge=1, le=25)
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 21, 63])
    earnings_guidance_per_company_year: int = Field(default=2, ge=0, le=8)
    minimum_earnings_guidance: int = Field(default=500, ge=0)
    maximum_other_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    minimum_sectors: int = Field(default=11, ge=1)
    maximum_ticker_fraction: float = Field(default=0.02, gt=0.0, le=1.0)
    random_seed: int = 20260718
    frozen_sample_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: list[int]) -> list[int]:
        if sorted(value) != [1, 3, 5, 10, 21, 63]:
            raise ValueError("Hybrid sample horizons must be exactly 1, 3, 5, 10, 21, 63")
        return value

    @model_validator(mode="after")
    def validate_design(self) -> HybridSampleConfig:
        if self.news_end < self.news_start:
            raise ValueError("news_end must not precede news_start")
        years = self.news_end.year - self.news_start.year + 1
        if self.articles_per_company != self.articles_per_company_year * years:
            raise ValueError("Per-company total must equal per-year total times year count")
        if len(self.universe) * self.articles_per_company != self.max_articles:
            raise ValueError("Universe size times per-company total must equal max_articles")
        if len({item.ticker for item in self.universe}) != len(self.universe):
            raise ValueError("Hybrid universe tickers must be unique")
        if len({item.sector for item in self.universe}) < self.minimum_sectors:
            raise ValueError("Hybrid universe has too few sectors")
        if self.articles_per_company / self.max_articles > self.maximum_ticker_fraction:
            raise ValueError("Per-company sample exceeds maximum ticker fraction")
        return self


class HybridDataClient(Protocol):
    def fetch_news(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        max_articles: int,
        refresh: bool = False,
    ) -> list[NewsArticle]: ...

    def fetch_eod_prices(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        refresh: bool = False,
    ) -> list[EODPrice]: ...


@dataclass(frozen=True)
class ScoredCandidate:
    article: NewsArticle
    member: ValidationUniverseMember
    relevance: PreInferenceScore
    cluster_id: str = ""
    cluster_primary: bool = True

    @property
    def primary_event_type(self) -> HybridEventType:
        return self.relevance.event_type_candidates[0]


@dataclass(frozen=True)
class FrozenHybridSample:
    sample_hash: str
    article_count: int
    articles_path: Path
    candidates_path: Path
    prices_path: Path
    manifest_path: Path


def _quarter_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = date(start.year, ((start.month - 1) // 3) * 3 + 1, 1)
    while cursor <= end:
        next_month = cursor.month + 3
        next_year = cursor.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        next_quarter = date(next_year, next_month, 1)
        window_start = max(start, cursor)
        window_end = min(end, next_quarter - timedelta(days=1))
        windows.append((window_start, window_end))
        cursor = next_quarter
    return windows


def _complete_returns(
    article: NewsArticle, prices: list[EODPrice], horizons: list[int]
) -> bool:
    publication_day = article.provider_timestamp.astimezone(NEW_YORK).date()
    entry_index = next(
        (index for index, item in enumerate(prices) if item.date > publication_day), None
    )
    return entry_index is not None and entry_index + max(horizons) - 1 < len(prices)


def _candidate_sort_key(candidate: ScoredCandidate, seed: int) -> tuple[float, str, str]:
    tie = hashlib.sha256(f"{seed}:{candidate.article.article_id}".encode()).hexdigest()
    return (-candidate.relevance.score, tie, candidate.article.article_id)


def _select_year(
    candidates: list[ScoredCandidate],
    *,
    count: int,
    earnings_guidance_target: int,
    seed: int,
) -> list[ScoredCandidate]:
    available = sorted(candidates, key=lambda item: _candidate_sort_key(item, seed))
    selected: list[ScoredCandidate] = []
    selected_ids: set[str] = set()
    event_counts: Counter[HybridEventType] = Counter()
    quarter_counts: Counter[int] = Counter()
    earnings_guidance = [
        item
        for item in available
        if item.primary_event_type in {HybridEventType.earnings, HybridEventType.guidance}
    ]
    for item in earnings_guidance[:earnings_guidance_target]:
        selected.append(item)
        selected_ids.add(item.article.article_id)
        event_counts[item.primary_event_type] += 1
        quarter_counts[(item.article.provider_timestamp.month - 1) // 3] += 1
    while len(selected) < count:
        remaining = [item for item in available if item.article.article_id not in selected_ids]
        if not remaining:
            break
        ranked = sorted(
            remaining,
            key=lambda item: (
                item.primary_event_type is HybridEventType.other,
                event_counts[item.primary_event_type],
                quarter_counts[(item.article.provider_timestamp.month - 1) // 3],
                *_candidate_sort_key(item, seed),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        selected_ids.add(chosen.article.article_id)
        event_counts[chosen.primary_event_type] += 1
        quarter_counts[(chosen.article.provider_timestamp.month - 1) // 3] += 1
    return sorted(selected, key=lambda item: (item.article.provider_timestamp, item.article.article_id))


def _aligned_rows(
    selected: list[ScoredCandidate],
    prices_by_ticker: dict[str, list[EODPrice]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        prices = prices_by_ticker[item.member.ticker]
        local_publication = item.article.provider_timestamp.astimezone(NEW_YORK)
        entry_index = next(
            (
                index
                for index, price in enumerate(prices)
                if price.date > local_publication.date()
            ),
            None,
        )
        if entry_index is None:
            raise RuntimeError(f"No conservative entry for {item.article.article_id}")
        entry = prices[entry_index]
        entry_timestamp = datetime.combine(
            entry.date, time(9, 30), tzinfo=NEW_YORK
        ).astimezone(UTC)
        row: dict[str, Any] = {
            **item.article.model_dump(mode="python"),
            "ticker": item.member.ticker,
            "company_name": item.member.company_name,
            "sector": item.member.sector,
            "article_content_hash": article_content_hash(
                item.article.title, item.article.content
            ),
            "story_body_hash": story_body_hash(item.article.content),
            "story_cluster_id": item.cluster_id,
            "pre_inference_relevance_score": item.relevance.score,
            "pre_inference_score_components_json": stable_json(item.relevance.components),
            "pre_inference_event_candidates": [
                value.value for value in item.relevance.event_type_candidates
            ],
            "pre_inference_primary_event_type": item.primary_event_type.value,
            "entry_date": entry.date,
            "entry_timestamp_utc": entry_timestamp,
            "entry_adjusted_open": entry.adjusted_open,
        }
        for horizon in horizons:
            exit_price = prices[entry_index + horizon - 1]
            row[f"exit_date_{horizon}d"] = exit_price.date
            row[f"future_return_{horizon}d"] = (
                exit_price.adjusted_close / entry.adjusted_open - 1.0
            )
        rows.append(row)
    return rows


def sync_hybrid_sample(
    config: HybridSampleConfig,
    client: HybridDataClient,
    *,
    data_root: Path,
    duckdb_path: Path,
    refresh: bool = False,
) -> FrozenHybridSample:
    """Download candidates, freeze exactly 5,000, and align returns after selection."""

    windows = _quarter_windows(config.news_start, config.news_end)
    scored: list[ScoredCandidate] = []
    candidate_rows: list[dict[str, Any]] = []
    prices_by_ticker: dict[str, list[EODPrice]] = {}
    price_end = min(
        date.today(), config.news_end + timedelta(days=max(config.horizons) * 3 + 14)
    )
    totals: Counter[str] = Counter()
    seen_article_targets: set[tuple[str, str]] = set()

    for member_index, member in enumerate(config.universe, start=1):
        prices = sorted(
            client.fetch_eod_prices(
                member.ticker,
                config.news_start - timedelta(days=7),
                price_end,
                refresh=refresh,
            ),
            key=lambda item: item.date,
        )
        if not prices:
            raise RuntimeError(f"No prices for {member.ticker}")
        prices_by_ticker[member.ticker] = prices
        requested: list[NewsArticle] = []
        for window_start, window_end in windows:
            requested.extend(
                client.fetch_news(
                    member.ticker,
                    window_start,
                    window_end,
                    max_articles=config.candidates_per_quarter,
                    refresh=refresh,
                )
            )
        unique = {article.article_id: article for article in requested}
        local_body_hashes: set[str] = set()
        for article in sorted(
            unique.values(), key=lambda value: (value.provider_timestamp, value.article_id)
        ):
            target_identity = (article.article_id, member.ticker)
            if target_identity in seen_article_targets:
                continue
            seen_article_targets.add(target_identity)
            totals["considered"] += 1
            relevance = company_relevance_score(
                article,
                member,
                minimum_text_characters=config.minimum_text_characters,
                minimum_score=config.minimum_relevance_score,
                maximum_symbols=config.maximum_symbols,
            )
            body_hash = story_body_hash(article.content)
            reasons = list(relevance.exclusion_reasons)
            if body_hash in local_body_hashes:
                reasons.append("exact_body_duplicate")
            if not _complete_returns(article, prices, config.horizons):
                reasons.append("incomplete_forward_returns")
            eligible = relevance.eligible
            if "exact_body_duplicate" in reasons or "incomplete_forward_returns" in reasons:
                eligible = False
            candidate_rows.append(
                {
                    "article_id": article.article_id,
                    "ticker": member.ticker,
                    "provider_timestamp": article.provider_timestamp,
                    "title": article.title,
                    "content_characters": len(article.content),
                    "symbol_count": len(article.symbols),
                    "pre_inference_relevance_score": relevance.score,
                    "eligible_before_story_clustering": eligible,
                    "exclusion_reasons": reasons,
                    "score_components_json": stable_json(relevance.components),
                    "event_type_candidates": [
                        value.value for value in relevance.event_type_candidates
                    ],
                }
            )
            if eligible:
                local_body_hashes.add(body_hash)
                scored.append(ScoredCandidate(article, member, relevance))
                totals["eligible_before_story_clustering"] += 1
        log.info(
            "Hybrid sample sync %d/%d %s candidates=%d eligible_total=%d",
            member_index,
            len(config.universe),
            member.ticker,
            len(unique),
            totals["eligible_before_story_clustering"],
        )

    cluster_rows = cluster_syndicated_stories(item.article for item in scored)
    cluster_by_id = {item.article_id: item for item in cluster_rows}
    clustered = [
        ScoredCandidate(
            article=item.article,
            member=item.member,
            relevance=item.relevance,
            cluster_id=cluster_by_id[item.article.article_id].cluster_id,
            cluster_primary=cluster_by_id[item.article.article_id].primary,
        )
        for item in scored
        if cluster_by_id[item.article.article_id].primary
    ]
    by_ticker_year: defaultdict[tuple[str, int], list[ScoredCandidate]] = defaultdict(list)
    for item in clustered:
        by_ticker_year[(item.member.ticker, item.article.provider_timestamp.year)].append(item)

    selected: list[ScoredCandidate] = []
    years = range(config.news_start.year, config.news_end.year + 1)
    for member_index, member in enumerate(config.universe):
        company_selected: list[ScoredCandidate] = []
        for year in years:
            yearly = _select_year(
                by_ticker_year[(member.ticker, year)],
                count=config.articles_per_company_year,
                earnings_guidance_target=config.earnings_guidance_per_company_year,
                seed=config.random_seed + member_index * 100 + year,
            )
            if len(yearly) != config.articles_per_company_year:
                raise RuntimeError(
                    f"{member.ticker} year {year} has {len(yearly)} eligible primary stories; "
                    f"{config.articles_per_company_year} required"
                )
            company_selected.extend(yearly)
        if len(company_selected) != config.articles_per_company:
            raise RuntimeError(f"{member.ticker} did not produce its frozen allocation")
        selected.extend(company_selected)

    selected.sort(key=lambda item: (item.article.provider_timestamp, item.member.ticker, item.article.article_id))
    if len(selected) != config.max_articles:
        raise RuntimeError(f"Sample has {len(selected)} articles, expected {config.max_articles}")
    if len({item.article.article_id for item in selected}) != len(selected):
        raise RuntimeError("Frozen sample contains duplicate article IDs")
    if len({item.cluster_id for item in selected}) != len(selected):
        raise RuntimeError("Frozen sample contains duplicate story clusters")
    event_counts = Counter(item.primary_event_type for item in selected)
    earnings_guidance = event_counts[HybridEventType.earnings] + event_counts[HybridEventType.guidance]
    other_fraction = event_counts[HybridEventType.other] / len(selected)
    if earnings_guidance < config.minimum_earnings_guidance:
        raise RuntimeError(
            f"Frozen sample has only {earnings_guidance} earnings/guidance candidates"
        )
    if other_fraction > config.maximum_other_fraction:
        raise RuntimeError(f"Preclassified other fraction {other_fraction:.3f} exceeds target")

    aligned_rows = _aligned_rows(selected, prices_by_ticker, config.horizons)
    price_rows: list[dict[str, Any]] = []
    for ticker, prices in sorted(prices_by_ticker.items()):
        for price in prices:
            price_rows.append({"ticker": ticker, **price.model_dump(mode="python")})
    sample_material = {
        "config": config.model_dump(mode="json", exclude={"frozen_sample_hash"}),
        "articles": [
            {
                "article_id": item.article.article_id,
                "content_hash": article_content_hash(item.article.title, item.article.content),
                "ticker": item.member.ticker,
                "cluster_id": item.cluster_id,
            }
            for item in selected
        ],
        "prices": [
            {"ticker": ticker, **price.model_dump(mode="json")}
            for ticker, prices in sorted(prices_by_ticker.items())
            for price in prices
        ],
    }
    sample_hash = hashlib.sha256(stable_json(sample_material).encode()).hexdigest()
    if config.frozen_sample_hash is not None and config.frozen_sample_hash != sample_hash:
        raise RuntimeError(
            f"Frozen sample hash mismatch: expected {config.frozen_sample_hash}, got {sample_hash}"
        )
    root = data_root / "normalized" / "hybrid_5000" / sample_hash
    store = ArtifactStore(data_root, duckdb_path)
    articles_path = store.write_parquet(
        pl.DataFrame(aligned_rows, infer_schema_length=None), root / "articles.parquet"
    )
    candidates_path = store.write_parquet(
        pl.DataFrame(candidate_rows, infer_schema_length=None), root / "candidate_scores.parquet"
    )
    prices_path = store.write_parquet(
        pl.DataFrame(price_rows, infer_schema_length=None), root / "prices.parquet"
    )
    manifest = {
        "sample_hash": sample_hash,
        "frozen_before_local_inference": True,
        "article_count": len(selected),
        "company_count": len({item.member.ticker for item in selected}),
        "sector_count": len({item.member.sector for item in selected}),
        "years": sorted({item.article.provider_timestamp.year for item in selected}),
        "maximum_ticker_fraction": max(
            Counter(item.member.ticker for item in selected).values()
        )
        / len(selected),
        "event_type_candidates": {key.value: value for key, value in event_counts.items()},
        "earnings_guidance_count": earnings_guidance,
        "other_fraction": other_fraction,
        "candidate_filtering": dict(totals),
        "config": config.model_dump(mode="json"),
        "artifacts": {
            "articles": str(articles_path),
            "candidate_scores": str(candidates_path),
            "prices": str(prices_path),
        },
    }
    manifest_path = store.write_json(manifest, root / "manifest.json")
    store.register_parquet_view("hybrid_5000_frozen_articles", articles_path)
    store.register_parquet_view("hybrid_5000_candidate_scores", candidates_path)
    return FrozenHybridSample(
        sample_hash=sample_hash,
        article_count=len(selected),
        articles_path=articles_path,
        candidates_path=candidates_path,
        prices_path=prices_path,
        manifest_path=manifest_path,
    )
