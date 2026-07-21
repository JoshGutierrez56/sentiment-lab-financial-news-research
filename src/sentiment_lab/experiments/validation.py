"""Frozen 250-article validation without portfolio construction or factor modeling."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from sentiment_lab.backtest.event_engine import align_events
from sentiment_lab.config.models import (
    AppConfig,
    SentimentConfig,
    ValidationExperimentConfig,
    ValidationUniverseMember,
)
from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.schemas import EODPrice, NewsArticle
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.nlp.cache import story_body_hash
from sentiment_lab.nlp.classifier import (
    ArticleClassifier,
    ClassificationRun,
    ClassificationTarget,
)
from sentiment_lab.nlp.schemas import ClassificationRecord

NEW_YORK = ZoneInfo("America/New_York")

_EVENT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("earnings", ("earnings", "quarterly results", "revenue", "eps", "profit")),
    ("guidance", ("guidance", "forecast", "outlook", "expects", "raises forecast")),
    ("analyst_action", ("upgrade", "downgrade", "price target", "analyst", "rating")),
    ("product_news", ("launch", "product", "approval", "drug", "platform", "service")),
    ("litigation", ("lawsuit", "litigation", "court", "settlement", "sued")),
    ("regulation", ("regulator", "regulatory", "antitrust", "ftc", "sec ", "doj ")),
    ("capital_allocation", ("buyback", "repurchase", "dividend", "acquisition", "merger")),
    ("operational", ("outage", "disruption", "recall", "production", "strike", "cyber")),
)


class ValidationDataClient(Protocol):
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
class SampledArticle:
    article: NewsArticle
    member: ValidationUniverseMember
    sampling_event_bucket: str


@dataclass(frozen=True)
class ValidationSnapshot:
    snapshot_id: str
    sampled: list[SampledArticle]
    prices_by_ticker: dict[str, list[EODPrice]]
    articles_path: Path
    prices_path: Path
    filter_report: dict[str, Any]


def sampling_event_bucket(article: NewsArticle) -> str:
    text = f"{article.title} {article.content[:3000]}".casefold()
    for name, terms in _EVENT_TERMS:
        if any(term in text for term in terms):
            return name
    return "other"


def _full_text(article: NewsArticle, minimum_characters: int) -> bool:
    content = " ".join(article.content.split())
    return len(content) >= minimum_characters and content.casefold() != article.title.casefold()


def _market_summary(article: NewsArticle, symbol_threshold: int) -> bool:
    title = article.title.casefold()
    summary_words = ("market roundup", "market wrap", "stocks today", "premarket update")
    return len(article.symbols) >= symbol_threshold and any(word in title for word in summary_words)


def _has_complete_returns(
    article: NewsArticle, prices: list[EODPrice], horizons: list[int]
) -> bool:
    local_day = article.provider_timestamp.astimezone(NEW_YORK).date()
    entry_index = next(
        (index for index, price in enumerate(prices) if price.date > local_day),
        None,
    )
    return entry_index is not None and entry_index + max(horizons) - 1 < len(prices)


def _seeded_key(article: NewsArticle, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{article.article_id}".encode()).hexdigest()


def _select_company_articles(
    candidates: list[NewsArticle], *, count: int, seed: int
) -> list[NewsArticle]:
    """Greedily balance event buckets and months with deterministic tie-breaking."""

    remaining = list(candidates)
    selected: list[NewsArticle] = []
    month_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    while remaining and len(selected) < count:
        ranked = sorted(
            remaining,
            key=lambda article: (
                event_counts[sampling_event_bucket(article)],
                month_counts[article.provider_timestamp.strftime("%Y-%m")],
                _seeded_key(article, seed),
                article.article_id,
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        remaining.remove(chosen)
        event_counts[sampling_event_bucket(chosen)] += 1
        month_counts[chosen.provider_timestamp.strftime("%Y-%m")] += 1
    return sorted(selected, key=lambda item: (item.provider_timestamp, item.article_id))


def _article_rows(sampled: list[SampledArticle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sampled:
        row = item.article.model_dump(mode="python")
        row.update(
            {
                "ticker": item.member.ticker,
                "company_name": item.member.company_name,
                "sector": item.member.sector,
                "article_text_type": "full_text",
                "sampling_event_bucket": item.sampling_event_bucket,
            }
        )
        rows.append(row)
    return rows


def sync_validation_data(
    experiment: ValidationExperimentConfig,
    app_config: AppConfig,
    sentiment_config: SentimentConfig,
    client: ValidationDataClient,
    *,
    refresh: bool = False,
) -> ValidationSnapshot:
    """Create and verify the exact pre-OpenAI sample."""

    selected: list[SampledArticle] = []
    prices_by_ticker: dict[str, list[EODPrice]] = {}
    global_story_hashes: set[str] = set()
    totals: Counter[str] = Counter()
    per_ticker: dict[str, dict[str, int]] = {}
    price_end = experiment.news_end + timedelta(days=max(experiment.horizons) * 3 + 14)

    for member_index, member in enumerate(experiment.universe):
        requested = client.fetch_news(
            member.ticker,
            experiment.news_start,
            experiment.news_end,
            max_articles=experiment.news_candidate_pool_per_company,
            refresh=refresh,
        )
        prices = client.fetch_eod_prices(
            member.ticker,
            experiment.news_start,
            price_end,
            refresh=refresh,
        )
        prices = sorted(prices, key=lambda item: item.date)
        if not prices:
            raise RuntimeError(f"No EOD prices returned for {member.ticker}")
        prices_by_ticker[member.ticker] = prices
        counts: Counter[str] = Counter(considered=len(requested))
        eligible: list[NewsArticle] = []
        local_hashes: set[str] = set()
        for article in sorted(
            requested, key=lambda item: (item.provider_timestamp, item.article_id)
        ):
            local_day = article.provider_timestamp.astimezone(NEW_YORK).date()
            if not experiment.news_start <= local_day <= experiment.news_end:
                counts["outside_sample"] += 1
                continue
            if member.ticker not in article.symbols:
                counts["low_confidence_ticker_mapping"] += 1
                continue
            if not _full_text(article, sentiment_config.minimum_article_characters):
                counts["headline_only_or_low_text"] += 1
                continue
            if _market_summary(article, sentiment_config.market_summary_symbol_threshold):
                counts["irrelevant_market_summary"] += 1
                continue
            story_hash = story_body_hash(article.content)
            if story_hash in local_hashes or story_hash in global_story_hashes:
                counts["duplicate_story"] += 1
                continue
            if not _has_complete_returns(article, prices, experiment.horizons):
                counts["incomplete_forward_returns"] += 1
                continue
            local_hashes.add(story_hash)
            eligible.append(article)
        chosen = _select_company_articles(
            eligible,
            count=experiment.articles_per_company,
            seed=experiment.random_seed + member_index,
        )
        if len(chosen) != experiment.articles_per_company:
            raise RuntimeError(
                f"{member.ticker} has only {len(chosen)} eligible full-text articles; "
                f"{experiment.articles_per_company} are required. No OpenAI request was made."
            )
        for article in chosen:
            global_story_hashes.add(story_body_hash(article.content))
            selected.append(
                SampledArticle(
                    article=article,
                    member=member,
                    sampling_event_bucket=sampling_event_bucket(article),
                )
            )
        counts["eligible_before_sample_limit"] = len(eligible)
        counts["sample_limit"] = len(eligible) - len(chosen)
        counts["selected"] = len(chosen)
        per_ticker[member.ticker] = dict(counts)
        totals.update(counts)

    selected.sort(key=lambda item: (item.article.provider_timestamp, item.member.ticker))
    if len(selected) != experiment.max_articles:
        raise RuntimeError(
            f"Frozen sample has {len(selected)} articles, expected {experiment.max_articles}"
        )
    if len({item.article.article_id for item in selected}) != len(selected):
        raise RuntimeError("Frozen sample contains duplicate article IDs")
    if len({story_body_hash(item.article.content) for item in selected}) != len(selected):
        raise RuntimeError("Frozen sample contains duplicate article bodies")

    months = sorted({item.article.provider_timestamp.strftime("%Y-%m") for item in selected})
    sectors = sorted({item.member.sector for item in selected})
    event_buckets = sorted({item.sampling_event_bucket for item in selected})
    if len(months) < experiment.minimum_months:
        raise RuntimeError(f"Sample covers only {len(months)} months")
    if len(sectors) < experiment.minimum_sectors:
        raise RuntimeError(f"Sample covers only {len(sectors)} sectors")
    if len(event_buckets) < experiment.minimum_event_buckets:
        raise RuntimeError(f"Sample covers only {len(event_buckets)} event buckets")

    price_rows: list[dict[str, Any]] = []
    for ticker, prices in sorted(prices_by_ticker.items()):
        for price in prices:
            row = price.model_dump(mode="json")
            row["ticker"] = ticker
            price_rows.append(row)
    material = {
        "config": experiment.model_dump(mode="json", exclude={"frozen_snapshot_id"}),
        "article_ids": [item.article.article_id for item in selected],
        "article_raw_hashes": [item.article.raw_response_hash for item in selected],
        "prices": price_rows,
        "filter_policy": sentiment_config.model_dump(mode="json"),
    }
    snapshot_id = hashlib.sha256(stable_json(material).encode()).hexdigest()[:16]
    if experiment.frozen_snapshot_id and snapshot_id != experiment.frozen_snapshot_id:
        raise RuntimeError(
            "Recomputed sample does not match frozen_snapshot_id: "
            f"expected {experiment.frozen_snapshot_id}, got {snapshot_id}. No OpenAI request made."
        )
    root = app_config.storage.data_root / "normalized" / "validation_250" / snapshot_id
    store = ArtifactStore(app_config.storage.data_root, app_config.storage.duckdb_path)
    articles_path = store.write_parquet(
        pl.DataFrame(_article_rows(selected), infer_schema_length=None),
        root / "articles.parquet",
    )
    prices_path = store.write_parquet(
        pl.DataFrame(price_rows, infer_schema_length=None), root / "prices.parquet"
    )
    report: dict[str, Any] = {
        "total_articles_considered": totals["considered"],
        "articles_filtered_before_openai": totals["considered"] - len(selected),
        "filtered_by_reason": {
            key: totals[key]
            for key in (
                "outside_sample",
                "low_confidence_ticker_mapping",
                "headline_only_or_low_text",
                "irrelevant_market_summary",
                "duplicate_story",
                "incomplete_forward_returns",
                "sample_limit",
            )
        },
        "selected_full_text": len(selected),
        "selected_headline_only": 0,
        "months": months,
        "sectors": sectors,
        "sampling_event_buckets": event_buckets,
        "per_ticker": per_ticker,
        "ticker_mapping_method": "eodhd_direct_symbol",
        "ticker_mapping_confidence": 1.0,
    }
    store.write_json(report, root / "filter_report.json")
    return ValidationSnapshot(
        snapshot_id=snapshot_id,
        sampled=selected,
        prices_by_ticker=prices_by_ticker,
        articles_path=articles_path,
        prices_path=prices_path,
        filter_report=report,
    )


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _corr(signal: np.ndarray, returns: np.ndarray) -> dict[str, float | None]:
    if len(signal) < 3 or np.std(signal) == 0 or np.std(returns) == 0:
        return {"pearson_ic": None, "spearman_ic": None}
    return {
        "pearson_ic": _finite(float(pearsonr(signal, returns).statistic)),
        "spearman_ic": _finite(float(spearmanr(signal, returns).statistic)),
    }


def _bootstrap_company_equal(
    tickers: np.ndarray,
    signed_returns: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    unique = np.unique(tickers)
    if len(unique) < 2:
        return {"lower_95": None, "upper_95": None}
    by_ticker = {ticker: float(np.mean(signed_returns[tickers == ticker])) for ticker in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = float(np.mean([by_ticker[ticker] for ticker in sampled]))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"lower_95": float(lower), "upper_95": float(upper)}


def _bootstrap_signal_statistics(
    tickers: np.ndarray,
    scores: np.ndarray,
    weighted_signal: np.ndarray,
    returns: np.ndarray,
    labels: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """Company-cluster bootstrap CIs for association and label spread."""

    unique = np.unique(tickers)
    statistics: dict[str, list[float]] = {
        "pearson_ic": [],
        "spearman_ic": [],
        "weighted_pearson_ic": [],
        "weighted_spearman_ic": [],
        "bullish_minus_bearish_spread": [],
    }
    if len(unique) < 2:
        return {name: {"lower_95": None, "upper_95": None} for name in statistics}
    indices = {ticker: np.flatnonzero(tickers == ticker) for ticker in unique}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        sampled_tickers = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([indices[ticker] for ticker in sampled_tickers])
        raw = _corr(scores[draw], returns[draw])
        weighted = _corr(weighted_signal[draw], returns[draw])
        for name in ("pearson_ic", "spearman_ic"):
            value = raw[name]
            if value is not None:
                statistics[name].append(value)
        for name in ("pearson_ic", "spearman_ic"):
            value = weighted[name]
            if value is not None:
                statistics[f"weighted_{name}"].append(value)
        bullish = returns[draw][labels[draw] == "bullish"]
        bearish = returns[draw][labels[draw] == "bearish"]
        if len(bullish) and len(bearish):
            statistics["bullish_minus_bearish_spread"].append(
                float(np.mean(bullish) - np.mean(bearish))
            )
    output: dict[str, dict[str, float | None]] = {}
    for name, values in statistics.items():
        if values:
            lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
            output[name] = {"lower_95": float(lower), "upper_95": float(upper)}
        else:
            output[name] = {"lower_95": None, "upper_95": None}
    return output


def compute_validation_metrics(
    events: pl.DataFrame,
    run: ClassificationRun,
    *,
    horizons: list[int],
    neutral_return_bps: float,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    threshold = neutral_return_bps / 10_000.0
    labels = Counter(cast(list[str], events["sentiment_label"].to_list()))
    n_abstain = int(events["abstain"].sum())
    n_tradable = int(events["tradable"].sum())
    escalated = [
        index for index, record in enumerate(run.final_records) if record.stage == "escalation"
    ]
    escalation_successes = 0
    escalation_changes = 0
    for index in escalated:
        initial = run.first_pass_records[index] if run.first_pass_records else None
        final = run.final_records[index]
        if initial is None:
            escalation_successes += 1
            continue
        changed = (
            initial.assessment.sentiment_label != final.assessment.sentiment_label
            or initial.assessment.tradable != final.assessment.tradable
            or initial.assessment.abstain != final.assessment.abstain
        )
        escalation_changes += int(changed)
        escalation_successes += int(changed)

    result: dict[str, Any] = {
        "definition": (
            "Directional accuracy and IC use final tradable, non-abstaining classifications. "
            "Neutral realized returns are within the configured basis-point band. Returns "
            "overlap and are not a portfolio return series; no Sharpe ratio is calculated."
        ),
        "valid_classification_count": len(run.final_records),
        "cache_hits": run.summary()["cache_hits"],
        "abstention_rate": n_abstain / events.height if events.height else None,
        "tradable_coverage": n_tradable / events.height if events.height else None,
        "escalation_rate": len(escalated) / events.height if events.height else None,
        "escalation_success_rate": (escalation_successes / len(escalated) if escalated else None),
        "escalation_conclusion_change_rate": (
            escalation_changes / len(escalated) if escalated else None
        ),
        "label_counts": {
            label: labels.get(label, 0) for label in ("bullish", "neutral", "bearish")
        },
        "horizons": {},
    }
    usable = events.filter(pl.col("tradable") & ~pl.col("abstain"))
    for horizon in horizons:
        column = f"future_return_{horizon}d"
        complete = usable.filter(pl.col(column).is_not_null())
        returns = complete[column].to_numpy().astype(float)
        scores = complete["sentiment_score"].to_numpy().astype(float)
        confidence = complete["confidence"].to_numpy().astype(float)
        materiality = complete["materiality"].to_numpy().astype(float)
        signal = scores * confidence * materiality
        event_labels = np.asarray(complete["sentiment_label"].to_list())
        realized = np.where(
            returns > threshold, "bullish", np.where(returns < -threshold, "bearish", "neutral")
        )
        signed = np.sign(scores) * returns
        lower, upper = np.quantile(returns, [0.01, 0.99]) if len(returns) else (0.0, 0.0)
        winsorized = np.clip(returns, lower, upper)
        by_label: dict[str, Any] = {}
        for label in ("bullish", "neutral", "bearish"):
            values = returns[event_labels == label]
            by_label[label] = {
                "n": len(values),
                "average_return": float(np.mean(values)) if len(values) else None,
                "median_return": float(np.median(values)) if len(values) else None,
            }
        bullish = returns[event_labels == "bullish"]
        bearish = returns[event_labels == "bearish"]
        tickers = np.asarray(complete["ticker"].to_list())
        company_means = [float(np.mean(signed[tickers == ticker])) for ticker in np.unique(tickers)]
        result["horizons"][f"{horizon}d"] = {
            "n": len(returns),
            "directional_accuracy": float(np.mean(event_labels == realized))
            if len(returns)
            else None,
            "returns_by_sentiment": by_label,
            "bullish_minus_bearish_average_spread": (
                float(np.mean(bullish) - np.mean(bearish))
                if len(bullish) and len(bearish)
                else None
            ),
            **_corr(scores, returns),
            "weighted_signal": {
                **_corr(signal, returns),
                "average_signed_return": float(np.mean(np.sign(signal) * returns))
                if len(returns)
                else None,
            },
            "equal_weighted_company_average_signed_return": (
                float(np.mean(company_means)) if company_means else None
            ),
            "company_cluster_bootstrap_95_ci": _bootstrap_company_equal(
                tickers,
                signed,
                samples=bootstrap_samples,
                seed=random_seed + horizon,
            )
            if len(returns)
            else {"lower_95": None, "upper_95": None},
            "company_cluster_signal_bootstrap_95_ci": _bootstrap_signal_statistics(
                tickers,
                scores,
                signal,
                returns,
                event_labels,
                samples=bootstrap_samples,
                seed=random_seed + horizon * 101,
            )
            if len(returns)
            else {},
            "winsorized_1_99": {
                **_corr(scores, winsorized),
                "average_signed_return": float(np.mean(np.sign(scores) * winsorized))
                if len(returns)
                else None,
                "lower_cutoff": float(lower) if len(returns) else None,
                "upper_cutoff": float(upper) if len(returns) else None,
            },
        }

    group_rows: dict[str, list[dict[str, Any]]] = {}
    for group in ("ticker", "event_type"):
        rows: list[dict[str, Any]] = []
        for values in events.partition_by(group, as_dict=False):
            row: dict[str, Any] = {
                group: values[group][0],
                "n": values.height,
                "tradable": int(values["tradable"].sum()),
                "abstain": int(values["abstain"].sum()),
                "mean_sentiment_score": float(cast(float, values["sentiment_score"].mean())),
            }
            for horizon in horizons:
                non_null = values[f"future_return_{horizon}d"].drop_nulls()
                row[f"average_return_{horizon}d"] = (
                    float(cast(float, non_null.mean())) if len(non_null) else None
                )
                usable_group = values.filter(
                    pl.col("tradable")
                    & ~pl.col("abstain")
                    & pl.col(f"future_return_{horizon}d").is_not_null()
                )
                group_returns = usable_group[f"future_return_{horizon}d"].to_numpy().astype(float)
                group_scores = usable_group["sentiment_score"].to_numpy().astype(float)
                group_labels = np.asarray(usable_group["sentiment_label"].to_list())
                realized_group = np.where(
                    group_returns > threshold,
                    "bullish",
                    np.where(group_returns < -threshold, "bearish", "neutral"),
                )
                row[f"signal_n_{horizon}d"] = len(group_returns)
                row[f"directional_accuracy_{horizon}d"] = (
                    float(np.mean(group_labels == realized_group)) if len(group_returns) else None
                )
                row[f"average_signed_return_{horizon}d"] = (
                    float(np.mean(np.sign(group_scores) * group_returns))
                    if len(group_returns)
                    else None
                )
            rows.append(row)
        group_rows[f"by_{group}"] = sorted(rows, key=lambda row: str(row[group]))
    result.update(group_rows)
    return result


def validation_decision(metrics: dict[str, Any]) -> tuple[str, str]:
    """Apply the bounded-stage decision rule without tuning on individual returns."""

    horizons = metrics["horizons"]
    evidence_horizons = [key for key in ("5d", "21d") if key in horizons]
    useful_evidence = any(
        (horizons[key]["weighted_signal"]["spearman_ic"] or 0.0) > 0.0
        and (horizons[key]["company_cluster_bootstrap_95_ci"]["lower_95"] or 0.0) > 0.0
        for key in evidence_horizons
    )
    if not useful_evidence:
        return "STOP", "No positive medium-horizon weighted IC with a positive company-level CI."
    other_rows = [row for row in metrics["by_event_type"] if row["event_type"] == "other"]
    other_share = other_rows[0]["n"] / metrics["valid_classification_count"] if other_rows else 0.0
    if (metrics["abstention_rate"] or 0.0) > 0.50 or other_share > 0.50:
        return (
            "REVISE",
            "Medium-horizon evidence exists, but abstention/source-mix concentration exceeds "
            "50%; repair sampling before spending on 1,000 articles.",
        )
    return "PROCEED", "Coverage and medium-horizon directional evidence justify 1,000 articles."


def _assessment_rows(
    sampled: list[SampledArticle], records: list[ClassificationRecord]
) -> list[dict[str, Any]]:
    metadata = {item.article.article_id: item for item in sampled}
    rows: list[dict[str, Any]] = []
    for record in records:
        item = metadata[record.article_id]
        row = record.assessment.model_dump(mode="python")
        row.update(
            {
                "article_id": record.article_id,
                "ticker": record.ticker,
                "company_name": item.member.company_name,
                "sector": item.member.sector,
                "event_timestamp": record.event_timestamp,
                "requested_model": record.requested_model,
                "model": record.model,
                "classification_stage": record.stage,
                "escalation_reasons": record.escalation_reasons,
                "cache_key": record.cache_key,
                "input_hash": record.input_hash,
                "output_hash": record.output_hash,
                "input_tokens": record.usage.input_tokens,
                "cached_input_tokens": record.usage.cached_input_tokens,
                "output_tokens": record.usage.output_tokens,
                "reasoning_tokens": record.usage.reasoning_tokens,
                "estimated_cost_usd": record.usage.estimated_cost_usd,
            }
        )
        rows.append(row)
    return rows


class ValidationRunner:
    def __init__(
        self,
        experiment: ValidationExperimentConfig,
        app_config: AppConfig,
        sentiment_config: SentimentConfig,
        eodhd_client: ValidationDataClient,
        classifier: ArticleClassifier,
    ) -> None:
        self.experiment = experiment
        self.app_config = app_config
        self.sentiment_config = sentiment_config
        self.eodhd_client = eodhd_client
        self.classifier = classifier

    def run(self, *, refresh: bool = False) -> Path:
        started = datetime.now(UTC)
        snapshot = sync_validation_data(
            self.experiment,
            self.app_config,
            self.sentiment_config,
            self.eodhd_client,
            refresh=refresh,
        )
        budget = self.app_config.openai.spending_limits_usd.bounded_validation
        if budget > 2.0:
            raise RuntimeError("Bounded validation budget may not exceed $2")
        targets = [
            ClassificationTarget(item.article, item.member.ticker, item.member.company_name)
            for item in snapshot.sampled
        ]
        run = self.classifier.classify_targets(
            targets,
            prompt_variant=self.experiment.prompt_variant,
            budget_limit_usd=budget,
        )
        event_frames: list[pl.DataFrame] = []
        for member in self.experiment.universe:
            indices = [
                index
                for index, item in enumerate(snapshot.sampled)
                if item.member.ticker == member.ticker
            ]
            event_frames.append(
                align_events(
                    [snapshot.sampled[index].article for index in indices],
                    [run.final_records[index] for index in indices],
                    snapshot.prices_by_ticker[member.ticker],
                    horizons=self.experiment.horizons,
                )
            )
        events = pl.concat(event_frames, how="vertical_relaxed").sort(
            ["publication_timestamp_utc", "ticker"]
        )
        for horizon in self.experiment.horizons:
            if events[f"future_return_{horizon}d"].null_count():
                raise RuntimeError(f"Final events contain missing {horizon}-day returns")
        metrics = compute_validation_metrics(
            events,
            run,
            horizons=self.experiment.horizons,
            neutral_return_bps=self.experiment.neutral_return_bps,
            bootstrap_samples=self.experiment.bootstrap_samples,
            random_seed=self.experiment.random_seed,
        )
        config_hash = hashlib.sha256(
            stable_json(
                {
                    "application": self.app_config.model_dump(mode="json"),
                    "experiment": self.experiment.model_dump(mode="json"),
                    "sentiment": self.sentiment_config.model_dump(mode="json"),
                }
            ).encode()
        ).hexdigest()
        experiment_id = (
            f"{started.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{hashlib.sha256(f'{snapshot.snapshot_id}:{config_hash}'.encode()).hexdigest()[:8]}"
        )
        output = self.app_config.storage.data_root / "results" / experiment_id
        store = ArtifactStore(
            self.app_config.storage.data_root, self.app_config.storage.duckdb_path
        )
        artifacts: list[Path] = []
        artifacts.append(
            store.write_parquet(
                pl.DataFrame(_article_rows(snapshot.sampled), infer_schema_length=None),
                output / "articles.parquet",
            )
        )
        artifacts.append(
            store.write_parquet(
                pl.DataFrame(
                    _assessment_rows(snapshot.sampled, run.final_records), infer_schema_length=None
                ),
                output / "assessments.parquet",
            )
        )
        artifacts.append(
            store.write_parquet(
                pl.DataFrame(
                    [entry.model_dump(mode="python") for entry in run.ledger_entries],
                    infer_schema_length=None,
                ),
                output / "classification_ledger.parquet",
            )
        )
        artifacts.append(store.write_parquet(events, output / "events.parquet"))
        data_quality_issues = [
            "EODHD direct-symbol mapping is treated as confidence 1.0; ambiguous multi-company effects are left to model abstention.",
            "The universe is a current liquid-equity research set and therefore does not remove survivorship bias.",
            "Sampling-event buckets are deterministic keyword strata, not human-validated event labels.",
            "Syndicated rewrites can survive exact normalized-body deduplication.",
            "There is no independent human sentiment label set in this stage.",
            "Forward event returns overlap; ordinary ICs and company-cluster bootstrap intervals are descriptive, not a portfolio Sharpe series.",
        ]
        custom_id_method = getattr(self.classifier, "batch_custom_ids", None)
        if callable(custom_id_method):
            custom_ids = cast(
                set[str],
                custom_id_method(targets, prompt_variant=self.experiment.prompt_variant),
            )
        else:
            custom_ids = {
                entry.batch_custom_id
                for entry in run.ledger_entries
                if entry.batch_custom_id is not None
            }
        batch_client = getattr(self.classifier, "batch_client", None)
        audit_method = getattr(batch_client, "audit_saved_batch_usage", None)
        if callable(audit_method):
            api_usage_audit = cast(dict[str, int | float], audit_method(custom_ids))
        else:
            api_usage_audit = {
                "batch_count": 0,
                "request_attempts": 0,
                "structured_output_failures": 0,
                "input_tokens": sum(entry.input_tokens for entry in run.ledger_entries),
                "cached_input_tokens": sum(
                    entry.cached_input_tokens for entry in run.ledger_entries
                ),
                "output_tokens": sum(entry.output_tokens for entry in run.ledger_entries),
                "reasoning_tokens": sum(entry.reasoning_tokens for entry in run.ledger_entries),
                "actual_api_cost_usd": sum(
                    entry.estimated_cost_usd for entry in run.ledger_entries
                ),
                "preflight_maximum_estimated_cost_usd_all_attempts": 0.0,
                "largest_batch_preflight_maximum_usd": 0.0,
            }
        actual_api_cost = float(api_usage_audit["actual_api_cost_usd"])
        unique_escalated = int(api_usage_audit.get("unique_escalated_articles", 0))
        if unique_escalated:
            metrics["escalation_rate"] = unique_escalated / len(run.final_records)
            metrics["escalation_success_rate"] = 1.0
            metrics["unique_articles_ever_escalated"] = unique_escalated
            metrics["final_expensive_model_classifications"] = sum(
                record.stage == "escalation" for record in run.final_records
            )
        metrics["article_first_pass_cache_hits"] = sum(
            entry.stage == "first_pass" and entry.outcome == "cache_hit"
            for entry in run.ledger_entries
        )
        metrics["stage_cache_hits"] = run.summary()["cache_hits"]
        decision, decision_rationale = validation_decision(metrics)
        if int(api_usage_audit["structured_output_failures"]):
            data_quality_issues.append(
                f"{int(api_usage_audit['structured_output_failures'])} structured-output "
                "attempts required cached, cost-accounted repair batches."
            )
        artifacts.append(store.write_json(metrics, output / "metrics.json"))
        report_payload = {
            "experiment_id": experiment_id,
            "snapshot_id": snapshot.snapshot_id,
            "metrics": metrics,
            "filtering": snapshot.filter_report,
            "actual_api_cost_usd": actual_api_cost,
            "cost_per_article_usd": actual_api_cost / len(run.final_records),
            "api_usage_audit": api_usage_audit,
            "data_quality_issues": data_quality_issues,
            "decision": decision,
            "decision_rationale": decision_rationale,
        }
        report_path = output / "report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "<!doctype html><meta charset='utf-8'><title>Bounded 250-Article Validation</title>"
            "<h1>Bounded 250-Article Validation</h1><p>No portfolio or Sharpe ratio was constructed.</p>"
            f"<pre>{html.escape(json.dumps(report_payload, indent=2, default=str))}</pre>",
            encoding="utf-8",
        )
        artifacts.append(report_path)
        manifest = {
            "experiment_id": experiment_id,
            "created_at": started.isoformat(),
            "status": "complete",
            "config_hash": config_hash,
            "data_snapshot_id": snapshot.snapshot_id,
            "frozen_snapshot_verified": self.experiment.frozen_snapshot_id == snapshot.snapshot_id,
            "experiment_config": self.experiment.model_dump(mode="json"),
            "filtering": snapshot.filter_report,
            "classification": run.summary(),
            "cost_control": {
                "spending_limit_usd": budget,
                "largest_batch_preflight_maximum_usd": api_usage_audit[
                    "largest_batch_preflight_maximum_usd"
                ],
                "all_attempts_preflight_maximum_usd": api_usage_audit[
                    "preflight_maximum_estimated_cost_usd_all_attempts"
                ],
                "actual_api_cost_usd": actual_api_cost,
                "cost_per_article_usd": actual_api_cost / len(run.final_records),
                "within_budget": actual_api_cost <= budget,
            },
            "api_usage_audit": api_usage_audit,
            "metrics": metrics,
            "data_quality_issues": data_quality_issues,
            "decision": decision,
            "decision_rationale": decision_rationale,
            "artifacts": {
                path.name: {"path": str(path), "sha256": file_sha256(path)} for path in artifacts
            },
        }
        store.write_json(manifest, output / "manifest.json")
        store.register_parquet_view("validation_250_events_latest", output / "events.parquet")
        store.register_parquet_view(
            "validation_250_assessments_latest", output / "assessments.parquet"
        )
        return output
