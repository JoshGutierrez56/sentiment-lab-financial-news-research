"""One focused experiment: EODHD article → OpenAI assessment → future return."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from sentiment_lab.backtest.event_engine import align_events
from sentiment_lab.backtest.metrics import compute_event_metrics
from sentiment_lab.config.models import AppConfig, ExperimentConfig, SentimentConfig
from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.eodhd_client import EODHDClient
from sentiment_lab.data.schemas import EODPrice, NewsArticle
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.nlp.cache import story_body_hash
from sentiment_lab.nlp.classifier import ArticleClassifier
from sentiment_lab.nlp.schemas import ClassificationLedgerEntry, ClassificationRecord
from sentiment_lab.reporting.report import build_milestone_report

NEW_YORK = ZoneInfo("America/New_York")
_MARKET_SUMMARY_PATTERN = re.compile(
    r"\b(?:market|stocks?|wall street|premarket)\b.*"
    r"\b(?:roundup|wrap|recap|update|today|higher|lower|rally|selloff)\b",
    re.IGNORECASE,
)


def _usage_totals(records: list[ClassificationRecord]) -> dict[str, int | float]:
    return {
        "input_tokens": sum(record.usage.input_tokens for record in records),
        "cached_input_tokens": sum(record.usage.cached_input_tokens for record in records),
        "output_tokens": sum(record.usage.output_tokens for record in records),
        "reasoning_tokens": sum(record.usage.reasoning_tokens for record in records),
        "total_tokens": sum(
            record.usage.input_tokens + record.usage.output_tokens for record in records
        ),
        "estimated_cost_usd": sum(record.usage.estimated_cost_usd for record in records),
    }


def _ledger_usage(
    entries: list[ClassificationLedgerEntry], *, current_run_only: bool
) -> dict[str, int | float]:
    selected = [
        entry
        for entry in entries
        if not current_run_only or entry.outcome in {"api_success", "api_failure"}
    ]
    return {
        "input_tokens": sum(entry.input_tokens for entry in selected),
        "cached_input_tokens": sum(entry.cached_input_tokens for entry in selected),
        "output_tokens": sum(entry.output_tokens for entry in selected),
        "reasoning_tokens": sum(entry.reasoning_tokens for entry in selected),
        "total_tokens": sum(entry.input_tokens + entry.output_tokens for entry in selected),
        "estimated_cost_usd": sum(entry.estimated_cost_usd for entry in selected),
        "current_run_cost_usd": sum(entry.run_cost_usd for entry in selected),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class DataSnapshot:
    snapshot_id: str
    articles: list[NewsArticle]
    prices: list[EODPrice]
    articles_path: Path
    prices_path: Path
    article_text_types: dict[str, str]
    filter_report: dict[str, Any]


def _frame_from_models(models: list[Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [model.model_dump(mode="python") for model in models], infer_schema_length=None
    )


def _article_text_type(article: NewsArticle, minimum_characters: int) -> str:
    content = " ".join(article.content.split())
    title = " ".join(article.title.split())
    if len(content) >= minimum_characters and content.casefold() != title.casefold():
        return "full_text"
    return "headline_only"


def _is_market_summary(article: NewsArticle, symbol_threshold: int) -> bool:
    return len(article.symbols) >= symbol_threshold and bool(
        _MARKET_SUMMARY_PATTERN.search(article.title)
    )


def _select_diverse_articles(
    candidate_groups: list[list[NewsArticle]], experiment: ExperimentConfig
) -> list[NewsArticle]:
    """Select date-diverse articles while preserving group priority.

    Full-text and headline-only candidates are passed as separate groups so an
    earlier headline can never displace an eligible full article.
    """

    selected: list[NewsArticle] = []
    selected_per_day: dict[date, int] = {}
    for candidates in candidate_groups:
        by_day: dict[date, list[NewsArticle]] = {}
        for article in sorted(
            candidates, key=lambda item: (item.provider_timestamp, item.article_id)
        ):
            local_day = article.provider_timestamp.astimezone(NEW_YORK).date()
            by_day.setdefault(local_day, []).append(article)
        for daily_index in range(experiment.max_articles_per_day):
            for local_day in sorted(by_day):
                if selected_per_day.get(local_day, 0) >= experiment.max_articles_per_day:
                    continue
                daily_articles = by_day[local_day]
                if daily_index < len(daily_articles):
                    selected.append(daily_articles[daily_index])
                    selected_per_day[local_day] = selected_per_day.get(local_day, 0) + 1
                    if len(selected) >= experiment.max_articles:
                        return selected
    return selected


def _article_frame(articles: list[NewsArticle], text_types: dict[str, str]) -> pl.DataFrame:
    rows = []
    for article in articles:
        row = article.model_dump(mode="python")
        row["article_text_type"] = text_types[article.article_id]
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def sync_milestone_data(
    experiment: ExperimentConfig,
    app_config: AppConfig,
    sentiment_config: SentimentConfig,
    client: EODHDClient,
    *,
    refresh: bool = False,
) -> DataSnapshot:
    """Fetch a real, directly mapped article sample and enough future prices."""

    requested = client.fetch_news(
        experiment.ticker,
        experiment.news_start,
        experiment.news_end,
        max_articles=experiment.news_candidate_pool,
        refresh=refresh,
    )
    filtered = {
        "outside_sample": 0,
        "low_confidence_ticker_mapping": 0,
        "inadequate_text": 0,
        "irrelevant_market_summary": 0,
        "duplicate_story": 0,
        "sample_limit": 0,
    }
    full_text: list[NewsArticle] = []
    headline_only: list[NewsArticle] = []
    text_types: dict[str, str] = {}
    seen_story_hashes: set[str] = set()
    for article in sorted(requested, key=lambda item: (item.provider_timestamp, item.article_id)):
        local_day = article.provider_timestamp.astimezone(NEW_YORK).date()
        if not experiment.news_start <= local_day <= experiment.news_end:
            filtered["outside_sample"] += 1
            continue
        mapping_confidence = 1.0 if experiment.ticker in article.symbols else 0.0
        if mapping_confidence < sentiment_config.ticker_mapping_confidence_threshold:
            filtered["low_confidence_ticker_mapping"] += 1
            continue
        text_type = _article_text_type(article, sentiment_config.minimum_article_characters)
        text_types[article.article_id] = text_type
        if text_type == "headline_only" and not sentiment_config.classify_headline_only:
            filtered["inadequate_text"] += 1
            continue
        if _is_market_summary(article, sentiment_config.market_summary_symbol_threshold):
            filtered["irrelevant_market_summary"] += 1
            continue
        duplicate_key = story_body_hash(article.content or article.title)
        if duplicate_key in seen_story_hashes:
            filtered["duplicate_story"] += 1
            continue
        seen_story_hashes.add(duplicate_key)
        if text_type == "full_text":
            full_text.append(article)
        else:
            headline_only.append(article)

    articles = _select_diverse_articles([full_text, headline_only], experiment)
    filtered["sample_limit"] = len(full_text) + len(headline_only) - len(articles)
    if not articles:
        raise RuntimeError(
            "EODHD returned no eligible articles after text, mapping, duplicate, and relevance "
            "filters."
        )
    price_end = experiment.news_end + timedelta(days=max(experiment.horizons) * 3 + 10)
    prices = client.fetch_eod_prices(
        experiment.ticker,
        experiment.news_start,
        price_end,
        refresh=refresh,
    )
    if not prices:
        raise RuntimeError("EODHD returned no EOD prices for the milestone window.")
    snapshot_material = {
        "article_ids": [article.article_id for article in articles],
        "article_raw_hashes": [article.raw_response_hash for article in articles],
        "prices": [price.model_dump(mode="json") for price in prices],
        "filter_policy": sentiment_config.model_dump(mode="json"),
    }
    snapshot_id = hashlib.sha256(stable_json(snapshot_material).encode("utf-8")).hexdigest()[:16]
    root = app_config.storage.data_root / "normalized" / "milestone" / snapshot_id
    store = ArtifactStore(app_config.storage.data_root, app_config.storage.duckdb_path)
    selected_text_types = {
        article.article_id: text_types[article.article_id] for article in articles
    }
    filter_report: dict[str, Any] = {
        "total_articles_considered": len(requested),
        "articles_filtered_before_openai": len(requested) - len(articles),
        "filtered_by_reason": filtered,
        "eligible_full_text": len(full_text),
        "eligible_headline_only": len(headline_only),
        "selected_full_text": sum(
            selected_text_types[article.article_id] == "full_text" for article in articles
        ),
        "selected_headline_only": sum(
            selected_text_types[article.article_id] == "headline_only" for article in articles
        ),
        "ticker_mapping_method": "eodhd_direct_symbol",
        "ticker_mapping_confidence": 1.0,
    }
    articles_path = store.write_parquet(
        _article_frame(articles, selected_text_types), root / "articles.parquet"
    )
    prices_path = store.write_parquet(_frame_from_models(prices), root / "prices.parquet")
    store.register_parquet_view("milestone_articles_latest", articles_path)
    store.register_parquet_view("milestone_prices_latest", prices_path)
    return DataSnapshot(
        snapshot_id=snapshot_id,
        articles=articles,
        prices=prices,
        articles_path=articles_path,
        prices_path=prices_path,
        article_text_types=selected_text_types,
        filter_report=filter_report,
    )


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


class MilestoneRunner:
    def __init__(
        self,
        experiment: ExperimentConfig,
        app_config: AppConfig,
        sentiment_config: SentimentConfig,
        eodhd_client: EODHDClient,
        classifier: ArticleClassifier,
    ) -> None:
        self.experiment = experiment
        self.app_config = app_config
        self.sentiment_config = sentiment_config
        self.eodhd_client = eodhd_client
        self.classifier = classifier

    def run(
        self,
        *,
        refresh: bool = False,
    ) -> Path:
        started = datetime.now(UTC)
        snapshot = sync_milestone_data(
            self.experiment,
            self.app_config,
            self.sentiment_config,
            self.eodhd_client,
            refresh=refresh,
        )
        budget_limit_usd = float(
            getattr(
                self.app_config.openai.spending_limits_usd,
                self.experiment.spending_limit_tier,
            )
        )
        classification_run = self.classifier.classify_many(
            snapshot.articles,
            ticker=self.experiment.ticker,
            company_name=self.experiment.company_name,
            prompt_variant=self.experiment.prompt_variant,
            budget_limit_usd=budget_limit_usd,
        )
        classifications = classification_run.final_records
        events = align_events(
            snapshot.articles,
            classifications,
            snapshot.prices,
            horizons=self.experiment.horizons,
        )
        metrics = compute_event_metrics(
            events,
            horizons=self.experiment.horizons,
            neutral_return_bps=self.experiment.neutral_return_bps,
        )
        config_material = {
            "application": self.app_config.model_dump(mode="json"),
            "experiment": self.experiment.model_dump(mode="json"),
            "sentiment": self.sentiment_config.model_dump(mode="json"),
        }
        config_hash = hashlib.sha256(stable_json(config_material).encode("utf-8")).hexdigest()
        run_suffix = hashlib.sha256(
            f"{started.isoformat()}:{snapshot.snapshot_id}:{config_hash}".encode()
        ).hexdigest()[:8]
        experiment_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{run_suffix}"
        output = self.app_config.storage.data_root / "results" / experiment_id
        store = ArtifactStore(
            self.app_config.storage.data_root, self.app_config.storage.duckdb_path
        )
        articles_path = store.write_parquet(
            _article_frame(snapshot.articles, snapshot.article_text_types),
            output / "articles.parquet",
        )
        assessment_rows = []
        for record in classifications:
            row = record.assessment.model_dump(mode="python")
            row.update(
                {
                    "article_id": record.article_id,
                    "ticker": record.ticker,
                    "event_timestamp": record.event_timestamp,
                    "article_text_type": snapshot.article_text_types[record.article_id],
                    "requested_model": record.requested_model,
                    "model": record.model,
                    "classification_stage": record.stage,
                    "escalation_reasons": record.escalation_reasons,
                    "prompt_version": record.prompt_version,
                    "schema_version": record.schema_version,
                    "cache_key": record.cache_key,
                    "input_hash": record.input_hash,
                    "output_hash": record.output_hash,
                    "classified_at": record.classified_at,
                    "response_id": record.response_id,
                    "batch_id": record.batch_id,
                    "batch_custom_id": record.batch_custom_id,
                    "input_tokens": record.usage.input_tokens,
                    "cached_input_tokens": record.usage.cached_input_tokens,
                    "output_tokens": record.usage.output_tokens,
                    "reasoning_tokens": record.usage.reasoning_tokens,
                    "estimated_cost_usd": record.usage.estimated_cost_usd,
                }
            )
            assessment_rows.append(row)
        assessments_path = store.write_parquet(
            pl.DataFrame(assessment_rows, infer_schema_length=None),
            output / "assessments.parquet",
        )
        ledger_path = store.write_parquet(
            _frame_from_models(classification_run.ledger_entries),
            output / "classification_ledger.parquet",
        )
        events_path = store.write_parquet(events, output / "events.parquet")
        metrics_path = store.write_json(metrics, output / "metrics.json")
        classification_summary = classification_run.summary()
        report_path = build_milestone_report(
            events,
            metrics,
            output_path=output / "report.html",
            experiment_id=experiment_id,
            ticker=self.experiment.ticker,
            horizons=self.experiment.horizons,
            generated_at=started.isoformat(),
            filtering=snapshot.filter_report,
            classification=classification_summary,
            budget_limit_usd=budget_limit_usd,
            spending_limit_tier=self.experiment.spending_limit_tier,
        )
        returned_models = sorted(
            {entry.response_model for entry in classification_run.ledger_entries}
        )
        artifacts = {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in [
                articles_path,
                assessments_path,
                ledger_path,
                events_path,
                metrics_path,
                report_path,
            ]
        }
        manifest = {
            "experiment_id": experiment_id,
            "created_at": started.isoformat(),
            "status": "complete",
            "git": _git_state(),
            "config_hash": config_hash,
            "experiment_config": self.experiment.model_dump(mode="json"),
            "data_snapshot_id": snapshot.snapshot_id,
            "data_snapshot_hashes": {
                "articles": file_sha256(snapshot.articles_path),
                "prices": file_sha256(snapshot.prices_path),
            },
            "provider_api": {
                "eodhd": {
                    "version": "unversioned",
                    "news_endpoint": "/api/news",
                    "price_endpoint": f"/api/eod/{self.experiment.ticker}",
                },
                "openai": {
                    "surface": "Batch API over /v1/responses with Structured Outputs",
                    "completion_window": "24h",
                },
            },
            "software_versions": {
                "python": platform.python_version(),
                "sentiment_lab": _package_version("sentiment-lab"),
                "openai_sdk": _package_version("openai"),
                "polars": _package_version("polars"),
                "duckdb": _package_version("duckdb"),
            },
            "openai_models_requested": {
                "first_pass": self.app_config.openai.first_pass_model,
                "escalation": self.app_config.openai.escalation_model,
            },
            "openai_models_returned": returned_models,
            "prompt_version": classifications[0].prompt_version,
            "schema_version": classifications[0].schema_version,
            "classification_cache": {
                "hits": classification_summary["cache_hits"],
                "api_attempts": sum(
                    entry.outcome in {"api_success", "api_failure"}
                    for entry in classification_run.ledger_entries
                ),
            },
            "filtering": snapshot.filter_report,
            "classification": classification_summary,
            "batch_executions": [
                {
                    "stage": execution.stage,
                    "requested_model": execution.requested_model,
                    "batch_id": execution.batch_id,
                    "input_file_id": execution.input_file_id,
                    "output_file_id": execution.output_file_id,
                    "requests": len(execution.calls) + len(execution.failures),
                    "failed_requests": len(execution.failures),
                    "maximum_estimated_cost_usd": execution.maximum_estimated_cost_usd,
                }
                for execution in classification_run.batch_executions
            ],
            "cost_control": {
                "spending_limit_tier": self.experiment.spending_limit_tier,
                "spending_limit_usd": budget_limit_usd,
                "preflight_maximum_estimates_usd": [
                    execution.maximum_estimated_cost_usd
                    for execution in classification_run.batch_executions
                ],
                "actual_estimated_cost_usd": classification_run.current_run_cost_usd,
                "within_budget": classification_run.current_run_cost_usd <= budget_limit_usd,
                "pricing_source_url": self.app_config.openai.pricing_source_url,
                "pricing_as_of": self.app_config.openai.pricing_as_of.isoformat(),
                "regional_processing_multiplier": (
                    self.app_config.openai.regional_processing_multiplier
                ),
            },
            "token_usage": _ledger_usage(classification_run.ledger_entries, current_run_only=True),
            "classification_ledger_usage": _ledger_usage(
                classification_run.ledger_entries, current_run_only=False
            ),
            "metrics": metrics,
            "artifacts": artifacts,
        }
        store.write_json(manifest, output / "manifest.json")
        store.register_parquet_view("milestone_events_latest", events_path)
        store.register_parquet_view("milestone_assessments_latest", assessments_path)
        store.register_parquet_view("milestone_classification_ledger_latest", ledger_path)
        return output
