"""One focused experiment: EODHD article → OpenAI assessment → future return."""

from __future__ import annotations

import hashlib
import platform
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
from sentiment_lab.nlp.classifier import ArticleClassifier
from sentiment_lab.nlp.schemas import ClassificationRecord
from sentiment_lab.reporting.report import build_milestone_report

NEW_YORK = ZoneInfo("America/New_York")


def _usage_totals(records: list[ClassificationRecord]) -> dict[str, int | float | None]:
    costs = [record.usage.estimated_cost_usd for record in records]
    if not records:
        estimated_cost: float | None = 0.0
    elif any(cost is None for cost in costs):
        estimated_cost = None
    else:
        estimated_cost = sum(cost for cost in costs if cost is not None)
    return {
        "input_tokens": sum(record.usage.input_tokens for record in records),
        "output_tokens": sum(record.usage.output_tokens for record in records),
        "estimated_cost_usd": estimated_cost,
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


def _frame_from_models(models: list[Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [model.model_dump(mode="python") for model in models], infer_schema_length=None
    )


def sync_milestone_data(
    experiment: ExperimentConfig,
    app_config: AppConfig,
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
    eligible = [
        article
        for article in requested
        if article.content.strip() and experiment.ticker in article.symbols
    ]
    by_day: dict[date, list[NewsArticle]] = {}
    for article in sorted(eligible, key=lambda item: (item.provider_timestamp, item.article_id)):
        local_day = article.provider_timestamp.astimezone(NEW_YORK).date()
        by_day.setdefault(local_day, []).append(article)
    articles: list[NewsArticle] = []
    for daily_index in range(experiment.max_articles_per_day):
        for local_day in sorted(by_day):
            daily_articles = by_day[local_day]
            if daily_index < len(daily_articles):
                articles.append(daily_articles[daily_index])
                if len(articles) >= experiment.max_articles:
                    break
        if len(articles) >= experiment.max_articles:
            break
    if not articles:
        raise RuntimeError(
            "EODHD returned no full-text articles directly mapped to the requested ticker."
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
    }
    snapshot_id = hashlib.sha256(stable_json(snapshot_material).encode("utf-8")).hexdigest()[:16]
    root = app_config.storage.data_root / "normalized" / "milestone" / snapshot_id
    store = ArtifactStore(app_config.storage.data_root, app_config.storage.duckdb_path)
    articles_path = store.write_parquet(_frame_from_models(articles), root / "articles.parquet")
    prices_path = store.write_parquet(_frame_from_models(prices), root / "prices.parquet")
    store.register_parquet_view("milestone_articles_latest", articles_path)
    store.register_parquet_view("milestone_prices_latest", prices_path)
    return DataSnapshot(
        snapshot_id=snapshot_id,
        articles=articles,
        prices=prices,
        articles_path=articles_path,
        prices_path=prices_path,
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
        force_classify: bool = False,
    ) -> Path:
        started = datetime.now(UTC)
        snapshot = sync_milestone_data(
            self.experiment,
            self.app_config,
            self.eodhd_client,
            refresh=refresh,
        )
        classifications = self.classifier.classify_many(
            snapshot.articles,
            ticker=self.experiment.ticker,
            company_name=self.experiment.company_name,
            prompt_variant=self.experiment.prompt_variant,
            force=force_classify,
        )
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
            _frame_from_models(snapshot.articles), output / "articles.parquet"
        )
        assessment_rows = []
        for record in classifications:
            row = record.assessment.model_dump(mode="python")
            row.update(
                {
                    "model": record.model,
                    "prompt_version": record.prompt_version,
                    "schema_version": record.schema_version,
                    "cache_key": record.cache_key,
                    "input_hash": record.input_hash,
                    "output_hash": record.output_hash,
                    "classified_at": record.classified_at,
                    "response_id": record.response_id,
                    "input_tokens": record.usage.input_tokens,
                    "output_tokens": record.usage.output_tokens,
                    "estimated_cost_usd": record.usage.estimated_cost_usd,
                }
            )
            assessment_rows.append(row)
        assessments_path = store.write_parquet(
            pl.DataFrame(assessment_rows, infer_schema_length=None),
            output / "assessments.parquet",
        )
        events_path = store.write_parquet(events, output / "events.parquet")
        metrics_path = store.write_json(metrics, output / "metrics.json")
        report_path = build_milestone_report(
            events,
            metrics,
            output_path=output / "report.html",
            experiment_id=experiment_id,
            ticker=self.experiment.ticker,
            horizons=self.experiment.horizons,
            generated_at=started.isoformat(),
        )
        cache_hits = sum(record.from_cache for record in classifications)
        run_model_calls = [record for record in classifications if not record.from_cache]
        returned_models = sorted({record.model for record in classifications})
        artifacts = {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in [articles_path, assessments_path, events_path, metrics_path, report_path]
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
                "openai": {"surface": "Responses API Structured Outputs"},
            },
            "software_versions": {
                "python": platform.python_version(),
                "sentiment_lab": _package_version("sentiment-lab"),
                "openai_sdk": _package_version("openai"),
                "polars": _package_version("polars"),
                "duckdb": _package_version("duckdb"),
            },
            "openai_model_requested": self.classifier.model_client.model,
            "openai_models_returned": returned_models,
            "prompt_version": classifications[0].prompt_version,
            "schema_version": classifications[0].schema_version,
            "classification_cache": {
                "hits": cache_hits,
                "misses": len(classifications) - cache_hits,
            },
            "token_usage": _usage_totals(run_model_calls),
            "classification_ledger_usage": _usage_totals(classifications),
            "metrics": metrics,
            "artifacts": artifacts,
        }
        store.write_json(manifest, output / "manifest.json")
        store.register_parquet_view("milestone_events_latest", events_path)
        store.register_parquet_view("milestone_assessments_latest", assessments_path)
        return output
