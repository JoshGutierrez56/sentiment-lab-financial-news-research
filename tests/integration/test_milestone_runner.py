"""Mocked end-to-end article -> assessment -> return artifact test."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from conftest import make_article, make_price, make_record
from sentiment_lab.config.models import (
    AppConfig,
    ExperimentConfig,
    SentimentConfig,
    StorageConfig,
)
from sentiment_lab.data.storage import file_sha256
from sentiment_lab.experiments.runner import MilestoneRunner, sync_milestone_data
from sentiment_lab.nlp.classifier import ClassificationRun
from sentiment_lab.nlp.schemas import ClassificationLedgerEntry, SentimentLabel


class FakeEODHD:
    def __init__(self, articles: list[object], prices: list[object]) -> None:
        self.articles = articles
        self.prices = prices
        self.news_requests: list[dict[str, object]] = []
        self.price_requests: list[dict[str, object]] = []

    def fetch_news(self, ticker: str, start: date, end: date, **kwargs: object) -> list[object]:
        self.news_requests.append({"ticker": ticker, "start": start, "end": end, **kwargs})
        return self.articles

    def fetch_eod_prices(
        self, ticker: str, start: date, end: date, **kwargs: object
    ) -> list[object]:
        self.price_requests.append({"ticker": ticker, "start": start, "end": end, **kwargs})
        return self.prices


class FakeClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def classify_many(self, articles: list[object], **_: object) -> ClassificationRun:
        self.calls += 1
        labels = [
            (SentimentLabel.bullish, 0.8),
            (SentimentLabel.neutral, 0.0),
            (SentimentLabel.bearish, -0.7),
        ]
        records = [
            make_record(article, label=label, score=score).model_copy(
                update={
                    "cache_key": article.article_id,
                    "from_cache": self.calls > 1,
                }
            )
            for article, (label, score) in zip(articles, labels, strict=True)
        ]
        ledger = [
            ClassificationLedgerEntry(
                article_id=record.article_id,
                ticker=record.ticker,
                event_timestamp=record.event_timestamp,
                cache_key=record.cache_key,
                input_hash=record.input_hash,
                requested_model=record.requested_model,
                response_model=record.model,
                prompt_version=record.prompt_version,
                schema_version=record.schema_version,
                stage=record.stage,
                outcome="cache_hit" if record.from_cache else "api_success",
                response_id=record.response_id,
                batch_id=record.batch_id,
                batch_custom_id=record.batch_custom_id,
                input_tokens=record.usage.input_tokens,
                cached_input_tokens=record.usage.cached_input_tokens,
                output_tokens=record.usage.output_tokens,
                reasoning_tokens=record.usage.reasoning_tokens,
                estimated_cost_usd=record.usage.estimated_cost_usd,
                run_cost_usd=0.0 if record.from_cache else record.usage.estimated_cost_usd,
            )
            for record in records
        ]
        return ClassificationRun(
            final_records=records,
            ledger_entries=ledger,
            batch_executions=[],
        )


def _setup(tmp_path: Path) -> tuple[ExperimentConfig, AppConfig, list[object], list[object]]:
    experiment = ExperimentConfig(
        name="integration_milestone",
        ticker="AAPL.US",
        company_name="Apple Inc.",
        news_start=date(2026, 5, 1),
        news_end=date(2026, 5, 3),
        max_articles=3,
        news_candidate_pool=9,
        horizons=[1, 3],
    )
    app_config = AppConfig(
        storage=StorageConfig(
            data_root=tmp_path / "data",
            duckdb_path=tmp_path / "data" / "research.duckdb",
        )
    )
    articles = [
        make_article(
            article_id=str(index) * 64,
            timestamp=datetime(2026, 5, index, 21, 0, tzinfo=UTC),
            title=f"Article {index}",
            content=f"Apple company-specific full-text development number {index}. " * 2,
        )
        for index in (1, 2, 3)
    ]
    prices = [
        make_price(date(2026, 5, 4), open_=100, close=102),
        make_price(date(2026, 5, 5), open_=102, close=101),
        make_price(date(2026, 5, 6), open_=101, close=105),
    ]
    return experiment, app_config, articles, prices


def test_runner_emits_auditable_reproducible_milestone_artifacts(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    provider = FakeEODHD(articles, prices)
    classifier = FakeClassifier()
    runner = MilestoneRunner(
        experiment,
        app_config,
        SentimentConfig(minimum_article_characters=50),
        provider,  # type: ignore[arg-type]
        classifier,  # type: ignore[arg-type]
    )
    first_output = runner.run()
    second_output = runner.run()
    required = {
        "articles.parquet",
        "assessments.parquet",
        "classification_ledger.parquet",
        "events.parquet",
        "metrics.json",
        "manifest.json",
        "report.html",
    }
    assert {path.name for path in first_output.iterdir()} == required
    first_events = pl.read_parquet(first_output / "events.parquet")
    second_events = pl.read_parquet(second_output / "events.parquet")
    assert first_events.equals(second_events)
    assert file_sha256(first_output / "events.parquet") == file_sha256(
        second_output / "events.parquet"
    )
    for artifact in ("articles.parquet", "assessments.parquet", "events.parquet", "metrics.json"):
        assert file_sha256(first_output / artifact) == file_sha256(second_output / artifact)
    assert first_events["article_text"].str.len_chars().min() > 0
    assert first_events["reasoning"].str.len_chars().min() > 0
    assert first_events["entry_date"].unique().to_list() == [date(2026, 5, 4)]
    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["data_snapshot_id"]
    assert manifest["config_hash"]
    assert manifest["token_usage"]["input_tokens"] == 300
    assert manifest["classification_ledger_usage"]["input_tokens"] == 300
    assert manifest["classification_cache"] == {"hits": 0, "api_attempts": 3}
    assert manifest["openai_models_requested"] == {
        "first_pass": "gpt-5.4-mini",
        "escalation": "gpt-5.4",
    }
    assert manifest["openai_models_returned"] == ["test-model"]
    assert manifest["provider_api"]["openai"]["surface"] == (
        "Batch API over /v1/responses with Structured Outputs"
    )
    assert manifest["software_versions"]["python"]
    second_manifest = json.loads((second_output / "manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["token_usage"] == {
        "cached_input_tokens": 0,
        "current_run_cost_usd": 0.0,
        "estimated_cost_usd": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    assert second_manifest["classification_ledger_usage"]["input_tokens"] == 300
    assert second_manifest["classification_cache"] == {"hits": 3, "api_attempts": 0}
    assert "EODHD → ChatGPT → Future Returns" in (first_output / "report.html").read_text(
        encoding="utf-8"
    )
    with duckdb.connect(str(app_config.storage.duckdb_path)) as connection:
        assert connection.execute("select count(*) from milestone_events_latest").fetchone() == (3,)
    assert classifier.calls == 2
    stored_articles = pl.read_parquet(first_output / "articles.parquet")
    stored_assessments = pl.read_parquet(first_output / "assessments.parquet")
    assert stored_articles.schema["provider_timestamp"] == pl.Datetime("us", "UTC")
    assert stored_assessments.schema["event_timestamp"] == pl.Datetime("us", "UTC")
    assert stored_assessments["input_hash"].str.len_chars().unique().to_list() == [64]
    assert stored_assessments["output_hash"].str.len_chars().unique().to_list() == [64]
    assert "from_cache" not in stored_assessments.columns


def test_sync_filters_to_full_text_direct_ticker_mapping(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    empty = articles[0].model_copy(update={"article_id": "e" * 64, "content": " "})
    wrong = articles[1].model_copy(update={"article_id": "w" * 64, "symbols": ["MSFT.US"]})
    provider = FakeEODHD([empty, wrong, articles[2]], prices)
    snapshot = sync_milestone_data(
        experiment,
        app_config,
        SentimentConfig(minimum_article_characters=50),
        provider,  # type: ignore[arg-type]
    )
    assert [article.article_id for article in snapshot.articles] == [articles[2].article_id]
    assert snapshot.articles_path.is_file()
    assert snapshot.prices_path.is_file()
    assert pl.read_parquet(snapshot.articles_path).schema["provider_timestamp"] == pl.Datetime(
        "us", "UTC"
    )
    assert pl.read_parquet(snapshot.prices_path).schema["date"] == pl.Date
    assert provider.news_requests[0]["max_articles"] == 9
    assert provider.price_requests[0]["end"] > experiment.news_end


def test_sync_accounts_for_every_pre_openai_filter(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    good = articles[0]
    outside = good.model_copy(
        update={
            "article_id": "o" * 64,
            "provider_timestamp": datetime(2026, 4, 30, 12, 0, tzinfo=UTC),
        }
    )
    duplicate = good.model_copy(
        update={
            "article_id": "d" * 64,
            "provider_timestamp": datetime(2026, 5, 1, 22, 0, tzinfo=UTC),
            "link": "https://example.com/duplicate",
        }
    )
    wrong_ticker = articles[1].model_copy(update={"article_id": "w" * 64, "symbols": ["MSFT.US"]})
    inadequate = articles[1].model_copy(
        update={"article_id": "i" * 64, "title": "Tiny headline", "content": "Tiny."}
    )
    market_summary = articles[2].model_copy(
        update={
            "article_id": "m" * 64,
            "title": "Market roundup: stocks rally today",
            "content": "A broad recap mentions many unrelated companies and indices. " * 2,
            "symbols": ["AAPL.US", "MSFT.US", "NVDA.US", "AMZN.US", "META.US"],
        }
    )
    provider = FakeEODHD(
        [outside, good, duplicate, wrong_ticker, inadequate, market_summary], prices
    )
    snapshot = sync_milestone_data(
        experiment,
        app_config,
        SentimentConfig(minimum_article_characters=50),
        provider,  # type: ignore[arg-type]
    )

    assert [article.article_id for article in snapshot.articles] == [good.article_id]
    assert snapshot.filter_report == {
        "total_articles_considered": 6,
        "articles_filtered_before_openai": 5,
        "filtered_by_reason": {
            "outside_sample": 1,
            "low_confidence_ticker_mapping": 1,
            "inadequate_text": 1,
            "irrelevant_market_summary": 1,
            "duplicate_story": 1,
            "sample_limit": 0,
        },
        "eligible_full_text": 1,
        "eligible_headline_only": 0,
        "selected_full_text": 1,
        "selected_headline_only": 0,
        "ticker_mapping_method": "eodhd_direct_symbol",
        "ticker_mapping_confidence": 1.0,
    }


def test_full_text_has_priority_over_earlier_headline_only_articles(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    experiment = experiment.model_copy(update={"max_articles": 2})
    headline = articles[0].model_copy(
        update={
            "article_id": "h" * 64,
            "provider_timestamp": datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
            "title": "Apple headline only",
            "content": "Apple headline only",
        }
    )
    provider = FakeEODHD([headline, articles[0], articles[1]], prices)
    snapshot = sync_milestone_data(
        experiment,
        app_config,
        SentimentConfig(
            minimum_article_characters=50,
            classify_headline_only=True,
        ),
        provider,  # type: ignore[arg-type]
    )

    assert [article.article_id for article in snapshot.articles] == [
        articles[0].article_id,
        articles[1].article_id,
    ]
    assert snapshot.filter_report["eligible_full_text"] == 2
    assert snapshot.filter_report["eligible_headline_only"] == 1
    assert snapshot.filter_report["selected_full_text"] == 2
    assert snapshot.filter_report["selected_headline_only"] == 0


def test_sync_reports_empty_article_and_price_samples(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    with pytest.raises(RuntimeError, match="no eligible articles"):
        sync_milestone_data(
            experiment,
            app_config,
            SentimentConfig(minimum_article_characters=50),
            FakeEODHD([], prices),  # type: ignore[arg-type]
        )
    with pytest.raises(RuntimeError, match="no EOD prices"):
        sync_milestone_data(
            experiment,
            app_config,
            SentimentConfig(minimum_article_characters=50),
            FakeEODHD(articles, []),  # type: ignore[arg-type]
        )
