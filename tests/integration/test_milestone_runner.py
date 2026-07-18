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
from sentiment_lab.nlp.schemas import SentimentLabel


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

    def classify_many(self, articles: list[object], **_: object) -> list[object]:
        self.calls += 1
        labels = [
            (SentimentLabel.bullish, 0.8),
            (SentimentLabel.neutral, 0.0),
            (SentimentLabel.bearish, -0.7),
        ]
        return [
            make_record(article, label=label, score=score).model_copy(
                update={"cache_key": article.article_id}
            )
            for article, (label, score) in zip(articles, labels, strict=True)
        ]


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
        SentimentConfig(),
        provider,  # type: ignore[arg-type]
        classifier,  # type: ignore[arg-type]
    )
    first_output = runner.run()
    second_output = runner.run()
    required = {
        "articles.parquet",
        "assessments.parquet",
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
    assert first_events["article_text"].str.len_chars().min() > 0
    assert first_events["reasoning"].str.len_chars().min() > 0
    assert first_events["entry_date"].unique().to_list() == [date(2026, 5, 4)]
    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["data_snapshot_id"]
    assert manifest["config_hash"]
    assert manifest["token_usage"]["input_tokens"] == 300
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


def test_sync_filters_to_full_text_direct_ticker_mapping(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    empty = articles[0].model_copy(update={"article_id": "e" * 64, "content": " "})
    wrong = articles[1].model_copy(update={"article_id": "w" * 64, "symbols": ["MSFT.US"]})
    provider = FakeEODHD([empty, wrong, articles[2]], prices)
    snapshot = sync_milestone_data(
        experiment,
        app_config,
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


def test_sync_reports_empty_article_and_price_samples(tmp_path: Path) -> None:
    experiment, app_config, articles, prices = _setup(tmp_path)
    with pytest.raises(RuntimeError, match="no full-text articles"):
        sync_milestone_data(
            experiment,
            app_config,
            FakeEODHD([], prices),  # type: ignore[arg-type]
        )
    with pytest.raises(RuntimeError, match="no EOD prices"):
        sync_milestone_data(
            experiment,
            app_config,
            FakeEODHD(articles, []),  # type: ignore[arg-type]
        )
