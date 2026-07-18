"""Diversified frozen-sample and validation-report regression tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from conftest import make_article, make_price, make_record
from sentiment_lab.config.models import (
    AppConfig,
    SentimentConfig,
    StorageConfig,
    ValidationExperimentConfig,
    ValidationUniverseMember,
)
from sentiment_lab.experiments.validation import (
    ValidationRunner,
    sampling_event_bucket,
    sync_validation_data,
)
from sentiment_lab.nlp.classifier import ClassificationRun, ClassificationTarget
from sentiment_lab.nlp.schemas import SentimentLabel


class FakeValidationProvider:
    def __init__(self, articles: dict[str, list[object]], prices: dict[str, list[object]]) -> None:
        self.articles = articles
        self.prices = prices

    def fetch_news(self, ticker: str, *_: object, **__: object) -> list[object]:
        return self.articles[ticker]

    def fetch_eod_prices(self, ticker: str, *_: object, **__: object) -> list[object]:
        return self.prices[ticker]


class FakeTargetClassifier:
    def classify_targets(
        self,
        targets: list[ClassificationTarget],
        **_: object,
    ) -> ClassificationRun:
        records = []
        for index, target in enumerate(targets):
            label, score = (
                (SentimentLabel.bullish, 0.8) if index % 2 == 0 else (SentimentLabel.bearish, -0.7)
            )
            record = make_record(target.article, label=label, score=score).model_copy(
                update={"ticker": target.ticker, "cache_key": f"{index:064x}"}
            )
            records.append(record)
        return ClassificationRun(
            final_records=records,
            first_pass_records=list(records),
            ledger_entries=[],
            batch_executions=[],
        )


def _config(*, frozen_snapshot_id: str | None = None) -> ValidationExperimentConfig:
    return ValidationExperimentConfig(
        name="test_validation",
        news_start=date(2026, 1, 1),
        news_end=date(2026, 2, 10),
        universe=[
            ValidationUniverseMember(
                ticker="AAPL.US", company_name="Apple Inc.", sector="Technology"
            ),
            ValidationUniverseMember(
                ticker="JPM.US", company_name="JPMorgan Chase", sector="Financials"
            ),
        ],
        articles_per_company=2,
        max_articles=4,
        news_candidate_pool_per_company=10,
        horizons=[1, 21],
        minimum_months=2,
        minimum_sectors=2,
        minimum_event_buckets=2,
        bootstrap_samples=100,
        frozen_snapshot_id=frozen_snapshot_id,
    )


def _provider() -> FakeValidationProvider:
    apple = [
        make_article(
            article_id="1" * 64,
            timestamp=datetime(2026, 1, 5, 15, tzinfo=UTC),
            title="Apple reports earnings beat",
            content="Apple reported quarterly earnings and revenue above estimates. " * 5,
        ),
        make_article(
            article_id="2" * 64,
            timestamp=datetime(2026, 2, 5, 15, tzinfo=UTC),
            title="Apple launches product",
            content="Apple launched a new company-specific product for customers. " * 5,
        ),
        make_article(
            article_id="3" * 64,
            timestamp=datetime(2026, 1, 7, 15, tzinfo=UTC),
            title="Apple headline only",
            content="short",
        ),
    ]
    jpm = [
        make_article(
            article_id="4" * 64,
            timestamp=datetime(2026, 1, 8, 15, tzinfo=UTC),
            title="JPMorgan analyst upgrades shares",
            content="An analyst upgraded JPMorgan after reviewing company fundamentals. " * 5,
        ).model_copy(update={"symbols": ["JPM.US"]}),
        make_article(
            article_id="5" * 64,
            timestamp=datetime(2026, 2, 8, 15, tzinfo=UTC),
            title="JPMorgan announces dividend",
            content="JPMorgan announced a dividend and capital allocation update. " * 5,
        ).model_copy(update={"symbols": ["JPM.US"]}),
    ]
    start = date(2026, 1, 2)
    prices = [
        make_price(start + timedelta(days=index), open_=100 + index, close=101 + index)
        for index in range(90)
    ]
    return FakeValidationProvider(
        {"AAPL.US": apple, "JPM.US": jpm},
        {"AAPL.US": prices, "JPM.US": prices},
    )


def _app(tmp_path: Path) -> AppConfig:
    return AppConfig(
        storage=StorageConfig(
            data_root=tmp_path / "data",
            duckdb_path=tmp_path / "data" / "research.duckdb",
        )
    )


def test_sampling_event_bucket_is_deterministic() -> None:
    assert sampling_event_bucket(make_article(title="Company earnings report")) == "earnings"
    assert (
        sampling_event_bucket(
            make_article(title="Court lawsuit filed", content="A court lawsuit was filed.")
        )
        == "litigation"
    )
    assert (
        sampling_event_bucket(
            make_article(title="Uncategorized development", content="Company update.")
        )
        == "other"
    )


def test_sync_freezes_balanced_complete_sample(tmp_path: Path) -> None:
    snapshot = sync_validation_data(
        _config(),
        _app(tmp_path),
        SentimentConfig(minimum_article_characters=50),
        _provider(),  # type: ignore[arg-type]
    )
    assert len(snapshot.sampled) == 4
    assert snapshot.snapshot_id
    assert snapshot.filter_report["selected_full_text"] == 4
    assert snapshot.filter_report["selected_headline_only"] == 0
    assert snapshot.filter_report["filtered_by_reason"]["headline_only_or_low_text"] == 1
    assert pl.read_parquet(snapshot.articles_path)["ticker"].n_unique() == 2
    assert pl.read_parquet(snapshot.prices_path)["ticker"].n_unique() == 2

    verified = sync_validation_data(
        _config(frozen_snapshot_id=snapshot.snapshot_id),
        _app(tmp_path),
        SentimentConfig(minimum_article_characters=50),
        _provider(),  # type: ignore[arg-type]
    )
    assert verified.snapshot_id == snapshot.snapshot_id


def test_sync_rejects_snapshot_drift_and_underfilled_ticker(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not match frozen_snapshot_id"):
        sync_validation_data(
            _config(frozen_snapshot_id="0" * 16),
            _app(tmp_path),
            SentimentConfig(minimum_article_characters=50),
            _provider(),  # type: ignore[arg-type]
        )
    provider = _provider()
    provider.articles["JPM.US"] = provider.articles["JPM.US"][:1]
    with pytest.raises(RuntimeError, match="eligible full-text articles"):
        sync_validation_data(
            _config(),
            _app(tmp_path),
            SentimentConfig(minimum_article_characters=50),
            provider,  # type: ignore[arg-type]
        )


def test_validation_runner_reports_requested_metrics_without_sharpe(tmp_path: Path) -> None:
    initial = sync_validation_data(
        _config(),
        _app(tmp_path),
        SentimentConfig(minimum_article_characters=50),
        _provider(),  # type: ignore[arg-type]
    )
    runner = ValidationRunner(
        _config(frozen_snapshot_id=initial.snapshot_id),
        _app(tmp_path),
        SentimentConfig(minimum_article_characters=50),
        _provider(),  # type: ignore[arg-type]
        FakeTargetClassifier(),  # type: ignore[arg-type]
    )
    output = runner.run()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["valid_classification_count"] == 4
    assert metrics["article_first_pass_cache_hits"] == 0
    assert metrics["stage_cache_hits"] == 0
    assert metrics["label_counts"] == {"bearish": 2, "bullish": 2, "neutral": 0}
    assert set(metrics["horizons"]) == {"1d", "21d"}
    assert len(metrics["by_ticker"]) == 2
    assert metrics["horizons"]["1d"]["company_cluster_bootstrap_95_ci"]["lower_95"] is not None
    assert manifest["frozen_snapshot_verified"] is True
    assert manifest["decision"] in {"PROCEED", "REVISE", "STOP"}
    assert manifest["cost_control"]["spending_limit_usd"] == 2.0
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "No portfolio or Sharpe ratio was constructed" in report
    assert not (output / "portfolio.parquet").exists()
