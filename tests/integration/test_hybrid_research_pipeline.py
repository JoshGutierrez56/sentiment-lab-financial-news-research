from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.hybrid import analysis
from sentiment_lab.hybrid.analysis import PredictionAnalysisConfig, run_prediction_analysis
from sentiment_lab.hybrid.baselines import BaselineConfig, run_baselines
from sentiment_lab.hybrid.openai_calibration import (
    AdditionalCalibrationConfig,
    freeze_additional_openai_sample,
)
from sentiment_lab.hybrid.portfolio import PortfolioRunConfig, run_portfolio_backtests


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    tickers = [f"T{index:03d}.US" for index in range(125)]
    sectors = [f"Sector {index}" for index in range(11)]
    start = datetime(2022, 1, 1, 12, tzinfo=UTC)
    article_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, str]] = []
    labels = ("bullish", "neutral", "bearish")
    scores = (0.6, 0.0, -0.6)
    for index in range(5000):
        ticker = tickers[index % len(tickers)]
        timestamp = start + timedelta(hours=index)
        entry_date = date(2022, 2, 1) + timedelta(days=index % 80)
        label = labels[index % 3]
        score = scores[index % 3]
        article_id = f"article-{index:05d}"
        article_rows.append(
            {
                "article_id": article_id,
                "provider": "eodhd",
                "provider_timestamp": timestamp,
                "retrieved_at": timestamp,
                "title": f"Company {ticker} reports strong earnings {index}",
                "content": "Company reported revenue, profit and operating results. " * 10,
                "link": f"https://example.test/{index}",
                "symbols": [ticker],
                "tags": ["earnings"],
                "provider_sentiment_polarity": score / 2,
                "raw_response_hash": "a" * 64,
                "ticker": ticker,
                "company_name": f"Company {ticker}",
                "sector": sectors[index % len(sectors)],
                "article_content_hash": f"{index:064x}",
                "story_cluster_id": f"cluster-{index}",
                "pre_inference_event_candidates": ["earnings"],
                "entry_date": entry_date,
                **{
                    f"future_return_{horizon}d": score * horizon / 10_000 + 0.001
                    for horizon in (1, 3, 5, 10, 21, 63)
                },
            }
        )
        classification_rows.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "sentiment_score": score,
                "sentiment_label": label,
                "confidence": 0.9 if index % 4 else 0.4,
                "relevance": 0.9,
                "materiality": 0.8,
                "novelty": 0.7,
                "event_type": "other" if index % 10 == 0 else "earnings",
                "expected_horizon": "5d",
                "tradable": True,
                "abstain": False,
                "abstain_reason": None,
                "concise_reasoning": "Material company result.",
            }
        )
        split = "development" if index < 3000 else ("validation" if index < 4000 else "holdout")
        split_rows.append({"article_id": article_id, "research_split": split})
    price_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for offset in range(160):
            close = 100.0 + offset / 10
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": date(2022, 1, 1) + timedelta(days=offset),
                    "open": close - 0.1,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000_000,
                }
            )
    paths = {
        "articles": tmp_path / "articles.parquet",
        "classifications": tmp_path / "classifications.parquet",
        "splits": tmp_path / "splits.parquet",
        "prices": tmp_path / "prices.parquet",
        "original": tmp_path / "original.parquet",
    }
    pl.DataFrame(article_rows, infer_schema_length=None).write_parquet(paths["articles"])
    pl.DataFrame(classification_rows, infer_schema_length=None).write_parquet(
        paths["classifications"]
    )
    pl.DataFrame(split_rows).write_parquet(paths["splits"])
    pl.DataFrame(price_rows).write_parquet(paths["prices"])
    pl.DataFrame({"article_id": [f"article-{index:05d}" for index in range(250)]}).write_parquet(
        paths["original"]
    )
    return paths


def test_prediction_orchestration_is_split_locked_and_return_artifact_is_complete(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = _artifacts(tmp_path)
    monkeypatch.setattr(
        analysis,
        "_horizon_metrics",
        lambda frame, **_: {"n": frame.height, "stubbed_integration_statistic": True},
    )
    config = PredictionAnalysisConfig(
        name="development_only",
        articles_path=paths["articles"],
        classifications_path=paths["classifications"],
        splits_path=paths["splits"],
        expected_sample_hash="a" * 64,
        expected_articles_sha256=file_sha256(paths["articles"]),
        expected_classifications_sha256=file_sha256(paths["classifications"]),
        expected_splits_sha256=file_sha256(paths["splits"]),
        included_splits=["development"],
        bootstrap_samples=100,
    )
    metrics_path, events_path = run_prediction_analysis(
        config, data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
    )
    assert pl.read_parquet(events_path).height == 3000
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["included_splits"] == ["development"]
    assert "holdout" not in metrics["event_level"]


def test_baselines_portfolio_and_additional_selection_run_on_frozen_inputs(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    hashes = {name: file_sha256(path) for name, path in paths.items() if name != "original"}
    baseline_path = run_baselines(
        BaselineConfig(
            name="baseline_test",
            articles_path=paths["articles"],
            classifications_path=paths["classifications"],
            splits_path=paths["splits"],
            prices_path=paths["prices"],
            expected_hashes=hashes,
            evaluation_splits=["development", "validation"],
        ),
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["splits"]["validation"]["5d"]["keyword_sentiment"]["n"] == 1000

    portfolio_path = run_portfolio_backtests(
        PortfolioRunConfig(
            name="portfolio_test",
            articles_path=paths["articles"],
            classifications_path=paths["classifications"],
            splits_path=paths["splits"],
            prices_path=paths["prices"],
            expected_hashes=hashes,
            evaluation_splits=["development"],
        ),
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
    )
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    assert "sharpe" in portfolio["splits"]["development"]["5d"]["long_only"]["base_net"]

    sample = freeze_additional_openai_sample(
        AdditionalCalibrationConfig(
            name="additional_test",
            articles_path=paths["articles"],
            local_classifications_path=paths["classifications"],
            splits_path=paths["splits"],
            original_calibration_path=paths["original"],
            expected_articles_sha256=file_sha256(paths["articles"]),
            expected_local_sha256=file_sha256(paths["classifications"]),
            expected_splits_sha256=file_sha256(paths["splits"]),
            expected_original_calibration_sha256=file_sha256(paths["original"]),
            maximum_articles=50,
        ),
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
    )
    selected = pl.read_parquet(sample.sample_path)
    assert selected.height == 50
    assert not set(selected["article_id"]) & {
        f"article-{index:05d}" for index in range(250)
    }
