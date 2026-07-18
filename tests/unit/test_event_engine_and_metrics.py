"""Dedicated leakage/timestamp alignment and basic inference tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from conftest import make_article, make_price, make_record
from sentiment_lab.backtest.event_engine import align_events
from sentiment_lab.backtest.metrics import compute_event_metrics


def test_conservative_alignment_skips_publication_day_and_weekend() -> None:
    article = make_article(timestamp=datetime(2026, 5, 1, 21, 0, tzinfo=UTC))  # Friday 5pm ET
    prices = [
        make_price(date(2026, 5, 1), open_=90, close=95),
        make_price(date(2026, 5, 4), open_=100, close=102),
        make_price(date(2026, 5, 5), open_=102, close=105),
        make_price(date(2026, 5, 6), open_=105, close=110),
    ]
    events = align_events([article], [make_record(article)], prices, horizons=[1, 3])
    row = events.row(0, named=True)
    assert row["publication_timestamp_local"].date() == date(2026, 5, 1)
    assert row["entry_date"] == date(2026, 5, 4)
    assert row["entry_timestamp_utc"] == datetime(2026, 5, 4, 13, 30, tzinfo=UTC)
    assert row["future_return_1d"] == pytest.approx(0.02)
    assert row["future_return_3d"] == pytest.approx(0.10)
    assert row["exit_date_3d"] == date(2026, 5, 6)


def test_adjusted_entry_and_missing_horizon_are_explicit() -> None:
    article = make_article(timestamp=datetime(2026, 5, 4, 11, 0, tzinfo=UTC))
    prices = [
        make_price(date(2026, 5, 4), open_=190, close=200, adjusted_close=100),
        make_price(date(2026, 5, 5), open_=200, close=200, adjusted_close=100),
    ]
    events = align_events([article], [make_record(article)], prices, horizons=[1, 5])
    row = events.row(0, named=True)
    assert row["entry_date"] == date(2026, 5, 5)
    assert row["entry_adjusted_open"] == 100
    assert row["future_return_1d"] == 0
    assert row["future_return_5d"] is None


def test_alignment_rejects_identity_timestamp_and_bad_market_data() -> None:
    article = make_article()
    other = make_article(article_id="z" * 64)
    prices = [make_price(date(2026, 5, 4), open_=100, close=101)]
    with pytest.raises(ValueError, match="equal length"):
        align_events([article], [], prices, horizons=[1])
    with pytest.raises(ValueError, match="identity mismatch"):
        align_events([article], [make_record(other)], prices, horizons=[1])
    bad_timestamp_record = make_record(article).model_copy(
        update={
            "assessment": make_record(article).assessment.model_copy(
                update={"event_timestamp": datetime(2026, 5, 1, 20, 0, tzinfo=UTC)}
            )
        }
    )
    with pytest.raises(ValueError, match="timestamp mismatch"):
        align_events([article], [bad_timestamp_record], prices, horizons=[1])
    with pytest.raises(ValueError, match="must not be empty"):
        align_events([article], [make_record(article)], [], horizons=[1])
    with pytest.raises(ValueError, match="must be positive"):
        align_events([article], [make_record(article)], prices, horizons=[0])


def test_directional_accuracy_and_ic_are_computed_on_tradable_events() -> None:
    events = pl.DataFrame(
        {
            "sentiment_score": [-0.8, 0.0, 0.8, -1.0],
            "confidence": [0.9, 0.7, 0.8, 1.0],
            "sentiment_label": ["bearish", "neutral", "bullish", "bearish"],
            "tradable": [True, True, True, False],
            "future_return_1d": [-0.02, 0.0005, 0.03, 0.10],
        }
    )
    metrics = compute_event_metrics(events, horizons=[1], neutral_return_bps=10)
    one_day = metrics["horizons"]["1d"]
    assert metrics["n_articles"] == 4
    assert metrics["n_tradable"] == 3
    assert metrics["coverage"] == 0.75
    assert one_day["n"] == 3
    assert one_day["directional_accuracy"] == 1.0
    assert one_day["information_coefficient_spearman"] == 1.0
    assert one_day["confidence_weighted_ic_spearman"] == 1.0
    assert one_day["by_label"]["bearish"]["mean_future_return"] == -0.02


def test_metrics_handle_empty_and_constant_samples_without_nan() -> None:
    empty = pl.DataFrame(
        schema={
            "sentiment_score": pl.Float64,
            "confidence": pl.Float64,
            "sentiment_label": pl.String,
            "tradable": pl.Boolean,
            "future_return_1d": pl.Float64,
        }
    )
    empty_metrics = compute_event_metrics(empty, horizons=[1], neutral_return_bps=10)
    assert empty_metrics["coverage"] is None
    assert empty_metrics["horizons"]["1d"]["directional_accuracy"] is None
    constant = pl.DataFrame(
        {
            "sentiment_score": [0.5, 0.5, 0.5],
            "confidence": [1.0, 1.0, 1.0],
            "sentiment_label": ["bullish", "bullish", "bullish"],
            "tradable": [True, True, True],
            "future_return_1d": [0.01, 0.02, 0.03],
        }
    )
    metrics = compute_event_metrics(constant, horizons=[1], neutral_return_bps=10)
    assert metrics["horizons"]["1d"]["information_coefficient_spearman"] is None
