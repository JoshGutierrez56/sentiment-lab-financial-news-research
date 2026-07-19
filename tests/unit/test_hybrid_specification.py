from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.hybrid.specification import (
    SpecificationSearchConfig,
    _aggregate,
    _bh_adjust,
    freeze_primary_specification,
)


def test_bh_adjustment_is_monotone_and_bounded() -> None:
    candidates = [
        {"combined_validation_p_value": 0.001},
        {"combined_validation_p_value": 0.02},
        {"combined_validation_p_value": 0.5},
    ]
    _bh_adjust(candidates)
    values = [candidate["validation_bh_q_value"] for candidate in candidates]
    assert values == sorted(values)
    assert all(0 <= value <= 1 for value in values)


def test_company_day_aggregation_prevents_story_multiplication() -> None:
    frame = pl.DataFrame(
        {
            "research_split": ["development", "development"],
            "ticker": ["A.US", "A.US"],
            "entry_date": ["2024-01-02", "2024-01-02"],
            "next_split_entry_date": [None, None],
            "exit_date_5d": ["2024-01-09", "2024-01-09"],
            "exit_date_21d": ["2024-02-01", "2024-02-01"],
            "signal": [0.4, 0.8],
            "future_return_5d": [0.1, 0.1],
            "future_return_21d": [0.2, 0.2],
        }
    )
    aggregated = _aggregate(frame, "company_day_aggregate", "signal")
    assert aggregated.height == 1
    assert math.isclose(aggregated["signal"][0], 0.6)


def test_primary_specification_freezes_without_holdout_rows(tmp_path: Path) -> None:
    rows = []
    for index in range(200):
        score = (index % 20 - 10) / 10
        rows.append(
            {
                "research_split": "development" if index < 100 else "validation",
                "ticker": f"T{index % 20}.US",
                "entry_date": date(2024, 1, 1) + timedelta(days=index),
                "next_split_entry_date": (
                    date(2025, 1, 1) if index < 100 else date(2026, 1, 1)
                ),
                "exit_date_5d": date(2024, 1, 1) + timedelta(days=index + 5),
                "exit_date_21d": date(2024, 1, 1) + timedelta(days=index + 21),
                "tradable": True,
                "abstain": False,
                "confidence": 0.8,
                "relevance": 0.9,
                "materiality": 0.7,
                "raw_sentiment": score,
                "future_return_5d": score * 0.01,
                "future_return_21d": score * 0.02,
            }
        )
    events = tmp_path / "events.parquet"
    pl.DataFrame(rows).write_parquet(events)
    output = freeze_primary_specification(
        SpecificationSearchConfig(
            name="test_spec",
            development_validation_events_path=events,
            expected_events_sha256=file_sha256(events),
            sample_hash="a" * 64,
            split_assignment_hash="b" * 64,
            signals=["raw_sentiment"],
            aggregations=["event_level"],
            confidence_thresholds=[0.0],
            relevance_thresholds=[0.0],
            materiality_thresholds=[0.0],
            minimum_validation_events=50,
        ),
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
    )
    manifest = output.read_text(encoding="utf-8")
    assert '"frozen_before_holdout": true' in manifest
    assert '"holdout_metrics_read": false' in manifest
