from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.hybrid.calibration import CalibrationRegistryEntry, register_calibration_v1


def _write_sources(root: Path) -> CalibrationRegistryEntry:
    experiment = root / "results" / "exp1"
    experiment.mkdir(parents=True)
    rows = 250
    ids = [f"{index:064x}" for index in range(rows)]
    articles = pl.DataFrame(
        {
            "article_id": ids,
            "provider_timestamp": [datetime(2025, 1, 1, tzinfo=UTC)] * rows,
            "title": [f"Company reports result {index}" for index in range(rows)],
            "content": [f"Company-specific full article {index}." for index in range(rows)],
            "ticker": ["AAA.US"] * rows,
            "company_name": ["Alpha Inc."] * rows,
            "sector": ["Technology"] * rows,
            "provider_sentiment_polarity": [None] * rows,
        }
    )
    assessments = pl.DataFrame(
        {
            "article_id": ids,
            "sentiment_score": [0.5] * rows,
            "sentiment_label": ["bullish"] * rows,
            "confidence": [0.8] * rows,
            "relevance": [0.9] * rows,
            "materiality": [0.7] * rows,
            "novelty": [0.6] * rows,
            "event_type": ["earnings_results"] * rows,
            "expected_horizon": ["5d"] * rows,
            "tradable": [True] * rows,
            "abstain": [False] * rows,
            "concise_reasoning": ["Results beat expectations."] * rows,
            "requested_model": ["gpt-mini"] * rows,
            "model": ["gpt-mini-date"] * rows,
            "classification_stage": ["first_pass"] * rows,
            "escalation_reasons": [[] for _ in range(rows)],
            "cache_key": [f"cache-{index}" for index in range(rows)],
            "input_hash": [f"input-{index}" for index in range(rows)],
            "output_hash": [f"output-{index}" for index in range(rows)],
        }
    )
    events = pl.DataFrame(
        {
            "article_id": ids,
            "entry_date": [date(2025, 1, 2)] * rows,
            "entry_timestamp_utc": [datetime(2025, 1, 2, 14, 30, tzinfo=UTC)] * rows,
            "future_return_1d": [0.01] * rows,
            "future_return_3d": [0.02] * rows,
            "future_return_5d": [0.03] * rows,
            "future_return_21d": [0.04] * rows,
        }
    )
    paths = {
        "articles.parquet": articles,
        "assessments.parquet": assessments,
        "events.parquet": events,
    }
    for name, frame in paths.items():
        frame.write_parquet(experiment / name)
    return CalibrationRegistryEntry(
        dataset_version="openai_calibration_v1",
        experiment_id="exp1",
        source_commit="b703114",
        data_snapshot_id="0123456789abcdef",
        prompt_version="evidence_v2.1.0-cost",
        schema_version="article_assessment.v2",
        source_artifacts={
            name: file_sha256(experiment / name) for name in paths
        },
    )


def test_register_calibration_is_reproducible_and_refuses_source_drift(tmp_path: Path) -> None:
    entry = _write_sources(tmp_path)
    first = register_calibration_v1(
        entry, data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
    )
    second = register_calibration_v1(
        entry, data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
    )
    assert first.dataset_hash == second.dataset_hash
    assert first.row_count == 250
    assert pl.read_parquet(first.parquet_path)["article_content_hash"].n_unique() == 250

    source = tmp_path / "results" / "exp1" / "articles.parquet"
    source.write_bytes(source.read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="source drift"):
        register_calibration_v1(
            entry, data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
        )
