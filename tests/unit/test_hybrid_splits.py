from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from sentiment_lab.hybrid.splits import freeze_chronological_splits


def test_freeze_chronological_splits_is_exact_and_return_blind(tmp_path: Path) -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    articles = pl.DataFrame(
        {
            "article_id": [f"a{index:04d}" for index in range(5000)],
            "ticker": [f"T{index % 125:03d}.US" for index in range(5000)],
            "provider_timestamp": [start + timedelta(hours=index) for index in range(5000)],
            "future_return_21d": [999.0] * 5000,
        }
    )
    source = tmp_path / "articles.parquet"
    articles.write_parquet(source)
    output = freeze_chronological_splits(
        source,
        sample_hash="a" * 64,
        output_root=tmp_path / "split",
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
    )
    assignments = pl.read_parquet(output.assignments_path)
    assert assignments.group_by("research_split").len().sort("research_split").to_dicts() == [
        {"research_split": "development", "len": 3000},
        {"research_split": "holdout", "len": 1000},
        {"research_split": "validation", "len": 1000},
    ]
    manifest = json.loads(output.manifest_path.read_text(encoding="utf-8"))
    assert manifest["performance_fields_read"] == []
    assert manifest["holdout_performance_inspected"] is False
