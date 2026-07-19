"""Immutable chronological split assignment without performance inspection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256


@dataclass(frozen=True)
class SplitArtifacts:
    assignments_path: Path
    manifest_path: Path
    assignment_hash: str


def freeze_chronological_splits(
    articles_path: Path,
    *,
    sample_hash: str,
    output_root: Path,
    data_root: Path,
    duckdb_path: Path,
) -> SplitArtifacts:
    """Assign exact 60/20/20 rows before any holdout performance analysis."""

    articles = pl.read_parquet(
        articles_path, columns=["article_id", "ticker", "provider_timestamp"]
    ).sort(["provider_timestamp", "ticker", "article_id"])
    if articles.height != 5000:
        raise RuntimeError(f"Chronological split requires exactly 5,000 rows, got {articles.height}")
    assignments = articles.with_row_index("chronological_index").with_columns(
        pl.when(pl.col("chronological_index") < 3000)
        .then(pl.lit("development"))
        .when(pl.col("chronological_index") < 4000)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("holdout"))
        .alias("research_split")
    )
    material: list[dict[str, Any]] = assignments.select(
        "article_id", "research_split", "chronological_index"
    ).to_dicts()
    assignment_hash = hashlib.sha256(stable_json(material).encode()).hexdigest()
    store = ArtifactStore(data_root, duckdb_path)
    assignments_path = store.write_parquet(assignments, output_root / "splits.parquet")
    counts = assignments.group_by("research_split").len().sort("research_split")
    boundaries: dict[str, Any] = {}
    for split in ("development", "validation", "holdout"):
        subset = assignments.filter(pl.col("research_split") == split)
        boundaries[split] = {
            "count": subset.height,
            "first_timestamp": subset["provider_timestamp"].min(),
            "last_timestamp": subset["provider_timestamp"].max(),
            "first_article_id": subset["article_id"][0],
            "last_article_id": subset["article_id"][-1],
        }
    manifest = {
        "sample_hash": sample_hash,
        "articles_sha256": file_sha256(articles_path),
        "method": "global chronology, exact 60/20/20 row counts",
        "performance_fields_read": [],
        "holdout_performance_inspected": False,
        "assignment_hash": assignment_hash,
        "counts": dict(zip(counts["research_split"], counts["len"], strict=True)),
        "boundaries": boundaries,
        "artifacts": {"assignments": str(assignments_path)},
    }
    manifest_path = store.write_json(manifest, output_root / "split_manifest.json")
    store.register_parquet_view("hybrid_5000_chronological_splits", assignments_path)
    return SplitArtifacts(assignments_path, manifest_path, assignment_hash)
