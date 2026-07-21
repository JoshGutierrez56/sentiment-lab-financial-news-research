"""Immutable registration of the original OpenAI calibration dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.nlp.cache import article_content_hash


class CalibrationRegistryEntry(BaseModel):
    """Tracked, fail-closed pointer to an immutable completed experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str = Field(pattern=r"^[a-z0-9_.-]+$")
    experiment_id: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    data_snapshot_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    prompt_version: str
    schema_version: str
    source_artifacts: dict[str, str]

    @field_validator("source_artifacts")
    @classmethod
    def require_core_artifacts(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"articles.parquet", "assessments.parquet", "events.parquet"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Missing calibration artifacts: {sorted(missing)}")
        for name, digest in value.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Invalid SHA256 for {name}")
        return value


@dataclass(frozen=True)
class RegisteredCalibration:
    dataset_version: str
    dataset_hash: str
    row_count: int
    parquet_path: Path
    manifest_path: Path


def _verify_sources(entry: CalibrationRegistryEntry, results_root: Path) -> Path:
    experiment_root = results_root / entry.experiment_id
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"Calibration experiment not found: {experiment_root}")
    for name, expected in entry.source_artifacts.items():
        source = experiment_root / name
        if not source.is_file():
            raise FileNotFoundError(f"Calibration source artifact missing: {source}")
        actual = file_sha256(source)
        if actual != expected:
            raise RuntimeError(
                f"Immutable calibration source drift for {name}: expected {expected}, got {actual}"
            )
    return experiment_root


def _build_frame(experiment_root: Path, entry: CalibrationRegistryEntry) -> pl.DataFrame:
    articles = pl.read_parquet(experiment_root / "articles.parquet").select(
        "article_id",
        "provider_timestamp",
        "title",
        "content",
        "ticker",
        "company_name",
        "sector",
        "provider_sentiment_polarity",
    )
    assessments = pl.read_parquet(experiment_root / "assessments.parquet").select(
        "article_id",
        "sentiment_score",
        "sentiment_label",
        "confidence",
        "relevance",
        "materiality",
        "novelty",
        "event_type",
        "expected_horizon",
        "tradable",
        "abstain",
        "concise_reasoning",
        "requested_model",
        "model",
        "classification_stage",
        "escalation_reasons",
        "cache_key",
        "input_hash",
        "output_hash",
    )
    events = pl.read_parquet(experiment_root / "events.parquet").select(
        "article_id",
        "entry_date",
        "entry_timestamp_utc",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d",
        "future_return_21d",
    )
    frame = articles.join(assessments, on="article_id", how="inner", validate="1:1").join(
        events, on="article_id", how="inner", validate="1:1"
    )
    if frame.height != 250 or frame["article_id"].n_unique() != 250:
        raise RuntimeError(
            f"Calibration v1 must contain exactly 250 unique articles, got {frame.height}"
        )
    content_hashes = [
        article_content_hash(row["title"], row["content"])
        for row in frame.select("title", "content").iter_rows(named=True)
    ]
    return frame.with_columns(
        pl.Series("article_content_hash", content_hashes),
        pl.lit(entry.dataset_version).alias("calibration_dataset_version"),
        pl.lit(entry.experiment_id).alias("source_experiment_id"),
        pl.lit(entry.source_commit).alias("source_commit"),
        pl.lit(entry.data_snapshot_id).alias("source_data_snapshot_id"),
        pl.lit(entry.prompt_version).alias("prompt_version"),
        pl.lit(entry.schema_version).alias("schema_version"),
    ).sort("provider_timestamp", "ticker", "article_id")


def _logical_dataset_hash(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.iter_rows(named=True):
        canonical: list[str] = []
        for name in frame.columns:
            value = row[name]
            canonical.append(f"{name}={value!s}")
        digest.update("\x1f".join(canonical).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def register_calibration_v1(
    entry: CalibrationRegistryEntry,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> RegisteredCalibration:
    """Verify source bytes and materialize a read-only derived calibration table."""

    experiment_root = _verify_sources(entry, data_root / "results")
    frame = _build_frame(experiment_root, entry)
    dataset_hash = _logical_dataset_hash(frame)
    destination = data_root / "normalized" / "calibration" / entry.dataset_version
    store = ArtifactStore(data_root, duckdb_path)
    parquet_path = destination / "calibration.parquet"
    manifest_path = destination / "manifest.json"
    manifest: dict[str, Any] = {
        **entry.model_dump(mode="json"),
        "row_count": frame.height,
        "dataset_hash": dataset_hash,
        "source_experiment_path": str(experiment_root),
        "derived_artifact": str(parquet_path),
    }
    if parquet_path.is_file():
        existing = pl.read_parquet(parquet_path)
        if _logical_dataset_hash(existing) != dataset_hash:
            raise RuntimeError(f"Refusing to overwrite conflicting calibration: {parquet_path}")
    else:
        store.write_parquet(frame, parquet_path)
    if manifest_path.is_file():
        import json

        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError(f"Refusing to overwrite conflicting manifest: {manifest_path}")
    else:
        store.write_json(manifest, manifest_path)
    store.register_parquet_view("openai_calibration_v1", parquet_path)
    return RegisteredCalibration(
        dataset_version=entry.dataset_version,
        dataset_hash=dataset_hash,
        row_count=frame.height,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
    )
