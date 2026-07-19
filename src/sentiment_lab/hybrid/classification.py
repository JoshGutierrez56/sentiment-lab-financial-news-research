"""Resumable local classification for an immutable hybrid sample."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sentiment_lab.config.models import ValidationUniverseMember
from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.hybrid.hardware import NvidiaTelemetrySampler
from sentiment_lab.hybrid.local_model import (
    LocalClassificationCache,
    LocalClassificationRecord,
    LocalModelSpec,
    LocalTarget,
    OllamaStructuredClient,
)

log = logging.getLogger(__name__)


class LocalRunQualityGates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_invalid_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    maximum_other_fraction: float = Field(default=0.40, ge=0.0, le=1.0)
    minimum_tradable_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    maximum_label_fraction: float = Field(default=0.85, ge=0.0, le=1.0)
    maximum_projected_runtime_days: float = Field(default=7.0, gt=0.0)
    minimum_observations: int = Field(default=500, ge=100)


class HybridLocalRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sample_manifest_path: Path
    sample_articles_path: Path
    expected_sample_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_articles_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_article_count: int = Field(default=5000, ge=1, le=5000)
    model: LocalModelSpec
    quality_gates: LocalRunQualityGates = Field(default_factory=LocalRunQualityGates)
    checkpoint_interval: int = Field(default=25, ge=1, le=500)
    telemetry_interval_seconds: float = Field(default=1.0, gt=0.0, le=10.0)
    electricity_rate_usd_per_kwh: float = Field(default=0.25, ge=0.0)


@dataclass(frozen=True)
class HybridLocalRunOutput:
    run_id: str
    classifications_path: Path
    manifest_path: Path


class LocalRunQualityError(RuntimeError):
    """Raised only after partial results and a diagnostic manifest are persisted."""


def _target(row: dict[str, Any]) -> LocalTarget:
    article_fields = set(NewsArticle.model_fields)
    article = NewsArticle.model_validate(
        {key: value for key, value in row.items() if key in article_fields}
    )
    member = ValidationUniverseMember(
        ticker=str(row["ticker"]),
        company_name=str(row["company_name"]),
        sector=str(row["sector"]),
        aliases=[],
    )
    return LocalTarget(article=article, member=member)


def _row(record: LocalClassificationRecord) -> dict[str, Any]:
    return {
        "article_id": record.article_id,
        "article_content_hash": record.article_content_hash,
        "ticker": record.ticker,
        "model": record.model,
        "quantization": record.quantization,
        "prompt_version": record.prompt_version,
        "schema_version": record.schema_version,
        "cache_key": record.cache_key,
        "from_cache": record.from_cache,
        "response_hash": record.response_hash,
        "initial_output_valid": record.initial_output_valid,
        "validation_attempts": record.validation_attempts,
        "prompt_tokens": record.usage.prompt_tokens,
        "output_tokens": record.usage.output_tokens,
        "total_duration_ns": record.usage.total_duration_ns,
        "eval_duration_ns": record.usage.eval_duration_ns,
        **record.assessment.model_dump(mode="python"),
    }


def _gate_failures(
    records: list[LocalClassificationRecord],
    failures: list[dict[str, str]],
    *,
    elapsed_seconds: float,
    total_articles: int,
    gates: LocalRunQualityGates,
) -> list[str]:
    processed = len(records) + len(failures)
    if processed < gates.minimum_observations:
        return []
    problems: list[str] = []
    invalid_fraction = len(failures) / processed
    if invalid_fraction > gates.maximum_invalid_fraction:
        problems.append(
            f"invalid structured output {invalid_fraction:.3%} exceeds "
            f"{gates.maximum_invalid_fraction:.3%}"
        )
    if records:
        other_fraction = sum(
            record.assessment.event_type.value == "other" for record in records
        ) / len(records)
        tradable_fraction = sum(record.assessment.tradable for record in records) / len(
            records
        )
        labels = Counter(record.assessment.sentiment_label.value for record in records)
        dominant_label, dominant_count = labels.most_common(1)[0]
        dominant_fraction = dominant_count / len(records)
        if other_fraction > gates.maximum_other_fraction:
            problems.append(
                f"other event type {other_fraction:.3%} exceeds "
                f"{gates.maximum_other_fraction:.3%}"
            )
        if tradable_fraction < gates.minimum_tradable_fraction:
            problems.append(
                f"tradable coverage {tradable_fraction:.3%} is below "
                f"{gates.minimum_tradable_fraction:.3%}"
            )
        if dominant_fraction > gates.maximum_label_fraction:
            problems.append(
                f"{dominant_label} label share {dominant_fraction:.3%} exceeds "
                f"{gates.maximum_label_fraction:.3%}"
            )
    projected_days = (elapsed_seconds / processed * total_articles) / 86_400
    if projected_days > gates.maximum_projected_runtime_days:
        problems.append(
            f"runtime projection {projected_days:.2f} days exceeds "
            f"{gates.maximum_projected_runtime_days:.2f} days"
        )
    return problems


def run_local_classification(
    config: HybridLocalRunConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
    client_factory: Callable[[], OllamaStructuredClient] = OllamaStructuredClient,
    telemetry_factory: Callable[..., NvidiaTelemetrySampler] = NvidiaTelemetrySampler,
) -> HybridLocalRunOutput:
    """Classify the frozen sample, checkpointing without mutating its artifacts."""

    manifest = json.loads(config.sample_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sample_hash") != config.expected_sample_hash:
        raise RuntimeError("Frozen sample manifest hash does not match preregistration")
    if manifest.get("frozen_before_local_inference") is not True:
        raise RuntimeError("Sample was not frozen before local inference")
    if file_sha256(config.sample_articles_path) != config.expected_articles_sha256:
        raise RuntimeError("Frozen article artifact hash does not match preregistration")
    articles = pl.read_parquet(config.sample_articles_path).sort(
        ["provider_timestamp", "ticker", "article_id"]
    )
    if articles.height != config.expected_article_count:
        raise RuntimeError(
            f"Expected {config.expected_article_count} frozen articles, got {articles.height}"
        )

    material = config.model_dump(mode="json")
    config_hash = hashlib.sha256(stable_json(material).encode()).hexdigest()
    run_id = f"hybrid_local_{config_hash[:16]}"
    root = data_root / "results" / run_id
    classifications_path = root / "classifications.parquet"
    failures_path = root / "failures.json"
    manifest_path = root / "manifest.json"

    # A completed run is an immutable research artifact.  A cache-only rerun
    # must not replace its measured runtime, token, or GPU telemetry with the
    # cost of reading the cache.
    if classifications_path.is_file() and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_ids = pl.read_parquet(classifications_path, columns=["article_id"])
        if (
            existing_manifest.get("status") == "complete"
            and existing_manifest.get("config_hash") == config_hash
            and existing_manifest.get("sample_hash") == config.expected_sample_hash
            and existing_manifest.get("sample_articles_sha256")
            == config.expected_articles_sha256
            and existing_ids.height == articles.height
            and existing_ids.get_column("article_id").n_unique() == articles.height
        ):
            return HybridLocalRunOutput(run_id, classifications_path, manifest_path)

    store = ArtifactStore(data_root, duckdb_path)
    cache = LocalClassificationCache(data_root)
    telemetry = telemetry_factory(interval_seconds=config.telemetry_interval_seconds)
    telemetry.start()
    started = time.monotonic()
    records: list[LocalClassificationRecord] = []
    failures: list[dict[str, str]] = []
    stopped_for: list[str] = []

    def checkpoint() -> None:
        if records:
            store.write_parquet(
                pl.DataFrame([_row(record) for record in records], infer_schema_length=None),
                classifications_path,
            )
        store.write_json(failures, failures_path)

    try:
        with client_factory() as client:
            for index, row in enumerate(articles.iter_rows(named=True), start=1):
                try:
                    records.append(client.classify(_target(row), config.model, cache))
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    failures.append(
                        {"article_id": str(row["article_id"]), "error": str(exc)[:2000]}
                    )
                if index % config.checkpoint_interval == 0:
                    checkpoint()
                    log.info(
                        "Local 5,000 progress %d/%d valid=%d failures=%d cache_hits=%d",
                        index,
                        articles.height,
                        len(records),
                        len(failures),
                        sum(record.from_cache for record in records),
                    )
                    stopped_for = _gate_failures(
                        records,
                        failures,
                        elapsed_seconds=time.monotonic() - started,
                        total_articles=articles.height,
                        gates=config.quality_gates,
                    )
                    if stopped_for:
                        break
    finally:
        elapsed = time.monotonic() - started
        gpu = telemetry.stop(
            electricity_rate_usd_per_kwh=config.electricity_rate_usd_per_kwh
        )
        checkpoint()

    counts = Counter(record.assessment.sentiment_label.value for record in records)
    events = Counter(record.assessment.event_type.value for record in records)
    uncached = [record for record in records if not record.from_cache]
    valid = len(records)
    run_manifest = {
        "run_id": run_id,
        "config_hash": config_hash,
        "preregistered_config": material,
        "sample_hash": config.expected_sample_hash,
        "sample_articles_sha256": config.expected_articles_sha256,
        "status": "stopped_quality_gate" if stopped_for else "complete",
        "quality_gate_failures": stopped_for,
        "processed_count": valid + len(failures),
        "valid_count": valid,
        "failure_count": len(failures),
        "valid_fraction": valid / max(valid + len(failures), 1),
        "cache_hits": sum(record.from_cache for record in records),
        "tradable_count": sum(record.assessment.tradable for record in records),
        "tradable_fraction": (
            sum(record.assessment.tradable for record in records) / valid if valid else 0.0
        ),
        "abstention_fraction": (
            sum(record.assessment.abstain for record in records) / valid if valid else 0.0
        ),
        "label_counts": dict(counts),
        "event_type_counts": dict(events),
        "other_fraction": events["other"] / valid if valid else 0.0,
        "initial_valid_fraction": (
            sum(record.initial_output_valid for record in records) / valid if valid else 0.0
        ),
        "prompt_tokens": sum(record.usage.prompt_tokens for record in uncached),
        "output_tokens": sum(record.usage.output_tokens for record in uncached),
        "elapsed_seconds": elapsed,
        "articles_per_minute": len(uncached) / max(elapsed / 60.0, 1e-12),
        "gpu": gpu.__dict__,
        "artifacts": {
            "classifications": str(classifications_path),
            "failures": str(failures_path),
        },
    }
    store.write_json(run_manifest, manifest_path)
    if classifications_path.is_file():
        store.register_parquet_view("hybrid_5000_local_classifications", classifications_path)
    if stopped_for:
        raise LocalRunQualityError("; ".join(stopped_for))
    if valid != articles.height:
        raise LocalRunQualityError(
            f"Local classification completed with {valid}/{articles.height} valid outputs"
        )
    return HybridLocalRunOutput(run_id, classifications_path, manifest_path)
