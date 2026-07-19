"""Return-blind selection of at most 250 new OpenAI calibration articles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sentiment_lab.config.models import AppConfig, SentimentConfig
from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.nlp.cache import ClassificationCache
from sentiment_lab.nlp.classifier import (
    ArticleClassifier,
    ClassificationTarget,
)
from sentiment_lab.nlp.openai_client import OpenAIBatchClient
from sentiment_lab.nlp.schemas import ClassificationRecord


class AdditionalCalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    articles_path: Path
    local_classifications_path: Path
    splits_path: Path
    original_calibration_path: Path
    expected_articles_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_local_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_original_calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_articles: int = Field(default=250, ge=1, le=250)
    maximum_per_ticker: int = Field(default=3, ge=1, le=10)
    maximum_per_sector: int = Field(default=30, ge=1, le=100)
    high_confidence_threshold: float = Field(default=0.80, ge=0.5, le=1.0)
    low_confidence_threshold: float = Field(default=0.55, ge=0.0, le=0.8)
    random_seed: int = 20260718


@dataclass(frozen=True)
class AdditionalCalibrationSample:
    sample_path: Path
    manifest_path: Path
    sample_hash: str


class AdditionalOpenAIRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sample_path: Path
    sample_manifest_path: Path
    expected_sample_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_variant: str = "evidence_v2"
    budget_limit_usd: float = Field(default=1.0, gt=0.0, le=1.0)


def _priority_bucket(
    row: dict[str, Any],
    rare_events: set[str],
    *,
    high_confidence_threshold: float = 0.80,
    low_confidence_threshold: float = 0.55,
) -> tuple[str, ...]:
    buckets: list[str] = []
    label = str(row["sentiment_label"])
    confidence = float(row["confidence"])
    if label == "bullish" and confidence >= high_confidence_threshold:
        buckets.append("high_confidence_bullish")
    if label == "bearish" and confidence >= high_confidence_threshold:
        buckets.append("high_confidence_bearish")
    if bool(row["abstain"]):
        buckets.append("local_abstention")
    if confidence < low_confidence_threshold:
        buckets.append("low_confidence")
    if str(row["event_type"]) in rare_events:
        buckets.append("rare_event_type")
    if not buckets:
        buckets.append("general_calibration")
    return tuple(buckets)


def freeze_additional_openai_sample(
    config: AdditionalCalibrationConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> AdditionalCalibrationSample:
    """Select calibration cases from development/validation only, never returns."""

    expected = {
        config.articles_path: config.expected_articles_sha256,
        config.local_classifications_path: config.expected_local_sha256,
        config.splits_path: config.expected_splits_sha256,
        config.original_calibration_path: config.expected_original_calibration_sha256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Calibration selection input hash mismatch: {path}")
    article_fields = [
        "article_id",
        "provider",
        "retrieved_at",
        "ticker",
        "company_name",
        "sector",
        "provider_timestamp",
        "title",
        "content",
        "link",
        "symbols",
        "tags",
        "provider_sentiment_polarity",
        "raw_response_hash",
        "article_content_hash",
        "pre_inference_event_candidates",
    ]
    articles = pl.read_parquet(config.articles_path, columns=article_fields)
    local = pl.read_parquet(config.local_classifications_path).select(
        "article_id",
        "sentiment_label",
        "sentiment_score",
        "confidence",
        "relevance",
        "materiality",
        "novelty",
        "event_type",
        "tradable",
        "abstain",
    )
    splits = pl.read_parquet(config.splits_path, columns=["article_id", "research_split"])
    original_ids = set(
        pl.read_parquet(config.original_calibration_path, columns=["article_id"])[
            "article_id"
        ].to_list()
    )
    frame = (
        articles.join(local, on="article_id", validate="1:1")
        .join(splits, on="article_id", validate="1:1")
        .filter(pl.col("research_split").is_in(["development", "validation"]))
        .filter(~pl.col("article_id").is_in(original_ids))
    )
    event_counts = Counter(frame["event_type"].to_list())
    rare_events = {
        event for event, count in event_counts.items() if count < max(50, frame.height // 100)
    }
    rows: list[dict[str, Any]] = []
    for row in frame.iter_rows(named=True):
        buckets = _priority_bucket(
            row,
            rare_events,
            high_confidence_threshold=config.high_confidence_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
        )
        rank_material = (
            f"{config.random_seed}:{row['article_id']}:{row['ticker']}:" + ",".join(buckets)
        )
        rows.append(
            {
                **row,
                "calibration_buckets": list(buckets),
                "deterministic_rank": hashlib.sha256(rank_material.encode()).hexdigest(),
            }
        )
    candidates = sorted(
        rows,
        key=lambda row: (
            len(row["calibration_buckets"]) == 1
            and row["calibration_buckets"][0] == "general_calibration",
            row["deterministic_rank"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    ticker_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    bucket_targets = {
        "high_confidence_bullish": 40,
        "high_confidence_bearish": 40,
        "local_abstention": 40,
        "low_confidence": 40,
        "rare_event_type": 50,
        "general_calibration": 40,
    }

    def allowed(row: dict[str, Any]) -> bool:
        return (
            row["article_id"] not in selected_ids
            and ticker_counts[row["ticker"]] < config.maximum_per_ticker
            and sector_counts[row["sector"]] < config.maximum_per_sector
        )

    for bucket, target in bucket_targets.items():
        for row in candidates:
            if bucket_counts[bucket] >= target:
                break
            if bucket not in row["calibration_buckets"] or not allowed(row):
                continue
            selected.append(row)
            selected_ids.add(row["article_id"])
            ticker_counts[row["ticker"]] += 1
            sector_counts[row["sector"]] += 1
            for value in row["calibration_buckets"]:
                bucket_counts[value] += 1
    for row in candidates:
        if len(selected) >= config.maximum_articles:
            break
        if not allowed(row):
            continue
        selected.append(row)
        selected_ids.add(row["article_id"])
        ticker_counts[row["ticker"]] += 1
        sector_counts[row["sector"]] += 1
        for value in row["calibration_buckets"]:
            bucket_counts[value] += 1
    if len(selected) != config.maximum_articles:
        raise RuntimeError(
            f"Additional calibration produced {len(selected)} rows, "
            f"expected {config.maximum_articles}"
        )
    selected.sort(key=lambda row: (row["provider_timestamp"], row["ticker"], row["article_id"]))
    material = [
        {
            "article_id": row["article_id"],
            "article_content_hash": row["article_content_hash"],
            "ticker": row["ticker"],
            "buckets": row["calibration_buckets"],
        }
        for row in selected
    ]
    sample_hash = hashlib.sha256(stable_json(material).encode()).hexdigest()
    root = data_root / "normalized" / "calibration" / f"openai_additional_{sample_hash}"
    store = ArtifactStore(data_root, duckdb_path)
    sample_path = store.write_parquet(
        pl.DataFrame(selected, infer_schema_length=None), root / "sample.parquet"
    )
    manifest = {
        "sample_hash": sample_hash,
        "row_count": len(selected),
        "selection_used_future_returns": False,
        "holdout_articles_selected": 0,
        "original_openai_articles_selected": 0,
        "bucket_counts": dict(bucket_counts),
        "ticker_count": len(ticker_counts),
        "sector_count": len(sector_counts),
        "year_counts": dict(
            Counter(str(row["provider_timestamp"].year) for row in selected)
        ),
        "event_type_counts": dict(Counter(str(row["event_type"]) for row in selected)),
        "input_hashes": {str(path): digest for path, digest in expected.items()},
        "artifacts": {"sample": str(sample_path)},
    }
    manifest_path = store.write_json(manifest, root / "manifest.json")
    store.register_parquet_view("openai_additional_calibration_sample", sample_path)
    return AdditionalCalibrationSample(sample_path, manifest_path, sample_hash)


def _openai_row(record: ClassificationRecord) -> dict[str, Any]:
    return {
        "article_id": record.article_id,
        "ticker": record.ticker,
        "event_timestamp": record.event_timestamp,
        "requested_model": record.requested_model,
        "model": record.model,
        "prompt_version": record.prompt_version,
        "schema_version": record.schema_version,
        "stage": record.stage,
        "escalation_reasons": record.escalation_reasons,
        "cache_key": record.cache_key,
        "input_hash": record.input_hash,
        "output_hash": record.output_hash,
        "input_tokens": record.usage.input_tokens,
        "cached_input_tokens": record.usage.cached_input_tokens,
        "output_tokens": record.usage.output_tokens,
        "reasoning_tokens": record.usage.reasoning_tokens,
        "estimated_cost_usd": record.usage.estimated_cost_usd,
        **record.assessment.model_dump(mode="python"),
    }


def run_additional_openai_calibration(
    config: AdditionalOpenAIRunConfig,
    *,
    api_key: str,
    app_config: AppConfig,
    sentiment_config: SentimentConfig,
) -> Path:
    """Run one cost-guarded Batch calibration without touching OpenAI v1 articles."""

    manifest = json.loads(config.sample_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sample_hash") != config.expected_sample_hash:
        raise RuntimeError("Additional calibration sample hash mismatch")
    if manifest.get("original_openai_articles_selected") != 0:
        raise RuntimeError("Additional sample contains an original OpenAI calibration article")
    if manifest.get("holdout_articles_selected") != 0:
        raise RuntimeError("Additional sample contains a holdout article")
    if file_sha256(config.sample_path) != config.expected_sample_sha256:
        raise RuntimeError("Additional calibration Parquet hash mismatch")
    sample = pl.read_parquet(config.sample_path)
    if sample.height > 250:
        raise RuntimeError("Additional OpenAI calibration may not exceed 250 articles")
    targets: list[ClassificationTarget] = []
    article_fields = set(NewsArticle.model_fields)
    for row in sample.iter_rows(named=True):
        article = NewsArticle.model_validate(
            {key: value for key, value in row.items() if key in article_fields}
        )
        targets.append(
            ClassificationTarget(article, str(row["ticker"]), str(row["company_name"]))
        )
    client = OpenAIBatchClient(api_key, app_config.openai, app_config.storage.data_root)
    classifier = ArticleClassifier(
        client,
        ClassificationCache(app_config.storage.data_root),
        app_config.openai,
        schema_version=sentiment_config.schema_version,
        max_article_characters=sentiment_config.max_article_characters,
        escalation_confidence_threshold=sentiment_config.escalation_confidence_threshold,
        escalation_materiality_threshold=sentiment_config.escalation_materiality_threshold,
        escalation_ambiguity_relevance_threshold=(
            sentiment_config.escalation_ambiguity_relevance_threshold
        ),
        escalation_ambiguity_materiality_threshold=(
            sentiment_config.escalation_ambiguity_materiality_threshold
        ),
    )
    run = classifier.classify_targets(
        targets,
        prompt_variant=config.prompt_variant,
        budget_limit_usd=config.budget_limit_usd,
    )
    if len(run.final_records) != sample.height:
        raise RuntimeError("OpenAI calibration did not return every requested classification")
    if run.current_run_cost_usd > config.budget_limit_usd + 1e-9:
        raise RuntimeError("Additional OpenAI calibration exceeded the $1 hard guard")
    root = (
        app_config.storage.data_root
        / "results"
        / f"openai_additional_{config.expected_sample_hash}"
    )
    store = ArtifactStore(app_config.storage.data_root, app_config.storage.duckdb_path)
    records_path = store.write_parquet(
        pl.DataFrame([_openai_row(record) for record in run.final_records]),
        root / "classifications.parquet",
    )
    ledger_path = store.write_parquet(
        pl.DataFrame(
            [entry.model_dump(mode="python") for entry in run.ledger_entries],
            infer_schema_length=None,
        ),
        root / "ledger.parquet",
    )
    output_manifest = {
        "sample_hash": config.expected_sample_hash,
        "valid_classification_count": len(run.final_records),
        "budget_limit_usd": config.budget_limit_usd,
        "actual_openai_cost_usd": run.current_run_cost_usd,
        "summary": run.summary(),
        "models": dict(Counter(record.model for record in run.final_records)),
        "artifacts": {
            "classifications": str(records_path),
            "ledger": str(ledger_path),
        },
    }
    output_path = store.write_json(output_manifest, root / "manifest.json")
    store.register_parquet_view("openai_additional_classifications", records_path)
    return output_path
