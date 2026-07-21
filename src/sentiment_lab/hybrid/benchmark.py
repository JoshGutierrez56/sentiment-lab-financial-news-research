"""Preregistered local-model benchmark on immutable OpenAI calibration v1."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import pearsonr, spearmanr

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
    ollama_model_metadata,
)

log = logging.getLogger(__name__)

_EVENT_MAP = {
    "earnings_results": "earnings",
    "analyst_rating": "analyst_action",
    "product_launch": "product",
    "regulatory_action": "regulatory",
    "management_change": "management",
    "operational_disruption": "operations",
    "bankruptcy": "fraud_accounting",
}


class SelectionWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictive_performance: float = Field(ge=0.0, le=1.0)
    openai_agreement: float = Field(ge=0.0, le=1.0)
    valid_output: float = Field(ge=0.0, le=1.0)
    tradable_coverage: float = Field(ge=0.0, le=1.0)
    runtime: float = Field(ge=0.0, le=1.0)
    electricity_cost: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> SelectionWeights:
        if not math.isclose(sum(self.model_dump().values()), 1.0, abs_tol=1e-9):
            raise ValueError("Model selection weights must sum to 1.0")
        return self


class LocalBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    calibration_dataset_path: Path
    calibration_manifest_path: Path
    models: list[LocalModelSpec] = Field(min_length=2)
    electricity_rate_usd_per_kwh: float = Field(default=0.25, ge=0.0)
    telemetry_interval_seconds: float = Field(default=0.5, gt=0.0, le=10.0)
    selection_weights: SelectionWeights
    predictive_floor: float = -0.10
    predictive_target: float = 0.20
    runtime_target_articles_per_minute: float = Field(default=10.0, gt=0.0)
    projected_electricity_target_usd: float = Field(default=0.50, gt=0.0)

    @model_validator(mode="after")
    def unique_models(self) -> LocalBenchmarkConfig:
        identifiers = [item.identifier for item in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Benchmark model identifiers must be unique")
        return self


@dataclass(frozen=True)
class BenchmarkOutput:
    benchmark_id: str
    selected_model: str
    results_path: Path
    metrics_path: Path


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, float | None]:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None, None
    return (
        _finite(float(pearsonr(left, right).statistic)),
        _finite(float(spearmanr(left, right).statistic)),
    )


def _target(row: dict[str, Any]) -> LocalTarget:
    timestamp = row["provider_timestamp"]
    article = NewsArticle(
        article_id=row["article_id"],
        provider_timestamp=timestamp,
        retrieved_at=timestamp,
        title=row["title"],
        content=row["content"],
        link="",
        symbols=[row["ticker"]],
        tags=[],
        provider_sentiment_polarity=row["provider_sentiment_polarity"],
        raw_response_hash="0" * 64,
    )
    member = ValidationUniverseMember(
        ticker=row["ticker"],
        company_name=row["company_name"],
        sector=row["sector"],
        aliases=[],
    )
    return LocalTarget(article=article, member=member)


def _local_row(record: LocalClassificationRecord) -> dict[str, Any]:
    assessment = record.assessment.model_dump(mode="python")
    return {
        "article_id": record.article_id,
        **{f"local_{key}": value for key, value in assessment.items()},
        "local_model": record.model,
        "local_quantization": record.quantization,
        "local_cache_key": record.cache_key,
        "local_from_cache": record.from_cache,
        "local_initial_output_valid": record.initial_output_valid,
        "local_validation_attempts": record.validation_attempts,
        "local_prompt_tokens": record.usage.prompt_tokens,
        "local_output_tokens": record.usage.output_tokens,
        "local_total_duration_ns": record.usage.total_duration_ns,
        "local_eval_duration_ns": record.usage.eval_duration_ns,
    }


def _prediction_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    tradable = frame.filter(pl.col("local_tradable") & ~pl.col("local_abstain"))
    result: dict[str, Any] = {
        "tradable_count": tradable.height,
        "tradable_coverage": tradable.height / frame.height,
    }
    score = tradable["local_sentiment_score"].to_numpy().astype(float)
    for horizon in (5, 21):
        returns = tradable[f"future_return_{horizon}d"].to_numpy().astype(float)
        pearson, spearman = _correlation(score, returns)
        direction = np.sign(score)
        directional = direction != 0
        result[f"{horizon}d"] = {
            "pearson_ic": pearson,
            "spearman_ic": spearman,
            "average_signed_return": (
                float(np.mean(direction[directional] * returns[directional]))
                if np.any(directional)
                else None
            ),
            "directional_accuracy": (
                float(np.mean(direction[directional] == np.sign(returns[directional])))
                if np.any(directional)
                else None
            ),
            "n": len(returns),
            "directional_n": int(np.sum(directional)),
        }
    return result


def _agreement_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    local_score = frame["local_sentiment_score"].to_numpy().astype(float)
    openai_score = frame["sentiment_score"].to_numpy().astype(float)
    score_pearson, score_spearman = _correlation(local_score, openai_score)
    local_labels = frame["local_sentiment_label"].to_numpy()
    openai_labels = frame["sentiment_label"].to_numpy()
    directional = np.isin(local_labels, ["bullish", "bearish"]) & np.isin(
        openai_labels, ["bullish", "bearish"]
    )
    local_events = frame["local_event_type"].to_list()
    openai_events = [_EVENT_MAP.get(str(value), str(value)) for value in frame["event_type"]]
    return {
        "exact_sentiment_label_agreement": float(np.mean(local_labels == openai_labels)),
        "sentiment_score_pearson": score_pearson,
        "sentiment_score_spearman": score_spearman,
        "bullish_bearish_directional_agreement": (
            float(np.mean(local_labels[directional] == openai_labels[directional]))
            if np.any(directional)
            else None
        ),
        "bullish_bearish_overlap_n": int(np.sum(directional)),
        "tradable_agreement": float(
            np.mean(frame["local_tradable"].to_numpy() == frame["tradable"].to_numpy())
        ),
        "abstain_agreement": float(
            np.mean(frame["local_abstain"].to_numpy() == frame["abstain"].to_numpy())
        ),
        "event_type_agreement": float(
            np.mean(np.asarray(local_events) == np.asarray(openai_events))
        ),
    }


def _bounded(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


def _selection_score(metrics: dict[str, Any], config: LocalBenchmarkConfig) -> dict[str, float]:
    prediction = metrics["prediction"]
    primary_spearman = np.mean(
        [float(prediction[key]["spearman_ic"] or 0.0) for key in ("5d", "21d")]
    )
    predictive = _bounded(
        float(primary_spearman), config.predictive_floor, config.predictive_target
    )
    agreement = metrics["agreement"]
    agreement_values = [
        float(agreement["exact_sentiment_label_agreement"]),
        (float(agreement["sentiment_score_spearman"] or 0.0) + 1.0) / 2.0,
        float(agreement["tradable_agreement"]),
        float(agreement["event_type_agreement"]),
    ]
    openai_agreement = float(np.mean(agreement_values))
    valid = 0.75 * float(metrics["final_valid_rate"]) + 0.25 * float(metrics["initial_valid_rate"])
    openai_coverage = float(metrics["openai_tradable_coverage"])
    local_coverage = float(prediction["tradable_coverage"])
    coverage = max(0.0, 1.0 - abs(local_coverage - openai_coverage) / 0.75)
    runtime = min(
        1.0,
        float(metrics["articles_per_minute"]) / config.runtime_target_articles_per_minute,
    )
    cost = min(
        1.0,
        config.projected_electricity_target_usd
        / max(float(metrics["projected_5000_electricity_cost_usd"]), 1e-12),
    )
    components = {
        "predictive_performance": predictive,
        "openai_agreement": openai_agreement,
        "valid_output": valid,
        "tradable_coverage": coverage,
        "runtime": runtime,
        "electricity_cost": cost,
    }
    weights = config.selection_weights.model_dump()
    components["total"] = sum(components[key] * weights[key] for key in weights)
    return components


def run_local_benchmark(
    config: LocalBenchmarkConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> BenchmarkOutput:
    """Run/resume both models and select using weights frozen in YAML."""

    calibration = pl.read_parquet(config.calibration_dataset_path)
    if calibration.height != 250:
        raise RuntimeError(f"Benchmark calibration must contain 250 rows, got {calibration.height}")
    manifest = json.loads(config.calibration_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("row_count") != 250:
        raise RuntimeError("Calibration manifest row_count is not 250")
    config_material = config.model_dump(mode="json")
    config_hash = hashlib.sha256(stable_json(config_material).encode()).hexdigest()
    benchmark_id = f"local_benchmark_{config_hash[:16]}"
    root = data_root / "results" / benchmark_id
    store = ArtifactStore(data_root, duckdb_path)
    summaries: list[dict[str, Any]] = []
    all_results: list[pl.DataFrame] = []

    for spec in config.models:
        log.info("Benchmarking %s on 250 immutable calibration articles", spec.identifier)
        cache = LocalClassificationCache(data_root)
        sampler = NvidiaTelemetrySampler(interval_seconds=config.telemetry_interval_seconds)
        sampler.start()
        started = time.monotonic()
        records: list[LocalClassificationRecord] = []
        failures: list[dict[str, str]] = []
        with OllamaStructuredClient() as client:
            for index, row in enumerate(calibration.iter_rows(named=True), start=1):
                try:
                    records.append(client.classify(_target(row), spec, cache))
                except (httpx.HTTPError, RuntimeError) as exc:
                    failures.append({"article_id": row["article_id"], "error": str(exc)})
                if index % 10 == 0:
                    log.info("%s progress %d/250 failures=%d", spec.model, index, len(failures))
        elapsed = time.monotonic() - started
        telemetry = sampler.stop(electricity_rate_usd_per_kwh=config.electricity_rate_usd_per_kwh)
        local = pl.DataFrame([_local_row(item) for item in records], infer_schema_length=None)
        joined = calibration.join(local, on="article_id", how="left", validate="1:1")
        valid = local.height
        uncached = [item for item in records if not item.from_cache]
        api_elapsed = sum(item.usage.total_duration_ns for item in uncached) / 1e9
        eval_elapsed = sum(item.usage.eval_duration_ns for item in uncached) / 1e9
        output_tokens = sum(item.usage.output_tokens for item in uncached)
        articles_per_minute = len(uncached) / max(api_elapsed / 60.0, 1e-12)
        tokens_per_second = output_tokens / max(eval_elapsed, 1e-12)
        projection_factor = 5000 / max(len(uncached), 1)
        metadata = ollama_model_metadata(spec.model)
        complete = joined.filter(pl.col("local_model").is_not_null())
        summary: dict[str, Any] = {
            "model": spec.model,
            "quantization": spec.quantization,
            "model_metadata": {
                "modified_at": metadata.get("modified_at"),
                "details": metadata.get("details", {}),
            },
            "valid_count": valid,
            "failure_count": len(failures),
            "failures": failures,
            "final_valid_rate": valid / 250,
            "initial_valid_rate": (
                sum(bool(value) for value in local["local_initial_output_valid"].to_list()) / valid
                if valid
                else 0.0
            ),
            "cache_hits": sum(item.from_cache for item in records),
            "wall_time_seconds": elapsed,
            "articles_per_minute": articles_per_minute,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens": sum(item.usage.prompt_tokens for item in uncached),
            "output_tokens": output_tokens,
            "gpu": telemetry.__dict__,
            "projected_5000_runtime_seconds": api_elapsed * projection_factor,
            "projected_5000_energy_kwh": telemetry.energy_kwh * projection_factor,
            "projected_5000_electricity_cost_usd": (
                telemetry.electricity_cost_usd * projection_factor
            ),
            "openai_tradable_coverage": sum(
                bool(value) for value in calibration["tradable"].to_list()
            )
            / calibration.height,
            "agreement": _agreement_metrics(complete),
            "prediction": _prediction_metrics(complete),
        }
        summary["selection"] = _selection_score(summary, config)
        summaries.append(summary)
        all_results.append(joined.with_columns(pl.lit(spec.identifier).alias("model_id")))

    selected = max(summaries, key=lambda item: float(item["selection"]["total"]))
    results = pl.concat(all_results, how="diagonal_relaxed")
    results_path = store.write_parquet(results, root / "classifications.parquet")
    metrics = {
        "benchmark_id": benchmark_id,
        "config_hash": config_hash,
        "calibration_dataset_hash": manifest["dataset_hash"],
        "calibration_source_hash": file_sha256(config.calibration_dataset_path),
        "preregistered_config": config_material,
        "models": summaries,
        "selected_model": selected["model"],
        "selected_quantization": selected["quantization"],
        "selection_score": selected["selection"]["total"],
    }
    metrics_path = store.write_json(metrics, root / "metrics.json")
    store.register_parquet_view("local_model_benchmark", results_path)
    return BenchmarkOutput(
        benchmark_id=benchmark_id,
        selected_model=str(selected["model"]),
        results_path=results_path,
        metrics_path=metrics_path,
    )
