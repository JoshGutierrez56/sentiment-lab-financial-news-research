from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.hybrid import benchmark
from sentiment_lab.hybrid.benchmark import (
    LocalBenchmarkConfig,
    SelectionWeights,
    run_local_benchmark,
)
from sentiment_lab.hybrid.hardware import GPUTelemetrySummary
from sentiment_lab.hybrid.local_model import (
    LocalClassificationRecord,
    LocalModelSpec,
    LocalUsage,
)
from sentiment_lab.hybrid.schemas import LocalArticleAssessment


def _calibration(path: Path) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    labels = ("bearish", "neutral", "bullish")
    scores = (-0.6, 0.0, 0.6)
    for index in range(250):
        label = labels[index % 3]
        rows.append(
            {
                "article_id": f"{index:064x}",
                "provider_timestamp": datetime(2025, 1, 1, tzinfo=UTC),
                "title": f"Acme earnings {index}",
                "content": "Acme reported quarterly earnings and revenue. " * 20,
                "ticker": "ACME.US",
                "company_name": "Acme Corporation",
                "sector": "Industrials",
                "provider_sentiment_polarity": 0.0,
                "sentiment_score": scores[index % 3],
                "sentiment_label": label,
                "tradable": label != "neutral",
                "abstain": label == "neutral",
                "event_type": "earnings_results",
                "future_return_5d": scores[index % 3] * 0.02 + index / 1_000_000,
                "future_return_21d": scores[index % 3] * 0.04 + index / 1_000_000,
            }
        )
    calibration = path / "calibration.parquet"
    manifest = path / "manifest.json"
    pl.DataFrame(rows).write_parquet(calibration)
    manifest.write_text('{"row_count":250,"dataset_hash":"dataset-test"}', encoding="utf-8")
    return calibration, manifest


class _FakeOllama:
    def __enter__(self) -> _FakeOllama:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def classify(self, target: Any, spec: LocalModelSpec, _cache: Any) -> LocalClassificationRecord:
        index = int(target.article.article_id, 16)
        labels = ("bearish", "neutral", "bullish")
        scores = (-0.6, 0.0, 0.6)
        label = labels[index % 3]
        abstain = label == "neutral"
        assessment = LocalArticleAssessment.model_validate(
            {
                "sentiment_score": scores[index % 3],
                "sentiment_label": label,
                "confidence": 0.8,
                "relevance": 0.9,
                "materiality": 0.7,
                "novelty": 0.6,
                "event_type": "earnings",
                "expected_horizon": "5d",
                "tradable": not abstain,
                "abstain": abstain,
                "abstain_reason": "Balanced event." if abstain else None,
                "concise_reasoning": "Earnings direction determines the label.",
            }
        )
        return LocalClassificationRecord(
            cache_key=f"cache-{spec.model}-{index}",
            article_id=target.article.article_id,
            article_content_hash=f"{index:064x}",
            ticker="ACME.US",
            model=spec.model,
            quantization=spec.quantization,
            prompt_version="test",
            schema_version="test",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            from_cache=index == 0,
            response_hash=f"{index:064x}",
            initial_output_valid=index % 2 == 0,
            validation_attempts=1 if index % 2 == 0 else 2,
            attempt_output_hashes=[f"{index:064x}"],
            usage=LocalUsage(
                prompt_tokens=100,
                output_tokens=50,
                total_duration_ns=1_000_000_000,
                load_duration_ns=0,
                prompt_eval_duration_ns=100_000_000,
                eval_duration_ns=500_000_000,
            ),
            assessment=assessment,
        )


class _FakeSampler:
    def __init__(self, *, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds

    def start(self) -> None:
        pass

    def stop(self, *, electricity_rate_usd_per_kwh: float) -> GPUTelemetrySummary:
        return GPUTelemetrySummary(
            sample_count=10,
            duration_seconds=10.0,
            average_utilization_percent=80.0,
            maximum_utilization_percent=100.0,
            maximum_memory_used_mib=20_000.0,
            average_power_watts=300.0,
            maximum_power_watts=400.0,
            energy_kwh=0.001,
            electricity_cost_usd=0.001 * electricity_rate_usd_per_kwh,
        )


def test_benchmark_runs_two_models_and_writes_preregistered_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calibration, manifest = _calibration(tmp_path)
    monkeypatch.setattr(benchmark, "OllamaStructuredClient", _FakeOllama)
    monkeypatch.setattr(benchmark, "NvidiaTelemetrySampler", _FakeSampler)
    monkeypatch.setattr(
        benchmark,
        "ollama_model_metadata",
        lambda model: {"modified_at": "now", "details": {"model": model}},
    )
    config = LocalBenchmarkConfig(
        name="test",
        calibration_dataset_path=calibration,
        calibration_manifest_path=manifest,
        models=[
            LocalModelSpec(model="fast", quantization="q4"),
            LocalModelSpec(model="strong", quantization="q4"),
        ],
        selection_weights=SelectionWeights(
            predictive_performance=0.35,
            openai_agreement=0.25,
            valid_output=0.15,
            tradable_coverage=0.10,
            runtime=0.10,
            electricity_cost=0.05,
        ),
    )
    output = run_local_benchmark(
        config, data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
    )
    assert output.selected_model in {"fast", "strong"}
    assert pl.read_parquet(output.results_path).height == 500
    assert output.metrics_path.is_file()


def test_selection_config_rejects_nonunit_weights() -> None:
    try:
        SelectionWeights(
            predictive_performance=0.5,
            openai_agreement=0.5,
            valid_output=0.5,
            tradable_coverage=0.0,
            runtime=0.0,
            electricity_cost=0.0,
        )
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid weights were accepted")

