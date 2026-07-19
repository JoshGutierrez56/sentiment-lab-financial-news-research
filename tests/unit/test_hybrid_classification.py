from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.hybrid.classification import (
    HybridLocalRunConfig,
    LocalRunQualityGates,
    _gate_failures,
    run_local_classification,
)
from sentiment_lab.hybrid.hardware import GPUTelemetrySummary
from sentiment_lab.hybrid.local_model import (
    LocalClassificationRecord,
    LocalUsage,
)
from sentiment_lab.hybrid.schemas import LocalArticleAssessment
from sentiment_lab.nlp.schemas import ExpectedHorizon, SentimentLabel


def _record(*, event_type: str = "earnings", label: str = "bullish") -> LocalClassificationRecord:
    score = 0.5 if label == "bullish" else (-0.5 if label == "bearish" else 0.0)
    return LocalClassificationRecord(
        cache_key="a" * 64,
        article_id="b" * 64,
        article_content_hash="c" * 64,
        ticker="AAPL.US",
        model="local",
        quantization="Q4",
        prompt_version="v1",
        schema_version="v1",
        created_at=datetime.now(UTC),
        response_hash="d" * 64,
        initial_output_valid=True,
        validation_attempts=1,
        attempt_output_hashes=["e" * 64],
        usage=LocalUsage(
            prompt_tokens=10,
            output_tokens=5,
            total_duration_ns=1,
            load_duration_ns=0,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
        ),
        assessment=LocalArticleAssessment(
            sentiment_score=score,
            sentiment_label=SentimentLabel(label),
            confidence=0.8,
            relevance=0.9,
            materiality=0.8,
            novelty=0.7,
            event_type=event_type,
            expected_horizon=ExpectedHorizon.five_days,
            tradable=True,
            abstain=False,
            concise_reasoning="Material company event.",
        ),
    )


def test_quality_gates_wait_for_minimum_and_then_stop_bad_outputs() -> None:
    gates = LocalRunQualityGates(minimum_observations=100)
    records = [_record(event_type="other", label="neutral") for _ in range(99)]
    assert not _gate_failures(
        records, [], elapsed_seconds=1.0, total_articles=5000, gates=gates
    )
    records.append(_record(event_type="other", label="neutral"))
    problems = _gate_failures(
        records, [], elapsed_seconds=1.0, total_articles=5000, gates=gates
    )
    assert any("other event type" in value for value in problems)
    assert any("neutral label share" in value for value in problems)


def test_local_run_config_rejects_unknown_fields(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "name": "locked",
        "sample_manifest_path": tmp_path / "manifest.json",
        "sample_articles_path": tmp_path / "articles.parquet",
        "expected_sample_hash": "a" * 64,
        "expected_articles_sha256": "b" * 64,
        "model": {"model": "qwen", "quantization": "Q4"},
        "unexpected": True,
    }
    try:
        HybridLocalRunConfig.model_validate(payload)
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown configuration field accepted")


class _FakeClient:
    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def classify(self, target: Any, spec: Any, _cache: Any) -> LocalClassificationRecord:
        record = _record()
        return record.model_copy(
            update={
                "article_id": target.article.article_id,
                "ticker": target.member.ticker,
                "model": spec.model,
                "quantization": spec.quantization,
            }
        )


class _FakeTelemetry:
    def __init__(self, **_: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self, **_: Any) -> GPUTelemetrySummary:
        return GPUTelemetrySummary(1, 1.0, 50.0, 50.0, 1000.0, 100.0, 100.0, 0.1, 0.025)


def test_local_run_verifies_sample_and_persists_valid_results(tmp_path: Path) -> None:
    rows = []
    for index in range(10):
        timestamp = datetime(2024, 1, 1, tzinfo=UTC)
        rows.append(
            {
                "article_id": f"article-{index}",
                "provider": "eodhd",
                "provider_timestamp": timestamp,
                "retrieved_at": timestamp,
                "title": f"Company reports earnings {index}",
                "content": "Company reported material earnings results. " * 20,
                "link": "",
                "symbols": ["AAPL.US"],
                "tags": [],
                "provider_sentiment_polarity": None,
                "raw_response_hash": "a" * 64,
                "ticker": "AAPL.US",
                "company_name": "Apple Inc.",
                "sector": "Technology",
            }
        )
    source = tmp_path / "articles.parquet"
    pl.DataFrame(rows).write_parquet(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"sample_hash":"' + "b" * 64 + '","frozen_before_local_inference":true}',
        encoding="utf-8",
    )
    config = HybridLocalRunConfig(
        name="test",
        sample_manifest_path=manifest,
        sample_articles_path=source,
        expected_sample_hash="b" * 64,
        expected_articles_sha256=file_sha256(source),
        expected_article_count=10,
        model={"model": "qwen", "quantization": "Q4"},
        checkpoint_interval=5,
    )
    result = run_local_classification(
        config,
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
        client_factory=_FakeClient,
        telemetry_factory=_FakeTelemetry,
    )
    assert pl.read_parquet(result.classifications_path).height == 10
    manifest_result = result.manifest_path.read_text(encoding="utf-8")
    assert '"valid_count": 10' in manifest_result
    assert '"status": "complete"' in manifest_result

    def unexpected_client() -> _FakeClient:
        raise AssertionError("completed local run attempted inference again")

    rerun = run_local_classification(
        config,
        data_root=tmp_path,
        duckdb_path=tmp_path / "research.duckdb",
        client_factory=unexpected_client,
        telemetry_factory=_FakeTelemetry,
    )
    assert rerun == result
    assert rerun.manifest_path.read_text(encoding="utf-8") == manifest_result
