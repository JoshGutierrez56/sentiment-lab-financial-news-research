from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.hybrid.calibration_analysis import (
    CalibrationAnalysisConfig,
    run_calibration_analysis,
)


def _write_inputs(root: Path, *, holdout: bool = False) -> dict[str, Path]:
    ids = [f"article-{index}" for index in range(12)]
    labels = ["bullish", "bearish", "neutral"] * 4
    scores = [0.8, -0.7, 0.0] * 4
    returns = [0.02, -0.015, 0.001] * 4
    articles = pl.DataFrame(
        {
            "article_id": ids,
            "ticker": [f"T{index % 4}.US" for index in range(12)],
            "sector": ["Technology", "Financials"] * 6,
            "entry_date": [date(2024, 1, index + 1) for index in range(12)],
            **{
                f"exit_date_{horizon}d": [date(2024, 1, index + 1) for index in range(12)]
                for horizon in (1, 3, 5, 10, 21, 63)
            },
            **{
                f"future_return_{horizon}d": [value * horizon / 5 for value in returns]
                for horizon in (1, 3, 5, 10, 21, 63)
            },
        }
    )
    common = {
        "article_id": ids,
        "sentiment_score": scores,
        "sentiment_label": labels,
        "confidence": [0.8] * 12,
        "materiality": [0.7] * 12,
        "novelty": [0.6] * 12,
        "event_type": ["earnings"] * 12,
        "tradable": [True] * 12,
        "abstain": [False] * 12,
    }
    paths = {
        "articles": root / "articles.parquet",
        "local": root / "local.parquet",
        "sample": root / "sample.parquet",
        "openai": root / "openai.parquet",
        "splits": root / "splits.parquet",
    }
    articles.write_parquet(paths["articles"])
    pl.DataFrame(common).write_parquet(paths["local"])
    pl.DataFrame({"article_id": ids}).write_parquet(paths["sample"])
    pl.DataFrame(common).write_parquet(paths["openai"])
    pl.DataFrame(
        {
            "article_id": ids,
            "research_split": [
                "holdout" if holdout and index == 11 else "development" for index in range(12)
            ],
        }
    ).write_parquet(paths["splits"])
    return paths


def _config(paths: dict[str, Path]) -> CalibrationAnalysisConfig:
    return CalibrationAnalysisConfig(
        name="test",
        articles_path=paths["articles"],
        local_classifications_path=paths["local"],
        additional_sample_path=paths["sample"],
        openai_classifications_path=paths["openai"],
        splits_path=paths["splits"],
        expected_hashes={name: file_sha256(path) for name, path in paths.items()},
        bootstrap_samples=100,
    )


def test_calibration_analysis_compares_models_and_returns(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    metrics_path, comparisons_path = run_calibration_analysis(
        _config(paths), data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
    )

    metrics = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))
    assert comparisons_path.is_file()
    assert metrics["count"] == 12
    assert metrics["agreement"]["exact_sentiment_label"] == 1.0
    assert metrics["agreement"]["event_type"] == 1.0
    assert metrics["predictive"]["local"]["development"]["5d"]["spearman_ic"] > 0
    assert metrics["predictive"]["openai"]["development"]["21d"]["spearman_ic"] > 0


def test_calibration_analysis_rejects_holdout(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, holdout=True)

    with pytest.raises(RuntimeError, match="holdout"):
        run_calibration_analysis(
            _config(paths), data_root=tmp_path, duckdb_path=tmp_path / "research.duckdb"
        )
