from datetime import date

import numpy as np
import polars as pl

from sentiment_lab.hybrid.analysis import (
    PredictionAnalysisConfig,
    _clustered_slope,
    _corr,
    _date_block_bootstrap,
    _purge_overlapping_split_returns,
)


def test_dependence_aware_statistics_detect_positive_signal() -> None:
    signal = np.linspace(-1.0, 1.0, 100)
    returns = signal * 0.02
    companies = np.asarray([f"T{index % 10}" for index in range(100)])
    dates = np.asarray([f"2024-01-{index % 20 + 1:02d}" for index in range(100)])
    correlations = _corr(signal, returns)
    assert correlations["pearson_ic"] == 1.0
    clustered = _clustered_slope(signal, returns, companies, dates)
    assert clustered["company_clustered"]["slope"] > 0
    intervals = _date_block_bootstrap(
        dates,
        signal,
        returns,
        block_length=5,
        samples=100,
        seed=7,
    )
    assert intervals["signed_return"]["lower_95"] > 0


def test_holdout_config_fails_closed_without_frozen_specification() -> None:
    payload = {
        "name": "bad_holdout",
        "articles_path": "articles.parquet",
        "classifications_path": "classifications.parquet",
        "splits_path": "splits.parquet",
        "expected_sample_hash": "a" * 64,
        "expected_articles_sha256": "b" * 64,
        "expected_classifications_sha256": "c" * 64,
        "expected_splits_sha256": "d" * 64,
        "included_splits": ["holdout"],
    }
    try:
        PredictionAnalysisConfig.model_validate(payload)
    except ValueError as exc:
        assert "frozen primary specification" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("holdout access was accepted without a frozen specification")


def test_forward_windows_crossing_next_split_are_purged() -> None:
    frame = pl.DataFrame(
        {
            "exit_date_5d": [date(2025, 12, 31), date(2026, 1, 2)],
            "next_split_entry_date": [date(2026, 1, 2), date(2026, 1, 2)],
        }
    )
    purged = _purge_overlapping_split_returns(frame, 5)
    assert purged.height == 1
    assert purged["exit_date_5d"][0] == date(2025, 12, 31)
