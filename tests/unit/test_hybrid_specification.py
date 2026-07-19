from __future__ import annotations

import math

import polars as pl

from sentiment_lab.hybrid.specification import _aggregate, _bh_adjust


def test_bh_adjustment_is_monotone_and_bounded() -> None:
    candidates = [
        {"combined_validation_p_value": 0.001},
        {"combined_validation_p_value": 0.02},
        {"combined_validation_p_value": 0.5},
    ]
    _bh_adjust(candidates)
    values = [candidate["validation_bh_q_value"] for candidate in candidates]
    assert values == sorted(values)
    assert all(0 <= value <= 1 for value in values)


def test_company_day_aggregation_prevents_story_multiplication() -> None:
    frame = pl.DataFrame(
        {
            "research_split": ["development", "development"],
            "ticker": ["A.US", "A.US"],
            "entry_date": ["2024-01-02", "2024-01-02"],
            "signal": [0.4, 0.8],
            "future_return_5d": [0.1, 0.1],
            "future_return_21d": [0.2, 0.2],
        }
    )
    aggregated = _aggregate(frame, "company_day_aggregate", "signal")
    assert aggregated.height == 1
    assert math.isclose(aggregated["signal"][0], 0.6)
