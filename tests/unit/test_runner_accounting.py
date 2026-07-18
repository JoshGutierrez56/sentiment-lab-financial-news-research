"""Experiment usage-ledger accounting tests."""

from __future__ import annotations

from conftest import make_article, make_record
from sentiment_lab.experiments.runner import _usage_totals
from sentiment_lab.nlp.schemas import ModelUsage


def test_usage_totals_include_cache_and_reasoning_detail() -> None:
    record = make_record(make_article()).model_copy(
        update={
            "usage": ModelUsage(
                input_tokens=10,
                cached_input_tokens=2,
                output_tokens=3,
                reasoning_tokens=1,
                estimated_cost_usd=0.01,
            )
        }
    )
    assert _usage_totals([record]) == {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 3,
        "reasoning_tokens": 1,
        "total_tokens": 13,
        "estimated_cost_usd": 0.01,
    }
    assert _usage_totals([]) == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
