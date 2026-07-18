"""Experiment usage-ledger accounting tests."""

from __future__ import annotations

from conftest import make_article, make_record
from sentiment_lab.experiments.runner import _usage_totals
from sentiment_lab.nlp.schemas import ModelUsage


def test_usage_totals_distinguish_unknown_cost_from_zero_calls() -> None:
    record = make_record(make_article()).model_copy(
        update={"usage": ModelUsage(input_tokens=10, output_tokens=3)}
    )
    assert _usage_totals([record]) == {
        "input_tokens": 10,
        "output_tokens": 3,
        "estimated_cost_usd": None,
    }
    assert _usage_totals([]) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
