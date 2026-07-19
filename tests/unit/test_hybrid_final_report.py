from __future__ import annotations

from sentiment_lab.hybrid.final_report import research_decision


def _portfolio(total: float, sharpe: float = 1.0) -> dict[str, object]:
    return {
        "conservative_net": {"total_return": total, "sharpe": sharpe},
        "maximum_exposure_hhi": 0.05,
    }


def test_pass_requires_incremental_holdout_and_cost_survival() -> None:
    local = {"spearman_ic": 0.15, "average_signed_return": 0.01}
    baseline = {
        "keyword_sentiment": {"spearman_ic": 0.02},
        "event_type_signal": {"spearman_ic": 0.03},
        "eodhd_sentiment": {"spearman_ic": 0.01},
    }
    conclusion, scale, gates = research_decision(
        holdout_5d=local,
        holdout_21d=local,
        baseline_5d=baseline,
        baseline_21d=baseline,
        portfolio_5d=_portfolio(0.1),
        portfolio_21d=_portfolio(0.1),
        tradable_coverage=0.5,
    )
    assert conclusion == "PASS"
    assert scale == "Expand to 25,000"
    assert all(gates.values())


def test_mixed_evidence_stops_inconclusive() -> None:
    local = {"spearman_ic": 0.01, "average_signed_return": 0.001}
    baseline = {"keyword_sentiment": {"spearman_ic": 0.02}}
    conclusion, scale, _ = research_decision(
        holdout_5d=local,
        holdout_21d=local,
        baseline_5d=baseline,
        baseline_21d=baseline,
        portfolio_5d=_portfolio(-0.1),
        portfolio_21d=_portfolio(0.1),
        tradable_coverage=0.5,
    )
    assert conclusion == "INCONCLUSIVE"
    assert scale == "Stop at 5,000"
