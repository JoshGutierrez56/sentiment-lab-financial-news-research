from __future__ import annotations

import numpy as np

from sentiment_lab.hybrid.portfolio import PortfolioSpecification, _performance, _weights


def test_portfolio_specification_locks_horizons_and_company_cap() -> None:
    specification = PortfolioSpecification()
    assert specification.holding_periods == [5, 21]
    assert specification.maximum_company_weight == 0.02


def test_daily_performance_is_calculated_from_portfolio_series() -> None:
    metrics = _performance(np.asarray([0.01, -0.005, 0.002]))
    assert metrics["total_return"] is not None
    assert metrics["maximum_drawdown"] < 0


def test_market_neutral_requires_both_sides() -> None:
    class Position:
        def __init__(self, ticker: str, direction: int) -> None:
            self.ticker = ticker
            self.direction = direction

    one_sided = [Position("A", 1)]
    assert _weights(one_sided, mode="market_neutral", maximum_company_weight=0.02) == {}  # type: ignore[arg-type]
