"""Predefined risk scaling only; market regime never flips a signal."""

from __future__ import annotations

import numpy as np


def market_regime_multiplier(
    closes: list[float], *, window: int = 100, volatility_cap: float = 0.03
) -> float:
    if len(closes) < window + 2:
        raise ValueError("Insufficient closes for predefined regime")
    values = np.asarray(closes, dtype=float)
    trend = values[-1] >= values[-window:].mean()
    returns = np.diff(np.log(values[-window:]))
    volatile = np.std(returns, ddof=1) > volatility_cap
    return 1.0 if trend and not volatile else 0.5 if trend else 0.0
