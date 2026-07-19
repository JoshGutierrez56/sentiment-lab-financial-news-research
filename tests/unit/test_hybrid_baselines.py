from __future__ import annotations

import polars as pl

from sentiment_lab.hybrid.baselines import _lagged_price_features, keyword_sentiment


def test_keyword_sentiment_is_directional_and_neutral_when_unknown() -> None:
    assert keyword_sentiment("Company beats", "Strong profit growth") > 0
    assert keyword_sentiment("Company misses", "Weak loss and decline") < 0
    assert keyword_sentiment("Company holds meeting", "No defined dictionary words") == 0


def test_price_features_use_only_closes_before_entry() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["A.US"] * 30,
            "date": list(range(30)),
            "adjusted_close": [float(value + 100) for value in range(30)],
        }
    )
    features = _lagged_price_features(prices)
    row = features.filter(pl.col("entry_date") == 25).row(0, named=True)
    assert row["momentum_21d"] == 124.0 / 103.0 - 1.0
    assert row["short_term_reversal_5d"] == -(124.0 / 119.0 - 1.0)
