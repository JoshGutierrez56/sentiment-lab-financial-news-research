"""Basic directional accuracy and information-coefficient report."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _correlations(scores: np.ndarray, returns: np.ndarray) -> dict[str, float | None]:
    if len(scores) < 3 or np.std(scores) == 0 or np.std(returns) == 0:
        return {
            "information_coefficient_spearman": None,
            "information_coefficient_p_value": None,
            "pearson_correlation": None,
            "pearson_p_value": None,
        }
    spearman = spearmanr(scores, returns)
    pearson = pearsonr(scores, returns)
    return {
        "information_coefficient_spearman": _finite_or_none(float(spearman.statistic)),
        "information_coefficient_p_value": _finite_or_none(float(spearman.pvalue)),
        "pearson_correlation": _finite_or_none(float(pearson.statistic)),
        "pearson_p_value": _finite_or_none(float(pearson.pvalue)),
    }


def compute_event_metrics(
    events: pl.DataFrame,
    *,
    horizons: list[int],
    neutral_return_bps: float,
) -> dict[str, Any]:
    """Measure score/return association; "accuracy" is explicitly directional."""

    threshold = neutral_return_bps / 10_000.0
    n_tradable = int(events["tradable"].sum()) if events.height else 0
    result: dict[str, Any] = {
        "definition": (
            "Directional accuracy compares ChatGPT's bullish/bearish/neutral label with "
            f"the sign of future return; |return| <= {neutral_return_bps:g} bps is neutral. "
            "This is not human-label sentiment accuracy. IC and ordinary p-values are "
            "descriptive for this small milestone and do not correct overlapping returns."
        ),
        "n_articles": events.height,
        "n_tradable": n_tradable,
        "coverage": n_tradable / events.height if events.height else None,
        "horizons": {},
    }
    for horizon in horizons:
        return_column = f"future_return_{horizon}d"
        complete = events.filter(pl.col("tradable") & pl.col(return_column).is_not_null())
        scores = complete["sentiment_score"].to_numpy().astype(float)
        confidences = complete["confidence"].to_numpy().astype(float)
        returns = complete[return_column].to_numpy().astype(float)
        labels = complete["sentiment_label"].to_list()
        realized = np.where(
            returns > threshold, "bullish", np.where(returns < -threshold, "bearish", "neutral")
        )
        accuracy = float(np.mean(np.asarray(labels) == realized)) if len(labels) else float("nan")
        horizon_result: dict[str, Any] = {
            "n": len(labels),
            "directional_accuracy": _finite_or_none(accuracy),
            "mean_future_return": _finite_or_none(float(np.mean(returns)))
            if len(returns)
            else None,
            "median_future_return": _finite_or_none(float(np.median(returns)))
            if len(returns)
            else None,
            **_correlations(scores, returns),
        }
        confidence_correlations = _correlations(scores * confidences, returns)
        horizon_result["confidence_weighted_ic_spearman"] = confidence_correlations[
            "information_coefficient_spearman"
        ]
        by_label: dict[str, Any] = {}
        for label in ("bearish", "neutral", "bullish"):
            values = returns[np.asarray(labels) == label]
            by_label[label] = {
                "n": len(values),
                "mean_future_return": (
                    _finite_or_none(float(np.mean(values))) if len(values) else None
                ),
            }
        horizon_result["by_label"] = by_label
        result["horizons"][f"{horizon}d"] = horizon_result
    return result
