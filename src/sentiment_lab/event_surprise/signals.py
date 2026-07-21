"""Development-only score normalization and preregistered event signals."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Normalizer:
    mean: float
    scale: float
    fitted_split: str = "development"


def fit_normalizer(
    frame: pl.DataFrame, column: str, *, split_column: str = "research_split"
) -> Normalizer:
    development = frame.filter(pl.col(split_column) == "development").drop_nulls([column])
    if not development.height:
        raise ValueError("Cannot normalize without development observations")
    raw_scale = development[column].std()
    raw_mean = development[column].mean()
    if not isinstance(raw_scale, (int, float)) or not isinstance(raw_mean, (int, float)):
        raise ValueError("Normalization column must be numeric")
    scale = float(raw_scale)
    return Normalizer(mean=float(raw_mean), scale=scale if scale > 0 else 1.0)


def add_event_signals(frame: pl.DataFrame, normalizer: Normalizer) -> pl.DataFrame:
    required = {
        "llm_direction_score",
        "finbert_score",
        "company_specificity",
        "materiality",
        "novelty",
        "confidence",
        "direction_score",
        "article_id",
        "ticker",
        "entry_date",
        "abstain",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing signal fields: {sorted(missing)}")
    output = frame.with_columns(
        ((pl.col("llm_direction_score") - normalizer.mean) / normalizer.scale).alias(
            "calibrated_llm_score"
        ),
        (pl.col("llm_direction_score") - pl.col("finbert_score")).alias(
            "llm_minus_finbert_residual"
        ),
        (pl.col("llm_direction_score") - pl.col("finbert_score"))
        .abs()
        .alias("llm_finbert_disagreement"),
    ).with_columns(
        (pl.col("direction_score") * pl.col("confidence")).alias("event_surprise_confidence"),
        (pl.col("direction_score") * pl.col("confidence") * pl.col("materiality")).alias(
            "event_surprise_confidence_materiality"
        ),
        (
            pl.col("direction_score")
            * pl.col("confidence")
            * pl.col("materiality")
            * pl.col("novelty")
        ).alias("event_surprise_score"),
        (
            pl.col("llm_minus_finbert_residual")
            * pl.col("company_specificity")
            * pl.col("materiality")
            * pl.col("novelty")
            * pl.col("confidence")
        ).alias("event_surprise_signal"),
    )
    return output.with_columns(
        pl.when(pl.col("abstain"))
        .then(0.0)
        .otherwise(pl.col("event_surprise_signal"))
        .alias("event_surprise_signal")
    )


def strongest_qualifying_event_per_company_day(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.filter(~pl.col("abstain"))
        .with_columns(pl.col("event_surprise_signal").abs().alias("_signal_strength"))
        .sort(
            ["ticker", "entry_date", "_signal_strength", "article_id"],
            descending=[False, False, True, False],
        )
        .group_by(["ticker", "entry_date"], maintain_order=True)
        .first()
        .drop("_signal_strength")
    )
