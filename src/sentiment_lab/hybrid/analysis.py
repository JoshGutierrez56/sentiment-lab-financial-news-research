"""Dependence-aware predictive analysis for frozen hybrid classifications."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import pearsonr, spearmanr

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256

HORIZONS = (1, 3, 5, 10, 21, 63)
PRIMARY_HORIZONS = (5, 21)
SIGNALS = (
    "raw_sentiment",
    "sentiment_confidence",
    "sentiment_confidence_materiality",
    "sentiment_confidence_materiality_novelty",
)


class PredictionAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    articles_path: Path
    classifications_path: Path
    splits_path: Path
    expected_sample_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_articles_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_classifications_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    included_splits: list[str] = Field(default_factory=lambda: ["development", "validation"])
    bootstrap_samples: int = Field(default=1000, ge=100, le=10_000)
    random_seed: int = 20260718
    neutral_return_bps: float = Field(default=0.0, ge=0.0, le=100.0)
    primary_specification_manifest: Path | None = None

    @model_validator(mode="after")
    def protect_holdout(self) -> PredictionAnalysisConfig:
        allowed = {"development", "validation", "holdout"}
        if not self.included_splits or not set(self.included_splits) <= allowed:
            raise ValueError("included_splits contains an unknown research split")
        if "holdout" in self.included_splits and self.primary_specification_manifest is None:
            raise ValueError("Holdout analysis requires a frozen primary specification manifest")
        return self


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _corr(signal: np.ndarray, returns: np.ndarray) -> dict[str, float | None]:
    if len(signal) < 3 or np.std(signal) == 0 or np.std(returns) == 0:
        return {"pearson_ic": None, "spearman_ic": None}
    return {
        "pearson_ic": _finite(float(pearsonr(signal, returns).statistic)),
        "spearman_ic": _finite(float(spearmanr(signal, returns).statistic)),
    }


def _clustered_slope(
    signal: np.ndarray,
    returns: np.ndarray,
    companies: np.ndarray,
    dates: np.ndarray,
) -> dict[str, Any]:
    if len(signal) < 10 or np.std(signal) == 0:
        return {}
    design = sm.add_constant(signal)
    output: dict[str, Any] = {}
    company_codes = np.unique(companies, return_inverse=True)[1]
    date_codes = np.unique(dates, return_inverse=True)[1]
    variants = {
        "company_clustered": company_codes,
        "date_clustered": date_codes,
        "two_way_clustered": np.column_stack([company_codes, date_codes]),
    }
    for name, groups in variants.items():
        try:
            fit = sm.OLS(returns, design).fit(
                cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}
            )
            output[name] = {
                "slope": _finite(float(fit.params[1])),
                "standard_error": _finite(float(fit.bse[1])),
                "t_statistic": _finite(float(fit.tvalues[1])),
                "p_value": _finite(float(fit.pvalues[1])),
                "n": len(signal),
            }
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            output[name] = {}
    return output


def _company_bootstrap(
    companies: np.ndarray,
    signed_returns: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    unique = np.unique(companies)
    if len(unique) < 2:
        return {"lower_95": None, "upper_95": None}
    means = {company: np.mean(signed_returns[companies == company]) for company in unique}
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.mean([means[value] for value in rng.choice(unique, len(unique), replace=True)])
            for _ in range(samples)
        ],
        dtype=float,
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"lower_95": float(lower), "upper_95": float(upper)}


def _date_block_bootstrap(
    dates: np.ndarray,
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    block_length: int,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    unique = np.unique(dates)
    if len(unique) < block_length * 2:
        empty: dict[str, float | None] = {"lower_95": None, "upper_95": None}
        return {"pearson_ic": empty, "spearman_ic": empty, "signed_return": empty}
    by_date = {value: np.flatnonzero(dates == value) for value in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "pearson_ic": [],
        "spearman_ic": [],
        "signed_return": [],
    }
    starts = np.arange(0, len(unique) - block_length + 1)
    blocks_needed = math.ceil(len(unique) / block_length)
    for _ in range(samples):
        sampled_dates: list[Any] = []
        for start in rng.choice(starts, blocks_needed, replace=True):
            sampled_dates.extend(unique[start : start + block_length])
        sampled_dates = sampled_dates[: len(unique)]
        draw = np.concatenate([by_date[value] for value in sampled_dates])
        correlations = _corr(signal[draw], returns[draw])
        if correlations["pearson_ic"] is not None:
            values["pearson_ic"].append(float(correlations["pearson_ic"]))
        if correlations["spearman_ic"] is not None:
            values["spearman_ic"].append(float(correlations["spearman_ic"]))
        values["signed_return"].append(float(np.mean(np.sign(signal[draw]) * returns[draw])))
    output: dict[str, dict[str, float | None]] = {}
    for name, observations in values.items():
        if observations:
            lower, upper = np.quantile(np.asarray(observations), [0.025, 0.975])
            output[name] = {"lower_95": float(lower), "upper_95": float(upper)}
        else:
            output[name] = {"lower_95": None, "upper_95": None}
    return output


def _horizon_metrics(
    frame: pl.DataFrame,
    *,
    signal_column: str,
    horizon: int,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    usable = frame.filter(
        pl.col("tradable")
        & ~pl.col("abstain")
        & pl.col(f"future_return_{horizon}d").is_not_null()
    )
    returns = usable[f"future_return_{horizon}d"].to_numpy().astype(float)
    signal = usable[signal_column].to_numpy().astype(float)
    labels = np.asarray(usable["sentiment_label"].to_list())
    companies = np.asarray(usable["ticker"].to_list())
    dates = np.asarray(usable["entry_date"].cast(pl.String).to_list())
    directional = np.abs(signal) > 1e-12
    lower, upper = np.quantile(returns, [0.01, 0.99]) if len(returns) else (0.0, 0.0)
    winsorized = np.clip(returns, lower, upper)
    by_label: dict[str, Any] = {}
    for label in ("bullish", "neutral", "bearish"):
        observations = returns[labels == label]
        by_label[label] = {
            "n": len(observations),
            "average_return": float(np.mean(observations)) if len(observations) else None,
            "median_return": float(np.median(observations)) if len(observations) else None,
        }
    bullish = returns[labels == "bullish"]
    bearish = returns[labels == "bearish"]
    signed = np.sign(signal) * returns
    company_means = [np.mean(signed[companies == value]) for value in np.unique(companies)]
    realized_sign = np.where(returns > threshold, 1, np.where(returns < -threshold, -1, 0))
    return {
        "n": len(returns),
        "directional_n": int(np.sum(directional)),
        "directional_accuracy": (
            float(np.mean(np.sign(signal[directional]) == realized_sign[directional]))
            if np.any(directional)
            else None
        ),
        "returns_by_sentiment": by_label,
        "bullish_minus_bearish_spread": (
            float(np.mean(bullish) - np.mean(bearish))
            if len(bullish) and len(bearish)
            else None
        ),
        **_corr(signal, returns),
        "average_signed_return": float(np.mean(signed)) if len(signed) else None,
        "company_equal_average_signed_return": (
            float(np.mean(company_means)) if company_means else None
        ),
        "company_cluster_bootstrap_95_ci": _company_bootstrap(
            companies, signed, samples=bootstrap_samples, seed=seed
        )
        if len(returns)
        else {"lower_95": None, "upper_95": None},
        "date_block_bootstrap_95_ci": _date_block_bootstrap(
            dates,
            signal,
            returns,
            block_length=max(5, horizon),
            samples=bootstrap_samples,
            seed=seed + 1,
        )
        if len(returns)
        else {},
        "clustered_regression": _clustered_slope(signal, returns, companies, dates),
        "winsorized_1_99": {
            **_corr(signal, winsorized),
            "average_signed_return": (
                float(np.mean(np.sign(signal) * winsorized)) if len(returns) else None
            ),
            "lower_cutoff": float(lower) if len(returns) else None,
            "upper_cutoff": float(upper) if len(returns) else None,
        },
    }


def _purge_overlapping_split_returns(frame: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Exclude outcomes whose measurement window reaches the next split."""

    boundary = pl.col("next_split_entry_date")
    return frame.filter(
        boundary.is_null() | (pl.col(f"exit_date_{horizon}d") < boundary)
    )


def _add_signals(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("sentiment_score").alias("raw_sentiment"),
        (pl.col("sentiment_score") * pl.col("confidence")).alias(
            "sentiment_confidence"
        ),
        (
            pl.col("sentiment_score")
            * pl.col("confidence")
            * pl.col("materiality")
        ).alias("sentiment_confidence_materiality"),
        (
            pl.col("sentiment_score")
            * pl.col("confidence")
            * pl.col("materiality")
            * pl.col("novelty")
        ).alias("sentiment_confidence_materiality_novelty"),
    )


def _company_day(frame: pl.DataFrame, *, strongest: bool) -> pl.DataFrame:
    if strongest:
        return (
            frame.with_columns(
                pl.col("sentiment_confidence_materiality")
                .abs()
                .alias("_absolute_strength")
            )
            .sort("_absolute_strength", descending=True, nulls_last=True)
            .group_by(["research_split", "ticker", "entry_date"], maintain_order=True)
            .first()
            .drop("_absolute_strength")
        )
    aggregations: list[pl.Expr] = [pl.col(name).mean().alias(name) for name in SIGNALS]
    aggregations.extend(
        [
            pl.col("confidence").mean(),
            pl.col("materiality").mean(),
            pl.col("novelty").mean(),
            pl.col("tradable").any(),
            pl.col("abstain").all(),
            pl.col("sector").first(),
            pl.col("event_type").first(),
            pl.col("provider_timestamp").min(),
            pl.col("next_split_entry_date").first(),
            *[pl.col(f"exit_date_{horizon}d").first() for horizon in HORIZONS],
            *[pl.col(f"future_return_{horizon}d").first() for horizon in HORIZONS],
        ]
    )
    grouped = frame.group_by(
        ["research_split", "ticker", "entry_date"], maintain_order=True
    ).agg(aggregations)
    return grouped.with_columns(
        pl.when(pl.col("raw_sentiment") > 0.25)
        .then(pl.lit("bullish"))
        .when(pl.col("raw_sentiment") < -0.25)
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral"))
        .alias("sentiment_label")
    )


def _breakdown(frame: pl.DataFrame, group: str) -> list[dict[str, Any]]:
    primary_expressions: list[pl.Expr] = []
    for horizon in PRIMARY_HORIZONS:
        eligible = pl.col("next_split_entry_date").is_null() | (
            pl.col(f"exit_date_{horizon}d") < pl.col("next_split_entry_date")
        )
        primary_expressions.extend(
            [
                eligible.cast(pl.Int64).sum().alias(f"eligible_{horizon}d"),
                pl.when(eligible)
                .then(
                    pl.col("sentiment_score").sign()
                    * pl.col(f"future_return_{horizon}d")
                )
                .otherwise(None)
                .mean()
                .alias(f"average_signed_return_{horizon}d"),
            ]
        )
    return (
        frame.group_by(group)
        .agg(
            pl.len().alias("n"),
            pl.col("tradable").sum().alias("tradable"),
            pl.col("abstain").sum().alias("abstain"),
            pl.col("sentiment_score").mean().alias("mean_sentiment_score"),
            *primary_expressions,
        )
        .sort("n", descending=True)
        .to_dicts()
    )


def run_prediction_analysis(
    config: PredictionAnalysisConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> tuple[Path, Path]:
    """Analyze only preregistered splits; holdout access is fail-closed."""

    expected = {
        config.articles_path: config.expected_articles_sha256,
        config.classifications_path: config.expected_classifications_sha256,
        config.splits_path: config.expected_splits_sha256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Immutable analysis input hash mismatch: {path}")
    if "holdout" in config.included_splits:
        assert config.primary_specification_manifest is not None
        specification = json.loads(
            config.primary_specification_manifest.read_text(encoding="utf-8")
        )
        if specification.get("frozen_before_holdout") is not True:
            raise RuntimeError("Primary specification was not frozen before holdout access")
        if specification.get("sample_hash") != config.expected_sample_hash:
            raise RuntimeError("Primary specification sample hash mismatch")

    article_columns = [
        "article_id",
        "ticker",
        "sector",
        "provider_timestamp",
        "entry_date",
        "story_cluster_id",
        "provider_sentiment_polarity",
        *[f"exit_date_{horizon}d" for horizon in HORIZONS],
        *[f"future_return_{horizon}d" for horizon in HORIZONS],
    ]
    articles = pl.read_parquet(config.articles_path, columns=article_columns)
    classifications = pl.read_parquet(config.classifications_path)
    splits = pl.read_parquet(config.splits_path, columns=["article_id", "research_split"])
    if articles.height != 5000 or classifications.height != 5000 or splits.height != 5000:
        raise RuntimeError("Prediction inputs must each contain exactly 5,000 rows")
    joined = articles.join(
        classifications, on=["article_id", "ticker"], validate="1:1"
    ).join(splits, on="article_id", validate="1:1")
    split_minimums = {
        split: joined.filter(pl.col("research_split") == split)["entry_date"].min()
        for split in ("development", "validation", "holdout")
    }
    joined = joined.with_columns(
        pl.when(pl.col("research_split") == "development")
        .then(pl.lit(split_minimums["validation"]))
        .when(pl.col("research_split") == "validation")
        .then(pl.lit(split_minimums["holdout"]))
        .otherwise(pl.lit(None, dtype=pl.Date))
        .alias("next_split_entry_date")
    )
    frame = _add_signals(
        joined
        .filter(pl.col("research_split").is_in(config.included_splits))
        .with_columns(
            pl.col("provider_timestamp").dt.year().alias("year"),
            pl.when(pl.col("confidence") < 0.50)
            .then(pl.lit("[0.00,0.50)"))
            .when(pl.col("confidence") < 0.70)
            .then(pl.lit("[0.50,0.70)"))
            .when(pl.col("confidence") < 0.85)
            .then(pl.lit("[0.70,0.85)"))
            .otherwise(pl.lit("[0.85,1.00]"))
            .alias("confidence_bucket"),
            pl.when(pl.col("materiality") < 0.25)
            .then(pl.lit("[0.00,0.25)"))
            .when(pl.col("materiality") < 0.50)
            .then(pl.lit("[0.25,0.50)"))
            .when(pl.col("materiality") < 0.75)
            .then(pl.lit("[0.50,0.75)"))
            .otherwise(pl.lit("[0.75,1.00]"))
            .alias("materiality_bucket"),
        )
    )
    threshold = config.neutral_return_bps / 10_000
    metrics: dict[str, Any] = {
        "definition": (
            "Event returns overlap. ICs, clustered regressions, and block-bootstrap intervals "
            "are predictive diagnostics, not a portfolio Sharpe series."
        ),
        "sample_hash": config.expected_sample_hash,
        "included_splits": config.included_splits,
        "counts": {
            "events": frame.height,
            "labels": dict(Counter(frame["sentiment_label"].to_list())),
            "tradable": int(frame["tradable"].sum()),
            "abstain": int(frame["abstain"].sum()),
            "tradable_coverage": float(cast(float, frame["tradable"].mean())),
            "abstention_rate": float(cast(float, frame["abstain"].mean())),
        },
        "event_level": {},
        "strongest_company_day": {},
        "company_day_aggregate": {},
        "split_purge_boundaries": {
            "development": split_minimums["validation"],
            "validation": split_minimums["holdout"],
            "holdout": None,
        },
        "breakdowns": {
            split: {
                group: _breakdown(
                    frame.filter(pl.col("research_split") == split), group
                )
                for group in (
                    "year",
                    "sector",
                    "ticker",
                    "event_type",
                    "confidence_bucket",
                    "materiality_bucket",
                )
            }
            for split in config.included_splits
        },
        "excluding_other": {},
    }
    representations = {
        "event_level": frame,
        "strongest_company_day": _company_day(frame, strongest=True),
        "company_day_aggregate": _company_day(frame, strongest=False),
    }
    for representation, values in representations.items():
        for split in config.included_splits:
            split_frame = values.filter(pl.col("research_split") == split)
            metrics[representation][split] = {}
            for signal in SIGNALS:
                metrics[representation][split][signal] = {
                    f"{horizon}d": _horizon_metrics(
                        _purge_overlapping_split_returns(split_frame, horizon),
                        signal_column=signal,
                        horizon=horizon,
                        threshold=threshold,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.random_seed + horizon,
                    )
                    for horizon in HORIZONS
                }
    without_other = frame.filter(pl.col("event_type") != "other")
    for split in config.included_splits:
        split_frame = without_other.filter(pl.col("research_split") == split)
        metrics["excluding_other"][split] = {
            f"{horizon}d": _horizon_metrics(
                _purge_overlapping_split_returns(split_frame, horizon),
                signal_column="sentiment_confidence_materiality",
                horizon=horizon,
                threshold=threshold,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.random_seed + 10_000 + horizon,
            )
            for horizon in HORIZONS
        }

    config_hash = hashlib.sha256(
        stable_json(config.model_dump(mode="json")).encode()
    ).hexdigest()
    root = data_root / "results" / f"prediction_{config_hash[:16]}"
    store = ArtifactStore(data_root, duckdb_path)
    events_path = store.write_parquet(frame, root / "events.parquet")
    metrics["config_hash"] = config_hash
    metrics["input_hashes"] = {str(path): digest for path, digest in expected.items()}
    metrics_path = store.write_json(metrics, root / "metrics.json")
    store.register_parquet_view("hybrid_prediction_events", events_path)
    return metrics_path, events_path
