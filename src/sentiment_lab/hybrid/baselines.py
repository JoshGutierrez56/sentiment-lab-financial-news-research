"""Point-in-time baselines and placebos for the frozen hybrid sample."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.hybrid.analysis import HORIZONS, _corr, _purge_overlapping_split_returns

_POSITIVE = {
    "approval",
    "beat",
    "beats",
    "growth",
    "increase",
    "profit",
    "raises",
    "record",
    "strong",
    "upgrade",
}
_NEGATIVE = {
    "bankruptcy",
    "decline",
    "downgrade",
    "fraud",
    "investigation",
    "loss",
    "miss",
    "misses",
    "reduced",
    "weak",
}


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    articles_path: Path
    classifications_path: Path
    splits_path: Path
    prices_path: Path
    expected_hashes: dict[str, str]
    evaluation_splits: list[str] = Field(default_factory=lambda: ["development", "validation"])
    random_seed: int = 20260718
    primary_specification_manifest: Path | None = None

    @model_validator(mode="after")
    def protect_holdout(self) -> BaselineConfig:
        if "holdout" in self.evaluation_splits and self.primary_specification_manifest is None:
            raise ValueError("Holdout baselines require a frozen primary specification")
        return self


def keyword_sentiment(title: str, content: str) -> float:
    words = re.findall(r"[a-z]+", f"{title} {content}".casefold())
    positive = sum(word in _POSITIVE for word in words)
    negative = sum(word in _NEGATIVE for word in words)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def _lagged_price_features(prices: pl.DataFrame) -> pl.DataFrame:
    return (
        prices.sort(["ticker", "date"])
        .with_columns(
            (
                pl.col("adjusted_close").shift(1).over("ticker")
                / pl.col("adjusted_close").shift(22).over("ticker")
                - 1.0
            ).alias("momentum_21d"),
            -(
                pl.col("adjusted_close").shift(1).over("ticker")
                / pl.col("adjusted_close").shift(6).over("ticker")
                - 1.0
            ).alias("short_term_reversal_5d"),
        )
        .select("ticker", pl.col("date").alias("entry_date"), "momentum_21d", "short_term_reversal_5d")
    )


def _permute_within(frame: pl.DataFrame, column: str, group: str, *, seed: int) -> np.ndarray:
    output = np.empty(frame.height, dtype=float)
    rng = np.random.default_rng(seed)
    values = frame[column].to_numpy().astype(float)
    for indices in frame.with_row_index("_index").partition_by(group, as_dict=False):
        locations = indices["_index"].to_numpy().astype(int)
        output[locations] = rng.permutation(values[locations])
    return output


def _metrics(signal: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    usable = np.isfinite(signal) & np.isfinite(returns)
    selected_signal = signal[usable]
    selected_returns = returns[usable]
    directional = np.abs(selected_signal) > 1e-12
    return {
        "n": int(np.sum(usable)),
        **_corr(selected_signal, selected_returns),
        "average_signed_return": (
            float(np.mean(np.sign(selected_signal) * selected_returns))
            if len(selected_returns)
            else None
        ),
        "directional_accuracy": (
            float(
                np.mean(
                    np.sign(selected_signal[directional])
                    == np.sign(selected_returns[directional])
                )
            )
            if np.any(directional)
            else None
        ),
    }


def run_baselines(
    config: BaselineConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> Path:
    """Fit the event-type baseline on development only and evaluate locked splits."""

    paths = {
        "articles": config.articles_path,
        "classifications": config.classifications_path,
        "splits": config.splits_path,
        "prices": config.prices_path,
    }
    for name, path in paths.items():
        if file_sha256(path) != config.expected_hashes.get(name):
            raise RuntimeError(f"Baseline input hash mismatch: {name}")
    if "holdout" in config.evaluation_splits:
        assert config.primary_specification_manifest is not None
        specification = json.loads(
            config.primary_specification_manifest.read_text(encoding="utf-8")
        )
        if specification.get("frozen_before_holdout") is not True:
            raise RuntimeError("Primary specification was not frozen before holdout baselines")
    articles = pl.read_parquet(config.articles_path).select(
        "article_id",
        "ticker",
        "provider_timestamp",
        "entry_date",
        "title",
        "content",
        "provider_sentiment_polarity",
        *[f"exit_date_{horizon}d" for horizon in HORIZONS],
        *[f"future_return_{horizon}d" for horizon in HORIZONS],
    )
    local = pl.read_parquet(config.classifications_path).select(
        "article_id", "sentiment_score", "sentiment_label", "event_type", "tradable", "abstain"
    )
    splits = pl.read_parquet(config.splits_path).select("article_id", "research_split")
    prices = pl.read_parquet(config.prices_path)
    frame = (
        articles.join(local, on="article_id", validate="1:1")
        .join(splits, on="article_id", validate="1:1")
        .join(_lagged_price_features(prices), on=["ticker", "entry_date"], how="left")
        .with_columns(
            pl.struct(["title", "content"])
            .map_elements(
                lambda value: keyword_sentiment(value["title"], value["content"]),
                return_dtype=pl.Float64,
            )
            .alias("keyword_sentiment"),
            pl.col("provider_sentiment_polarity").cast(pl.Float64).alias("eodhd_sentiment"),
        )
        .sort(["provider_timestamp", "ticker", "article_id"])
    )
    split_minimums = {
        split: frame.filter(pl.col("research_split") == split)["entry_date"].min()
        for split in ("development", "validation", "holdout")
    }
    frame = frame.with_columns(
        pl.when(pl.col("research_split") == "development")
        .then(pl.lit(split_minimums["validation"]))
        .when(pl.col("research_split") == "validation")
        .then(pl.lit(split_minimums["holdout"]))
        .otherwise(pl.lit(None, dtype=pl.Date))
        .alias("next_split_entry_date")
    )
    development = frame.filter(pl.col("research_split") == "development")
    class_frequencies = Counter(development["sentiment_label"].to_list())
    labels = np.asarray([-1.0, 0.0, 1.0])
    probabilities = np.asarray(
        [
            class_frequencies["bearish"],
            class_frequencies["neutral"],
            class_frequencies["bullish"],
        ],
        dtype=float,
    )
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(config.random_seed)
    frame = frame.with_columns(
        pl.Series("random_matched", rng.choice(labels, frame.height, p=probabilities)),
        pl.lit(0.0).alias("always_neutral"),
        pl.Series(
            "shuffled_ticker_placebo",
            _permute_within(frame, "sentiment_score", "entry_date", seed=config.random_seed + 1),
        ),
        pl.Series(
            "shuffled_timestamp_placebo",
            _permute_within(frame, "sentiment_score", "ticker", seed=config.random_seed + 2),
        ),
    )
    signal_names = (
        "sentiment_score",
        "eodhd_sentiment",
        "keyword_sentiment",
        "momentum_21d",
        "short_term_reversal_5d",
        "always_neutral",
        "random_matched",
        "shuffled_ticker_placebo",
        "shuffled_timestamp_placebo",
    )
    results: dict[str, Any] = {
        "definition": "All price baselines use closes available before conservative entry.",
        "development_class_frequencies": dict(class_frequencies),
        "split_purge_boundaries": {
            "development": split_minimums["validation"],
            "validation": split_minimums["holdout"],
            "holdout": None,
        },
        "splits": {},
    }
    for split in config.evaluation_splits:
        selected = frame.filter(pl.col("research_split") == split)
        results["splits"][split] = {}
        for horizon in HORIZONS:
            selected_horizon = _purge_overlapping_split_returns(selected, horizon)
            development_horizon = _purge_overlapping_split_returns(
                development, horizon
            )
            returns = selected_horizon[f"future_return_{horizon}d"].to_numpy().astype(float)
            event_means = (
                development_horizon.group_by("event_type")
                .agg(pl.col(f"future_return_{horizon}d").mean().alias("event_type_signal"))
            )
            evaluated = selected_horizon.join(
                event_means, on="event_type", how="left"
            ).with_columns(pl.col("event_type_signal").fill_null(0.0))
            horizon_result = {
                name: _metrics(evaluated[name].to_numpy().astype(float), returns)
                for name in (*signal_names, "event_type_signal")
            }
            results["splits"][split][f"{horizon}d"] = horizon_result
    config_hash = hashlib.sha256(stable_json(config.model_dump(mode="json")).encode()).hexdigest()
    root = data_root / "results" / f"baselines_{config_hash[:16]}"
    store = ArtifactStore(data_root, duckdb_path)
    store.write_parquet(frame.drop("content"), root / "baseline_events.parquet")
    output = store.write_json(results, root / "metrics.json")
    return output
