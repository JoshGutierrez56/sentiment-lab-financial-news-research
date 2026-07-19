"""Read-only comparison of local and additional OpenAI calibration outputs."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.hybrid.analysis import HORIZONS, _corr, _horizon_metrics


class CalibrationAnalysisConfig(BaseModel):
    """Hash-locked inputs for a development/validation-only comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    articles_path: Path
    local_classifications_path: Path
    additional_sample_path: Path
    openai_classifications_path: Path
    splits_path: Path
    expected_hashes: dict[str, str]
    bootstrap_samples: int = Field(default=1000, ge=100, le=10_000)
    random_seed: int = 20260718


def _predictive_metrics(
    frame: pl.DataFrame,
    *,
    prefix: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    renamed = frame.rename(
        {
            f"{prefix}_sentiment_score": "sentiment_score",
            f"{prefix}_sentiment_label": "sentiment_label",
            f"{prefix}_confidence": "confidence",
            f"{prefix}_materiality": "materiality",
            f"{prefix}_novelty": "novelty",
            f"{prefix}_tradable": "tradable",
            f"{prefix}_abstain": "abstain",
        }
    ).with_columns(
        (
            pl.col("sentiment_score")
            * pl.col("confidence")
            * pl.col("materiality")
        ).alias("sentiment_confidence_materiality")
    )
    return {
        split: {
            f"{horizon}d": _horizon_metrics(
                renamed.filter(pl.col("research_split") == split),
                signal_column="sentiment_confidence_materiality",
                horizon=horizon,
                threshold=0.0,
                bootstrap_samples=bootstrap_samples,
                seed=seed + horizon + (0 if split == "development" else 1000),
            )
            for horizon in HORIZONS
        }
        for split in ("development", "validation")
    }


def run_calibration_analysis(
    config: CalibrationAnalysisConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> tuple[Path, Path]:
    """Compare the new calibration pair without reading holdout observations."""

    paths = {
        "articles": config.articles_path,
        "local": config.local_classifications_path,
        "sample": config.additional_sample_path,
        "openai": config.openai_classifications_path,
        "splits": config.splits_path,
    }
    for name, path in paths.items():
        if file_sha256(path) != config.expected_hashes.get(name):
            raise RuntimeError(f"Calibration analysis input hash mismatch: {name}")

    sample_ids = pl.read_parquet(config.additional_sample_path, columns=["article_id"])
    splits = pl.read_parquet(config.splits_path, columns=["article_id", "research_split"])
    selected_splits = sample_ids.join(splits, on="article_id", validate="1:1")
    if selected_splits.height > 250:
        raise RuntimeError("Additional calibration analysis may not exceed 250 articles")
    if set(selected_splits["research_split"].to_list()) - {"development", "validation"}:
        raise RuntimeError("Additional calibration analysis may not access holdout articles")

    article_columns = [
        "article_id",
        "ticker",
        "sector",
        "entry_date",
        *[f"future_return_{horizon}d" for horizon in HORIZONS],
    ]
    articles = pl.read_parquet(config.articles_path, columns=article_columns)
    common = [
        "article_id",
        "sentiment_score",
        "sentiment_label",
        "confidence",
        "materiality",
        "novelty",
        "event_type",
        "tradable",
        "abstain",
    ]
    local = pl.read_parquet(config.local_classifications_path, columns=common).rename(
        {name: f"local_{name}" for name in common if name != "article_id"}
    )
    openai = pl.read_parquet(config.openai_classifications_path, columns=common).rename(
        {name: f"openai_{name}" for name in common if name != "article_id"}
    )
    frame = (
        selected_splits.join(articles, on="article_id", validate="1:1")
        .join(local, on="article_id", validate="1:1")
        .join(openai, on="article_id", validate="1:1")
        .sort("entry_date", "ticker", "article_id")
    )
    if frame.height != selected_splits.height:
        raise RuntimeError("Additional calibration outputs are incomplete")

    local_scores = frame["local_sentiment_score"].to_numpy().astype(float)
    openai_scores = frame["openai_sentiment_score"].to_numpy().astype(float)
    agreement = {
        "exact_sentiment_label": float(
            np.mean(
                frame["local_sentiment_label"].to_numpy()
                == frame["openai_sentiment_label"].to_numpy()
            )
        ),
        "sentiment_score": _corr(local_scores, openai_scores),
        "bullish_bearish_directional": None,
        "bullish_bearish_overlap_n": 0,
        "tradable": float(
            np.mean(
                frame["local_tradable"].to_numpy()
                == frame["openai_tradable"].to_numpy()
            )
        ),
        "abstain": float(
            np.mean(
                frame["local_abstain"].to_numpy()
                == frame["openai_abstain"].to_numpy()
            )
        ),
        "event_type": float(
            np.mean(
                frame["local_event_type"].to_numpy()
                == frame["openai_event_type"].to_numpy()
            )
        ),
    }
    local_labels = frame["local_sentiment_label"].to_numpy()
    openai_labels = frame["openai_sentiment_label"].to_numpy()
    directional = np.isin(local_labels, ["bullish", "bearish"]) & np.isin(
        openai_labels, ["bullish", "bearish"]
    )
    agreement["bullish_bearish_overlap_n"] = int(np.sum(directional))
    if np.any(directional):
        agreement["bullish_bearish_directional"] = float(
            np.mean(local_labels[directional] == openai_labels[directional])
        )

    metrics: dict[str, Any] = {
        "definition": (
            "Return-blind additional calibration selection; development and validation only. "
            "OpenAI is a reference model, not assumed ground truth."
        ),
        "count": frame.height,
        "split_counts": dict(Counter(frame["research_split"].to_list())),
        "agreement": agreement,
        "label_counts": {
            "local": dict(Counter(frame["local_sentiment_label"].to_list())),
            "openai": dict(Counter(frame["openai_sentiment_label"].to_list())),
        },
        "coverage": {
            "local_tradable": float(cast(float, frame["local_tradable"].mean())),
            "openai_tradable": float(cast(float, frame["openai_tradable"].mean())),
        },
        "predictive": {
            "local": _predictive_metrics(
                frame,
                prefix="local",
                bootstrap_samples=config.bootstrap_samples,
                seed=config.random_seed,
            ),
            "openai": _predictive_metrics(
                frame,
                prefix="openai",
                bootstrap_samples=config.bootstrap_samples,
                seed=config.random_seed + 10_000,
            ),
        },
        "input_hashes": config.expected_hashes,
    }
    config_hash = hashlib.sha256(
        stable_json(config.model_dump(mode="json")).encode()
    ).hexdigest()
    metrics["config_hash"] = config_hash
    root = data_root / "results" / f"calibration_analysis_{config_hash[:16]}"
    store = ArtifactStore(data_root, duckdb_path)
    comparisons_path = store.write_parquet(frame, root / "comparisons.parquet")
    metrics_path = store.write_json(metrics, root / "metrics.json")
    store.register_parquet_view("hybrid_additional_calibration_comparison", comparisons_path)
    return metrics_path, comparisons_path
