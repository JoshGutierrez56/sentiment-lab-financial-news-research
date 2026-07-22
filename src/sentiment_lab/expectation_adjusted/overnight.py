"""Locked historical exploratory expectation-adjusted overnight benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.expectation_adjusted.wrds_ibes import (
    MAX_OVERNIGHT_EVENTS,
    QueryRunner,
    run_wrds_ibes_eps_overnight_snapshot,
)

EXPLORATORY_LABEL = "exploratory_historical_locked_not_confirmatory"
ORIGINAL_REPO = Path(
    "C:/Users/Owner/.graphify/repos/JoshGutierrez56/"
    "DeepSeek-Generative-AI-Sentiment-Analysis-Algorithm"
)
PRIVATE_ROOT = Path("data/private/expectation_adjusted_overnight")
RESULTS_ROOT = Path("data/results/expectation_adjusted_overnight")
REPORT_PATH = Path("docs/EXPECTATION_ADJUSTED_OVERNIGHT_EXPLORATORY_REPORT.md")


class FrozenInput(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OvernightConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    study: dict[str, Any]
    immutable_inputs: dict[str, Any]
    scope: dict[str, Any]
    evaluation: dict[str, Any]
    modeling: dict[str, Any]

    @property
    def articles(self) -> FrozenInput:
        return FrozenInput.model_validate(self.immutable_inputs["articles"])

    @property
    def prices(self) -> FrozenInput:
        return FrozenInput.model_validate(self.immutable_inputs["prices"])

    @property
    def cached_text_signals(self) -> FrozenInput:
        return FrozenInput.model_validate(self.immutable_inputs["cached_text_signals"])


@dataclass(frozen=True)
class BenchmarkArtifacts:
    status: Literal["completed", "coverage_gate_unmet", "blocked"]
    manifest_path: Path
    modeling_table_path: Path | None
    metrics_path: Path | None
    report_path: Path
    coverage: dict[str, Any]
    metrics: dict[str, Any]
    limitations: list[str]


def load_overnight_config(path: Path) -> OvernightConfig:
    return OvernightConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _input_path(config: OvernightConfig, item: FrozenInput) -> Path:
    source_repo = Path(str(config.immutable_inputs.get("source_repo", ORIGINAL_REPO)))
    return source_repo / item.path


def _write_progress(repository_root: Path, payload: dict[str, Any]) -> None:
    path = repository_root / PRIVATE_ROOT / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )


def _verify_input(path: Path, expected_sha256: str) -> str:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"Immutable input hash mismatch: {path.name}")
    return observed


def frozen_universe(articles_path: Path) -> tuple[tuple[str, ...], str, dict[str, int]]:
    articles = pl.read_parquet(articles_path, columns=["article_id", "ticker"])
    tickers = tuple(
        sorted(str(value).upper() for value in articles["ticker"].drop_nulls().unique())
    )
    ticker_hash = hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest()
    return (
        tickers,
        ticker_hash,
        {
            "article_rows": articles.height,
            "distinct_article_ids": articles["article_id"].n_unique(),
            "universe_tickers": len(tickers),
        },
    )


def _price_return_features(prices: pl.DataFrame, tickers: tuple[str, ...]) -> pl.DataFrame:
    adjusted_open = pl.col("open") * pl.col("adjusted_close") / pl.col("close")
    features = (
        prices.filter(pl.col("ticker").is_in(tickers))
        .sort(["ticker", "date"])
        .with_columns(adjusted_open.alias("adjusted_open"))
        .with_columns(
            pl.col("adjusted_close").shift(-4).over("ticker").alias("_exit_close_5session"),
            pl.col("date").shift(-4).over("ticker").alias("exit_date_5session"),
            pl.col("adjusted_close").shift(5).over("ticker").alias("_lag_close_5session"),
            pl.col("adjusted_close").shift(21).over("ticker").alias("_lag_close_21session"),
        )
        .with_columns(
            (pl.col("_exit_close_5session") / pl.col("adjusted_open") - 1.0).alias(
                "asset_return_5session"
            ),
            (pl.col("adjusted_close") / pl.col("_lag_close_5session") - 1.0).alias(
                "lagged_return_5session"
            ),
            (pl.col("adjusted_close") / pl.col("_lag_close_21session") - 1.0).alias(
                "lagged_return_21session"
            ),
            (pl.col("volume").cast(pl.Float64) * pl.col("adjusted_close")).alias("dollar_volume"),
            pl.col("volume").cast(pl.Float64).log1p().alias("log_volume"),
        )
        .select(
            "ticker",
            pl.col("date").alias("entry_date"),
            "exit_date_5session",
            "asset_return_5session",
            "lagged_return_5session",
            "lagged_return_21session",
            "dollar_volume",
            "log_volume",
        )
    )
    universe_returns = (
        features.group_by("entry_date")
        .agg(pl.col("asset_return_5session").mean().alias("universe_return_5session"))
        .drop_nulls(["universe_return_5session"])
    )
    return features.join(universe_returns, on="entry_date", how="left")


def add_wrds_surprise_fields(frame: pl.DataFrame) -> pl.DataFrame:
    actual_on_basis = (
        pl.col("actual_unadjusted")
        * pl.col("cfacshr_consensus_date")
        / pl.col("cfacshr_report_date")
    )
    return frame.with_columns(
        actual_on_basis.alias("actual_on_consensus_share_basis"),
        (actual_on_basis - pl.col("consensus_mean_unadjusted")).alias("actual_minus_mean_estimate"),
        pl.when(pl.col("consensus_stdev_unadjusted") > 0)
        .then(
            (actual_on_basis - pl.col("consensus_mean_unadjusted"))
            / pl.col("consensus_stdev_unadjusted")
        )
        .otherwise(None)
        .alias("standardized_eps_surprise"),
        (
            pl.col("consensus_stdev_unadjusted").is_null()
            | (pl.col("consensus_stdev_unadjusted") <= 0)
        ).alias("missing_dispersion"),
        (pl.col("contributor_count").is_null() | (pl.col("contributor_count") <= 0)).alias(
            "missing_contributor_count"
        ),
        (pl.col("revisions_up").fill_null(0) + pl.col("revisions_down").fill_null(0)).alias(
            "revision_count"
        ),
        (pl.col("revisions_up").is_null() & pl.col("revisions_down").is_null()).alias(
            "missing_revision_count"
        ),
    )


def deterministic_news_match(wrds: pl.DataFrame, cached_signals: pl.DataFrame) -> pl.DataFrame:
    wrds_events = (
        add_wrds_surprise_fields(wrds)
        .with_columns(
            pl.col("actual_announce_date").cast(pl.Date),
            pl.col("actual_activation_date").cast(pl.Date),
        )
        .with_row_index("wrds_event_id")
        .with_columns(
            pl.max_horizontal("actual_announce_date", "actual_activation_date").alias("_event_date")
        )
    )
    signal_columns = [
        "ticker",
        "entry_date",
        "article_id",
        "article_hash",
        "company_specificity",
        "direction_score",
        "confidence",
        "relevance",
        "materiality",
        "novelty",
        "already_priced_in",
        "finbert_score",
        "llm_direction_score",
        "calibrated_llm_score",
        "llm_minus_finbert_residual",
        "llm_finbert_disagreement",
        "event_surprise_confidence",
        "event_surprise_confidence_materiality",
        "event_surprise_score",
        "event_surprise_signal",
        "abstain",
        "event_type",
    ]
    candidates = (
        wrds_events.join(
            cached_signals.select(signal_columns).with_columns(pl.col("entry_date").cast(pl.Date)),
            left_on="source_ticker",
            right_on="ticker",
            how="inner",
        )
        .filter(
            (pl.col("entry_date") > pl.col("_event_date"))
            & (pl.col("entry_date") <= pl.col("_event_date") + pl.duration(days=2))
        )
        .sort(["wrds_event_id", "entry_date", "article_id"])
    )
    return (
        candidates.group_by("wrds_event_id", maintain_order=True)
        .first()
        .with_columns(pl.col("source_ticker").alias("ticker"))
    )


def _modeling_table(
    *,
    wrds_snapshot: Path,
    articles_path: Path,
    prices_path: Path,
    cached_signals_path: Path,
    tickers: tuple[str, ...],
) -> pl.DataFrame:
    wrds = pl.read_parquet(wrds_snapshot)
    cached_signals = pl.read_parquet(cached_signals_path)
    articles = pl.read_parquet(articles_path, columns=["article_id", "sector"])
    price_features = _price_return_features(pl.read_parquet(prices_path), tickers)
    matched = deterministic_news_match(wrds, cached_signals)
    table = (
        matched.join(articles, on="article_id", how="left")
        .join(price_features, on=["ticker", "entry_date"], how="left")
        .with_columns(
            (pl.col("asset_return_5session") - pl.col("universe_return_5session")).alias(
                "target_residual_return_5session"
            ),
            pl.when(pl.col("entry_date") <= date(2024, 12, 31))
            .then(pl.lit("development"))
            .when(
                (pl.col("entry_date") >= date(2025, 1, 1))
                & (pl.col("entry_date") <= date(2025, 12, 31))
            )
            .then(pl.lit("validation"))
            .otherwise(pl.lit("outside_window"))
            .alias("research_split"),
            pl.col("entry_date").dt.month().cast(pl.Utf8).alias("entry_month"),
            pl.col("entry_date").dt.weekday().cast(pl.Utf8).alias("entry_weekday"),
        )
        .filter(pl.col("research_split").is_in(["development", "validation"]))
        .filter(
            pl.when(pl.col("research_split") == "development")
            .then(pl.col("exit_date_5session") < date(2025, 1, 1))
            .otherwise(pl.col("exit_date_5session") < date(2026, 1, 1))
        )
        .drop_nulls(["target_residual_return_5session"])
    )
    return table


def coverage_counts(table: pl.DataFrame, *, wrds_rows: int, universe_count: int) -> dict[str, Any]:
    validation = table.filter(pl.col("research_split") == "validation")
    return {
        "wrds_event_rows": wrds_rows,
        "matched_observations_total": table.height,
        "development_observations": table.filter(pl.col("research_split") == "development").height,
        "validation_observations": validation.height,
        "distinct_validation_entry_dates": validation["entry_date"].n_unique()
        if validation.height
        else 0,
        "universe_tickers": universe_count,
    }


def coverage_gate_passed(coverage: dict[str, Any], config: OvernightConfig) -> bool:
    gate = config.evaluation["minimum_coverage_gate"]
    return (
        int(coverage["matched_observations_total"]) >= int(gate["matched_observations_total"])
        and int(coverage["validation_observations"]) >= int(gate["validation_observations"])
        and int(coverage["distinct_validation_entry_dates"])
        >= int(gate["distinct_validation_entry_dates"])
    )


NUMERIC_FEATURES = {
    "price_only": [
        "lagged_return_5session",
        "lagged_return_21session",
        "log_volume",
        "dollar_volume",
    ],
    "expectations_fundamentals_only": [
        "actual_minus_mean_estimate",
        "standardized_eps_surprise",
        "consensus_stdev_unadjusted",
        "contributor_count",
        "revision_count",
        "missing_dispersion",
        "missing_contributor_count",
        "missing_revision_count",
    ],
    "cached_text_only": [
        "company_specificity",
        "direction_score",
        "confidence",
        "relevance",
        "materiality",
        "novelty",
        "already_priced_in",
        "finbert_score",
        "llm_direction_score",
        "calibrated_llm_score",
        "llm_minus_finbert_residual",
        "llm_finbert_disagreement",
        "event_surprise_confidence",
        "event_surprise_confidence_materiality",
        "event_surprise_score",
        "event_surprise_signal",
        "abstain",
    ],
}
CATEGORICAL_FEATURES = {
    "sector_calendar_only": ["sector", "entry_month", "entry_weekday"],
}
MODEL_FEATURE_GROUPS = {
    "price_only": ["price_only"],
    "expectations_fundamentals_only": ["expectations_fundamentals_only"],
    "sector_calendar_only": ["sector_calendar_only"],
    "cached_text_only": ["cached_text_only"],
    "combined": [
        "price_only",
        "expectations_fundamentals_only",
        "sector_calendar_only",
        "cached_text_only",
    ],
}


def _feature_columns(model_name: str) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for group in MODEL_FEATURE_GROUPS[model_name]:
        numeric.extend(NUMERIC_FEATURES.get(group, []))
        categorical.extend(CATEGORICAL_FEATURES.get(group, []))
    return numeric, categorical


def _design_matrix(
    frame: pl.DataFrame,
    development: pl.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    matrices: list[np.ndarray] = []
    names: list[str] = []
    for column in numeric_columns:
        dev_values = development[column].cast(pl.Float64).to_numpy()
        dev_values = dev_values[np.isfinite(dev_values)]
        mean = float(np.mean(dev_values)) if len(dev_values) else 0.0
        scale = float(np.std(dev_values)) if len(dev_values) else 1.0
        if scale <= 0 or not math.isfinite(scale):
            scale = 1.0
        values = frame[column].cast(pl.Float64).to_numpy()
        values = np.where(np.isfinite(values), values, mean)
        matrices.append(((values - mean) / scale).reshape(-1, 1))
        names.append(column)
    for column in categorical_columns:
        categories = sorted(str(value) for value in development[column].drop_nulls().unique())
        raw_values = np.asarray(
            [str(value) if value is not None else "" for value in frame[column].to_list()]
        )
        for category in categories:
            matrices.append((raw_values == category).astype(float).reshape(-1, 1))
            names.append(f"{column}={category}")
    if not matrices:
        return np.zeros((frame.height, 0)), names
    return np.hstack(matrices), names


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    design = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return cast(np.ndarray, np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train)


def _predict_ridge(x: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.hstack([np.ones((x.shape[0], 1)), x]) @ coefficient)


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if len(prediction) < 3 or np.std(prediction) == 0 or np.std(target) == 0:
        return None
    value = float(spearmanr(prediction, target).statistic)
    return value if math.isfinite(value) else None


def _bootstrap_interval(
    validation: pl.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    dates = np.asarray(validation["entry_date"].cast(pl.Utf8).to_list())
    target = validation["target_residual_return_5session"].to_numpy().astype(float)
    unique_dates = np.unique(dates)
    if len(unique_dates) < 2:
        return {"draws": 0, "combined_minus_best_nontext_95_ci": [None, None]}
    rng = np.random.default_rng(seed)
    nontext = ["price_only", "expectations_fundamentals_only", "sector_calendar_only"]
    increments: list[float] = []
    for _ in range(draws):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        indices = np.concatenate([np.flatnonzero(dates == value) for value in sampled_dates])
        combined = _spearman(predictions["combined"][indices], target[indices])
        baseline_values = [
            _spearman(predictions[name][indices], target[indices]) for name in nontext
        ]
        valid_baselines = [value for value in baseline_values if value is not None]
        if combined is not None and valid_baselines:
            increments.append(combined - max(valid_baselines))
    if not increments:
        return {"draws": 0, "combined_minus_best_nontext_95_ci": [None, None]}
    lower, upper = np.quantile(np.asarray(increments), [0.025, 0.975])
    return {
        "draws": len(increments),
        "combined_minus_best_nontext_95_ci": [float(lower), float(upper)],
    }


def evaluate_nested_models(table: pl.DataFrame, config: OvernightConfig) -> dict[str, Any]:
    development = table.filter(pl.col("research_split") == "development")
    validation = table.filter(pl.col("research_split") == "validation")
    y_train = development["target_residual_return_5session"].to_numpy().astype(float)
    y_validation = validation["target_residual_return_5session"].to_numpy().astype(float)
    alpha = float(config.modeling.get("ridge_alpha", 1.0))
    model_metrics: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for model_name in MODEL_FEATURE_GROUPS:
        numeric, categorical = _feature_columns(model_name)
        x_train, feature_names = _design_matrix(
            development,
            development,
            numeric_columns=numeric,
            categorical_columns=categorical,
        )
        x_validation, _ = _design_matrix(
            validation,
            development,
            numeric_columns=numeric,
            categorical_columns=categorical,
        )
        coefficient = _fit_ridge(x_train, y_train, alpha)
        predicted = _predict_ridge(x_validation, coefficient)
        validation_predictions[model_name] = predicted
        model_metrics[model_name] = {
            "validation_spearman_ic": _spearman(predicted, y_validation),
            "feature_count": len(feature_names),
            "validation_observations": validation.height,
        }
    nontext = ["price_only", "expectations_fundamentals_only", "sector_calendar_only"]
    best_nontext = max(
        (
            (name, model_metrics[name]["validation_spearman_ic"])
            for name in nontext
            if model_metrics[name]["validation_spearman_ic"] is not None
        ),
        key=lambda item: float(item[1]),
        default=(None, None),
    )
    combined_ic = model_metrics["combined"]["validation_spearman_ic"]
    bootstrap = _bootstrap_interval(
        validation,
        validation_predictions,
        draws=int(config.evaluation["bootstrap"]["draws"]),
        seed=int(config.evaluation["bootstrap"]["seed"]),
    )
    return {
        "status": EXPLORATORY_LABEL,
        "models": model_metrics,
        "best_nontext_model": best_nontext[0],
        "best_nontext_validation_spearman_ic": best_nontext[1],
        "combined_minus_best_nontext_validation_spearman_ic": (
            None
            if combined_ic is None or best_nontext[1] is None
            else float(combined_ic) - float(best_nontext[1])
        ),
        "bootstrap": bootstrap,
        "portfolio": {
            "status": "not_run",
            "reason": "predeclared reuse gate not met for this expectation-adjusted table without changing assumptions",
        },
    }


def _safe_manifest(
    *,
    config_hash: str,
    input_hashes: dict[str, str],
    ticker_hash: str,
    coverage: dict[str, Any],
    wrds_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": EXPLORATORY_LABEL,
        "config_sha256": config_hash,
        "input_hashes": input_hashes,
        "ticker_list_sha256": ticker_hash,
        "coverage_counts": coverage,
        "wrds": {
            "rows_written": wrds_receipt.get("rows_written"),
            "query_sha256": wrds_receipt.get("query_sha256"),
            "schema_sha256": wrds_receipt.get("schema_sha256"),
            "snapshot_parquet_sha256": wrds_receipt.get("events_parquet_sha256"),
            "source_tables": wrds_receipt.get("source_tables"),
        },
    }


def write_report(
    repository_root: Path,
    *,
    status: str,
    coverage: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    limitations: list[str],
) -> Path:
    report_path = repository_root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_summary = json.dumps(metrics, indent=2, sort_keys=True, default=str, allow_nan=False)
    manifest_summary = json.dumps(
        {
            "config_sha256": manifest.get("config_sha256"),
            "input_hashes": manifest.get("input_hashes"),
            "ticker_list_sha256": manifest.get("ticker_list_sha256"),
            "coverage_counts": coverage,
        },
        indent=2,
        sort_keys=True,
    )
    report_path.write_text(
        "\n".join(
            [
                "# Expectation-Adjusted News Overnight Exploratory Report",
                "",
                f"**Status:** {EXPLORATORY_LABEL}",
                f"**Terminal state:** {status}",
                "",
                "## PM Summary",
                "",
                "This historical exploratory run is not confirmatory and is not evidence of deployable alpha. "
                "The prior 2022-2026 news holdout was already viewed, so the only permissible decision is whether this design has enough coverage and clean enough plumbing to justify a separately frozen future study.",
                "",
                "Portfolio output was not produced because the predeclared reuse gate requires existing stateful execution and cost machinery without changing economic assumptions.",
                "",
                "## Coverage",
                "",
                "```json",
                json.dumps(coverage, indent=2, sort_keys=True),
                "```",
                "",
                "## Aggregate Metrics",
                "",
                "```json",
                metrics_summary,
                "```",
                "",
                "## Source Lineage",
                "",
                "```json",
                manifest_summary,
                "```",
                "",
                "## Open-Source Context",
                "",
                "- ProsusAI FinBERT is a finance-domain sentiment classification reference; classification success is separate from verified costed trading success: https://github.com/ProsusAI/finBERT",
                "- AI4Finance FinGPT is a financial NLP model and benchmark ecosystem; it does not make this historical benchmark a deployed trading result: https://github.com/AI4Finance-Foundation/FinGPT",
                "- Stefan Jansen's Machine Learning for Trading materials emphasize point-in-time validation, walk-forward testing, robustness, and costs; those standards are consistent with keeping this result exploratory: https://github.com/stefan-jansen/machine-learning-for-trading",
                "",
                "## Limitations",
                "",
                *[f"- {item}" for item in limitations],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def run_overnight_benchmark(
    config: OvernightConfig,
    *,
    repository_root: Path,
    wrds_runner: QueryRunner | None,
) -> BenchmarkArtifacts:
    if config.study.get("status") != EXPLORATORY_LABEL:
        raise RuntimeError("Overnight benchmark requires the exploratory locked status label")
    config_path = (
        repository_root
        / "config/experiments/expectation_adjusted_news_overnight_exploratory_v0.yaml"
    )
    config_hash = file_sha256(config_path)
    articles_path = _input_path(config, config.articles)
    prices_path = _input_path(config, config.prices)
    cached_signals_path = _input_path(config, config.cached_text_signals)
    input_hashes = {
        "articles": _verify_input(articles_path, config.articles.sha256),
        "prices": _verify_input(prices_path, config.prices.sha256),
        "cached_text_signals": _verify_input(
            cached_signals_path, config.cached_text_signals.sha256
        ),
    }
    tickers, ticker_hash, universe_counts = frozen_universe(articles_path)
    if len(tickers) != int(config.scope["required_universe_ticker_count"]):
        raise RuntimeError("Frozen article universe ticker count mismatch")

    private_root = repository_root / PRIVATE_ROOT
    results_root = repository_root / RESULTS_ROOT
    private_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    _write_progress(
        repository_root,
        {
            "status": "running",
            "phase": "hashes_and_universe_verified",
            "exploratory_label": EXPLORATORY_LABEL,
            "coverage_counts": universe_counts,
            "hashes": {**input_hashes, "ticker_list": ticker_hash, "config": config_hash},
        },
    )

    snapshot_path = private_root / "wrds_eps_snapshot.parquet"
    wrds_receipt_path = private_root / "wrds_eps_snapshot_receipt.json"
    if snapshot_path.exists() and wrds_receipt_path.exists():
        wrds_receipt = json.loads(wrds_receipt_path.read_text(encoding="utf-8"))
    elif wrds_runner is not None:
        artifacts = run_wrds_ibes_eps_overnight_snapshot(
            wrds_runner,
            repository_root=repository_root,
            output_directory=private_root,
            tickers=tickers,
            start_date=date.fromisoformat(str(config.scope["announcement_start"])),
            end_date_exclusive=date.fromisoformat(str(config.scope["announcement_end_exclusive"])),
            max_events=min(MAX_OVERNIGHT_EVENTS, int(config.scope["maximum_wrds_event_rows"])),
        )
        snapshot_path = artifacts.data_path
        wrds_receipt = artifacts.receipt
    else:
        raise RuntimeError("No private WRDS snapshot exists and no live WRDS runner was supplied")

    table = _modeling_table(
        wrds_snapshot=snapshot_path,
        articles_path=articles_path,
        prices_path=prices_path,
        cached_signals_path=cached_signals_path,
        tickers=tickers,
    )
    wrds_rows = pl.read_parquet(snapshot_path).height
    coverage = coverage_counts(table, wrds_rows=wrds_rows, universe_count=len(tickers))
    store = ArtifactStore(private_root, private_root / "overnight.duckdb")
    modeling_path = store.write_parquet(table, private_root / "joined_modeling_table.parquet")
    manifest = _safe_manifest(
        config_hash=config_hash,
        input_hashes=input_hashes,
        ticker_hash=ticker_hash,
        coverage=coverage,
        wrds_receipt=wrds_receipt,
    )
    manifest_path = store.write_json(manifest, private_root / "hash_coverage_manifest.json")
    _write_progress(
        repository_root,
        {
            "status": "running",
            "phase": "coverage_checked",
            "exploratory_label": EXPLORATORY_LABEL,
            "coverage_counts": coverage,
            "hashes": manifest,
        },
    )

    limitations = [
        "Historical exploratory benchmark only; the prior news holdout was already viewed.",
        "Raw I/B/E/S times are preserved without UTC assignment.",
        "Licensed WRDS rows and company-level EPS values remain only in ignored private artifacts.",
        "Portfolio evaluation was skipped unless the predeclared reuse gate could be met.",
    ]
    if not coverage_gate_passed(coverage, config):
        metrics = {
            "status": "coverage_gate_unmet",
            "models_fit": False,
            "reason": "minimum matched-observation validation gate was not met",
        }
        metrics_path = store.write_json(metrics, results_root / "aggregate_metrics.json")
        report_path = write_report(
            repository_root,
            status="coverage_gate_unmet",
            coverage=coverage,
            metrics=metrics,
            manifest=manifest,
            limitations=limitations,
        )
        return BenchmarkArtifacts(
            status="coverage_gate_unmet",
            manifest_path=manifest_path,
            modeling_table_path=modeling_path,
            metrics_path=metrics_path,
            report_path=report_path,
            coverage=coverage,
            metrics=metrics,
            limitations=limitations,
        )

    metrics = evaluate_nested_models(table, config)
    metrics["config_sha256"] = config_hash
    metrics["modeling_table_sha256"] = file_sha256(modeling_path)
    metrics_path = store.write_json(metrics, results_root / "aggregate_metrics.json")
    report_path = write_report(
        repository_root,
        status="completed",
        coverage=coverage,
        metrics=metrics,
        manifest=manifest,
        limitations=limitations,
    )
    return BenchmarkArtifacts(
        status="completed",
        manifest_path=manifest_path,
        modeling_table_path=modeling_path,
        metrics_path=metrics_path,
        report_path=report_path,
        coverage=coverage,
        metrics=metrics,
        limitations=limitations,
    )
