"""Simple daily portfolio backtest built only from explicit marked positions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256


class PortfolioSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: str = "sentiment_confidence_materiality"
    aggregation: Literal["strongest_company_day", "company_day_aggregate"] = "strongest_company_day"
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_materiality: float = Field(default=0.0, ge=0.0, le=1.0)
    holding_periods: list[int] = Field(default_factory=lambda: [5, 21])
    maximum_company_weight: float = Field(default=0.02, gt=0.0, le=0.02)
    base_cost_bps: float = Field(default=10.0, ge=0.0)
    conservative_cost_bps: float = Field(default=25.0, ge=0.0)

    @model_validator(mode="after")
    def validate_holding_periods(self) -> PortfolioSpecification:
        if sorted(self.holding_periods) != [5, 21]:
            raise ValueError("Portfolio holding periods must be exactly 5 and 21 days")
        return self


class PortfolioRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    articles_path: Path
    classifications_path: Path
    splits_path: Path
    prices_path: Path
    expected_hashes: dict[str, str]
    evaluation_splits: list[str] = Field(default_factory=lambda: ["development", "validation"])
    specification: PortfolioSpecification = Field(default_factory=PortfolioSpecification)
    primary_specification_manifest: Path | None = None

    @model_validator(mode="after")
    def protect_holdout(self) -> PortfolioRunConfig:
        if "holdout" in self.evaluation_splits and self.primary_specification_manifest is None:
            raise ValueError("Holdout portfolio requires a frozen primary specification")
        return self


@dataclass(frozen=True)
class PositionDay:
    position_id: str
    ticker: str
    sector: str
    date: date
    direction: int
    asset_return: float


def _signal_expression(name: str) -> pl.Expr:
    expressions = {
        "raw_sentiment": pl.col("sentiment_score"),
        "sentiment_confidence": pl.col("sentiment_score") * pl.col("confidence"),
        "sentiment_confidence_materiality": (
            pl.col("sentiment_score") * pl.col("confidence") * pl.col("materiality")
        ),
        "sentiment_confidence_materiality_novelty": (
            pl.col("sentiment_score")
            * pl.col("confidence")
            * pl.col("materiality")
            * pl.col("novelty")
        ),
    }
    try:
        return expressions[name]
    except KeyError as exc:
        raise ValueError(f"Unknown portfolio signal: {name}") from exc


def _position_days(
    events: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    holding_period: int,
) -> tuple[list[PositionDay], int]:
    price_maps: dict[str, list[dict[str, Any]]] = {
        str(key[0] if isinstance(key, tuple) else key): values.sort("date").to_dicts()
        for key, values in prices.partition_by("ticker", as_dict=True).items()
    }
    date_indices = {
        ticker: {row["date"]: index for index, row in enumerate(rows)}
        for ticker, rows in price_maps.items()
    }
    output: list[PositionDay] = []
    suppressed_overlap = 0
    for ticker_values in events.partition_by("ticker", as_dict=False):
        ticker = str(ticker_values["ticker"][0])
        last_exit = -1
        for row in ticker_values.sort(["entry_date", "article_id"]).iter_rows(named=True):
            entry_index = date_indices[ticker][row["entry_date"]]
            exit_index = entry_index + holding_period - 1
            if entry_index <= last_exit:
                suppressed_overlap += 1
                continue
            rows = price_maps[ticker]
            if exit_index >= len(rows):
                continue
            direction = 1 if float(row["portfolio_signal"]) > 0 else -1
            position_id = f"{row['article_id']}:{holding_period}"
            for index in range(entry_index, exit_index + 1):
                price = rows[index]
                if index == entry_index:
                    adjusted_open = (
                        float(price["open"])
                        * float(price["adjusted_close"])
                        / float(price["close"])
                    )
                    asset_return = float(price["adjusted_close"]) / adjusted_open - 1.0
                else:
                    asset_return = (
                        float(price["adjusted_close"]) / float(rows[index - 1]["adjusted_close"])
                        - 1.0
                    )
                output.append(
                    PositionDay(
                        position_id,
                        ticker,
                        str(row["sector"]),
                        price["date"],
                        direction,
                        asset_return,
                    )
                )
            last_exit = exit_index
    return output, suppressed_overlap


def _weights(
    active: list[PositionDay], *, mode: str, maximum_company_weight: float
) -> dict[str, float]:
    companies = {item.ticker: item.direction for item in active}
    longs = sorted(ticker for ticker, direction in companies.items() if direction > 0)
    shorts = sorted(ticker for ticker, direction in companies.items() if direction < 0)
    weights: dict[str, float] = {}
    if mode == "long_only":
        weight = min(1.0 / len(longs), maximum_company_weight) if longs else 0.0
        return {ticker: weight for ticker in longs}
    if mode != "market_neutral":
        raise ValueError(f"Unknown portfolio mode: {mode}")
    if not longs or not shorts:
        return {}
    long_weight = min(0.5 / len(longs), maximum_company_weight)
    short_weight = min(0.5 / len(shorts), maximum_company_weight)
    weights.update({ticker: long_weight for ticker in longs})
    weights.update({ticker: -short_weight for ticker in shorts})
    return weights


def _performance(returns: np.ndarray) -> dict[str, float | None]:
    if not len(returns):
        return {}
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    volatility = float(np.std(returns, ddof=1) * math.sqrt(252)) if len(returns) > 1 else 0.0
    mean = float(np.mean(returns) * 252)
    downside = returns[returns < 0]
    downside_vol = float(np.std(downside, ddof=1) * math.sqrt(252)) if len(downside) > 1 else 0.0
    years = len(returns) / 252
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else None
    return {
        "total_return": float(equity[-1] - 1.0),
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": mean / volatility if volatility > 0 else None,
        "sortino": mean / downside_vol if downside_vol > 0 else None,
        "maximum_drawdown": float(np.min(drawdown)),
    }


def _daily_portfolio(
    positions: list[PositionDay],
    calendar: list[date],
    *,
    mode: str,
    maximum_company_weight: float,
    base_cost_bps: float,
    conservative_cost_bps: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    by_date: defaultdict[date, list[PositionDay]] = defaultdict(list)
    for item in positions:
        by_date[item.date].append(item)
    previous: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for day in calendar:
        active = by_date[day]
        weights = _weights(active, mode=mode, maximum_company_weight=maximum_company_weight)
        returns_by_ticker = {item.ticker: item.asset_return for item in active}
        gross_return = sum(weights[ticker] * returns_by_ticker[ticker] for ticker in weights)
        turnover = sum(
            abs(weights.get(ticker, 0.0) - previous.get(ticker, 0.0))
            for ticker in set(weights) | set(previous)
        )
        long_contribution = sum(
            weight * returns_by_ticker[ticker] for ticker, weight in weights.items() if weight > 0
        )
        short_contribution = sum(
            weight * returns_by_ticker[ticker] for ticker, weight in weights.items() if weight < 0
        )
        gross_exposure = sum(abs(value) for value in weights.values())
        concentration = (
            sum(value * value for value in weights.values()) / gross_exposure**2
            if gross_exposure
            else 0.0
        )
        rows.append(
            {
                "date": day,
                "gross_return": gross_return,
                "base_net_return": gross_return - turnover * base_cost_bps / 10_000,
                "conservative_net_return": (
                    gross_return - turnover * conservative_cost_bps / 10_000
                ),
                "turnover": turnover,
                "cost_drag_base": turnover * base_cost_bps / 10_000,
                "cost_drag_conservative": turnover * conservative_cost_bps / 10_000,
                "long_contribution": long_contribution,
                "short_contribution": short_contribution,
                "gross_exposure": gross_exposure,
                "net_exposure": sum(weights.values()),
                "position_count": len(weights),
                "exposure_hhi": concentration,
            }
        )
        previous = weights
    frame = pl.DataFrame(rows, infer_schema_length=None)
    summary = {
        "gross": _performance(frame["gross_return"].to_numpy()),
        "base_net": _performance(frame["base_net_return"].to_numpy()),
        "conservative_net": _performance(frame["conservative_net_return"].to_numpy()),
        "turnover": float(frame["turnover"].sum()),
        "cost_drag_base": float(frame["cost_drag_base"].sum()),
        "cost_drag_conservative": float(frame["cost_drag_conservative"].sum()),
        "long_contribution": float(frame["long_contribution"].sum()),
        "short_contribution": float(frame["short_contribution"].sum()),
        "active_trading_days": int((frame["position_count"] > 0).sum()),
        "average_positions": float(cast(float, frame["position_count"].mean())),
        "maximum_positions": int(cast(int, frame["position_count"].max())),
        "maximum_exposure_hhi": float(cast(float, frame["exposure_hhi"].max())),
    }
    return frame, summary


def run_portfolio_backtests(
    config: PortfolioRunConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> Path:
    """Create daily long-only and market-neutral series after predictive testing."""

    paths = {
        "articles": config.articles_path,
        "classifications": config.classifications_path,
        "splits": config.splits_path,
        "prices": config.prices_path,
    }
    for name, path in paths.items():
        if file_sha256(path) != config.expected_hashes.get(name):
            raise RuntimeError(f"Portfolio input hash mismatch: {name}")
    if "holdout" in config.evaluation_splits:
        assert config.primary_specification_manifest is not None
        manifest = json.loads(config.primary_specification_manifest.read_text(encoding="utf-8"))
        if manifest.get("frozen_before_holdout") is not True:
            raise RuntimeError("Portfolio specification was not frozen before holdout")
        if manifest.get("portfolio_specification") != config.specification.model_dump(mode="json"):
            raise RuntimeError("Portfolio specification differs from frozen primary manifest")
    articles = pl.read_parquet(config.articles_path).select(
        "article_id", "ticker", "sector", "entry_date"
    )
    classifications = pl.read_parquet(config.classifications_path).select(
        "article_id",
        "sentiment_score",
        "sentiment_label",
        "confidence",
        "relevance",
        "materiality",
        "novelty",
        "tradable",
        "abstain",
    )
    splits = pl.read_parquet(config.splits_path).select("article_id", "research_split")
    prices = pl.read_parquet(config.prices_path)
    filtered_events = (
        articles.join(classifications, on="article_id", validate="1:1")
        .join(splits, on="article_id", validate="1:1")
        .with_columns(_signal_expression(config.specification.signal).alias("portfolio_signal"))
        .filter(
            pl.col("tradable")
            & ~pl.col("abstain")
            & pl.col("sentiment_label").is_in(["bullish", "bearish"])
            & (pl.col("confidence") >= config.specification.minimum_confidence)
            & (pl.col("relevance") >= config.specification.minimum_relevance)
            & (pl.col("materiality") >= config.specification.minimum_materiality)
        )
    )
    if config.specification.aggregation == "strongest_company_day":
        events = (
            filtered_events.with_columns(pl.col("portfolio_signal").abs().alias("_strength"))
            .sort("_strength", descending=True)
            .group_by(["research_split", "ticker", "entry_date"], maintain_order=True)
            .first()
            .drop("_strength")
        )
    else:
        events = (
            filtered_events.group_by(
                ["research_split", "ticker", "entry_date"], maintain_order=True
            )
            .agg(
                pl.col("portfolio_signal").mean(),
                pl.col("article_id").first(),
                pl.col("sector").first(),
            )
            .filter(pl.col("portfolio_signal").abs() > 1e-12)
        )
    config_hash = hashlib.sha256(stable_json(config.model_dump(mode="json")).encode()).hexdigest()
    root = data_root / "results" / f"portfolio_{config_hash[:16]}"
    store = ArtifactStore(data_root, duckdb_path)
    results: dict[str, Any] = {
        "definition": (
            "Sharpe ratios use explicit daily marked portfolio returns. Same-ticker event "
            "signals arriving during an active holding period are suppressed."
        ),
        "splits": {},
    }
    for split in config.evaluation_splits:
        selected = events.filter(pl.col("research_split") == split)
        results["splits"][split] = {}
        for holding in config.specification.holding_periods:
            position_days, suppressed = _position_days(selected, prices, holding_period=holding)
            if position_days:
                first = min(item.date for item in position_days)
                last = max(item.date for item in position_days)
                calendar = sorted(
                    value for value in set(prices["date"].to_list()) if first <= value <= last
                )
            else:
                calendar = []
            holding_result: dict[str, Any] = {
                "accepted_positions": len({item.position_id for item in position_days}),
                "suppressed_overlapping_same_ticker_events": suppressed,
            }
            for mode in ("long_only", "market_neutral"):
                daily, summary = _daily_portfolio(
                    position_days,
                    calendar,
                    mode=mode,
                    maximum_company_weight=config.specification.maximum_company_weight,
                    base_cost_bps=config.specification.base_cost_bps,
                    conservative_cost_bps=config.specification.conservative_cost_bps,
                )
                path = store.write_parquet(daily, root / f"{split}_{holding}d_{mode}_daily.parquet")
                holding_result[mode] = {**summary, "daily_returns_path": str(path)}
            results["splits"][split][f"{holding}d"] = holding_result
    output = store.write_json(results, root / "metrics.json")
    return output
