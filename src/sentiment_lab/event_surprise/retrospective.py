"""Frozen, one-shot event-surprise portfolio retrospective.

The implementation deliberately separates article-level prediction evidence from
portfolio evidence.  It fits one edge scale on the development split, applies a
predeclared execution-cost hurdle, and never uses validation or holdout outcomes
to choose a threshold, holding period, or cost assumption.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sentiment_lab.event_surprise.signals import strongest_qualifying_event_per_company_day
from sentiment_lab.execution.costs import (
    BorrowCostModel,
    CommissionModel,
    CostBreakdown,
    MarketImpactModel,
    ResearchCostModel,
    SlippageModel,
    SpreadModel,
)
from sentiment_lab.redesign.experiment import RedesignConfig


class CostScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commission_per_share_usd: float = Field(ge=0)
    commission_minimum_usd: float = Field(ge=0)
    half_spread_bps_per_side: float = Field(ge=0)
    slippage_bps_per_side: float = Field(ge=0)
    market_impact_coefficient: float = Field(ge=0)
    annual_short_borrow_rate: float = Field(ge=0)


class PromotionGates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_holdout_trades: int = Field(ge=1)
    minimum_holdout_base_net_sharpe: float
    minimum_holdout_bootstrap_ci_lower: float
    minimum_holdout_conservative_net_sharpe: float
    require_positive_validation_base_net_return: bool
    maximum_single_ticker_absolute_pnl_share: float = Field(gt=0, le=1)


class RetrospectiveSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_column: str
    edge_target_column: str
    primary_split: str
    diagnostic_splits: tuple[str, ...]
    holding_sessions: int = Field(ge=1)
    bootstrap_iterations: int = Field(ge=100)
    bootstrap_block_length_sessions: int = Field(ge=1)
    starting_capital_usd: float = Field(gt=0)
    position_weight: float = Field(gt=0, le=1)
    maximum_gross_exposure: float = Field(gt=0, le=2)
    maximum_long_exposure: float = Field(gt=0, le=1)
    maximum_short_exposure: float = Field(gt=0, le=1)
    maximum_volume_participation: float = Field(gt=0, le=1)
    minimum_absolute_signal: float = Field(ge=0)
    cost_safety_multiple: float = Field(ge=0)
    research_cost_per_entry_usd: float = Field(ge=0)
    annualization_sessions: int = Field(ge=1)
    base: CostScenario
    conservative: CostScenario
    gates: PromotionGates
    random_seed: int

    @classmethod
    def from_config(cls, config: RedesignConfig) -> RetrospectiveSpecification:
        calibration = config.model["edge_calibration"]
        return cls(
            signal_column=str(config.feature_generation["signal_column"]),
            edge_target_column=str(calibration["target"]),
            primary_split=str(config.validation["primary_split"]),
            diagnostic_splits=tuple(config.validation["diagnostic_splits"]),
            holding_sessions=int(config.execution["holding_sessions"]),
            bootstrap_iterations=int(config.validation["bootstrap_iterations"]),
            bootstrap_block_length_sessions=int(
                config.validation["bootstrap_block_length_sessions"]
            ),
            starting_capital_usd=float(config.execution["starting_capital_usd"]),
            position_weight=float(config.execution["position_weight"]),
            maximum_gross_exposure=float(config.execution["maximum_gross_exposure"]),
            maximum_long_exposure=float(config.execution["maximum_long_exposure"]),
            maximum_short_exposure=float(config.execution["maximum_short_exposure"]),
            maximum_volume_participation=float(config.execution["maximum_volume_participation"]),
            minimum_absolute_signal=float(config.signal["minimum_absolute_signal"]),
            cost_safety_multiple=float(config.costs["cost_safety_multiple"]),
            research_cost_per_entry_usd=float(config.costs["research_cost_per_entry_usd"]),
            annualization_sessions=int(config.portfolio["annualization_sessions"]),
            base=CostScenario.model_validate(config.costs["base"]),
            conservative=CostScenario.model_validate(config.costs["conservative"]),
            gates=PromotionGates.model_validate(config.validation["gates"]),
            random_seed=config.random_seed,
        )


@dataclass(frozen=True)
class PricePoint:
    ticker: str
    date: date
    open: float
    close: float
    adjusted_close: float
    volume: float

    @property
    def adjusted_open(self) -> float:
        if self.close <= 0:
            raise ValueError("close must be positive to compute adjusted open")
        return self.open * self.adjusted_close / self.close


@dataclass(frozen=True)
class AcceptedTrade:
    article_id: str
    story_cluster_id: str
    ticker: str
    split: str
    signal: float
    direction: int
    expected_return: float
    predicted_edge_usd: float
    entry: PricePoint
    exit: PricePoint
    path: tuple[PricePoint, ...]
    notional_usd: float
    entry_shares: float
    volume_limited: bool


@dataclass(frozen=True)
class RetrospectiveResult:
    frames: dict[str, pl.DataFrame]
    metrics: dict[str, Any]


def fit_development_edge_slope(
    events: pl.DataFrame,
    *,
    signal_column: str,
    target_column: str,
) -> float:
    """Fit a no-intercept edge scale using development outcomes only."""
    development = events.filter(pl.col("research_split") == "development").drop_nulls(
        [signal_column, target_column]
    )
    if not development.height:
        raise ValueError("Development data are required for edge calibration")
    x = development[signal_column].to_numpy().astype(float)
    y = development[target_column].to_numpy().astype(float)
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        raise ValueError("Development signal has no variance around zero")
    return float(np.dot(x, y) / denominator)


def _price_history(prices: pl.DataFrame) -> dict[str, tuple[PricePoint, ...]]:
    required = {"ticker", "date", "open", "close", "adjusted_close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price fields: {sorted(missing)}")
    histories: dict[str, list[PricePoint]] = defaultdict(list)
    for row in prices.select(sorted(required)).iter_rows(named=True):
        point = PricePoint(
            ticker=str(row["ticker"]),
            date=row["date"],
            open=float(row["open"]),
            close=float(row["close"]),
            adjusted_close=float(row["adjusted_close"]),
            volume=float(row["volume"]),
        )
        if min(point.open, point.close, point.adjusted_close, point.volume) > 0:
            histories[point.ticker].append(point)
    return {
        ticker: tuple(sorted(points, key=lambda point: point.date))
        for ticker, points in histories.items()
    }


def _holding_path(
    histories: dict[str, tuple[PricePoint, ...]],
    ticker: str,
    entry_date: date,
    holding_sessions: int,
) -> tuple[PricePoint, ...] | None:
    history = histories.get(ticker, ())
    start = next((index for index, point in enumerate(history) if point.date == entry_date), None)
    if start is None:
        return None
    path = history[start : start + holding_sessions]
    return path if len(path) == holding_sessions else None


def _transaction_costs(
    scenario: CostScenario,
    *,
    quantity: float,
    price: float,
    volume: float,
    maximum_volume_participation: float,
    research_cost: float = 0.0,
) -> CostBreakdown:
    return CostBreakdown(
        commissions=CommissionModel(
            scenario.commission_per_share_usd, scenario.commission_minimum_usd
        ).estimate(quantity, price),
        half_spread=SpreadModel(scenario.half_spread_bps_per_side).estimate(quantity, price),
        slippage=SlippageModel(scenario.slippage_bps_per_side).estimate(quantity, price),
        volume_impact=MarketImpactModel(
            maximum_volume_participation, scenario.market_impact_coefficient
        ).estimate(quantity, price, volume),
        research_cost=research_cost,
    )


def _estimated_round_trip_costs(
    scenario: CostScenario,
    spec: RetrospectiveSpecification,
    *,
    direction: int,
    notional: float,
    shares: float,
    point: PricePoint,
) -> CostBreakdown:
    one_way = _transaction_costs(
        scenario,
        quantity=shares,
        price=point.open,
        volume=point.volume,
        maximum_volume_participation=spec.maximum_volume_participation,
    )
    doubled = one_way.add(one_way)
    return CostBreakdown(
        commissions=doubled.commissions,
        half_spread=doubled.half_spread,
        slippage=doubled.slippage,
        volume_impact=doubled.volume_impact,
        short_borrow=BorrowCostModel(scenario.annual_short_borrow_rate).estimate(
            notional if direction < 0 else 0.0, spec.holding_sessions
        ),
        research_cost=ResearchCostModel(spec.research_cost_per_entry_usd).estimate(1),
    )


def _frame(rows: list[dict[str, Any]], schema: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)


def _float_scalar(value: Any) -> float:
    if value is None:
        raise ValueError("Expected a finite numeric scalar")
    return float(value)


def _sharpe(values: np.ndarray, annualization: int) -> float | None:
    if values.size < 2:
        return None
    volatility = float(np.std(values, ddof=1))
    if volatility <= 0:
        return None
    return float(np.mean(values) / volatility * math.sqrt(annualization))


def _performance(frame: pl.DataFrame, return_column: str, annualization: int) -> dict[str, Any]:
    values = frame[return_column].to_numpy().astype(float)
    if not values.size:
        return {
            "sessions": 0,
            "total_return": 0.0,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "maximum_drawdown": None,
        }
    equity = np.cumprod(1.0 + values)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    annualized_volatility = (
        float(np.std(values, ddof=1) * math.sqrt(annualization)) if values.size > 1 else None
    )
    return {
        "sessions": int(values.size),
        "total_return": float(equity[-1] - 1.0),
        "annualized_return": float(np.mean(values) * annualization),
        "annualized_volatility": annualized_volatility,
        "sharpe": _sharpe(values, annualization),
        "maximum_drawdown": float(np.min(drawdowns)),
    }


def _block_bootstrap_sharpe_interval(
    values: np.ndarray,
    *,
    iterations: int,
    block_length: int,
    annualization: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if values.size < 2 or float(np.std(values, ddof=1)) <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    length = values.size
    block = min(block_length, length)
    starts = np.arange(0, length - block + 1)
    samples: list[float] = []
    blocks_needed = math.ceil(length / block)
    for _ in range(iterations):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in selected])[:length]
        result = _sharpe(sample, annualization)
        if result is not None:
            samples.append(result)
    if not samples:
        return None, None
    lower, upper = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(lower), float(upper)


def _safe_spearman(frame: pl.DataFrame, signal_column: str, target_column: str) -> float | None:
    complete = frame.drop_nulls([signal_column, target_column])
    if complete.height < 3:
        return None
    value = complete.select(pl.corr(signal_column, target_column, method="spearman")).item()
    return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None


def run_retrospective(
    signals: pl.DataFrame,
    articles: pl.DataFrame,
    prices: pl.DataFrame,
    spec: RetrospectiveSpecification,
) -> RetrospectiveResult:
    """Run the frozen portfolio once and return canonical evidence frames."""
    required_signals = {
        "article_id",
        "story_cluster_id",
        "ticker",
        "entry_date",
        "research_split",
        "abstain",
        spec.signal_column,
        spec.edge_target_column,
    }
    missing = required_signals - set(signals.columns)
    if missing:
        raise ValueError(f"Missing retrospective signal fields: {sorted(missing)}")
    events = strongest_qualifying_event_per_company_day(signals).filter(
        pl.col(spec.signal_column).abs() > spec.minimum_absolute_signal
    )
    article_entry = articles.select("article_id", "entry_adjusted_open").unique("article_id")
    events = events.join(article_entry, on="article_id", how="left", validate="1:1")
    all_splits = (*spec.diagnostic_splits, spec.primary_split)
    split_starts = {
        split: cast(date, signals.filter(pl.col("research_split") == split)["entry_date"].min())
        for split in all_splits
    }
    next_split_starts = {
        split: split_starts[all_splits[index + 1]] for index, split in enumerate(all_splits[:-1])
    }
    histories = _price_history(prices)
    development_next_start = next_split_starts.get("development")
    if development_next_start is None:
        raise ValueError("A chronological split after development is required")
    purged_development_ids: list[str] = []
    for row in events.filter(pl.col("research_split") == "development").iter_rows(named=True):
        path = _holding_path(
            histories,
            str(row["ticker"]),
            row["entry_date"],
            spec.holding_sessions,
        )
        if path is not None and path[-1].date < development_next_start:
            purged_development_ids.append(str(row["article_id"]))
    slope = fit_development_edge_slope(
        events.filter(pl.col("article_id").is_in(purged_development_ids)),
        signal_column=spec.signal_column,
        target_column=spec.edge_target_column,
    )
    events = events.with_columns(
        (pl.col(spec.signal_column) * slope).alias("development_fitted_expected_return"),
        pl.col(spec.signal_column).abs().alias("_signal_strength"),
    ).sort(
        ["research_split", "entry_date", "_signal_strength", "article_id"],
        descending=[False, False, True, False],
    )
    accepted: list[AcceptedTrade] = []
    prediction_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    active_by_split: dict[str, list[AcceptedTrade]] = defaultdict(list)

    for row in events.iter_rows(named=True):
        split = str(row["research_split"])
        ticker = str(row["ticker"])
        article_id = str(row["article_id"])
        entry_date = row["entry_date"]
        signal = float(row[spec.signal_column])
        expected_return = float(row["development_fitted_expected_return"])
        direction = 1 if signal > 0 else -1
        prediction = {
            "article_id": article_id,
            "story_cluster_id": str(row["story_cluster_id"]),
            "ticker": ticker,
            "entry_date": entry_date,
            "research_split": split,
            "signal": signal,
            "expected_return": expected_return,
            "realized_return_5d": row[spec.edge_target_column],
            "accepted": False,
            "rejection_reason": None,
        }
        path = _holding_path(histories, ticker, entry_date, spec.holding_sessions)
        reason: str | None = None
        if path is None:
            reason = "missing_holding_path"
        elif split in next_split_starts and path[-1].date >= next_split_starts[split]:
            reason = "split_boundary"
        elif not math.isclose(
            path[0].adjusted_open,
            float(row["entry_adjusted_open"]),
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            reason = "entry_price_mismatch"
        active = [trade for trade in active_by_split[split] if trade.exit.date >= entry_date]
        active_by_split[split] = active
        if reason is None and any(trade.ticker == ticker for trade in active):
            reason = "same_ticker_overlap"
        desired_notional = spec.starting_capital_usd * spec.position_weight
        if reason is None and path is not None:
            side_used = sum(trade.notional_usd for trade in active if trade.direction == direction)
            gross_used = sum(trade.notional_usd for trade in active)
            side_limit = spec.starting_capital_usd * (
                spec.maximum_long_exposure if direction > 0 else spec.maximum_short_exposure
            )
            gross_limit = spec.starting_capital_usd * spec.maximum_gross_exposure
            capacity = min(side_limit - side_used, gross_limit - gross_used)
            liquidity = path[0].volume * path[0].open * spec.maximum_volume_participation
            notional = max(0.0, min(desired_notional, capacity, liquidity))
            if notional <= 0:
                reason = "portfolio_capacity"
            else:
                shares = notional / path[0].open
                estimated_costs = _estimated_round_trip_costs(
                    spec.base,
                    spec,
                    direction=direction,
                    notional=notional,
                    shares=shares,
                    point=path[0],
                )
                predicted_edge = abs(expected_return) * notional
                if slope <= 0:
                    reason = "development_calibration_not_directionally_positive"
                elif predicted_edge < spec.cost_safety_multiple * estimated_costs.total:
                    reason = "insufficient_cost_buffer"
                else:
                    trade = AcceptedTrade(
                        article_id=article_id,
                        story_cluster_id=str(row["story_cluster_id"]),
                        ticker=ticker,
                        split=split,
                        signal=signal,
                        direction=direction,
                        expected_return=expected_return,
                        predicted_edge_usd=predicted_edge,
                        entry=path[0],
                        exit=path[-1],
                        path=path,
                        notional_usd=notional,
                        entry_shares=shares,
                        volume_limited=notional < desired_notional,
                    )
                    accepted.append(trade)
                    active_by_split[split].append(trade)
                    prediction["accepted"] = True
        prediction["rejection_reason"] = reason
        prediction_rows.append(prediction)
        if reason is not None:
            rejection_rows.append(
                {
                    "article_id": article_id,
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "research_split": split,
                    "direction": direction,
                    "reason": reason,
                }
            )

    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for trade in accepted:
        for order_type, point, quantity in (
            ("entry", trade.entry, trade.direction * trade.entry_shares),
            (
                "exit",
                trade.exit,
                -trade.direction
                * (trade.notional_usd * trade.exit.adjusted_close / trade.entry.adjusted_open)
                / trade.exit.close,
            ),
        ):
            order = {
                "article_id": trade.article_id,
                "ticker": trade.ticker,
                "research_split": trade.split,
                "date": point.date,
                "order_type": order_type,
                "quantity": quantity,
                "price": point.open if order_type == "entry" else point.close,
            }
            order_rows.append(order)
            fill_rows.append({**order, "status": "filled"})

        daily_borrow: dict[str, list[float]] = {"base": [], "conservative": []}
        for scenario_name, scenario in (
            ("base", spec.base),
            ("conservative", spec.conservative),
        ):
            entry_costs = _transaction_costs(
                scenario,
                quantity=trade.entry_shares,
                price=trade.entry.open,
                volume=trade.entry.volume,
                maximum_volume_participation=spec.maximum_volume_participation,
                research_cost=spec.research_cost_per_entry_usd,
            )
            exit_notional = (
                trade.notional_usd * trade.exit.adjusted_close / trade.entry.adjusted_open
            )
            exit_shares = exit_notional / trade.exit.close
            exit_costs = _transaction_costs(
                scenario,
                quantity=exit_shares,
                price=trade.exit.close,
                volume=trade.exit.volume,
                maximum_volume_participation=spec.maximum_volume_participation,
            )
            borrow_values = [
                BorrowCostModel(scenario.annual_short_borrow_rate).estimate(
                    trade.notional_usd * point.adjusted_close / trade.entry.adjusted_open
                    if trade.direction < 0
                    else 0.0,
                    1,
                )
                for point in trade.path
            ]
            daily_borrow[scenario_name] = borrow_values
            combined = entry_costs.add(exit_costs).add(
                CostBreakdown(short_borrow=float(sum(borrow_values)))
            )
            cost_rows.append(
                {
                    "article_id": trade.article_id,
                    "ticker": trade.ticker,
                    "research_split": trade.split,
                    "scenario": scenario_name,
                    "commissions": combined.commissions,
                    "half_spread": combined.half_spread,
                    "slippage": combined.slippage,
                    "volume_impact": combined.volume_impact,
                    "short_borrow": combined.short_borrow,
                    "research_cost": combined.research_cost,
                    "total": combined.total,
                }
            )

        previous_adjusted_price = trade.entry.adjusted_open
        for index, point in enumerate(trade.path):
            gross_pnl = (
                trade.direction
                * trade.notional_usd
                * (point.adjusted_close - previous_adjusted_price)
                / trade.entry.adjusted_open
            )
            previous_adjusted_price = point.adjusted_close
            marked_notional = trade.notional_usd * point.adjusted_close / trade.entry.adjusted_open
            base_cost = daily_borrow["base"][index]
            conservative_cost = daily_borrow["conservative"][index]
            if index == 0:
                for scenario_name, scenario in (
                    ("base", spec.base),
                    ("conservative", spec.conservative),
                ):
                    entry = _transaction_costs(
                        scenario,
                        quantity=trade.entry_shares,
                        price=trade.entry.open,
                        volume=trade.entry.volume,
                        maximum_volume_participation=spec.maximum_volume_participation,
                        research_cost=spec.research_cost_per_entry_usd,
                    ).total
                    if scenario_name == "base":
                        base_cost += entry
                    else:
                        conservative_cost += entry
            if index == len(trade.path) - 1:
                exit_notional = marked_notional
                exit_shares = exit_notional / trade.exit.close
                for scenario_name, scenario in (
                    ("base", spec.base),
                    ("conservative", spec.conservative),
                ):
                    exit_cost = _transaction_costs(
                        scenario,
                        quantity=exit_shares,
                        price=trade.exit.close,
                        volume=trade.exit.volume,
                        maximum_volume_participation=spec.maximum_volume_participation,
                    ).total
                    if scenario_name == "base":
                        base_cost += exit_cost
                    else:
                        conservative_cost += exit_cost
            position_rows.append(
                {
                    "article_id": trade.article_id,
                    "ticker": trade.ticker,
                    "research_split": trade.split,
                    "date": point.date,
                    "direction": trade.direction,
                    "signal": trade.signal,
                    "notional_usd": trade.notional_usd,
                    "marked_notional_usd": marked_notional,
                    "gross_pnl_usd": gross_pnl,
                    "base_cost_usd": base_cost,
                    "conservative_cost_usd": conservative_cost,
                    "base_net_pnl_usd": gross_pnl - base_cost,
                    "conservative_net_pnl_usd": gross_pnl - conservative_cost,
                }
            )

    predictions = _frame(
        prediction_rows,
        {
            "article_id": pl.String,
            "story_cluster_id": pl.String,
            "ticker": pl.String,
            "entry_date": pl.Date,
            "research_split": pl.String,
            "signal": pl.Float64,
            "expected_return": pl.Float64,
            "realized_return_5d": pl.Float64,
            "accepted": pl.Boolean,
            "rejection_reason": pl.String,
        },
    )
    orders = _frame(
        order_rows,
        {
            "article_id": pl.String,
            "ticker": pl.String,
            "research_split": pl.String,
            "date": pl.Date,
            "order_type": pl.String,
            "quantity": pl.Float64,
            "price": pl.Float64,
        },
    )
    fills = _frame(
        fill_rows,
        {
            **orders.schema,
            "status": pl.String,
        },
    )
    rejected = _frame(
        rejection_rows,
        {
            "article_id": pl.String,
            "ticker": pl.String,
            "entry_date": pl.Date,
            "research_split": pl.String,
            "direction": pl.Int64,
            "reason": pl.String,
        },
    )
    positions = _frame(
        position_rows,
        {
            "article_id": pl.String,
            "ticker": pl.String,
            "research_split": pl.String,
            "date": pl.Date,
            "direction": pl.Int64,
            "signal": pl.Float64,
            "notional_usd": pl.Float64,
            "marked_notional_usd": pl.Float64,
            "gross_pnl_usd": pl.Float64,
            "base_cost_usd": pl.Float64,
            "conservative_cost_usd": pl.Float64,
            "base_net_pnl_usd": pl.Float64,
            "conservative_net_pnl_usd": pl.Float64,
        },
    )
    cost_breakdown = _frame(
        cost_rows,
        {
            "article_id": pl.String,
            "ticker": pl.String,
            "research_split": pl.String,
            "scenario": pl.String,
            "commissions": pl.Float64,
            "half_spread": pl.Float64,
            "slippage": pl.Float64,
            "volume_impact": pl.Float64,
            "short_borrow": pl.Float64,
            "research_cost": pl.Float64,
            "total": pl.Float64,
        },
    )

    fill_turnover = (
        fills.with_columns((pl.col("quantity").abs() * pl.col("price")).alias("turnover_usd"))
        .group_by(["research_split", "date"])
        .agg(pl.col("turnover_usd").sum())
        if fills.height
        else pl.DataFrame(
            schema={"research_split": pl.String, "date": pl.Date, "turnover_usd": pl.Float64}
        )
    )
    daily_rows: list[dict[str, Any]] = []
    split_metrics: dict[str, Any] = {}
    calendar = sorted({point.date for history in histories.values() for point in history})
    for split_index, split in enumerate(all_splits):
        split_start = split_starts[split]
        if split in next_split_starts:
            split_calendar = [
                day for day in calendar if split_start <= day < next_split_starts[split]
            ]
        else:
            candidate_exit_dates = [
                path[-1].date
                for row in events.filter(pl.col("research_split") == split).iter_rows(named=True)
                if (
                    path := _holding_path(
                        histories,
                        str(row["ticker"]),
                        row["entry_date"],
                        spec.holding_sessions,
                    )
                )
                is not None
            ]
            split_end = max(candidate_exit_dates, default=split_start)
            split_calendar = [day for day in calendar if split_start <= day <= split_end]
        split_positions = positions.filter(pl.col("research_split") == split)
        aggregates = (
            split_positions.group_by("date").agg(
                pl.col("gross_pnl_usd").sum(),
                pl.col("base_net_pnl_usd").sum(),
                pl.col("conservative_net_pnl_usd").sum(),
                pl.when(pl.col("direction") > 0)
                .then(pl.col("marked_notional_usd"))
                .otherwise(0.0)
                .sum()
                .alias("long_exposure_usd"),
                pl.when(pl.col("direction") < 0)
                .then(pl.col("marked_notional_usd"))
                .otherwise(0.0)
                .sum()
                .alias("short_exposure_usd"),
            )
            if split_positions.height
            else pl.DataFrame()
        )
        by_date = {row["date"]: row for row in aggregates.iter_rows(named=True)}
        turnover_by_date = {
            row["date"]: float(row["turnover_usd"])
            for row in fill_turnover.filter(pl.col("research_split") == split).iter_rows(named=True)
        }
        for day in split_calendar:
            values = by_date.get(day, {})
            gross_pnl = float(values.get("gross_pnl_usd", 0.0))
            base_net_pnl = float(values.get("base_net_pnl_usd", 0.0))
            conservative_net_pnl = float(values.get("conservative_net_pnl_usd", 0.0))
            long_exposure = float(values.get("long_exposure_usd", 0.0))
            short_exposure = float(values.get("short_exposure_usd", 0.0))
            daily_rows.append(
                {
                    "research_split": split,
                    "date": day,
                    "gross_return": gross_pnl / spec.starting_capital_usd,
                    "base_net_return": base_net_pnl / spec.starting_capital_usd,
                    "conservative_net_return": conservative_net_pnl / spec.starting_capital_usd,
                    "turnover": turnover_by_date.get(day, 0.0) / spec.starting_capital_usd,
                    "gross_exposure": (long_exposure + short_exposure) / spec.starting_capital_usd,
                    "net_exposure": (long_exposure - short_exposure) / spec.starting_capital_usd,
                }
            )
        split_daily = _frame(
            [row for row in daily_rows if row["research_split"] == split],
            {
                "research_split": pl.String,
                "date": pl.Date,
                "gross_return": pl.Float64,
                "base_net_return": pl.Float64,
                "conservative_net_return": pl.Float64,
                "turnover": pl.Float64,
                "gross_exposure": pl.Float64,
                "net_exposure": pl.Float64,
            },
        )
        lower, upper = _block_bootstrap_sharpe_interval(
            split_daily["base_net_return"].to_numpy().astype(float),
            iterations=spec.bootstrap_iterations,
            block_length=spec.bootstrap_block_length_sessions,
            annualization=spec.annualization_sessions,
            seed=spec.random_seed + split_index,
        )
        split_trades = [trade for trade in accepted if trade.split == split]
        split_costs = cost_breakdown.filter(pl.col("research_split") == split)
        ticker_pnl = (
            split_positions.group_by("ticker")
            .agg(pl.col("base_net_pnl_usd").sum())
            .with_columns(pl.col("base_net_pnl_usd").abs().alias("absolute_pnl"))
            if split_positions.height
            else pl.DataFrame()
        )
        absolute_total = (
            _float_scalar(ticker_pnl["absolute_pnl"].sum()) if ticker_pnl.height else 0.0
        )
        concentration = (
            _float_scalar(ticker_pnl["absolute_pnl"].max()) / absolute_total
            if absolute_total > 0
            else None
        )
        split_events = events.filter(pl.col("research_split") == split)
        split_predictions = predictions.filter(pl.col("research_split") == split)
        evaluation_predictions = split_predictions.filter(
            ~pl.col("rejection_reason").is_in(
                ["missing_holding_path", "split_boundary", "entry_price_mismatch"]
            )
            | pl.col("rejection_reason").is_null()
        )
        split_metrics[split] = {
            "candidate_events": split_events.height,
            "evaluation_events": evaluation_predictions.height,
            "accepted_trades": len(split_trades),
            "long_trades": sum(trade.direction > 0 for trade in split_trades),
            "short_trades": sum(trade.direction < 0 for trade in split_trades),
            "event_signal_spearman_5d": _safe_spearman(
                evaluation_predictions, "signal", "realized_return_5d"
            ),
            "gross": _performance(split_daily, "gross_return", spec.annualization_sessions),
            "base_net": _performance(split_daily, "base_net_return", spec.annualization_sessions),
            "conservative_net": _performance(
                split_daily, "conservative_net_return", spec.annualization_sessions
            ),
            "base_net_sharpe_block_bootstrap_95pct": {"lower": lower, "upper": upper},
            "average_daily_turnover": (
                _float_scalar(split_daily["turnover"].mean()) if split_daily.height else 0.0
            ),
            "maximum_gross_exposure": (
                _float_scalar(split_daily["gross_exposure"].max()) if split_daily.height else 0.0
            ),
            "maximum_absolute_net_exposure": (
                _float_scalar(split_daily["net_exposure"].abs().max())
                if split_daily.height
                else 0.0
            ),
            "single_ticker_absolute_pnl_share": concentration,
            "base_cost_usd": (
                float(split_costs.filter(pl.col("scenario") == "base")["total"].sum())
                if split_costs.height
                else 0.0
            ),
            "conservative_cost_usd": (
                float(split_costs.filter(pl.col("scenario") == "conservative")["total"].sum())
                if split_costs.height
                else 0.0
            ),
        }

    daily_returns = _frame(
        daily_rows,
        {
            "research_split": pl.String,
            "date": pl.Date,
            "gross_return": pl.Float64,
            "base_net_return": pl.Float64,
            "conservative_net_return": pl.Float64,
            "turnover": pl.Float64,
            "gross_exposure": pl.Float64,
            "net_exposure": pl.Float64,
        },
    )
    primary = split_metrics[spec.primary_split]
    validation = split_metrics.get("validation", {})
    lower = primary["base_net_sharpe_block_bootstrap_95pct"]["lower"]
    concentration = primary["single_ticker_absolute_pnl_share"]
    gate_results = {
        "minimum_holdout_trades": primary["accepted_trades"] >= spec.gates.minimum_holdout_trades,
        "minimum_holdout_base_net_sharpe": (
            primary["base_net"]["sharpe"] is not None
            and primary["base_net"]["sharpe"] >= spec.gates.minimum_holdout_base_net_sharpe
        ),
        "minimum_holdout_bootstrap_ci_lower": (
            lower is not None and lower > spec.gates.minimum_holdout_bootstrap_ci_lower
        ),
        "minimum_holdout_conservative_net_sharpe": (
            primary["conservative_net"]["sharpe"] is not None
            and primary["conservative_net"]["sharpe"]
            >= spec.gates.minimum_holdout_conservative_net_sharpe
        ),
        "positive_validation_base_net_return": (
            not spec.gates.require_positive_validation_base_net_return
            or validation.get("base_net", {}).get("total_return", 0.0) > 0
        ),
        "maximum_single_ticker_absolute_pnl_share": (
            concentration is not None
            and concentration <= spec.gates.maximum_single_ticker_absolute_pnl_share
        ),
    }
    metrics: dict[str, Any] = {
        "status": "promoted" if all(gate_results.values()) else "not_promoted",
        "interpretation": (
            "All predeclared promotion gates passed."
            if all(gate_results.values())
            else "The event-surprise strategy failed at least one predeclared promotion gate."
        ),
        "holdout_disclosure": (
            "Holdout event-level IC had been viewed before this portfolio specification was frozen; "
            "this is a one-shot final retrospective, not a pristine confirmatory test."
        ),
        "implementation_audit": (
            "A pre-canonical verification run exposed split-boundary handling that retained an "
            "earlier split's boundary day and discarded valid terminal-holdout exits. The runner "
            "was corrected to purge outcomes reaching the next split and retain terminal holdout "
            "paths. No horizon, signal threshold, weight, cost, gate, or model was changed."
        ),
        "development_edge_slope": slope,
        "specification": spec.model_dump(mode="json"),
        "selection_counts": {
            "article_rows": signals.height,
            "qualifying_company_day_events": events.height,
            "purged_development_edge_fit_events": len(purged_development_ids),
            "accepted_trades": len(accepted),
            "rejections": rejected.group_by("reason").len().sort("reason").to_dicts(),
        },
        "splits": split_metrics,
        "promotion_gates": gate_results,
        "all_promotion_gates_passed": all(gate_results.values()),
    }
    return RetrospectiveResult(
        frames={
            "predictions.parquet": predictions.drop("_signal_strength", strict=False),
            "orders.parquet": orders,
            "fills.parquet": fills,
            "rejected_orders.parquet": rejected,
            "positions.parquet": positions,
            "daily_returns.parquet": daily_returns,
            "cost_breakdown.parquet": cost_breakdown,
        },
        metrics=metrics,
    )
