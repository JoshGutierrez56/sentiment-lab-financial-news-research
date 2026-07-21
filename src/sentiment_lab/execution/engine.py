"""Small stateful portfolio engine with netting, locates, and partial fills."""

from __future__ import annotations

from dataclasses import dataclass, field

from sentiment_lab.execution.costs import (
    BorrowCostModel,
    CommissionModel,
    CostBreakdown,
    LocateAvailabilityModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
)


@dataclass(frozen=True)
class Target:
    ticker: str
    target_quantity: float
    estimated_gross_edge: float
    story_id: str


@dataclass(frozen=True)
class Fill:
    ticker: str
    quantity: float
    price: float
    costs: CostBreakdown


@dataclass(frozen=True)
class RejectedOrder:
    ticker: str
    quantity: float
    reason: str


@dataclass
class PortfolioState:
    positions: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    rejected: list[RejectedOrder] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionModels:
    commission: CommissionModel
    spread: SpreadModel
    slippage: SlippageModel
    impact: MarketImpactModel
    borrow: BorrowCostModel
    locate: LocateAvailabilityModel
    cost_safety_multiple: float = 2.0


class StatefulPortfolioEngine:
    def __init__(self, models: ExecutionModels) -> None:
        self.models = models
        self.state = PortfolioState()

    def rebalance(
        self, targets: list[Target], prices: dict[str, float], volumes: dict[str, float]
    ) -> PortfolioState:
        unique: dict[str, Target] = {}
        for target in targets:
            if target.ticker in unique:
                raise ValueError("One primary signal per company-day is required")
            unique[target.ticker] = target
        all_tickers = set(self.state.positions) | set(unique)
        for ticker in sorted(all_tickers):
            desired = unique.get(ticker, Target(ticker, 0.0, float("inf"), "exit"))
            current = self.state.positions.get(ticker, 0.0)
            quantity = desired.target_quantity - current
            if not quantity:
                continue
            if ticker not in prices or ticker not in volumes:
                self.state.rejected.append(RejectedOrder(ticker, quantity, "missing_market_data"))
                continue
            if not self.models.locate.permitted(ticker, desired.target_quantity):
                self.state.rejected.append(RejectedOrder(ticker, quantity, "locate_unavailable"))
                continue
            price, volume = prices[ticker], volumes[ticker]
            allowed = min(abs(quantity), volume * self.models.impact.volume_cap)
            filled = allowed if quantity > 0 else -allowed
            estimated = self._costs(filled, price, volume)
            if desired.estimated_gross_edge < self.models.cost_safety_multiple * estimated.total:
                self.state.rejected.append(
                    RejectedOrder(ticker, quantity, "insufficient_cost_buffer")
                )
                continue
            self.state.fills.append(Fill(ticker, filled, price, estimated))
            self.state.positions[ticker] = current + filled
            if self.state.positions[ticker] == 0:
                del self.state.positions[ticker]
            if abs(filled) < abs(quantity):
                self.state.rejected.append(
                    RejectedOrder(ticker, quantity - filled, "partial_fill_liquidity")
                )
        return self.state

    def _costs(self, quantity: float, price: float, volume: float) -> CostBreakdown:
        return CostBreakdown(
            commissions=self.models.commission.estimate(quantity, price),
            half_spread=self.models.spread.estimate(quantity, price),
            slippage=self.models.slippage.estimate(quantity, price),
            volume_impact=self.models.impact.estimate(quantity, price, volume),
        )
