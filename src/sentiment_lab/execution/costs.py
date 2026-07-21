"""Auditable cost components; no opaque combined friction bucket."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostBreakdown:
    commissions: float = 0.0
    half_spread: float = 0.0
    slippage: float = 0.0
    volume_impact: float = 0.0
    short_borrow: float = 0.0
    research_cost: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(float(value) for value in asdict(self).values()))

    def add(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            **{key: value + getattr(other, key) for key, value in asdict(self).items()}
        )


@dataclass(frozen=True)
class CommissionModel:
    per_share: float = 0.0
    minimum: float = 0.0

    def estimate(self, quantity: float, price: float) -> float:
        del price
        return max(self.minimum, abs(quantity) * self.per_share) if quantity else 0.0


@dataclass(frozen=True)
class SpreadModel:
    half_spread_bps: float

    def estimate(self, quantity: float, price: float) -> float:
        return abs(quantity * price) * self.half_spread_bps / 10_000


@dataclass(frozen=True)
class SlippageModel:
    bps: float = 0.0

    def estimate(self, quantity: float, price: float) -> float:
        return abs(quantity * price) * self.bps / 10_000


@dataclass(frozen=True)
class MarketImpactModel:
    volume_cap: float
    price_impact_coefficient: float

    def estimate(self, quantity: float, price: float, available_volume: float) -> float:
        if available_volume <= 0:
            raise ValueError("available_volume must be positive")
        volume_share = min(abs(quantity) / available_volume, self.volume_cap)
        impact_pct = self.price_impact_coefficient * volume_share**2
        return abs(quantity * price) * impact_pct


@dataclass(frozen=True)
class BorrowCostModel:
    annual_rate: float = 0.0
    trading_days: int = 252

    def estimate(self, short_notional: float, holding_days: int) -> float:
        return abs(short_notional) * self.annual_rate * holding_days / self.trading_days


@dataclass(frozen=True)
class LocateAvailabilityModel:
    available_tickers: frozenset[str]

    def permitted(self, ticker: str, quantity: float) -> bool:
        return quantity >= 0 or ticker in self.available_tickers


@dataclass(frozen=True)
class ResearchCostModel:
    per_article: float = 0.0

    def estimate(self, article_count: int) -> float:
        return self.per_article * article_count
