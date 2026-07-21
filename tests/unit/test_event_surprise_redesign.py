from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from sentiment_lab.baselines.finance_local import benchmark_predictions, required_benchmark_fields
from sentiment_lab.baselines.finbert import FinBERTAdapter, FinBERTPrediction, cache_key, text_hash
from sentiment_lab.event_surprise.retrospective import (
    CostScenario,
    PromotionGates,
    RetrospectiveSpecification,
    fit_development_edge_slope,
    run_retrospective,
)
from sentiment_lab.event_surprise.schemas import (
    EventSurpriseAssessment,
    EventType,
    SurpriseDirection,
)
from sentiment_lab.event_surprise.signals import (
    add_event_signals,
    fit_normalizer,
    strongest_qualifying_event_per_company_day,
)
from sentiment_lab.execution.costs import (
    BorrowCostModel,
    CommissionModel,
    LocateAvailabilityModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
)
from sentiment_lab.execution.engine import ExecutionModels, StatefulPortfolioEngine, Target
from sentiment_lab.redesign.experiment import RedesignConfig, write_cache_only_run
from sentiment_lab.redesign.regime import market_regime_multiplier
from sentiment_lab.validation.purged_cv import purged_walk_forward_folds


class _Cache:
    def __init__(self) -> None:
        self.rows: dict[str, FinBERTPrediction] = {}

    def get(self, key: str) -> FinBERTPrediction | None:
        return self.rows.get(key)

    def put(self, key: str, prediction: FinBERTPrediction) -> None:
        self.rows[key] = prediction


def test_finbert_cache_only_and_key() -> None:
    cache = _Cache()
    adapter = FinBERTAdapter(cache, model_revision="r1")
    article = {"article_id": "a", "title": "Good quarter", "content": "details"}
    digest = text_hash(article["title"], article["content"], mode="headline")
    key = cache_key(
        article_id="a", text_digest=digest, model="ProsusAI/finbert", revision="r1", mode="headline"
    )
    with pytest.raises(RuntimeError, match="absent"):
        adapter.score([article])
    cache.put(
        key,
        FinBERTPrediction(
            "a", 0.7, 0.2, 0.1, 0.6, "ProsusAI/finbert", "r1", "main", digest, "now", 1, "headline"
        ),
    )
    assert adapter.score([article])[0].finbert_score == 0.6
    assert digest != text_hash(article["title"], article["content"], mode="full_text")


def test_cached_finance_model_benchmark_contract() -> None:
    frame = pl.DataFrame(
        {
            "local_label": ["bullish", "bearish"],
            "openai_label": ["bullish", "bullish"],
            "local_abstain": [False, True],
            "openai_abstain": [False, False],
            "score": [0.8, -0.2],
            "return": [0.02, -0.01],
        }
    )
    result = benchmark_predictions(frame, score_column="score", return_column="return")
    assert result["n"] == 2
    assert result["sentiment_label_agreement"] == 0.5
    assert "structured_valid" in required_benchmark_fields()
    with pytest.raises(ValueError, match="Missing"):
        benchmark_predictions(pl.DataFrame(), score_column="score", return_column="return")


def test_sparse_event_schema_rejects_generic_and_accepts_surprise() -> None:
    with pytest.raises(ValueError, match="must abstain"):
        EventSurpriseAssessment(
            company_specificity=1,
            event_type=EventType.other,
            surprise_direction=SurpriseDirection.positive,
            surprise_magnitude=0.8,
            direction_score=0.8,
            confidence=0.9,
            relevance=0.9,
            materiality=0.9,
            novelty=0.9,
            already_priced_in=0.1,
            expected_horizon="5d",
            abstain=False,
        )
    accepted = EventSurpriseAssessment(
        primary_company="Acme",
        primary_ticker="ACME",
        company_specificity=0.9,
        event_type=EventType.earnings,
        actual_information="EPS beat",
        expected_or_prior_information="Consensus lower",
        surprise_direction=SurpriseDirection.positive,
        surprise_magnitude=0.8,
        direction_score=0.8,
        confidence=0.9,
        relevance=0.9,
        materiality=0.9,
        novelty=0.9,
        already_priced_in=0.1,
        expected_horizon="5d",
        abstain=False,
    )
    assert accepted.primary_ticker == "ACME"


def test_development_normalization_and_one_event_per_day() -> None:
    frame = pl.DataFrame(
        {
            "article_id": ["a", "b", "c"],
            "ticker": ["A", "A", "B"],
            "entry_date": ["2026-01-01"] * 2 + ["2026-01-02"],
            "research_split": ["development", "development", "validation"],
            "llm_direction_score": [0.1, 0.5, 0.3],
            "finbert_score": [0.0, 0.1, 0.2],
            "direction_score": [0.1, 0.5, 0.3],
            "company_specificity": [1.0] * 3,
            "materiality": [1.0] * 3,
            "novelty": [1.0] * 3,
            "confidence": [1.0] * 3,
            "abstain": [False, False, True],
        }
    )
    output = add_event_signals(frame, fit_normalizer(frame, "llm_direction_score"))
    assert output.filter(pl.col("abstain"))["event_surprise_signal"][0] == 0
    assert strongest_qualifying_event_per_company_day(output).height == 1


def test_company_day_selection_uses_absolute_signal_strength() -> None:
    frame = pl.DataFrame(
        {
            "article_id": ["negative", "positive"],
            "ticker": ["A", "A"],
            "entry_date": ["2026-01-02", "2026-01-02"],
            "event_surprise_signal": [-0.8, 0.3],
            "abstain": [False, False],
        }
    )
    selected = strongest_qualifying_event_per_company_day(frame)
    assert selected["article_id"].to_list() == ["negative"]


def _retrospective_spec() -> RetrospectiveSpecification:
    costs = CostScenario(
        commission_per_share_usd=0,
        commission_minimum_usd=0,
        half_spread_bps_per_side=0,
        slippage_bps_per_side=0,
        market_impact_coefficient=0,
        annual_short_borrow_rate=0,
    )
    return RetrospectiveSpecification(
        signal_column="event_surprise_signal",
        edge_target_column="future_return_5d",
        primary_split="holdout",
        diagnostic_splits=("development", "validation"),
        holding_sessions=5,
        bootstrap_iterations=100,
        bootstrap_block_length_sessions=2,
        starting_capital_usd=1_000_000,
        position_weight=0.02,
        maximum_gross_exposure=1,
        maximum_long_exposure=0.5,
        maximum_short_exposure=0.5,
        maximum_volume_participation=0.01,
        minimum_absolute_signal=0,
        cost_safety_multiple=0,
        research_cost_per_entry_usd=0,
        annualization_sessions=252,
        base=costs,
        conservative=costs,
        gates=PromotionGates(
            minimum_holdout_trades=1,
            minimum_holdout_base_net_sharpe=0,
            minimum_holdout_bootstrap_ci_lower=-100,
            minimum_holdout_conservative_net_sharpe=0,
            require_positive_validation_base_net_return=True,
            maximum_single_ticker_absolute_pnl_share=1,
        ),
        random_seed=7,
    )


def test_development_edge_fit_and_stateful_retrospective() -> None:
    split_dates = {
        "development": [datetime(2025, 1, day).date() for day in range(2, 7)],
        "validation": [datetime(2025, 2, day).date() for day in range(3, 8)],
        "holdout": [datetime(2025, 3, day).date() for day in range(3, 8)],
    }
    rows: list[dict[str, object]] = []
    articles: list[dict[str, object]] = []
    prices: list[dict[str, object]] = []
    for index, (split, dates) in enumerate(split_dates.items()):
        article_id = f"event-{index}"
        rows.append(
            {
                "article_id": article_id,
                "story_cluster_id": f"story-{index}",
                "ticker": f"T{index}",
                "entry_date": dates[0],
                "research_split": split,
                "abstain": False,
                "event_surprise_signal": 1.0,
                "future_return_5d": 0.05,
            }
        )
        rows.append(
            {
                "article_id": f"boundary-{index}",
                "story_cluster_id": f"boundary-story-{index}",
                "ticker": f"B{index}",
                "entry_date": dates[-1],
                "research_split": split,
                "abstain": True,
                "event_surprise_signal": 0.0,
                "future_return_5d": None,
            }
        )
        articles.append({"article_id": article_id, "entry_adjusted_open": 100.0})
        articles.append({"article_id": f"boundary-{index}", "entry_adjusted_open": 100.0})
        for day_index, day in enumerate(dates):
            close = 101.0 + day_index
            prices.append(
                {
                    "ticker": f"T{index}",
                    "date": day,
                    "open": 100.0 if day_index == 0 else close - 1,
                    "high": close,
                    "low": close - 1,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000_000.0,
                }
            )
    signal_frame = pl.DataFrame(rows)
    selected = strongest_qualifying_event_per_company_day(signal_frame)
    assert fit_development_edge_slope(
        selected,
        signal_column="event_surprise_signal",
        target_column="future_return_5d",
    ) == pytest.approx(0.05)
    result = run_retrospective(
        signal_frame,
        pl.DataFrame(articles),
        pl.DataFrame(prices),
        _retrospective_spec(),
    )
    assert result.metrics["selection_counts"]["accepted_trades"] == 3
    assert result.frames["orders.parquet"].height == 6
    assert result.frames["positions.parquet"].height == 15
    assert result.metrics["splits"]["holdout"]["base_net"]["total_return"] > 0


def test_purged_walk_forward_removes_overlapping_labels() -> None:
    times = [datetime(2026, 1, day) for day in range(1, 7)]
    ends = [time + timedelta(days=2) for time in times]
    folds = purged_walk_forward_folds(times, ends, n_splits=3, embargo=timedelta(days=2))
    assert len(folds) == 2
    assert all(
        ends[index] < fold.validation_start for fold in folds for index in fold.train_indices
    )


def test_inventory_netting_partial_fill_locate_and_cost_buffer() -> None:
    models = ExecutionModels(
        CommissionModel(0.01),
        SpreadModel(5),
        SlippageModel(1),
        MarketImpactModel(0.1, 0.1),
        BorrowCostModel(),
        LocateAvailabilityModel(frozenset({"LONG"})),
    )
    engine = StatefulPortfolioEngine(models)
    state = engine.rebalance(
        [Target("LONG", 100, 100, "s1"), Target("SHORT", -50, 100, "s2")],
        {"LONG": 10, "SHORT": 10},
        {"LONG": 500, "SHORT": 500},
    )
    assert state.positions["LONG"] == 50
    assert any(order.reason == "locate_unavailable" for order in state.rejected)
    state = engine.rebalance([Target("LONG", 50, 100, "s3")], {"LONG": 10}, {"LONG": 500})
    assert len(state.fills) == 1


def test_regime_and_canonical_run_artifacts(tmp_path: Path) -> None:
    assert market_regime_multiplier([100.0] * 104) == 1.0
    config = RedesignConfig(
        dataset={},
        feature_generation={},
        model={},
        signal={},
        validation={},
        execution={},
        costs={},
        portfolio={},
        reporting={},
    )
    paths = write_cache_only_run(
        config,
        root=tmp_path / "runs",
        repository_root=Path.cwd(),
        data_hashes={},
        feature_hashes={},
        model_metadata={},
        frames={},
        metrics={"exploratory": True},
    )
    assert (paths.root / "resolved_config.yaml").exists()
    assert (paths.root / "operational.json").exists()
    with pytest.raises(FileExistsError):
        write_cache_only_run(
            config,
            root=tmp_path / "runs",
            repository_root=Path.cwd(),
            data_hashes={},
            feature_hashes={},
            model_metadata={},
            frames={},
            metrics={},
        )
