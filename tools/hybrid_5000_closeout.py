"""Bounded, cache-only closeout diagnostics for the hybrid 5,000 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.hybrid.analysis import _horizon_metrics, _purge_overlapping_split_returns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "results" / "hybrid_5000_closeout"

PATHS = {
    "canonical_classifications": DATA
    / "results/hybrid_local_3c4cdaf2fd9d9a16/classifications.parquet",
    "reproducibility_manifest": DATA / "results/hybrid_local_3c4cdaf2fd9d9a16/reproducibility.json",
    "recovered_original_classifications": OUTPUT / "expected_classifications_recovered.parquet",
    "mismatch_comparison": OUTPUT / "parquet_mismatch_comparison.json",
    "development_validation_events": DATA / "results/prediction_84c8d119cbebb62c/events.parquet",
    "holdout_events": DATA / "results/prediction_8855ba6a467e9a46/events.parquet",
    "corrected_holdout_metrics": DATA / "results/prediction_8855ba6a467e9a46/metrics.json",
    "original_openai_250": DATA
    / "normalized/calibration/openai_calibration_v1/calibration.parquet",
    "original_local_benchmark": DATA
    / "results/local_benchmark_4fbd283166050ee9/classifications.parquet",
    "additional_comparisons": DATA
    / "results/calibration_analysis_afe836eaa2aee842/comparisons.parquet",
    "portfolio_metrics": DATA / "results/portfolio_ab46d89f86695a40/metrics.json",
}

EXPECTED_HASHES = {
    "canonical_classifications": "a1bd6afa5d015b17c16412c1116342cb2203f64f1a4d14d102d8d9bae7180df7",
    "recovered_original_classifications": "43bdba5e10e5337ecb676e829bdf28633f304b29e3c131baaf8df6fb0167202e",
    "development_validation_events": "f2f036ec07960bff7b8afac56864c5df91abfc0904e9bda603c2da1c2426d272",
    "holdout_events": "50d6dbdc467c19d1f9228a3cd3ce11f9cbd2f842d1783d743828d6b21e934fa7",
    "original_openai_250": "ee34c656c394d79d2bb1b9f698398ec7fffa8987dba1c403f0427db5de843928",
    "original_local_benchmark": "6adec90d834ecad766a0e0a8ae3061e64c7e8120cd4b69ec485ed64ec08a6866",
    "additional_comparisons": "1db5277de15588764e18ff6fb41fba10fa0964a09586ba3cdd75bc3866169687",
    "portfolio_metrics": "77ad5d12bb06e9f35814dafcec2f2c6adce84923a29580d41ca6db3b337e3231",
}

OPERATIONAL_COLUMNS = [
    "from_cache",
    "prompt_tokens",
    "output_tokens",
    "total_duration_ns",
    "eval_duration_ns",
]
SPLITS = ("development", "validation", "holdout")
HORIZONS = (5, 21)
SIGNAL = "sentiment_confidence"
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 20260718


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _verify_inputs() -> dict[str, str]:
    actual = {name: file_sha256(PATHS[name]) for name in EXPECTED_HASHES}
    for name, expected in EXPECTED_HASHES.items():
        if actual[name] != expected:
            raise RuntimeError(f"Immutable closeout input hash mismatch: {name}")
    old = pl.read_parquet(PATHS["recovered_original_classifications"])
    canonical = pl.read_parquet(PATHS["canonical_classifications"])
    if not old.drop(OPERATIONAL_COLUMNS).equals(canonical):
        raise RuntimeError(
            "Canonical rebuild is not scientifically equal to original metrics input"
        )
    return actual


def _distribution(frame: pl.DataFrame, column: str, threshold: float) -> dict[str, Any]:
    series = frame[column]
    return {
        "mean": float(series.mean()),
        "p25": float(series.quantile(0.25, interpolation="linear")),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75, interpolation="linear")),
        f"fraction_at_least_{threshold}": float((series >= threshold).mean()),
    }


def _permissiveness() -> dict[str, Any]:
    benchmark = pl.read_parquet(PATHS["original_local_benchmark"]).filter(
        pl.col("model_id") == "qwen3.6:35b-a3b|Q4_K_M"
    )
    if benchmark.height != 250:
        raise RuntimeError("Expected one 250-row qwen3.6 benchmark")
    agreed = benchmark.filter(pl.col("local_sentiment_label") == pl.col("sentiment_label"))
    summaries: dict[str, Any] = {}
    for measure, threshold in (("relevance", 0.7), ("confidence", 0.7), ("materiality", 0.5)):
        local = _distribution(agreed, f"local_{measure}", threshold)
        openai = _distribution(agreed, measure, threshold)
        summaries[measure] = {
            "local": local,
            "openai": openai,
            "paired_mean_difference_local_minus_openai": float(
                (agreed[f"local_{measure}"] - agreed[measure]).mean()
            ),
        }
    cross = {
        "both_tradable": agreed.filter(pl.col("local_tradable") & pl.col("tradable")).height,
        "local_only_tradable": agreed.filter(pl.col("local_tradable") & ~pl.col("tradable")).height,
        "openai_only_tradable": agreed.filter(
            ~pl.col("local_tradable") & pl.col("tradable")
        ).height,
        "both_abstained": agreed.filter(pl.col("local_abstain") & pl.col("abstain")).height,
    }
    canonical = pl.read_parquet(PATHS["canonical_classifications"])
    original_openai = pl.read_parquet(PATHS["original_openai_250"])
    local_250_coverage = float(benchmark["local_tradable"].mean())
    openai_250_coverage = float(original_openai["tradable"].mean())
    local_5000_coverage = float(canonical["tradable"].mean())
    return {
        "restriction": (
            "All paired distribution comparisons use only exact three-way directional-label "
            "agreement (bullish/neutral/bearish) between local and OpenAI."
        ),
        "directionally_agreed_n": agreed.height,
        "coverage_within_agreement": {
            "local_tradable": float(agreed["local_tradable"].mean()),
            "openai_tradable": float(agreed["tradable"].mean()),
            "local_abstention": float(agreed["local_abstain"].mean()),
            "openai_abstention": float(agreed["abstain"].mean()),
            "cross_counts": cross,
        },
        "distributions": summaries,
        "coverage_gap_decomposition_percentage_points": {
            "openai_original_250": 100 * openai_250_coverage,
            "local_same_original_250": 100 * local_250_coverage,
            "local_hybrid_5000": 100 * local_5000_coverage,
            "model_policy_gap_same_sample": 100 * (local_250_coverage - openai_250_coverage),
            "sample_selection_shift_same_local_model": 100
            * (local_5000_coverage - local_250_coverage),
        },
        "diagnosis": (
            "The local model is not numerically overconfident relative to OpenAI. It is "
            "systematically more permissive through lower abstention and much higher "
            "relevance/materiality assessments; the company-specific 5,000-sample design "
            "adds a separate selection-driven coverage increase."
        ),
    }


def _trim_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metric.get(key)
        for key in (
            "n",
            "average_signed_return",
            "bullish_minus_bearish_spread",
            "spearman_ic",
            "pearson_ic",
            "directional_accuracy",
            "company_cluster_bootstrap_95_ci",
            "date_block_bootstrap_95_ci",
        )
    }


def _evaluate_subset(frame: pl.DataFrame, seed_offset: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split_index, split in enumerate(SPLITS):
        split_frame = frame.filter(pl.col("research_split") == split)
        if split_frame.is_empty():
            continue
        output[split] = {}
        for horizon in HORIZONS:
            metric = _horizon_metrics(
                _purge_overlapping_split_returns(split_frame, horizon),
                signal_column=SIGNAL,
                horizon=horizon,
                threshold=0.0,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                seed=RANDOM_SEED + seed_offset + split_index * 100 + horizon,
            )
            output[split][f"{horizon}d"] = _trim_metric(metric)
    return output


def _subsets() -> dict[str, Any]:
    frame = pl.concat(
        [
            pl.read_parquet(PATHS["development_validation_events"]),
            pl.read_parquet(PATHS["holdout_events"]),
        ]
    )
    base = frame.filter(pl.col("tradable") & ~pl.col("abstain"))
    strongest = (
        base.with_columns(pl.col(SIGNAL).abs().alias("_strength"))
        .sort("_strength", descending=True, nulls_last=True)
        .group_by(["research_split", "ticker", "entry_date"], maintain_order=True)
        .first()
        .drop("_strength")
    )
    definitions = {
        "exclude_event_type_other": base.filter(pl.col("event_type") != "other"),
        "high_relevance_ge_0_7": base.filter(pl.col("relevance") >= 0.7),
        "high_materiality_ge_0_5": base.filter(pl.col("materiality") >= 0.5),
        "high_confidence_ge_0_7": base.filter(pl.col("confidence") >= 0.7),
        "strongest_event_per_company_day": strongest,
        "earnings_and_guidance_only": base.filter(
            pl.col("event_type").is_in(["earnings", "guidance"])
        ),
    }
    additional = pl.read_parquet(PATHS["additional_comparisons"]).filter(
        pl.col("local_sentiment_label").is_in(["bullish", "bearish"])
        & (pl.col("local_sentiment_label") == pl.col("openai_sentiment_label"))
        & pl.col("local_tradable")
        & ~pl.col("local_abstain")
        & pl.col("openai_tradable")
        & ~pl.col("openai_abstain")
    )
    definitions["local_openai_directional_agreement"] = additional.with_columns(
        (pl.col("local_sentiment_score") * pl.col("local_confidence")).alias(SIGNAL),
        pl.col("local_sentiment_label").alias("sentiment_label"),
        pl.col("local_tradable").alias("tradable"),
        pl.col("local_abstain").alias("abstain"),
    )
    return {
        "label": (
            "EXPLORATORY PREREGISTERED DIAGNOSTICS ONLY; none is a validated strategy, "
            "and no threshold was selected from holdout returns."
        ),
        "fixed_rules": {
            "signal": SIGNAL,
            "horizons": list(HORIZONS),
            "confidence_threshold": 0.7,
            "relevance_threshold": 0.7,
            "materiality_threshold": 0.5,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "results": {
            name: _evaluate_subset(values, index * 1000)
            for index, (name, values) in enumerate(definitions.items(), start=1)
        },
    }


def _cost_attribution() -> dict[str, Any]:
    portfolio = _load_json(PATHS["portfolio_metrics"])
    holding = portfolio["splits"]["holdout"]["5d"]
    market_neutral = holding["market_neutral"]
    daily = pl.read_parquet(ROOT / market_neutral["daily_returns_path"])
    accepted = int(holding["accepted_positions"])
    suppressed = int(holding["suppressed_overlapping_same_ticker_events"])
    additive_gross = float(market_neutral["long_contribution"]) + float(
        market_neutral["short_contribution"]
    )
    base_cost = float(market_neutral["cost_drag_base"])
    active = daily.filter(pl.col("position_count") > 0)
    return {
        "gross_to_net": {
            "gross_sharpe": market_neutral["gross"]["sharpe"],
            "base_net_sharpe": market_neutral["base_net"]["sharpe"],
            "gross_total_return": market_neutral["gross"]["total_return"],
            "base_net_total_return": market_neutral["base_net"]["total_return"],
        },
        "turnover": market_neutral["turnover"],
        "bid_ask_slippage": {
            "separately_identifiable": False,
            "reason": (
                "The engine applies one combined 10 bps one-way friction to turnover; "
                "it does not allocate that bucket among spread, slippage, or commission."
            ),
        },
        "commissions": {
            "separately_identifiable": False,
            "additional_modeled_amount": 0.0,
        },
        "shorting_costs": {
            "borrow_fee_modeled": 0.0,
            "locate_fee_modeled": 0.0,
        },
        "combined_base_friction": {
            "one_way_bps": 10.0,
            "total_cost_drag": base_cost,
            "total_cost_drag_bps": 10_000 * base_cost,
        },
        "average_per_accepted_position": {
            "additive_gross_return": additive_gross / accepted,
            "additive_gross_return_bps": 10_000 * additive_gross / accepted,
            "base_cost": base_cost / accepted,
            "base_cost_bps": 10_000 * base_cost / accepted,
        },
        "gross_return_per_unit_turnover_bps": 10_000 * additive_gross / market_neutral["turnover"],
        "holding_period_overlap": {
            "accepted_positions": accepted,
            "suppressed_same_ticker_events": suppressed,
            "suppressed_fraction_of_candidate_events": suppressed / (accepted + suppressed),
            "holding_sessions": 5,
        },
        "active_positions": {
            "calendar_days": daily.height,
            "active_trading_days": market_neutral["active_trading_days"],
            "average_all_calendar_days": market_neutral["average_positions"],
            "average_active_days": float(active["position_count"].mean()),
            "maximum": market_neutral["maximum_positions"],
            "total_position_days": int(daily["position_count"].sum()),
        },
        "books": {
            "long_additive_contribution": market_neutral["long_contribution"],
            "short_additive_contribution": market_neutral["short_contribution"],
            "long_contribution_bps": 10_000 * market_neutral["long_contribution"],
            "short_contribution_bps": 10_000 * market_neutral["short_contribution"],
        },
        "diagnosis": (
            "The signal earned about the same gross return per dollar turned over as the "
            "fixed one-way cost assumption. Turnover therefore consumed essentially all "
            "five-session gross performance before any borrow or separately modeled fees."
        ),
    }


def main() -> None:
    hashes = _verify_inputs()
    mismatch = _load_json(PATHS["mismatch_comparison"])
    corrected = _load_json(PATHS["corrected_holdout_metrics"])
    results = {
        "scope": "Bounded closeout; no inference, OpenAI calls, tuning, expansion, or options work.",
        "reproducibility": {
            "status": "PASS",
            "canonical_rebuild_manifest": str(PATHS["reproducibility_manifest"].relative_to(ROOT)),
            "verified_input_hashes": hashes,
            "mismatch_comparison": mismatch,
            "exact_original_metrics_source": (
                "The 28-column artifact with SHA-256 43bdba5e... generated the original "
                "downstream metrics; it is recovered byte-for-byte from the two locked "
                "prediction event artifacts."
            ),
            "canonical_equivalence": (
                "Dropping the five execution-only columns from the recovered original "
                "artifact equals the 23-column permanent-cache rebuild cell-for-cell."
            ),
        },
        "corrected_frozen_primary_holdout": corrected["frozen_primary_specification"],
        "permissiveness": _permissiveness(),
        "five_session_cost_attribution": _cost_attribution(),
        "exploratory_subsets": _subsets(),
        "closeout_decision": "REDESIGN",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "closeout_results.json"
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(destination.relative_to(ROOT))
    print(file_sha256(destination))


if __name__ == "__main__":
    main()
