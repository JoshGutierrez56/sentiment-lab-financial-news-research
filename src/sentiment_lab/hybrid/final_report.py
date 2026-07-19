"""Final hybrid report assembly and predeclared PASS/FAIL/INCONCLUSIVE decision."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentiment_lab.data.storage import ArtifactStore, file_sha256

Conclusion = Literal["PASS", "FAIL", "INCONCLUSIVE"]
ScaleRecommendation = Literal["Stop at 5,000", "Expand to 10,000", "Expand to 25,000"]


class FinalReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_manifest: Path
    benchmark_metrics: Path
    local_manifest: Path
    primary_specification: Path
    prediction_metrics: Path
    baseline_metrics: Path
    portfolio_metrics: Path
    additional_openai_manifest: Path
    expected_hashes: dict[str, str]
    original_openai_cost_usd: float = Field(default=0.348443, ge=0.0)


@dataclass(frozen=True)
class FinalReportOutput:
    report_path: Path
    results_path: Path
    conclusion: Conclusion
    scale_recommendation: ScaleRecommendation


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def research_decision(
    *,
    holdout_5d: dict[str, Any],
    holdout_21d: dict[str, Any],
    baseline_5d: dict[str, Any],
    baseline_21d: dict[str, Any],
    portfolio_5d: dict[str, Any],
    portfolio_21d: dict[str, Any],
    tradable_coverage: float,
    nearby_positive_fraction: float,
) -> tuple[Conclusion, ScaleRecommendation, dict[str, bool]]:
    """Apply fixed scale gates; no event-level Sharpe enters this decision."""

    local_ic_positive = all(
        float(item.get("spearman_ic") or 0.0) > 0
        and float(item.get("average_signed_return") or 0.0) > 0
        for item in (holdout_5d, holdout_21d)
    )
    baseline_incremental = all(
        float(local.get("spearman_ic") or 0.0)
        > max(
            float(baseline.get("keyword_sentiment", {}).get("spearman_ic") or 0.0),
            float(baseline.get("event_type_signal", {}).get("spearman_ic") or 0.0),
            float(baseline.get("eodhd_sentiment", {}).get("spearman_ic") or 0.0),
        )
        for local, baseline in (
            (holdout_5d, baseline_5d),
            (holdout_21d, baseline_21d),
        )
    )
    cost_survival = all(
        float(item.get("conservative_net", {}).get("total_return") or 0.0) > 0
        for item in (portfolio_5d, portfolio_21d)
    )
    coverage_ok = tradable_coverage >= 0.25
    dependence_adjusted = all(
        float(item.get("company_cluster_bootstrap_95_ci", {}).get("lower_95") or 0.0) > 0
        and float(
            item.get("date_block_bootstrap_95_ci", {})
            .get("signed_return", {})
            .get("lower_95")
            or 0.0
        )
        > 0
        for item in (holdout_5d, holdout_21d)
    )
    nearby_stability = nearby_positive_fraction >= 0.60
    concentration_ok = all(
        float(item.get("maximum_exposure_hhi") or 1.0) <= 0.25
        for item in (portfolio_5d, portfolio_21d)
    )
    gates = {
        "positive_holdout_5d_and_21d": local_ic_positive,
        "incremental_to_keyword_event_type_and_eodhd": baseline_incremental,
        "positive_under_conservative_costs": cost_survival,
        "tradable_coverage_at_least_25_percent": coverage_ok,
        "dependence_adjusted_intervals_positive": dependence_adjusted,
        "nearby_specifications_directionally_stable": nearby_stability,
        "portfolio_not_extremely_concentrated": concentration_ok,
    }
    if all(gates.values()):
        strong_sharpe = all(
            float(item.get("conservative_net", {}).get("sharpe") or 0.0) >= 0.75
            for item in (portfolio_5d, portfolio_21d)
        )
        return (
            "PASS",
            "Expand to 25,000" if strong_sharpe else "Expand to 10,000",
            gates,
        )
    both_negative = all(
        float(item.get("spearman_ic") or 0.0) <= 0
        and float(item.get("average_signed_return") or 0.0) <= 0
        for item in (holdout_5d, holdout_21d)
    )
    if both_negative and not cost_survival:
        return "FAIL", "Stop at 5,000", gates
    return "INCONCLUSIVE", "Stop at 5,000", gates


def _html_document(results: dict[str, Any]) -> str:
    conclusion = html.escape(str(results["conclusion"]))
    recommendation = html.escape(str(results["scale_recommendation"]))
    payload = html.escape(json.dumps(results, indent=2, sort_keys=True, default=str))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Hybrid 5,000 Research Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}}
.decision{{padding:1rem;border:2px solid #34495e;border-radius:8px;background:#f8f9f9}}
pre{{white-space:pre-wrap;background:#f4f6f7;padding:1rem;border-radius:6px;font-size:.82rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd1d1;padding:.4rem;text-align:left}}
</style></head><body>
<h1>Hybrid 5,000-Article Equity News Sentiment Study</h1>
<div class="decision"><h2>{conclusion}</h2><p>Scale recommendation: {recommendation}</p></div>
<h2>Decision gates</h2><table><tr><th>Gate</th><th>Result</th></tr>
{''.join(f'<tr><td>{html.escape(key)}</td><td>{value}</td></tr>' for key,value in results['decision_gates'].items())}
</table><h2>Machine-readable evidence</h2><pre>{payload}</pre></body></html>"""


def build_final_report(
    config: FinalReportConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> FinalReportOutput:
    """Verify every evidence hash, decide once, and write HTML plus results JSON."""

    paths = {
        "sample": config.sample_manifest,
        "benchmark": config.benchmark_metrics,
        "local": config.local_manifest,
        "specification": config.primary_specification,
        "prediction": config.prediction_metrics,
        "baselines": config.baseline_metrics,
        "portfolio": config.portfolio_metrics,
        "additional_openai": config.additional_openai_manifest,
    }
    for name, path in paths.items():
        if file_sha256(path) != config.expected_hashes.get(name):
            raise RuntimeError(f"Final report evidence hash mismatch: {name}")
    evidence = {name: _load(path) for name, path in paths.items()}
    specification = evidence["specification"]["selected_predictive_specification"]
    aggregation = specification["aggregation"]
    signal = specification["signal"]
    prediction = evidence["prediction"][aggregation]["holdout"][signal]
    baseline = evidence["baselines"]["splits"]["holdout"]
    portfolio = evidence["portfolio"]["splits"]["holdout"]
    portfolio_mode = "market_neutral"
    decision = research_decision(
        holdout_5d=prediction["5d"],
        holdout_21d=prediction["21d"],
        baseline_5d=baseline["5d"],
        baseline_21d=baseline["21d"],
        portfolio_5d=portfolio["5d"][portfolio_mode],
        portfolio_21d=portfolio["21d"][portfolio_mode],
        tradable_coverage=float(evidence["prediction"]["counts"]["tradable_coverage"]),
        nearby_positive_fraction=float(specification["nearby_positive_fraction"]),
    )
    conclusion, recommendation, gates = decision
    additional_cost = float(evidence["additional_openai"]["actual_openai_cost_usd"])
    local_gpu = evidence["local"]["gpu"]
    results = {
        "conclusion": conclusion,
        "scale_recommendation": recommendation,
        "decision_gates": gates,
        "sample": {
            key: evidence["sample"].get(key)
            for key in (
                "sample_hash",
                "article_count",
                "company_count",
                "sector_count",
                "years",
                "other_fraction",
                "earnings_guidance_count",
            )
        },
        "local_run": evidence["local"],
        "selected_specification": specification,
        "holdout_prediction": {"5d": prediction["5d"], "21d": prediction["21d"]},
        "holdout_baselines": {"5d": baseline["5d"], "21d": baseline["21d"]},
        "holdout_portfolio": portfolio,
        "costs": {
            "additional_openai_usd": additional_cost,
            "cumulative_openai_usd": config.original_openai_cost_usd + additional_cost,
            "local_electricity_kwh": local_gpu["energy_kwh"],
            "local_electricity_usd": local_gpu["electricity_cost_usd"],
            "total_hybrid_usd": (
                config.original_openai_cost_usd
                + additional_cost
                + float(local_gpu["electricity_cost_usd"])
            ),
        },
        "evidence_hashes": config.expected_hashes,
        "limitations": [
            "The holdout covers only the first quarter of 2026.",
            "The fixed 125-company universe can retain survivorship bias.",
            "Daily adjusted prices cannot reconstruct intraday reaction paths or quoted spreads.",
            "Transaction costs are scenarios rather than historical order-book reconstruction.",
        ],
    }
    root = data_root / "results" / "hybrid_5000_final"
    store = ArtifactStore(data_root, duckdb_path)
    results_path = store.write_json(results, root / "results.json")
    report_path = root / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_html_document(results), encoding="utf-8")
    return FinalReportOutput(report_path, results_path, conclusion, recommendation)
