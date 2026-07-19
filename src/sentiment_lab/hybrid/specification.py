"""Development/validation-only primary specification selection and freeze."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.hybrid.analysis import SIGNALS
from sentiment_lab.hybrid.portfolio import PortfolioSpecification


class SpecificationSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    development_validation_events_path: Path
    expected_events_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_assignment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    signals: list[str] = Field(default_factory=lambda: list(SIGNALS))
    aggregations: list[str] = Field(
        default_factory=lambda: [
            "event_level",
            "strongest_company_day",
            "company_day_aggregate",
        ]
    )
    confidence_thresholds: list[float] = Field(default_factory=lambda: [0.0, 0.5, 0.7])
    relevance_thresholds: list[float] = Field(default_factory=lambda: [0.0, 0.5, 0.7])
    materiality_thresholds: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5])
    minimum_validation_events: int = Field(default=100, ge=50)
    minimum_validation_coverage: float = Field(default=0.25, ge=0.0, le=1.0)
    maximum_company_weight: float = Field(default=0.02, gt=0.0, le=0.02)
    base_cost_bps: float = Field(default=10.0, ge=0.0)
    conservative_cost_bps: float = Field(default=25.0, ge=0.0)


def _aggregate(frame: pl.DataFrame, method: str, signal: str) -> pl.DataFrame:
    if method == "event_level":
        return frame
    keys = ["research_split", "ticker", "entry_date"]
    if method == "strongest_company_day":
        return (
            frame.with_columns(pl.col(signal).abs().alias("_strength"))
            .sort("_strength", descending=True)
            .group_by(keys, maintain_order=True)
            .first()
            .drop("_strength")
        )
    if method != "company_day_aggregate":
        raise ValueError(f"Unknown aggregation method: {method}")
    return frame.group_by(keys, maintain_order=True).agg(
        pl.col(signal).mean(),
        pl.col("future_return_5d").first(),
        pl.col("future_return_21d").first(),
    )


def _association(frame: pl.DataFrame, signal: str, horizon: int) -> dict[str, Any]:
    values = frame.select(signal, f"future_return_{horizon}d").drop_nulls()
    if values.height < 3:
        return {"n": values.height, "spearman_ic": None, "p_value": None, "signed_return": None}
    scores = values[signal].to_numpy().astype(float)
    returns = values[f"future_return_{horizon}d"].to_numpy().astype(float)
    if np.std(scores) == 0 or np.std(returns) == 0:
        return {"n": len(scores), "spearman_ic": None, "p_value": None, "signed_return": None}
    statistic = spearmanr(scores, returns)
    return {
        "n": len(scores),
        "spearman_ic": float(statistic.statistic),
        "p_value": float(statistic.pvalue),
        "signed_return": float(np.mean(np.sign(scores) * returns)),
    }


def _bh_adjust(candidates: list[dict[str, Any]]) -> None:
    valid = [
        (index, float(candidate["combined_validation_p_value"]))
        for index, candidate in enumerate(candidates)
        if candidate["combined_validation_p_value"] is not None
    ]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    adjusted: dict[int, float] = {}
    total = len(ordered)
    for rank_from_end, (index, p_value) in enumerate(reversed(ordered), start=1):
        rank = total - rank_from_end + 1
        running = min(running, p_value * total / rank)
        adjusted[index] = running
    for index, candidate in enumerate(candidates):
        candidate["validation_bh_q_value"] = adjusted.get(index)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def freeze_primary_specification(
    config: SpecificationSearchConfig,
    *,
    data_root: Path,
    duckdb_path: Path,
) -> Path:
    """Search only development/validation and freeze before any holdout metrics."""

    if file_sha256(config.development_validation_events_path) != config.expected_events_sha256:
        raise RuntimeError("Development/validation event artifact hash mismatch")
    frame = pl.read_parquet(config.development_validation_events_path)
    observed_splits = set(frame["research_split"].to_list())
    if observed_splits != {"development", "validation"}:
        raise RuntimeError(
            "Specification search input must contain development and validation only"
        )
    validation_total = frame.filter(pl.col("research_split") == "validation").height
    candidates: list[dict[str, Any]] = []
    for signal, aggregation, confidence, relevance, materiality in itertools.product(
        config.signals,
        config.aggregations,
        config.confidence_thresholds,
        config.relevance_thresholds,
        config.materiality_thresholds,
    ):
        eligible = frame.filter(
            pl.col("tradable")
            & ~pl.col("abstain")
            & (pl.col("confidence") >= confidence)
            & (pl.col("relevance") >= relevance)
            & (pl.col("materiality") >= materiality)
        )
        evaluated = _aggregate(eligible, aggregation, signal)
        metrics = {
            split: {
                f"{horizon}d": _association(
                    evaluated.filter(pl.col("research_split") == split), signal, horizon
                )
                for horizon in (5, 21)
            }
            for split in ("development", "validation")
        }
        validation_ics = [metrics["validation"][f"{horizon}d"]["spearman_ic"] for horizon in (5, 21)]
        development_ics = [metrics["development"][f"{horizon}d"]["spearman_ic"] for horizon in (5, 21)]
        validation_signed = [metrics["validation"][f"{horizon}d"]["signed_return"] for horizon in (5, 21)]
        validation_n = min(metrics["validation"][f"{horizon}d"]["n"] for horizon in (5, 21))
        coverage = validation_n / validation_total if validation_total else 0.0
        complete = all(value is not None for value in (*validation_ics, *development_ics, *validation_signed))
        if complete:
            val_ic = float(np.mean(validation_ics))
            dev_ic = float(np.mean(development_ics))
            signed = float(np.mean(validation_signed))
            consistent = all(float(value) > 0 for value in (*validation_ics, *development_ics))
            utility = 0.45 * val_ic + 0.25 * dev_ic + 0.20 * signed * 100 + 0.10 * coverage
            if not consistent:
                utility -= 0.25
        else:
            utility = -math.inf
            consistent = False
        p_values = [metrics["validation"][f"{horizon}d"]["p_value"] for horizon in (5, 21)]
        combined_p = min(1.0, min(float(value) for value in p_values) * 2) if all(
            value is not None for value in p_values
        ) else None
        candidates.append(
            {
                "signal": signal,
                "aggregation": aggregation,
                "minimum_confidence": confidence,
                "minimum_relevance": relevance,
                "minimum_materiality": materiality,
                "validation_coverage": coverage,
                "directionally_consistent": consistent,
                "utility": utility if math.isfinite(utility) else None,
                "combined_validation_p_value": combined_p,
                "metrics": metrics,
            }
        )
    _bh_adjust(candidates)
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["utility"] is not None
        and candidate["validation_coverage"] >= config.minimum_validation_coverage
        and min(
            candidate["metrics"]["validation"][f"{horizon}d"]["n"]
            for horizon in (5, 21)
        )
        >= config.minimum_validation_events
    ]
    if not eligible_candidates:
        raise RuntimeError("No development/validation specification passed coverage gates")
    selected = max(eligible_candidates, key=lambda value: float(value["utility"]))
    confidence_index = {
        value: index for index, value in enumerate(config.confidence_thresholds)
    }
    relevance_index = {
        value: index for index, value in enumerate(config.relevance_thresholds)
    }
    materiality_index = {
        value: index for index, value in enumerate(config.materiality_thresholds)
    }
    neighbors = [
        candidate
        for candidate in eligible_candidates
        if candidate is not selected
        and candidate["signal"] == selected["signal"]
        and candidate["aggregation"] == selected["aggregation"]
        and (
            abs(
                confidence_index[float(candidate["minimum_confidence"])]
                - confidence_index[float(selected["minimum_confidence"])]
            )
            + abs(
                relevance_index[float(candidate["minimum_relevance"])]
                - relevance_index[float(selected["minimum_relevance"])]
            )
            + abs(
                materiality_index[float(candidate["minimum_materiality"])]
                - materiality_index[float(selected["minimum_materiality"])]
            )
        )
        == 1
    ]
    nearby_positive = sum(
        all(
            float(candidate["metrics"]["validation"][f"{horizon}d"]["spearman_ic"] or 0.0)
            > 0
            for horizon in (5, 21)
        )
        for candidate in neighbors
    )
    selected["nearby_candidate_count"] = len(neighbors)
    selected["nearby_positive_fraction"] = (
        nearby_positive / len(neighbors) if neighbors else 0.0
    )
    selected_aggregation = str(selected["aggregation"])
    portfolio_aggregation = (
        selected_aggregation
        if selected_aggregation in {"strongest_company_day", "company_day_aggregate"}
        else "strongest_company_day"
    )
    portfolio = PortfolioSpecification(
        signal=str(selected["signal"]),
        aggregation=portfolio_aggregation,
        minimum_confidence=float(selected["minimum_confidence"]),
        minimum_relevance=float(selected["minimum_relevance"]),
        minimum_materiality=float(selected["minimum_materiality"]),
        holding_periods=[5, 21],
        maximum_company_weight=config.maximum_company_weight,
        base_cost_bps=config.base_cost_bps,
        conservative_cost_bps=config.conservative_cost_bps,
    )
    selection_material = {
        "config": config.model_dump(mode="json"),
        "selected": selected,
        "candidate_count": len(candidates),
    }
    selection_hash = hashlib.sha256(stable_json(selection_material).encode()).hexdigest()
    commit, dirty = _git_state()
    manifest = {
        "specification_id": f"primary_{selection_hash[:16]}",
        "frozen_at": datetime.now(UTC),
        "frozen_before_holdout": True,
        "holdout_metrics_read": False,
        "sample_hash": config.sample_hash,
        "split_assignment_hash": config.split_assignment_hash,
        "development_validation_events_sha256": config.expected_events_sha256,
        "selection_hash": selection_hash,
        "candidate_count": len(candidates),
        "multiple_testing_method": "Benjamini-Hochberg across searched specifications",
        "selected_predictive_specification": selected,
        "portfolio_specification": portfolio.model_dump(mode="json"),
        "git_commit": commit,
        "dirty_worktree": dirty,
    }
    root = data_root / "results" / str(manifest["specification_id"])
    store = ArtifactStore(data_root, duckdb_path)
    store.write_parquet(pl.DataFrame(candidates, infer_schema_length=None), root / "candidates.parquet")
    path = root / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for volatile in ("frozen_at", "git_commit", "dirty_worktree"):
            existing.pop(volatile, None)
            manifest.pop(volatile, None)
        if existing != manifest:
            raise RuntimeError("Refusing to alter an existing primary specification")
        return path
    return store.write_json(manifest, path)
