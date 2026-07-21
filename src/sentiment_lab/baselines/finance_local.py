"""Contracts for a cache-only finance-specialised local-model benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class LocalModelSpec:
    identifier: str
    revision: str
    prompt_version: str
    local_only: bool = True


def benchmark_predictions(
    frame: pl.DataFrame, *, score_column: str, return_column: str
) -> dict[str, float | int | None]:
    """Compare cached model outputs; never invokes a model or API."""
    needed = {
        "local_label",
        "openai_label",
        "local_abstain",
        "openai_abstain",
        score_column,
        return_column,
    }
    absent = needed - set(frame.columns)
    if absent:
        raise ValueError(f"Missing benchmark columns: {sorted(absent)}")
    usable = frame.drop_nulls([score_column, return_column])
    score = usable[score_column].to_numpy().astype(float)
    returns = usable[return_column].to_numpy().astype(float)
    correlation = (
        float(np.corrcoef(score, returns)[0, 1])
        if len(score) > 1 and np.std(score) and np.std(returns)
        else None
    )
    directional = usable.filter(pl.col("local_label") == pl.col("openai_label"))
    result: dict[str, float | int | None] = {
        "n": usable.height,
        "sentiment_label_agreement": _fraction(usable["local_label"] == usable["openai_label"]),
        "directional_agreement": float(directional.height / usable.height)
        if usable.height
        else None,
        "abstention_agreement": _fraction(usable["local_abstain"] == usable["openai_abstain"]),
        "prediction_ic": correlation,
        "signed_forward_return": float(np.mean(np.sign(score) * returns)) if len(score) else None,
    }
    optional_pairs = {
        "event_type_agreement": ("local_event_type", "openai_event_type"),
        "structured_output_validity": ("structured_valid",),
    }
    for metric, columns in optional_pairs.items():
        if set(columns).issubset(usable.columns):
            values = (
                usable[columns[0]]
                if len(columns) == 1
                else usable[columns[0]] == usable[columns[1]]
            )
            result[metric] = _fraction(values)
    for metric, column in (
        ("runtime_ms", "runtime_ms"),
        ("gpu_memory_mb", "gpu_memory_mb"),
        ("estimated_cost", "estimated_cost"),
    ):
        if column in usable.columns:
            value = usable[column].mean()
            result[metric] = float(value) if isinstance(value, (int, float)) else None
    return result


def _fraction(values: pl.Series) -> float | None:
    value = values.mean()
    return float(value) if isinstance(value, (int, float)) else None


def required_benchmark_fields() -> tuple[str, ...]:
    return (
        "sentiment_label",
        "direction_score",
        "event_type",
        "abstain",
        "structured_valid",
        "runtime_ms",
        "gpu_memory_mb",
        "estimated_cost",
        "forward_return",
    )
