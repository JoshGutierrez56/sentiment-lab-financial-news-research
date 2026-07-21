"""Resume the local-only Qwen event-surprise extraction for the frozen 5,000 run.

This intentionally reuses the preregistered prompt/helper from
event_surprise_inference_5000.py, but checkpoints JSONL rows before writing
the canonical parquet so a stopped background run can resume without starting
from scratch.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import polars as pl
from event_surprise_inference_5000 import (
    ARTICLES,
    EVENT_TYPES,
    OUT,
    UNIT_FIELDS,
    article_hash,
    structured_one,
)

CHECKPOINT = OUT / "event_surprise_predictions.checkpoint.jsonl"
FINAL = OUT / "event_surprise_predictions.parquet"
BENCHMARK = OUT / "operational_benchmark.json"

REQUIRED_FIELDS = {
    "article_id",
    "article_hash",
    "ticker",
    "primary_ticker",
    "primary_company",
    "company_specificity",
    "event_type",
    "actual_information",
    "expected_or_prior_information",
    "surprise_direction",
    "surprise_magnitude",
    "direction_score",
    "confidence",
    "relevance",
    "materiality",
    "novelty",
    "already_priced_in",
    "expected_horizon",
    "abstain",
    "abstain_reason",
    "concise_evidence",
    "valid_json",
    "eval_count",
    "eval_duration_ns",
}

FINAL_SCHEMA = {
    "article_id": pl.String,
    "article_hash": pl.String,
    "ticker": pl.String,
    "primary_ticker": pl.String,
    "primary_company": pl.String,
    "company_specificity": pl.Float64,
    "event_type": pl.String,
    "actual_information": pl.String,
    "expected_or_prior_information": pl.String,
    "surprise_direction": pl.Float64,
    "surprise_magnitude": pl.Float64,
    "direction_score": pl.Float64,
    "confidence": pl.Float64,
    "relevance": pl.Float64,
    "materiality": pl.Float64,
    "novelty": pl.Float64,
    "already_priced_in": pl.Float64,
    "expected_horizon": pl.String,
    "abstain": pl.Boolean,
    "abstain_reason": pl.String,
    "concise_evidence": pl.String,
    "valid_json": pl.Boolean,
    "eval_count": pl.Int64,
    "eval_duration_ns": pl.Int64,
}


def contract_errors(row: dict[str, object]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        errors.append(f"missing={sorted(missing)}")
    if row.get("valid_json") is not True:
        errors.append("valid_json is not true")
    row_hash = row.get("article_hash")
    if not isinstance(row_hash, str) or len(row_hash) != 64:
        errors.append("article_hash is invalid")
    for field in UNIT_FIELDS:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"{field} is not numeric in [0, 1]")
    for field in ("surprise_direction", "direction_score"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1 <= value <= 1:
            errors.append(f"{field} is not numeric in [-1, 1]")
    if row.get("event_type") not in EVENT_TYPES:
        errors.append("event_type is outside the frozen taxonomy")
    if not isinstance(row.get("abstain"), bool):
        errors.append("abstain is not boolean")
    if not isinstance(row.get("concise_evidence"), str):
        errors.append("concise_evidence is not text")
    return errors


def canonicalize(row: dict[str, object]) -> dict[str, object]:
    text_or_none = (
        "primary_ticker",
        "primary_company",
        "actual_information",
        "expected_or_prior_information",
        "abstain_reason",
    )
    output = {field: row.get(field) for field in FINAL_SCHEMA}
    output["article_id"] = str(row["article_id"])
    output["article_hash"] = str(row["article_hash"])
    output["ticker"] = str(row["ticker"])
    for field in text_or_none:
        value = row.get(field)
        output[field] = None if value is None else str(value)
    horizon = row.get("expected_horizon")
    output["expected_horizon"] = None if horizon is None else str(horizon)
    for field in (*UNIT_FIELDS, "surprise_direction", "direction_score"):
        output[field] = float(row[field])
    output["event_type"] = str(row["event_type"])
    output["concise_evidence"] = str(row["concise_evidence"])
    output["abstain"] = bool(row["abstain"])
    output["valid_json"] = True
    output["eval_count"] = int(row["eval_count"])
    output["eval_duration_ns"] = int(row["eval_duration_ns"])
    return output


def load_checkpoint() -> dict[object, dict[str, object]]:
    if not CHECKPOINT.exists():
        return {}
    rows: dict[object, dict[str, object]] = {}
    ignored = 0
    with CHECKPOINT.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if contract_errors(row):
                    ignored += 1
                    continue
                rows[row["article_id"]] = row
    print(f"Checkpoint accepted {len(rows)} contract-valid rows; ignored {ignored} invalid rows")
    return rows


def append_checkpoint(row: dict[str, object]) -> None:
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def write_final(rows: list[dict[str, object]], started: str, elapsed: float) -> None:
    valid = sum(bool(row.get("valid_json", False)) for row in rows)
    invalid = len(rows) - valid
    canonical = [canonicalize(row) for row in rows]
    pl.DataFrame(canonical, schema=FINAL_SCHEMA, strict=True).write_parquet(
        FINAL, compression="zstd"
    )
    benchmark = {
        "started": started,
        "completed": datetime.now(UTC).isoformat(),
        "qwen": "qwen3.6:35b-a3b",
        "n": len(rows),
        "structured": {
            "elapsed_seconds": elapsed,
            "valid": valid,
            "invalid": invalid,
            "articles_per_minute": len(rows) / (elapsed / 60) if elapsed else None,
        },
        "checkpoint": str(CHECKPOINT),
    }
    BENCHMARK.write_text(json.dumps(benchmark, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    rows = pl.read_parquet(ARTICLES).sort("article_id").to_dicts()
    completed = load_checkpoint()
    source_by_id = {row["article_id"]: row for row in rows}
    completed = {
        article_id: result
        for article_id, result in completed.items()
        if article_id in source_by_id
        and result["article_hash"] == article_hash(source_by_id[article_id])
    }
    print(f"Loaded {len(rows)} articles; checkpoint has {len(completed)} rows")

    began = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        article_id = row["article_id"]
        if article_id in completed:
            continue
        if (index - 1) % 25 == 0:
            pct = (index - 1) / len(rows) * 100
            print(f"Article {index - 1}/{len(rows)} ({pct:.1f}%)")
        try:
            result = structured_one(row)
        except Exception as error:
            result = {
                "article_id": article_id,
                "article_hash": article_hash(row),
                "ticker": row["ticker"],
                "valid_json": False,
                "error": type(error).__name__,
            }
        append_checkpoint(result)
        if contract_errors(result):
            continue
        completed[article_id] = result

    missing_ids = [row["article_id"] for row in rows if row["article_id"] not in completed]
    if missing_ids:
        raise RuntimeError(
            f"Qwen repair finished with {len(missing_ids)} unresolved rows; refusing final Parquet"
        )
    ordered = [completed[row["article_id"]] for row in rows]
    elapsed = time.perf_counter() - began
    write_final(ordered, started, elapsed)
    valid = sum(bool(row.get("valid_json", False)) for row in ordered)
    print(f"Done. Wrote {FINAL} with {valid}/{len(ordered)} valid JSON rows")


if __name__ == "__main__":
    main()
