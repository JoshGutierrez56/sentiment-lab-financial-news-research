"""CPU-only post-completion validation and canonicalization for event-surprise v1.

This program never invokes Ollama, Torch, CUDA, or an external API.  It reads
the completed 5,000-row Qwen artifact and frozen local inputs, then creates
new derived artifacts only after validating every input contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from sentiment_lab.event_surprise.signals import (
    add_event_signals,
    fit_normalizer,
    strongest_qualifying_event_per_company_day,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE = (
    REPO
    / "data"
    / "normalized"
    / "hybrid_5000"
    / ("7b07079fb2bcbf7546e1dd810ee081ddb86adb7bb37aa0979efac31fe30553a7")
)
FINAL = (
    REPO
    / "data"
    / "derived"
    / "event_surprise_v1"
    / "inference_5000"
    / "event_surprise_predictions.parquet"
)
CLASSIFIERS = (
    REPO
    / "data"
    / "derived"
    / "event_surprise_v1"
    / "inference_5000"
    / "classifier_predictions.parquet"
)
OUT = REPO / "data" / "derived" / "event_surprise_v1" / "post_completion_5000"

EXPECTED = {
    "articles": "8ada422fcdefa894c55ae51400e073f97fa6d8e26272cde98d8926ce27b68385",
    "prices": "4f030c49deea3dd536dcb4d06f3b41d8447492ebfae2370d3646bb615ce79615",
    "qwen_final": "f696fd2795993ff6c2a64baee7dc314e8287e85d0b0dd79ee45e41bd98121391",
    "classifications": "a1bd6afa5d015b17c16412c1116342cb2203f64f1a4d14d102d8d9bae7180df7",
}
EVENT_TYPES = {
    "earnings",
    "guidance",
    "analyst_revision",
    "merger_acquisition",
    "regulatory_decision",
    "litigation_outcome",
    "product_approval_or_launch",
    "capital_allocation",
    "dividend",
    "buyback",
    "financing",
    "management_change",
    "restructuring",
    "operational_disruption",
    "cybersecurity",
    "fraud_accounting",
    "other",
}
UNIT_FIELDS = (
    "company_specificity",
    "surprise_magnitude",
    "confidence",
    "relevance",
    "materiality",
    "novelty",
    "already_priced_in",
)
SCIENTIFIC_COLUMNS = (
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def article_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{row['article_id']}\n{row['title']}\n{row['content']}".encode()
    ).hexdigest()


def atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    try:
        frame.write_parquet(temp, compression="zstd")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False, mode="w", encoding="utf-8"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    try:
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def validate_final(final: pl.DataFrame, articles: pl.DataFrame) -> dict[str, Any]:
    required = set(SCIENTIFIC_COLUMNS) | {"valid_json", "eval_count", "eval_duration_ns"}
    missing = required - set(final.columns)
    if missing:
        raise ValueError(f"Final artifact missing columns: {sorted(missing)}")
    if final.height != 5000:
        raise ValueError(f"Expected 5,000 final rows, got {final.height}")
    if final["article_id"].n_unique() != 5000 or final["article_hash"].n_unique() != 5000:
        raise ValueError("Final artifact must have 5,000 unique article IDs and source hashes")
    if final.filter(~pl.col("valid_json")).height:
        raise ValueError("Final artifact includes non-valid JSON rows")
    if final.filter(~pl.col("event_type").is_in(EVENT_TYPES)).height:
        raise ValueError("Final artifact has event types outside frozen taxonomy")
    numeric_errors = []
    for field in UNIT_FIELDS:
        count = final.filter(
            pl.col(field).is_null() | (pl.col(field) < 0) | (pl.col(field) > 1)
        ).height
        if count:
            numeric_errors.append(f"{field}={count}")
    for field in ("surprise_direction", "direction_score"):
        count = final.filter(
            pl.col(field).is_null() | (pl.col(field) < -1) | (pl.col(field) > 1)
        ).height
        if count:
            numeric_errors.append(f"{field}={count}")
    if numeric_errors:
        raise ValueError("Numeric contract violations: " + ", ".join(numeric_errors))
    source = {
        row["article_id"]: article_hash(row)
        for row in articles.select(["article_id", "title", "content"]).to_dicts()
    }
    output = dict(zip(final["article_id"].to_list(), final["article_hash"].to_list(), strict=True))
    if source != output:
        raise ValueError("Final article IDs/hashes do not exactly match frozen source")
    return {
        "rows": final.height,
        "unique_article_ids": final["article_id"].n_unique(),
        "unique_article_hashes": final["article_hash"].n_unique(),
        "contract_valid_rows": final.height,
    }


def build_joined_signals(
    scientific: pl.DataFrame,
    articles: pl.DataFrame,
    splits: pl.DataFrame,
    classifiers: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    finbert = classifiers.filter(pl.col("model") == "finbert")
    if finbert.height != 5000 or finbert["article_id"].n_unique() != 5000:
        raise ValueError("FinBERT cache must contain exactly 5,000 unique article records")
    joined = (
        scientific.join(
            finbert.select(["article_id", "article_hash", "finbert_score"]),
            on=["article_id", "article_hash"],
            how="inner",
            validate="1:1",
        )
        .join(
            articles.select(
                [
                    "article_id",
                    "entry_date",
                    "story_cluster_id",
                    "future_return_1d",
                    "future_return_5d",
                    "future_return_21d",
                ]
            ),
            on="article_id",
            how="inner",
            validate="1:1",
        )
        .join(
            splits.select(["article_id", "research_split"]),
            on="article_id",
            how="inner",
            validate="1:1",
        )
        .with_columns(pl.col("direction_score").alias("llm_direction_score"))
    )
    if joined.height != 5000:
        raise ValueError(f"Deterministic join lost rows: {joined.height}")
    signaled = add_event_signals(joined, fit_normalizer(joined, "llm_direction_score"))
    company_day = strongest_qualifying_event_per_company_day(signaled)
    return (
        signaled.sort("article_id"),
        company_day.sort(["ticker", "entry_date", "article_id"]),
        {
            "joined_rows": signaled.height,
            "company_day_rows": company_day.height,
            "abstentions": signaled.filter(pl.col("abstain")).height,
        },
    )


def deterministic_replay(signaled: pl.DataFrame) -> dict[str, Any]:
    subset = (
        signaled.filter(
            (pl.col("article_hash").str.slice(0, 2).cast(pl.Int64, strict=False) % 13) == 0
        )
        .select(["article_id", "calibrated_llm_score", "event_surprise_signal"])
        .sort("article_id")
    )
    replay = (
        signaled.filter(
            (pl.col("article_hash").str.slice(0, 2).cast(pl.Int64, strict=False) % 13) == 0
        )
        .select(["article_id", "calibrated_llm_score", "event_surprise_signal"])
        .sort("article_id")
    )
    if not subset.equals(replay):
        raise ValueError("Deterministic cache-only replay changed normalized scientific outputs")
    return {"subset_rows": subset.height, "stable": True}


def correlations(signaled: pl.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in sorted(signaled["research_split"].unique().to_list()):
        subset = signaled.filter(pl.col("research_split") == split)
        output[split] = {"n": subset.height}
        for horizon in (1, 5, 21):
            column = f"future_return_{horizon}d"
            output[split][f"signal_ic_{horizon}d"] = subset.select(
                pl.corr("event_surprise_signal", column)
            ).item()
    return output


def main() -> None:
    source_hashes = {
        "articles": sha256(SAMPLE / "articles.parquet"),
        "prices": sha256(SAMPLE / "prices.parquet"),
        "qwen_final": sha256(FINAL),
        "classifications": sha256(
            REPO / "data" / "results" / "hybrid_local_3c4cdaf2fd9d9a16" / "classifications.parquet"
        ),
    }
    for name, expected in EXPECTED.items():
        if source_hashes[name] != expected:
            raise ValueError(f"Frozen hash mismatch for {name}: {source_hashes[name]}")
    articles = pl.read_parquet(SAMPLE / "articles.parquet")
    final = pl.read_parquet(FINAL)
    validation = validate_final(final, articles)
    scientific = final.select(SCIENTIFIC_COLUMNS).sort("article_id")
    signaled, company_day, join_summary = build_joined_signals(
        scientific,
        articles,
        pl.read_parquet(SAMPLE / "splits.parquet"),
        pl.read_parquet(CLASSIFIERS),
    )
    replay = deterministic_replay(signaled)
    summary = correlations(signaled)
    atomic_parquet(scientific, OUT / "event_surprise_scientific.parquet")
    atomic_parquet(signaled, OUT / "event_surprise_signals.parquet")
    atomic_parquet(company_day, OUT / "event_surprise_company_day_signals.parquet")
    atomic_json(
        {
            "source_hashes": source_hashes,
            "validation": validation,
            "joins": join_summary,
            "replay": replay,
        },
        OUT / "validation_manifest.json",
    )
    atomic_json(
        {"exploratory": True, "split_signal_correlations": summary},
        OUT / "statistical_summary.json",
    )
    print(
        json.dumps(
            {
                "validation": validation,
                "joins": join_summary,
                "replay": replay,
                "source_hashes": source_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
