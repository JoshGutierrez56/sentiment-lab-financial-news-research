"""Authorized local-only 5,000-article inference: FinBERT, Financial RoBERTa, Qwen structured extraction.

Writes only to data/derived/event_surprise_v1/. Does not modify original artifacts.
No OpenAI or paid API calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ARTICLES = Path(
    "data/normalized/hybrid_5000/7b07079fb2bcbf7546e1dd810ee081ddb86adb7bb37aa0979efac31fe30553a7/articles.parquet"
)
OUT = Path("data/derived/event_surprise_v1/inference_5000")
DEVICE = "cuda"
MODELS = {
    "finbert": ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43"),
    "finance_roberta": (
        "soleimanian/financial-roberta-large-sentiment",
        "f8804d31111d7c3569e88abaad6969918e858fbd",
    ),
}
QWEN = "qwen3.6:35b-a3b"
BATCH_SIZE = 32
QWEN_BATCH_SIZE = 10  # articles per Qwen batch via concurrent requests

EVENT_TYPES = [
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
]

STRUCTURED_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "required": [
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
    ],
    "properties": {
        "primary_ticker": {"type": ["string", "null"]},
        "primary_company": {"type": ["string", "null"]},
        "company_specificity": {"type": "number", "minimum": 0, "maximum": 1},
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "actual_information": {"type": ["string", "null"]},
        "expected_or_prior_information": {"type": ["string", "null"]},
        "surprise_direction": {"type": "number", "minimum": -1, "maximum": 1},
        "surprise_magnitude": {"type": "number", "minimum": 0, "maximum": 1},
        "direction_score": {"type": "number", "minimum": -1, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "materiality": {"type": "number", "minimum": 0, "maximum": 1},
        "novelty": {"type": "number", "minimum": 0, "maximum": 1},
        "already_priced_in": {"type": "number", "minimum": 0, "maximum": 1},
        "expected_horizon": {"type": ["string", "null"]},
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
        "concise_evidence": {"type": "string"},
    },
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


def _validate_structured_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Qwen structured output must be a JSON object")
    missing = set(STRUCTURED_FORMAT["required"]) - set(value)
    if missing:
        raise ValueError(f"Qwen structured output omitted fields: {sorted(missing)}")
    for field in UNIT_FIELDS:
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 1:
            raise ValueError(f"Qwen field {field} must be numeric in [0, 1]")
    for field in ("surprise_direction", "direction_score"):
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not -1 <= number <= 1
        ):
            raise ValueError(f"Qwen field {field} must be numeric in [-1, 1]")
    if value["event_type"] not in EVENT_TYPES:
        raise ValueError("Qwen event_type is outside the frozen taxonomy")
    if not isinstance(value["abstain"], bool):
        raise ValueError("Qwen abstain field must be boolean")
    if value["expected_horizon"] is not None and not isinstance(value["expected_horizon"], str):
        raise ValueError("Qwen expected_horizon must be text or null")
    if not isinstance(value["concise_evidence"], str):
        raise ValueError("Qwen concise_evidence must be text")
    return value


def article_hash(row: dict[str, object]) -> str:
    return hashlib.sha256(
        f"{row['article_id']}\n{row['title']}\n{row['content']}".encode()
    ).hexdigest()


def score_classifier(
    name: str, repo: str, revision: str, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            repo, revision=revision, local_files_only=True
        )
        .to(DEVICE)
        .eval()
    )
    labels = {str(label).casefold(): int(index) for index, label in model.config.id2label.items()}
    output: list[dict[str, object]] = []
    started = time.perf_counter()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        text = [str(row["title"]) + "\n\n" + str(row["content"]) for row in batch]
        tokens = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(model(**tokens).logits, dim=1).cpu().tolist()
        for row, values, count in zip(
            batch, probabilities, tokens["attention_mask"].sum(dim=1).cpu().tolist(), strict=True
        ):
            probabilities_by_label = {
                label: float(values[index]) for label, index in labels.items()
            }
            positive = probabilities_by_label.get("positive", 0.0)
            negative = probabilities_by_label.get("negative", 0.0)
            output.append(
                {
                    "article_id": row["article_id"],
                    "article_hash": article_hash(row),
                    "ticker": row["ticker"],
                    "model": name,
                    "model_revision": revision,
                    "probability_positive": positive,
                    "probability_neutral": probabilities_by_label.get("neutral", 0.0),
                    "probability_negative": negative,
                    "finbert_score": positive - negative,
                    "label": max(probabilities_by_label, key=probabilities_by_label.get),
                    "input_token_count": int(count),
                }
            )
    elapsed = time.perf_counter() - started
    print(
        f"  {name}: {len(rows)} rows in {elapsed:.1f}s ({len(rows) / (elapsed / 60):.0f} articles/min)"
    )
    del model
    torch.cuda.empty_cache()
    return output


def structured_one(row: dict[str, object]) -> dict[str, object]:
    prompt = (
        """Return only valid compact JSON. Assess one equity-news article. Default abstain for generic, stale, syndicated, incidental, low relevance/materiality, no identifiable new information, or non-company primary subject. Facts are not surprise. event_type must be earnings,guidance,analyst_revision,merger_acquisition,regulatory_decision,litigation_outcome,product_approval_or_launch,capital_allocation,dividend,buyback,financing,management_change,restructuring,operational_disruption,cybersecurity,fraud_accounting,other. Fields: primary_ticker,primary_company,company_specificity,event_type,actual_information,expected_or_prior_information,surprise_direction,surprise_magnitude,direction_score,confidence,relevance,materiality,novelty,already_priced_in,expected_horizon,abstain,abstain_reason,concise_evidence. All numeric fields 0..1 except direction_score -1..1. concise_evidence <=35 words and only article facts.\n\nARTICLE:\n"""
        + str(row["title"])
        + "\n"
        + str(row["content"])
    )
    eval_count = 0
    eval_duration_ns = 0
    last_error: Exception | None = None
    for attempt in range(1, 4):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\n\nREPAIR REQUIREMENT: include every requested field. "
                "company_specificity must be a JSON number from 0 to 1, never text or null."
            )
        try:
            http_response = httpx.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": QWEN,
                    "prompt": attempt_prompt,
                    "stream": False,
                    "think": False,
                    "format": STRUCTURED_FORMAT,
                    "options": {
                        "temperature": 0,
                        "seed": 20260719,
                        "top_k": 1,
                        "top_p": 1,
                        "num_predict": 420,
                    },
                },
                timeout=600,
            )
            http_response.raise_for_status()
            response = http_response.json()
            eval_count += int(response.get("eval_count", 0))
            eval_duration_ns += int(response.get("eval_duration", 0))
            value = _validate_structured_value(json.loads(response["response"]))
            return {
                "article_id": row["article_id"],
                "article_hash": article_hash(row),
                "ticker": row["ticker"],
                **value,
                "valid_json": True,
                "eval_count": eval_count,
                "eval_duration_ns": eval_duration_ns,
            }
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            last_error = error
    raise RuntimeError(
        "Qwen failed the frozen structured-output contract after 3 attempts"
    ) from last_error


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pl.read_parquet(ARTICLES).sort("article_id").to_dicts()
    n = len(rows)
    print(f"Loaded {n} articles from frozen corpus")

    # Phase 3: FinBERT + Financial RoBERTa
    print("Phase 3: Running classifier baselines...")
    all_classifier_rows = []
    benchmark: dict[str, object] = {
        "started": datetime.now(UTC).isoformat(),
        "models": MODELS,
        "qwen": QWEN,
        "n": n,
    }
    for name, (repo, revision) in MODELS.items():
        scored = score_classifier(name, repo, revision, rows)
        benchmark[name] = {
            "elapsed_seconds": time.perf_counter(),
            "rows": len(scored),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        }
        all_classifier_rows.extend(scored)
    pl.DataFrame(all_classifier_rows).write_parquet(
        OUT / "classifier_predictions.parquet", compression="zstd"
    )
    print(f"  Wrote {len(all_classifier_rows)} classifier predictions")

    # Phase 4: Qwen structured extraction
    print("Phase 4: Running Qwen structured extraction...")
    structured_rows: list[dict[str, object]] = []
    began = time.perf_counter()
    for i, row in enumerate(rows):
        if i % 50 == 0:
            print(f"  Article {i}/{n} ({i / n * 100:.1f}%)")
        try:
            structured_rows.append(structured_one(row))
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as error:
            structured_rows.append(
                {
                    "article_id": row["article_id"],
                    "article_hash": article_hash(row),
                    "ticker": row["ticker"],
                    "valid_json": False,
                    "error": type(error).__name__,
                }
            )
    elapsed = time.perf_counter() - began
    valid = sum(bool(r.get("valid_json", False)) for r in structured_rows)
    invalid = n - valid
    benchmark["structured"] = {"elapsed_seconds": elapsed, "valid": valid, "invalid": invalid}
    print(
        f"  Qwen: {valid} valid, {invalid} invalid in {elapsed:.1f}s ({n / (elapsed / 60):.1f} articles/min)"
    )

    pl.DataFrame(structured_rows).write_parquet(
        OUT / "event_surprise_predictions.parquet", compression="zstd"
    )
    (OUT / "operational_benchmark.json").write_text(
        json.dumps(benchmark, indent=2, default=str), encoding="utf-8"
    )
    print("Done. Outputs in", OUT)


if __name__ == "__main__":
    main()
