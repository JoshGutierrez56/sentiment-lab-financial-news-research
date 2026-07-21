"""Authorized local-only 100-article model gate; writes only derived artifacts."""

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
OUT = Path("data/derived/event_surprise_v1/benchmark_100")
DEVICE = "cuda"
MODELS = {
    "finbert": ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43"),
    "finance_roberta": (
        "soleimanian/financial-roberta-large-sentiment",
        "f8804d31111d7c3569e88abaad6969918e858fbd",
    ),
}
QWEN = "qwen3.6:35b-a3b"


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
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
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
                    "score": positive - negative,
                    "label": max(probabilities_by_label, key=probabilities_by_label.get),
                    "input_token_count": int(count),
                }
            )
    output.append(
        {
            "_benchmark": name,
            "elapsed_seconds": time.perf_counter() - started,
            "rows": len(rows),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    del model
    torch.cuda.empty_cache()
    return output


def structured(row: dict[str, object]) -> dict[str, object]:
    prompt = (
        """Return only valid compact JSON. Assess one equity-news article. Default abstain for generic, stale, syndicated, incidental, low relevance/materiality, no identifiable new information, or non-company primary subject. Facts are not surprise. event_type must be earnings,guidance,analyst_revision,merger_acquisition,regulatory_decision,litigation_outcome,product_approval_or_launch,capital_allocation,dividend,buyback,financing,management_change,restructuring,operational_disruption,cybersecurity,fraud_accounting,other. Fields: primary_ticker,primary_company,company_specificity,event_type,actual_information,expected_or_prior_information,surprise_direction,surprise_magnitude,direction_score,confidence,relevance,materiality,novelty,already_priced_in,expected_horizon,abstain,abstain_reason,concise_evidence. All numeric fields 0..1 except direction_score -1..1. concise_evidence <=35 words and only article facts.\n\nARTICLE:\n"""
        + str(row["title"])
        + "\n"
        + str(row["content"])
    )
    response = httpx.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": QWEN,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "seed": 20260719,
                "top_k": 1,
                "top_p": 1,
                "num_predict": 420,
            },
        },
        timeout=600,
    ).json()
    value = json.loads(response["response"])
    return {
        "article_id": row["article_id"],
        "article_hash": article_hash(row),
        "ticker": row["ticker"],
        **value,
        "valid_json": True,
        "eval_count": response.get("eval_count", 0),
        "eval_duration_ns": response.get("eval_duration", 0),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows = pl.read_parquet(ARTICLES).sort("article_id").head(100).to_dicts()
    classifiers = []
    benchmark: dict[str, object] = {
        "started": datetime.now(UTC).isoformat(),
        "models": MODELS,
        "qwen": QWEN,
        "n": len(rows),
    }
    for name, (repo, revision) in MODELS.items():
        scored = score_classifier(name, repo, revision, rows)
        benchmark[name] = scored.pop()
        classifiers.extend(scored)
    structured_rows = []
    began = time.perf_counter()
    for row in rows:
        try:
            structured_rows.append(structured(row))
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
    benchmark["structured"] = {
        "elapsed_seconds": time.perf_counter() - began,
        "valid": sum(bool(row["valid_json"]) for row in structured_rows),
        "invalid": sum(not bool(row["valid_json"]) for row in structured_rows),
    }
    pl.DataFrame(classifiers).write_parquet(
        OUT / "classifier_predictions.parquet", compression="zstd"
    )
    pl.DataFrame(structured_rows).write_parquet(
        OUT / "event_surprise_predictions.parquet", compression="zstd"
    )
    (OUT / "operational_benchmark.json").write_text(
        json.dumps(benchmark, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
