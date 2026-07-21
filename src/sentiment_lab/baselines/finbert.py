"""Cache-first Hugging Face FinBERT adapter.

This module deliberately contains no network download path.  A model must be
present in the local Hugging Face cache and inference is opt-in, keeping the
redesign's cache-only stage from silently changing the completed experiment.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol

DEFAULT_MODEL = "ProsusAI/finbert"


def text_hash(headline: str, body: str, *, mode: str) -> str:
    """Hash the exact text passed to a model, including inference mode."""
    text = headline if mode == "headline" else f"{headline}\n\n{body}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FinBERTPrediction:
    article_id: str
    probability_positive: float
    probability_neutral: float
    probability_negative: float
    finbert_score: float
    model_identifier: str
    model_revision: str
    tokenizer_revision: str
    text_hash: str
    inference_timestamp: str
    inference_duration_ms: int
    mode: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


class FinBERTCache(Protocol):
    def get(self, key: str) -> FinBERTPrediction | None: ...

    def put(self, key: str, prediction: FinBERTPrediction) -> None: ...


def cache_key(*, article_id: str, text_digest: str, model: str, revision: str, mode: str) -> str:
    material = "|".join((article_id, text_digest, model, revision, mode))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class FinBERTAdapter:
    """Batch adapter with deterministic label ordering and cache keys."""

    def __init__(
        self,
        cache: FinBERTCache,
        *,
        model_identifier: str = DEFAULT_MODEL,
        model_revision: str = "main",
        tokenizer_revision: str = "main",
    ) -> None:
        self.cache = cache
        self.model_identifier = model_identifier
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision

    def score(
        self,
        articles: Iterable[dict[str, str]],
        *,
        mode: str = "headline",
        batch_size: int = 16,
        allow_inference: bool = False,
    ) -> list[FinBERTPrediction]:
        if mode not in {"headline", "full_text"}:
            raise ValueError("mode must be headline or full_text")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        prepared = list(articles)
        cached: list[FinBERTPrediction] = []
        missing: list[tuple[dict[str, str], str, str]] = []
        for article in prepared:
            digest = text_hash(article["title"], article.get("content", ""), mode=mode)
            key = cache_key(
                article_id=article["article_id"],
                text_digest=digest,
                model=self.model_identifier,
                revision=self.model_revision,
                mode=mode,
            )
            prediction = self.cache.get(key)
            if prediction is None:
                missing.append((article, digest, key))
            else:
                cached.append(prediction)
        if missing and not allow_inference:
            raise RuntimeError(
                f"{len(missing)} FinBERT predictions are absent from the immutable cache; "
                "cache-only mode will not run inference"
            )
        if missing:
            cached.extend(self._infer(missing, mode=mode, batch_size=batch_size))
        return sorted(cached, key=lambda row: row.article_id)

    def _infer(
        self, missing: list[tuple[dict[str, str], str, str]], *, mode: str, batch_size: int
    ) -> list[FinBERTPrediction]:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install sentiment-lab[finbert] with a local model cache") from error
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_identifier, revision=self.tokenizer_revision, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier, revision=self.model_revision, local_files_only=True
        )
        model.eval()
        result: list[FinBERTPrediction] = []
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            texts = [
                row[0]["title"]
                if mode == "headline"
                else f"{row[0]['title']}\n\n{row[0].get('content', '')}"
                for row in chunk
            ]
            began = time.perf_counter()
            tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                probabilities = torch.softmax(model(**tokens).logits, dim=1).tolist()
            elapsed = round((time.perf_counter() - began) * 1000)
            labels = {
                str(value).casefold(): index for index, value in model.config.id2label.items()
            }
            for (article, digest, key), values in zip(chunk, probabilities, strict=True):
                positive = float(values[labels["positive"]])
                neutral = float(values[labels["neutral"]])
                negative = float(values[labels["negative"]])
                prediction = FinBERTPrediction(
                    article_id=article["article_id"],
                    probability_positive=positive,
                    probability_neutral=neutral,
                    probability_negative=negative,
                    finbert_score=positive - negative,
                    model_identifier=self.model_identifier,
                    model_revision=self.model_revision,
                    tokenizer_revision=self.tokenizer_revision,
                    text_hash=digest,
                    inference_timestamp=datetime.now(UTC).isoformat(),
                    inference_duration_ms=elapsed,
                    mode=mode,
                )
                self.cache.put(key, prediction)
                result.append(prediction)
        return result
