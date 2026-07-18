"""Cached, bounded-concurrency article classification orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Protocol

from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.nlp.cache import ClassificationCache, assessment_hash
from sentiment_lab.nlp.openai_client import ModelCall, OpenAIClassificationError
from sentiment_lab.nlp.prompts import PROMPT_VERSIONS, build_messages
from sentiment_lab.nlp.schemas import ClassificationRecord


class ArticleModel(Protocol):
    model: str

    def classify(self, messages: list[dict[str, str]]) -> ModelCall: ...


class ArticleClassifier:
    def __init__(
        self,
        model_client: ArticleModel,
        cache: ClassificationCache,
        *,
        schema_version: str,
        max_article_characters: int,
        max_concurrency: int,
    ) -> None:
        self.model_client = model_client
        self.cache = cache
        self.schema_version = schema_version
        self.max_article_characters = max_article_characters
        self.max_concurrency = max_concurrency

    def classify_one(
        self,
        article: NewsArticle,
        *,
        ticker: str,
        company_name: str,
        prompt_variant: str,
        force: bool = False,
    ) -> ClassificationRecord:
        prompt_version = PROMPT_VERSIONS[prompt_variant]
        content_for_hash = "\n".join(
            [
                article.article_id,
                article.provider_timestamp.isoformat(),
                article.title,
                article.content,
            ]
        )
        cache_key, input_hash = self.cache.key(
            article_content=content_for_hash,
            ticker=ticker,
            company_name=company_name,
            prompt_version=prompt_version,
            schema_version=self.schema_version,
            model=self.model_client.model,
        )
        if not force:
            cached = self.cache.load(cache_key)
            if cached is not None:
                return cached

        messages = build_messages(
            article,
            ticker=ticker,
            company_name=company_name,
            variant=prompt_variant,
            max_characters=self.max_article_characters,
        )
        call: ModelCall | None = None
        mismatch = ""
        for semantic_attempt in range(2):
            call = self.model_client.classify(messages)
            assessment = call.assessment
            mismatches: list[str] = []
            if assessment.article_id != article.article_id:
                mismatches.append("article_id")
            if assessment.ticker != ticker.upper():
                mismatches.append("ticker")
            if assessment.event_timestamp != article.provider_timestamp:
                mismatches.append("event_timestamp")
            if not mismatches:
                break
            mismatch = ", ".join(mismatches)
            if semantic_attempt == 0:
                messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "Repair the structured response. Echo the authoritative metadata "
                            f"exactly; mismatched fields: {mismatch}."
                        ),
                    },
                ]
        else:
            raise OpenAIClassificationError(
                f"OpenAI structured assessment mismatched authoritative fields: {mismatch}"
            )

        if call is None:  # pragma: no cover - loop always executes
            raise AssertionError("classification loop did not execute")
        record = ClassificationRecord(
            cache_key=cache_key,
            input_hash=input_hash,
            output_hash=assessment_hash(call.assessment),
            model=call.response_model,
            prompt_version=prompt_version,
            schema_version=self.schema_version,
            classified_at=datetime.now(UTC),
            response_id=call.response_id,
            from_cache=False,
            usage=call.usage,
            assessment=call.assessment,
        )
        self.cache.store(record)
        return record

    def classify_many(
        self,
        articles: Sequence[NewsArticle],
        *,
        ticker: str,
        company_name: str,
        prompt_variant: str,
        force: bool = False,
    ) -> list[ClassificationRecord]:
        def classify(article: NewsArticle) -> ClassificationRecord:
            return self.classify_one(
                article,
                ticker=ticker,
                company_name=company_name,
                prompt_variant=prompt_variant,
                force=force,
            )

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            return list(executor.map(classify, articles))
