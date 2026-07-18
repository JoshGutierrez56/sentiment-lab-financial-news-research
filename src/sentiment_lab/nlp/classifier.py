"""Permanent-cache Batch API classification with selective post-pass escalation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from sentiment_lab.config.models import OpenAIConfig
from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.nlp.cache import ClassificationCache, article_content_hash, assessment_hash
from sentiment_lab.nlp.openai_client import (
    BatchExecution,
    BatchFailure,
    BatchItem,
    ModelCall,
    OpenAIClassificationError,
)
from sentiment_lab.nlp.prompts import PROMPT_VERSIONS, build_messages
from sentiment_lab.nlp.schemas import (
    ClassificationLedgerEntry,
    ClassificationRecord,
    EventType,
)

Stage = Literal["first_pass", "escalation"]

_ALWAYS_ESCALATE_EVENTS = {
    EventType.earnings_results,
    EventType.guidance,
    EventType.merger_acquisition,
    EventType.regulatory_action,
    EventType.fraud_accounting,
    EventType.bankruptcy,
}
_POSITIVE_TERMS = {
    "approval",
    "beat",
    "growth",
    "improved",
    "profit",
    "raised",
    "record",
    "strong",
    "upgrade",
}
_NEGATIVE_TERMS = {
    "bankruptcy",
    "decline",
    "downgrade",
    "fraud",
    "investigation",
    "loss",
    "missed",
    "reduced",
    "weak",
}
_CONTRAST_TERMS = {"although", "but", "despite", "however", "while", "yet"}


class BatchGateway(Protocol):
    def run_batch(
        self,
        items: list[BatchItem],
        *,
        model: str,
        max_output_tokens: int,
        prompt_version: str,
        schema_version: str,
        stage: str,
        budget_remaining_usd: float,
    ) -> BatchExecution: ...


@dataclass(frozen=True)
class _PreparedArticle:
    article: NewsArticle
    cache_key: str
    input_hash: str
    custom_id: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class ClassificationRun:
    final_records: list[ClassificationRecord]
    ledger_entries: list[ClassificationLedgerEntry]
    batch_executions: list[BatchExecution]

    @property
    def current_run_cost_usd(self) -> float:
        return sum(entry.run_cost_usd for entry in self.ledger_entries)

    def summary(self) -> dict[str, int | float]:
        api_entries = [entry for entry in self.ledger_entries if entry.outcome.startswith("api_")]
        first_pass_api = [entry for entry in api_entries if entry.stage == "first_pass"]
        escalation_api = [entry for entry in api_entries if entry.stage == "escalation"]
        first_pass_successes = [entry for entry in first_pass_api if entry.outcome == "api_success"]
        total_cost = sum(entry.run_cost_usd for entry in api_entries)
        return {
            "articles_classified_by_mini": len(first_pass_successes),
            "mini_model_api_requests": len(first_pass_api),
            "mini_model_failures": sum(entry.outcome == "api_failure" for entry in first_pass_api),
            "articles_escalated": sum(
                record.stage == "escalation" for record in self.final_records
            ),
            "expensive_model_api_calls": len(escalation_api),
            "expensive_model_failures": sum(
                entry.outcome == "api_failure" for entry in escalation_api
            ),
            "cache_hits": sum(entry.outcome == "cache_hit" for entry in self.ledger_entries),
            "input_tokens": sum(entry.input_tokens for entry in api_entries),
            "cached_input_tokens": sum(entry.cached_input_tokens for entry in api_entries),
            "output_tokens": sum(entry.output_tokens for entry in api_entries),
            "reasoning_tokens": sum(entry.reasoning_tokens for entry in api_entries),
            "total_tokens": sum(entry.input_tokens + entry.output_tokens for entry in api_entries),
            "total_cost_usd": total_cost,
            "average_cost_per_article_usd": (
                total_cost / len(self.final_records) if self.final_records else 0.0
            ),
        }


def contains_contradictory_information(article: NewsArticle) -> bool:
    words = set(re.findall(r"[a-z]+", f"{article.title} {article.content}".casefold()))
    return bool(words & _CONTRAST_TERMS and words & _POSITIVE_TERMS and words & _NEGATIVE_TERMS)


class ArticleClassifier:
    def __init__(
        self,
        batch_client: BatchGateway,
        cache: ClassificationCache,
        openai_config: OpenAIConfig,
        *,
        schema_version: str,
        max_article_characters: int,
        escalation_confidence_threshold: float,
        escalation_materiality_threshold: float,
    ) -> None:
        self.batch_client = batch_client
        self.cache = cache
        self.openai_config = openai_config
        self.schema_version = schema_version
        self.max_article_characters = max_article_characters
        self.escalation_confidence_threshold = escalation_confidence_threshold
        self.escalation_materiality_threshold = escalation_materiality_threshold

    def _prepare(
        self,
        article: NewsArticle,
        *,
        ticker: str,
        company_name: str,
        prompt_variant: str,
        model: str,
        stage: Stage,
    ) -> _PreparedArticle:
        prompt_version = PROMPT_VERSIONS[prompt_variant]
        input_hash = article_content_hash(article.title, article.content)
        cache_key = self.cache.key(
            article_content_hash=input_hash,
            ticker=ticker,
            prompt_version=prompt_version,
            schema_version=self.schema_version,
            model=model,
        )
        stage_prefix = "fp" if stage == "first_pass" else "esc"
        return _PreparedArticle(
            article=article,
            cache_key=cache_key,
            input_hash=input_hash,
            custom_id=f"{stage_prefix}-{cache_key[:56]}",
            messages=build_messages(
                article,
                ticker=ticker,
                company_name=company_name,
                variant=prompt_variant,
                max_characters=self.max_article_characters,
            ),
        )

    @staticmethod
    def _bind_cached(
        record: ClassificationRecord,
        prepared: _PreparedArticle,
        *,
        stage: Stage,
        escalation_reasons: list[str],
    ) -> ClassificationRecord:
        return record.model_copy(
            update={
                "article_id": prepared.article.article_id,
                "ticker": record.ticker,
                "event_timestamp": prepared.article.provider_timestamp,
                "stage": stage,
                "escalation_reasons": escalation_reasons,
                "from_cache": True,
            }
        )

    def _record_from_call(
        self,
        prepared: _PreparedArticle,
        call: ModelCall,
        *,
        ticker: str,
        requested_model: str,
        prompt_version: str,
        stage: Stage,
        escalation_reasons: list[str],
    ) -> ClassificationRecord:
        record = ClassificationRecord(
            cache_key=prepared.cache_key,
            input_hash=prepared.input_hash,
            output_hash=assessment_hash(call.assessment),
            article_id=prepared.article.article_id,
            ticker=ticker,
            event_timestamp=prepared.article.provider_timestamp,
            requested_model=requested_model,
            model=call.response_model,
            prompt_version=prompt_version,
            schema_version=self.schema_version,
            stage=stage,
            escalation_reasons=escalation_reasons,
            classified_at=datetime.now(UTC),
            response_id=call.response_id,
            batch_id=call.batch_id,
            batch_custom_id=call.batch_custom_id,
            from_cache=False,
            usage=call.usage,
            assessment=call.assessment,
        )
        self.cache.store(record)
        return record

    @staticmethod
    def _ledger_from_record(record: ClassificationRecord) -> ClassificationLedgerEntry:
        outcome: Literal["api_success", "cache_hit"] = (
            "cache_hit" if record.from_cache else "api_success"
        )
        return ClassificationLedgerEntry(
            article_id=record.article_id,
            ticker=record.ticker,
            event_timestamp=record.event_timestamp,
            cache_key=record.cache_key,
            input_hash=record.input_hash,
            requested_model=record.requested_model,
            response_model=record.model,
            prompt_version=record.prompt_version,
            schema_version=record.schema_version,
            stage=record.stage,
            outcome=outcome,
            escalation_reasons=record.escalation_reasons,
            response_id=record.response_id,
            batch_id=record.batch_id,
            batch_custom_id=record.batch_custom_id,
            input_tokens=record.usage.input_tokens,
            cached_input_tokens=record.usage.cached_input_tokens,
            output_tokens=record.usage.output_tokens,
            reasoning_tokens=record.usage.reasoning_tokens,
            estimated_cost_usd=record.usage.estimated_cost_usd,
            run_cost_usd=0.0 if record.from_cache else record.usage.estimated_cost_usd,
        )

    @staticmethod
    def _failure_ledger(
        prepared: _PreparedArticle,
        failure: BatchFailure,
        *,
        ticker: str,
        requested_model: str,
        prompt_version: str,
        schema_version: str,
        stage: Stage,
        escalation_reasons: list[str],
        charge_to_run: bool,
    ) -> ClassificationLedgerEntry:
        return ClassificationLedgerEntry(
            article_id=prepared.article.article_id,
            ticker=ticker,
            event_timestamp=prepared.article.provider_timestamp,
            cache_key=prepared.cache_key,
            input_hash=prepared.input_hash,
            requested_model=requested_model,
            response_model=failure.response_model,
            prompt_version=prompt_version,
            schema_version=schema_version,
            stage=stage,
            outcome="api_failure",
            failure_reason=failure.reason,
            escalation_reasons=escalation_reasons,
            response_id=failure.response_id,
            batch_id=failure.batch_id,
            batch_custom_id=failure.batch_custom_id,
            input_tokens=failure.usage.input_tokens if charge_to_run else 0,
            cached_input_tokens=failure.usage.cached_input_tokens if charge_to_run else 0,
            output_tokens=failure.usage.output_tokens if charge_to_run else 0,
            reasoning_tokens=failure.usage.reasoning_tokens if charge_to_run else 0,
            estimated_cost_usd=failure.usage.estimated_cost_usd if charge_to_run else 0.0,
            run_cost_usd=failure.usage.estimated_cost_usd if charge_to_run else 0.0,
        )

    def _escalation_reasons(
        self,
        article: NewsArticle,
        first_pass: ClassificationRecord | None,
    ) -> list[str]:
        if first_pass is None:
            return ["structured_output_validation_repeated_failure"]
        assessment = first_pass.assessment
        reasons: list[str] = []
        if assessment.confidence < self.escalation_confidence_threshold:
            reasons.append("confidence_below_threshold")
        if assessment.materiality >= self.escalation_materiality_threshold:
            reasons.append("high_materiality")
        if assessment.event_type in _ALWAYS_ESCALATE_EVENTS:
            reasons.append(f"event_type:{assessment.event_type.value}")
        if (
            assessment.event_type is EventType.litigation
            and assessment.materiality >= self.escalation_materiality_threshold
        ):
            reasons.append("event_type:major_litigation")
        if contains_contradictory_information(article):
            reasons.append("contradictory_article_evidence")
        return reasons

    def _run_stage(
        self,
        prepared_articles: list[tuple[int, _PreparedArticle]],
        *,
        model: str,
        max_output_tokens: int,
        prompt_version: str,
        stage: Stage,
        budget_remaining_usd: float,
        ticker: str,
        escalation_reasons_by_index: dict[int, list[str]],
        ledger: list[ClassificationLedgerEntry],
    ) -> tuple[dict[int, ClassificationRecord], BatchExecution | None]:
        records: dict[int, ClassificationRecord] = {}
        groups: dict[str, list[tuple[int, _PreparedArticle]]] = {}
        for index, prepared in prepared_articles:
            cached = self.cache.load(prepared.cache_key)
            reasons = escalation_reasons_by_index.get(index, [])
            if cached is not None:
                bound = self._bind_cached(
                    cached,
                    prepared,
                    stage=stage,
                    escalation_reasons=reasons,
                )
                records[index] = bound
                ledger.append(self._ledger_from_record(bound))
            else:
                groups.setdefault(prepared.cache_key, []).append((index, prepared))
        if not groups:
            return records, None

        representatives = [group[0][1] for group in groups.values()]
        execution = self.batch_client.run_batch(
            [
                BatchItem(custom_id=item.custom_id, messages=item.messages)
                for item in representatives
            ],
            model=model,
            max_output_tokens=max_output_tokens,
            prompt_version=prompt_version,
            schema_version=self.schema_version,
            stage=stage,
            budget_remaining_usd=budget_remaining_usd,
        )
        for group in groups.values():
            representative_index, representative = group[0]
            reasons = escalation_reasons_by_index.get(representative_index, [])
            call = execution.calls.get(representative.custom_id)
            if call is not None:
                canonical = self._record_from_call(
                    representative,
                    call,
                    ticker=ticker,
                    requested_model=model,
                    prompt_version=prompt_version,
                    stage=stage,
                    escalation_reasons=reasons,
                )
                records[representative_index] = canonical
                ledger.append(self._ledger_from_record(canonical))
                for index, duplicate in group[1:]:
                    bound = self._bind_cached(
                        canonical,
                        duplicate,
                        stage=stage,
                        escalation_reasons=escalation_reasons_by_index.get(index, []),
                    )
                    records[index] = bound
                    ledger.append(self._ledger_from_record(bound))
                continue
            failure = execution.failures[representative.custom_id]
            for group_index, (index, failed) in enumerate(group):
                ledger.append(
                    self._failure_ledger(
                        failed,
                        failure,
                        ticker=ticker,
                        requested_model=model,
                        prompt_version=prompt_version,
                        schema_version=self.schema_version,
                        stage=stage,
                        escalation_reasons=escalation_reasons_by_index.get(index, []),
                        charge_to_run=group_index == 0,
                    )
                )
        return records, execution

    def classify_many(
        self,
        articles: Sequence[NewsArticle],
        *,
        ticker: str,
        company_name: str,
        prompt_variant: str,
        budget_limit_usd: float,
    ) -> ClassificationRun:
        if not articles:
            raise ValueError("articles must not be empty")
        if budget_limit_usd <= 0:
            raise ValueError("budget_limit_usd must be positive")
        prompt_version = PROMPT_VERSIONS[prompt_variant]
        ledger: list[ClassificationLedgerEntry] = []
        executions: list[BatchExecution] = []

        first_prepared = [
            (
                index,
                self._prepare(
                    article,
                    ticker=ticker,
                    company_name=company_name,
                    prompt_variant=prompt_variant,
                    model=self.openai_config.first_pass_model,
                    stage="first_pass",
                ),
            )
            for index, article in enumerate(articles)
        ]
        first_records, first_execution = self._run_stage(
            first_prepared,
            model=self.openai_config.first_pass_model,
            max_output_tokens=self.openai_config.first_pass_max_output_tokens,
            prompt_version=prompt_version,
            stage="first_pass",
            budget_remaining_usd=budget_limit_usd,
            ticker=ticker,
            escalation_reasons_by_index={},
            ledger=ledger,
        )
        if first_execution is not None:
            executions.append(first_execution)

        # The entire mini-model batch is complete before any expensive-model request is built.
        reasons_by_index = {
            index: self._escalation_reasons(article, first_records.get(index))
            for index, article in enumerate(articles)
        }
        escalation_indices = [index for index, reasons in reasons_by_index.items() if reasons]
        final_records: dict[int, ClassificationRecord] = dict(first_records)
        if escalation_indices:
            spent = sum(entry.run_cost_usd for entry in ledger)
            escalation_prepared = [
                (
                    index,
                    self._prepare(
                        articles[index],
                        ticker=ticker,
                        company_name=company_name,
                        prompt_variant=prompt_variant,
                        model=self.openai_config.escalation_model,
                        stage="escalation",
                    ),
                )
                for index in escalation_indices
            ]
            escalation_records, escalation_execution = self._run_stage(
                escalation_prepared,
                model=self.openai_config.escalation_model,
                max_output_tokens=self.openai_config.escalation_max_output_tokens,
                prompt_version=prompt_version,
                stage="escalation",
                budget_remaining_usd=max(0.0, budget_limit_usd - spent),
                ticker=ticker,
                escalation_reasons_by_index=reasons_by_index,
                ledger=ledger,
            )
            if escalation_execution is not None:
                executions.append(escalation_execution)
            missing_escalations = sorted(set(escalation_indices) - set(escalation_records))
            if missing_escalations:
                raise OpenAIClassificationError(
                    "Expensive-model escalation did not return valid structured output for "
                    f"{len(missing_escalations)} article(s); no final milestone was produced"
                )
            final_records.update(escalation_records)

        missing_final = sorted(set(range(len(articles))) - set(final_records))
        if missing_final:
            raise OpenAIClassificationError(
                f"No valid final classification for {len(missing_final)} article(s)"
            )
        ordered = [final_records[index] for index in range(len(articles))]
        run = ClassificationRun(
            final_records=ordered,
            ledger_entries=ledger,
            batch_executions=executions,
        )
        if run.current_run_cost_usd > budget_limit_usd + 1e-9:
            raise OpenAIClassificationError(
                "OpenAI usage exceeded the configured budget despite conservative preflight; "
                "stop and review pricing configuration"
            )
        return run
