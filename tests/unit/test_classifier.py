"""Batch API, permanent cache, budget, and escalation-policy tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_article, make_assessment
from sentiment_lab.config.models import OpenAIConfig
from sentiment_lab.nlp.cache import ClassificationCache
from sentiment_lab.nlp.classifier import ArticleClassifier, contains_contradictory_information
from sentiment_lab.nlp.openai_client import (
    BatchExecution,
    BatchFailure,
    BatchItem,
    ModelCall,
    OpenAIBatchClient,
    OpenAIBudgetError,
)
from sentiment_lab.nlp.prompts import PROMPT_VERSIONS, build_messages
from sentiment_lab.nlp.schemas import ArticleAssessment, EventType, ModelUsage


def _assessment(article: object, **updates: object) -> ArticleAssessment:
    base = make_assessment(article)  # type: ignore[arg-type]
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "event_type": EventType.other,
            "materiality": 0.20,
            "confidence": 0.90,
            **updates,
        }
    )
    return ArticleAssessment.model_validate(payload)


class FakeBatchGateway:
    def __init__(
        self,
        articles: list[object],
        factory: Callable[[object, str], ArticleAssessment],
        *,
        fail_first_pass: bool = False,
        fail_escalation: bool = False,
    ) -> None:
        self.articles = articles
        self.factory = factory
        self.fail_first_pass = fail_first_pass
        self.fail_escalation = fail_escalation
        self.stage_calls: list[tuple[str, str, int]] = []

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
    ) -> BatchExecution:
        del max_output_tokens, prompt_version, schema_version, budget_remaining_usd
        self.stage_calls.append((stage, model, len(items)))
        calls: dict[str, ModelCall] = {}
        failures: dict[str, BatchFailure] = {}
        should_fail = (stage == "first_pass" and self.fail_first_pass) or (
            stage == "escalation" and self.fail_escalation
        )
        for item in items:
            prompt = "\n".join(message["content"] for message in item.messages)
            matches = [article for article in self.articles if article.title in prompt]
            assert matches
            article = matches[0]
            usage = ModelUsage(
                input_tokens=100,
                cached_input_tokens=10,
                output_tokens=30,
                estimated_cost_usd=0.001,
            )
            if should_fail:
                failures[item.custom_id] = BatchFailure(
                    reason="structured_output_validation_failed",
                    usage=usage,
                    response_id="resp_failed",
                    response_model=model,
                    batch_id=f"batch_{stage}",
                    batch_custom_id=item.custom_id,
                )
            else:
                calls[item.custom_id] = ModelCall(
                    assessment=self.factory(article, stage),
                    usage=usage,
                    response_id=f"resp_{stage}",
                    response_model=model,
                    batch_id=f"batch_{stage}",
                    batch_custom_id=item.custom_id,
                )
        return BatchExecution(
            requested_model=model,
            stage=stage,
            batch_id=f"batch_{stage}",
            input_file_id=f"file_in_{stage}",
            output_file_id=f"file_out_{stage}",
            calls=calls,
            failures=failures,
            maximum_estimated_cost_usd=0.01,
        )


def _classifier(tmp_path: Path, gateway: FakeBatchGateway) -> ArticleClassifier:
    config = OpenAIConfig()
    return ArticleClassifier(
        gateway,
        ClassificationCache(tmp_path),
        config,
        schema_version="article_assessment.v2",
        max_article_characters=2000,
        escalation_confidence_threshold=0.70,
        escalation_materiality_threshold=0.80,
    )


def test_classifier_never_reclassifies_identical_content(tmp_path: Path) -> None:
    article = make_article()
    duplicate_id = make_article(article_id="z" * 64)
    gateway = FakeBatchGateway([article, duplicate_id], lambda item, _: _assessment(item))
    classifier = _classifier(tmp_path, gateway)

    first = classifier.classify_many(
        [article],
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
        budget_limit_usd=1.0,
    )
    second = classifier.classify_many(
        [duplicate_id],
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
        budget_limit_usd=1.0,
    )

    assert gateway.stage_calls == [("first_pass", "gpt-5.4-mini", 1)]
    assert first.final_records[0].from_cache is False
    assert second.final_records[0].from_cache is True
    assert second.final_records[0].article_id == duplicate_id.article_id
    assert first.final_records[0].cache_key == second.final_records[0].cache_key
    assert second.summary()["cache_hits"] == 1
    assert second.summary()["total_cost_usd"] == 0


def test_expensive_escalation_waits_for_complete_first_pass(tmp_path: Path) -> None:
    first = make_article(article_id="1" * 64, title="First unique article")
    second = make_article(article_id="2" * 64, title="Second unique article")

    def factory(article: object, stage: str) -> ArticleAssessment:
        if stage == "escalation":
            return _assessment(article, confidence=0.95, materiality=0.40)
        if article.title.startswith("First"):
            return _assessment(article, confidence=0.60)
        return _assessment(article, event_type=EventType.earnings_results)

    gateway = FakeBatchGateway([first, second], factory)
    result = _classifier(tmp_path, gateway).classify_many(
        [first, second],
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
        budget_limit_usd=1.0,
    )

    assert gateway.stage_calls == [
        ("first_pass", "gpt-5.4-mini", 2),
        ("escalation", "gpt-5.4", 2),
    ]
    assert [record.stage for record in result.final_records] == ["escalation", "escalation"]
    assert result.summary()["articles_classified_by_mini"] == 2
    assert result.summary()["articles_escalated"] == 2
    assert result.summary()["expensive_model_api_calls"] == 2


def test_contradiction_detection_handles_punctuation() -> None:
    contradictory = make_article(
        title="Apple reports record growth, but warns of weak profit",
        content="Results improved; however, management also disclosed a loss.",
    )
    one_sided = make_article(
        title="Apple reports record growth and strong profit",
        content="Results improved across the business.",
    )
    assert contains_contradictory_information(contradictory)
    assert not contains_contradictory_information(one_sided)


def test_validation_failure_is_charged_then_escalated(tmp_path: Path) -> None:
    article = make_article()
    gateway = FakeBatchGateway(
        [article],
        lambda item, _: _assessment(item),
        fail_first_pass=True,
    )
    result = _classifier(tmp_path, gateway).classify_many(
        [article],
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
        budget_limit_usd=1.0,
    )
    assert gateway.stage_calls == [
        ("first_pass", "gpt-5.4-mini", 1),
        ("escalation", "gpt-5.4", 1),
    ]
    assert result.current_run_cost_usd == pytest.approx(0.002)
    assert result.ledger_entries[0].outcome == "api_failure"
    assert result.summary()["articles_classified_by_mini"] == 0
    assert result.summary()["mini_model_api_requests"] == 1
    assert result.summary()["mini_model_failures"] == 1
    assert result.final_records[0].escalation_reasons == [
        "structured_output_validation_repeated_failure"
    ]


class FakeOpenAISDK:
    def __init__(self, assessment: ArticleAssessment) -> None:
        self.assessment = assessment
        self.uploaded_requests: list[dict[str, object]] = []
        self.upload_count = 0
        self.batch_create_count = 0
        self.files = SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve_batch)

    def _create_file(self, *, file: tuple[str, bytes, str], purpose: str) -> object:
        assert purpose == "batch"
        self.upload_count += 1
        self.uploaded_requests = [json.loads(line) for line in file[1].decode().splitlines()]
        return SimpleNamespace(id="file_input")

    def _batch(self) -> object:
        return SimpleNamespace(
            id="batch_live_shape",
            status="completed",
            output_file_id="file_output",
            metadata={"model": "gpt-5.4-mini", "stage": "first_pass"},
        )

    def _create_batch(self, **kwargs: object) -> object:
        assert kwargs["endpoint"] == "/v1/responses"
        assert kwargs["completion_window"] == "24h"
        self.batch_create_count += 1
        return self._batch()

    def _retrieve_batch(self, _: str) -> object:
        return self._batch()

    def _content(self, _: str) -> object:
        output_lines = []
        for request in reversed(self.uploaded_requests):
            body = {
                "id": f"resp_{request['custom_id']}",
                "model": "gpt-5.4-mini-2026-03-17",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": self.assessment.model_dump_json(),
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 100},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
            output_lines.append(
                json.dumps(
                    {
                        "custom_id": request["custom_id"],
                        "response": {"status_code": 200, "body": body},
                    }
                )
            )
        return SimpleNamespace(text="\n".join(output_lines))


def test_batch_adapter_uses_compact_schema_exact_cost_and_resume(tmp_path: Path) -> None:
    article = make_article()
    sdk = FakeOpenAISDK(_assessment(article))
    config = OpenAIConfig(batch_poll_interval_seconds=0.1)
    client = OpenAIBatchClient("key", config, tmp_path, sdk_client=sdk)
    messages = build_messages(
        article,
        ticker="AAPL.US",
        company_name="Apple Inc.",
        variant="evidence_v2",
        max_characters=2000,
    )
    kwargs = {
        "model": "gpt-5.4-mini",
        "max_output_tokens": 320,
        "prompt_version": PROMPT_VERSIONS["evidence_v2"],
        "schema_version": "article_assessment.v2",
        "stage": "first_pass",
        "budget_remaining_usd": 1.0,
    }
    first = client.run_batch([BatchItem(custom_id="request-1", messages=messages)], **kwargs)
    second = client.run_batch([BatchItem(custom_id="request-1", messages=messages)], **kwargs)

    request_body = sdk.uploaded_requests[0]["body"]
    assert isinstance(request_body, dict)
    assert request_body["model"] == "gpt-5.4-mini"
    assert request_body["max_output_tokens"] == 320
    assert request_body["reasoning"] == {"effort": "none"}
    assert "temperature" not in request_body
    schema = request_body["text"]["format"]["schema"]
    assert set(schema["properties"]) == {
        "sentiment_score",
        "sentiment_label",
        "confidence",
        "relevance",
        "materiality",
        "novelty",
        "event_type",
        "expected_horizon",
        "tradable",
        "abstain",
        "concise_reasoning",
    }
    usage = first.calls["request-1"].usage
    assert usage.input_tokens == 1000
    assert usage.cached_input_tokens == 100
    assert usage.output_tokens == 100
    assert usage.estimated_cost_usd == pytest.approx(0.00056625)
    assert first.calls["request-1"].response_model == "gpt-5.4-mini-2026-03-17"
    assert second.calls["request-1"].assessment == first.calls["request-1"].assessment
    assert sdk.upload_count == 1
    assert sdk.batch_create_count == 1


def test_batch_budget_preflight_stops_before_upload(tmp_path: Path) -> None:
    article = make_article()
    sdk = FakeOpenAISDK(_assessment(article))
    client = OpenAIBatchClient("key", OpenAIConfig(), tmp_path, sdk_client=sdk)
    messages = build_messages(
        article,
        ticker="AAPL.US",
        company_name="Apple Inc.",
        variant="evidence_v2",
        max_characters=2000,
    )
    with pytest.raises(OpenAIBudgetError, match="nothing was submitted"):
        client.run_batch(
            [BatchItem(custom_id="request-1", messages=messages)],
            model="gpt-5.4-mini",
            max_output_tokens=320,
            prompt_version=PROMPT_VERSIONS["evidence_v2"],
            schema_version="article_assessment.v2",
            stage="first_pass",
            budget_remaining_usd=0.0000001,
        )
    assert sdk.upload_count == 0


def test_openai_status_detail_is_short_and_redacts_credentials() -> None:
    detail = OpenAIBatchClient._status_error_detail(
        SimpleNamespace(
            body={
                "message": "  Invalid request with Bearer secret-token and "
                "sk-proj-do-not-print-this-value  "
            }
        )
    )

    assert detail == "Invalid request with Bearer [REDACTED] and [REDACTED]"
