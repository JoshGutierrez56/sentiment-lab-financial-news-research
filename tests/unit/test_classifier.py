"""OpenAI adapter, semantic repair, and classification-cache tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_article, make_call
from sentiment_lab.config.models import OpenAIConfig
from sentiment_lab.nlp.cache import ClassificationCache
from sentiment_lab.nlp.classifier import ArticleClassifier
from sentiment_lab.nlp.openai_client import OpenAIArticleClient, OpenAIClassificationError


class FakeModel:
    model = "test-model"

    def __init__(self, calls: list[object]) -> None:
        self.outputs = list(calls)
        self.messages: list[list[dict[str, str]]] = []

    def classify(self, messages: list[dict[str, str]]) -> object:
        self.messages.append(messages)
        return self.outputs.pop(0)


class ArticleAwareModel:
    """Return by authoritative article ID so concurrency cannot reorder fixtures."""

    model = "test-model"

    def __init__(self, outputs: dict[str, object]) -> None:
        self.outputs = outputs

    def classify(self, messages: list[dict[str, str]]) -> object:
        prompt = "\n".join(message["content"] for message in messages)
        matches = [output for article_id, output in self.outputs.items() if article_id in prompt]
        assert len(matches) == 1
        return matches[0]


def _classifier(tmp_path: Path, model: object) -> ArticleClassifier:
    return ArticleClassifier(
        model,  # type: ignore[arg-type]
        ClassificationCache(tmp_path),
        schema_version="article_assessment.v1",
        max_article_characters=2000,
        max_concurrency=2,
    )


def test_classifier_caches_structured_assessment(tmp_path: Path) -> None:
    article = make_article()
    model = FakeModel([make_call(article)])
    classifier = _classifier(tmp_path, model)
    first = classifier.classify_one(
        article,
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
    )
    second = classifier.classify_one(
        article,
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
    )
    assert first.from_cache is False
    assert second.from_cache is True
    assert first.cache_key == second.cache_key
    assert len(model.messages) == 1


def test_classifier_repairs_authoritative_metadata_once(tmp_path: Path) -> None:
    article = make_article()
    wrong_article = make_article(article_id="z" * 64)
    model = FakeModel([make_call(wrong_article), make_call(article)])
    record = _classifier(tmp_path, model).classify_one(
        article,
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="directional_v1",
    )
    assert record.assessment.article_id == article.article_id
    assert "mismatched fields: article_id" in model.messages[1][-1]["content"]


def test_classifier_rejects_persistent_metadata_mismatch(tmp_path: Path) -> None:
    article = make_article()
    wrong_article = make_article(article_id="z" * 64)
    model = FakeModel([make_call(wrong_article), make_call(wrong_article)])
    with pytest.raises(OpenAIClassificationError, match="authoritative fields"):
        _classifier(tmp_path, model).classify_one(
            article,
            ticker="AAPL.US",
            company_name="Apple Inc.",
            prompt_variant="evidence_v2",
        )


def test_classifier_many_preserves_input_order(tmp_path: Path) -> None:
    first = make_article(article_id="1" * 64)
    second = make_article(article_id="2" * 64)
    model = ArticleAwareModel(
        {
            first.article_id: make_call(first),
            second.article_id: make_call(second),
        }
    )
    records = _classifier(tmp_path, model).classify_many(
        [first, second],
        ticker="AAPL.US",
        company_name="Apple Inc.",
        prompt_variant="evidence_v2",
    )
    assert [record.assessment.article_id for record in records] == [
        first.article_id,
        second.article_id,
    ]


def test_openai_adapter_uses_responses_parse_and_accounts_cost() -> None:
    article = make_article()
    response = SimpleNamespace(
        output_parsed=make_call(article).assessment,
        usage=SimpleNamespace(input_tokens=250, output_tokens=50),
        id="resp_live_shape",
        model="structured-test-model",
    )
    parse_calls: list[dict[str, object]] = []

    def parse(**kwargs: object) -> object:
        parse_calls.append(kwargs)
        return response

    sdk = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    config = OpenAIConfig(
        max_retries=1,
        temperature=0,
        input_cost_per_million=2,
        output_cost_per_million=8,
    )
    client = OpenAIArticleClient("key", "configured-model", config, sdk_client=sdk)
    call = client.classify([{"role": "user", "content": "test"}])
    assert call.response_model == "structured-test-model"
    assert call.usage.estimated_cost_usd == pytest.approx(0.0009)
    assert parse_calls[0]["text_format"] is call.assessment.__class__
    assert parse_calls[0]["temperature"] == 0


def test_openai_adapter_rejects_empty_parse_and_blank_credentials() -> None:
    sdk = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **_: SimpleNamespace(output_parsed=None, output_text="refused")
        )
    )
    client = OpenAIArticleClient("key", "model", OpenAIConfig(max_retries=1), sdk_client=sdk)
    with pytest.raises(OpenAIClassificationError, match="no parsed assessment"):
        client.classify([{"role": "user", "content": "test"}])
    with pytest.raises(ValueError, match="must not be blank"):
        OpenAIArticleClient(" ", "model", OpenAIConfig(), sdk_client=sdk)
