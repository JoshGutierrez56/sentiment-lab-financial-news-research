from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from sentiment_lab.config.models import ValidationUniverseMember
from sentiment_lab.data.schemas import NewsArticle
from sentiment_lab.hybrid.local_model import (
    LocalClassificationCache,
    LocalModelSpec,
    LocalTarget,
    OllamaStructuredClient,
    build_local_messages,
)


def _target() -> LocalTarget:
    article = NewsArticle(
        article_id="a" * 64,
        provider_timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        retrieved_at=datetime(2025, 1, 2, tzinfo=UTC),
        title="Acme raises guidance",
        content="Acme raised revenue guidance after reporting strong orders. " * 20,
        link="https://example.test/acme",
        symbols=["ACME.US"],
        tags=[],
        raw_response_hash="b" * 64,
    )
    member = ValidationUniverseMember(
        ticker="ACME.US", company_name="Acme Corporation", sector="Industrials", aliases=["Acme"]
    )
    return LocalTarget(article=article, member=member)


def test_local_prompt_provides_candidates_and_treats_text_as_data() -> None:
    messages = build_local_messages(_target())
    assert "untrusted quoted data" in messages[0]["content"]
    assert "guidance" in messages[1]["content"]


def test_ollama_result_is_validated_cached_and_not_repeated(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = request.read().decode()
        assert '"think":false' in body
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": (
                        '{"sentiment_score":0.7,"sentiment_label":"bullish",'
                        '"confidence":0.9,"relevance":1.0,"materiality":0.8,"novelty":0.7,'
                        '"event_type":"guidance","expected_horizon":"5d","tradable":true,'
                        '"abstain":false,"abstain_reason":null,'
                        '"concise_reasoning":"Raised guidance signals stronger expected revenue."}'
                    )
                },
                "prompt_eval_count": 100,
                "eval_count": 60,
                "total_duration": 1_000_000_000,
                "load_duration": 1,
                "prompt_eval_duration": 2,
                "eval_duration": 3,
            },
        )

    cache = LocalClassificationCache(tmp_path)
    spec = LocalModelSpec(model="qwen-test", quantization="q4")
    with (
        httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as http,
        OllamaStructuredClient(http=http) as client,
    ):
        first = client.classify(_target(), spec, cache)
        second = client.classify(_target(), spec, cache)
    assert calls == 1
    assert not first.from_cache
    assert second.from_cache
    assert first.assessment.event_type.value == "guidance"


def test_invalid_first_attempt_is_repaired_and_accounted(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "not-json" if calls == 1 else (
            '{"sentiment_score":0.0,"sentiment_label":"neutral",'
            '"confidence":0.5,"relevance":0.5,"materiality":0.1,"novelty":0.1,'
            '"event_type":"other","expected_horizon":"1d","tradable":false,'
            '"abstain":true,"abstain_reason":"Insufficient information.",'
            '"concise_reasoning":"No material company-specific event."}'
        )
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as http,
        OllamaStructuredClient(http=http) as client,
    ):
        result = client.classify(
            _target(),
            LocalModelSpec(model="repair", quantization="q4"),
            LocalClassificationCache(tmp_path),
        )
    assert calls == 2
    assert not result.initial_output_valid
    assert result.validation_attempts == 2
    assert result.usage.prompt_tokens == 20


def test_invalid_local_output_is_not_cached(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"message": {"content": "not-json"}})
    )
    with (
        httpx.Client(transport=transport, base_url="http://test") as http,
        OllamaStructuredClient(http=http) as client,
        pytest.raises(RuntimeError, match="Invalid structured"),
    ):
        client.classify(
            _target(),
            LocalModelSpec(model="bad", quantization="q4"),
            LocalClassificationCache(tmp_path),
        )
    assert not list(tmp_path.rglob("*.json"))
