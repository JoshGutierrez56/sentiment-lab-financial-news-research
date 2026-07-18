"""Mocked HTTP tests for the documented EODHD endpoints."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import httpx
import pytest

from sentiment_lab.config.models import EODHDConfig
from sentiment_lab.data.cache import RawResponseCache
from sentiment_lab.data.eodhd_client import (
    EODHDClient,
    EODHDRequestError,
    EODHDSchemaError,
)


def _news_item(index: int) -> dict[str, object]:
    return {
        "date": f"2026-05-0{index + 1}T12:00:00+00:00",
        "title": f"Article {index}",
        "content": f"Full article body {index}",
        "link": f"https://example.test/{index}",
        "symbols": ["AAPL.US"],
        "tags": ["technology"],
        "sentiment": {"polarity": index / 10},
    }


def _client(
    tmp_path: Path,
    handler: object,
    *,
    sleeps: list[float] | None = None,
    page_size: int = 2,
) -> EODHDClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http = httpx.Client(transport=transport, base_url="https://eodhd.test")
    config = EODHDConfig(
        base_url="https://eodhd.test",
        max_retries=2,
        backoff_base_seconds=0,
        jitter_seconds=0,
        news_page_size=page_size,
    )
    return EODHDClient(
        "secret-token",
        config,
        RawResponseCache(tmp_path),
        http=http,
        sleeper=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_news_paginates_deduplicates_and_reuses_raw_cache(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    pages = {
        0: [_news_item(0), _news_item(1)],
        2: [_news_item(1), _news_item(2)],
        4: [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/api/news"
        assert request.url.params["api_token"] == "secret-token"
        return httpx.Response(200, json=pages[int(request.url.params["offset"])])

    client = _client(tmp_path, handler)
    first = client.fetch_news("aapl.us", date(2026, 5, 1), date(2026, 5, 10), max_articles=4)
    second = client.fetch_news("AAPL.US", date(2026, 5, 1), date(2026, 5, 10), max_articles=4)
    assert [item.title for item in first] == ["Article 0", "Article 1", "Article 2"]
    assert [item.article_id for item in second] == [item.article_id for item in first]
    assert len(calls) == 3
    all_cache_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (tmp_path / "raw").rglob("*")
        if path.is_file()
    )
    assert "secret-token" not in all_cache_text


def test_repeated_news_page_is_detected(tmp_path: Path) -> None:
    repeated = [_news_item(0), _news_item(1)]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=repeated)

    client = _client(tmp_path, handler)
    with pytest.raises(EODHDSchemaError, match="repeated a news page"):
        client.fetch_news("AAPL.US", date(2026, 5, 1), date(2026, 5, 10), max_articles=4)


def test_eod_price_retry_validation_and_adjustment(tmp_path: Path) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.url.path == "/api/eod/AAPL.US"
        assert request.url.params["period"] == "d"
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})
        return httpx.Response(
            200,
            json=[
                {
                    "date": "2026-05-04",
                    "open": 200,
                    "high": 204,
                    "low": 198,
                    "close": 200,
                    "adjusted_close": 100,
                    "volume": 1000,
                },
                {
                    "date": "2026-05-04",
                    "open": 202,
                    "high": 205,
                    "low": 200,
                    "close": 202,
                    "adjusted_close": 101,
                    "volume": 1200,
                },
            ],
        )

    client = _client(tmp_path, handler, sleeps=sleeps)
    prices = client.fetch_eod_prices("aapl.us", date(2026, 5, 1), date(2026, 5, 5))
    assert attempts == 2
    assert sleeps == [0.0]
    assert len(prices) == 1
    assert prices[0].adjusted_open == 101


def test_invalid_json_is_preserved_and_reported(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client = _client(tmp_path, handler)
    with pytest.raises(EODHDSchemaError, match="invalid JSON"):
        client.fetch_eod_prices("AAPL.US", date(2026, 5, 1), date(2026, 5, 5))
    bodies = list((tmp_path / "raw" / "eodhd" / "responses").rglob("*.json"))
    assert len(bodies) == 1
    assert bodies[0].read_bytes() == b"not-json"


def test_provider_schema_and_http_errors_are_clear_and_secret_free(tmp_path: Path) -> None:
    responses = iter(
        [
            httpx.Response(200, json={"not": "an array"}),
            httpx.Response(403, json={"error": "forbidden"}),
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    client = _client(tmp_path, handler)
    with pytest.raises(EODHDSchemaError, match="JSON array"):
        client.fetch_news(
            "AAPL.US", date(2026, 5, 1), date(2026, 5, 2), max_articles=1, refresh=True
        )
    with pytest.raises(EODHDRequestError, match="HTTP 403") as captured:
        client.fetch_eod_prices("AAPL.US", date(2026, 5, 1), date(2026, 5, 2), refresh=True)
    assert "secret-token" not in str(captured.value)


def test_bad_inputs_and_malformed_news_schema(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps([{"date": "not-a-date"}]).encode())

    client = _client(tmp_path, handler)
    with pytest.raises(ValueError, match="max_articles"):
        client.fetch_news("AAPL.US", date.today(), date.today(), max_articles=0)
    with pytest.raises(EODHDSchemaError, match="schema validation"):
        client.fetch_news("AAPL.US", date.today(), date.today(), max_articles=1)
    with pytest.raises(ValueError, match="must not be blank"):
        EODHDClient(" ", EODHDConfig(), RawResponseCache(tmp_path))


def test_request_logging_is_sanitized(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    caplog.set_level(logging.INFO, logger="sentiment_lab.data.eodhd_client")
    client = _client(tmp_path, handler)
    client.fetch_news("AAPL.US", date.today(), date.today(), max_articles=1, refresh=True)
    assert "endpoint=/api/news" in caplog.text
    assert "secret-token" not in caplog.text
    assert "api_token" not in caplog.text
