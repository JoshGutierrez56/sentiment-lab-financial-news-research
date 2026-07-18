"""Documented EODHD news and EOD-price client for the mandatory milestone."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any, cast
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from sentiment_lab.config.models import EODHDConfig
from sentiment_lab.data.cache import CachedPayload, RawResponseCache
from sentiment_lab.data.schemas import EODHDNewsItem, EODPrice, NewsArticle

log = logging.getLogger(__name__)


class _SecretRedactionFilter(logging.Filter):
    """Redact an EODHD query token from third-party transport log records."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._secrets = (token, quote(token, safe=""))

    def _redact(self, value: object) -> object:
        rendered = str(value)
        if not any(secret and secret in rendered for secret in self._secrets):
            return value
        for secret in self._secrets:
            if secret:
                rendered = rendered.replace(secret, "[REDACTED]")
        return rendered

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(self._redact(value) for value in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: self._redact(value) for key, value in record.args.items()}
        return True


class EODHDError(RuntimeError):
    """Base error that never embeds credentials."""


class EODHDRequestError(EODHDError):
    pass


class EODHDSchemaError(EODHDError):
    pass


class EODHDClient:
    """Small injectable client using only current documented endpoints."""

    _RETRYABLE = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        token: str,
        config: EODHDConfig,
        cache: RawResponseCache,
        *,
        http: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("EODHD token must not be blank")
        self._token = token.strip()
        self.config = config
        self.cache = cache
        self._owns_http = http is None
        self.http = http or httpx.Client(
            base_url=config.base_url.rstrip("/"), timeout=config.timeout_seconds
        )
        self._sleep = sleeper
        self._rng = rng or random.Random()
        self._transport_loggers = (logging.getLogger("httpx"), logging.getLogger("httpcore"))
        self._redaction_filter = _SecretRedactionFilter(self._token)
        for transport_logger in self._transport_loggers:
            transport_logger.addFilter(self._redaction_filter)

    def close(self) -> None:
        for transport_logger in self._transport_loggers:
            transport_logger.removeFilter(self._redaction_filter)
        if self._owns_http:
            self.http.close()

    def __enter__(self) -> EODHDClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            value = response.headers.get("Retry-After")
            if value:
                try:
                    return min(float(value), self.config.backoff_max_seconds)
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(value)
                        now = parsedate_to_datetime(response.headers["Date"])
                        return max(
                            0.0,
                            min((retry_at - now).total_seconds(), self.config.backoff_max_seconds),
                        )
                    except (KeyError, TypeError, ValueError, OverflowError):
                        pass
        exponential = min(
            self.config.backoff_base_seconds * (2**attempt),
            self.config.backoff_max_seconds,
        )
        jitter = float(self._rng.uniform(0.0, self.config.jitter_seconds))
        return cast(float, exponential + jitter)

    def _request_json(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        refresh: bool,
    ) -> tuple[Any, CachedPayload]:
        public_params = {**params, "fmt": "json"}
        sanitized_params = {
            key: value for key, value in public_params.items() if key.lower() != "api_token"
        }
        request_key = self.cache.request_key(endpoint, public_params)
        if not refresh:
            cached = self.cache.load(request_key)
            if cached is not None:
                log.info("EODHD cache hit endpoint=%s params=%s", endpoint, sanitized_params)
                try:
                    return json.loads(cached.body), cached
                except json.JSONDecodeError as exc:
                    raise EODHDSchemaError(
                        f"Cached EODHD response is not valid JSON for {endpoint}"
                    ) from exc

        authenticated = {**public_params, "api_token": self._token}
        log.info("EODHD request endpoint=%s params=%s", endpoint, sanitized_params)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            response: httpx.Response | None = None
            try:
                response = self.http.get(endpoint, params=authenticated)
                if response.status_code in self._RETRYABLE:
                    if attempt + 1 >= self.config.max_retries:
                        raise EODHDRequestError(
                            f"EODHD {endpoint} failed with HTTP {response.status_code} "
                            f"after {self.config.max_retries} attempts"
                        )
                    delay = self._delay(attempt, response)
                    log.warning(
                        "EODHD retry endpoint=%s status=%d attempt=%d delay=%.2fs",
                        endpoint,
                        response.status_code,
                        attempt + 1,
                        delay,
                    )
                    self._sleep(delay)
                    continue
                response.raise_for_status()
                cached = self.cache.store(
                    endpoint=endpoint,
                    params=public_params,
                    status_code=response.status_code,
                    body=response.content,
                )
                try:
                    return response.json(), cached
                except json.JSONDecodeError as exc:
                    raise EODHDSchemaError(
                        f"EODHD returned invalid JSON for {endpoint}; raw body was preserved"
                    ) from exc
            except EODHDError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                delay = self._delay(attempt, response)
                log.warning(
                    "EODHD transport retry endpoint=%s attempt=%d delay=%.2fs",
                    endpoint,
                    attempt + 1,
                    delay,
                )
                self._sleep(delay)
            except httpx.HTTPStatusError as exc:
                raise EODHDRequestError(
                    f"EODHD {endpoint} rejected the request with HTTP {exc.response.status_code}"
                ) from exc
        raise EODHDRequestError(
            f"EODHD {endpoint} failed after {self.config.max_retries} attempts"
        ) from last_error

    def fetch_news(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        max_articles: int,
        refresh: bool = False,
    ) -> list[NewsArticle]:
        """Download, validate, normalize, paginate, and deduplicate news."""

        if max_articles <= 0:
            raise ValueError("max_articles must be positive")
        articles: dict[str, NewsArticle] = {}
        offset = 0
        seen_pages: set[str] = set()
        while len(articles) < max_articles:
            limit = min(self.config.news_page_size, max_articles - len(articles))
            payload, cached = self._request_json(
                "/api/news",
                {
                    "s": ticker.upper(),
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                    "limit": limit,
                    "offset": offset,
                },
                refresh=refresh,
            )
            if not isinstance(payload, list):
                raise EODHDSchemaError("EODHD news response must be a JSON array")
            page_signature = cached.metadata.response_hash
            if page_signature in seen_pages:
                raise EODHDSchemaError("EODHD repeated a news page during pagination")
            seen_pages.add(page_signature)
            try:
                page = [EODHDNewsItem.model_validate(item) for item in payload]
            except ValidationError as exc:
                raise EODHDSchemaError(
                    "EODHD news response failed schema validation; raw response was preserved"
                ) from exc
            for item in page:
                article = NewsArticle.from_provider(
                    item,
                    retrieved_at=cached.metadata.fetched_at,
                    raw_response_hash=cached.metadata.response_hash,
                )
                articles.setdefault(article.article_id, article)
                if len(articles) >= max_articles:
                    break
            if len(page) < limit:
                break
            offset += limit
        return sorted(
            articles.values(), key=lambda item: (item.provider_timestamp, item.article_id)
        )

    def fetch_eod_prices(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        refresh: bool = False,
    ) -> list[EODPrice]:
        payload, _ = self._request_json(
            f"/api/eod/{ticker.upper()}",
            {
                "from": start.isoformat(),
                "to": end.isoformat(),
                "period": "d",
                "order": "a",
            },
            refresh=refresh,
        )
        if not isinstance(payload, list):
            raise EODHDSchemaError("EODHD EOD response must be a JSON array")
        try:
            prices = [EODPrice.model_validate(item) for item in payload]
        except ValidationError as exc:
            raise EODHDSchemaError(
                "EODHD EOD response failed schema validation; raw response was preserved"
            ) from exc
        unique = {item.date: item for item in prices}
        return [unique[key] for key in sorted(unique)]
