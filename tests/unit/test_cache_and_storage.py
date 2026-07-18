"""Audit-cache and artifact-storage tests."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from conftest import make_article, make_record
from sentiment_lab.data.cache import RawResponseCache
from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.nlp.cache import ClassificationCache, article_content_hash


def test_raw_cache_is_content_addressed_and_redacts_token(tmp_path: Path) -> None:
    cache = RawResponseCache(tmp_path)
    payload = cache.store(
        endpoint="/api/news",
        params={"s": "AAPL.US", "api_token": "never-store-me", "fmt": "json"},
        status_code=200,
        body=b'[{"title":"test"}]',
    )
    assert payload.from_cache is False
    assert "api_token" not in payload.metadata.sanitized_params
    all_cache_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (tmp_path / "raw" / "eodhd").rglob("*")
        if path.is_file()
    )
    assert "never-store-me" not in all_cache_text
    loaded = cache.load(payload.metadata.request_key)
    assert loaded is not None
    assert loaded.from_cache is True
    assert loaded.body == payload.body
    assert cache.request_key(
        "/api/news", {"s": "AAPL.US", "api_token": "one"}
    ) == cache.request_key("/api/news", {"s": "AAPL.US", "api_token": "two"})


def test_raw_cache_detects_corruption_and_missing_body(tmp_path: Path) -> None:
    cache = RawResponseCache(tmp_path)
    stored = cache.store(endpoint="/api/eod/AAPL.US", params={}, status_code=200, body=b"[]")
    body_path = cache.root / stored.metadata.body_path
    body_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.load(stored.metadata.request_key)
    body_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing body"):
        cache.load(stored.metadata.request_key)


def test_classification_cache_round_trip_marks_cache_hit(tmp_path: Path) -> None:
    article = make_article()
    record = make_record(article)
    cache = ClassificationCache(tmp_path)
    path = cache.store(record)
    assert path.is_file()
    loaded = cache.load(record.cache_key)
    assert loaded is not None
    assert loaded.from_cache is True
    input_hash = article_content_hash(article.title, article.content)
    key = cache.key(
        article_content_hash=input_hash,
        ticker="aapl.us",
        prompt_version="v1",
        schema_version="s1",
        model="m1",
    )
    assert len(key) == len(input_hash) == 64
    variants = {
        cache.key(
            article_content_hash=content_hash,
            ticker=ticker,
            prompt_version=prompt,
            schema_version=schema,
            model=model,
        )
        for content_hash, ticker, prompt, schema, model in [
            (input_hash, "AAPL.US", "v1", "s1", "m1"),
            ("0" * 64, "AAPL.US", "v1", "s1", "m1"),
            (input_hash, "MSFT.US", "v1", "s1", "m1"),
            (input_hash, "AAPL.US", "v2", "s1", "m1"),
            (input_hash, "AAPL.US", "v1", "s2", "m1"),
            (input_hash, "AAPL.US", "v1", "s1", "m2"),
        ]
    }
    assert len(variants) == 6
    with pytest.raises(ValueError, match="Refusing to overwrite conflicting"):
        cache.store(record.model_copy(update={"output_hash": "0" * 64}))


def test_classification_cache_detects_output_tampering(tmp_path: Path) -> None:
    record = make_record(make_article()).model_copy(update={"output_hash": "0" * 64})
    cache = ClassificationCache(tmp_path)
    cache.store(record)
    with pytest.raises(ValueError, match="output hash mismatch"):
        cache.load(record.cache_key)


def test_artifact_store_writes_parquet_json_and_duckdb_view(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, tmp_path / "research.duckdb")
    frame = pl.DataFrame({"article_id": ["a", "b"], "score": [0.1, -0.2]})
    parquet = store.write_parquet(frame, tmp_path / "out" / "events.parquet")
    metadata = store.write_json({"ok": True}, tmp_path / "out" / "metrics.json")
    assert pl.read_parquet(parquet).equals(frame)
    assert json.loads(metadata.read_text(encoding="utf-8")) == {"ok": True}
    assert len(file_sha256(parquet)) == 64
    store.register_parquet_view("events_latest", parquet)
    with duckdb.connect(str(tmp_path / "research.duckdb")) as connection:
        assert connection.execute("select count(*) from events_latest").fetchone() == (2,)
    with pytest.raises(ValueError, match="Unsafe DuckDB view name"):
        store.register_parquet_view("events; drop table x", parquet)


def test_write_models_rejects_empty_schema(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, tmp_path / "research.duckdb")
    with pytest.raises(ValueError, match="empty model list"):
        store.write_models([], tmp_path / "empty.parquet")
