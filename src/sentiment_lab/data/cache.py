"""Immutable raw-response cache with a mutable latest-request index."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentiment_lab.data.schemas import RawResponseMetadata


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class CachedPayload:
    body: bytes
    metadata: RawResponseMetadata
    from_cache: bool


class RawResponseCache:
    """Store every successful provider body before it is normalized."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "raw" / "eodhd"

    @staticmethod
    def request_key(endpoint: str, params: dict[str, Any]) -> str:
        safe = {key: value for key, value in params.items() if key.lower() != "api_token"}
        material = stable_json({"endpoint": endpoint, "params": safe})
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def load(self, request_key: str) -> CachedPayload | None:
        index_path = self.root / "index" / f"{request_key}.json"
        if not index_path.is_file():
            return None
        metadata = RawResponseMetadata.model_validate_json(index_path.read_text(encoding="utf-8"))
        body_path = self.root / metadata.body_path
        if not body_path.is_file():
            raise FileNotFoundError(f"Raw cache index points to missing body: {body_path}")
        body = body_path.read_bytes()
        actual_hash = hashlib.sha256(body).hexdigest()
        if actual_hash != metadata.response_hash:
            raise ValueError(f"Raw cache hash mismatch for {body_path}")
        return CachedPayload(body=body, metadata=metadata, from_cache=True)

    def store(
        self,
        *,
        endpoint: str,
        params: dict[str, Any],
        status_code: int,
        body: bytes,
    ) -> CachedPayload:
        safe_params = {key: value for key, value in params.items() if key.lower() != "api_token"}
        request_key = self.request_key(endpoint, safe_params)
        response_hash = hashlib.sha256(body).hexdigest()
        fetched_at = datetime.now(UTC)
        endpoint_slug = endpoint.strip("/").replace("/", "__") or "root"
        relative_body = Path("responses") / endpoint_slug / f"{response_hash}.json"
        body_path = self.root / relative_body
        if not body_path.exists():
            body_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = body_path.with_name(f".{body_path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(body)
            os.replace(temporary, body_path)

        metadata = RawResponseMetadata(
            provider="eodhd",
            endpoint=endpoint,
            sanitized_params=safe_params,
            request_key=request_key,
            fetched_at=fetched_at,
            status_code=status_code,
            response_hash=response_hash,
            body_path=relative_body.as_posix(),
        )
        immutable_name = (
            f"{request_key[:24]}_{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}"
            f"_{response_hash[:16]}.json"
        )
        immutable_meta = self.root / "metadata" / request_key[:2] / immutable_name
        rendered = metadata.model_dump_json(indent=2)
        _atomic_text(immutable_meta, rendered)
        _atomic_text(self.root / "index" / f"{request_key}.json", rendered)
        return CachedPayload(body=body, metadata=metadata, from_cache=False)
