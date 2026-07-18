"""Cost-bounded OpenAI Batch API adapter for structured article classification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError

from sentiment_lab.config.models import BatchModelPricing, OpenAIConfig
from sentiment_lab.data.cache import stable_json
from sentiment_lab.nlp.schemas import ArticleAssessment, ModelUsage


class OpenAIClassificationError(RuntimeError):
    """A sanitized, actionable Batch API failure."""


class OpenAIBudgetError(OpenAIClassificationError):
    """Raised before submission when conservative maximum cost exceeds the run budget."""


@dataclass(frozen=True)
class BatchItem:
    custom_id: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class ModelCall:
    assessment: ArticleAssessment
    usage: ModelUsage
    response_id: str | None
    response_model: str
    batch_id: str
    batch_custom_id: str


@dataclass(frozen=True)
class BatchFailure:
    reason: str
    usage: ModelUsage
    response_id: str | None
    response_model: str
    batch_id: str
    batch_custom_id: str


@dataclass(frozen=True)
class BatchExecution:
    requested_model: str
    stage: str
    batch_id: str
    input_file_id: str
    output_file_id: str | None
    calls: dict[str, ModelCall]
    failures: dict[str, BatchFailure]
    maximum_estimated_cost_usd: float


_TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatchClient:
    """Submit one-model JSONL batches to `/v1/responses` and resume them idempotently."""

    def __init__(
        self,
        api_key: str,
        config: OpenAIConfig,
        data_root: str | Path,
        *,
        sdk_client: Any | None = None,
        sleeper: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank")
        self.config = config
        self.client = sdk_client or OpenAI(
            api_key=api_key.strip(),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self.state_root = Path(data_root) / "raw" / "openai_batches"
        self._sleep = sleeper
        self._monotonic = monotonic

    def _pricing(self, model: str) -> BatchModelPricing:
        try:
            return self.config.batch_pricing[model]
        except KeyError as exc:  # pragma: no cover - guarded by configuration validation
            raise OpenAIClassificationError(f"No configured Batch API pricing for {model}") from exc

    def prompt_cache_key(self, model: str, prompt_version: str, schema_version: str) -> str:
        material = (
            f"{self.config.prompt_cache_key_prefix}:{model}:{prompt_version}:{schema_version}"
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        prefix = self.config.prompt_cache_key_prefix[:32]
        return f"{prefix}:{digest}"

    def request_body(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_output_tokens: int,
        prompt_version: str,
        schema_version: str,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort},
            "store": False,
            "prompt_cache_key": self.prompt_cache_key(model, prompt_version, schema_version),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "article_assessment_v2",
                    "strict": True,
                    "schema": ArticleAssessment.model_json_schema(),
                }
            },
        }

    def estimate_maximum_cost(
        self,
        items: list[BatchItem],
        *,
        model: str,
        max_output_tokens: int,
        prompt_version: str,
        schema_version: str,
    ) -> float:
        """Conservatively treat each UTF-8 request byte as an input token."""

        pricing = self._pricing(model)
        total = 0.0
        for item in items:
            body = self.request_body(
                item.messages,
                model=model,
                max_output_tokens=max_output_tokens,
                prompt_version=prompt_version,
                schema_version=schema_version,
            )
            input_upper_bound = (
                len(stable_json(body).encode("utf-8")) + self.config.input_token_estimate_overhead
            )
            total += (
                input_upper_bound * pricing.input_per_million
                + max_output_tokens * pricing.output_per_million
            ) / 1_000_000.0
        return total * self.config.regional_processing_multiplier

    def _usage(self, body: dict[str, Any], requested_model: str) -> ModelUsage:
        raw = body.get("usage") or {}
        input_tokens = int(raw.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or 0)
        input_details = raw.get("input_tokens_details") or {}
        output_details = raw.get("output_tokens_details") or {}
        cached_tokens = int(input_details.get("cached_tokens") or 0)
        reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
        pricing = self._pricing(requested_model)
        uncached_tokens = max(0, input_tokens - cached_tokens)
        cost = (
            uncached_tokens * pricing.input_per_million
            + cached_tokens * pricing.cached_input_per_million
            + output_tokens * pricing.output_per_million
        ) / 1_000_000.0
        return ModelUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            estimated_cost_usd=cost * self.config.regional_processing_multiplier,
        )

    def audit_saved_batch_usage(self, custom_ids: set[str]) -> dict[str, int | float]:
        """Reconcile billed usage across completed original and repair batches.

        A cached rerun has zero *new* cost, but its experiment still incurred the
        original calls. Scanning immutable Batch output files also captures malformed
        structured outputs that consumed tokens before a successful repair.
        """

        totals: dict[str, int | float] = {
            "batch_count": 0,
            "request_attempts": 0,
            "first_pass_request_attempts": 0,
            "escalation_request_attempts": 0,
            "unique_first_pass_articles": 0,
            "unique_escalated_articles": 0,
            "structured_output_failures": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "actual_api_cost_usd": 0.0,
            "preflight_maximum_estimated_cost_usd_all_attempts": 0.0,
            "largest_batch_preflight_maximum_usd": 0.0,
        }
        if not custom_ids or not self.state_root.is_dir():
            return totals
        unique_by_stage: dict[str, set[str]] = {"first_pass": set(), "escalation": set()}
        for state_path in sorted(self.state_root.glob("*.json")):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_ids = {str(value) for value in state.get("custom_ids") or []}
            relevant = state_ids & custom_ids
            output_file_id = state.get("output_file_id")
            model = str(state.get("model") or "")
            stage = str(state.get("stage") or "")
            if not relevant or state.get("status") != "completed" or not output_file_id:
                continue
            input_file_id = state.get("input_file_id")
            batch_maximum = 0.0
            if input_file_id:
                input_response = self._safe_api_call(
                    lambda input_file_id=input_file_id: self.client.files.content(input_file_id),
                    f"Could not audit saved OpenAI input {input_file_id}",
                )
                pricing = self._pricing(model)
                for line in self._file_text(input_response).splitlines():
                    if not line.strip():
                        continue
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(request.get("custom_id") or "") not in relevant:
                        continue
                    body = request.get("body") or {}
                    input_upper_bound = (
                        len(stable_json(body).encode("utf-8"))
                        + self.config.input_token_estimate_overhead
                    )
                    max_output_tokens = int(body.get("max_output_tokens") or 0)
                    batch_maximum += (
                        input_upper_bound * pricing.input_per_million
                        + max_output_tokens * pricing.output_per_million
                    ) / 1_000_000.0
                batch_maximum *= self.config.regional_processing_multiplier
                totals["preflight_maximum_estimated_cost_usd_all_attempts"] += batch_maximum
                totals["largest_batch_preflight_maximum_usd"] = max(
                    float(totals["largest_batch_preflight_maximum_usd"]), batch_maximum
                )
            response = self._safe_api_call(
                lambda output_file_id=output_file_id: self.client.files.content(output_file_id),
                f"Could not audit saved OpenAI output {output_file_id}",
            )
            totals["batch_count"] += 1
            for line in self._file_text(response).splitlines():
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(result.get("custom_id") or "") not in relevant:
                    continue
                wrapper = result.get("response") or {}
                if wrapper.get("status_code") != 200:
                    continue
                body = wrapper.get("body") or {}
                usage = self._usage(body, model)
                totals["request_attempts"] += 1
                if stage in unique_by_stage:
                    totals[f"{stage}_request_attempts"] += 1
                    unique_by_stage[stage].add(str(result.get("custom_id")))
                totals["input_tokens"] += usage.input_tokens
                totals["cached_input_tokens"] += usage.cached_input_tokens
                totals["output_tokens"] += usage.output_tokens
                totals["reasoning_tokens"] += usage.reasoning_tokens
                totals["actual_api_cost_usd"] += usage.estimated_cost_usd
                output_text = self._output_text(body)
                try:
                    if output_text is None:
                        raise ValueError("missing structured output")
                    ArticleAssessment.model_validate_json(output_text)
                except (ValidationError, ValueError):
                    totals["structured_output_failures"] += 1
        totals["unique_first_pass_articles"] = len(unique_by_stage["first_pass"])
        totals["unique_escalated_articles"] = len(unique_by_stage["escalation"])
        return totals

    @staticmethod
    def _output_text(body: dict[str, Any]) -> str | None:
        for output in body.get("output") or []:
            if output.get("type") != "message":
                continue
            for part in output.get("content") or []:
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    return str(part["text"])
                if part.get("type") == "refusal":
                    return None
        return None

    def _state_path(self, request_set_hash: str) -> Path:
        return self.state_root / f"{request_set_hash}.json"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _file_text(response: Any) -> str:
        text_value = getattr(response, "text", "")
        if callable(text_value):
            text_value = text_value()
        if isinstance(text_value, bytes):
            return text_value.decode("utf-8")
        return str(text_value)

    def _safe_api_call(self, operation: Any, failure_message: str) -> Any:
        try:
            return operation()
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            raise OpenAIClassificationError(failure_message) from exc
        except APIStatusError as exc:
            detail = self._status_error_detail(exc)
            suffix = f": {detail}" if detail else ""
            raise OpenAIClassificationError(
                f"{failure_message}; OpenAI returned HTTP {exc.status_code}{suffix}"
            ) from exc

    @staticmethod
    def _status_error_detail(exc: Any) -> str | None:
        """Return only the API's short message, with credentials defensively redacted."""

        body = getattr(exc, "body", None)
        message: Any = None
        if isinstance(body, dict):
            message = body.get("message")
            nested = body.get("error")
            if message is None and isinstance(nested, dict):
                message = nested.get("message")
        if not isinstance(message, str):
            message = getattr(exc, "message", None)
        if not isinstance(message, str) or not message.strip():
            return None
        normalized = " ".join(message.split())
        normalized = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", normalized)
        normalized = re.sub(r"\bsk-[A-Za-z0-9_-]{10,}\b", "[REDACTED]", normalized)
        return normalized[:500]

    def _submit_or_resume(
        self,
        *,
        request_set_hash: str,
        jsonl_bytes: bytes,
        model: str,
        stage: str,
        custom_ids: list[str],
    ) -> tuple[Any, str, Path]:
        state_path = self._state_path(request_set_hash)
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("model") != model or state.get("custom_ids") != custom_ids:
                raise OpenAIClassificationError(
                    f"Stored OpenAI batch state does not match request set {request_set_hash}"
                )
            batch_id = str(state["batch_id"])
            batch = self._safe_api_call(
                lambda batch_id=batch_id: self.client.batches.retrieve(batch_id),
                f"Could not resume OpenAI batch {batch_id}",
            )
            return batch, str(state["input_file_id"]), state_path

        uploaded = self._safe_api_call(
            lambda: self.client.files.create(
                file=(f"sentiment-{stage}.jsonl", jsonl_bytes, "application/jsonl"),
                purpose="batch",
            ),
            "Could not upload the OpenAI batch input file",
        )
        batch = self._safe_api_call(
            lambda: self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={"project": "sentiment-lab", "stage": stage, "model": model},
            ),
            "Could not create the OpenAI batch",
        )
        self._write_json_atomic(
            state_path,
            {
                "request_set_hash": request_set_hash,
                "model": model,
                "stage": stage,
                "custom_ids": custom_ids,
                "input_file_id": uploaded.id,
                "batch_id": batch.id,
                "status": str(batch.status),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return batch, str(uploaded.id), state_path

    def _wait(self, batch: Any, input_file_id: str, state_path: Path) -> Any:
        deadline = self._monotonic() + self.config.batch_wait_timeout_seconds
        while str(batch.status) not in _TERMINAL_BATCH_STATUSES:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise OpenAIClassificationError(
                    f"OpenAI batch {batch.id} is still {batch.status}; rerun the same command "
                    "to resume it without resubmitting any article"
                )
            self._sleep(min(self.config.batch_poll_interval_seconds, remaining))
            batch_id = batch.id
            batch = self._safe_api_call(
                lambda batch_id=batch_id: self.client.batches.retrieve(batch_id),
                f"Could not poll OpenAI batch {batch_id}",
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(
                {
                    "input_file_id": input_file_id,
                    "batch_id": batch.id,
                    "status": str(batch.status),
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            self._write_json_atomic(state_path, state)
        return batch

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
        if not items:
            raise ValueError("Batch API submission requires at least one item")
        maximum_cost = self.estimate_maximum_cost(
            items,
            model=model,
            max_output_tokens=max_output_tokens,
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
        if maximum_cost > budget_remaining_usd + 1e-12:
            raise OpenAIBudgetError(
                f"OpenAI {stage} batch maximum estimate ${maximum_cost:.6f} exceeds "
                f"remaining budget ${budget_remaining_usd:.6f}; nothing was submitted"
            )

        lines: list[str] = []
        request_fingerprints: list[dict[str, Any]] = []
        for item in items:
            body = self.request_body(
                item.messages,
                model=model,
                max_output_tokens=max_output_tokens,
                prompt_version=prompt_version,
                schema_version=schema_version,
            )
            request = {
                "custom_id": item.custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            lines.append(stable_json(request))
            request_fingerprints.append(
                {
                    "custom_id": item.custom_id,
                    "body_hash": hashlib.sha256(lines[-1].encode()).hexdigest(),
                }
            )
        request_set_hash = hashlib.sha256(
            stable_json(request_fingerprints).encode("utf-8")
        ).hexdigest()
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        custom_ids = [item.custom_id for item in items]
        batch, input_file_id, state_path = self._submit_or_resume(
            request_set_hash=request_set_hash,
            jsonl_bytes=jsonl_bytes,
            model=model,
            stage=stage,
            custom_ids=custom_ids,
        )
        batch = self._wait(batch, input_file_id, state_path)
        status = str(batch.status)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update(
            {
                "status": status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        self._write_json_atomic(state_path, state)
        if status != "completed":
            raise OpenAIClassificationError(
                f"OpenAI batch {batch.id} ended with status {status}; it will not be resubmitted"
            )
        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            raise OpenAIClassificationError(
                f"OpenAI batch {batch.id} completed without an output file"
            )
        response = self._safe_api_call(
            lambda: self.client.files.content(output_file_id),
            f"Could not download OpenAI batch {batch.id} output",
        )
        calls: dict[str, ModelCall] = {}
        failures: dict[str, BatchFailure] = {}
        for line in self._file_text(response).splitlines():
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = str(result.get("custom_id") or "")
            if custom_id not in custom_ids:
                continue
            response_wrapper = result.get("response") or {}
            if response_wrapper.get("status_code") != 200:
                failures[custom_id] = BatchFailure(
                    reason="batch_request_failed",
                    usage=ModelUsage(),
                    response_id=None,
                    response_model=model,
                    batch_id=str(batch.id),
                    batch_custom_id=custom_id,
                )
                continue
            body = response_wrapper.get("body") or {}
            usage = self._usage(body, model)
            response_id = str(body.get("id")) if body.get("id") else None
            response_model = str(body.get("model") or model)
            text = self._output_text(body)
            if text is None:
                failures[custom_id] = BatchFailure(
                    reason="structured_output_missing_or_refused",
                    usage=usage,
                    response_id=response_id,
                    response_model=response_model,
                    batch_id=str(batch.id),
                    batch_custom_id=custom_id,
                )
                continue
            try:
                assessment = ArticleAssessment.model_validate_json(text)
            except (ValidationError, ValueError):
                failures[custom_id] = BatchFailure(
                    reason="structured_output_validation_failed",
                    usage=usage,
                    response_id=response_id,
                    response_model=response_model,
                    batch_id=str(batch.id),
                    batch_custom_id=custom_id,
                )
                continue
            calls[custom_id] = ModelCall(
                assessment=assessment,
                usage=usage,
                response_id=response_id,
                response_model=response_model,
                batch_id=str(batch.id),
                batch_custom_id=custom_id,
            )
        for custom_id in custom_ids:
            if custom_id not in calls and custom_id not in failures:
                failures[custom_id] = BatchFailure(
                    reason="batch_output_missing",
                    usage=ModelUsage(),
                    response_id=None,
                    response_model=model,
                    batch_id=str(batch.id),
                    batch_custom_id=custom_id,
                )
        return BatchExecution(
            requested_model=model,
            stage=stage,
            batch_id=str(batch.id),
            input_file_id=input_file_id,
            output_file_id=str(output_file_id),
            calls=calls,
            failures=failures,
            maximum_estimated_cost_usd=maximum_cost,
        )
