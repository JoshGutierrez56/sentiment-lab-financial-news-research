"""Official OpenAI Responses API structured-output adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from sentiment_lab.config.models import OpenAIConfig
from sentiment_lab.nlp.schemas import ArticleAssessment, ModelUsage


class OpenAIClassificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelCall:
    assessment: ArticleAssessment
    usage: ModelUsage
    response_id: str | None
    response_model: str


class OpenAIArticleClient:
    """Use `responses.parse` with a Pydantic schema, per current official docs."""

    def __init__(
        self,
        api_key: str,
        model: str,
        config: OpenAIConfig,
        *,
        sdk_client: Any | None = None,
        sleeper: Any = time.sleep,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("OpenAI API key and model must not be blank")
        self.model = model.strip()
        self.config = config
        self.client = sdk_client or OpenAI(
            api_key=api_key.strip(),
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._sleep = sleeper

    def classify(self, messages: list[dict[str, str]]) -> ModelCall:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": messages,
                    "text_format": ArticleAssessment,
                    "max_output_tokens": self.config.max_output_tokens,
                }
                if self.config.temperature is not None:
                    kwargs["temperature"] = self.config.temperature
                response = self.client.responses.parse(**kwargs)
                parsed = response.output_parsed
                if parsed is None:
                    refusal = getattr(response, "output_text", "")
                    raise OpenAIClassificationError(
                        f"OpenAI returned no parsed assessment: {refusal or 'empty response'}"
                    )
                assessment = ArticleAssessment.model_validate(parsed)
                usage_obj = getattr(response, "usage", None)
                input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
                output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
                estimated_cost: float | None = None
                if (
                    self.config.input_cost_per_million is not None
                    and self.config.output_cost_per_million is not None
                ):
                    estimated_cost = (
                        input_tokens * self.config.input_cost_per_million
                        + output_tokens * self.config.output_cost_per_million
                    ) / 1_000_000.0
                return ModelCall(
                    assessment=assessment,
                    usage=ModelUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        estimated_cost_usd=estimated_cost,
                    ),
                    response_id=getattr(response, "id", None),
                    response_model=str(getattr(response, "model", self.model)),
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500 and exc.status_code != 429:
                    raise OpenAIClassificationError(
                        f"OpenAI rejected the classification request with HTTP {exc.status_code}"
                    ) from exc
                last_error = exc
            if attempt + 1 < self.config.max_retries:
                self._sleep(min(2**attempt, 20))
        raise OpenAIClassificationError(
            f"OpenAI classification failed after {self.config.max_retries} attempts"
        ) from last_error
