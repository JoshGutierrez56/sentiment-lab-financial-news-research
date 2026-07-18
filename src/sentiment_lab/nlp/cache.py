"""Content-addressed OpenAI classification cache."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from sentiment_lab.data.cache import stable_json
from sentiment_lab.nlp.schemas import ArticleAssessment, ClassificationRecord


def assessment_hash(assessment: ArticleAssessment) -> str:
    material = stable_json(assessment.model_dump(mode="json"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ClassificationCache:
    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root) / "features" / "openai_cache"

    @staticmethod
    def key(
        *,
        article_content: str,
        ticker: str,
        company_name: str,
        prompt_version: str,
        schema_version: str,
        model: str,
    ) -> tuple[str, str]:
        input_hash = hashlib.sha256(article_content.encode("utf-8")).hexdigest()
        material = stable_json(
            {
                "input_hash": input_hash,
                "ticker": ticker.upper(),
                "company_name": company_name,
                "prompt_version": prompt_version,
                "schema_version": schema_version,
                "model": model,
            }
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest(), input_hash

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> ClassificationRecord | None:
        path = self._path(key)
        if not path.is_file():
            return None
        record = ClassificationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if assessment_hash(record.assessment) != record.output_hash:
            raise ValueError(f"OpenAI cache output hash mismatch for {path}")
        return record.model_copy(update={"from_cache": True})

    def store(self, record: ClassificationRecord) -> Path:
        path = self._path(record.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path
