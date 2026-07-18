"""Atomic Parquet/JSON artifact storage and DuckDB view registration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from pydantic import BaseModel


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, data_root: str | Path, duckdb_path: str | Path) -> None:
        self.data_root = Path(data_root)
        self.duckdb_path = Path(duckdb_path)

    @staticmethod
    def _atomic_replace(temporary: Path, final: Path) -> None:
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)

    def write_parquet(self, frame: pl.DataFrame, path: str | Path) -> Path:
        final = Path(path)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_suffix(final.suffix + ".tmp")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        self._atomic_replace(temporary, final)
        return final

    def write_models(self, models: list[BaseModel], path: str | Path) -> Path:
        if not models:
            raise ValueError("Cannot infer a Parquet schema from an empty model list")
        rows = [model.model_dump(mode="python") for model in models]
        return self.write_parquet(pl.DataFrame(rows, infer_schema_length=None), path)

    def write_json(self, value: Any, path: str | Path) -> Path:
        final = Path(path)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_suffix(final.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False),
            encoding="utf-8",
        )
        self._atomic_replace(temporary, final)
        return final

    def register_parquet_view(self, name: str, path: str | Path) -> None:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Unsafe DuckDB view name: {name!r}")
        parquet = Path(path).resolve().as_posix().replace("'", "''")
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.duckdb_path)) as connection:
            connection.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{parquet}')"
            )
