"""Qlib-style configuration and canonical, cache-only run artifacts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict

from sentiment_lab.data.cache import stable_json
from sentiment_lab.data.storage import ArtifactStore, file_sha256


class RedesignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: dict[str, Any]
    feature_generation: dict[str, Any]
    model: dict[str, Any]
    signal: dict[str, Any]
    validation: dict[str, Any]
    execution: dict[str, Any]
    costs: dict[str, Any]
    portfolio: dict[str, Any]
    reporting: dict[str, Any]
    random_seed: int = 20260719


REQUIRED_PARQUET = (
    "predictions.parquet",
    "orders.parquet",
    "fills.parquet",
    "rejected_orders.parquet",
    "positions.parquet",
    "daily_returns.parquet",
    "cost_breakdown.parquet",
)


def configuration_hash(config: RedesignConfig) -> str:
    return hashlib.sha256(stable_json(config.model_dump(mode="json")).encode()).hexdigest()


def _git(command: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path


def write_cache_only_run(
    config: RedesignConfig,
    *,
    root: Path,
    repository_root: Path,
    data_hashes: dict[str, str],
    feature_hashes: dict[str, str],
    model_metadata: dict[str, Any],
    frames: dict[str, pl.DataFrame],
    metrics: dict[str, Any],
) -> RunPaths:
    """Write an immutable new result directory; never touches source study artifacts."""
    run_root = root / configuration_hash(config)[:16]
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite canonical run {run_root}")
    store = ArtifactStore(root, root / "runs.duckdb")
    run_root.mkdir(parents=True)
    (run_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    for name in REQUIRED_PARQUET:
        frame = frames.get(name)
        if frame is None:
            frame = pl.DataFrame({"_empty": pl.Series([], dtype=pl.String)})
        store.write_parquet(frame, run_root / name)
    manifest = {
        "git_commit": _git(["git", "rev-parse", "HEAD"], repository_root),
        "dirty_worktree": bool(_git(["git", "status", "--porcelain"], repository_root)),
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": config.random_seed,
        "configuration_hash": configuration_hash(config),
        "canonical_artifacts": {name: file_sha256(run_root / name) for name in REQUIRED_PARQUET},
        "operational_artifact": "operational.json",
    }
    store.write_json(manifest, run_root / "manifest.json")
    store.write_json(data_hashes, run_root / "data_hashes.json")
    store.write_json(feature_hashes, run_root / "feature_hashes.json")
    store.write_json(model_metadata, run_root / "model_metadata.json")
    store.write_json(metrics, run_root / "metrics.json")
    (run_root / "report.html").write_text(
        "<html><body><h1>Event-surprise cache-only run</h1><pre>"
        + stable_json(metrics)
        + "</pre></body></html>",
        encoding="utf-8",
    )
    # Timing/cache hits vary by machine and are intentionally non-canonical.
    store.write_json({"cache_only": True}, run_root / "operational.json")
    return RunPaths(root=run_root, manifest=run_root / "manifest.json")
