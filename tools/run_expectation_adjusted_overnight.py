"""Run the locked expectation-adjusted overnight exploratory benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentiment_lab.expectation_adjusted.overnight import (
    EXPLORATORY_LABEL,
    PRIVATE_ROOT,
    load_overnight_config,
    run_overnight_benchmark,
)
from sentiment_lab.expectation_adjusted.wrds_ibes import (
    WRDSConnectionRunner,
    derive_wrds_username_from_pgpass,
)


def _progress_path(repository_root: Path) -> Path:
    return repository_root / PRIVATE_ROOT / "progress.json"


def _write_progress(repository_root: Path, payload: dict[str, object]) -> None:
    path = _progress_path(repository_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/expectation_adjusted_news_overnight_exploratory_v0.yaml"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--live-wrds", action="store_true")
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    config = load_overnight_config(repository_root / args.config)
    runner: WRDSConnectionRunner | None = None
    try:
        if args.live_wrds:
            runner = WRDSConnectionRunner(username=derive_wrds_username_from_pgpass())
        artifacts = run_overnight_benchmark(
            config,
            repository_root=repository_root,
            wrds_runner=runner,
        )
        payload: dict[str, object] = {
            "status": artifacts.status,
            "phase": "terminal",
            "exploratory_label": EXPLORATORY_LABEL,
            "coverage_counts": artifacts.coverage,
            "aggregate_metrics": artifacts.metrics,
            "limitations": artifacts.limitations,
            "report_path": str(artifacts.report_path),
            "manifest_path": str(artifacts.manifest_path),
            "metrics_path": str(artifacts.metrics_path) if artifacts.metrics_path else None,
        }
        _write_progress(repository_root, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except Exception as exc:
        payload = {
            "status": "blocked",
            "phase": "terminal",
            "exploratory_label": EXPLORATORY_LABEL,
            "blocker": str(exc),
            "limitations": [
                "Historical exploratory benchmark only.",
                "Run did not broaden WRDS scope or relax coverage gates.",
            ],
        }
        _write_progress(repository_root, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        raise SystemExit(1) from None
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
