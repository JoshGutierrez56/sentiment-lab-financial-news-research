"""Run the bounded WRDS I/B/E/S quarterly-EPS source audit."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from sentiment_lab.expectation_adjusted.wrds_ibes import (
    LIVE_WRDS_ENVIRONMENT_FLAG,
    MAX_PILOT_EVENTS,
    WRDSConnectionRunner,
    build_wrds_ibes_eps_pilot_sql,
    live_wrds_is_enabled,
    query_sha256,
    run_wrds_ibes_eps_pilot,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dates must use YYYY-MM-DD") from error


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write at most 25 licensed EPS observations under data/private."
    )
    parser.add_argument("--start-date", type=_date, default=date(2025, 1, 1))
    parser.add_argument("--end-date-exclusive", type=_date, default=date(2026, 1, 1))
    parser.add_argument("--max-events", type=int, default=MAX_PILOT_EVENTS)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    repository_root = Path(__file__).resolve().parents[1]
    output_directory = repository_root / "data" / "private" / "wrds_ibes_eps_pilot"
    query = build_wrds_ibes_eps_pilot_sql(
        start_date=arguments.start_date,
        end_date_exclusive=arguments.end_date_exclusive,
        max_events=arguments.max_events,
    )

    if not arguments.live:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "query_sha256": query_sha256(query),
                    "requested_event_cap": arguments.max_events,
                    "performance_metrics_computed": False,
                    "live_enable_flag_required": LIVE_WRDS_ENVIRONMENT_FLAG,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not live_wrds_is_enabled():
        raise RuntimeError(
            f"Set {LIVE_WRDS_ENVIRONMENT_FLAG}=1 to authorize the bounded live pilot"
        )
    username = os.getenv("WRDS_USERNAME", "")
    if not username:
        raise RuntimeError("WRDS_USERNAME is required; the password must remain in pgpass")

    runner = WRDSConnectionRunner(username=username)
    try:
        artifacts = run_wrds_ibes_eps_pilot(
            runner,
            repository_root=repository_root,
            output_directory=output_directory,
            start_date=arguments.start_date,
            end_date_exclusive=arguments.end_date_exclusive,
            max_events=arguments.max_events,
        )
    finally:
        runner.close()

    print(
        json.dumps(
            {
                "status": artifacts.receipt["status"],
                "rows_written": artifacts.receipt["rows_written"],
                "requested_event_cap": artifacts.receipt["requested_event_cap"],
                "performance_metrics_computed": False,
                "private_output_directory": str(output_directory.relative_to(repository_root)),
                "manual_validation_file": str(
                    artifacts.manual_validation_path.relative_to(repository_root)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
