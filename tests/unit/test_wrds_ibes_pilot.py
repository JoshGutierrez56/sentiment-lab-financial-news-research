from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from sentiment_lab.expectation_adjusted import wrds_ibes
from sentiment_lab.expectation_adjusted.wrds_ibes import (
    LIVE_WRDS_ENVIRONMENT_FLAG,
    MAX_PILOT_EVENTS,
    PILOT_COLUMNS,
    WRDSConnectionRunner,
    build_wrds_ibes_eps_pilot_sql,
    live_wrds_is_enabled,
    run_wrds_ibes_eps_pilot,
)


def _pilot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ibes_ticker": ["ACME", "BETA"],
            "ibes_cusip": ["00000010", "00000020"],
            "official_ticker": ["ACME", "BETA"],
            "company_name": ["Acme Corp", "Beta Corp"],
            "fiscal_period_end": [date(2025, 12, 31), date(2025, 12, 31)],
            "measure": ["EPS", "EPS"],
            "periodicity": ["QTR", "QTR"],
            "actual_announce_date": [date(2026, 2, 3), date(2026, 2, 4)],
            "actual_announce_time": [time(16, 5), time(7, 0)],
            "actual_activation_date": [date(2026, 2, 3), date(2026, 2, 4)],
            "actual_activation_time": [time(16, 6), time(7, 1)],
            "actual_unadjusted": [1.2, 0.8],
            "actual_currency": ["USD", "USD"],
            "consensus_statistical_period": [date(2026, 1, 15), date(2026, 1, 15)],
            "consensus_fiscal_period": ["QTR", "QTR"],
            "forecast_period_indicator": ["6", "6"],
            "estimate_flag": ["P", "P"],
            "consensus_currency": ["USD", "USD"],
            "contributor_count": [12.0, 8.0],
            "revisions_up": [2.0, 1.0],
            "revisions_down": [1.0, 2.0],
            "consensus_median_unadjusted": [1.0, 0.9],
            "consensus_mean_unadjusted": [1.0, 0.9],
            "consensus_stdev_unadjusted": [0.1, 0.2],
            "consensus_high_unadjusted": [1.1, 1.0],
            "consensus_low_unadjusted": [0.9, 0.7],
            "permno": [10001, 10002],
            "link_start": [date(2020, 1, 1), date(2020, 1, 1)],
            "link_end": [None, None],
            "link_score": [0.0, 1.0],
            "crsp_consensus_trading_date": [date(2026, 1, 15), date(2026, 1, 15)],
            "cfacshr_consensus_date": [2.0, 1.0],
            "crsp_report_trading_date": [date(2026, 2, 3), date(2026, 2, 4)],
            "cfacshr_report_date": [1.0, 1.0],
        }
    )


class FakeRunner:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.query = ""

    def run(self, query: str) -> pl.DataFrame:
        self.query = query
        return self.frame


def test_sql_uses_strictly_prior_unadjusted_eps_and_crsp_split_factors() -> None:
    query = build_wrds_ibes_eps_pilot_sql(
        start_date=date(2025, 1, 1),
        end_date_exclusive=date(2026, 1, 1),
        max_events=25,
    )
    assert "ibes.actu_epsus" in query
    assert "ibes.statsumu_epsus" in query
    assert "wrdsapps_link_crsp_ibes.ibcrsphist" in query
    assert "crsp.dsf_v2" in query
    assert "daily.dlycumfacshr" in query
    assert "summary.statpers < a.anndats" in query
    assert "candidate_link.score <= 1" in query
    assert query.endswith("LIMIT 25")


@pytest.mark.parametrize("max_events", [0, MAX_PILOT_EVENTS + 1])
def test_sql_rejects_requests_outside_hard_cap(max_events: int) -> None:
    with pytest.raises(ValueError, match="max_events"):
        build_wrds_ibes_eps_pilot_sql(
            start_date=date(2025, 1, 1),
            end_date_exclusive=date(2026, 1, 1),
            max_events=max_events,
        )


def test_sql_rejects_invalid_date_window() -> None:
    with pytest.raises(ValueError, match="start_date"):
        build_wrds_ibes_eps_pilot_sql(
            start_date=date(2026, 1, 1),
            end_date_exclusive=date(2026, 1, 1),
            max_events=1,
        )


def test_pilot_writes_only_private_artifacts_and_a_safe_receipt(tmp_path: Path) -> None:
    frame = _pilot_frame()
    runner = FakeRunner(frame)
    private_output = tmp_path / "data" / "private" / "pilot"
    artifacts = run_wrds_ibes_eps_pilot(
        runner,
        repository_root=tmp_path,
        output_directory=private_output,
        start_date=date(2025, 1, 1),
        end_date_exclusive=date(2026, 7, 22),
        max_events=2,
    )

    assert artifacts.data_path.exists()
    assert artifacts.receipt_path.exists()
    assert artifacts.manual_validation_path.exists()
    assert artifacts.data_path.resolve().is_relative_to((tmp_path / "data" / "private").resolve())
    assert artifacts.receipt["rows_written"] == 2
    assert artifacts.receipt["split_factor_change_rows"] == 1
    assert artifacts.receipt["performance_metrics_computed"] is False
    assert "summary.statpers < a.anndats" in runner.query

    receipt_text = artifacts.receipt_path.read_text(encoding="utf-8")
    assert "Acme Corp" not in receipt_text
    assert "ACME" not in receipt_text
    assert "sharpe" not in receipt_text.lower()
    assert '"ic"' not in receipt_text.lower()

    manual = json.loads(artifacts.manual_validation_path.read_text(encoding="utf-8"))
    assert manual["contains_restricted_wrds_data"] is True
    assert manual["observation"]["ibes_ticker"] == "ACME"


def test_pilot_rejects_non_private_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data/private"):
        run_wrds_ibes_eps_pilot(
            FakeRunner(_pilot_frame()),
            repository_root=tmp_path,
            output_directory=tmp_path / "reports" / "pilot",
            start_date=date(2025, 1, 1),
            end_date_exclusive=date(2026, 7, 22),
            max_events=2,
        )


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda frame: frame.drop("permno"), "missing required columns"),
        (
            lambda frame: frame.with_columns(
                pl.col("actual_announce_date").alias("consensus_statistical_period")
            ),
            "strictly precede",
        ),
        (lambda frame: frame.with_columns(pl.lit("EUR").alias("consensus_currency")), "currencies"),
        (lambda frame: frame.with_columns(pl.lit(2.0).alias("link_score")), "link score"),
        (
            lambda frame: frame.with_columns(pl.lit(0.0).alias("cfacshr_report_date")),
            "must be positive",
        ),
    ],
)
def test_pilot_rejects_invalid_source_contract(
    tmp_path: Path,
    mutator: object,
    error: str,
) -> None:
    mutate = mutator
    assert callable(mutate)
    with pytest.raises(ValueError, match=error):
        run_wrds_ibes_eps_pilot(
            FakeRunner(mutate(_pilot_frame())),
            repository_root=tmp_path,
            output_directory=tmp_path / "data" / "private" / "pilot",
            start_date=date(2025, 1, 1),
            end_date_exclusive=date(2026, 7, 22),
            max_events=2,
        )


def test_pilot_rejects_empty_oversized_and_duplicate_frames(tmp_path: Path) -> None:
    private_output = tmp_path / "data" / "private" / "pilot"
    empty = _pilot_frame().head(0)
    with pytest.raises(ValueError, match="no eligible"):
        run_wrds_ibes_eps_pilot(
            FakeRunner(empty),
            repository_root=tmp_path,
            output_directory=private_output,
            start_date=date(2025, 1, 1),
            end_date_exclusive=date(2026, 7, 22),
            max_events=2,
        )

    oversized = pl.concat([_pilot_frame(), _pilot_frame().slice(0, 1)])
    with pytest.raises(ValueError, match="exceeded"):
        run_wrds_ibes_eps_pilot(
            FakeRunner(oversized),
            repository_root=tmp_path,
            output_directory=private_output,
            start_date=date(2025, 1, 1),
            end_date_exclusive=date(2026, 7, 22),
            max_events=2,
        )

    duplicated = pl.concat([_pilot_frame().slice(0, 1), _pilot_frame().slice(0, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        run_wrds_ibes_eps_pilot(
            FakeRunner(duplicated),
            repository_root=tmp_path,
            output_directory=private_output,
            start_date=date(2025, 1, 1),
            end_date_exclusive=date(2026, 7, 22),
            max_events=2,
        )


def test_live_gate_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIVE_WRDS_ENVIRONMENT_FLAG, raising=False)
    assert live_wrds_is_enabled() is False
    monkeypatch.setenv(LIVE_WRDS_ENVIRONMENT_FLAG, "1")
    assert live_wrds_is_enabled() is True


def test_repository_ignores_restricted_private_output() -> None:
    repository_root = Path(__file__).parents[2]
    ignore_text = (repository_root / ".gitignore").read_text(encoding="utf-8")
    assert "data/private/**" in ignore_text


def test_wrds_connection_runner_converts_results_without_printing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeConnection:
        def __init__(self, **_: object) -> None:
            print("client chatter")

        def raw_sql(self, _: str) -> object:
            print("query chatter")
            return _pilot_frame().to_pandas()

        def close(self) -> None:
            print("close chatter")

    monkeypatch.setattr(
        wrds_ibes,
        "importlib",
        SimpleNamespace(import_module=lambda _: SimpleNamespace(Connection=FakeConnection)),
    )
    runner = WRDSConnectionRunner(username="private-user")
    frame = runner.run("SELECT fixed_query")
    runner.close()
    assert frame.columns == list(PILOT_COLUMNS)
    assert capsys.readouterr().out == ""


def test_wrds_connection_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(_: str) -> object:
        raise RuntimeError("secret-host private-user")

    monkeypatch.setattr(
        wrds_ibes,
        "importlib",
        SimpleNamespace(import_module=fail_import),
    )
    with pytest.raises(RuntimeError, match="connection failed") as error:
        WRDSConnectionRunner(username="private-user")
    assert "private-user" not in str(error.value)
