"""Bounded, credential-safe WRDS I/B/E/S EPS pilot ingestion.

The pilot deliberately stops before research feature construction. It preserves
the provider's raw date/time fields, obtains a strictly prior summary estimate,
and collects CRSP share-adjustment factors needed to put unadjusted actuals and
expectations on the same per-share basis.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from sentiment_lab.data.storage import ArtifactStore, file_sha256

MAX_PILOT_EVENTS = 25
MAX_OVERNIGHT_EVENTS = 2500
LIVE_WRDS_ENVIRONMENT_FLAG = "SENTIMENT_LAB_ENABLE_LIVE_WRDS_IBES"

SOURCE_TABLES = (
    "ibes.actu_epsus",
    "ibes.statsumu_epsus",
    "wrdsapps_link_crsp_ibes.ibcrsphist",
    "crsp.dsf_v2",
)

PILOT_COLUMNS = (
    "ibes_ticker",
    "ibes_cusip",
    "official_ticker",
    "company_name",
    "fiscal_period_end",
    "measure",
    "periodicity",
    "actual_announce_date",
    "actual_announce_time",
    "actual_activation_date",
    "actual_activation_time",
    "actual_unadjusted",
    "actual_currency",
    "consensus_statistical_period",
    "consensus_fiscal_period",
    "forecast_period_indicator",
    "estimate_flag",
    "consensus_currency",
    "contributor_count",
    "revisions_up",
    "revisions_down",
    "consensus_median_unadjusted",
    "consensus_mean_unadjusted",
    "consensus_stdev_unadjusted",
    "consensus_high_unadjusted",
    "consensus_low_unadjusted",
    "permno",
    "link_start",
    "link_end",
    "link_score",
    "crsp_consensus_trading_date",
    "cfacshr_consensus_date",
    "crsp_report_trading_date",
    "cfacshr_report_date",
)

OVERNIGHT_COLUMNS = (
    "source_ticker",
    "ibes_ticker",
    "ibes_cusip",
    "official_ticker",
    "fiscal_period_end",
    "measure",
    "periodicity",
    "actual_announce_date",
    "actual_announce_time",
    "actual_activation_date",
    "actual_activation_time",
    "actual_unadjusted",
    "actual_currency",
    "consensus_statistical_period",
    "consensus_fiscal_period",
    "forecast_period_indicator",
    "estimate_flag",
    "consensus_currency",
    "contributor_count",
    "revisions_up",
    "revisions_down",
    "consensus_mean_unadjusted",
    "consensus_stdev_unadjusted",
    "permno",
    "link_start",
    "link_end",
    "link_score",
    "crsp_consensus_trading_date",
    "cfacshr_consensus_date",
    "crsp_report_trading_date",
    "cfacshr_report_date",
)


class QueryRunner(Protocol):
    """Minimal query interface so tests never need a live WRDS dependency."""

    def run(self, query: str) -> pl.DataFrame: ...


@dataclass(frozen=True)
class PilotArtifacts:
    data_path: Path
    receipt_path: Path
    manual_validation_path: Path
    receipt: dict[str, Any]


def _validate_pilot_request(*, start_date: date, end_date_exclusive: date, max_events: int) -> None:
    if not 1 <= max_events <= MAX_PILOT_EVENTS:
        raise ValueError(f"max_events must be between 1 and {MAX_PILOT_EVENTS}")
    if start_date >= end_date_exclusive:
        raise ValueError("start_date must precede end_date_exclusive")


def _base_wrds_symbol(source_ticker: str) -> str:
    normalized = source_ticker.strip().upper()
    if not normalized or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for character in normalized
    ):
        raise ValueError(f"unsafe ticker in frozen universe: {source_ticker!r}")
    return normalized.removesuffix(".US")


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_overnight_request(
    *, tickers: tuple[str, ...], start_date: date, end_date_exclusive: date, max_events: int
) -> None:
    if not tickers:
        raise ValueError("at least one frozen ticker is required")
    if len(set(tickers)) != len(tickers):
        raise ValueError("frozen ticker universe contains duplicates")
    if not 1 <= max_events <= MAX_OVERNIGHT_EVENTS:
        raise ValueError(f"max_events must be between 1 and {MAX_OVERNIGHT_EVENTS}")
    if start_date >= end_date_exclusive:
        raise ValueError("start_date must precede end_date_exclusive")
    for ticker in tickers:
        _base_wrds_symbol(ticker)


def build_wrds_ibes_eps_pilot_sql(
    *, start_date: date, end_date_exclusive: date, max_events: int
) -> str:
    """Build fixed-table SQL for a small, non-performance EPS source audit."""

    _validate_pilot_request(
        start_date=start_date,
        end_date_exclusive=end_date_exclusive,
        max_events=max_events,
    )
    start = start_date.isoformat()
    end = end_date_exclusive.isoformat()
    return f"""
WITH ranked_actuals AS (
    SELECT
        a.*,
        ROW_NUMBER() OVER (
            PARTITION BY a.ticker, a.pends, a.measure, a.pdicity
            ORDER BY a.actdats ASC, a.acttims ASC NULLS LAST,
                     a.anndats ASC, a.anntims ASC NULLS LAST
        ) AS actual_version_rank
    FROM ibes.actu_epsus AS a
    WHERE a.usfirm = 1
      AND UPPER(a.measure) = 'EPS'
      AND UPPER(a.pdicity) = 'QTR'
      AND UPPER(a.curr_act) = 'USD'
      AND a.anndats >= DATE '{start}'
      AND a.anndats < DATE '{end}'
      AND a.pends IS NOT NULL
      AND a.value IS NOT NULL
), candidate_actuals AS (
    SELECT *
    FROM ranked_actuals
    WHERE actual_version_rank = 1
    ORDER BY anndats DESC, ticker, pends DESC
    LIMIT 500
)
SELECT
    a.ticker AS ibes_ticker,
    a.cusip AS ibes_cusip,
    a.oftic AS official_ticker,
    a.cname AS company_name,
    a.pends AS fiscal_period_end,
    a.measure AS measure,
    a.pdicity AS periodicity,
    a.anndats AS actual_announce_date,
    a.anntims AS actual_announce_time,
    a.actdats AS actual_activation_date,
    a.acttims AS actual_activation_time,
    a.value AS actual_unadjusted,
    a.curr_act AS actual_currency,
    s.statpers AS consensus_statistical_period,
    s.fiscalp AS consensus_fiscal_period,
    s.fpi AS forecast_period_indicator,
    s.estflag AS estimate_flag,
    s.curcode AS consensus_currency,
    s.numest AS contributor_count,
    s.numup AS revisions_up,
    s.numdown AS revisions_down,
    s.medest AS consensus_median_unadjusted,
    s.meanest AS consensus_mean_unadjusted,
    s.stdev AS consensus_stdev_unadjusted,
    s.highest AS consensus_high_unadjusted,
    s.lowest AS consensus_low_unadjusted,
    link.permno AS permno,
    link.sdate AS link_start,
    link.edate AS link_end,
    link.score AS link_score,
    split_consensus.dlycaldt AS crsp_consensus_trading_date,
    split_consensus.dlycumfacshr AS cfacshr_consensus_date,
    split_report.dlycaldt AS crsp_report_trading_date,
    split_report.dlycumfacshr AS cfacshr_report_date
FROM candidate_actuals AS a
JOIN LATERAL (
    SELECT
        statpers, fiscalp, fpi, estflag, curcode, numest, numup, numdown,
        medest, meanest, stdev, highest, lowest
    FROM ibes.statsumu_epsus AS summary
    WHERE summary.ticker = a.ticker
      AND summary.usfirm = 1
      AND UPPER(summary.measure) = 'EPS'
      AND UPPER(summary.fiscalp) = 'QTR'
      AND UPPER(summary.curcode) = UPPER(a.curr_act)
      AND summary.fpedats = a.pends
      AND summary.statpers < a.anndats
      AND summary.meanest IS NOT NULL
    ORDER BY summary.statpers DESC,
             (summary.estflag = 'P') DESC,
             summary.numest DESC NULLS LAST,
             summary.fpi
    LIMIT 1
) AS s ON TRUE
JOIN LATERAL (
    SELECT permno, sdate, edate, score
    FROM wrdsapps_link_crsp_ibes.ibcrsphist AS candidate_link
    WHERE candidate_link.ticker = a.ticker
      AND candidate_link.sdate <= a.anndats
      AND (candidate_link.edate IS NULL OR candidate_link.edate >= a.anndats)
      AND candidate_link.score <= 1
    ORDER BY candidate_link.score ASC, candidate_link.sdate DESC, candidate_link.permno
    LIMIT 1
) AS link ON TRUE
JOIN LATERAL (
    SELECT dlycaldt, dlycumfacshr
    FROM crsp.dsf_v2 AS daily
    WHERE daily.permno = link.permno
      AND daily.dlycaldt <= s.statpers
      AND daily.dlycumfacshr IS NOT NULL
      AND daily.dlycumfacshr > 0
    ORDER BY daily.dlycaldt DESC
    LIMIT 1
) AS split_consensus ON TRUE
JOIN LATERAL (
    SELECT dlycaldt, dlycumfacshr
    FROM crsp.dsf_v2 AS daily
    WHERE daily.permno = link.permno
      AND daily.dlycaldt <= a.anndats
      AND daily.dlycumfacshr IS NOT NULL
      AND daily.dlycumfacshr > 0
    ORDER BY daily.dlycaldt DESC
    LIMIT 1
) AS split_report ON TRUE
ORDER BY a.anndats DESC, a.ticker, a.pends DESC
LIMIT {max_events}
""".strip()


def build_wrds_ibes_eps_overnight_sql(
    *,
    tickers: tuple[str, ...],
    start_date: date,
    end_date_exclusive: date,
    max_events: int,
) -> str:
    """Build the frozen-universe WRDS SQL for the overnight exploratory run."""

    _validate_overnight_request(
        tickers=tickers,
        start_date=start_date,
        end_date_exclusive=end_date_exclusive,
        max_events=max_events,
    )
    values = ",\n        ".join(
        f"({_sql_literal(ticker)}, {_sql_literal(_base_wrds_symbol(ticker))})" for ticker in tickers
    )
    start = start_date.isoformat()
    end = end_date_exclusive.isoformat()
    return f"""
WITH frozen_universe(source_ticker, wrds_symbol) AS (
    VALUES
        {values}
), ranked_actuals AS (
    SELECT
        u.source_ticker,
        a.*,
        ROW_NUMBER() OVER (
            PARTITION BY u.source_ticker, a.pends, a.measure, a.pdicity
            ORDER BY a.actdats ASC, a.acttims ASC NULLS LAST,
                     a.anndats ASC, a.anntims ASC NULLS LAST
        ) AS actual_version_rank
    FROM frozen_universe AS u
    JOIN ibes.actu_epsus AS a
      ON UPPER(a.oftic) = u.wrds_symbol
      OR UPPER(a.ticker) = u.wrds_symbol
    WHERE a.usfirm = 1
      AND UPPER(a.measure) = 'EPS'
      AND UPPER(a.pdicity) = 'QTR'
      AND UPPER(a.curr_act) = 'USD'
      AND a.anndats >= DATE '{start}'
      AND a.anndats < DATE '{end}'
      AND a.pends IS NOT NULL
      AND a.value IS NOT NULL
), candidate_actuals AS (
    SELECT *
    FROM ranked_actuals
    WHERE actual_version_rank = 1
)
SELECT
    a.source_ticker AS source_ticker,
    a.ticker AS ibes_ticker,
    a.cusip AS ibes_cusip,
    a.oftic AS official_ticker,
    a.pends AS fiscal_period_end,
    a.measure AS measure,
    a.pdicity AS periodicity,
    a.anndats AS actual_announce_date,
    a.anntims AS actual_announce_time,
    a.actdats AS actual_activation_date,
    a.acttims AS actual_activation_time,
    a.value AS actual_unadjusted,
    a.curr_act AS actual_currency,
    s.statpers AS consensus_statistical_period,
    s.fiscalp AS consensus_fiscal_period,
    s.fpi AS forecast_period_indicator,
    s.estflag AS estimate_flag,
    s.curcode AS consensus_currency,
    s.numest AS contributor_count,
    s.numup AS revisions_up,
    s.numdown AS revisions_down,
    s.meanest AS consensus_mean_unadjusted,
    s.stdev AS consensus_stdev_unadjusted,
    link.permno AS permno,
    link.sdate AS link_start,
    link.edate AS link_end,
    link.score AS link_score,
    split_consensus.dlycaldt AS crsp_consensus_trading_date,
    split_consensus.dlycumfacshr AS cfacshr_consensus_date,
    split_report.dlycaldt AS crsp_report_trading_date,
    split_report.dlycumfacshr AS cfacshr_report_date
FROM candidate_actuals AS a
JOIN LATERAL (
    SELECT
        statpers, fiscalp, fpi, estflag, curcode, numest, numup, numdown,
        meanest, stdev
    FROM ibes.statsumu_epsus AS summary
    WHERE summary.ticker = a.ticker
      AND summary.usfirm = 1
      AND UPPER(summary.measure) = 'EPS'
      AND UPPER(summary.fiscalp) = 'QTR'
      AND UPPER(summary.curcode) = UPPER(a.curr_act)
      AND summary.fpedats = a.pends
      AND summary.statpers < a.anndats
      AND summary.meanest IS NOT NULL
    ORDER BY summary.statpers DESC,
             (summary.estflag = 'P') DESC,
             summary.numest DESC NULLS LAST,
             summary.fpi
    LIMIT 1
) AS s ON TRUE
JOIN LATERAL (
    SELECT permno, sdate, edate, score
    FROM wrdsapps_link_crsp_ibes.ibcrsphist AS candidate_link
    WHERE candidate_link.ticker = a.ticker
      AND candidate_link.sdate <= a.anndats
      AND (candidate_link.edate IS NULL OR candidate_link.edate >= a.anndats)
      AND candidate_link.score <= 1
    ORDER BY candidate_link.score ASC, candidate_link.sdate DESC, candidate_link.permno
    LIMIT 1
) AS link ON TRUE
JOIN LATERAL (
    SELECT dlycaldt, dlycumfacshr
    FROM crsp.dsf_v2 AS daily
    WHERE daily.permno = link.permno
      AND daily.dlycaldt <= s.statpers
      AND daily.dlycumfacshr IS NOT NULL
      AND daily.dlycumfacshr > 0
    ORDER BY daily.dlycaldt DESC
    LIMIT 1
) AS split_consensus ON TRUE
JOIN LATERAL (
    SELECT dlycaldt, dlycumfacshr
    FROM crsp.dsf_v2 AS daily
    WHERE daily.permno = link.permno
      AND daily.dlycaldt <= a.anndats
      AND daily.dlycumfacshr IS NOT NULL
      AND daily.dlycumfacshr > 0
    ORDER BY daily.dlycaldt DESC
    LIMIT 1
) AS split_report ON TRUE
ORDER BY a.anndats ASC, a.source_ticker, a.pends ASC
LIMIT {max_events}
""".strip()


def query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _ensure_private_output(repository_root: Path, output_directory: Path) -> Path:
    private_root = (repository_root / "data" / "private").resolve()
    resolved_output = output_directory.resolve()
    if not resolved_output.is_relative_to(private_root):
        raise ValueError("WRDS pilot output must stay under data/private")
    return resolved_output


def _validate_overnight_frame(
    frame: pl.DataFrame, *, tickers: tuple[str, ...], max_events: int
) -> None:
    if frame.height > max_events or frame.height > MAX_OVERNIGHT_EVENTS:
        raise ValueError("WRDS overnight snapshot exceeded the hard observation cap")
    missing_columns = sorted(set(OVERNIGHT_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"WRDS overnight snapshot is missing required columns: {missing_columns}")
    if frame.height == 0:
        return
    source_tickers = set(frame["source_ticker"].cast(pl.Utf8).to_list())
    if not source_tickers.issubset(set(tickers)):
        raise ValueError("WRDS overnight snapshot contains a ticker outside the frozen universe")
    if frame.select(
        (pl.col("consensus_statistical_period") >= pl.col("actual_announce_date")).any()
    ).item():
        raise ValueError("consensus snapshot must strictly precede the announcement date")
    if frame.select((pl.col("link_score") > 1).any()).item():
        raise ValueError("I/B/E/S-CRSP link score exceeds the quality threshold")
    if frame.select(
        ((pl.col("cfacshr_consensus_date") <= 0) | (pl.col("cfacshr_report_date") <= 0)).any()
    ).item():
        raise ValueError("CRSP share-adjustment factors must be positive")
    duplicate_count = frame.select(
        pl.struct("source_ticker", "fiscal_period_end", "actual_announce_date")
        .is_duplicated()
        .sum()
    ).item()
    if duplicate_count:
        raise ValueError("WRDS overnight snapshot contains duplicate source ticker-period events")


def _validate_pilot_frame(frame: pl.DataFrame, *, max_events: int) -> None:
    if frame.height < 1:
        raise ValueError("WRDS pilot returned no eligible observations")
    if frame.height > max_events or frame.height > MAX_PILOT_EVENTS:
        raise ValueError("WRDS pilot exceeded the hard observation cap")
    missing_columns = sorted(set(PILOT_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"WRDS pilot is missing required columns: {missing_columns}")

    required_non_null = (
        "ibes_ticker",
        "fiscal_period_end",
        "actual_announce_date",
        "actual_unadjusted",
        "actual_currency",
        "consensus_statistical_period",
        "consensus_currency",
        "consensus_mean_unadjusted",
        "permno",
        "link_score",
        "cfacshr_consensus_date",
        "cfacshr_report_date",
    )
    for column in required_non_null:
        if frame[column].null_count() > 0:
            raise ValueError(f"WRDS pilot contains nulls in required column {column}")

    if frame.select(
        (pl.col("consensus_statistical_period") >= pl.col("actual_announce_date")).any()
    ).item():
        raise ValueError("consensus snapshot must strictly precede the announcement date")
    if frame.select(
        (
            pl.col("actual_currency").str.to_uppercase()
            != pl.col("consensus_currency").str.to_uppercase()
        ).any()
    ).item():
        raise ValueError("actual and consensus currencies must match")
    if frame.select((pl.col("link_score") > 1).any()).item():
        raise ValueError("I/B/E/S-CRSP link score exceeds the pilot quality threshold")
    if frame.select(
        ((pl.col("cfacshr_consensus_date") <= 0) | (pl.col("cfacshr_report_date") <= 0)).any()
    ).item():
        raise ValueError("CRSP share-adjustment factors must be positive")

    duplicate_count = frame.select(
        pl.struct("ibes_ticker", "fiscal_period_end", "actual_announce_date").is_duplicated().sum()
    ).item()
    if duplicate_count:
        raise ValueError("WRDS pilot contains duplicate ticker-period-announcement observations")


def _manual_validation_payload(frame: pl.DataFrame) -> dict[str, Any]:
    split_change = frame.filter(pl.col("cfacshr_consensus_date") != pl.col("cfacshr_report_date"))
    exercise_frame = split_change if split_change.height else frame
    first_observation = exercise_frame.row(0, named=True)
    return {
        "contains_restricted_wrds_data": True,
        "do_not_commit_or_share": True,
        "observation": first_observation,
        "exercise": [
            "Verify consensus_statistical_period is strictly before actual_announce_date.",
            "Compute actual_on_consensus_basis = actual_unadjusted * cfacshr_consensus_date / cfacshr_report_date.",
            "Compute raw_eps_surprise = actual_on_consensus_basis - consensus_mean_unadjusted.",
            "Explain why the two CRSP factors are required for unadjusted EPS.",
            "Identify which raw time fields still need a documented time-zone convention before UTC normalization.",
        ],
    }


def run_wrds_ibes_eps_pilot(
    runner: QueryRunner,
    *,
    repository_root: Path,
    output_directory: Path,
    start_date: date,
    end_date_exclusive: date,
    max_events: int,
) -> PilotArtifacts:
    """Fetch and privately persist at most 25 source-audit observations."""

    query = build_wrds_ibes_eps_pilot_sql(
        start_date=start_date,
        end_date_exclusive=end_date_exclusive,
        max_events=max_events,
    )
    resolved_output = _ensure_private_output(repository_root, output_directory)
    frame = runner.run(query)
    _validate_pilot_frame(frame, max_events=max_events)

    store = ArtifactStore(resolved_output, resolved_output / "pilot.duckdb")
    data_path = store.write_parquet(frame.select(PILOT_COLUMNS), resolved_output / "events.parquet")
    manual_path = store.write_json(
        _manual_validation_payload(frame), resolved_output / "manual_validation_observation.json"
    )

    schema_digest = hashlib.sha256(
        "\n".join(f"{name}:{frame.schema[name]}" for name in PILOT_COLUMNS).encode("utf-8")
    ).hexdigest()
    receipt: dict[str, Any] = {
        "status": "source_audit_only",
        "contains_restricted_rows": True,
        "git_commit_permitted": False,
        "performance_metrics_computed": False,
        "source_tables": list(SOURCE_TABLES),
        "announcement_window": {
            "start": start_date.isoformat(),
            "end_exclusive": end_date_exclusive.isoformat(),
        },
        "hard_event_cap": MAX_PILOT_EVENTS,
        "requested_event_cap": max_events,
        "rows_written": frame.height,
        "split_factor_change_rows": frame.filter(
            pl.col("cfacshr_consensus_date") != pl.col("cfacshr_report_date")
        ).height,
        "query_sha256": query_sha256(query),
        "schema_sha256": schema_digest,
        "events_parquet_sha256": file_sha256(data_path),
        "manual_validation_sha256": file_sha256(manual_path),
        "consensus_rule": "latest_statsumu_epsus_statpers_strictly_before_actual_anndats",
        "split_basis_rule": "wrds_method_3_crsp_cfacshr_ratio_consensus_date_to_report_date",
        "raw_vendor_time_zone_status": "unresolved_preserved_without_utc_conversion",
    }
    receipt_path = store.write_json(receipt, resolved_output / "receipt.json")
    return PilotArtifacts(
        data_path=data_path,
        receipt_path=receipt_path,
        manual_validation_path=manual_path,
        receipt=receipt,
    )


def run_wrds_ibes_eps_overnight_snapshot(
    runner: QueryRunner,
    *,
    repository_root: Path,
    output_directory: Path,
    tickers: tuple[str, ...],
    start_date: date,
    end_date_exclusive: date,
    max_events: int,
) -> PilotArtifacts:
    """Fetch and privately persist the frozen-universe WRDS EPS snapshot."""

    query = build_wrds_ibes_eps_overnight_sql(
        tickers=tickers,
        start_date=start_date,
        end_date_exclusive=end_date_exclusive,
        max_events=max_events,
    )
    resolved_output = _ensure_private_output(repository_root, output_directory)
    frame = runner.run(query)
    _validate_overnight_frame(frame, tickers=tickers, max_events=max_events)

    store = ArtifactStore(resolved_output, resolved_output / "overnight.duckdb")
    data_path = store.write_parquet(
        frame.select(OVERNIGHT_COLUMNS), resolved_output / "wrds_eps_snapshot.parquet"
    )
    schema_digest = hashlib.sha256(
        "\n".join(f"{name}:{frame.schema[name]}" for name in OVERNIGHT_COLUMNS).encode("utf-8")
    ).hexdigest()
    receipt: dict[str, Any] = {
        "status": "bounded_snapshot_only",
        "contains_restricted_rows": True,
        "git_commit_permitted": False,
        "performance_metrics_computed": False,
        "source_tables": list(SOURCE_TABLES),
        "announcement_window": {
            "start": start_date.isoformat(),
            "end_exclusive": end_date_exclusive.isoformat(),
        },
        "frozen_ticker_count": len(tickers),
        "frozen_ticker_list_sha256": hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest(),
        "hard_event_cap": MAX_OVERNIGHT_EVENTS,
        "requested_event_cap": max_events,
        "rows_written": frame.height,
        "query_sha256": query_sha256(query),
        "schema_sha256": schema_digest,
        "events_parquet_sha256": file_sha256(data_path),
        "consensus_rule": "latest_statsumu_epsus_statpers_strictly_before_actual_anndats",
        "split_basis_rule": "actual_unadjusted_times_cfacshr_consensus_date_divided_by_cfacshr_report_date",
        "raw_vendor_time_zone_status": "unresolved_preserved_without_utc_conversion",
    }
    receipt_path = store.write_json(receipt, resolved_output / "wrds_eps_snapshot_receipt.json")
    manual_path = store.write_json(
        {
            "contains_restricted_wrds_data": False,
            "message": "No row-level values are exposed in this overnight manifest.",
        },
        resolved_output / "wrds_eps_snapshot_manual_placeholder.json",
    )
    return PilotArtifacts(
        data_path=data_path,
        receipt_path=receipt_path,
        manual_validation_path=manual_path,
        receipt=receipt,
    )


class WRDSConnectionRunner:
    """Lazily import WRDS and suppress client chatter that could reveal context."""

    def __init__(self, *, username: str) -> None:
        if not username.strip():
            raise ValueError("WRDS username is required")
        try:
            wrds_module = importlib.import_module("wrds")
            connection_type = wrds_module.Connection
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self._connection: Any = connection_type(
                    wrds_username=username,
                    verbose=False,
                )
        except Exception:
            raise RuntimeError("WRDS connection failed; no source data were written") from None

    def run(self, query: str) -> pl.DataFrame:
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                pandas_frame = self._connection.raw_sql(query)
            return pl.from_pandas(pandas_frame)
        except Exception:
            raise RuntimeError("WRDS pilot query failed; no source data were written") from None

    def close(self) -> None:
        with (
            contextlib.suppress(Exception),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self._connection.close()


def live_wrds_is_enabled() -> bool:
    return os.getenv(LIVE_WRDS_ENVIRONMENT_FLAG) == "1"


def derive_wrds_username_from_pgpass(pgpass_path: Path | None = None) -> str:
    """Return only the WRDS username field from an existing PostgreSQL password file."""

    candidates: list[Path] = []
    if pgpass_path is not None:
        candidates.append(pgpass_path)
    if os.getenv("PGPASSFILE"):
        candidates.append(Path(os.environ["PGPASSFILE"]))
    if os.getenv("APPDATA"):
        candidates.append(Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf")
    if os.getenv("USERPROFILE"):
        candidates.append(Path(os.environ["USERPROFILE"]) / ".pgpass")

    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(":")
            if len(fields) < 5:
                continue
            host, _port, database, username = fields[:4]
            if username and ("wrds" in host.lower() or "wrds" in database.lower()):
                return username
    raise RuntimeError("WRDS username could not be derived from pgpass")
