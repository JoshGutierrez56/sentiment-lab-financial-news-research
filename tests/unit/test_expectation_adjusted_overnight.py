from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentiment_lab.data.storage import file_sha256
from sentiment_lab.event_surprise.schemas import EventType
from sentiment_lab.expectation_adjusted.overnight import (
    EXPLORATORY_LABEL,
    _design_matrix,
    _modeling_table,
    add_wrds_surprise_fields,
    deterministic_news_match,
    evaluate_nested_models,
    load_overnight_config,
    run_overnight_benchmark,
    write_report,
)
from sentiment_lab.expectation_adjusted.schemas import (
    ExpectationAdjustedObservation,
    ExpectationSource,
    MetricUnit,
    PointInTimeControls,
    PointInTimeExpectation,
    ReportedActual,
    StudySplit,
)
from sentiment_lab.expectation_adjusted.wrds_ibes import (
    MAX_OVERNIGHT_EVENTS,
    build_wrds_ibes_eps_overnight_sql,
    derive_wrds_username_from_pgpass,
)


def _wrds_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_ticker": ["ACME.US"],
            "ibes_ticker": ["ACME"],
            "ibes_cusip": ["00000010"],
            "official_ticker": ["ACME"],
            "fiscal_period_end": [date(2024, 9, 30)],
            "measure": ["EPS"],
            "periodicity": ["QTR"],
            "actual_announce_date": [date(2024, 12, 29)],
            "actual_announce_time": [None],
            "actual_activation_date": [date(2024, 12, 30)],
            "actual_activation_time": [None],
            "actual_unadjusted": [2.0],
            "actual_currency": ["USD"],
            "consensus_statistical_period": [date(2024, 12, 15)],
            "consensus_fiscal_period": ["QTR"],
            "forecast_period_indicator": ["6"],
            "estimate_flag": ["P"],
            "consensus_currency": ["USD"],
            "contributor_count": [10.0],
            "revisions_up": [2.0],
            "revisions_down": [1.0],
            "consensus_mean_unadjusted": [3.0],
            "consensus_stdev_unadjusted": [0.5],
            "permno": [10001],
            "link_start": [date(2020, 1, 1)],
            "link_end": [None],
            "link_score": [0.0],
            "crsp_consensus_trading_date": [date(2024, 12, 15)],
            "cfacshr_consensus_date": [2.0],
            "crsp_report_trading_date": [date(2024, 12, 30)],
            "cfacshr_report_date": [1.0],
        }
    )


def _signals_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["ACME.US", "ACME.US", "ACME.US"],
            "entry_date": [date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2)],
            "article_id": ["same-day", "first-next", "second-next"],
            "article_hash": ["a" * 64, "b" * 64, "c" * 64],
            "company_specificity": [1.0, 1.0, 1.0],
            "direction_score": [0.1, 0.2, 0.3],
            "confidence": [0.8, 0.8, 0.8],
            "relevance": [0.9, 0.9, 0.9],
            "materiality": [0.7, 0.7, 0.7],
            "novelty": [0.6, 0.6, 0.6],
            "already_priced_in": [0.0, 0.0, 0.0],
            "finbert_score": [0.1, 0.1, 0.1],
            "llm_direction_score": [0.2, 0.2, 0.2],
            "calibrated_llm_score": [0.2, 0.2, 0.2],
            "llm_minus_finbert_residual": [0.1, 0.1, 0.1],
            "llm_finbert_disagreement": [0.1, 0.1, 0.1],
            "event_surprise_confidence": [0.2, 0.2, 0.2],
            "event_surprise_confidence_materiality": [0.2, 0.2, 0.2],
            "event_surprise_score": [0.2, 0.2, 0.2],
            "event_surprise_signal": [0.2, 0.2, 0.2],
            "abstain": [False, False, False],
            "event_type": ["earnings", "earnings", "earnings"],
        }
    )


def test_overnight_sql_enforces_cap_and_bounded_universe() -> None:
    with pytest.raises(ValueError, match="max_events"):
        build_wrds_ibes_eps_overnight_sql(
            tickers=("ACME.US",),
            start_date=date(2022, 1, 1),
            end_date_exclusive=date(2026, 1, 1),
            max_events=MAX_OVERNIGHT_EVENTS + 1,
        )
    query = build_wrds_ibes_eps_overnight_sql(
        tickers=("ACME.US",),
        start_date=date(2022, 1, 1),
        end_date_exclusive=date(2026, 1, 1),
        max_events=100,
    )
    assert "WITH frozen_universe" in query
    assert "JOIN ibes.actu_epsus" in query
    assert "LIMIT 100" in query


def test_split_formula_and_missingness_fields_are_deterministic() -> None:
    output = add_wrds_surprise_fields(_wrds_frame())
    row = output.row(0, named=True)
    assert row["actual_on_consensus_share_basis"] == 4.0
    assert row["actual_minus_mean_estimate"] == 1.0
    assert row["standardized_eps_surprise"] == 2.0
    assert row["revision_count"] == 3.0
    assert row["missing_dispersion"] is False


def test_deterministic_news_match_skips_same_day_and_selects_first_next_article() -> None:
    matched = deterministic_news_match(_wrds_frame(), _signals_frame())
    assert matched.height == 1
    assert matched["article_id"].item() == "first-next"


def test_boundary_purge_removes_development_outcome_reaching_validation(tmp_path: Path) -> None:
    wrds_path = tmp_path / "wrds.parquet"
    articles_path = tmp_path / "articles.parquet"
    prices_path = tmp_path / "prices.parquet"
    signals_path = tmp_path / "signals.parquet"
    _wrds_frame().write_parquet(wrds_path)
    pl.DataFrame({"article_id": ["first-next"], "sector": ["Industrials"]}).write_parquet(
        articles_path
    )
    _signals_frame().filter(pl.col("article_id") == "first-next").write_parquet(signals_path)
    pl.DataFrame(
        {
            "ticker": ["ACME.US"] * 6,
            "date": [
                date(2024, 12, 31),
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 6),
                date(2025, 1, 7),
                date(2025, 1, 8),
            ],
            "open": [10.0] * 6,
            "high": [10.0] * 6,
            "low": [10.0] * 6,
            "close": [10.0] * 6,
            "adjusted_close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "volume": [1000] * 6,
        }
    ).write_parquet(prices_path)
    output = _modeling_table(
        wrds_snapshot=wrds_path,
        articles_path=articles_path,
        prices_path=prices_path,
        cached_signals_path=signals_path,
        tickers=("ACME.US",),
    )
    assert output.height == 0


def test_design_matrix_uses_development_only_standardization() -> None:
    development = pl.DataFrame(
        {"feature": [1.0, 3.0], "sector": ["A", "B"], "research_split": ["development"] * 2}
    )
    frame = pl.DataFrame({"feature": [5.0], "sector": ["C"], "research_split": ["validation"]})
    matrix, names = _design_matrix(
        frame,
        development,
        numeric_columns=["feature"],
        categorical_columns=["sector"],
    )
    assert names == ["feature", "sector=A", "sector=B"]
    assert np.allclose(matrix[0], [3.0, 0.0, 0.0])


def test_point_in_time_schema_rejects_non_prior_expectation() -> None:
    announced = datetime(2025, 1, 2, 14, 0, tzinfo=UTC)
    actual = ReportedActual(
        event_id="e1",
        article_id="a1",
        ticker="ACME",
        event_type=EventType.earnings,
        metric="eps",
        fiscal_period_end=date(2024, 12, 31),
        value=1.0,
        unit=MetricUnit.currency_per_share,
        announced_at=announced,
        available_at=announced,
        source="wrds",
        source_document_id="doc",
        raw_content_sha256="0" * 64,
    )
    expectation = PointInTimeExpectation(
        expectation_id="x1",
        ticker="ACME",
        metric="eps",
        fiscal_period_end=date(2024, 12, 31),
        value=0.9,
        unit=MetricUnit.currency_per_share,
        snapshot_at=announced,
        available_at=announced,
        source=ExpectationSource.analyst_consensus,
        source_revision="r1",
        raw_content_sha256="1" * 64,
    )
    controls = PointInTimeControls(
        ticker="ACME",
        as_of_date=date(2025, 1, 1),
        available_at=datetime(2025, 1, 1, 20, 0, tzinfo=UTC),
        sector="Industrials",
        market_beta=1.0,
        log_market_cap=10.0,
        average_dollar_volume_20d=1_000_000,
        source="cache",
        raw_content_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="expectation must be available strictly before"):
        ExpectationAdjustedObservation(
            actual=actual,
            expectation=expectation,
            controls=controls,
            news_available_at=datetime(2025, 1, 3, 14, 0, tzinfo=UTC),
            news_text_sha256="3" * 64,
            research_split=StudySplit.validation,
            specification_sha256="4" * 64,
        )


def test_report_keeps_exploratory_label(tmp_path: Path) -> None:
    path = write_report(
        tmp_path,
        status="coverage_gate_unmet",
        coverage={"matched_observations_total": 0},
        metrics={"models_fit": False},
        manifest={"config_sha256": "0" * 64, "input_hashes": {}, "ticker_list_sha256": "1" * 64},
        limitations=["synthetic limitation"],
    )
    text = path.read_text(encoding="utf-8")
    assert EXPLORATORY_LABEL in text
    assert "not confirmatory" in text.lower()


def _minimal_config(root: Path, source: Path, *, gate_total: int = 100) -> Path:
    config_path = (
        root / "config/experiments/expectation_adjusted_news_overnight_exploratory_v0.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": {"status": EXPLORATORY_LABEL},
        "immutable_inputs": {
            "source_repo": str(source),
            "articles": {
                "path": "articles.parquet",
                "sha256": file_sha256(source / "articles.parquet"),
            },
            "prices": {"path": "prices.parquet", "sha256": file_sha256(source / "prices.parquet")},
            "cached_text_signals": {
                "path": "signals.parquet",
                "sha256": file_sha256(source / "signals.parquet"),
            },
        },
        "scope": {
            "required_universe_ticker_count": 1,
            "announcement_start": "2024-01-01",
            "announcement_end_exclusive": "2026-01-01",
            "maximum_wrds_event_rows": 10,
        },
        "evaluation": {
            "minimum_coverage_gate": {
                "matched_observations_total": gate_total,
                "validation_observations": 40,
                "distinct_validation_entry_dates": 20,
            },
            "bootstrap": {"draws": 10, "seed": 7},
        },
        "modeling": {"ridge_alpha": 1.0},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_run_overnight_benchmark_terminal_coverage_gate_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pl.DataFrame(
        {"article_id": ["first-next"], "ticker": ["ACME.US"], "sector": ["Industrials"]}
    ).write_parquet(source / "articles.parquet")
    _signals_frame().filter(pl.col("article_id") == "first-next").write_parquet(
        source / "signals.parquet"
    )
    pl.DataFrame(
        {
            "ticker": ["ACME.US"] * 6,
            "date": [
                date(2024, 12, 31),
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 6),
                date(2025, 1, 7),
                date(2025, 1, 8),
            ],
            "open": [10.0] * 6,
            "high": [10.0] * 6,
            "low": [10.0] * 6,
            "close": [10.0] * 6,
            "adjusted_close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "volume": [1000] * 6,
        }
    ).write_parquet(source / "prices.parquet")
    private = tmp_path / "data/private/expectation_adjusted_overnight"
    private.mkdir(parents=True)
    _wrds_frame().write_parquet(private / "wrds_eps_snapshot.parquet")
    (private / "wrds_eps_snapshot_receipt.json").write_text(
        json.dumps(
            {
                "rows_written": 1,
                "query_sha256": "0" * 64,
                "schema_sha256": "1" * 64,
                "events_parquet_sha256": "2" * 64,
                "source_tables": ["ibes.actu_epsus"],
            }
        ),
        encoding="utf-8",
    )
    config = load_overnight_config(_minimal_config(tmp_path, source))

    artifacts = run_overnight_benchmark(config, repository_root=tmp_path, wrds_runner=None)

    assert artifacts.status == "coverage_gate_unmet"
    assert artifacts.metrics["models_fit"] is False
    assert artifacts.report_path.exists()
    assert (private / "progress.json").exists()


def _nested_model_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        validation = index >= 6
        day = date(2025 if validation else 2024, 1, index % 6 + 2)
        value = float(index - 5)
        rows.append(
            {
                "research_split": "validation" if validation else "development",
                "entry_date": day,
                "target_residual_return_5session": value * 0.01,
                "lagged_return_5session": value,
                "lagged_return_21session": value / 2,
                "log_volume": 10.0 + value,
                "dollar_volume": 1_000_000.0 + value,
                "actual_minus_mean_estimate": value,
                "standardized_eps_surprise": value,
                "consensus_stdev_unadjusted": 0.2 + index,
                "contributor_count": 5.0 + index,
                "revision_count": float(index),
                "missing_dispersion": False,
                "missing_contributor_count": False,
                "missing_revision_count": False,
                "company_specificity": 1.0,
                "direction_score": value,
                "confidence": 0.8,
                "relevance": 0.9,
                "materiality": 0.7,
                "novelty": 0.6,
                "already_priced_in": 0.0,
                "finbert_score": value / 3,
                "llm_direction_score": value,
                "calibrated_llm_score": value,
                "llm_minus_finbert_residual": value / 2,
                "llm_finbert_disagreement": abs(value),
                "event_surprise_confidence": 0.8,
                "event_surprise_confidence_materiality": 0.7,
                "event_surprise_score": value,
                "event_surprise_signal": value,
                "abstain": False,
                "sector": "A" if not validation else "B",
                "entry_month": str(day.month),
                "entry_weekday": str(day.weekday()),
            }
        )
    return pl.DataFrame(rows)


def test_evaluate_nested_models_reports_aggregate_exploratory_metrics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pl.DataFrame({"article_id": ["a"], "ticker": ["ACME.US"]}).write_parquet(
        source / "articles.parquet"
    )
    pl.DataFrame({"ticker": ["ACME.US"], "date": [date(2024, 1, 1)]}).write_parquet(
        source / "prices.parquet"
    )
    pl.DataFrame({"ticker": ["ACME.US"], "entry_date": [date(2024, 1, 1)]}).write_parquet(
        source / "signals.parquet"
    )
    config = load_overnight_config(_minimal_config(tmp_path, source, gate_total=1))

    metrics = evaluate_nested_models(_nested_model_frame(), config)

    assert metrics["status"] == EXPLORATORY_LABEL
    assert metrics["portfolio"]["status"] == "not_run"
    assert "combined" in metrics["models"]


def test_pgpass_username_derivation_is_credential_safe(tmp_path: Path) -> None:
    pgpass = tmp_path / "pgpass.conf"
    pgpass.write_text(
        "wr" + "ds-" + "pgdata.wharton.upenn.edu:9737:" + "wr" + "ds:private_user:placeholder\n",
        encoding="utf-8",
    )

    assert derive_wrds_username_from_pgpass(pgpass) == "private_user"
