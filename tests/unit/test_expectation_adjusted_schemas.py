from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sentiment_lab.event_surprise.schemas import EventType
from sentiment_lab.expectation_adjusted.schemas import (
    ExpectationAdjustedObservation,
    ExpectationSource,
    MetricUnit,
    PointInTimeControls,
    PointInTimeExpectation,
    ReportedActual,
    StudySplit,
)

DIGEST = "a" * 64
PROTOCOL_CONFIG = (
    Path(__file__).parents[2] / "config" / "experiments" / "expectation_adjusted_news_v0.yaml"
)


def _actual() -> ReportedActual:
    announced = datetime(2026, 7, 21, 12, tzinfo=UTC)
    return ReportedActual(
        event_id="event-1",
        article_id="article-1",
        ticker="ACME",
        event_type=EventType.earnings,
        metric="eps_diluted",
        fiscal_period_end=date(2026, 6, 30),
        value=1.20,
        unit=MetricUnit.currency_per_share,
        announced_at=announced,
        available_at=announced + timedelta(seconds=5),
        source="company-filing",
        source_document_id="filing-1",
        raw_content_sha256=DIGEST,
    )


def _expectation(*, available_at: datetime | None = None) -> PointInTimeExpectation:
    available = available_at or datetime(2026, 7, 20, 20, tzinfo=UTC)
    return PointInTimeExpectation(
        expectation_id="expectation-1",
        ticker="ACME",
        metric="eps_diluted",
        fiscal_period_end=date(2026, 6, 30),
        value=1.00,
        unit=MetricUnit.currency_per_share,
        snapshot_at=available - timedelta(minutes=1),
        available_at=available,
        source=ExpectationSource.analyst_consensus,
        source_revision="vendor-revision-7",
        dispersion=0.10,
        contributor_count=12,
        raw_content_sha256=DIGEST,
    )


def _controls(*, available_at: datetime | None = None) -> PointInTimeControls:
    return PointInTimeControls(
        ticker="ACME",
        as_of_date=date(2026, 7, 20),
        available_at=available_at or datetime(2026, 7, 21, 0, tzinfo=UTC),
        sector="Industrials",
        industry="Machinery",
        market_beta=1.1,
        log_market_cap=22.0,
        average_dollar_volume_20d=25_000_000,
        book_to_market=0.4,
        momentum_12_1=0.08,
        return_on_equity=0.15,
        asset_growth=0.05,
        idiosyncratic_volatility=0.22,
        source="point-in-time-factor-store",
        raw_content_sha256=DIGEST,
    )


def _observation(**overrides: object) -> ExpectationAdjustedObservation:
    values: dict[str, object] = {
        "actual": _actual(),
        "expectation": _expectation(),
        "controls": _controls(),
        "news_available_at": datetime(2026, 7, 21, 12, 0, 8, tzinfo=UTC),
        "news_text_sha256": DIGEST,
        "research_split": StudySplit.development,
        "specification_sha256": DIGEST,
    }
    values.update(overrides)
    return ExpectationAdjustedObservation.model_validate(values)


def test_valid_observation_exposes_auditable_surprise_and_decision_time() -> None:
    observation = _observation()
    assert observation.raw_surprise == pytest.approx(0.20)
    assert observation.dispersion_scaled_surprise == pytest.approx(2.0)
    assert observation.decision_at == datetime(2026, 7, 21, 12, 0, 8, tzinfo=UTC)


def test_expectation_must_predate_announcement() -> None:
    announcement = _actual().announced_at
    with pytest.raises(ValidationError, match="strictly before"):
        _observation(expectation=_expectation(available_at=announcement))


def test_controls_must_predate_announcement() -> None:
    announcement = _actual().announced_at
    with pytest.raises(ValidationError, match="controls must be available"):
        _observation(controls=_controls(available_at=announcement + timedelta(seconds=1)))


def test_join_keys_and_units_must_match() -> None:
    mismatched = _expectation().model_copy(update={"ticker": "OTHER"})
    with pytest.raises(ValidationError, match="one ticker"):
        _observation(expectation=mismatched)


def test_naive_timestamps_and_non_hashes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _observation(news_available_at=datetime(2026, 7, 21, 12))
    with pytest.raises(ValidationError):
        _observation(news_text_sha256="not-a-hash")


def test_design_config_cannot_masquerade_as_confirmatory_evidence() -> None:
    config = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    assert config["study"]["status"] == "design_only_not_frozen"
    assert config["study"]["results_available"] is False
    assert config["study"]["prior_holdout_policy"] == "prohibited_for_confirmatory_claims"
    assert (
        config["evaluation"]["splits"]["prospective_holdout"]
        == "observations_available_only_after_specification_freeze"
    )
    assert "feature_selection_using_prospective_holdout" in config["modeling"]["prohibited"]
    source = config["expectations_source_contract"]
    assert source["status"] == "selected_for_bounded_pilot_not_frozen"
    assert source["tables"] == {
        "unadjusted_actuals": "ibes.actu_epsus",
        "unadjusted_summary": "ibes.statsumu_epsus",
        "historical_ibes_crsp_link": "wrdsapps_link_crsp_ibes.ibcrsphist",
        "crsp_daily_split_factors": "crsp.dsf_v2",
    }
    assert source["consensus_rule"] == "latest_statpers_strictly_before_actual_anndats"
    assert config["source_audit_pilot"]["maximum_events"] == 25
    assert config["source_audit_pilot"]["performance_metrics_permitted"] is False
