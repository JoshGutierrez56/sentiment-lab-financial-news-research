"""Command line entry point for the mandatory milestone."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, NoReturn, TypeVar

import typer
import yaml
from pydantic import BaseModel

from sentiment_lab.config.loader import (
    load_app_config,
    load_experiment_config,
    load_sentiment_config,
    load_validation_experiment_config,
)
from sentiment_lab.config.models import (
    AppConfig,
    ExperimentConfig,
    RuntimeSecrets,
    SentimentConfig,
    ValidationExperimentConfig,
)
from sentiment_lab.data.cache import RawResponseCache
from sentiment_lab.data.eodhd_client import EODHDClient, EODHDError
from sentiment_lab.experiments.runner import MilestoneRunner, sync_milestone_data
from sentiment_lab.experiments.validation import ValidationRunner, sync_validation_data
from sentiment_lab.hybrid.analysis import PredictionAnalysisConfig, run_prediction_analysis
from sentiment_lab.hybrid.baselines import BaselineConfig, run_baselines
from sentiment_lab.hybrid.calibration_analysis import (
    CalibrationAnalysisConfig,
    run_calibration_analysis,
)
from sentiment_lab.hybrid.classification import HybridLocalRunConfig, run_local_classification
from sentiment_lab.hybrid.final_report import FinalReportConfig, build_final_report
from sentiment_lab.hybrid.openai_calibration import (
    AdditionalCalibrationConfig,
    AdditionalOpenAIRunConfig,
    freeze_additional_openai_sample,
    run_additional_openai_calibration,
)
from sentiment_lab.hybrid.portfolio import PortfolioRunConfig, run_portfolio_backtests
from sentiment_lab.hybrid.sample import HybridSampleConfig, sync_hybrid_sample
from sentiment_lab.hybrid.specification import (
    SpecificationSearchConfig,
    freeze_primary_specification,
)
from sentiment_lab.hybrid.splits import freeze_chronological_splits
from sentiment_lab.nlp.cache import ClassificationCache
from sentiment_lab.nlp.classifier import ArticleClassifier
from sentiment_lab.nlp.openai_client import OpenAIBatchClient, OpenAIClassificationError

app = typer.Typer(no_args_is_help=True, help="EODHD → OpenAI sentiment research")
data_app = typer.Typer(no_args_is_help=True, help="Download and cache milestone data")
milestone_app = typer.Typer(no_args_is_help=True, help="Run the article-to-return milestone")
validation_app = typer.Typer(no_args_is_help=True, help="Run the bounded 250-article validation")
hybrid_app = typer.Typer(no_args_is_help=True, help="Run the locked 5,000-article hybrid study")
app.add_typer(data_app, name="data")
app.add_typer(milestone_app, name="milestone")
app.add_typer(validation_app, name="validation")
app.add_typer(hybrid_app, name="hybrid")

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


def _yaml_config(path: Path, model: type[ConfigModel]) -> ConfigModel:
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return model.model_validate(value)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _load(
    experiment_path: Path,
    settings_path: Path,
    sentiment_path: Path,
) -> tuple[RuntimeSecrets, AppConfig, SentimentConfig, ExperimentConfig]:
    runtime = RuntimeSecrets()
    logging.basicConfig(
        level=getattr(logging, runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return (
        runtime,
        load_app_config(settings_path, secrets=runtime),
        load_sentiment_config(sentiment_path),
        load_experiment_config(experiment_path),
    )


def _load_validation(
    experiment_path: Path,
    settings_path: Path,
    sentiment_path: Path,
) -> tuple[RuntimeSecrets, AppConfig, SentimentConfig, ValidationExperimentConfig]:
    runtime = RuntimeSecrets()
    logging.basicConfig(
        level=getattr(logging, runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return (
        runtime,
        load_app_config(settings_path, secrets=runtime),
        load_sentiment_config(sentiment_path),
        load_validation_experiment_config(experiment_path),
    )


def _classifier(
    runtime: RuntimeSecrets,
    app_config: AppConfig,
    sentiment_config: SentimentConfig,
) -> ArticleClassifier:
    model_client = OpenAIBatchClient(
        runtime.require_openai_key(),
        app_config.openai,
        app_config.storage.data_root,
    )
    return ArticleClassifier(
        model_client,
        ClassificationCache(app_config.storage.data_root),
        app_config.openai,
        schema_version=sentiment_config.schema_version,
        max_article_characters=sentiment_config.max_article_characters,
        escalation_confidence_threshold=sentiment_config.escalation_confidence_threshold,
        escalation_materiality_threshold=sentiment_config.escalation_materiality_threshold,
        escalation_ambiguity_relevance_threshold=(
            sentiment_config.escalation_ambiguity_relevance_threshold
        ),
        escalation_ambiguity_materiality_threshold=(
            sentiment_config.escalation_ambiguity_materiality_threshold
        ),
    )


@data_app.command("sync")
def data_sync(
    config: Annotated[Path, typer.Option(exists=True, readable=True, help="Experiment YAML")],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
    sentiment: Annotated[Path, typer.Option(exists=True)] = Path("config/sentiment.yaml"),
    refresh: Annotated[
        bool, typer.Option(help="Bypass request cache and append a raw snapshot")
    ] = False,
) -> None:
    """Download real EODHD articles and EOD prices without requiring OpenAI."""

    runtime, app_config, sentiment_config, experiment = _load(config, settings, sentiment)
    try:
        token = runtime.require_eodhd_token()
    except RuntimeError as exc:
        _fail(str(exc))
    cache = RawResponseCache(app_config.storage.data_root)
    try:
        with EODHDClient(token, app_config.eodhd, cache) as client:
            snapshot = sync_milestone_data(
                experiment,
                app_config,
                sentiment_config,
                client,
                refresh=refresh,
            )
    except (EODHDError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "articles": len(snapshot.articles),
                "prices": len(snapshot.prices),
                "articles_path": str(snapshot.articles_path),
                "prices_path": str(snapshot.prices_path),
                "filtering": snapshot.filter_report,
            },
            indent=2,
        )
    )


@milestone_app.command("run")
def milestone_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True, help="Experiment YAML")],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
    sentiment: Annotated[Path, typer.Option(exists=True)] = Path("config/sentiment.yaml"),
    refresh: Annotated[bool, typer.Option(help="Append fresh EODHD raw responses")] = False,
) -> None:
    """Run cached/real articles through cost-bounded OpenAI batches and align returns."""

    runtime, app_config, sentiment_config, experiment = _load(config, settings, sentiment)
    try:
        token = runtime.require_eodhd_token()
        api_key = runtime.require_openai_key()
    except RuntimeError as exc:
        _fail(str(exc))
    raw_cache = RawResponseCache(app_config.storage.data_root)
    model_client = OpenAIBatchClient(
        api_key,
        app_config.openai,
        app_config.storage.data_root,
    )
    classifier = ArticleClassifier(
        model_client,
        ClassificationCache(app_config.storage.data_root),
        app_config.openai,
        schema_version=sentiment_config.schema_version,
        max_article_characters=sentiment_config.max_article_characters,
        escalation_confidence_threshold=(sentiment_config.escalation_confidence_threshold),
        escalation_materiality_threshold=(sentiment_config.escalation_materiality_threshold),
        escalation_ambiguity_relevance_threshold=(
            sentiment_config.escalation_ambiguity_relevance_threshold
        ),
        escalation_ambiguity_materiality_threshold=(
            sentiment_config.escalation_ambiguity_materiality_threshold
        ),
    )
    try:
        with EODHDClient(token, app_config.eodhd, raw_cache) as eodhd:
            runner = MilestoneRunner(
                experiment,
                app_config,
                sentiment_config,
                eodhd,
                classifier,
            )
            output = runner.run(refresh=refresh)
    except (EODHDError, OpenAIClassificationError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@validation_app.command("sync")
def validation_sync(
    config: Annotated[Path, typer.Option(exists=True, readable=True, help="Validation YAML")],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
    sentiment: Annotated[Path, typer.Option(exists=True)] = Path("config/sentiment.yaml"),
    refresh: Annotated[
        bool, typer.Option(help="Bypass EODHD request cache and append raw responses")
    ] = False,
) -> None:
    """Freeze and validate the 250-article sample without calling OpenAI."""

    runtime, app_config, sentiment_config, experiment = _load_validation(
        config, settings, sentiment
    )
    try:
        token = runtime.require_eodhd_token()
        with EODHDClient(
            token,
            app_config.eodhd,
            RawResponseCache(app_config.storage.data_root),
        ) as client:
            snapshot = sync_validation_data(
                experiment,
                app_config,
                sentiment_config,
                client,
                refresh=refresh,
            )
    except (EODHDError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "articles": len(snapshot.sampled),
                "articles_path": str(snapshot.articles_path),
                "prices_path": str(snapshot.prices_path),
                "filtering": snapshot.filter_report,
            },
            indent=2,
        )
    )


@validation_app.command("run")
def validation_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True, help="Validation YAML")],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
    sentiment: Annotated[Path, typer.Option(exists=True)] = Path("config/sentiment.yaml"),
) -> None:
    """Run exactly one frozen, cost-bounded 250-article Batch validation."""

    runtime, app_config, sentiment_config, experiment = _load_validation(
        config, settings, sentiment
    )
    if experiment.frozen_snapshot_id is None:
        _fail("frozen_snapshot_id is required; run `sentiment-lab validation sync` first")
    try:
        token = runtime.require_eodhd_token()
        classifier = _classifier(runtime, app_config, sentiment_config)
        with EODHDClient(
            token,
            app_config.eodhd,
            RawResponseCache(app_config.storage.data_root),
        ) as client:
            output = ValidationRunner(
                experiment,
                app_config,
                sentiment_config,
                client,
                classifier,
            ).run()
    except (EODHDError, OpenAIClassificationError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("sample-sync")
def hybrid_sample_sync(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
    refresh: Annotated[bool, typer.Option(help="Bypass EODHD raw cache")] = False,
) -> None:
    """Build or verify the return-blind, hash-locked 5,000-article sample."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        with EODHDClient(
            runtime.require_eodhd_token(),
            app_config.eodhd,
            RawResponseCache(app_config.storage.data_root),
        ) as client:
            output = sync_hybrid_sample(
                _yaml_config(config, HybridSampleConfig),
                client,
                data_root=app_config.storage.data_root,
                duckdb_path=app_config.storage.duckdb_path,
                refresh=refresh,
            )
    except (EODHDError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("local-run")
def hybrid_local_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Run/resume locked local inference with permanent cache and QA gates."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = run_local_classification(
            _yaml_config(config, HybridLocalRunConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("splits-freeze")
def hybrid_splits_freeze(
    articles: Annotated[Path, typer.Option(exists=True, readable=True)],
    sample_hash: Annotated[str, typer.Option()],
    output_root: Annotated[Path, typer.Option()],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Freeze exact chronological 60/20/20 assignments before analysis."""

    invalid_digest = len(sample_hash) != 64 or any(
        character not in "0123456789abcdef" for character in sample_hash
    )
    if invalid_digest:
        _fail("sample-hash must be a lowercase SHA-256 digest")
    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = freeze_chronological_splits(
            articles,
            sample_hash=sample_hash,
            output_root=output_root,
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("prediction-run")
def hybrid_prediction_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Run dependence-aware predictive tests on permitted chronological splits."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = run_prediction_analysis(
            _yaml_config(config, PredictionAnalysisConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("spec-freeze")
def hybrid_spec_freeze(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Search development/validation only and freeze the primary specification."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = freeze_primary_specification(
            _yaml_config(config, SpecificationSearchConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("calibration-select")
def hybrid_calibration_select(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Freeze at most 250 bias-revealing development/validation calibration cases."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = freeze_additional_openai_sample(
            _yaml_config(config, AdditionalCalibrationConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("calibration-run")
def hybrid_calibration_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
    sentiment: Annotated[Path, typer.Option(exists=True)] = Path("config/sentiment.yaml"),
) -> None:
    """Run the optional additional Batch calibration under the hard $1 guard."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = run_additional_openai_calibration(
            _yaml_config(config, AdditionalOpenAIRunConfig),
            api_key=runtime.require_openai_key(),
            app_config=app_config,
            sentiment_config=load_sentiment_config(sentiment),
        )
    except (OpenAIClassificationError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("baselines-run")
def hybrid_baselines_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Run point-in-time baselines and matched placebos."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = run_baselines(
            _yaml_config(config, BaselineConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("calibration-analyze")
def hybrid_calibration_analyze(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Compare local and additional OpenAI outputs without holdout access."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = run_calibration_analysis(
            _yaml_config(config, CalibrationAnalysisConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("portfolio-run")
def hybrid_portfolio_run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Build explicit daily long-only and market-neutral portfolios."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = run_portfolio_backtests(
            _yaml_config(config, PortfolioRunConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


@hybrid_app.command("report-build")
def hybrid_report_build(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    settings: Annotated[Path, typer.Option(exists=True)] = Path("config/settings.yaml"),
) -> None:
    """Verify evidence hashes and build the final HTML/results decision artifacts."""

    runtime = RuntimeSecrets()
    app_config = load_app_config(settings, secrets=runtime)
    try:
        output = build_final_report(
            _yaml_config(config, FinalReportConfig),
            data_root=app_config.storage.data_root,
            duckdb_path=app_config.storage.duckdb_path,
        )
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


if __name__ == "__main__":  # pragma: no cover
    app()
