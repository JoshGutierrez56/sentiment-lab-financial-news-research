"""Command line entry point for the mandatory milestone."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from sentiment_lab.config.loader import (
    load_app_config,
    load_experiment_config,
    load_sentiment_config,
)
from sentiment_lab.config.models import (
    AppConfig,
    ExperimentConfig,
    RuntimeSecrets,
    SentimentConfig,
)
from sentiment_lab.data.cache import RawResponseCache
from sentiment_lab.data.eodhd_client import EODHDClient, EODHDError
from sentiment_lab.experiments.runner import MilestoneRunner, sync_milestone_data
from sentiment_lab.nlp.cache import ClassificationCache
from sentiment_lab.nlp.classifier import ArticleClassifier
from sentiment_lab.nlp.openai_client import OpenAIArticleClient, OpenAIClassificationError

app = typer.Typer(no_args_is_help=True, help="EODHD → OpenAI sentiment research")
data_app = typer.Typer(no_args_is_help=True, help="Download and cache milestone data")
milestone_app = typer.Typer(no_args_is_help=True, help="Run the article-to-return milestone")
app.add_typer(data_app, name="data")
app.add_typer(milestone_app, name="milestone")


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
    # EODHD authenticates through a query parameter, which HTTPX includes in
    # its INFO request line. Keep third-party transport logging from exposing it.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return (
        runtime,
        load_app_config(settings_path, secrets=runtime),
        load_sentiment_config(sentiment_path),
        load_experiment_config(experiment_path),
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

    runtime, app_config, _, experiment = _load(config, settings, sentiment)
    try:
        token = runtime.require_eodhd_token()
    except RuntimeError as exc:
        _fail(str(exc))
    cache = RawResponseCache(app_config.storage.data_root)
    try:
        with EODHDClient(token, app_config.eodhd, cache) as client:
            snapshot = sync_milestone_data(experiment, app_config, client, refresh=refresh)
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
    force_classify: Annotated[bool, typer.Option(help="Bypass the OpenAI response cache")] = False,
) -> None:
    """Run real articles through ChatGPT and align conservative future returns."""

    runtime, app_config, sentiment_config, experiment = _load(config, settings, sentiment)
    try:
        token = runtime.require_eodhd_token()
        api_key, model = runtime.require_openai()
    except RuntimeError as exc:
        _fail(str(exc))
    raw_cache = RawResponseCache(app_config.storage.data_root)
    model_client = OpenAIArticleClient(api_key, model, app_config.openai)
    classifier = ArticleClassifier(
        model_client,
        ClassificationCache(app_config.storage.data_root),
        schema_version=sentiment_config.schema_version,
        max_article_characters=sentiment_config.max_article_characters,
        max_concurrency=app_config.openai.max_concurrency,
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
            output = runner.run(refresh=refresh, force_classify=force_classify)
    except (EODHDError, OpenAIClassificationError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


if __name__ == "__main__":  # pragma: no cover
    app()
