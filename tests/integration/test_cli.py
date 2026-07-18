"""User-facing CLI failure behavior."""

from __future__ import annotations

from typer.testing import CliRunner

from sentiment_lab.cli import app


def test_missing_openai_key_is_actionable_without_traceback() -> None:
    result = CliRunner().invoke(
        app,
        ["milestone", "run", "--config", "config/experiments/milestone.yaml"],
        env={
            "EODHD_API_TOKEN": "test-token",
            "OPENAI_API_KEY": "",
            "OPENAI_MODEL": "test-model",
        },
    )
    assert result.exit_code == 1
    assert "OPENAI_API_KEY is required" in result.output
    assert "Traceback" not in result.output
