.PHONY: install format lint type test check milestone-sync milestone-run

install:
	uv sync --extra dev

format:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run mypy src/sentiment_lab

test:
	uv run pytest

check: lint type test

milestone-sync:
	uv run sentiment-lab data sync --config config/experiments/milestone.yaml

milestone-run:
	uv run sentiment-lab milestone run --config config/experiments/milestone.yaml
