# Sentiment Lab

Sentiment Lab is an evidence-first research pipeline for one deliberately narrow
question: does ChatGPT's interpretation of an EODHD news article predict the
specified company's subsequent stock return?

The current milestone does exactly this:

1. Download and preserve EODHD news and adjusted EOD prices.
2. Select a deterministic, date-diverse sample of full-text articles directly
   mapped by EODHD to one ticker.
3. Ask OpenAI for a strict bullish, bearish, or neutral assessment with score,
   confidence, relevance, event type, horizon, concise reasoning, and an
   explicit abstention decision.
4. Cache the validated structured response by the complete classification
   input, model, prompt version, and schema version.
5. Enter at the first market open strictly after the article's New York
   publication date and measure adjusted 1/3/5-trading-day returns.
6. Write article-level evidence, directional accuracy, and information
   coefficients to Parquet, JSON, DuckDB views, and a self-contained HTML
   report.

Advanced portfolio construction, factors, dashboards, broad universe tooling,
and parameter sweeps are intentionally deferred until this vertical slice has
run with real ChatGPT classifications.

## Verified status

As of 2026-07-18:

- A live EODHD sync succeeded for `AAPL.US`: 500 candidate records were cached,
  then 12 unique full-text articles were selected across 11 publication dates.
- Article timestamps are stored as timezone-aware UTC Parquet values; price
  dates are native Parquet dates.
- The real sample has seven distinct conservative entry dates and complete
  adjusted 1/3/5-day returns. Every entry is later than publication.
- Raw and normalized artifacts were scanned for the EODHD credential: zero
  persisted matches.
- Ruff and strict MyPy pass. The test suite passes with more than 90% package
  coverage on both Python 3.11 and 3.12, including mocked HTTP/OpenAI
  integration and cached-rerun tests.
- A live OpenAI run has not run because `OPENAI_API_KEY` is absent from the
  runtime environment. Therefore no trading conclusion is claimed yet.

## Install

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```powershell
git clone https://github.com/JoshGutierrez56/DeepSeek-Generative-AI-Sentiment-Analysis-Algorithm.git
cd DeepSeek-Generative-AI-Sentiment-Analysis-Algorithm
git switch agent/openai-eodhd-rebuild
uv sync --extra dev
Copy-Item .env.example .env
```

Populate the untracked `.env` file:

```dotenv
EODHD_API_TOKEN=...
OPENAI_API_KEY=...
OPENAI_MODEL=...
DATA_ROOT=./data
DUCKDB_PATH=./data/research.duckdb
LOG_LEVEL=INFO
```

`OPENAI_MODEL` is intentionally not hardcoded. Select a currently supported
model that supports Structured Outputs in the Responses API. Credentials are
loaded only at runtime. Do not commit `.env`.

## Run the core milestone

Download fresh EODHD data and preserve each raw response before normalization:

```powershell
uv run sentiment-lab data sync --config config/experiments/milestone.yaml --refresh
```

Run the complete real article-to-return pipeline:

```powershell
uv run sentiment-lab milestone run --config config/experiments/milestone.yaml
```

Do not add `--refresh` or `--force-classify` on a reproducibility rerun. The
command will then use the raw EODHD cache and content-addressed OpenAI cache.
The scientific artifacts (`articles.parquet`, `assessments.parquet`,
`events.parquet`, and `metrics.json`) are byte-stable. The report and manifest
receive a new run identity and timestamp; the manifest records cache hits and
zero per-run OpenAI tokens/cost while retaining original ledger usage.

Each completed run creates `data/results/<experiment-id>/` containing:

- `articles.parquet` — provider text, original publication time, retrieval time,
  symbols, provider sentiment, and raw-response hash.
- `assessments.parquet` — ChatGPT label, score, confidence, relevance, concise
  reasoning, abstention, model/prompt/schema versions, token usage, and cost.
- `events.parquet` — the joined article/assessment plus entry timestamp and
  future adjusted returns.
- `metrics.json` — coverage, directional accuracy, Spearman IC, Pearson
  correlation, confidence-weighted IC, and per-label returns.
- `manifest.json` — git/config/data hashes, exact versions and parameters,
  provider endpoints, Python/SDK/library versions, artifact hashes, cache
  hits/misses, per-run token/cost totals, classification-ledger usage, and
  summary metrics.
- `report.html` — a compact human-readable evidence table and metrics report.

The latest normalized tables are also exposed in `data/research.duckdb` as
`milestone_articles_latest`, `milestone_prices_latest`,
`milestone_assessments_latest`, and `milestone_events_latest`.

## Validate the implementation

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src/sentiment_lab
uv run pytest
```

Or run all checks with:

```powershell
make check
```

Tests never require paid credentials. They cover strict config rejection,
EODHD pagination/retry/rate-limit/schema behavior, immutable raw caching, token
redaction, OpenAI structured parsing and semantic repair, concurrent cache
safety, after-hours/weekend alignment, adjusted-open math, accuracy/IC metrics,
DuckDB views, reports, and deterministic event artifacts.

## Timing and return definition

The active policy is `conservative_next_day_open`:

- Convert the provider timestamp from UTC to `America/New_York`.
- Ignore every price on that local publication date, even for pre-market news.
- Enter at 09:30 New York time on the first later EODHD trading date.
- Estimate adjusted open as `raw_open × adjusted_close / raw_close` for that
  session so entry and exit values use a consistent split-adjusted scale.
- A 1-day horizon exits at that entry session's adjusted close; 3-day and 5-day
  horizons exit at the third and fifth trading-session adjusted closes.

The engine rejects article/classification identity or timestamp mismatches and
tests explicitly prove that no entry precedes its article.

## Interpretation of the milestone metrics

Directional accuracy maps realized returns within ±10 bps to neutral and uses
their sign outside that band. IC is the Spearman correlation between ChatGPT's
continuous sentiment score and the future return. Non-tradable/abstained rows
remain in the evidence artifact but are excluded from accuracy and IC.

These are descriptive smoke-test metrics. A 12-article sample is not evidence of
statistical significance; overlapping returns violate IID assumptions, and the
ordinary correlation p-values in the compact report do not correct dependence.
HAC inference, block bootstrap, controls, placebos, costs, and untouched
out-of-sample validation remain gated behind successful completion of this core
pipeline.

## Current limitations

- The live sample covers one ticker and is selected from EODHD's most recent 500
  candidate rows in the configured interval, with at most two articles per New
  York publication date.
- Mapping currently requires the exact requested ticker in EODHD's article
  symbols. Ambiguous entity resolution abstention is a later milestone.
- Provider timestamps are treated as first availability; article revision
  history is unavailable in this endpoint and cannot yet be controlled.
- EOD data cannot represent intraday reaction. The deliberately conservative
  next-day-open rule avoids pretending otherwise.
- Returns are raw company returns, not market- or factor-adjusted abnormal
  returns.
- No human-labeled sentiment evaluation has been performed.
- Most importantly, the real ChatGPT classifications and resulting accuracy/IC
  report remain blocked until an OpenAI key and model are supplied.

See [repository audit](docs/REPOSITORY_AUDIT.md) and
[implementation plan](docs/IMPLEMENTATION_PLAN.md) for provenance, reuse, and
the gated roadmap.

## License

MIT. See [LICENSE](LICENSE).
