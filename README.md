# Sentiment Lab

Sentiment Lab currently answers one deliberately narrow question: does a
structured OpenAI assessment of an EODHD article predict the specified
company's subsequent stock return?

Advanced portfolios, factors, dashboards, and large experiment frameworks are
deferred until this article-to-return milestone has run successfully with real
OpenAI classifications.

## Core execution path

1. Download and immutably cache EODHD news and adjusted EOD prices.
2. Before OpenAI, reject out-of-window articles, uncertain ticker mappings,
   inadequate text, broad market summaries, and duplicate stories.
3. Keep full text and headline-only records separate; the active configuration
   excludes headline-only records and always prioritizes full text.
4. Submit all uncached first-pass requests through the OpenAI Batch API to
   `/v1/responses` using `gpt-5.4-mini` and a strict structured-output schema.
5. After that batch is complete, submit a second `gpt-5.4` batch only for
   low-confidence, high-materiality, contradictory, specified major-event, or
   invalid-output cases. Pro models are rejected by configuration validation.
6. Permanently cache each validated result by article-content hash, ticker,
   model, prompt version, and schema version. Identical work is never submitted
   twice, including duplicates within one run.
7. Enter at the first market open strictly after the New York publication date
   and measure adjusted 1/3/5-trading-day returns.
8. Write article evidence, accuracy/IC metrics, per-attempt token/cost ledgers,
   an immutable manifest, DuckDB views, and a self-contained HTML report.

The model returns only sentiment score/label, confidence, relevance,
materiality, novelty, event type, expected horizon, tradable/abstain flags, and
concise reasoning capped at 40 words. Article identity and timestamps come from
the trusted local record rather than being echoed by the model.

## Cost controls

The fixed system prompt is shared across requests and a stable prompt-cache key
is supplied. Output is capped at 256 tokens for both models. Published Batch
rates are versioned in `config/settings.yaml`; every API result records exact
input, cached-input, output, and reasoning tokens plus estimated USD cost.

| Run tier | Hard limit |
|---|---:|
| Smoke test | $1 |
| First research sample | $5 |
| Expanded validation | $20 |

Before uploading JSONL, the client computes a conservative maximum using one
input token per UTF-8 request byte, an additional overhead allowance, and the
full output cap. A batch that cannot fit within the remaining run budget stops
before upload. Batch state is content-addressed, so an interrupted local run
resumes the same remote batch instead of resubmitting it.

The implementation follows the official [Batch API guide](https://developers.openai.com/api/docs/guides/batch),
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs),
[prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching),
and [pricing page](https://developers.openai.com/api/docs/pricing).

## Verified status

As of 2026-07-18, the cached EODHD smoke selection ran successfully:

- 500 `AAPL.US` articles considered; 488 filtered before OpenAI.
- Pre-request exclusions: 33 inadequate-text records, 33 broad market
  summaries, four duplicate stories, and 418 records beyond the 12-item sample
  cap. No ticker-mapping or date-window failures occurred.
- 12 full-text articles selected across 11 UTC publication dates; no
  headline-only records selected.
- 48 adjusted price rows; all 12 events have conservative entries and available
  adjusted 1/3/5-day returns, with every entry strictly after publication.
- Conservative no-submit cost ceiling: $0.044043 for the mini batch and
  $0.146733 more if all 12 articles escalate, or $0.190775 combined against the
  $1 smoke limit.
- Ruff, strict MyPy, and 41 tests pass; package coverage is 90.16% against an
  85% gate.

`OPENAI_API_KEY` is not configured in the current local runtime. Consequently,
no paid batch was submitted and no real sentiment, accuracy, or IC result is
claimed. The research conclusion remains **INCONCLUSIVE**.

## Install

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```powershell
git clone https://github.com/JoshGutierrez56/DeepSeek-Generative-AI-Sentiment-Analysis-Algorithm.git
cd DeepSeek-Generative-AI-Sentiment-Analysis-Algorithm
git switch agent/openai-eodhd-rebuild
uv sync --extra dev
Copy-Item .env.example .env
```

Populate the untracked `.env` without pasting credentials into chat:

```dotenv
EODHD_API_TOKEN=...
OPENAI_API_KEY=...
DATA_ROOT=./data
DUCKDB_PATH=./data/research.duckdb
LOG_LEVEL=INFO
```

The first-pass and escalation models live in validated YAML, not environment
variables. The defaults are `gpt-5.4-mini` and `gpt-5.4`.

## Run the smoke milestone

Reuse the cached provider responses and inspect all filter counts:

```powershell
uv run sentiment-lab data sync --config config/experiments/milestone.yaml
```

Run the complete cost-bounded Batch workflow after configuring the OpenAI key:

```powershell
uv run sentiment-lab milestone run --config config/experiments/milestone.yaml
```

Do not add `--refresh` to a reproducibility rerun. EODHD and permanent OpenAI
caches are reused automatically; there is intentionally no force-reclassify
option.

Each completed result directory contains:

- `articles.parquet` — provider text, timestamps, symbols, provider sentiment,
  raw-response hash, and full-text/headline-only type.
- `assessments.parquet` — the final structured assessment, model/stage,
  prompt/schema/cache hashes, tokens, and estimated historical classification
  cost.
- `classification_ledger.parquet` — every mini/escalation/cache attempt with
  exact usage, estimated cost, current-run cost, batch IDs, and failure reason.
- `events.parquet` — article, final assessment, conservative entry, and future
  adjusted returns.
- `metrics.json` — coverage, directional accuracy, Spearman IC, Pearson
  correlation, confidence-weighted IC, and per-label returns.
- `manifest.json` — git/config/data/artifact hashes, filter counts, Batch IDs,
  requested/returned models, budget ceiling, exact run usage/cost, and metrics.
- `report.html` — compact evidence, filter, classification, cost, and return
  tables.

## Validate

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src/sentiment_lab
uv run pytest
```

## Timing and interpretation

The active policy is `conservative_next_day_open`: ignore the publication's
local market date, enter at 09:30 New York time on the first later EODHD trading
date, and back-adjust the raw open using that session's close adjustment factor.
The engine rejects identity/timestamp mismatches and proves the entry is after
the article.

Directional accuracy maps returns within +/-10 bps to neutral. IC is Spearman
correlation between the continuous score and future return. Abstained rows stay
in evidence artifacts but are excluded from accuracy and IC.

A 12-article, one-ticker smoke sample cannot establish significance. HAC/block
bootstrap inference, baselines, costs, walk-forward validation, and an untouched
test remain gated behind successful real classification of this vertical slice.
See [milestone status](docs/MILESTONE_STATUS.md), [repository audit](docs/REPOSITORY_AUDIT.md),
and [implementation plan](docs/IMPLEMENTATION_PLAN.md).

## License

MIT. See [LICENSE](LICENSE).
