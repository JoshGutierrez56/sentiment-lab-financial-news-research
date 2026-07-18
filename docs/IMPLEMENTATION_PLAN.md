# Implementation plan

> **Priority override — 2026-07-18:** Only the mandatory vertical slice is
> active: real EODHD articles → OpenAI structured sentiment → conservatively
> aligned future returns → basic accuracy/IC report. Phases 2–4 below are paused
> and must not begin until that milestone works end to end.

This is an execution plan. Each phase ends in runnable tests and artifacts, not
only scaffolding. The default primary specification remains locked until Phase 4.

## Engineering principles

1. Raw provider responses are immutable and stored before normalization.
2. Every normalized row carries source, retrieval, hash, and availability time.
3. All configuration is typed and rejects unknown fields.
4. Secrets enter only at runtime and never appear in URLs, logs, manifests, or
   cache keys.
5. Signal time, decision time, and execution time are separate fields.
6. Cached-data reruns make no network or model calls and reproduce artifact
   hashes within declared tolerances.
7. Negative/null experiments are first-class registry entries.
8. Final-test thresholds and parameters are frozen before test-period access.

## Mandatory milestone — article to measured return

The smallest accepted run must produce real EODHD article rows containing the
article text, publication timestamp, requested company/ticker, OpenAI label,
score, confidence, concise evidence-based reasoning, next tradable session,
future 1/3/5-day adjusted returns, direction accuracy, and Spearman information
coefficient. It must write machine-readable JSON/Parquet plus a compact HTML
report and be rerunnable from cache without external calls.

Advanced portfolio construction, factor neutralization, dashboards, universe
infrastructure, and broad experiment frameworks are explicitly deferred.

## Phase 1 — provider-to-classification vertical slice

- Modern Python 3.11+ package, CLI, lint/type/test/coverage configuration.
- Strict YAML/runtime configuration and named config registries.
- EODHD HTTP client with injected transport, timeouts, exponential backoff plus
  jitter, `Retry-After`, redacted request logs, pagination, and immutable raw
  cache.
- Validated EOD, intraday, news, provider-sentiment, fundamentals, symbols, and
  exchange-calendar responses.
- Normalizers with stable article IDs, UTC timestamps, deduplication, Parquet
  storage, and DuckDB views.
- OpenAI Batch API over `/v1/responses` with a strict Pydantic schema, fixed
  system prompt, two versioned prompt variants, permanent exact-configuration
  cache, per-article usage/cost ledger, conservative budget preflight, and a
  selective second-model escalation batch after the first pass completes.
- Unit/integration tests use injected HTTP transports and deterministic model
  fixtures. A licensed EODHD probe runs separately from the test suite.

Exit gate: `ruff`, core `mypy`, unit tests, mocked integration tests, and a
real EODHD smoke probe pass; OpenAI live smoke is run only if its key exists.

## Phase 2 — event research slice

- Effective-dated ticker/entity resolver using symbol metadata, aliases,
  article symbols, ISIN/CUSIP/CIK/OpenFIGI when supplied, and mapping confidence.
- Exchange-session calendar including weekends, holidays, early closes, and
  pre/post-market classifications.
- Four execution policies with conservative next-day/next-open default.
- Registered event and aggregate sentiment signals, contradiction/flood caps,
  provider-minus-LLM and recent-tone surprise.
- Event-study engine and independent-event trade engine with overlapping-window
  flags and benchmark-adjusted returns.
- Keyword, provider sentiment, event type, momentum/reversal/gap, randomized,
  timestamp-placebo, and ticker-placebo baselines.
- Turnover-aware commission/spread/slippage/impact/borrow cost scenarios.
- Immutable manifest and DuckDB experiment registry.

Exit gate: timestamp/leakage tests, metric regression tests, cost tests, placebo
tests, and cached rerun hash comparison pass.

## Phase 3 — portfolio and inference slice

- Named historical universes: `liquid_100`, `liquid_500`, `us_large_cap`,
  `us_tradeable`, and `research_sample`.
- Point-in-time control factors: momentum, reversal, size, value,
  profitability, asset growth, volatility, beta, liquidity, and available
  earnings features.
- Long-only, long/short, market-neutral, and sector-neutral construction with
  equal/signal/rank/volatility weights and exposure/turnover constraints.
- Walk-forward engine and strict development/validation/final-test partitions.
- HAC/Newey-West inference, event-study CARs, IID and block bootstrap confidence
  intervals, multiple-test correction, and deflated-Sharpe-equivalent diagnostic.
- Stability tables by period, regime, sector, event type, size, confidence,
  model, and prompt.

Exit gate: all folds and controls are materialized; final period remains
unopened and its lock hash is recorded.

## Phase 4 — freeze, evaluation, and reporting

- Versioned human-label template and scoring/calibration workflow.
- Compare prompt variants on a stratified labeled sample before trading-model
  selection.
- Run sensitivity surfaces on development/validation only.
- Freeze primary strategy, gates, model/prompt/schema, universe, costs, dates,
  and random seed in a signed/hash-addressed specification.
- Run the untouched final period once.
- Produce `results.json`, `metrics.parquet`, `trades.parquet`, `events.parquet`,
  `manifest.json`, and `report.html` for every major experiment.
- Report PASS, FAIL, or INCONCLUSIVE without changing thresholds after the run.
- Commit coherent milestones, push the feature branch, and open a draft PR with
  evidence and exact reproduction commands.

## Predeclared primary research gates

The suggested gates in the mission are adopted initially. Before final-test
access, the freeze artifact will specify exact dates, maximum drawdown limit,
minimum event coverage/mapping thresholds, acceptable placebo p-values,
dependence-adjusted confidence threshold, concentration thresholds, and the
definition of “does not collapse” under conservative costs. Any revision must
be made using development/validation evidence and recorded before the final
test hash is opened.

## Current external-state assessment

- `EODHD_API_TOKEN`: present in the runtime environment; entitlement probes are
  authorized and will redact the token.
- `OPENAI_API_KEY`: absent. This does not block implementation, mocked tests,
  cached runs, prompt evaluation tooling, or non-LLM baselines. It blocks only
  live classifications and therefore blocks a definitive primary result.
- GitHub CLI: authenticated as the repository owner; branch push/draft PR can
  occur after verified commits.
