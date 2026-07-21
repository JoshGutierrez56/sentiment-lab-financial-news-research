# Repository audit

Date: 2026-07-18

Target: `JoshGutierrez56/sentiment-lab-financial-news-research` (renamed from the original DeepSeek-era repository)

Reference: `JoshGutierrez56/Optlab-Research`

## Executive finding

The target is not an installable working repository in its current form. Its
actual modular package and tests are stored only inside
`news_sentiment_trader.zip`; the checked-out tree contains a CLI that imports
that absent package. The original idea is useful, and several corrected metric
tests are worth adapting, but its data, entity, timing, portfolio, inference,
and experiment design cannot support a defensible research conclusion.

Optlab-Research contains stronger reusable patterns: strict Pydantic registries,
named point-in-time universes, DuckDB views over Parquet, Polars transforms,
signal dispatch, portfolio primitives, transaction-cost tests, workbench
facades, and explicit point-in-time tests. Those patterns will be adapted rather
than copied wholesale. Optlab also contains material technical debt (duplicated
package trees, tracked caches/data, incomplete workbench imports, incomplete
lineage manifests, and a market-impact implementation that does not actually
scale by dollar volume), so it is not a drop-in base.

## Audit method

- Cloned both repositories and fetched their full histories.
- Inspected every tracked target artifact, including every source and test file
  embedded in the ZIP.
- Built a target semantic/code graph and an Optlab AST graph, then merged them
  into an 820-node, 1,334-edge cross-repository graph.
- Queried the graph for registry, universe, DuckDB, workbench, backtest, and
  sentiment dependencies and verified the returned paths against source.
- Consulted current official OpenAI and EODHD documentation before selecting
  API surfaces.

Graphify semantic extraction used local Ollama and incurred no external API
cost. The Optlab semantic pass was terminated after seven minutes without a
completed chunk; its deterministic code graph completed successfully with 796
nodes and 1,314 edges.

## Target repository: tracked-file inventory

| Artifact | Finding | Decision |
|---|---|---|
| `.env.example` | Correctly avoids real secrets, but is tied to FactSet and DeepSeek. | Replace with EODHD/OpenAI/runtime paths. |
| `.gitignore` | Basic Python/data exclusions; no research-cache, report, DuckDB WAL, or experiment rules. | Expand while retaining auditable directory placeholders. |
| `Gen AI News Algo` | Extensionless, monolithic prototype with placeholder credentials, broad exception swallowing, no timeouts, guessed API contracts, and pytest code in production. | Preserve findings in this audit; remove from the production tree after migration. |
| `README.md` | Describes a modular package and tests not present in the tree, reports example performance unsupported by an artifact, and claims MIT while no `LICENSE` exists. | Rewrite with verified commands, evidence status, and limitations. |
| `news_sentiment_trader.zip` | Contains the only modular source and tests. It also contains generated `egg-info` and malformed brace-expansion directory entries. | Adapt selected logic/tests into normal tracked files, then remove archive. |
| `requirements.txt` | Python 3.10-era flat dependencies; lacks EODHD/OpenAI, typed config, Parquet/DuckDB, lint/type tooling, and reproducible locking. | Replace with `pyproject.toml`; retain a compatibility export only if useful. |
| `run.py` | Imports the ZIP-only `trader` package, performs row-wise synchronous inference, and consumes a precomputed-return CSV. It cannot run from checkout. | Replace with the package CLI/compatibility shim. |
| `setup.py` | Minimal legacy setuptools metadata for a package absent from checkout. | Replace with PEP 517/621 packaging. |

The target has two commits. No credentials were found in tracked source; the
credential-looking strings are placeholders.

## Reusable target components

The following ideas are sound enough to adapt with attribution to the original
repository history:

- Runtime-only secret loading and dependency-passed settings.
- HTTP retry intent and request timeouts from the ZIP client.
- A clean sentiment-to-action boundary, generalized to continuous scores and
  configurable aggregation.
- Mocked LLM tests with injected responses.
- Regression tests for geometric compounding and non-positive drawdown.
- The correction that event-frequency annualization must not blindly assume
  252 IID daily observations.

They still require redesign. In particular, retry behavior needs jitter and
explicit `Retry-After` handling, response schemas need validation, and event
metrics need dependence-aware inference rather than only a rescaled Sharpe.

## Target methodological weaknesses

### Data and lineage

- FactSet request formats are assumed rather than documented or validated.
- No raw-response archive, request manifest, content hash, pagination, or
  incremental refresh.
- Price input is an ad hoc CSV of already-computed returns, preventing price,
  corporate-action, execution, and vendor-revision auditing.
- No stable article ID or deduplication; syndicated/revised stories can be
  traded repeatedly.

### Entity resolution

- The provider ticker is accepted as truth.
- No alias, identifier, exchange suffix, share-class, rename, delisting, or
  effective-date handling.
- Multi-company articles are not represented.

### NLP

- Headline-only YES/NO/UNKNOWN parsing is brittle and loses relevance,
  confidence, novelty, materiality, horizon, source quality, event type, and
  abstention reasons.
- The prompt has no version, schema version, hashes, cache, calibrated cost
  accounting, duplicate awareness, or evaluation sample.
- Row-wise inference cannot be reproduced and encourages accidental re-spend.

### Timing and leakage

- `calendar day + 1` is not a trading calendar and fails on weekends/holidays.
- Publication time is converted without proving timezone awareness.
- No pre-market/intraday/after-hours/early-close classification.
- No signal availability time, provider ingestion delay, execution delay,
  article revision policy, or conservative next-open default.
- No point-in-time universe/fundamental membership or inactive-symbol checks.

### Portfolio and costs

- One article becomes one unit trade; a news flood creates unbounded exposure.
- No aggregation, overlap accounting, capital constraint, sector/beta/factor
  neutrality, turnover control, borrow cost, spread, slippage, or impact.
- Trade returns are treated as an independent sequence even when positions
  overlap.

### Validation and evidence

- No chronological split, walk-forward validation, untouched test, freeze
  record, baselines, placebos, multiple-testing adjustment, HAC inference, or
  block bootstrap.
- The README's example performance is not linked to data, trades, code version,
  or a manifest and must not be treated as evidence.

## Optlab-Research: reusable architecture

| Pattern | Source evidence | Adaptation |
|---|---|---|
| Strict registries (`extra="forbid"`, unique names, kind validation) | `optlab_research/signals/registry.py` | Typed signal, universe, strategy, prompt, and experiment registries. |
| Point-in-time universe SQL and tests | `optlab_research/universes/builder.py`, `tests/test_signals.py` | Effective-dated EODHD metadata/fundamentals and dedicated leakage tests. |
| DuckDB over Parquet | `optlab_research/db.py` | Local raw/normalized/features/results layers and stable views. |
| Polars transforms | signal/universe/backtest modules | Default research frame; Pandas only at compatibility boundaries. |
| Signal library dispatch | `signals/registry.py`, `signals/compute.py` | Registered sentiment transforms, baselines, and control factors without unsafe user-provided eval. |
| Portfolio primitives | `backtest/portfolio.py` | Equal/signal/rank/volatility weighting with explicit exposure constraints. |
| Costs on changes in weights | `backtest/tcost.py` | Spread, slippage, impact, borrow, commissions, and scenario grids on actual turnover. |
| Result persistence | `backtest/result.py` | Required Parquet/JSON/HTML artifact contract. |
| Workbench facade | `workbench/api.py` | A small stable research API over internal modules. |

## Optlab patterns not reused unchanged

- Two divergent package trees (`optlab_research/` and
  `optlab-research/optlab_research/`) make import provenance ambiguous.
- Compiled `__pycache__`, a DuckDB file, Parquet data, notebook outputs, and many
  debug harnesses are committed.
- `workbench/api.py` references symbols/modules absent from the checked-out
  package and has inconsistent registry access.
- Manifests omit git commit/dirty state, config/data snapshot hashes, output
  hashes, random seed, provider/model versions, and result metrics.
- The `sqrt_adv` path checks for `dvol` but its formula never uses the `dvol`
  values or portfolio dollars; it is not a genuine liquidity-impact model.
- Missing-return reweighting can introduce an availability bias unless missing
  outcomes and delistings are handled before portfolio renormalization.
- Formula strings are evaluated with Python `eval`; the new platform uses
  registered callables/controlled enums for untrusted configs.
- Documentation overstates completion in places; source and tests are treated
  as authoritative.

## Verified provider/API choices

Current official documentation supports these integrations:

- EOD prices: `GET /api/eod/{ticker}` with JSON, inclusive `from`/`to`, adjusted
  close, and explicit ordering.
- News: `GET /api/news` with ticker/topic, date filters, `limit` up to 1000, and
  `offset` pagination. Articles include ISO-8601 publication time, content,
  symbols, tags, and provider sentiment.
- Provider sentiment benchmark: `GET /api/sentiments`.
- Intraday bars: `GET /api/intraday/{ticker}`, UTC Unix bounds, and documented
  1m/5m/1h range limits.
- Fundamentals: recommended `GET /api/v1.1/fundamentals/{ticker}` with partial
  `filter` retrieval and identifier/delisting fields.
- Metadata: `/api/exchanges-list/`, `/api/exchange-symbol-list/{exchange}` with
  `delisted=1`, and `/api/v2/exchange-details/{exchange}` for IANA timezone,
  sessions, holidays, and early closes.
- OpenAI historical classification: Batch API JSONL requests to
  `/v1/responses` with Responses Structured Outputs. The active cost override
  configures `gpt-5.4-mini` first pass and selective `gpt-5.4` escalation in
  validated YAML, uses published Batch pricing, and prohibits Pro models.

## Initial risk register

| Risk | Control |
|---|---|
| Missing OpenAI key | Complete mocked/cached implementation and evaluation fixtures; provide exact live command. |
| EODHD plan excludes news/intraday/fundamentals | Run endpoint probes, record HTTP/status/entitlement without token leakage, and degrade to licensed subsets. |
| Historical constituent gaps | Effective-dated active/delisted symbol snapshots where available; disclose remaining survivorship bias. |
| Provider revisions | Immutable raw responses plus retrieval timestamp/content hash; never overwrite a snapshot. |
| LLM drift | Model/prompt/schema/hash in cache and manifest; compare prompt variants and report by version. |
| Overlapping events | HAC and block-bootstrap inference plus company/day aggregation and exposure caps. |
| Researcher degrees of freedom | Predeclared gates, chronological splits, sweep correction, primary-spec freeze, one final-test run. |

## Audit conclusion

The rebuild is justified. The target contributes a small set of corrected tests
and operational ideas; Optlab contributes the architectural vocabulary. The
production system must be newly implemented around auditable provider snapshots,
structured classifications, explicit event clocks, constrained portfolios, and
immutable experiments. No existing result answers the primary research question.
