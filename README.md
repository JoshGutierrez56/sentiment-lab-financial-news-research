# Sentiment Lab

Sentiment Lab is an evidence-first financial-news research pipeline. It joins full-text equity news to point-in-time market data, produces structured model assessments, and evaluates daily portfolios only after timestamps, data lineage, split boundaries, execution rules, and costs have been locked.

The main result is negative and intentionally reported that way: **neither generic news sentiment nor the final event-surprise redesign was promoted as a standalone trading strategy.** The repository is useful as a reproducible research and model-risk case study—not as a claim of production alpha.

> Research software only. Nothing in this repository is investment advice or a representation of live trading performance.

## Final results

### Frozen 5,000-article study

The corpus contains 5,000 unique full-text articles covering 125 U.S. companies across 11 sectors and five years. Sampling was balanced at 40 articles per company. All studies use a chronological 60/20/20 development, validation, and holdout split; conservative next-open entry; adjusted returns; explicit daily portfolio series; and immutable source hashes.

| Specification | Evaluation | Gross Sharpe | Base-cost net Sharpe | Conservative net Sharpe | Decision |
|---|---|---:|---:|---:|---|
| Generic structured sentiment | 5-session holdout portfolio | 1.1396 | 0.0082 | -1.6210 | Not promoted |
| Generic structured sentiment | 21-session holdout portfolio | -1.9490 | -2.6790 | -3.7360 | Not promoted |
| Event-surprise redesign | 5-session holdout portfolio | -0.2776 | -0.7014 | -1.1471 | Not promoted |

The generic five-session signal looked attractive before costs, but its gross contribution was only slightly larger than modeled turnover costs. The event-surprise redesign asked a better economic question—whether news was company-specific, material, novel, and surprising relative to prior information—but it also failed the frozen portfolio gates.

For the event-surprise holdout:

- 517 company-day events were evaluable;
- 46 passed the development-fitted 2x round-trip cost hurdle and stateful portfolio rules;
- event-level five-session Spearman IC was 0.0349;
- base-cost net return was -0.25% over 66 market sessions;
- base-cost net Sharpe was -0.7014;
- the five-session block-bootstrap 95% Sharpe interval was [-2.9615, 3.1299]; and
- three of six promotion gates failed, including base Sharpe, bootstrap lower bound, and conservative Sharpe.

Positive event-level IC did not translate into a robust portfolio. That distinction is central to the project: classification quality, predictive association, and deployable performance are different claims.

Read the complete [event-surprise retrospective](docs/EVENT_SURPRISE_RETROSPECTIVE_REPORT.md) and its [machine-readable evidence](docs/evidence/event_surprise_retrospective_summary.json). The earlier generic-sentiment result is documented in the [5,000-article final report](docs/HYBRID_5000_FINAL_REPORT.md) and [artifact closeout](docs/HYBRID_5000_CLOSEOUT.md).

## What this repository demonstrates

- **Point-in-time alignment:** publication timestamps are converted to New York time and entered at the first eligible adjusted market open after publication.
- **Immutable data lineage:** article, price, model-output, configuration, and result artifacts are content-addressed and hash-checked.
- **Strict structured outputs:** model assessments are validated against typed schemas; invalid, ambiguous, or generic records abstain.
- **Local and hosted model workflows:** an immutable 250-article OpenAI calibration set is preserved; the 5,000-article extraction uses cached Qwen outputs plus pinned FinBERT baselines.
- **Leakage controls:** chronological splits, outcome-boundary purging, development-only calibration, story deduplication, and one company-day event rule.
- **Stateful execution:** same-ticker overlap suppression, side and gross exposure caps, volume participation limits, and auditable order/fill/rejection ledgers.
- **Decomposed costs:** commissions, half-spread, slippage, nonlinear volume impact, short borrow, and a research-cost allocation are reported separately.
- **Falsifiable promotion gates:** a strategy is closed when it fails rather than retuned on holdout.
- **Reproducible engineering:** Ruff, strict mypy, a two-version GitHub Actions matrix, and 110 tests with an 85% coverage gate.

## Research lineage

The repository preserves three distinct stages rather than overwriting earlier evidence:

1. **OpenAI calibration (250 articles).** Operational validation and comparison dataset; not large enough for a trading conclusion.
2. **Generic sentiment study (5,000 articles).** Qwen structured sentiment evaluated against forward returns and explicit daily portfolios. Gross five-session performance did not survive costs.
3. **Event-surprise redesign (5,000 articles).** Sparse surprise-relative-to-expectations signal, development-only edge calibration, a 2x cost hurdle, purged boundaries, and one frozen stateful retrospective. It was not promoted.

The final redesign did not rewrite the old strategy or tune it until it passed. A pre-canonical verification run did expose a split-boundary implementation defect: the earlier split retained its boundary day while valid terminal-holdout exits were discarded. The runner was mechanically corrected to purge outcomes reaching the next split and retain complete terminal paths. No horizon, signal threshold, weight, cost assumption, model, or promotion gate changed in response to performance.

## Method summary

The primary event-surprise signal is:

```text
(Qwen direction score - FinBERT score)
× company specificity
× materiality
× novelty
× confidence
```

Abstentions are zeroed and excluded from portfolio selection. The strongest absolute qualifying event is selected per company-day with article ID as the deterministic tie-break. A no-intercept slope is fitted on purged development observations only, then used to estimate dollar edge for the fixed five-session portfolio. Validation and holdout outcomes do not set the threshold or calibrator.

The portfolio uses $1 million starting capital, 2% maximum event sleeves, 50% per-side caps, 100% maximum gross exposure, 1% maximum volume participation, no same-ticker overlap, and no rebalance between entry and exit. Inactive market sessions remain in the daily series as zero-return cash days.

The complete frozen specification is [event_surprise_retrospective.yaml](config/experiments/event_surprise_retrospective.yaml).

## Install and validate

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```powershell
git clone https://github.com/JoshGutierrez56/sentiment-lab-financial-news-research.git
cd sentiment-lab-financial-news-research
uv sync --locked --extra dev

uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src/sentiment_lab
uv run --locked pytest
uv build
```

The core test suite does not require API credentials, proprietary news data, Ollama, or a GPU.

## Reproduce the frozen retrospective

Large and licensed research artifacts are intentionally excluded from Git. To reproduce the canonical portfolio, place the three frozen Parquets at the paths declared in the configuration or rebuild them from your own authorized EODHD data and local model cache. The runner refuses any input whose SHA-256 does not match the frozen specification.

```powershell
uv run --locked python tools/run_event_surprise_retrospective.py
```

The command is cache-only and makes no network, OpenAI, Ollama, CUDA, or EODHD calls. It writes an immutable directory keyed by configuration hash containing:

- `predictions.parquet`
- `orders.parquet`
- `fills.parquet`
- `rejected_orders.parquet`
- `positions.parquet`
- `daily_returns.parquet`
- `cost_breakdown.parquet`
- `metrics.json`
- `manifest.json`
- `report.html`

Canonical configuration SHA-256: `3b45aad9a1774ad53d5f325185b66a13012a2a1296ceffe8f9850396480bffc0`.

## Repository map

| Path | Purpose |
|---|---|
| `src/sentiment_lab/event_surprise/` | Strict event schema, signal construction, frozen portfolio retrospective |
| `src/sentiment_lab/execution/` | Stateful execution and decomposed cost models |
| `src/sentiment_lab/hybrid/` | Sampling, local inference, calibration, baselines, portfolios, and reporting |
| `src/sentiment_lab/validation/` | Purged walk-forward validation primitives |
| `config/experiments/` | Hash-locked experiment and promotion specifications |
| `tools/` | Cache-only inference repair, closeout, and final retrospective runners |
| `tests/` | Unit and integration tests, including leakage and accounting invariants |
| `docs/` | Methodology, model benchmark, closeouts, audits, and final reports |

## Evidence and limitations

The canonical retrospective manifest records implementation commit `ca9c8ae67d3c93b4bdc402530f00e000940962c0`, a clean worktree, the random seed, platform, exact input hashes, and SHA-256 for every canonical output Parquet.

Important limitations remain:

- holdout event-level IC was viewed before the final portfolio specification, so the result is a transparent one-shot retrospective rather than a pristine confirmatory test;
- the balanced sample is not a production universe or capacity estimate;
- end-of-day data cannot represent intraday latency, queue position, or realized spreads;
- historical locate availability was unavailable, so shorts are assumed locatable while borrow is modeled;
- model output is not ground truth, and hash validation proves lineage rather than semantic correctness; and
- even a passed retrospective would require prospective validation before any alpha or deployability claim.

## Development disclosure

The research questions, economic constraints, validation standard, and project direction were set by Josh Gutierrez. AI coding agents generated substantial portions of the implementation under those specifications. The repository therefore emphasizes inspectable source, tests, immutable artifacts, explicit failure gates, and reproducible evidence rather than claiming unaided authorship.

## License

MIT. See [LICENSE](LICENSE).
