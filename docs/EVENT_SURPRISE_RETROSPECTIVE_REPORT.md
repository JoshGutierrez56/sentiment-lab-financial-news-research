# Event-Surprise Retrospective

## Decision

**NOT PROMOTED.** The frozen event-surprise portfolio did not pass all predeclared promotion gates. The primary holdout base-cost net Sharpe was **-0.7014**; the conservative-cost net Sharpe was **-1.1471**; and the five-session block-bootstrap 95% interval for base-cost net Sharpe was **[-2.9615, 3.1299]**.

This is a strategy decision, not a claim that NLP cannot classify financial text or help analysts. It asks only whether this specific event-surprise construction produced a sufficiently robust, costed daily portfolio in the frozen 5,000-article sample.

## Research question

Can a sparse signal based on the disagreement between a structured event-surprise assessment and FinBERT, scaled by company specificity, materiality, novelty, and confidence, support a tradable five-session long/short event portfolio after explicit costs?

The economic specification was frozen in `config/experiments/event_surprise_retrospective.yaml` before the first portfolio calculation. A pre-canonical verification run then exposed a boundary-handling defect: it retained the earlier split's boundary day and discarded valid terminal-holdout exits. The runner was corrected to purge outcomes reaching the next split and retain complete terminal-holdout paths. No horizon, threshold, weight, cost assumption, promotion gate, or model was changed in response to performance.

## Frozen portfolio specification

- 5,000 full-text articles, 125 companies, 2022-2026, with a chronological 60/20/20 development/validation/holdout split.
- Abstentions excluded; one event per company-day selected by greatest absolute signal, then article ID.
- Direction is the sign of `event_surprise_signal`; no post-freeze signal threshold search.
- Expected edge is scaled by a no-intercept OLS coefficient fitted only on development observations.
- Entry is the frozen next adjusted market open; exit is the adjusted close after five sessions including entry.
- Same-ticker overlaps and split-crossing labels are rejected.
- Each event sleeve is capped at 2% of $1 million; each side is capped at 50%; gross exposure is capped at 100%; volume participation is capped at 1%.
- A trade must have development-fitted expected gross edge at least 2x its estimated base round-trip cost.
- Base costs: $0.005/share with $1 minimum, 5 bps half-spread per side, 2 bps slippage per side, nonlinear volume impact, 3% annualized short borrow, and $0.02 research allocation per entry.
- Conservative costs: 10 bps half-spread, 5 bps slippage, doubled impact coefficient, and 8% borrow; commission and research allocation unchanged.
- Inactive market sessions remain in the daily return series as zero-return cash days.

## Results

| Split | Candidate events | Evaluation events | Accepted trades | Event IC (5d) | Gross Sharpe | Base net Sharpe | Conservative net Sharpe | Base Sharpe bootstrap 95% CI | Base net return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 1927 | 1913 | 160 | 0.0419 | 1.2767 | 0.9785 | 0.6526 | [-0.2988, 2.2499] | 1.71% |
| validation | 661 | 614 | 41 | -0.0178 | 0.3646 | 0.0560 | -0.2919 | [-2.0317, 2.4999] | 0.02% |
| holdout | 517 | 517 | 46 | 0.0349 | -0.2776 | -0.7014 | -1.1471 | [-2.9615, 3.1299] | -0.25% |

The development-only edge slope was `0.0088270800` return units per signal unit. Across all splits, 247 trades passed the cost and capacity rules.

## Promotion gates

| Gate | Result |
|---|---|
| minimum holdout trades | PASS |
| minimum holdout base net sharpe | FAIL |
| minimum holdout bootstrap ci lower | FAIL |
| minimum holdout conservative net sharpe | FAIL |
| positive validation base net return | PASS |
| maximum single ticker absolute pnl share | PASS |

The gate thresholds were a minimum of 30 holdout trades, holdout base net Sharpe of at least 0.75, a block-bootstrap lower bound above zero, conservative holdout Sharpe above zero, positive validation base net return, and no single ticker contributing more than 20% of absolute holdout P&L.

## Reconciliation and rejection audit

| Rejection reason | Count |
|---|---:|
| insufficient_cost_buffer | 2713 |
| same_ticker_overlap | 84 |
| split_boundary | 61 |

Holdout base costs totaled `$1536.26` and conservative costs totaled `$3223.72`. Canonical output includes every prediction, requested order, fill, rejection, position-day, daily portfolio return, and decomposed cost row, allowing dollar P&L to be reconciled from source event through portfolio.

## Relation to the earlier broad-sentiment result

The earlier generic-sentiment five-session portfolio had gross Sharpe 1.1396, but base-cost net Sharpe 0.0082 and conservative-cost net Sharpe -1.6210. That result rejected the easy claim that positive language alone was a durable trading edge. This retrospective tested the narrower surprise-relative-to-expectations hypothesis without rewriting the prior result.

## Important limitations

- The event-level holdout IC was viewed before this portfolio specification was frozen. This is therefore a transparent one-shot final retrospective, not a pristine confirmatory holdout.
- The sample is intentionally balanced by company and is not a production universe or a capacity study.
- End-of-day prices cannot model intraday latency, queue position, or realized spreads.
- Borrow is modeled, but historical locate availability is unavailable; the test assumes every sampled short was locatable.
- The development-fitted linear edge scale is deliberately simple. No alternative calibrator was searched after the freeze.
- Qwen and FinBERT outputs are cached model assessments, not ground truth. Source-hash validation establishes lineage, not semantic correctness.
- Passing the gates would justify further prospective testing, not an alpha or deployability claim. Failing them closes this specification rather than inviting holdout tuning.

## Reproduce

```powershell
uv sync --locked --extra dev
uv run --locked python tools/run_event_surprise_retrospective.py
uv run --locked ruff check .
uv run --locked mypy src/sentiment_lab
uv run --locked pytest
```

The runner is cache-only: it does not call OpenAI, Ollama, EODHD, CUDA, or any network service. It verifies all three frozen input hashes and refuses to overwrite an existing canonical run.

## Evidence identity

- Configuration SHA-256: `3b45aad9a1774ad53d5f325185b66a13012a2a1296ceffe8f9850396480bffc0`
- Signals SHA-256: `9388d59a609fcbef9e5b9cb91bfdfcdccd5fffa71ad424c9b76f9ebfd47efdf5`
- Articles SHA-256: `8ada422fcdefa894c55ae51400e073f97fa6d8e26272cde98d8926ce27b68385`
- Prices SHA-256: `4f030c49deea3dd536dcb4d06f3b41d8447492ebfae2370d3646bb615ce79615`
- Canonical local run: `data/results/event_surprise_retrospective/3b45aad9a1774ad5`
