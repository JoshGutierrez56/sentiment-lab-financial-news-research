# Hybrid 5,000-Article Final Report

## Decision

**Research conclusion: INCONCLUSIVE**

**Scale recommendation: Stop at 5,000**

The local classifier and data pipeline completed reliably, but the frozen
holdout did not show stable incremental value after dependence adjustment,
simple baselines, and realistic costs. No expansion beyond the frozen 5,000
articles is justified by the present evidence.

## Frozen design

- Sample: 5,000 unique full-text articles, frozen before inference.
- Sample hash: `7b07079fb2bcbf7546e1dd810ee081ddb86adb7bb37aa0979efac31fe30553a7`.
- Chronological splits: 3,000 development, 1,000 validation, 1,000 holdout.
- Local model: `qwen3.6:35b-a3b`, Q4_K_M, deterministic non-thinking output.
- Original OpenAI calibration: immutable experiment
  `20260718T232828Z_70aaf344` (250 articles).
- Additional OpenAI calibration: 250 articles, hard $1 budget.
- Primary specification: sentiment times confidence, strongest event per
  company-day, minimum materiality 0.50, 5- and 21-session horizons.
- Holdout results were read only after the specification manifest was frozen.

## Local classification quality

- Valid outputs: 5,000/5,000.
- Tradable coverage: 97.44%.
- Labels: 1,208 bullish, 2,822 neutral, 970 bearish.
- Event type `other`: 26.6%.
- Initial structured-output validity: 99.28%; all repairable failures were
  resolved and permanently cached.

On the additional OpenAI calibration subset, local/OpenAI exact label agreement
was 70.0%, score Pearson/Spearman correlation was 0.743/0.719, directional
agreement was 95.9%, and tradable/abstain agreement was 78.0%. Event-type
agreement was only 41.2%, a material taxonomy limitation. OpenAI marked 71.6%
tradable versus 84.0% for the local model on this subset.

## Frozen holdout evidence

The selected company-day signal produced the following event-level holdout
results. These are predictive diagnostics, not portfolio Sharpe ratios.

| Horizon | Observations | Average signed return | Bull-minus-bear spread | Pearson IC | Spearman IC |
|---|---:|---:|---:|---:|---:|
| 5 sessions | 625 | +0.035% | +0.469% | 0.049 | 0.041 |
| 21 sessions | 625 | -0.253% | -1.774% | -0.042 | -0.097 |

The 5-session company-clustered interval crossed zero
(`[-0.053%, +0.832%]`), as did the date-block interval. The 21-session result
reversed sign. Development and validation were not consistently positive, and
the frozen model-selection q-value was approximately 0.708.

## Baselines and portfolio results

The local signal did not add consistent value over event-type, EODHD, keyword,
or short-term-reversal baselines. In particular, the event-type and reversal
baselines had stronger holdout rank ICs at important horizons.

Portfolio Sharpe ratios below use explicit daily marked returns with overlapping
positions handled by the engine. They are not calculated from event returns.

| Market-neutral portfolio | Gross Sharpe | Base-cost net Sharpe | Conservative-cost net Sharpe | Base-cost total return |
|---|---:|---:|---:|---:|
| 5 sessions | 1.140 | 0.008 | -1.621 | -0.002% |
| 21 sessions | -1.949 | -2.679 | -3.736 | -1.950% |

The 5-session gross result was fully consumed by costs. The 21-session
market-neutral portfolio was negative before costs. Exposure concentration also
failed the preregistered decision gate.

## Cost and runtime

- Original OpenAI calibration: $0.348443.
- Additional OpenAI calibration: $0.394876.
- Cumulative OpenAI cost: $0.743319.
- Estimated local energy: 0.6450 kWh.
- Estimated local electricity cost: $0.1613.
- Estimated total hybrid cost: $0.9046.

The local run was interrupted when the GPU was needed for a game and resumed
from permanent cache. The final inference segment was directly metered; the
earlier segment's runtime and electricity are estimated from observed duration
and benchmark power, so electricity figures are estimates rather than utility
meter readings.

## Limitations

- The untouched holdout covers only 2026 Q1, limiting regime inference.
- Return windows overlap; clustered and block-bootstrap methods were used where
  the split length permitted. The 21-session holdout was too short for a useful
  date-block interval.
- The fixed liquid-company universe leaves survivorship-bias risk.
- Daily adjusted data cannot reproduce intraday spreads or exact fills;
  transaction costs are scenario assumptions.
- Local/OpenAI event-taxonomy agreement is weak even when directional sentiment
  agreement is high.
- Local tradable coverage is much higher than OpenAI coverage, suggesting a
  meaningful calibration difference in abstention behavior.
- Syndicated stories were clustered and suppressed, but imperfect provider text
  and entity metadata can still leave residual duplicates or mapping errors.

## Reproduction

Run the hash-locked command sequence in the project README. The final report and
machine-readable decision are in:

- `data/results/hybrid_5000_final/report.html`
- `data/results/hybrid_5000_final/results.json`

Quality gates at completion: Ruff passed, strict MyPy passed, and 100 tests
passed with 86.07% coverage.
