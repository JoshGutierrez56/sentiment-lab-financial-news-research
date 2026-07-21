# Hybrid 5,000-Article Study: Bounded Closeout

## Decision

**REDESIGN**

The predictive conclusion remains **INCONCLUSIVE** and the scale recommendation remains
**Stop at 5,000**. No diagnostic subset is promoted. No additional classifications,
OpenAI requests, threshold tuning, sample expansion, or options work were performed.

## Reproducibility status: PASS

End-to-end reproducibility has been restored from the frozen 5,000-article sample and
the permanent schema-validated local cache.

- Canonical classification SHA-256:
  `a1bd6afa5d015b17c16412c1116342cb2203f64f1a4d14d102d8d9bae7180df7`.
- Two independent cache-only rebuilds produced that same physical Parquet hash.
- The canonical table is 5,000 rows by 23 deterministic scientific/provenance columns.
- The selected development/validation specification reproduced exactly after the
  canonical rebuild: signal `sentiment_confidence`, strongest company-day aggregation,
  minimum materiality 0.5, and unchanged selection statistics.
- Prediction, baseline, portfolio, calibration-analysis, and final-report artifacts were
  rerun against the canonical hash. The final report still concludes `INCONCLUSIVE` and
  `Stop at 5,000`.
- The immutable original OpenAI 250 artifact and all permanent caches remain unchanged.

The reproducibility manifest is
`data/results/hybrid_local_3c4cdaf2fd9d9a16/reproducibility.json`.

## `classifications.parquet` root cause

### When and why it changed

The originally reported metrics were computed before the later rewrite and pinned the
classification SHA-256
`43bdba5e10e5337ecb676e829bdf28633f304b29e3c131baaf8df6fb0167202e`.
The downstream development/validation and holdout event artifacts were created between
03:34 and 03:38 EDT on July 19, 2026.

The classification file was then recreated at approximately 03:45:42 EDT and its
manifest completed at approximately 03:46:45 EDT. The exact code path was a completed
`run_local_classification` invocation using
`config/experiments/hybrid_local_5000.yaml`. Its manifest makes the cache-only nature
unambiguous: 5,000 cache hits, zero prompt/output tokens, and 64.422 seconds elapsed.
The old implementation replayed all permanent cache rows and unconditionally rewrote
the Parquet and manifest even though the scientific run had already completed. Commit
`df46911` added a completed-run guard immediately afterward.

### File-level comparison

The original expected file was recovered byte-for-byte from the classification columns
embedded in the locked 4,000-row development/validation events artifact and the locked
1,000-row holdout events artifact. Its recovered SHA-256 equals the pinned hash.

- Rows: 5,000 versus 5,000.
- Columns: the same 28 columns in the same order.
- Dtypes: identical for every column.
- Row ordering: identical by article/ticker; no duplicate article IDs.
- Parquet metadata: both Polars Parquet 1.0, one row group, ZSTD compression, identical
  Arrow schema, encodings, and statistics layout. Serialized metadata differed by one
  byte because the values differed.
- Scientific/model values: identical in every cell.
- Differences were limited to execution state:
  - `from_cache`: 2,163 rows. Original: 2,163 false and 2,837 true; replay: all 5,000 true.
  - `total_duration_ns`: three rows.
  - `eval_duration_ns`: the same three rows.

The three timing differences were CVS, CAT, and NEE cache-writer races. Concurrent local
workers produced the same validated response hash, so the permanent cache correctly
accepted one response; the original caller retained its own timing record while a later
cache replay loaded the winning writer's timings. No assessment value differed.

### Exact source of the original metrics

The 28-column `43bdba5e...` artifact generated the original final metrics. This is proven
by the downstream input hash locks, artifact timestamps, and byte-identical recovery
from the downstream event artifacts. The later `0071dfd1...` replay did not generate
those metrics.

The repair does not replace the expected hash with a new arbitrary value. It changes
the scientific artifact contract: cache-hit state, token counts, and inference timings
are execution telemetry and are excluded from canonical classifications. The permanent
cache remains the immutable source of those operational records.

## Frozen-primary reporting correction

The closeout found a separate downstream defect: the old final report read a generic
strongest-company-day metric instead of applying the exact frozen primary thresholds,
and generic strongest selection ranked by a different signal. The corrected holdout
block applies minimum materiality 0.5 and ranks by the selected
`sentiment_confidence` signal.

- Corrected 5-session holdout: 556 company-days, +0.0904% average signed return,
  +0.4163% bull/bear spread, Spearman IC 0.0419. Company and date-block intervals
  cross zero.
- Corrected 21-session holdout: 556 company-days, -0.1643% average signed return,
  -1.8073% spread, Spearman IC -0.1024. The available dependence-adjusted interval
  crosses zero.

This changes magnitudes but not the conclusion.

## Why 97.44% was tradable versus 45.6%

The comparison below uses only the 191 original-250 observations where local and OpenAI
had exact three-way directional-label agreement.

- Tradable: local 61.26%, OpenAI 36.13%.
- Abstained: local 38.74%, OpenAI 63.87%.
- Cross-counts: 66 both tradable, 51 local-only tradable, 3 OpenAI-only tradable, and
  71 both abstained.
- Relevance: local mean 0.621 versus OpenAI 0.304; local-minus-OpenAI paired difference
  +0.317. Shares at least 0.7 were 57.07% and 25.13%.
- Materiality: local mean 0.293 versus OpenAI 0.131; paired difference +0.162. Shares at
  least 0.5 were 25.65% and 9.95%.
- Confidence: local mean 0.880 versus OpenAI 0.924; paired difference -0.044. The local
  model was not more confident numerically.

The 51-to-3 imbalance in one-sided trade decisions shows systematic permissiveness,
but not generalized numeric overconfidence. The mechanism was lower abstention plus
higher relevance and materiality judgments.

The full coverage gap has two approximately equal components:

- Model-policy effect on the same original sample: OpenAI 45.6% to local 70.0%,
  or +24.4 percentage points.
- Sample-selection effect under the same local model: original 250 local 70.0% to
  hybrid 5,000 local 97.44%, or +27.44 percentage points.

The hybrid sample deliberately preferred company-specific, full-text articles with
complete return windows and deterministic relevance screens. It therefore cannot be
expected to reproduce the broad original calibration sample's abstention rate.

## Five-session gross-to-net Sharpe attribution

- Gross Sharpe: 1.1396; base-net Sharpe: 0.0082.
- Gross total return: +0.6876%; base-net total return: -0.0021%.
- Turnover: 6.8722 times portfolio capital.
- Combined base friction: 10 bps one-way applied to turnover, producing 68.722 bps of
  additive cost drag.
- Bid-ask/slippage: not separately identifiable. The engine has one combined friction
  bucket and does not allocate it among spread, slippage, and commission.
- Commissions: no separate parameter or additional modeled charge outside that combined
  bucket; a commission-only attribution is therefore unavailable.
- Shorting costs: borrow and locate fees were not modeled, so both are zero in the test.
- Accepted positions: 197. Average additive gross return per accepted position was
  0.3514 bps of portfolio capital; average base cost was 0.3488 bps.
- Gross return per unit of turnover: 10.073 bps, essentially equal to the assumed
  10 bps one-way friction.
- Holding-period overlap: 138 same-ticker events were suppressed versus 197 accepted,
  a 41.19% suppression rate among otherwise eligible events.
- Active positions: 979 position-days, 14.83 average over 66 calendar days, 21.28 average
  on 46 active days, and 79 maximum.
- Long book additive contribution: +44.662 bps.
- Short book additive contribution: +24.563 bps.

Both books contributed positively, but their combined 69.224 bps additive gross
contribution was almost exactly consumed by the 68.722 bps base friction. This happened
before any borrow cost or separately modeled commission.

## Exploratory preregistered diagnostics

These are diagnostics only. All use the fixed `sentiment_confidence` signal, 5- and
21-session horizons, preregistered thresholds (confidence 0.7, relevance 0.7,
materiality 0.5), and no holdout-driven threshold selection. Values below are average
signed returns.

| Exploratory subset | Development 5d / 21d | Validation 5d / 21d | Holdout 5d / 21d |
|---|---:|---:|---:|
| Exclude `event_type=other` | -0.148% / -0.209% | +0.121% / +0.445% | +0.145% / -0.139% |
| High relevance | -0.030% / -0.067% | +0.165% / +0.285% | +0.114% / -0.127% |
| High materiality | -0.089% / -0.280% | +0.244% / +0.484% | +0.276% / +0.072% |
| High confidence | -0.039% / -0.062% | +0.182% / +0.302% | +0.133% / -0.112% |
| Local/OpenAI directional agreement | +0.533% / +0.886% | -0.404% / -2.396% | Not sampled |
| Strongest event per company-day | -0.023% / +0.018% | +0.183% / +0.240% | -0.007% / -0.305% |
| Earnings and guidance only | -0.014% / -0.600% | +0.144% / -0.173% | +0.682% / +0.808% |

Every available company-cluster interval for the superficially positive subset results
crosses zero, and no subset is directionally stable across development, validation, and
holdout at both primary horizons. Earnings/guidance is positive in the holdout but fails
to replicate in development and validation, especially at 21 sessions. No subset
warrants an entirely new future study from this evidence alone.

## Final interpretation

- The original 250 was a small classification-calibration sample, not prospective
  predictive validation. Its broader article mix, high abstention, and different sample
  construction did not represent the curated 5,000-sample distribution.
- Even before holdout, the frozen primary specification was negative in development,
  positive in validation, not directionally consistent, and had a Benjamini-Hochberg
  validation q-value of 0.708. Generalization was weak before transaction costs entered.
- The short 2026 Q1 holdout then showed a small, uncertainty-bound 5-session effect and
  a reversed 21-session effect.
- Classifier permissiveness likely diluted the signal because local-only trade decisions
  were common and relevance/materiality were systematically higher. The evidence does
  not support numeric confidence overstatement as the cause.
- Five-session turnover converted a gross return of roughly 10.07 bps per unit turned
  over into essentially zero net performance under a 10 bps one-way cost assumption.
- The requested exploratory subsets contain no stable replication pattern and remain
  unvalidated.

**Closeout decision: REDESIGN.** Archive this study's evidence as immutable. Any future
work should begin with a genuinely new preregistration that redesigns abstention and
materiality calibration, sample comparability, and a decomposed transaction-cost model.
This closeout does not initiate that future study.

## Verification

- Ruff format: 27 files reformatted; subsequent format check clean.
- Ruff lint: passed.
- Strict MyPy: passed for 41 source files.
- Full test suite: 101 passed with 85.43% coverage.
- Machine evidence:
  `data/results/hybrid_5000_closeout/closeout_results.json`.
- Parquet comparison:
  `data/results/hybrid_5000_closeout/parquet_mismatch_comparison.json`.
- Corrected final results:
  `data/results/hybrid_5000_final/results.json`.
