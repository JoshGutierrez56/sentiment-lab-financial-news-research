# Expectation-Adjusted News Study Protocol

**Evidence stage:** design only  
**Freeze state:** not frozen  
**Results:** none  
**Confirmatory status:** no confirmatory test may begin until every freeze requirement is complete

## Purpose

The completed sentiment and event-surprise studies showed that a weak positive
event-level association can disappear after portfolio construction and costs.
This follow-up asks a different question: does interpretation of company news
add information after the market's measurable expectations and standard
fundamental and price controls are already known?

This is a new study. The viewed 2022–2026 holdout from the prior project is
prohibited from supporting confirmatory claims here. It may be used only for
engineering tests explicitly labeled exploratory.

## Research question

Does a combined model containing point-in-time expectations, reported actuals,
fundamental controls, price controls, event type, and structured text features
produce incremental out-of-sample information beyond the strongest comparable
non-text baseline?

The primary economic unit for version 1 is a U.S. company's quarterly EPS
announcement. Guidance and analyst revisions require different definitions of
the actual, expectation, and decision clock, so they are deferred to separate
protocols. This narrow event universe makes the meaning of “expected” auditable
instead of asking a language model to infer the market's full information set.

## Falsifiable hypotheses

1. **Incremental information:** the combined model's prospective five-session
   factor-residual IC exceeds the best declared non-text baseline, and the
   date-block-bootstrap lower bound for that difference is above zero.
2. **Economic implementation:** a stateful long/short portfolio based on the
   frozen combined model passes every base- and conservative-cost promotion
   gate.
3. **Concentration:** no single ticker or sector explains more than the frozen
   share of absolute prospective P&L.

Failure of any required gate prevents promotion. Classification quality alone
cannot satisfy an investment hypothesis.

## Point-in-time data contract

Every normalized row must retain an immutable source identifier, retrieval or
availability timestamp, source revision, and content hash. Licensed source rows
must never be committed to Git.

| Entity | Required timing rule | Minimum fields |
|---|---|---|
| Reported actual | `available_at >= announced_at` | ticker, event, metric, period, value, unit, source, hash |
| Expectation snapshot | `available_at < event.announced_at` | ticker, metric, period, value, dispersion, contributor count, revision, hash |
| Fundamental/price controls | `available_at < event.announced_at` | sector, beta, size, liquidity, value, quality, growth, momentum, volatility, hash |
| News | immutable provider availability time | article ID, company mapping, full-text hash, source revision |
| Returns and factors | observable under the declared execution clock | adjusted prices, volume, factor returns, corporate-action state |

The typed contract is implemented in
[`src/sentiment_lab/expectation_adjusted/schemas.py`](../src/sentiment_lab/expectation_adjusted/schemas.py).
It rejects expectations or control records that became available at or after
the event announcement.

### Selected expectations source and pilot

The bounded source audit uses authorized WRDS access to four tables:

| Purpose | Table | Fields that establish the contract |
|---|---|---|
| Unadjusted quarterly EPS actual | `ibes.actu_epsus` | `pends`, `anndats`, `anntims`, `actdats`, `acttims`, `value`, `curr_act` |
| Unadjusted consensus snapshot | `ibes.statsumu_epsus` | `statpers`, `fpedats`, `meanest`, `medest`, `stdev`, `numest`, `curcode` |
| Historical security link | `wrdsapps_link_crsp_ibes.ibcrsphist` | `ticker`, `permno`, `sdate`, `edate`, `score` |
| Split basis | `crsp.dsf_v2` | `permno`, `dlycaldt`, `dlycumfacshr` |

For the pilot, the consensus is the latest `statsumu_epsus` observation whose
`statpers` is **strictly earlier** than the actual's `anndats`. The historical
I/B/E/S-to-CRSP link must be effective on the announcement date and have a link
score no greater than 1. The query selects USD, U.S.-file, quarterly EPS rows
only and retains the earliest activation when duplicate actual versions exist.

I/B/E/S unadjusted estimates and actuals can refer to different share bases
when a split occurs between the consensus date and earnings report. Following
Method 3 in WRDS's *A Note on IBES Unadjusted Data*, the actual placed on the
consensus share basis is:

```text
actual_unadjusted
× CRSP cfacshr at consensus statistical period
÷ CRSP cfacshr at report date
```

The adapter collects both factors but does not calculate returns, IC, Sharpe,
or a portfolio. One observation must be derived manually before the pilot may
scale. WRDS's database catalog labels the raw announcement and activation
times, but it does not document their time zone. The pilot therefore preserves
those raw fields without assigning UTC. A normalized study row cannot be built
until the provider time-zone convention has been documented and tested.

Licensed observations are limited to 25 and written only beneath
`data/private/wrds_ibes_eps_pilot`, which is ignored by Git. The runner requires
both `--live` and the explicit `SENTIMENT_LAB_ENABLE_LIVE_WRDS_IBES=1` gate. The
password remains in PostgreSQL's password file and is never accepted as a
command-line argument.

See the [WRDS I/B/E/S pilot runbook](WRDS_IBES_EPS_PILOT.md) for the exact safe
workflow and the official WRDS references.

## Surprise definitions

The raw structured surprise is:

```text
reported actual - point-in-time expectation
```

When valid dispersion is available, the primary standardized surprise is:

```text
(reported actual - point-in-time expectation) / expectation dispersion
```

Metric, fiscal period, unit, and ticker must match before either value is
computed. Guidance ranges require a separately frozen normalization rule; the
midpoint cannot be chosen after results are observed.

## Nested models

The study will compare five feature families using the same event eligibility,
splits, targets, and evaluation clock:

1. price and liquidity only;
2. structured expectations and fundamentals only;
3. event type and industry only;
4. structured text only; and
5. all declared features combined.

The primary test is the combined model's incremental performance relative to
the best non-text baseline. This prevents a strong earnings-surprise or momentum
effect from being misattributed to NLP.

The primary model is an interpretable regularized linear model. Scaling,
imputation, and hyperparameter selection use development data only. More complex
models are secondary until the linear baseline and its failure modes are fully
understood.

## Targets and validation

- Primary target: five-session factor-residual return.
- Diagnostic targets: one- and 21-session factor-residual returns.
- Training outcomes that reach a later split are purged.
- Development-only purged walk-forward folds select hyperparameters.
- The validation segment may reject or freeze the design but cannot be merged
  into training after prospective evaluation begins.
- The prospective holdout contains only observations available after the final
  specification hash is signed.
- Declared secondary tests use false-discovery-rate control; all undeclared
  slices are exploratory and require a new holdout before confirmation.

## Portfolio evaluation

The existing stateful execution engine will be reused only after predictive
testing is complete. The portfolio retains explicit orders, fills, rejections,
positions, daily returns, turnover, exposures, and decomposed commissions,
spread, slippage, volume impact, borrow, and research cost.

Required reporting adds:

- market, sector, and style-factor exposure;
- P&L attribution by ticker, sector, event type, and cost component;
- signal decay by horizon;
- capacity and participation sensitivity; and
- the combined model's incremental contribution over each nested baseline.

The draft numerical gates live in
[`expectation_adjusted_news_v0.yaml`](../config/experiments/expectation_adjusted_news_v0.yaml).
They become binding only when all data sources, date boundaries, normalization
rules, and input hashes are frozen together.

## Required decisions before freeze

1. Finish the WRDS I/B/E/S source audit: manually validate one split-adjusted
   observation, document raw time-zone semantics, and freeze revision rules. A
   current consensus value backfilled historically is invalid.
2. Freeze the eligible security universe, event metrics, fiscal-period mapping,
   and exact historical development/validation dates.
3. Freeze the factor model and residual-return construction.
4. Write executable tests for every as-of join, provider revision, metric/unit
   mapping, and corporate action.
5. Run a data-availability audit without computing predictive returns.
6. Sign the final YAML and input snapshot hashes before the prospective clock
   begins.

## Implementation acceptance checklist

Before any result is presented, the primary researcher should be able to:

- derive raw and dispersion-scaled surprise by hand;
- explain every timestamp and why each join is point-in-time;
- reproduce one observation from source records to portfolio entry;
- modify one schema or control and update its tests;
- explain why factor residualization changes the research claim; and
- name the result that would falsify the investment hypothesis.

This checklist is part of the evidence standard. A generated pipeline is not a
substitute for research ownership.
