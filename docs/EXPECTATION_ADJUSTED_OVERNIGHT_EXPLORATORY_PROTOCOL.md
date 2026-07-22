# Expectation-Adjusted News Overnight Exploratory Protocol

**Status:** exploratory_historical_locked_not_confirmatory  
**Freeze time:** 2026-07-21 21:05 America/New_York  
**Evidence standard:** locked historical exploratory benchmark only  
**Confirmatory or deployment claims:** prohibited

This run is an isolated stacked-branch benchmark. The prior 2022-2026 news
holdout has already been viewed, so every result from this run must be described
as historical and exploratory. It cannot be called untouched, prospective,
promoted, successful, production-ready, or evidence of deployable alpha.

## Frozen Inputs

The benchmark derives its 125-name universe from the immutable 5,000-article
input in the original read-only data repository. Only safe source hashes and
aggregate coverage counts may be reported.

| Input | SHA-256 |
|---|---|
| `articles.parquet` | `8ada422fcdefa894c55ae51400e073f97fa6d8e26272cde98d8926ce27b68385` |
| `prices.parquet` | `4f030c49deea3dd536dcb4d06f3b41d8447492ebfae2370d3646bb615ce79615` |
| `event_surprise_company_day_signals.parquet` | `79b60edce36d69e6302d46c56143fd6a4180dea08fc3ed7d2253d377d2329ba6` |

## Frozen Event Scope

- Announcement window: `2022-01-01 <= announcement < 2026-01-01`.
- Development entries: through `2024-12-31`.
- Validation entries: calendar 2025 only.
- WRDS scope: quarterly USD EPS announcements for only the exact distinct
  tickers derived from the immutable article input, capped at 2,500 resulting
  event rows.
- Licensed and intermediate rows must be written only under ignored
  `data/private/` or `data/results/` paths.

## Frozen WRDS Contract

The only WRDS tables permitted for the expectation snapshot are
`ibes.actu_epsus`, `ibes.statsumu_epsus`, `wrdsapps_link_crsp_ibes.ibcrsphist`,
and `crsp.dsf_v2`. Links must be active on the announcement date with quality
score no greater than 1. The consensus snapshot is the latest `statpers`
strictly before `anndats`.

Raw I/B/E/S times are preserved as vendor fields and are not assigned UTC. The
actual EPS value is put on the consensus share basis with WRDS Method 3:

```text
actual_unadjusted * dlycumfacshr_at_statpers / dlycumfacshr_at_report_date
```

The benchmark subtracts mean estimate and records dispersion, contributor
count, revisions, and missingness.

## Frozen News Match

For each WRDS announcement, the matched news row is the first cached article for
the same ticker whose America/New_York calendar date is strictly after
`max(anndats, actdats)` and no more than two calendar days later. This
conservative next-day rule avoids undocumented raw I/B/E/S time semantics.

## Frozen Model And Evaluation

The primary target is five-session return residualized by the equal-weight
125-name universe return for the same entry date. Five-session outcomes reaching
a development/validation boundary are purged.

The fixed estimator is ridge regression with `alpha=1.0`, development-only
standardization and imputation, and seed `20260721`. Nested models are:
price-only, expectations/fundamentals-only, sector/calendar-only, cached
text-only, and combined.

The primary statistic is validation Spearman IC. The incremental statistic is
combined-minus-best-nontext validation Spearman IC with 2,000 date-block
bootstrap draws. The benchmark stops at coverage without fitting or evaluating
if it has fewer than 100 matched observations total, 40 validation
observations, or 20 distinct validation entry dates.

Portfolio output may run only if the existing stateful execution and cost
machinery can be reused without changing its economic assumptions. Otherwise
the report must present predictive results only and must not invent a new cost
model.
