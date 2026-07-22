# Expectation-Adjusted Overnight Exploratory Protocol Freeze

**Status:** exploratory_historical_locked_not_confirmatory  
**Created:** 2026-07-21 21:04 America/New_York  
**Evidence label:** locked historical exploratory benchmark, not prospective, not confirmatory, not deployable alpha

This note freezes the overnight exploratory run before any return, IC, Sharpe,
portfolio, or model-result calculation. The prior 2022-2026 news holdout has
already been viewed and cannot support untouched, prospective, promoted,
successful, production-ready, or confirmatory claims.

The immutable article input hashes to
`8ada422fcdefa894c55ae51400e073f97fa6d8e26272cde98d8926ce27b68385`.
The exact 125-ticker universe is derived only from distinct uppercase tickers
in that input and hashes to
`649112c580450cedd122e349349f1920fc1fea4f28cb1603c7970992d9ca60c3`.
The cached text-signal input hashes to
`79b60edce36d69e6302d46c56143fd6a4180dea08fc3ed7d2253d377d2329ba6`.

The WRDS scope is bounded to those tickers, 2022-01-01 through 2025-12-31
announcements, quarterly USD EPS, I/B/E/S unadjusted actual and summary EPS,
CRSP Method-3 split-basis factors, link score no greater than 1, and at most
2,500 resulting event rows. Raw I/B/E/S times are preserved without assigning
UTC. Consensus is the latest `statsumu_epsus.statpers` strictly before
`actu_epsus.anndats`.

News is matched deterministically: for the same ticker, select the first cached
article whose America/New_York calendar date is strictly after
`max(anndats, actdats)` and no more than two calendar days later. Development
entries end on 2024-12-31; validation entries are calendar 2025 only; five-
session outcomes that reach a split boundary are purged.

The fixed model is ridge regression with alpha 1.0 after development-only
standardization and imputation, seed 20260721. Nested families are price-only,
expectations/fundamentals-only, sector/calendar-only, cached text-only, and
combined. The primary target is the five-session return residualized by the
equal-weight 125-name universe return for the same entry date. The primary
statistics are validation Spearman IC and combined-minus-best-nontext IC using
2,000 date-block bootstrap draws.

If coverage is below 100 matched observations total, 40 validation
observations, or 20 distinct validation entry dates, the run stops at coverage
and does not fit or evaluate models. Portfolio output is secondary and may run
only if existing stateful execution and cost machinery can be reused without
changing its economic assumptions.
