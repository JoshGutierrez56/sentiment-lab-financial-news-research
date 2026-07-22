# Expectation-Adjusted News Overnight Exploratory Report

**Status:** exploratory_historical_locked_not_confirmatory
**Terminal state:** coverage_gate_unmet

## PM Summary

This historical exploratory run is not confirmatory and is not evidence of deployable alpha. The prior 2022-2026 news holdout was already viewed, so the only permissible decision is whether this design has enough coverage and clean enough plumbing to justify a separately frozen future study.

Portfolio output was not produced because the predeclared reuse gate requires existing stateful execution and cost machinery without changing economic assumptions.

## Coverage

```json
{
  "development_observations": 91,
  "distinct_validation_entry_dates": 9,
  "matched_observations_total": 100,
  "universe_tickers": 125,
  "validation_observations": 9,
  "wrds_event_rows": 1995
}
```

## Aggregate Metrics

```json
{
  "models_fit": false,
  "reason": "minimum matched-observation validation gate was not met",
  "status": "coverage_gate_unmet"
}
```

## Source Lineage

```json
{
  "config_sha256": "95f13760c7028dadb0698ff64ecdee2f642425d406abe741e12c7329cdb1bb36",
  "coverage_counts": {
    "development_observations": 91,
    "distinct_validation_entry_dates": 9,
    "matched_observations_total": 100,
    "universe_tickers": 125,
    "validation_observations": 9,
    "wrds_event_rows": 1995
  },
  "input_hashes": {
    "articles": "8ada422fcdefa894c55ae51400e073f97fa6d8e26272cde98d8926ce27b68385",
    "cached_text_signals": "79b60edce36d69e6302d46c56143fd6a4180dea08fc3ed7d2253d377d2329ba6",
    "prices": "4f030c49deea3dd536dcb4d06f3b41d8447492ebfae2370d3646bb615ce79615"
  },
  "ticker_list_sha256": "649112c580450cedd122e349349f1920fc1fea4f28cb1603c7970992d9ca60c3"
}
```

## Open-Source Context

- ProsusAI FinBERT is a finance-domain sentiment classification reference; classification success is separate from verified costed trading success: https://github.com/ProsusAI/finBERT
- AI4Finance FinGPT is a financial NLP model and benchmark ecosystem; it does not make this historical benchmark a deployed trading result: https://github.com/AI4Finance-Foundation/FinGPT
- Stefan Jansen's Machine Learning for Trading materials emphasize point-in-time validation, walk-forward testing, robustness, and costs; those standards are consistent with keeping this result exploratory: https://github.com/stefan-jansen/machine-learning-for-trading

## Limitations

- Historical exploratory benchmark only; the prior news holdout was already viewed.
- Raw I/B/E/S times are preserved without UTC assignment.
- Licensed WRDS rows and company-level EPS values remain only in ignored private artifacts.
- Portfolio evaluation was skipped unless the predeclared reuse gate could be met.
