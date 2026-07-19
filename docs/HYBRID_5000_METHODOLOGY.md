# Hybrid 5,000-Article Methodology

This document records the research design frozen before local-model prediction results or
holdout returns were inspected.

## Immutable inputs

- Prior OpenAI experiment: `20260718T232828Z_70aaf344` (calibration dataset v1, 250 rows).
- Frozen hybrid sample hash:
  `7b07079fb2bcbf7546e1dd810ee081ddb86adb7bb37aa0979efac31fe30553a7`.
- Frozen article Parquet SHA-256:
  `8ada422fcdefa894c55ae51400e073f97fa6d8e26272cde98d8926ce27b68385`.
- Chronological assignment hash:
  `0a96016b3398c9bb43fe30f088316af2604e4026d1c8bc92cfa6131f7f708b43`.
- Selected local model: `qwen3.6:35b-a3b`, Q4_K_M, non-thinking, temperature zero.
- Local prompt/schema: `hybrid_local_v1.1.0` / `local_article_assessment.v1`.

The source 250 OpenAI classifications and all associated artifacts remain unchanged. They are
never resubmitted to OpenAI.

## Return-blind sample construction

The candidate selector downloaded EODHD news and adjusted daily prices for 125 liquid US
companies across 11 sectors. It considered 123,113 ticker/article candidates and retained 45,770
after deterministic company-relevance, full-text, symbol, and complete-return checks. Selection
did not read realized returns.

Company relevance rewards target-company mentions in the headline and opening text, a direct
provider symbol match, low symbol count, adequate full text, and defined corporate-event terms.
It rejects incidental mentions, generic/listicle language, broad market summaries, short text,
and incomplete horizons. Exact bodies are removed before near-duplicate/syndicated clustering.
Only one primary story per cluster is eligible.

The frozen sample contains exactly:

- 5,000 unique article IDs, content hashes, and story-cluster IDs;
- 125 companies, 40 articles each (0.8% maximum ticker share);
- 11 sectors;
- 2022–2026 publication dates;
- 1,788 deterministic earnings/guidance candidates;
- complete 1-, 3-, 5-, 10-, 21-, and 63-session adjusted returns; and
- zero conservative-entry timestamps at or before publication.

Deterministic event candidates are hints to the local model, not forced final labels. Their very
low preclassification `other` rate is therefore not treated as model evidence.

## Chronological split and holdout lock

Articles are globally sorted by publication timestamp, ticker, and article ID, then assigned:

- development: first 3,000 rows, 2022-01-03 through 2024-12-31;
- validation: next 1,000 rows, 2024-12-31 through 2025-12-31; and
- untouched holdout: final 1,000 rows, 2025-12-31 through 2026-03-31.

The split freezer reads only identifiers and timestamps. Predictive and baseline runners refuse
holdout access until an immutable primary-specification manifest exists. Specification search is
limited to development and validation, records all 324 nearby configurations, and applies
Benjamini-Hochberg correction across the search.

## Local inference gates

Each result is cached by article content hash, ticker, model/quantization, prompt version, and
schema version. A run resumes from successful cached records and never counts an output until it
passes the strict schema.

The run stops for diagnosis if, after the initial 500 observations:

- invalid outputs exceed 2%;
- final `other` exceeds 40%;
- tradable coverage falls below 25%;
- one sentiment label exceeds 85%; or
- projected runtime exceeds seven days.

## Predictive inference

Event-return diagnostics explicitly account for overlapping horizons. Reported measures include
raw and weighted IC, signed returns, company-equal results, 1/99 winsorization, company-clustered
intervals, date-block bootstrap intervals, and company/date/two-way clustered regressions.
These event-level diagnostics are not portfolio return series and do not receive Sharpe ratios.

Signals are tested as raw sentiment and sentiment multiplied successively by confidence,
materiality, and novelty. Event-level, strongest-company-day, and company-day-aggregate forms are
compared. Primary horizons are 5 and 21 sessions; 1 day is retained as a timing falsification.

## Baselines and portfolio

Baselines use EODHD sentiment, fixed keyword sentiment, pre-entry 21-day momentum, pre-entry
5-day reversal, always-neutral, matched-random labels, shuffled ticker, shuffled timestamp, and a
development-fitted event-type signal.

Only after predictive analysis is complete is a daily portfolio constructed. It uses conservative
next-session entry, company-day aggregation, 2% maximum company weights, explicit 5- and 21-day
tests, base and conservative one-way cost schedules, and separate long-only and market-neutral
series. A new same-ticker signal arriving during an active holding period is suppressed, preventing
duplicate-story or overlapping-cohort multiplication. Sharpe is calculated only from these daily
marked series.

## Known limitations frozen before results

- The holdout is concentrated in the first quarter of 2026 and is not a multi-year holdout.
- The 125-company universe is fixed for this experiment and can retain survivorship bias.
- EODHD adjusted daily data cannot reproduce intraday reaction paths or quoted spreads.
- Deterministic event hints may anchor the local classifier; final event-label diagnostics must be
  checked against OpenAI calibration and returns.
- Local-model electricity cost uses sampled GPU power and a configured $0.25/kWh rate; it excludes
  embodied hardware and host-system power.
- If local inference is interrupted, the final cost ledger combines sampled telemetry from the
  resumed segment with a separately disclosed estimate for the interrupted segment. That estimate
  is observed model duration multiplied by the benchmark model's average GPU power.
- Transaction costs are scenario assumptions, not reconstructed historical order-book costs.
