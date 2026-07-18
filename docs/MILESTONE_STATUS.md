# Core milestone status

Status date: 2026-07-18

Branch: `agent/openai-eodhd-rebuild`

Research conclusion: **INCONCLUSIVE — live ChatGPT classification is blocked**

## What ran against the real provider

Command:

```powershell
uv run sentiment-lab data sync --config config/experiments/milestone.yaml --refresh
```

EODHD returned a candidate pool of 500 unique `AAPL.US` news records and 48
daily price rows. The deterministic date-diverse selector retained 12 unique
full-text articles:

- Snapshot ID: `4cc8af37fb51617d`
- Selected publication range: 2026-06-05 16:14:00 UTC through
  2026-06-15 06:31:07 UTC
- Distinct UTC publication dates: 11
- Maximum selected articles on one date: 2
- Non-empty article bodies: 12 of 12
- Article-body length: 171 minimum, 3,719 median, 6,992 maximum characters
- Exact article-provided `AAPL.US` mapping: 12 of 12
- Price range: 2026-05-01 through 2026-07-10
- Duplicate price dates: 0
- Article Parquet SHA-256:
  `c8299ce723aff6ad11d3f8bdd9ef5bb244a3f171e0d9aeb0d73a1f488ef4fb6a`
- Price Parquet SHA-256:
  `132bc48335976ecc05d09c52a02488eae491077c619e396d1bfd3dc5fd8098bf`

Provider-licensed article bodies and generated data files remain ignored by
Git. The hashes make the local snapshot identifiable without republishing it.

## Timing validation on the real snapshot

The real articles and prices were passed through the production alignment
engine with clearly labeled neutral fixture assessments solely to validate
timing; these fixtures were not saved as research results and are not ChatGPT
outputs.

- Entries after publication: 12 of 12
- Available conservative entries: 12 of 12
- Available adjusted 1-day returns: 12 of 12
- Available adjusted 3-day returns: 12 of 12
- Available adjusted 5-day returns: 12 of 12
- Distinct entry dates: 7
- Distinct 1-day realized returns: 7

Dedicated automated tests also cover Friday after-hours publication, weekend
roll-forward, same-day exclusion, adjusted-open scaling, missing horizons, and
article/classification timestamp mismatch rejection.

## Automated verification

```text
Ruff:       PASS
MyPy:      PASS (23 source files)
Pytest:    PASS (38 tests)
Coverage:  PASS (91.74%; gate 85%)
Python:    PASS on 3.11.15 and 3.12.10
```

The mocked integration test runs the full article → structured assessment →
future return → metrics/report pipeline twice. Articles, assessments, events,
and metrics have identical hashes; the second manifest records three cache hits
and zero new model tokens while preserving the original classification ledger.

## Security finding and remediation

The first live sync exposed a defect in console logging: HTTPX's INFO request
line included EODHD's query-parameter credential. No credential was written to
the repository or data cache, but it appeared in the private execution output.

Remediation completed:

1. HTTPX/HTTPCore request logging is forced to WARNING by the CLI.
2. The EODHD client emits its own sanitized endpoint/parameter log.
3. A fresh live request confirmed that only the sanitized log is rendered.
4. Every persisted file under `data/` was scanned against the runtime token;
   credential matches: zero.
5. A regression test asserts that request logs contain neither the token nor an
   `api_token` field.

Because the credential appeared in execution output, rotating the EODHD token
is still prudent before longer experiments.

## Exact blocker

`OPENAI_API_KEY` is absent. Running:

```powershell
uv run sentiment-lab milestone run --config config/experiments/milestone.yaml
```

therefore exits cleanly with:

```text
Error: OPENAI_API_KEY is required for real ChatGPT classification.
```

No real sentiment labels, confidence values, reasoning, token costs, accuracy,
or IC values have been fabricated. To unblock, set both `OPENAI_API_KEY` and a
current Structured-Outputs-capable `OPENAI_MODEL` in the untracked `.env`, then
run the command above once. A completed run will write `articles.parquet`,
`assessments.parquet`, `events.parquet`, `metrics.json`, `manifest.json`, and
`report.html`. Repeating it without `--refresh` or `--force-classify` performs
the cached reproducibility run.
