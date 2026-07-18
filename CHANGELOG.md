# Changelog

## Unreleased

- Begin the OpenAI + EODHD rebuild on `agent/openai-eodhd-rebuild`.
- Add the repository audit and milestone-first implementation plan.
- Add modern Python packaging and strict configuration for the focused
  article-to-future-return vertical slice.
- Add immutable/redacted EODHD raw caching, validated native-type Parquet and
  DuckDB views, OpenAI Responses API structured classification, content-addressed
  assessment caching, conservative event alignment, accuracy/IC metrics,
  manifests, and an HTML evidence report.
- Validate a live EODHD AAPL sample and correct three defects found during the
  smoke run: Windows cache path length, credential-bearing HTTPX INFO logs, and
  string-typed temporal Parquet columns.
- Add 37 tests with more than 90% package coverage, including a deterministic
  mocked end-to-end rerun.
