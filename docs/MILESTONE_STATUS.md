# Core milestone status

Status date: 2026-07-18

Branch: `agent/openai-eodhd-rebuild`

Research conclusion: **INCONCLUSIVE — live OpenAI classification is blocked**

## Current state

The cost-optimized EODHD-to-OpenAI-to-return implementation is complete and
validated with deterministic mocked Batch responses. The existing real
12-article provider sample was rerun from cache, but no OpenAI request was made
because the local runtime has no `OPENAI_API_KEY`.

The live-data selection command was:

```powershell
uv run sentiment-lab data sync --config config/experiments/milestone.yaml
```

It produced snapshot `0892e81701aab621`:

- 500 articles considered.
- 488 filtered before OpenAI.
- 33 inadequate-text articles skipped.
- 33 broad market summaries skipped.
- Four duplicate stories skipped.
- 418 otherwise eligible records excluded by the 12-article smoke cap.
- Zero out-of-window or low-confidence direct ticker mappings.
- 12 full-text articles selected; zero headline-only articles.
- Publication range: 2026-06-05 16:30:12 UTC through 2026-06-15 06:31:07 UTC.
- 11 distinct UTC publication dates.
- Normalized article bodies: 2,019 minimum and 6,958 maximum characters.
- 48 adjusted daily price rows.

Provider-licensed text and generated data remain ignored by Git.

## Timing validation

Neutral fixture assessments were used only to exercise the alignment engine;
they were not stored or represented as ChatGPT research results.

- Entries strictly after publication: 12 of 12.
- Available adjusted 1-day returns: 12 of 12.
- Available adjusted 3-day returns: 12 of 12.
- Available adjusted 5-day returns: 12 of 12.
- Distinct conservative entry dates: seven.

## OpenAI cost controls

- First pass: `gpt-5.4-mini` through Batch `/v1/responses`.
- Selective escalation: `gpt-5.4` only after the complete first pass.
- Pro models: prohibited.
- Strict output: the 11 requested fields only; reasoning is at most 40 words.
- Output cap: 256 tokens for either model.
- Permanent key: content hash + ticker + model + prompt version + schema
  version.
- Smoke/first-sample/expanded hard limits: $1/$5/$20.
- Batch state is resumable and output is joined by `custom_id`, not file order.

The conservative pre-submit ceiling for this 12-article snapshot is:

| Stage | Maximum estimate |
|---|---:|
| All 12 mini first passes | $0.04404225 |
| All 12 expensive escalations | $0.14673250 |
| Combined worst case | $0.19077475 |
| Smoke limit | $1.00000000 |

No upload occurred during this estimate.

## Automated verification

```text
Ruff format: PASS
Ruff lint:   PASS
MyPy:       PASS (23 source files)
Pytest:     PASS (41 tests)
Coverage:   PASS (90.16%; gate 85%)
```

Tests cover JSONL request shape, strict output fields, unordered Batch output,
remote-batch resume without re-upload, conservative preflight rejection before
upload, exact Batch pricing with cached tokens, permanent cache hits, within-run
deduplication, all escalation triggers, filter accounting, full-text priority,
timestamp leakage, adjusted returns, report/manifests, and deterministic reruns.

## Exact blocker and next command

The command:

```powershell
uv run sentiment-lab milestone run --config config/experiments/milestone.yaml
```

stops cleanly before provider or OpenAI work with:

```text
Error: OPENAI_API_KEY is required for real ChatGPT classification. Cached downloads and mocked tests do not require it.
```

Configure the key locally in the untracked `.env`—do not paste it into chat—and
rerun that command. A complete run will report articles considered/filtered,
mini classifications, escalations, cache hits, exact token totals, total and
average cost, correctly aligned returns, accuracy, and IC.

The next size is 100 articles under the $5 tier, but it must not run until the
12-article classifications are inspected for schema quality, abstention,
relevance, ticker specificity, escalation behavior, and return alignment.

## Security note

An earlier EODHD smoke request exposed its query token in private HTTPX INFO
output. Transport logging was subsequently forced to WARNING, sanitized client
logging was added, and persisted data was scanned with zero token matches.
Rotating that EODHD token remains prudent.
