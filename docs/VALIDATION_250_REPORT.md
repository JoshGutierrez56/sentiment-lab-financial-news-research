# Bounded 250-Article Validation Report

## Decision

**REVISE** — the pipeline works and medium-horizon evidence exists, but sampling should be
made more company-specific before paying for 1,000 classifications. The final sample has a
54.4% abstention rate and 175 of 250 articles were assigned the model event type `other`.
No larger run was started.

Experiment: `20260718T232828Z_70aaf344`

Frozen snapshot: `b2ab96dc53588be1`

## Phase 1: cached 12-article audit

No article was sent to OpenAI again for this audit. The exact table, including full abstain
reasons and unrounded returns, is stored in
`data/results/20260718T221911Z_3146948f/classification_audit.csv`.

| Article | Headline (shortened) | Published UTC | Mini | Final | Score | Conf. | Mat. | Nov. | Trade / abstain | Abstention category | Escalation / changed | 1d | 3d | 5d |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| `d5ecc63f` | SpaceX and S&P 500 | 2026-06-05 16:30 | neutral | neutral | 0.00 | .99 | .00 | .00 | no / yes | Correctly irrelevant | contradiction / no | -2.33% | -5.56% | -5.70% |
| `31b8f169` | Dell best stock | 2026-06-05 16:48 | neutral | bearish | -.26 | .74 | .27 | .62 | yes / no | — | contradiction / yes | -2.33% | -5.56% | -5.70% |
| `0ce98ca3` | Broad market history | 2026-06-06 04:20 | neutral | neutral | 0.00 | .96 | .02 | .01 | no / yes | Correctly irrelevant | none / no | -2.33% | -5.56% | -5.70% |
| `9c55a97b` | Buffett portfolio moves | 2026-06-07 06:20 | invalid | neutral | 0.00 | .96 | .08 | .10 | no / yes | Duplicate or stale | validation / yes | -2.33% | -5.56% | -5.70% |
| `e83c3490` | VOO vs SPY | 2026-06-08 04:20 | invalid | neutral | 0.00 | .98 | .00 | .00 | no / yes | Correctly irrelevant | validation / yes | -3.24% | -1.55% | -1.29% |
| `de156eb1` | Nedap / Albert Heijn | 2026-06-09 05:00 | neutral | neutral | 0.00 | .99 | .00 | .00 | no / yes | Correctly irrelevant | none / no | +0.29% | +0.13% | +2.92% |
| `7b70d09c` | TSMC May revenue | 2026-06-10 07:37 | invalid | neutral | 0.00 | .95 | .12 | .45 | no / yes | Insufficient company info | validation / yes | +0.65% | +0.92% | +0.76% |
| `4d8aa5c8` | ARKQ vs QQQ | 2026-06-11 04:20 | neutral | neutral | 0.00 | .98 | .01 | .01 | no / yes | Correctly irrelevant | none / no | -1.66% | +1.08% | +0.67% |
| `0a73e554` | Buffett wide-moat stocks | 2026-06-12 08:23 | neutral | neutral | 0.00 | .95 | .05 | .02 | no / yes | Duplicate or stale | none / no | +0.78% | +0.62% | +0.98% |
| `03705950` | Tata iPhone-parts pollution | 2026-06-13 08:33 | bearish | bearish | -.32 | .83 | .42 | .79 | yes / no | — | contradiction / no | +0.78% | +0.62% | +0.98% |
| `5b67a8ef` | Stocks instead of SpaceX | 2026-06-14 08:05 | neutral | neutral | 0.00 | .99 | .00 | .00 | no / yes | Correctly irrelevant | contradiction / yes | +0.78% | +0.62% | +0.98% |
| `24f62b0c` | France digital tax | 2026-06-15 06:31 | neutral | neutral | 0.00 | .82 | .18 | .20 | yes / no | — | none / no | +1.35% | +0.93% | -0.32% |

The nine abstentions comprise six correctly irrelevant stories, two duplicate/stale opinion
pieces, and one supplier story without enough Apple-specific information. None was judged
potentially tradable but blocked by an overly conservative prompt. The 75% smoke-test
abstention rate was therefore caused by source mix.

## Phase 2: escalation correction

- Contradiction no longer escalates an abstention or zero-materiality result.
- Neutral low-materiality results do not escalate because of low confidence.
- Substantive ambiguity requires relevance at least 0.50 and materiality at least 0.25.
- Structured-output validation failures still escalate automatically.
- The cache-key material, prompt version, schema version, and deterministic cache behavior
  were unchanged.

Ruff, strict MyPy, and 51 tests pass. Coverage is 88.46%.

## Phase 3: frozen sample and cost

- 5,000 EODHD candidates considered; 250 selected and frozen.
- 25 liquid US equities, exactly 10 articles per company, 11 sectors, five publication
  months, and all nine sampling event buckets.
- 250 full-text and zero headline-only records.
- Excluded before OpenAI: 256 headline-only/low-text, 93 duplicate bodies, 36 direct-symbol
  mapping failures, and 4,365 beyond the balanced sample limit.
- All selected records have complete adjusted 1-, 3-, 5-, and 21-day returns.
- Largest individual Batch preflight: $0.892928; all submitted attempts' conservative
  preflight maxima total $1.505776, below the $2 hard limit.
- Usage reconciliation: 480,648 input tokens, 8,960 cached input tokens, 36,997 output
  tokens, 306 request attempts, and 17 malformed structured-output attempts repaired from
  cache-resumable batches.
- Usage-derived OpenAI cost: **$0.348443**, or **$0.001394 per selected article**.

## Phase 4: classification and return evidence

- Valid final classifications: 250; all 250 first-pass outputs are cached.
- Labels: 47 bullish, 187 neutral, and 16 bearish.
- Abstention: 54.4%; tradable coverage: 45.6% (114 events).
- Unique articles ever escalated: 39 (15.6%); all received a valid cached resolution.
- The expensive-model assessment remained the final record for 24 articles. It changed the
  label/tradable/abstain conclusion for 25% of those 24.

| Horizon | Directional accuracy | Pearson IC | Spearman IC | Weighted Pearson | Weighted Spearman | Bull − bear mean spread | Company-equal signed return | Company-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d | 32.46% | .005 | .020 | -.006 | .034 | +0.11% | +0.10% | [-0.11%, +0.33%] |
| 3d | 28.07% | -.029 | .031 | -.028 | .043 | -0.15% | +0.27% | [-0.16%, +0.74%] |
| 5d | 33.33% | .122 | .152 | .133 | .166 | +1.16% | +0.68% | [+0.06%, +1.28%] |
| 21d | 36.84% | .209 | .169 | .195 | .164 | +5.30% | +2.26% | [+0.69%, +4.06%] |

The company-cluster IC intervals still include zero at every horizon. The 21-day
bullish-minus-bearish spread interval is positive ([+0.43%, +9.69%]); its 5-day interval
crosses zero. Winsorization at the 1st/99th percentiles leaves 5-day Pearson/Spearman IC at
.121/.152 and 21-day IC at .204/.169.

Average / median returns for tradable records:

| Horizon | Bullish | Neutral | Bearish |
|---|---:|---:|---:|
| 1d | +0.23% / +0.45% | +0.09% / +0.17% | +0.12% / +0.65% |
| 3d | +0.85% / +0.74% | +0.58% / +1.07% | +1.00% / +0.85% |
| 5d | +1.22% / +0.84% | +0.49% / +0.40% | +0.06% / +0.49% |
| 21d | +5.05% / +4.06% | +3.62% / +3.91% | -0.25% / -1.31% |

Full ticker and model-event-type tables are in `metrics.json` and `report.html`. The
strongest 21-day company-equal signed results were GE, HD, and JNJ; the weakest were XOM,
TSLA, and WMT. Of the model event types, `other` dominated (175 records, 126 abstentions),
which is the principal reason to revise sampling before expansion.

## Data-quality limitations

- Direct EODHD symbol membership is treated as mapping confidence 1.0; multi-company impact
  ambiguity is left to model abstention.
- The named current liquid-equity universe does not eliminate survivorship bias.
- Sampling event buckets are keyword strata, not human labels; their mismatch with the
  model's 175 `other` assignments shows that the sampler needs a company-specific event gate.
- Exact body hashing cannot remove every syndicated rewrite.
- There is no independent human sentiment label set yet.
- Returns overlap, are not market- or factor-adjusted, and can contain common market drift.
  The bootstrap clusters by company, not time. These are event associations, not a
  non-overlapping portfolio series, so no Sharpe ratio is reported.

## Reproduction

```powershell
uv run sentiment-lab validation sync --config config/experiments/validation_250.yaml
uv run sentiment-lab validation run --config config/experiments/validation_250.yaml
uv run ruff check src tests
uv run mypy src/sentiment_lab
uv run pytest
```
