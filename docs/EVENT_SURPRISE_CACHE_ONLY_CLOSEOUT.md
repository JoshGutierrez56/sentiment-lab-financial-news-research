# Event-surprise cache-only engineering closeout

Status date: 2026-07-19  
Branch: `agent/event-surprise-redesign`

## Scope and immutability

This branch adds redesign infrastructure only.  It made no OpenAI request, did
not run a local language model, did not collect or classify an article, did
not modify the completed hybrid-5,000 artifacts, and did not create a fresh
sample.  The frozen 5,000 study remains development evidence only.

## Implemented engineering controls

- `baselines/finbert.py` is a deterministic batch adapter for the current
  Hugging Face `ProsusAI/finbert` model.  Its cache key includes article,
  selected text, model revision, and mode; it persists all requested
  probabilities, score, revisions, text hash, timestamp, and duration.
  Cache-only operation fails explicitly when a prediction is missing.
- `baselines/finance_local.py` defines the cache-only finance-local benchmark
  contract and reports agreement, IC, signed return, event agreement,
  structured-output validity, runtime, GPU memory, and cost where cached
  fields are available.
- `event_surprise/` supplies the controlled taxonomy, mandatory factual
  comparison fields, sparse abstention rules, development-only normalization,
  residual/disagreement signals, and one strongest event per company-day.
- `validation/purged_cv.py` prevents any train observation whose outcome
  window overlaps a validation window; folds record the embargo endpoint.
- `execution/` decomposes commissions, half spread, slippage, volume impact,
  borrow, locate availability, and research cost.  The stateful engine trades
  target minus inventory, nets positions, rejects unavailable locates, records
  partial fills, and requires a fixed 2.0x estimated-edge/cost buffer.
- `redesign/experiment.py` writes separate canonical scientific artifacts and
  a non-canonical operational artifact.  It refuses to overwrite a run and
  records configuration/data/model/hash/provenance metadata.

## Cache-only finding

The local Hugging Face cache has no `ProsusAI/finbert` weights or predictions.
There is likewise no cached finance-specialized model output nor strict
event-surprise extraction cache for the frozen articles.  Downloading a model
or running either model now would be new classification work, prohibited by
the cache-only directive.  Consequently no FinBERT IC, residual IC,
event-surprise IC, redesigned turnover, or redesigned net return was
calculated.  Reporting those values would be invented.

## Gate decision

**Engineering decision: PASS.** The code and tests pass.  The missing local
model/output caches are expected input-generation work for the authorized
local-only retrospective, not an engineering failure.

**Research decision: INCONCLUSIVE.** The 5,000-article broad-sentiment result
does not validate this distinct event-surprise hypothesis, and no redesigned
retrospective result exists yet.

Recommendation: **PROCEED TO BOUNDED LOCAL-ONLY RETROSPECTIVE.** No
preregistration document or fresh-2,000 study was created.  A fresh study
remains conditional on the retrospective gates.
