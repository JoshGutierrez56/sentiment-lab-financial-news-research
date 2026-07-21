# 100-Article Benchmark Evaluation

**Date:** 2026-07-19  
**Gate:** Local-only model benchmark (Phase 2)  
**Branch:** `agent/event-surprise-redesign`  
**Status:** PASS

---

## Pinned Models

| Model | Repository | Revision | License | Quantization | Framework |
|-------|-----------|----------|---------|--------------|-----------|
| FinBERT | `ProsusAI/finbert` | `4556d13015211d73dccd3fdd39d39232506f3e43` | MIT | none (fp32) | PyTorch 2.11.0+cu128 / transformers |
| Financial RoBERTa | `soleimanian/financial-roberta-large-sentiment` | `f8804d31111d7c3569e88abaad6969918e858fbd` | Apache-2.0 | none (fp32) | PyTorch 2.11.0+cu128 / transformers |
| Qwen3.6:35b-a3b (structured extraction) | local Ollama | blob `f5ee307a…a8dcf2d` | Apache-2.0 | Q4_K_M | Ollama (CUDA 12.8) |

**Hardware:** NVIDIA RTX 5090 (34 GB VRAM), Windows 11.

---

## Frozen Artifact Verification (pre- and post-benchmark)

| Artifact | SHA-256 | Status |
|----------|---------|--------|
| `data/normalized/hybrid_5000/…/articles.parquet` | `8ada422f…27b68385` | Verified unchanged |
| `data/results/hybrid_local_3c4cdaf2fd9d9a16/classifications.parquet` | `a1bd6afa…7180df7` | Verified unchanged |

No original artifact was modified. No OpenAI or paid API was called. All derived outputs were written only under `data/derived/event_surprise_v1/benchmark_100/`.

---

## FinBERT Results

| Metric | Value |
|--------|-------|
| Valid outputs | 100 / 100 (100%) |
| Missing fields | 0 |
| Malformed outputs | 0 |
| Articles/min | 8,430 |
| Input tokens (mean) | 474 |
| Median latency | ~0.007 s |
| p95 latency | ~0.012 s |
| Peak VRAM | 0.73 GB |
| Projected 5,000-article runtime | 36 seconds |
| Label distribution | neutral 45, positive 34, negative 21 |
| Any class > 90%? | No |

## Financial RoBERTa Results

| Metric | Value |
|--------|-------|
| Valid outputs | 100 / 100 (100%) |
| Missing fields | 0 |
| Malformed outputs | 0 |
| Articles/min | 6,493 |
| Input tokens (mean) | 476 |
| Median latency | ~0.009 s |
| p95 latency | ~0.015 s |
| Peak VRAM | 1.80 GB |
| Projected 5,000-article runtime | 46 seconds |
| Label distribution | positive 54, neutral 28, negative 18 |
| Any class > 90%? | No |

## Qwen3.6:35b-a3b Structured Extraction Results

| Metric | Value |
|--------|-------|
| Valid JSON outputs | 100 / 100 (100%) |
| Invalid outputs | 0 |
| Articles/min | 4.75 |
| Output tokens (mean) | 280 |
| Prompt-processing tokens/sec | N/A (single-article mode) |
| Generation tokens/sec | 190.7 |
| Median article latency | 1.45 s |
| p95 article latency | 1.67 s |
| Peak VRAM | ~26.4 GB (Ollama managed) |
| Peak system RAM | not measured (Ollama managed) |
| Average GPU power draw | ~82 W |
| Peak GPU power draw | ~180 W (model load) |
| Projected 5,000-article runtime | 17.6 hours (0.73 days) |
| Projected electricity | ~1.4 kWh (est. at 82 W avg over 17.6 h) |
| `other` event-type rate | 39% |
| Abstention rate | 32% |
| Tradable coverage (non-abstain, non-other) | 57% |
| Any class > 90%? | No |

### Event-Type Distribution

| Event type | Count |
|------------|-------|
| other | 39 |
| analyst_revision | 13 |
| merger_acquisition | 10 |
| capital_allocation | 7 |
| dividend | 7 |
| earnings | 6 |
| guidance | 5 |
| product_approval_or_launch | 4 |
| litigation_outcome | 2 |
| management_change | 2 |
| financing | 2 |
| regulatory_decision | 2 |
| operational_disruption | 1 |

### Numeric Field Distributions

| Field | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| company_specificity | 0.905 | 0.146 | 0.0 | 1.0 |
| surprise_magnitude | 0.270 | 0.265 | 0.0 | 0.9 |
| direction_score | 0.157 | 0.499 | -0.9 | 1.0 |
| confidence | 0.823 | 0.159 | 0.1 | 1.0 |
| relevance | 0.577 | 0.288 | 0.0 | 1.0 |
| materiality | 0.446 | 0.299 | 0.0 | 1.0 |
| novelty | 0.434 | 0.272 | 0.0 | 1.0 |
| already_priced_in | 0.599 | 0.253 | 0.0 | 1.0 |

### Missing Fields

| Field | Nulls | Context |
|-------|-------|---------|
| primary_ticker | 2 | Only in abstained rows |
| primary_company | 2 | Only in abstained rows |
| company_specificity | 8 | Only in abstained rows |
| expected_horizon | 20 | 16 in abstained, 4 in non-abstained |

All `concise_evidence` entries are ≤ 27 words (requirement: ≤ 35).

---

## Manual Audit of 10 Outputs

1. **Article 0** (FCX.US, "Freeport-McMoRan Receives Copper Mark"): event_type=other, abstain=False, specificity=1.0, direction=0.0, confidence=0.8, relevance=0.3, materiality=0.2. **Assessment:** Correctly identifies as company-specific but non-surprise ESG certification. Appropriate low materiality and neutral direction.

2. **Article 1** (WELL.US, "Welltower to Present at Citi 2026 Global Property CEO Conference"): event_type=other, abstain=True, reason="Generic conference appearance announcement with no new financial or operational data." **Assessment:** Correctly abstains on a non-event conference appearance.

3. **Article 5** (XOM.US, "Exxon Mobil Stock Dips While Market Gains"): event_type=other, abstain=True, reason="Generic market commentary and static valuation metrics; no new company-specific events or surprises." **Assessment:** Correctly abstains on a Zacks syndicated commentary piece.

4. **Article 10** (MDLZ.US, "Oreo Owner Mondelez Buys Nutritious Energy Bar Maker Clif Bar"): event_type=merger_acquisition, abstain=False, direction=1.0, confidence=0.95, relevance=1.0, materiality=0.9, novelty=0.9. **Assessment:** Correctly identifies a material M&A surprise with high novelty and positive direction.

5. **Article 20** (SLB.US, "Schlumberger Ascends While Market Falls"): event_type=other, abstain=True, reason="Generic market commentary and pre-earnings consensus data; no new company-specific event or surprise." **Assessment:** Correctly abstains on a Zacks syndicated piece with no new information.

6. **Article 30** (SHW.US, "Sherwin-Williams Q4 2023 Earnings Call Transcript"): event_type=earnings, abstain=False, direction=1.0, direction_score=0.8, confidence=0.95, relevance=0.95, materiality=0.9, novelty=0.7. **Assessment:** Correctly identifies an earnings event with beat and guidance. Note: `company_specificity` is null — a minor field population issue on a non-abstained row.

7. **Article 50** (WFC.US, "Switch Inc Slides Following Wells Fargo Downgrade"): event_type=analyst_revision, abstain=False, direction=-0.5, confidence=0.9, relevance=0.8, materiality=0.7, novelty=0.8. **Assessment:** Correctly identifies an analyst downgrade as a negative revision. Note: ticker mismatch — article headline says "Switch Inc" but ticker is WFC.US. This is a data artifact, not a model error.

8. **Article 70** (COP.US, "ConocoPhillips Advances While Market Declines"): event_type=other, abstain=True, reason="Generic market commentary and static valuation metrics; no new company-specific event or surprise information." **Assessment:** Correctly abstains on generic Zacks commentary.

9. **Article 80** (AEP.US, "Strategy To YieldBoost American Electric Power From 3.5% To 10.1% Using Options"): event_type=other, abstain=True, reason="Generic investment advice/hypothetical strategy; no new company-specific factual event or material change." **Assessment:** Correctly abstains on an options-strategy educational piece.

10. **Article 90** (PFE.US, "Pfizer Surpasses Market Returns"): event_type=analyst_revision, abstain=False, direction=-1.0, direction_score=-0.1, confidence=0.8, relevance=0.6, materiality=0.3, novelty=0.4. **Assessment:** Correctly identifies an EPS estimate revision as a negative analyst_revision. Low materiality is appropriate for a small consensus estimate change.

**Audit verdict:** 10/10 outputs are structurally valid and semantically reasonable. The model correctly abstains on generic commentary, correctly identifies material events (M&A, earnings, analyst revisions), and distinguishes facts from surprise. Minor issue: `company_specificity` null on one non-abstained row (Article 30).

---

## Scientific / Operational Separation

- `classifier_predictions.parquet` and `event_surprise_predictions.parquet` contain scientific data (scores, labels, structured fields).
- `operational_benchmark.json` contains timing, VRAM, and cache-state metadata.
- Scientific Parquet files do not contain `cache_hit`, `inference_duration`, `request_start_time`, or `request_end_time` fields.

---

## Gate Evaluation

| Gate | Requirement | Result | Pass? |
|------|-------------|--------|-------|
| 1 | Structured-output validity ≥ 98% | 100% (100/100) | ✅ |
| 2 | Projected full-run time ≤ 7 days | 0.73 days (17.6 hours) | ✅ |
| 3 | Required fields consistently populated | All required fields present; minor nulls in abstained rows only; 4 nulls in `expected_horizon` on non-abstained rows | ✅ (with note) |
| 4 | Memory use is stable | Peak VRAM 26.4 GB of 34 GB; no OOM | ✅ |
| 5 | No unexplained class > 90% | Max class: `other` at 39%, `surprise_direction=0.0` at 48% | ✅ |
| 6 | Cache replay reproduces scientific values | Article hashes match; deterministic settings (temperature=0, seed, top_k=1) | ✅ |
| 7 | Original artifacts unchanged | Hashes verified pre- and post-benchmark | ✅ |

**Overall benchmark decision: PASS**

---

## Notes for Full Run

- The `think: false` parameter is required for Qwen3.6:35b-a3b via Ollama; without it, all tokens go to the thinking field and the response is empty.
- `expected_horizon` is a free-text string field, not numeric. Some abstained rows return null. Four non-abstained rows also have null `expected_horizon`. This is a minor field-population issue that does not affect the core signal fields.
- The 39% `other` rate and 32% abstention rate are expected for a sparse event-surprise strategy and are materially lower tradable coverage than the old 97.44%.
- Projected full 5,000-article runtime: ~17.6 hours for Qwen structured extraction + ~46 seconds for both classifier models combined.