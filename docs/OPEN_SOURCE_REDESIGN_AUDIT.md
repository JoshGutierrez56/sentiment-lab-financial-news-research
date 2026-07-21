# Open-Source Redesign Audit

## Scope and method

This audit informs an independently implemented, cache-first event-surprise research
pipeline. It does not copy source code. Repositories were inspected at the immutable
commits below on 2026-07-19. No framework is added as a runtime dependency unless a
future written benchmark demonstrates that it is preferable to the small local component.

| Source | Commit inspected | License | Relevant pattern | Use in this repository | Copy status and failure addressed |
|---|---|---|---|---|---|
| [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) | `44995e0c5870c4ab37a189d756550654ae87cdf0` | Apache-2.0 | `finbert/finbert.py` shows a finance-domain classifier layered over Hugging Face tokenization and batched evaluation. | `src/sentiment_lab/baselines/finbert.py` | Independently reimplemented with the current Hugging Face auto classes and content-addressed cache. It supplies a cheap, fixed financial-text comparator rather than treating a permissive LLM as ground truth. |
| [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | `e5e516470e7a25ed3690889b7b54d9946dd17520` | MIT | `fingpt/FinGPT_Sentiment_Analysis_v3/README.md` documents instruction-formatted finance sentiment inference and separates base and adapter identifiers. | `src/sentiment_lab/baselines/finance_local.py` | Independently reimplemented as a local structured-model adapter contract; no LoRA/QLoRA or FinGPT code/model is copied. This tests whether a finance-specialized local model contributes information rather than merely agreeing with OpenAI. |
| [microsoft/qlib](https://github.com/microsoft/qlib) | `d5379c520f66a39953bad76234a7019a72796fd0` | MIT | `docs/component/workflow.rst` defines config-driven data, model, inference, evaluation, backtest, and recorded artifacts. | `src/sentiment_lab/redesign/experiment.py`, `config/experiments/event_surprise_cache_only.yaml` | Independently reimplemented lightweight run manifest/artifact contract. It addresses the prior run's artifact-hash mismatch and makes cache-only scientific outputs distinct from operational timing. |
| [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | `0269115d3cfbf691c7a0b7cfcc9ed412cafb91f6` | Apache-2.0 | `Common/Orders/Slippage/VolumeShareSlippageModel.cs` uses capped order-volume share and quadratic price impact; order/fill separation is pervasive. | `src/sentiment_lab/execution/costs.py`, `src/sentiment_lab/execution/engine.py` | Independently reimplemented the stated volume-share equation and stateful target-minus-current order logic. It replaces the old undifferentiated 10-bps cost bucket and full daily recreation. |
| [waylonli/FINSABER](https://github.com/waylonli/FINSABER) | `59fa1dfbf9ed00ad65c786475a0135656332b2c3` | Apache-2.0 | `backtest/finsaber_bt.py` demonstrates rolling-window execution driven by a trade configuration and persisted artifacts. | `src/sentiment_lab/validation/purged_cv.py`, `src/sentiment_lab/redesign/experiment.py` | Independently reimplemented chronological fold and run-artifact boundaries. This prevents the old study's one-shot broad-signal workflow from being mistaken for a robust research loop. |
| [sam31415/timeseriescv](https://github.com/sam31415/timeseriescv) | `cb04fb6ea7a0b2c15920ca253f882336fe336ba8` | MIT | `README.rst` documents `PurgedWalkForwardCV` and embargoing based on prediction and evaluation timestamps. | `src/sentiment_lab/validation/purged_cv.py` | Independently reimplemented, with no code copied. It removes outcome-window overlap from train/validation/test and explicitly ties embargo to the longest horizon. |
| [lambdaclass/options_portfolio_backtester](https://github.com/lambdaclass/options_portfolio_backtester) | `e53ef86928777de6ee0721424762ea3dc133f993` | MIT | `options_portfolio_backtester/engine/engine.py` composes data, strategy, execution, portfolio, risk, and analytics rather than conflating them. | `src/sentiment_lab/execution/engine.py`, `src/sentiment_lab/execution/costs.py` | Independently reimplemented a minimal equities-only composition. No options code or dependency is used. This supports auditable rejected orders, fills, inventory, and component costs. |
| [anthonymakarewicz/volatility-trading](https://github.com/anthonymakarewicz/volatility-trading) | `a866b6eee79d9a0c0bf81a1b82208dc6da6aae66` | MIT | `examples/backtesting/execution/models_and_costs.py` compares explicit execution scenarios rather than burying costs in performance. | `src/sentiment_lab/execution/costs.py`, `src/sentiment_lab/redesign/regime.py` | Independently reimplemented component cost reporting and a limited trend/realized-volatility risk scaler. It addresses the old gross-to-net collapse and avoids reversing labels based on regime. |

## Dependency decision

Qlib, LEAN, FINSABER, `timeseriescv`, the options backtester, and the volatility-trading
project are **not** runtime dependencies. Their relevant patterns are smaller than the
integration surface required to preserve the repository's immutable artifacts, Polars data
model, and local execution assumptions. The implementation remains local and testable.

The FinBERT adapter uses the supported Hugging Face `transformers` interface when its
model and tokenizer are available locally. It does not reuse the legacy FinBERT training
implementation. Any downloaded model revision is recorded in the model metadata and cache
key.
