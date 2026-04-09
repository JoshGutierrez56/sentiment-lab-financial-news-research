# News-Sentiment Algorithmic Trading System
### FactSet RTNews + DeepSeek Chat | Event-Driven Backtesting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

An algorithmic trading pipeline that ingests news headlines via the FactSet RTNews API, classifies them as bullish/bearish/ambiguous using the DeepSeek Chat LLM, converts the classifications into directional trading signals, and backtests those signals against historical price data.

---

## Bug Fixes (vs. original)

13 bugs were identified and corrected in the rewrite:

| # | Bug | Fix |
|---|-----|-----|
| 1 | API keys hardcoded in `CONFIG` dict | `load_settings()` reads from env vars only |
| 2 | Wrong model name `'deepseek-llm'` | Corrected to `'deepseek-chat'` |
| 3 | `max_drawdown` formula inverted (returned max run-up) | `(nav/peak - 1).min()` — correct negative value |
| 4 | Sharpe annualised with flat `×252` on non-daily event returns | Annualises by observed signals-per-year from date range |
| 5 | Cumulative return used `cumsum()` (arithmetic) | Fixed to `(1+r).cumprod()` (geometric) |
| 6 | `pytest` fixtures in the main production module | Moved entirely to `tests/` |
| 7 | No retry logic — single `raise_for_status()` | `HTTPAdapter` with exponential backoff on 429/5xx |
| 8 | Token rate limiter used a simple counter with no rolling window | Sliding-window `deque` of `(timestamp, tokens)` pairs |
| 9 | `datetime.min` sentinel | `Optional[float] = None` with `time.monotonic()` |
| 10 | `apply(result_type='expand')` with no guaranteed 2-tuple contract | Explicit return type in all error paths |
| 11 | Chained `.loc` on MultiIndex — silent `KeyError` risk | `pd.IndexSlice` + proper `try/except` |
| 12 | Entire system in one flat file | Proper package structure with isolated modules |
| 13 | `CONFIG` dict with placeholder secrets at module scope | Eliminated — secrets only ever touch memory at runtime |

---

## Architecture

```
news-sentiment-trader/
│
├── run.py                           ← Pipeline entry point (CLI)
├── .env.example                     ← Credential template (never commit .env)
│
├── src/trader/
│   ├── config.py                    ← Settings loaded from env vars
│   ├── data/
│   │   └── factset.py               ← FactSet RTNews client (retry, rate limit)
│   ├── nlp/
│   │   └── sentiment.py             ← DeepSeek Chat analyser (sliding-window RL)
│   ├── strategy/
│   │   └── signals.py               ← Sentiment → LONG / SHORT / HOLD
│   └── backtest/
│       └── engine.py                ← Event-driven backtester + metrics
│
└── tests/
    ├── test_strategy.py             ← Signal generation (no API calls)
    ├── test_backtest.py             ← Metric correctness (MDD, Sharpe, total return)
    └── test_sentiment.py            ← Sentiment analyser (fully mocked)
```

---

## Setup

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/news-sentiment-trader.git
cd news-sentiment-trader
pip install -e ".[dev]"
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in FACTSET_API_KEY and DEEPSEEK_API_KEY
export $(cat .env | xargs)
```

### 3. Run

```bash
python run.py \
    --tickers AAPL MSFT GOOGL \
    --start   2022-01-01 \
    --end     2022-12-31 \
    --prices  data/prices.csv \
    --term    short
```

**Expected output:**
```
==================================================
  BACKTEST RESULTS
==================================================
  sharpe_ratio          : 0.8234
  max_drawdown          : -0.0412        ← always ≤ 0
  total_return          : 0.1847
  hit_rate              : 0.5800
  n_trades              : 142
  avg_trade_return      : 0.001302
==================================================
```

### 4. Test

```bash
pytest tests/ -v
```

All tests run without API keys — the sentiment analyser is fully mocked.

---

## Price Data Format

`data/prices.csv` must contain:

| Column | Type | Description |
|--------|------|-------------|
| `date`   | YYYY-MM-DD | Trading date |
| `ticker` | str | Ticker symbol matching FactSet feed |
| `return` | float | One-day simple return (e.g. `0.0123` = +1.23%) |

---

## Key Design Decisions

**Rate limiting** — FactSet uses token-bucket limiting (`time.monotonic()` based, 100 ms floor). DeepSeek uses a **sliding-window deque** that tracks actual token consumption over the past 60 seconds — not a simple counter that forgets history.

**Retry** — Both API clients share a `requests.Session` mounted with `HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504]))`. Transient 5xx errors and rate-limit 429s are retried automatically with exponential backoff.

**Metrics** — All three performance metrics use geometrically compounded returns `(1+r).cumprod()`. Sharpe ratio is annualised by the *observed* signal frequency (signals per year in the backtest window), not the hardcoded 252-day constant.

**Secrets** — No credentials ever appear in source code. `load_settings()` calls `os.environ.get()` at runtime; the `Settings` dataclass is frozen and passed as a dependency to every client. `CONFIG` dict is gone entirely.

---

## Extending the System

**Swap the LLM** — Replace `DeepSeekAnalyzer` with any chat-completion API (OpenAI, Claude, Gemini) by subclassing and overriding `analyze_sentiment()`. The rest of the pipeline is unchanged.

**Add more signals** — `TradingStrategy.generate_signals()` currently maps one headline → one signal. You can aggregate multiple headlines per stock per day before calling `generate_signals()` to reduce noise.

**Position sizing** — The backtester currently assumes unit position size. Add a `position_size` column to signals (e.g. Kelly criterion or vol-targeting) and multiply `trade_return × position_size` in `run_backtest()`.

---

## License

MIT
