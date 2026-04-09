#!/usr/bin/env python3
"""
run.py
======
Main pipeline entry point.

Usage
-----
    # Set credentials first:
    cp .env.example .env
    # Fill in FACTSET_API_KEY and DEEPSEEK_API_KEY, then:
    export $(cat .env | xargs)

    python run.py \
        --tickers AAPL MSFT \
        --start 2022-01-01 \
        --end   2022-12-31 \
        --prices data/prices.csv
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="News-Sentiment Trading Pipeline")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"],
                        help="Ticker symbols to analyse")
    parser.add_argument("--start",   default="2022-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",     default="2022-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--prices",  default="data/prices.csv",
                        help="Path to historical prices CSV (date, ticker, return)")
    parser.add_argument("--term",    default="short", choices=["short", "long"],
                        help="Investment horizon for the sentiment prompt")
    args = parser.parse_args()

    # ── 0. Load config ────────────────────────────────────────────────
    from trader.config import load_settings
    try:
        settings = load_settings()
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # ── 1. Fetch news ─────────────────────────────────────────────────
    from trader.data.factset import FactSetClient
    logger.info("Fetching news for %s (%s → %s)...", args.tickers, args.start, args.end)
    factset  = FactSetClient(settings)
    news_df  = factset.get_bulk_news(args.tickers, args.start, args.end)

    if news_df.empty:
        logger.error("No news returned — check tickers and date range.")
        sys.exit(1)

    logger.info("Retrieved %d headlines.", len(news_df))

    # ── 2. Sentiment analysis ─────────────────────────────────────────
    from trader.nlp.sentiment import DeepSeekAnalyzer
    logger.info("Analysing sentiment with DeepSeek Chat...")
    analyzer = DeepSeekAnalyzer(settings)

    news_df[["sentiment", "explanation"]] = news_df.apply(
        lambda row: analyzer.analyze_sentiment(
            row["headline"], row["ticker"], term=args.term
        ),
        axis=1,
        result_type="expand",
    )

    # ── 3. Signal generation ──────────────────────────────────────────
    from trader.strategy.signals import TradingStrategy
    all_signals      = TradingStrategy.generate_signals(news_df)
    actionable       = TradingStrategy.actionable(all_signals)

    logger.info(
        "Signals: %d total | %d LONG | %d SHORT | %d HOLD",
        len(all_signals),
        (all_signals["action"] == "LONG").sum(),
        (all_signals["action"] == "SHORT").sum(),
        (all_signals["action"] == "HOLD").sum(),
    )

    # ── 4. Backtest ───────────────────────────────────────────────────
    try:
        price_data = pd.read_csv(args.prices)
    except FileNotFoundError:
        logger.error("Price file not found: %s", args.prices)
        logger.error("Provide a CSV with columns: date, ticker, return")
        sys.exit(1)

    from trader.backtest.engine import Backtester
    backtester = Backtester(price_data)
    results    = backtester.run_backtest(actionable)

    if results.empty:
        logger.warning("Backtest produced no results — check price data alignment.")
        sys.exit(0)

    # ── 5. Report ─────────────────────────────────────────────────────
    metrics = Backtester.calculate_metrics(results)

    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print("=" * 50)
    print("\nSample trades:")
    print(results[["timestamp", "ticker", "sentiment", "action", "trade_return"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
