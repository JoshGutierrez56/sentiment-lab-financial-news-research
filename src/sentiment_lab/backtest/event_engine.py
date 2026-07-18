"""Align classified articles to future returns without using an earlier price."""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from sentiment_lab.data.schemas import EODPrice, NewsArticle
from sentiment_lab.nlp.schemas import ClassificationRecord

NEW_YORK = ZoneInfo("America/New_York")


def align_events(
    articles: list[NewsArticle],
    classifications: list[ClassificationRecord],
    prices: list[EODPrice],
    *,
    horizons: list[int],
) -> pl.DataFrame:
    """Enter at the first trading-day open strictly after publication's local date.

    EODHD publishes raw open and adjusted close. The entry open is adjusted by
    that day's `adjusted_close / close` factor, avoiding split-scale mismatches.
    A horizon of 1 exits at the entry day's adjusted close; 3 and 5 exit at the
    third and fifth trading-day closes respectively.
    """

    if len(articles) != len(classifications):
        raise ValueError("articles and classifications must have equal length")
    if not prices:
        raise ValueError("prices must not be empty")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")

    ordered_prices = sorted(prices, key=lambda item: item.date)
    records: list[dict[str, object]] = []
    for article, classification in zip(articles, classifications, strict=True):
        if classification.assessment.article_id != article.article_id:
            raise ValueError("classification/article identity mismatch")
        if classification.assessment.event_timestamp != article.provider_timestamp:
            raise ValueError("classification/article timestamp mismatch")
        local_publication = article.provider_timestamp.astimezone(NEW_YORK)
        entry_index = next(
            (
                index
                for index, price in enumerate(ordered_prices)
                if price.date > local_publication.date()
            ),
            None,
        )
        row: dict[str, object] = {
            "article_id": article.article_id,
            "ticker": classification.assessment.ticker,
            "publication_timestamp_utc": article.provider_timestamp,
            "publication_timestamp_local": local_publication,
            "title": article.title,
            "article_text": article.content,
            "link": article.link,
            "provider_sentiment_polarity": article.provider_sentiment_polarity,
            "sentiment_label": classification.assessment.sentiment_label.value,
            "sentiment_score": classification.assessment.sentiment_score,
            "confidence": classification.assessment.confidence,
            "relevance": classification.assessment.relevance,
            "event_type": classification.assessment.event_type.value,
            "expected_horizon": classification.assessment.expected_horizon.value,
            "reasoning": classification.assessment.concise_reasoning,
            "tradable": classification.assessment.tradable,
            "abstain_reason": classification.assessment.abstain_reason,
            "openai_model": classification.model,
            "prompt_version": classification.prompt_version,
            "classification_cache_key": classification.cache_key,
        }
        if entry_index is None:
            row.update(
                {
                    "entry_date": None,
                    "entry_timestamp_utc": None,
                    "entry_adjusted_open": None,
                }
            )
            for horizon in horizons:
                row[f"exit_date_{horizon}d"] = None
                row[f"future_return_{horizon}d"] = None
        else:
            entry = ordered_prices[entry_index]
            entry_timestamp = datetime.combine(entry.date, time(9, 30), tzinfo=NEW_YORK).astimezone(
                UTC
            )
            row.update(
                {
                    "entry_date": entry.date,
                    "entry_timestamp_utc": entry_timestamp,
                    "entry_adjusted_open": entry.adjusted_open,
                }
            )
            for horizon in horizons:
                exit_index = entry_index + horizon - 1
                if exit_index >= len(ordered_prices):
                    row[f"exit_date_{horizon}d"] = None
                    row[f"future_return_{horizon}d"] = None
                else:
                    exit_price = ordered_prices[exit_index]
                    row[f"exit_date_{horizon}d"] = exit_price.date
                    row[f"future_return_{horizon}d"] = (
                        exit_price.adjusted_close / entry.adjusted_open - 1.0
                    )
        records.append(row)
    return pl.DataFrame(records, infer_schema_length=None)
