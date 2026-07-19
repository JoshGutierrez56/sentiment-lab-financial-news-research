from __future__ import annotations

from sentiment_lab.hybrid.openai_calibration import _priority_bucket


def test_priority_buckets_target_bias_revealing_cases() -> None:
    bullish = {
        "sentiment_label": "bullish",
        "confidence": 0.9,
        "abstain": False,
        "event_type": "earnings",
    }
    assert _priority_bucket(bullish, set()) == ("high_confidence_bullish",)
    abstain_rare = {
        "sentiment_label": "neutral",
        "confidence": 0.4,
        "abstain": True,
        "event_type": "cybersecurity",
    }
    assert _priority_bucket(abstain_rare, {"cybersecurity"}) == (
        "local_abstention",
        "low_confidence",
        "rare_event_type",
    )


def test_general_bucket_is_used_only_when_no_target_applies() -> None:
    row = {
        "sentiment_label": "neutral",
        "confidence": 0.7,
        "abstain": False,
        "event_type": "earnings",
    }
    assert _priority_bucket(row, set()) == ("general_calibration",)
