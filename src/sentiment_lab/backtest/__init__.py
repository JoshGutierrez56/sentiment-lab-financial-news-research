"""Conservative event-to-return alignment and basic metrics."""

from sentiment_lab.backtest.event_engine import align_events
from sentiment_lab.backtest.metrics import compute_event_metrics

__all__ = ["align_events", "compute_event_metrics"]
