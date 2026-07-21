"""Purged, embargoed walk-forward folds using prediction and outcome timestamps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_start: datetime
    validation_end: datetime
    embargo_end: datetime


def purged_walk_forward_folds(
    prediction_times: Sequence[datetime],
    evaluation_end_times: Sequence[datetime],
    *,
    n_splits: int,
    embargo: timedelta,
) -> list[PurgedFold]:
    if len(prediction_times) != len(evaluation_end_times):
        raise ValueError("prediction and evaluation timestamp lengths differ")
    if n_splits < 2 or len(prediction_times) < n_splits:
        raise ValueError("Need at least two non-empty chronological folds")
    ordered = sorted(range(len(prediction_times)), key=lambda index: prediction_times[index])
    size = len(ordered) // n_splits
    folds: list[PurgedFold] = []
    for fold in range(1, n_splits):
        start, end = fold * size, (fold + 1) * size if fold < n_splits - 1 else len(ordered)
        validation = ordered[start:end]
        validation_start = prediction_times[validation[0]]
        validation_end = max(evaluation_end_times[index] for index in validation)
        embargo_end = validation_end + embargo
        # In walk-forward form training is historical.  The embargo remains
        # recorded for the next admissible decision timestamp; purging removes
        # every historical label whose outcome window enters validation.
        train = tuple(
            index for index in ordered[:start] if evaluation_end_times[index] < validation_start
        )
        folds.append(
            PurgedFold(
                fold, train, tuple(validation), validation_start, validation_end, embargo_end
            )
        )
    return folds
