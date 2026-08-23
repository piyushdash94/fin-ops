"""Walk-forward splitting for time series.

K-fold cross-validation is invalid on price data: shuffling lets a model train
on Thursday to predict Wednesday. Walk-forward evaluation instead retrains on a
growing (or rolling) window of the past and tests only on the period that
followed -- the same information a live model would have had.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

__all__ = ["Fold", "walk_forward", "n_folds"]


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray

    def __len__(self) -> int:
        return len(self.test)


def walk_forward(
    n_samples: int,
    *,
    train_size: int = 750,
    test_size: int = 126,
    step: int | None = None,
    expanding: bool = True,
    embargo: int = 1,
) -> Iterator[Fold]:
    """Yield successive train/test folds ordered in time.

    ``embargo`` drops rows between train and test. It must be at least the
    forecast horizon: with a next-day target, the final training row's label is
    only known on the first test day, so training on it leaks one day of the
    future. Defaults to 1 for the standard next-day setup.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    step = step or test_size
    start = 0
    fold_index = 0
    while True:
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples or test_end - test_start < 1:
            break

        train_start = 0 if expanding else start
        yield Fold(
            index=fold_index,
            train=np.arange(train_start, train_end),
            test=np.arange(test_start, test_end),
        )
        fold_index += 1
        start += step


def n_folds(n_samples: int, **kwargs) -> int:
    return sum(1 for _ in walk_forward(n_samples, **kwargs))
