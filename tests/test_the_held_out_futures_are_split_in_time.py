"""The carrier's held-out futures are cross-validated in contiguous blocks of anchors.

`core/subject/estimate.py` says why every fit in the battery splits in time:
rows next to each other in a trajectory are nearly the same row, and a shuffled
split lets the model see the answer sitting beside the question. The carrier's
anchors are consecutive turns and an anchor's older turns are exactly a
neighbour's recent ones, and the two estimators that read held-out futures
shuffled their folds. On the carrier at f78644bbd that read as history still
needed at the longest rung; in blocks, older turns add no more than shuffled
ones at any rung (22 October).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from core.subject import intrinsic_v25


@pytest.fixture
def held_out_blocks(monkeypatch):
    seen: list[np.ndarray] = []
    split = KFold.split

    def recording(self, x, y=None, groups=None):
        for train, test in split(self, x, y, groups):
            seen.append(np.asarray(test))
            yield train, test

    monkeypatch.setattr(KFold, "split", recording)
    return seen


def _contiguous(block: np.ndarray) -> bool:
    return bool(np.all(np.diff(np.sort(block)) == 1)) and bool(np.all(np.diff(block) == 1))


def _rows(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 6)), rng.normal(size=(n, 3)), rng.normal(size=(n, 5))


def test_the_sufficiency_gain_holds_out_blocks_of_anchors(held_out_blocks):
    history, state, future = _rows()
    intrinsic_v25.heldout_sufficiency_gain(history, state, future)
    assert held_out_blocks and all(_contiguous(block) for block in held_out_blocks)


def test_the_new_direction_gain_holds_out_blocks_of_anchors(held_out_blocks):
    history, state, future = _rows()
    intrinsic_v25.heldout_new_direction_gain(history, state, future)
    assert held_out_blocks and all(_contiguous(block) for block in held_out_blocks)
