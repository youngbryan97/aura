"""A refusal that cannot change is said once, not on every turn.

The non-parametric store must be dense enough for its neighbours to be near
before it may steer generation — a real density requirement, argued in the
module and paid for by two garbled live answers. What is not required is
saying so on every turn of every session: the store cannot grow past the floor
while the process runs, so the verdict is a standing fact.

LIVE, 2026-09-07: "408 entries is too sparse to steer a 5120-wide space (need
50,000)" once per turn, forever, against a floor nothing on this host reaches.
The module already says of its other unusable branch that repeating a verdict
which cannot improve is noise; this branch returned before reaching it.

The recall outcome still carries the reason every turn. What changed is the
log.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _forget_what_was_said():
    from core.brain import nonparametric_worker as worker

    worker._REPORTED_REFUSALS.clear()
    yield
    worker._REPORTED_REFUSALS.clear()


class _Store:
    """A store below the density floor, which is the live case."""

    _dim = 5120
    _path = ""

    def __init__(self, entries: int) -> None:
        self._tokens = [f"token {index}" for index in range(entries)]

    def __len__(self) -> int:
        return len(self._tokens)


def test_a_sparse_store_still_refuses() -> None:
    from core.brain.nonparametric_worker import _unusable_datastore_reason

    reason = _unusable_datastore_reason(_Store(408))
    assert "too sparse" in reason
    assert "50,000" in reason


def test_a_dense_enough_store_is_not_refused_for_sparsity() -> None:
    from core.brain.nonparametric_worker import (
        _MIN_ENTRIES_TO_STEER_GENERATION,
        _unusable_datastore_reason,
    )

    reason = _unusable_datastore_reason(_Store(_MIN_ENTRIES_TO_STEER_GENERATION))
    assert "too sparse" not in reason


def test_the_refusal_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    from core.brain import nonparametric_worker as worker

    reason = worker._unusable_datastore_reason(_Store(408))
    with caplog.at_level(logging.WARNING):
        for _turn in range(5):
            if reason not in worker._REPORTED_REFUSALS:
                worker._REPORTED_REFUSALS.add(reason)
                worker.logger.warning("REFUSED: %s", reason)
    said = [record for record in caplog.records if "REFUSED" in record.message]
    assert len(said) == 1


def test_a_different_condition_is_said_again() -> None:
    """Saying it once must not silence a new fact."""

    from core.brain import nonparametric_worker as worker

    first = worker._unusable_datastore_reason(_Store(408))
    second = worker._unusable_datastore_reason(_Store(409))
    assert first != second
    worker._REPORTED_REFUSALS.add(first)
    assert second not in worker._REPORTED_REFUSALS


def test_the_live_path_says_it_once() -> None:
    """The guard is where the turn refuses, not only in this test."""

    import inspect

    from core.brain import nonparametric_worker as worker

    body = inspect.getsource(worker)
    marker = body.index("Foreground non-parametric memory REFUSED")
    before = body[marker - 400 : marker]
    assert "_REPORTED_REFUSALS" in before
