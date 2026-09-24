"""A deadline sent to the model worker is stamped on the clock the worker reads.

An experiment run replaces `time.time` with a clock a restore rewinds
(core/subject/clock.py). The client stamped the worker's deadline with it, and
the worker, another process, compared it against the machine's clock, so every
report arm after an anchor's first reached the worker already past its deadline:
"deadline_exceeded_before_decode", and on 23 September no whole run's report arm
after the first got an answer from her cortex.
"""

from __future__ import annotations

import time

import pytest

from core.brain.llm.mlx_client import _worker_deadline
from core.brain.llm.mlx_worker import _generation_deadline_open, _seconds_left_on
from core.runtime.wall_clock import wall_time
from core.subject.clock import ExperimentClock

pytestmark = pytest.mark.unit


def _rewound() -> ExperimentClock:
    """An experiment clock a restore has put an hour behind the machine."""
    return ExperimentClock(step=0.03, start=time.time() - 3600.0)


def test_the_machines_clock_is_read_whatever_time_time_has_become() -> None:
    before = wall_time()
    with _rewound():
        assert time.time() < before - 3000.0
        assert wall_time() >= before


def test_a_deadline_stamped_under_a_rewound_clock_is_still_ahead_of_the_worker() -> None:
    with _rewound():
        deadline = _worker_deadline(30.0)
    assert deadline > wall_time() + 25.0
    assert _generation_deadline_open({"deadline_unix": deadline}, started=False)
    assert _seconds_left_on({"deadline_unix": deadline}) > 25.0


def test_a_worker_in_process_reads_the_same_clock_the_client_stamped_with() -> None:
    """In process the worker shares the run's `time.time`; the deadline must still hold."""
    with _rewound():
        deadline = _worker_deadline(30.0)
        assert _generation_deadline_open({"deadline_unix": deadline}, started=False)
        assert not _generation_deadline_open({"deadline_unix": wall_time() - 1.0}, started=False)
