"""What the loser of a construction race leaves behind.

`get_ontogeny` and `get_experience_spine` both build outside their lock and
publish under it, so a race builds two and keeps one. That is the right shape —
the constructor opens a store and fsyncs, and holding a process-wide lock
across an fsync stalls every other caller. What the shape costs is a candidate
that has already touched shared state by the time it loses:

* it subscribed to the spine's resolutions, and nothing took the subscription
  off, so every outcome in the system was tallied twice — once into an organ
  nobody holds;
* it handed the authority ledger its own calibration monitor, replacing the
  winner's, so the gate that decides whether a head may decide was reading
  measurements nothing was recording;
* it had started a sweeper thread and a maintenance loop before anyone looked;
* and when it was stopped it closed the spine — a singleton it did not build —
  which stops the flusher for every other holder. Every episode recorded after
  that sat in a queue nothing would ever write, with no error anywhere.

These run before a long developmental run, because all four are silent.
"""

from __future__ import annotations

import threading

import pytest

from core.ontogeny.experience import (
    ExperienceSpine,
    get_experience_spine,
    reset_experience_spine_for_test,
)
from core.ontogeny.service import OntogenyCore, get_ontogeny, reset_ontogeny_for_test

pytestmark = pytest.mark.unit


@pytest.fixture
def fresh(tmp_path):
    """A spine of this test's own, and no organ built on it yet."""
    reset_ontogeny_for_test(None)
    spine = ExperienceSpine(db_path=tmp_path / "experience.db")
    reset_experience_spine_for_test(spine)
    spine.shared = True
    yield spine
    reset_ontogeny_for_test(None)
    reset_experience_spine_for_test(None)


def test_eight_threads_asking_at_once_get_one_organ(fresh) -> None:
    seen: list[object] = []
    barrier = threading.Barrier(8)

    def ask() -> None:
        barrier.wait()
        seen.append(get_ontogeny())

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(seen) == 8
    assert len({id(core) for core in seen}) == 1


def test_a_loser_does_not_leave_a_subscriber_behind(fresh) -> None:
    """The tally that would have been kept twice."""
    winner = get_ontogeny()
    assert fresh.subscriber_count() == 1

    loser = OntogenyCore(autostart=False)
    assert fresh.subscriber_count() == 2
    loser.dispose()
    assert fresh.subscriber_count() == 1
    assert winner is get_ontogeny()


def test_a_loser_does_not_start_anything(fresh) -> None:
    """It had a sweeper and a maintenance loop before anyone had looked."""
    loser = OntogenyCore(autostart=False)
    assert loser._sweeper is None
    assert loser._maintenance is None
    loser.dispose()


def test_the_ledger_keeps_the_monitor_that_is_being_fed(fresh) -> None:
    """The loser's constructor replaces it; recommissioning puts it back."""
    winner = get_ontogeny()
    loser = OntogenyCore(autostart=False)
    assert winner._authority._calibration is loser._candidate_calibration
    loser.dispose()
    winner.recommission()
    assert winner._authority._calibration is winner._candidate_calibration


def test_stopping_the_organ_does_not_stop_the_spine(fresh) -> None:
    """A store you borrow is owed a flush, not a close."""
    core = get_ontogeny()
    core.stop()
    assert not fresh._stopped.is_set(), (
        "stopping the organ closed the shared spine; every episode recorded "
        "after this point would sit in a queue nothing writes"
    )


def test_the_organ_is_started_after_it_is_published(fresh) -> None:
    core = get_ontogeny()
    assert core._sweeper is not None
    assert core._maintenance is not None


def test_subscribing_twice_subscribes_once(fresh) -> None:
    core = get_ontogeny()
    before = fresh.subscriber_count()
    core.recommission()
    core.recommission()
    assert fresh.subscriber_count() == before


def test_unsubscribing_something_that_never_subscribed_is_quiet(fresh) -> None:
    fresh.off_resolve(lambda episode_id, outcome: None)
