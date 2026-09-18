"""Four loop-blocking holds that were contention, not work.

R06's pattern is "the lock is held across the WORK rather than across the
read the work needs". These four were the other half of it: the section
is already small, and lockdep still measured 273ms, 148ms, 87ms and 53ms
on them, because every caller in the process queues on a lock protecting
something written once.

A register written by rebinding does not need a reader to hold anything.
Reading a module global is atomic, so a reader either sees the version
before a write or the version after it, never half of one.
"""

from __future__ import annotations

import threading

import numpy as np

from core.canonical.state import get_canonical_state
from core.runtime import health_fragments as hf
from core.verify.earned_metric import EarnedAxis


def _axis(capacity: int) -> EarnedAxis:
    return EarnedAxis(
        "valence",
        min_samples=4,
        holdout_fraction=0.3,
        min_holdout_r=0.35,
        max_p=0.05,
        ridge_penalty=1.0,
        permutations=16,
        capacity=capacity,
    )


def test_the_canonical_state_is_built_once_and_then_read_without_the_lock():
    first = get_canonical_state()
    assert get_canonical_state() is first

    # Every thread gets the same object, and the construction still
    # happens exactly once.
    seen: list[object] = []
    threads = [
        threading.Thread(target=lambda: seen.append(get_canonical_state()))
        for _ in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(state is first for state in seen)


def test_the_fragment_register_is_published_by_rebinding():
    # Mutating in place is what makes a lockless reader unsafe.
    with hf.health_fragments_reset():
        before = hf._PROVIDERS
        hf.register_health_fragment("probe_one", lambda: {"available": True})
        after = hf._PROVIDERS
        assert after is not before, "a reader with no lock must see a whole version"
        assert "probe_one" in after
        assert "probe_one" not in before


def test_a_reset_rebinds_rather_than_clearing_under_a_reader():
    with hf.health_fragments_reset():
        hf.register_health_fragment("probe_two", lambda: {"available": True})
        held = hf._PROVIDERS
        saved = hf.reset_health_fragments_for_test()
        assert hf._PROVIDERS is not held
        # The version a reader was already holding is untouched.
        assert "probe_two" in held
        hf.restore_health_fragments_for_test(saved)
        assert "probe_two" in hf._PROVIDERS


def test_collecting_fragments_survives_registration_during_the_walk():
    # The reader takes no lock, so this is the case that must not break.
    with hf.health_fragments_reset():
        hf.register_health_fragment("steady", lambda: {"available": True})

        def _churn() -> None:
            for index in range(200):
                hf.register_health_fragment(
                    f"late_{index}", lambda: {"available": True}
                )

        writer = threading.Thread(target=_churn)
        writer.start()
        for _ in range(50):
            fragments = hf.collect_health_fragments()
            assert fragments["steady"]["registered"] is True
        writer.join()


def test_the_surface_register_is_a_rebound_tuple():
    from core.language import learned_matcher

    assert isinstance(learned_matcher._SURFACES, tuple)
    assert learned_matcher.registered_surfaces() is learned_matcher._SURFACES


def test_an_axis_evicts_in_constant_time_rather_than_shifting_a_list():
    # The trim was ``del self._states[:n]`` inside the lock, on every
    # observation once full.
    axis = _axis(8)
    for step in range(40):
        axis.observe([float(step), float(step) * 0.5], float(step))
    assert len(axis._observations) == 8
    assert axis._observations.maxlen == 8
    # The newest observations are the ones kept.
    assert axis._observations[-1][1] == 39.0
    assert axis.snapshot()["observations"] == 8


def test_an_axis_still_forgets_a_resized_substrate():
    axis = _axis(8)
    axis.observe([1.0, 2.0], 1.0)
    axis.observe([1.0, 2.0], 2.0)
    axis.observe([1.0, 2.0, 3.0], 3.0)
    assert len(axis._observations) == 1
    assert np.asarray(axis._observations[-1][0]).size == 3
