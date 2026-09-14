"""Owned RLock recursion adds depth, not a new wait dependency."""

import threading

import pytest

from core.runtime import lockdep


@pytest.fixture
def validator(monkeypatch):
    validator = lockdep.LockdepValidator()
    monkeypatch.setattr(lockdep, "_VALIDATOR", validator)
    monkeypatch.setattr(validator, "_report_splats", lambda pending: None)
    return validator


@pytest.mark.parametrize("ranked", [False, True])
def test_owned_recursion_does_not_invent_reverse_dependencies(validator, ranked):
    low = lockdep.LockRank.REGISTRY if ranked else lockdep.LockRank.UNRANKED
    high = lockdep.LockRank.LEAF if ranked else lockdep.LockRank.UNRANKED
    first = lockdep.checked_lock("first", rank=low, reentrant=True)
    second = lockdep.checked_lock("second", rank=high, reentrant=True)
    with first, second:
        with first:
            assert validator.held_names() == ["first", "second", "first"]
        assert validator.held_names() == ["first", "second"]
    report = validator.report()
    assert report["splats"] == []
    assert report["order_edges"] == {"first": ["second"]}
    assert report["currently_held"] == {}


def test_reentry_does_not_hide_a_later_real_reversed_acquisition(validator):
    first = lockdep.checked_lock("first", reentrant=True)
    second = lockdep.checked_lock("second", reentrant=True)
    with first, second, first:
        pass
    with second, first:
        pass
    assert [s["kind"] for s in validator.report()["splats"]] == ["order_inversion"]


def test_another_threads_ownership_is_not_reentry(validator):
    first = lockdep.checked_lock("first", reentrant=True)
    second = lockdep.checked_lock("second", reentrant=True)
    with first, second:
        pass
    attempted = threading.Event()
    results = []
    def reverse():
        with second:
            results.append(first.acquire(blocking=False))
        attempted.set()
    with first:
        thread = threading.Thread(target=reverse, daemon=True)
        thread.start()
        assert attempted.wait(2)
        thread.join(2)
    assert results == [False]
    assert [s["kind"] for s in validator.report()["splats"]] == ["order_inversion"]
    assert validator.report()["currently_held"] == {}


def test_nonreentrant_lock_still_reports_self_deadlock(validator):
    lock = lockdep.checked_lock("plain")
    with lock:
        assert lock.acquire(blocking=False) is False
    assert [s["kind"] for s in validator.report()["splats"]] == ["self_deadlock"]
