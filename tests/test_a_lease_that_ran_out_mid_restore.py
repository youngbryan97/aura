"""Thirteen hours of recording lost to a thirty-second lease.

A governance lease lasts thirty seconds, short on purpose so that an async task
which went to sleep inside one cannot wake up still holding authority. The
fork's restore held a single lease while it put a filesystem back, and a
campaign's state root grows as the run proceeds. On the seed-7 campaign of
16 September the restore finally took longer than the lease and the run died
in its thirteenth hour:

    GovernanceViolationError: file_write_gateway.ensure_directory:
    subject_core.fork called outside governed context

Nothing was wrong except that there was more to put back than there had been at
the start. The lease is renewed as the work proceeds now, at half its length.
"""

from __future__ import annotations

import time

from core.governance_context import is_governed
from core.subject.snapshot import _ForkLease


def test_the_scope_governs_while_it_is_held() -> None:
    assert not is_governed()
    with _ForkLease("subject_core.fork"):
        assert is_governed()
    assert not is_governed()


def test_an_unrenewed_lease_runs_out(monkeypatch) -> None:
    """The mechanism the campaign died on, held so it stays understood."""
    with _ForkLease("subject_core.fork") as lease:
        token = lease._token
        assert is_governed()
        monkeypatch.setattr(
            time, "monotonic", lambda: token.mono_timestamp + token.ttl + 1.0
        )
        assert not is_governed(), "a lease is supposed to expire"


def test_keeping_it_renews_before_it_runs_out(monkeypatch) -> None:
    with _ForkLease("subject_core.fork") as lease:
        first = lease._token
        monkeypatch.setattr(
            time, "monotonic", lambda: first.mono_timestamp + first.ttl * 0.6
        )
        lease.keep()
        assert lease._token is not first, "the lease was not renewed"
        monkeypatch.undo()
        assert is_governed()


def test_keeping_it_early_does_not_churn_the_ledger() -> None:
    with _ForkLease("subject_core.fork") as lease:
        first = lease._token
        for _ in range(20):
            lease.keep()
        assert lease._token is first, "a lease with time left was replaced"


def test_the_restore_renews_as_it_walks() -> None:
    """Both restore loops ask to keep the lease, not only one of them."""
    import inspect

    from core.subject import snapshot

    for name in ("_restore_world", "_restore_stores"):
        source = inspect.getsource(getattr(snapshot, name))
        assert "_ForkLease" in source, f"{name} does not take a renewable lease"
        assert "lease.keep()" in source, f"{name} never renews it"


def test_a_long_restore_stays_governed(monkeypatch, tmp_path) -> None:
    """The failing case end to end: a walk that outlasts a lease."""
    from core.subject.snapshot import _restore_world

    root = tmp_path / "scratch"
    root.mkdir()
    saved = {
        "files": {f"file{i}.txt": b"x" for i in range(40)},
        "directories": [f"dir{i}" for i in range(40)],
    }

    clock = {"now": time.monotonic()}
    real = time.monotonic

    def creeping() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(time, "monotonic", creeping)
    try:
        _restore_world(root, saved)
    finally:
        monkeypatch.setattr(time, "monotonic", real)

    assert (root / "file0.txt").exists()
    assert (root / "dir39").is_dir()
