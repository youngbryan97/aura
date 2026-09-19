"""What she invented is written when it changes, not every time she is asked.

LIVE 2026-09-19: the autonomy conductor asks every five minutes, and every
five minutes the same two properties were written and fsynced again, each
time under the store's I/O lane, which lockdep reported as a blocking
operation under a lock.
"""
from __future__ import annotations

from core.agency import what_she_invented
from core.agency.how_good_is_this import INVENTED, forget, promote
from core.agency.inventing_a_measure import Measure
from core.runtime import lockdep


def _one_measure() -> str:
    return promote(Measure(at="neighbours", of="how big it is", summed="on average"), 0.4)


def test_the_same_body_is_written_once(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_invented, "_kept_at", lambda: tmp_path / "invented.json")
    monkeypatch.setattr(what_she_invented, "_LAST_KEPT", {"body": None})
    written: list[object] = []
    real_store = what_she_invented._the_store

    class Counting:
        def __init__(self) -> None:
            self.inner = real_store()

        def save(self, body):
            written.append(body)
            return self.inner.save(body)

    monkeypatch.setattr(what_she_invented, "_the_store", Counting)
    name = _one_measure()
    try:
        assert what_she_invented.keep()
        assert what_she_invented.keep()
        assert len(written) == 1
        forget(name)
        assert what_she_invented.keep(), "a change is written"
        assert len(written) == 2
    finally:
        forget(name)
        INVENTED.pop(name, None)


def test_a_file_that_went_away_is_written_again(tmp_path, monkeypatch):
    kept_at = tmp_path / "invented.json"
    monkeypatch.setattr(what_she_invented, "_kept_at", lambda: kept_at)
    monkeypatch.setattr(what_she_invented, "_LAST_KEPT", {"body": None})
    name = _one_measure()
    try:
        assert what_she_invented.keep()
        kept_at.unlink()
        assert what_she_invented.keep()
        assert kept_at.exists()
    finally:
        forget(name)
        INVENTED.pop(name, None)


def test_the_store_lane_is_sanctioned_with_its_reason():
    reason = lockdep.SANCTIONED_BLOCKING_LOCKS["core.persistence.a_versioned_store.self._io_lane"]
    assert "older flush" in reason and "to_thread" in reason
