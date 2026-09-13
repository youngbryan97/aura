"""A handle a background writer is using is not a leaked handle.

Ontogeny's experience flusher is a daemon thread on a two-second timer. The
store it holds open for the length of one write shows up under whichever test
happens to be tearing down, and the hermetic fixture used to call that a leak.
Nothing could tell the two apart from outside: sqlite objects are thread-affine,
so the sweeper's `PRAGMA database_list` raises rather than answers when the
connection belongs to the flusher.
"""
from __future__ import annotations

import sys
import threading
import time

import core.ontogeny.experience as experience
from core.ontogeny.experience import Episode, ExperienceSpine

from tests.conftest import wait_out_declared_background_writers


def _an_episode() -> Episode:
    return Episode(
        control_point="a_test",
        features={"x": 1.0},
        decision="left",
        options=("left", "right"),
    )


def test_a_write_raises_the_flag_and_the_end_of_it_lowers_it(tmp_path, monkeypatch):
    spine = ExperienceSpine(db_path=tmp_path / "experience.db", autoflush=False)
    assert not spine.a_write_is_in_flight()

    during: list[bool] = []
    real = experience.connecting
    monkeypatch.setattr(
        experience, "connecting", lambda conn: (during.append(spine.a_write_is_in_flight()), real(conn))[1]
    )
    spine.record(_an_episode())
    assert spine.flush() == 1

    assert during == [True], "the flag has to be up while the connection is open"
    assert not spine.a_write_is_in_flight(), "and down once it is shut"


def test_waiting_when_nothing_is_writing_costs_nothing(tmp_path):
    spine = ExperienceSpine(db_path=tmp_path / "experience.db", autoflush=False)
    started = time.monotonic()
    assert spine.wait_until_quiet(5.0) is True
    assert time.monotonic() - started < 0.5


def test_the_wait_returns_when_the_writer_finishes(tmp_path, monkeypatch):
    spine = ExperienceSpine(db_path=tmp_path / "experience.db", autoflush=False)
    holding = threading.Event()
    release = threading.Event()
    real = experience.connecting

    def slow(conn):
        holding.set()
        release.wait(5.0)
        return real(conn)

    monkeypatch.setattr(experience, "connecting", slow)
    spine.record(_an_episode())
    writer = threading.Thread(target=spine.flush, name="test-slow-flush")
    writer.start()
    try:
        assert holding.wait(5.0), "the flush never reached its connection"
        assert spine.a_write_is_in_flight()
        assert spine.wait_until_quiet(0.2) is False, "a wait shorter than the write gives up"
        release.set()
        assert spine.wait_until_quiet(5.0) is True
    finally:
        release.set()
        writer.join(5.0)
    assert not writer.is_alive()


def test_the_fixture_waits_the_shared_spine_out_instead_of_blaming_a_test(tmp_path, monkeypatch):
    spine = ExperienceSpine(db_path=tmp_path / "experience.db", autoflush=False)
    monkeypatch.setattr(experience, "_spine", spine)
    assert experience.a_background_write_is_in_flight() is False
    assert wait_out_declared_background_writers() == []

    holding = threading.Event()
    release = threading.Event()
    real = experience.connecting

    def slow(conn):
        holding.set()
        release.wait(5.0)
        return real(conn)

    monkeypatch.setattr(experience, "connecting", slow)
    spine.record(_an_episode())
    writer = threading.Thread(target=spine.flush, name="test-slow-flush")
    writer.start()
    try:
        assert holding.wait(5.0)
        assert experience.a_background_write_is_in_flight() is True
        threading.Timer(0.2, release.set).start()
        waited = wait_out_declared_background_writers(5.0)
    finally:
        release.set()
        writer.join(5.0)

    assert waited == ["core.ontogeny.experience"], "the fixture has to say what it waited for"
    assert experience.a_background_write_is_in_flight() is False


def test_a_subsystem_a_test_never_imported_is_never_consulted(monkeypatch):
    monkeypatch.delitem(sys.modules, "core.ontogeny.experience", raising=False)
    assert wait_out_declared_background_writers() == []
