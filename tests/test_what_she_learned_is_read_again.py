"""Every world she has learned is read again by the code as it stands, and what no longer holds is named.

LIVE 23 November 2026 a change to how she learns a world passed every test, all of which start learning from nothing,
and the first world she went back to began, by her reading of her own record, with her already lost.
"""
from __future__ import annotations

import pytest

from core.runtime import what_she_learned
from core.runtime import what_she_learned_still_holds as still

pytestmark = pytest.mark.unit


@pytest.fixture
def kept(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    monkeypatch.setattr(still, "_REPLAYS", {})
    (tmp_path / "a-world.json").write_text('{"_kept_for": "a-world", "counts": {"left": 3}, "other": 1}')
    (tmp_path / "another-world.json").write_text('{"_kept_for": "another-world", "unread": 2}')
    return tmp_path


def test_a_part_that_no_longer_holds_is_named_with_its_world(kept) -> None:
    @still.replays("counts")
    def _counts(world, record):
        return [] if record["counts"].get("left") == 3 else ["the count was lost"]

    assert still.what_no_longer_holds() == ({}, 1)

    @still.replays("counts")
    def _counts_changed(world, record):
        return ["the count was lost"]

    found, read = still.what_no_longer_holds()
    assert read == 1 and found == {"a-world": ["counts: the count was lost"]}


def test_a_reader_that_raises_on_her_record_is_a_finding(kept) -> None:
    @still.replays("other")
    def _other(world, record):
        raise ValueError("a number was expected")

    found, _read = still.what_no_longer_holds(["a-world"])
    assert found == {"a-world": ["other: reading it raised ValueError: a number was expected"]}


def test_a_world_no_reader_knows_is_not_counted_as_read(kept) -> None:
    @still.replays("counts")
    def _counts(world, record):
        return []

    assert still.what_no_longer_holds(["another-world"]) == ({}, 0)


def test_a_screen_world_part_is_read_again_by_the_reader_her_pursuit_uses(kept) -> None:
    from core.skills.screen_pursuit import _read_back

    _read_back("moves", "core.perception.how_it_moves:HowItMoves")
    (kept / "a-screen.json").write_text('{"_kept_for": "a-screen", "moves": {}}')
    assert still.readers() == ["moves"]
    assert still.what_no_longer_holds(["a-screen"]) == ({}, 1)
