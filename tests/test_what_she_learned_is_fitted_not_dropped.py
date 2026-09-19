"""A record too long to keep is made shorter, not thrown away.

Live, 2026-09-18: "what she learned about '2048-game' is too big to keep
(124122)" at the end of an eighty-eight-move run, and the next run began from
what the run before that had known. The longest list loses its older half
until the record fits, because these are records appended as she goes.
"""

from __future__ import annotations

import json

import pytest

from core.runtime import what_she_learned


@pytest.fixture
def kept_in(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    return tmp_path


def test_a_long_record_is_kept_with_its_newest_part(kept_in, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 2_000)
    record = {"moves": {"record": [f"pair {i}" for i in range(400)]}, "rule": "slides and combines"}
    assert what_she_learned.remember("a board", record) is True
    back = what_she_learned.recall("a board")
    assert back["rule"] == "slides and combines"
    kept = back["moves"]["record"]
    assert kept and kept[-1] == "pair 399"
    assert "pair 0" not in kept
    assert len(json.dumps(back)) <= 2_000


def test_short_lists_are_never_cut_to_make_room(kept_in, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 200)
    record = {"down_at": [0.1, 0.2, 0.3, 0.4], "note": "x" * 400}
    assert what_she_learned.remember("a board", record) is False
