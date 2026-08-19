"""The one question a person asks to find out whether something has a life.

LIVE, 2026-08-19. "how many turns have we had today, and how long have you
actually been awake across all your restarts?" was answered:

    That's a complex question. The number of turns depends on how you count —
    full conversations, partial exchanges within sessions?

Both halves were exact and both were on disk. ``continuity.json`` carries
``total_uptime_seconds`` and ``session_count`` — forty days across 1,523
sessions at that moment — and the episodic store holds every turn of the day
with a timestamp. The record is written on every shutdown and nothing read it
back to her.

A deflection is the worst available answer here: it reads as evasion about
exactly the thing being asked, when the truth is more impressive than any
hedge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core.self.lifetime import describe_lifetime, read_lifetime


@pytest.fixture
def continuity(tmp_path: Path) -> Path:
    path = tmp_path / "continuity.json"
    path.write_text(
        json.dumps(
            {
                "total_uptime_seconds": 3_490_422.47,
                "session_count": 1523,
                "last_shutdown": time.time() - 600,
                "last_shutdown_reason": "checkpoint",
            }
        )
    )
    return path


def test_the_cumulative_record_is_read(continuity: Path):
    lifetime = read_lifetime(continuity)
    assert lifetime is not None
    assert lifetime.session_count == 1523
    assert lifetime.total() == "40.4 days"


def test_a_missing_record_reads_as_nothing(tmp_path: Path):
    assert read_lifetime(tmp_path / "absent.json") is None


def test_an_empty_record_is_not_a_lifetime_of_zero(tmp_path: Path):
    path = tmp_path / "continuity.json"
    path.write_text(json.dumps({"total_uptime_seconds": 0, "session_count": 0}))
    assert read_lifetime(path) is None


def test_a_corrupt_record_does_not_break_the_turn(tmp_path: Path):
    path = tmp_path / "continuity.json"
    path.write_text("{not json")
    assert read_lifetime(path) is None


def test_the_answer_states_both_halves(monkeypatch, continuity: Path):
    """The question is usually asked as one: how much of me, how much today."""
    import core.self.lifetime as module

    monkeypatch.setattr(module, "_continuity_path", lambda: continuity)
    monkeypatch.setattr(module, "turns_today", lambda: 26)
    described = describe_lifetime()
    assert "40.4 days" in described
    assert "1523 sessions" in described
    assert "26 things" in described


def test_this_session_is_a_different_question():
    """Answering "how long this run?" with forty days is wrong.

    And wrong in the direction that sounds impressive, which is the worst
    direction for a claim about herself.
    """
    from core.brain.observable_registry import _matches_lifetime

    assert _matches_lifetime("how long have you been awake across all your restarts?")
    assert _matches_lifetime("how many turns have we had today?")
    assert not _matches_lifetime("how long have you been running this session?")
    assert not _matches_lifetime("how long have you been running right now?")


def test_nothing_is_served_when_nothing_is_recorded(monkeypatch, tmp_path: Path):
    from interface.routes.chat import _serve_lifetime
    import core.self.lifetime as module

    monkeypatch.setattr(module, "_continuity_path", lambda: tmp_path / "absent.json")
    assert _serve_lifetime("how long have you been alive?", "her own words") == "her own words"
