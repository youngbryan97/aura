"""A draft of hers that waits on her own yes is asked again once her source moves, and is not said to wait for Bryan (F16).

LIVE 22 November 2026, her changes page: two drafts "waits for Bryan". Neither did. Her mend of the subprocess
gateway's argument check had waited on her own gates since 17 November, after tests not its own failed, and her young
collection in the MLX worker's lifecycle since 20 November, after a restart stopped its outside check. A draft was asked
again only where what made it came back, a fault heard again or a cost measured again, and neither came back. Her own
yes takes a draft (Bryan, 9 and 18 November 2026), and only one that changes her own governance waits for him.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import her_goal_at_work as goals
from tests.test_her_goal_is_worked_by_her import her_source  # noqa: F401

pytestmark = pytest.mark.unit


def _offered(slug: str, files: list[str], at: float) -> goals.GoalRecord:
    record = goals.GoalRecord(goal=f"goal {slug}", slug=slug)
    record.status = goals.OFFERED
    record.offer = {"commit": f"{slug:0<40}"[:40], "files": files, "at": at}
    asyncio.run(goals.save(record))
    return record


def test_the_longest_waiting_draft_is_asked_again_once_her_source_moves(her_source, monkeypatch):
    from core.agency import mending_a_fault_of_hers as mending

    _offered("newer", ["core/notes.py"], at=200.0)
    _offered("older", ["core/notes.py"], at=100.0)
    _offered("governs", ["core/governance/will.py"], at=50.0)
    heads, asked = ["head-a"], []
    monkeypatch.setattr(goals, "source_head", lambda: heads[0])
    monkeypatch.setattr(goals, "ensure_workspace", lambda record: None)
    monkeypatch.setattr(goals, "take_it_herself", lambda slug: asked.append(slug) or {"ok": False, "why": "tests failed"})

    def ask():
        return asyncio.run(mending._ask_again_of_a_draft_that_waits())

    assert ask()["slug"] == "older", "the longest-waiting first"
    assert ask()["slug"] == "newer"
    assert ask() is None, "nothing is asked twice on the same source"
    heads[0] = "head-b"
    assert ask()["slug"] == "older", "her source moved"
    assert "governs" not in asked, "a draft that changes her governance waits for Bryan"


def test_her_changes_page_says_who_a_draft_waits_for(her_source):
    from core.agency.what_she_changed import _state

    assert _state(_offered("hers", ["core/notes.py"], at=1.0)) == "waits for her own yes"
    assert _state(_offered("his", ["core/governance/will.py"], at=1.0)) == "waits for Bryan"
