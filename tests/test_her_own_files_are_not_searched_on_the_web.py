"""A question about her own files is read on her disk, and the web is not searched for it (U07).

LIVE 20 November 2026, typed into her window: "Find the two largest Python files under core/brain by line count, count
their lines by actually running something, and tell me by what percentage the larger one is bigger than the smaller."
The desktop route's own search check said no, because core/brain is on her disk. Her world evidence then searched the
web for the whole message anyway, through the other ways a question reaches outside: nothing it read bore on her
files, and her answer was held up by it. The check that says a request names something on her disk now speaks for
both.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit

HER_FILES = (
    "Find the two largest Python files under core/brain by line count, count their lines by actually running "
    "something, and tell me by what percentage the larger one is bigger than the smaller."
)
THE_WORLDS = "Who designed the Kaseya Center, and what did it cost to build?"


@pytest.mark.parametrize(("message", "on_her_disk"), [
    (HER_FILES, True),
    ("How many Python files are under core/consciousness in your source tree?", True),
    ("Read the file CONTRIBUTING.md and tell me the first rule it states.", True),
    (THE_WORLDS, False),
    ("What's the latest stable release of Python?", False),
])
def test_what_names_a_place_on_her_disk_is_read_from_the_request(message, on_her_disk):
    from interface.routes.chat_desktop_evidence import _its_evidence_is_on_her_disk

    assert _its_evidence_is_on_her_disk(message) is on_her_disk


def _the_search_her_world_evidence_was_given(monkeypatch, message: str):
    import core.conversation.what_the_world_says as world
    import interface.routes.chat_desktop_evidence as evidence_route

    given: dict[str, object] = {}

    async def gathered(*_args, **kwargs):
        given.update(kwargs)
        return world.WorldEvidence()

    async def checked(*_args, **_kwargs):
        return []

    async def nothing_earlier(**_kwargs):
        return []

    monkeypatch.setattr(world, "gather_world_evidence", gathered)
    monkeypatch.setattr(world, "her_check_reads", checked)
    monkeypatch.setattr(evidence_route._chat_memory_state, "_recent_completed_conversation_exchanges", nothing_earlier)
    monkeypatch.setattr(evidence_route, "_should_collect_desktop_required_search_evidence", lambda _m: (False, "", None))
    asyncio.run(evidence_route._what_the_world_says_for_the_turn(message, message, session_id="s"))
    assert "search" in given, "her world evidence was gathered"
    return given["search"]


def test_her_own_files_are_not_searched_on_the_web(monkeypatch):
    assert _the_search_her_world_evidence_was_given(monkeypatch, HER_FILES) is None


def test_a_question_about_the_world_can_still_be_searched(monkeypatch):
    assert _the_search_her_world_evidence_was_given(monkeypatch, THE_WORLDS) is not None
