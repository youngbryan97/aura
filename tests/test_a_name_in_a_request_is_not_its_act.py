"""A capability named by a word in a request is not the one it asks for when its meaning clearly asks for another.

LIVE 19 November 2026: "Can you look up what the latest stable release of Python is right now?" matched code_repl's
declaration and its trigger patterns on the word "Python", so the tool loop was handed code_repl alone and ran Python
to scrape pages, twice, and her answer was written again. Her reading by meaning had chosen a reference search. Two
readings of the words agreeing now still ask her reading by meaning; where it clearly means another capability, that
one is offered. Where it agrees, or cannot tell, the words stand.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

LIVE = "Can you look up what the latest stable release of Python is right now?"


@pytest.fixture
def catalogue(monkeypatch):
    from core.container import ServiceContainer

    skills = {
        "code_repl": SimpleNamespace(description="Execute Python code in a sandbox and return its output.", enabled=True,
                                     trigger_patterns=[r"\bpython\b"]),
        "local_reference_search": SimpleNamespace(description="Look something up in reference sources.", enabled=True,
                                                  trigger_patterns=[]),
    }
    fake = SimpleNamespace(skills=skills)
    original = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get",
                        lambda name, default=None: fake if name == "capability_engine" else original(name, default=default))
    monkeypatch.setattr("core.intent.capability_selection.the_one_asked_for", lambda text, offered: next(iter(offered)))
    return skills


def _meaning_says(monkeypatch, meant):
    monkeypatch.setattr("core.brain.requests_by_meaning.meant_capability", lambda text, skills: meant)


def test_where_the_meaning_clearly_asks_for_another_capability_that_one_is_offered(catalogue, monkeypatch):
    from core.brain.her_reading_of_a_request import her_reading

    _meaning_says(monkeypatch, "local_reference_search")
    required, _ceiling, _ = asyncio.run(her_reading(LIVE, ["code_repl"], "sandboxed_compute", frozenset()))
    assert required[0] == "local_reference_search", required


@pytest.mark.parametrize("meant", [None, "code_repl"])
def test_where_the_meaning_agrees_or_cannot_tell_the_words_stand(catalogue, monkeypatch, meant):
    from core.brain.her_reading_of_a_request import her_reading

    _meaning_says(monkeypatch, meant)
    asked = "Run this Python snippet and tell me what it prints."
    required, ceiling, scopes = asyncio.run(her_reading(asked, ["code_repl"], "sandboxed_compute", frozenset()))
    assert (required, ceiling, scopes) == (["code_repl"], "sandboxed_compute", frozenset())


ASKED_TO_RUN = ("Write a Python function that groups a list of file paths by extension, then run it on a few example "
                "paths and show me the output.")


@pytest.mark.parametrize(("work_on_code", "offered"), [
    (True, ["file_operation", "code_repl"]),
    (None, ["file_operation"]),
    (False, ["file_operation"]),
])
def test_work_on_code_keeps_the_runner_its_words_found(catalogue, monkeypatch, work_on_code, offered):
    """LIVE 20 November 2026: meaning chose file_operation by "file paths", and nothing could run the code asked for."""
    from core.brain.her_reading_of_a_request import her_reading
    from core.intent import asks_for_code_work

    catalogue["file_operation"] = SimpleNamespace(description="Read, write and list files on disk.", enabled=True,
                                                  trigger_patterns=[])
    monkeypatch.setattr("core.intent.capability_selection.the_one_asked_for", lambda text, offered: None)
    monkeypatch.setattr("core.together.taking_it_up.asked_to_do_something", _yes)
    monkeypatch.setattr(asks_for_code_work, "asks_for_code_work", lambda text: work_on_code)
    _meaning_says(monkeypatch, "file_operation")
    required, _ceiling, _ = asyncio.run(her_reading(ASKED_TO_RUN, ["file_operation", "code_repl", "build_document"],
                                                    "sandboxed_compute", frozenset()))
    assert required == offered


async def _yes(_text):
    return True

