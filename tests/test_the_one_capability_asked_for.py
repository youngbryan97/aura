"""A request that plainly names one capability's job is that capability's, by two readings that agree."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _engine():
    def skill(description, *patterns):
        return SimpleNamespace(description=description, trigger_patterns=list(patterns), enabled=True)

    return SimpleNamespace(skills={
        "repair_a_program": skill(
            "Fix a broken program somebody else wrote (a game, a web page, a script in a file) that does not "
            "work as it should: run it and watch what it does, find the bugs in its code, try their fixes on copies.",
            r"\b(?:fix|repair|mend)\b.*\b(?:game|program|page|script)\b|\bbroken\b",
        ),
        "file_operation": skill("Read, write, list or delete a file on disk.", r"\bread (?:the )?file\b"),
        "web_search": skill("Search the web for pages about a topic.", r"\bsearch\b"),
    })


@pytest.fixture
def engine(monkeypatch):
    from core.container import ServiceContainer

    fake = _engine()
    original = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get", lambda name, default=None: fake if name == "capability_engine" else original(name, default=default))
    return fake


def test_a_request_for_exactly_one_capability_s_job_is_that_capability(engine):
    from core.intent.capability_selection import the_one_asked_for

    asked = "The game at /tmp/pong.html is broken. Fix the program, run it and watch what it does, and find the bugs in its code."
    assert the_one_asked_for(asked, {"repair_a_program": {}}) == "repair_a_program"


def test_two_offered_is_a_choice_and_not_settled_here(engine):
    from core.intent.capability_selection import the_one_asked_for

    asked = "The game at /tmp/pong.html is broken. Fix the program, run it and watch what it does, and find the bugs in its code."
    assert the_one_asked_for(asked, {"repair_a_program": {}, "file_operation": {}}) is None


def test_its_own_phrases_must_match_too(engine):
    from core.intent.capability_selection import the_one_asked_for

    assert the_one_asked_for("a program that watches what it does and finds bugs in code is a nice idea", {"repair_a_program": {}}) is None
