"""Asking for something to exist is asking for an effect.

LIVE, 2026-08-22, twice. "I have to present you to a funding panel in 10
minutes. Six slides, no fluff..." was handed code_repl, diagnose_repo and
quantum_lab. The model invented a tool called `create_slides`, and on the next
attempt one called `generate_slides`, and wrote one slide of six as prose.

Two gates, each with the right principle and too narrow a reader.

The effect ceiling's own docstring says "Asking for a page to exist is asking
for that effect" — and it asked `asks_to_build_software`. A deck is not
software, so every capability that produces a file was filtered out before it
could be offered. build_document was ranked FIRST by the selector and dropped
by the ceiling, exactly as build_app was on 2026-08-20.

And ranking reads a verb acting on an object. "Six slides, no fluff" is a noun
with a count in front of it, so it ranked nothing at all.
"""

from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine
from core.intent.artifact_request import asks_for_an_artifact
from core.intent.capability_selection import select_capabilities
from core.phases.response_contract import requested_effect_ceiling

PANEL = (
    "I have to present you to a funding panel in 10 minutes. Six slides, no fluff: "
    "what you are, what you can actually do today, your honest limitations."
)


@pytest.fixture(scope="module")
def skills():
    return CapabilityEngine().skills


def offered(text: str, skills) -> list[str]:
    ceiling, scopes = requested_effect_ceiling(text)
    return select_capabilities(text, skills, ceiling=ceiling, admissible_scopes=scopes, limit=4)


@pytest.mark.parametrize(
    "asked",
    [
        PANEL,
        "make me a deck for the funding panel",
        "write me a one-pager on it",
        "can you put together a short report",
        "give me a checklist for the move",
        "build me a little web app for tracking water",
    ],
)
def test_a_request_for_a_thing_raises_the_ceiling(asked: str):
    assert asks_for_an_artifact(asked), asked
    ceiling, scopes = requested_effect_ceiling(asked)
    assert "read_write_artifacts" in scopes, asked


@pytest.mark.parametrize(
    "asked",
    [
        "what is a deck?",
        "explain how slides work",
        "how are you feeling today?",
        "who founded Hugging Face?",
        "tell me about Anthropic the company",
    ],
)
def test_a_request_for_an_answer_does_not(asked: str):
    assert not asks_for_an_artifact(asked), asked
    _ceiling, scopes = requested_effect_ceiling(asked)
    assert "read_write_artifacts" not in scopes, asked


def test_the_builders_are_offered_for_a_thing_named_without_a_verb(skills):
    """Ranking needs a verb; this request is a noun with a count."""
    assert "build_document" in offered(PANEL, skills)


def test_the_right_builder_leads(skills):
    assert offered("make me a deck for the funding panel", skills)[0] == "build_document"
    assert offered("build me a little web app for tracking water", skills)[0] == "build_app"


def test_prose_requests_are_still_offered_nothing(skills):
    for asked in (
        "explain how dijkstra works and show the distance updates",
        "how are you feeling today?",
        "what do you think about consciousness?",
    ):
        assert offered(asked, skills) == [], asked


def test_the_producers_are_chosen_by_what_they_declare(skills):
    """No skill is named here, so one registered tomorrow joins by describing
    itself."""
    from core.intent.declared_capability import declared_vocabulary, producing_capabilities

    catalogue = {
        name: declared_vocabulary(name, str(getattr(meta, "description", "") or ""))
        for name, meta in skills.items()
        if getattr(meta, "enabled", True)
    }
    producers = producing_capabilities(catalogue)
    assert "build_document" in producers
    assert "build_app" in producers
    # Something that answers rather than produces is not in the set.
    assert "clock" not in producers
