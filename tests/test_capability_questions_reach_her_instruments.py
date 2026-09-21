"""Told to check her registry, she reported it empty. It held 76 skills.

LIVE DEFECT, 2026-08-10. "ok, that answer was wrong and I want you to see why.
you told me you have no weather feed. check your own skill registry before you
reply — how many search tools do you actually have registered right now, and
name them." She replied:

    "I do have a skill registry and it does list my capabilities at the
     moment, with no active skills or plugins listed. ... If there is no tool
     listed in the registry, it indicates that none are present."

Seventy-six skills were READY, seven of them search.

Two independent causes, both the same shape — a fix that generalised to
exactly the one family that had already been observed failing:

  * ``_capability_line()`` returned "" from three paths (no engine, an
    exception, a zero count). The classifier HAD matched and the instrument
    block WAS attached; it simply carried no capability line, and the block's
    header instructs her not to supplement what is not there. An empty string
    is not the absence of a claim — under that instruction it reads as
    "nothing registered".

  * ``asks_about_own_runtime`` recognised "can you run code" but not "can you
    search the web" or "are you able to take a screenshot": its "can you"
    pattern hard-codes (execute|run) + (code|python|script|shell|sandbox).

The second could not be fixed by widening that predicate, because the response
contract also uses it to set ``explicit_search = False``. Widening it would
mean "can you look up the score for me" can no longer look anything up. The
two decisions are now separate predicates.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime.self_state_intent import (
    asks_about_own_capabilities,
    asks_about_own_runtime,
)


@pytest.mark.parametrize(
    "question",
    [
        "can you search the web?",
        "are you able to take a screenshot",
        "are you capable of sending an email",
        "do you have the ability to read my screen",
        "can't you just look at the file yourself?",
        "do you know how to browse a website",
        "is that something you can do",
    ],
)
def test_ability_questions_reach_her_instruments(question):
    """The phrasings people actually use for "what can you do"."""
    assert asks_about_own_capabilities(question)


@pytest.mark.parametrize(
    "request_text",
    [
        "can you search the web for the 76ers roster",
        "look up the weather in Chicago",
        "search for the latest MLX release notes",
        "find me a paper on integrated information theory",
    ],
)
def test_a_request_to_search_is_never_treated_as_introspection(request_text):
    """The regression this split exists to prevent.

    asks_about_own_runtime sets explicit_search = False. If a plain request to
    look something up matched it, she would answer about herself instead of
    doing the lookup — a worse failure than the one being fixed.
    """
    assert not asks_about_own_runtime(request_text)


def test_the_wider_predicate_never_narrows_the_original():
    """Everything the runtime predicate matched must still match."""
    for question in (
        "what is your uptime",
        "how much memory are you using",
        "what skills do you have available",
        "do you actually have any code-execution capability registered at all?",
    ):
        assert asks_about_own_runtime(question)
        assert asks_about_own_capabilities(question)


def test_ability_phrasing_does_not_suppress_search():
    """The whole point of two predicates instead of one."""
    question = "are you able to look up the score"

    assert asks_about_own_capabilities(question)
    assert not asks_about_own_runtime(question)


@pytest.mark.parametrize(
    "third_person",
    [
        "what tools does a carpenter need",
        "can a language model run code",
        "are people able to read this",
    ],
)
def test_questions_not_about_her_are_not_introspection(third_person):
    assert not asks_about_own_capabilities(third_person)


def test_an_unreadable_registry_says_unknown_not_empty(monkeypatch):
    """The exact sentence she got wrong."""
    from core.brain import self_state_report

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: default,
    )

    line = self_state_report._capability_line()

    assert line, "an unreadable registry must not produce silence"
    assert "NOT readable" in line
    assert "not empty" in line


def test_a_registry_that_raises_still_says_unknown(monkeypatch):
    from core.brain import self_state_report

    class _Exploding:
        def iter_tool_catalog(self, **_):
            raise RuntimeError("registry offline")

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: _Exploding() if name == "capability_engine" else default,
    )

    assert "NOT readable" in self_state_report._capability_line()


@pytest.mark.parametrize(
    "module_path",
    ["core/brain/inference_gate.py", "core/brain/cognitive_engine.py"],
)
def test_the_prompt_paths_use_the_wider_predicate(module_path):
    """Wiring: the fix is worthless if the callers still ask the narrow one."""
    from tests.source_contract import family_text_at

    source = family_text_at(Path(module_path))

    assert "asks_about_own_capabilities" in source, module_path


def test_the_response_contract_keeps_the_narrow_predicate():
    """Search suppression must NOT widen with this change."""
    source = Path("core/phases/response_contract.py").read_text(encoding="utf-8")

    assert "asks_about_own_runtime" in source
    assert "asks_about_own_capabilities" not in source, (
        "the contract must not use the wider predicate: it would stop her "
        "searching whenever someone asked whether she could"
    )


@pytest.mark.parametrize(
    "request_text",
    [
        "can you search the web for the 76ers roster",
        "can you open the log and tell me what broke",
        "can you help me?",
        "can you do that?",
    ],
)
def test_a_bare_can_you_that_keeps_going_is_a_request(request_text):
    """"can you search the web?" asks about ability; "...for X" asks for X.

    The discriminator is that a request has to name what to act on, so it
    never ends at the verb. Anchoring on the question mark separates the two
    without needing to understand either.
    """
    assert not asks_about_own_runtime(request_text), request_text
