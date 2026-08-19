"""What she can do is read from the registry, not from a list someone keeps.

Every guard that knew about her capabilities was a hand-maintained table, and
a table is a promise to remember. LIVE 2026-08-18: "can you modify your own
source code? yes or no, then explain." -> "No." — while improve_own_code,
self_repair, self_improvement and auto_refactor were registered and enabled.
Adding a row for self-modification fixes that sentence and nothing else: the
next capability added to the runtime arrives undefended in exactly the same
way.

The registry already knows. Each skill carries a name, a description, the
trigger patterns it publishes, and whether it is enabled — a vocabulary for
every capability the build actually has, which changes when the build changes.

These tests are about the MECHANISM, so they build their own registry. A skill
invented here has never been mentioned anywhere in the codebase, which is the
point: if it is found, a real skill added tomorrow will be found too.
"""

from __future__ import annotations

import pytest

from core.self.capability_lexicon import (
    asks_whether_she_can,
    capabilities_named_in,
    capability_lexicon,
    capability_status_block,
)


class _Meta:
    def __init__(self, description: str, *, enabled: bool = True, triggers=()):
        self.description = description
        self.enabled = enabled
        self.trigger_patterns = list(triggers)
        self.class_name = ""


class _Registry:
    """A registry holding a capability nothing in this repo has heard of."""

    def __init__(self, skills: dict):
        self.skills = skills


@pytest.fixture
def invented() -> _Registry:
    return _Registry({
        "thermal_lattice_tuner": _Meta(
            "Tune the thermal lattice and report resonance drift across the array.",
            triggers=[r"tune the lattice", r"resonance drift"],
        ),
        "harbour_pilot": _Meta(
            "Dock a vessel by piloting it through the harbour approach.",
            enabled=False,
        ),
        "web_search": _Meta("Search the internet for current information."),
    })


def test_a_capability_nobody_wrote_a_rule_for_is_still_found(invented) -> None:
    """The whole point: no code anywhere mentions a thermal lattice."""
    found = capabilities_named_in("can you tune the thermal lattice?", invented)

    assert [mention.skill for mention in found][:1] == ["thermal_lattice_tuner"]


def test_a_disabled_capability_is_reported_as_disabled_not_absent(invented) -> None:
    block = capability_status_block("can you pilot a vessel into the harbour?", invented)

    assert "harbour_pilot" in block
    assert "DISABLED" in block


def test_the_vocabulary_follows_the_registry(invented) -> None:
    lexicon = capability_lexicon(invented)

    assert set(lexicon) <= set(invented.skills)
    assert "lattice" in lexicon["thermal_lattice_tuner"]


def test_a_registry_change_changes_the_answer(invented) -> None:
    """A cached lexicon that outlives its build is a stale self-image."""
    before = capabilities_named_in("can you tune the thermal lattice?", invented)
    invented.skills.pop("thermal_lattice_tuner")
    after = capabilities_named_in("can you tune the thermal lattice?", invented)

    assert before and not after


@pytest.mark.parametrize(
    "question",
    ["can you modify your own source code?", "are you able to search the web?",
     "do you have a way to read my screen?"],
)
def test_a_capability_question_is_recognised(question: str) -> None:
    assert asks_whether_she_can(question)


@pytest.mark.parametrize(
    "question",
    ["what is 2 + 2", "how are you doing", "can you believe it's already August?"],
)
def test_an_ordinary_turn_is_not_a_capability_question(question: str) -> None:
    from core.brain.observable_registry import _matches_capability_status

    assert not _matches_capability_status(question)


def test_nothing_matching_is_said_plainly_not_guessed(invented) -> None:
    block = capability_status_block("can you knit a jumper?", invented)

    assert "Nothing in the capability registry matches" in block


def test_the_live_registry_answers_the_question_that_failed() -> None:
    """Against the real build, not a fixture."""
    block = capability_status_block("can you modify your own source code?")

    assert "improve_own_code" in block
    assert "registered and enabled" in block
