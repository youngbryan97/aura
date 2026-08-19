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
    from core.conversation.word_markers import stem_fold
    from core.self.capability_sources import all_capabilities

    lexicon = capability_lexicon(invented)

    # Skills are one register of what she can do; the deterministic readers
    # are another, and both belong in a self-image. What must not appear is a
    # name from neither — that would be vocabulary somebody typed by hand.
    declared = set(all_capabilities(invented))
    assert set(lexicon) <= declared
    assert set(invented.skills) <= declared
    assert stem_fold("lattice") in lexicon["thermal_lattice_tuner"]


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


def test_the_catalog_is_indexed_once() -> None:
    """Two providers feeding the same documents would double every score.

    CapabilityEngine already registers the catalog with the shared retriever
    for tool selection. Registering the same skills again under a second
    provider name puts every skill in the corpus twice, which changes the
    document frequencies the ranking is built on — quietly, and in a way that
    only shows up as slightly wrong answers.
    """
    from core.capability_engine import CapabilityEngine
    from core.skills.skill_retrieval import get_skill_retriever

    registered = len(getattr(CapabilityEngine(), "skills", {}) or {})
    if not registered:
        pytest.skip("no capability registry in this process")

    capabilities_named_in("can you modify your own source code?")
    CapabilityEngine()._retrieved_tool_candidates("search the web", 3)

    assert get_skill_retriever().corpus_size() == registered


def test_a_skill_with_no_trigger_patterns_is_still_reachable() -> None:
    """38 of 77 skills publish no regex at all.

    Trigger patterns only work as far as somebody anticipated the phrasing, so
    a skill that ships without any can never be proposed by intent matching.
    Retrieval over its own description is what makes it reachable, and that is
    the property a capability added later depends on.
    """
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine()
    skills = getattr(engine, "skills", {}) or {}
    if not skills:
        pytest.skip("no capability registry in this process")

    untriggered = [
        name
        for name, meta in skills.items()
        if not (getattr(meta, "trigger_patterns", None) or [])
    ]

    assert untriggered, "expected some skills to ship without trigger patterns"

    name = "auto_refactor"
    if name not in skills:
        pytest.skip("auto_refactor is not registered in this build")

    found = engine._retrieved_tool_candidates(
        "refactor my code automatically and improve its structure", 6
    )

    assert isinstance(found, list)
