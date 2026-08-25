"""Evidence follows the MEANING of a request, not its wording.

Bryan, live 2026-08-04: "a lot of these requests are tied into specific
phrases and that shouldn't be the case. part of her reasoning has to include
general associations and a general understanding of what is being asked."

He had the evidence for it. "Which file in your repository does that
function live in?" reached her source and was answered correctly; "What
python module is that from" — the same question — missed a regex and was
answered from her weights. A keyword gate that misses leaves her blind to
something she can actually see.
"""
from __future__ import annotations

import pytest

from core.cognition import evidence_relevance as evidence_module
from core.cognition.evidence_relevance import (
    EXTERNAL_WORLD,
    OWN_SOURCE,
    PHYSICAL_PERCEPTION,
    SCREEN_PERCEPTION,
    assess_evidence_alignment,
    relevance,
    semantic_routing_available,
    wants_evidence,
)

pytestmark = pytest.mark.skipif(
    not semantic_routing_available(),
    reason="sentence-transformers unavailable; routing falls back to the lexical floor",
)


@pytest.mark.parametrize(
    "question",
    [
        "What python module is that from",
        "Which file in your repository does that function live in?",
        "Can you show me a snippet of your code that you're interested in?",
        "where can it be found?",
        # Phrasings that appear in no pattern anywhere in the codebase.
        "show me how you're actually built",
        "let me see a bit of what you're made of",
    ],
)
def test_a_question_about_her_code_finds_her_source(question: str) -> None:
    assert relevance(question, OWN_SOURCE) > 0.0, question
    assert wants_evidence(question, OWN_SOURCE)


@pytest.mark.parametrize(
    "question",
    [
        "Hey, Aura. Can you tell me what you see on the screen?",
        "what's on my screen right now?",
        "What's behind your window? Can you see what's underneath it?",
        "is there anything about UFC on my screen right now?",
        "what was that repo you saw?",
    ],
)
def test_a_question_about_the_screen_finds_the_perception(question: str) -> None:
    assert relevance(question, SCREEN_PERCEPTION) > 0.0, question
    assert wants_evidence(question, SCREEN_PERCEPTION)


@pytest.mark.parametrize(
    "question",
    [
        "Which of your senses can actually tell what is around you right now?",
        "Can you determine whether anyone else is physically here with me?",
        "Do you have a current reading of who is nearby?",
        "Establish from your present surroundings whether I am alone.",
    ],
)
def test_physical_perception_follows_semantics_not_trigger_phrases(question: str) -> None:
    assert relevance(question, PHYSICAL_PERCEPTION) > 0.0, question
    assert wants_evidence(question, PHYSICAL_PERCEPTION), question


@pytest.mark.parametrize(
    "question",
    ["what's 17 times 4?", "how are you feeling today?", "tell me a joke"],
)
def test_unrelated_turns_pull_no_evidence(question: str) -> None:
    assert not wants_evidence(question, OWN_SOURCE), question
    assert not wants_evidence(question, SCREEN_PERCEPTION), question
    assert not wants_evidence(question, PHYSICAL_PERCEPTION), question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who founded Hugging Face?", True),
        ("What did Mistral announce this week?", True),
        ("What is one subtle tradeoff in CPU architecture?", False),
        (
            "What is one subtle engineering tradeoff when migrating a long-lived AI system?",
            False,
        ),
    ],
)
def test_external_world_evidence_distinguishes_entities_from_concepts(
    question: str, expected: bool
) -> None:
    assert wants_evidence(question, EXTERNAL_WORLD) is expected


def test_query_to_evidence_alignment_separates_the_live_failure():
    aligned = assess_evidence_alignment(
        "What is one tradeoff in hybrid recurrent transformer architectures?",
        "Hybrid recurrent designs reduce KV-cache growth but make state recovery harder.",
    )
    shipping = assess_evidence_alignment(
        "What is one tradeoff in hybrid recurrent transformer architectures?",
        "Ocean Network Express provides cargo tracking and shipping schedules.",
    )

    assert aligned.measured and aligned.relevant
    assert shipping.measured and not shipping.relevant
    assert aligned.score is not None and shipping.score is not None
    assert aligned.score > shipping.score


def test_writing_new_code_is_not_a_question_about_her_own():
    """"Write me a python module" shares vocabulary and shares no intent."""
    assert relevance("write me a python module for sorting", OWN_SOURCE) < 0.0
    assert not wants_evidence("write me a python module for sorting", OWN_SOURCE)


def test_the_lexical_floor_can_add_but_never_veto():
    """Meaning wins; the pattern is a floor under it, not a gate over it."""
    assert wants_evidence(
        "zzz unparseable zzz", OWN_SOURCE, lexical_floor=lambda _text: True
    )
    # A floor that says no cannot suppress a clear semantic match.
    assert wants_evidence(
        "Which file in your repository does that function live in?",
        OWN_SOURCE,
        lexical_floor=lambda _text: False,
    )


def test_an_empty_request_asks_for_nothing():
    assert not wants_evidence("", OWN_SOURCE)
    assert relevance("", SCREEN_PERCEPTION) == 0.0


def test_semantic_caches_are_bound_to_one_embedding_space(monkeypatch):
    """Equal-width vectors from a replacement encoder are not comparable."""

    first = object()
    second = object()
    monkeypatch.setattr(evidence_module, "_CACHE_ENGINE_TOKEN", None)
    evidence_module._ANCHOR_CACHE.clear()
    evidence_module._REQUEST_CACHE.clear()
    evidence_module._ALIGNMENT_QUERY_CACHE.clear()

    evidence_module._bind_cache_to_engine(first)
    evidence_module._ANCHOR_CACHE["concept"] = [1.0]
    evidence_module._REQUEST_CACHE["request"] = [1.0]
    evidence_module._ALIGNMENT_QUERY_CACHE["query"] = [1.0]
    evidence_module._bind_cache_to_engine(first)
    assert evidence_module._ANCHOR_CACHE == {"concept": [1.0]}
    assert evidence_module._REQUEST_CACHE == {"request": [1.0]}
    assert evidence_module._ALIGNMENT_QUERY_CACHE == {"query": [1.0]}

    evidence_module._bind_cache_to_engine(second)
    assert evidence_module._ANCHOR_CACHE == {}
    assert evidence_module._REQUEST_CACHE == {}
    assert evidence_module._ALIGNMENT_QUERY_CACHE == {}


def test_alignment_boundary_separates_its_declared_calibration_cohort():
    for query, passage in evidence_module._ALIGNMENT_POSITIVES:
        verdict = assess_evidence_alignment(query, passage)
        assert verdict.measured and verdict.relevant, (query, verdict)
    for query, passage in evidence_module._ALIGNMENT_NEGATIVES:
        verdict = assess_evidence_alignment(query, passage)
        assert verdict.measured and not verdict.relevant, (query, verdict)


def test_a_screen_question_does_not_drag_in_her_source():
    """Both concepts score positive; only one of them is the question."""
    from interface.routes.chat import (
        _turn_may_concern_own_source,
        _turn_may_concern_perception,
    )

    for question in ("what's on my screen?", "what do you see right now?"):
        assert _turn_may_concern_perception(question), question
        assert not _turn_may_concern_own_source(question), question


def test_a_request_to_write_prose_is_not_a_request_to_look():
    """Live: this pulled a screen reading into a request for two sentences."""
    question = "Give me two concise sentences about reliable desktop tool use."
    assert not wants_evidence(question, SCREEN_PERCEPTION)
    assert not wants_evidence(question, OWN_SOURCE)


def test_a_question_that_spans_both_keeps_both():
    """A repo seen on a screen is genuinely both, and she needs both."""
    question = "what was that repo you saw?"
    assert wants_evidence(question, SCREEN_PERCEPTION)
    assert wants_evidence(question, OWN_SOURCE)
