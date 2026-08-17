"""The local corpus must be consulted before answering a question about the world.

LIVE 2026-08-17. Asked to explain correlation versus causation, she produced
"correlation means two things happen together without any clear relationship
between them", which is not what correlation means.

The corpus that contains the right answer was already on this disk, already
indexed, and already had a reader. The response path already had a grounding
channel. No wire ran between them: grounding was only ever built from a
web_search or sovereign_browser skill result, so a question answered from
weights alone was never grounded at all.
"""

from __future__ import annotations

from core.knowledge.corpus_grounding import (
    CorpusGrounding,
    _topical_query,
    corpus_grounding_for,
    is_corpus_groundable,
)


class _Hit:
    def __init__(self, title: str, text: str) -> None:
        self.title = title
        self.text = text


class _Store:
    """Records what it was asked, so the query shape itself is testable."""

    def __init__(self, hits: list[_Hit] | None = None) -> None:
        self.hits = hits or []
        self.queries: list[str] = []

    def search(self, query, limit=5, deadline_s=1.0):
        self.queries.append(query)
        return self.hits[:limit]


# ── eligibility ──────────────────────────────────────────────────────────────

def test_a_question_about_the_world_is_eligible() -> None:
    assert is_corpus_groundable(
        "Explain the difference between correlation and causation to a smart 12 year old."
    )


def test_asking_how_she_is_doing_is_not() -> None:
    """A corpus cannot answer this, and injecting an article would hurt."""
    assert not is_corpus_groundable("how are you doing right now?")


def test_recall_about_the_conversation_is_not() -> None:
    assert not is_corpus_groundable("what did I ask you first today?")


def test_an_action_request_is_not() -> None:
    assert not is_corpus_groundable("open my notes folder and save the draft")


def test_a_greeting_is_not() -> None:
    assert not is_corpus_groundable("hey")


# ── the query shape, which is what made it work ──────────────────────────────

def test_scaffolding_is_stripped_from_the_query() -> None:
    """The whole sentence AND-matches nothing over 7M pages and times out.

    Measured: the raw sentence returned 0 passages in 384ms (the full
    deadline); the extracted terms returned 3 passages in 105ms.
    """
    query = _topical_query(
        "Explain the difference between correlation and causation to a smart 12 year old."
    )

    assert query == "correlation causation"


def test_the_store_receives_the_topical_query_not_the_sentence() -> None:
    store = _Store([_Hit("Correlation does not imply causation", "The phrase ...")])

    corpus_grounding_for(
        "Explain the difference between correlation and causation to a smart 12 year old.",
        store=store,
    )

    assert store.queries == ["correlation causation"]


def test_the_query_stays_short() -> None:
    """A long AND-query over an encyclopaedia matches nothing."""
    long_question = (
        "what is the difference between a transformer and a recurrent neural "
        "network and a convolutional network and an autoencoder and a diffusion model"
    )

    assert len(_topical_query(long_question).split()) <= 6


# ── retrieval behaviour ──────────────────────────────────────────────────────

def test_passages_come_back_when_the_corpus_has_them() -> None:
    store = _Store(
        [_Hit("Correlation does not imply causation", "The phrase refers to ...")]
    )

    grounding = corpus_grounding_for("what is correlation versus causation", store=store)

    assert grounding.grounded
    assert "Correlation does not imply causation" in grounding.render()[0]


def test_an_ineligible_turn_never_touches_the_store() -> None:
    """Latency matters: a conversational turn must not pay for a lookup."""
    store = _Store([_Hit("X", "y")])

    grounding = corpus_grounding_for("how are you doing right now?", store=store)

    assert store.queries == []
    assert not grounding.grounded


def test_a_raising_store_yields_an_empty_grounding() -> None:
    class _Broken:
        @staticmethod
        def search(query, limit=5, deadline_s=1.0):
            raise OSError("corpus unreadable")

    grounding = corpus_grounding_for("what is a confounding variable", store=_Broken())

    assert not grounding.grounded


def test_empty_hits_render_nothing() -> None:
    grounding = CorpusGrounding(query="q", passages=())

    assert grounding.render() == []
    assert not grounding.grounded


def test_blank_passages_are_dropped() -> None:
    store = _Store([_Hit("Title", "   "), _Hit("Real", "actual text")])

    grounding = corpus_grounding_for("what is entropy in physics", store=store)

    assert [t for t, _ in grounding.passages] == ["Real"]


def test_garbage_input_is_safe() -> None:
    for value in (None, "", 0, [], {"a": 1}):
        assert not corpus_grounding_for(value).grounded
        assert not is_corpus_groundable(value)
