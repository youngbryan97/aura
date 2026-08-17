"""A hedge is a decision to look something up, not a place to stop.

She holds an offline Wikipedia index, a web-search skill, and several checkers,
and none of them ran during ordinary conversation because nothing converted
"I'm not sure" into an action. A person who says that reaches for their phone.
"""

from __future__ import annotations

from core.knowledge.knowledge_gap import detect_knowledge_gap, gap_query


def test_an_uncertain_date_is_a_gap() -> None:
    gap = detect_knowledge_gap(
        "I think it's around 1969, but I'm not certain when Apollo 11 landed.",
        "when did apollo 11 land",
    )

    assert gap is not None
    assert "1969" in gap.query and "Apollo" in gap.query


def test_a_flat_admission_is_a_gap() -> None:
    gap = detect_knowledge_gap(
        "I do not know who wrote the Rust borrow checker originally.",
        "who wrote it",
    )

    assert gap is not None
    assert "Rust" in gap.query


def test_from_memory_is_a_gap() -> None:
    gap = detect_knowledge_gap(
        "Off the top of my head, Django was released around 2005.",
        "when was django released",
    )

    assert gap is not None
    assert "Django" in gap.query


def test_a_confident_answer_is_not_a_gap() -> None:
    assert detect_knowledge_gap("The capital of France is Paris.", "capital of france") is None


def test_a_hedged_opinion_is_not_a_gap() -> None:
    """No reference work settles taste. A lookup here is pure cost."""
    assert detect_knowledge_gap("I think you'd enjoy that film, honestly.", "should i watch it") is None


def test_uncertainty_about_the_user_is_not_a_gap() -> None:
    assert detect_knowledge_gap("I'm not sure how you're feeling about it.", "im upset") is None


def test_a_bare_hedge_with_nothing_to_look_up_is_not_a_gap() -> None:
    assert detect_knowledge_gap("I'm not sure I follow.", "huh?") is None


def test_contractions_do_not_poison_the_query() -> None:
    """"it's" and "I'm" survive a plain stopword check and wreck the search."""
    query = gap_query("I think it's around 1969, but I'm not certain when Apollo 11 landed.")

    assert "it's" not in query.lower()
    assert "i'm" not in query.lower()


def test_the_query_stays_short() -> None:
    gap = detect_knowledge_gap(
        "I'm not sure whether the Antikythera mechanism was built in Rhodes or "
        "Corinth or Syracuse during the Hellenistic period honestly.",
        "where was it built",
    )

    assert gap is not None
    assert len(gap.query.split()) <= 6


def test_garbage_input_is_safe() -> None:
    for value in (None, "", 0, [], {"a": 1}):
        assert detect_knowledge_gap(value, value) is None
