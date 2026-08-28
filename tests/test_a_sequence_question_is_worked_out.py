"""The induction machinery meets a person.

It could learn a transformation from a few examples, keep it, compose with it
and carry it to the next problem, and none of that ever met a live turn: the
architecture had the mechanism and the agent did not use it. Its only consumers
were its own battery, a comparison tool and these tests.

Where the rule accounts for every example shown there is nothing to generate,
the same as a seating arrangement or a product of two numbers.
"""

from __future__ import annotations

import pytest

from core.cognition.sequence_induction import (
    answer_sequence_question,
    read_sequence_question,
)


def test_a_reversal_is_worked_out_and_the_rule_is_said() -> None:
    asked = (
        "Work out the rule from these examples and apply it.\n\n"
        "[3, 4, 5, 6, 7] becomes [7, 6, 5, 4, 3]\n"
        "[10, 11, 12] becomes [12, 11, 10]\n\n"
        "What does [20, 21, 22, 23] become?"
    )
    answered = answer_sequence_question(asked)
    assert answered.startswith("[23, 22, 21, 20]")
    assert "n-1-i" in answered


@pytest.mark.parametrize(
    ("asked", "starts"),
    [
        (
            "[1,2,3,4] -> [3,4,1,2] and [5,6,7,8] -> [7,8,5,6]. What about [9,10,11,12]?",
            "[11, 12, 9, 10]",
        ),
        (
            "[a, b, c, d] becomes [a, d, c, b]. [p, q, r, s] becomes [p, s, r, q]. "
            "So [w, x, y, z] becomes?",
            "['w', 'z', 'y', 'x']",
        ),
    ],
)
def test_other_shapes_and_other_cells(asked: str, starts: str) -> None:
    assert answer_sequence_question(asked).startswith(starts)


def test_single_values_are_left_alone() -> None:
    """A relation between numbers is not a rearrangement of positions."""

    assert read_sequence_question("45 becomes 15. 28 becomes 14. What does 91 become?") is None
    assert answer_sequence_question("45 becomes 15. 28 becomes 14. What does 91 become?") == ""


def test_prose_with_lists_in_it_is_not_a_question() -> None:
    assert answer_sequence_question(
        "I have a list [1,2,3] and another [4,5,6] and I like them both."
    ) == ""
    assert answer_sequence_question("what's the weather like") == ""


def test_examples_with_no_rule_in_them_answer_nothing() -> None:
    """Read as a question, and still silent, because nothing explains it."""

    asked = "[1,2,3] becomes [9,4,7]. [4,5,6] becomes [2,8,1]. What does [7,8,9] become?"
    assert read_sequence_question(asked) is not None
    assert answer_sequence_question(asked) == ""


def test_a_pair_must_be_joined_by_something_meaning_becomes() -> None:
    assert read_sequence_question("[1,2,3] [3,2,1] [4,5,6]") is None


def test_the_live_route_consults_it() -> None:
    from pathlib import Path

    body = Path("interface/routes/chat.py").read_text()
    assert "_serve_worked_out_sequence(user_message, corrected)" in body
    start = body.index("def _serve_worked_out_sequence(")
    window = body[start : start + 1800]
    assert "answer_sequence_question" in window


def test_one_turn_makes_the_next_one_possible(tmp_path, monkeypatch) -> None:
    """The loop, end to end: experience, representation, reuse.

    LIVE, 2026-08-28, two consecutive turns in the window. The first showed a
    two-deep composition and was answered. The second showed a three-deep one —
    UNREACHABLE from a blank language however many examples are offered — and
    was answered [36, 34, 33, 32, 31, 30, 35], naming all three parts, because
    the turn before it had left the two-deep shape behind.
    """

    import core.cognition.sequence_induction as seam

    store = tmp_path / "relation_language.json"
    monkeypatch.setattr(seam, "_language_path", lambda: store)

    def mirror(row):
        return tuple(reversed(row))

    def rot1(row):
        return row[1:] + row[:1]

    def ends(row):
        out = list(row)
        out[0], out[-1] = out[-1], out[0]
        return tuple(out)

    def two(row):
        return rot1(mirror(row))

    def three(row):
        return ends(rot1(mirror(row)))

    def asked(rule, lengths, query_length):
        parts = []
        for length in lengths:
            row = tuple(range(1, length + 1))
            parts.append(f"{list(row)} becomes {list(rule(row))}")
        query = tuple(range(30, 30 + query_length))
        parts.append(f"What does {list(query)} become?")
        return ". ".join(parts), rule(query)

    deep_question, deep_answer = asked(three, (5, 6), 7)

    # Cold: the three-deep shape is out of reach.
    assert seam.answer_sequence_question(deep_question) == ""

    # One ordinary turn, showing the two-deep shape.
    shallow_question, shallow_answer = asked(two, (5, 6), 7)
    served = seam.answer_sequence_question(shallow_question)
    assert served.startswith(str(list(shallow_answer)))
    assert store.exists(), "the shape has to outlive the turn that learned it"

    # And now the harder question is answerable, in a fresh read of the store.
    served = seam.answer_sequence_question(deep_question)
    assert served.startswith(str(list(deep_answer)))
    assert "then" in served, "the rule it names is a composition"
