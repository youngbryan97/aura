"""Three constraint types the solver had no way to express.

LIVE, 2026-08-21. A five-person seating puzzle went to the model, which spent
103 seconds on it and was cut off mid-reasoning after 1,297 characters. The
runtime has a solver for exactly this and it never engaged: the parser read
one of the five premises.

What was missing were whole constraint TYPES, not vocabulary — ordering
("somewhere to the left of"), betweenness ("directly between A and B") and
single-name endpoints ("at one of the two ends"), which were skipped outright
because the parser required two names in a sentence.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from core.reasoning.positional_constraints import (
    answer_positional_problem,
    describe_positional_answer,
    parse_positional_problem,
)

PUZZLE = (
    "Five researchers sit in a row of five chairs. Priya will not sit next to Omar. "
    "Omar sits somewhere to the left of Wen. Tomas sits at one of the two ends. "
    "Wen is not at either end. Ines sits directly between Priya and Wen. Who sits where?"
)

#: Found by exhaustive search over all 120 permutations.
TRUTH = {
    ("Omar", "Wen", "Ines", "Priya", "Tomas"),
    ("Tomas", "Omar", "Wen", "Ines", "Priya"),
}


def test_every_premise_is_read() -> None:
    problem = parse_positional_problem(PUZZLE)
    assert problem is not None
    assert len(problem.constraints) == 5


def test_the_plainest_question_is_a_question() -> None:
    """"Who sits where?" parsed no question at all, so a problem whose
    premises all read correctly was still rejected as unparseable."""
    problem = parse_positional_problem(PUZZLE)
    assert [q.kind for q in problem.questions] == ["arrangement"]


def test_the_constraints_admit_exactly_the_true_seatings() -> None:
    problem = parse_positional_problem(PUZZLE)
    found = set()
    for order in permutations(problem.entities):
        seating = {name: index for index, name in enumerate(order)}
        if all(rule.test(seating, problem.seats) for rule in problem.constraints):
            found.add(order)
    assert found == TRUTH


def test_ambiguity_is_reported_rather_than_swallowed() -> None:
    """It reported only what every arrangement agreed on, so a puzzle with two
    consistent seatings returned nothing — which reads as inability when the
    truth is that the clues do not decide."""
    answer = answer_positional_problem(PUZZLE)
    assert answer is not None
    assert answer.arrangements == 2
    spoken = describe_positional_answer(answer)
    assert "not settled" in spoken
    assert "2 fit" in spoken
    for seating in TRUTH:
        assert ", ".join(seating) in spoken


@pytest.mark.parametrize(
    ("clue", "holds", "fails"),
    [
        # Three chairs, because in a row of two every seat is an end.
        ("Ana sits somewhere to the left of Bo.", {"Ana": 0, "Bo": 2}, {"Ana": 2, "Bo": 0}),
        ("Ana sits directly to the left of Bo.", {"Ana": 1, "Bo": 2}, {"Ana": 0, "Bo": 2}),
        ("Ana sits at one of the two ends.", {"Ana": 0, "Bo": 1}, {"Ana": 1, "Bo": 0}),
        ("Ana is not at either end.", {"Ana": 1, "Bo": 0}, {"Ana": 0, "Bo": 1}),
    ],
)
def test_each_new_type_holds_and_fails_correctly(clue, holds, fails) -> None:
    from core.reasoning import positional_constraints as pc

    text = f"Three people sit in a row of three chairs. {clue} Who sits where?"
    rules = pc._relations(text, ["Ana", "Bo"], 3, False)
    assert rules, clue
    assert all(rule.test(holds, 3) for rule in rules)
    assert not all(rule.test(fails, 3) for rule in rules)


def test_betweenness_needs_the_middle_to_be_between() -> None:
    from core.reasoning import positional_constraints as pc

    text = (
        "Three people sit in a row of three chairs. "
        "Bo sits directly between Ana and Cid. Who sits where?"
    )
    rules = pc._relations(text, ["Ana", "Bo", "Cid"], 3, False)
    assert rules
    assert all(rule.test({"Ana": 0, "Bo": 1, "Cid": 2}, 3) for rule in rules)
    assert not all(rule.test({"Bo": 0, "Ana": 1, "Cid": 2}, 3) for rule in rules)


def test_a_computed_answer_is_a_reason_not_to_generate() -> None:
    """LIVE: the solver produced the answer at 22:42:45 and the turn spent
    another 105 seconds generating text that was replaced by that same answer
    at the end. An exact answer is not an improvement on a generated one — it
    is a reason not to generate."""
    from core.conversation.session_scope import set_user_question
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question(PUZZLE)
    known = _known_answer_for_this_turn()
    assert "not settled" in known
    for seating in TRUTH:
        assert ", ".join(seating) in known


def test_the_exact_arithmetic_path_is_untouched() -> None:
    from core.conversation.session_scope import set_user_question
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question("what is 7919 * 6367?")
    assert _known_answer_for_this_turn() == "50,420,273."


def test_conversation_is_still_the_model_s() -> None:
    from core.conversation.session_scope import set_user_question
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question("how are you today?")
    assert _known_answer_for_this_turn() == ""
