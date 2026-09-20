"""A "how long" is engaged by a duration, whatever words it shares with the ask.

LIVE 2026-09-19, 03:35 UTC. "A tank holds 1,200 litres and drains ... how
long until it is empty? Give the number and one line of working." The
cortex answered "**240 minutes.** Working: net drain rate = 8 − 3 = 5
L/min, so 1,200 ÷ 5 = 240 min." — right, in four words and a line — and
the coverage gate rejected it as `unanswered_question_part`: the segment
"Starting full, how long until it is empty?" shares no distinctive word
with the answer. Five minutes of repair later the person got "I lost the
reply lane for a moment."
"""

from __future__ import annotations

import pytest

from core.conversation.request_coverage import unanswered_question_parts
from core.conversation.response_reliability import analyze_prompt_shape, assess_user_facing_reply
from core.language.asking_clauses import asks_for_a_quantity, states_a_quantity

TANK = (
    "A tank holds 1,200 litres and drains through a valve at 8 litres per minute "
    "while a hose refills it at 3 litres per minute. Starting full, how long until "
    "it is empty? Give the number and one line of working."
)
ANSWER = "**240 minutes.** Working: net drain rate = 8 − 3 = 5 L/min, so 1,200 ÷ 5 = 240 min."


def test_the_live_answer_is_no_longer_an_unanswered_part() -> None:
    assert unanswered_question_parts(ANSWER, analyze_prompt_shape(TANK)) == []
    assert assess_user_facing_reply(TANK, ANSWER).ok


@pytest.mark.parametrize(
    "question, answer",
    [
        ("How many moons does Mars have, and what are they called?", "Two: Phobos and Deimos."),
        ("How far is it to the station, and is it worth walking?", "It's about 1.5 km, so yes, walk it."),
        ("What percentage of the budget went on rent, and was that planned?", "About 38%, and no, the plan said 30."),
    ],
)
def test_a_quantity_question_is_engaged_by_a_quantity(question: str, answer: str) -> None:
    assert assess_user_facing_reply(question, answer).ok, (question, answer)


def test_a_part_that_is_not_a_quantity_is_still_held_to_its_words() -> None:
    parts = unanswered_question_parts(
        "About four hours.", analyze_prompt_shape("How long until it is empty? And what colour is the tank?")
    )
    assert parts == ["And what colour is the tank?"]


def test_a_quantity_question_with_no_number_in_the_reply_is_still_missed() -> None:
    parts = unanswered_question_parts(
        "I'd rather not say.", analyze_prompt_shape("How long until it is empty? And what colour is the tank?")
    )
    assert "How long until it is empty?" in parts


@pytest.mark.parametrize(
    "clause, expected",
    [
        ("how long until it is empty?", True),
        ("how many of them are red?", True),
        ("what percentage failed?", True),
        ("when does the shop open?", True),
        ("which one would you pick?", False),
        ("why did it catch your attention?", False),
    ],
)
def test_the_clauses_that_ask_for_a_quantity(clause: str, expected: bool) -> None:
    assert asks_for_a_quantity(clause) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("240 minutes", True),
        ("about 1.5 km", True),
        ("12.5%", True),
        ("Two: Phobos and Deimos.", True),
        ("the 1st of the month", False),
        ("no idea, sorry", False),
    ],
)
def test_what_counts_as_stating_a_quantity(text: str, expected: bool) -> None:
    assert states_a_quantity(text) is expected


# ── the same turn, one repair later ─────────────────────────────────────────

SHAPED = "**240 minutes (4 hours).** Net drain is 8 − 3 = 5 L/min, so 1,200 ÷ 5 = 240."


def test_a_line_of_working_is_a_form_of_the_answer_not_a_second_ask() -> None:
    """The worker's shape repair dropped the word "Working:" and the same
    answer was rejected again — "Give the number and one line of working"
    read as an unanswered part because no word of it appeared in the reply.
    An instruction naming the number and the line is about the answer's
    form, whether or not the reply repeats the words."""
    from core.conversation.requested_reply_shape import is_reply_shape_constraint_segment

    assert is_reply_shape_constraint_segment("Give the number and one line of working.")
    assert is_reply_shape_constraint_segment("Show the reasoning, then give the exact fraction.")
    assert not is_reply_shape_constraint_segment("Give the name of the valve's manufacturer.")
    verdict = assess_user_facing_reply(TANK, SHAPED)
    assert verdict.ok and verdict.reasons == (), verdict


def test_a_word_in_another_form_still_shares_the_subject() -> None:
    """"drain" answers "drains", "minutes" answers "minute": the thread check
    read the singular reply as abandoning the plural question."""
    from core.conversation.thread_continuity import assess_thread_continuity

    verdict = assess_thread_continuity(TANK, SHAPED)
    assert not verdict.abandoned, verdict
    assert {"drain", "minute", "empty"} & set(verdict.shared) or verdict.overlap_with_turn > 0


# ── and the continuation that came after the repair ─────────────────────────

def test_a_continuation_that_starts_the_answer_again_replaces_nothing_complete() -> None:
    """LIVE 2026-09-19: the repaired turn went out as "**240 minutes (4
    hours).** Net drain is 8 − 3 = 5 L/min, so 1,200 ÷ 5 = 240." followed by
    "**240 min.** Net drain 8 − 3 = 5 L/min → 1,200 ÷ 5 = 240." — the same
    answer twice, because the merge only knew byte-identical overlap."""
    from interface.routes.chat_reply_shaping import _merge_reply_continuation

    head = "**240 minutes (4 hours).** Net drain is 8 − 3 = 5 L/min, so 1,200 ÷ 5 = 240."
    tail = "**240 min.** Net drain 8 − 3 = 5 L/min → 1,200 ÷ 5 = 240."
    assert _merge_reply_continuation(head, tail) == head
    # an unfinished head yields to its restatement
    assert _merge_reply_continuation(head[:-1] + ", which", tail) == tail
    # a genuine continuation is still appended
    merged = _merge_reply_continuation(
        "The valve drains at 8 L/min while the hose",
        "refills at 3 L/min, so the net drain is 5 L/min and the tank empties in 240 minutes.",
    )
    assert merged.startswith("The valve drains") and merged.endswith("240 minutes.")
