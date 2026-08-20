"""A problem outside the template is still a problem to be checked.

LIVE, 2026-08-19. The bridge-and-torch puzzle — four people, times 1, 2, 7 and
10, at most two crossing, the torch must come back — classified as `generic`,
so the verifier-backed amplifier never saw it. She derived the correct
schedule and then stated the total as 19 where her own steps sum to 17.

Sampling and checking is exactly what catches that, and it was switched off by
vocabulary: every hint in the classifier is a keyword list, and a novel problem
uses none of those words. That is the definition of a task outside the
template, which is the case the amplifier matters most for.

Recognised by shape instead: several quantities, a rule constraining them, and
a definite question. Conservative, because amplification costs samples — a
chatty message that merely contains numbers must not trigger it.
"""

from __future__ import annotations

import pytest

from core.brain.reasoning_amplifier_v2 import is_amplifiable

PUZZLES = [
    "four people cross a bridge at night with one torch, 1 2 7 and 10 minutes "
    "each, at most two at a time and a pair moves at the slower pace. who "
    "crosses together, in order?",
    "a farmer has 12 sheep, 4 pens and must put an odd number in each pen with "
    "no pen empty. how many ways can she do it?",
    "three switches downstairs control three bulbs upstairs. you may go up only "
    "once. how do you tell which switch is which?",
    "you have two ropes that each burn for exactly one hour but not evenly. how "
    "do you measure forty five minutes?",
]

CONVERSATION = [
    "how are you feeling today",
    "i slept about 5 hours and had 2 coffees, feeling rough",
    "tell me a story about the sea",
    "what did i ask you about earlier today, before you restarted?",
    "read /tmp/x/accounts.py and tell me what close() does",
    "what's something you've genuinely changed your mind about since you started running?",
    "how do you feel about being restarted so often?",
]


@pytest.mark.parametrize("puzzle", PUZZLES)
def test_a_puzzle_earns_verification_whatever_words_it_uses(puzzle: str):
    assert is_amplifiable(puzzle) is not None


@pytest.mark.parametrize("message", CONVERSATION)
def test_conversation_does_not_pay_for_sampling(message: str):
    assert is_amplifiable(message) is None


def test_quantities_count_whether_or_not_they_are_digits():
    """The three-switches puzzle names none in figures and is no less a puzzle."""
    from core.brain.reasoning_amplifier_v2 import _looks_like_a_quantitative_puzzle

    assert _looks_like_a_quantitative_puzzle(PUZZLES[2])
    assert _looks_like_a_quantitative_puzzle(PUZZLES[3])


def test_a_puzzle_needs_a_constraint_not_just_numbers():
    """Numbers alone are a receipt, a date, a phone number."""
    from core.brain.reasoning_amplifier_v2 import _looks_like_a_quantitative_puzzle

    assert not _looks_like_a_quantitative_puzzle(
        "the meeting is at 3, there are 4 of us and it runs 90 minutes, see you there"
    )


def test_a_puzzle_needs_a_definite_question():
    from core.brain.reasoning_amplifier_v2 import _looks_like_a_quantitative_puzzle

    assert not _looks_like_a_quantitative_puzzle(
        "each of the 3 teams must submit 2 reports and every report needs 4 sections."
    )


def test_a_short_message_is_never_a_puzzle():
    from core.brain.reasoning_amplifier_v2 import _looks_like_a_quantitative_puzzle

    assert not _looks_like_a_quantitative_puzzle("2 + 2 = ?")
