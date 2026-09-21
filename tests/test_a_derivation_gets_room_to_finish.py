"""An answer cut off before its conclusion is not a partial answer.

Live, 2026-07-27. Asked when a second train catches the first and how far from
the station, she worked it correctly and stopped here:

    "5. Calculate where they meet.
     - The first train has been traveling for 3:00pm + 2.25 "

Recorded as ``desktop_quick_reply_midsentence_cutoff`` and trimmed back to the
last complete sentence, which is the right salvage — but the number was in the
sentence that never arrived, so the user got working and no answer. The same
budget cut a recall answer at "Probably just read".

The budget was 512 tokens because the turn came in on the quick lane. How much
room an answer needs is a property of the question: a two-part derivation is a
two-part derivation whichever lane carries it. The extended band already
existed and was gated behind ``require_full_foreground_mind_reply``, which the
quick lane does not set.

The quick lane still exists for latency, but capacity follows the requested
work. Natural EOS keeps ordinary answers short without clipping compound ones.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.cognitive_engine import _turn_wants_a_derivation
from tests.source_contract import family_text_at

SOURCE = Path("core/brain/cognitive_engine.py")


@pytest.mark.parametrize(
    "question",
    [
        "A train leaves at 2:15pm going 60mph. Another leaves the same station "
        "at 3:00pm going 80mph on the same track. When does the second catch "
        "the first, and how far from the station?",
        "Work out how many marbles are left and show your working.",
        "Walk me through how you got that.",
        "Calculate the compound interest over five years.",
        "How long would it take, and how much would it cost?",
    ],
)
def test_a_question_that_needs_working_is_recognised(question: str) -> None:
    assert _turn_wants_a_derivation(question)


@pytest.mark.parametrize(
    "question",
    [
        "Morning. What's it actually like in there right now?",
        "What was the very first thing I asked you in this conversation?",
        "What is 17 times 23?",
        "hey",
        "",
        # Conversation, not derivation. Treating "and why" as working to be
        # shown lengthened ordinary turns for nothing.
        "If you could put one song on right now, what would it be and why that one?",
        "Which would you choose, and why?",
        "Is a river the same river over time? I'm curious how you reason about it.",
    ],
)
def test_an_ordinary_turn_keeps_the_conversational_budget(question: str) -> None:
    """Over-claiming here would slow every turn to pay for a few."""
    assert not _turn_wants_a_derivation(question)


def test_a_very_long_message_is_not_treated_as_a_derivation() -> None:
    """A pasted document is not a request to derive anything."""
    assert not _turn_wants_a_derivation("and how " * 400)


def test_the_shape_is_consulted_independently_of_the_lane() -> None:
    """The bug was the extended band being reachable only from the full lane."""
    src = family_text_at(SOURCE)
    assert "shape_wants_room = bool(" in src
    assert (
        'extended_full_mind_reply = bool(\n'
        '            context.get("require_full_foreground_mind_reply", False) '
        'and shape_wants_room\n'
        '        )'
    ) in src


def test_the_quick_lane_uses_the_structural_answer_floor() -> None:
    src = family_text_at(SOURCE)
    assert "elif shape_wants_room:" in src
    assert "max_tokens = max(896, structural_answer_floor, min(max_tokens, 4096))" in src
    assert "max_tokens = max(1024, structural_answer_floor, min(max_tokens, 4096))" in src


def test_the_conversational_floor_is_unchanged_for_everything_else() -> None:
    src = family_text_at(SOURCE)
    assert "max_tokens = max(512, min(max_tokens, 1024))" in src


def test_tight_contracts_still_win() -> None:
    """Status and inventory answers stay short; they are answered, not derived."""
    src = family_text_at(SOURCE)
    # Anchor on the clamp chain itself. The old anchor also matched the
    # request_timeout chain further down, which silently moved the slice.
    clamp = src[src.index("max_tokens = max(128, min(max_tokens, 256))") :]
    clamp = clamp[: clamp.index("request_timeout =")]
    assert clamp.index("elif capability_inventory_contract") < clamp.index(
        "elif shape_wants_room:"
    ), "a status or inventory answer must clamp before the derivation band widens it"


# ── The reason for the budget travels with it ──────────────────────────────

GATE = Path("core/brain/inference_gate.py")


def test_the_engine_says_why_it_asked_for_more() -> None:
    """A number alone cannot survive a pressure cap; a reason can."""
    assert '"reply_needs_room": shape_wants_room,' in family_text_at(SOURCE)


def test_the_starvation_floor_answers_the_caller_not_a_constant() -> None:
    """Measured live: caller asked 896, pressure cut it to 459, the flat floor
    lifted it to 512, and the derivation stopped at "- The" in step 5 of 5."""
    src = family_text_at(GATE)
    assert 'needs_room = bool(context.get("reply_needs_room", False))' in src
    assert "AURA_FOREGROUND_CHAT_DERIVATION_FLOOR_TOKENS" in src
    assert "1024 if needs_room else 512" in src


def test_the_floor_never_exceeds_what_was_asked_for() -> None:
    """The floor stops starvation; it does not hand out budget nobody wanted."""
    src = family_text_at(GATE)
    block = src[src.index("needs_room = bool(") :]
    block = block[: block.index("if max_tokens < starvation_floor")]
    assert "min(\n                    requested_budget," in block
