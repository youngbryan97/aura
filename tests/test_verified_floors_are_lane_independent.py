"""A verified answer must be reachable from every lane that composes a reply.

Live 2026-08-03: "show me a piece of your code you find interesting" was
answered "I couldn't get a clear enough answer together, and I'd rather say
that than hand you something thin" — while a real, correctly-cited, disk-read
excerpt sat one call away. The short phrasing answered correctly the whole
time, which made it look like a phrasing bug.

It was a lane boundary. The floors lived inside _direct_answer_floor on the
synthesis lane; that turn went to full cognition, whose draft the quality gate
filtered, and the last-resort composer apologised. Nothing on the cognition
lane could see them.

Saying "I have nothing" while holding something is the worst failure available
at that point, so this pins the shape rather than the one phrasing: the floors
are one shared function, and every lane consults it.
"""
from __future__ import annotations

import inspect

import pytest

from core.synthesis import verified_answer_floor

ANSWERABLE_WITHOUT_THE_MODEL = (
    "show me your code",
    "show me a piece of your code you find interesting",
    "what's behind your window?",
)


class TestOneDefinitionForBothLanes:
    @pytest.mark.parametrize("message", ANSWERABLE_WITHOUT_THE_MODEL)
    def test_both_lanes_return_the_same_answer(self, message):
        from interface.routes.chat import _verified_floor_answer

        shared = verified_answer_floor(message)
        assert shared.strip(), f"no verified answer for {message!r}"
        assert _verified_floor_answer(message) == shared.strip()

    def test_the_cognition_lane_names_no_individual_floor(self):
        """Naming one floor is how the next floor becomes invisible again."""
        from interface.routes import chat

        body = inspect.getsource(chat._verified_floor_answer)
        assert "verified_answer_floor" in body
        for individual in (
            "own_source_excerpt_floor",
            "occluded_screen_view_floor",
            "live_chat_diagnostic_floor",
        ):
            assert individual not in body, (
                f"{individual} named directly — add it to verified_answer_floor instead"
            )

    def test_the_synthesis_lane_goes_through_the_same_function(self):
        from core import synthesis

        body = inspect.getsource(synthesis._direct_answer_floor)
        assert "verified_answer_floor" in body

    def test_every_floor_is_inside_the_shared_function(self):
        from core import synthesis

        shared = inspect.getsource(synthesis.verified_answer_floor)
        for individual in (
            "own_source_excerpt_floor",
            "occluded_screen_view_floor",
            "live_chat_diagnostic_floor",
        ):
            assert individual in shared, f"{individual} is not reachable from both lanes"


class TestTheLastResortPrefersReadEvidence:
    @pytest.mark.parametrize("message", ANSWERABLE_WITHOUT_THE_MODEL)
    def test_a_degraded_turn_answers_instead_of_apologising(self, message):
        from interface.routes.chat import _build_degraded_live_reply

        reply = _build_degraded_live_reply({}, message, reason="filtered_draft")
        assert "couldn't get a clear enough answer" not in reply
        # What was read reaches the person, in the runtime's notice under its own heading.
        assert reply.strip().endswith(verified_answer_floor(message).strip())
        assert "What the runtime held for it:" in reply

    def test_a_turn_with_no_read_answer_still_degrades_honestly(self):
        from interface.routes.chat import _build_degraded_live_reply

        reply = _build_degraded_live_reply({}, "what is the weather", reason="filtered_draft")
        assert "Send it again" in reply


class TestTheFloorsStayHonest:
    def test_nothing_is_invented_for_an_unrelated_turn(self):
        for message in ("what is the weather", "tell me a story", ""):
            assert verified_answer_floor(message) == ""

    def test_whitespace_only_is_not_a_question(self):
        assert verified_answer_floor("   \n  ") == ""
