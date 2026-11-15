"""A gesture is not said, in a reply as in a remark, however it runs on into what she says.

LIVE 8 November 2026, rehearsal 9: her reply on the episode she picked had "*settles in beside her screen as it starts
rolling.*...okay, and there we go." in it, and her thoughts after it opened "*letting the credits wash over without
turning away right away.*...and that's it."
"""
from __future__ import annotations

import pytest

from core.skills.watching_along import without_asides, without_gestures


def test_a_gesture_run_straight_on_into_her_words_is_left_off():
    said = "*letting the credits wash over without turning away right away.*...and that's it. Just names on black."
    assert without_asides(said) == "And that's it. Just names on black."


def test_a_reply_loses_its_gestures_and_keeps_its_emphasis():
    reply = ("Four minutes ten seconds of cheerful animals.\n\n"
             "*settles in beside her screen as it starts rolling.*...okay, and there we go. It's that *bright*.\n\n"
             "*leans back.*\n\n"
             "You've got me curious who goes first.")
    assert without_gestures(reply) == ("Four minutes ten seconds of cheerful animals.\n\n"
                                       "Okay, and there we go. It's that *bright*.\n\n"
                                       "You've got me curious who goes first.")


def test_stressed_words_and_exclaimed_asides_are_still_said():
    for said in ("*This* is the pivot. That little moment.", "Oh — *the restroom.* Of course it's the restroom.",
                 "*one hundred and eighty times its own body weight?* Wow."):
        assert without_gestures(said) == said


@pytest.mark.asyncio
async def test_her_quick_replies_go_through_it(monkeypatch):
    """A quick reply her model drafts with a gesture in it reaches the person without the gesture."""
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.phases import kept_procedures_in_a_turn

    class Router:
        async def think(self, **kwargs):
            return "*leans back.*\n\nYou've got me curious who goes first."

        def get_last_generation_metadata(self):
            return {"surface_control_receipt": {"generation_stop_reason": "eos", "surface_quality_gate_reasons": []}}

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    async def nothing_kept(objective, *, router, origin, tier):
        return None

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())
    monkeypatch.setattr(kept_procedures_in_a_turn, "kept_answer_for", nothing_kept)
    request = "Who do you think goes first?"
    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        request, ThinkingMode.FAST, "user",
        {"desktop_quick_reply_contract": True, "visible_user_message": request, "max_tokens": 512}, timeout_s=30.0,
    )
    assert thought.content == "You've got me curious who goes first."
