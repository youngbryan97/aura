"""A gesture is not said, in a reply as in a remark, however it runs on into what she says.

LIVE 8 November 2026, rehearsal 9: her reply on the episode she picked had "*settles in beside her screen as it starts
rolling.*...okay, and there we go." in it, and her thoughts after it opened "*letting the credits wash over without
turning away right away.*...and that's it."
"""
from __future__ import annotations

import inspect

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


def test_her_quick_replies_go_through_it():
    from core.brain import cognitive_engine_quick_reply as quick

    assert "without_gestures(said_once(" in inspect.getsource(quick)
