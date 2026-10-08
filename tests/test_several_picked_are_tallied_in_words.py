"""Several things picked and done in turn end with a tally in words, not the machinery's step count.

LIVE 2026-10-08 three games' account closed with "That much is verified. The
rest did not complete: step 3 not ok (2/3 steps)."
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import _how_many_as_asked


@pytest.mark.unit
def test_none_one_and_all_are_said_as_words():
    assert _how_many_as_asked([{"had": True, "completed": False}] * 3) == "Of the three, none ended the way you asked."
    assert _how_many_as_asked([{"had": True, "completed": True}] * 3) == "All three ended the way you asked."
    assert _how_many_as_asked([{"had": True, "completed": True}, {"had": True}, {"had": False}]) == (
        "Of the three, one ended the way you asked; one could not be had anywhere.")


@pytest.mark.unit
def test_a_pursuit_that_ended_is_not_a_pick_done_as_asked():
    # LIVE 2026-10-08 three games, none won, each pursuit ended: "All three ended the way you asked."
    from core.skills.sovereign_browser_picking import _as_asked

    lost = _as_asked({"completed": True, "steps": [{"won": False, "completed": False}]})
    maker = _as_asked({"completed": True, "steps": [{"won": False, "completed": False, "the_ask_does_not_apply": True}]})
    picked = [{"had": True, **lost}, {"had": True, **lost}, {"had": True, **maker}]
    assert _how_many_as_asked(picked) == "Of the three, none ended the way you asked; for one, what you asked could not apply."


@pytest.mark.unit
def test_one_thing_is_not_tallied():
    assert _how_many_as_asked([{"had": True, "completed": False}]) == ""
    assert _how_many_as_asked(None) == ""
