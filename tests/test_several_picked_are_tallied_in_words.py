"""Several things picked and done in turn end with a tally in words, not the machinery's step count.

LIVE 2026-10-08 three games' account closed with "That much is verified. The
rest did not complete: step 3 not ok (2/3 steps)."
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import _how_many_as_asked


@pytest.mark.unit
def test_none_one_and_all_are_said_as_words():
    assert _how_many_as_asked([{"had": True, "completed": False}] * 3) == (
        "Of the three picked: three I did not win in the time each had.")
    assert _how_many_as_asked([{"had": True, "completed": True}] * 3) == "All three ended the way you asked: I won each."
    assert _how_many_as_asked([{"had": True, "completed": True}, {"had": True}, {"had": False}]) == (
        "Of the three picked: I won one; one I did not win in the time it had; and one I could not find anywhere.")


@pytest.mark.unit
def test_a_pursuit_that_ended_is_not_a_pick_done_as_asked():
    # LIVE 2026-10-08 three games, none won, each pursuit ended: "All three ended the way you asked."
    from core.skills.sovereign_browser_picking import _as_asked

    lost = _as_asked({"completed": True, "steps": [{"won": False, "completed": False}]})
    maker = _as_asked({"completed": True, "steps": [{"won": False, "completed": False, "the_ask_does_not_apply": True}]})
    picked = [{"had": True, **lost}, {"had": True, **lost}, {"had": True, **maker}]
    assert _how_many_as_asked(picked) == (
        "Of the three picked: one had nothing to win; and two I did not win in the time each had.")


@pytest.mark.unit
def test_which_ended_how_is_said_so_a_listener_can_follow():
    """LIVE 2026-10-10 "Of the four, one ended the way you asked; for one, what you asked could not apply, and one could
    not be had anywhere": which was which, and the fourth, left to the listener."""
    picked = [{"had": False}, {"had": True, "as_asked": True, "could_not_apply": True},
              {"had": True, "as_asked": False}, {"had": True, "as_asked": False}]
    assert _how_many_as_asked(picked) == (
        "Of the four picked: one had nothing to win, and I did as you said for it; two I did not win in the time each "
        "had; and one I could not find anywhere.")


@pytest.mark.unit
def test_one_thing_is_not_tallied():
    assert _how_many_as_asked([{"had": True, "completed": False}]) == ""
    assert _how_many_as_asked(None) == ""


@pytest.mark.unit
def test_a_game_played_for_its_best_says_what_the_best_was():
    """LIVE 2026-10-10 "...so I played it as many times as you said and kept my best", and not what it was."""
    from core.skills.sovereign_browser_picking import _with_the_best

    said = "played, and not won; there is nothing to win in it, so I played it as many times as you said and kept my best"
    assert _with_the_best(said, {"steps": [{"best_score": 522}]}).endswith("and my best was 522")
    assert _with_the_best(said, {"steps": [{}]}) == said
