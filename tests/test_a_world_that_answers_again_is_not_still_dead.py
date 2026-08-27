"""Starting again clears the verdict that nothing answers.

Measured live on 2026-08-26: she noticed a finished game, decided the attempt
was over, clicked New Game — and then made no move at all for the rest of the
run, because while that verdict stood she was only ever offered the ways out.
"""

from __future__ import annotations

from core.perception.where_it_responds import Responsive


def _dead() -> Responsive:
    state = Responsive()
    state.unanswered = Responsive.DEAD_AFTER
    return state


def test_a_dead_world_says_so():
    assert _dead().nothing_answers()


def test_starting_again_clears_it():
    state = _dead()
    state.began_again()
    assert not state.nothing_answers()


def test_where_things_happen_still_holds_across_a_restart():
    state = _dead()
    state.answered = {(50, 50): 6}
    state.regardless = {(50, 50): 1}
    state.effective = 6
    state.idle = 2
    state.began_again()
    assert state.answered == {(50, 50): 6}
    assert state.effective == 6
    assert state.settled()


def test_it_can_die_again_afterwards():
    state = _dead()
    state.began_again()
    state.unanswered = Responsive.DEAD_AFTER
    assert state.nothing_answers()


# ── and deciding to start again is not starting again ────────────────────

from pathlib import Path  # noqa: E402

SOURCE = Path("core/skills/screen_pursuit.py").read_text()
RESTART = SOURCE[SOURCE.index("async def begin_again") : SOURCE.index("async def begin_again") + 1600]


def test_the_verdict_is_not_cleared_when_she_merely_decides_to_restart():
    """A click that landed on nothing left her believing the world was fresh."""
    decides = SOURCE.index('intending["value"] = START_OVER')
    clears = SOURCE.index('responds["state"].began_again()')
    clicks = SOURCE.index("clicked = await click_normalized(")
    assert decides < clicks < clears


def test_the_screen_is_read_again_before_anything_is_believed():
    assert "after = await observe()" in RESTART
    assert "now_showing.strip() == was_showing.strip()" in RESTART


def test_a_restart_that_did_not_take_is_reported_as_not_taking():
    assert "the restart did not take" in RESTART
    assert "return False" in RESTART


def test_and_it_is_only_counted_when_it_worked():
    counted = RESTART.index('restarts["count"] += 1')
    checked = RESTART.index("now_showing.strip() == was_showing.strip()")
    assert checked < counted
