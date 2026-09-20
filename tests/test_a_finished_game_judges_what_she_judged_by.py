"""What she judges by is played out when a game ends, not only when a run does.

LIVE 2026-09-20: a run asked to play until the 2048 tile carried two invented
measures that had already been measured to cost three games in four. The
rehearsal that drops them runs at the end of the run, and that run would not
end for hours, so a whole demo was played by measures she had the evidence to
drop after the first game.
"""
from __future__ import annotations

from tests.source_support import inlined_function_source
from tests.source_contract import function_containing

PURSUIT = "core/skills/screen_pursuit.py"


def test_the_run_installs_what_to_do_when_a_game_ends():
    from core.skills import screen_pursuit

    _name, body = function_containing(screen_pursuit, 'pending["when_a_game_ends"] = ')
    assert "_judge_what_she_judges_by_in_her_model" in body


def test_a_confirmed_restart_calls_it():
    from core.skills import screen_pursuit_decision

    _name, body = function_containing(screen_pursuit_decision, 'responds["state"].began_again()')
    assert 'pending.get("when_a_game_ends")' in body
    # Only after the restart is confirmed by the screen, never on the click.
    assert body.index("the restart did not take") < body.index('pending.get("when_a_game_ends")')


def test_the_run_end_still_rehearses_and_forgets_the_hook():
    from core.skills import screen_pursuit

    _name, body = function_containing(screen_pursuit, 'pending.pop("when_a_game_ends", None)')
    assert "_judge_what_she_judges_by_in_her_model(" in body


def test_a_rehearsal_that_hurts_is_let_go_of():
    """The mechanism itself, unchanged: what does worse is dropped."""
    body = inlined_function_source(PURSUIT, "_judge_what_she_judges_by_in_her_model")
    assert "rehearsed.hurts()" in body and "forget(name)" in body
