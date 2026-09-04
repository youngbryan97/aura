"""The moves after this one have to be moves about the situation after it.

A ranking is a judgement about alternatives at one moment. Sent as a sequence
it is four answers to the same question, executed in order.

LIVE 2026-09-04, driving the real 2048 app: every cycle committed to four
moves — down, up, left, right, in ranking order — because every prediction
"the view will be different" held and the commitment length is read off how
often predictions hold. Each key changed the whole board, so the second, third
and fourth were answers to a board that no longer existed. Sixty moves a game,
no rule ever formed, and the pair the learner was handed spanned four acts
while being labelled as one.

A named run keeps going while each step is still what she would choose from
the board the step before it leaves. That is the same judgement carried
forward, which is what committing without words claims to be — and unlike the
claim, it can be checked.
"""

from __future__ import annotations

from core.skills.screen_pursuit import a_run_she_can_carry

CHOICES = ["up", "down", "left", "right"]


def _boards(*moves: str) -> dict:
    """A tiny world: every act leads somewhere new, named by the path taken."""
    return {}


def _foresee(where, step):
    if where is None:
        return None
    return f"{where}/{step}"


def _would_pick(best: dict[str, str]):
    """What she would choose from each board, by name."""

    def pick(board, names):
        return best.get(board, "")

    return pick


def test_a_run_holds_while_every_step_is_what_she_would_choose():
    pick = _would_pick({"b": "left", "b/left": "down"})
    assert a_run_she_can_carry(
        ["left", "down"], "b", _foresee, pick, CHOICES, moved=9
    ) == ["left", "down"]


def test_it_stops_at_the_first_step_she_would_not_choose():
    pick = _would_pick({"b": "left", "b/left": "up"})
    assert a_run_she_can_carry(
        ["left", "down", "right"], "b", _foresee, pick, CHOICES, moved=9
    ) == ["left"]


def test_a_ranking_of_every_key_is_cut_to_the_one_she_meant():
    """down, up, left, right — the live case."""
    pick = _would_pick({"b": "up"})
    assert (
        a_run_she_can_carry(["up", "left", "right"], "b", _foresee, pick, CHOICES, moved=40)
        == ["up"]
    )


def test_a_world_whose_acts_change_it_gets_one_act_at_a_time_without_a_model():
    assert a_run_she_can_carry(["up", "left"], None, None, None, CHOICES, moved=12) == []


def test_a_page_where_nothing_has_moved_yet_still_tries_the_next_thing():
    """A key that may simply do nothing: the ranking IS the list to try."""
    assert a_run_she_can_carry(
        ["tab", "return"], None, None, None, CHOICES, moved=0
    ) == ["tab", "return"]


def test_a_step_that_changes_nothing_ends_the_run():
    pick = _would_pick({"b": "left", "b/left": "down"})

    def stuck(where, step):
        return where if step == "down" else f"{where}/{step}"

    assert a_run_she_can_carry(
        ["left", "down"], "b", stuck, pick, CHOICES, moved=9
    ) == ["left"]


def test_no_run_named_is_no_run_taken():
    assert a_run_she_can_carry([], "b", _foresee, _would_pick({}), CHOICES, moved=0) == []


def test_the_pursuit_checks_a_named_sequence_before_committing_to_it():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    named = source.index('getattr(made, "then", ())')
    assert "a_run_she_can_carry(" in source[named : named + 900]


def test_every_act_of_a_run_is_counted_however_the_run_was_made():
    """The guard that refuses to learn from several acts and one reading
    reads this number, and it was only ever set for the run she built
    herself."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index('expected["took"] = len(follow_on) + 1')
    before = source[:at]
    line_start = before.rindex("\n") + 1
    assert source[line_start:at] == " " * 8
