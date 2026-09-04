"""In a world she has just arrived in, the unknown is her own acts.

Everything else this measures needs a state she can read and rules that
disagree about it. A grid is worked out from what moves, and what moves under
one act is wherever that act puts things — so arriving with no grid, no rule
and no reason to vary, she takes the same act until the run ends.

LIVE 2026-09-04: two hundred seconds of the same key, no grid ever formed, and
nothing anywhere said why.
"""

from __future__ import annotations

from core.agency.looking_ahead import worth_finding_out

ACTS = ["up", "down", "left", "right"]


class _SaysNothing:
    """A model with no opinion, which is what a fresh world gives her."""

    def what_this_would_settle(self, state, action):
        return 0.0


def test_the_acts_she_has_not_taken_come_first():
    told = worth_finding_out(_SaysNothing(), None, ACTS, None, never_tried=ACTS)
    assert set(told) == set(ACTS)
    assert len(set(told.values())) == 1, "none of them settles more than another"


def test_it_empties_as_she_takes_them():
    told = worth_finding_out(_SaysNothing(), None, ACTS, None, never_tried=["left"])
    assert set(told) == {"left"}


def test_nothing_is_left_once_they_have_all_been_taken():
    assert worth_finding_out(_SaysNothing(), None, ACTS, None, never_tried=[]) == {}


def test_an_act_she_could_not_take_here_is_not_offered():
    told = worth_finding_out(
        _SaysNothing(), None, ["up", "down"], None, never_tried=["up", "escape"]
    )
    assert set(told) == {"up"}


def test_a_model_with_something_to_say_is_still_asked_once_they_are_all_tried():
    class _Splits:
        def what_this_would_settle(self, state, action):
            return 1.0 if action == "left" else 0.0

    told = worth_finding_out(_Splits(), object(), ACTS, None, never_tried=[])
    assert told.get("left", 0.0) > 0.0


def test_the_pursuit_names_the_acts_she_has_not_taken():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index("telling = worth_finding_out(")
    nearby = source[at : at + 600]
    assert "never_tried=[" in nearby
    assert 'responds["state"].tried' in nearby
