"""A situation read a part at a time and put back together: where to act, and what each act comes to in the whole.

Bryan, 24 November 2026: "if this area gives me 10 possibilities, 4 lead to a negative outcome, 3 put me in a worse
position than not, 3 put me in a better position ... stitch together the outcomes of other areas and map out how this
will play out ... which put me in a worse position or give the opponent the option of a better position"; "if you know
a few, you may be able to deduce what other paths forward may look like" (core/cognition/what_each_area_comes_to.py).
Tested on worlds that are not boards.
"""
from __future__ import annotations

import pytest

from core.cognition.what_each_area_comes_to import (
    HERS,
    THEIRS,
    PartsSeen,
    Reading,
    put_together,
    read_part,
    threats,
    weigh_acts,
    what_it_changes,
    what_to_do,
    where_they_work,
)


def _a_switch(name: str, worth: float) -> Reading:
    """A part either side can take once: worth ``worth`` to her if she takes it, as much to them if they do."""
    def acts(state, side):
        return [side] if state == "open" else []

    def judge(state):
        return {"open": 0.0, HERS: worth, THEIRS: -worth}[state]

    return read_part(name, "open", acts=acts, after=lambda state, act: act, judge=judge, plies=1)


def test_the_parts_are_taken_most_urgent_first_and_the_whole_is_their_sum() -> None:
    parts = [_a_switch("small", 1.0), _a_switch("large", 5.0), _a_switch("middle", 3.0)]
    assert [part.urgency for part in parts] == [1.0, 5.0, 3.0]
    whole = put_together(parts, first=HERS)
    assert whole.order == (("large", HERS), ("middle", THEIRS), ("small", HERS))
    assert whole.comes_to == pytest.approx(5 - 3 + 1)
    assert put_together(parts, first=THEIRS).comes_to == pytest.approx(-5 + 3 - 1)


def test_an_act_in_a_part_that_is_not_the_most_urgent_leaves_the_other_side_the_one_that_is() -> None:
    parts = [_a_switch("small", 1.0), _a_switch("large", 5.0), _a_switch("middle", 3.0)]
    weighed = {w.part: w for w in weigh_acts(parts)}
    assert weighed["large"].whole == pytest.approx(5 - 3 + 1)
    assert weighed["small"].whole == pytest.approx(1 - 5 + 3)
    assert weighed["small"].leaves_them == ("large", 5.0)
    assert [w.part for w in weigh_acts(parts)][0] == "large"


def test_a_part_acting_in_first_costs_the_actor_is_left_as_it_stands_and_acting_there_is_worse_than_not() -> None:
    # Whoever acts here first gives the other side the better of it: neither side wants to.
    def acts(state, side):
        return [side] if state == "open" else []

    cold = read_part("cold", "open", acts=acts, after=lambda state, act: act,
                     judge=lambda state: {"open": 0.0, HERS: -2.0, THEIRS: 2.0}[state], plies=1)
    assert cold.urgency < 0
    hot = _a_switch("hot", 1.0)
    assert put_together([cold, hot], first=HERS).comes_to == pytest.approx(0.0 + 1.0)
    weighed = {w.part: w for w in weigh_acts([cold, hot])}
    assert weighed["cold"].worse_than_acting_nowhere and not weighed["hot"].worse_than_acting_nowhere


def test_a_grab_the_other_side_can_punish_is_seen_for_what_it_comes_to() -> None:
    # Grabbing stands at +3 until they answer it, and their answer leaves her at -4; holding back stands at +1.
    moves = {("here", HERS): ["grab", "hold"], ("grabbed", THEIRS): ["punish"]}
    leads = {"grab": "grabbed", "hold": "held", "punish": "punished"}
    stands = {"here": 0.0, "grabbed": 3.0, "held": 1.0, "punished": -4.0}
    part = read_part("one", "here", acts=lambda state, side: moves.get((state, side), []),
                     after=lambda state, act: leads[act], judge=stands.__getitem__, plies=2)
    assert part.acts[0] == ("hold", 1.0) and dict(part.acts)["grab"] == -4.0
    assert part.if_hers_first == 1.0


def test_ten_acts_are_told_apart_as_better_worse_and_losing() -> None:
    worth = [-10.0] * 4 + [-1.0] * 3 + [2.0] * 3
    part = read_part("ten", "here", acts=lambda state, side: list(range(10)) if state == "here" and side == HERS else [],
                     after=lambda state, act: act, judge=lambda state: 0.0 if state == "here" else worth[state],
                     plies=1)
    kinds = part.kinds_of_act(losing_at=-10.0)
    assert (len(kinds["losing"]), len(kinds["worse"]), len(kinds["better"])) == (4, 3, 3)


def test_a_part_not_looked_at_is_read_from_the_parts_like_it_and_says_so() -> None:
    seen = PartsSeen(most=10)
    seen.keep((0.0, 0.0), _a_switch("near", 4.0))
    seen.keep((10.0, 10.0), _a_switch("far", 0.0))
    like = seen.like("new", (1.0, 1.0))
    assert not like.looked and like.urgency > 3.5 and like.spread > 0
    same = seen.like("again", (0.0, 0.0))
    assert same.urgency == 4.0 and same.spread == 0.0 and not same.looked
    assert PartsSeen(most=10).like("none", (0.0,)) is None


def test_what_they_can_do_is_read_as_what_it_would_cost_her_worst_first() -> None:
    parts = [_a_switch("small", 1.0), _a_switch("large", 5.0)]
    found = threats(parts)
    assert [(t.part, t.costs_her) for t in found] == [("large", 5.0), ("small", 1.0)]


def test_where_they_keep_acting_and_what_they_make_hotter_is_where_they_are_going() -> None:
    before = [_a_switch("left", 1.0), _a_switch("right", 1.0)]
    after = [_a_switch("left", 1.0), _a_switch("right", 4.0)]
    going = where_they_work(["left", "right", "right"], before, after)
    assert going[0][0] == "right" and going[0][1] > 0.5
    changed = what_it_changes(before, after)
    assert set(changed) == {"right"} and changed["right"]["urgency"] == 3.0


def test_the_most_good_act_where_there_is_one_and_the_least_bad_where_there_is_not() -> None:
    assert what_to_do([_a_switch("large", 5.0), _a_switch("small", 1.0)]).most_good

    def acts(state, side):
        return [side] if state == "open" else []

    # Every act of hers leaves her worse than leaving things alone: she takes the one that costs least.
    def costly(name: str, cost: float) -> Reading:
        return read_part(name, "open", acts=acts, after=lambda state, act: act,
                         judge=lambda state: {"open": 0.0, HERS: -cost, THEIRS: cost}[state], plies=1)

    choice = what_to_do([costly("dear", 3.0), costly("cheap", 1.0)])
    assert not choice.most_good and choice.weighed.part == "cheap"


def test_what_they_can_do_after_each_act_and_her_answer_to_each_of_theirs_are_worked_out_before_they_act() -> None:
    """A take-away count: a pile of tokens, each side takes one or two in turn, and whoever takes the last wins. Worked
    out to the end, every line: what each of her takes lets them force, and her answer to each of theirs."""
    from core.cognition.what_each_area_comes_to import what_can_happen

    def acts(state, side):
        pile, _last = state
        return [n for n in (1, 2) if n <= pile]

    def after(state, take):
        pile, _last = state
        return pile - take, None

    # The side to act is whoever did not act last; the win is read where the pile is gone.
    def judge(state):
        return 0.0

    def acts_by_side(state, side):
        return [(side, n) for n in acts(state, side)]

    def after_by_side(state, act):
        side, take = act
        pile, _ = state
        return pile - take, side

    def judge_by_side(state):
        pile, last = state
        return 0.0 if pile else (1.0 if last == HERS else -1.0)

    found = what_can_happen("pile", (4, None), acts=acts_by_side, after=after_by_side, judge=judge_by_side,
                            plies_at_most=8, seconds=5.0, may_leave=False)
    assert found.deep == 8
    forced = {act[1]: worst for act, worst, _best, _n, _hurt in found.hers}
    # From four, taking one leaves three, from which she wins however they answer; taking two leaves two, which they
    # take and win.
    assert forced[1] == 1.0 and forced[2] == -1.0
    assert found.hers[0][0] == (HERS, 1)
    # Her answer to each of their takes, ready before they take: where they take two, leaving two, she takes both and
    # wins; where they take one, leaving three, nothing she does wins, and she knows it before they take.
    assert found.counter_to((THEIRS, 2)) == ((HERS, 2), 1.0)
    assert found.counter_to((THEIRS, 1))[1] == -1.0
