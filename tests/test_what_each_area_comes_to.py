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
    forced = {path.act[1]: path.for_certain for path in found.hers}
    # From four, taking one leaves three, from which she wins however they answer; taking two leaves two, which they
    # take and win.
    assert forced[1] == 1.0 and forced[2] == -1.0
    assert found.hers[0].act == (HERS, 1) and found.complete and found.hers[0].settled and not found.hers[1].settled
    # Her answer to each of their takes, ready before they take: where they take two, leaving two, she takes both and
    # wins; where they take one, leaving three, nothing she does wins, and she knows it before they take.
    assert found.counter_to((THEIRS, 2)) == ((HERS, 2), 1.0)
    assert found.counter_to((THEIRS, 1))[1] == -1.0


def _a_neutral_part(name: str = "quiet") -> Reading:
    """A part where she has a move that changes nothing, and so does nobody else."""
    return read_part(name, "here", acts=lambda state, side: ["shuffle"] if side == HERS else [],
                     after=lambda state, act: state, judge=lambda state: 0.0, plies=1)


def test_the_neutral_move_is_taken_where_nothing_she_could_do_beats_standing_still() -> None:
    from core.cognition.what_each_area_comes_to import what_to_do

    def acts(state, side):
        return [side] if state == "open" else []

    # A part where whoever moves first gives the other side the better of it: acting there costs her 2.
    cold = read_part("cold", "open", acts=acts, after=lambda state, act: act,
                     judge=lambda state: {"open": 0.0, HERS: -2.0, THEIRS: 2.0}[state], plies=1)
    choice = what_to_do([cold, _a_neutral_part()])
    assert choice.kind == "wait" and choice.weighed.part == "quiet"
    assert not choice.most_good


def test_who_outlasts_whom_in_a_waiting_game_turns_on_the_count_and_on_who_spends_first() -> None:
    from core.cognition.what_each_area_comes_to import the_waiting_game

    assert the_waiting_game(3, 2, hers_to_move=True) is True
    assert the_waiting_game(2, 2, hers_to_move=True) is False, "moving first with as many, she runs out first"
    assert the_waiting_game(2, 2, hers_to_move=False) is True, "the same count with them to move: they run out first"
    assert the_waiting_game(float("inf"), 5, hers_to_move=True) is True
    assert the_waiting_game(float("inf"), float("inf"), hers_to_move=True) is None


def test_where_the_world_deals_by_odds_what_luck_decides_is_told_from_what_she_makes_sure_of() -> None:
    """A gamble against a sure thing: the gamble deals +10 or -10 at even odds; the sure thing is +1 for certain."""
    from core.cognition.what_each_area_comes_to import what_can_happen

    def acts(state, side):
        return ["gamble", "sure"] if state == "start" and side == HERS else []

    def after(state, act):
        return {"gamble": "dealing", "sure": "plus one"}[act]

    def dealt(state):
        return [("plus ten", 0.5), ("minus ten", 0.5)] if state == "dealing" else []

    stands = {"start": 0.0, "dealing": 0.0, "plus one": 1.0, "plus ten": 10.0, "minus ten": -10.0}
    found = what_can_happen("bet", "start", acts=acts, after=after, judge=stands.__getitem__, plies_at_most=3,
                            seconds=2.0, may_leave=False, dealt=dealt)
    paths = {path.act: path for path in found.hers}
    assert paths["gamble"].for_certain == -10.0 and paths["gamble"].at_best == 10.0
    assert paths["gamble"].expected == 0.0 and paths["gamble"].only_if_helped == 20.0
    assert paths["sure"].settled and paths["sure"].for_certain == 1.0
    assert found.hers[0].act == "sure", "what she can make sure of comes first"


def test_played_as_one_game_the_turn_order_over_the_parts_decides_what_the_quick_sum_misses() -> None:
    """Parts where whoever moves first gives the other side a point, and no turn may be let go: with three of them she
    moves in two and they in one, so she ends a point down; with two, even. The quick sum takes no part no one gains by
    acting in first, and calls both even (suggestions of 24 November 2026: the turn order over the parts interacts)."""
    from core.cognition.what_each_area_comes_to import play_the_sum

    def acts(state, side):
        return [side] if state == "open" else []

    def judge(state):
        return {"open": 0.0, HERS: -1.0, THEIRS: 1.0}[state]

    def cold(name):
        return read_part(name, "open", acts=acts, after=lambda state, act: act, judge=judge, plies=1)

    three = play_the_sum([(n, "open") for n in "abc"], acts=acts, after=lambda state, act: act, judge=judge,
                         plies_at_most=6, seconds=5.0)
    two = play_the_sum([(n, "open") for n in "ab"], acts=acts, after=lambda state, act: act, judge=judge,
                       plies_at_most=6, seconds=5.0)
    assert three.complete and three.comes_to == -1.0
    assert two.complete and two.comes_to == 0.0
    assert put_together([cold(n) for n in "abc"], first=HERS).comes_to == 0.0


def test_working_out_what_can_happen_ends_when_the_world_says_the_time_is_over_and_keeps_what_it_finished() -> None:
    """A count that never ends, given a minute: told the time is over a fifth of a second in, it stops at once and
    keeps the deepest pass it finished, marked unfinished. The time it was given was a guess at how long the other side
    would take; their move arriving ends it (24 November 2026: run on into her own turn, it slowed her own look)."""
    import time

    from core.cognition.what_each_area_comes_to import play_the_sum, what_can_happen

    def acts(state, side):
        return [1, 2]

    # Every line its own state: nothing reached two ways, so nothing is worked out once for two.
    def after(state, act):
        return (*state, act)

    def judge(state):
        return float(sum(state) % 3)

    for work in (
        lambda over: what_can_happen("count", (), acts=acts, after=after, judge=judge, plies_at_most=60,
                                     seconds=60.0, may_leave=False, stop=over),
        lambda over: play_the_sum([("a", ()), ("b", ())], acts=acts, after=after, judge=judge, plies_at_most=60,
                                  seconds=60.0, stop=over),
    ):
        began = time.monotonic()
        found = work(lambda began=began: time.monotonic() > began + 0.2)
        assert time.monotonic() - began < 5.0
        assert not found.complete and found.deep >= 1
        assert all(path.uncertain for path in getattr(found, "hers", ()))


def test_where_every_act_comes_to_the_same_nothing_is_chosen_and_standing_still_is_no_better() -> None:
    """A part where every act of hers leaves it as it stands, and one where her two acts change it but come to the
    same: nothing counted tells her acts apart, so nothing is chosen. A tie with standing still is not a reason to
    stand still."""
    from core.cognition.what_each_area_comes_to import what_to_do

    still = read_part("still", "here", acts=lambda state, side: ["a", "b"] if side == HERS else ["x"],
                      after=lambda state, act: state, judge=lambda state: 0.0, plies=2)
    choice = what_to_do([still])
    assert choice.kind == "even" and choice.weighed is None
    moving = read_part("moving", "here", acts=lambda state, side: ["left", "right"] if state == "here" else [],
                       after=lambda state, act: act, judge=lambda state: 0.0, plies=2)
    choice = what_to_do([moving, _a_neutral_part()])
    assert choice.kind == "even" and choice.weighed is None


def test_played_as_one_game_a_spare_move_that_only_delays_what_must_come_is_no_reason_to_wait() -> None:
    """A part where whoever moves first loses two, a spare move only she has and a spare move only they have, and no
    turn may be let go. The quick sum calls her spare move the neutral choice; played as one game, she spends hers,
    they spend theirs, and she must move first where it costs her two all the same: every act comes to the same, and
    nothing is chosen (a review Bryan passed on, 24 November 2026: the exact sum was not yet deciding anything)."""
    from core.cognition.what_each_area_comes_to import play_the_sum

    def acts(state, side):
        name, now = state
        if name == "cold":
            return [side] if now == "open" else []
        owner = HERS if name == "mine" else THEIRS
        return ["spare"] if now == "open" and side == owner else []

    def after(state, act):
        name, _now = state
        return name, (act if name == "cold" else "spent")

    def judge(state):
        return {"open": 0.0, HERS: -2.0, THEIRS: 2.0, "spent": 0.0}[state[1]]

    parts = [(name, (name, "open")) for name in ("cold", "mine", "yours")]
    readings = [read_part(name, state, acts=acts, after=after, judge=judge, plies=2) for name, state in parts]
    assert what_to_do(readings).kind == "neutral"
    played = tuple(play_the_sum(parts, acts=acts, after=after, judge=judge, plies_at_most=8, seconds=5.0, first=side)
                   for side in (HERS, THEIRS))
    assert all(one.complete and one.ended for one in played)
    assert {comes_to for _part, _act, comes_to in played[0].each} == {-2.0}
    choice = what_to_do(readings, played=played)
    assert choice.kind == "even" and choice.weighed is None


def test_what_luck_decides_after_their_answer_is_not_counted_as_certain() -> None:
    """A gamble whose dealing comes after the other side's answer, against a sure point: what she makes sure of has the
    world deal against her at every deal, not only at the first (a review Bryan passed on, 24 November 2026: "Deeper
    stochastic branches currently use expected values, which means for_certain is not necessarily a genuine worst-case
    guarantee")."""
    from core.cognition.what_each_area_comes_to import what_can_happen

    def acts(state, side):
        if state == "start" and side == HERS:
            return ["gamble", "sure"]
        return ["go on"] if state == "their answer" and side == THEIRS else []

    def after(state, act):
        return {"gamble": "their answer", "sure": "plus one", "go on": "dealing"}[act]

    def dealt(state):
        return [("plus ten", 0.5), ("minus ten", 0.5)] if state == "dealing" else []

    stands = {"start": 0.0, "their answer": 0.0, "dealing": 0.0, "plus one": 1.0, "plus ten": 10.0, "minus ten": -10.0}
    found = what_can_happen("bet", "start", acts=acts, after=after, judge=stands.__getitem__, plies_at_most=4,
                            seconds=2.0, may_leave=False, dealt=dealt)
    paths = {path.act: path for path in found.hers}
    assert paths["gamble"].for_certain == -10.0 and paths["gamble"].expected == 0.0
    assert paths["gamble"].at_best == 10.0
    assert paths["gamble"].proven and paths["gamble"].made_sure_of == -10.0
    assert found.hers[0].act == "sure" and found.proven


def test_a_number_that_rests_on_a_line_cut_at_the_depth_asked_is_not_proven() -> None:
    """The take-away count from four: two moves deep, no line has reached the last token, and what each take comes to
    is a judgement of a pile still standing; eight deep, every line ends, and what she makes sure of is proven (a review
    Bryan passed on, 24 November 2026: "reaching the requested search depth doesn't establish certainty beyond that
    depth")."""
    from core.cognition.what_each_area_comes_to import what_can_happen

    def acts(state, side):
        pile, _last = state
        return [(side, n) for n in (1, 2) if n <= pile]

    def after(state, act):
        side, take = act
        return state[0] - take, side

    def judge(state):
        pile, last = state
        return 0.0 if pile else (1.0 if last == HERS else -1.0)

    shallow = what_can_happen("pile", (4, None), acts=acts, after=after, judge=judge, plies_at_most=2, seconds=5.0,
                              may_leave=False)
    assert shallow.complete and not shallow.proven
    assert all(path.made_sure_of == float("-inf") for path in shallow.hers)
    deep = what_can_happen("pile", (4, None), acts=acts, after=after, judge=judge, plies_at_most=8, seconds=5.0,
                           may_leave=False)
    best = deep.hers[0]
    assert deep.proven and best.act == (HERS, 1) and best.made_sure_of == best.for_certain == 1.0


def test_a_move_that_keeps_everything_held_but_leaves_fewer_ways_changes_something_where_ways_are_counted() -> None:
    """A camp where her shuffle keeps everything she holds but closes one of her ways out. Judged by what is held alone,
    the shuffle changes nothing; judged by what is held and the ways open, it costs her (a review Bryan passed on,
    24 November 2026: "unchanged material doesn't necessarily mean unchanged strategic position")."""
    from core.cognition.what_each_area_comes_to import changes_nothing

    def acts(state, side):
        return ["shuffle"] if side == HERS else []

    def after(state, _act):
        held, ways = state
        return held, ways - 1

    def held_alone(state):
        return float(state[0])

    def held_and_ways(state):
        return float(state[0]) + state[1] / 10

    for judge, it_changes in ((held_alone, False), (held_and_ways, True)):
        reading = read_part("camp", (3, 4), acts=acts, after=after, judge=judge, plies=2)
        shuffle = weigh_acts([reading])[0]
        assert changes_nothing(shuffle, [reading]) is not it_changes
    # Read by what is held, with the fuller judgement told of it: the shuffle weakens her, and is no waiting move.
    from dataclasses import replace

    held = read_part("camp", (3, 4), acts=acts, after=after, judge=held_alone, plies=2)
    told = replace(held, weakens=("shuffle",))
    assert not changes_nothing(weigh_acts([told])[0], [told])


def _a_shared_pool_world():
    """Three plots that draw on one pool of tokens: a side claims an unclaimed plot by spending a token, and on the first
    plot a side may instead spend a token to double what every claim is worth. A claim is worth its worth to the side
    that holds it. Each part's state carries the pool and the worth it last saw."""
    def acts(state, side):
        (pool, worth), plot, held = state
        if pool <= 0:
            return []
        mine = [("claim", side)] if held is None else []
        return mine + ([("double", side)] if plot == "a" and worth == 1 else [])

    def after(state, act):
        (pool, worth), plot, held = state
        kind, side = act
        if kind == "double":
            return (pool - 1, worth * 2), plot, held
        return (pool - 1, worth), plot, side

    def judge(state):
        (_pool, worth), _plot, held = state
        return 0.0 if held is None else (float(worth) if held == HERS else -float(worth))

    def joined(states, i):
        shared = states[i][0]
        return [(shared, plot, held) for _seen, plot, held in states]

    def whole(states):
        return sum(judge(state) for state in states)

    return acts, after, judge, joined, whole


def _the_whole_searched(states, acts, after, joined, whole, side, left):
    """Every line of the whole world, followed: the plain search a sum must agree with."""
    options = [(i, act) for i, state in enumerate(states) for act in acts(state, side)] if left > 0 else []
    if not options:
        return whole(states)
    found = [_the_whole_searched(tuple(joined(states[:i] + (after(states[i], act),) + states[i + 1:], i)), acts,
                                 after, joined, whole, 1 - side, left - 1) for i, act in options]
    return max(found) if side == HERS else min(found)


def test_parts_that_draw_on_one_pool_are_played_as_one_world_and_come_to_what_searching_the_whole_does() -> None:
    """Two tokens for three plots, and an act on one plot that changes what every claim is worth. Kept apart, each plot
    spends the pool as if it were its own and knows nothing of the doubling: the sum is wrong. Held together, every act's
    whole is what a plain search of the whole world finds, whichever side acts first (a review Bryan passed on,
    24 November 2026: "Verify the composed results against exhaustive global search on small interacting worlds,
    including cases where an action in one region changes the legal actions, resources, or strategic value of another
    region")."""
    from core.cognition.what_each_area_comes_to import play_the_sum

    acts, after, judge, joined, whole = _a_shared_pool_world()
    start = tuple(((2, 1), plot, None) for plot in "abc")
    parts = [(state[1], state) for state in start]
    for first in (HERS, THEIRS):
        held = play_the_sum(parts, acts=acts, after=after, judge=judge, plies_at_most=6, seconds=10.0, first=first,
                            joined=joined, whole=whole)
        apart = play_the_sum(parts, acts=acts, after=after, judge=judge, plies_at_most=6, seconds=10.0, first=first)
        assert held.complete and apart.complete
        for part, act, comes_to in held.each:
            i = next(n for n, state in enumerate(start) if state[1] == part)
            moved = tuple(joined(start[:i] + (after(start[i], act),) + start[i + 1:], i))
            assert comes_to == _the_whole_searched(moved, acts, after, joined, whole, 1 - first, 5), (part, act)
        assert held.comes_to == _the_whole_searched(start, acts, after, joined, whole, first, 6)
        assert apart.comes_to != held.comes_to, "kept apart, the pool is spent three times over"
