"""She learns what a kind of thing does just before it hurts her, gets clear when she sees it, and keeps clear of a
thing she does not yet know that is closing on her faster than she can answer.

The tells of action games (an enemy that stops and draws back before it lunges, a block that shakes before it drops)
are given so a player can read them; nobody learns them as a list.
"""
from __future__ import annotations

from types import SimpleNamespace as Thing

import pytest

from core.agency.warning_signs import STOPS, WarningSigns

pytestmark = pytest.mark.unit

ME = Thing(number=0, x=100.0, y=100.0, kind=0)


def _brute(at_rest: bool, x: float = 300.0) -> dict[int, Thing]:
    return {7: Thing(number=7, kind=3, x=x, y=100.0, w=20.0, h=20.0, vx=0.0 if at_rest else 30.0, vy=0.0, still=at_rest)}


def test_a_change_that_comes_before_losses_more_than_chance_becomes_a_tell_and_is_read_after():
    signs = WarningSigns()
    losses: list[dict] = []
    at = 0.0
    for _round in range(3):                                   # it paces, stops, and a moment later she is hit
        for _ in range(6):
            signs.saw(_brute(False), ME, at, losses)
            at += 0.1
        stopped = at
        losses.append({"what": "loss", "at": stopped + 0.3})
        for _ in range(25):                                   # it stands a while after, then paces on
            signs.saw(_brute(True), ME, at, losses)
            at += 0.1
        for _ in range(30):                                   # long enough for each change to be settled
            signs.saw(_brute(False), ME, at, losses)
            at += 0.1
    assert signs.is_a_tell(3, STOPS)
    said = signs.news(lambda kind: "the brutes")
    assert said == ["When the brutes stop, harm follows: I'll get clear when I see it."] and signs.news(lambda k: "") == []
    for _ in range(6):
        signs.saw(_brute(False), ME, at, losses)
        at += 0.1
    for _ in range(5):
        shown = signs.saw(_brute(True), ME, at, losses)
        at += 0.1
        if shown:
            break
    assert (3, STOPS) in shown and signs.showing(_brute(True)[7], at)


def test_a_change_after_which_nothing_comes_is_no_tell_and_a_thing_coming_fast_is_kept_clear_of():
    signs = WarningSigns()
    at = 0.0
    for _round in range(4):
        for at_rest in (False,) * 6 + (True,) * 3:
            signs.saw(_brute(at_rest), ME, at, [])
            at += 0.1
        for _ in range(25):
            signs.saw(_brute(False), ME, at, [])
            at += 0.1
    assert not signs.is_a_tell(3, STOPS)
    rushing = {9: Thing(number=9, kind=5, x=160.0, y=100.0, w=10.0, h=10.0, vx=-200.0, vy=0.0, still=False)}
    signs.saw(rushing, ME, at, [])
    signs.saw(rushing, ME, at + 0.05, [])
    assert signs.closing_in(rushing[9], ME, within_s=0.5)                      # 60 px at 200 px/s: 0.3 s away
    leaving = Thing(number=9, kind=5, x=160.0, y=100.0, w=10.0, h=10.0, vx=200.0, vy=0.0, still=False)
    assert not signs.closing_in(leaving, ME, within_s=0.5)


def test_what_she_keeps_clear_of_that_went_out_of_sight_is_still_danger_where_it_was_going():
    from core.agency.playing_as_it_happens import AVOID, _Choosing
    from core.perception.what_moves_in_the_picture import Thing as Seen

    def seen(number, x, y, vx=0.0):
        import numpy as np
        return Seen(number, x, y, 10.0, 10.0, np.zeros(4), (200, 30, 30), np.zeros((1, 1, 3)), born=0.0, seen=0.0, kind=4,
                    vx=vx)

    choosing = object.__new__(_Choosing)
    choosing.mine, choosing.at, choosing.reach = seen(0, 100.0, 50.0), 10.0, None
    choosing.moves = Thing(shape=(120, 200), out_of_sight={9: (seen(9, 40.0, 50.0, vx=60.0), 9.5)})
    choosing.meeting = Thing(stance=lambda kind: AVOID if kind == 4 else "meet")
    choosing.others = lambda: []
    staying = choosing.danger_along(lambda after: (0.0, 0.0))
    keeping_ahead = choosing.danger_along(lambda after: (60.0 * after, 0.0))
    assert staying > 0.0 and keeping_ahead < staying                            # it comes out where she stands
