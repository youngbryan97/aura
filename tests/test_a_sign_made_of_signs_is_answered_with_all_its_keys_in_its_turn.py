"""A line of signs with everything a real one has that fooled a first reading of it.

Nothing here is a spell book. A strip with arrows coming along it to a pad, and four things about it that are true
of drawn games and were each a lost game to a reference runner before they were built for:

- from the second showing a sign may be two arrows drawn over one another, asking for both keys at once;
- the strip's own edge is drawn in the arrows' ink, and the lower arrow of a pair lies on it;
- the pad at the end is shaded with a shape in that ink that reads, by its outline, as an arrow;
- a sign cannot be seen while it is judged: on the pad it is drawn in another colour.

The sign is judged over the pad by exactly which keys are down, as each comes to it.
"""
from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest

from core.agency.each_sign_in_its_turn import InTurn
from core.agency.signs_that_come_along import SignsComing
from core.perception.signs_made_of_signs import made_of
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit

WIDE, TALL = 640, 360
STRIP = (170, 222, 150)
INK = (50, 120, 50)
ON_THE_PAD = (235, 240, 120)
NEAR_THE_INK = (48, 150, 50)
PAD_FROM, PAD_TO = 520, 610
WAYS = ("left", "right", "up", "down")


def _arrow(way: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (44, 44), 0)
    draw, m = ImageDraw.Draw(canvas), 22
    draw.rectangle([m - 15, m - 3, m + 4, m + 3], fill=255)
    draw.polygon([(m + 2, m - 12), (m + 16, m), (m + 2, m + 12)], fill=255)
    return np.asarray(canvas.rotate({"right": 0, "up": 90, "left": 180, "down": 270}[way])) > 128


def _drawn(ways: tuple[str, ...]) -> np.ndarray:
    """A sign as drawn: one arrow, or two laid over one another a little apart, the second lower and to the left."""
    canvas = np.zeros((70, 70), bool)
    places = [(13, 13)] if len(ways) == 1 else [(5, 22), (21, 4)]
    for way, (top, left) in zip(ways, places, strict=True):
        canvas[top:top + 44, left:left + 44] |= _arrow(way)
    return canvas


class _Strip:
    """Signs come along a strip to a pad; each is judged over the pad by exactly what is down."""

    def __init__(self, seed: int = 3, pace: float = 170.0, pairs_from: int = 8, alone_for: float = 0.75,
                 judged_from: int = PAD_FROM, whirl: bool = False, drawn_again: bool = False,
                 near_edge: bool = False) -> None:
        self.random = random.Random(seed)
        self.pace, self.pairs_from = pace, pairs_from
        #: How far along a sign must be to be judged: it is not to be seen from the pad's edge on, wherever that is.
        self.judged_from = judged_from
        #: Whether what the signs come out of shows arrows in their ink going round in it.
        self.whirl = whirl
        #: Whether a sign is drawn again in its own ink, bigger, on the far half of the pad as it leaves.
        self.drawn_again = drawn_again
        #: Whether the lower part of the strip is a band of a green near the signs' own, under the lower arrow of a pair.
        self.near_edge = near_edge
        #: How long the first sign is on the strip alone before the second is sent after it.
        self.alone_for = alone_for
        self.coming: list[dict] = []
        self.sent = 0
        self.now, self.next_at = 0.0, 0.4
        self.down: set[str] = set()
        self.good = self.wrong = 0
        self.good_pairs = self.pairs = 0

    async def look(self):
        self.now += 1 / 30
        if self.now >= self.next_at:
            self.sent += 1
            self.next_at = self.now + (self.alone_for if self.sent == 1 else 0.75)
            pair = self.sent > self.pairs_from and self.random.random() < 0.5
            ways = tuple(self.random.sample(WAYS, 2)) if pair else (self.random.choice(WAYS),)
            self.coming.append({"ways": ways, "x": 70.0, "ok": False})
            self.pairs += len(ways) == 2
        for sign in self.coming:
            sign["x"] += self.pace / 30
        for sign in list(self.coming):
            if self.judged_from <= sign["x"] < PAD_TO and self.down == set(sign["ways"]):
                sign["ok"] = True
            if sign["x"] >= PAD_TO:
                self.good += sign["ok"]
                self.wrong += not sign["ok"]
                self.good_pairs += sign["ok"] and len(sign["ways"]) == 2
                self.coming.remove(sign)
        picture = np.full((TALL, WIDE, 3), (228, 240, 246), np.uint8)
        picture[262:340] = STRIP
        if self.near_edge:
            picture[316:336] = NEAR_THE_INK
        picture[258:262] = INK                                       # the strip's edges, in the arrows' own ink
        picture[336:340] = INK
        picture[264:338, PAD_FROM + 24:PAD_TO - 4] = (150, 205, 135)   # the pad
        shading = _arrow("down")                                     # and a shape on it that reads as an arrow
        picture[278:322, 545:589][shading] = INK
        for sign in self.coming:
            drawn = _drawn(sign["ways"])
            left = round(sign["x"]) - 35
            if left < 0 or left + 70 > WIDE:
                continue
            colour = ON_THE_PAD if sign["x"] >= PAD_FROM else INK      # not to be seen as ink while it is judged
            if self.drawn_again and sign["x"] >= PAD_FROM + 45:
                big = np.kron(_arrow(sign["ways"][0]), np.ones((2, 2), bool))[10:78, 10:78]
                picture[266:334, left:left + 68][big[:, :min(68, WIDE - left)][:, :68]] = INK
                continue
            picture[266:336, left:left + 70][drawn] = colour
        if self.whirl:
            # What the signs come out of: a page over the start of the strip, arrows in their ink going round on it.
            picture[240:352, 8:128] = (236, 250, 228)
            for number, way in enumerate(("up", "left", "down", "right", "up")):
                turn = 2.3 * self.now + 1.26 * number
                top, left = round(274 + 22 * np.sin(turn)), round(46 + 34 * np.cos(0.7 * turn + number))
                picture[top:top + 44, left:left + 44][_arrow(way)] = INK
        return picture, self.now


class _Hands:
    def __init__(self, strip: _Strip) -> None:
        self.strip = strip

    async def down(self, key):
        self.strip.down.add(key)

    async def up(self, key):
        self.strip.down.discard(key)


async def _played(strip: _Strip, seconds: float, signs: SignsComing | None = None) -> SignsComing:
    moves, signs, hands = WhatMoves(), signs or SignsComing(told=True), _Hands(strip)     # a place that asks for signs
    while strip.now < seconds:
        picture, at = await strip.look()
        moves.see(picture, at)
        signs.see(moves, picture, at)
        if signs.asks(at):
            await signs.hold(hands, at)
        else:
            await signs.let_go(hands)
    return signs


def test_single_signs_teach_their_shapes_and_pairs_are_then_answered_with_both_keys():
    strip = _Strip()
    signs = asyncio.run(_played(strip, 40.0))
    assert signs.strip is not None and len(signs.strip.known) == 4, signs.strip and sorted(signs.strip.known)
    judged = strip.good + strip.wrong
    assert judged >= 45 and strip.pairs >= 10, (judged, strip.pairs)
    # The first few go by while she is still finding that it is a line of signs; after them nearly all, pairs with them.
    assert strip.wrong <= 6, (strip.good, strip.wrong)
    assert strip.good_pairs >= strip.pairs - 3, (strip.good_pairs, strip.pairs)


def test_a_line_known_before_is_answered_from_its_first_sign_pairs_and_all():
    first = _Strip(seed=3)
    learned = asyncio.run(_played(first, 30.0))
    second = _Strip(seed=11, pairs_from=0)
    asyncio.run(_played(second, 25.0, SignsComing(learned.known(), told=True)))
    assert second.wrong <= 2 and second.good >= 25, (second.good, second.wrong)
    assert second.good_pairs >= second.pairs - 2, (second.good_pairs, second.pairs)


def test_a_mark_is_read_as_the_known_signs_it_is_made_of():
    known = {way: _drawn((way,))[13:57, 13:57] for way in WAYS}
    for ways in (("up",), ("left",), ("up", "down"), ("left", "right"), ("down", "left"), ("right", "up")):
        mark = _drawn(ways)
        assert made_of(mark, known) == frozenset(ways), ways
    # With its foot cut off, as where it overlaps an edge that is left out: still what it is.
    cut = _drawn(("up", "down"))
    cut[-10:] = False
    assert made_of(cut, known) == frozenset({"up", "down"})
    # And a thing that is none of them is none of them.
    blob = np.zeros((40, 40), bool)
    blob[8:32, 8:32] = True
    assert made_of(blob, known) == frozenset()


class _Sign:
    def __init__(self, along: float, keys: str) -> None:
        self.along, self.keys = along, frozenset(keys.split("+"))


def test_a_sign_s_keys_stay_down_until_it_is_judged_and_then_the_next_one_s_go_down_alone():
    turn, read = InTurn(), lambda sign: sign.keys
    held = []
    # Two signs 0.25 apart going 0.4 of the picture a second; each goes from sight at 0.8, where it is judged.
    for step in range(60):
        at = step / 30
        a, b = 0.45 + 0.4 * at, 0.20 + 0.4 * at
        signs = [_Sign(x, keys) for x, keys in ((a, "up"), (b, "left+right")) if x < 0.8]
        held.append((at, a, b, turn.take(signs, at, read)))
    # Once the first has come some way along the line, while it is still on it, and for a while after it has gone
    # from sight, its key is the one down.
    assert all(keys == {"up"} for _at, a, _b, keys in held if 0.56 < a < 0.9), [h for h in held if 0.56 < h[1] < 0.9][:3]
    assert held[0][3] == frozenset()                          # a thing only just seen on the line is not yet answered
    # By the time the second gets to where signs are judged, its two keys are down and nothing else.
    assert all(keys == {"left", "right"} for _at, _a, b, keys in held if 0.74 < b < 0.8)
    # And once the line is empty and the last has been judged, nothing is held.
    assert held[-1][3] == frozenset()


def test_what_shows_beyond_where_signs_go_from_sight_is_not_gone_by():
    turn, read = InTurn(), lambda sign: sign.keys
    for step in range(30):                                   # the first goes from sight at 0.8, the second behind it
        at = step / 30
        signs = [_Sign(x, keys) for x, keys in ((0.5 + 0.4 * at, "up"), (0.25 + 0.4 * at, "down")) if x < 0.8]
        turn.take(signs, at, read)
    # The first shows again for a look on the pad, further on: it has been judged, and the second's key stays down.
    assert turn.take([_Sign(0.95, "up"), _Sign(0.70, "down")], 1.12, read) == {"down"}


def test_told_that_signs_are_to_be_followed_she_answers_the_very_first_one():
    """Offline 2026-10-12: the first arrow of a spell is most of the way to its mark before a second shows behind it.
    Waiting for two going together before taking either for a sign, she lost three lives in the first six seconds of
    every game, in a place whose words had said to follow the arrows."""
    strip = _Strip(seed=5, alone_for=3.5)                 # the first is judged before the second is even sent
    asyncio.run(_played(strip, 14.0, SignsComing(told=True)))
    assert strip.wrong == 0 and strip.good >= 11, (strip.good, strip.wrong)


def test_the_line_a_watch_saw_signs_on_is_read_whole_once_enough_readings_of_their_ink_agree():
    from core.agency.signs_that_come_along import SignsAreComing, watching_for_signs

    strip = _Strip(seed=7)

    async def watched_then_played() -> tuple[int, int]:
        look = watching_for_signs(strip.look)
        try:
            while strip.now < 6.0:
                await look()
        except SignsAreComing as seen:
            known = seen.args[0]
        assert known.get("going") and known.get("on_line"), known
        signs = SignsComing(known, told=True)
        await _played(strip, 14.0, signs)
        assert signs.strip is not None and signs.strip.ink is not None
        return strip.good, strip.wrong

    good, wrong = asyncio.run(watched_then_played())
    assert wrong <= 1 and good >= 12, (good, wrong)


def test_the_loop_that_only_answers_signs_answers_them_and_gives_the_screen_back_when_none_come():
    """Offline 2026-10-12: read inside the loop that plays everything else as it happens, the line was seen three to
    ten times a second and every round was lost. This loop does nothing but look, read the line and hold."""
    from core.agency.signs_that_come_along import NOTHING_COMING_S, answer_signs_as_they_come

    strip = _Strip(seed=11, alone_for=3.5)

    async def played() -> tuple[dict, dict]:
        first = await answer_signs_as_they_come(strip.look, _Hands(strip), seconds=14.0)
        strip.next_at = float("inf")                      # the showing is over: nothing more is sent
        after = await answer_signs_as_they_come(strip.look, _Hands(strip), seconds=60.0, known=first["known"])
        return first, after

    first, after = asyncio.run(played())
    assert strip.wrong <= 1 and strip.good >= 11, (strip.good, strip.wrong)
    assert first["ended"] == "out of time" and first["judged"] >= 9, first
    assert first["known"]["ink"] is not None and first["known"]["shapes"], first["known"].keys()
    assert after["ended"] == "no signs are coming" and after["seconds"] < NOTHING_COMING_S + 6.0, after
    assert not strip.down


def test_the_ink_most_readings_agree_on_is_the_signs_and_not_the_one_they_are_judged_in():
    """Offline 2026-10-12: a line's ink was taken from two readings, both of something drawn in another colour, and
    nothing was seen on the line for the rest of the game."""
    signs = SignsComing(told=True)
    signs._inks, signs._sizes = [ON_THE_PAD, INK, INK, ON_THE_PAD, INK], [70, 44, 45, 71, 44]
    assert signs._ink_and_size() is None                      # three agree: not yet enough
    signs._inks.append(INK)
    signs._sizes.append(43)
    ink, size = signs._ink_and_size()
    assert ink == INK and 43 <= size <= 45


LINE = {"going": (170.0 / WIDE, 0.0), "on_line": (0.5, 301 / TALL), "ink": INK, "size": 32.0, "shapes": {}}


def test_a_sign_s_keys_are_still_down_where_it_is_judged_well_onto_its_mark():
    """Offline 2026-10-12: a sign went from sight at the edge of its mark and was judged a twentieth of the picture
    further on. Keys let go a moment after it went from sight were let go before that about one time in six, and the
    next sign's were down in their place."""
    strip = _Strip(seed=13, judged_from=PAD_FROM + 50, pairs_from=14)     # pairs once each single has come alone
    asyncio.run(_played(strip, 16.0, SignsComing(dict(LINE), told=True)))
    assert strip.wrong <= 1 and strip.good >= 14, (strip.good, strip.wrong)



def test_arrows_going_round_in_what_the_signs_come_out_of_are_not_signs_on_their_way():
    """Offline 2026-10-12: a book the arrows came out of showed arrows in their ink whirling on its page, in line with
    the strip. They were read as signs for as long as the game ran, and what they read as was held for."""
    strip = _Strip(seed=17, whirl=True)
    signs = asyncio.run(_played(strip, 22.0, SignsComing(dict(LINE), told=True)))
    assert strip.wrong <= 2 and strip.good >= 20, (strip.good, strip.wrong)
    strip.next_at = float("inf")                             # nothing more is sent along the strip
    asyncio.run(_played(strip, 27.0, signs))                 # and the last of them have gone by

    async def with_only_the_page_going() -> list[set[str]]:
        held, hands = [], _Hands(strip)
        moves = WhatMoves()
        for _look in range(90):
            picture, at = await strip.look()
            signs.see(moves, picture, at)
            await (signs.hold(hands, at) if signs.asks(at) else signs.let_go(hands))
            held.append(set(strip.down))
        return held

    assert not any(asyncio.run(with_only_the_page_going()))


def test_a_sign_drawn_again_bigger_past_where_it_went_from_sight_is_not_the_one_at_the_front():
    """Offline 2026-10-12: a sign that had been judged was drawn again on its mark, twice the size, in the signs' own
    ink. Taken up as the line was first read, it was the front sign and where signs go from sight at once, and the keys
    of the one really coming went down a sign late for the rest of the round."""
    strip = _Strip(seed=19, drawn_again=True)
    asyncio.run(_played(strip, 16.0, SignsComing(dict(LINE), told=True)))
    assert strip.wrong <= 1 and strip.good >= 14, (strip.good, strip.wrong)


def test_a_sign_passing_over_a_part_of_the_strip_that_is_nearly_its_own_colour_is_still_read_whole():
    """Offline 2026-10-12: the strip's lower edge was a green near the arrows' own. Kept as ink that stays, with a
    little left out round it, it cut the head off the lower arrow of every pair, and an arrow up beside an arrow
    down was held for as up alone. In the fourth room, where most signs are pairs, that was the game."""
    strip = _Strip(seed=23, near_edge=True, pairs_from=14)
    asyncio.run(_played(strip, 26.0, SignsComing(dict(LINE), told=True)))
    assert strip.pairs >= 6 and strip.wrong <= 1, (strip.good, strip.wrong, strip.pairs, strip.good_pairs)
    assert strip.good_pairs >= strip.pairs - 4            # all but those still on the strip at the end


def test_things_going_together_down_a_line_where_nothing_asks_for_signs_are_not_read_as_a_line_of_signs():
    """Offline 2026-10-12: eggs falling in a game of catching them were read whole as a line of signs at every look
    of its play, in a place that had asked for none."""
    strip = _Strip(seed=3)

    async def watched() -> SignsComing:
        moves, signs = WhatMoves(), SignsComing()
        while strip.now < 8.0:
            picture, at = await strip.look()
            moves.see(picture, at)
            signs.see(moves, picture, at)
        return signs

    signs = asyncio.run(watched())
    assert signs.going is not None and signs.strip is None
    assert "ink" not in (signs.known() or {}) or SignsComing(signs.known()).strip is None
