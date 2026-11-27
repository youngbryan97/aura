"""Where a game showed her wrong, looked at again for longer and learned from.

Results against a player who beats her every game teach her nothing, but the games themselves show her where she was
wrong. At each of her turns she judged where she stood; where that judgement fell by her next turn, out of line with
how it moved through the rest of the game, the other side showed her something her look had not seen. Out of line is
the usual fence for a value apart from the rest: more than one and a half times the spread of the middle half of the
game's changes above its upper quarter (Tukey 1977).

Between games each such place is looked at again for far longer than a turn of hers could take, by every core she can
spare. The line that longer look expects is followed to its end, and where the position there is quiet, it is kept with
what the longer look found it worth: a position whose standing her quick judgement can learn, chosen by where a
stronger player exposed her and judged with time she did not have. These are kept apart from her practice's own
positions and are never pushed out by them, for they are few and come only from games in the world; when how the pieces
stand together is learned again, they are learned from with the rest (core/agency/what_a_kind_is_worth.py).

Nothing here knows a game: a place is a position written out, a judgement a number.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.WhereSheWasShownWrong")

__all__ = ["ShownWrong", "look_back_on_some", "places_shown_wrong"]

#: Places waiting to be looked at again, at most: the latest are kept.
WAITING_AT_MOST = 400
#: How long a look back takes, in seconds: about ten times a turn of hers at a game on a clock.
LOOK_BACK_S = 20.0
#: Positions kept from looking back, at most: games in the world are few, so these fill slowly.
KEPT_AT_MOST = 20_000


def places_shown_wrong(judged: list[tuple[str, str, int]]) -> list[str]:
    """The places of her turns after which her judgement fell by her next turn more than the rest of the game's changes
    would lead one to expect: above Tukey's upper fence of the changes, and a fall."""
    if len(judged) < 4:
        return []
    changes = np.array([judged[i + 1][2] - judged[i][2] for i in range(len(judged) - 1)], dtype=np.float64)
    falls = -changes
    lower, upper = np.percentile(falls, [25, 75])
    fence = upper + 1.5 * (upper - lower)
    return [judged[i][0] for i, fall in enumerate(falls) if fall > 0 and fall > fence]


@dataclass
class ShownWrong:
    """Places a game showed her wrong, waiting to be looked at again, and the positions kept from looking back."""

    waiting: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    parts: list[list[int]] = field(default_factory=list)
    glance: list[float] = field(default_factory=list)
    slow: list[float] = field(default_factory=list)
    game: list[int] = field(default_factory=list)
    games: int = 0

    def shown(self, places: list[str]) -> None:
        """One more game's places, to be looked at again."""
        self.games += 1
        self.waiting.extend(place for place in places if place not in self.waiting)
        del self.waiting[:-WAITING_AT_MOST]

    def kept_from(self, kind_order: list[str], cells: list[int], glance: float, slow: float, squares: int) -> None:
        """A quiet position kept from a look back: its parts under this store's order of kinds."""
        for letter in kind_order:
            if letter not in self.kinds:
                self.kinds.append(letter)
        slot = [self.kinds.index(letter) for letter in kind_order]
        parts = []
        for index in cells:
            kind, rest = divmod(int(index), 2 * squares)
            parts.append(slot[kind] * 2 * squares + rest)
        self.parts.append(parts)
        self.glance.append(float(glance))
        self.slow.append(float(slow))
        # Each game in the world an occasion of its own, apart from her practice's rounds: held back whole.
        self.game.append(-1 - self.games)
        del self.parts[:-KEPT_AT_MOST], self.glance[:-KEPT_AT_MOST], self.slow[:-KEPT_AT_MOST]
        del self.game[:-KEPT_AT_MOST]

    def as_bytes(self) -> bytes:
        return json.dumps({"waiting": self.waiting, "kinds": self.kinds, "parts": self.parts, "glance": self.glance,
                           "slow": self.slow, "game": self.game, "games": self.games},
                          separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes | None) -> ShownWrong:
        if not data:
            return cls()
        try:
            held = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return cls()
        if not isinstance(held, dict):
            return cls()
        return cls(waiting=[str(p) for p in held.get("waiting", [])], kinds=[str(k) for k in held.get("kinds", [])],
                   parts=[[int(i) for i in p] for p in held.get("parts", [])],
                   glance=[float(x) for x in held.get("glance", [])], slow=[float(x) for x in held.get("slow", [])],
                   game=[int(x) for x in held.get("game", [])], games=int(held.get("games", 0)))


def look_back_on_some(shown: ShownWrong, family: Any, weights: Any, start: dict, *, seconds: float,
                      threads: int = 1) -> int:
    """Places waiting, looked at again for LOOK_BACK_S each, for about ``seconds`` in all: the quiet position at the end
    of each longer look's line kept with what it found. How many places were looked at."""
    from core.agency.pieces_by_their_kind import after
    from core.agency.playing_pieces_by_kind import _position_from_place
    from core.agency.searching_by_kind import _compiled, described, placed, search

    engine = _compiled()
    if engine is None or not hasattr(engine, "seen_as_practice_sees"):
        return 0
    began, looked = time.monotonic(), 0
    squares = family.rows * family.columns
    without_the_layer_too = weights.as_json()
    while shown.waiting and time.monotonic() - began < seconds:
        place = shown.waiting.pop(0)
        position = _position_from_place(place, start)
        if position is None:
            continue
        looked += 1
        found = search(family, position, seconds=LOOK_BACK_S, weights=weights, threads=threads)
        if found.move is None or getattr(found, "mate", False):
            continue
        leaf, score = position, int(found.score)
        try:
            for step in found.line:
                leaf = after(family, leaf, step)
                score = -score
        # not a failure: a line her rules no longer take is followed only as far as they do.
        except (KeyError, ValueError):
            continue
        order, rows = engine.seen_as_practice_sees(described(family), [placed(leaf)], without_the_layer_too)
        cells, first_look, quiet, _turn = rows[0]
        if quiet and abs(score) < 3000:
            shown.kept_from(list(order), list(cells), float(first_look), float(score), squares)
    if looked:
        logger.info("looked again at %d place(s) a game showed her wrong; %d kept, %d waiting", looked,
                    len(shown.parts), len(shown.waiting))
    return looked
