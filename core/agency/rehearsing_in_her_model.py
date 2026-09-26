"""Trying a change to how she judges things in her own model of a world, before living with it.

A property she invents is a change to her own judgement, and the only trial it
had was a live one: take it on, play, and compare how fast things moved with
how fast they had moved before. That compares two different stretches of a
life — an early game and a late one, a good start and a bad one — and a
property kept on it stays at full weight in every run after. Two came back
this way on 2048. Played out from the same four starts with her own rule and
search, she reached 2048 in all four without them and in one of four with
them (2026-09-19).

Where she has compiled a world — her own rule for what her acts do, her own
record of what arrives between them — she does not need a life to try a
change on. She can play it out in her head: the same starts, the same dice,
once judging the old way and once the new, and see which gets further. Nothing
here is about any one world. The model is whatever she learned, the judging is
her own search with her own weights, and "further" is the furthest thing she
reached, which is what every rung she climbs is measured in.

It is only as good as her model. Where her model is wrong the rehearsal is
wrong in the same way, which is why this decides whether a change is worth
living with rather than replacing the living.
"""

from __future__ import annotations

import logging
import math
import random
import time as _clock
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Rehearsing")

__all__ = ["Played", "Rehearsed", "played_out", "rehearse"]

#: How many times each way. The same starts and the same dice both ways, so
#: what differs between the two is the change and not the luck.
TIMES = 3

#: How far each goes, in acts: nought is until the world stops answering or
#: what she is after is reached.
#:
#: It was two hundred. On 2048 two hundred moves reach about 256 whichever
#: way she judges, and a game that reaches 2048 takes about a thousand, so
#: every rehearsal came out level and a property measured to cost her three
#: games in four was kept after every game (2026-09-20 to 23). Played to its
#: end, a game takes her search a few seconds at this depth.
HOW_FAR = 0

#: How far ahead she looks while rehearsing. Shallower than when she plays:
#: this asks which way of judging is better, and a shallow search leans on the
#: judging more, which is the thing being asked about.
LOOKS_AHEAD = 2


@dataclass(frozen=True)
class Rehearsed:
    """How far each way got, rehearsal by rehearsal, from the same starts."""

    without: tuple[float, ...]
    with_it: tuple[float, ...]
    #: What it was, for saying so.
    what: str = ""

    def _middle(self, got: tuple[float, ...]) -> float:
        # Reached things are compared as ratios: getting from 64 to 128 is the
        # same step as from 512 to 1024 wherever doubling is how things grow.
        # Where nothing was reached, nothing is claimed for it.
        kept = [math.log(value) for value in got if value > 0]
        return sum(kept) / len(kept) if kept else 0.0

    def helps(self) -> bool:
        """Whether it got further with the change than without, over the same starts."""
        if not self.without or len(self.without) != len(self.with_it):
            return False
        return self._middle(self.with_it) > self._middle(self.without)

    def hurts(self) -> bool:
        """Whether it got less far with the change than without.

        Not the same as not helping. Two rehearsals that both went nowhere —
        from a finished position, say — are no evidence either way, and
        letting a property go on no evidence is the mistake the live trial
        made in the other direction.
        """
        if not self.without or len(self.without) != len(self.with_it):
            return False
        return self._middle(self.with_it) < self._middle(self.without)

    def says(self) -> str:
        return (
            f"played out in my own model of this, {self.what or 'the change'} got to "
            f"{', '.join(f'{v:g}' for v in self.with_it)} against "
            f"{', '.join(f'{v:g}' for v in self.without)} without it"
        )


def _furthest(state: Any) -> float:
    numbers = [value for value in (state.numbers() or ()) if value is not None]
    return float(max(numbers)) if numbers else 0.0


@dataclass(frozen=True)
class Played:
    """One game played out in her head: the furthest thing reached, and in how many acts."""

    furthest: float
    moves: int
    #: Whether what she was after was reached, where it names something.
    reached: bool = False
    #: The nearest the game came to what she was after, where reaching it is one.
    nearest: float = 0.0


def _one(knows: Any, world: Any, start: Any, actions: Sequence[str], **how: Any) -> float | None:
    """One rehearsal: her own search, her own model, her own dice. How far it got."""
    played = played_out(knows, world, start, actions, **how)
    return None if played is None else played.furthest


def played_out(
    knows: Any,
    world: Any,
    start: Any,
    actions: Sequence[str],
    *,
    weights: dict[str, float],
    toward: str,
    how_far: int = HOW_FAR,
    looks_ahead: int = LOOKS_AHEAD,
    seed: int = 0,
    choose: Callable[..., dict[str, tuple[float, str]]] | None = None,
    until: float = 0.0,
) -> Played | None:
    """One game in her own model, from ``start``, with her own search judging by ``weights``.

    None where there is no model to play in, or where ``until`` came first: a
    game cut off by the clock is not evidence about how far it would have got.
    """
    if choose is None:
        from core.agency.looking_ahead import look_ahead as choose  # noqa: PLC0415
    from core.agency.a_world_compiled import compiled  # noqa: PLC0415
    from core.agency.what_she_is_after import goal_in  # noqa: PLC0415

    made = compiled(knows, world, start, actions)
    if made is None:
        return None
    goal = goal_in(toward)
    roll = random.Random(seed)
    board = made.board(start)
    state = start
    furthest = _furthest(state)
    nearest = goal.nearness(state)
    reached = goal.reached(state)
    moves = 0
    while how_far <= 0 or moves < how_far:
        moves += 1
        if reached:
            break
        if until and _clock.monotonic() > until:
            return None
        scores = choose(
            knows, state, list(actions), toward=toward, world=world, weights=weights, depth=looks_ahead
        )
        if not scores:
            break
        act = max(scores, key=lambda name: scores[name][0])
        after = made.act(board, act)
        if after == board:
            break
        ways = made.replies(after)
        total = sum(share for _way, share in ways)
        landed, pick = ways[-1][0], roll.random() * total
        for way, share in ways:
            pick -= share
            if pick <= 0.0:
                landed = way
                break
        board = landed
        state = made.arrangement(board, like=start)
        furthest = max(furthest, _furthest(state))
        nearest = max(nearest, goal.nearness(state))
        reached = reached or goal.reached(state)
    return Played(furthest, moves, reached=reached, nearest=nearest)


def rehearse(
    knows: Any,
    world: Any,
    start: Any,
    actions: Sequence[str],
    *,
    weights: dict[str, float],
    trying: dict[str, float],
    toward: str = "",
    times: int = TIMES,
    how_far: int = HOW_FAR,
    looks_ahead: int = LOOKS_AHEAD,
    seed: int = 0,
    choose: Callable[..., dict[str, tuple[float, str]]] | None = None,
    within_s: float = 0.0,
) -> Rehearsed | None:
    """Play a change to her judging out in her own model, against not making it.

    ``trying`` is what the change adds to her weights, by name — a property
    she invented at the worth she would give it. None where she has no model
    to rehearse in, which is most of the time early in a world, or where
    ``within_s`` ran out before every game was played to its end.
    """
    until = _clock.monotonic() + within_s if within_s > 0.0 else 0.0
    if choose is None:
        from core.agency.looking_ahead import look_ahead as choose  # noqa: PLC0415
    without: list[float] = []
    with_it: list[float] = []
    changed = {**weights, **{name: float(worth) for name, worth in trying.items()}}
    for time in range(max(1, int(times))):
        kwargs = dict(
            toward=toward, how_far=how_far, looks_ahead=looks_ahead, seed=seed + time,
            choose=choose, until=until,
        )
        before = _one(knows, world, start, actions, weights=dict(weights), **kwargs)
        after = _one(knows, world, start, actions, weights=changed, **kwargs)
        if before is None or after is None:
            return None
        without.append(before)
        with_it.append(after)
    rehearsed = Rehearsed(tuple(without), tuple(with_it), what=", ".join(trying))
    logger.info("%s", rehearsed.says())
    return rehearsed
