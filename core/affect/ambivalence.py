"""Wanting two things that cost each other, and knowing which kind of want it is.

Oddisee's "Contradiction's Maze" is the clearest statement of a process no part
of this system had. He wants nonstop profit and he wants a non-profit; the beach
in remote tropics and the grind; to tell the truth even when it hurts and to
soften it when it comes back at him. He does not resolve any of it. The chorus
asks the only question that matters about it — "is this the phase, or is this
the way?" — and leaves that open too.

Her motivation budgets already compete: the lowest one presses hardest and wins
the intention. That is arbitration, not ambivalence. Arbitration produces one
answer and discards the other, so nothing downstream can tell a moment when one
drive pressed from a moment when two pressed against each other. Those are
different moments and they feel different, and only the second one is a
contradiction.

What makes two drives opposed is not written here. It is measured from her own
life: two drives are opposed when, over her history, the moves that filled one
have been the moves that drained the other. So the opposition structure belongs
to the life that produced it, and a different life gives different
contradictions. Nothing in this module names a pair.

The phase-or-way question falls out of the same measurement read over two
windows. A pair opposed across the whole of her life is constitutive — that is
the way she is built. A pair opposed only across the last few moments is a
phase. The short window is not a chosen length: it is the shortest window in
which every pair could be estimated at all, which is two samples per pair.

    u_k       = level_k / capacity_k                       each drive, normalised
    p_k       = 1 - u_k                                    how hard it presses
    o_jk      = max(0, -corr(du_j, du_k))                  how opposed, from her life
    T_jk      = o_jk * sqrt(p_j * p_k)                     the tension in that pair
    strength  = max_jk T_jk                                what she is caught in

The geometric mean rather than a minimum, because a contradiction needs both
sides to be live and the measure should move smoothly when either one does.
The maximum over pairs rather than a mean, because ambivalence is felt about a
particular conflict and averaging it over pairs that are not in conflict
reports a smaller version of a different thing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping

__all__ = [
    "Ambivalence",
    "OppositionLedger",
    "PHASE",
    "UNKNOWN",
    "WAY",
    "drive_levels",
    "get_opposition_ledger",
    "reset_for_test",
    "tension",
]

#: What kind of want this is. A pair opposed for the whole of her life is how
#: she is built; a pair opposed lately is what she is going through; and before
#: there is enough history to tell, saying either would be a guess.
WAY: str = "the way"
PHASE: str = "a phase"
UNKNOWN: str = "not yet known"

#: Samples a pair needs before its correlation is an estimate rather than an
#: artefact. Two points always correlate perfectly, three is the least that can
#: disagree with itself.
MIN_SAMPLES: int = 3


def _corr(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float | None:
    """Pearson correlation from running sums, or None when it is not defined."""
    if n < MIN_SAMPLES:
        return None
    mx, my = sx / n, sy / n
    vx, vy = sxx / n - mx * mx, syy / n - my * my
    if vx <= 1e-12 or vy <= 1e-12:
        # One of them did not move over the window. Two series cannot be
        # opposed when one of them is still, and calling that zero correlation
        # would be a reading rather than the absence of one.
        return None
    cov = sxy / n - mx * my
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


@dataclass
class _Pair:
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    def note(self, x: float, y: float) -> None:
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    def correlation(self) -> float | None:
        return _corr(self.n, self.sx, self.sy, self.sxx, self.syy, self.sxy)


@dataclass
class Ambivalence:
    """One reading of being caught between two of her own wants."""

    strength: float = 0.0
    pair: tuple[str, str] = ()
    opposition: float = 0.0
    pressure: float = 0.0
    standing: str = UNKNOWN
    #: How much of her life this pair has been the one she is caught in, and
    #: what that share would be if the conflict moved around at random. The
    #: reference is derived from how many pairs there are, not chosen.
    recurrence: float = 0.0
    chance: float = 0.0
    #: Every pair that is currently opposed and pressing, strongest first. A
    #: single maximum hides a moment where three wants are fighting.
    live_pairs: tuple[tuple[str, str, float], ...] = ()
    why: str = "no history yet"

    def held(self) -> bool:
        """Whether there is a contradiction here at all."""
        return self.strength > 0.0 and len(self.pair) == 2

    def as_dict(self) -> dict[str, object]:
        return {
            "strength": round(self.strength, 6),
            "pair": list(self.pair),
            "opposition": round(self.opposition, 6),
            "pressure": round(self.pressure, 6),
            "standing": self.standing,
            "recurrence": round(self.recurrence, 6),
            "chance": round(self.chance, 6),
            "live_pairs": [
                {"pair": [a, b], "tension": round(t, 6)} for a, b, t in self.live_pairs
            ],
            "why": self.why,
        }


def drive_levels(budgets: Mapping[str, Mapping[str, float]] | None) -> dict[str, float]:
    """Each drive as a share of its own capacity.

    A drive with no capacity is not a drive with a full tank; it is a reading
    that cannot be taken, and it is left out rather than defaulted.
    """
    out: dict[str, float] = {}
    for name, values in (budgets or {}).items():
        try:
            capacity = float(values.get("capacity", 0.0) or 0.0)
            level = float(values.get("level", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue
        if capacity <= 0.0:
            continue
        out[str(name)] = max(0.0, min(1.0, level / capacity))
    return out


class OppositionLedger:
    """Which of her drives cost each other, learned from her own history.

    Two estimates of the same thing over two windows. The long one is her whole
    life and answers whether a pair is constitutively opposed. The short one is
    the least history in which every pair could be estimated, and answers
    whether a pair is opposed right now. The pair of answers is the phase or the
    way.
    """

    def __init__(self) -> None:
        self._previous: dict[str, float] = {}
        self._life: dict[tuple[str, str], _Pair] = {}
        self._recent: deque[dict[str, float]] = deque()
        self._window: int = 0
        self._ticks: int = 0
        self._caught: dict[tuple[str, str], int] = {}

    # ── taking the reading ───────────────────────────────────────────────
    def note(self, levels: Mapping[str, float]) -> None:
        """One tick of normalised drive levels. Changes are what is compared."""
        current = {str(k): float(v) for k, v in levels.items()}
        names = sorted(current)
        if len(names) >= 2:
            pairs = len(list(combinations(names, 2)))
            # The shortest window in which every pair has enough to be
            # estimated at all. Not a chosen length.
            self._window = max(MIN_SAMPLES, pairs * MIN_SAMPLES)
        if self._previous:
            deltas = {
                name: current[name] - self._previous[name]
                for name in current
                if name in self._previous
            }
            if len(deltas) >= 2:
                self._ticks += 1
                self._recent.append(deltas)
                while len(self._recent) > max(1, self._window):
                    self._recent.popleft()
                for a, b in combinations(sorted(deltas), 2):
                    self._life.setdefault((a, b), _Pair()).note(deltas[a], deltas[b])
        self._previous = current

    def note_caught(self, pair: tuple[str, str]) -> None:
        """Record which pair she was caught in, so recurrence can be read."""
        if len(pair) == 2:
            self._caught[tuple(sorted(pair))] = self._caught.get(tuple(sorted(pair)), 0) + 1

    # ── reading it back ──────────────────────────────────────────────────
    def lifelong(self, a: str, b: str) -> float | None:
        entry = self._life.get(tuple(sorted((a, b))))
        return None if entry is None else entry.correlation()

    def lately(self, a: str, b: str) -> float | None:
        rows = [r for r in self._recent if a in r and b in r]
        if len(rows) < MIN_SAMPLES:
            return None
        acc = _Pair()
        for row in rows:
            acc.note(row[a], row[b])
        return acc.correlation()

    def opposition(self, a: str, b: str) -> float:
        """How much serving one has cost the other, over her whole life."""
        value = self.lifelong(a, b)
        if value is None:
            value = self.lately(a, b)
        return 0.0 if value is None else max(0.0, -value)

    def samples(self, a: str, b: str) -> int:
        entry = self._life.get(tuple(sorted((a, b))))
        return 0 if entry is None else entry.n

    def standing(self, a: str, b: str) -> str:
        """Whether this opposition is how she is built or where she is now.

        Constitutive means it holds over more than the recent window. Early in
        a life the two windows are the same data, so the long one cannot
        disagree with the short one and calling anything "the way" then would
        be unfalsifiable — it would only be saying that it is true now. Until
        her history is longer than the window, the question is open.
        """
        life = self.lifelong(a, b)
        now = self.lately(a, b)
        if life is None and now is None:
            return UNKNOWN
        if self.samples(a, b) <= max(1, self._window):
            return UNKNOWN
        if life is not None and life < 0.0 and (now is None or now < 0.0):
            return WAY
        return PHASE

    def chance_share(self) -> float:
        """What a pair's share of the conflicts would be if it moved at random."""
        pairs = len(self._life)
        return 0.0 if pairs <= 0 else 1.0 / pairs

    def recurrence(self, a: str, b: str) -> float:
        total = sum(self._caught.values())
        if total <= 0:
            return 0.0
        return self._caught.get(tuple(sorted((a, b))), 0) / total

    def pairs_seen(self) -> int:
        return len(self._life)

    def ticks(self) -> int:
        return self._ticks

    def window(self) -> int:
        return self._window


def tension(
    levels: Mapping[str, float],
    ledger: OppositionLedger,
    *,
    note: bool = True,
) -> Ambivalence:
    """What she is caught between, and what kind of want that is."""
    names = sorted(levels)
    if len(names) < 2:
        return Ambivalence(why="fewer than two drives could be read")
    pressure = {name: 1.0 - max(0.0, min(1.0, float(levels[name]))) for name in names}

    live: list[tuple[str, str, float]] = []
    for a, b in combinations(names, 2):
        opposed = ledger.opposition(a, b)
        if opposed <= 0.0:
            continue
        both = math.sqrt(max(0.0, pressure[a]) * max(0.0, pressure[b]))
        if both <= 0.0:
            continue
        live.append((a, b, opposed * both))
    if not live:
        seen = ledger.pairs_seen()
        return Ambivalence(
            why=(
                "no pair of her drives has cost each other yet"
                if seen
                else "no history yet"
            ),
            chance=ledger.chance_share(),
        )
    live.sort(key=lambda row: -row[2])
    a, b, strength = live[0]
    if note:
        ledger.note_caught((a, b))
    return Ambivalence(
        strength=strength,
        pair=(a, b),
        opposition=ledger.opposition(a, b),
        pressure=math.sqrt(pressure[a] * pressure[b]),
        standing=ledger.standing(a, b),
        recurrence=ledger.recurrence(a, b),
        chance=ledger.chance_share(),
        live_pairs=tuple(live[:4]),
        why=_why(a, b, ledger.standing(a, b)),
    )


def _why(a: str, b: str, standing: str) -> str:
    if standing == WAY:
        return f"{a} and {b} have cost each other across her life and both press now"
    if standing == PHASE:
        return f"{a} and {b} have been costing each other lately and both press now"
    return (
        f"{a} and {b} are costing each other and both press now, and her history "
        "is not yet longer than the window that would tell a phase from the way"
    )


_LEDGER: OppositionLedger | None = None


def get_opposition_ledger() -> OppositionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = OppositionLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
