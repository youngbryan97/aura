"""When more thought changes the answer: the chance that thinking one pass further would change what she has decided,
learned from her own decisions that went on thinking.

A decision made by thinking longer and longer (a search that looks a ply deeper each pass, a plan worked out a step
further each time) leaves a record: after each pass, what it would decide now and how it judges that. Some passes
change the decision; most do not. Whether the next one will is what a thinker on a clock must judge to know when to
stop, and it can be learned. Each pass she completes is a case: how often the answer has changed lately, how many
passes it has held, how far its judgement just fell, how far it has looked, and whether the pass after it changed the
answer. A logistic curve is fitted to the cases.

Each kind of thinking she does this way keeps its own record and curve, since what a pass is and how far a judgement
moves differ between them. Her searches over a board (core/agency/searching_by_kind.py) add their passes, and her
compiled search reads the curve as habits (rust_extensions/aura_board_search, Habits, the flip_ ones). Her looks ahead
in a world she acts in (core/agency/looking_ahead.py) add theirs and stop by theirs. Either way the time meant is
scaled by the chance the next pass changes the answer against how often a pass does: steady decisions give their time
to unsteady ones, and on average the time is what was meant. A curve is fitted again and kept when what she played is
kept.

Measured 23 November 2026: fitted on 40,650 of her passes from clocked games (a pass changed her answer 10.8 per cent
of the time; recent changes and a falling judgement raise the chance, holding and depth lower it), then 160 games each
on a 30-second clock against Stockfish held to depth 6: 0.450 +- 0.029 spending by the curve, 0.391 +- 0.030 by the
rule set by hand.

Nothing here knows what is decided: a pass is a depth, a judgement, and whether the answer changed.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhenMoreThoughtChanges")

__all__ = ["BOARD_SEARCHES", "ENOUGH_CASES", "FADES", "LOOKS_AHEAD", "WhenMoreThoughtChanges", "cases_from",
           "how_far_apart", "kept", "noted", "settled", "signs_of"]

#: Cases kept at most, the latest: a search adds one for each pass after its first but its last.
KEPT_AT_MOST = 20_000
#: Cases, at least, before the curve is used, and passes that changed the answer among them: fewer is a few stories.
ENOUGH_CASES = 300
ENOUGH_CHANGES = 30
#: How much of an earlier change still counts at each later pass: the compiled search's own (Habits.unsteady_kept).
FADES = 0.5
#: How far a judgement's fall counts, in the search's margins, either way.
FALL_COUNTED = 4.0
#: Passes held counted at most: past this, holding longer says little more.
HELD_COUNTED = 6

#: The kinds of thinking a record is kept for, each with its own curve.
BOARD_SEARCHES = "board-searches"
LOOKS_AHEAD = "looks-ahead"

#: Where the records are kept beside her other records (core/runtime/what_she_learned.py).
_KEPT_FOR = "her own searches"
_KEPT_AS = "when-more-thought-changes"


def _signs(rows: list[tuple[float, float, bool]], unit: float) -> list[tuple[float, float, float, float]]:
    """The signs at each pass but the first: how often the answer has changed lately, passes held, how far the
    judgement fell in margins, and how far it had looked."""
    signs = []
    changes, held = 0.0, 0
    for i in range(1, len(rows)):
        depth, judged, changed = rows[i]
        changes = changes * FADES + (1.0 if changed else 0.0)
        held = 0 if changed else held + 1
        fell = max(-FALL_COUNTED, min(FALL_COUNTED, (rows[i - 1][1] - judged) / unit))
        signs.append((changes, float(min(held, HELD_COUNTED)), fell, depth))
    return signs


def _rows(passes: Sequence[Sequence[Any]]) -> list[tuple[float, float, bool]]:
    return [(float(p[0]), float(p[1]), bool(p[2])) for p in passes if len(p) >= 3]


def how_far_apart(worths: Sequence[float]) -> float:
    """How far apart the worths of a pass's choices stood, finite ones only: the unit a judgement's move is counted in
    where nothing else gives one."""
    finite = [float(w) for w in worths if math.isfinite(float(w))]
    return max(1e-6, max(finite) - min(finite)) if len(finite) > 1 else 1.0


def signs_of(passes: Sequence[Sequence[Any]], margin: float) -> tuple[float, float, float, float] | None:
    """The signs at the last of ``passes``, for asking whether to go on after it; None before a second pass."""
    signs = _signs(_rows(passes), max(1e-9, float(margin or 0)))
    return signs[-1] if signs else None


def cases_from(passes: Sequence[Sequence[Any]], margin: float) -> list[tuple[float, float, float, float, int]]:
    """Each pass of one decision but its first and its last, as (changes lately, passes held, how far its judgement
    fell in margins, how far it had looked, whether the next pass changed the answer). ``passes`` are (depth,
    judgement, whether the answer changed at it, ...) in order."""
    rows = _rows(passes)
    signs = _signs(rows, max(1e-9, float(margin or 0)))
    # The sign at pass i + 1 (the second pass on) with whether the pass after it changed the answer; the last has none.
    return [(*signs[i - 1], int(rows[i + 1][2])) for i in range(1, len(rows) - 1)]


@dataclass
class WhenMoreThoughtChanges:
    """Her passes of thought with whether the next changed the answer, and the curve fitted to them."""

    cases: list[tuple[float, float, float, float, int]] = field(default_factory=list)
    #: The curve: chance = 1 / (1 + exp(-(level + changes, held, fell and depth each by its weight))).
    weights: tuple[float, float, float, float, float] | None = None
    #: How often a pass changed the answer, over the cases fitted.
    mean: float = 0.0
    added: int = 0

    def passed(self, passes: Sequence[Sequence[Any]], margin: float) -> None:
        """One decision's passes, as its search recorded them."""
        new = cases_from(passes, margin)
        if new:
            self.cases.extend(new)
            del self.cases[:-KEPT_AT_MOST]
            self.added += len(new)

    def fit(self) -> bool:
        """The curve, by Newton's method on the log-likelihood with each sign scaled to its own spread; False where
        there are too few cases, or too few that changed, to say anything."""
        import numpy as np

        cases = np.asarray(self.cases, dtype=float).reshape(-1, 5)
        changed = int(cases[:, 4].sum())
        if len(cases) < ENOUGH_CASES or changed < ENOUGH_CHANGES or changed == len(cases):
            return False
        spreads = np.maximum(1e-9, np.sqrt((cases[:, :4] ** 2).mean(axis=0)))
        xs = np.hstack([np.ones((len(cases), 1)), cases[:, :4] / spreads])
        ys = cases[:, 4]
        # A pull toward a flat curve, so a sign seen rarely does not decide alone; the level is not pulled.
        pull = np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
        w = np.zeros(5)
        for _ in range(30):
            p = 1.0 / (1.0 + np.exp(-np.clip(xs @ w, -30.0, 30.0)))
            grad = xs.T @ (ys - p) - pull @ w
            hess = (xs * (p * (1 - p))[:, None]).T @ xs + pull
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            w = w + step
            if np.abs(step).sum() < 1e-7:
                break
        self.weights = (float(w[0]), *(float(w[k + 1] / spreads[k]) for k in range(4)))
        self.mean = changed / len(cases)
        return True

    def goes_on(self, passes: Sequence[Sequence[Any]], margin: float, spent: float, meant: float) -> bool | None:
        """Whether to think another pass after ``passes``, having spent ``spent`` of the ``meant`` time: while what has
        gone is short of the time meant scaled by the chance the next pass changes the answer against how often one
        does. None where the curve says nothing yet."""
        signs = signs_of(passes, margin)
        chance = None if signs is None else self.chance(*signs)
        if chance is None or self.mean <= 0:
            return None
        return spent < meant * chance / self.mean

    def chance(self, changes: float, held: float, fell: float, depth: float) -> float | None:
        """The chance the next pass changes the answer, given these signs; None until fitted."""
        if self.weights is None:
            return None
        level, *by = self.weights
        z = level + sum(b * s for b, s in zip(by, (changes, held, fell, depth), strict=True))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def as_habits(self) -> dict[str, float]:
        """The curve as the compiled search reads it; nothing until fitted."""
        if self.weights is None or self.mean <= 0:
            return {}
        level, changes, held, fell, depth = self.weights
        return {"flip_mean": self.mean, "flip_level": level, "flip_changes": changes, "flip_held": held,
                "flip_fell": fell, "flip_depth": depth}

    def as_bytes(self) -> bytes:
        return json.dumps({"cases": [[round(c[0], 3), c[1], round(c[2], 2), c[3], c[4]] for c in self.cases]}).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> WhenMoreThoughtChanges:
        held = json.loads(data)
        kept_now = cls(cases=[(float(a), float(b), float(c), float(d), int(e))
                              for a, b, c, d, e in held.get("cases", [])][-KEPT_AT_MOST:])
        kept_now.fit()
        return kept_now


_KEPT: dict[str, WhenMoreThoughtChanges] = {}


def kept(what: str = BOARD_SEARCHES) -> WhenMoreThoughtChanges:
    """Her record of her own passes of one kind of thinking, read back once."""
    if what not in _KEPT:
        _KEPT[what] = WhenMoreThoughtChanges()
        try:
            from core.runtime.what_she_learned import recall_beside

            data = recall_beside(_KEPT_FOR, f"{_KEPT_AS}-{what}")
            if data:
                _KEPT[what] = WhenMoreThoughtChanges.from_bytes(data)
        # not a failure: a record that cannot be read back is gathered afresh.
        except (OSError, TypeError, ValueError, KeyError) as why:
            logger.info("her record of when more thought changed her answer was not read back (%s)", why)
    return _KEPT[what]


def noted(passes: Sequence[Sequence[Any]], margin: float, what: str = BOARD_SEARCHES) -> None:
    """A decision's passes, added to the record in memory; the curve is not fitted again until ``settled``, so the
    decisions of one game all spend their time by the same curve."""
    kept(what).passed(passes, margin)


def settled(what: str = BOARD_SEARCHES) -> dict[str, float]:
    """Fit the curve again on what has been added, and keep the record: called where what she played is kept, off
    the event loop. Returns the curve as habits."""
    record = kept(what)
    if record.added:
        record.fit()
        record.added = 0
        try:
            from core.runtime.what_she_learned import remember_beside

            remember_beside(_KEPT_FOR, f"{_KEPT_AS}-{what}", record.as_bytes())
        # not a failure: an unkept record is gathered again.
        except OSError as why:
            logger.info("her record of when more thought changed her answer was not kept (%s)", why)
    return record.as_habits()
