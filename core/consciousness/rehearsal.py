"""What she has newly set herself to do, rehearsed while her self is in question.

People picture themselves meeting what they have just decided to do, and set
themselves up to respond that way (Bryan, 23 October). Imagining a future
episode is recurrent work: the same machinery that recalls a scene builds the
one that has not happened yet (Schacter and Addis, Philosophical Transactions B
362, 2007). And it is a self that is unsure of itself that rehearses: a settled
self meets a new intention without picturing itself in it, and a self in
question with nothing new to do has nothing to rehearse.

Self and intention met at recurrent cognition through the conjunctive codes'
current, whose self is the self-predictor's last miss. Setting her self moves
her stability, not that miss, so the self half of the meeting never saw it, and
S,D -> C read direction p 0.34 on her organism on 23 October. A band of the
substrate written at the top of a turn did no better: its own dynamics took the
value back long before the turn ended.

So the rehearsal is an input current. Each intention is anticipated at its
urgency when it first appears, and the anticipation fades at her affect's
momentum as it becomes familiar; it is kept on the intention, so it lives and
forks with her state. The rehearsal is how far her self is in question beyond
her usual (`self_conscious.in_question`) times the sum of what she anticipates.
It enters the substrate's transition along one fixed input column seeded from
its name; a rehearsal of one enters at the strength of the recurrent drive
itself, which is the strength the conjunctive codes' current has at a typical
standing.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from typing import Any

import numpy as np

__all__ = ["ANTICIPATED", "anticipating", "rehearsal_current", "rehearsing"]

#: The mark an intention carries once she has met it: how strongly she still anticipates it.
ANTICIPATED: str = "anticipated"

#: The window her ledgers share: a rate of 1 / min(n, WINDOW).
_WINDOW: int = 256

_COLUMNS: dict[int, np.ndarray] = {}


def anticipating(state: Any) -> float:
    """How much she anticipates of what she has newly set herself to do. Fades what she has met."""
    affect = getattr(state, "affect", None)
    try:
        keep = max(0.0, min(1.0, float(getattr(affect, "momentum", 0.85))))
    # not a failure: a momentum that is not a number is her default.
    except (TypeError, ValueError):
        keep = 0.85
    cognition = getattr(state, "cognition", None)
    total = 0.0
    for attribute in ("active_goals", "pending_initiatives"):
        for item in list(getattr(cognition, attribute, []) or []):
            if not isinstance(item, MutableMapping):
                continue
            if ANTICIPATED in item:
                item[ANTICIPATED] = float(item[ANTICIPATED]) * keep
            else:
                try:
                    item[ANTICIPATED] = max(0.0, min(1.0, float(item.get("urgency", 0.0) or 0.0)))
                # not a failure: an intention with no urgency is anticipated by nothing.
                except (TypeError, ValueError):
                    item[ANTICIPATED] = 0.0
            total += float(item[ANTICIPATED])
    return total


def rehearsing(state: Any, usual: Any = None) -> float:
    """How far her self is in question beyond her usual, times what she anticipates. Once a turn."""
    from core.consciousness.self_conscious import in_question

    identity = getattr(state, "identity", None)
    try:
        stability = float(getattr(identity, "stability", 1.0))
    # not a failure: a stability that is not a number is a settled self.
    except (TypeError, ValueError):
        stability = 1.0
    anticipated = anticipating(state)
    return in_question(stability, usual) * anticipated


def _column(neurons: int) -> np.ndarray:
    held = _COLUMNS.get(neurons)
    if held is None:
        import hashlib

        digest = hashlib.sha256(f"substrate\x1frehearsal\x1f{neurons}".encode()).digest()
        held = np.random.default_rng(int.from_bytes(digest[:8], "big")).standard_normal(neurons)
        held /= np.linalg.norm(held) or 1.0
        _COLUMNS[neurons] = held
    return held


def rehearsal_current(substrate: Any, recurrent: np.ndarray) -> np.ndarray | None:
    """The rehearsal as an input current. None when she is not rehearsing.

    A unit-norm column puts 1 / sqrt(n) per neuron; scaled by the running root
    mean square of the recurrent drive times sqrt(n), a rehearsal of one enters
    at that drive's own strength.
    """
    level = float(getattr(substrate, "rehearsal", 0.0) or 0.0)
    recurrent = np.asarray(recurrent, dtype=np.float64).reshape(-1)
    count, mean_square = getattr(substrate, "_rehearsal_power", None) or (0, 0.0)
    count += 1
    mean_square += (float(np.mean(recurrent * recurrent)) - mean_square) / min(count, _WINDOW)
    substrate._rehearsal_power = (count, mean_square)
    if level <= 0.0 or mean_square <= 0.0:
        return None
    return math.sqrt(mean_square * recurrent.size) * level * _column(recurrent.size)
