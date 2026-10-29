"""The eight ordinary situations a campaign lives through, and how each one arrives.

A condition is the world arriving before a turn: an objective, who gave it, and
what the body and the screen are doing. It is E, not K, so it is written down
before the run and results can be split by it. Moved out of core/subject/driver.py,
which re-exports every name here because 44 files ask the driver for them.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.state.percepts import emit_percept


@dataclass(frozen=True, slots=True)
class Condition:
    """One kind of life, named before the run so results can be split by it."""

    name: str
    objective: str
    origin: str = "user"
    #: Applied to the state before the turn. This is the world arriving, not
    #: the core acting, so it belongs to E rather than to K.
    prepare: Callable[[Any, random.Random], dict[str, float]] | None = None
    #: Real subsystem work done after the phases, inside the same turn.
    after: str = ""


def _stress(state: Any, rng: random.Random) -> dict[str, float]:
    load = 0.75 + 0.2 * rng.random()
    state.soma.hardware["cpu_usage"] = round(load * 100.0, 2)
    state.soma.hardware["temperature"] = round(70.0 + 25.0 * load, 2)
    state.soma.hardware["vram_usage"] = round(80.0 + 15.0 * rng.random(), 2)
    state.soma.latency["last_thought_ms"] = round(1500.0 + 2500.0 * load, 1)
    return {"host_load": load, "host_thermal": 1.0}


def _calm(state: Any, rng: random.Random) -> dict[str, float]:
    load = 0.05 + 0.15 * rng.random()
    state.soma.hardware["cpu_usage"] = round(load * 100.0, 2)
    state.soma.hardware["temperature"] = round(38.0 + 8.0 * load, 2)
    state.soma.hardware["vram_usage"] = round(30.0 + 10.0 * rng.random(), 2)
    state.soma.latency["last_thought_ms"] = round(200.0 + 400.0 * load, 1)
    return {"host_load": load, "host_thermal": 0.0}


def _percept(
    state: Any, rng: random.Random, kind: str, source: str, content: str
) -> None:
    """The world arriving, in the shape the organism's own percepts arrive in.

    The first version wrote `source` and `salience` and no `type`. The affect
    phase keys on the type and dropped every one of them; the workspace prices
    its perception bid from a strength no producer writes. So for the whole of
    every recording, the world arrived and could not be felt. The salience is
    still drawn — the environment is not uniform — and the draw is matched
    across arms because the generator's state is carried through the fork.
    """
    emit_percept(
        state.world,
        kind,
        content=content,
        intensity=round(rng.random(), 3),
        source=source,
    )


def _seen(
    source: str, contents: Sequence[str], *, kind: str = "interaction"
) -> Callable[[Any, random.Random], dict[str, float]]:
    def prepare(state: Any, rng: random.Random) -> dict[str, float]:
        reading = _calm(state, rng)
        _percept(state, rng, kind, source, rng.choice(list(contents)))
        reading["percept_arrived"] = 1.0
        return reading

    return prepare


#: The eight ordinary situations. A mechanism that only appears in a ninth,
#: written for the battery, is not a property of the ordinary agent.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "conversation",
        "Tell me what you have been thinking about today.",
        prepare=_seen("chat", ("Bryan is typing", "Bryan sent a message", "the window has focus")),
    ),
    Condition(
        "problem_solving",
        "Every A is a B, and some B are C. Does it follow that some A are C?",
        prepare=_seen("chat", ("a question arrived", "a hard question arrived")),
    ),
    Condition(
        "autonomy",
        "Review the last hour of your own behaviour and pick one thing to improve.",
        origin="motivation",
        prepare=_calm,
    ),
    Condition("idle", "", origin="system", prepare=_calm),
    Condition(
        "salience",
        "Something you did earlier caused a problem for someone. Sit with that.",
        prepare=_seen("chat", ("a correction arrived", "a complaint arrived")),
    ),
    Condition(
        "stress",
        "Keep working while the machine is under load.",
        origin="system",
        prepare=_stress,
    ),
    Condition(
        "memory",
        "What did we establish earlier that still matters?",
        prepare=_seen("chat", ("a recall request arrived",)),
        after="retrieve",
    ),
    Condition(
        "tool_use",
        "Write today's plan into notes.txt.",
        prepare=_seen("chat", ("a file request arrived",)),
        after="act",
    ),
)
