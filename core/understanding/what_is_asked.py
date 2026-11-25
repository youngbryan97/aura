"""What a turn asks for, read from her own model's states where her turn opens.

The reading is a measurement and not a question. A transcript is run through the model she thinks with, and the states
at the middle of it, at the position where she would begin to answer, are read by a linear classifier fitted on invented
examples (core/understanding/examples.py). Nothing is generated and nothing is said to the model. The meaning of the
words is what the model made of them; the classifier says which of thirteen kinds that is.

The reading is fitted in her own process, on the model that is running: the states belong to a model, so a fit does not
outlive it. The examples are read while nobody is waiting, kept by the hash of each transcript, and only what is new is
read again.

Three things a point estimate does not give are kept. The probabilities are calibrated on predictions made out of fold,
since a classifier fitted on a few hundred examples in thousands of dimensions is surer than it is right. What is done is
decided by what each way of being wrong would cost (core/understanding/effects.py): act, ask one question first, or take
it as talk. And a reading that is not ready answers nothing, so that what asks for one does what it did before.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock
from core.runtime.what_this_turn_asks import Asked
from core.understanding.effects import ASKING_FIRST, BY_NAME, TALK
from core.understanding.examples import Example, Turn, examples, transcript
from core.understanding.fitted_on_her_model import FittedOnHerModel

logger = logging.getLogger("Aura.Understanding")

__all__ = ["WhatIsAsked", "decide", "get_what_is_asked"]


def decide(p: dict[str, float]) -> tuple[str, str, float]:
    """The kind, what to do about it, and how sure it is of being asked for anything: by what each mistake would cost.

    Acting costs what doing that unasked costs, as likely as nothing was asked; where something else was asked it costs
    that and the request left unmet. Taking it as talk costs a request that went unmet, as far as an answer leaves it unmet: a
    question about the world is answered from what she knows. One question first costs a little, and settles only whether that
    kind was asked for: a request for another is still unmet after it.
    """
    asking = 1.0 - p.get(TALK, 0.0)
    kinds = [name for name in p if name != TALK]
    if not kinds:
        return TALK, "talk", 0.0

    def unmet(name: str) -> float:
        return BY_NAME[name].unmet if name in BY_NAME else 1.0

    def others(name: str) -> float:
        return sum(p[kind] * unmet(kind) for kind in kinds if kind != name)

    def acting(name: str) -> float:
        wrong = BY_NAME[name].unasked if name in BY_NAME else 1.0
        another = max(0.0, asking - p[name])
        return p.get(TALK, 0.0) * wrong + another * wrong + others(name)

    likeliest = max(kinds, key=lambda name: p[name] * unmet(name))
    ways = {"act": acting(likeliest), "ask_first": ASKING_FIRST + others(likeliest), "talk": others("")}
    decision = min(ways, key=lambda way: ways[way])
    return (TALK if decision == "talk" else likeliest), decision, asking


class WhatIsAsked(FittedOnHerModel):
    """What a turn asks for, of the thirteen kinds in core/understanding/effects.py."""

    def __init__(self, *, root: Path | None = None, given: Sequence[Example] | None = None) -> None:
        def pairs() -> list[tuple[str, str]]:
            return [(e.effect, transcript(e.said, e.before)) for e in (given if given is not None else examples())]

        super().__init__("what_is_asked", pairs, kind_of_nothing=TALK, root=root)

    async def read(self, said: str, before: Sequence[Turn] = (), *, client: Any) -> Asked | None:
        """What this turn asks for, or None where it is not read."""
        if not str(said or "").strip():
            return None
        began = time.monotonic()
        p = await self.read_one(transcript(said, before), client=client)
        if p is None:
            return None
        effect, decision, asking = decide(p)
        logger.info("🧭 Read as %s (%.2f), asking %.2f: %s, in %.2f s over %d characters.", effect, p.get(effect, 0.0),
                    asking, decision, time.monotonic() - began, len(said))
        return Asked(said=str(said), effect=effect, decision=decision, sure=p.get(effect, 0.0), asking=asking,
                     by_effect=p, at=time.time())


_READS: WhatIsAsked | None = None
_READS_LOCK = checked_lock("understanding.what_is_asked.singleton")


def get_what_is_asked() -> WhatIsAsked:
    global _READS
    with _READS_LOCK:
        if _READS is None:
            _READS = WhatIsAsked()
        return _READS
