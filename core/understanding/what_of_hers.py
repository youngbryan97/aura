"""Which of the things she does a turn is about, read as what it asks is: by her model's states at its end.

Told by its topic, "my daughter plays the violin" is near to her playing, and "did you like it?" is near to nothing; what a
turn is about is in the exchange. The kinds and the messages on record are those of core/self/how_she_does_it.py.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock
from core.understanding.examples import Turn, transcript
from core.understanding.fitted_on_her_model import FittedOnHerModel

__all__ = ["WhatOfHers", "get_what_of_hers"]

NONE = "none"
#: How sure a reading is before how she does it is put before her.
SURE = 0.5


class WhatOfHers(FittedOnHerModel):
    def __init__(self, *, root: Path | None = None) -> None:
        def pairs() -> list[tuple[str, str]]:
            from core.self.how_she_does_it import examples

            return [(key or NONE, transcript(said, [Turn(role, words) for role, words in before])) for key, said, before in examples()]

        super().__init__("what_of_hers", pairs, kind_of_nothing=NONE, root=root)

    async def read(self, said: str, before: Sequence[Turn] = (), *, client: Any) -> list[str] | None:
        """The things she does that a turn is about, or None where it is not read."""
        p = await self.read_one(transcript(said, before), client=client) if str(said or "").strip() else None
        return None if p is None else [key for key, value in p.items() if key != NONE and value >= SURE]


_READS: WhatOfHers | None = None
_READS_LOCK = checked_lock("understanding.what_of_hers.singleton")


def get_what_of_hers() -> WhatOfHers:
    global _READS
    with _READS_LOCK:
        if _READS is None:
            _READS = WhatOfHers()
        return _READS
