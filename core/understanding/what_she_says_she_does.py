"""What a reply of hers says of doing anything: that she is doing it, that she has, that she could, or nothing.

Saying "I'm on it" is doing something only where something has begun. LIVE 23 November 2026, asked again for a game she
had not built, her whole reply was "Alright — this time I'm actually building it, not just narrating that I would", and
nothing was. So a reply is read for it, in her model's own states at the end of her turn, beside what it answers, by a
classifier fitted on invented replies (heard/she_says.tsv). Measured 23 November on 26 replies kept out of the fitting,
with a 9B of her cortex's family: 25 read as what they are; of the 142 fitted on, 97.2% out of fold.

What a reply undertakes is then begun (interface/routes/chat_carrying_out.py).
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock
from core.understanding.examples import MOST_OF_AN_EARLIER_TURN
from core.understanding.fitted_on_her_model import FittedOnHerModel

__all__ = ["REPORTS", "UNDERTAKES", "WhatSheSaysSheDoes", "get_what_she_says_she_does", "reply_transcript", "she_says"]

UNDERTAKES, REPORTS, OFFERS, NONE = "undertakes", "reports", "offers", "none"


def reply_transcript(reply: str, said: str) -> str:
    """A reply after what it answers, to the end of her turn."""
    said = " ".join(str(said or "").split())[-MOST_OF_AN_EARLIER_TURN:]
    reply = " ".join(str(reply or "").split())
    reply = reply if len(reply) <= 2 * MOST_OF_AN_EARLIER_TURN else f"{reply[:MOST_OF_AN_EARLIER_TURN]} … {reply[-MOST_OF_AN_EARLIER_TURN:]}"
    return f"<|im_start|>user\n{said}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>"


@lru_cache(maxsize=1)
def she_says() -> tuple[tuple[str, str, str], ...]:
    """Every example: (kind, reply, what it answers)."""
    rows = []
    for line in (Path(__file__).resolve().parent / "heard" / "she_says.tsv").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] in (UNDERTAKES, REPORTS, OFFERS, NONE) and parts[1].strip():
            rows.append((parts[0], parts[1].strip(), parts[2].strip() if len(parts) > 2 else ""))
    return tuple(rows)


class WhatSheSaysSheDoes(FittedOnHerModel):
    def __init__(self, *, root: Path | None = None, given: Sequence[tuple[str, str, str]] | None = None) -> None:
        super().__init__("what_she_says_she_does", lambda: [(kind, reply_transcript(reply, said)) for kind, reply, said in
                                                            (given if given is not None else she_says())],
                         kind_of_nothing=NONE, root=root)

    async def read(self, reply: str, said: str, *, client: Any) -> dict[str, float] | None:
        """The probability of each kind for a reply, or None where it is not read."""
        return await self.read_one(reply_transcript(reply, said), client=client) if str(reply or "").strip() else None


_READS: WhatSheSaysSheDoes | None = None
_READS_LOCK = checked_lock("understanding.what_she_says_she_does.singleton")


def get_what_she_says_she_does() -> WhatSheSaysSheDoes:
    global _READS
    with _READS_LOCK:
        if _READS is None:
            _READS = WhatSheSaysSheDoes()
        return _READS
