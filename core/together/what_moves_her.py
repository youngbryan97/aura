"""What moves her in what she watches, as it plays, read off her own expectations.

Nothing in her reacted to a video between her remarks. Her senses kept what she saw and heard as text, and what decided
whether she spoke was how unlike one look's words were from the last's (LIVE 22 October 2026). In people, most of what
makes a moment of a film catch them runs on expectation:

- the eye goes to what is surprising, where surprise is how much new data changes what the watcher believes (Itti and
  Baldi 2005: 72% of gaze shifts on TV and games went to places more surprising than average);
- people mark where one event ends and the next begins where their predictions of what comes next fail, and remember
  those moments best (Zacks's event segmentation theory). With a language model reading a story, the marks followed
  how much each word moved its prediction of the next, not how unlikely each word was (Kumar, Zacks, Norman and Hasson);
  surprise read over the next several words tracks what people call surprising, over one word it does not (Zhang 2025);
- surprise is how far a belief moves, suspense how far it may move next (Ely, Frankel and Kamenica 2015);
- interest needs something new and the means to make sense of it, or it is confusion (Silvia), and what is rewarding
  is getting better at it, while what cannot be learned at all, like static, is left (Schmidhuber's compression
  progress; Ten, Kaushik, Oudeyer and Gottlieb 2021);
- people watching together talk when something on screen is notable, and need not talk at all (Gerhardt).

Her language model is the expectation she has. So each stretch of what she hears is read against what she heard before
it, in one forward pass of her own model with nothing sampled (core/brain/llm/what_she_expects_next.py
``surprise_of``): how unexpected its words were, how much it moved what she expects next, and how open what comes next
is after it. A stretch moves her as much as it stands out among this video's own stretches on the first two. A stretch
far more unexpected than any she made sense of was not made out (a misheard line), and moves nothing.

What moved her most since she last spoke is what a remark of hers is worth (core/skills/watching_together.py
``WorthSaying``). Every reading is kept with the watching, so what moved her can be set beside what moves a person.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Together.Moves")

__all__ = ["FEWEST_READINGS", "MOVED_AT", "Reading", "WhatMovesHer", "her_expectations"]

#: The fewest words of what she heard one reading is taken over: a stretch of it, not a word.
FEWEST_WORDS = 6
#: The most words one stretch holds; more waiting than this and the oldest go unread.
MOST_WORDS = 60
#: The characters of what she heard before a stretch that it is read against.
BEFORE_CHARS = 700
#: The fewest readings of a video she made sense of before what moves her in it can be told from what is usual in it.
FEWEST_READINGS = 6
#: How far into what has moved her in this video a stretch must stand to be worth a remark: its top fifth.
MOVED_AT = 0.8
#: A stretch more unexpected than the middle of this video's by this many of their spreads, scaled as a standard
#: deviation (1.4826 x the median absolute deviation), was not made out.
NOT_MADE_OUT = 3.0
#: The least spread that is read as one, in nats a token: a video read alike for a while is not a reason to call the
#: next small rise a misheard line.
LEAST_SPREAD = 0.1
#: And it is not made out only if her model also finds it less expected than this, in nats a token: a striking line is
#: still speech, and word salad is far less expected than any. To be set from her readings in the first rehearsal.
LEAST_UNMADE = 7.0


@dataclass
class Reading:
    """How one stretch of what she heard landed on her expectations."""

    at: float
    words: str
    surprisal: float
    shift: float
    open: float
    made_out: bool = True


def _share_below(values: list[float], value: float) -> float:
    """Where ``value`` falls among ``values``, in [0, 1], ties counted half: alike readings stand in the middle."""
    if not values:
        return 0.0
    return (sum(1 for other in values if other < value) + 0.5 * sum(1 for other in values if other == value)) / len(values)


def _middle(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


class WhatMovesHer:
    """What moved her in one watching, reading by reading. ``ask(before, then)`` is her model's reading of a stretch;
    ``keep`` keeps each reading with the watching."""

    def __init__(self, title: str = "", *, ask: Callable[[str, str], Awaitable[dict[str, float]]] | None = None,
                 keep: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.title = " ".join(str(title or "").split())
        self.ask, self.keep = ask, keep or (lambda _row: None)
        self.heard_before = ""
        self.waiting: list[str] = []
        self.last_heard = ""
        self.readings: list[Reading] = []
        self.since: list[Reading] = []
        self._reading: asyncio.Task | None = None

    # ── what she heard ────────────────────────────────────────────────

    def heard(self, at: float, words: str) -> None:
        """What was heard at ``at``; a stretch long enough, with no reading under way, is read."""
        words = " ".join(str(words or "").split())
        if not words or words == self.last_heard:
            return
        self.last_heard = words
        self.waiting.append(words)
        stretch = " ".join(self.waiting).split()
        if len(stretch) > MOST_WORDS:
            self.waiting = [" ".join(stretch[-MOST_WORDS:])]
        if self.ask is None or len(stretch) < FEWEST_WORDS or (self._reading is not None and not self._reading.done()):
            return
        then = " ".join(self.waiting)
        before = "\n".join(part for part in (self.title, self.heard_before[-BEFORE_CHARS:]) if part)
        self.heard_before = f"{self.heard_before} {then}".strip()[-4 * BEFORE_CHARS:]
        self.waiting = []
        self._reading = asyncio.get_running_loop().create_task(self._read(at, before, then), name="what_moves_her")

    async def _read(self, at: float, before: str, then: str) -> None:
        try:
            read = await self.ask(before, then) if self.ask is not None else {}
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as exc:
            record_degradation("what_moves_her", exc, severity="info", action="a stretch she heard was not read")
            return
        if not read or not all(isinstance(read.get(key), (int, float)) for key in ("surprisal", "shift", "open")):
            return
        reading = Reading(at=at, words=then, surprisal=float(read["surprisal"]), shift=float(read["shift"]),
                          open=float(read["open"]))
        reading.made_out = self._made_out(reading.surprisal)
        self.readings.append(reading)
        self.since.append(reading)
        moved = self.how_moved(reading)
        logger.info("🎬 %s at %.0fs: surprisal %.2f, shift %.2f, open %.2f%s: %s", f"moved {moved:.2f}" if moved is not None
                    else "read", at, reading.surprisal, reading.shift, reading.open,
                    "" if reading.made_out else " (not made out)", then[:120])
        self.keep({"kind": "moved", "at": at, "words": then[:400], "surprisal": round(reading.surprisal, 4),
                   "shift": round(reading.shift, 4), "open": round(reading.open, 4), "made_out": reading.made_out,
                   "moved": None if moved is None else round(moved, 4)})

    # ── what moved her ────────────────────────────────────────────────

    def _made_out(self, surprisal: float) -> bool:
        """Whether a stretch this unexpected is one she made sense of: no further above the middle of what she made
        sense of in this video than ``NOT_MADE_OUT`` of its spreads, or less unexpected than ``LEAST_UNMADE``."""
        known = [r.surprisal for r in self.readings if r.made_out]
        if len(known) < FEWEST_READINGS:
            return True
        middle = _middle(known)
        spread = max(LEAST_SPREAD, 1.4826 * _middle([abs(value - middle) for value in known]))
        return surprisal <= max(LEAST_UNMADE, middle + NOT_MADE_OUT * spread)

    def how_moved(self, reading: Reading) -> float | None:
        """How far a stretch stands among this video's stretches she made sense of, in [0, 1]: the mean of where its
        move of her expectation and its unexpectedness fall among theirs. None before there are enough to tell."""
        made_out = [r for r in self.readings if r.made_out]
        if not reading.made_out:
            return 0.0 if len(made_out) >= FEWEST_READINGS else None
        if len(made_out) < FEWEST_READINGS:
            return None
        return (_share_below([r.shift for r in made_out], reading.shift)
                + _share_below([r.surprisal for r in made_out], reading.surprisal)) / 2.0

    def moved(self) -> tuple[float, str] | None:
        """What moved her most since she last spoke, and the words that did: None when nothing she heard since has
        been read, or too little of this video has to tell what moves her in it."""
        read = [(self.how_moved(r), r.words) for r in self.since]
        read = [(value, words) for value, words in read if value is not None]
        return max(read) if read else None

    def remarked(self) -> None:
        self.since = []

    def getting_it(self) -> float | None:
        """How much more she expects of it now than at its start: the middle surprisal of her first third of readings
        she made sense of less that of her last third, in nats a token. What makes something rewarding to take in is
        getting better at it (Schmidhuber; Ten et al. 2021)."""
        made_out = [r.surprisal for r in self.readings if r.made_out]
        third = len(made_out) // 3
        return _middle(made_out[:third]) - _middle(made_out[-third:]) if third >= 2 else None

    def done(self) -> None:
        if self._reading is not None and not self._reading.done():
            self._reading.cancel()
        progress = self.getting_it()
        self.keep({"kind": "moved overall", "readings": len(self.readings),
                   "not made out": sum(1 for r in self.readings if not r.made_out),
                   "getting it": None if progress is None else round(progress, 4)})


def her_expectations() -> Callable[[str, str], Awaitable[dict[str, float]]] | None:
    """Her model's reading of how a stretch of what she heard landed: the client her turns are served by."""
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        ask = getattr(getattr(gate, "_mlx_client", None), "surprised_by", None)
        return ask if callable(ask) else None
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("her expectations could not be reached: %s", exc)
        return None
