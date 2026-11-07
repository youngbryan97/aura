"""What moves her in what she watches, as it plays: read off her own expectations, by sight, sound and words.

Nothing in her reacted to a video between her remarks. Her senses kept what she saw and heard as text, and what decided
whether she spoke was how unlike one look's words were from the last's (LIVE 22 October 2026). In people, most of what
makes a moment of a film catch them runs on expectation:

- the eye goes to what is surprising, where surprise is how much new data changes what the watcher believes (Itti and
  Baldi 2005: 72% of gaze shifts on TV and games went to places more surprising than average);
- people mark where one event ends and the next begins where their predictions of what comes next fail, and remember
  those moments best (Zacks's event segmentation theory). With a language model reading a story, the marks followed
  how much each word moved its prediction of the next, not how unlikely each word was (Kumar, Zacks, Norman and Hasson);
  surprise read over the next several words tracks what people call surprising, over one word it does not (Zhang 2025);
- sudden, loud sounds move people by a brainstem reflex, the startle (Juslin's BRECVEMA), which habituates fast;
- surprise is how far a belief moves, suspense how far it may move next (Ely, Frankel and Kamenica 2015);
- interest needs something new and the means to make sense of it, or it is confusion (Silvia), and what is rewarding
  is getting better at it, while what cannot be learned at all, like static, is left (Schmidhuber's compression
  progress; Ten, Kaushik, Oudeyer and Gottlieb 2021);
- the same stimulus again moves less, the more often it comes; what was habituated recovers with time; and the
  decrement is specific to the stimulus (Rankin et al. 2009, "Habituation revisited");
- boredom is wanting to engage and being unable to (Eastwood et al. 2012), and attention samples more where there is
  more to learn (Gottlieb and Oudeyer's active sampling);
- people watching together talk when something on screen is notable, and need not talk at all (Gerhardt).

So she reads what plays on three channels, each against her own expectation of it:

- **words**: each stretch she hears, read by her own language model against what she heard before it, with nothing
  sampled (core/brain/llm/what_she_expects_next.py ``surprise_of``): how unexpected its words were, and how far it
  moved what she expects next. A stretch far less expected than any she made sense of was not made out (a misheard
  line), and moves nothing;
- **sight**: how far each look departed from what her eyes expected of it (core/perception/what_her_senses_expect.py);
- **sound**: how far each second's loudness departed from what her ear expected, and its onsets, the startle's
  stimulus.

A reading moves her as far as it stands out from this video's readings on its channel: how many of their spreads it
lies above their middle (a robust standard score). Within ``Z_ORDINARY`` it is what this video is like and is no
evidence of anything; at ``Z_FULL`` it is all the evidence there is. A startle needs no history of the video to stand
out. (Ranked instead, the top third of ordinary speech's loudness read as moving and habituated her to the bang
after it.) Each channel habituates: a reading's evidence is
divided by one plus the moving readings before it on that channel, each weighted by how alike it is (for words, the
stems they share) and by how recently it came (recovering over ``RECOVERS_S`` of the video). The channels join as
independent evidence (noisy-OR): two together move her more than either alone.

What moved her most since she last spoke is what a remark of hers is worth (core/skills/watching_together.py
``WorthSaying``), and what her appraisal of a turn reads as her latest surprise while she watches
(core/phases/affect_appraisal.py). It sets how often she looks (``look_in``): sooner when something is moving her,
later when nothing has for a while. Its spikes split what she watches into events, which are what she keeps of it. Every
reading is kept with the watching, to be set beside what moves people (tools/what_moved_her_and_people.py).
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Together.Moves")

__all__ = ["FEWEST_READINGS", "MOVED_AT", "Reading", "WhatMovesHer", "her_expectations", "moved_while_watching"]

#: The fewest words of what she heard one reading is taken over: a stretch of it, not a word.
FEWEST_WORDS = 6
#: The most words one stretch holds; more waiting than this and the oldest go unread.
MOST_WORDS = 60
#: The characters of what she heard before a stretch that it is read against.
BEFORE_CHARS = 700
#: The fewest readings on a channel before where one stands among them can be told: of words, of looks, of seconds.
FEWEST_READINGS, FEWEST_LOOKS, FEWEST_SECONDS = 6, 6, 10
#: How far she must have been moved, all channels together, for a remark to be worth it: one channel's reading half
#: way from ordinary to as moving as it gets (3.5 spreads above the middle) on its own, or less on two. To be set from
#: her readings and the human curves.
MOVED_AT = 0.5
#: How far from the middle of a channel's readings, in their spreads, a reading is still what the video is like, and
#: how far it has to be to be all the evidence there is.
Z_ORDINARY, Z_FULL = 2.0, 5.0
#: The least spread read as one on each channel: words in nats a token, a look and a second of sound in nats.
LEAST_SPREADS = {"words": 0.1, "sight": 0.05, "sound": 0.05}
#: A stretch more unexpected than the middle of this video's by this many of their spreads, scaled as a standard
#: deviation (1.4826 x the median absolute deviation), was not made out.
NOT_MADE_OUT = 3.0
#: The least spread that is read as one, in nats a token.
LEAST_SPREAD = 0.1
#: And it is not made out only if her model also finds it less expected than this, in nats a token: a striking line is
#: still speech, and word salad is far less expected than any. To be set from her readings in the first rehearsal.
LEAST_UNMADE = 7.0
#: How far above what her ear expected a second must rise, beyond an onset, to startle her fully, in decibels.
STARTLE_FULL_DB = 10.0
#: How long, in seconds of the video, a habituated response takes to recover by e (Rankin et al.: spontaneous recovery
#: over time). To be set from her readings.
RECOVERS_S = 60.0
#: A reading that splits what she watches into events: one standing out this far on how far it moved her expectation
#: of what comes next (words) or on how far it departed from what her eyes expected (sight), and no sooner than this
#: after the last split, in seconds of the video.
EVENT_STANDS, EVENT_AT_LEAST_S = 0.5, 20.0
#: How often she looks while something is moving her, and while nothing has for ``BORED_AFTER_S``, against her usual.
LOOK_SOONER, LOOK_LATER, BORED_AFTER_S = 0.5, 1.5, 90.0
#: Evidence this small is not something that moved her, for boredom and for attention.
STIRRED = 0.2


@dataclass
class Reading:
    """How one stretch of what she heard landed on her expectations."""

    at: float
    words: str
    surprisal: float
    shift: float
    open: float
    made_out: bool = True


@dataclass
class Felt:
    """One reading on one channel: where it stood, what it is evidence of, and what it was of."""

    channel: str
    at: float
    standing: float | None
    evidence: float
    what: str = ""
    stems: frozenset[str] = frozenset()


#: The watchings under way, each with what has moved her in it.
_UNDER_WAY: list[WhatMovesHer] = []


def moved_while_watching() -> float | None:
    """How far what she took in since she last spoke moved her, in the watching under way; None when she is watching
    nothing, or nothing has been read enough to tell. Her appraisal of a turn of hers reads it as her latest surprise
    while she watches (core/phases/affect_appraisal.py), so a remark of hers is felt for what moved her."""
    read = [moved[0] for moves in list(_UNDER_WAY) if (moved := moves.moved()) is not None]
    return max(read) if read else None


def _middle(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def _stands_out(before: list[float], value: float, *, least_spread: float, fewest: int) -> float | None:
    """How far ``value`` stands out from the readings before it, as evidence in [0, 1]: its robust standard score (spreads
    above their middle, the spread 1.4826 x their median absolute deviation, at least ``least_spread``) from
    ``Z_ORDINARY`` to ``Z_FULL``. None before there are ``fewest`` readings to set it against."""
    if len(before) < fewest:
        return None
    middle = _middle(before)
    spread = max(least_spread, 1.4826 * _middle([abs(other - middle) for other in before]))
    score = (float(value) - middle) / spread
    return max(0.0, min(1.0, (score - Z_ORDINARY) / (Z_FULL - Z_ORDINARY)))


def _stems(text: str) -> frozenset[str]:
    import re

    try:
        import snowballstemmer

        return frozenset(snowballstemmer.stemmer("english").stemWords(
            [w.lower() for w in re.findall(r"[A-Za-z]{4,}", text or "")]))
    except ImportError:
        return frozenset(w.lower() for w in re.findall(r"[A-Za-z]{4,}", text or ""))


def _alike(one: frozenset[str], other: frozenset[str]) -> float:
    """How alike two readings' stimuli are, in [0, 1]: the stems they share (Jaccard), or wholly where neither has any
    (a look, a sound: their channel is all there is to tell them apart by)."""
    if not one and not other:
        return 1.0
    union = one | other
    return len(one & other) / len(union) if union else 0.0


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class _Event:
    began: float
    ended: float = 0.0
    heard: list[str] = field(default_factory=list)
    seen: str = ""
    moved: float = 0.0
    what: str = ""


class WhatMovesHer:
    """What moved her in one watching, reading by reading, on each channel. ``ask(before, then)`` is her model's
    reading of a stretch of words; ``keep`` keeps each reading with the watching."""

    def __init__(self, title: str = "", *, ask: Callable[[str, str], Awaitable[dict[str, float]]] | None = None,
                 keep: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.title = " ".join(str(title or "").split())
        self.ask, self.keep = ask, keep or (lambda _row: None)
        self.heard_before = ""
        self.waiting: list[str] = []
        self.last_heard = ""
        self.readings: list[Reading] = []
        self.felt: list[Felt] = []
        self.since: list[Felt] = []
        self._raw: dict[str, list[float]] = {"sight": [], "sound": []}
        self._reading: asyncio.Task | None = None
        self.now = 0.0
        self.last_stirred = 0.0
        self.events: list[_Event] = [_Event(began=0.0)]

    # ── what she heard ────────────────────────────────────────────────

    def heard(self, at: float, words: str) -> None:
        """What was heard at ``at``; a stretch long enough, with no reading under way, is read."""
        words = " ".join(str(words or "").split())
        self.now = max(self.now, float(at or 0.0))
        if not words or words == self.last_heard:
            return
        self.last_heard = words
        self.events[-1].heard.append(words)
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
        standing = self.how_moved(reading)
        felt = self._take(Felt("words", at, standing, 0.0, then, _stems(then)))
        logger.info("🎬 words at %.0fs: surprisal %.2f, shift %.2f, open %.2f%s, stands %s, evidence %.2f: %s", at,
                    reading.surprisal, reading.shift, reading.open, "" if reading.made_out else " (not made out)",
                    "-" if standing is None else f"{standing:.2f}", felt.evidence, then[:120])
        self.keep({"kind": "moved", "channel": "words", "at": at, "words": then[:400],
                   "surprisal": round(reading.surprisal, 4), "shift": round(reading.shift, 4),
                   "open": round(reading.open, 4), "made_out": reading.made_out,
                   "moved": None if standing is None else round(standing, 4), "evidence": round(felt.evidence, 4)})
        if reading.made_out and standing is not None:
            shift_stands = _stands_out([r.shift for r in self.readings[:-1] if r.made_out], reading.shift,
                                       least_spread=LEAST_SPREADS["words"], fewest=FEWEST_READINGS)
            self._maybe_a_new_event(at, shift_stands or 0.0)

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
        """How far a stretch stands out from this video's stretches she made sense of before it, as evidence in [0, 1]:
        the mean of how far its move of her expectation and its unexpectedness stand out from theirs. A stretch she did
        not make out moves nothing. None before there are enough to tell."""
        before = [r for r in self.readings if r.made_out and r is not reading]
        if len(before) < FEWEST_READINGS:
            return None
        if not reading.made_out:
            return 0.0
        spread = LEAST_SPREADS["words"]
        shift = _stands_out([r.shift for r in before], reading.shift, least_spread=spread, fewest=FEWEST_READINGS)
        surprisal = _stands_out([r.surprisal for r in before], reading.surprisal, least_spread=spread,
                                fewest=FEWEST_READINGS)
        return ((shift or 0.0) + (surprisal or 0.0)) / 2.0

    # ── what she saw and how it sounded ───────────────────────────────

    def saw(self, at: float, surprise: float | None, seen: str = "") -> None:
        """One look: how far it departed from what her eyes expected (core/perception/what_her_senses_expect.py)."""
        self.now = max(self.now, float(at or 0.0))
        if surprise is not None:
            raw = self._raw["sight"]
            standing = _stands_out(raw, float(surprise), least_spread=LEAST_SPREADS["sight"], fewest=FEWEST_LOOKS)
            raw.append(float(surprise))
            if standing is not None:
                self._maybe_a_new_event(at, standing)  # the look that departs is the new event's first
            felt = self._take(Felt("sight", at, standing, 0.0, " ".join(str(seen or "").split())[:200]))
            self.keep({"kind": "moved", "channel": "sight", "at": at, "surprise": round(float(surprise), 4),
                       "moved": None if standing is None else round(standing, 4), "evidence": round(felt.evidence, 4)})
        if seen and not self.events[-1].seen:
            self.events[-1].seen = " ".join(seen.split())[:300]

    def sounded(self, at: float, surprise: float, onset: float) -> None:
        """One second of sound: how far its loudness departed from what her ear expected, and how far it rose above
        that as an onset (dB beyond an onset's threshold). A startle needs no history of the video to stand."""
        self.now = max(self.now, float(at or 0.0))
        raw = self._raw["sound"]
        standing = _stands_out(raw, float(surprise), least_spread=LEAST_SPREADS["sound"], fewest=FEWEST_SECONDS)
        raw.append(float(surprise))
        if onset > 0.0:
            startle = 0.5 + 0.5 * min(1.0, float(onset) / STARTLE_FULL_DB)
            standing = startle if standing is None else max(standing, startle)
        felt = self._take(Felt("sound", at, standing, 0.0))
        if onset > 0.0 or (standing is not None and standing > 0.0):
            self.keep({"kind": "moved", "channel": "sound", "at": at, "surprise": round(float(surprise), 4),
                       "onset": round(float(onset), 2), "moved": None if standing is None else round(standing, 4),
                       "evidence": round(felt.evidence, 4)})

    # ── habituation, and how the channels join ────────────────────────

    def _take(self, felt: Felt) -> Felt:
        """Habituate a reading against the moving readings before it on its channel, and keep it."""
        if felt.standing is not None:
            fresh = felt.standing
            used = sum(prior.evidence * _alike(felt.stems, prior.stems) * math.exp(-max(0.0, felt.at - prior.at) / RECOVERS_S)
                       for prior in self.felt if prior.channel == felt.channel and prior.evidence > 0.0)
            felt.evidence = fresh / (1.0 + used)
        self.felt.append(felt)
        self.since.append(felt)
        if felt.evidence >= STIRRED:
            self.last_stirred = max(self.last_stirred, felt.at)
        if felt.evidence > self.events[-1].moved:
            self.events[-1].moved, self.events[-1].what = felt.evidence, felt.what
        return felt

    def moved(self) -> tuple[float, str] | None:
        """How far what she took in since she last spoke moved her, all channels together (noisy-OR of each channel's
        strongest evidence), and what moved her most; None when nothing since has been read enough to tell."""
        stood = [felt for felt in self.since if felt.standing is not None]
        if not stood:
            return None
        strongest: dict[str, Felt] = {}
        for felt in stood:
            if felt.channel not in strongest or felt.evidence > strongest[felt.channel].evidence:
                strongest[felt.channel] = felt
        unmoved = 1.0
        for felt in strongest.values():
            unmoved *= 1.0 - felt.evidence
        most = max(strongest.values(), key=lambda felt: felt.evidence)
        return 1.0 - unmoved, most.what

    def remarked(self) -> None:
        self.since = []

    # ── what it does to her attention, and what she keeps of it ───────

    def look_in(self, usual: float) -> float:
        """How long until she looks again: sooner while something is moving her, later once nothing has for a while.
        Attention samples where there is more to take in (active sampling); boredom is not being able to engage."""
        moved = self.moved()
        stirred_lately = self.last_stirred > 0.0 and self.now - self.last_stirred <= usual * 3
        if (moved is not None and moved[0] >= STIRRED) or stirred_lately:
            return usual * LOOK_SOONER
        if self.bored():
            return usual * LOOK_LATER
        return usual

    def bored(self) -> bool:
        """Whether nothing has moved her for ``BORED_AFTER_S`` of the video, once there was enough of it to tell."""
        enough = len(self._raw["sight"]) >= FEWEST_LOOKS or len(self.readings) >= FEWEST_READINGS
        return enough and self.now - self.last_stirred >= BORED_AFTER_S

    def _maybe_a_new_event(self, at: float, standing: float) -> None:
        current = self.events[-1]
        if standing < EVENT_STANDS or at - current.began < EVENT_AT_LEAST_S:
            return
        current.ended = at
        self.keep({"kind": "event", "from": round(current.began, 1), "to": round(at, 1),
                   "heard": " ".join(current.heard)[:600], "seen": current.seen, "moved": round(current.moved, 4),
                   "what": current.what[:200]})
        self.events.append(_Event(began=at))

    def events_text(self, room: int = 2400) -> str:
        """The events she took it in as, each with what was heard in it and what it showed, to fit ``room``."""
        lines = []
        for event in self.events:
            heard = " ".join(event.heard)
            if not heard and not event.seen:
                continue
            ended = event.ended or self.now
            parts = [f"{_clock(event.began)}-{_clock(ended)}"]
            if event.seen:
                parts.append(f"showed: {event.seen[:160]}")
            if heard:
                parts.append(f"said: {heard[:260]}")
            lines.append("; ".join(parts))
        text = "\n".join(lines)
        while len(text) > room and len(lines) > 2:
            lines = lines[::2] if len(lines) > 3 else lines[-2:]
            text = "\n".join(lines)
        return text[-room:]

    def getting_it(self) -> float | None:
        """How much more she expects of it now than at its start: the middle surprisal of her first third of readings
        she made sense of less that of her last third, in nats a token. What makes something rewarding to take in is
        getting better at it (Schmidhuber; Ten et al. 2021)."""
        made_out = [r.surprisal for r in self.readings if r.made_out]
        third = len(made_out) // 3
        return _middle(made_out[:third]) - _middle(made_out[-third:]) if third >= 2 else None

    # ── under way ─────────────────────────────────────────────────────

    def begin(self) -> None:
        """Under way: what moves her in it is her latest surprise while it plays."""
        if self not in _UNDER_WAY:
            _UNDER_WAY.append(self)

    def done(self) -> None:
        if self in _UNDER_WAY:
            _UNDER_WAY.remove(self)
        if self._reading is not None and not self._reading.done():
            self._reading.cancel()
        last = self.events[-1]
        if last.heard or last.seen:
            self.keep({"kind": "event", "from": round(last.began, 1), "to": round(self.now, 1),
                       "heard": " ".join(last.heard)[:600], "seen": last.seen, "moved": round(last.moved, 4),
                       "what": last.what[:200]})
        progress = self.getting_it()
        self.keep({"kind": "moved overall", "readings": len(self.readings),
                   "not made out": sum(1 for r in self.readings if not r.made_out),
                   "looks read": len(self._raw["sight"]), "seconds heard": len(self._raw["sound"]),
                   "events": len([e for e in self.events if e.heard or e.seen]),
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
