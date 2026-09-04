"""core/consciousness/narrative_provenance.py — a rendering is not a reading.

A phenomenal narrative is produced by giving a language model a state and
asking it to write. What comes back is a rendering of that state: real, useful,
and worth keeping, but carrying no information the state did not already have.
Read it back as though it were an observation and the loop closes — the text
becomes evidence for the state that generated it, and the confidence rises
with each pass while nothing new has been measured.

Three of those loops were running here:

* the witness reflection asked what keeps returning in the experiential
  stream, and was shown three previous narratives. What keeps returning in a
  sampled text is a property of the sampler;
* synthesis depth, a measure of how rich a moment is, rose when the narrative
  was longer than a hundred characters, so a verbose model produced deeper
  experience;
* the deep narrative prompt included the last narrative, so each one was
  written partly from the one before it.

None of them were wrong about anything. Each was a plausible way to use text
that was right there. What they lacked was a way to tell a rendering from a
reading, which is what this module supplies.

:class:`Rendering` binds text to a digest of the state it came from.
:func:`evidence_grade` then answers the only question that matters at a
consumer: is this text about a state I am now looking at, an earlier one, or
is it the state I am looking at, rendered? The third case is the loop, and it
returns :data:`Grade.SELF`, which no consumer may treat as evidence.

The digest is over numbers only. Text derived from a state must never enter
the digest, or two renderings of the same state would look like two states.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

#: How long a rendering describes a state anyone should still act on. Beyond
#: this the state it was made from is gone and the text is a record, not a
#: report of now.
FRESH_FOR_S = 120.0

#: Digest precision. Two states that differ below this are the same state for
#: the purpose of asking whether a narrative is about them; without rounding,
#: float noise in an unrelated channel would make every rendering look novel.
_PLACES = 4


class Grade(StrEnum):
    """What a narrative can be evidence for."""

    #: Rendered from the state being asked about. Not evidence for it — this
    #: is the loop, and a consumer that takes it is confirming its own input.
    SELF = "self"
    #: Rendered from an earlier state. Evidence about that state, and about
    #: the fact that it was rendered, which is a real event with a timestamp.
    PRIOR = "prior"
    #: Rendered from a state old enough that nothing should act on it now.
    STALE = "stale"
    #: No state was recorded with it, so nothing can be said about what it is
    #: evidence for. Treated as SELF, because an unattributed narrative is the
    #: case the loops above were all made of.
    UNATTRIBUTED = "unattributed"


def digest(state: Mapping[str, Any]) -> str:
    """A stable digest of the numeric part of a state.

    Strings are dropped rather than hashed. A narrative is a string derived
    from the state, so admitting strings would let a rendering change the
    digest of the very state it renders, and every rendering would then look
    like it came from a state of its own.
    """
    numbers: dict[str, float] = {}
    for key, value in sorted(state.items()):
        if isinstance(value, bool):
            numbers[str(key)] = float(value)
            continue
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        numbers[str(key)] = round(number, _PLACES)
    payload = json.dumps(numbers, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Rendering:
    """Text produced from a state, and the state it was produced from."""

    text: str
    state_digest: str
    generator: str
    rendered_at: float = field(default_factory=time.time)
    #: The numbers it was written from. Kept, not just digested, because
    #: calibration needs the distance between two states and a hash has none.
    state: Mapping[str, float] = field(default_factory=dict)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.rendered_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "state_digest": self.state_digest,
            "generator": self.generator,
            "rendered_at": self.rendered_at,
            "age_s": round(self.age_s, 2),
        }


def evidence_grade(rendering: Rendering | None, current: str | None) -> Grade:
    """What this narrative may be used as evidence for, against a state."""
    if rendering is None or not rendering.state_digest:
        return Grade.UNATTRIBUTED
    if rendering.age_s > FRESH_FOR_S:
        return Grade.STALE
    if current and rendering.state_digest == current:
        return Grade.SELF
    if not current:
        return Grade.UNATTRIBUTED
    return Grade.PRIOR


def usable_as_evidence(rendering: Rendering | None, current: str | None) -> bool:
    """Whether a consumer may take this narrative as saying something new."""
    return evidence_grade(rendering, current) is Grade.PRIOR


class RenderingLog:
    """Recent renderings, with the states they came from.

    The log is what makes a series of narratives usable at all. Asked for
    what has been rendered, it can say which of them describe distinct states
    — which is the honest version of "what keeps returning", because a phrase
    repeating across renderings of one unchanged state is repetition in the
    writing and a phrase repeating across renderings of six different states
    is something about the states.
    """

    def __init__(self, maxlen: int = 32) -> None:
        self._entries: deque[Rendering] = deque(maxlen=maxlen)

    def record(self, text: str, state: Mapping[str, Any], generator: str) -> Rendering:
        numbers = {
            str(k): float(v)
            for k, v in state.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        }
        entry = Rendering(
            text=str(text),
            state_digest=digest(state),
            generator=str(generator)[:64],
            state=MappingProxyType(numbers),
        )
        self._entries.append(entry)
        return entry

    def latest(self) -> Rendering | None:
        return self._entries[-1] if self._entries else None

    def recent(self, n: int = 5) -> tuple[Rendering, ...]:
        return tuple(self._entries)[-max(0, n):]

    def over_distinct_states(self, n: int = 5) -> tuple[Rendering, ...]:
        """The most recent renderings, one per distinct state.

        Several renderings of one unchanged state say one thing several
        times. Anything counting across them counts the generator.
        """
        seen: set[str] = set()
        picked: list[Rendering] = []
        for entry in reversed(self._entries):
            if entry.state_digest in seen:
                continue
            seen.add(entry.state_digest)
            picked.append(entry)
            if len(picked) >= max(0, n):
                break
        return tuple(reversed(picked))

    def distinct_states(self) -> int:
        return len({e.state_digest for e in self._entries})

    def fidelity(self) -> Fidelity:
        """How much this generator's reports track the states behind them.

        The number the whole module exists to make possible. Cutting the
        loops stops a narrative confirming its own state; this says whether
        the narrative was worth anything, and it can come back saying no.
        """
        return fidelity((e.text, e.state) for e in self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self._entries),
            "distinct_states": self.distinct_states(),
            "latest": self.latest().to_dict() if self.latest() else None,
            "fidelity": self.fidelity().to_dict(),
        }


def dominant_label(labels: Iterable[str], *, default: str = "") -> str:
    """The most common label, ties broken by name rather than by hash order.

    `max` over a dict built from a `set` picks by insertion order on a tie,
    and a set's order varies between processes. A tie then resolves one way
    on one boot and the other way on the next, which shows up as a dominant
    mood that changes when nothing changed.
    """
    counts: dict[str, int] = {}
    for label in labels:
        name = str(label)
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return default
    return min(counts, key=lambda name: (-counts[name], name))


__all__ = [
    "FRESH_FOR_S",
    "Fidelity",
    "Grade",
    "Rendering",
    "RenderingLog",
    "digest",
    "dominant_label",
    "evidence_grade",
    "fidelity",
    "usable_as_evidence",
]


# ── introspective calibration ────────────────────────────────────────────
#
# Cutting the loops stops a narrative from confirming its own state. It does
# not say whether the narrative was any good. An introspective report is a
# measurement made by an instrument, and an instrument nobody has calibrated
# is not evidence either — it is just a reading with no error bar.
#
# The obvious calibration is to parse the report and compare what it claims
# against telemetry: "extremely distressed" against a distress channel sitting
# at 0.1. That needs a vocabulary, and a vocabulary is a list somebody wrote
# that decides in advance what she is allowed to have said. Every such list
# here has eventually been found to be the thing being measured.
#
# So the calibration is on discrimination instead. An instrument that reports
# is informative exactly to the degree that different states produce different
# reports and similar states produce similar ones. That is measurable with no
# vocabulary at all, on any generator, in any language: correlate the distance
# between two reports with the distance between the two states they were made
# from. A generator writing the same three metaphors whatever is happening
# scores zero and deserves to.


def _tokens(text: str) -> frozenset[str]:
    out: set[str] = set()
    word: list[str] = []
    for char in str(text).lower():
        if char.isalnum():
            word.append(char)
        elif word:
            out.add("".join(word))
            word = []
    if word:
        out.add("".join(word))
    return frozenset(t for t in out if len(t) > 2)


def _text_distance(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 0.0
    union = a | b
    return 1.0 - (len(a & b) / len(union)) if union else 0.0


def _state_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    shared = sorted(set(left) & set(right))
    if not shared:
        return 0.0
    total = sum((float(left[k]) - float(right[k])) ** 2 for k in shared)
    return math.sqrt(total / len(shared))


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties do not manufacture an ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    rx, ry = _rank(xs), _rank(ys)
    n = float(len(rx))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        # Every distance identical. That is a real finding — the instrument
        # is not discriminating — and it is zero, not undefined.
        return 0.0
    return num / (dx * dy)


@dataclass(frozen=True)
class Fidelity:
    """How much a generator's reports track the states they were made from."""

    #: Spearman correlation between report distance and state distance.
    observed: float
    #: The same statistic with the pairing shuffled. No null, no verdict.
    null: float
    #: How many (report, state) pairs it was computed over.
    samples: int
    #: How many distinct states those pairs covered.
    distinct_states: int

    @property
    def margin(self) -> float:
        return self.observed - self.null

    @property
    def informative(self) -> bool:
        """Whether the reports say anything about the states at all.

        Three pairs cannot settle this and the threshold is not a hedge: with
        fewer, the shuffled null has too few arrangements to be a null.
        """
        return self.samples >= _MIN_FIDELITY_SAMPLES and self.margin > _FIDELITY_MARGIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": round(self.observed, 4),
            "null": round(self.null, 4),
            "margin": round(self.margin, 4),
            "samples": self.samples,
            "distinct_states": self.distinct_states,
            "informative": self.informative,
        }


#: Pairs below this cannot separate a real correlation from an arrangement of
#: a handful of points.
_MIN_FIDELITY_SAMPLES = 6

#: How far above its own shuffled null an instrument has to score before its
#: reports are treated as saying anything.
_FIDELITY_MARGIN = 0.15

#: Shuffles averaged for the null. One shuffle is one arrangement, not a null.
_NULL_SHUFFLES = 32


def fidelity(pairs: Iterable[tuple[str, Mapping[str, float]]]) -> Fidelity:
    """Calibrate an introspective instrument against its own shuffled null.

    Pass (report, state) pairs. Every pair of pairs contributes a text
    distance and a state distance, and the statistic is how those two orders
    agree. A generator that writes the same thing whatever is happening has no
    agreement to find, which is the answer rather than a failure to compute
    one.
    """
    import random as _random

    items = [(str(text), dict(state)) for text, state in pairs]
    if len(items) < 2:
        return Fidelity(0.0, 0.0, len(items), len({digest(s) for _t, s in items}))

    text_d: list[float] = []
    state_d: list[float] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            text_d.append(_text_distance(items[i][0], items[j][0]))
            state_d.append(_state_distance(items[i][1], items[j][1]))

    observed = _spearman(text_d, state_d)

    rng = _random.Random(0xA17A)
    nulls: list[float] = []
    order = list(range(len(items)))
    for _ in range(_NULL_SHUFFLES):
        rng.shuffle(order)
        shuffled: list[float] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                shuffled.append(_state_distance(items[order[i]][1], items[order[j]][1]))
        nulls.append(_spearman(text_d, shuffled))
    null = sum(nulls) / len(nulls) if nulls else 0.0

    return Fidelity(
        observed=observed,
        null=null,
        samples=len(items),
        distinct_states=len({digest(state) for _t, state in items}),
    )
