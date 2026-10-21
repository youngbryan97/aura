"""Where two of her domains meet, units that answer to both at once.

Her domains combine by sums. A workspace bid is a priority times a few factors
that each read one thing; a drive grows at a rate that reads the substrate; the
world model learns at one step size whatever she perceived. So nothing in her
answers to a combination of two domains that neither carries alone, and the
battery said so in the only way it can: on seed 7 at 370 rounds the held-out
gain from letting two domains multiply was 0.0 for three of the four meetings
the specification names, and most of what synergy there was followed the eight
conditions of the schedule rather than what happened in a turn.

Cortex does not combine by sums. Most neurons in prefrontal cortex respond to
nonlinear mixtures of the variables of a task (Rigotti, Barak, Warden, Wang,
Daw, Miller and Fusi, "The importance of mixed selectivity in complex cognitive
tasks", Nature 497, 2013), and that mixing is what lets a downstream readout
separate situations a sum would fold together (Fusi, Miller and Rigotti, "Why
neurons mix", Current Opinion in Neurobiology 37, 2016). The simplest such
response is a gain field: a unit tuned to one variable whose gain another sets,
the product of two tuning curves (Salinas and Abbott, "A model of multiplicative
neural responses in parietal cortex", PNAS 93, 1996).

So at each of the four meetings there are `UNITS` units, and unit k answers

    tanh(u_k . x) * tanh(v_k . y)

where x and y are the two domains' readings, each field in units of its own
recent spread, and u_k, v_k are fixed random directions seeded from the
meeting's name. Random because Rigotti's units are random mixtures and a
direction chosen to move a criterion would be tuned to it; fixed because a
code that rewired itself between two arms of a trial is not one code.

    feeling and self       meet at attention            A, S -> G
    perception and memory  meet at the world model      P, M -> W
    the world and feeling  meet at deliberation         W, A -> D
    self and intention     meet at recurrent cognition  S, D -> C

The units are bound as a cycle closes, from where each domain ended it, and they
are the context the next cycle runs in. Each meeting's units live in the domain
they meet at and act there:

- attention: each source's bid is scaled by a gain the units set;
- the world model: the size of each learning step is;
- deliberation: how fast each drive grows is;
- recurrent cognition: all sixteen units drive their own band of the
  substrate, so every binding reaches the recurrent core.

A gain is exp(w . m / sqrt(UNITS)), w a fixed random direction per source,
drive or learner seeded from its name, so a unit at rest leaves everything as
it was and no scale is chosen here.

`AURA_DISABLE_CONJUNCTIONS` is the ablation.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Conjunctions")

__all__ = [
    "BAND_BASE",
    "DISABLE_ENV",
    "MEETINGS",
    "UNITS",
    "Conjunctions",
    "Meeting",
    "bind",
    "disabled",
    "gain",
    "gain_from_units",
    "into_substrate",
    "readings",
]

#: Units at each meeting.
UNITS: int = 4

#: Where the units land in the substrate. The named bands run to 70 (what she
#: means to do, at 68); the next free block holds the four meetings' sixteen.
BAND_BASE: int = 72

#: The ablation, read on every call so a probe can set it for one process.
DISABLE_ENV = "AURA_DISABLE_CONJUNCTIONS"

#: The window her ledgers share: a rate of 1 / min(n, 256).
_WINDOW = 256

#: Which ways a field has moved.
_UP, _DOWN = 1, 2


def disabled() -> bool:
    """Whether the conjunctive codes are switched off for this process."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Meeting:
    """Two domains, and the domain their conjunction lives in."""

    name: str
    target: str
    sources: tuple[str, str]


MEETINGS: tuple[Meeting, ...] = (
    Meeting("attention", "G", ("A", "S")),
    Meeting("world", "W", ("P", "M")),
    Meeting("deliberation", "D", ("W", "A")),
    Meeting("recurrent", "C", ("S", "D")),
)


def _seeded(*names: str) -> np.random.Generator:
    digest = hashlib.sha256("\x1f".join(names).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _direction(width: int, *names: str) -> np.ndarray:
    """A fixed unit direction of the given width, drawn from the names."""
    draw = _seeded(*names).standard_normal(max(1, width))
    norm = float(np.linalg.norm(draw))
    return draw / norm if norm > 0.0 else draw


# ── what each domain reads as, from her side ─────────────────────────────


def _dig(state: Any, path: str, default: Any = None) -> Any:
    node: Any = state
    for part in path.split("."):
        if node is None:
            return default
        node = node.get(part, None) if isinstance(node, Mapping) else getattr(node, part, None)
    return default if node is None else node


def _number(value: Any) -> float:
    try:
        number = float(value)
    # not a failure: a reading that is not a number is no reading.
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


#: The feelings read by name, in a fixed order, so a position is always one feeling.
_FEELINGS: tuple[str, ...] = (
    "joy", "trust", "fear", "surprise", "sadness", "anger", "anticipation",
    "interest", "wonder", "dread", "longing", "confused",
)

#: Her drives, in a fixed order (core/motivation/constants.py).
_DRIVES: tuple[str, ...] = ("curiosity", "energy", "growth", "integrity", "social")

#: Fixed directions over a text's character encoding: the gist of what a text is.
_GIST = 4


def _gist(text: str) -> list[float]:
    """Where a text falls along four fixed directions of the substrate's own encoding of it.

    `LiquidSubstrate.encode_text_to_stimulus` is how text reaches recurrent
    cognition: character frequencies and four structural features projected
    through a fixed matrix. Its body is repeated here without the instance, so
    reading a text's gist needs no substrate to be running.
    """
    if not text:
        return [0.0] * _GIST
    try:
        from core.consciousness.liquid_substrate import _text_projection
    # not a failure: no substrate module here, so the text has no gist to read.
    except ImportError:
        return [0.0] * _GIST
    hist = np.zeros(256, dtype=np.float64)
    for char in text[:512]:
        hist[ord(char) & 0xFF] += 1.0
    hist /= hist.sum() or 1.0
    size = max(1, len(text))
    features = np.array(
        [
            min(1.0, len(text) / 500.0),
            sum(1 for c in text if c in ".,!?;:") / size,
            sum(1 for c in text if c.isupper()) / size,
            sum(1 for c in text if c.isdigit()) / size,
        ]
    )
    encoded = np.tanh(_text_projection(512).astype(np.float64) @ np.concatenate([hist, features]))
    return [float(_direction(encoded.size, "gist", str(k)) @ encoded) for k in range(_GIST)]


def _urgency(state: Any) -> float:
    hardest = 0.0
    for attribute in ("active_goals", "pending_initiatives"):
        for item in list(_dig(state, f"cognition.{attribute}", []) or []):
            value = item.get("urgency") if isinstance(item, Mapping) else getattr(item, "urgency", None)
            hardest = max(hardest, _number(value))
    return hardest


def _self_prediction() -> Mapping[str, Any]:
    try:
        from core.runtime.service_registry import get_runtime_service

        organ = get_runtime_service("self_prediction", default=None)
    # not a failure: no registry here, so there is no self-prediction to read.
    except (ImportError, AttributeError, RuntimeError):
        return {}
    snapshot = getattr(organ, "get_snapshot", None)
    if not callable(snapshot):
        return {}
    try:
        reading = snapshot()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("self-prediction unread for the conjunctions: %s", exc)
        return {}
    return reading if isinstance(reading, Mapping) else {}


def readings(state: Any) -> dict[str, list[float]]:
    """Each source domain as a few signed numbers in her own units, in a fixed order."""
    emotions = _dig(state, "affect.emotions", {}) or {}
    budgets = _dig(state, "motivation.budgets", {}) or {}
    partner = _dig(state, "world.partner_register", {}) or {}
    together = _dig(state, "cognition.togetherness", {}) or {}
    working = _dig(state, "cognition.working_memory", []) or []
    newest = working[-1] if isinstance(working, list) and working else {}
    newest_text = str(newest.get("content", "") if isinstance(newest, Mapping) else newest or "")
    percepts = _dig(state, "world.recent_percepts", []) or []
    predicted = _self_prediction()
    return {
        "A": [
            _number(_dig(state, "affect.valence")),
            _number(_dig(state, "affect.arousal")),
            _number(_dig(state, "affect.curiosity")),
            _number(_dig(state, "affect.ambivalence")),
            *(_number(emotions.get(name)) if isinstance(emotions, Mapping) else 0.0 for name in _FEELINGS),
        ],
        "S": [
            _number(_dig(state, "identity.stability")),
            _number(_dig(state, "identity.bonding_level")),
            _number(predicted.get("smoothed_error")),
            _number(predicted.get("confirmation_rate")),
            _number(predicted.get("surprise_rate")),
            _number(predicted.get("valence_error_ema")),
            _number(predicted.get("focus_error_ema")),
        ],
        "P": [
            float(len(percepts) if isinstance(percepts, list) else 0),
            max([_number(getattr(item, "salience", None)) for item in percepts[-8:]] or [0.0])
            if isinstance(percepts, list)
            else 0.0,
            float(len(str(_dig(state, "cognition.current_objective", "") or ""))),
            *_gist(str(_dig(state, "cognition.current_objective", "") or "")),
        ],
        "M": [
            float(len(working) if isinstance(working, list) else 0),
            float(len(_dig(state, "cognition.memory_scores", []) or [])),
            max([_number(v) for v in (_dig(state, "cognition.memory_scores", []) or [])] or [0.0]),
            *_gist(newest_text),
        ],
        "W": [
            _number(partner.get("asking")) if isinstance(partner, Mapping) else 0.0,
            _number(partner.get("first")) if isinstance(partner, Mapping) else 0.0,
            _number(partner.get("second")) if isinstance(partner, Mapping) else 0.0,
            _number(partner.get("persistence")) if isinstance(partner, Mapping) else 0.0,
            _number(together.get("together")) if isinstance(together, Mapping) else 0.0,
            _number(_dig(state, "world.prediction_error")),
            _number(_dig(state, "cognition.turns_since_user_spoke")),
        ],
        "D": [
            *(
                _number((budgets.get(name) or {}).get("level")) if isinstance(budgets, Mapping) else 0.0
                for name in _DRIVES
            ),
            _urgency(state),
            float(len(_dig(state, "cognition.active_goals", []) or [])),
            float(len(_dig(state, "cognition.pending_initiatives", []) or [])),
        ],
    }


# ── the code itself ──────────────────────────────────────────────────────


@dataclass
class Conjunctions:
    """The units at each meeting, and the running scale each field is read in.

    Rides on her state so a fork carries it, as the relay's scope does.
    """

    units: dict[str, tuple[float, ...]] = field(default_factory=dict)
    _stats: dict[tuple[str, int], tuple[int, float, float]] = field(default_factory=dict)
    _last: dict[tuple[str, int], float] = field(default_factory=dict)
    _moved: dict[tuple[str, int], int] = field(default_factory=dict)

    def standardise(self, domain: str, values: Sequence[float]) -> np.ndarray:
        """Each field from its recent mean in its recent spread; zero until it has moved both ways.

        A field that has only ever grown is a clock, and a clock standardised is
        a trend, so it reads zero, the same as a field that has not moved. Its
        position is kept either way, so a position is always one field.
        """
        out = np.zeros(len(values), dtype=np.float64)
        for position, value in enumerate(values):
            key = (domain, position)
            last = self._last.get(key)
            if last is not None and value != last:
                self._moved[key] = self._moved.get(key, 0) | (_UP if value > last else _DOWN)
            self._last[key] = value
            n, mean, var = self._stats.get(key, (0, 0.0, 0.0))
            n += 1
            rate = 1.0 / min(n, _WINDOW)
            delta = value - mean
            mean += rate * delta
            var = (1.0 - rate) * (var + rate * delta * delta)
            self._stats[key] = (n, mean, var)
            if self._moved.get(key, 0) == _UP | _DOWN and var > 0.0:
                out[position] = (value - mean) / math.sqrt(var)
        return out

    def of(self, target: str) -> tuple[float, ...]:
        """The units living in a domain, or rest when none has been bound."""
        for meeting in MEETINGS:
            if meeting.target == target:
                return self.units.get(meeting.name, (0.0,) * UNITS)
        return (0.0,) * UNITS


def _units(meeting: Meeting, x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    out = []
    for k in range(UNITS):
        u = _direction(x.size, meeting.name, "u", str(k))
        v = _direction(y.size, meeting.name, "v", str(k))
        out.append(float(np.tanh(u @ x) * np.tanh(v @ y)))
    return tuple(out)


def bind(state: Any) -> Conjunctions | None:
    """Bind this cycle's domains into the four codes, on her state. None when switched off."""
    if disabled():
        return None
    held = getattr(state, "conjunctions", None)
    if not isinstance(held, Conjunctions):
        held = Conjunctions()
        try:
            state.conjunctions = held
        except (AttributeError, TypeError) as exc:
            logger.warning("Her state cannot hold the conjunctive codes: %s", exc)
            return None
    read = readings(state)
    scaled = {domain: held.standardise(domain, values) for domain, values in read.items()}
    held.units = {
        meeting.name: _units(meeting, scaled[meeting.sources[0]], scaled[meeting.sources[1]])
        for meeting in MEETINGS
    }
    return held


def gain(state_or_codes: Any, meeting: str, consumer: str) -> float:
    """The multiplier one consumer at a meeting takes from its units; 1.0 at rest or when off."""
    if disabled():
        return 1.0
    codes = state_or_codes if isinstance(state_or_codes, Conjunctions) else getattr(state_or_codes, "conjunctions", None)
    if not isinstance(codes, Conjunctions):
        return 1.0
    units = np.asarray(codes.units.get(meeting, ()), dtype=np.float64)
    if units.size == 0:
        return 1.0
    w = _direction(units.size, meeting, "consumer", consumer) * math.sqrt(units.size)
    return float(math.exp(float(w @ units) / math.sqrt(units.size)))


def band(codes: Conjunctions) -> list[float]:
    """All sixteen units in meeting order, for the substrate's conjunction band."""
    return [value for meeting in MEETINGS for value in codes.units.get(meeting.name, (0.0,) * UNITS)]


def gain_from_units(units: Sequence[float] | None, meeting: str, consumer: str) -> float:
    """`gain` for a consumer that holds its meeting's units rather than her state."""
    if disabled() or not units:
        return 1.0
    values = np.asarray(tuple(units), dtype=np.float64)
    w = _direction(values.size, meeting, "consumer", consumer) * math.sqrt(values.size)
    return float(math.exp(float(w @ values) / math.sqrt(values.size)))


#: How far the band is blended toward the units each time they are bound: the
#: weight every other band of the substrate is blended at
#: (`LiquidSubstrate.inject_perceptual_frame`).
_BAND_WEIGHT = 0.25


def into_substrate(substrate: Any, codes: Conjunctions) -> float:
    """Blend the sixteen units into their band of the substrate. Returns how far it moved."""
    values = np.asarray(band(codes), dtype=np.float64)
    x = getattr(substrate, "x", None)
    lock = getattr(substrate, "sync_lock", None)
    if x is None or lock is None or len(x) < BAND_BASE + values.size:
        return 0.0
    with lock:
        block = np.array(substrate.x[BAND_BASE : BAND_BASE + values.size], dtype=np.float64)
        blended = np.clip(block * (1.0 - _BAND_WEIGHT) + values * _BAND_WEIGHT, -1.0, 1.0)
        substrate.x[BAND_BASE : BAND_BASE + values.size] = blended
        marker = getattr(substrate, "mark_state_mutated_locked", None)
        if callable(marker):
            marker("conjunctive_codes")
    return float(np.abs(blended - block).mean())
