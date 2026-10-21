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

So each source domain is read along its two leading axes, learned from the
domain's own running statistics, and each meeting has a unit for every pairing
of its two domains' axes: a rate along one axis times a rate along the other,

    sigma(x_i) * sigma(y_j),   sigma(v) = (1 + tanh(v)) / 2,

a product of two non-negative tuning curves. A signed product of two centred
signals is nearly invisible to anything that reads rank-linear structure, and it
is not what a rate is; a product of rates carries each domain's own effect and
their interaction together, which is what a gain field is. Each unit carries
over from the cycle before at her affect's momentum.

    feeling and self       meet at attention            A, S -> G
    perception and memory  meet at the world model      P, M -> W
    the world and feeling  meet at deliberation         W, A -> D
    self and intention     meet at recurrent cognition  S, D -> C

The units are bound as a cycle closes, from where each domain ended it, and they
are the context the next cycle runs in. Each meeting's units live in the domain
they meet at and act there:

- attention: each source's bid is scaled by a gain the units set;
- the world model: the size of each learning step is;
- deliberation: each initiative's score is;
- recurrent cognition: all sixteen units drive their own band of the
  substrate, so every binding reaches the recurrent core.

A gain is exp(w . m / 2), m the units each in its own spread and w a fixed unit
direction per source, initiative kind, tier or learner, seeded from its name, so
units at their own mean leave everything as they were.

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
    "standing_of",
]

#: Units at each meeting.
UNITS: int = 4  # AXES * AXES

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


def _organ_reading(module: str, getter: str, method: str) -> Mapping[str, Any]:
    """A mapping one of her organs publishes about itself, or nothing when it cannot say."""
    try:
        organ = getattr(__import__(module, fromlist=[getter]), getter)()
        reading = getattr(organ, method)()
    # not a failure: an organ that is not built here has nothing to say about itself.
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("%s.%s unread for the conjunctions: %s", module, method, exc)
        return {}
    return reading if isinstance(reading, Mapping) else {}


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


#: Deliberation's own readings of the turn, each a field some module of hers
#: writes under `cognition` and reads back when she weighs what to do.
_DELIBERATION_FIELDS: tuple[str, ...] = (
    "scale.reach", "scale.pressure", "persona_gap.gap", "constancy.theirs",
    "averted.share_declined", "averted.held", "particular.unique", "particular.anchored",
    "closing_window.closing", "impulse.distrust", "fuel.share_self", "returning.pull",
    "telling.urge", "catharsis.drain", "witness.company", "the_kind.cost_to_source",
    "never_told.scarcity",
)

#: The forces on her drives this turn (core/phases/motivation_update.py).
_FORCES: tuple[str, ...] = ("pressure", "social_hold", "warmth_return", "attended_credit", "resolve_hold")


def _texts(items: Any, last: int) -> str:
    held = list(items)[-last:] if isinstance(items, list) else []
    return " ".join(
        str(item.get("content") or item.get("text") or "") if isinstance(item, Mapping) else str(getattr(item, "content", item))
        for item in held
    )


def readings(state: Any) -> dict[str, list[float]]:
    """Each source domain as the numbers she keeps about it, in a fixed order.

    In full. The first version read a handful of numbers a domain: identity
    stability for the self, which barely moves, and the drive budgets for
    deliberation, which move by a hundred-thousandth of their range a turn. On seed 7 at 370 rounds the units bound from those carried 0.02 of
    joint information about the domains they were meant to bind, because they
    were not about what those domains were doing. A position is always one field.
    """
    emotions = _dig(state, "affect.emotions", {}) or {}
    budgets = _dig(state, "motivation.budgets", {}) or {}
    forces = _dig(state, "motivation.forces", {}) or {}
    partner = _dig(state, "world.partner_register", {}) or {}
    cadence = _dig(state, "cognition.partner_cadence", {}) or {}
    together = _dig(state, "cognition.togetherness", {}) or {}
    working = _dig(state, "cognition.working_memory", []) or []
    retrieved = _dig(state, "cognition.long_term_memory", []) or []
    relived = _dig(state, "cognition.relived", {}) or {}
    habits = _dig(state, "cognition.habits", {}) or {}
    percepts = _dig(state, "world.recent_percepts", []) or []
    predicted = _self_prediction()
    agency = _organ_reading("core.agency.authorship", "get_agency_ledger", "snapshot")
    comparator = _organ_reading("core.consciousness.agency_comparator", "get_agency_comparator", "get_status")
    portrait = _organ_reading("core.agency.what_she_is_like", "get_portrait_reader", "columns")

    def field(mapping: Any, name: str) -> float:
        return _number(mapping.get(name)) if isinstance(mapping, Mapping) else 0.0

    return {
        "A": [
            _number(_dig(state, "affect.valence")),
            _number(_dig(state, "affect.arousal")),
            _number(_dig(state, "affect.curiosity")),
            _number(_dig(state, "affect.engagement")),
            _number(_dig(state, "affect.ambivalence")),
            _number(_dig(state, "affect.confirmation")),
            _number(_dig(state, "affect.delivery_z")),
            *(field(emotions, name) for name in _FEELINGS),
        ],
        "S": [
            _number(_dig(state, "identity.stability")),
            _number(_dig(state, "identity.bonding_level")),
            field(predicted, "smoothed_error"),
            field(predicted, "confirmation_rate"),
            field(predicted, "surprise_rate"),
            field(predicted, "valence_error_ema"),
            field(predicted, "focus_error_ema"),
            field(agency, "efficacy"),
            field(agency, "authored_share"),
            field(comparator, "agency_score"),
            field(portrait, "narrowness"),
            field(portrait, "values_over_drive"),
            field(portrait, "values_foretell_choice"),
        ],
        "P": [
            float(len(percepts) if isinstance(percepts, list) else 0),
            max([_number(getattr(item, "salience", None)) for item in percepts[-8:]] or [0.0])
            if isinstance(percepts, list)
            else 0.0,
            float(len(str(_dig(state, "cognition.current_objective", "") or ""))),
            *_gist(str(_dig(state, "cognition.current_objective", "") or "")),
            *_gist(_texts(percepts, 8)),
        ],
        "M": [
            float(len(working) if isinstance(working, list) else 0),
            float(len(retrieved) if isinstance(retrieved, list) else 0),
            max([_number(v) for v in (_dig(state, "cognition.memory_scores", []) or [])] or [0.0]),
            1.0 if isinstance(relived, Mapping) and relived.get("relived") else 0.0,
            field(relived, "intensity"),
            field(relived, "returns"),
            *_gist(_texts(retrieved, 4)),
            *_gist(_texts(working, 4)),
        ],
        "W": [
            field(partner, "asking"),
            field(partner, "first"),
            field(partner, "second"),
            field(partner, "plural"),
            field(partner, "persistence"),
            field(together, "together"),
            field(together, "edge"),
            field(cadence, "chars"),
            field(cadence, "gap"),
            field(cadence, "placement"),
            _number(_dig(state, "world.prediction_error")),
            _number(_dig(state, "cognition.turns_since_user_spoke")),
        ],
        "D": [
            *(field(budgets.get(name) if isinstance(budgets, Mapping) else {}, "level") for name in _DRIVES),
            *(field(forces, name) for name in _FORCES),
            _urgency(state),
            float(len(_dig(state, "cognition.active_goals", []) or [])),
            float(len(_dig(state, "cognition.pending_initiatives", []) or [])),
            1.0 if field(_dig(state, "cognition.borrowed_resolve", {}) or {}, "borrowed") else 0.0,
            *(_number(_dig(state, f"cognition.{path}")) for path in _DELIBERATION_FIELDS),
            field(habits, "to_change"),
            field(habits, "largest_deficit"),
            *_gist(str(_dig(state, "cognition.last_action_source", "") or "")),
        ],
    }


# ── the code itself ──────────────────────────────────────────────────────

#: Leading axes each domain is read along. Two axes from each of two domains
#: make the four units of a meeting, one per pairing.
AXES: int = 2

#: Her affect's own carry-over from one update to the next, used when her state
#: does not say (core/state/aura_state.py, `AffectVector.momentum`).
_MOMENTUM_AT_REST: float = 0.85


def _tuned(value: float) -> float:
    """A non-negative tuning curve: a rate, from none at one extreme to full at the other."""
    return 0.5 * (1.0 + math.tanh(value))


@dataclass
class Conjunctions:
    """The units at each meeting, and what binding them has learned.

    Rides on her state so a fork carries it, as the relay's scope does.
    """

    units: dict[str, tuple[float, ...]] = field(default_factory=dict)
    _stats: dict[tuple[str, int], tuple[int, float, float]] = field(default_factory=dict)
    _last: dict[tuple[str, int], float] = field(default_factory=dict)
    _moved: dict[tuple[str, int], int] = field(default_factory=dict)
    #: Per domain: how many readings, their running covariance, and the axes read last.
    _seen: dict[str, int] = field(default_factory=dict)
    _covariance: dict[str, np.ndarray] = field(default_factory=dict)
    _axes: dict[str, np.ndarray] = field(default_factory=dict)
    _projected: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)
    #: Per unit of each meeting: its running mean and spread, for the gains.
    _unit_stats: dict[tuple[str, int], tuple[int, float, float]] = field(default_factory=dict)

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

    def along_its_axes(self, domain: str, z: np.ndarray) -> np.ndarray:
        """Where a domain stands along its leading axes, each in units of its own spread.

        The axes are the leading eigenvectors of the domain's running covariance,
        the directions Hebbian learning of a domain's inputs converges to (Oja
        1982; Sanger 1989), computed from the running statistics directly so no
        learning rate is chosen. Each is signed to agree with the one before it,
        so an axis does not flip between two cycles.
        """
        n = self._seen.get(domain, 0) + 1
        self._seen[domain] = n
        rate = 1.0 / min(n, _WINDOW)
        held = self._covariance.get(domain)
        if held is None or held.shape != (z.size, z.size):
            held = np.zeros((z.size, z.size))
        held = (1.0 - rate) * held + rate * np.outer(z, z)
        self._covariance[domain] = held
        values, vectors = np.linalg.eigh(held)
        axes = vectors[:, ::-1][:, :AXES].T
        before = self._axes.get(domain)
        if before is not None and before.shape == axes.shape:
            signs = np.sign(np.sum(axes * before, axis=1))
            axes = axes * np.where(signs == 0, 1.0, signs)[:, None]
        self._axes[domain] = axes
        out = np.zeros(AXES)
        for k in range(AXES):
            value = float(axes[k] @ z) if values[::-1][k] > 0.0 else 0.0
            count, mean, var = self._projected.get((domain, k), (0, 0.0, 0.0))
            count += 1
            r = 1.0 / min(count, _WINDOW)
            mean += r * (value - mean)
            var = (1.0 - r) * (var + r * (value - mean) ** 2)
            self._projected[(domain, k)] = (count, mean, var)
            out[k] = (value - mean) / math.sqrt(var) if var > 0.0 else 0.0
        return out

    def standing(self, meeting: str, values: Sequence[float]) -> np.ndarray:
        """Each unit of a meeting from its running mean, in its running spread."""
        out = np.zeros(len(values))
        for k, value in enumerate(values):
            n, mean, var = self._unit_stats.get((meeting, k), (0, 0.0, 0.0))
            n += 1
            rate = 1.0 / min(n, _WINDOW)
            mean += rate * (value - mean)
            var = (1.0 - rate) * (var + rate * (value - mean) ** 2)
            self._unit_stats[(meeting, k)] = (n, mean, var)
            out[k] = (value - mean) / math.sqrt(var) if var > 0.0 else 0.0
        return out

    def of(self, target: str) -> tuple[float, ...]:
        """The units living in a domain, or rest when none has been bound."""
        for meeting in MEETINGS:
            if meeting.target == target:
                return self.units.get(meeting.name, (0.0,) * UNITS)
        return (0.0,) * UNITS


def _units(x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    """A gain field per pairing of the two domains' leading axes: rate times rate."""
    return tuple(_tuned(float(x[i])) * _tuned(float(y[j])) for i in range(AXES) for j in range(AXES))


def bind(state: Any) -> Conjunctions | None:
    """Bind this cycle's domains into the four codes, on her state. None when switched off.

    Each unit carries over from the cycle before at the rate her feelings carry
    over, her affect's momentum, and takes the rest from this cycle's gain field.
    Bound fresh each cycle, the units moved with nothing that persists, and the
    synergy between two domains and the third read at a level the eight-condition
    schedule reproduced; integrated, a cycle's change follows what the two
    domains held across the cycles before it.
    """
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
    placed = {domain: held.along_its_axes(domain, held.standardise(domain, values)) for domain, values in read.items()}
    momentum = _number(_dig(state, "affect.momentum", _MOMENTUM_AT_REST))
    momentum = min(1.0, max(0.0, momentum if momentum > 0.0 else _MOMENTUM_AT_REST))
    for meeting in MEETINGS:
        fresh = _units(placed[meeting.sources[0]], placed[meeting.sources[1]])
        before = held.units.get(meeting.name)
        if before is None or len(before) != len(fresh):
            held.units[meeting.name] = fresh
        else:
            held.units[meeting.name] = tuple(
                momentum * old + (1.0 - momentum) * new for old, new in zip(before, fresh, strict=True)
            )
        held.standing(meeting.name, held.units[meeting.name])
    return held


def gain(state_or_codes: Any, meeting: str, consumer: str) -> float:
    """The multiplier one consumer at a meeting takes from its units; 1.0 at rest or when off."""
    if disabled():
        return 1.0
    codes = state_or_codes if isinstance(state_or_codes, Conjunctions) else getattr(state_or_codes, "conjunctions", None)
    if not isinstance(codes, Conjunctions):
        return 1.0
    return gain_from_units(standing_of(codes, meeting), meeting, consumer)


def standing_of(codes: Conjunctions, meeting: str) -> tuple[float, ...]:
    """Each unit of a meeting from its running mean in its spread, without updating either."""
    values = codes.units.get(meeting, ())
    out = []
    for k, value in enumerate(values):
        _n, mean, var = codes._unit_stats.get((meeting, k), (0, 0.0, 0.0))
        out.append((value - mean) / math.sqrt(var) if var > 0.0 else 0.0)
    return tuple(out)


def band(codes: Conjunctions) -> list[float]:
    """All sixteen units in meeting order, each where it stands against its own range, for the substrate."""
    return [
        math.tanh(value)
        for meeting in MEETINGS
        for value in (standing_of(codes, meeting.name) or (0.0,) * UNITS)
    ]


def gain_from_units(units: Sequence[float] | None, meeting: str, consumer: str) -> float:
    """`gain` for a consumer that holds its meeting's units rather than her state."""
    if disabled() or not units:
        return 1.0
    values = np.asarray(tuple(units), dtype=np.float64)
    w = _direction(values.size, meeting, "consumer", consumer)
    return float(math.exp(0.5 * float(w @ values)))


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


#: Input matrices for the substrate, one per size, drawn once.
_INPUT: dict[tuple[int, int], np.ndarray] = {}


def _input_matrix(neurons: int, units: int) -> np.ndarray:
    held = _INPUT.get((neurons, units))
    if held is None:
        held = _seeded("substrate", "conjunction input", str(neurons), str(units)).standard_normal((neurons, units))
        held /= np.linalg.norm(held, axis=0, keepdims=True)
        _INPUT[(neurons, units)] = held
    return held


def substrate_current(substrate: Any, recurrent: np.ndarray) -> np.ndarray | None:
    """Self and intention as an input current to the substrate's own transition.

    dx/dt = -x + tanh(W x + B phi): phi is where the units of the meeting of her
    self and what she means to do stand against their own range, and B a fixed
    input matrix seeded from its name. The current is linear in phi, so units at
    their running mean put nothing in. At a typical standing, one spread per
    unit, its root mean square equals the running root mean square of the
    recurrent drive W x: the two enter the transition at the same strength, and
    no weight is chosen here. The band reached sixteen of 512 neurons by setting
    their values; this reaches every neuron through the dynamics. None when
    nothing is bound.
    """
    drive = getattr(substrate, "conjunction_drive", None)
    if disabled() or not drive:
        return None
    phi = np.asarray(tuple(drive), dtype=np.float64)
    recurrent = np.asarray(recurrent, dtype=np.float64).reshape(-1)
    count, mean_square = getattr(substrate, "_recurrence_power", None) or (0, 0.0)
    count += 1
    mean_square += (float(np.mean(recurrent * recurrent)) - mean_square) / min(count, _WINDOW)
    substrate._recurrence_power = (count, mean_square)
    if mean_square <= 0.0:
        return None
    # A unit-norm column puts 1 / sqrt(n) per neuron, and a typical phi has
    # norm sqrt(units).
    scale = math.sqrt(mean_square * recurrent.size / phi.size)
    return scale * (_input_matrix(recurrent.size, phi.size) @ phi)


def with_conjunction_current(substrate: Any, recurrent: Any) -> Any:
    """The recurrent drive, a torch tensor, with the current added when one is bound."""
    current = substrate_current(substrate, recurrent.detach().cpu().numpy())
    if current is None:
        return recurrent
    import torch

    return recurrent + torch.from_numpy(current.astype(np.float32)).to(recurrent.device)
