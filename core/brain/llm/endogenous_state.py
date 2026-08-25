"""z_Aura: the endogenous cognitive state, with every dimension named.

The substrate vector alone is 64 numbers of affect and liquid-ODE dynamics.
That is not what the architecture means by endogenous state. What is meant is
the ongoing state of the whole organism — what she wants, what she has
recalled, how sure she is, how much recurrent work she has done, where her
attention sits, whether she is oriented to what happened or to what is coming.

This module assembles that, and it does three things that a plain
concatenation would not:

* **Every dimension has a name.** ``uncertainty.confidence`` is one float at a
  fixed index, declared once. A readout trained on this vector can be asked
  which named dimension moved it, and an experiment can intervene on exactly
  one.
* **An unreachable source reads absent, not zero.** A goal system that fails
  to answer must not look like a system reporting no goals. Every state
  carries a presence mask alongside the values, and a consumer that ignores
  the mask is reading zeros it was never given.
* **Nothing here imports the organs it reads.** Sources are resolved through
  the runtime registry and duck-typed. A missing organ is a missing channel,
  not an ImportError, and no new layering edge appears because this file
  exists.

Interventions (``do``) return a new state. The live state is never mutated by
an experiment, because an experiment that edits the thing it measures has
measured nothing.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.EndogenousState")

#: Bump when the feature list changes. A head trained against one layout is
#: meaningless under another, and the digest below is what catches that.
LAYOUT_VERSION = 1

#: How many substrate dimensions are pooled into the state. Band-mean pooling,
#: not a projection: a random projection here would reintroduce exactly the
#: untrained matrix this pathway exists to replace.
SUBSTRATE_BANDS = 32


@dataclass(frozen=True)
class Feature:
    """One named dimension of z_Aura."""

    name: str
    channel: str
    meaning: str
    low: float = -1.0
    high: float = 1.0

    def clamp(self, value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(v):
            return 0.0
        return max(self.low, min(self.high, v))


def _band_features() -> tuple[Feature, ...]:
    return tuple(
        Feature(
            name=f"substrate.band_{i:02d}",
            channel="substrate",
            meaning="mean of one contiguous band of the liquid substrate state",
        )
        for i in range(SUBSTRATE_BANDS)
    )


FEATURES: tuple[Feature, ...] = (
    # ── affect ────────────────────────────────────────────────────────────
    Feature("affect.valence", "affect", "pleasant/unpleasant appraisal"),
    Feature("affect.arousal", "affect", "activation level", 0.0, 1.0),
    Feature("affect.dominance", "affect", "sense of control over the situation"),
    Feature("affect.engagement", "affect", "involvement with what is happening", 0.0, 1.0),
    Feature("affect.curiosity", "affect", "pull towards the unknown", 0.0, 1.0),
    Feature("affect.stress", "affect", "somatic stress index", 0.0, 1.0),
    Feature("affect.mobilization", "affect", "readiness to act", 0.0, 1.0),
    Feature("affect.conductance", "affect", "somatic conductance index", 0.0, 1.0),
    # ── substrate dynamics ────────────────────────────────────────────────
    *_band_features(),
    Feature("substrate.energy", "substrate", "state norm, scaled by dimension", 0.0, 1.0),
    Feature("substrate.phi", "substrate", "integration estimate", 0.0, 1.0),
    # ── goals ─────────────────────────────────────────────────────────────
    Feature("goal.active", "goal", "whether a goal is currently held", 0.0, 1.0),
    Feature("goal.priority", "goal", "priority of the top goal", 0.0, 1.0),
    Feature("goal.progress", "goal", "how far the top goal has got", 0.0, 1.0),
    Feature("goal.age", "goal", "age of the top goal, log-scaled to an hour", 0.0, 1.0),
    Feature("goal.blocked", "goal", "the top goal cannot currently advance", 0.0, 1.0),
    Feature("goal.conflict", "goal", "two held goals pull opposite ways", 0.0, 1.0),
    # ── memory activation ─────────────────────────────────────────────────
    Feature("memory.recall_hits", "memory", "how much was recalled for this moment", 0.0, 1.0),
    Feature("memory.recall_confidence", "memory", "how well grounded the recall is", 0.0, 1.0),
    Feature("memory.episodic_recency", "memory", "how recent the strongest episode is", 0.0, 1.0),
    Feature("memory.semantic_density", "memory", "how connected the active concepts are", 0.0, 1.0),
    Feature("memory.working_load", "memory", "working-set pressure", 0.0, 1.0),
    Feature("memory.contradiction", "memory", "recalled items disagree", 0.0, 1.0),
    # ── uncertainty ───────────────────────────────────────────────────────
    Feature("uncertainty.confidence", "uncertainty", "current confidence in the answer forming", 0.0, 1.0),
    Feature("uncertainty.calibration_error", "uncertainty", "recent gap between confidence and outcome", 0.0, 1.0),
    Feature("uncertainty.evidence_support", "uncertainty", "how much evidence is attached", 0.0, 1.0),
    Feature("uncertainty.abstention_pressure", "uncertainty", "pull towards saying nothing", 0.0, 1.0),
    # ── self-state ────────────────────────────────────────────────────────
    Feature("self.continuity", "self_state", "continuity with the previous cycle", 0.0, 1.0),
    Feature("self.integrity", "self_state", "runtime integrity as she can sense it", 0.0, 1.0),
    Feature("self.drift", "self_state", "distance from the stored self-model", 0.0, 1.0),
    Feature("self.agency", "self_state", "how much of the next step is hers to choose", 0.0, 1.0),
    # ── attention ─────────────────────────────────────────────────────────
    Feature("attention.focus", "attention", "narrowness of the current focus", 0.0, 1.0),
    Feature("attention.salience_peak", "attention", "strength of the most salient item", 0.0, 1.0),
    Feature("attention.novelty", "attention", "how new the salient item is", 0.0, 1.0),
    Feature("attention.load", "attention", "how much is competing for it", 0.0, 1.0),
    # ── recurrent cognition ───────────────────────────────────────────────
    Feature("recurrence.depth", "recurrence", "recurrent passes taken this turn", 0.0, 1.0),
    Feature("recurrence.convergence", "recurrence", "how much the passes agreed"),
    Feature("recurrence.delta", "recurrence", "how much the last pass changed", 0.0, 1.0),
    Feature("recurrence.budget_used", "recurrence", "share of the recurrent budget spent", 0.0, 1.0),
    # ── temporal orientation ──────────────────────────────────────────────
    Feature("temporal.past", "temporal", "orientation towards what happened", 0.0, 1.0),
    Feature("temporal.present", "temporal", "orientation towards what is happening", 0.0, 1.0),
    Feature("temporal.future", "temporal", "orientation towards what is coming", 0.0, 1.0),
    Feature("temporal.horizon", "temporal", "how far ahead the current concern reaches", 0.0, 1.0),
)

STATE_DIM = len(FEATURES)
FEATURE_INDEX: dict[str, int] = {f.name: i for i, f in enumerate(FEATURES)}
CHANNELS: tuple[str, ...] = tuple(dict.fromkeys(f.channel for f in FEATURES))
CHANNEL_SLICES: dict[str, tuple[int, ...]] = {
    channel: tuple(i for i, f in enumerate(FEATURES) if f.channel == channel)
    for channel in CHANNELS
}


def layout_digest() -> str:
    """Fingerprint of the exact feature list a head was trained against."""
    payload = json.dumps(
        {
            "version": LAYOUT_VERSION,
            "features": [[f.name, f.channel, f.low, f.high] for f in FEATURES],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


@dataclass(frozen=True)
class Intervention:
    """One ``do(feature = value)`` applied to a state."""

    feature: str
    before: float
    after: float
    was_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "before": round(self.before, 6),
            "after": round(self.after, 6),
            "was_present": self.was_present,
        }


@dataclass(frozen=True)
class EndogenousState:
    """z_Aura at one instant, with a presence mask and a provenance record."""

    values: np.ndarray
    present: np.ndarray
    sources: Mapping[str, str] = field(default_factory=dict)
    captured_at: float = 0.0
    interventions: tuple[Intervention, ...] = ()
    digest: str = ""

    def __post_init__(self) -> None:
        if self.values.shape != (STATE_DIM,):
            raise ValueError(f"z_Aura must be {STATE_DIM}-dimensional, got {self.values.shape}")
        if self.present.shape != (STATE_DIM,):
            raise ValueError("presence mask must match the state width")

    # ── reading ───────────────────────────────────────────────────────────
    def get(self, feature: str) -> float:
        return float(self.values[FEATURE_INDEX[feature]])

    def is_present(self, feature: str) -> bool:
        return bool(self.present[FEATURE_INDEX[feature]])

    def channel(self, channel: str) -> np.ndarray:
        return self.values[list(CHANNEL_SLICES[channel])]

    @property
    def coverage(self) -> float:
        """Share of dimensions a live source actually answered for."""
        return float(np.mean(self.present)) if self.present.size else 0.0

    @property
    def live_channels(self) -> tuple[str, ...]:
        return tuple(c for c, s in self.sources.items() if s == "live")

    def named(self) -> dict[str, float | None]:
        """Every dimension by name; ``None`` where nothing answered."""
        return {
            f.name: (float(self.values[i]) if self.present[i] else None)
            for i, f in enumerate(FEATURES)
        }

    # ── intervening ───────────────────────────────────────────────────────
    def do(self, **assignments: float) -> EndogenousState:
        """Return a copy with named dimensions forced to given values.

        The live state is untouched. An intervention on a dimension nothing
        answered for is allowed and recorded as such — that is the ablation
        where the value was never there to begin with.
        """
        values = self.values.copy()
        present = self.present.copy()
        applied: list[Intervention] = list(self.interventions)
        for name, value in assignments.items():
            if name not in FEATURE_INDEX:
                raise KeyError(f"no such endogenous dimension: {name}")
            index = FEATURE_INDEX[name]
            feature = FEATURES[index]
            before = float(values[index])
            values[index] = feature.clamp(value)
            applied.append(
                Intervention(name, before, float(values[index]), bool(present[index]))
            )
            present[index] = True
        return replace(
            self,
            values=values,
            present=present,
            interventions=tuple(applied),
            digest=_digest_of(values),
        )

    def ablate(self, channel: str) -> EndogenousState:
        """Return a copy with a whole channel removed — values zeroed, mask cleared.

        This is the "remove the memory and see whether she stops referring to
        it" experiment. Zeroing without clearing the mask would forge a
        reading of zero; clearing without zeroing would leave the value
        readable by anything that ignores the mask. Both happen here.
        """
        if channel not in CHANNEL_SLICES:
            raise KeyError(f"no such endogenous channel: {channel}")
        indices = list(CHANNEL_SLICES[channel])
        values = self.values.copy()
        present = self.present.copy()
        applied = list(self.interventions)
        for index in indices:
            applied.append(
                Intervention(FEATURES[index].name, float(values[index]), 0.0, bool(present[index]))
            )
            values[index] = 0.0
            present[index] = False
        sources = dict(self.sources)
        sources[channel] = "ablated"
        return replace(
            self,
            values=values,
            present=present,
            sources=sources,
            interventions=tuple(applied),
            digest=_digest_of(values),
        )

    # ── crossing a process boundary ───────────────────────────────────────
    def to_payload(self) -> dict[str, Any]:
        """Compact form for the MLX worker, which has no access to the organs.

        The worker subprocess cannot reach the substrate, the goal system or
        anything else in the parent. It does not need to: the state is 74
        floats, and shipping them with the job is cheaper than any attempt to
        share the organs across the boundary.
        """
        return {
            "layout": layout_digest(),
            "layout_version": LAYOUT_VERSION,
            "values": [round(float(v), 6) for v in self.values],
            "present": [bool(p) for p in self.present],
            "captured_at": self.captured_at,
            "coverage": round(self.coverage, 4),
            "interventions": [i.as_dict() for i in self.interventions],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EndogenousState | None:
        """Rebuild a state, or refuse if it was built against another layout."""
        if not isinstance(payload, Mapping):
            return None
        if str(payload.get("layout") or "") != layout_digest():
            return None
        try:
            values = np.asarray(payload.get("values") or [], dtype=np.float32)
            present = np.asarray(payload.get("present") or [], dtype=bool)
        except (TypeError, ValueError):
            return None
        if values.shape != (STATE_DIM,) or present.shape != (STATE_DIM,):
            return None
        if not np.all(np.isfinite(values)):
            return None
        # Interventions survive the crossing. Dropping them would let an
        # experimental state come back looking observed, and the pair recorder
        # would fold a constructed condition into the training corpus as if
        # the runtime had actually held it.
        applied: list[Intervention] = []
        for entry in payload.get("interventions") or []:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("feature") or "")
            if name not in FEATURE_INDEX:
                continue
            try:
                applied.append(
                    Intervention(
                        feature=name,
                        before=float(entry.get("before") or 0.0),
                        after=float(entry.get("after") or 0.0),
                        was_present=bool(entry.get("was_present")),
                    )
                )
            except (TypeError, ValueError):
                continue
        return cls(
            values=values,
            present=present,
            sources={},
            captured_at=float(payload.get("captured_at") or 0.0),
            interventions=tuple(applied),
            digest=_digest_of(values),
        )


def _digest_of(values: np.ndarray) -> str:
    return hashlib.blake2b(
        np.asarray(values, dtype=np.float32).tobytes(), digest_size=8
    ).hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Probes. One per channel, each duck-typed, each fail-open.
# ──────────────────────────────────────────────────────────────────────────


#: What a registry lookup is allowed to raise. Narrow on purpose: an organ
#: that is missing, half-registered, or mid-shutdown produces one of these,
#: and anything else is a fault this module should not be hiding.
_LOOKUP_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _service(name: str) -> Any:
    """Resolve a runtime organ without importing it.

    Two registries exist and neither is a superset of the other, so both are
    asked. A lookup that fails is a missing organ, which is a missing channel,
    not an error — but the failure is recorded, because a registry that raises
    on every call would otherwise look exactly like a runtime with no organs.
    """
    for source, resolve in (
        ("service_registry", _resolve_from_registry),
        ("container", _resolve_from_container),
    ):
        try:
            found = resolve(name)
        except _LOOKUP_ERRORS as exc:
            logger.debug("endogenous lookup of %s via %s failed: %s", name, source, exc)
            continue
        if found is not None:
            return found
    return None


def _resolve_from_registry(name: str) -> Any:
    from core.runtime.service_registry import get_runtime_service

    return get_runtime_service(name, default=None)


def _resolve_from_container(name: str) -> Any:
    from core.container import ServiceContainer

    return ServiceContainer.get(name, default=None)


def _first_number(source: Any, keys: Sequence[str]) -> float | None:
    """First key that yields a finite number, from a mapping or an object."""
    for key in keys:
        value: Any = None
        if isinstance(source, Mapping):
            value = source.get(key)
        else:
            value = getattr(source, key, None)
        if callable(value):
            continue
        if value is None or isinstance(value, bool):
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _log_age(seconds: Any, *, span: float = 3600.0) -> float:
    try:
        age = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(age) or age <= 0:
        return 0.0
    return min(1.0, math.log1p(age) / math.log1p(span))


def _probe_affect() -> dict[str, float] | None:
    engine = _service("affect_engine") or _service("affect_manager")
    substrate = _service("continuous_substrate") or _service("liquid_state")
    out: dict[str, float] = {}
    snapshot: Any = None
    if engine is not None and hasattr(engine, "get_snapshot"):
        try:
            snapshot = engine.get_snapshot()
        except _LOOKUP_ERRORS as exc:
            logger.debug("affect snapshot unavailable: %s", exc)
    if isinstance(snapshot, Mapping):
        for name, keys in (
            ("affect.valence", ("valence",)),
            ("affect.arousal", ("arousal",)),
            ("affect.engagement", ("engagement",)),
        ):
            value = _first_number(snapshot, keys)
            if value is not None:
                out[name] = value
        somatic = snapshot.get("somatic_indices")
        if isinstance(somatic, Mapping):
            for name, keys in (
                ("affect.stress", ("stress",)),
                ("affect.mobilization", ("mobilization",)),
                ("affect.conductance", ("conductance",)),
            ):
                value = _first_number(somatic, keys)
                if value is not None:
                    out[name] = value
    summary = _substrate_summary(substrate)
    if summary is not None:
        for name, keys in (
            ("affect.valence", ("valence",)),
            ("affect.arousal", ("arousal",)),
            ("affect.dominance", ("dominance",)),
            ("affect.curiosity", ("curiosity",)),
        ):
            if name in out:
                continue
            value = _first_number(summary, keys)
            if value is not None:
                out[name] = value
    return out or None


def _substrate_summary(substrate: Any) -> Mapping[str, Any] | None:
    if substrate is None:
        return None
    getter = getattr(substrate, "get_state_summary_nowait", None)
    if not callable(getter):
        getter = getattr(substrate, "get_state_summary", None)
    if not callable(getter) or inspect.iscoroutinefunction(getter):
        return None
    try:
        summary = getter()
    except _LOOKUP_ERRORS as exc:
        logger.debug("substrate summary unavailable: %s", exc)
        return None
    if inspect.isawaitable(summary):
        # A duck-typed source can return an awaitable without declaring an
        # async function. This probe runs on the synchronous model-request
        # path, so it must neither await nor leak that object.
        if inspect.iscoroutine(summary):
            summary.close()
        return None
    return summary if isinstance(summary, Mapping) else None


def pool_substrate(vector: Any, bands: int = SUBSTRATE_BANDS) -> np.ndarray:
    """Mean-pool a substrate state into a fixed number of contiguous bands.

    Declared and reproducible, which a random projection would not be. A state
    shorter than the band count is padded so the layout stays fixed; the
    padding reads as zero and the caller marks it present only if the source
    answered at all.
    """
    array = np.asarray(vector, dtype=np.float32).ravel()
    if array.size == 0:
        return np.zeros(bands, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.tanh(array)
    if array.size < bands:
        array = np.pad(array, (0, bands - array.size))
    edges = np.linspace(0, array.size, bands + 1).astype(int)
    return np.asarray(
        [float(array[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()) for i in range(bands)],
        dtype=np.float32,
    )


def _probe_substrate() -> dict[str, float] | None:
    substrate = _service("continuous_substrate") or _service("liquid_state")
    if substrate is None:
        return None
    out: dict[str, float] = {}
    getter = getattr(substrate, "get_state_vector", None)
    if callable(getter):
        try:
            pooled = pool_substrate(getter())
        except _LOOKUP_ERRORS as exc:
            logger.debug("substrate vector unavailable: %s", exc)
        else:
            for i, value in enumerate(pooled):
                out[f"substrate.band_{i:02d}"] = float(value)
    summary = _substrate_summary(substrate)
    if summary is not None:
        energy = _first_number(summary, ("energy",))
        if energy is not None:
            out["substrate.energy"] = energy
        phi = _first_number(summary, ("phi",))
        if phi is not None:
            out["substrate.phi"] = phi
    return out or None


def _probe_goal() -> dict[str, float] | None:
    """What she is actually trying to do, from the durable goal engine.

    ``get_active_goals`` serves a cached snapshot with a five-second TTL
    precisely because every tool authorization already descends through it, so
    reading it here costs a dict copy rather than a query.
    """
    engine = _service("goal_engine") or _service("goal_manager")
    goals: list[Any] = []
    if engine is not None and hasattr(engine, "get_active_goals"):
        try:
            goals = list(engine.get_active_goals(limit=8) or [])
        except _LOOKUP_ERRORS as exc:
            logger.debug("active goals unavailable: %s", exc)
            goals = []
    if goals:
        return _goal_features(goals)
    return _probe_goal_by_accessor()


def _goal_features(goals: Sequence[Any]) -> dict[str, float]:
    """Reduce the active set to the declared goal dimensions."""
    priorities = [
        value
        for value in (_first_number(goal, ("priority",)) for goal in goals)
        if value is not None
    ]
    top = max(range(len(goals)), key=lambda i: _first_number(goals[i], ("priority",)) or 0.0)
    leader = goals[top]
    out: dict[str, float] = {"goal.active": 1.0}
    if priorities:
        out["goal.priority"] = max(0.0, min(1.0, max(priorities)))
        # Two goals both near the top is the shape of a pull in two
        # directions. One goal, however urgent, is not a conflict.
        ranked = sorted(priorities, reverse=True)
        if len(ranked) > 1 and ranked[0] >= 0.6 and ranked[1] >= 0.6:
            out["goal.conflict"] = min(1.0, ranked[1])
        else:
            out["goal.conflict"] = 0.0
    progress = _first_number(leader, ("progress",))
    if progress is not None:
        out["goal.progress"] = max(0.0, min(1.0, progress))
    created = _first_number(leader, ("created_at", "started_at"))
    if created is not None and created > 0:
        out["goal.age"] = _log_age(time.time() - created)
    status = ""
    if isinstance(leader, Mapping):
        status = str(leader.get("status") or "")
    else:
        status = str(getattr(leader, "status", "") or "")
    out["goal.blocked"] = 1.0 if status.lower() in _BLOCKED_STATUSES else 0.0
    return out


#: Goal statuses that mean the top goal cannot advance right now. Read from the
#: engine's own vocabulary rather than inferred from progress being flat.
_BLOCKED_STATUSES = frozenset({"blocked", "stalled", "waiting", "paused", "deferred"})


def _probe_goal_by_accessor() -> dict[str, float] | None:
    """Any other organ that will hand over a single current goal."""
    for key in ("volition", "objective_manager", "goals"):
        organ = _service(key)
        if organ is None:
            continue
        goal: Any = None
        for accessor in ("current_goal", "active_goal", "get_current_goal", "top_goal"):
            candidate = getattr(organ, accessor, None)
            try:
                goal = candidate() if callable(candidate) else candidate
            except _LOOKUP_ERRORS as exc:
                logger.debug("goal accessor %s declined: %s", accessor, exc)
                goal = None
            if goal:
                break
        if goal:
            return _goal_features([goal])
    return None


def _probe_memory() -> dict[str, float] | None:
    """How much recall is live, from the ring the recall path already fills.

    The observation ring is in memory and bounded, which is what makes it
    usable here: the state is assembled on the request path, and a probe that
    opened a database would put a query in front of every generation. It is
    PEEKED, never fetched — building the ring resolves a store path and
    touches disk, and a probe must not be the thing that does that.
    """
    try:
        from core.memory.recall_observations import peek_recall_observations

        ring = peek_recall_observations()
        samples = list(ring.samples()) if ring is not None else []
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("recall observations unavailable: %s", exc)
        samples = []
    out: dict[str, float] = {}
    if samples:
        recent = samples[-64:]
        returned = [1.0 if count else 0.0 for _activation, count in recent]
        activations = [float(activation) for activation, _count in recent]
        out["memory.recall_hits"] = float(np.mean(returned))
        out["memory.recall_confidence"] = max(
            0.0, min(1.0, float(np.mean(activations)))
        )

    facade = _service("memory_facade") or _service("episodic_memory")
    stats = _mapping_from(
        facade, ("get_activation_summary", "get_stats", "stats", "get_status")
    )
    if isinstance(stats, Mapping):
        for name, keys in (
            ("memory.semantic_density", ("semantic_density", "graph_density")),
            ("memory.working_load", ("working_load", "working_set_pressure")),
            ("memory.contradiction", ("contradiction_rate", "contradictions")),
        ):
            value = _first_number(stats, keys)
            if value is not None:
                out[name] = min(1.0, abs(value))
        recency = _first_number(
            stats, ("seconds_since_last_episode", "last_episode_age_s")
        )
        if recency is not None:
            out["memory.episodic_recency"] = 1.0 - _log_age(recency)
    return out or None


def _mapping_from(organ: Any, accessors: Sequence[str]) -> Mapping[str, Any] | None:
    """First accessor that hands back a non-empty mapping."""
    if organ is None:
        return None
    for accessor in accessors:
        candidate = getattr(organ, accessor, None)
        try:
            value = candidate() if callable(candidate) else candidate
        except _LOOKUP_ERRORS as exc:
            logger.debug("organ accessor %s declined: %s", accessor, exc)
            continue
        if isinstance(value, Mapping) and value:
            return value
    return None


def _probe_uncertainty() -> dict[str, float] | None:
    """How sure she is, from whatever organ is registered to say so.

    This channel has NO confirmed in-memory source today, and it is left
    duck-typed rather than bound to a number that means something else. The
    calibration tracker computes its report from a database, which a probe on
    the request path may not do; the self-model's prediction error is about
    the self-model, not about an answer. Until an organ publishes a cheap
    confidence reading, this reads absent — which is the honest state, and is
    what the arbitration checks skip on.
    """
    for key in ("calibration_tracker", "confidence_calibrator", "epistemics", "uncertainty_engine"):
        organ = _service(key)
        if organ is None:
            continue
        report: Any = None
        for accessor in ("current", "get_report", "summary", "get_summary"):
            candidate = getattr(organ, accessor, None)
            try:
                report = candidate() if callable(candidate) else candidate
            except _LOOKUP_ERRORS as exc:
                logger.debug("organ accessor %s declined: %s", accessor, exc)
                report = None
            if isinstance(report, Mapping) and report:
                break
            report = None
        if not isinstance(report, Mapping):
            continue
        out: dict[str, float] = {}
        for name, keys in (
            ("uncertainty.confidence", ("confidence", "mean_confidence")),
            ("uncertainty.calibration_error", ("calibration_error", "ece", "brier")),
            ("uncertainty.evidence_support", ("evidence_support", "support")),
            ("uncertainty.abstention_pressure", ("abstention_rate", "abstain_pressure")),
        ):
            value = _first_number(report, keys)
            if value is not None:
                out[name] = min(1.0, abs(value))
        if out:
            return out
    return None


def _probe_self_state() -> dict[str, float] | None:
    """Her sense of being the same system she was a moment ago.

    From the continuous recurrent self-model, PEEKED rather than fetched: a
    model this probe started would report a continuity score of one for a self
    that has never ticked, which reads as perfect continuity rather than as an
    absence.
    """
    out: dict[str, float] = {}
    try:
        from core.consciousness.crsm import peek_crsm

        model = peek_crsm()
        snapshot = getattr(model, "current_snapshot", None) if model else None
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("self model unavailable: %s", exc)
        snapshot = None
    if snapshot is not None:
        continuity = _first_number(snapshot, ("continuity_score",))
        if continuity is not None:
            out["self.continuity"] = max(0.0, min(1.0, continuity))
        error = _first_number(snapshot, ("prediction_error",))
        if error is not None:
            # Distance from the self-model's own expectation IS the drift.
            out["self.drift"] = max(0.0, min(1.0, abs(error)))

    ghost = _service("ghost") or _service("soul")
    if ghost is not None:
        agency = _first_number(ghost, ("agency", "agency_score"))
        if agency is not None:
            out["self.agency"] = min(1.0, abs(agency))
    health = _service("health_monitor") or _service("watchdog")
    integrity = _first_number(health, ("integrity", "health_score")) if health else None
    if integrity is not None:
        out["self.integrity"] = min(1.0, abs(integrity))
    return out or None


def _probe_attention() -> dict[str, float] | None:
    """Where attention sits, from the ECAN focus the atomspace maintains.

    ``attentional_focus`` ranks the atoms that hold any short-term importance
    at all. Three of the four attention dimensions fall straight out of that
    ranking; novelty does not, and is left absent rather than invented.
    """
    space = _service("atomspace")
    focus: list[Any] = []
    if space is not None and hasattr(space, "attentional_focus"):
        try:
            focus = list(space.attentional_focus(_ATTENTION_FOCUS_SIZE) or [])
        except _LOOKUP_ERRORS as exc:
            logger.debug("attentional focus unavailable: %s", exc)
            focus = []
    if focus:
        weights = []
        for entry in focus:
            try:
                weights.append(float(entry[1]))
            except (IndexError, TypeError, ValueError):
                continue
        if weights:
            total = sum(abs(w) for w in weights) or 1.0
            peak = max(abs(w) for w in weights)
            return {
                "attention.salience_peak": min(1.0, peak / max(1.0, total)),
                "attention.focus": min(1.0, peak / total),
                "attention.load": min(1.0, len(weights) / _ATTENTION_FOCUS_SIZE),
            }
    return _probe_attention_by_accessor()


#: How many focus atoms are read. Bounded because the atomspace ranks every
#: atom holding importance, and this runs on the request path.
_ATTENTION_FOCUS_SIZE = 16


def _probe_attention_by_accessor() -> dict[str, float] | None:
    for key in ("attention_manager", "workspace", "global_workspace"):
        report = _mapping_from(
            _service(key),
            ("attention_summary", "get_attention", "get_focus", "summary"),
        )
        if not isinstance(report, Mapping):
            continue
        out: dict[str, float] = {}
        for name, keys in (
            ("attention.focus", ("focus", "concentration")),
            ("attention.salience_peak", ("peak_salience", "max_sti", "salience")),
            ("attention.novelty", ("novelty",)),
            ("attention.load", ("load", "competition", "n_competing")),
        ):
            value = _first_number(report, keys)
            if value is not None:
                out[name] = min(1.0, abs(value))
        if out:
            return out
    return None


def _probe_recurrence() -> dict[str, float] | None:
    """How much recurrent work this surface is admitted to do.

    The admitted loop count and its ceiling are the same policy the worker
    enforces, computed in-process from configuration rather than read from an
    organ. Convergence and per-turn delta belong to a running turn and are
    left absent unless an organ reports them.
    """
    out: dict[str, float] = {}
    try:
        from core.brain.llm.user_surface_recurrence import (
            admit_user_surface_recurrent_loops,
            user_surface_recurrent_ceiling,
        )

        admitted = float(admit_user_surface_recurrent_loops())
        ceiling = float(user_surface_recurrent_ceiling())
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("recurrent admission unavailable: %s", exc)
    else:
        out["recurrence.depth"] = min(1.0, admitted / 8.0)
        out["recurrence.budget_used"] = min(1.0, admitted / max(1.0, ceiling))

    for key in ("latent_cortex", "cognitive_engine"):
        report = _mapping_from(
            _service(key), ("recurrence_summary", "last_recurrence", "get_recurrence")
        )
        if not isinstance(report, Mapping):
            continue
        for name, keys in (
            ("recurrence.convergence", ("convergence", "cosine", "agreement")),
            ("recurrence.delta", ("delta", "change")),
        ):
            value = _first_number(report, keys)
            if value is not None:
                out[name] = max(-1.0, min(1.0, value))
        break
    return out or None


def _probe_temporal() -> dict[str, float] | None:
    """Temporal orientation, derived from what the other organs are doing.

    This channel has no organ of its own, and inventing one would be worse
    than deriving it: orientation to the past IS memory activity, orientation
    to the future IS an open goal. The derivation is stated here so it is not
    mistaken for a measurement.
    """
    memory = _probe_memory() or {}
    goal = _probe_goal() or {}
    past = float(memory.get("memory.recall_hits", 0.0))
    future = float(goal.get("goal.active", 0.0)) * float(goal.get("goal.priority", 0.5) or 0.5)
    if not memory and not goal:
        return None
    present = max(0.0, 1.0 - 0.5 * (past + future))
    horizon = float(goal.get("goal.age", 0.0))
    return {
        "temporal.past": min(1.0, past),
        "temporal.present": min(1.0, present),
        "temporal.future": min(1.0, future),
        "temporal.horizon": min(1.0, horizon),
    }


PROBES: dict[str, Callable[[], dict[str, float] | None]] = {
    "affect": _probe_affect,
    "substrate": _probe_substrate,
    "goal": _probe_goal,
    "memory": _probe_memory,
    "uncertainty": _probe_uncertainty,
    "self_state": _probe_self_state,
    "attention": _probe_attention,
    "recurrence": _probe_recurrence,
    "temporal": _probe_temporal,
}


#: How long an assembled state may be reused. Every probe is memory-only, but
#: "cheap" is not "free" and the state is read once per generation while
#: generation takes seconds. A quarter second is shorter than any turn and
#: long enough that a burst of calls costs one assembly.
CACHE_TTL_S = 0.25

_CACHE: tuple[float, EndogenousState] | None = None
_CACHE_LOCK = threading.Lock()


def reset_state_cache() -> None:
    """Forget the cached state. For tests, and after an organ is replaced."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


def assemble_state(
    *,
    overrides: Mapping[str, float] | None = None,
    probes: Mapping[str, Callable[[], dict[str, float] | None]] | None = None,
    max_age_s: float = CACHE_TTL_S,
) -> EndogenousState:
    """Read every organ once and build z_Aura.

    ``overrides`` is for experiments and for tests: named dimensions forced to
    a value, marked present, and recorded as interventions so no downstream
    consumer can mistake a constructed state for an observed one.

    A caller supplying its own probes or overrides never reads the cache and
    never writes to it, because a constructed state must not be handed to the
    next caller as an observation.
    """
    global _CACHE
    constructed = probes is not None or bool(overrides)
    if not constructed and max_age_s > 0.0:
        with _CACHE_LOCK:
            cached = _CACHE
        if cached is not None and (time.time() - cached[0]) <= max_age_s:
            return cached[1]

    values = np.zeros(STATE_DIM, dtype=np.float32)
    present = np.zeros(STATE_DIM, dtype=bool)
    sources: dict[str, str] = {}
    active_probes = dict(PROBES if probes is None else probes)

    for channel, probe in active_probes.items():
        try:
            reading = probe()
        except Exception as exc:  # noqa: BLE001 — one dead organ is one dead channel
            # Deliberately broad: a probe calls arbitrary organ code, and an
            # exception type nobody anticipated must cost that channel rather
            # than the whole state. Recorded, not swallowed — a channel that
            # fails on every assembly is a fault someone has to see.
            sources[channel] = "error"
            record_degradation(
                "endogenous_state",
                exc,
                severity="warning",
                action=f"assembled z_Aura without the {channel} channel",
            )
            continue
        if not reading:
            sources[channel] = "absent"
            continue
        wrote = False
        for name, value in reading.items():
            index = FEATURE_INDEX.get(name)
            if index is None:
                continue
            values[index] = FEATURES[index].clamp(value)
            present[index] = True
            wrote = True
        sources[channel] = "live" if wrote else "absent"

    state = EndogenousState(
        values=values,
        present=present,
        sources=sources,
        captured_at=time.time(),
        digest=_digest_of(values),
    )
    if overrides:
        state = state.do(**{k: float(v) for k, v in overrides.items()})
    elif not constructed:
        with _CACHE_LOCK:
            _CACHE = (time.time(), state)
    return state


def empty_state() -> EndogenousState:
    """A state nothing answered for. Every dimension absent, and it says so."""
    zeros = np.zeros(STATE_DIM, dtype=np.float32)
    return EndogenousState(
        values=zeros,
        present=np.zeros(STATE_DIM, dtype=bool),
        sources={c: "absent" for c in CHANNELS},
        captured_at=time.time(),
        digest=_digest_of(zeros),
    )


def describe_layout() -> list[dict[str, Any]]:
    """The declared layout, for receipts and for anyone auditing a fit."""
    return [
        {"index": i, "name": f.name, "channel": f.channel, "meaning": f.meaning,
         "range": [f.low, f.high]}
        for i, f in enumerate(FEATURES)
    ]


__all__ = [
    "CHANNELS",
    "CHANNEL_SLICES",
    "FEATURES",
    "FEATURE_INDEX",
    "LAYOUT_VERSION",
    "STATE_DIM",
    "SUBSTRATE_BANDS",
    "EndogenousState",
    "Feature",
    "Intervention",
    "CACHE_TTL_S",
    "assemble_state",
    "describe_layout",
    "empty_state",
    "layout_digest",
    "pool_substrate",
    "reset_state_cache",
]
