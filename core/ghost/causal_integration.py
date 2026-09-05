"""core/ghost/causal_integration.py — system-level causal integration (Φ_system).

The honest "unity vs federation" instrument.

Bryan's recurring critique of Aura is that it "feels like a federation, not a
unity" — many capable subsystems that each ultimately influence a prompt but do
not genuinely integrate. That is exactly the question Integrated Information
Theory formalises: a system is *integrated* to the degree that its cause-effect
structure is irreducible — you cannot partition it without losing information
about how the whole behaves.

Aura already measures integration on two axes:
  * activation-level φ (``core/consciousness/grassmann_phi.py``) — integration
    *inside a single forward pass* of the resident model;
  * co-presence unity (``core/unity/runtime.py``) — integration of the contents
    bound *within one mind-moment*.

Neither measures integration of the **running organism over time**: do the
subsystems actually cause one another, in feedback, or do they fire as isolated
islands? This module answers that from real data — the live inter-subsystem
consequence stream on ``ConsequenceBus`` (every consequential action publishes
its source subsystem, so the stream *is* the organism's causal activity).

What it computes (all operational, all disclosed — this is a proxy, not a claim
to have solved IIT, whose exact Φ over a real transition matrix is intractable):

  cross_subsystem_influence : fraction of consecutive activations that hand off
                              between *different* subsystems (vs a subsystem
                              talking only to itself) — the opposite of islands.
  feedback_recurrence       : fraction of cross-subsystem edges that sit on a
                              directed cycle (A influences B influences ... A) —
                              Hofstadter's strange loop, at the organ level.
  min_partition_mi          : the IIT-faithful core. Bipartition the active
                              subsystems every way; measure the mutual
                              information between the two halves' co-activation
                              across time; take the MINIMUM over partitions. A
                              high minimum means *no cut is clean* — the whole is
                              irreducible.
  subsystem_diversity       : normalised entropy of which subsystems are active
                              (a mind driven by one organ is not integrated).
  core_participation         : fraction of windows in which a broad set of
                              subsystems co-participate (integration = many parts
                              active together, not one at a time).

These blend into ``phi_system`` ∈ [0,1] with disclosed heuristic weights, and a
plain-language label from ``integrated`` down to ``federated``.

This is deliberately cheap (pure Python over a bounded in-memory ring) and
side-effect-free, so it is safe to read on the response hot path behind a short
TTL cache.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Ghost.CausalIntegration")

# Full bipartition enumeration is 2^n; above this many subsystems we sample
# partitions instead so the measure stays bounded on the hot path.
_FULL_ENUMERATION_CEILING = 9
_SAMPLED_PARTITIONS = 96

# Consecutive consequence events further apart than this do not count as a causal
# hand-off — they bracket an idle gap, not an influence.
_MAX_HANDOFF_GAP_S = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# Information-theoretic primitives (pure, unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _entropy(seq: Sequence[Any]) -> float:
    """Shannon entropy (bits) of the empirical distribution of ``seq``."""
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _mutual_information(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    """Mutual information I(X;Y) in bits over paired samples."""
    if not xs or len(xs) != len(ys):
        return 0.0
    n = len(xs)
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys, strict=False))
    total = 0.0
    for (x, y), nxy in cxy.items():
        pxy = nxy / n
        px = cx[x] / n
        py = cy[y] / n
        total += pxy * math.log2((pxy + 1e-12) / (px * py + 1e-12))
    return max(0.0, total)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemIntegrationReport:
    """A single reading of the organism's over-time causal integration."""

    phi_system: float
    label: str
    events: int
    subsystems: list[str] = field(default_factory=list)
    cross_subsystem_influence: float = 0.0
    feedback_recurrence: float = 0.0
    min_partition_mi: float = 0.0
    subsystem_diversity: float = 0.0
    core_participation: float = 0.0
    window_seconds: float = 0.0
    #: Events dropped for being reports of a measurement rather than records
    #: of an action. Named, not silent: a reader comparing Φ across versions
    #: needs to see that the denominator changed and why.
    measurement_events_excluded: int = 0
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phi_system": round(self.phi_system, 4),
            "label": self.label,
            "events": self.events,
            "measurement_events_excluded": self.measurement_events_excluded,
            "subsystems": list(self.subsystems),
            "cross_subsystem_influence": round(self.cross_subsystem_influence, 4),
            "feedback_recurrence": round(self.feedback_recurrence, 4),
            "min_partition_mi": round(self.min_partition_mi, 4),
            "subsystem_diversity": round(self.subsystem_diversity, 4),
            "core_participation": round(self.core_participation, 4),
            "window_seconds": round(self.window_seconds, 2),
            "computed_at": self.computed_at,
        }

    @property
    def is_integrated(self) -> bool:
        return self.phi_system >= 0.6

    @property
    def is_federated(self) -> bool:
        """True when the live evidence says 'islands, not a mind'."""
        return self.events >= 8 and self.phi_system < 0.2


def _label_for(phi: float, events: int, min_events: int) -> str:
    if events < min_events:
        return "insufficient_history"
    if phi >= 0.6:
        return "integrated"
    if phi >= 0.4:
        return "partial"
    if phi >= 0.2:
        return "loosely_coupled"
    return "federated"


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class SystemIntegration:
    """Computes Φ_system from the live ConsequenceBus stream.

    Pull-based and side-effect-free: it reads a bounded window of recent events
    and derives the graph on demand, memoising the last report for ``ttl``
    seconds so repeated reads on the hot path are free.
    """

    def __init__(
        self,
        bus: Any = None,
        *,
        window: int = 200,
        ttl: float = 2.0,
        min_events: int = 8,
        max_partition_subsystems: int = _FULL_ENUMERATION_CEILING,
        co_activation_window: int = 6,
    ) -> None:
        self._bus = bus
        self._window = int(window)
        self._ttl = float(ttl)
        self._min_events = int(min_events)
        self._max_partition_subsystems = int(max_partition_subsystems)
        self._co_activation_window = int(co_activation_window)
        self._cached: SystemIntegrationReport | None = None
        self._cached_at: float = 0.0

    # ── bus access ────────────────────────────────────────────────────────
    def _get_bus(self) -> Any:
        if self._bus is not None:
            return self._bus
        from core.runtime.consequence_bus import ConsequenceBus
        return ConsequenceBus.get()

    def _recent_events(self) -> list[Any]:
        bus = self._get_bus()
        try:
            return list(bus.recent_events(self._window))
        except (AttributeError, TypeError, ValueError):
            return []

    # ── public API ────────────────────────────────────────────────────────
    def report(self, *, now: float | None = None, force: bool = False) -> SystemIntegrationReport:
        now = time.time() if now is None else now
        if (
            not force
            and self._cached is not None
            and (now - self._cached_at) < self._ttl
        ):
            return self._cached
        raw_events = self._recent_events()
        # CP126 594d43b7: an event published to appear here is not
        # evidence of causal integration. Φ measures how much the organs
        # actually cause one another; counting a measurement's own
        # publication is the instrument reading its own reflection.
        events = [
            event
            for event in raw_events
            if not bool(getattr(event, "measurement_only", False))
        ]
        excluded = len(raw_events) - len(events)
        report = self._compute(events, now=now, measurement_excluded=excluded)
        self._cached = report
        self._cached_at = now
        return report

    # ── computation ───────────────────────────────────────────────────────
    def _compute(
        self,
        events: list[Any],
        *,
        now: float,
        measurement_excluded: int = 0,
    ) -> SystemIntegrationReport:
        if len(events) < self._min_events:
            return SystemIntegrationReport(
                phi_system=0.0,
                label=_label_for(0.0, len(events), self._min_events),
                events=len(events),
                measurement_events_excluded=measurement_excluded,
                computed_at=now,
            )

        # Order oldest→newest (recent_events already yields chronological order,
        # but sort defensively so the measure never depends on caller behaviour).
        ordered = sorted(events, key=lambda e: float(getattr(e, "timestamp", 0.0) or 0.0))
        sources = [str(getattr(e, "source", "") or "unknown") for e in ordered]
        timestamps = [float(getattr(e, "timestamp", 0.0) or 0.0) for e in ordered]

        subsystems = sorted(set(sources))
        window_seconds = max(0.0, timestamps[-1] - timestamps[0])

        cross = self._cross_subsystem_influence(sources, timestamps)
        edges = self._transition_edges(sources, timestamps)
        recurrence = self._feedback_recurrence(edges)
        diversity = self._subsystem_diversity(sources, subsystems)
        min_mi, core = self._partition_and_participation(sources)

        phi = _clamp(
            0.22 * cross
            + 0.22 * recurrence
            + 0.20 * _clamp(min_mi / 2.0)
            + 0.18 * diversity
            + 0.18 * core
        )

        return SystemIntegrationReport(
            phi_system=phi,
            label=_label_for(phi, len(events), self._min_events),
            events=len(events),
            measurement_events_excluded=measurement_excluded,
            subsystems=subsystems,
            cross_subsystem_influence=cross,
            feedback_recurrence=recurrence,
            min_partition_mi=min_mi,
            subsystem_diversity=diversity,
            core_participation=core,
            window_seconds=window_seconds,
            computed_at=now,
        )

    @staticmethod
    def _cross_subsystem_influence(sources: list[str], timestamps: list[float]) -> float:
        """Fraction of consecutive hand-offs that cross subsystem boundaries."""
        handoffs = 0
        cross = 0
        for i in range(1, len(sources)):
            if (timestamps[i] - timestamps[i - 1]) > _MAX_HANDOFF_GAP_S:
                continue
            handoffs += 1
            if sources[i] != sources[i - 1]:
                cross += 1
        return (cross / handoffs) if handoffs else 0.0

    @staticmethod
    def _transition_edges(sources: list[str], timestamps: list[float]) -> dict[tuple[str, str], int]:
        edges: dict[tuple[str, str], int] = defaultdict(int)
        for i in range(1, len(sources)):
            if (timestamps[i] - timestamps[i - 1]) > _MAX_HANDOFF_GAP_S:
                continue
            a, b = sources[i - 1], sources[i]
            if a != b:
                edges[(a, b)] += 1
        return dict(edges)

    @staticmethod
    def _feedback_recurrence(edges: dict[tuple[str, str], int]) -> float:
        """Fraction of cross-subsystem edges that participate in a directed cycle."""
        if not edges:
            return 0.0
        adj: dict[str, set[str]] = defaultdict(set)
        for (a, b) in edges:
            adj[a].add(b)

        def reaches(start: str, target: str) -> bool:
            seen: set[str] = set()
            stack = list(adj.get(start, ()))
            while stack:
                node = stack.pop()
                if node == target:
                    return True
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(adj.get(node, ()))
            return False

        cyclic = sum(1 for (a, b) in edges if reaches(b, a))
        return cyclic / len(edges)

    @staticmethod
    def _subsystem_diversity(sources: list[str], subsystems: list[str]) -> float:
        if len(subsystems) <= 1:
            return 0.0
        return _clamp(_entropy(sources) / math.log2(len(subsystems)))

    def _partition_and_participation(self, sources: list[str]) -> tuple[float, float]:
        """Return (min_partition_mi, core_participation).

        Slides a window over the activation stream, recording which subsystems
        are active in each window, then:
          * min_partition_mi — minimum, over all bipartitions of the (top-K
            most active) subsystems, of the MI between the two halves' window
            activation patterns. High minimum ⇒ irreducible.
          * core_participation — fraction of windows with broad co-activation.
        """
        w = self._co_activation_window
        if len(sources) < w * 2:
            return 0.0, 0.0

        # Windows of subsystem-activity sets.
        active_sets: list[frozenset[str]] = []
        for i in range(0, len(sources) - w + 1):
            active_sets.append(frozenset(sources[i:i + w]))

        # Rank subsystems by activity; bound the partition space to top-K.
        activity = Counter(sources)
        ranked = [s for s, _ in activity.most_common(self._max_partition_subsystems)]
        m = len(ranked)

        # core_participation: windows where at least a broad set is co-active.
        broad_threshold = min(4, max(2, m))
        broad = sum(1 for s in active_sets if len(s & set(ranked)) >= broad_threshold)
        core = broad / len(active_sets)

        if m < 2:
            return 0.0, core

        # Per-window binary activation vector over the ranked subsystems.
        vectors: list[tuple[int, ...]] = [
            tuple(1 if ranked[j] in s else 0 for j in range(m))
            for s in active_sets
        ]

        masks = self._partition_masks(m)
        if not masks:
            return 0.0, core

        min_mi = math.inf
        for mask in masks:
            a_states = [tuple(bit for j, bit in enumerate(v) if mask & (1 << j)) for v in vectors]
            b_states = [tuple(bit for j, bit in enumerate(v) if not (mask & (1 << j))) for v in vectors]
            mi = _mutual_information(a_states, b_states)
            if mi < min_mi:
                min_mi = mi
            if min_mi == 0.0:
                break  # a clean cut exists; cannot get lower
        return (0.0 if min_mi is math.inf else float(min_mi)), core

    @staticmethod
    def _partition_masks(m: int) -> list[int]:
        """Non-trivial bipartitions of ``m`` items, keyed so each is counted once.

        We fix bit 0 to the 'A' side to avoid enumerating a partition and its
        mirror. Full enumeration below the ceiling, random sampling above.
        """
        if m < 2:
            return []
        if m <= _FULL_ENUMERATION_CEILING:
            masks = []
            for mask in range(1, (1 << m) - 1):
                if mask & 1:  # fix item 0 to side A → each partition once
                    masks.append(mask)
            return masks
        rng = random.Random(1729)
        seen: set[int] = set()
        while len(seen) < _SAMPLED_PARTITIONS:
            mask = 1  # item 0 on side A
            for j in range(1, m):
                if rng.random() < 0.5:
                    mask |= (1 << j)
            if mask != (1 << m) - 1:  # not everyone on side A
                seen.add(mask)
        return list(seen)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE: SystemIntegration | None = None


def get_system_integration() -> SystemIntegration:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SystemIntegration()
    return _ENGINE


def reset_system_integration() -> None:
    global _ENGINE
    _ENGINE = None


__all__ = [
    "SystemIntegration",
    "SystemIntegrationReport",
    "get_system_integration",
    "reset_system_integration",
]
