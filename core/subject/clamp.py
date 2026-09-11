"""Holding a domain still, so nothing downstream can depend on its changes.

A lesion in this substrate is not a severed wire. The ten domains are views on
one shared state object, and the pathway from affect to planning is whatever
phase reads one and writes the other, spread across a pipeline. Cutting that
by hand would mean editing the phases, which changes the organism rather than
testing it.

Clamping does the same work from the other side. The clamped domain's fields
are captured once and written back after every phase, so the domain still
exists, is still read, and never varies. Nothing downstream can carry
information about it, because there is none to carry. That severs every
outgoing edge of a node at once, which is a node lesion rather than an edge
lesion, and the difference matters when reading the result: a deficit shows
that something depended on that domain changing, not which pathway carried it.

The clamp is exact and reversible. Restoring is dropping the context, and the
rescue arm is the same run with the clamp released, which is what makes a
recovered measurement evidence rather than a second baseline.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from core.subject.state import DOMAINS

__all__ = ["CLAMPED_FIELDS", "Clamp", "clamped", "compose"]

#: What holding a domain still means, field by field. These are the same
#: attributes the readers read and the writers write, so a clamped domain
#: cannot move and cannot be moved.
CLAMPED_FIELDS: dict[str, tuple[str, ...]] = {
    "P": ("world.recent_percepts", "world.spatial_context"),
    "I": ("soma.hardware", "soma.latency", "soma.expressive", "soma.sensors", "vitality"),
    "A": (
        "affect.valence",
        "affect.arousal",
        "affect.curiosity",
        "affect.engagement",
        "affect.social_hunger",
        "affect.emotions",
        "affect.dominant_emotion",
        "free_energy",
    ),
    "G": (
        "cognition.attention_focus",
        "cognition.coherence_score",
        "cognition.fragmentation_score",
        "cognition.contradiction_count",
        "cognition.conversation_energy",
        "cognition.discourse_depth",
        "cognition.discourse_branches",
        "cognition.selfhood_reading",
        "phi",
    ),
    "C": (
        "cognition.current_mode",
        "cognition.phenomenal_state",
        "phi_estimate",
        "loop_cycle",
    ),
    "S": (
        "identity.stability",
        "identity.evolution_score",
        "identity.bonding_level",
        "identity.narrative_version",
        "identity.current_narrative",
        "identity.core_values",
        "identity.self_preferences",
        "identity.personality_growth",
    ),
    "M": (
        "cognition.working_memory",
        "cognition.long_term_memory",
        "cognition.rolling_summary",
        "cognition.continuity_ledger",
        "cognition.active_thread_id",
        "cold.long_term_memory",
        "cold.evolution_log",
    ),
    "W": (
        "world.known_entities",
        "world.relationship_graph",
        "world.facts",
        "world.user_preferences",
        "cold.concept_graph",
        "cognition.user_emotional_trend",
    ),
    "D": (
        "cognition.active_goals",
        "cognition.pending_initiatives",
        "cognition.current_origin",
        "cognition.last_action_source",
        "motivation.budgets",
    ),
    "N": (),  # the reservoir is clamped through the runtime, not the state
}


def _get(root: Any, path: str) -> tuple[Any, str, Any]:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        node = getattr(node, part, None)
        if node is None:
            return None, parts[-1], None
    name = parts[-1]
    return node, name, getattr(node, name, None)


class Clamp:
    """Captured values for a set of domains, and the way to put them back."""

    def __init__(self, state_holder: Any, domains: Sequence[str]) -> None:
        self.holder = state_holder
        self.domains = tuple(key for key in DOMAINS if key in set(domains))
        self.values: dict[str, Any] = {}
        self.hidden: Any = None
        self.capture()

    def capture(self) -> None:
        state = self.holder.state
        self.values.clear()
        for domain in self.domains:
            for path in CLAMPED_FIELDS.get(domain, ()):
                owner, name, current = _get(state, path)
                if owner is None:
                    continue
                self.values[path] = copy.deepcopy(current)
        if "N" in self.domains:
            import numpy as np

            self.hidden = np.array(self.holder.ontogeny.h, copy=True)

    def apply(self) -> None:
        state = self.holder.state
        for path, value in self.values.items():
            owner, name, _ = _get(state, path)
            if owner is None:
                continue
            try:
                setattr(owner, name, copy.deepcopy(value))
            except (AttributeError, TypeError):
                continue
        if self.hidden is not None:
            import numpy as np

            self.holder.ontogeny.h = np.array(self.hidden, copy=True)


@contextmanager
def clamped(runtime: Any, domains: Sequence[str]) -> Iterator[Clamp]:
    """Hold these domains still for the duration, then let them go.

    The runtime is asked to call `apply` after every phase; that hook is what
    makes the clamp bite inside a turn rather than only between turns.
    """
    clamp = Clamp(runtime, domains)
    previous = getattr(runtime, "after_phase", None)
    runtime.after_phase = clamp.apply
    try:
        yield clamp
    finally:
        runtime.after_phase = previous


def compose(left_rows: Sequence[Any], right_rows: Sequence[Any], left: Sequence[str]) -> list[Any]:
    """One recording of the cut system, from the two halves that ran apart.

    `left_rows` is the run in which the right-hand side was held still, so its
    left-hand columns are the left side evolving with no information from the
    right; `right_rows` is the mirror of that. Taking each side's columns from
    the run where it was free gives the system with `E(A*, B*)` and `E(B*, A*)`
    removed and both sides' internal dynamics intact — which is the lesion the
    equation states, rather than the node clamp that reads as one.
    """
    import dataclasses

    side = set(left)
    out: list[Any] = []
    for a, b in zip(left_rows, right_rows, strict=False):
        values = {
            key: (a.values[key] if key in side else b.values[key]) for key in DOMAINS
        }
        misses = dict(getattr(a, "misses", {}) or {})
        misses.update(getattr(b, "misses", {}) or {})
        out.append(dataclasses.replace(a, values=values, misses=misses))
    return out
