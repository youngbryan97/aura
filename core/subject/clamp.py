"""Holding a domain still, so nothing downstream can depend on its changes.

A lesion in this substrate is not a severed wire. The ten domains are views on
one shared state object, and the pathway from affect to planning is whatever
phase reads one and writes the other, spread across a pipeline. Cutting that
by hand would mean editing the phases, which changes the organism rather than
testing it.

Clamping does the same work from the other side. The clamped domain's fields,
and the organs its readers read, are captured once and written back after
every phase and before every reading, so the domain still exists, is still
read, and never varies. Nothing downstream can carry
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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from core.subject.snapshot import _organ_state, _restore_organ
from core.subject.state import _SCHEMAS, DOMAINS, domain_slices, feature_names

__all__ = [
    "CLAMPED_FIELDS",
    "HELD_TOLERANCE",
    "RESERVOIR_FIELDS",
    "Clamp",
    "clamped",
    "compose",
    "organs_read_by",
    "still",
]

#: How far a held column may move and still count as held: floats accumulated
#: in a different order when a copy is written back. The fork test's bound, far
#: below the 0.3 standardised effect an edge has to clear.
HELD_TOLERANCE: float = 1e-6

#: What holding a domain still means, field by field. These are the same
#: attributes the readers read and the writers write, so a clamped domain
#: cannot move and cannot be moved.
CLAMPED_FIELDS: dict[str, tuple[str, ...]] = {
    "P": ("world.recent_percepts", "world.spatial_context", "cognition.current_objective"),
    "I": (
        "soma.hardware",
        "soma.latency",
        "soma.expressive",
        "soma.sensors",
        "vitality",
        # Her own exertion and what she spent it on. The effort ledger is the
        # only channel the body has while the host is held still, so a clamp
        # that did not hold it left interoception moving inside its own lesion.
        "soma.exertion",
        "soma.effort",
    ),
    "A": (
        "affect.valence",
        "affect.arousal",
        "affect.curiosity",
        "affect.engagement",
        "affect.social_hunger",
        "affect.emotions",
        "affect.dominant_emotion",
        "free_energy",
        "affect.momentum",
        # What she has come to expect of each feeling, which is what an arriving
        # one is priced against, and the three physiological channels A reads.
        "affect.mood_baselines",
        "affect.physiology.heart_rate",
        "affect.physiology.cortisol",
        "affect.physiology.adrenaline",
        # The organs added since: what she is caught between and how it stands,
        # whether the moment came out as expected, and the level she is
        # speaking from. A reads every one, so a clamp on A that left them free
        # let the lesion's held side move by a third of a unit.
        "affect.ambivalence",
        "affect.ambivalent_about",
        "affect.ambivalence_standing",
        "affect.markers.ambivalence",
        "affect.confirmation",
        "affect.delivery_z",
        "affect.breakthrough",
        "affect.steadiness",
        "affect.lift",
        "affect.frisson",
        "affect.turn",
        "affect.safety",
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
        # The six modifiers the workspace hands the rest of the cycle. They are
        # what attention does to everything downstream, so a clamp on the
        # workspace that left them free was not holding the workspace.
        "cognition.modifiers.temperature_mod",
        "cognition.modifiers.depth_mod",
        "cognition.modifiers.focus_mod",
        "cognition.modifiers.creativity_mod",
        "cognition.modifiers.urgency_flag",
        "cognition.modifiers.overall_vitality",
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
        # How somebody else read her, which changes how much she trusts her own
        # reading. See core/self/recognition.py.
        "identity.read_by_other",
    ),
    "M": (
        "cognition.working_memory",
        "cognition.long_term_memory",
        "cognition.rolling_summary",
        "cognition.continuity_ledger",
        "cognition.active_thread_id",
        "cold.long_term_memory",
        "cold.evolution_log",
        # How strongly each recollection is in mind. Recall writes a match
        # score beside every one and the workspace prices its memory bid from
        # it, so a clamp that held the text and not the scores held what she
        # remembered and not how much it counted — and the memory displacement
        # writes exactly this.
        "cognition.memory_scores",
        # What recall brought back and how often it has been asked for.
        # See core/memory/reliving.py.
        "cognition.relived",
    ),
    "W": (
        "world.known_entities",
        "world.relationship_graph",
        "world.facts",
        "world.user_preferences",
        "cold.concept_graph",
        "cognition.user_emotional_trend",
        # The shape of what the person just said. A fact about them that W
        # reads, written by the conversation phase on every message.
        "world.partner_register",
        # The pulse they are keeping, which sizes her own next turn.
        # See core/expression/entrainment.py.
        "cognition.partner_cadence",
        # A we both of them are saying. See core/social/togetherness.py.
        "cognition.togetherness",
    ),
    "D": (
        "cognition.active_goals",
        "cognition.pending_initiatives",
        "cognition.current_origin",
        "cognition.last_action_source",
        "motivation.budgets",
        # The stance she is taking toward the person, which decides whether she
        # searches for something to do at all. See core/social/witness.py.
        "cognition.witness",
        # And how much of what she wants to say she has already said, and
        # whether she has something worth handing over at all.
        "cognition.catharsis",
        "cognition.telling",
        "cognition.borrowed_resolve",
        "cognition.scale",
        "cognition.persona_gap",
        "cognition.constancy",
    ),
    # The reservoir is clamped through the runtime rather than the state; see
    # `RESERVOIR_FIELDS` and `Clamp.capture`.
    "N": (),
}


#: What holding development still means. The hidden units are the reservoir,
#: and the four beside them are what N's own columns read: how far it has come,
#: which era it is in, and what the last step sensed. Holding only the units
#: left four of the thirteen columns moving inside their own lesion.
RESERVOIR_FIELDS: tuple[str, ...] = (
    "steps",
    "era",
    "last_novelty",
    "last_displacement",
    "last_relative_displacement",
)


def _step(node: Any, name: str) -> Any:
    """One step down a path, through an attribute or a key.

    Attributes only, this was. `cognition.modifiers` and `affect.physiology`
    are dicts, so every field held under them resolved to no owner and was
    never captured: a held workspace's modifiers and a held heart rate moved
    inside their own lesion.
    """
    if isinstance(node, Mapping):
        return node.get(name)
    return getattr(node, name, None)


def _get(root: Any, path: str) -> tuple[Any, str, Any]:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        node = _step(node, part)
        if node is None:
            return None, parts[-1], None
    name = parts[-1]
    return node, name, _step(node, name)


def organs_read_by(domains: Sequence[str]) -> tuple[str, ...]:
    """The organs whose own state the readers of these domains read."""
    names = {
        source[len("organ:"):].split(".", 1)[0]
        for domain in domains
        for source in _SCHEMAS[domain].sources
        if source.startswith("organ:")
    }
    return tuple(sorted(names))


class Clamp:
    """Captured values for a set of domains, and the way to put them back."""

    def __init__(self, state_holder: Any, domains: Sequence[str]) -> None:
        self.holder = state_holder
        self.domains = tuple(key for key in DOMAINS if key in set(domains))
        self.values: dict[str, Any] = {}
        self.hidden: Any = None
        self.reservoir: dict[str, Any] = {}
        self.organs: dict[str, tuple[Any, dict[str, Any]]] = {}
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
        self.reservoir.clear()
        if "N" in self.domains:
            import numpy as np

            ontogeny = self.holder.ontogeny
            self.hidden = np.array(ontogeny.h, copy=True)
            for field in RESERVOIR_FIELDS:
                if hasattr(ontogeny, field):
                    self.reservoir[field] = copy.deepcopy(getattr(ontogeny, field))
        # The organs these domains are read from. A held workspace whose own
        # object kept choosing winners was not held: a probe holding P, I, A, G
        # and C saw the winner columns move by more than a whole unit, because
        # they are read off the workspace and not off the state. Each organ
        # feeds one domain, so holding it holds nothing on the other side.
        self.organs.clear()
        kit = getattr(self.holder, "organs", None)
        for name in organs_read_by(self.domains):
            organ = getattr(kit, name, None)
            if organ is not None:
                self.organs[name] = (organ, _organ_state(organ))

    def apply(self) -> None:
        state = self.holder.state
        for path, value in self.values.items():
            owner, name, _ = _get(state, path)
            if owner is None:
                continue
            try:
                if isinstance(owner, Mapping):
                    owner[name] = copy.deepcopy(value)
                else:
                    setattr(owner, name, copy.deepcopy(value))
            except (AttributeError, TypeError):
                continue
        if self.hidden is not None:
            import numpy as np

            ontogeny = self.holder.ontogeny
            ontogeny.h = np.array(self.hidden, copy=True)
            for field, value in self.reservoir.items():
                try:
                    setattr(ontogeny, field, copy.deepcopy(value))
                except (AttributeError, TypeError):
                    continue
        for organ, saved in self.organs.values():
            _restore_organ(organ, saved)


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


def still(rows: Sequence[Any], domains: Sequence[str], *, tolerance: float = HELD_TOLERANCE) -> dict[str, float]:
    """Every column of these domains that moved across these readings, and how far.

    Empty is the bar for a held side. A column that moves while its domain is
    held is a channel the lesion did not sever.
    """
    import numpy as np

    if not rows:
        return {}
    matrix = np.vstack([row.vector() for row in rows])
    slices = domain_slices()
    moved: dict[str, float] = {}
    for domain in domains:
        block = matrix[:, slices[domain]]
        change = np.max(np.abs(block - block[0]), axis=0)
        for name, amount in zip(feature_names(domain), change, strict=True):
            if amount > tolerance:
                moved[name] = round(float(amount), 6)
    return moved

