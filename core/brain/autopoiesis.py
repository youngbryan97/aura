# core/brain/autopoiesis.py
"""A bounded friction ledger over recurring objectives.

**What this is, said plainly.** It keeps a capped list of objective keys, each
with a weight and an accumulated friction score. Friction rises when an
objective fails to resolve and decays when it succeeds. When one key's
friction crosses a threshold the entry is split so the pressure is recorded
against a narrower key; entries that stay weak and frictionless are dropped;
the list is capped by weight.

That is a small hand-written controller. It is not autopoiesis, and the
previous version's vocabulary — "self-creating topology", "mitosis",
"apoptosis", "spontaneous generation of a new pathway" — described a
biological process the code does not perform. The names below say what the
operations do. The old names remain as aliases because two live call sites
use them, but they are aliases, not the description.

**Three defects came with the overclaim, and they are why this was rewritten
rather than renamed.**

1. *Pruning was unreachable.* The old `_apoptosis` fired on
   `friction < 0.0`, but friction only ever accumulated — `friction *= 0.9;
   friction += dissonance` with a non-negative dissonance can never go
   negative. Both live call sites pass 0.05 and 0.45. The branch could not
   execute, so nothing was ever pruned for being obsolete; the only removal
   path was the capacity cap.

2. *Splitting produced duplicates, not nuance.* Each split appended a node
   literally named `Nuance_of_<concept>`, so a repeatedly-failing objective
   generated twenty identical entries in forty calls — measured, not
   supposed. "Splitting a high-friction node into two nuanced concepts"
   made one concept and nineteen copies of it.

3. *Nothing read any of it.* Two writers in `cognitive_engine`
   (`experience_friction(objective[:20], 0.05)` on an assistant response,
   `0.45` without one) and no reader anywhere. The graph accumulated state
   that could not influence a single output — the half-wired shape this
   codebase keeps finding, where a writer with no reader makes a
   measurement structurally impossible.

The reader is the substantive change. :meth:`friction_for` and
:meth:`pressure_report` make the accumulated signal available, so "this kind
of objective keeps failing" can reach something rather than being recorded
into a void.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger("Aura.ObjectiveFriction")

# Constants for graph health
MAX_NODES = 500          # Prevent unbounded growth
FRICTION_DECAY = 0.9     # Friction retained per observation
MUTATION_THRESHOLD = 0.85

#: Friction at or above this counts as a sustained-pressure objective in
#: :meth:`pressure_report`. Below the split threshold on purpose: the report
#: exists to surface pressure BEFORE the entry is split and reset.
PRESSURE_THRESHOLD = 0.5

#: An entry this weak with no friction left is carrying no information.
#: The old code intended this and could never reach it.
OBSOLETE_WEIGHT = 0.1
OBSOLETE_FRICTION = 0.05

#: How many splits one lineage may accumulate. Without this the split path
#: appends forever under the capacity cap, which is what produced twenty
#: entries with two distinct names.
MAX_REFINEMENT_DEPTH = 3


class SynapticNode:
    """One objective key, its split trigger, and its durable pressure.

    Two numbers, because they answer different questions and collapsing them
    made the reader unobservable. ``friction`` is the short-term trigger and
    is reset by a split — that is what makes splitting terminate. ``pressure``
    is the durable signal a reader wants, and a split does NOT clear it:
    refining how an objective is represented does not mean it stopped
    failing. With one number, `pressure_report` saw zero immediately after
    every split, which is precisely when pressure was highest.
    """

    __slots__ = (
        "id",
        "concept",
        "weight",
        "friction",
        "pressure",
        "depth",
        "observations",
    )

    def __init__(self, concept: str, weight: float = 0.5, depth: int = 0):
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.weight = weight
        self.friction = 0.0  # short-term trigger, reset by a split
        self.pressure = 0.0  # durable signal, survives a split
        self.depth = depth
        self.observations = 0


class ObjectiveFrictionGraph:
    """Tracks which recurring objectives keep failing to resolve."""

    def __init__(self) -> None:
        self.nodes: list[SynapticNode] = []
        self.mutation_threshold = MUTATION_THRESHOLD
        self._by_concept: dict[str, SynapticNode] = {}
        self.splits = 0
        self.pruned = 0

    # -- writing ------------------------------------------------------

    def experience_friction(self, concept: str, dissonance_level: float) -> None:
        """Record one observation against ``concept``.

        A negative ``dissonance_level`` relieves pressure, which is what
        makes the obsolescence path reachable at all.
        """
        concept = str(concept or "").strip()
        if not concept:
            return

        target_node = self._by_concept.get(concept)
        if target_node is None:
            logger.debug("New objective key: %s", concept)
            target_node = SynapticNode(concept, weight=0.1)
            self._add(target_node)
            # Fall through rather than returning. The old code created the
            # entry and discarded the observation that created it, so the
            # first failure of any objective was never counted — and an
            # objective seen twice looked like one seen once.

        target_node.observations += 1
        # Decay first, so both reflect recent history rather than a lifetime
        # total.
        target_node.friction = target_node.friction * FRICTION_DECAY + dissonance_level
        target_node.pressure = max(
            0.0, target_node.pressure * FRICTION_DECAY + dissonance_level
        )

        if target_node.friction >= self.mutation_threshold:
            self._refine(target_node)
        elif (
            target_node.friction <= OBSOLETE_FRICTION
            and target_node.pressure <= OBSOLETE_FRICTION
            and target_node.weight < OBSOLETE_WEIGHT
        ):
            self._retire(target_node)

    def _refine(self, node: SynapticNode) -> None:
        """Split sustained pressure onto a narrower key.

        Bounded by ``MAX_REFINEMENT_DEPTH``: past that, the pressure stays on
        the existing key and only the friction resets. The old version had no
        bound and no uniqueness, so it appended the same name indefinitely.
        """
        node.weight *= 0.5
        node.friction = 0.0
        self.splits += 1

        if node.depth >= MAX_REFINEMENT_DEPTH:
            logger.debug("Refinement depth reached for '%s'; pressure reset only", node.concept)
            return

        refined = f"{node.concept}#{node.depth + 1}"
        if refined in self._by_concept:
            # Already tracking this refinement. Weighting it up is the honest
            # response to repeated pressure; a duplicate entry is not.
            self._by_concept[refined].weight = min(
                1.0, self._by_concept[refined].weight + 0.1
            )
            return
        logger.debug("Sustained friction on '%s'; refining to '%s'", node.concept, refined)
        child = SynapticNode(refined, weight=0.5, depth=node.depth + 1)
        # Inherit the pressure that caused the split. Starting at zero would
        # discard the signal the split was a response to.
        child.pressure = node.pressure
        self._add(child)

    def _retire(self, node: SynapticNode) -> None:
        """Drop a key carrying no weight and no pressure."""
        logger.debug("Retiring inert objective key '%s'", node.concept)
        self._remove(node)
        self.pruned += 1

    # -- reading ------------------------------------------------------
    #
    # The half that did not exist. Two writers and no reader meant none of
    # this could affect anything.

    def friction_for(self, concept: str) -> float:
        """Current unresolved pressure on an objective key, 0.0 if unknown."""
        node = self._by_concept.get(str(concept or "").strip())
        # `pressure`, not `friction`: a split zeroes friction, so reading it
        # here reported nothing at the moment the objective was failing most.
        return float(node.pressure) if node is not None else 0.0

    def is_under_pressure(self, concept: str) -> bool:
        """Whether this objective has been failing enough to be worth noting."""
        return self.friction_for(concept) >= PRESSURE_THRESHOLD

    def pressure_report(self, *, limit: int = 5) -> dict[str, Any]:
        """The objectives currently carrying the most unresolved pressure.

        What a caller actually wants to know: which kinds of request keep
        not resolving. Bounded so it can be attached to telemetry.
        """
        ranked = sorted(self.nodes, key=lambda n: n.pressure, reverse=True)
        under_pressure = [n for n in ranked if n.pressure >= PRESSURE_THRESHOLD]
        return {
            "tracked_objectives": len(self.nodes),
            "under_pressure": len(under_pressure),
            "splits": self.splits,
            "retired": self.pruned,
            "top": [
                {
                    "concept": n.concept,
                    "pressure": round(n.pressure, 4),
                    "friction": round(n.friction, 4),
                    "weight": round(n.weight, 4),
                    "observations": n.observations,
                    "refinement_depth": n.depth,
                }
                for n in ranked[:limit]
                if n.pressure > 0.0
            ],
        }

    # -- bookkeeping --------------------------------------------------

    def _add(self, node: SynapticNode) -> None:
        self.nodes.append(node)
        self._by_concept[node.concept] = node
        self._enforce_capacity()

    def _remove(self, node: SynapticNode) -> None:
        try:
            self.nodes.remove(node)
        # not a failure: a node already gone is the state this is removing it into.
        except ValueError:
            return
        if self._by_concept.get(node.concept) is node:
            del self._by_concept[node.concept]

    def _enforce_capacity(self) -> None:
        """Cap the list, keeping the heaviest keys."""
        if len(self.nodes) <= MAX_NODES:
            return
        self.nodes.sort(key=lambda n: n.weight)
        dropped = len(self.nodes) - MAX_NODES
        for node in self.nodes[:dropped]:
            if self._by_concept.get(node.concept) is node:
                del self._by_concept[node.concept]
        self.nodes = self.nodes[dropped:]
        self.pruned += dropped
        logger.info(
            "Pruned %d low-weight objective keys to stay within %d", dropped, MAX_NODES
        )


#: The previous name. Kept because two live call sites in
#: `core/brain/cognitive_engine.py` construct it, and renaming a class out
#: from under a caller is a separate change from correcting what it claims.
AutopoieticGraph = ObjectiveFrictionGraph
