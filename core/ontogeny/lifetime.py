"""Advancing the lifetime state on every cognitive cycle, not only on retrievals.

The reservoir in `core/ontogeny/state.py` is meant to be the state that carries
her whole life forward. Until now it only stepped inside
`OntogenyCore.consider`, which the runtime calls at one place: choosing how wide
to cast a memory retrieval. A state that advances only when memory is queried
is a retrieval-history state wearing the name of a developmental one, and a
day of conversation with no retrieval left it exactly where it started.

So this advances it once per cognitive cycle on the current core features, and
publishes what it senses — how unlike her ordinary life this moment is, and how
far the step moved her. Those two numbers are the only thing a developmental
state can honestly offer a moment, and they were being computed and thrown
away.

Advancing a state is not the same as letting it decide. Nothing here changes a
choice. What reads the reading is a separate question, answered at the reading
site and gated the way every other ontogenetic authority is gated.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

from core.ontogeny.features import FeatureSchema
from core.runtime.errors import record_degradation

__all__ = ["LIFETIME", "LIFETIME_SCHEMA", "advance", "last_reading", "reset_for_test", "state"]

logger = logging.getLogger("Aura.Ontogeny.Lifetime")

LIFETIME = "lifetime_state"

#: What the lifetime state is advanced on. Deliberately coarse and stable: one
#: number per cognitive domain rather than every field of each, because the
#: reservoir's width is fixed for the life of the state and a schema that grows
#: with the codebase would reincarnate her every time a field was added.
LIFETIME_SCHEMA = FeatureSchema(
    control_point=LIFETIME,
    version=1,
    names=(
        "perception",
        "interoception",
        "affect_valence",
        "affect_arousal",
        "workspace",
        "cognition",
        "self_state",
        "memory",
        "world",
        "deliberation",
    ),
    sources={
        "perception": "AuraState.world.recent_percepts",
        "interoception": "AuraState.soma.hardware",
        "affect_valence": "AuraState.affect.valence",
        "affect_arousal": "AuraState.affect.arousal",
        "workspace": "AuraState.cognition.conversation_energy",
        "cognition": "AuraState.phi",
        "self_state": "AuraState.identity.stability",
        "memory": "AuraState.cognition.working_memory",
        "world": "AuraState.world.facts",
        "deliberation": "AuraState.cognition.active_goals",
    },
)

_lock = threading.Lock()
_last: Any = None


def advance(features: Mapping[str, float]) -> Any:
    """Step her lifetime state on this moment. Returns the reading, or None.

    Total: a reservoir that will not step is a degradation, never an exception
    into a cognitive phase.
    """
    global _last
    try:
        from core.ontogeny.service import get_ontogeny

        reading = get_ontogeny().advance(LIFETIME_SCHEMA, features)
    except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
        record_degradation(
            "ontogeny_lifetime",
            exc,
            severity="warning",
            action="the lifetime state did not advance on this cycle",
        )
        return None
    with _lock:
        _last = reading
    return reading


def state() -> Any:
    """The reservoir itself, for anything that has to read or restore it.

    One object, shared with every control point, because it is her state and
    not a per-subsystem scratchpad. A caller that forks a run has to carry it
    across the fork or the two arms will share a lifetime.
    """
    try:
        from core.ontogeny.service import get_ontogeny

        return get_ontogeny()._state_for(LIFETIME_SCHEMA)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "ontogeny_lifetime", exc, severity="debug", action="lifetime state unavailable"
        )
        return None


def last_reading() -> Any:
    """The most recent step, or None if the state has not advanced yet."""
    with _lock:
        return _last


def reset_for_test() -> None:
    global _last
    with _lock:
        _last = None
