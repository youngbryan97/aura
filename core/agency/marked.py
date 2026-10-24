"""Something new in her world, marked by how strongly she feels it.

Her world reached deliberation through the partner's register, the cadence of
the exchange and how surprising things were (`conjunctive_codes.readings`,
which her choices read through `consequence.world_of`). What her world holds,
the entities and facts she knows to be there, entered nothing deliberation
weighs, and in a campaign they never changed: one entity and one fact on every
turn of seed 7 at `82549c823`. So setting her world moved nothing she weighs,
and ISC-v6's W,A -> D read a direction p of 0.71 at `281bd4518`.

People weigh what a situation offers by how it makes them feel. Damasio's
somatic markers are the bodily feelings that mark an option before reasoning
gets to it, and without them choices about what is in front of you go wrong
(Bechara, Damasio, Tranel and Damasio, Science 275, 1997). A feeling informs a
judgment about whatever it is taken to be about (Schwarz and Clore 1983), so a
feeling with nothing new to be about marks nothing, and something new met
without feeling is noted and left.

So each thing that has arrived in her world since she last looked is marked at
her arousal times how much it stands out. The mark is the urgency of looking
into it, and an intention to do so goes through governance like any other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["SEEN", "marked"]

#: The mark an entity carries once it has been weighed.
SEEN: str = "marked"


def _unit(value: Any, default: float) -> float:
    try:
        number = float(value)
    # not a failure: a reading that is not a number is the default.
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return max(0.0, min(1.0, number))


def marked(state: Any) -> list[tuple[str, float]]:
    """Each thing new to her world, and the urgency her feeling gives it. Marks them seen."""
    world = getattr(state, "world", None)
    entities = getattr(world, "known_entities", None) if world is not None else None
    if not isinstance(entities, Mapping):
        return []
    affect = getattr(state, "affect", None)
    arousal = _unit(getattr(affect, "arousal", 0.0) if affect is not None else 0.0, 0.0)
    out: list[tuple[str, float]] = []
    for name, entity in entities.items():
        if not isinstance(entity, dict) or entity.get(SEEN):
            continue
        entity[SEEN] = True
        out.append((str(name), arousal * _unit(entity.get("salience"), 0.5)))
    return out
