"""Something new in her world, marked by how stirred she is when it appears.

Her world reached deliberation through the partner's register, the cadence of
the exchange and how surprising things were (`conjunctive_codes.readings`,
which her choices read through `consequence.world_of`). What her world holds,
the entities and facts she knows to be there, entered nothing deliberation
weighs, and in a campaign they never changed: one entity and one fact on every
turn of seed 7 at `82549c823`. So setting her world moved nothing she weighs,
and ISC-v6's W,A -> D read a direction p of 0.71 at `281bd4518`.

People weigh what a situation offers by how it makes them feel. Damasio's
somatic markers are the bodily states that mark an option before reasoning gets
to it (Bechara, Damasio, Tranel and Damasio, Science 275, 1997), and arousal
present when something new appears is taken to be about the new thing, even
when something else raised it (Dutton and Aron, Journal of Personality and
Social Psychology 30, 1974). A new thing that stirs her opens a gap in what she
knows, and curiosity is the want to close it (Loewenstein, Psychological
Bulletin 116, 1994). A new thing met at her ordinary level is noted and left;
being stirred with nothing new there is only being stirred.

So each thing that has arrived in her world since she last looked is marked
by how far her arousal stands above where it usually sits, in its usual
spread, as the affect phase keeps them, times how much the thing stands out. The
mark is a share of her curiosity that the thing now claims, and the urgency of
an intention to look into it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["SEEN", "how_stirred", "marked", "stirred"]

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


def stirred(arousal: float, centre: float, scatter: float) -> float:
    """How far arousal stands above where it usually sits, as a share in [0, 1).

    Homeostasis's scale for a felt deviation: a rise the size of her usual spread reads a half,
    a much larger one approaches one, and her usual level reads nothing.
    """
    over = max(0.0, float(arousal) - float(centre))
    if over <= 0.0:
        return 0.0
    return over / (over + max(float(scatter), 1e-6))


def how_stirred(state: Any) -> float:
    """How far her arousal stands above where it usually sits, right now; 0 with no usual kept yet.

    Her own arousal's usual level and spread, which the affect phase keeps
    (`AffectUpdatePhase._carry_arousal`).
    """
    affect = getattr(state, "affect", None)
    if affect is None:
        return 0.0
    markers = getattr(affect, "markers", None)
    usual = markers.get("arousal_usual") if isinstance(markers, Mapping) else None
    if not isinstance(usual, Mapping) or not usual.get("n"):
        return 0.0
    return stirred(
        _unit(getattr(affect, "arousal", 0.0), 0.0),
        float(usual.get("level", 0.0) or 0.0),
        float(usual.get("spread", 0.0) or 0.0),
    )


def marked(state: Any, stir: float | None = None) -> list[tuple[str, float]]:
    """Each thing new to her world, and the mark her arousal puts on it. Marks them seen.

    ``stir`` is how stirred she is (`how_stirred`), read from her state when not given.
    """
    world = getattr(state, "world", None)
    entities = getattr(world, "known_entities", None) if world is not None else None
    if not isinstance(entities, Mapping):
        return []
    fresh = [(str(name), entity) for name, entity in entities.items() if isinstance(entity, dict) and not entity.get(SEEN)]
    if not fresh:
        return []
    stir = how_stirred(state) if stir is None else _unit(stir, 0.0)
    out: list[tuple[str, float]] = []
    for name, entity in fresh:
        entity[SEEN] = True
        out.append((name, stir * _unit(entity.get("salience"), 0.5)))
    return out
