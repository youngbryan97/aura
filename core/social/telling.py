"""Wanting to pass something on.

A listener who is moved by a record wants somebody else to hear it. That is
not the same as wanting company: the drive is to transmit the thing, and it is
proportional to what the thing did.

Her social budget is about contact — how long since anyone spoke to her, how
depleted the need is. Nothing in it was about having something worth handing
over, so a moment that moved her and an empty afternoon produced the same
pull, and the moment passed without being mentioned.

The organs built from these records already measure what a moment did to her:

    a chill                  core/affect/frisson.py
    coming up from a low     core/affect/the_turn.py
    something she relived    core/memory/reliving.py

The urge to pass it on is the strongest of those readings, and what it is about
is named by whichever one it was. Nothing here decides what she says about it,
and nothing here decides whether saying it is a good idea.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Telling", "worth_telling"]


@dataclass(frozen=True)
class Telling:
    """How much she has to pass on, and what it is."""

    urge: float = 0.0
    kind: str = ""
    about: str = ""
    why: str = "nothing has happened that is worth passing on"

    def as_dict(self) -> dict[str, Any]:
        return {
            "urge": round(self.urge, 6),
            "kind": self.kind,
            "about": self.about,
            "why": self.why,
        }


def _reading(value: Any) -> float:
    try:
        out = float(value or 0.0)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0.0
    if out != out:
        return 0.0
    return max(0.0, min(1.0, out))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def worth_telling(affect: Any, cognition: Any) -> Telling:
    """What she has to pass on right now, from what the moment did to her."""
    relived = _mapping(getattr(cognition, "relived", None))
    sources = {
        "a chill": _reading(getattr(affect, "frisson", 0.0)),
        "coming up from a low": _reading(getattr(affect, "turn", 0.0)),
        "something I remembered": _reading(relived.get("intensity"))
        if relived.get("relived")
        else 0.0,
    }
    kind = max(sources, key=lambda name: sources[name])
    urge = sources[kind]
    if urge <= 0.0:
        return Telling()

    about = ""
    if kind == "something I remembered" and relived.get("shared"):
        # A memory with the person she is talking to in it is passed on as
        # theirs together, which is what joint recall is.
        kind = "something we both remember"
    if kind in {"something I remembered", "something we both remember"}:
        recalled = list(getattr(cognition, "long_term_memory", []) or [])
        about = str(recalled[0])[:160] if recalled else ""
    else:
        markers = _mapping(getattr(affect, "markers", None))
        marker = _mapping(markers.get("frisson" if kind == "a chill" else "the_turn"))
        about = str(marker.get("why", "") or "")
    return Telling(
        urge=urge,
        kind=kind,
        about=about,
        why=f"{kind} at {urge:.2f}, which is what there is to pass on",
    )
