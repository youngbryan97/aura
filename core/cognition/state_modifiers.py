"""Who owns a cognition modifier, and what happens when two writers disagree.

``state.cognition.modifiers`` is a plain dict that any subsystem can assign
into. Eleven modules do. Nothing records who wrote a key, when, or on what
evidence, so two writers can hold contradictory beliefs about the same
name and the last one to run silently wins — and the loser has no way to
find out it lost.

CP126 ``ad5752a2`` found it through the social modeller, which wrote
``social_tension`` straight into kernel cognition from a keyword
heuristic. The write itself is reasonable; doing it anonymously, with no
revision, no ownership and no conflict detection, is not.

So a modifier write goes through :func:`set_modifier`, which:

* stamps the owner, the time and a monotonically increasing revision
  beside the value, in a parallel ``modifier_provenance`` map
* refuses a write to a key another owner set recently, and records the
  conflict rather than resolving it silently
* leaves the modifier value itself in the same place and the same shape,
  so every existing reader keeps working

The provenance map is what makes a wrong modifier debuggable: "who set
social_tension to 0.9" now has an answer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "MODIFIER_PROVENANCE_KEY",
    "ModifierConflict",
    "set_modifier",
    "set_modifiers",
    "modifier_owner",
]

#: Where provenance lives, inside the same dict so it travels with the
#: state and no reader needs a second lookup to find it.
MODIFIER_PROVENANCE_KEY = "__provenance__"

#: How long one owner's claim on a key holds against a different owner. A
#: modifier is a live reading, not a lock, so the claim expires — this is
#: long enough to catch two subsystems fighting over a name in the same
#: conversation and short enough that a retired writer never blocks a new
#: one forever.
CLAIM_TTL_S = 300.0


class ModifierConflict(RuntimeError):
    """Two owners wrote the same modifier inside the claim window."""


def _provenance(modifiers: Any) -> dict[str, dict[str, Any]]:
    record = modifiers.get(MODIFIER_PROVENANCE_KEY)
    if not isinstance(record, dict):
        record = {}
        modifiers[MODIFIER_PROVENANCE_KEY] = record
    return record


def modifier_owner(modifiers: Any, key: str) -> str | None:
    """Who last set ``key``, or None when nothing recorded it."""
    try:
        entry = _provenance(modifiers).get(str(key))
    except (AttributeError, TypeError):
        return None
    if isinstance(entry, dict):
        owner = entry.get("owner")
        return str(owner) if owner else None
    return None


def set_modifier(
    modifiers: Any,
    key: str,
    value: Any,
    *,
    owner: str,
    evidence: str = "",
) -> bool:
    """Set one modifier with its owner recorded. Returns whether it landed.

    A write to a key a DIFFERENT owner set within :data:`CLAIM_TTL_S` is
    refused and recorded. Same owner overwrites its own value freely: a
    modifier is a reading, and a reading is expected to change.
    """
    name = str(key)
    try:
        provenance = _provenance(modifiers)
    except (AttributeError, TypeError) as exc:
        record_degradation(
            "cognition.state_modifiers",
            exc,
            severity="info",
            action=f"did not set modifier {name}; the modifier map was not a mapping",
        )
        return False

    now = time.time()
    previous = provenance.get(name)
    if isinstance(previous, dict):
        prior_owner = str(previous.get("owner") or "")
        prior_at = float(previous.get("at") or 0.0)
        if prior_owner and prior_owner != owner and (now - prior_at) < CLAIM_TTL_S:
            record_degradation(
                "cognition.state_modifiers",
                ModifierConflict(
                    f"{owner} tried to set cognition modifier {name!r}, "
                    f"which {prior_owner} set {now - prior_at:.0f}s ago"
                ),
                severity="warning",
                action=(
                    f"kept {prior_owner}'s value for {name}; two owners disagree "
                    "about one modifier and the write was refused rather than "
                    "resolved silently"
                ),
            )
            return False
        revision = int(previous.get("revision") or 0) + 1
    else:
        revision = 1

    modifiers[name] = value
    provenance[name] = {
        "owner": owner,
        "at": now,
        "revision": revision,
        "evidence": str(evidence)[:200],
    }
    return True


def set_modifiers(
    modifiers: Any,
    values: Mapping[str, Any],
    *,
    owner: str,
    evidence: str = "",
) -> int:
    """Set several modifiers under one owner. Returns how many landed."""
    return sum(
        1
        for key, value in values.items()
        if set_modifier(modifiers, key, value, owner=owner, evidence=evidence)
    )
