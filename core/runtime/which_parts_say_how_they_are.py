"""Which registered parts can say what state they are in, and which cannot.

:mod:`core.runtime.what_a_part_of_her_declares` gives a part five things: its
state, why it is in that state, who owns it, what it needs, and — added
alongside this — when it entered that state. What nothing checked is how many
of the runtime's actual parts declare any of it.

The answer matters because of what a health report can say without it. A
service that is registered and constructed reads as present, and present is
not a state. "Registered" cannot distinguish a service that started cleanly
from one that refused to start and was registered anyway, and it cannot say
when either happened. So a runtime with a dead organ reports the same shape as
a runtime with a live one, and the only difference is in a log.

This asks every canonical service name in :mod:`core.service_names` whether
the thing behind it declares a lifecycle, and names the ones that do not.
Measured rather than asserted: the count comes from asking the objects, so it
moves when the runtime does.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.WhichPartsSayHowTheyAre")

__all__ = [
    "AStatus",
    "THE_FIVE_THINGS",
    "what_a_part_says",
    "how_the_parts_answer",
    "parts_that_say_nothing",
]

#: What a part has to be able to say. Four of these existed; `since` is the
#: one a report needs to tell a part that has just started from one that has
#: been wedged since boot.
THE_FIVE_THINGS: tuple[str, ...] = (
    "state",
    "reason",
    "since",
    "owner",
    "dependencies",
)


@dataclass(frozen=True, slots=True)
class AStatus:
    """One part's answer, and which of the five it could not give."""

    name: str
    state: str = ""
    reason: str = ""
    since: float = 0.0
    owner: str = ""
    dependencies: tuple[str, ...] = ()
    #: Empty where the object was there but said nothing at all.
    present: bool = False

    @property
    def silent_about(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.state:
            missing.append("state")
        # A reason is only owed where the state is one that needs explaining.
        if self.state in ("refused", "stopped", "stopping") and not self.reason:
            missing.append("reason")
        if self.since <= 0.0:
            missing.append("since")
        if not self.owner or self.owner == "unclassified":
            missing.append("owner")
        if not self.dependencies:
            missing.append("dependencies")
        return tuple(missing)

    @property
    def says_everything(self) -> bool:
        return self.present and not self.silent_about

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "state": self.state,
            "reason": self.reason,
            "since": self.since,
            "owner": self.owner,
            "dependencies": list(self.dependencies),
            "silent_about": list(self.silent_about),
        }


def _first(obj: Any, *names: str) -> Any:
    """The first of these attributes the object actually has."""
    for name in names:
        # A property that raises is still an answer of a kind, and asking must
        # not take the report down with it.
        try:
            found = getattr(obj, name, None)
        except Exception:  # noqa: BLE001
            continue
        if found is None:
            continue
        if callable(found):
            try:
                found = found()
            except Exception:  # noqa: BLE001 - asking must never break the report
                continue
        if found not in (None, ""):
            return found
    return None


def what_a_part_says(name: str, part: Any) -> AStatus:
    """Ask one object for the five things, without requiring it to know them."""
    if part is None:
        return AStatus(name=name, present=False)
    state = _first(part, "alive", "lifecycle_state", "state", "status")
    reason = _first(part, "why_refused", "lifecycle_reason", "reason")
    since = _first(part, "alive_since", "lifecycle_since", "started_at", "since")
    owner = _first(part, "authority", "owner", "lifecycle_owner")
    needs = _first(part, "needs", "dependencies", "requires")
    if isinstance(needs, str):
        needs = (needs,)
    return AStatus(
        name=name,
        present=True,
        state=str(state or ""),
        reason=str(reason or ""),
        since=float(since) if isinstance(since, (int, float)) else 0.0,
        owner=str(owner or ""),
        dependencies=tuple(str(one) for one in (needs or ())),
    )


def _the_service_names() -> tuple[str, ...]:
    try:
        from core.service_names import ServiceNames
    except ImportError:
        return ()
    return tuple(
        sorted(
            str(value)
            for key, value in vars(ServiceNames).items()
            if not key.startswith("_") and isinstance(value, str)
        )
    )


def how_the_parts_answer(parts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask every registered service, and report what the runtime can say.

    Never raises and never starts anything: a service that is not registered
    is reported as absent rather than constructed to ask it, because
    constructing it to find out changes the thing being measured. In a bare
    process nothing is registered and ``registered`` is zero — which is the
    truth about that process, not about Aura, and is why the report separates
    what was asked from what answered.

    ``parts`` supplies the objects directly, for a caller that has them.
    """
    if parts is None:
        try:
            from core.container import ServiceContainer
        except ImportError:
            return {"asked": 0, "note": "no service container in this process"}

        def look(name: str) -> Any:
            try:
                return ServiceContainer.get(name, default=None)
            except Exception:  # noqa: BLE001 - a lookup must not break the report
                return None
    else:
        def look(name: str) -> Any:
            return parts.get(name)

    names = _the_service_names() if parts is None else tuple(sorted(parts))
    statuses = [what_a_part_says(name, look(name)) for name in names]

    registered = [s for s in statuses if s.present]
    complete = [s for s in registered if s.says_everything]
    return {
        "asked": len(statuses),
        "registered": len(registered),
        "say_all_five": len(complete),
        "say_nothing": [s.name for s in registered if len(s.silent_about) == 5],
        "silent_about": {
            thing: sum(1 for s in registered if thing in s.silent_about)
            for thing in THE_FIVE_THINGS
        },
        "the_five_things": list(THE_FIVE_THINGS),
        "each": [s.to_dict() for s in registered],
    }


def parts_that_say_nothing() -> tuple[str, ...]:
    """Registered services that declare none of the five. The number to move."""
    answer = how_the_parts_answer()
    return tuple(answer.get("say_nothing", ()))


def now() -> float:
    """The stamp a part records when it enters a state."""
    return time.time()
