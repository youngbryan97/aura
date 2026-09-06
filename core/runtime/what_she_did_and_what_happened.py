"""One action history, read off the event log rather than kept beside it.

AutoGPT keeps one action-history component; the closure asked for a canonical
ActionEpisode projection sourced from the event spine, with adapters rather
than a second store.

Aura has several places that remember what was done — the record of her own
work, the delivery journal, the influence ledger, the degradation register —
and each was written for its own question. None of them can answer "what did
she do, in order, and what came of it", because each holds a different slice
and none holds the join.

A projection rather than a store, and the distinction is the whole design. The
spine is append-only and never edited; this reads it. If the projection is
wrong it can be thrown away and rebuilt, which is exactly what cannot be said
of a fifth store — and a fifth store is what "one action history" would become
if it were written down anywhere else.

An action that was started and never finished is kept as unfinished rather
than dropped. A history that only contains completed actions cannot answer the
question anybody actually has after a crash.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatSheDidAndWhatHappened")

__all__ = [
    "AnActionEpisode",
    "how_the_history_stands",
    "the_action_history",
    "what_never_finished",
]

#: Event kinds that begin an action, and the ones that end one. Declared
#: rather than pattern-matched: an event kind that merely contains the word
#: "action" is not the start of one, and a projection built on that guess
#: reports things that never happened.
THE_STARTS: frozenset[str] = frozenset(
    {"action_started", "tool_called", "skill_invoked", "turn_started"}
)
THE_ENDS: frozenset[str] = frozenset(
    {"action_finished", "tool_returned", "skill_returned", "turn_recorded"}
)


@dataclass
class AnActionEpisode:
    """One thing she did, and what came of it."""

    what: str
    began_at_event: int
    began_at: float
    by: str = ""
    ended_at_event: int = 0
    ended_at: float = 0.0
    outcome: str = ""
    #: Everything the start and end events carried, joined.
    said: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.ended_at_event > 0

    @property
    def took_s(self) -> float:
        return max(0.0, self.ended_at - self.began_at) if self.finished else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "by": self.by,
            "began_at_event": self.began_at_event,
            "ended_at_event": self.ended_at_event,
            "finished": self.finished,
            "took_s": round(self.took_s, 4),
            "outcome": self.outcome,
            "said": dict(self.said),
        }


def _key(event: Any) -> str:
    """What ties a start to its end.

    The payload's own id where it has one, and the actor otherwise. Two
    concurrent actions by one actor with no id cannot be told apart, and that
    is a fact about the events rather than something to guess around.
    """
    payload = getattr(event, "payload", None) or {}
    for name in ("action_id", "call_id", "id", "turn_id", "receipt_id"):
        found = payload.get(name)
        if found:
            return f"{name}:{found}"
    return f"actor:{getattr(event, 'actor', '') or 'nobody'}"


def the_action_history(
    log: Any = None, *, since: int = 0, most: int = 500
) -> list[AnActionEpisode]:
    """Every action in the log, in order, joined to how it ended.

    Reads the spine. Nothing is written, so a projection that turns out wrong
    is thrown away rather than migrated.
    """
    if log is None:
        try:
            from core.runtime.event_spine import get_spine

            log = get_spine().log
        except Exception as exc:  # noqa: BLE001 — no spine is an empty history
            logger.debug("no spine to read a history from: %s", exc)
            return []

    open_ones: dict[str, AnActionEpisode] = {}
    done: list[AnActionEpisode] = []
    for event in log.events(since=since):
        kind = str(getattr(event, "kind", ""))
        if kind in THE_STARTS:
            episode = AnActionEpisode(
                what=str((getattr(event, "payload", None) or {}).get("what") or kind),
                began_at_event=int(getattr(event, "seq", 0)),
                began_at=float(getattr(event, "at", 0.0)),
                by=str(getattr(event, "actor", "")),
                said=dict(getattr(event, "payload", None) or {}),
            )
            open_ones[_key(event)] = episode
            done.append(episode)
            continue
        if kind in THE_ENDS:
            episode = open_ones.pop(_key(event), None)
            if episode is None:
                # An end with no start. Kept as its own episode rather than
                # dropped: an action whose beginning is off the front of the
                # log still happened.
                episode = AnActionEpisode(
                    what=str(
                        (getattr(event, "payload", None) or {}).get("what") or kind
                    ),
                    began_at_event=0,
                    began_at=0.0,
                    by=str(getattr(event, "actor", "")),
                )
                done.append(episode)
            payload = dict(getattr(event, "payload", None) or {})
            episode.ended_at_event = int(getattr(event, "seq", 0))
            episode.ended_at = float(getattr(event, "at", 0.0))
            episode.outcome = str(
                payload.get("outcome") or payload.get("status") or "ended"
            )
            episode.said.update(payload)
    return done[-int(most):] if most else done


def what_never_finished(log: Any = None) -> list[dict[str, Any]]:
    """Actions that began and have no end.

    The question anybody actually has after a crash, and the one a history of
    completed actions cannot answer.
    """
    return [
        one.to_dict() for one in the_action_history(log) if not one.finished
    ]


def how_the_history_stands(log: Any = None) -> dict[str, Any]:
    episodes = the_action_history(log)
    unfinished = [one for one in episodes if not one.finished]
    return {
        "episodes": len(episodes),
        "unfinished": len(unfinished),
        "what_never_finished": [one.what for one in unfinished][:20],
        "starts": sorted(THE_STARTS),
        "ends": sorted(THE_ENDS),
        "what_this_is": (
            "a projection off the append-only log, not a fifth store; if it "
            "is wrong it is thrown away and read again"
        ),
    }
