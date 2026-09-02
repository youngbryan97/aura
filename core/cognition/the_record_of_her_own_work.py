"""What she did, what it cost, and what it was worth — kept, because a decision needs it.

The previous mandate asked whether the language of her future learning could be
a product of experience. This one asks whether she DECIDES to change it, and
that is a different kind of question with a different prerequisite. A system
cannot choose between courses of action whose values it cannot estimate, and it
cannot estimate them from what it does not record.

So this is the prerequisite, and it is deliberately small. An episode is one
occasion of trying to say something: what was asked, which route answered it,
what that cost in candidates walked, what it used, and what — if anything — was
admitted as a result. Everything the decision rule needs is a statistic of
these, and nothing here is a category of opportunity.

Three statistics, and each answers a question that was previously unanswerable
from inside:

    how often has this come up      the recurrence estimate, which is what
                                    turns "worth doing once" into "worth
                                    carrying"
    what has this route cost        the attribution, which is what makes
                                    "which part of me is slow" a fact
    when was this last used         the disuse, which is what makes dropping
                                    something a measurement

Bounded, because memory is. The ring keeps the most recent episodes and the
counts survive the episodes they were taken from, so a recurrence seen a
thousand episodes ago still counts even though the episode itself is gone. That
is the shape finite memory forces: keep the statistics, forget the instances.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.runtime.errors import record_degradation

__all__ = [
    "Episode",
    "HOW_MANY_EPISODES_ARE_KEPT",
    "attribution",
    "episodes",
    "forget_the_record",
    "how_long_since",
    "how_often",
    "keep_the_record",
    "note_an_episode",
    "note_a_use",
    "recall_the_record",
    "the_record",
    "what_it_has_cost",
]

logger = logging.getLogger("Aura.TheRecordOfHerOwnWork")

#: How many episodes are kept whole. The counts outlive them, so a structure
#: seen long ago still counts; what is lost is the instance, not the statistic.
#: Read off what a decision needs rather than chosen: the estimates below use
#: counts, and only the attribution reads whole episodes.
HOW_MANY_EPISODES_ARE_KEPT = 512

_KEPT_AT = Path.home() / ".aura" / "state" / "the_record_of_her_own_work.json"


@dataclass(frozen=True, slots=True)
class Episode:
    """One occasion of trying to say something, and what it cost."""

    #: What was asked, as a key that recurs when the same shape recurs.
    family: str
    #: Which route answered it, or nothing where none did.
    route: str | None
    #: Candidates walked. The unit everything here is priced in.
    walked: int
    #: Library entries the answer was built from.
    used: tuple[str, ...] = ()
    #: What was admitted because of it, where anything was.
    admitted: str | None = None
    when: float = field(default_factory=time.monotonic)

    def describes(self) -> str:
        said = self.route or "nothing"
        return f"{self.family}: {said} after {self.walked:,}"


@dataclass
class Record:
    """Everything a decision about developing needs, and nothing else."""

    kept: list[Episode] = field(default_factory=list)
    #: How often each family has come up, past the episodes still held.
    families: Counter = field(default_factory=Counter)
    #: How often each library entry has been used.
    uses: Counter = field(default_factory=Counter)
    #: Which episode each entry was last used at, counted in episodes.
    last_used: dict[str, int] = field(default_factory=dict)
    #: How many episodes there have ever been.
    seen: int = 0

    def note(self, episode: Episode) -> None:
        self.seen += 1
        self.families[episode.family] += 1
        for name in episode.used:
            self.uses[name] += 1
            self.last_used[name] = self.seen
        self.kept.append(episode)
        if len(self.kept) > HOW_MANY_EPISODES_ARE_KEPT:
            # The instance goes, the counts stay. That is what finite memory
            # forces, and it is why the counts are kept beside the ring rather
            # than computed from it.
            del self.kept[: len(self.kept) - HOW_MANY_EPISODES_ARE_KEPT]


_RECORD = Record()


def the_record() -> Record:
    """The record itself, for anything that needs more than a statistic."""
    return _RECORD


def episodes() -> tuple[Episode, ...]:
    return tuple(_RECORD.kept)


def note_an_episode(
    family: str,
    *,
    route: str | None,
    walked: int,
    used: Sequence[str] = (),
    admitted: str | None = None,
) -> Episode:
    """Write down one occasion. Called from the answering path, not from a test."""
    made = Episode(
        family=str(family),
        route=route,
        walked=max(0, int(walked)),
        used=tuple(str(one) for one in used),
        admitted=admitted,
    )
    _RECORD.note(made)
    return made


def note_a_use(name: str) -> None:
    """Record that something was used, where no whole episode is being written.

    A library entry used inside ordinary cognition is used, and counting it
    only when an episode is written would make everything look disused.
    """
    _RECORD.uses[str(name)] += 1
    _RECORD.last_used[str(name)] = _RECORD.seen


def how_often(family: str) -> int:
    """How many times this shape has come up. The recurrence estimate."""
    return int(_RECORD.families.get(str(family), 0))


def how_long_since(name: str) -> int | None:
    """Episodes since this entry was last used, or nothing if it never was."""
    at = _RECORD.last_used.get(str(name))
    return None if at is None else max(0, _RECORD.seen - at)


def what_it_has_cost(route: str) -> int | None:
    """What this route has cost on average, or nothing where it never ran.

    The measured cost of a developmental action, so nothing has to estimate
    what has already been observed.
    """
    spent = [one.walked for one in _RECORD.kept if one.route == route]
    return int(round(sum(spent) / len(spent))) if spent else None


def attribution() -> dict[str, dict[str, Any]]:
    """What each route has cost and how often it has answered.

    The self-model, and it is a plain one: which part of her spends the search
    is a fact about the record rather than a thing she believes about herself.
    A route that answers rarely and costs much is a bottleneck, and that is
    readable here without anything having to say the word.
    """
    spent: Counter = Counter()
    answered: Counter = Counter()
    tried: Counter = Counter()
    for one in _RECORD.kept:
        where = one.route or "nothing answered"
        spent[where] += one.walked
        tried[where] += 1
        if one.route is not None:
            answered[where] += 1
    return {
        where: {
            "walked": int(spent[where]),
            "answered": int(answered[where]),
            "episodes": int(tried[where]),
            "each": round(spent[where] / max(1, tried[where]), 1),
        }
        for where in sorted(tried)
    }


def keep_the_record() -> bool:
    """Write it down, so a developmental history survives a restart."""
    body = {
        "seen": _RECORD.seen,
        "families": dict(_RECORD.families),
        "uses": dict(_RECORD.uses),
        "last_used": dict(_RECORD.last_used),
        "kept": [
            {
                "family": one.family,
                "route": one.route,
                "walked": one.walked,
                "used": list(one.used),
                "admitted": one.admitted,
            }
            for one in _RECORD.kept
        ],
    }
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "the_record_of_her_own_work.keep", domain="state_mutation"
        ):
            get_file_write_gateway().ensure_directory(
                _KEPT_AT.parent, source="the_record_of_her_own_work"
            )
            get_file_write_gateway().write_text(
                _KEPT_AT, json.dumps(body), source="the_record_of_her_own_work"
            )
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "the_record_of_her_own_work", exc, severity="info",
            action="keep the record of her own work",
        )
        return False


def recall_the_record() -> int:
    """Put it back. Returns how many episodes came back."""
    try:
        held = json.loads(_KEPT_AT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(held, dict):
        return 0
    _RECORD.seen = int(held.get("seen") or 0)
    _RECORD.families = Counter(
        {str(k): int(v) for k, v in (held.get("families") or {}).items()}
    )
    _RECORD.uses = Counter(
        {str(k): int(v) for k, v in (held.get("uses") or {}).items()}
    )
    _RECORD.last_used = {
        str(k): int(v) for k, v in (held.get("last_used") or {}).items()
    }
    _RECORD.kept = []
    for row in held.get("kept") or ():
        if not isinstance(row, dict):
            continue
        _RECORD.kept.append(
            Episode(
                family=str(row.get("family") or ""),
                route=row.get("route"),
                walked=int(row.get("walked") or 0),
                used=tuple(str(one) for one in row.get("used") or ()),
                admitted=row.get("admitted"),
            )
        )
    return len(_RECORD.kept)


def forget_the_record() -> None:
    """Start again. Used by tests, and by nothing else."""
    _RECORD.kept.clear()
    _RECORD.families.clear()
    _RECORD.uses.clear()
    _RECORD.last_used.clear()
    _RECORD.seen = 0
