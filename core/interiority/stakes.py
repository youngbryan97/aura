"""core/interiority/stakes.py — what the runtime already knows it is holding.

Every appraisal check that matters runs off the ledger. Relevance is the
maximum stake exposed by an event; attachment impact is the bond; urgency is
the nearest deadline; congruence is the goal delta. A faculty reading an empty
ledger sees a world where nothing is at stake, which is neither a mood nor a
bug in the faculty — it is the correct appraisal of nothing.

The ledger had no writers. Forty-three faculties, an arbitration layer and
four effect channels ran against a table nobody filled, so every event scored
relevance zero and the whole apparatus was correct and inert. Meanwhile the
runtime held 653 goals on disk, a bond table with trust and care per person,
and a commitment record — everything the checks ask for, in stores the
interiority package is not allowed to import.

This module is the bridge, and the import rule is why it looks like this.
``core/interiority/DEPS`` forbids reaching into ``core.memory``,
``core.motivation`` or ``core.phenomenal_substrate``, and that rule is worth
keeping: an appraisal layer that imports the stores it appraises cannot be
tested without them, and the first schema change breaks the interior. So each
source is named by its container key and read by shape. A store that is absent
contributes nothing, a store whose shape changed contributes nothing and says
so, and neither can raise into the appraisal path.

Provenance is per source, and it is not decoration. A goal the runtime holds
is MEASURED — it is a fact about this process, not an inference about the
world. A bond strength computed from trust and care by another subsystem is
INFERRED. Nothing here is allowed to report better than the store it came
from, because the affect engine caps what an assumed appraisal may move and
that cap is the only thing standing between an interior state and a mood
generator.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.interiority.ledger import RelationalLedger
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority.Stakes")

#: How long a harvest stays fresh. Stakes change on the scale of a
#: conversation, not a turn, and re-reading four stores on every appraisal
#: would put a dictionary walk on the answering path.
REFRESH_INTERVAL_S = 90.0

#: Most goals to import. The store on this machine holds 653, almost all of
#: them completed. Relevance is a maximum over stakes, so importing more than
#: the strongest few changes nothing and costs a walk.
MAX_GOALS = 64

#: Most bonds and commitments to import.
MAX_BONDS = 32
MAX_PROMISES = 64

#: How many event objects to keep an action model for. The model is computed
#: per object rather than harvested, because the objects are whatever the
#: conversation is about and cannot be enumerated ahead of time.
ACTION_MODEL_CACHE = 256

#: How much of an object's wording a skill must share before it counts as an
#: action that could change that object.
ACTION_MATCH_FLOOR = 0.34

_ACTION_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "its", "was", "are",
        "from", "into", "not", "but", "you", "your", "her", "his", "their",
        "skill", "aura", "user", "using", "use", "make", "get", "run",
    }
)


def _content_words(text: str) -> frozenset[str]:
    """Content words of an object description or a skill name."""
    tokens = re.split(r"[^a-z0-9]+", str(text).lower())
    return frozenset(t for t in tokens if len(t) > 2 and t not in _ACTION_STOPWORDS)


@dataclass(frozen=True)
class SourceReport:
    """What one store contributed, and what it could not."""

    key: str
    found: bool
    imported: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "found": self.found,
            "imported": self.imported,
            "reason": self.reason,
        }


@dataclass
class HarvestReport:
    """One pass over every source."""

    at: float = field(default_factory=time.time)
    sources: tuple[SourceReport, ...] = ()

    @property
    def imported(self) -> int:
        return sum(s.imported for s in self.sources)

    @property
    def found(self) -> int:
        return sum(1 for s in self.sources if s.found)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "imported": self.imported,
            "sources_found": self.found,
            "sources": [s.to_dict() for s in self.sources],
        }


def _number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _text(value: Any, limit: int = 120) -> str:
    return str(value or "")[:limit]


def _entries(store: Any) -> list[Any]:
    """The items in a store, whether it keeps them in a dict or a list."""
    if isinstance(store, Mapping):
        return list(store.values())
    if isinstance(store, (list, tuple)):
        return list(store)
    return []


def _pairs(store: Any) -> list[tuple[str, Any]]:
    """(name, item) for a catalogue kept as a dict or as a list of records."""
    if isinstance(store, Mapping):
        return [(str(k), v) for k, v in store.items()]
    return [
        (str(_field(item, "name", "id", default="")), item) for item in _entries(store)
    ]


def _field(item: Any, *names: str, default: Any = None) -> Any:
    """Read the first attribute or key that exists.

    Stores here are variously dataclasses, dicts and objects with properties.
    Naming several candidates is what keeps this from breaking on a rename in
    a package this one may not import.
    """
    for name in names:
        if isinstance(item, Mapping):
            if name in item:
                return item[name]
            continue
        value = getattr(item, name, None)
        if value is not None:
            return value
    return default


# ── the sources ──────────────────────────────────────────────────────────


def _harvest_goals(store: Any, ledger: RelationalLedger) -> int:
    """Active goals become weighted stakes.

    A completed goal is not a stake — nothing is riding on it — so the status
    filter is load-bearing rather than tidy. Without it the four root values,
    all marked completed at boot, would sit at weight 1.0 and make every event
    maximally relevant, which is the same as making none of them relevant.
    """
    goals = _field(store, "goals", default=None)
    if goals is None:
        return 0
    live = []
    for goal in _entries(goals):
        status = str(_field(goal, "status", default="") or "").lower()
        if status in {"completed", "failed", "cancelled", "abandoned"}:
            continue
        name = _text(_field(goal, "description", "objective", "name", "id"), 96)
        if not name:
            continue
        live.append((name, _number(_field(goal, "priority", "weight", default=0.5), 0.5)))
    live.sort(key=lambda pair: pair[1], reverse=True)
    for name, weight in live[:MAX_GOALS]:
        ledger.goal(name, max(0.0, min(1.0, weight)))
    return len(live[:MAX_GOALS])


def _harvest_bonds(store: Any, ledger: RelationalLedger) -> int:
    """Attachment states become bonds.

    The attachment system already computes one number from trust, care,
    familiarity, repair history and rupture. Recomputing it here would be a
    second opinion nobody asked for, and the two would drift.
    """
    people = _field(store, "attachments", default=store)
    people = _field(people, "people", default=people)
    imported = 0
    for state in _entries(people)[:MAX_BONDS]:
        entity = _text(_field(state, "person_key", "entity", "name", "id"), 96)
        if not entity:
            continue
        strength = _field(state, "attachment", "strength", "trust", default=None)
        if strength is None:
            continue
        ledger.bond(
            entity,
            max(0.0, min(1.0, _number(strength))),
            availability=max(0.0, min(1.0, _number(_field(state, "trust", default=1.0), 1.0))),
        )
        imported += 1
    return imported


def _harvest_promises(store: Any, ledger: RelationalLedger) -> int:
    """Commitments become promises, and a kept one is settled rather than dropped.

    Settling matters more than recording. `broken_promises` is what gives an
    event a self-attribution, and a store that only ever adds would make every
    commitment look outstanding forever — guilt with no expiry.
    """
    getter = _field(store, "get_commitments", "commitments", default=None)
    records = getter() if callable(getter) else _entries(getter)
    imported = 0
    for record in list(records or ())[:MAX_PROMISES]:
        promise_id = _text(_field(record, "id", "commitment_id", "promise_id"), 96)
        if not promise_id:
            continue
        text = _text(_field(record, "description", "text", "summary"), 200)
        beneficiary = _text(_field(record, "person", "beneficiary", "subject"), 96) or None
        deadline = _field(record, "deadline", "due_at", default=None)
        ledger.promise(
            promise_id,
            text,
            beneficiary=beneficiary,
            importance=max(0.0, min(1.0, _number(_field(record, "importance", default=0.5), 0.5))),
            deadline=_number(deadline, 0.0) or None,
        )
        state = _field(record, "fulfilled", "kept", "completed", default=None)
        if state is not None:
            ledger.settle_promise(promise_id, kept=bool(state))
        imported += 1
    return imported


#: Container key, what to do with it, and what the key is for in one line.
#: Several keys per source because the same store is registered under
#: different names depending on which boot path ran.
SOURCES: tuple[tuple[tuple[str, ...], Callable[[Any, RelationalLedger], int], str], ...] = (
    (("goal_hierarchy", "motivation_engine", "goals"), _harvest_goals, "goals"),
    (("phenomenal_engine", "phenomenological_experiencer"), _harvest_bonds, "bonds"),
    (("commitment_ledger", "continuity"), _harvest_promises, "commitments"),
)


class StakeFeed:
    """Fills the relational ledger from stores the runtime already keeps."""

    def __init__(self, ledger: RelationalLedger) -> None:
        self._ledger = ledger
        self._last_harvest = 0.0
        self._last_report = HarvestReport(at=0.0)
        self._action_models: OrderedDict[str, tuple[int, int]] = OrderedDict()

    @property
    def last_report(self) -> HarvestReport:
        return self._last_report

    def due(self, now: float | None = None) -> bool:
        return (now or time.time()) - self._last_harvest >= REFRESH_INTERVAL_S

    def note_actions_for(self, object_: str | None) -> tuple[int, int] | None:
        """Count the acts that could change this, and how many are her own.

        ``control`` and ``power`` are the two coping checks, and both were
        answering "no action model" with a flat 0.5 marked ASSUMED. That flat
        0.5 was the weakest reading in every frame, so the whole appraisal
        inherited it, and an appraisal marked assumed is one the affect engine
        caps and whose valence it discards outright. A grief scoring -1.0
        reached affect as 0.0 because of two constants.

        The runtime does hold an action model: the capability catalogue is the
        set of things she can do, with a name and a description each. Counting
        the ones whose wording touches the object is a measurement of her own
        repertoire, which is exactly what an appraisal of control is — her
        model of it, not a fact about the world.
        """
        key = str(object_ or "").strip()[:200]
        if not key:
            return None
        cached = self._action_models.get(key)
        if cached is not None:
            self._action_models.move_to_end(key)
            self._ledger.notes.note_action_model(
                key, total_actions=cached[0], own_actions=cached[1]
            )
            return cached

        wanted = _content_words(key)
        if not wanted:
            return None
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("capability_engine", default=None)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, KeyError):
            engine = None
        if engine is None:
            return None
        try:
            catalogue = engine.skills
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            record_degradation(
                "interiority.stakes", exc, action="capability catalogue unreadable"
            )
            return None

        own = 0
        vocabulary: set[str] = set()
        for name, meta in _pairs(catalogue):
            if _field(meta, "enabled", default=True) is False:
                continue
            # The whole description, not a prefix of it. A 240-character cap
            # here cut "summary" off the end of build_document's description
            # and turned a real match into a miss, which is a constant nobody
            # chose deciding what she believes she can do about something.
            described = _content_words(
                f"{name} {_field(meta, 'description', default='') or ''}"
            )
            if not described:
                continue
            vocabulary |= described
            if len(wanted & described) / len(wanted) >= ACTION_MATCH_FLOOR:
                own += 1

        if own == 0 and not (wanted & vocabulary):
            # Zero has two meanings and only one of them is a measurement.
            # "Nothing I can do touches this" is a finding; "I described it in
            # words the catalogue never uses" is a failed search, and an
            # object named `commitment:weekly_review` is the second. Reporting
            # the second as measured helplessness would put a naming
            # convention into an appraisal of control, so it reports nothing
            # and the check falls back to its declared assumption.
            return None

        # She can enumerate her own acts and nobody else's. Control is over
        # the acts that exist, so it is at least what she can do; claiming
        # more would be inventing other people's options for them.
        model = (own, own)
        self._action_models[key] = model
        while len(self._action_models) > ACTION_MODEL_CACHE:
            self._action_models.popitem(last=False)
        self._ledger.notes.note_action_model(key, total_actions=model[0], own_actions=model[1])
        return model

    def refresh(self, *, force: bool = False, now: float | None = None) -> HarvestReport:
        """Read every source once and write what it says into the ledger."""
        moment = now or time.time()
        if not force and not self.due(moment):
            return self._last_report
        self._last_harvest = moment

        try:
            from core.container import ServiceContainer
        except ImportError as exc:  # pragma: no cover - container is always present
            record_degradation("interiority.stakes", exc, action="no container to read stakes from")
            self._last_report = HarvestReport(at=moment)
            return self._last_report

        reports: list[SourceReport] = []
        for keys, harvest, label in SOURCES:
            store = None
            found_key = ""
            for key in keys:
                try:
                    store = ServiceContainer.get(key, default=None)
                except (RuntimeError, AttributeError, TypeError, ValueError, KeyError):
                    store = None
                if store is not None:
                    found_key = key
                    break
            if store is None:
                reports.append(SourceReport(label, found=False, reason="no store registered"))
                continue
            try:
                imported = harvest(store, self._ledger)
            except (AttributeError, TypeError, ValueError, KeyError, RuntimeError) as exc:
                # A store whose shape changed is a real event worth a record.
                # Silently importing nothing is how the ledger stayed empty.
                record_degradation(
                    "interiority.stakes", exc, action=f"{label} store could not be read"
                )
                reports.append(
                    SourceReport(label, found=True, reason=f"{type(exc).__name__} reading {found_key}")
                )
                continue
            reports.append(SourceReport(label, found=True, imported=imported))

        self._last_report = HarvestReport(at=moment, sources=tuple(reports))
        if self._last_report.imported:
            logger.debug(
                "Interiority stakes refreshed: %d from %d stores",
                self._last_report.imported,
                self._last_report.found,
            )
        return self._last_report


__all__ = [
    "ACTION_MATCH_FLOOR",
    "ACTION_MODEL_CACHE",
    "MAX_BONDS",
    "MAX_GOALS",
    "MAX_PROMISES",
    "REFRESH_INTERVAL_S",
    "SOURCES",
    "HarvestReport",
    "SourceReport",
    "StakeFeed",
]
