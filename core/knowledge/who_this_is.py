"""core/knowledge/who_this_is.py — one name for one thing, across every store.

OpenCog interns every atom in one space, so two references to the same concept
are the same object and a graph relationship is enforced rather than hoped
for. Aura keeps knowledge in several places that each name things their own
way: the associative entity memory content-addresses a kind and a name into
``ent_<digest>``; the perception belief state builds ``label:glyph:position``;
the metagraph uses node identity; conversation memory uses free text.

None of those are wrong for what they do. What is missing is the join. Two
stores holding a fact about the same person cannot tell that they do, so a
reference across them is a string comparison and a duplicate is invisible.

Three things here, and the third is what makes the first two worth having.

* One id. Content-addressed on a kind and a normalised name, which is the
  scheme the entity memory already uses — adopted rather than replaced, so
  every id it has already minted is already canonical.
* Equivalence with a reason. Two ids can be declared the same thing, and the
  declaration says why. Merging is by union with the smaller id winning, so
  the canonical form does not depend on the order the equivalences arrived.
* Duplicates are findable. Two stores holding the same name under different
  ids is the failure this exists to catch, and it is a query rather than a
  hope.

The equivalences persist. An identity worked out once and forgotten on restart
is worse than never working it out, because the work looks done.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhoThisIs")

__all__ = [
    "AnEquivalence",
    "also_known_as",
    "an_id_for",
    "duplicates_among",
    "forget_the_equivalences",
    "keep_the_equivalences",
    "normalise",
    "the_equivalences",
    "the_same",
    "what_it_was_called",
]

_LOCK = checked_lock("core.knowledge.who_this_is", reentrant=True)
#: id -> the id it was merged into. A chain, compressed on read.
_MERGED_INTO: dict[str, str] = {}
#: canonical id -> what it has been called, so a duplicate can be explained.
_CALLED: dict[str, set[str]] = {}
_WHY: list["AnEquivalence"] = []
_LOADED = False


@dataclass(frozen=True)
class AnEquivalence:
    """Two ids declared the same thing, and on what evidence."""

    one: str
    other: str
    because: str
    #: How sure. A declared equivalence with weak evidence is still worth
    #: recording; what it must not do is silently win over a strong one.
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "one": self.one,
            "other": self.other,
            "because": self.because,
            "confidence": round(float(self.confidence), 4),
        }


def normalise(name: Any) -> str:
    """The form two spellings of one name agree on.

    Case, surrounding space and internal runs of space. Nothing cleverer:
    a normaliser that strips punctuation makes "St. John" and "St John" the
    same person and "C++" and "C" the same language, and only one of those is
    wanted.
    """

    return re.sub(r"\s+", " ", str(name or "").strip()).casefold()


def an_id_for(kind: Any, name: Any) -> str:
    """The canonical id for a thing of this kind with this name.

    The same scheme the entity memory already uses, so every id it minted is
    already canonical and adopting this changes nothing it has stored.
    """

    seed = f"{str(kind or '').strip()}|{normalise(name)}".encode()
    made = "ent_" + hashlib.blake2b(seed, digest_size=10).hexdigest()
    with _LOCK:
        _CALLED.setdefault(the_same(made), set()).add(str(name or ""))
    return made


def the_same(entity_id: Any) -> str:
    """The canonical id this one resolves to, following any merges."""

    found = str(entity_id or "")
    with _LOCK:
        seen: list[str] = []
        while found in _MERGED_INTO:
            seen.append(found)
            found = _MERGED_INTO[found]
        # Compress, so a long chain is walked once.
        for one in seen:
            _MERGED_INTO[one] = found
    return found


def also_known_as(
    one: Any, other: Any, *, because: str, confidence: float = 1.0
) -> str:
    """Declare two ids the same thing. Returns the canonical id.

    The smaller id wins, so the canonical form does not depend on the order
    the declarations arrived in — which is the property that makes two
    processes reaching the same conclusion agree about it.
    """

    first, second = the_same(one), the_same(other)
    if not first or not second:
        raise ValueError("an equivalence needs two ids")
    if first == second:
        return first
    keep, merge = (first, second) if first < second else (second, first)
    with _LOCK:
        _MERGED_INTO[merge] = keep
        _CALLED.setdefault(keep, set()).update(_CALLED.pop(merge, set()))
        _WHY.append(
            AnEquivalence(one=merge, other=keep, because=str(because), confidence=confidence)
        )
    return keep


def what_it_was_called(entity_id: Any) -> tuple[str, ...]:
    """Every name that has resolved to this thing."""

    with _LOCK:
        return tuple(sorted(_CALLED.get(the_same(entity_id), set())))


def the_equivalences() -> tuple[AnEquivalence, ...]:
    with _LOCK:
        return tuple(_WHY)


def duplicates_among(
    stores: Mapping[str, Iterable[tuple[Any, Any, Any]]],
) -> list[dict[str, Any]]:
    """Things two stores hold under different ids and the same name.

    Each store yields ``(kind, name, id)``. The finding this answers asks for
    duplicate detection spanning memories and graphs, and spanning them is the
    whole point: a duplicate inside one store is that store's business, and one
    across two is nobody's until something looks.
    """

    by_name: dict[tuple[str, str], dict[str, set[str]]] = {}
    for store, rows in stores.items():
        for kind, name, entity_id in rows:
            key = (str(kind or ""), normalise(name))
            by_name.setdefault(key, {}).setdefault(str(store), set()).add(
                the_same(entity_id)
            )
    found: list[dict[str, Any]] = []
    for (kind, name), where in sorted(by_name.items()):
        ids = {one for got in where.values() for one in got}
        if len(ids) < 2:
            continue
        found.append(
            {
                "kind": kind,
                "name": name,
                "ids": sorted(ids),
                "stores": {one: sorted(got) for one, got in sorted(where.items())},
                "the_canonical_one": an_id_for(kind, name),
            }
        )
    return found


def _where_it_is_kept() -> Any:
    from pathlib import Path

    from core.runtime.state_ownership import state_root

    return Path(state_root()) / "who_this_is.json"


def keep_the_equivalences() -> int:
    """Write the equivalences. Returns how many, or -1 where it could not.

    An identity worked out once and forgotten on restart is worse than never
    working it out, because the work looks done.
    """

    with _LOCK:
        body = {
            "merged_into": dict(_MERGED_INTO),
            "called": {one: sorted(names) for one, names in _CALLED.items()},
            "why": [one.to_dict() for one in _WHY],
        }
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("who_this_is.keep", domain="state_mutation"):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(_where_it_is_kept().parent, source="who_this_is")
            gateway.write_text(
                _where_it_is_kept(), json.dumps(body), source="who_this_is"
            )
        return len(body["why"])
    except Exception as exc:  # noqa: BLE001 — losing this is not fatal to a turn
        from core.runtime.errors import record_degradation

        record_degradation(
            "who_this_is",
            exc,
            severity="warning",
            action="the equivalences were not written and will not survive a restart",
        )
        return -1


def _remember_what_was_worked_out() -> None:
    """Read the equivalences once per process, at the first question."""

    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True
    place = _where_it_is_kept()
    if not place.exists():
        return
    try:
        body = json.loads(place.read_text())
        with _LOCK:
            _MERGED_INTO.update({str(k): str(v) for k, v in (body.get("merged_into") or {}).items()})
            for one, names in (body.get("called") or {}).items():
                _CALLED.setdefault(str(one), set()).update(str(n) for n in names)
            for row in body.get("why") or ():
                _WHY.append(
                    AnEquivalence(
                        one=str(row.get("one", "")),
                        other=str(row.get("other", "")),
                        because=str(row.get("because", "")),
                        confidence=float(row.get("confidence", 1.0)),
                    )
                )
    except Exception as exc:  # noqa: BLE001 — a bad file is not a dead boot
        logger.info("could not read the equivalences: %s", exc)


def forget_the_equivalences() -> None:
    """For tests. Nothing in production clears identity."""

    global _LOADED
    with _LOCK:
        _MERGED_INTO.clear()
        _CALLED.clear()
        _WHY.clear()
        _LOADED = True
