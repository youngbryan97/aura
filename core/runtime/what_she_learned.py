"""What she worked out about a thing, kept for the next time she is in it.

Everything she learns about a world she is acting in — which part of it answers
to her, how it moves when she pushes it, which lines held — has been dying with
the process. So the fortieth run started exactly as ignorant as the first, and
experience was something she had during a run rather than something she had.

Kept per world rather than in general, because that is the honest scope: how a
game board moves is not how a spreadsheet moves, and a rule that held on one
thing is a guess about the next. What is remembered here is remembered about
the named thing it was learned in.

Nothing is trusted on the strength of having been written down. What comes
back is a starting point that has to keep earning its place against what she
sees now, which is why the counts come back discounted rather than whole.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.verify import invariant

__all__ = ["forget", "kept_worlds", "named", "recall", "remember", "validate_indexed_state", "TRUST_CARRIED_OVER"]

logger = logging.getLogger("Aura.WhatSheLearned")

#: Where it is kept. One file per thing she has acted in. Resolved on each
#: call: fixed at import, a test run inherits the live instance's directory and
#: writes practice worlds into the mind that has to act in the real one.
#: Set to send it somewhere else. Left alone in the live runtime; a test
#: that wants its own file names one here.
_KEPT_IN: Path | None = None


def _kept_in() -> Path:
    if _KEPT_IN is not None:
        return _KEPT_IN
    from core.runtime.state_ownership import state_root

    return state_root() / "state" / "worlds"

#: How much of what she knew comes back. Under one on purpose: something she
#: worked out yesterday is evidence about today and not a fact about it, and a
#: handful of things that disagree should be able to overturn it.
TRUST_CARRIED_OVER = 0.5

#: How much can be kept about any one thing. A record that grows without a
#: bound is a record nobody reads.
_MOST_KEPT = 60_000
_INDEXED_TABLES = "_indexed_tables"
_MISSING = object()


@dataclass(frozen=True)
class _IndexedTable:
    table: tuple[str, ...]
    keyed_by: tuple[tuple[str, ...], ...] = ()
    references: tuple[tuple[str, ...], ...] = ()
    histories: tuple[tuple[str, ...], ...] = ()


def _path(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(p, str) and p for p in value):
        raise ValueError("an indexed-state path must contain field names")
    return tuple(value)


def _relations(what: Mapping[str, Any]) -> tuple[_IndexedTable, ...]:
    declared = what.get(_INDEXED_TABLES, [])
    if not isinstance(declared, list):
        raise ValueError("indexed-state declarations must be a list")
    found = []
    occupied: set[tuple[str, ...]] = set()
    for item in declared:
        if not isinstance(item, dict):
            raise ValueError("an indexed-state declaration must be a mapping")
        relation = _IndexedTable(
            _path(item.get("table")),
            tuple(_path(p) for p in item.get("keyed_by", [])),
            tuple(_path(p) for p in item.get("references", [])),
            tuple(_path(p) for p in item.get("histories", [])),
        )
        paths = (relation.table, *relation.keyed_by, *relation.references)
        for path in paths:
            if path[0] == _INDEXED_TABLES or "*" in path:
                raise ValueError("an indexed-state relation must name a concrete data path")
            if any(path[:len(other)] == other or other[:len(path)] == path for other in occupied):
                raise ValueError("indexed-state relations overlap")
            occupied.add(path)
        found.append(relation)
    for relation in found:
        for pattern in relation.histories:
            owned = any(len(pattern) > len(root) and pattern[:len(root)] == root
                        for root in (relation.table, *relation.keyed_by, *relation.references))
            if not owned:
                raise ValueError("a history must lie strictly inside its own indexed-state closure")
            if any(len(pattern) <= len(other.table)
                   and all(a == b or a == "*" for a, b in zip(pattern, other.table[:len(pattern)], strict=True))
                   for other in found):
                raise ValueError("a history cannot match an indexed table or its ancestor")
    return tuple(found)


def _at(what: Any, path: tuple[str, ...], *, default: Any = None) -> Any:
    for field in path:
        if not isinstance(what, dict) or field not in what:
            return default
        what = what[field]
    return what


def _put(what: dict, path: tuple[str, ...], value: Any) -> None:
    parent = _at(what, path[:-1]) if len(path) > 1 else what
    if isinstance(parent, dict):
        parent[path[-1]] = value


def _drop(what: dict, path: tuple[str, ...]) -> None:
    parent = what
    parents = []
    for field in path[:-1]:
        if not isinstance(parent, dict) or not isinstance(parent.get(field), dict):
            return
        parents.append((parent, field))
        parent = parent[field]
    parent.pop(path[-1], None)
    for outer, field in reversed(parents):
        if outer[field]:
            break
        outer.pop(field)


def _index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or re.fullmatch(r"0|[1-9][0-9]*", str(value)) is None:
        raise ValueError("an indexed-state reference must be a nonnegative integer")
    return int(value)


def _check_relation(what: dict, relation: _IndexedTable) -> None:
    table = _at(what, relation.table)
    if table is None:
        if any(_at(what, p) for p in relation.keyed_by) or any(_at(what, p) is not None for p in relation.references):
            raise ValueError("indexed-state references have no table")
        return
    if not isinstance(table, list):
        raise ValueError("an indexed-state table must be a list")
    for path in relation.keyed_by:
        references = _at(what, path)
        if references is None:
            continue
        if not isinstance(references, dict):
            raise ValueError("indexed-state keyed references must be mappings")
        if any(_index(key) >= len(table) for key in references):
            raise ValueError("an indexed-state keyed reference lies outside its table")
    for path in relation.references:
        reference = _at(what, path)
        if reference is not None and _index(reference) >= len(table):
            raise ValueError("an indexed-state scalar reference lies outside its table")


def validate_indexed_state(what: Mapping[str, Any], *, indexed_tables: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Copy a record, rejecting damaged relation groups while keeping unrelated knowledge.

    Callers can declare older records' tables without rewriting their files.
    Once any reference is outside a table, its in-range references cannot be
    assumed sound either: an older writer may have removed the table's prefix.
    """
    held = json.loads(json.dumps(what))
    if indexed_tables:
        declared = json.loads(json.dumps(list(indexed_tables)))
        existing = held.get(_INDEXED_TABLES, [])
        if isinstance(existing, list):
            declared.extend(item for item in existing if item not in declared)
        held[_INDEXED_TABLES] = declared
    try:
        relations = _relations(held)
    except (TypeError, ValueError) as why:
        record_degradation("what_she_learned", why, severity="info", action="rejected knowledge with an invalid indexed-state declaration")
        return {}
    for relation in relations:
        try:
            _check_relation(held, relation)
        except (TypeError, ValueError) as why:
            for path in (relation.table, *relation.keyed_by, *relation.references):
                _drop(held, path)
            record_degradation("what_she_learned", why, severity="info",
                               action=f"excluded damaged indexed state at {'.'.join(relation.table)}; kept unrelated knowledge")
    return held


def named(*parts: str) -> str:
    """A name for the thing she is acting in, from whatever identifies it.

    Whatever the caller has: an app, a page, an address. Made into something
    that can be a filename without pretending two different things are one.
    """
    said = " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
    cleaned = re.sub(r"[^a-z0-9]+", "-", said).strip("-")
    if len(cleaned) > 80:
        digest = hashlib.sha256(said.encode("utf-8")).hexdigest()[:16]
        return f"{cleaned[:63]}-{digest}"
    return cleaned or "somewhere"



def _history(path: tuple[str, ...], relations: tuple[_IndexedTable, ...]) -> bool:
    return any(any(len(path) > len(root) and path[:len(root)] == root
                   for root in (relation.table, *relation.keyed_by, *relation.references))
               and len(pattern) == len(path) and all(a == b or a == "*" for a, b in zip(pattern, path, strict=True))
               for relation in relations for pattern in relation.histories)


def _protected(path: tuple[str, ...], relations: tuple[_IndexedTable, ...]) -> bool:
    if path[:1] == (_INDEXED_TABLES,):
        return True
    return any(path[:len(root)] == root for relation in relations
               for root in (relation.table, *relation.keyed_by, *relation.references))


def _the_longest_list(value: Any, path: tuple[str, ...] = (), *,
                      relations: tuple[_IndexedTable, ...] = ()) -> tuple[tuple[str, ...], list] | None:
    """The longest history that can be shortened without changing structure."""
    best: tuple[tuple[str, ...], list] | None = None
    if isinstance(value, list):
        if not _protected(path, relations) or _history(path, relations):
            best = (path, value)
        items = enumerate(value)
    elif isinstance(value, dict):
        items = value.items()
    else:
        return None
    for key, inner in items:
        found = _the_longest_list(inner, (*path, str(key)), relations=relations)
        if found is not None and (best is None or len(found[1]) > len(best[1])):
            best = found
    return best


def _compact_table(what: dict, relation: _IndexedTable, keep: list[int]) -> None:
    """Renumber a table and every declared reference in one operation."""
    table = _at(what, relation.table)
    renamed = {old: new for new, old in enumerate(keep)}
    _put(what, relation.table, [table[old] for old in keep])
    for path in relation.keyed_by:
        references = _at(what, path)
        if isinstance(references, dict):
            _put(what, path, {str(renamed[_index(old)]): value for old, value in references.items() if _index(old) in renamed})
    for path in relation.references:
        reference = _at(what, path)
        if reference is not None:
            if _index(reference) in renamed:
                _put(what, path, renamed[_index(reference)])
            else:
                _drop(what, path)
    _check_relation(what, relation)


def _compact_one(what: dict, relations: tuple[_IndexedTable, ...]) -> str | None:
    """Release the least connected row, retaining explicitly selected rows."""
    choices = []
    for order, relation in enumerate(relations):
        table = _at(what, relation.table)
        if not isinstance(table, list) or not table:
            continue
        selected = {_index(value) for path in relation.references if (value := _at(what, path)) is not None}
        mappings = [_at(what, path) for path in relation.keyed_by]
        for index, item in enumerate(table):
            if index in selected:
                continue
            linked = [values[str(index)] for values in mappings if isinstance(values, dict) and str(index) in values]
            size = len(json.dumps(item)) + sum(len(json.dumps(value)) for value in linked)
            # Keep one measured subject in each relation while another
            # relation can yield a row without losing all its knowledge.
            choices.append((len(table) == 1 and bool(linked), len(linked), -size, index, order))
    if not choices:
        return None
    _last_subject, _links, _size, index, order = min(choices)
    relation = relations[order]
    _compact_table(what, relation, [i for i in range(len(_at(what, relation.table))) if i != index])
    return f"{'.'.join(relation.table)} released row {index} with its dependent state; remaining references were remapped"


def _fitted(what: dict[str, Any], most: int) -> tuple[dict[str, Any], list[str]]:
    """The record made small enough to keep, and what was let go to do it.

    A record over the bound used to be refused whole, so a long run lost
    everything it learned for being long: live, 2026-09-18, "too big to keep
    (124122)" at the end of eighty-eight moves, and the next run began from
    what the run before that knew. The longest list goes first, and the older
    half of it: these are records she appends to as she goes, so the newer
    half is what is most like now.
    """
    fitted = json.loads(json.dumps(what))
    relations = _relations(fitted)
    for relation in relations:
        _check_relation(fitted, relation)
    let_go: list[str] = []
    while len(json.dumps(fitted)) > most:
        found = _the_longest_list(fitted, relations=relations)
        if found is None or len(found[1]) < 8:
            break
        path, longest = found
        dropped = len(longest) // 2
        del longest[:dropped]
        let_go.append(f"{'.'.join(path) or 'the record'} lost its oldest {dropped}")
    while len(json.dumps(fitted)) > most:
        released = _compact_one(fitted, relations)
        if released is None:
            break
        let_go.append(released)
    # A part no list in it can be shortened enough, let go whole and by name,
    # so the rest is kept. Refusing the record for one part lost everything
    # else a two-hour run had learned: live, 26 Sep, "too big to keep
    # (1364611)" at the end of the run that reached 2048.
    while len(json.dumps(fitted)) > most and fitted:
        before = len(json.dumps(fitted))
        biggest = max((part for part in fitted if part != _INDEXED_TABLES),
                      key=lambda part: len(json.dumps(fitted[part])), default=None)
        if biggest is None:
            fitted.clear()
            let_go.append("the indexed-state declaration could not fit; its data was already released")
            break
        affected = [relation for relation in relations if any(path[0] == biggest and _at(fitted, path, default=_MISSING) is not _MISSING
                    for path in (relation.table, *relation.keyed_by, *relation.references))]
        drop = {path for relation in affected for path in (relation.table, *relation.keyed_by, *relation.references)} if affected else {(biggest,)}
        for path in sorted(drop):
            value = _at(fitted, path, default=_MISSING)
            if value is not _MISSING:
                size = len(json.dumps(value))
                _drop(fitted, path)
                let_go.append(f"{'.'.join(path)} ({size}) could not be made smaller and was not kept")
        if len(json.dumps(fitted)) >= before:
            raise ValueError("fitting knowledge could not release an existing field")
    for relation in relations:
        _check_relation(fitted, relation)
    if len(json.dumps(fitted)) > most:
        raise ValueError("the knowledge budget cannot hold an empty record")
    return fitted, let_go


@invariant("runtime.indexed_knowledge_keeps_relationships", scope="runtime",
           owner="core/runtime/what_she_learned.py", observational=False)
def _indexed_knowledge_invariant() -> tuple:
    schema = [{"table": ["symbols"], "keyed_by": [["findings"]], "references": [["selected"]]}]
    record = {"symbols": [{"name": f"object {n}"} for n in range(30)],
              "findings": {str(n): [n] * 9 for n in range(30)}, "selected": 29, _INDEXED_TABLES: schema}
    fitted, _released = _fitted(record, 650)
    assert fitted["symbols"][fitted["selected"]]["name"] == "object 29", "a selected symbol changed identity"
    for index, finding in fitted["findings"].items():
        assert len(finding) == 9 and fitted["symbols"][int(index)]["name"] == f"object {finding[0]}", "a finding changed its subject or shape"
    return ()


def remember(world: str, what: dict[str, Any], *, indexed_tables: Sequence[Mapping[str, Any]] = ()) -> bool:
    """Keep what she worked out about this thing."""
    key = named(world)
    if not what:
        return False
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        # Its own bookkeeping, under a name a caller cannot mean.
        #
        # This was `{"world": key, **what}`, so a caller with something of its
        # own called "world" — and the pursuit has one, the model of what the
        # world does on its own — silently replaced the record's name with it,
        # or had its own replaced, depending which way round the merge went.
        # Both happened: one world file on this machine has the model where
        # the name should be and another has the name where the model should
        # be. Whichever way a collision resolves, one of the two is lost.
        what = validate_indexed_state(what, indexed_tables=indexed_tables)
        if not what:
            return False
        body = json.dumps({"_kept_for": key, **what})
        if len(body) > _MOST_KEPT:
            fitted, let_go = _fitted(what, _MOST_KEPT - len(json.dumps({"_kept_for": key})))
            body = json.dumps({"_kept_for": key, **fitted})
            if len(body) > _MOST_KEPT:
                logger.info("what she learned about %r is too big to keep (%d)", key, len(body))
                return False
            logger.info(
                "what she learned about %r was %d over, so %s",
                key, len(json.dumps({"_kept_for": key, **what})) - _MOST_KEPT, "; ".join(let_go),
            )
        with local_internal_governed_scope(
            "what_she_learned.remember",
            domain="state_mutation",
            constraints={"world": key},
        ):
            get_file_write_gateway().ensure_directory(_kept_in(), source="what_she_learned")
            get_file_write_gateway().write_text(
                _kept_in() / f"{key}.json", body, source="what_she_learned"
            )
        return True
    except Exception as exc:  # noqa: BLE001 - remembering is never the task
        record_degradation(
            "what_she_learned", exc, severity="info", action="carried on without remembering"
        )
        return False


def recall(world: str, *, indexed_tables: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """What she worked out about this thing last time, if she has been here."""
    key = named(world)
    try:
        held = json.loads((_kept_in() / f"{key}.json").read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(held, dict):
        return {}
    # Older records kept the name under "world", which is also what the
    # pursuit calls the model of what the world does on its own. A name there
    # is not a model, and every reader of it already refuses a string, so it
    # is left alone rather than guessed at.
    held.pop("_kept_for", None)
    return validate_indexed_state(held, indexed_tables=indexed_tables)


def kept_worlds() -> list[str]:
    """The name of every thing she has kept a record of, for a caller looking across them."""
    try:
        return sorted(path.stem for path in _kept_in().glob("*.json"))
    except OSError:
        return []


def forget(world: str) -> bool:
    """Drop what she knew about a thing, for a caller that has reason to."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    try:
        get_file_write_gateway().delete_file(
            _kept_in() / f"{named(world)}.json", source="what_she_learned"
        )
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        # False says "there was nothing to forget", and a delete that failed
        # means she still knows it.
        logger.warning(
            "could not forget %r (%s: %s)",
            world,
            type(exc).__name__,
            exc,
        )
        return False
