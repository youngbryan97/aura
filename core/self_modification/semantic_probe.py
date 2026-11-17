"""What her program's values were where a repair is being looked for: recorded while her tests run, and sometimes
steered (Master Todo F16, semantic program repair).

Semantic repair (core/self_modification/semantic_repair.py) puts calls to this module into a copy of a file, at the
condition of a branch, around a statement, or before one, and runs her tests on that copy. Each call writes down the
values in scope there, what the branch decided, and which test was running. A call can also be told what to decide
instead, which is how the outcome a failing test needed is found: force the branch the other way and see whether the
test passes (angelic fix localisation, Nopol, Xuan et al., TSE 2017).

The plan and the record are files whose paths are written into the instrumented copy, so nothing here reads her
environment or her state. Values are kept as plain numbers, booleans, None, short strings and sizes; anything else
is kept only as whether it is None.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["cond", "guard", "leave", "summarise", "value"]

#: Evaluations kept for one location in one test. A condition in a loop can run millions of times, and the synthesis
#: needs examples, not all of them.
KEPT_PER_TEST = 40
#: Characters of a string kept as its value; a longer string is kept as its length.
SHORT_TEXT = 40
#: Names in one scope that are looked at, and attributes of one object.
NAMES_LOOKED_AT = 60
ATTRIBUTES_LOOKED_AT = 20

_plans: dict[str, dict[str, Any]] = {}
_counts: dict[tuple[str, str, str], int] = {}
#: Which of a session's plans is running, when one session runs many (forcing_session.py); None when it runs one.
FORCING: int | None = None


def _plan(path: str) -> dict[str, Any]:
    if path not in _plans:
        try:
            with open(path, encoding="utf-8") as handle:
                _plans[path] = json.load(handle)
        except (OSError, ValueError):
            _plans[path] = {}
    return _plans[path]


def _test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]


class _Opaque:
    """A value that is neither plain nor sized: only whether it is None is kept."""


_OPAQUE = _Opaque()


def _plain(value: Any) -> Any:
    """``value`` when it is a plain value worth comparing, else _OPAQUE."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else _OPAQUE
    if isinstance(value, str) and len(value) <= SHORT_TEXT:
        return value
    return _OPAQUE


def _sized(value: Any) -> int | None:
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        return len(value)
    return None


#: The containers whose membership of a plain value in scope is kept as a term.
_HOLDERS = (dict, list, tuple, set, frozenset)


def summarise(scope: Mapping[str, Any]) -> dict[str, Any]:
    """The terms a condition here could be built from, by the expression that reads each: ``x`` for a plain value,
    ``len(x)`` for a size, ``x is None`` for anything else, ``x.a`` for an object's plain attributes, and ``k in d``
    for a plain value and a container both in scope.

    Membership is what a lookup depends on. LIVE 17 November 2026 her loop mended ``prices[item]``, which raised
    KeyError, with ``if len(prices) <= 1: return None``: it told the failing run's one-item table from the passing
    run's two, and returned None for an item a one-item table holds. Whether the table holds the item was not a term
    it could use."""
    terms: dict[str, Any] = {}
    plain_names = [n for n, v in list(scope.items())[:NAMES_LOOKED_AT]
                   if not n.startswith(("__", "_aura_")) and _plain(v) is not _OPAQUE and not isinstance(v, bool)]
    for holder, held in list(scope.items())[:NAMES_LOOKED_AT]:
        if holder.startswith(("__", "_aura_")) or not isinstance(held, _HOLDERS):
            continue
        for name in plain_names:
            try:
                terms[f"{name} in {holder}"] = scope[name] in held
            except TypeError:
                continue  # not a value this container can hold
    for name, value in list(scope.items())[:NAMES_LOOKED_AT]:
        if name.startswith("__") or name.startswith("_aura_"):
            continue
        if callable(value) and _sized(value) is None:
            continue
        plain = _plain(value)
        size = _sized(value)
        if plain is not _OPAQUE:
            terms[name] = plain
        else:
            terms[f"{name} is None"] = False
        if size is not None:
            terms[f"len({name})"] = size
        fields = getattr(value, "__dict__", None) if plain is _OPAQUE else None
        if isinstance(fields, dict):
            for attribute, inner in list(fields.items())[:ATTRIBUTES_LOOKED_AT]:
                if attribute.startswith("__"):
                    continue
                inner_plain = _plain(inner)
                if inner_plain is not _OPAQUE:
                    terms[f"{name}.{attribute}"] = inner_plain
                inner_size = _sized(inner)
                if inner_size is not None:
                    terms[f"len({name}.{attribute})"] = inner_size
    return terms


def _values_of(parts: Mapping[str, Callable[[], Any]] | None) -> dict[str, Any]:
    """Each part of a condition, evaluated on its own where it can be; a part that raises is left out."""
    out: dict[str, Any] = {}
    for text, read in (parts or {}).items():
        try:
            value = _plain(read())
        except Exception:  # noqa: BLE001 - a part read out of its short-circuit may raise anything; it is left out
            continue
        if value is not _OPAQUE:
            out[text] = value
    return out


def _count(record: str, location: str) -> int:
    """Which evaluation of this location in the running test this is, counting from one."""
    key = (record, location, _test())
    _counts[key] = _counts.get(key, 0) + 1
    return _counts[key]


def _record(record: str, location: str, kind: str, scope: Mapping[str, Any], decided: Any, n: int,
            **more: Any) -> None:
    if n > KEPT_PER_TEST:
        return
    try:
        row = {"location": location, "kind": kind, "test": _test(), "n": n, "decided": decided,
               "terms": summarise(scope), "forcing": FORCING, **more}
        # A raw append, not her write gateway: this module runs inside the test process of whatever copy is being
        # repaired, an old commit or a planted package, where her governance is not running, and the record is a
        # scratch file of the search's own (governance baseline, 16 November 2026).
        with open(record, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass  # not a failure: a row that cannot be written is one example fewer, never a change to what ran


def _forced(plan_path: str, location: str, n: int, decided: Any) -> Any:
    """What the plan says this evaluation decides, or None to leave it be.

    A step is either ``{"decide": v}``, the same for every evaluation (in the tests it names, or all), or
    ``{"flip": {test: [n, ...]}}``, the named evaluations of a test decided the other way: a branch in a loop can need
    one evaluation changed and the rest kept (angelic values per evaluation, as Angelix searches them).
    """
    step = _plan(plan_path).get(location)
    if step is None:
        return None
    test = _test()
    if "flip" in step:
        flips = step["flip"].get(test) or step["flip"].get("*") or ()
        return (not decided) if n in flips else None
    tests = step.get("tests")
    if tests and test not in tests:
        return None
    return step.get("decide")


def cond(location: str, value: Any, scope: Mapping[str, Any], plan: str, record: str,
         parts: Mapping[str, Callable[[], Any]] | None = None) -> bool:
    """A branch's condition: its own value, recorded with the values in scope, or what the plan decides instead."""
    decided = bool(value)
    n = _count(record, location)
    forced = _forced(plan, location, n, decided)
    _record(record, location, "cond", scope, decided, n, parts=_values_of(parts), forced=forced)
    return decided if forced is None else bool(forced)


def guard(location: str, scope: Mapping[str, Any], plan: str, record: str) -> bool:
    """Whether a statement runs: always, unless the plan says to skip it here."""
    n = _count(record, location)
    forced = _forced(plan, location, n, True)
    _record(record, location, "guard", scope, True, n, forced=forced)
    return True if forced is None else bool(forced)


def leave(location: str, scope: Mapping[str, Any], plan: str, record: str) -> int:
    """Which of the values written beside it to return before a statement, or -1 to go on: always -1, unless the
    plan names one."""
    n = _count(record, location)
    forced = _forced(plan, location, n, -1)
    _record(record, location, "leave", scope, -1, n, forced=forced)
    return -1 if forced is None or isinstance(forced, bool) else int(forced)


def value(location: str, read: Callable[[], Any], scope: Mapping[str, Any], plan: str, record: str,
          candidates: Mapping[str, Callable[[], Any]] | None = None, pure: tuple[str, ...] = ()) -> Any:
    """A value read on a suspect line: its own, recorded with its type and the types of the candidates that can be
    read without a call (``pure``), or the candidate the plan picks instead."""
    n = _count(record, location)
    step = _plan(plan).get(location) or {}
    pick = step.get("pick")
    tests = step.get("tests")
    if pick is not None and pick in (candidates or {}) and (not tests or _test() in tests):
        _record(record, location, "value", {}, None, n, forced=pick)
        return candidates[pick]()
    got = read()
    if n <= KEPT_PER_TEST:
        types: dict[str, str] = {}
        same: list[str] = []
        for text in pure:
            try:
                theirs = candidates[text]()
            except Exception:  # noqa: BLE001 - a candidate read where it is not bound says only that it is not
                continue
            types[text] = type(theirs).__name__
            if _plain(theirs) is not _OPAQUE and _plain(got) is not _OPAQUE and theirs == got:
                same.append(text)
        _record(record, location, "value", scope, type(got).__name__, n, types=types, same=same)
    return got
