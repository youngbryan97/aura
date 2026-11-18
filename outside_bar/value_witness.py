"""A pytest plugin that writes down what the functions of the named files returned, call by call.

Loaded by outside_bar/verify_change.py into a test run. It reads two environment variables: OUTSIDE_BAR_WITNESS_FUNCTIONS,
JSON mapping absolute paths to the functions a commit changed, and OUTSIDE_BAR_WITNESS_OUT, the file to write. Each call
is keyed by the running test, the function, how many times that test had called it with those arguments, and the
arguments. A value whose text holds a memory address cannot be compared between two runs, so it is left out. The plain
arguments of module-level functions are kept too, for the differential run on inputs near them.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from typing import Any

_TOOL = sys.monitoring.PROFILER_ID
_EVENTS = sys.monitoring.events
#: Calls kept for one function in one test: enough witnesses, and a loop of a million calls stays cheap.
_KEPT = 200
_SKIPPED_KINDS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR

_functions = {path: frozenset(names) for path, names in
              json.loads(os.environ.get("OUTSIDE_BAR_WITNESS_FUNCTIONS") or "{}").items()}
_seen: dict[str, str] = {}
_inputs: dict[str, list[dict[str, Any]]] = {}
_ran: set[str] = set()
_counts: dict[tuple[str, str, str], int] = {}
_open: dict[int, list[str | None]] = {}


def _text(value: Any) -> str | None:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a value that cannot be shown is a value that cannot be compared
        return None
    return None if " at 0x" in text else hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _named(code: Any) -> bool:
    return code.co_qualname in _functions.get(code.co_filename, ())


def _watched(code: Any) -> bool:
    return _named(code) and not code.co_flags & _SKIPPED_KINDS


def _keep_the_input(code: Any, frame: Any, names: tuple[str, ...]) -> None:
    if "." in code.co_qualname or len(_inputs.get(f"{code.co_filename}::{code.co_qualname}", ())) >= _KEPT:
        return
    try:
        plain = json.loads(json.dumps({name: frame.f_locals[name] for name in names}))
    except (KeyError, TypeError, ValueError):
        return
    _inputs.setdefault(f"{code.co_filename}::{code.co_qualname}", []).append(plain)


def _key_of(code: Any, frame: Any) -> str | None:
    names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
    _keep_the_input(code, frame, names)
    shown = []
    for name in names:
        if name in ("self", "cls"):
            continue
        text = _text(frame.f_locals.get(name))
        if text is None:
            return None
        shown.append(text)
    test = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
    arguments = ",".join(shown)
    count = _counts.get((test, code.co_qualname, arguments), 0) + 1
    _counts[(test, code.co_qualname, arguments)] = count
    return None if count > _KEPT else f"{test}|{code.co_qualname}|{count}|{arguments}"


def _started(code: Any, _offset: int) -> Any:
    if _named(code):
        _ran.add(f"{code.co_filename}::{code.co_qualname}")
    if not _watched(code):
        return sys.monitoring.DISABLE
    _open.setdefault(id(code), []).append(_key_of(code, sys._getframe(1)))
    return None


def _returned(code: Any, _offset: int, value: Any) -> Any:
    if not _watched(code):
        return sys.monitoring.DISABLE
    key = _open.get(id(code), [None]).pop() if _open.get(id(code)) else None
    text = _text(value)
    if key is not None and text is not None:
        _seen[key] = text
    return None


def _unwound(code: Any, _offset: int, error: BaseException) -> None:
    if _watched(code) and _open.get(id(code)):
        key = _open[id(code)].pop()
        if key is not None:
            _seen[key] = f"raised {type(error).__name__}"


def pytest_sessionstart(session: Any) -> None:
    sys.monitoring.use_tool_id(_TOOL, "outside_bar_value_witness")
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_START, _started)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_RETURN, _returned)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_UNWIND, _unwound)
    sys.monitoring.set_events(_TOOL, _EVENTS.PY_START | _EVENTS.PY_RETURN | _EVENTS.PY_UNWIND)


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    sys.monitoring.set_events(_TOOL, 0)
    sys.monitoring.free_tool_id(_TOOL)
    with open(os.environ["OUTSIDE_BAR_WITNESS_OUT"], "w", encoding="utf-8") as handle:
        json.dump({"values": _seen, "inputs": _inputs, "ran": sorted(_ran)}, handle)
