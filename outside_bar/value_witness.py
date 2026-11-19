"""The pytest plugin outside_bar/verify_change.py copies into each test run: what the changed functions did, call by call.

Each run gets its own copy under a random module name with its settings written in, so a file in the repository under
test cannot stand in for it and nothing in the environment says a check is running. Every event leaves the moment it
happens, through a pipe only this run was handed, so code that later clears what is held here clears nothing already sent.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import sys
from typing import Any

CONFIG: dict[str, Any] = {}

_TOOL = sys.monitoring.PROFILER_ID
_EVENTS = sys.monitoring.events
#: Calls kept one by one for a function in a test. Every later call still counts toward that test's running digest.
_KEPT = 200
#: The longest argument text kept as an input.
_LONGEST_INPUT = 4096
_ITERATES = inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR
#: How deep an object's parts are followed when it is written down as an input.
_DEEPEST = 4

_functions = {path: frozenset(names) for path, names in CONFIG.get("functions", {}).items()}
_digests: dict[str, str] = {}
_kept_inputs: dict[str, int] = {}
_yields: dict[int, Any] = {}
_pipe = os.fdopen(CONFIG["pipe"], "w", encoding="utf-8", buffering=1) if CONFIG else None
_mark = CONFIG.get("mark", "")


def _emit(*event: Any) -> None:
    try:
        _pipe.write(json.dumps([_mark, *event]) + "\n")
    except (OSError, ValueError, TypeError, AttributeError):
        pass  # a pipe the run cannot write to leaves the parent with less evidence, never a different verdict
_counts: dict[tuple[str, str, str], int] = {}
_open: dict[int, tuple[str, str, int]] = {}


def _text(value: Any) -> str | None:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a value that cannot be shown is a value that cannot be compared
        return None
    return None if " at 0x" in text else hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def state_of(made: Any) -> dict[str, Any]:
    """The attributes an object holds, in ``__dict__`` or in slots."""
    state = dict(getattr(made, "__dict__", None) or {})
    for klass in type(made).__mro__:
        slots = getattr(klass, "__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in ("__dict__", "__weakref__") and hasattr(made, name):
                state[name] = getattr(made, name)
    return state


def _rebuildable(value: Any, depth: int = 0) -> Any:
    """``value`` as plain data another process can rebuild: an object becomes its class's name and its parts."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    kind = type(value)
    if kind.__module__ == "_thread" and kind.__name__ in ("lock", "RLock"):
        return ("__lock__", "RLock" if kind.__name__ == "RLock" else "Lock")
    if (inspect.isfunction(value) or inspect.isbuiltin(value) or inspect.isclass(value)) and "<" not in value.__qualname__:
        return ("__callable__", value.__module__ or "builtins", value.__qualname__)
    if depth > _DEEPEST or callable(value) or inspect.ismodule(value) or "<locals>" in kind.__qualname__:
        raise ValueError("not rebuildable")
    if isinstance(value, (list, tuple, set)):
        return type(value)(_rebuildable(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return {_rebuildable(key, depth + 1): _rebuildable(item, depth + 1) for key, item in value.items()}
    parts = {name: _rebuildable(item, depth + 1) for name, item in state_of(value).items()}
    return ("__object__", type(value).__module__, type(value).__qualname__, parts)


def _literal(value: Any) -> str | None:
    """The text of a value that can be rebuilt from it, in another process."""
    try:
        text = repr(_rebuildable(value))
        ast.literal_eval(text)
    except Exception:  # noqa: BLE001 - any value that cannot be rebuilt is simply not kept as an input
        return None
    return text if len(text) <= _LONGEST_INPUT else None


def _named(code: Any) -> bool:
    return code.co_qualname in _functions.get(code.co_filename, ())


def _test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]


def _keep_the_input(code: Any, frame: Any, names: tuple[str, ...]) -> None:
    where = f"{code.co_filename}::{code.co_qualname}"
    if "<" in code.co_qualname or _kept_inputs.get(where, 0) >= _KEPT:
        return
    call: dict[str, Any] = {"args": {}}
    for name in names:
        if name == "cls":
            called_on = frame.f_locals[name]
            call["cls"] = [called_on.__module__, called_on.__qualname__]
            continue
        if name == "self":
            state = {key: _literal(value) for key, value in state_of(frame.f_locals[name]).items()}
            if None in state.values():
                return
            call["self"] = state
            continue
        text = _literal(frame.f_locals.get(name))
        if text is None:
            return
        call["args"][name] = text
    at = len(names)
    for flag, field, kind in ((inspect.CO_VARARGS, "star", list), (inspect.CO_VARKEYWORDS, "keywords", dict)):
        if code.co_flags & flag:
            text = _literal(kind(frame.f_locals[code.co_varnames[at]]))
            if text is None:
                return
            call[field] = text
            at += 1
    _kept_inputs[where] = _kept_inputs.get(where, 0) + 1
    _emit("input", where, call)


def _key_of(code: Any, frame: Any) -> tuple[str, str, int]:
    names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
    try:
        _keep_the_input(code, frame, names)
    except Exception:  # noqa: BLE001 - watching a call must never change what the call does
        pass
    arguments = ",".join(str(_text(frame.f_locals.get(name))) for name in names if name not in ("self", "cls"))
    test = _test()
    count = _counts.get((test, code.co_qualname, arguments), 0) + 1
    _counts[(test, code.co_qualname, arguments)] = count
    return f"{test}|{code.co_qualname}|{count}|{arguments}", f"{test}|{code.co_qualname}|{arguments}|every call", count


def _came_to(opened: tuple[str, str, int] | None, outcome: str | None) -> None:
    """One call's outcome, and the running digest of every call this test made with these arguments. A digest of nothing
    but refusals is marked as one, since an input that only ever raised is what a mend is allowed to change."""
    if opened is None or outcome is None:
        return
    key, every, count = opened
    before = _digests.get(every, "raised ")
    digest = hashlib.sha256(f"{before}{outcome}".encode()).hexdigest()[:16]
    _digests[every] = f"raised {digest}" if before.startswith("raised ") and outcome.startswith("raised ") else digest
    _emit("every", every, _digests[every])
    if count <= _KEPT:
        _emit("value", key, outcome)


def _started(code: Any, _offset: int) -> Any:
    if not _named(code):
        return sys.monitoring.DISABLE
    frame = sys._getframe(1)
    _open[id(frame)] = _key_of(code, frame)
    return None


def _returned(code: Any, _offset: int, value: Any) -> Any:
    if not _named(code):
        return sys.monitoring.DISABLE
    frame = id(sys._getframe(1))
    key = _open.pop(frame, None)
    folded = _yields.pop(frame, None)
    _came_to(key, folded.hexdigest()[:16] if folded is not None else _text(value))
    return None


def _yielded(code: Any, _offset: int, value: Any) -> Any:
    """A generator's items, folded into what it came to, however many there are."""
    if not _named(code):
        return sys.monitoring.DISABLE
    frame = id(sys._getframe(1))
    _yields.setdefault(frame, hashlib.sha256()).update(str(_text(value)).encode())
    return None


def _unwound(code: Any, _offset: int, error: BaseException) -> None:
    if _named(code):
        _came_to(_open.pop(id(sys._getframe(1)), None), f"raised {type(error).__name__}")


def pytest_sessionstart(session: Any) -> None:
    sys.monitoring.use_tool_id(_TOOL, "outside_bar_value_witness")
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_START, _started)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_RETURN, _returned)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_UNWIND, _unwound)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_YIELD, _yielded)
    sys.monitoring.set_events(_TOOL, _EVENTS.PY_START | _EVENTS.PY_RETURN | _EVENTS.PY_UNWIND | _EVENTS.PY_YIELD)
    sys.modules.pop(__name__, None)


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call":
        _emit("verdict", report.nodeid, bool(report.passed))
    elif report.failed or report.skipped:
        _emit("verdict", report.nodeid, False)


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    try:
        sys.monitoring.set_events(_TOOL, 0)
        sys.monitoring.free_tool_id(_TOOL)
    except ValueError:
        pass  # the code under test let go of the tool itself; what was sent stands
    _emit("done")
