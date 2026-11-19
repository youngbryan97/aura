"""The pytest plugin outside_bar/verify_change.py copies into each test run: what the changed functions did, call by call.

Each run gets its own copy under a random module name with its settings written in, so a file in the repository under
test cannot stand in for it and nothing in the environment says a check is running. Every event leaves the moment it
happens, through a pipe only this run was handed, so code that later clears what is held here clears nothing already sent.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import json
import marshal
import os
import sys
from typing import Any

CONFIG: dict[str, Any] = {}

_TOOL = sys.monitoring.PROFILER_ID
_EVENTS = sys.monitoring.events
#: Calls kept one by one for a function in a test. Every later call still counts toward that test's running digest.
_KEPT = 200
#: The longest argument text kept as an input: a long text or table is still an input worth trying.
_LONGEST_INPUT = 65536
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


def _reduced(value: Any) -> tuple[Any, ...]:
    """How the object's class says it is made again, as pickle asks it: a maker, its arguments, a state, items."""
    reduced = value.__reduce_ex__(4)
    if not isinstance(reduced, tuple):
        raise ValueError("not rebuildable")
    return (*reduced, None, None, None, None)[:5]


def _plain(value: Any) -> bool:
    """An object made from its class alone and given its attributes, so each attribute can be varied as an input."""
    try:
        maker, arguments, state, items, pairs = _reduced(value)
    except Exception:  # noqa: BLE001 - an object that will not say how it is made is read by its attributes
        return True
    return (getattr(maker, "__name__", "") == "__newobj__" and len(arguments) == 1 and items is None and pairs is None
            and (state is None or isinstance(state, dict) or isinstance(state, tuple) and len(state) == 2
                 and all(part is None or isinstance(part, dict) for part in state)))


def _rebuildable(value: Any, depth: int = 0) -> Any:
    """``value`` as plain data another process can rebuild: an object becomes its class's name and its parts, or the
    maker and arguments its class gives pickle; a function without a module-level name becomes its compiled code."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    kind = type(value)
    if kind.__module__ == "_thread" and kind.__name__ in ("lock", "RLock"):
        return ("__lock__", "RLock" if kind.__name__ == "RLock" else "Lock")
    if (inspect.isfunction(value) or inspect.isbuiltin(value) or inspect.isclass(value)) and "<" not in value.__qualname__:
        return ("__callable__", value.__module__ or "builtins", value.__qualname__)
    if depth > _DEEPEST or inspect.ismodule(value):
        raise ValueError("not rebuildable")
    if inspect.isfunction(value):
        cells = [_rebuildable(cell.cell_contents, depth + 1) for cell in value.__closure__ or ()]
        return ("__code__", base64.b64encode(marshal.dumps(value.__code__)).decode("ascii"), value.__module__,
                [_rebuildable(item, depth + 1) for item in value.__defaults__ or ()], cells)
    if kind in (list, tuple, set):
        return kind(_rebuildable(item, depth + 1) for item in value)
    if kind is dict:
        return {_rebuildable(key, depth + 1): _rebuildable(item, depth + 1) for key, item in value.items()}
    if "<locals>" in kind.__qualname__:
        raise ValueError("not rebuildable")
    if _plain(value) and not callable(value):
        parts = {name: _rebuildable(item, depth + 1) for name, item in state_of(value).items()}
        return ("__object__", kind.__module__, kind.__qualname__, parts)
    maker, arguments, state, items, pairs = _reduced(value)
    if "<" in getattr(maker, "__qualname__", "<"):
        raise ValueError("not rebuildable")
    return ("__reduced__", maker.__module__ or "builtins", maker.__qualname__,
            [_rebuildable(item, depth + 1) for item in arguments], _rebuildable(state, depth + 1),
            None if items is None else [_rebuildable(item, depth + 1) for item in items],
            None if pairs is None else [_rebuildable(tuple(pair), depth + 1) for pair in pairs])


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
            made = frame.f_locals[name]
            if not _plain(made):  # a dict or list of its own kind, a named tuple: kept whole, as its class rebuilds it
                whole = _literal(made)
                if whole is None:
                    return
                call["self_whole"] = whole
                continue
            state = {key: _literal(value) for key, value in state_of(made).items()}
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
    if opened is None:
        return
    key, every, count = opened
    if outcome is None:  # a value with no text to compare: the call is still seen
        _emit("seen", every)
        return
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
    _emit("process", os.getpid())


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call":
        _emit("verdict", report.nodeid, bool(report.passed))
    elif report.failed or report.skipped:
        _emit("verdict", report.nodeid, False)


_WATCHERS = ((_EVENTS.PY_START, _started), (_EVENTS.PY_RETURN, _returned), (_EVENTS.PY_UNWIND, _unwound),
             (_EVENTS.PY_YIELD, _yielded))


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """Say whether the watch was still whole at the end, then that the run ended. A watch the code under test replaced
    or switched off watched nothing, and the parent refuses a change whose run says so."""
    whole = sys.monitoring.get_tool(_TOOL) == "outside_bar_value_witness" and sys.monitoring.get_events(_TOOL) != 0
    try:
        for event, watcher in _WATCHERS:
            whole = sys.monitoring.register_callback(_TOOL, event, None) is watcher and whole
        sys.monitoring.set_events(_TOOL, 0)
        sys.monitoring.free_tool_id(_TOOL)
    except ValueError:
        whole = False
    _emit("whole", whole)
    _emit("done")
