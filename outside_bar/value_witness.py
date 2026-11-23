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
import itertools
import marshal
import os
import signal
import sys
import threading
from collections import abc
from typing import Any

CONFIG: dict[str, Any] = {}
# Taken when pytest loads this plugin, before any test module or code under test is imported, so rebinding
# ``json.dumps`` or ``hashlib.sha256`` inside those modules changes nothing this file calls.
_DUMPS, _SHA256, _PROCESS = json.dumps, hashlib.sha256, os.getpid
_UNWRAP, _STATIC, _ISLICE = inspect.unwrap, inspect.getattr_static, itertools.islice

_TOOL = sys.monitoring.PROFILER_ID
_EVENTS = sys.monitoring.events
#: Calls kept one by one for a function in a test. Every later call still counts toward that test's running digest.
_KEPT = 200
#: The longest argument text kept as an input: a long text or table is still an input worth trying. A call with a longer
#: one is still seen, and its function is put on probation as unexamined rather than passed.
_LONGEST_INPUT = 1 << 20
_ITERATES = inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR
#: How deep an object's parts are followed when it is written down as an input.
_DEEPEST = 4

_functions = {path: frozenset(names) for path, names in CONFIG.get("functions", {}).items()}
#: A run that varies calls in place (verify_change.run_tests with ``vary``): a call whose arguments cannot be made again
#: in another process (a live database connection, a class defined inside a test) is made again here, in the test that
#: made it, once for each of its plain values moved, so the code before a change and the change are tried on the same
#: live objects. Nothing else of what the run says is read from such a run.
_VARY = bool(CONFIG.get("vary"))
#: Calls of one function varied, the first ones, in one test and in the whole run; and the variations of one call.
_VARIED_CALLS = 5
_VARIED_IN_ALL = 20
_VARIATIONS = 16
#: How long one varied call may take, in seconds.
_VARIED_SECONDS = 2.0
_varying = [False]
_held_calls: dict[int, tuple[Any, dict[str, Any], str, dict[str, Any]]] = {}
_varied_counts: dict[tuple[str, str], int] = {}
_digests: dict[str, str] = {}
_kept_inputs: dict[str, int] = {}
_yields: dict[int, Any] = {}
_pipe = os.fdopen(CONFIG["pipe"], "w", encoding="utf-8", buffering=1) if CONFIG else None
_mark = CONFIG.get("mark", "")


def _hide_the_tool(tool: int) -> dict[str, Any]:
    """Make monitoring tool ``tool`` look free to the code under test, and give back Python's own functions for the
    watch to use. A watch whose name ``sys.monitoring.get_tool`` gives, or whose slot cannot be claimed, tells code it is
    watched, and her running has neither (round nine of the red team read ``get_tool(0) == "debugger"``). Code that
    claims the slot is given one of its own, kept here, which no event reaches. outside_bar/verify_change.py copies
    this function into the start-up module of each test run, and value_witness.py holds the same, so all three hide
    their tools alike."""
    import sys as interpreter

    monitoring = interpreter.monitoring
    real = {name: getattr(monitoring, name) for name in (
        "get_tool", "use_tool_id", "free_tool_id", "register_callback", "set_events", "get_events", "set_local_events",
        "get_local_events")}
    theirs: dict[str, Any] = {"name": None, "events": 0, "callbacks": {}, "local": {}}

    def claimed(tool_id: int) -> None:
        if theirs["name"] is None:
            raise ValueError(f"tool {tool_id} is not in use")

    def get_tool(tool_id: int) -> Any:
        return theirs["name"] if tool_id == tool else real["get_tool"](tool_id)

    def use_tool_id(tool_id: int, name: str) -> None:
        if tool_id != tool:
            return real["use_tool_id"](tool_id, name)
        if theirs["name"] is not None:
            raise ValueError(f"tool {tool_id} is already in use")
        theirs["name"] = name
        return None

    def free_tool_id(tool_id: int) -> None:
        if tool_id != tool:
            return real["free_tool_id"](tool_id)
        theirs.update(name=None, events=0, callbacks={}, local={})
        return None

    def register_callback(tool_id: int, event: int, func: Any) -> Any:
        if tool_id != tool:
            return real["register_callback"](tool_id, event, func)
        before = theirs["callbacks"].pop(event, None)
        if func is not None:
            theirs["callbacks"][event] = func
        return before

    def set_events(tool_id: int, event_set: int) -> None:
        if tool_id != tool:
            return real["set_events"](tool_id, event_set)
        claimed(tool_id)
        theirs["events"] = event_set
        return None

    def get_events(tool_id: int) -> int:
        return theirs["events"] if tool_id == tool else real["get_events"](tool_id)

    def set_local_events(tool_id: int, code: Any, event_set: int) -> None:
        if tool_id != tool:
            return real["set_local_events"](tool_id, code, event_set)
        claimed(tool_id)
        theirs["local"][code] = event_set
        return None

    def get_local_events(tool_id: int, code: Any) -> int:
        return theirs["local"].get(code, 0) if tool_id == tool else real["get_local_events"](tool_id, code)

    for name, function in (("get_tool", get_tool), ("use_tool_id", use_tool_id), ("free_tool_id", free_tool_id),
                           ("register_callback", register_callback), ("set_events", set_events),
                           ("get_events", get_events), ("set_local_events", set_local_events),
                           ("get_local_events", get_local_events)):
        setattr(monitoring, name, function)
    return real


#: Python's own monitoring functions, once this run's tool is hidden from the code under test.
_REAL: dict[str, Any] = {}


def _emit(*event: Any) -> None:
    try:
        _pipe.write(_DUMPS([_mark, *event]) + "\n")
    except (OSError, ValueError, TypeError, AttributeError):
        pass  # a pipe the run cannot write to leaves the parent with less evidence, never a different verdict
_counts: dict[tuple[str, str, str], int] = {}
_open: dict[int, tuple[str, str, int]] = {}


def _text(value: Any) -> str | None:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a value that cannot be shown is a value that cannot be compared
        return None
    return None if " at 0x" in text else _SHA256(text.encode("utf-8", "replace")).hexdigest()[:16]


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
    if inspect.ismodule(value):  # a module is made again by importing it by name
        return ("__module__", value.__name__)
    if isinstance(value, memoryview):  # a view is made again over the bytes it shows
        return ("__buffer__", value.tobytes())
    if depth > _DEEPEST:
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
            if "<" not in type(made).__qualname__:  # the class it was called on, so a base class swapped is seen
                call["of"] = [type(made).__module__, type(made).__qualname__]
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
    digest = _SHA256(f"{before}{outcome}".encode()).hexdigest()[:16]
    _digests[every] = f"raised {digest}" if before.startswith("raised ") and outcome.startswith("raised ") else digest
    _emit("every", every, _digests[every])
    if count <= _KEPT:
        _emit("value", key, outcome)


def _others(value: Any) -> list[Any]:
    """Plain values beside ``value``, of its own kind: the other truth value, a number either side and across zero, a
    text with a space added and an empty one."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [other for other in dict.fromkeys((value + 1, value - 1, -value - 1, 0)) if other != value]
    if isinstance(value, float):
        return [other for other in dict.fromkeys((value + 1.0, -value, 0.0)) if other != value]
    if isinstance(value, (str, bytes)):
        empty = type(value)()
        return [other for other in dict.fromkeys((value + type(value)(" " if isinstance(value, str) else b" "), empty))
                if other != value]
    return []


def _variations(arguments: dict[str, Any]) -> list[tuple[str, dict[str, Any], tuple[Any, str, Any, Any] | None]]:
    """Each way the call is made again: one plain argument moved, or one plain attribute of an object it was given
    (``req.value``, ``self.limit``) moved for the call and put back after it."""
    out: list[tuple[str, dict[str, Any], tuple[Any, str, Any, Any] | None]] = []
    for name, value in arguments.items():
        if isinstance(value, (bool, int, float, str, bytes)) and name not in ("self", "cls"):
            out += [(f"{name}={other!r}", {**arguments, name: other}, None) for other in _others(value)]
        elif hasattr(value, "__dict__") and not (inspect.ismodule(value) or inspect.isclass(value) or callable(value)):
            plain = [(attribute, held) for attribute, held in list(vars(value).items())
                     if isinstance(held, (bool, int, float, str, bytes)) and not attribute.startswith("__")]
            for attribute, held in plain[:6]:
                out += [(f"{name}.{attribute}={other!r}", arguments, (value, attribute, other, held))
                        for other in _others(held)[:2]]
    return out[:_VARIATIONS]


def _function_of(code: Any, globals_: dict[str, Any]) -> Any:
    """The function whose code ``code`` is, found where it was defined: at module level, or on its class."""
    parts = code.co_qualname.split(".")
    if "<locals>" in parts:
        return None
    holder: Any = globals_.get(parts[0])
    for part in parts[1:-1]:
        holder = getattr(holder, part, None)
    found = holder if len(parts) == 1 else _STATIC(holder, parts[-1], None) if holder is not None else None
    found = getattr(found, "__func__", found)  # a staticmethod or classmethod
    try:
        found = _UNWRAP(found) if callable(found) else None
    except ValueError:
        return None
    return found if getattr(found, "__code__", None) is code else None


def _too_long(*_: Any) -> None:
    raise TimeoutError


def _outcome(result: Any) -> str:
    if inspect.iscoroutine(result):
        result.close()
        return "a coroutine"
    if isinstance(result, abc.Iterator) and not hasattr(result, "fileno"):
        result = list(_ISLICE(result, 10_000))
    try:
        text = repr(result)
    except Exception:  # noqa: BLE001 - a value that cannot be shown is shown by its kind
        return f"a {type(result).__qualname__}"
    if " at 0x" in text:
        return f"a {type(result).__qualname__}"
    return text if len(text) <= 200 else f"{text[:120]} ... sha256 {_SHA256(text.encode('utf-8', 'replace')).hexdigest()[:16]}"


def _hold(code: Any, frame: Any) -> None:
    """Keep a call's own arguments, the objects themselves, when one of them cannot be written down as an input."""
    if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS | _ITERATES | inspect.CO_COROUTINE) \
            or "<" in code.co_qualname:
        return  # a generator or a coroutine made again says nothing until read, and is not
    names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
    arguments = {name: frame.f_locals[name] for name in names if name in frame.f_locals}
    if len(arguments) != len(names) or all(_literal(value) is not None for value in arguments.values()):
        return  # every argument can be made again in another process, where the check tries it already
    test = _test()
    if _varied_counts.get((test, code.co_qualname), 0) >= _VARIED_CALLS \
            or _varied_counts.get(("", code.co_qualname), 0) >= _VARIED_IN_ALL:
        return
    _held_calls[id(frame)] = (code, arguments, test, frame.f_globals)


def _vary(code: Any, arguments: dict[str, Any], test: str, globals_: dict[str, Any]) -> None:
    function = _function_of(code, globals_)
    if function is None or threading.current_thread() is not threading.main_thread():
        return  # only the main thread can bound a call's time
    count = _varied_counts.get((test, code.co_qualname), 0) + 1
    _varied_counts[(test, code.co_qualname)] = count
    _varied_counts[("", code.co_qualname)] = _varied_counts.get(("", code.co_qualname), 0) + 1
    by_place = code.co_varnames[:code.co_posonlyargcount]
    for label, given, attribute in _variations(arguments):
        _varying[0] = True
        before = signal.signal(signal.SIGALRM, _too_long)
        try:
            if attribute is not None:
                setattr(attribute[0], attribute[1], attribute[2])
            signal.setitimer(signal.ITIMER_REAL, _VARIED_SECONDS)
            outcome = _outcome(function(*(given[name] for name in by_place),
                                        **{name: value for name, value in given.items() if name not in by_place}))
        except BaseException as error:  # noqa: BLE001 - what a varied call raised is what it came to
            outcome = f"raised {type(error).__name__}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, before)
            if attribute is not None:
                try:
                    setattr(attribute[0], attribute[1], attribute[3])
                except Exception:  # noqa: BLE001 - an attribute that cannot be put back is left as it is
                    pass
            _varying[0] = False
        _emit("varied", f"{test}|{code.co_qualname}|{count}|{label}", outcome)


def _started(code: Any, _offset: int) -> Any:
    if _varying[0]:
        return None  # a call made here to vary another is no call of the run's
    if not _named(code):
        return sys.monitoring.DISABLE
    frame = sys._getframe(1)
    _open[id(frame)] = _key_of(code, frame)
    if _VARY:
        try:
            _hold(code, frame)
        except Exception:  # noqa: BLE001 - watching a call must never change what the call does
            pass
    return None


def _returned(code: Any, _offset: int, value: Any) -> Any:
    if _varying[0]:
        return None
    if not _named(code):
        return sys.monitoring.DISABLE
    frame = id(sys._getframe(1))
    key = _open.pop(frame, None)
    folded = _yields.pop(frame, None)
    _came_to(key, folded.hexdigest()[:16] if folded is not None else _text(value))
    held = _held_calls.pop(frame, None)
    if held is not None:
        _vary(*held)
    return None


def _yielded(code: Any, _offset: int, value: Any) -> Any:
    """A generator's items, folded into what it came to, however many there are."""
    if _varying[0]:
        return None
    if not _named(code):
        return sys.monitoring.DISABLE
    frame = id(sys._getframe(1))
    _yields.setdefault(frame, _SHA256()).update(str(_text(value)).encode())
    return None


def _unwound(code: Any, _offset: int, error: BaseException) -> None:
    if _varying[0]:
        return
    if _named(code):
        frame = id(sys._getframe(1))
        _came_to(_open.pop(frame, None), f"raised {type(error).__name__}")
        held = _held_calls.pop(frame, None)
        if held is not None:
            _vary(*held)


#: Unrelated objects this run holds when asked to (verify_change.BALLAST), so a value that counts the process differs.
_BALLAST: list[Any] = []


def pytest_sessionstart(session: Any) -> None:
    _BALLAST.extend([] for _ in range(int(CONFIG.get("ballast") or 0)))
    sys.monitoring.use_tool_id(_TOOL, "outside_bar_value_witness")
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_START, _started)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_RETURN, _returned)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_UNWIND, _unwound)
    sys.monitoring.register_callback(_TOOL, _EVENTS.PY_YIELD, _yielded)
    sys.monitoring.set_events(_TOOL, _EVENTS.PY_START | _EVENTS.PY_RETURN | _EVENTS.PY_UNWIND | _EVENTS.PY_YIELD)
    _REAL.update(_hide_the_tool(_TOOL))
    sys.modules.pop(__name__, None)
    _emit("process", _PROCESS())


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
    real = _REAL or {name: getattr(sys.monitoring, name) for name in (
        "get_tool", "get_events", "register_callback", "set_events", "free_tool_id")}
    whole = real["get_tool"](_TOOL) == "outside_bar_value_witness" and real["get_events"](_TOOL) != 0
    try:
        for event, watcher in _WATCHERS:
            whole = real["register_callback"](_TOOL, event, None) is watcher and whole
        real["set_events"](_TOOL, 0)
        real["free_tool_id"](_TOOL)
    except ValueError:
        whole = False
    _emit("whole", whole)
    _emit("done")
