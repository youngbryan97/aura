"""Call the changed functions on the given inputs in this checkout, and say what each call came to.

outside_bar/verify_change.py copies this next to each run under a plain name, hands it the inputs on stdin and reads the
answers from a pipe. The file the answers end up in belongs to the parent, which never imports the code under test.

Everything this needs from the standard library is taken before the code under test is imported, and kept inside one
function's closure, so code that rebinds ``time.perf_counter`` or a name on ``__main__`` reaches nothing used here.
Each call starts from the module state the import left, and the run ends by saying it ended, so a process that dies
part way through cannot pass for one that answered.
"""
from __future__ import annotations

import ast
import asyncio
import base64
import builtins
import copy
import dis
import hashlib
import importlib
import inspect
import io
import itertools
import json
import marshal
import os
import signal
import sqlite3
import sys
import tempfile
import time
import types
import urllib.parse
import weakref
from collections import abc
from typing import Any

# Taken now, before any code under test is imported, so rebinding ``json.dumps`` or ``copy.deepcopy`` inside those
# modules changes nothing this file calls.
_DUMPS, _LOADS, _DEEPCOPY, _SHA256 = json.dumps, json.load, copy.deepcopy, hashlib.sha256
_LITERAL, _INSTRUCTIONS, _SIGNATURE, _ISLICE = ast.literal_eval, dis.get_instructions, inspect.signature, itertools.islice
_IMPORT, _REALPATH = importlib.import_module, os.path.realpath
_TEMPORARY, _DUP, _DUP2, _CLOSE = tempfile.TemporaryFile, os.dup, os.dup2, os.close
#: Python's own filesystem audit events other than open: what a call removes, renames, empties or makes. Read only, so
#: code under test that reaches it cannot empty it in place.
_FILE_EVENTS = types.MappingProxyType({
    "os.remove": "removed", "os.rename": "renamed", "os.rmdir": "removed the folder", "shutil.rmtree": "removed the tree",
    "os.truncate": "truncated", "os.mkdir": "made the folder", "os.symlink": "linked", "os.link": "linked",
    "os.chmod": "changed the mode of"})
#: How much of what a call printed or wrote is compared, in bytes: enough to tell two outputs apart.
_READ_BACK = 1 << 20
#: The files a call wrote that its outcome names, at most.
_FILES_SHOWN = 8

#: How long one call may take, in seconds. A call that takes longer says nothing about the value it would have returned.
_CALL_SECONDS = 2
#: Values read from a generator: enough to compare, and an endless one still ends.
_ITEMS = 10_000
#: How deep an object's parts are followed when it is shown.
_DEEPEST = 4
#: The longest outcome sent as text. A longer one is sent as its start and a digest of the whole, which compare the same.
_LONGEST_OUTCOME = 2000


def _too_long(*_: Any) -> None:
    raise TimeoutError


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


def _named_thing(module: str, qualname: str) -> Any:
    holder: Any = _IMPORT(module)
    for part in qualname.split("."):
        holder = getattr(holder, part)
    return holder


def _given_state(made: Any, state: Any) -> None:
    """State put back the way pickle puts it back: a ``__setstate__`` of its own, or a dict, or a dict and slots."""
    setter = getattr(type(made), "__setstate__", None)
    if setter is not None and setter is not getattr(object, "__setstate__", None):
        made.__setstate__(state)
        return
    held, slots = state if isinstance(state, tuple) and len(state) == 2 else (state, None)
    for name, item in (held or {}).items():
        object.__setattr__(made, name, item)
    for name, item in (slots or {}).items():
        object.__setattr__(made, name, item)


def _rebuilt(value: Any) -> Any:
    """A value as the witness wrote it down, with each object made again the way its class says it is made."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__module__":
        return _IMPORT(value[1])
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__buffer__":
        return memoryview(value[1])
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__lock__":
        import threading

        return getattr(threading, value[1])()
    if isinstance(value, tuple) and len(value) == 3 and value[0] == "__callable__":
        return _named_thing(value[1], value[2])
    if isinstance(value, tuple) and len(value) == 4 and value[0] == "__object__":
        holder = _named_thing(value[1], value[2])
        made = holder.__new__(holder)
        for name, item in value[3].items():
            object.__setattr__(made, name, _rebuilt(item))
        return made
    if isinstance(value, tuple) and len(value) == 7 and value[0] == "__reduced__":
        made = _named_thing(value[1], value[2])(*_rebuilt(list(value[3])))
        if value[4] is not None:
            _given_state(made, _rebuilt(value[4]))
        for item in _rebuilt(list(value[5] or [])):
            made.append(item)
        for key, item in _rebuilt(list(value[6] or [])):
            made[key] = item
        return made
    if isinstance(value, tuple) and len(value) == 5 and value[0] == "__code__":
        try:
            room = vars(_IMPORT(value[2]))
        except Exception:  # noqa: BLE001 - a callable from a module this checkout lacks still runs on builtins
            room = {"__builtins__": builtins}
        cells = tuple(types.CellType(_rebuilt(item)) for item in value[4]) or None
        defaults = tuple(_rebuilt(item) for item in value[3]) or None
        return types.FunctionType(marshal.loads(base64.b64decode(value[1])), room, None, defaults, cells)
    if type(value) in (list, tuple, set):
        return type(value)(_rebuilt(item) for item in value)
    if type(value) is dict:
        return {_rebuilt(key): _rebuilt(item) for key, item in value.items()}
    return value


def _state_of(made: Any) -> dict[str, Any]:
    state = dict(getattr(made, "__dict__", None) or {})
    for klass in type(made).__mro__:
        slots = getattr(klass, "__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in ("__dict__", "__weakref__") and hasattr(made, name):
                state[name] = getattr(made, name)
    if isinstance(made, dict) and type(made) is not dict:
        state["<items>"] = dict(made)
    elif isinstance(made, (list, set)) and type(made) not in (list, set):
        state["<items>"] = list(made)
    return state


def _shown(value: Any, depth: int = 0) -> str:
    """A value as text two runs can compare: an object without a text of its own is shown by what it holds, and one that
    holds nothing Python can see (a C type such as ``io.StringIO``) by what it gives pickle to be made again."""
    text = repr(value)
    if " at 0x" not in text or depth > _DEEPEST:
        return text
    if isinstance(value, weakref.ref):  # a weak reference is shown by what it still refers to
        return f"weakref({_shown(value(), depth + 1)})"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}({[_shown(item, depth + 1) for item in value]})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_shown(key, depth + 1)}: {_shown(item, depth + 1)}" for key, item in value.items()) + "}"
    state = _state_of(value)
    if not state:
        try:
            state = {"made again from": value.__reduce_ex__(4)[1:]}
        except Exception:  # noqa: BLE001 - an object that will not say what it holds is read as a buffer, if it is one
            try:
                state = {"bytes": memoryview(value).tobytes()[:_READ_BACK]}
            except TypeError:
                state = {}
    return f"{type(value).__qualname__}({_shown(state, depth + 1)})"


def _found(path: str, qualname: str, call: dict[str, Any]) -> tuple[Any, Any]:
    """The function to call, and the object it was called on when it is a method."""
    module = _IMPORT(path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
    parts = qualname.split(".")
    holder: Any = module
    for part in parts[:-1]:
        holder = getattr(holder, part)
    made = None
    if "cls" in call:  # a class method is called on the class it was called on, a subclass included
        holder = _named_thing(*call["cls"])
    elif len(parts) > 1 and "of" in call:  # and a method on the object's own class, so it resolves as each checkout has it
        holder = _named_thing(*call["of"])
    if len(parts) > 1 and "self_whole" in call:
        made = _rebuilt(_LITERAL(call["self_whole"]))
    elif len(parts) > 1 and "self" in call:
        made = holder.__new__(holder)
        for key, text in call["self"].items():
            object.__setattr__(made, key, _rebuilt(_LITERAL(text)))
    found = inspect.getattr_static(holder, parts[-1], None)
    if made is not None and isinstance(found, property):
        if call["args"]:
            return (lambda *place, **name: found.fset(made, *place, *name.values())), made
        return (lambda: found.fget(made)), made
    return getattr(made if made is not None else holder, parts[-1]), made


def _ours(module: Any) -> bool:
    return _REALPATH(getattr(module, "__file__", None) or "/").startswith(_REALPATH(os.getcwd()))


def _places_around(path: str, made: Any) -> list[Any]:
    """What a call can change besides its arguments: the module a function lives in and every module of this checkout it
    reaches through the names it holds, at any depth (``mod`` calls ``mid``, which writes to ``store``), the classes they
    define, the functions among them that hold attributes, and the class of the object a method is called on."""
    module = sys.modules.get(path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
    places = [type(made)] if made is not None else []
    if module is None:
        return places
    modules, waiting = [module], [module]
    while waiting:
        here = waiting.pop(0)
        values = list(vars(here).values())
        reached = [value for value in values if inspect.ismodule(value)]
        reached += [sys.modules.get(getattr(value, "__module__", "") or "") for value in values
                    if inspect.isroutine(value) or inspect.isclass(value)]
        for value in reached:
            if value is not None and value not in modules and _ours(value):
                modules.append(value)
                waiting.append(value)
    for held in modules:
        places.append(held)
        for value in list(vars(held).values()):
            if inspect.isclass(value) and value.__module__ == held.__name__ and value not in places:
                places.append(value)
            elif inspect.isfunction(value) and value.__module__ == held.__name__ and vars(value) and value not in places:
                places.append(value)  # record.entries = [] kept on the function itself
    return places


#: The names Python itself gives every module and class, which say nothing about what a call did.
_PYTHONS_OWN = frozenset({*vars(types.ModuleType("_")), *vars(type("_", (), {})), "__file__", "__cached__", "__builtins__",
                          "__path__", "__all__", "__annotations__", "__slots__", "__match_args__", "__orig_bases__",
                          "__parameters__", "__static_attributes__", "__firstlineno__", "__dataclass_fields__",
                          "__dataclass_params__", "__abstractmethods__", "_abc_impl", "__version__"})


def _cells_around(path: str, made: Any) -> list[tuple[str, Any]]:
    """The closure cells the functions of the watched modules hold, by name, and the object a module-level bound method
    is bound to: ``_push, read = _make()`` keeps its list in a cell no module name reaches, and ``record = [].append``
    keeps it on the method. A function held in a cell is followed into its own cells."""
    waiting = [(f"{getattr(place, '__name__', '')}.{name}", value) for place in _places_around(path, made)
               if inspect.ismodule(place) for name, value in list(vars(place).items())
               if (inspect.isfunction(value) and value.__module__ == place.__name__)
               or isinstance(value, types.BuiltinMethodType) and not inspect.ismodule(getattr(value, "__self__", None))]
    out: list[tuple[str, Any]] = []
    seen: set[int] = set()
    while waiting:
        label, function = waiting.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        if isinstance(function, types.BuiltinMethodType):
            out.append((f"{label}.__self__", function))
            continue
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            try:
                held = cell.cell_contents
            except ValueError:  # a cell not yet filled
                continue
            if inspect.isfunction(held):
                waiting.append((f"{label}.<{name}>", held))
            elif not (inspect.isroutine(held) or inspect.isclass(held) or inspect.ismodule(held)):
                out.append((f"{label}.<{name}>", cell))
    return out


def _in_cell(holder: Any) -> Any:
    return holder.__self__ if isinstance(holder, types.BuiltinMethodType) else holder.cell_contents


def _plain_names(place: Any) -> list[tuple[str, Any]]:
    """What a place holds that a call can change: everything but its functions, classes and modules and the names Python
    gives every module and class, so an object that can be called and holds state counts, and so does ``__ledger__``."""
    return [(name, value) for name, value in list(vars(place).items())
            if name not in _PYTHONS_OWN and not inspect.isroutine(value)
            and not inspect.isclass(value) and not inspect.ismodule(value)
            and not isinstance(value, (staticmethod, classmethod, property))]


def _held_around(path: str, made: Any) -> str:
    """What the module, its classes and the object's class hold, so a call that changes them changes what it came to."""
    held = {f"{getattr(place, '__qualname__', getattr(place, '__name__', ''))}.{name}": _shown(value)
            for place in _places_around(path, made) for name, value in _plain_names(place)}
    held.update({label: _shown(_in_cell(cell)) for label, cell in _cells_around(path, made)})
    return _shown(held)


def _kept_state(path: str) -> list[tuple[Any, str, Any]]:
    """A copy of what the module and its classes hold just after import, to start every call from."""
    kept = []
    for place in _places_around(path, None):
        for name, value in _plain_names(place):
            try:
                kept.append((place, name, _DEEPCOPY(value)))
            except Exception:  # noqa: BLE001 - what cannot be copied is left as each call leaves it
                continue
    for _, cell in _cells_around(path, None):  # a cell, or the object a bound method holds, is put back in place
        try:
            kept.append((cell, None, _DEEPCOPY(_in_cell(cell))))
        except Exception:  # noqa: BLE001
            continue
    return kept


def _restore(kept: list[tuple[Any, str, Any]]) -> None:
    for place, name, value in kept:
        current = vars(place).get(name) if name is not None else _in_cell(place)
        fresh = _DEEPCOPY(value)
        try:
            if name is None and not isinstance(place, types.BuiltinMethodType) and type(current) not in (dict, list, set):
                place.cell_contents = fresh
            elif type(current) is dict and type(fresh) is dict:
                current.clear()
                current.update(fresh)
            elif type(current) is list and type(fresh) is list:
                current[:] = fresh
            elif type(current) is set and type(fresh) is set:
                current.clear()
                current.update(fresh)
            elif name is not None:
                setattr(place, name, fresh)
        except Exception:  # noqa: BLE001 - a name that cannot be set back is left as it is
            continue


async def _gathered(stream: Any) -> list[Any]:
    out: list[Any] = []
    async for item in stream:
        out.append(item)
        if len(out) >= _ITEMS:
            break
    return out


def _finished(result: Any) -> Any:
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    if inspect.isasyncgen(result):
        return asyncio.run(_gathered(result))
    if isinstance(result, abc.Iterator) and not isinstance(result, io.IOBase):  # generators, map, zip, list_iterator
        return list(_ISLICE(result, _ITEMS))
    return result


def _by_the_code(error: BaseException, home: str) -> bool:
    """Whether a ``raise`` statement in this checkout's own code raised it, as opposed to an operation that failed."""
    trace = error.__traceback__
    while trace is not None and trace.tb_next is not None:
        trace = trace.tb_next
    if trace is None or not os.path.realpath(trace.tb_frame.f_code.co_filename).startswith(home):
        return False
    return any(item.offset == trace.tb_lasti and item.opname in ("RAISE_VARARGS", "RERAISE")
               for item in _INSTRUCTIONS(trace.tb_frame.f_code))


def _refusal(error: BaseException, arguments: dict[str, Any], home: str) -> str:
    """The kind of refusal, with the input taken out of its words, so one refusal given two inputs reads as one."""
    words = str(error)
    forms = {form for value in arguments.values() for form in (repr(value), str(value)) if form}
    for form in sorted(forms, key=len, reverse=True):  # the longest first: 'milk' before milk
        words = words.replace(form, "<input>")
    by = " by the code" if _by_the_code(error, home) else ""
    return f"raised {type(error).__name__}{by}: {words[:120]}"


def _invariants_broken(names: frozenset[str]) -> list[str]:
    """The invariants named in ``names`` (core/verify/invariants.py, written next to what they protect) that report a
    violation now, or raise, by name: what a call left behind that its own module says must never be. A message can
    carry a time or a count, so only the name is kept, which two runs share."""
    registry = getattr(sys.modules.get("core.verify.invariants"), "get_registry", None)
    if not names or registry is None:
        return []
    broken = []
    for spec in registry().specs():
        if spec.name not in names or not getattr(spec, "observational", True):
            continue
        try:
            if any(True for _ in spec.check()):
                broken.append(spec.name)
        except Exception:  # noqa: BLE001 - a check that raises is itself a violation, as the verifier counts it
            broken.append(f"{spec.name} raised")
    return sorted(broken)


def _short(outcome: str) -> str:
    if len(outcome) <= _LONGEST_OUTCOME:
        return outcome
    return f"{outcome[:300]} ... sha256 {_SHA256(outcome.encode('utf-8', 'replace')).hexdigest()[:24]}"


def main() -> int:
    clock, alarm, home = time.perf_counter, signal.alarm, os.path.realpath(os.getcwd())
    monitoring, relative, real = sys.monitoring, os.path.relpath, os.path.realpath
    writing_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    # Side effects a call leaves outside its value: what it printed to either stream, the files it wrote and what they
    # hold after it, and the files it removed, renamed or emptied. Audited from here, which no code under test can
    # remove, and only while a call runs. The streams are caught at their file descriptors, so ``sys.stdout`` is the
    # object it always is and says nothing about this check.
    wrote: list[str] = []
    #: Files a call may have written, named only when they are there after it: the sidecar files of a database, and a
    #: database's name read both as a path and as a URI, since its connect event does not say which it was.
    may_have_written: list[str] = []
    touched: list[str] = []
    watching_writes: list[bool] = [False]

    def named(path: Any) -> str:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return "a file it held open"
        where = real(os.fsdecode(path))
        return relative(where, home) if where.startswith(home) else "a file outside the checkout"

    def database(name: Any) -> None:
        """A database a call opens is a file it writes, through C: as the path it names, and as the file a ``file:``
        URI names (``file:records.db?mode=rwc``), each with the journal and write-ahead log beside it."""
        if not isinstance(name, (str, bytes, os.PathLike)):
            return
        text = os.fsdecode(name)
        if text in (":memory:", ""):
            return
        paths = [text]
        if text.startswith("file:"):
            where, _, query = text[len("file:"):].partition("?")
            where = urllib.parse.unquote(where.split("#", 1)[0])
            if where.startswith("//"):  # file://host/path: the path after the host
                where = "/" + where[2:].partition("/")[2]
            if "mode=memory" not in query and where not in ("", ":memory:", "/"):
                paths.append(where)
        for path in paths:
            may_have_written.append(real(path))
            may_have_written.extend(real(path) + side for side in ("-journal", "-wal"))

    def authorized(action: int, first: Any, *_: Any) -> int:
        if action == sqlite3.SQLITE_ATTACH and watching_writes[0]:
            database(first)  # ATTACH DATABASE 'other.db': a second file the connection writes
        return sqlite3.SQLITE_OK

    def audited(event: str, args: tuple[Any, ...]) -> None:
        if not watching_writes[0]:
            return
        if event == "sqlite3.connect" and args:
            database(args[0])
        elif event == "sqlite3.connect/handle" and args:
            try:
                args[0].set_authorizer(authorized)
            except Exception:  # noqa: BLE001 - a connection that takes no authorizer is watched by its file alone
                pass
        elif event == "open" and len(args) >= 3 and isinstance(args[0], (str, bytes)):
            path, mode, flags = args[0], args[1], args[2]
            if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
                    isinstance(flags, int) and flags & writing_flags):
                wrote.append(real(os.fsdecode(path)))
        elif event in _FILE_EVENTS and args:
            touched.append(f"{_FILE_EVENTS[event]} {named(args[0])}"
                           + (f" to {named(args[1])}" if event == "os.rename" and len(args) > 1 else ""))

    def what_it_wrote() -> list[str]:
        out = []
        maybe = {where for where in may_have_written if os.path.isfile(where)} - set(wrote)
        for where in sorted({*wrote, *maybe}):
            try:
                with open(where, "rb") as held:
                    digest = _SHA256(held.read(_READ_BACK)).hexdigest()[:16]
            except OSError:
                digest = "gone"
            shown = relative(where, home) if where.startswith(home) else "a file outside the checkout"
            out.append(f"{shown}: {digest}")
        return out

    questions = _LOADS(sys.stdin)
    # Both streams go, for the whole run, to named files in a folder the parent made for them and named to no one else,
    # opened before any code under test is imported: a file descriptor that changes kind or inode during a call, or a
    # file with no name, would tell the code it is watched, and so would a file in its own temporary folder whose
    # name it could guess (round nine of the red team compared fd 1's inode with $TMPDIR/out.log). Each call's share is
    # what was added while it ran.
    folder = questions.pop("streams", None) or os.environ.get("TMPDIR") or home
    held_streams = []
    for number in (1, 2):
        where = os.path.join(folder, os.urandom(6).hex())
        descriptor = os.open(where, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        _DUP2(descriptor, number)
        _CLOSE(descriptor)
        held_streams.append(where)

    def caught_streams() -> tuple[int, int]:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - a stream the code replaced is flushed by whoever replaced it
                pass
        return os.fstat(1).st_size, os.fstat(2).st_size

    def released(sizes: tuple[int, int]) -> tuple[str, str]:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        texts = []
        for where, start in zip(held_streams, sizes):
            with open(where, "rb") as held:
                held.seek(start)
                texts.append(held.read(_READ_BACK).decode("utf-8", "replace"))
        return texts[0], texts[1]

    sys.addaudithook(audited)
    sys.path.insert(0, os.getcwd())
    signal.signal(signal.SIGALRM, _too_long)
    # Unrelated objects, held for the whole run when the check asks (verify_change.BALLAST): a value that moves with
    # them counts the process, and the check leaves it uncompared.
    ballast = [[] for _ in range(int(questions.pop("ballast", 0) or 0))]
    pipe = os.fdopen(questions.pop("pipe"), "w", encoding="utf-8", buffering=1)
    mark = questions.pop("mark")
    invariants = frozenset(questions.pop("invariants", None) or ())
    caught: list[str] = []
    ran: set[str] = set()
    target_code: list[tuple[str, str]] = [("", "")]
    tool = monitoring.DEBUGGER_ID

    def handled(code: Any, _offset: int, error: BaseException) -> Any:
        if os.path.realpath(code.co_filename).startswith(home):
            caught.append(type(error).__name__)
        return None

    def lined(code: Any, line: int) -> Any:
        """The lines of the function asked about that a call ran: which way through it the answer came."""
        qualname, where = target_code[0]
        if code.co_qualname == qualname and real(code.co_filename) == where:
            ran.add(f"line {line}")
            return None
        return monitoring.DISABLE

    def branched(code: Any, offset: int, destination: int) -> Any:
        """Which way each branch of the function went, inside a line too: ``a if p else b`` written on one line has one
        line and two ways through it."""
        qualname, where = target_code[0]
        if code.co_qualname == qualname and real(code.co_filename) == where:
            ran.add(f"branch {offset} to {destination}")
            return None
        return monitoring.DISABLE

    monitoring.use_tool_id(tool, "debugger")
    monitoring.register_callback(tool, monitoring.events.EXCEPTION_HANDLED, handled)
    monitoring.register_callback(tool, monitoring.events.LINE, lined)
    monitoring.register_callback(tool, monitoring.events.BRANCH, branched)
    own_tools = _hide_the_tool(tool)
    set_events, register_callback, get_tool = (own_tools["set_events"], own_tools["register_callback"],
                                                own_tools["get_tool"])
    kept: dict[str, list[tuple[Any, str, Any]]] = {}
    read_as: list[str] = ["plain"]

    def came_to(path: str, qualname: str, call: dict[str, Any]) -> str:
        arguments = {name: _rebuilt(_LITERAL(text)) for name, text in call["args"].items()}
        star = _rebuilt(_LITERAL(call.get("star", "[]")))
        keywords = _rebuilt(_LITERAL(call.get("keywords", "{}")))
        target, made = _found(path, qualname, call)
        if path not in kept:
            kept[path] = _kept_state(path)
        _restore(kept[path])
        by_place = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) if star \
            else (inspect.Parameter.POSITIONAL_ONLY,)
        placed = [item.name for item in _SIGNATURE(target).parameters.values()
                  if item.kind in by_place and item.name in arguments]
        named_here = {name: value for name, value in arguments.items() if name not in placed}
        before, held_before = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
        around_before = _held_around(path, made)
        target_code[0] = (qualname, real(os.path.join(home, path)))
        monitoring.restart_events()
        sizes = caught_streams()
        wrote.clear()
        may_have_written.clear()
        touched.clear()
        watching_writes[0] = True
        set_events(tool, monitoring.events.EXCEPTION_HANDLED | monitoring.events.LINE | monitoring.events.BRANCH)
        try:
            result = target(*(arguments[name] for name in placed), *star, **named_here, **keywords)
            # How the value was read, for a test written of it later: awaited, or read out of an iterator, which a
            # function returns without a yield of its own as often as with one (``return (x * 2 for x in items)``).
            read_as[0] = "coroutine" if inspect.iscoroutine(result) else "async iterator" if inspect.isasyncgen(result) \
                else "iterator" if isinstance(result, abc.Iterator) and not isinstance(result, io.IOBase) else "plain"
            outcome = _shown(_finished(result))
        except TimeoutError:
            return "took too long"
        except BaseException as error:  # noqa: BLE001 - what it raised is what the call came to, an exit included
            outcome = _refusal(error, arguments, home)
        finally:
            set_events(tool, 0)
            watching_writes[0] = False
            said, warned = released(sizes)
        outcome = outcome if not said else f"{outcome}; it printed {said[:200]!r}"
        outcome = outcome if not warned else f"{outcome}; it wrote to stderr {warned[:200]!r}"
        written = what_it_wrote()
        outcome = outcome if not written else f"{outcome}; it wrote {written[:_FILES_SHOWN]}"
        outcome = outcome if not touched else f"{outcome}; it {', '.join(sorted(set(touched))[:5])}"
        after, held_after = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
        outcome = outcome if after == before else f"{outcome}; its arguments became {after[:200]}"
        outcome = outcome if held_after == held_before else f"{outcome}; the object became {held_after[:200]}"
        around_after = _held_around(path, made)
        outcome = outcome if around_after == around_before else f"{outcome}; what is around it became {around_after[:200]}"
        broken = _invariants_broken(invariants)
        return outcome if not broken else f"{outcome}; its invariants say otherwise: {broken[:5]}"

    # What this file's own names were before any code under test ran: a call that rebinds one (``_shown`` through
    # ``sys.modules["__main__"]``) has changed the check, and the run says so and stops.
    own = {name: value for name, value in globals().items() if not name.startswith("__")}
    for name, calls in questions.pop("inputs").items():
        path, qualname = name.split("::")
        for call in calls:
            caught.clear()
            ran.clear()
            read_as[0] = "plain"
            started = clock()
            alarm(_CALL_SECONDS)
            try:
                outcome = came_to(path, qualname, call)
            except BaseException as error:  # noqa: BLE001 - a function that cannot be reached has no value to compare
                outcome = "took too long" if isinstance(error, TimeoutError) else f"unavailable: {type(error).__name__}"
            finally:
                alarm(0)
            watching = [register_callback(tool, event, None) for event in
                        (monitoring.events.EXCEPTION_HANDLED, monitoring.events.LINE, monitoring.events.BRANCH)]
            for event, callback in zip((monitoring.events.EXCEPTION_HANDLED, monitoring.events.LINE,
                                        monitoring.events.BRANCH), (handled, lined, branched)):
                register_callback(tool, event, callback)
            if any(globals().get(key) is not value for key, value in own.items()) or \
                    watching != [handled, lined, branched] or get_tool(tool) != "debugger":
                pipe.write(_DUMPS([mark, "", "", "tampered", 0.0, [], [], ""]) + "\n")
                return 1
            pipe.write(_DUMPS([mark, name, _DUMPS(call, sort_keys=True), _short(outcome), clock() - started,
                               sorted(set(caught)), sorted(ran), read_as[0]]) + "\n")
    pipe.write(_DUMPS([mark, "", "", "done", 0.0, [], [], ""]) + "\n")
    del ballast
    return 0


if __name__ == "__main__":
    sys.exit(main())
