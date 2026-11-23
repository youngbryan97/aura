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
import sys
import tempfile
import time
import types
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


def _plain_names(place: Any) -> list[tuple[str, Any]]:
    """What a place holds that a call can change: everything but its functions, classes and modules and the names Python
    gives every module and class, so an object that can be called and holds state counts, and so does ``__ledger__``."""
    return [(name, value) for name, value in list(vars(place).items())
            if name not in _PYTHONS_OWN and not inspect.isroutine(value)
            and not inspect.isclass(value) and not inspect.ismodule(value)
            and not isinstance(value, (staticmethod, classmethod, property))]


def _held_around(path: str, made: Any) -> str:
    """What the module, its classes and the object's class hold, so a call that changes them changes what it came to."""
    return _shown({f"{getattr(place, '__qualname__', getattr(place, '__name__', ''))}.{name}": _shown(value)
                   for place in _places_around(path, made) for name, value in _plain_names(place)})


def _kept_state(path: str) -> list[tuple[Any, str, Any]]:
    """A copy of what the module and its classes hold just after import, to start every call from."""
    kept = []
    for place in _places_around(path, None):
        for name, value in _plain_names(place):
            try:
                kept.append((place, name, _DEEPCOPY(value)))
            except Exception:  # noqa: BLE001 - what cannot be copied is left as each call leaves it
                continue
    return kept


def _restore(kept: list[tuple[Any, str, Any]]) -> None:
    for place, name, value in kept:
        current = vars(place).get(name)
        fresh = _DEEPCOPY(value)
        try:
            if type(current) is dict and type(fresh) is dict:
                current.clear()
                current.update(fresh)
            elif type(current) is list and type(fresh) is list:
                current[:] = fresh
            elif type(current) is set and type(fresh) is set:
                current.clear()
                current.update(fresh)
            else:
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
    touched: list[str] = []
    watching_writes: list[bool] = [False]

    def named(path: Any) -> str:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return "a file it held open"
        where = real(os.fsdecode(path))
        return relative(where, home) if where.startswith(home) else "a file outside the checkout"

    def audited(event: str, args: tuple[Any, ...]) -> None:
        if not watching_writes[0]:
            return
        if event == "sqlite3.connect" and args and isinstance(args[0], (str, bytes, os.PathLike)) \
                and os.fsdecode(args[0]) not in (":memory:", ""):
            wrote.append(real(os.fsdecode(args[0])))  # a database is a file a call writes, through C
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
        for where in sorted(set(wrote)):
            try:
                with open(where, "rb") as held:
                    digest = _SHA256(held.read(_READ_BACK)).hexdigest()[:16]
            except OSError:
                digest = "gone"
            shown = relative(where, home) if where.startswith(home) else "a file outside the checkout"
            out.append(f"{shown}: {digest}")
        return out

    # Both streams go, for the whole run, to files with names in this run's own temporary folder, opened before any code
    # under test is imported: a file descriptor that changes kind or inode during a call, or a file with no name, would
    # tell the code it is being watched. Each call's share is what was added while it ran.
    folder = os.environ.get("TMPDIR") or home
    held_streams = []
    for number, name in ((1, "out.log"), (2, "err.log")):
        where = os.path.join(folder, name)
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
    questions = _LOADS(sys.stdin)
    # Unrelated objects, held for the whole run when the check asks (verify_change.BALLAST): a value that moves with
    # them counts the process, and the check leaves it uncompared.
    ballast = [[] for _ in range(int(questions.pop("ballast", 0) or 0))]
    pipe = os.fdopen(questions.pop("pipe"), "w", encoding="utf-8", buffering=1)
    mark = questions.pop("mark")
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
    kept: dict[str, list[tuple[Any, str, Any]]] = {}

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
        touched.clear()
        watching_writes[0] = True
        monitoring.set_events(tool, monitoring.events.EXCEPTION_HANDLED | monitoring.events.LINE
                              | monitoring.events.BRANCH)
        try:
            outcome = _shown(_finished(target(*(arguments[name] for name in placed), *star, **named_here, **keywords)))
        except TimeoutError:
            return "took too long"
        except BaseException as error:  # noqa: BLE001 - what it raised is what the call came to, an exit included
            outcome = _refusal(error, arguments, home)
        finally:
            monitoring.set_events(tool, 0)
            watching_writes[0] = False
            said, warned = released(sizes)
        outcome = outcome if not said else f"{outcome}; it printed {said[:200]!r}"
        outcome = outcome if not warned else f"{outcome}; it wrote to stderr {warned[:200]!r}"
        written = what_it_wrote()
        outcome = outcome if not written else f"{outcome}; it wrote {written[:5]}"
        outcome = outcome if not touched else f"{outcome}; it {', '.join(sorted(set(touched))[:5])}"
        after, held_after = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
        outcome = outcome if after == before else f"{outcome}; its arguments became {after[:200]}"
        outcome = outcome if held_after == held_before else f"{outcome}; the object became {held_after[:200]}"
        around_after = _held_around(path, made)
        return outcome if around_after == around_before else f"{outcome}; what is around it became {around_after[:200]}"

    # What this file's own names were before any code under test ran: a call that rebinds one (``_shown`` through
    # ``sys.modules["__main__"]``) has changed the check, and the run says so and stops.
    own = {name: value for name, value in globals().items() if not name.startswith("__")}
    for name, calls in questions.pop("inputs").items():
        path, qualname = name.split("::")
        for call in calls:
            caught.clear()
            ran.clear()
            started = clock()
            alarm(_CALL_SECONDS)
            try:
                outcome = came_to(path, qualname, call)
            except BaseException as error:  # noqa: BLE001 - a function that cannot be reached has no value to compare
                outcome = "took too long" if isinstance(error, TimeoutError) else f"unavailable: {type(error).__name__}"
            finally:
                alarm(0)
            watching = [monitoring.register_callback(tool, event, None) for event in
                        (monitoring.events.EXCEPTION_HANDLED, monitoring.events.LINE, monitoring.events.BRANCH)]
            for event, callback in zip((monitoring.events.EXCEPTION_HANDLED, monitoring.events.LINE,
                                        monitoring.events.BRANCH), (handled, lined, branched)):
                monitoring.register_callback(tool, event, callback)
            if any(globals().get(key) is not value for key, value in own.items()) or \
                    watching != [handled, lined, branched] or monitoring.get_tool(tool) != "debugger":
                pipe.write(_DUMPS([mark, "", "", "tampered", 0.0, [], []]) + "\n")
                return 1
            pipe.write(_DUMPS([mark, name, _DUMPS(call, sort_keys=True), _short(outcome), clock() - started,
                               sorted(set(caught)), sorted(ran)]) + "\n")
    pipe.write(_DUMPS([mark, "", "", "done", 0.0, [], []]) + "\n")
    del ballast
    return 0


if __name__ == "__main__":
    sys.exit(main())
