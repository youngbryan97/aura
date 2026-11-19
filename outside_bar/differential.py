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
import time
import types
from collections import abc
from typing import Any

# Taken now, before any code under test is imported, so rebinding ``json.dumps`` or ``copy.deepcopy`` inside those
# modules changes nothing this file calls.
_DUMPS, _LOADS, _DEEPCOPY, _SHA256 = json.dumps, json.load, copy.deepcopy, hashlib.sha256
_LITERAL, _INSTRUCTIONS, _SIGNATURE, _ISLICE = ast.literal_eval, dis.get_instructions, inspect.signature, itertools.islice
_IMPORT, _REALPATH = importlib.import_module, os.path.realpath

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
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}({[_shown(item, depth + 1) for item in value]})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_shown(key, depth + 1)}: {_shown(item, depth + 1)}" for key, item in value.items()) + "}"
    state = _state_of(value)
    if not state:
        try:
            state = {"made again from": value.__reduce_ex__(4)[1:]}
        except Exception:  # noqa: BLE001 - an object that will not say what it holds is shown by its kind alone
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
    """The module a function lives in, the modules of this checkout it holds by name, the classes they define, and the
    class of the object a method is called on: what a call can change besides its arguments."""
    module = sys.modules.get(path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
    places = [type(made)] if made is not None else []
    if module is None:
        return places
    modules = [module, *(value for value in list(vars(module).values())
                         if inspect.ismodule(value) and value is not module and _ours(value))]
    for held in modules:
        places.append(held)
        places += [value for value in list(vars(held).values())
                   if inspect.isclass(value) and value.__module__ == held.__name__ and value not in places]
    return places


def _plain_names(place: Any) -> list[tuple[str, Any]]:
    """What a place holds that a call can change: everything but its functions, classes and modules, so an object that
    can be called and also holds state still counts."""
    return [(name, value) for name, value in list(vars(place).items())
            if not (name.startswith("__") and name.endswith("__")) and not inspect.isroutine(value)
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
    monitoring, printing, relative, real = sys.monitoring, io.StringIO, os.path.relpath, os.path.realpath
    writing_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    # Side effects a call leaves outside its value: what it printed, and the files it opened to write. Audited from here,
    # which no code under test can remove, and only while a call runs.
    wrote: list[str] = []
    watching_writes: list[bool] = [False]

    def audited(event: str, args: tuple[Any, ...]) -> None:
        if not watching_writes[0] or event != "open" or len(args) < 3 or not isinstance(args[0], (str, bytes)):
            return
        path, mode, flags = args[0], args[1], args[2]
        if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
                isinstance(flags, int) and flags & writing_flags):
            where = real(path if isinstance(path, str) else path.decode("utf-8", "replace"))
            wrote.append(relative(where, home) if where.startswith(home) else "a file outside the checkout")

    sys.addaudithook(audited)
    sys.path.insert(0, os.getcwd())
    signal.signal(signal.SIGALRM, _too_long)
    questions = _LOADS(sys.stdin)
    pipe = os.fdopen(questions.pop("pipe"), "w", encoding="utf-8", buffering=1)
    mark = questions.pop("mark")
    caught: list[str] = []
    tool = monitoring.DEBUGGER_ID

    def handled(code: Any, _offset: int, error: BaseException) -> Any:
        if os.path.realpath(code.co_filename).startswith(home):
            caught.append(type(error).__name__)
        return None

    monitoring.use_tool_id(tool, "outside_bar_differential")
    monitoring.register_callback(tool, monitoring.events.EXCEPTION_HANDLED, handled)
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
        named = {name: value for name, value in arguments.items() if name not in placed}
        before, held_before = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
        around_before = _held_around(path, made)
        monitoring.set_events(tool, monitoring.events.EXCEPTION_HANDLED)
        printed, shown_to = printing(), sys.stdout
        sys.stdout = printed
        wrote.clear()
        watching_writes[0] = True
        try:
            outcome = _shown(_finished(target(*(arguments[name] for name in placed), *star, **named, **keywords)))
        except TimeoutError:
            return "took too long"
        except BaseException as error:  # noqa: BLE001 - what it raised is what the call came to, an exit included
            outcome = _refusal(error, arguments, home)
        finally:
            watching_writes[0] = False
            monitoring.set_events(tool, 0)
            sys.stdout = shown_to
        said = printed.getvalue()
        outcome = outcome if not said else f"{outcome}; it printed {said[:200]!r}"
        outcome = outcome if not wrote else f"{outcome}; it opened to write {sorted(set(wrote))[:5]}"
        after, held_after = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
        outcome = outcome if after == before else f"{outcome}; its arguments became {after[:200]}"
        outcome = outcome if held_after == held_before else f"{outcome}; the object became {held_after[:200]}"
        around_after = _held_around(path, made)
        return outcome if around_after == around_before else f"{outcome}; what is around it became {around_after[:200]}"

    for name, calls in questions.pop("inputs").items():
        path, qualname = name.split("::")
        for call in calls:
            caught.clear()
            started = clock()
            alarm(_CALL_SECONDS)
            try:
                outcome = came_to(path, qualname, call)
            except BaseException as error:  # noqa: BLE001 - a function that cannot be reached has no value to compare
                outcome = "took too long" if isinstance(error, TimeoutError) else f"unavailable: {type(error).__name__}"
            finally:
                alarm(0)
            pipe.write(_DUMPS([mark, name, _DUMPS(call, sort_keys=True), _short(outcome), clock() - started,
                               sorted(set(caught))]) + "\n")
    pipe.write(_DUMPS([mark, "", "", "done", 0.0, []]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
