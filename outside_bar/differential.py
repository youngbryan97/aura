"""Call the changed functions on the given inputs in this checkout, and say what each call came to.

outside_bar/verify_change.py copies this next to each run under a plain name, hands it the inputs on stdin and reads the
answers from a pipe. The file the answers end up in belongs to the parent, which never imports the code under test.
"""
from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import itertools
import json
import os
import signal
import sys
import time
from typing import Any

#: How long one call may take, in seconds. A call that takes longer says nothing about the value it would have returned.
_CALL_SECONDS = 2
#: Values read from a generator: enough to compare, and an endless one still ends.
_ITEMS = 500
#: How deep an object's parts are followed when it is shown.
_DEEPEST = 4


def _too_long(*_: Any) -> None:
    raise TimeoutError


def _rebuilt(value: Any) -> Any:
    """A value as the witness wrote it down, with each object made again from its class's name and its parts."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__lock__":
        import threading

        return getattr(threading, value[1])()
    if isinstance(value, tuple) and len(value) == 3 and value[0] == "__callable__":
        holder: Any = importlib.import_module(value[1])
        for part in value[2].split("."):
            holder = getattr(holder, part)
        return holder
    if isinstance(value, tuple) and len(value) == 4 and value[0] == "__object__":
        holder: Any = importlib.import_module(value[1])
        for part in value[2].split("."):
            holder = getattr(holder, part)
        made = holder.__new__(holder)
        for name, item in value[3].items():
            object.__setattr__(made, name, _rebuilt(item))
        return made
    if isinstance(value, (list, tuple, set)):
        return type(value)(_rebuilt(item) for item in value)
    if isinstance(value, dict):
        return {_rebuilt(key): _rebuilt(item) for key, item in value.items()}
    return value


def _state_of(made: Any) -> dict[str, Any]:
    state = dict(getattr(made, "__dict__", None) or {})
    for klass in type(made).__mro__:
        slots = getattr(klass, "__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in ("__dict__", "__weakref__") and hasattr(made, name):
                state[name] = getattr(made, name)
    return state


def _shown(value: Any, depth: int = 0) -> str:
    """A value as text two runs can compare: an object without a text of its own is shown by what it holds."""
    text = repr(value)
    if " at 0x" not in text or depth > _DEEPEST:
        return text
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}({[_shown(item, depth + 1) for item in value]})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_shown(key, depth + 1)}: {_shown(item, depth + 1)}" for key, item in value.items()) + "}"
    return f"{type(value).__qualname__}({_shown(_state_of(value), depth + 1)})"


def _found(path: str, qualname: str, call: dict[str, Any]) -> tuple[Any, Any]:
    """The function to call, and the object it was called on when it is a method."""
    module = importlib.import_module(path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
    parts = qualname.split(".")
    holder: Any = module
    for part in parts[:-1]:
        holder = getattr(holder, part)
    made = None
    if "cls" in call:  # a class method is called on the class it was called on, a subclass included
        holder = importlib.import_module(call["cls"][0])
        for part in call["cls"][1].split("."):
            holder = getattr(holder, part)
    if len(parts) > 1 and "self" in call:
        made = holder.__new__(holder)
        for key, text in call["self"].items():
            object.__setattr__(made, key, _rebuilt(ast.literal_eval(text)))
    found = inspect.getattr_static(holder, parts[-1], None)
    if made is not None and isinstance(found, property):
        if call["args"]:
            return (lambda *place, **name: found.fset(made, *place, *name.values())), made
        return (lambda: found.fget(made)), made
    return getattr(made if made is not None else holder, parts[-1]), made


def _held_around(path: str, qualname: str, made: Any) -> str:
    """What the class and the module around a call hold, so a call that changes them changes what it came to."""
    module = sys.modules.get(path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
    places = [type(made)] if made is not None else []
    places += [module] if module is not None else []
    held = {}
    for place in places:
        for name, value in list(vars(place).items()):
            if not name.startswith("__") and not callable(value) and not inspect.ismodule(value):
                held[f"{getattr(place, '__name__', '')}.{name}"] = _shown(value)
    return _shown(held)


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
    if inspect.isgenerator(result):
        return list(itertools.islice(result, _ITEMS))
    return result


def _refusal(error: BaseException, arguments: dict[str, Any]) -> str:
    """The kind of refusal, with the input taken out of its words, so one refusal given two inputs reads as one."""
    words = str(error)
    forms = {form for value in arguments.values() for form in (repr(value), str(value)) if form}
    for form in sorted(forms, key=len, reverse=True):  # the longest first: 'milk' before milk
        words = words.replace(form, "<input>")
    return f"raised {type(error).__name__}: {words[:120]}"


def _came_to(path: str, qualname: str, call: dict[str, Any]) -> str:
    arguments = {name: _rebuilt(ast.literal_eval(text)) for name, text in call["args"].items()}
    star = _rebuilt(ast.literal_eval(call.get("star", "[]")))
    keywords = _rebuilt(ast.literal_eval(call.get("keywords", "{}")))
    target, made = _found(path, qualname, call)
    by_place = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) if star \
        else (inspect.Parameter.POSITIONAL_ONLY,)
    placed = [item.name for item in inspect.signature(target).parameters.values()
              if item.kind in by_place and item.name in arguments]
    named = {name: value for name, value in arguments.items() if name not in placed}
    before, held_before = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
    around_before = _held_around(path, qualname, made)
    try:
        outcome = _shown(_finished(target(*(arguments[name] for name in placed), *star, **named, **keywords)))
    except TimeoutError:
        return "took too long"
    except Exception as error:  # noqa: BLE001 - what it raised is what the call came to
        outcome = _refusal(error, arguments)
    after, held_after = _shown(arguments), _shown(_state_of(made)) if made is not None else ""
    outcome = outcome if after == before else f"{outcome}; its arguments became {after[:200]}"
    outcome = outcome if held_after == held_before else f"{outcome}; the object became {held_after[:200]}"
    around_after = _held_around(path, qualname, made)
    return outcome if around_after == around_before else f"{outcome}; what is around it became {around_after[:200]}"


def _answering(asked: dict[str, Any]) -> Any:
    """A writer for this run's answers that holds the pipe and the mark itself, so nothing the code under test can reach
    on the stack names either."""
    pipe = os.fdopen(asked.pop("pipe"), "w", encoding="utf-8", buffering=1)
    mark = asked.pop("mark")

    def answer(name: str, key: str, outcome: str, took: float) -> None:
        pipe.write(json.dumps([mark, name, key, outcome, took]) + "\n")

    return answer


def main() -> int:
    sys.path.insert(0, os.getcwd())
    signal.signal(signal.SIGALRM, _too_long)
    questions = json.load(sys.stdin)
    answer = _answering(questions)
    for name, calls in questions.pop("inputs").items():
        path, qualname = name.split("::")
        for call in calls:
            started = time.perf_counter()
            signal.alarm(_CALL_SECONDS)
            try:
                outcome = _came_to(path, qualname, call)
            except Exception as error:  # noqa: BLE001 - a function that cannot be reached has no value to compare
                outcome = f"unavailable: {type(error).__name__}"
            finally:
                signal.alarm(0)
            answer(name, json.dumps(call, sort_keys=True), outcome, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
