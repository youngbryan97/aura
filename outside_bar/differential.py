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
_ITEMS = 50


def _too_long(*_: Any) -> None:
    raise TimeoutError


def _found(path: str, qualname: str, call: dict[str, Any]) -> Any:
    module = importlib.import_module(path.removesuffix(".py").removesuffix("/__init__").replace("/", "."))
    parts = qualname.split(".")
    holder: Any = module
    for part in parts[:-1]:
        holder = getattr(holder, part)
    if len(parts) > 1 and "self" in call:
        made = holder.__new__(holder)
        made.__dict__.update({key: ast.literal_eval(text) for key, text in call["self"].items()})
        holder = made
    return getattr(holder, parts[-1])


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
    arguments = {name: ast.literal_eval(text) for name, text in call["args"].items()}
    target = _found(path, qualname, call)
    only_by_place = [item.name for item in inspect.signature(target).parameters.values()
                     if item.kind is inspect.Parameter.POSITIONAL_ONLY and item.name in arguments]
    placed = [arguments[name] for name in only_by_place]
    named = {name: value for name, value in arguments.items() if name not in only_by_place}
    before = repr(arguments)
    try:
        outcome = repr(_finished(target(*placed, **named)))
    except TimeoutError:
        return "took too long"
    except Exception as error:  # noqa: BLE001 - what it raised is what the call came to
        outcome = _refusal(error, arguments)
    after = repr(arguments)
    return outcome if after == before else f"{outcome}; its arguments became {after[:200]}"


def main() -> int:
    asked = json.load(sys.stdin)
    sys.path.insert(0, os.getcwd())
    signal.signal(signal.SIGALRM, _too_long)
    with os.fdopen(asked["pipe"], "w", encoding="utf-8") as pipe:
        for name, calls in asked["inputs"].items():
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
                pipe.write(json.dumps([asked["mark"], name, json.dumps(call, sort_keys=True), outcome,
                                       time.perf_counter() - started]) + "\n")
                pipe.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
