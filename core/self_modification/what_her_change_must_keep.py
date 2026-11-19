"""What a refused draft of hers lost, written as a test her next try must pass (Master Todo F16, to build).

The outside check (outside_bar/verify_change.py) refuses a draft that changes what an input came to, and since 19
November it returns each such input as data: the function, the call, what it came to before and after. Her repair
search mends from two sets of tests, the failing ones it must make pass and the passing ones it must keep
(semantic_repair.py, after SemFix and Angelix). An input the check counted as lost is one of the second kind. Written
as a test, it is a guard her next search must keep, so a try that loses the same input again is never offered.

Only what can be written back as plain source is kept: a call whose arguments are literals, of a function at module
level, that returned a literal or raised. A call the watch kept only as digests is left out. The inputs come from her
repository's own tests and the inputs near them, never from her real calls, so the test can land with her change.
"""
from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["keepable", "keeping_test_source", "path_of_what_it_must_keep"]


def _literal(text: Any) -> tuple[bool, Any]:
    try:
        return True, ast.literal_eval(str(text))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _module_and_function(function: str) -> tuple[str, str] | None:
    path, _, qualname = str(function).partition("::")
    if not path.endswith(".py") or not qualname.isidentifier():
        return None
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(part.isidentifier() for part in parts) or parts[0] in ("tests", "test"):
        return None
    return ".".join(parts), qualname


def keepable(counterexample: Mapping[str, Any]) -> bool:
    """Whether a counterexample can be written as a test: counted as lost, readable, a plain call of a module function
    that came to a literal value or a refusal, with nothing else changed on the way."""
    if counterexample.get("counted") != "lost" or counterexample.get("as") == "digests":
        return False
    if _module_and_function(str(counterexample.get("function") or "")) is None:
        return False
    call = counterexample.get("input")
    if not isinstance(call, Mapping) or set(call) - {"args", "keywords"} or not isinstance(call.get("args"), Mapping):
        return False
    if not all(str(name).isidentifier() and _literal(text)[0] for name, text in call["args"].items()):
        return False
    before = str(counterexample.get("before") or "")
    if ";" in before:  # it also changed its arguments, its object or what is around it: not one value to keep
        return False
    if before.startswith("raised "):
        return bool(re.match(r"raised [A-Za-z_][A-Za-z0-9_]*", before))
    return _literal(before)[0]


def _kind_raised(before: str) -> str:
    return re.match(r"raised ([A-Za-z_][A-Za-z0-9_]*)", before).group(1)  # type: ignore[union-attr]


def path_of_what_it_must_keep(slug: str) -> str:
    """Where the test of what a goal's change must keep lives in her source."""
    safe = re.sub(r"[^a-z0-9]+", "_", str(slug).lower()).strip("_")[:60] or "her_change"
    return f"tests/test_what_{safe}_must_keep.py"


def keeping_test_source(counterexamples: Iterable[Mapping[str, Any]], *, goal: str) -> str:
    """A test file with one test per input the refused drafts of a goal lost, asserting what the code before them gave;
    "" when none can be written. Her record keeps every refusal's counterexamples, so the file is written whole each
    time and a second refusal adds to the first."""
    calls: list[str] = []
    imports: set[str] = set()
    seen: set[str] = set()
    for one in counterexamples:
        if not keepable(one):
            continue
        module, function = _module_and_function(str(one["function"]))  # type: ignore[misc]
        arguments = ", ".join(f"{name}={text}" for name, text in one["input"]["args"].items())
        call = f"{function}({arguments})"
        before = str(one["before"])
        kind = _kind_raised(before) if before.startswith("raised ") else ""
        if kind and isinstance(getattr(builtins, kind, None), type) and issubclass(getattr(builtins, kind), BaseException):
            body = f"    with pytest.raises({kind}):\n        {call}\n"
        elif kind:  # a refusal of the module's own kind, known here only by its name
            body = (f"    with pytest.raises(Exception) as raised:\n        {call}\n"
                    f"    assert type(raised.value).__name__ == {kind!r}\n")
        else:
            body = f"    assert {call} == {before}\n"
        if body in seen:
            continue
        seen.add(body)
        imports.add(f"from {module} import {function}")
        calls.append(body)
    if not calls:
        return ""
    if any("pytest.raises" in body for body in calls):
        imports.add("import pytest")
    tests = [f"\n\ndef test_it_keeps_{number}():\n{body}" for number, body in enumerate(calls)]
    head = (f'"""What a change for this goal of hers must keep: {goal[:100]!s}.\n\n'
            "Each test is an input that worked before a draft of hers and that the outside check counted as lost when it\n"
            "refused the draft (core/self_modification/what_her_change_must_keep.py). Her next try keeps them.\n"
            '"""\n')
    ordered = sorted(imports, key=lambda line: (not line.startswith("import "), line))
    return head + "\n".join(ordered) + "\n" + "".join(tests)
