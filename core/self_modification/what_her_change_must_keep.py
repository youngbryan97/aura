"""What a refused draft of hers lost, written as a test her next try must pass (Master Todo F16, to build).

The outside check (outside_bar/verify_change.py) refuses a draft that changes what an input came to, and since 19
November it returns each such input as data: the function, the call, what it came to before and after. Her repair
search mends from two sets of tests, the failing ones it must make pass and the passing ones it must keep
(semantic_repair.py, after SemFix and Angelix). An input the check counted as lost is one of the second kind. Written
as a test, it is a guard her next search must keep, so a try that loses the same input again is never offered.

Only what can be written back as plain source is kept: a call whose arguments are literals, of a function at module
level, that returned a literal or raised. A call the watch kept only as digests is left out. The inputs come from her
repository's own tests and the inputs near them, never from her real calls, so the test can land with her change.

A change of hers the check doubted lands on probation, and one that survives it is written the other way round: what
it gives now, on the inputs where it gave something else before, as a guard for later changes to keep (``guardable``,
``guarding_test_source``). Those inputs come into every later check of that module, since the check runs the tests that
name it.
"""
from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["guardable", "guarding_test_source", "keepable", "keeping_test_source", "only_these_tests", "path_of_its_guard",
           "path_of_what_it_must_keep"]


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


def _writable(counterexample: Mapping[str, Any], outcome: str) -> bool:
    """Whether a counterexample's call, coming to ``outcome``, can be written as plain source: a call of a module
    function with literal arguments, that came to a literal value or a refusal with nothing else changed on the way."""
    if counterexample.get("as") == "digests":
        return False
    if _module_and_function(str(counterexample.get("function") or "")) is None:
        return False
    call = counterexample.get("input")
    if not isinstance(call, Mapping) or set(call) - {"args", "keywords", "star"} or not isinstance(call.get("args"), Mapping):
        return False
    if not all(str(name).isidentifier() and _literal(text)[0] for name, text in call["args"].items()):
        return False
    if not all(_literal(call[field])[0] for field in ("keywords", "star") if field in call):
        return False
    if ";" in outcome:  # it also changed its arguments, its object or what is around it: not one value to keep
        return False
    if outcome.startswith("raised "):
        # Only a refusal the code raised on purpose is kept. A crash in an operation (a KeyError on an unknown item, a
        # TypeError on a None the check put in) may be the very fault a goal mends, and a test that keeps it would
        # hold every later try to the fault.
        return bool(re.match(r"raised [A-Za-z_][A-Za-z0-9_]* by the code", outcome))
    return _literal(outcome)[0]


def keepable(counterexample: Mapping[str, Any]) -> bool:
    """Whether a counterexample can be written as a test of what the code before a refused draft gave: counted as
    lost, and writable as plain source."""
    return counterexample.get("counted") == "lost" and _writable(counterexample, str(counterexample.get("before") or ""))


#: The differences a change kept after probation is guarded on. A refusal gone is left out. What settles probation is
#: the outside benchmarks, which ask only valid questions, and her real calls, which cannot be written into her public
#: source; neither speaks for an input that was refused before, and a deleted guard looks exactly like that (rounds five
#: and six of the red team). A fault that still happens is not something to keep.
_GUARDED_KINDS = frozenset({"another value", "now raises", "another refusal"})


def guardable(counterexample: Mapping[str, Any]) -> bool:
    """Whether a counterexample can be written as a guard of what a change kept after probation gives now: counted as
    a doubt, a difference settling speaks for, and writable as plain source."""
    return (counterexample.get("counted") == "doubt" and counterexample.get("kind") in _GUARDED_KINDS
            and _writable(counterexample, str(counterexample.get("after") or "")))


def _kind_raised(before: str) -> str:
    return re.match(r"raised ([A-Za-z_][A-Za-z0-9_]*)", before).group(1)  # type: ignore[union-attr]


def _safe(slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slug).lower()).strip("_")[:60] or "her_change"


def path_of_what_it_must_keep(slug: str) -> str:
    """Where the test of what a goal's change must keep lives in her source."""
    return f"tests/test_what_{_safe(slug)}_must_keep.py"


def path_of_its_guard(slug: str) -> str:
    """Where the guard of what a goal's change, kept after probation, gives now lives in her source."""
    return f"tests/test_what_{_safe(slug)}_changed_is_guarded.py"


def _definition(function: str, root: Path | None) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The module-level definition of a function in its source, inside an ``if``, ``try``, ``with`` or ``match`` at
    module level too (a version guard), and through a module-level name bound to another (``score = _fast``); None
    when the source cannot be read."""
    path, _, name = function.partition("::")
    try:
        tree = ast.parse((root / path).read_bytes()) if root is not None else None
    except (OSError, SyntaxError, ValueError):
        return None
    for _ in range(8):  # an alias of an alias, a few deep
        waiting, alias = list(getattr(tree, "body", ())), None
        while waiting:
            node = waiting.pop(0)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id == name and isinstance(node.value, ast.Name):
                alias = node.value.id
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.Match)):
                waiting += [*getattr(node, "body", []), *getattr(node, "orelse", []), *getattr(node, "finalbody", []),
                            *(item for handler in getattr(node, "handlers", []) for item in handler.body),
                            *(item for case in getattr(node, "cases", []) for item in case.body)]
        if alias is None:
            return None
        name = alias
    return None


def _yields(node: ast.AST) -> bool:
    """Whether a function's own body yields, not that of a function or class inside it."""
    waiting = list(ast.iter_child_nodes(node))
    while waiting:
        item = waiting.pop()
        if isinstance(item, (ast.Yield, ast.YieldFrom)):
            return True
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            waiting += list(ast.iter_child_nodes(item))
    return False


def _kind_of(function: str, root: Path | None) -> str:
    """How the check read what a call came to: "plain", "coroutine" (awaited), "generator" (read out as a list), or
    "async generator", which a plain test cannot read back."""
    node = _definition(function, root)
    if node is None:
        return "plain"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async generator" if _yields(node) else "coroutine"
    return "generator" if _yields(node) else "plain"


def _placed(function: str, root: Path | None) -> tuple[list[str], list[str]]:
    """The parameters of a module-level function its source makes positional-only, and its ordinary ones, in order;
    nothing when the source cannot be read."""
    node = _definition(function, root)
    if node is None:
        return [], []
    return [item.arg for item in node.args.posonlyargs], [item.arg for item in node.args.args]


def _call_of(function: str, call: Mapping[str, Any], root: Path | None) -> str:
    """The call as source, the way the check made it: positional-only arguments in their places, every one before an
    extra positional argument in its place too, the rest by name, then the extra positional and keyword arguments."""
    module_function = _module_and_function(function)[1]  # type: ignore[index]
    only, ordinary = _placed(function, root)
    arguments = dict(call["args"])
    by_place = [*only, *(ordinary if "star" in call else [])]
    parts = [arguments.pop(name) for name in by_place if name in arguments]
    parts += [f"*{call['star']}"] if "star" in call else []
    parts += [f"{name}={text}" for name, text in arguments.items()]
    parts += [f"**{call['keywords']}"] if "keywords" in call else []
    return f"{module_function}({', '.join(parts)})"


def _tests_of(counterexamples: Iterable[Mapping[str, Any]], *, admits, outcome: str,
              root: Path | None = None) -> tuple[set[str], list[str]]:
    """The imports and the test bodies asserting each admitted counterexample's call comes to its ``outcome`` side.
    ``root`` is the source the calls are of, read for the parameters a call must pass by place."""
    calls: list[str] = []
    imports: set[str] = set()
    for one in counterexamples:
        if not admits(one):
            continue
        module, function = _module_and_function(str(one["function"]))  # type: ignore[misc]
        call = _call_of(str(one["function"]), one["input"], root)
        # The check awaited a coroutine and read an iterator out before comparing (outside_bar/differential.py), and
        # says which it did; the test does the same. An async iterator it cannot read back plainly, and it is left out.
        # Where the check did not say (an older verdict), the source is read for it.
        read_as = (one.get("came_as") or {}).get(outcome) if isinstance(one.get("came_as"), Mapping) else None
        kind_of_call = {"coroutine": "coroutine", "async iterator": "async generator", "iterator": "generator",
                        "plain": "plain"}.get(str(read_as), "") or _kind_of(str(one["function"]), root)
        if kind_of_call == "async generator":
            continue
        if kind_of_call == "coroutine":
            call = f"asyncio.run({call})"
            imports.add("import asyncio")
        elif kind_of_call == "generator":
            call = f"list({call})"
        value = str(one[outcome])
        kind = _kind_raised(value) if value.startswith("raised ") else ""
        if kind and isinstance(getattr(builtins, kind, None), type) and issubclass(getattr(builtins, kind), BaseException):
            body = f"    with pytest.raises({kind}):\n        {call}\n"
        elif kind:  # a refusal of the module's own kind, known here only by its name
            body = (f"    with pytest.raises(Exception) as raised:\n        {call}\n"
                    f"    assert type(raised.value).__name__ == {kind!r}\n")
        else:
            # The same value of the same type, as the check compared it: ``True == 1`` holds in Python, and a test of
            # ``== True`` would pass on the very draft that returned 1 (round nine of the red team).
            body = f"    assert _same({call}, {value})\n"
        if body in calls:
            continue
        imports.add(f"from {module} import {function}")
        calls.append(body)
    if any("pytest.raises" in body for body in calls):
        imports.add("import pytest")
    return imports, calls


def _for_a_docstring(goal: str) -> str:
    """A goal's words, on one line and at most 100 characters, that cannot end the docstring they are put in."""
    return " ".join(str(goal).split())[:100].replace("\\", "/").replace('"', "'")


_SAME = '''

def _same(got, want):
    """The same value of the same type, all the way down: True is not 1, and 1.0 is not 1."""
    if type(got) is not type(want):
        return False
    if isinstance(want, dict):
        return list(got) == list(want) and all(_same(got[key], want[key]) for key in want)
    if isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(_same(one, other) for one, other in zip(got, want))
    return got == want
'''


def _a_test_file(head: str, imports: set[str], calls: list[str], name: str) -> str:
    if not calls:
        return ""
    tests = [f"\n\ndef test_{name}_{number}():\n{body}" for number, body in enumerate(calls)]
    ordered = sorted(imports, key=lambda line: (not line.startswith("import "), line))
    helper = _SAME if any("_same(" in body for body in calls) else ""
    return head + "\n".join(ordered) + "\n" + helper + "".join(tests)


def only_these_tests(source: str, names: Iterable[str]) -> str:
    """A written test file with only the tests named ``names`` left in it, or "" when none is. Her mending runs each
    kept test on the code before a refused draft, where it must pass, and on the draft, where it must fail, and keeps
    only those that do: a test that fails on code that worked would hold her next try to a fault, and one that passes
    on the draft keeps nothing (round nine of the red team wrote both)."""
    keep = set(names)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    lines = source.splitlines(keepends=True)
    dropped = [(node.lineno, node.end_lineno or node.lineno) for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and node.name not in keep]
    if not any(isinstance(node, ast.FunctionDef) and node.name in keep for node in tree.body):
        return ""
    for first, last in sorted(dropped, reverse=True):
        start = first - 1
        while start > 0 and not lines[start - 1].strip():  # the blank lines before it go with it
            start -= 1
        del lines[start:last]
    return "".join(lines)


def keeping_test_source(counterexamples: Iterable[Mapping[str, Any]], *, goal: str, root: Path | None = None) -> str:
    """A test file with one test per input the refused drafts of a goal lost, asserting what the code before them gave;
    "" when none can be written. Her record keeps every refusal's counterexamples, so the file is written whole each
    time and a second refusal adds to the first."""
    imports, calls = _tests_of(counterexamples, admits=keepable, outcome="before", root=root)
    head = (f'"""What a change for this goal of hers must keep: {_for_a_docstring(goal)}.\n\n'
            "Each test is an input that worked before a draft of hers and that the outside check counted as lost when it\n"
            "refused the draft (core/self_modification/what_her_change_must_keep.py). Her next try keeps them.\n"
            '"""\n')
    return _a_test_file(head, imports, calls, "it_keeps")


def guarding_test_source(counterexamples: Iterable[Mapping[str, Any]], *, goal: str, commit: str,
                         root: Path | None = None) -> str:
    """A test file with one test per input a change kept after probation answers differently from the code before it,
    asserting what it gives now; "" when none can be written."""
    imports, calls = _tests_of(counterexamples, admits=guardable, outcome="after", root=root)
    head = (f'"""What her change {commit[:9]} gives now, kept after probation: {_for_a_docstring(goal)}.\n\n'
            "Each test is an input the outside check saw this change answer differently from the code before it. The\n"
            "outside benchmarks held with it and nothing she ran showed a loss, so a later change keeps what it gives\n"
            "(core/self_modification/what_her_change_must_keep.py).\n"
            '"""\n')
    return _a_test_file(head, imports, calls, "it_gives")
