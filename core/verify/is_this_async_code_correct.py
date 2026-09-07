"""The async mistakes that make generated code look right and behave wrong.

A source-matched live run finished and delivered a reply in 335 seconds. The
Python it had written did five things:

* queued job coroutines were never awaited, so the jobs never ran;
* it could deadlock waiting for queue data while holding the producer's lock;
* it mutated the queue's internal deque directly, bypassing ``put()`` and the
  bookkeeping that makes a queue a queue;
* four coroutines were created and dropped, which Python warns about at
  garbage-collection time and nothing was reading;
* and a comment asserted that cancellation releases an ``asyncio.Lock``, which
  it does not.

Delivery succeeded and semantic correctness was zero. That gap is the thing
this checks, because none of the five is visible in a reply: the code reads
fluently, imports cleanly, and passes anything that only asks whether it parses.

Every check here is a shape in the syntax tree rather than a name on a list, so
it fires on code written today for a library nobody has heard of. Each finding
says what will happen rather than which rule was broken — "the coroutine is
created and dropped, so the job never runs" beats "unawaited coroutine".
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.IsThisAsyncCodeCorrect")

__all__ = [
    "AMistake",
    "THE_CHECKS",
    "THE_SMELLS",
    "what_is_wrong_with",
    "what_is_worth_a_look_in",
    "is_it_correct",
]


@dataclass(frozen=True, slots=True)
class AMistake:
    """One thing that will go wrong, and where."""

    kind: str
    line: int
    what_happens: str
    #: The source line, so a reader does not have to go and find it.
    said: str = ""

    def __str__(self) -> str:
        return f"line {self.line}: {self.what_happens}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "line": self.line,
            "what_happens": self.what_happens,
            "said": self.said,
        }


#: Things that will go wrong. A finding here means the code does not do what
#: it says.
THE_CHECKS: tuple[str, ...] = (
    "a coroutine created and dropped",
    "reaching inside a queue instead of using it",
    "a lock released by cancellation",
    "a bare except swallowing cancellation",
)

#: Things worth a second look that are not defects. `async def` with no await
#: is a legitimate choice — an awaitable interface whose implementation is
#: synchronous today — and 253 of them in this tree are that rather than
#: mistakes. Reported separately so a real finding is not buried in them.
THE_SMELLS: tuple[str, ...] = (
    "an async function that never awaits",
    "a blocking await while holding a lock",
)

#: Calls that return a coroutine and do nothing unless awaited. Names rather
#: than resolved symbols: generated code names things this process has never
#: imported.
_ASYNC_LIBRARY_CALLS = frozenset(
    {
        "sleep",
        "wait",
        "wait_for",
        "to_thread",
        "start_server",
        "open_connection",
    }
)

#: Awaits that can block until something else happens. Holding a lock across
#: them merits inspection, but a deadlock requires a dependency cycle that
#: this local syntax scan cannot establish.
#:
#: ``put`` is not here. It blocks only on a bounded queue, and nothing in the
#: syntax says whether this queue has a bound, so flagging it calls correct
#: code wrong — which is the failure mode that gets a checker switched off.
_WAITS_FOR_SOMEONE_ELSE = frozenset({"get", "join", "acquire", "wait"})

#: The private innards of the standard queues. Touching them is how the
#: bookkeeping that makes a queue a queue gets skipped.
_INSIDE_A_QUEUE = frozenset({"_queue", "_getters", "_putters", "_unfinished_tasks"})


def _without_nested_functions(node: ast.AST) -> list[ast.AST]:
    """Every node inside this one, stopping at a nested function or class."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        one = stack.pop()
        if isinstance(
            one, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        out.append(one)
        stack.extend(ast.iter_child_nodes(one))
    return out


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _line(source: str, number: int) -> str:
    lines = source.splitlines()
    return lines[number - 1].strip() if 0 < number <= len(lines) else ""


def _a_coroutine_dropped(tree: ast.AST, source: str) -> list[AMistake]:
    """A call to something async whose result is thrown away.

    Only where the call can be resolved to an ``async def`` in this same
    source: a bare name matching a top-level one, or ``self.x()`` matching a
    method of the class it is written in. Matching any attribute against any
    async name in the file flagged ``orch.semantic_defrag.start()`` because
    something unrelated in that file was called ``start`` — which is the same
    kind of mistake as the ones being looked for.
    """
    top_level: set[str] = {
        node.name for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
    }
    methods: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods[node.name] = {
                one.name
                for one in node.body
                if isinstance(one, ast.AsyncFunctionDef)
            }
    inside: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for inner in ast.walk(node):
                inside[id(inner)] = node.name

    found: list[AMistake] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        called = _name_of(func)
        if not called:
            continue
        made_here = isinstance(func, ast.Name) and called in top_level
        if (
            not made_here
            and isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            owner = inside.get(id(node))
            made_here = bool(owner and called in methods.get(owner, set()))
        from_the_library = (
            isinstance(func, ast.Attribute)
            and _name_of(func.value) == "asyncio"
            and called in _ASYNC_LIBRARY_CALLS
        )
        if not (made_here or from_the_library):
            continue
        found.append(
            AMistake(
                kind="a coroutine created and dropped",
                line=node.lineno,
                what_happens=(
                    f"{called}() returns a coroutine and nothing awaits it, so the "
                    "work never runs and Python warns about it at collection time"
                ),
                said=_line(source, node.lineno),
            )
        )
    return found


def _blocking_await_under_a_lock(tree: ast.AST, source: str) -> list[AMistake]:
    """An await that waits for someone else, inside `async with` a lock."""
    found: list[AMistake] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncWith, ast.With)):
            continue
        holding = [
            _name_of(item.context_expr.func)
            if isinstance(item.context_expr, ast.Call)
            else _name_of(item.context_expr)
            for item in node.items
        ]
        looks_like_a_lock = any(
            "lock" in str(one).lower() or "semaphore" in str(one).lower()
            for one in holding
        )
        if not looks_like_a_lock:
            continue
        # Not into nested function bodies. A coroutine DEFINED inside the
        # block does not hold the lock when it later runs, and counting it
        # flagged a fallback orchestrator's run loop that holds nothing.
        for inner in _without_nested_functions(node):
            if not isinstance(inner, ast.Await):
                continue
            waited = inner.value
            if not isinstance(waited, ast.Call):
                continue
            called = _name_of(waited.func)
            if called not in _WAITS_FOR_SOMEONE_ELSE:
                continue
            found.append(
                AMistake(
                    kind="a blocking await while holding a lock",
                    line=inner.lineno,
                    what_happens=(
                        f"awaiting {called}() while holding the lock may deadlock "
                        "if the operation that unblocks it needs this same lock; "
                        "the syntax alone does not establish that dependency"
                    ),
                    said=_line(source, inner.lineno),
                )
            )
    return found


def _reaching_inside_a_queue(tree: ast.AST, source: str) -> list[AMistake]:
    """Touching a queue's private storage instead of calling put or get."""
    found: list[AMistake] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in _INSIDE_A_QUEUE:
            continue
        holder = _name_of(node.value).lower()
        if holder and "queue" not in holder and "q" != holder:
            continue
        found.append(
            AMistake(
                kind="reaching inside a queue instead of using it",
                line=node.lineno,
                what_happens=(
                    f"{node.attr} is the queue's own storage; writing to it skips "
                    "the waiter wake-up and the unfinished-task count, so get() "
                    "blocks on items that are already there"
                ),
                said=_line(source, node.lineno),
            )
        )
    return found


def _a_lock_released_by_cancellation(tree: ast.AST, source: str) -> list[AMistake]:
    """Code or comment relying on cancellation to release a lock. It does not."""
    found: list[AMistake] = []
    # Only where the file actually uses a lock, and only in a comment. The
    # claim is prose, so it can only be read as prose, and prose about locks
    # in a file with no locks is prose about something else.
    if "Lock(" not in source and ".lock" not in source:
        return found
    for number, line in enumerate(source.splitlines(), 1):
        said = line.lower()
        if "#" not in line:
            continue
        said = line.split("#", 1)[1].lower()
        if "cancel" not in said:
            continue
        # "lock-free" is a claim about there being no lock.
        if not re.search(r"\block\b(?!-free)", said):
            continue
        if not any(
            word in said for word in ("release", "releases", "released", "frees", "free")
        ):
            continue
        found.append(
            AMistake(
                kind="a lock released by cancellation",
                line=number,
                what_happens=(
                    "cancelling a task does not release a lock it holds; the lock "
                    "is released by leaving the block, and a cancellation that "
                    "skips the block leaves it held"
                ),
                said=line.strip(),
            )
        )
    return found


def _an_async_function_that_never_awaits(tree: ast.AST, source: str) -> list[AMistake]:
    """`async def` with no await in it. Callers pay a coroutine for nothing."""
    found: list[AMistake] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        awaits = any(
            isinstance(inner, (ast.Await, ast.AsyncFor, ast.AsyncWith))
            for inner in ast.walk(node)
        )
        yields = any(isinstance(inner, (ast.Yield, ast.YieldFrom)) for inner in ast.walk(node))
        if awaits or yields:
            continue
        found.append(
            AMistake(
                kind="an async function that never awaits",
                line=node.lineno,
                what_happens=(
                    f"{node.name} is declared async and never awaits, so every "
                    "caller must await a coroutine for work that was synchronous"
                ),
                said=_line(source, node.lineno),
            )
        )
    return found


def _swallowing_cancellation(tree: ast.AST, source: str) -> list[AMistake]:
    """`except Exception` around an await catches nothing — but bare does."""
    found: list[AMistake] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        catches_everything = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id == "BaseException"
        )
        if not catches_everything:
            continue
        reraises = any(isinstance(one, ast.Raise) for one in ast.walk(node))
        has_await = any(isinstance(one, ast.Await) for one in ast.walk(node))
        # An exception put somewhere for a caller to raise later is not
        # swallowed. The shutdown helpers do exactly that, and calling it a
        # defect would be reading the `except` and not the function.
        kept = bool(node.name) and any(
            isinstance(one, ast.Name) and one.id == node.name
            for one in ast.walk(node)
        )
        if reraises or has_await or kept:
            continue
        found.append(
            AMistake(
                kind="a bare except swallowing cancellation",
                line=node.lineno,
                what_happens=(
                    "this catches CancelledError as well, so a cancelled task "
                    "carries on and the thing that cancelled it waits forever"
                ),
                said=_line(source, node.lineno),
            )
        )
    return found


_THE_CHECKS = (
    _a_coroutine_dropped,
    _reaching_inside_a_queue,
    _a_lock_released_by_cancellation,
    _swallowing_cancellation,
)

_THE_SMELLS = (_an_async_function_that_never_awaits, _blocking_await_under_a_lock)


def what_is_wrong_with(code: str) -> tuple[AMistake, ...]:
    """Every async mistake in this source, in the order it appears.

    Unparsable source returns nothing rather than raising: whatever else is
    wrong with code that will not parse, it is not these.
    """
    source = str(code or "")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    found: list[AMistake] = []
    for check in _THE_CHECKS:
        found.extend(check(tree, source))
    return tuple(sorted(found, key=lambda one: (one.line, one.kind)))


def what_is_worth_a_look_in(code: str) -> tuple[AMistake, ...]:
    """Smells: not defects, and worth reading before the code ships."""
    source = str(code or "")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    found: list[AMistake] = []
    for check in _THE_SMELLS:
        found.extend(check(tree, source))
    return tuple(sorted(found, key=lambda one: one.line))


def is_it_correct(code: str) -> dict[str, Any]:
    """A verdict a caller can serve, refuse, or hand back to the writer."""
    wrong = what_is_wrong_with(code)
    smells = what_is_worth_a_look_in(code)
    return {
        "correct": not wrong,
        "checked": list(THE_CHECKS),
        "mistakes": [one.to_dict() for one in wrong],
        "worth_a_look": [one.to_dict() for one in smells],
        "what_to_say": (
            ""
            if not wrong
            else "; ".join(one.what_happens for one in wrong[:3])
        ),
    }
