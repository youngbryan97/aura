"""Exact mechanisms a written procedure can call, so each is worked out once and shared.

A procedure her model writes from solved examples
(core/learning/procedures_from_solved_examples.py) has two jobs: read the
problem, and carry it to the answer. The second job is mostly a handful of
mechanics that recur across kinds of problem: which assignments satisfy a
set of constraints, which orders respect a set of precedences, whether a
logical expression holds, what an arithmetic expression comes to exactly,
what a clock time or a date becomes, which stretches of a day are free. Each
is written here once, tested, and exact, so a procedure composes them
instead of rewriting a search inside every function it writes.

The set is limited on purpose. Every mechanism here is one a development
kind of G09 needs (Natural Plan calendar scheduling; BBEH boolean expressions
and hyperbaton) or one her runtime already had (the constraint search of
core/reasoning/settled_by_neighbours.py). A mechanism added because a
held-out kind needed it would carry that kind's answer into the measurement.

Procedures run in an OS sandbox with only the standard library, so this
module imports nothing else, and reaches her constraint engine as a sibling
module there.
"""

from __future__ import annotations

import ast
import datetime as _dt
import inspect
import operator
import re
from collections.abc import Callable, Hashable, Iterable, Sequence
from fractions import Fraction
from typing import Any

try:
    from core.reasoning import settled_by_neighbours as _settle
except ImportError:  # inside the sandbox this directory is on the path
    import settled_by_neighbours as _settle  # type: ignore[no-redef]

__all__ = [
    "solve_constraints",
    "only_solution",
    "orders",
    "precedences_from_sequences",
    "holds",
    "exact",
    "decimal_text",
    "minutes",
    "clock",
    "WEEKDAYS",
    "weekday_of",
    "add_to_date",
    "merge_intervals",
    "free_intervals",
    "reference",
]


# ── constraints ─────────────────────────────────────────────────────────────


def solve_constraints(
    variables: dict[str, Sequence[Hashable]],
    constraints: Iterable[tuple[Sequence[str], Callable[..., bool]]],
    *,
    all_different: Iterable[Sequence[str]] = (),
    limit: int = 2,
    seconds: float = 10.0,
) -> list[dict[str, Hashable]]:
    """Assignments, up to ``limit``, in which every constraint holds.

    ``variables`` maps each name to the values it may take; each constraint is
    ``(names, test)``, where ``test`` receives those variables' values in that
    order. ``all_different`` lists groups whose members take different values.
    With the default limit of two, one result proves the solution unique and
    two prove it is not. Raises TimeoutError if the search cannot finish.
    """
    names = list(variables)
    checks = []
    for scope, test in constraints:
        scope = tuple(scope)
        unknown = [name for name in scope if name not in variables]
        if unknown:
            raise ValueError(f"a constraint names unknown variables {unknown}")
        checks.append(_settle.Constraint(scope, test))
    groups = [tuple(group) for group in all_different]
    for group in groups:
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                checks.append(_settle.Constraint((first, second), operator.ne))
    problem = _settle.Problem({name: tuple(variables[name]) for name in names}, checks, groups)
    found, finished = _settle.solutions_up_to(problem, limit, budget_s=seconds)
    if not finished and len(found) < limit:
        raise TimeoutError("the constraint search did not finish")
    return found


def only_solution(variables: dict[str, Sequence[Hashable]],
                  constraints: Iterable[tuple[Sequence[str], Callable[..., bool]]],
                  *, all_different: Iterable[Sequence[str]] = ()) -> dict[str, Hashable]:
    """The one assignment satisfying every constraint; ValueError if there is none or more than one."""
    found = solve_constraints(variables, constraints, all_different=all_different, limit=2)
    if len(found) != 1:
        raise ValueError("no solution" if not found else "more than one solution")
    return found[0]


# ── orders ──────────────────────────────────────────────────────────────────


def orders(items: Iterable[Hashable], before: Iterable[tuple[Hashable, Hashable]], *, limit: int = 2) -> list[list]:
    """Orderings of ``items``, up to ``limit``, in which each ``(a, b)`` has a before b.

    One result means the precedences fix the order; none means they
    contradict each other.
    """
    items = list(dict.fromkeys(items))
    after: dict[Hashable, set] = {item: set() for item in items}
    waiting = dict.fromkeys(items, 0)
    for first, second in set(before):
        if first not in after or second not in after:
            raise ValueError(f"precedence names an item not given: {(first, second)}")
        if second not in after[first]:
            after[first].add(second)
            waiting[second] += 1
    found: list[list] = []

    def go(prefix: list, waiting: dict) -> None:
        if len(found) >= limit:
            return
        ready = [item for item in items if waiting.get(item) == 0]
        if not ready:
            if len(prefix) == len(items):
                found.append(list(prefix))
            return
        for item in ready:
            following = dict(waiting)
            del following[item]
            for later in after[item]:
                following[later] -= 1
            go(prefix + [item], following)

    go([], waiting)
    return found


def precedences_from_sequences(sequences: Iterable[Sequence[Hashable]]) -> set[tuple[Hashable, Hashable]]:
    """Every ``(a, b)`` with a seen before b in one of the sequences."""
    pairs = set()
    for sequence in sequences:
        for i, first in enumerate(sequence):
            for second in sequence[i + 1:]:
                if first != second:
                    pairs.add((first, second))
    return pairs


# ── logic and exact arithmetic ──────────────────────────────────────────────

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
            ast.Gt: operator.gt, ast.GtE: operator.ge}
_FUNCTIONS = {"max": max, "min": min, "abs": abs, "sum": lambda *values: sum(values)}


def _value(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _value(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
        return node.value if isinstance(node.value, bool) else Fraction(str(node.value))
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        operand = _value(node.operand, names)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_value(value, names) for value in node.values)
        return any(_value(value, names) for value in node.values)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _value(node.left, names), _value(node.right, names)
        if isinstance(node.op, ast.Pow) and isinstance(right, Fraction) and right.denominator != 1:
            raise ValueError("a fractional power is not exact")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Compare):
        left = _value(node.left, names)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _value(right_node, names)
            if type(op) not in _COMPARE or not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS \
            and not node.keywords:
        return _FUNCTIONS[node.func.id](*(_value(arg, names) for arg in node.args))
    raise ValueError(f"not an exact expression: {ast.dump(node)[:80]}")


def exact(expression: str, names: dict[str, Any] | None = None) -> Any:
    """The value of an arithmetic or logical expression, with every number an exact fraction.

    Python's syntax: ``+ - * / // % **``, comparisons (chained), ``and or
    not``, ``True False``, parentheses, and ``max min abs sum``. ``names``
    gives values to any other names. Nothing else is evaluated.
    """
    return _value(ast.parse(expression.strip(), mode="eval"), dict(names or {}))


def holds(expression: str, names: dict[str, Any] | None = None) -> bool:
    """Whether a logical expression is true; see ``exact`` for what it may contain."""
    return bool(exact(expression, names))


def decimal_text(value: Fraction | int, places: int) -> str:
    """``value`` rounded half away from zero to ``places`` decimals, as text."""
    value = Fraction(value)
    scaled = abs(value) * 10**places
    whole = int(scaled + Fraction(1, 2))
    text = f"{whole // 10**places}" + (f".{whole % 10**places:0{places}d}" if places else "")
    return ("-" if value < 0 and whole else "") + text


# ── clock times, days and dates ─────────────────────────────────────────────

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_CLOCK = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*$", re.IGNORECASE)


def minutes(text: str) -> int:
    """Minutes after midnight for a clock time: ``9:30``, ``17:00``, ``9am``, ``3:45 PM``, ``12:00 a.m.``"""
    match = _CLOCK.match(text)
    if not match:
        raise ValueError(f"not a clock time: {text!r}")
    hour, minute, half = int(match.group(1)), int(match.group(2) or 0), (match.group(3) or "").lower()
    if half:
        if not 1 <= hour <= 12:
            raise ValueError(f"not a twelve-hour time: {text!r}")
        hour = hour % 12 + (12 if half.startswith("p") else 0)
    if hour > 24 or minute > 59:
        raise ValueError(f"not a clock time: {text!r}")
    return hour * 60 + minute


def clock(total_minutes: int, *, twelve_hour: bool = False, pad_hour: bool = True) -> str:
    """A clock time from minutes after midnight, wrapping past a day: ``14:05`` or ``2:05 PM``."""
    hour, minute = divmod(int(total_minutes) % (24 * 60), 60)
    if twelve_hour:
        return f"{(hour % 12) or 12}:{minute:02d} {'PM' if hour >= 12 else 'AM'}"
    return f"{hour:02d}:{minute:02d}" if pad_hour else f"{hour}:{minute:02d}"


def weekday_of(day: _dt.date) -> str:
    """The English name of the day of the week a ``datetime.date`` falls on."""
    return WEEKDAYS[day.weekday()]


def add_to_date(day: _dt.date, *, days: int = 0, months: int = 0, years: int = 0) -> _dt.date:
    """``day`` moved by whole years and months (the day kept, or the month's last if shorter), then days."""
    month_index = day.year * 12 + (day.month - 1) + years * 12 + months
    year, month = divmod(month_index, 12)
    month += 1
    following = _dt.date(year + (month == 12), month % 12 + 1, 1)
    last = (following - _dt.timedelta(days=1)).day
    return _dt.date(year, month, min(day.day, last)) + _dt.timedelta(days=days)


# ── intervals ───────────────────────────────────────────────────────────────


def merge_intervals(intervals: Iterable[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Overlapping or touching ``(start, end)`` intervals joined, sorted by start."""
    merged: list[list] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def free_intervals(busy: Iterable[tuple[Any, Any]], start: Any, end: Any, *, length: Any = 0) -> list[tuple[Any, Any]]:
    """The stretches of ``[start, end)`` outside every busy interval, each at least ``length`` long."""
    free, cursor = [], start
    for busy_start, busy_end in merge_intervals(busy):
        if busy_end <= cursor:
            continue
        if busy_start >= end:
            break
        if busy_start > cursor:
            free.append((cursor, min(busy_start, end)))
        cursor = max(cursor, busy_end)
    if cursor < end:
        free.append((cursor, end))
    return [(a, b) for a, b in free if b - a >= length and b > a]


def reference() -> str:
    """What this module offers, as signatures and first docstring lines, for a reader of procedures."""
    lines = []
    for name in __all__:
        if name == "reference":
            continue
        thing = globals()[name]
        if callable(thing):
            doc = (inspect.getdoc(thing) or "").splitlines()[0] if inspect.getdoc(thing) else ""
            signature = str(inspect.signature(thing)).replace("'", "").replace("_dt.", "datetime.")
            lines.append(f"mechanisms.{name}{signature}\n    {doc}")
        else:
            lines.append(f"mechanisms.{name} = {thing!r}")
    return "\n".join(lines)
