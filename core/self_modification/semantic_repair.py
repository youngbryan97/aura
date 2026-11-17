"""Semantic program repair: a change computed from what her program did, not guessed (Master Todo F16, part 2).

Her own Python's fix templates (fix_templates.py) change a line in the ways one-line fixes usually go and let the
tests decide; on the faults of her own history they mended none (docs/evidence/SELF_REPAIR_ON_HER_HISTORY_2026-10-19.md).
Semantic repair works from evidence instead (SemFix, Nguyen et al., ICSE 2013; Nopol, Xuan et al., TSE 2017;
Angelix, Mechtaev et al., ICSE 2016):

1. Where: the lines her failing tests run and her passing tests do not, most suspicious first (fix_templates.ochiai).
   At each is a place a repair can be: the condition of a branch (``if``, ``while``, a conditional expression, a
   comprehension's filter, a returned test), a statement that may need a guard, a statement before which the
   function may need to return, or a value read there (an argument, an assigned or returned value) that may need to
   be another expression the function already reads.
2. What her program did there: a copy of the file is instrumented with calls to semantic_probe.py, and her failing
   and passing tests are run on it. Each time a place is reached, the values in scope and what was decided are kept.
3. What it should have done: each place is forced (the branch the other way, the statement skipped, the function
   left early with one of the values it already returns, the value read as another of the function's expressions of
   its type) while her failing tests run. A place where forcing makes
   every failing test pass is one a repair there can mend; what was forced is what the failing runs needed, and what
   the passing runs decided is what they need to keep (angelic fix localisation).
4. The smallest change that does both: condition_synthesis.conditions over the recorded values, the original
   condition with one part changed first, then the original joined with a missing test, then new conditions. A value
   is replaced outright, and only where no passing test that reaches it needs the old one.

Each repair is a SEARCH/REPLACE change on the original file, and none is taken for right: the tests decide, with the
guards and standards of repair_by_selection.py. A guard is never put around an exit (``return``, ``raise``,
``assert``) or a statement that decides control, for the reason fix_templates.py gives for dropping one: it removes
what the test reached instead of mending it.
"""
from __future__ import annotations

import ast
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.self_modification.condition_synthesis import Example, ranked_conditions

logger = logging.getLogger("SelfModification.SemanticRepair")

__all__ = ["Place", "RunForcings", "her_own_python", "instrumented", "places_at", "semantic_repairs"]

PROBE = "__aura_probe__"
CHOICE = "_aura_choice_"
#: Statements a guard may be put around: what a function does, not where it ends or which way it goes.
_GUARDABLE = (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)
#: Parts of a condition read on their own: those with no call but ``len``.
_PURE = (ast.Name, ast.Attribute, ast.Constant, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Call)
#: Values a function may be left with early, beside the ones it already returns.
_EMPTY_RETURNS = ("None",)
#: The values a function already returns that are offered for leaving it early.
RETURNS_OFFERED = 4

Run = Callable[[Sequence[str], Mapping[str, str]], Awaitable[tuple[dict[str, bool], str]]]
#: ``run_forcings(tests, sources, spec)`` runs ``tests`` on a copy holding ``sources`` in one session with the plans in
#: ``spec`` (forcing_session.py).
RunForcings = Callable[[Sequence[str], Mapping[str, str], Path], Awaitable[None]]


@dataclass(frozen=True)
class Place:
    """Somewhere on a line a repair can be made."""

    id: str
    kind: str  # "cond", "guard", "leave" or "value"
    path: str
    line: int
    #: For a condition, the source of the condition and its span in characters.
    text: str = ""
    start: int = 0
    end: int = 0
    #: For a condition, the parts of it read on their own.
    parts: tuple[str, ...] = ()
    #: For a guard or a leave, the statement's first and last line, column, kind and indentation.
    first: int = 0
    last: int = 0
    col: int = 0
    statement: str = ""
    indent: str = ""
    #: For a leave, the values the function may be left with.
    returns: tuple[str, ...] = ()
    #: Numbers written in the enclosing function: bounds a condition there may use.
    constants: tuple[Any, ...] = ()
    #: The names the code at the place reads: a condition over them is the likelier repair.
    names: tuple[str, ...] = ()
    #: For a value, the function's other expressions that may be read in its place, nearest first, and which of them
    #: are read without a call (their types are recorded; a call's is learnt only by making it).
    candidates: tuple[str, ...] = ()
    pure: tuple[str, ...] = ()


def _starts(source: str) -> list[int]:
    starts, at = [0], 0
    for line in source.splitlines(keepends=True):
        at += len(line)
        starts.append(at)
    return starts


def _offset(source: str, starts: list[int], line: int, col: int) -> int:
    text = source[starts[line - 1]:starts[line]] if line < len(starts) else source[starts[line - 1]:]
    return starts[line - 1] + len(text.encode("utf-8")[:col].decode("utf-8", errors="ignore"))


def _functions(tree: ast.AST) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _enclosing(functions: Sequence[ast.AST], line: int) -> ast.AST | None:
    holders = [f for f in functions if f.lineno <= line <= (f.end_lineno or f.lineno)]
    return max(holders, key=lambda f: f.lineno) if holders else None


def _pure(node: ast.AST) -> bool:
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and not (isinstance(inner.func, ast.Name) and inner.func.id == "len"
                                                and len(inner.args) == 1 and not inner.keywords):
            return False
        if not isinstance(inner, (*_PURE, ast.expr_context, ast.cmpop, ast.boolop, ast.unaryop, ast.operator)):
            return False
    return True


def _parts(test: ast.AST) -> tuple[str, ...]:
    """The pure parts of a condition, the condition itself left out, each once."""
    seen = []
    for node in ast.walk(test):
        if node is test or not isinstance(node, ast.expr) or isinstance(node, ast.Constant):
            continue
        if _pure(node):
            text = ast.unparse(node)
            if text not in seen:
                seen.append(text)
    return tuple(seen[:8])


def _constants(function: ast.AST | None) -> tuple[Any, ...]:
    if function is None:
        return ()
    values = {n.value for n in ast.walk(function) if isinstance(n, ast.Constant)
              and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    return tuple(sorted(values)[:12])


def _returns(function: ast.AST) -> tuple[str, ...]:
    """The plain values the function already returns, and None."""
    found: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if isinstance(value, ast.Constant) or (isinstance(value, (ast.List, ast.Dict, ast.Tuple, ast.Set))
                                                  and not getattr(value, "elts", getattr(value, "keys", None))):
                found.append(ast.unparse(value))
    return tuple(dict.fromkeys([*found[:RETURNS_OFFERED], *_EMPTY_RETURNS]))


def _read(node: ast.AST) -> tuple[str, ...]:
    """The names ``node`` reads, each once."""
    return tuple(dict.fromkeys(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)))


def _a_test(value: ast.AST | None) -> bool:
    """Whether a returned value is itself a test: a comparison, a conjunction or disjunction, or a negation."""
    return isinstance(value, (ast.Compare, ast.BoolOp)) or (isinstance(value, ast.UnaryOp)
                                                           and isinstance(value.op, ast.Not))


def _yields(function: ast.AST) -> bool:
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(function))


def _one_block_string(node: ast.AST) -> bool:
    """Whether a string in the statement spans lines, so that indenting its lines would change it."""
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and (n.end_lineno or n.lineno) != n.lineno
               for n in ast.walk(node))


#: What may not be in a value or an ingredient: what cannot be read again inside a lambda, or reads a scope of its own.
_NOT_READ_AGAIN = (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Lambda, ast.ListComp, ast.SetComp,
                   ast.DictComp, ast.GeneratorExp, ast.Starred)


def _readable_again(node: ast.AST) -> bool:
    return not any(isinstance(inner, _NOT_READ_AGAIN) for inner in ast.walk(node))


def _a_path(node: ast.AST) -> bool:
    """A name or an attribute of one: ``x``, ``x.a.b``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _an_ingredient(node: ast.AST, called: set[int]) -> bool:
    """An expression a fix may reuse in a value's place: a path, a call of a path with paths and constants for its
    arguments, or arithmetic over those; never the function a call names."""
    if id(node) in called or not _readable_again(node):
        return False
    if _a_path(node):
        return True
    if isinstance(node, ast.Call):
        return _a_path(node.func) and all(_a_path(a) or isinstance(a, ast.Constant) for a in node.args) and all(
            k.arg is not None and (_a_path(k.value) or isinstance(k.value, ast.Constant)) for k in node.keywords)
    if isinstance(node, ast.BinOp):
        return all(isinstance(side, ast.Constant) or _an_ingredient(side, called) for side in (node.left, node.right))
    return False


def _ingredients(function: ast.AST, target: ast.AST) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The function's expressions that may be read in ``target``'s place, nearest line first, each once; and those of
    them read without a call."""
    called = {id(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)}
    # Nor anything holding the value itself: the call it is an argument of is not a value for that argument.
    called |= {id(n) for n in ast.walk(function) if n is not target and any(m is target for m in ast.walk(n))}
    own = ast.unparse(target)
    found = sorted((abs(n.lineno - target.lineno), n.lineno, n.col_offset, ast.unparse(n)) for n in ast.walk(function)
                   if isinstance(n, ast.expr) and not isinstance(n, ast.Constant) and hasattr(n, "lineno")
                   and isinstance(getattr(n, "ctx", ast.Load()), ast.Load) and _an_ingredient(n, called))
    # And its parameters, in scope everywhere in it whether or not its body reads them.
    given = function.args  # type: ignore[attr-defined]
    found += sorted((abs(function.lineno - target.lineno), function.lineno, 0, a.arg)  # type: ignore[attr-defined]
                    for a in (*given.posonlyargs, *given.args, *given.kwonlyargs) if a.arg not in ("self", "cls"))
    texts = tuple(dict.fromkeys(text for *_, text in sorted(found) if text != own))
    pure = tuple(t for t in texts if _pure(ast.parse(t, mode="eval").body)
                 and not any(isinstance(n, ast.Call) for n in ast.walk(ast.parse(t, mode="eval").body)))
    # What can be read without a call first, then calls of nothing, then calls with arguments: each is tried in turn,
    # and a read value's type and value say before a run whether it can be what the failing runs needed.
    def cost(text: str) -> int:
        calls = [n for n in ast.walk(ast.parse(text, mode="eval").body) if isinstance(n, ast.Call)]
        return 0 if text in pure else 1 if all(not c.args and not c.keywords for c in calls) else 2
    return tuple(sorted(texts, key=cost)), pure


def _values_read(node: ast.AST) -> list[ast.AST]:
    """The values a statement or call on a line reads that a fix may replace: a call's arguments, an assigned value,
    a returned value that is not itself a test."""
    if isinstance(node, ast.Call):
        return [*(a for a in node.args if not isinstance(a, ast.Starred)), *(k.value for k in node.keywords)]
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        return [node.value]
    if isinstance(node, ast.Return) and node.value is not None and not _a_test(node.value):
        return [node.value]
    return []


def places_at(source: str, path: str, lines: Sequence[int]) -> list[Place]:
    """Every place on ``lines`` of ``source`` a repair can be made, in the order the lines were given."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    starts = _starts(source)
    functions = _functions(tree)
    rows = source.splitlines()
    wanted = list(dict.fromkeys(int(n) for n in lines))
    places: list[Place] = []
    valued: set[tuple[int, int]] = set()
    by_line: dict[int, list[ast.AST]] = {}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line in wanted:
            by_line.setdefault(line, []).append(node)
    for line in wanted:
        function = _enclosing(functions, line)
        constants = _constants(function)
        for node in by_line.get(line, ()):
            tests: list[ast.AST] = []
            if isinstance(node, (ast.If, ast.While, ast.IfExp)):
                tests.append(node.test)
            elif isinstance(node, ast.comprehension):
                tests.extend(node.ifs)
            elif isinstance(node, ast.Return) and _a_test(node.value):
                tests.append(node.value)
            for test in tests:
                if isinstance(test, ast.Constant) or function is None:
                    continue
                start = _offset(source, starts, test.lineno, test.col_offset)
                end = _offset(source, starts, test.end_lineno or test.lineno, test.end_col_offset or 0)
                places.append(Place(f"c{len(places)}", "cond", path, line, text=source[start:end], start=start,
                                    end=end, parts=_parts(test), constants=constants, names=_read(test)))
            for read in _values_read(node) if function is not None else ():
                if not _readable_again(read) or (read.lineno, read.col_offset) in valued:
                    continue
                valued.add((read.lineno, read.col_offset))
                candidates, pure = _ingredients(function, read)
                if not candidates:
                    continue
                start = _offset(source, starts, read.lineno, read.col_offset)
                end = _offset(source, starts, read.end_lineno or read.lineno, read.end_col_offset or 0)
                places.append(Place(f"v{len(places)}", "value", path, line, text=source[start:end], start=start,
                                    end=end, names=_read(read), candidates=candidates, pure=pure))
            if function is None or not isinstance(node, ast.stmt) or node is function:
                continue
            first, last = node.lineno, node.end_lineno or node.lineno
            indent = rows[first - 1][: len(rows[first - 1]) - len(rows[first - 1].lstrip())]
            if rows[first - 1].lstrip() != rows[first - 1][node.col_offset:]:
                continue  # not the first thing on its line (a statement after ";"): left alone
            where = {"first": first, "last": last, "col": node.col_offset, "statement": type(node).__name__,
                     "indent": indent, "constants": constants, "names": _read(node)}
            if isinstance(node, _GUARDABLE) and not _one_block_string(node):
                places.append(Place(f"g{len(places)}", "guard", path, line, **where))
            async_generator = isinstance(function, ast.AsyncFunctionDef) and _yields(function)
            # Never before a raise: leaving there turns a refusal into a return, which is how a repair deletes
            # what the test reached instead of mending it (fix_templates.py).
            if not async_generator and not _yields(function) and not isinstance(node, ast.Raise):
                places.append(Place(f"l{len(places)}", "leave", path, line, returns=_returns(function), **where))
    return places


class _Instrument(ast.NodeTransformer):
    def __init__(self, places: Sequence[Place], tree: ast.AST, source: str, plan: str, record: str) -> None:
        starts = _starts(source)
        self.tests: dict[int, Place] = {}
        self.values: dict[int, Place] = {}
        self.before: dict[int, list[Place]] = {}
        self.around: dict[int, Place] = {}
        self.plan, self.record = plan, record
        for node in ast.walk(tree):
            if isinstance(node, ast.expr) and hasattr(node, "lineno"):
                start = _offset(source, starts, node.lineno, node.col_offset)
                for place in places:
                    if place.kind in ("cond", "value") and place.start == start \
                            and ast.unparse(node) == ast.unparse(ast.parse(place.text, mode="eval").body):
                        (self.tests if place.kind == "cond" else self.values)[id(node)] = place
            elif isinstance(node, ast.stmt) and hasattr(node, "lineno"):
                for place in places:
                    if (place.first, place.col, place.statement) == (node.lineno, node.col_offset,
                                                                     type(node).__name__):
                        if place.kind == "guard":
                            self.around[id(node)] = place
                        elif place.kind == "leave":
                            self.before.setdefault(id(node), []).append(place)

    def _call(self, name: str, place: Place, *middle: ast.expr) -> ast.Call:
        return ast.Call(func=ast.Attribute(value=ast.Name(id=PROBE, ctx=ast.Load()), attr=name, ctx=ast.Load()),
                        args=[ast.Constant(place.id), *middle, ast.Call(func=ast.Name("locals", ast.Load()), args=[],
                                                                        keywords=[]),
                              ast.Constant(self.plan), ast.Constant(self.record)], keywords=[])

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for name, value in ast.iter_fields(node):
            if isinstance(value, list):
                out: list[Any] = []
                for item in value:
                    if isinstance(item, ast.AST):
                        visited = self.visit(item)
                        for one in visited if isinstance(visited, list) else [visited]:
                            out.append(one)
                    else:
                        out.append(item)
                setattr(node, name, out)
            elif isinstance(value, ast.AST):
                setattr(node, name, self.visit(value))
        return node

    @staticmethod
    def _lambda(body: ast.expr) -> ast.Lambda:
        return ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                          body=body)

    def visit(self, node: ast.AST) -> Any:
        valued = self.values.get(id(node))
        if valued is not None:
            node = super().visit(node)
            call = self._call("value", valued, self._lambda(node))
            call.args.append(ast.Dict(keys=[ast.Constant(text) for text in valued.candidates],
                                      values=[self._lambda(ast.parse(text, mode="eval").body)
                                              for text in valued.candidates]))
            call.args.append(ast.Tuple(elts=[ast.Constant(text) for text in valued.pure], ctx=ast.Load()))
            return call
        place = self.tests.get(id(node))
        if place is not None:
            node = super().visit(node)
            parts = ast.Dict(keys=[ast.Constant(text) for text in place.parts],
                             values=[ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                                   kw_defaults=[], defaults=[]),
                                                body=ast.parse(text, mode="eval").body) for text in place.parts])
            call = self._call("cond", place, node)
            call.args.append(parts)
            return call
        leaves = self.before.get(id(node), [])
        wrapped = self.around.get(id(node))
        node = super().visit(node)
        out: list[ast.stmt] = []
        for leave in leaves:
            if not leave.returns:
                continue
            choice = ast.NamedExpr(target=ast.Name(CHOICE, ast.Store()), value=self._call("leave", leave))
            values = ast.Tuple(elts=[ast.parse(v, mode="eval").body for v in leave.returns], ctx=ast.Load())
            out.append(ast.If(test=ast.Compare(left=choice, ops=[ast.GtE()], comparators=[ast.Constant(0)]),
                              body=[ast.Return(value=ast.Subscript(value=values, slice=ast.Name(CHOICE, ast.Load()),
                                                                   ctx=ast.Load()))], orelse=[]))
        if wrapped is not None:
            node = ast.If(test=self._call("guard", wrapped), body=[node], orelse=[])
        if out:
            return [*out, node]
        return node


def instrumented(source: str, places: Sequence[Place], *, plan: Path, record: Path) -> str:
    """``source`` with each place calling the probe: the code it runs is otherwise the same."""
    tree = ast.parse(source)
    transformer = _Instrument(places, tree, source, str(plan), str(record))
    tree = transformer.visit(tree)
    # The probe is loaded from this checkout's file, not imported from the package being repaired: a defect from her
    # history is repaired in a copy of her code as it was, where the probe did not yet exist.
    from core.self_modification import semantic_probe

    loader = ast.parse(
        "import importlib.util as _aura_util_\n"
        "import sys as _aura_sys_\n"
        f"{PROBE} = _aura_sys_.modules.get('aura_semantic_probe')\n"
        f"if {PROBE} is None:\n"
        f"    _aura_spec_ = _aura_util_.spec_from_file_location('aura_semantic_probe', {semantic_probe.__file__!r})\n"
        f"    {PROBE} = _aura_util_.module_from_spec(_aura_spec_)\n"
        f"    _aura_sys_.modules['aura_semantic_probe'] = {PROBE}\n"
        f"    _aura_spec_.loader.exec_module({PROBE})\n").body
    body = list(tree.body)
    at = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        at = 1
    while at < len(body) and isinstance(body[at], ast.ImportFrom) and body[at].module == "__future__":
        at += 1
    body[at:at] = loader
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree))


def _block(path: str, old: str, new: str) -> str:
    return f"{path}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE"


def _whole(path: str, before: str, after: str) -> str:
    from core.self_modification.proposals_without_a_model import _whole as whole

    return whole(path, before, after)


def _indent_unit(source: str) -> str:
    for row in source.splitlines():
        stripped = row.lstrip(" ")
        if stripped and row != stripped:
            width = len(row) - len(stripped)
            if width % 4 == 0:
                return "    "
            return " " * width
    return "    "


def repaired(source: str, place: Place, condition: str, value: str = "") -> str:
    """``source`` with ``condition`` put in at ``place``: in place of the condition, as a guard, or as a way out that
    returns ``value``."""
    if place.kind in ("cond", "value"):
        return source[:place.start] + condition + source[place.end:]
    rows = source.splitlines(keepends=True)
    unit = _indent_unit(source)
    if place.kind == "guard":
        body = "".join(unit + row if row.strip() else row for row in rows[place.first - 1:place.last])
        return "".join([*rows[:place.first - 1], f"{place.indent}if {condition}:\n", body, *rows[place.last:]])
    return "".join([*rows[:place.first - 1], f"{place.indent}if {condition}:\n", f"{place.indent}{unit}return {value}\n",
                    *rows[place.first - 1:]])


@dataclass
class Evidence:
    """What the recording and the forcing showed, kept for the receipt."""

    places: list[dict[str, Any]] = field(default_factory=list)
    runs: int = 0
    stopped: str = ""
    #: Forcings run in sessions of many (forcing_session.py), and the sessions they took.
    forcings: int = 0
    sessions: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"places": self.places, "runs": self.runs, "stopped": self.stopped, "forcings": self.forcings,
                "sessions": self.sessions}


def _plan_and_clear(plan: Path, record: Path, steps: Mapping[str, Any]) -> None:
    """Write the plan the probe reads and empty the record it writes, in a scratch folder of the search's own."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    with local_internal_governed_scope("semantic_repair.plan", domain="file_write",
                                       constraints={"artifact": "aura.semantic_repair.plan.v1"}):
        # The atomic writer makes a missing parent itself.
        gateway.write_text(plan, json.dumps(dict(steps)), source="semantic_repair")
        gateway.delete_file(record, source="semantic_repair")


def _rows(record: Path) -> list[dict[str, Any]]:
    if not record.exists():
        return []
    out = []
    for line in record.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _test_of(row: Mapping[str, Any]) -> str:
    return str(row.get("test") or "")


def _matches(node: str, test: str) -> bool:
    """Whether pytest's id of a running test names the node a test list gives, its parametrised ids included.

    The file part is compared only when both have one: the id pytest gives a running test is relative to where it
    was started, and in a folder of its own that can leave the file part empty ("::test_name").
    """
    if test == node or test.startswith(node + "["):
        return True
    if "::" not in node:
        return test.split("::")[0] == node
    node_file, _, node_name = node.partition("::")
    test_file, _, test_name = test.partition("::")
    if node_file and test_file and node_file != test_file:
        return False
    return test_name == node_name or test_name.startswith(node_name + "[")


async def semantic_repairs(*, sources: Mapping[str, str], failing: Sequence[str], passing: Sequence[str],
                           ranked: Sequence[tuple[tuple[str, int], float]], run: Run, workdir: Path,
                           deadline: float, lines: int = 15, evidence: Evidence | None = None,
                           run_forcings: RunForcings | None = None) -> AsyncIterator[tuple[str, str]]:
    """Each repair the recorded and forced runs support, as (origin, SEARCH/REPLACE text), the likeliest first.

    ``run(tests, sources)`` runs tests on a copy holding ``sources``. ``failing`` are the defect's tests, ``passing``
    tests that pass now and run the suspect lines. ``ranked`` is fix_templates.ochiai's order; its first ``lines``
    lines with a score are looked at. With ``run_forcings``, the forcings run ahead in sessions of many, and one that
    mends there is run again on its own before it is used.
    """
    import asyncio

    evidence = evidence if evidence is not None else Evidence()
    plan, record = workdir / "plan.json", workdir / "record.jsonl"
    chosen: dict[str, list[int]] = {}
    for (path, line), score in ranked:
        if score > 0 and path in sources and sum(len(v) for v in chosen.values()) < lines:
            chosen.setdefault(path, []).append(line)
    places = [p for path, at in chosen.items() for p in places_at(sources[path], path, at)]
    if not places:
        evidence.stopped = "no place for a repair on the lines the failing tests run"
        return
    probed: dict[str, str] = {}
    for path in {p.path for p in places}:
        try:
            probed[path] = instrumented(sources[path], [p for p in places if p.path == path], plan=plan,
                                        record=record)
        except (SyntaxError, ValueError, TypeError) as exc:
            logger.info("%s could not be instrumented: %s", path, exc)
    places = [p for p in places if p.path in probed]
    changed = {**sources, **probed}
    await asyncio.to_thread(_plan_and_clear, plan, record, {})
    tests = list(dict.fromkeys([*failing, *passing]))
    seen, _ = await run(tests, changed)
    evidence.runs += 1
    held = [t for t in passing if seen.get(t)]
    if any(seen.get(t) for t in failing):
        evidence.stopped = "a failing test passed under the probe, so what it records is not the fault"
        return
    unforced = await asyncio.to_thread(_rows, record)
    gathered: list[tuple[tuple[int, int, int], Place, str]] = []
    notes: dict[str, tuple[dict[str, Any], list[dict[str, Any]], int]] = {}
    queues: list[tuple[Place, list[tuple[dict[str, Any], Any]]]] = []
    for order, place in enumerate(places):
        kept_runs = [r for r in unforced if r.get("location") == place.id
                     and any(_matches(t, _test_of(r)) for t in held)]
        reached_failing = any(r.get("location") == place.id and any(_matches(t, _test_of(r)) for t in failing)
                              for r in unforced)
        note: dict[str, Any] = {"place": place.id, "kind": place.kind, "path": place.path, "line": place.line,
                                "text": place.text, "reached_by_failing": reached_failing,
                                "passing_runs": len(kept_runs)}
        evidence.places.append(note)
        notes[place.id] = (note, kept_runs, order)
        if reached_failing:
            queues.append((place, _forcings(place, unforced, failing)))
    # Every decision first, a place at a time: each has a few ways to be forced. Then the values, the nearest candidate
    # of every value before the next of any: a value has as many candidates as its function has expressions, and the
    # one a fix reuses is most often near it. LIVE 17 November 2026 the run on f5da1f5da spent its ten minutes on the
    # candidates of three values, place by place, and never reached the fourth, whose third candidate was the fix.
    deepest = max((len(steps) for place, steps in queues if place.kind == "value"), default=0)
    schedule = [(place, step, option) for place, steps in queues if place.kind != "value" for step, option in steps]
    schedule += [(place, *steps[depth]) for depth in range(deepest) for place, steps in queues
                 if place.kind == "value" and depth < len(steps)]
    done: set[str] = set()
    ahead: dict[int, bool] = {}

    async def a_session_from(first: int) -> None:
        """The scheduled forcings from ``first`` on, of places not yet mended, in one session; what each came to."""
        from core.self_modification.forcing_session import read_results, write_spec

        indexes = [k for k in range(first, len(schedule)) if schedule[k][0].id not in done]
        spec, results = workdir / f"session-{first}.json", workdir / f"session-{first}.jsonl"
        await asyncio.to_thread(_plan_and_clear, plan, record, {})
        await asyncio.to_thread(write_spec, spec, plan=plan, results=results, deadline=deadline,
                                steps=[{schedule[k][0].id: schedule[k][1]} for k in indexes])
        await run_forcings(list(failing), changed, spec)
        got = await asyncio.to_thread(read_results, results)
        evidence.sessions += 1
        evidence.forcings += len(got)
        for k, verdicts in zip(indexes, got, strict=False):
            ahead[k] = all(any(_matches(t, node) and ok for node, ok in verdicts.items()) for t in failing)
        if len(got) < len(indexes):
            ahead[indexes[len(got)]] = False  # it hung or ended the session: not a forcing that mends

    async def mends(index: int, place: Place, step: Mapping[str, Any]) -> bool:
        if run_forcings is not None:
            if index not in ahead:
                await a_session_from(index)
            if not ahead.get(index):
                return False
        await asyncio.to_thread(_plan_and_clear, plan, record, {place.id: step})
        outcome, _ = await run(list(failing), changed)
        evidence.runs += 1
        return all(outcome.get(t) for t in failing)

    for index, (place, step, option) in enumerate(schedule):
        if place.id in done:
            continue
        if time.time() >= deadline:
            evidence.stopped = "deadline"
            break
        note, kept_runs, order = notes[place.id]
        if not await mends(index, place, step):
            note.setdefault("forced_without_effect", []).append(step)
            continue
        note.setdefault("angelic", []).append(step)
        if place.kind == "value":
            # Replaced outright, so every passing test that reaches it reads the new value too: one that then fails
            # needs the old value, and this expression is not the fix (a condition choosing between the two would be
            # another kind of change).
            depends, ran = await _the_tests_that_depend_on(place, step, held, kept_runs, plan, record, changed, run)
            evidence.runs += ran
            if depends:
                note.setdefault("breaks_passing", []).append(option)
                continue
            after = repaired(sources[place.path], place, option)
            try:
                ast.parse(after)
            except SyntaxError:
                continue
            gathered.append(((1, order, len(option)), place, after))
            done.add(place.id)
            continue
        forced_rows = [r for r in await asyncio.to_thread(_rows, record) if r.get("location") == place.id]
        # A passing run constrains the condition only when its test depends on what is decided here: under the same
        # forcing it fails. One that passes either way says nothing about this place. LIVE 16 November 2026 the
        # drill's own test reached count_in('  ') and passed whether it raised or not, so its decision there
        # contradicted the failing run's at the same value and no condition could be found (Angelix keeps a passing
        # test's outcome, not every decision it happened to pass through).
        depends, ran = await _the_tests_that_depend_on(place, step, held, kept_runs, plan, record, changed, run)
        evidence.runs += ran
        note["passing_that_depend"] = len(depends)
        kept_here = [r for r in kept_runs if any(_matches(t, _test_of(r)) and "::" in t for t in depends)]
        examples = [Example(r.get("terms") or {}, want=_wanted(place, r), decided=r.get("decided"),
                            parts=r.get("parts")) for r in forced_rows]
        examples += [Example(r.get("terms") or {}, want=_kept(place, r), decided=r.get("decided"),
                             parts=r.get("parts")) for r in kept_here]
        original = place.text if place.kind == "cond" else ""
        found = ranked_conditions(examples, original=original, constants=place.constants, names=place.names)
        note.setdefault("synthesised", []).extend(text for _, text in found)
        for cost, condition in found:
            value = place.returns[option] if place.kind == "leave" else ""
            after = repaired(sources[place.path], place, condition, value)
            try:
                ast.parse(after)
            except SyntaxError:
                continue
            gathered.append(((cost, order, len(condition)), place, after))
        if found:
            done.add(place.id)  # the first forcing that mends is the outcome the failing runs needed here
    # Every place's repairs, cheapest first across places: a repair that changes less, over the names the faulty
    # code reads, is offered before one that pins a run at an earlier line.
    for _, place, after in sorted(gathered, key=lambda item: item[0]):
        yield f"semantic:{place.kind}:{place.path}:{place.line}", _whole(place.path, sources[place.path], after)
    evidence.stopped = evidence.stopped or "places spent"


async def _the_tests_that_depend_on(place: Place, step: Any, held: Sequence[str],
                                   kept_runs: Sequence[Mapping[str, Any]], plan: Path, record: Path,
                                   changed: Mapping[str, str], run: Run) -> tuple[list[str], int]:
    """The passing tests that reached ``place`` and fail when it is forced as ``step`` (the ones whose outcome
    depends on what is decided there), by the test ids the runs recorded, and how many test runs that took.

    Each test is asked on its own: passing tests are often given as a file, and a file holding one test that depends
    on the place would make every other test in it count as depending too."""
    import asyncio

    def runnable(test: str) -> str:
        if not test.startswith("::"):
            return test
        file = next((h for h in held if "::" not in h and _matches(h, test)), "")
        return f"{file}{test}" if file else ""

    reached = sorted({runnable(_test_of(r)) for r in kept_runs} - {""})
    if not reached:
        return [], 0
    await asyncio.to_thread(_plan_and_clear, plan, record, {place.id: step})
    outcome, _ = await run(reached, changed)
    return [t for t in reached if not outcome.get(t)], 1


def _forcings(place: Place, unforced: Sequence[Mapping[str, Any]],
              failing: Sequence[str]) -> list[tuple[dict[str, Any], Any]]:
    """The ways to force a place while the failing tests run, as (plan step, option), the bluntest first.

    A condition is forced to each value for every evaluation; then, when that cannot mend it, one evaluation at a
    time decided the other way, for as many evaluations as the failing tests made of it (a bound in a loop is wrong
    for one item and right for the rest). A guard skips its statement; a way out returns each value offered; a value
    is read as each of its candidates.
    """
    if place.kind == "guard":
        return [({"decide": False}, False)]
    if place.kind == "leave":
        return [({"decide": index}, index) for index in range(len(place.returns))]
    reached = [r for r in unforced if r.get("location") == place.id
               and any(_matches(t, _test_of(r)) for t in failing)]
    if place.kind == "value":
        # Each candidate in its turn, nearest first, but not one read without a call that held another type wherever
        # the failing runs reached the place: it cannot be the value they needed.
        had = {r.get("decided") for r in reached if r.get("decided")}
        out_values: list[tuple[dict[str, Any], Any]] = []
        for text in place.candidates:
            kinds = {(r.get("types") or {}).get(text) for r in reached} - {None}
            if had and kinds and not kinds & had:
                continue
            if reached and all(text in (r.get("same") or ()) for r in reached):
                continue  # the same value wherever the failing runs read it: reading it instead changes nothing
            out_values.append(({"pick": text}, text))
        return out_values
    decided = [bool(r.get("decided")) for r in reached]
    out: list[tuple[dict[str, Any], Any]] = []
    for value in (True, False):
        if not (decided and all(d == value for d in decided)):  # forcing what it already decides changes nothing
            out.append(({"decide": value}, value))
    tests = list(dict.fromkeys(_test_of(r) for r in reached))
    evaluations = max((int(r.get("n") or 0) for r in reached), default=0)
    for n in range(1, evaluations + 1):
        out.append(({"flip": {test: [n] for test in tests}}, n))
    return out


def _wanted(place: Place, row: Mapping[str, Any]) -> bool:
    """What a failing run needed at one evaluation, as the truth of the condition that will be written there."""
    forced = row.get("forced")
    if place.kind == "cond":
        return bool(row.get("decided")) if forced is None else bool(forced)
    if place.kind == "guard":
        return forced is None  # skipped where it was forced
    return forced is not None and int(forced) >= 0  # left where it was forced


def _kept(place: Place, row: Mapping[str, Any]) -> bool:
    """What a passing run decided there, as the same."""
    if place.kind == "cond":
        return bool(row.get("decided"))
    if place.kind == "guard":
        return True
    return False


Rank = Callable[[Mapping[str, str]], Awaitable[Sequence[tuple[tuple[str, int], float]]]]
Passing = Callable[[Mapping[str, str], Sequence[tuple[tuple[str, int], float]]], Awaitable[Sequence[str] | None]]
Propose = Callable[..., Awaitable[list[tuple[str, str]]]]


def her_own_python(*, repo: Path, rank: Rank, run: Run | None = None, passing: Passing | None = None,
                   workdir: Path | None = None, deadline: Callable[[], float] | None = None, error_type: str = "",
                   then: Propose | None = None, evidence: Evidence | None = None,
                   run_forcings: RunForcings | None = None) -> Propose:
    """A proposer for repair_by_selection, her own Python's in four stages, each exact and each judged by the tests:

    1. the exact changes of proposals_without_a_model (a missing import, a shape almost never meant, ruff's fixes);
    2. semantic repair from her program's values at the ranked lines (this module), when ``run`` is given;
    3. the fix templates at the ranked lines (fix_templates.py);
    4. ``then``'s proposals, when one is given.

    ``rank(start)`` gives the lines most suspicious first; ``passing(start, ranked)`` the tests that pass now and run
    them, which semantic repair keeps deciding as they did; ``deadline()`` when to stop forcing.
    """
    from core.self_modification.proposals_without_a_model import (
        proposals_without_a_model,
        template_proposals,
    )

    stages: dict[str, dict[str, Any]] = {}

    async def propose(defect: Any, start: Mapping[str, str], parent: Any, n: int) -> list[tuple[str, str]]:
        key = parent.id if parent is not None else ""
        if key not in stages:
            failure = parent.fitness.failures if parent is not None and parent.fitness else defect.seen
            ranked = list(await rank(start))
            stages[key] = {"ranked": ranked, "semantic": None,
                           "exact": iter(proposals_without_a_model(failure, start, repo, ranked, error_type,
                                                                   templates=False)),
                           "templates": iter(template_proposals(start, ranked, error_type))}
        stage = stages[key]
        offered = next(stage["exact"], None)
        if offered is not None:
            return [offered]
        # Semantic repair on the defect's own code only: a child of a candidate already carries a change, and its
        # values would be read on code nobody wrote.
        if key == "" and run is not None and workdir is not None and stage["semantic"] is None:
            tests = list(await passing(start, stage["ranked"]) or []) if passing is not None else []
            stage["semantic"] = semantic_repairs(
                sources=start, failing=list(defect.tests), passing=tests, ranked=stage["ranked"], run=run,
                workdir=workdir, deadline=deadline() if deadline is not None else time.time() + 600,
                evidence=evidence, run_forcings=run_forcings)
        if stage["semantic"]:
            offered = await anext(stage["semantic"], None)
            if offered is not None:
                return [offered]
            stage["semantic"] = False
        offered = next(stage["templates"], None)
        if offered is not None:
            return [offered]
        return await then(defect, start, parent, n) if then is not None else []

    return propose
