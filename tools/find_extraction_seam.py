#!/usr/bin/env python3
"""Find and score the extractable seams in an oversized function.

Cutting `_api_chat_turn` down took one afternoon of analysis and about four
minutes of editing, and almost all of the analysis was mechanical: which names
does this block read from the enclosing scope, which does it hand back, how
many early returns, does it await, is anything read before it is stored. That
analysis is the same for all 75 tracked functions, so it belongs in a tool
rather than in whoever next opens the file.

What makes a seam safe, in the order that matters:

* **Early returns.** Zero or one is a clean extraction — one needs a single
  optional-response sentinel. Several means general control-flow surgery and
  the seam is not worth taking.
* **Narrow interface.** Few names in, few out. A block that reads twenty locals
  is not a unit, it is a paragraph.
* **Conditionally-bound escapes.** A name bound only inside the block and read
  after it is the trap: returning a default converts a path that raised
  UnboundLocalError into one that quietly proceeds, which is a behaviour change
  wearing a refactor's clothes. The extraction has to hand those back through a
  sentinel, and this tool counts them so nobody discovers it afterwards.
* **Size.** A seam that removes 40 lines from a 4,000-line function is not
  worth the interface it costs.

The scoring is deliberately conservative: seams with multiple returns are
reported and marked unsafe rather than hidden, because "this function has no
clean seam" is a real and useful answer.

Run:
    python tools/find_extraction_seam.py interface/routes/chat.py::_api_chat_turn
    python tools/find_extraction_seam.py --tracked        # every baselined function
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "method_size_baseline.json"

_BUILTINS = set(dir(builtins))


@dataclass
class Seam:
    """One candidate block to lift out, with the contract it would need."""

    lineno: int
    end_lineno: int
    kind: str
    reads: list[str] = field(default_factory=list)
    escapes: list[str] = field(default_factory=list)
    conditional_escapes: list[str] = field(default_factory=list)
    returns: list[int] = field(default_factory=list)
    awaits: int = 0
    yields: int = 0
    jumps: int = 0
    depth: int = 0
    inside_loop: bool = False

    @property
    def lines(self) -> int:
        return self.end_lineno - self.lineno + 1

    @property
    def safe(self) -> bool:
        return (
            len(self.returns) <= 1
            and self.yields == 0
            and self.jumps == 0
            and not self.inside_loop
            and len(self.reads) <= 10
            and len(self.escapes) <= 10
        )

    @property
    def blockers(self) -> list[str]:
        out = []
        if len(self.returns) > 1:
            out.append(
                f"{len(self.returns)} early returns — needs control-flow surgery, "
                "not a sentinel"
            )
        if self.yields:
            out.append("contains yield; the block is a generator body")
        if len(self.reads) > 10:
            out.append(f"reads {len(self.reads)} enclosing names — not a unit")
        if len(self.escapes) > 10:
            out.append(f"hands back {len(self.escapes)} names — not a unit")
        if self.jumps:
            out.append(
                f"{self.jumps} break/continue — the jump targets a loop the "
                "helper would not contain"
            )
        if self.inside_loop:
            out.append("sits inside a loop; the body carries state between turns")
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "lines": self.lines,
            "range": f"{self.lineno}-{self.end_lineno}",
            "kind": self.kind,
            "reads": self.reads,
            "escapes": self.escapes,
            "conditional_escapes": self.conditional_escapes,
            "returns": self.returns,
            "awaits": self.awaits,
            "depth": self.depth,
            "safe": self.safe,
            "blockers": self.blockers,
        }


def _names(node: ast.AST, ctx: type) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _bound_in(node: ast.AST) -> set[str]:
    bound = _names(node, ast.Store)
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
    return bound


def _nested_scope_bindings(node: ast.AST) -> set[str]:
    """Names bound by scopes nested inside the seam, which are never inputs.

    Python opens a new scope for a lambda, a comprehension and a nested def, so
    their parameters and targets are not free variables of the block — they do
    not exist at the call site and passing one as an extraction argument would
    be a NameError, or worse, would silently shadow the real value.

    Patched twice by case before being done properly: comprehension targets
    first, then lambda arguments after ``install_runtime_validation`` reported
    ``p`` and ``o`` — the parameters of ``lambda p, o: ...`` — as inputs it
    needed passed in.
    """
    bound: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in sub.generators:
                bound |= _names(generator.target, ast.Store)
        elif isinstance(sub, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            args = sub.args
            for group in (args.posonlyargs, args.args, args.kwonlyargs):
                bound |= {a.arg for a in group}
            for special in (args.vararg, args.kwarg):
                if special is not None:
                    bound.add(special.arg)
    return bound


class _FlowScan:
    """Free variables and bound names, in evaluation order.

    The previous analysis compared line numbers: a name read on a line above
    its first write was an input, otherwise it was not. That is wrong for the
    assignment shape this file is full of —

        (
            repaired_reply,
            is_stale,
            is_off_topic,
        ) = await _repair(..., stale=is_stale, off_topic=is_off_topic, ...)

    — where the targets sit ABOVE the call that reads them. By line number
    ``is_stale`` is written first and looks local; at runtime the value is
    evaluated first and it is an input. An extraction built on the line-number
    answer produces a helper that raises ``UnboundLocalError`` the moment that
    branch is taken, and the docstring of the function it replaced records
    exactly that happening once already, on ``AuraKernel.tick``.

    This walks statements in the order Python executes them, so an assignment
    reads before it binds, a ``for`` reads its iterable before binding its
    target, and a nested ``def`` contributes its own free variables minus its
    parameters.
    """

    def __init__(self) -> None:
        self.free: set[str] = set()
        self.bound: set[str] = set()

    #: Expressions that open a scope of their own. Their bodies are analysed
    #: separately, with their own bindings.
    _OWN_SCOPE = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def load(self, node: ast.AST | None) -> None:
        """Every name this expression reads, outside its nested scopes.

        ``ast.walk`` was used here and it descends into comprehensions, so a
        comprehension's own target counted as a free variable of the enclosing
        block. Skipping the comprehension NODE while still visiting its
        children skips nothing: the generated helper took ``field=field`` as a
        parameter for a name that only ever existed inside a ``for`` clause,
        and the call site referenced a name that was not in scope.
        """
        if node is None:
            return
        stack: list[ast.AST] = [node]
        nested: list[ast.AST] = []
        while stack:
            current = stack.pop()
            if current is not node and isinstance(current, self._OWN_SCOPE):
                nested.append(current)
                continue
            if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
                if current.id not in self.bound:
                    self.free.add(current.id)
            stack.extend(ast.iter_child_nodes(current))

        if isinstance(node, self._OWN_SCOPE):
            nested.append(node)
        for scope in nested:
            if isinstance(scope, ast.Lambda):
                self._callable(scope.args, [scope.body])
            else:
                self._comprehension(scope)

    def _comprehension(self, node: ast.AST) -> None:
        inner = _FlowScan()
        inner.bound = set(self.bound)
        for generator in node.generators:
            inner.load(generator.iter)
            inner.store(generator.target)
            for condition in generator.ifs:
                inner.load(condition)
        for field in ("elt", "key", "value"):
            inner.load(getattr(node, field, None))
        self.free |= inner.free - self.bound

    def _callable(self, args: ast.arguments, body: list[ast.AST]) -> None:
        for default in [*args.defaults, *[d for d in args.kw_defaults if d]]:
            self.load(default)
        inner = _FlowScan()
        inner.bound = set(self.bound)
        for group in (args.posonlyargs, args.args, args.kwonlyargs):
            inner.bound |= {a.arg for a in group}
        for special in (args.vararg, args.kwarg):
            if special is not None:
                inner.bound.add(special.arg)
        for statement in body:
            if isinstance(statement, ast.stmt):
                inner.statement(statement)
            else:
                inner.load(statement)
        self.free |= inner.free - self.bound

    def store(self, node: ast.AST | None) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                self.bound.add(sub.id)
            elif isinstance(sub, (ast.Attribute, ast.Subscript)) and isinstance(
                sub.ctx, (ast.Store, ast.Del)
            ):
                # `a.b = x` reads `a`.
                self.load(sub.value)

    def body(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            if isinstance(node, ast.AugAssign):
                self.load(node.target)
            self.load(getattr(node, "value", None))
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.store(target)
            else:
                self.store(node.target)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self.load(node.iter)
            self.store(node.target)
            self.body(node.body)
            self.body(node.orelse)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.load(item.context_expr)
                self.store(item.optional_vars)
            self.body(node.body)
            return
        if isinstance(node, ast.Try):
            self.body(node.body)
            for handler in node.handlers:
                self.load(handler.type)
                if handler.name:
                    self.bound.add(handler.name)
                self.body(handler.body)
            self.body(node.orelse)
            self.body(node.finalbody)
            return
        if isinstance(node, (ast.If, ast.While)):
            self.load(node.test)
            self.body(node.body)
            self.body(node.orelse)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                self.load(decorator)
            self._callable(node.args, list(node.body))
            self.bound.add(node.name)
            return
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                self.load(decorator)
            for base in node.bases:
                self.load(base)
            inner = _FlowScan()
            inner.bound = set(self.bound)
            inner.body(node.body)
            self.free |= inner.free - self.bound
            self.bound.add(node.name)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                self.bound.add(alias.asname or alias.name.split(".")[0])
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            self.bound |= set(node.names)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.statement(child)
            elif isinstance(child, ast.expr):
                self.load(child)


_ALWAYS_BINDS = object()


def _must_bind(statements: "list[ast.stmt]") -> "tuple[set[str], bool]":
    """(names bound on EVERY path through this block, does it always exit).

    "Bound somewhere earlier in the function" is the wrong question and asking
    it produced a live crash. A block that assigns ``hard_final_quality_failed``
    inside an ``if`` hands the name back to its caller only when the branch
    fires; extracted into a helper that returns it unconditionally, the other
    path reaches ``return`` with the name never assigned and raises
    ``UnboundLocalError`` on the first turn that takes it.

    So the test is whether the block itself binds the name on every path.
    Conservative on purpose: a loop contributes nothing because it may not
    iterate, and a ``try`` contributes only its ``finally``, because the body
    can stop at any statement. Fewer must-binds means more refusals, which is
    the direction to be wrong in.
    """
    bound: set[str] = set()
    for statement in statements:
        names, aborts = _must_bind_statement(statement)
        bound |= names
        if aborts:
            return bound, True
    return bound, False


def _must_bind_statement(node: ast.stmt) -> "tuple[set[str], bool]":
    if isinstance(node, (ast.Return, ast.Raise)):
        return set(), True
    if isinstance(node, (ast.Break, ast.Continue)):
        return set(), True
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names |= _names(target, ast.Store)
        return names, False
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _names(node.target, ast.Store), False
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or a.name.split(".")[0] for a in node.names}, False
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}, False
    if isinstance(node, (ast.With, ast.AsyncWith)):
        names = set()
        for item in node.items:
            if item.optional_vars is not None:
                names |= _names(item.optional_vars, ast.Store)
        inner, aborts = _must_bind(node.body)
        return names | inner, aborts
    if isinstance(node, ast.If):
        then_names, then_aborts = _must_bind(node.body)
        else_names, else_aborts = _must_bind(node.orelse) if node.orelse else (set(), False)
        if not node.orelse:
            # No else: the name is bound only when the branch fires.
            return (then_names if then_aborts else set()), False
        if then_aborts and else_aborts:
            return then_names | else_names, True
        if then_aborts:
            return else_names, False
        if else_aborts:
            return then_names, False
        return then_names & else_names, False
    if isinstance(node, ast.Try):
        # The body can stop at any statement, and a handler may not run. Only
        # `finally` is guaranteed.
        final_names, final_aborts = _must_bind(node.finalbody) if node.finalbody else (set(), False)
        return final_names, final_aborts
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return set(), False
    return set(), False


def must_bind(statements: "list[ast.stmt]") -> set[str]:
    """Names this block binds on every path through it."""
    names, _aborts = _must_bind(statements)
    return names


def free_variables(statements: list[ast.stmt]) -> tuple[set[str], set[str]]:
    """(names read before this block binds them, names it binds)."""
    scan = _FlowScan()
    scan.body(statements)
    return scan.free, scan.bound


def _module_scope(tree: ast.Module) -> set[str]:
    """Names bound at MODULE level only.

    Walking the whole tree was wrong and quietly destructive. ``ast.walk``
    reaches functions nested inside other functions, so a helper defined in the
    body of a large function — ``positive_int``, ``finite_number_list`` and
    four siblings inside LatentCortexService._receipt_contract_errors — was
    treated as globally available and dropped from the seam's free-variable
    set. Extracting on that analysis produced a helper referencing six names
    that do not exist in its new scope; ruff caught it as F821 only after the
    file was written.
    """
    scope = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                scope.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            scope.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            scope.add(node.target.id)
    return scope


_BODY_FIELDS = ("body", "orelse", "finalbody")


def _walk_bodies(
    statements: "list[ast.stmt]",
    outer_before: "list[ast.stmt]",
    outer_after: "list[ast.stmt]",
    depth: int,
    inside_loop: bool,
) -> "Iterator[tuple[ast.stmt, list[ast.stmt], list[ast.stmt], int, bool]]":
    """Every statement in every nested body, with what runs before and after it.

    The original walk looked only at ``fn.body``. That is fine for a function
    whose statements sit at the top level and useless for the ones that most
    need cutting: ``_api_chat_turn``'s body is one ``try``, so the only seam
    the tool could see was the whole 4,798-line block, reported as unsafe and
    left there. Recursing means a block nested three levels down is a
    candidate, judged against everything that runs before and after it rather
    than against its siblings alone.

    ``after`` is deliberately over-inclusive: a branch's ``orelse`` is counted
    as running after its ``body`` even though only one of them will. Reading a
    name in code that cannot execute makes the seam look bigger than it is,
    which is the direction to be wrong in.
    """
    for index, statement in enumerate(statements):
        before = [*outer_before, *statements[:index]]
        after = [*statements[index + 1 :], *outer_after]
        yield statement, before, after, depth, inside_loop

        is_loop = isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
        for field_name in _BODY_FIELDS:
            nested = getattr(statement, field_name, None)
            if not isinstance(nested, list) or not nested:
                continue
            siblings = [
                item
                for other in _BODY_FIELDS
                if other != field_name
                for item in (getattr(statement, other, None) or [])
                if isinstance(item, ast.stmt)
            ]
            yield from _walk_bodies(
                nested,
                before,
                [*siblings, *after],
                depth + 1,
                inside_loop or is_loop,
            )
        for handler in getattr(statement, "handlers", []) or []:
            yield from _walk_bodies(
                handler.body, before, after, depth + 1, inside_loop or is_loop
            )


def analyse(
    path: Path, function: str, *, min_lines: int = 60, nested: bool = True
) -> list[Seam]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function.split(".")[-1]
        ),
        None,
    )
    if fn is None:
        raise SystemExit(f"no function named {function!r} in {path}")

    module_scope = _module_scope(tree)
    seams: list[Seam] = []

    candidates = (
        list(_walk_bodies(fn.body, [], [], 0, False))
        if nested
        else [(stmt, [], [], 0, False) for stmt in fn.body]
    )
    if not nested:
        candidates = [
            (stmt, fn.body[:i], fn.body[i + 1 :], 0, False)
            for i, stmt in enumerate(fn.body)
        ]

    for stmt, before_statements, after_statements, depth, inside_loop in candidates:
        end = getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
        if end - stmt.lineno + 1 < min_lines:
            continue

        free, bound = free_variables([stmt])
        reads = free - module_scope - _BUILTINS

        after: set[str] = set()
        for other in after_statements:
            after |= _names(other, ast.Load)

        # Names bound before the seam are carried through; names bound only
        # inside it and read afterwards are the sentinel cases.
        before: set[str] = set()
        for other in before_statements:
            before |= _bound_in(other)

        escapes = sorted(bound & after)
        # An escape is safe when the block binds it on every path, or when it
        # was already an input (so the helper receives a value and returns it).
        # "Bound somewhere earlier in the function" is not the same question:
        # earlier code may sit in a branch that did not run.
        guaranteed = must_bind([stmt]) | reads
        conditional = sorted(n for n in escapes if n not in guaranteed)

        seams.append(
            Seam(
                lineno=stmt.lineno,
                end_lineno=end,
                kind=type(stmt).__name__,
                reads=sorted(reads),
                escapes=escapes,
                conditional_escapes=conditional,
                returns=[
                    n.lineno for n in ast.walk(stmt) if isinstance(n, ast.Return)
                ],
                awaits=sum(1 for n in ast.walk(stmt) if isinstance(n, ast.Await)),
                yields=sum(
                    1 for n in ast.walk(stmt) if isinstance(n, (ast.Yield, ast.YieldFrom))
                ),
                jumps=_jump_count(stmt),
                depth=depth,
                inside_loop=inside_loop,
            )
        )

    seams.sort(key=lambda s: (not s.safe, -s.lines))
    return seams


def _jump_count(node: ast.AST) -> int:
    """`break`/`continue` that would leave the extracted helper.

    A jump inside a loop the block itself contains is fine; one that targets a
    loop outside it cannot survive being moved into a function.
    """
    count = 0
    for child in ast.walk(node):
        if not isinstance(child, (ast.Break, ast.Continue)):
            continue
        if not _has_enclosing_loop(node, child):
            count += 1
    return count


def _has_enclosing_loop(root: ast.AST, target: ast.AST) -> bool:
    def search(node: ast.AST, in_loop: bool) -> bool | None:
        if node is target:
            return in_loop
        is_loop = isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        for child in ast.iter_child_nodes(node):
            found = search(child, in_loop or is_loop)
            if found is not None:
                return found
        return None

    return bool(search(root, False))


def implementation_source(owner: object, name: str, *, depth: int = 3) -> str:
    """Source of a method plus the bodies it delegates to on ``self``.

    Source-inspection tests couple to function boundaries: assert that
    ``tick``'s source mentions something, extract half of ``tick`` into
    ``_tick_body``, and the assertion fails while the behaviour is identical.
    Three tests broke that way on the first kernel extraction, and every one of
    the remaining 50 seams would break more of them.

    Following ``self.<helper>()`` calls one level at a time keeps the assertion
    about the *implementation* rather than about where its curly braces happen
    to fall, which is what those tests meant in the first place.
    """
    import inspect

    seen: set[str] = set()
    parts: list[str] = []

    def walk(method_name: str, remaining: int) -> None:
        if method_name in seen or remaining < 0:
            return
        seen.add(method_name)
        method = getattr(owner, method_name, None)
        if method is None:
            return
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError):
            return
        parts.append(source)
        try:
            tree = ast.parse(textwrap_dedent(source))
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                walk(node.func.attr, remaining - 1)

    walk(name, depth)
    return "\n".join(parts)


def textwrap_dedent(text: str) -> str:
    import textwrap

    return textwrap.dedent(text)


def _tracked() -> list[tuple[Path, str]]:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    out = []
    for key in data.get("functions", {}):
        rel, _, func = key.partition("::")
        out.append((ROOT / rel, func))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="path/to/file.py::function_name")
    parser.add_argument("--tracked", action="store_true", help="every baselined function")
    parser.add_argument("--min-lines", type=int, default=60)
    parser.add_argument(
        "--top-level-only",
        action="store_true",
        help="the old behaviour: only statements directly in the function body",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.tracked:
        targets = _tracked()
    elif args.target:
        rel, _, func = args.target.partition("::")
        targets = [(ROOT / rel, func)]
    else:
        parser.error("give a target or --tracked")

    report: dict[str, object] = {}
    total_safe = 0
    for path, func in targets:
        if not path.is_file():
            continue
        try:
            seams = analyse(
                path, func, min_lines=args.min_lines, nested=not args.top_level_only
            )
        except (SyntaxError, SystemExit):
            continue
        safe = [s for s in seams if s.safe]
        total_safe += len(safe)
        key = f"{path.relative_to(ROOT)}::{func}"
        report[key] = [s.to_dict() for s in seams]

        if not args.json:
            if not seams:
                continue
            print(f"\n{key}")
            for seam in seams[: args.limit]:
                mark = "CUT " if seam.safe else "skip"
                print(
                    f"  [{mark}] {seam.lines:5d} lines  {seam.range if False else f'{seam.lineno}-{seam.end_lineno}':>13}"
                    f"  in={len(seam.reads):2d} out={len(seam.escapes):2d}"
                    f" sentinel={len(seam.conditional_escapes):2d}"
                    f" returns={len(seam.returns)} awaits={seam.awaits}"
                )
                for blocker in seam.blockers:
                    print(f"           blocked: {blocker}")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{total_safe} safe seam(s) across {len(report)} function(s)")
        print(
            "A safe seam still needs the sentinel treatment for its "
            "conditional escapes: returning a default for a name that was "
            "bound conditionally turns a path that raised into one that "
            "quietly proceeds."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
