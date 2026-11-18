"""Expressions her own file already writes, put in place of a suspect one: repair from ingredients (Master Todo F16).

A fix is often made of code the program already has (the plastic surgery hypothesis: Barr, Brun, Devanbu, Harman
and Sarro, FSE 2014). Repairs built on it take the replacement for a suspect expression from the same program and
rank it by the context it was written in: ssFix (Xin and Reiss, ASE 2017), CapGen (Wen, Chen, Wu, Hao and Cheung,
ICSE 2018), SimFix (Jiang, Xiong, Zhang, Gao and Chen, ISSTA 2018).

Here an expression on a suspect line can be replaced by an expression written elsewhere in the same file that reads
only names visible on that line and is used as the suspect one is: a truth (a branch's condition, an operand of
``and``, ``or`` or ``not``, a comparison) by a truth, any other value by a value, a number by a number. Those written
in a context nearer the suspect one come first:

1. in the same place of a call to the same function (the same keyword or the same position), or beside the suspect
   expression in another expression (two operands of one operator, two arguments of one call);
2. in the same function;
3. in the same class;
4. elsewhere in the file, reading a name the suspect expression reads (not ``self`` or ``cls``, which nearly every
   method reads, and not a builtin).

Anything else in the file is not offered. Semantic repair (semantic_repair.py) synthesises a condition from her
program's values; an ingredient brings what values cannot, such as a call her code already makes to decide the same
thing. Nothing inside a raise, an assert, or the condition of a branch that raises is replaced, so a refusal is never
what changes. Nothing is generated: each candidate is a copy of her own
code, and each is judged by her tests (repair_by_selection.py).
"""
from __future__ import annotations

import ast
import builtins
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from core.self_modification.fix_templates import _offsets, _replace, _span

__all__ = ["ingredient_changes", "ingredients_at"]

Line = tuple[str, int]

#: What a suspect value can be, and what can take its place.
_VALUES = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.BinOp, ast.IfExp, ast.Constant)
#: What a suspect truth can be, and what can take its place.
_TRUTHS = (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.Call, ast.Name, ast.Attribute, ast.Subscript)
#: How tightly each kind of expression binds, as Python's grammar orders them; an ingredient that binds less tightly
#: than the operator it goes into is put in brackets.
_BINDS = {ast.Lambda: 0, ast.IfExp: 1, ast.Or: 2, ast.And: 3, ast.Not: 4, ast.Compare: 5, ast.BitOr: 6,
          ast.BitXor: 7, ast.BitAnd: 8, ast.LShift: 9, ast.RShift: 9, ast.Add: 10, ast.Sub: 10, ast.Mult: 11,
          ast.Div: 11, ast.FloorDiv: 11, ast.Mod: 11, ast.MatMult: 11, ast.USub: 12, ast.UAdd: 12, ast.Invert: 12,
          ast.Pow: 13, ast.Await: 14}
_ATOM = 15
_BUILTINS = frozenset(dir(builtins))
_EVERYWHERE = frozenset({"self", "cls"})
#: Lines whose ingredients are ordered together, as many as semantic repair looks at (semantic_repairs' ``lines``);
#: the next lines' wait for these. Ordered over every line a failing test ran, an ingredient on the 140th line of
#: f5da1f5da653's ranking came before her own fix on the 5th.
LINES_AT_ONCE = 15
#: The names of the four kinds of context above, in their order.
CONTEXTS = ("same_place", "same_function", "same_class", "same_names")


@dataclass
class _Written:
    """One expression text and every context it is written in across the file."""

    text: str
    kind: type
    reads: frozenset[str]
    awaits: bool
    binds: int
    as_truth: bool = False
    as_value: bool = False
    functions: set[int] = field(default_factory=set)
    classes: set[int] = field(default_factory=set)
    places: set[tuple[str, str]] = field(default_factory=set)
    #: What is written beside it, by the occurrence it is written beside: the occurrence being replaced is left out
    #: of its own context.
    beside: dict[int, set[str]] = field(default_factory=dict)

    def beside_elsewhere(self, node: ast.AST) -> set[str]:
        return {text for at, texts in self.beside.items() if at != id(node) for text in texts}


def _binds(node: ast.AST) -> int:
    """How tightly ``node`` binds: its operator's place in _BINDS, or _ATOM for a name, call, literal and the like."""
    if isinstance(node, (ast.BoolOp, ast.BinOp, ast.UnaryOp)):
        return _BINDS.get(type(node.op), _ATOM)
    return _BINDS.get(type(node), _ATOM)


def _bracketed(binds: int, node: ast.AST, parent: ast.AST | None) -> bool:
    """Whether an expression binding as tightly as ``binds`` needs brackets put where ``node`` is."""
    if isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node or isinstance(parent, ast.Await):
        return binds < _ATOM
    if isinstance(parent, (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp)):
        return binds <= _binds(parent)
    return False


def _as_truth(node: ast.AST, parent: ast.AST | None) -> bool:
    """Whether ``node`` is used as a truth: a branch's condition, a filter, an operand of and, or, not, or a
    comparison, conjunction or negation wherever it is."""
    if isinstance(node, (ast.Compare, ast.BoolOp)) or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and parent.test is node:
        return True
    if isinstance(parent, ast.comprehension) and node in parent.ifs:
        return True
    return isinstance(parent, ast.BoolOp) or isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not)


def _a_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _holders(node: ast.AST, parents: Mapping[int, ast.AST]) -> list[ast.AST]:
    out, at = [], parents.get(id(node))
    while at is not None:
        out.append(at)
        at = parents.get(id(at))
    return out


def _reads(node: ast.AST) -> frozenset[str]:
    """The names an expression reads that it does not bind itself (a comprehension's or a lambda's own)."""
    bound = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {a.arg for n in ast.walk(node) if isinstance(n, ast.Lambda) for a in ast.walk(n.args)
              if isinstance(a, ast.arg)}
    return frozenset(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                     and n.id not in bound)


def _segmenter(source: str) -> Callable[[ast.AST], str]:
    """The text a node is written as, from line offsets counted once. ast.get_source_segment splits the whole
    source again on every call, which on a file of 1,700 lines took 26 seconds for one line's ingredients."""
    starts = _offsets(source)

    def segment(node: ast.AST) -> str:
        if getattr(node, "end_col_offset", None) is None:
            return ""
        a, b = _span(source, starts, node)
        return source[a:b]
    return segment


def _place(node: ast.AST, parents: Mapping[int, ast.AST], segment: Callable[[ast.AST], str]) -> tuple[str, str] | None:
    """The call and argument an expression is written as, by the callee's text and the keyword or position."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.keyword):
        call = parents.get(id(parent))
        if isinstance(call, ast.Call) and parent.arg:
            return segment(call.func), parent.arg
        return None
    if isinstance(parent, ast.Call) and node in parent.args:
        return segment(parent.func), str(parent.args.index(node))
    return None


def _siblings(node: ast.AST, parent: ast.AST | None) -> list[ast.AST]:
    """The expressions written beside ``node`` as operands of one operator. Two arguments of one call are not:
    each has a part of its own, which _place records."""
    if isinstance(parent, ast.BinOp):
        together = [parent.left, parent.right]
    elif isinstance(parent, ast.Compare):
        together = [parent.left, *parent.comparators]
    elif isinstance(parent, ast.BoolOp):
        together = list(parent.values)
    else:
        return []
    return [other for other in together if other is not node]


def _module_names(tree: ast.Module) -> set[str]:
    """The names the module binds at its top level, under any if, try or with there."""
    names: set[str] = set()

    def bind(statements: Sequence[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
            for block in ("body", "orelse", "finalbody"):
                if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    bind(getattr(node, block, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                bind(handler.body)

    bind(tree.body)
    return names


def _function_names(function: ast.AST, before: int) -> set[str]:
    """A function's parameters and the names it binds on lines before ``before``."""
    arguments = function.args  # type: ignore[attr-defined]
    names = {a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    names.update(a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None)
    for node in ast.walk(function):
        line = getattr(node, "lineno", before)
        if line >= before:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _not_a_place(node: ast.AST, parent: ast.AST | None, holders: Sequence[ast.AST]) -> bool:
    """Whether ``node`` is where no ingredient goes: a name bound, a callee, an annotation, a string's part, a
    decorator, or anything a refusal says or decides."""
    if isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
        return True
    if isinstance(parent, ast.Call) and parent.func is node:
        return True
    if isinstance(node, ast.Constant) and not _a_number(node.value):
        return True
    child = node
    for holder in holders:
        if isinstance(holder, (ast.JoinedStr, ast.Raise, ast.Assert, ast.arg)):
            return True
        if isinstance(holder, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if child in holder.decorator_list or child is getattr(holder, "returns", None):
                return True
            break
        if isinstance(holder, ast.AnnAssign) and child is holder.annotation:
            return True
        if isinstance(holder, (ast.If, ast.While)) and child is holder.test and any(
                isinstance(statement, ast.Raise) for statement in holder.body):
            return True
        child = holder
    return False


def _inventory(tree: ast.AST, segment: Callable[[ast.AST], str], parents: Mapping[int, ast.AST]) -> dict[str, _Written]:
    """Every expression the file writes on one line, by its text, with the contexts it is written in."""
    written: dict[str, _Written] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (*_VALUES, *_TRUTHS)) or isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
            continue
        if (node.end_lineno or node.lineno) != node.lineno:
            continue
        if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Starred)) for n in ast.walk(node)):
            continue
        text = segment(node)
        if not text:
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        holders = _holders(node, parents)
        if isinstance(node, ast.Constant) and not _a_number(node.value):
            continue
        truth = _as_truth(node, parent)
        if not truth and not isinstance(node, _VALUES):
            continue  # a negative number or an inverted one: not offered
        entry = written.setdefault(text, _Written(text, type(node), _reads(node),
                                                  any(isinstance(n, ast.Await) for n in ast.walk(node)), _binds(node)))
        entry.as_truth |= truth
        entry.as_value |= not truth
        for holder in holders:
            if isinstance(holder, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entry.functions.add(id(holder))
            elif isinstance(holder, ast.ClassDef):
                entry.classes.add(id(holder))
        place = _place(node, parents, segment)
        if place is not None:
            entry.places.add(place)
        entry.beside[id(node)] = {t for t in map(segment, _siblings(node, parent)) if t}
    return written


class _File:
    """One file read once: its tree, what each node is written in, and every expression it writes."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.tree = ast.parse(source)
        self.parents = _parents(self.tree)
        self.starts = _offsets(source)
        self.segment = _segmenter(source)
        self.written = _inventory(self.tree, self.segment, self.parents)
        self.module = _module_names(self.tree)
        self._names: dict[tuple[int, int], set[str]] = {}
        self._lines: dict[int, list[ast.AST]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (*_VALUES, *_TRUTHS)) and hasattr(node, "lineno"):
                self._lines.setdefault(node.lineno, []).append(node)

    def names(self, function: ast.AST, before: int) -> set[str]:
        key = (id(function), before)
        if key not in self._names:
            self._names[key] = _function_names(function, before)
        return self._names[key]

    def at(self, line: int) -> Iterator[tuple[tuple[int, int, int, int], str, tuple[int, int, str]]]:
        """Each ingredient for each expression on ``line``, as ((alternatives, -names shared, length, order),
        context, (start, end, text put there)), unsorted. ``alternatives`` is how many ingredients the context that
        offered this one offered for the same expression: the fewer, the more that context says. The change is kept
        as its span, not as the changed file: a long file has thousands of these."""
        order = 0
        for node in self._lines.get(line, ()):
            if (node.end_lineno or node.lineno) != node.lineno:
                continue
            parent = self.parents.get(id(node))
            holders = _holders(node, self.parents)
            if _not_a_place(node, parent, holders):
                continue
            truth = _as_truth(node, parent)
            if not truth and not isinstance(node, _VALUES):
                continue
            text = self.segment(node)
            functions = [h for h in holders if isinstance(h, (ast.FunctionDef, ast.AsyncFunctionDef))]
            function = functions[0] if functions else None
            cls = next((h for h in holders if isinstance(h, ast.ClassDef)), None)
            visible = set(self.module) | _BUILTINS
            for i, holder in enumerate(functions):
                visible |= self.names(holder, line if i == 0 else (holder.end_lineno or line) + 1)
            local = self.names(function, line) if function is not None else set()
            asynchronous = isinstance(function, ast.AsyncFunctionDef)
            own = self.written.get(text)
            beside = own.beside_elsewhere(node) if own is not None else set()
            # An operand replaced by its own partner (x < x, a and a) decides nothing.
            partners = set(map(self.segment, _siblings(node, parent)))
            place = _place(node, self.parents, self.segment)
            reads = _reads(node) - _EVERYWHERE - _BUILTINS
            found: list[tuple[int, _Written]] = []
            for other in self.written.values():
                if other.text == text or other.text in partners or not other.reads <= visible:
                    continue
                if (other.awaits and not asynchronous) or not (other.as_truth if truth else other.as_value):
                    continue
                if (other.kind is ast.Constant) != isinstance(node, ast.Constant):
                    continue
                if isinstance(node, ast.Name) and other.kind is ast.Name and other.text in local:
                    continue  # another local name is the fix templates' replace_name
                if place is not None and place in other.places or other.text in beside:
                    found.append((0, other))
                elif function is not None and id(function) in other.functions:
                    found.append((1, other))
                elif cls is not None and id(cls) in other.classes:
                    found.append((2, other))
                elif reads & other.reads:
                    found.append((3, other))
            offered = Counter(context for context, _ in found)
            a, b = _span(self.source, self.starts, node)
            for context, other in found:
                put = f"({other.text})" if _bracketed(other.binds, node, parent) else other.text
                order += 1
                yield ((offered[context], -len(reads & other.reads), len(other.text), order), CONTEXTS[context],
                       (a, b, put))


def ingredients_at(source: str, line: int) -> Iterator[tuple[tuple[int, int, int, int], str, tuple[int, int, str]]]:
    """_File.at for one line of ``source``; nothing when it does not parse."""
    try:
        file = _File(source)
    except SyntaxError:
        return  # not a failure: source that does not parse has no expression to replace
    yield from file.at(line)


def ingredient_changes(ranked: Sequence[tuple[Line, float]],
                       sources: Mapping[str, str]) -> Iterator[tuple[str, str, str]]:
    """(path, context, changed source) for every ingredient at every ranked line, LINES_AT_ONCE lines at a time:
    within those lines, those whose context offered the fewest alternatives first, then the more suspicious line,
    then the ingredient sharing more names, then the shorter."""
    lines = [(path, line) for (path, line), score in ranked if path in sources and score > 0]
    files: dict[str, _File | None] = {}
    for start in range(0, len(lines), LINES_AT_ONCE):
        yield from _window(lines[start:start + LINES_AT_ONCE], start, sources, files)


def _window(lines: Sequence[Line], first: int, sources: Mapping[str, str],
            files: dict[str, _File | None]) -> Iterator[tuple[str, str, str]]:
    found: list[tuple[tuple[int, int, int, int, int], str, str, tuple[int, int, str]]] = []
    for rank, (path, line) in enumerate(lines, first):
        if path not in files:
            try:
                files[path] = _File(sources[path])
            except SyntaxError:
                files[path] = None
        file = files[path]
        if file is None:
            continue
        for (alternatives, shared, length, order), context, span in file.at(line):
            found.append(((alternatives, rank, shared, length, order), path, context, span))
    found.sort(key=lambda item: item[0])
    seen: set[tuple[str, int, int, str]] = set()
    for _key, path, context, (a, b, put) in found:
        if (path, a, b, put) in seen:
            continue
        seen.add((path, a, b, put))
        changed = _replace(sources[path], a, b, put)
        try:
            ast.parse(changed)
        except SyntaxError:
            continue  # not a failure: a change that does not parse is not offered
        yield path, context, changed
