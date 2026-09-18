"""Read a function as it runs, with the blocks the size gate moved out put back.

`tools/refactor/extract_self_contained_blocks.py` moves a self-contained block
of an outsized function into a module-level helper named `_<function>_<slug>`
and leaves a call in its place. The code runs in the same order it did; the
text does not read in that order any more, and a test that pins "this line
comes before that one" or "these two lines sit within 400 characters" fails
against a refactor that changed nothing.

These readers expand each helper call back into the text at its call site, so
text order is execution order again. They are for reading, not running: a
helper's ``return`` becomes a bare line in the middle of the caller.
"""
from __future__ import annotations

import ast
import pathlib

_CALL_CARRIERS = (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)


def _helper_called(stmt: ast.stmt, owner: str, helpers: dict[str, ast.AST]) -> str | None:
    value = getattr(stmt, "value", None)
    if isinstance(stmt, ast.Expr):
        value = stmt.value
    if isinstance(value, ast.Await):
        value = value.value
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("self", "cls"):
        name = func.attr  # an earlier sweep made the helpers methods of the class
    else:
        return None
    if name not in helpers:
        return None
    # a helper of ``owner``, or a sibling helper of the same root function
    # (an earlier sweep named a helper's helpers after the root)
    for root in helpers.get("", ()):
        prefix = f"_{root.lstrip('_')}_"
        if name.startswith(prefix) and (owner == root or owner.startswith(prefix)):
            return name
    return None


def _statements(fn: ast.AST):
    """Every statement in ``fn``'s body, nested loops and branches included,
    but not those of a function or class defined inside it."""
    stack = list(reversed(getattr(fn, "body", [])))
    while stack:
        st = stack.pop()
        yield st
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            sub = getattr(st, field, None)
            if isinstance(sub, list):
                for child in reversed(sub):
                    if isinstance(child, ast.stmt):
                        stack.append(child)
                    elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                        stack.extend(reversed(child.body))


def _expanded(lines: list[str], fn: ast.AST, helpers: dict[str, ast.AST], indent: int, used: set[str], stack: tuple[str, ...] = ()) -> list[str]:
    """``fn``'s body lines, re-based to ``indent``, with helper calls expanded."""
    stack = (*stack, fn.name)
    # A helper moved into a sibling module carries its own file's lines.
    lines = getattr(fn, "_source_lines", lines)
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]  # the helper's docstring is not a line of the caller
    if not body:
        return []
    first, last = body[0].lineno, fn.end_lineno
    base = body[0].col_offset
    out = []
    replacements = {}
    for st in _statements(fn):
        name = _helper_called(st, fn.name, helpers)
        if name and name not in stack:  # a helper that calls itself stays a call
            used.add(name)
            replacements[st.lineno] = (st.end_lineno, _expanded(lines, helpers[name], helpers, st.col_offset - base + indent, used, stack))
    number = first
    while number <= last:
        if number in replacements:
            end, inserted = replacements[number]
            out.extend(inserted)
            number = end + 1
            continue
        line = lines[number - 1]
        stripped = line[base:] if line[:base].strip() == "" else line.lstrip()
        out.append((" " * indent + stripped) if line.strip() else line)
        number += 1
    return out


def _module_helpers(tree: ast.Module) -> dict[str, ast.AST]:
    """Every underscore-named function at module level or directly in a class."""
    found: dict = {}
    roots = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    found[""] = sorted(set(roots), key=len, reverse=True)  # the functions helpers can belong to
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_"):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("_"):
                    found.setdefault(member.name, member)
    return found


def inlined_module_source(path: str | pathlib.Path) -> str:
    """The module's text with every moved block back at its call site, and the
    helpers it came from removed."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    helpers = _module_helpers(tree)
    used: set[str] = set()
    sites: list[tuple[ast.AST, ast.stmt, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for st in _statements(node):
            name = _helper_called(st, node.name, helpers)
            if name:
                sites.append((node, st, name))
    expansions = {id(st): _expanded(lines, helpers[name], helpers, st.col_offset, used) for _n, st, name in sites}
    used |= {name for _n, _st, name in sites}
    # a helper that is itself expanded elsewhere is deleted whole; the calls
    # inside it were expanded on the way, and editing them here as well would
    # shift the lines its deletion counts on
    edits: list[tuple[int, int, list[str]]] = [
        (st.lineno, st.end_lineno, expansions[id(st)]) for node, st, _name in sites if node.name not in used
    ]
    for name in used:
        helper = helpers[name]
        start = min([helper.lineno] + [d.lineno for d in helper.decorator_list])
        edits.append((start, helper.end_lineno, []))
    for start, end, inserted in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[start - 1 : end] = inserted
    return "\n".join(lines) + "\n"


def inlined_function_source(
    path: str | pathlib.Path, qualname: str, *, helpers_also_in: tuple[str | pathlib.Path, ...] = ()
) -> str:
    """One function's text, as it runs. ``qualname`` may be ``Class.method``.

    ``helpers_also_in`` names sibling modules its moved blocks may have been
    moved on to: a second size sweep took the pursuit's decision helpers into
    a module of their own, and a function read from its own file alone then
    reported every call site in them as missing.
    """
    text = pathlib.Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    helpers = _module_helpers(tree)
    for other in helpers_also_in:
        other_text = pathlib.Path(other).read_text(encoding="utf-8")
        other_lines = other_text.splitlines()
        more = _module_helpers(ast.parse(other_text))
        helpers[""] = sorted(set(helpers[""]) | set(more.pop("", ())), key=len, reverse=True)
        for name, node in more.items():
            if name not in helpers:
                node._source_lines = other_lines  # noqa: SLF001 - read by _expanded
                helpers[name] = node
    *owners, name = qualname.split(".")
    scope: ast.AST = tree
    for owner in owners:
        scope = next(n for n in ast.walk(scope) if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in ast.walk(scope) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    head = lines[fn.lineno - 1 : fn.body[0].lineno - 1]
    used: set[str] = set()
    return "\n".join(head + _expanded(lines, fn, helpers, fn.body[0].col_offset, used)) + "\n"


def line_of(text: str, needle: str) -> int:
    """The first line of ``text`` containing ``needle``, 1-based."""
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"not found: {needle}")
