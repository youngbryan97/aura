"""Move module-level functions and constants into a sibling module, whole.

The shape the size ratchet's lifts take: the named definitions move to
`<module>_<part>.py` beside their module, the module imports them straight
back (every caller and every patch that names them there still finds them),
and what the moved code takes from the module is imported at CALL time, so
a test that patches a name on the module reaches the code that reads it.

Usage:
    lift_functions.py core/conversation/response_reliability.py \
        --into response_reliability_requests \
        --doc "What the person asked for, by count and by phrase." \
        NAME [NAME ...]

Names may be functions or module-level constants. A comment block directly
above a definition travels with it.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_BUILTINS = set(dir(builtins))


def _bound_inside(fn: ast.AST) -> set[str]:
    """Names a function binds itself: parameters and every local store."""
    out: set[str] = set()
    args = fn.args
    for a in args.args + args.posonlyargs + args.kwonlyargs:
        out.add(a.arg)
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            out.add(extra.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            out.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
    return out


def _loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _annotation_names(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs:
        if a.annotation is not None:
            out |= _loads(a.annotation)
    if fn.returns is not None:
        out |= _loads(fn.returns)
    return out


def _body_loads(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for d in fn.decorator_list:
        out |= _loads(d)
    for st in fn.body:
        out |= _loads(st)
    for a in fn.args.defaults + fn.args.kw_defaults:
        if a is not None:
            out |= _loads(a)
    return out


def lift(path: Path, into: str, doc: str, names: list[str]) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    top: dict[str, ast.stmt] = {}
    import_of: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        top[n.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                import_of[(alias.asname or alias.name).split(".")[0]] = node
    missing = [n for n in names if n not in top]
    if missing:
        raise SystemExit(f"not defined at the top of {path}: {', '.join(missing)}")
    moved = set(names)

    header_imports: list[str] = []
    seen_import_lines: set[str] = set()
    type_only: set[str] = set()
    pieces: list[tuple[int, int, str]] = []  # (start_line, end_line, rendered)
    ordered = sorted((top[n] for n in dict.fromkeys(names)), key=lambda n: n.lineno)
    done_ids: set[int] = set()
    for node in ordered:
        if id(node) in done_ids:
            continue
        done_ids.add(id(node))
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while start - 2 >= 0 and lines[start - 2].lstrip().startswith("#"):
            start -= 1
        end = node.end_lineno
        block = "".join(lines[start - 1 : end])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            free = (_body_loads(node) - _bound_inside(node)) - _BUILTINS - moved
            ann = (_annotation_names(node) - _bound_inside(node)) - _BUILTINS - moved
            call_time: list[str] = []
            for name in sorted(free):
                if name in import_of:
                    stmt = import_of[name]
                    rendered = ast.get_source_segment(text, stmt) or ""
                    if rendered and rendered not in seen_import_lines:
                        seen_import_lines.add(rendered)
                        header_imports.append(rendered)
                elif name in top:
                    call_time.append(name)
            for name in sorted(ann - free):
                if name in import_of:
                    stmt = import_of[name]
                    rendered = ast.get_source_segment(text, stmt) or ""
                    if rendered and rendered not in seen_import_lines:
                        seen_import_lines.add(rendered)
                        header_imports.append(rendered)
                elif name in top:
                    type_only.add(name)
            if call_time:
                body_first = node.body[0]
                # after a docstring, before the first statement
                first = node.body[1] if (
                    isinstance(body_first, ast.Expr) and isinstance(body_first.value, ast.Constant)
                    and isinstance(body_first.value.value, str) and len(node.body) > 1
                ) else body_first
                indent = " " * first.col_offset
                importer = (
                    f"{indent}from .{path.stem} import (\n"
                    + "".join(f"{indent}    {n},\n" for n in call_time)
                    + f"{indent})\n\n"
                )
                # insert at the line of the first statement, relative to the block
                rel = first.lineno - start
                block_lines = block.splitlines(keepends=True)
                block_lines.insert(rel, importer)
                block = "".join(block_lines)
        else:
            for name in sorted(_loads(node) - _BUILTINS - moved):
                if name in import_of:
                    stmt = import_of[name]
                    rendered = ast.get_source_segment(text, stmt) or ""
                    if rendered and rendered not in seen_import_lines:
                        seen_import_lines.add(rendered)
                        header_imports.append(rendered)
                elif name in top:
                    raise SystemExit(
                        f"{name} is read by a moved constant at import time and stays in {path.stem}; "
                        "move it too or leave the constant"
                    )
        pieces.append((start, end, block))

    header = f'"""{doc}\n\nLifted whole out of `{path.stem}`, which imports them straight back: every\ncaller and every patch that names them there still finds them. What they\ntake from that module is imported at CALL time, for the same reason.\n"""\nfrom __future__ import annotations\n\n'
    if header_imports:
        header += "\n".join(sorted(set(header_imports), key=lambda s: (not s.startswith("import "), s))) + "\n"
    if type_only:
        header += "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n" + f"    from .{path.stem} import (\n" + "".join(f"        {n},\n" for n in sorted(type_only)) + "    )\n"
    sibling = path.with_name(f"{into}.py")
    body = "\n\n".join(p[2].rstrip("\n") + "\n" for p in pieces)
    sibling.write_text(header + "\n\n" + body, encoding="utf-8")
    ast.parse(sibling.read_text(encoding="utf-8"))

    # cut from the parent, bottom-up, taking one following blank line
    for start, end, _ in sorted(pieces, key=lambda p: -p[0]):
        stop = end
        while stop < len(lines) and lines[stop].strip() == "" and stop - end < 2:
            stop += 1
        del lines[start - 1 : stop]
    # import back, after the last top-level import of the parent
    tree2 = ast.parse("".join(lines))
    last_import = 0
    for node in tree2.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = node.end_lineno
        elif last_import and not isinstance(node, (ast.Expr,)):
            break
    back = (
        f"from .{into} import (  # noqa: F401  (re-exported: they were defined here)\n"
        + "".join(f"    {n},\n" for n in names)
        + ")\n"
    )
    lines.insert(last_import, back)
    out = "".join(lines)
    ast.parse(out)
    path.write_text(out, encoding="utf-8")
    return sum(p[1] - p[0] + 1 for p in pieces), len(names)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--into", required=True)
    parser.add_argument("--doc", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args(argv)
    moved_lines, count = lift(ROOT / args.module, args.into, args.doc, args.names)
    print(f"{args.module}: {count} definition(s), {moved_lines} lines, moved to {args.into}.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
