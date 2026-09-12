#!/usr/bin/env python3
"""Assertions that read a module's source for a string that has moved.

A test that asserts ``"strip_prompt_artifacts" in inspect.getsource(chat)``
stops being true when the route is split for size and the call moves to a
sibling module. It does not fail honestly: it reads a shorter file, finds
nothing, and reports that the call site is gone — which is exactly what a
deleted call site looks like from there. Three of these turned up in one
offline suite run, in files nobody had touched.

The shape is mechanical, so it can be found mechanically. For every

    assert "<literal>" in inspect.getsource(<module>)

this resolves ``<module>`` through the test's own imports, reads that file,
and asks whether the literal is in it. When it is not, it looks for the
literal across the package and says where it went, which is the one fact the
failing test cannot tell you.

Reports only. A stale address is a test to fix, and which fix is right —
read the module that now holds it, or read the whole family through a
concatenating helper — depends on what the assertion is claiming.

    python tools/lint_source_assertions.py [paths...]
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _imported_modules(tree: ast.AST) -> dict[str, str]:
    """Local name to dotted module, for `import x.y` and `from x import y`.

    Walked over one scope at a time by the caller. These imports are almost
    always inside the test function, and a file-wide map let the last
    ``as mod`` in the file decide what ``mod`` meant in every other test —
    which is how a passing assertion was reported as reading the wrong
    module.
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return found


def _module_file(dotted: str) -> Path | None:
    candidate = ROOT / Path(*dotted.split("."))
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path
    return None


def _getsource_target(node: ast.AST) -> str | None:
    """The local name a ``getsource(...)`` call reads, if it reads a plain one."""
    if (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "getsource"
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def _source_reads(tree: ast.AST) -> list[tuple[int, str, str, dict[str, str]]]:
    """(line, local name read, literal looked for) for every spelling.

    Several spellings, because a gate that knows one of them is the defect it
    is looking for. ``"x" in getsource(m)`` is the direct form; the rest go
    through a local — ``source = getsource(m)`` and then ``"x" in source``,
    ``source.find("x")``, or ``source.split("x", 1)[1]``. The last raises
    IndexError rather than asserting, so it fails somewhere else entirely and
    a checker that matched only the compare called the file clean while it was
    failing.

    Per FUNCTION, not per file. ``source`` is the name every one of these
    tests uses, so a file-wide binding attributed one test's literal to
    another test's module and reported 169 stale assertions where there were
    three. A local name means what it means inside its own function.
    """
    out: list[tuple[int, str, str]] = []
    # Compares that sit inside an `or`: the assertion holds when ANY branch
    # does, so one stale spelling beside a live one is not a stale assertion.
    # `"_gates_required_for(bool(is_forged)" in source or "..." in source` is
    # written exactly that way, and flagging the first half reported a passing
    # test.
    alternatives: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for branch in node.values:
                for inner in ast.walk(branch):
                    alternatives.add(id(inner))

    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scope_imports = _imported_modules(scope)
        held: dict[str, str] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                read = _getsource_target(node.value)
                if isinstance(target, ast.Name):
                    # Rebound to something else: the name stops standing for a
                    # module's source, and keeping the old binding is how a
                    # checker invents a finding.
                    if read:
                        held[target.id] = read
                    else:
                        held.pop(target.id, None)
        for node in ast.walk(scope):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                if not isinstance(node.ops[0], ast.In):
                    continue
                if id(node) in alternatives:
                    continue
                left, right = node.left, node.comparators[0]
                if not (isinstance(left, ast.Constant) and isinstance(left.value, str)):
                    continue
                read = _getsource_target(right) or (
                    held.get(right.id) if isinstance(right, ast.Name) else None
                )
                if read:
                    out.append((node.lineno, read, left.value, scope_imports))
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") in {"find", "index", "count", "split", "partition", "rsplit"}
                and isinstance(getattr(node.func, "value", None), ast.Name)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                read = held.get(node.func.value.id)
                if read:
                    out.append((node.lineno, read, node.args[0].value, scope_imports))
    return out


def _where_it_lives(literal: str, root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            if literal in path.read_text(encoding="utf-8", errors="ignore"):
                found.append(_named(path))
        except OSError:
            continue
        if len(found) >= 4:
            break
    return found


def _named(path: Path) -> str:
    """The path as written, relative to the repo where it is under it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def look(paths: list[Path]) -> list[dict[str, object]]:
    stale: list[dict[str, object]] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        # Module-level imports are the fallback; a test's own import wins.
        outer = _imported_modules(tree)
        for line, name, literal, inner in _source_reads(tree):
            dotted = inner.get(name) or outer.get(name)
            if not dotted:
                continue
            target = _module_file(dotted)
            if target is None:
                continue
            try:
                text = target.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if literal in text:
                continue
            stale.append(
                {
                    "file": _named(path),
                    "line": line,
                    "reads": _named(target),
                    "literal": literal,
                    "now_in": _where_it_lives(literal, ROOT / "core")
                    or _where_it_lives(literal, ROOT / "interface"),
                }
            )
    return stale


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("paths", nargs="*", default=["tests"])
    ask.add_argument("--show", type=int, default=40)
    args = ask.parse_args()

    files: list[Path] = []
    for one in args.paths:
        target = ROOT / one
        files.extend(sorted(target.rglob("test_*.py")) if target.is_dir() else [target])

    stale = look(files)
    print(f"{len(stale)} assertion(s) reading a module that no longer holds the string")
    for row in stale[: args.show]:
        print(f"   {row['file']}:{row['line']}")
        print(f"      reads   {row['reads']}")
        print(f"      for     {row['literal'][:80]!r}")
        print(f"      now in  {row['now_in'] or 'nowhere under core/ or interface/'}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
