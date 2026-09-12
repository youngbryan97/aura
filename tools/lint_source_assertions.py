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


def _read_text_of(node: ast.AST, paths: dict[str, str]) -> str | None:
    """``CHAT.read_text()`` where CHAT names a source file in the repo."""
    if (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "read_text"
        and isinstance(getattr(node.func, "value", None), ast.Name)
    ):
        return paths.get(node.func.value.id)
    return None


def _module_file(dotted: str) -> Path | None:
    candidate = ROOT / Path(*dotted.split("."))
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path
    return None


def _is_repo_rooted(node: ast.AST, roots: set[str]) -> bool:
    """Whether a path chain starts at this repo rather than at a tmp dir.

    ``tmp_path / "core" / "terminal_monitor.py"`` has the same literal tail as
    the real module and is a file the test WROTE. Reading the repo's copy of
    it and reporting a missing string is a finding about nothing, so the head
    has to be anchored: ``Path(__file__)…`` directly, or a name bound to one.
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and (inner.id == "__file__" or inner.id in roots):
            return True
    return False


def _path_literal(node: ast.AST) -> str | None:
    """The repo-relative source file a ``Path(...) / "a/b.py"`` chain names.

    The other half of the same defect. Half of these tests never call
    ``getsource`` at all — they build ``CHAT = Path(__file__).parents[1] /
    "interface/routes/chat.py"`` and read it. When the route is split, that
    read finds a shorter file exactly the same way.

    Only the literal segments matter: whatever the ``Path(...)`` head is, the
    strings after it are the path inside the repo, and one of them has to end
    in ``.py`` for this to be a module read.
    """
    parts: list[str] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            parts.insert(0, node.right.value)
        else:
            return None
        node = node.left
    if not parts or not parts[-1].endswith(".py"):
        return None
    return "/".join(parts)


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
    # Module-level `CHAT = Path(...) / "interface/routes/chat.py"` and the
    # locals that read it.
    paths: dict[str, str] = {}
    # Names that stand for the repo root, so `ROOT / "interface/..."` counts
    # and `tmp_path / "core/..."` does not.
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if _path_literal(node.value) is None and _is_repo_rooted(node.value, roots):
                roots.add(target.id)
                continue
            named = _path_literal(node.value)
            if named and _is_repo_rooted(node.value, roots):
                paths[target.id] = named
    outer_paths: dict[str, str] = {}

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
        held: dict[str, str] = dict(outer_paths)
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                read = _getsource_target(node.value) or _read_text_of(node.value, paths)
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
            # `name` is either a local bound to an import, or — for the
            # `Path(...) / "a/b.py"` spelling — the repo-relative file itself.
            dotted = inner.get(name) or outer.get(name) or name
            target = _module_file(dotted)
            if target is None and dotted.endswith(".py"):
                candidate = ROOT / dotted
                target = candidate if candidate.is_file() else None
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
