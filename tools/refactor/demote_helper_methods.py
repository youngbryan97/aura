"""Move private helper methods out of a class to module level, in bulk.

The size gate holds a class to its method count. A sweep that moved blocks
out of a long method into `_<method>_<slug>` METHODS left the method shorter
and the class larger: `InferenceGate` went from 151 methods to 189 without a
line of new behaviour. A helper that needs `self` only to read and call it
is a function that takes `self`; nothing about it is a method.

For every method the selector names, and that qualifies:
  * no decorators, no `super()`, no `__class__`,
  * referenced in this file only as `self.<name>(...)` or `cls.<name>(...)`,
    and nowhere else in the repository (a test that patches it by class
    attribute would silently patch nothing),
the definition moves to just above the class, dedented one level, keeping
`self` as its first parameter, and each call becomes `<name>(self, ...)`.

Usage:
    demote_helper_methods.py core/brain/inference_gate.py::InferenceGate
    demote_helper_methods.py FILE::Class --pattern '^_salvage_'
    demote_helper_methods.py FILE::Class --dry-run

Without --pattern, the helpers of the sweep are selected: a method whose name
is `_<other method of the class>_<slug>`.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise SystemExit(f"no class {name}")


def _sweep_named(cls: ast.ClassDef) -> set[str]:
    methods = [n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    roots = sorted(methods, key=len, reverse=True)
    out = set()
    for name in methods:
        for root in roots:
            if root != name and name.startswith(f"_{root.lstrip('_')}_"):
                out.add(name)
                break
    return out


def _referenced_elsewhere(name: str, path: Path) -> list[str]:
    """Files other than ``path`` that mention ``name`` as a word."""
    try:
        out = subprocess.run(
            ["git", "grep", "-l", "-w", name, "--", "core", "interface", "tests", "tools", "skills", "executors"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return ["<grep failed>"]
    rel = str(path.resolve().relative_to(ROOT))
    return [f for f in out if f != rel]


def _uses_class_cell(fn: ast.AST) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "__class__":
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super":
            return True
    return False


def _docstrings(tree: ast.Module) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            out.add(id(body[0].value))
    return out


def _references_in_file(tree: ast.Module, name: str) -> tuple[list[ast.Call], int]:
    """Calls ``self.name(...)`` / ``cls.name(...)``, and how many other mentions there are.

    Another mention — an attribute read without a call, a string that could
    feed ``getattr`` — means the name is used as a member, and the method
    stays where it is.
    """
    calls: list[ast.Call] = []
    call_funcs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("self", "cls"):
            calls.append(node)
            call_funcs.add(id(node.func))
    docstrings = _docstrings(tree)
    others = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == name and id(node) not in call_funcs:
            others += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings \
                and re.search(rf"\b{re.escape(name)}\b", node.value):
            others += 1
    return calls, others


def demote(path: Path, class_name: str, pattern: str | None, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    cls = _class_node(tree, class_name)
    wanted = _sweep_named(cls) if pattern is None else {
        n.name for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and re.search(pattern, n.name)
    }
    chosen: list[ast.AST] = []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in wanted:
            continue
        why = None
        static = (
            len(node.decorator_list) == 1
            and isinstance(node.decorator_list[0], ast.Name)
            and node.decorator_list[0].id == "staticmethod"
        )
        if node.decorator_list and not static:
            why = "decorated"
        elif _uses_class_cell(node):
            why = "uses super() or __class__"
        elif not static and (not node.args.args or node.args.args[0].arg not in ("self", "cls")):
            why = "no self"
        else:
            calls, others = _references_in_file(tree, node.name)
            if others:
                why = f"{others} mention(s) in this file that are not self.{node.name}(...) calls"
            else:
                elsewhere = _referenced_elsewhere(node.name, path)
                if elsewhere:
                    why = "referenced in " + ", ".join(elsewhere[:3])
        if why:
            print(f"  keep {class_name}.{node.name}: {why}")
            continue
        chosen.append(node)
    if not chosen:
        print(f"{path}::{class_name}: nothing to demote")
        return 0
    names = {n.name for n in chosen}
    static_names = {n.name for n in chosen if n.decorator_list}

    # call sites, edited from the end of the file backwards
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in names
        and isinstance(n.func.value, ast.Name) and n.func.value.id in ("self", "cls")
    ]
    edits: list[tuple[int, int, int, int, str]] = []  # (line, col, end_line, end_col, text)
    for call in calls:
        func = call.func
        receiver = func.value.id
        # `self._name` -> `_name`
        edits.append((func.lineno, func.col_offset, func.end_lineno, func.end_col_offset, func.attr))
        # after the opening paren: `self, ` or `self`
        paren_line, paren_col = func.end_lineno, func.end_col_offset
        # the paren follows the attribute, possibly after whitespace
        seg = lines[paren_line - 1]
        j = paren_col
        while j < len(seg) and seg[j] != "(":
            j += 1
        if func.attr in static_names:
            continue  # a static helper takes no receiver
        has_args = bool(call.args or call.keywords)
        edits.append((paren_line, j + 1, paren_line, j + 1, f"{receiver}, " if has_args else receiver))

    def apply_point_edits(src_lines: list[str], point_edits):
        for line, col, end_line, end_col, new in sorted(point_edits, key=lambda e: (e[0], e[1]), reverse=True):
            if line != end_line:
                raise SystemExit("a call target spanning lines is not handled")
            row = src_lines[line - 1]
            src_lines[line - 1] = row[:col] + new + row[end_col:]
        return src_lines

    lines = apply_point_edits(lines, edits)

    # lift the definitions: cut from the class, paste above it
    lifted: list[str] = []
    cuts: list[tuple[int, int]] = []
    for node in chosen:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        # a comment block directly above the def travels with it
        while start - 2 >= 0 and lines[start - 2].strip().startswith("#"):
            start -= 1
        end = node.end_lineno
        block = lines[start - 1:end]
        if node.decorator_list:  # @staticmethod: a module function needs no marker
            block = [row for row in block if row.strip() != "@staticmethod"]
        indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
        dedented = [(row[indent:] if row[:indent].strip() == "" else row.lstrip()) if row.strip() else row for row in block]
        lifted.append("".join(dedented))
        cuts.append((start, end))
    for start, end in sorted(cuts, reverse=True):
        # also drop one blank line that separated it from the next method
        stop = end
        while stop < len(lines) and lines[stop].strip() == "" and stop - end < 1:
            stop += 1
        del lines[start - 1:stop]
    class_start = min([cls.lineno] + [d.lineno for d in cls.decorator_list])
    insert_at = class_start - 1
    # keep the class's own leading comment attached to the class
    while insert_at - 1 >= 0 and lines[insert_at - 1].strip().startswith("#"):
        insert_at -= 1
    lines[insert_at:insert_at] = [text_block + "\n\n" for text_block in lifted]
    new_text = "".join(lines)
    ast.parse(new_text)  # must still be a module
    if dry_run:
        print(f"{path}::{class_name}: would demote {len(chosen)}: " + ", ".join(sorted(names)))
        return len(chosen)
    path.write_text(new_text, encoding="utf-8")
    print(f"{path}::{class_name}: demoted {len(chosen)} method(s) to module level: " + ", ".join(sorted(names)))
    return len(chosen)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", help="FILE::Class")
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    total = 0
    for spec in args.specs:
        file_part, _, class_name = spec.partition("::")
        total += demote(ROOT / file_part, class_name, args.pattern, args.dry_run)
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
