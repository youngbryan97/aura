"""core/self_modification/coding_aci.py — a repository surface, not a shell.

Aura edits code through whatever a skill reached for: a shell command, a
subprocess, a direct write. The consequence is that a coding task is a sequence
of string operations against a filesystem, and every recurring problem in
repository work has to be solved again by whoever writes the next skill. Which
files matter. How to see a function without reading a 6,000-line file. Which
tests exercise a change. What a failure output actually said. How to undo.

The interface here is typed and small, and every operation is bounded:

* **Navigation returns structure.** :meth:`CodingSurface.outline` gives the
  definitions in a file with their line spans, so choosing what to read is a
  decision made on structure rather than on a full-file read.
* **Views are bounded.** :meth:`CodingSurface.view` refuses a request larger
  than its budget rather than returning it and letting the context absorb it.
* **Edits are structural.** :meth:`CodingSurface.replace_definition` swaps one
  function or class by name. A string replace that matches in two places is the
  edit that corrupts the second one.
* **Test selection is derived.** :meth:`CodingSurface.tests_touching` finds the
  tests that import or name what changed, so a change runs the tests that could
  fail rather than all of them or one guessed at.
* **Failure summarisation keeps the assertion.** A traceback is mostly frames;
  the line that matters is the assertion and the values in it, and
  :func:`summarise_failure` keeps those and drops the rest.
* **Every edit is undoable.** The surface holds the prior text and
  :meth:`CodingSurface.revert` restores it exactly.

The surface never runs anything. Execution goes through the sandbox and the
subprocess gateway, which already exist and already have the guarantees; adding
a second way to run a command is how a bypass gets built.
"""

from __future__ import annotations

import ast
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Definition",
    "Edit",
    "CodingSurface",
    "TooMuchToRead",
    "summarise_failure",
    "VIEW_BUDGET_LINES",
]

#: Lines a single view may return. Above this the caller is asked to narrow
#: rather than handed a file that will fill the context.
VIEW_BUDGET_LINES = 400


@dataclass(frozen=True, slots=True)
class Definition:
    """One function or class, and where it lives."""

    name: str
    kind: str
    start: int
    end: int
    parent: str = ""

    @property
    def qualified(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def lines(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "qualified": self.qualified, "kind": self.kind,
            "start": self.start, "end": self.end, "lines": self.lines,
        }


@dataclass(frozen=True, slots=True)
class Edit:
    """One change, with the text needed to undo it exactly."""

    path: str
    target: str
    before: str
    after: str

    @property
    def reversible(self) -> bool:
        return True


class TooMuchToRead(ValueError):
    """A view was larger than the budget. Narrow it rather than absorb it."""


class CodingSurface:
    """A bounded, structural view of a repository."""

    def __init__(self, root: Path, *, view_budget: int = VIEW_BUDGET_LINES) -> None:
        self._lock = threading.RLock()
        self.root = Path(root).resolve()
        self._budget = int(view_budget)
        self._edits: list[Edit] = []

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        target.relative_to(self.root)  # raises when outside
        return target

    def outline(self, path: str) -> list[Definition]:
        """Every definition in a file, so reading it whole is a choice not a default."""
        source = self._resolve(path).read_text(errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        found: list[Definition] = []

        def walk(node: ast.AST, parent: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(child, ast.ClassDef) else "function"
                    found.append(Definition(
                        name=child.name, kind=kind, start=child.lineno,
                        end=child.end_lineno or child.lineno, parent=parent,
                    ))
                    walk(child, child.name if isinstance(child, ast.ClassDef) else parent)

        walk(tree)
        return found

    def view(self, path: str, *, start: int = 1, end: int | None = None) -> str:
        """A bounded slice. Over budget, this refuses rather than returning it."""
        lines = self._resolve(path).read_text(errors="replace").splitlines()
        end = len(lines) if end is None else min(end, len(lines))
        span = end - start + 1
        if span > self._budget:
            raise TooMuchToRead(
                f"{path}:{start}-{end} is {span} lines against a budget of {self._budget}; "
                "narrow it with outline() rather than filling the context"
            )
        return "\n".join(lines[start - 1:end])

    def view_definition(self, path: str, name: str) -> str:
        """Read one function or class without reading the file around it."""
        definition = next(
            (d for d in self.outline(path) if d.qualified == name or d.name == name), None
        )
        if definition is None:
            raise KeyError(f"{name!r} is not defined in {path}")
        return self.view(path, start=definition.start, end=definition.end)

    def replace_definition(self, path: str, name: str, replacement: str) -> Edit:
        """Swap one definition by name. A string replace matching twice corrupts one."""
        target = self._resolve(path)
        definition = next(
            (d for d in self.outline(path) if d.qualified == name or d.name == name), None
        )
        if definition is None:
            raise KeyError(f"{name!r} is not defined in {path}")
        lines = target.read_text(errors="replace").splitlines()
        before = "\n".join(lines)
        after_lines = lines[: definition.start - 1] + replacement.splitlines() + lines[definition.end:]
        after = "\n".join(after_lines)
        target.write_text(after + "\n")
        edit = Edit(path=path, target=name, before=before, after=after)
        with self._lock:
            self._edits.append(edit)
        return edit

    def revert(self, edit: Edit) -> None:
        """Restore the file exactly as it was before this edit."""
        self._resolve(edit.path).write_text(edit.before + "\n")

    def tests_touching(self, names: Sequence[str], *, test_root: str = "tests") -> list[str]:
        """The tests that name or import what changed.

        A change should run the tests that could fail. Running all of them is
        slow and running a guessed subset is worse, because the guess is
        usually the tests the author already had in mind.
        """
        root = self._resolve(test_root)
        if not root.exists():
            return []
        needles = [n.split(".")[-1] for n in names]
        hits = []
        for path in sorted(root.rglob("test_*.py")):
            text = path.read_text(errors="replace")
            if any(re.search(rf"\b{re.escape(needle)}\b", text) for needle in needles):
                hits.append(str(path.relative_to(self.root)))
        return hits

    def report(self) -> dict[str, Any]:
        with self._lock:
            edits = list(self._edits)
        return {
            "root": str(self.root),
            "view_budget_lines": self._budget,
            "edits": [{"path": e.path, "target": e.target, "reversible": e.reversible} for e in edits],
            "all_reversible": all(e.reversible for e in edits),
        }


def summarise_failure(traceback_text: str, *, keep: int = 3) -> dict[str, Any]:
    """Keep the assertion and the values; drop the frames.

    A traceback is mostly frames and the frames are mostly the harness. What
    matters is the last file and line under test, the assertion, and the values
    it compared, and a summariser that keeps everything makes the model read
    the harness.
    """
    lines = [line.rstrip() for line in traceback_text.splitlines() if line.strip()]
    file_lines = [line for line in lines if line.strip().startswith("File ")]
    assertion = next(
        (line for line in reversed(lines)
         if line.lstrip().startswith(("E ", "assert", "AssertionError"))),
        "",
    )
    error = next(
        (line for line in reversed(lines) if re.match(r"^\w+(Error|Exception)\b", line.strip())),
        "",
    )
    values = [line for line in lines if line.lstrip().startswith("E ") and ("==" in line or "!=" in line)]
    return {
        "where": file_lines[-1].strip() if file_lines else "",
        "assertion": assertion.strip(),
        "error": error.strip(),
        "values": values[-keep:],
        "frames_dropped": max(0, len(file_lines) - 1),
        "original_lines": len(lines),
    }
