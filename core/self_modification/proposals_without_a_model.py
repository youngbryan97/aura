"""Changes to her code written by her own Python, not by a model: tried first, because each is exact.

A model can write any change and can be wrong in any way: an edit that does
not parse, a name that does not exist, a line it remembered differently. The
proposers here write only changes of shapes they know, each made exactly:

* the edits that mend a shape of code almost never meant
  (code_that_looks_wrong.py, the same suspicions the behaviour repair tries),
  each alone and those of one shape in one function together;
* the import a failure says is missing ("name 'X' is not defined"), copied
  from where her own code already imports that name, put after the file's
  imports instead of rewriting the file;
* what ruff's own fixes change in the file.

None of them is taken for right. Each is a candidate the tests decide on
(repair_by_selection.py), and a model is asked only once these are spent.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path

__all__ = ["import_for", "proposals_without_a_model"]

_UNDEFINED = re.compile(r"name '([A-Za-z_][A-Za-z0-9_]*)' is not defined")


def _block(path: str, old: str, new: str) -> str:
    return f"{path}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE"


def _whole(path: str, before: str, after: str) -> str:
    """A change to a file written as one block replacing the lines it touches, with a line either side."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    head = 0
    while head < min(len(a), len(b)) and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < min(len(a), len(b)) - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    lo, hi_a, hi_b = max(0, head - 1), len(a) - max(0, tail - 1), len(b) - max(0, tail - 1)
    old, new = "".join(a[lo:hi_a]).rstrip("\n"), "".join(b[lo:hi_b]).rstrip("\n")
    if before.count(old) != 1:
        old, new = before.rstrip("\n"), after.rstrip("\n")
    return _block(path, old, new)


def import_for(name: str, repo: Path) -> str:
    """The import of ``name`` her code uses most often, or "" when none imports it."""
    pattern = re.compile(rf"^(from [\w.]+ import (?:[\w, ]*\b{re.escape(name)}\b[\w, ]*)|import [\w.]+ as {re.escape(name)}|import {re.escape(name)})\s*$",
                         re.MULTILINE)
    seen: Counter[str] = Counter()
    for path in (repo / "core").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in pattern.finditer(text):
            line = m.group(1)
            if line.startswith("from "):
                module = line.split()[1]
                line = f"from {module} import {name}"
            seen[line] += 1
    return seen.most_common(1)[0][0] if seen else ""


def _with_import(source: str, statement: str) -> str:
    """The source with ``statement`` after its last top-level import, or after its docstring."""
    tree = ast.parse(source)
    after = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            after = node.end_lineno or node.lineno
        elif after == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            after = node.end_lineno or node.lineno
    lines = source.splitlines(keepends=True)
    return "".join([*lines[:after], statement + "\n", *lines[after:]])


def _ruff_fixed(source: str, path: str, repo: Path) -> str:
    try:
        done = subprocess.run([sys.executable, "-m", "ruff", "check", "--fix-only", "--quiet", "--stdin-filename", path, "-"],
                              input=source, capture_output=True, text=True, cwd=repo, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return source
    return done.stdout if done.returncode == 0 and done.stdout else source


def proposals_without_a_model(failure: str, sources: Mapping[str, str], repo: Path) -> Iterator[tuple[str, str]]:
    """Every exact change these proposers write for ``sources``, as (origin, SEARCH/REPLACE text)."""
    from core.self_modification.code_that_looks_wrong import applied, what_looks_wrong
    from core.self_modification.repairing_by_behaviour import _candidates

    for path, source in sources.items():
        for name in dict.fromkeys(_UNDEFINED.findall(failure)):
            statement = import_for(name, repo)
            if statement and statement not in source:
                yield "missing_import", _whole(path, source, _with_import(source, statement))
    for path, source in sources.items():
        suffix = Path(path).suffix
        try:
            suspicions = what_looks_wrong(source, suffix)
        except (ImportError, ValueError, RuntimeError):
            suspicions = []
        for suspicion, edits in _candidates(suspicions):
            changed = applied(source, edits)
            if changed != source:
                yield f"looks_wrong:{suspicion.pattern}", _whole(path, source, changed)
    for path, source in sources.items():
        if path.endswith(".py"):
            fixed = _ruff_fixed(source, path, repo)
            if fixed != source:
                yield "ruff", _whole(path, source, fixed)
