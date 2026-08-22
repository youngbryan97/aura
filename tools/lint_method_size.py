#!/usr/bin/env python3
"""The functions nobody can reason about, measured and held to a falling ceiling.

Four execution surfaces in this repository are large enough that the usual
tools stop working on them:

    interface/routes/chat.py::_api_chat_turn                   4,635 lines, CC ~731
    core/brain/llm/latent_cortex/engine.py::_latent_episode    4,432 lines, CC ~472
    core/brain/inference_gate.py::InferenceGate.generate       3,130 lines, CC ~630
    core/phases/response_generation_unitary.py::execute        3,084 lines, CC ~609

Cyclomatic complexity in the hundreds is not a style opinion. It is a
statement that the function has more independent paths than any test suite
will cover and more than any reader will hold at once. ``_api_chat_turn``
has 125 return points; 2,732 of this repo's 2,746 core files are smaller
than that one function.

The debt is unpaid, not unnoticed, and paying it down in one pass is the
wrong move: this is the code that serves every conversation, the branches
encode years of live incidents, and a behaviour-preserving rewrite of 4,635
lines cannot be validated by the offline suite alone. A rewrite that
silently drops one of those branches is worse than the size.

So this ratchets. Every listed function may only get smaller. That turns an
unbounded liability into a one-way one and puts the cost on whoever next
touches the function, which is the only time anyone has the context to
split it correctly.

A measured seam in ``_api_chat_turn``, so it is not re-derived
-------------------------------------------------------------
The hard part of splitting that function is not choosing where to cut, it is
proving the cut is behaviour-preserving. One seam has been measured and is
recorded here rather than left as folklore.

``interface/routes/chat.py`` lines 23398-23800 are the chat-preflight block —
one ``try`` with a coherent job (session identity, file references, resume
prefix, grounded recall, directive composition, affordance menu, context
clamp). Its interface with the rest of the function is small and exact:

* **reads 6** from the enclosing scope: ``body``, ``request``,
  ``_original_user_message``, ``_profile_user_id``, ``conversation_only_surface``,
  ``is_benchmark``
* **escapes 6**: ``_chat_session_id``, ``_grounded``, ``_grounded_recall_context``,
  ``_resume_prefix_for_response``, ``_shown``, ``status``
* **exactly one** ``return`` (line 23445, an early ``JSONResponse``), so the
  extraction needs a single optional-response sentinel rather than general
  control-flow surgery
* 7 ``await`` points, so the helper is ``async``
* no ``yield``, and no name is read before it is stored

The one hazard is that ``_grounded``, ``_shown`` and ``status`` are bound only
inside the block while being read after it. Giving them defaults in an
extracted result object would be a behaviour CHANGE, not a pure move — it
converts a possible ``UnboundLocalError`` into a default value. That has to be
resolved deliberately, with the chat-route suite, not folded silently into a
refactor. It is also the reason this seam is documented instead of already cut.

Run: ``python tools/lint_method_size.py`` / ``--write-baseline`` / ``--top``
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "method_size_baseline.json"

SCAN_ROOTS = ("core", "interface", "skills")
SKIP_DIR_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "archive",
    "node_modules",
    "tests",
}

#: Functions at or above this many lines are tracked. Set so the list is the
#: genuine outliers rather than every long function in the codebase.
TRACK_THRESHOLD_LINES = 400


def _complexity(node: ast.AST) -> int:
    branches = sum(
        1
        for child in ast.walk(node)
        if isinstance(
            child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                    ast.IfExp, ast.Assert, ast.With, ast.AsyncWith)
        )
    )
    booleans = sum(
        len(child.values) - 1
        for child in ast.walk(node)
        if isinstance(child, ast.BoolOp)
    )
    comprehensions = sum(
        len(child.generators)
        for child in ast.walk(node)
        if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    )
    return branches + booleans + comprehensions + 1


def _qualified_names(tree: ast.Module):
    """Yield (qualname, node) for every function, methods included."""

    def walk(node, prefix: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                # `yield from`, not a bare call: without it this generator
                # silently dropped every method in the codebase, which is how
                # a size gate comes to report that the largest methods do not
                # exist.
                yield from walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{prefix}{child.name}", child
                # Nested functions are part of the enclosing function's size;
                # counting them separately would let someone "shrink" a giant
                # by nesting more of it.
            else:
                yield from walk(child, prefix)

    yield from walk(tree, "")


def measure() -> dict[str, object]:
    tracked: dict[str, dict[str, int]] = {}
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(ROOT)
            if SKIP_DIR_PARTS.intersection(relative.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for qualname, node in _qualified_names(tree):
                lines = (node.end_lineno or node.lineno) - node.lineno + 1
                if lines < TRACK_THRESHOLD_LINES:
                    continue
                tracked[f"{relative}::{qualname}"] = {
                    "lines": lines,
                    "complexity": _complexity(node),
                    "returns": sum(
                        1 for c in ast.walk(node) if isinstance(c, ast.Return)
                    ),
                }
    return {
        "threshold_lines": TRACK_THRESHOLD_LINES,
        "tracked": len(tracked),
        "total_lines": sum(v["lines"] for v in tracked.values()),
        "functions": dict(sorted(tracked.items())),
    }


def _write_baseline(current: dict[str, object], argv: list[str]) -> int:
    """Record the measurement, and refuse to record growth.

    This used to write whatever it measured. A ratchet whose refresh command
    accepts any number is not a ratchet — it is a record of the last time
    somebody ran the refresh, and this repository has already had one refresh
    re-record 37 grown entries as the new normal.

    Growth is sometimes real and unavoidable. It is never silent: it needs
    ``--accept-growth --reason "..."``, and the reason is written into the
    baseline next to the function it excuses, where the next reader sees it.
    """
    functions: dict[str, dict[str, int]] = current["functions"]  # type: ignore[assignment]
    previous: dict[str, dict[str, int]] = {}
    notes: dict[str, str] = {}
    if BASELINE.is_file():
        recorded = json.loads(BASELINE.read_text(encoding="utf-8"))
        previous = recorded.get("functions") or {}
        notes = dict(recorded.get("growth_notes") or {})

    grew = {
        name: (previous[name]["lines"], stats["lines"])
        for name, stats in functions.items()
        if name in previous and stats["lines"] > previous[name]["lines"]
    }
    accept = "--accept-growth" in argv
    reason = ""
    if "--reason" in argv:
        index = argv.index("--reason")
        if index + 1 < len(argv):
            reason = argv[index + 1].strip()

    if grew and not accept:
        print(
            f"❌ refusing to write a baseline that records growth in "
            f"{len(grew)} function(s):"
        )
        for name, (was, now) in sorted(grew.items(), key=lambda kv: kv[1][0] - kv[1][1])[:15]:
            print(f"    {name}: {was} -> {now}")
        print(
            "\nShrink them, or record the growth deliberately with "
            '--accept-growth --reason "why this had to grow".'
        )
        return 1
    if grew and accept and not reason:
        print("❌ --accept-growth needs --reason; an unexplained ratchet reset is a reset")
        return 1

    for name in grew:
        notes[name] = reason
    for name in list(notes):
        if name not in functions:
            del notes[name]

    payload = dict(current)
    if notes:
        payload["growth_notes"] = dict(sorted(notes.items()))
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written: {BASELINE.relative_to(ROOT)}")
    if grew:
        print(f"   {len(grew)} function(s) recorded as grown, reason: {reason}")
    return 0


def _measure_source(source: str, relative: str) -> dict[str, dict[str, int]]:
    """Measure one file's oversized functions from a source string."""
    out: dict[str, dict[str, int]] = {}
    try:
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return out
    for qualname, node in _qualified_names(tree):
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        if lines < TRACK_THRESHOLD_LINES:
            continue
        out[f"{relative}::{qualname}"] = {
            "lines": lines,
            "complexity": _complexity(node),
            "returns": sum(1 for c in ast.walk(node) if isinstance(c, ast.Return)),
        }
    return out


def _changed_files(base: str) -> list[str]:
    def git(*args: str) -> str:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args], capture_output=True, text=True, cwd=ROOT, check=False
        )
        return result.stdout if result.returncode == 0 else ""

    merge_base = git("merge-base", "HEAD", base).strip() or base
    names = set()
    for diff in (
        git("diff", "--name-only", "--diff-filter=ACMR", merge_base, "HEAD"),
        git("diff", "--name-only", "--diff-filter=ACMR"),
        git("diff", "--name-only", "--diff-filter=ACMR", "--cached"),
    ):
        names |= {n for n in diff.splitlines() if n.endswith(".py")}
    return sorted(
        n for n in names if n.startswith(SCAN_ROOTS) and (ROOT / n).is_file()
    ), merge_base


def _check_changed(argv: list[str]) -> int:
    """No function you touched may get bigger.

    The whole-tree ratchet is red and will stay red until roughly twelve
    thousand lines of extraction land, which makes it a report rather than a
    gate — a job red on every push is a job everyone learns to ignore. This is
    the part that can block today and is the part that matters: the regression
    happens one edit at a time, in the file somebody is already changing.
    """
    base = "origin/main"
    if "--base" in argv:
        index = argv.index("--base")
        if index + 1 < len(argv):
            base = argv[index + 1]

    changed, merge_base = _changed_files(base)
    if not changed:
        print("✅ no production Python files changed")
        return 0

    grew: list[str] = []
    appeared: list[str] = []
    shrank: list[str] = []
    for relative in changed:
        after = _measure_source((ROOT / relative).read_text("utf-8", errors="ignore"), relative)
        before_source = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "show", f"{merge_base}:{relative}"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        before = (
            _measure_source(before_source.stdout, relative)
            if before_source.returncode == 0
            else {}
        )
        for name, stats in after.items():
            was = before.get(name)
            if was is None:
                if before_source.returncode == 0:
                    appeared.append(f"{name}: NEW at {stats['lines']} lines")
            elif stats["lines"] > was["lines"]:
                grew.append(f"{name}: {was['lines']} -> {stats['lines']} lines")
            elif stats["lines"] < was["lines"]:
                shrank.append(f"{name}: {was['lines']} -> {stats['lines']} lines")

    for line in shrank:
        print(f"  ⬇️  {line}")
    if grew or appeared:
        print(f"\n❌ this branch grew {len(grew) + len(appeared)} oversized function(s):")
        for line in grew + appeared:
            print(f"    {line}")
        print(
            "\nThe function you came to change is the one you have the context "
            "to split. Extract the part you touched."
        )
        return 1
    print(f"✅ no function grew across {len(changed)} changed file(s)")
    return 0


def main(argv: list[str]) -> int:
    if "--changed" in argv:
        return _check_changed(argv)

    current = measure()
    functions: dict[str, dict[str, int]] = current["functions"]  # type: ignore[assignment]
    print(
        f"functions at or over {TRACK_THRESHOLD_LINES} lines: {current['tracked']} "
        f"({current['total_lines']} lines total)"
    )

    if "--top" in argv:
        ranked = sorted(functions.items(), key=lambda kv: -kv[1]["lines"])
        for name, stats in ranked[:20]:
            print(
                f"  {stats['lines']:5d} lines  CC {stats['complexity']:4d}  "
                f"{stats['returns']:3d} returns  {name}"
            )
        return 0

    if "--write-baseline" in argv:
        return _write_baseline(current, argv)

    if not BASELINE.is_file():
        print(f"❌ no baseline at {BASELINE.relative_to(ROOT)}; run --write-baseline")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    previous: dict[str, dict[str, int]] = baseline.get("functions") or {}

    grew: list[str] = []
    appeared: list[str] = []
    shrank: list[str] = []
    for name, stats in functions.items():
        was = previous.get(name)
        if was is None:
            appeared.append(
                f"{name}: NEW at {stats['lines']} lines, CC {stats['complexity']}"
            )
        elif stats["lines"] > was["lines"]:
            grew.append(
                f"{name}: {was['lines']} -> {stats['lines']} lines "
                f"(CC {was['complexity']} -> {stats['complexity']})"
            )
        elif stats["lines"] < was["lines"]:
            shrank.append(f"{name}: {was['lines']} -> {stats['lines']} lines")

    for name in previous:
        if name not in functions:
            shrank.append(f"{name}: no longer over the threshold")

    if grew or appeared:
        print("\n❌ tracked functions grew, or a new one crossed the threshold:")
        for line in grew + appeared:
            print(f"    {line}")
        print(
            "\nThese are already past the point where a test suite can cover "
            "their paths or a reader can hold them. They may only shrink — "
            "extract the part you came to change."
        )
        return 1

    if shrank:
        print("\n⬇️  tracked functions shrank:")
        for line in shrank:
            print(f"    {line}")
        print("    refresh with: python tools/lint_method_size.py --write-baseline")
        return 1

    print("✅ no tracked function grew")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
