#!/usr/bin/env python3
"""Compiled output-filter patterns, counted and held to a falling ceiling.

``core/conversation/response_reliability.py`` reached 7,930 lines and 161
compiled regexes while its docstring said it "intentionally stays small".
The growth has one shape: a bad answer goes out, someone finds the substring
that characterised it, and a regex is added.

That is fixing the words. The standing rule in this codebase is to fix the
reasoning — never ban Aura from saying something, change what causes her to
say it — and each pattern added here is a cause nobody looked for, banked as
debt against the day a slightly different bad answer needs a slightly
different regex. ``re.compile(r"\\bm'?lol\\b")``, a regex for one garbled
token, was the clearest specimen; it is now
``has_malformed_contraction()``, which asks the general question the regex
was gesturing at.

Not every check in those files is lexical, and this tool does not pretend
otherwise — it counts ``re.compile`` calls, which is the population where
the debt lives. The number may only fall. Adding a pattern requires removing
one, which forces "what actually produced this output?" at exactly the
moment it is most tempting to skip.

Run: ``python tools/lint_lexical_debt.py`` / ``--write-baseline``
(the refresh refuses to record growth without ``--accept-growth --reason``)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "lexical_debt_baseline.json"

#: Modules whose job is deciding what Aura may say. Growth here is the debt.
WATCHED = (
    "core/conversation/response_reliability.py",
    "core/conversation/ontology_grounding.py",
    "core/dialogue/shared_history.py",
    "core/dialogue/referents.py",
)


def _compiled_patterns(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return 0
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "compile":
            if isinstance(func.value, ast.Name) and func.value.id == "re":
                count += 1
    return count


class Measurement(TypedDict):
    """What one run of this gate measured.

    `dict[str, object]` forced every reader to cast, and the casts were the
    only reason this file carried `type: ignore` comments at all.
    """

    total_patterns: int
    total_lines: int
    by_file: dict[str, dict[str, int]]


def measure() -> Measurement:
    by_file: dict[str, dict[str, int]] = {}
    for relative in WATCHED:
        path = ROOT / relative
        if not path.is_file():
            continue
        by_file[relative] = {
            "compiled_patterns": _compiled_patterns(path),
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
        }
    return {
        "total_patterns": sum(v["compiled_patterns"] for v in by_file.values()),
        "total_lines": sum(v["lines"] for v in by_file.values()),
        "by_file": by_file,
    }


def _pattern_counts(payload: dict[str, Any]) -> dict[str, int]:
    by_file = payload.get("by_file")
    if not isinstance(by_file, dict):
        return {}
    return {
        str(name): int(stats["compiled_patterns"])
        for name, stats in by_file.items()
        if isinstance(stats, dict)
    }


def main(argv: list[str]) -> int:
    current = measure()
    patterns = current["total_patterns"]
    lines = current["total_lines"]
    print(f"output-filter patterns: {patterns} across {lines} lines")
    for name, stats in current["by_file"].items():
        print(f"    {stats['compiled_patterns']:4d} patterns  {stats['lines']:6d} lines  {name}")

    if "--write-baseline" in argv:
        from tools.ratchet_baseline import guard_growth, load

        written: int = guard_growth(
            dict(current),
            load(BASELINE),
            BASELINE,
            argv,
            counts=_pattern_counts,
            tool="tools/lint_lexical_debt.py",
        )
        return written

    if not BASELINE.is_file():
        print(f"❌ no baseline at {BASELINE.relative_to(ROOT)}; run --write-baseline")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    allowed = int(baseline.get("total_patterns", 0))
    if patterns > allowed:
        print(f"\n❌ output-filter patterns rose: {allowed} -> {patterns}")
        previous = baseline.get("by_file") or {}
        for name, stats in current["by_file"].items():
            was = int((previous.get(name) or {}).get("compiled_patterns", 0))
            if stats["compiled_patterns"] > was:
                print(f"    {name}: {was} -> {stats['compiled_patterns']}")
        print(
            "\nA new pattern here is a bad answer whose cause nobody found. "
            "Ask what produced the output and change that; if the answer is "
            "genuinely lexical, retire a pattern to make room."
        )
        return 1

    if patterns < allowed:
        print(f"\n⬇️  output-filter patterns fell: {allowed} -> {patterns}")
        print("    refresh with: python tools/lint_lexical_debt.py --write-baseline")
        return 1

    print("\n✅ output-filter patterns held at baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
