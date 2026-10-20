"""Every place in her code where a string tells the model who to be or what to do.

Bryan, 13 October 2026: "i want to get rid of any part of this that is
prompting ... and implement as real systems engineering". This is the map of
what that covers, and the ratchet that holds what has been removed.

A string is counted when it is written to the model as an instruction: it
addresses the model in the second person ("You are", "Your task", "you
must"), or orders it ("Do not", "Never", "Always", "Respond with", "Return
only", "Output only"), or is a headed section of a prompt ("## "). Strings
under 80 characters are left out, since a label is not a prompt. What is
said, what she remembers and what a tool returned are content and are not
counted; only the text that shapes her is.

    where_she_is_still_prompted.py            the map, largest first
    where_she_is_still_prompted.py --check    fail if any file has grown past
                                              config/prompting_baseline.json
    where_she_is_still_prompted.py --baseline write the baseline as it stands
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


ROOTS = ("core", "interface", "skills", "llm", "executors")
BASELINE = REPO / "config" / "prompting_baseline.json"


_SHORTEST = 80


def _text_of(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str))
    return ""


def prompting_in(source: str) -> list[tuple[int, int]]:
    """(line, characters) for each instructing string in a module's source."""
    from core.brain.what_instructs import (
        INSTRUCTS as _INSTRUCTS,  # one reading for the map and the census
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    inside: set[int] = set()
    found: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if id(node) in docstrings or id(node) in inside:
            continue
        if isinstance(node, ast.JoinedStr):
            inside.update(id(part) for part in ast.walk(node))
        text = _text_of(node)
        if len(text) >= _SHORTEST and _INSTRUCTS.search(text):
            found.append((getattr(node, "lineno", 0), len(text)))
    return found


def the_map() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            found = prompting_in(path.read_text(encoding="utf-8", errors="replace"))
            if found:
                out[str(path.relative_to(REPO))] = {
                    "strings": len(found),
                    "characters": sum(size for _line, size in found),
                }
    return out


def main(argv: list[str]) -> int:
    found = the_map()
    total = sum(row["characters"] for row in found.values())
    if "--baseline" in argv:
        BASELINE.write_text(json.dumps({"total_characters": total, "files": found}, indent=1, sort_keys=True) + "\n")
        print(f"baseline written: {total} characters in {len(found)} files")
        return 0
    if "--check" in argv:
        held = json.loads(BASELINE.read_text())
        grown = [
            f"{name}: {row['characters']} > {held['files'].get(name, {}).get('characters', 0)}"
            for name, row in found.items()
            if row["characters"] > held["files"].get(name, {}).get("characters", 0)
        ]
        if grown:
            print("more prompting than the baseline holds:\n  " + "\n  ".join(grown))
            return 1
        print(f"OK: {total} characters of prompting, baseline {held['total_characters']}. It only goes down.")
        return 0
    print(f"{total} characters of prompting in {sum(r['strings'] for r in found.values())} strings across {len(found)} files")
    for name, row in sorted(found.items(), key=lambda kv: -kv[1]["characters"])[:40]:
        print(f"  {row['characters']:7d}  {row['strings']:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
