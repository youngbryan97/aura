"""`except X: pass` — caught, and nothing whatsoever done about it.

CONTRIBUTING forbids the shape and 585 of them are in the tree. Most are
defensible: a QueueFull that falls through to shedding a worse task, a state
transition the machine has already recorded, a substrate that is not loaded
yet during boot. The problem is that a defensible one and a real swallow look
exactly the same from outside, so the rule cannot be enforced and the real
ones survive by camouflage.

So the rule this checks is not "never swallow". It is **say why**. A silent
handler with a reason beside it is a decision somebody made; one without is a
failure nobody will ever hear about, and the difference is a sentence.

192 of the 585 carry a reason today. The other 393 are the baseline, and it
only goes down.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import Any

__all__ = ["silent_swallows", "unexplained", "load_baseline", "main"]

ROOTS = ("core", "interface", "skills", "llm", "executors", "security")
BASELINE = pathlib.Path("config/silent_swallow_baseline.json")


def _says_nothing(handler: ast.ExceptHandler) -> bool:
    """The body is exactly `pass` or `...` and nothing else."""
    body = handler.body
    if len(body) != 1:
        return False
    only = body[0]
    if isinstance(only, ast.Pass):
        return True
    return (
        isinstance(only, ast.Expr)
        and isinstance(only.value, ast.Constant)
        and only.value.value is Ellipsis
    )


def silent_swallows(repo: pathlib.Path) -> list[tuple[str, int, bool]]:
    """Every `except X: pass`, and whether a comment says why."""
    found: list[tuple[str, int, bool]] = []
    for name in ROOTS:
        base = repo / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(text)
            except (SyntaxError, ValueError, OSError):
                continue
            lines = text.splitlines()
            for node in ast.walk(tree):
                if not (isinstance(node, ast.ExceptHandler) and _says_nothing(node)):
                    continue
                # A reason may sit above the `except`, on it, or on the `pass`.
                window = "\n".join(
                    lines[max(0, node.lineno - 3) : node.body[0].lineno + 1]
                )
                found.append(
                    (str(path.relative_to(repo)), node.lineno, "#" in window)
                )
    return found


def unexplained(repo: pathlib.Path) -> list[str]:
    return [
        f"{path}:{line}"
        for path, line, explained in silent_swallows(repo)
        if not explained
    ]


def load_baseline(path: pathlib.Path) -> int:
    try:
        return int(json.loads(path.read_text("utf-8"))["unexplained"])
    except (OSError, KeyError, TypeError, ValueError):
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    repo = pathlib.Path(args.repo).resolve()

    every = silent_swallows(repo)
    mute = unexplained(repo)
    if args.write_baseline:
        held = load_baseline(repo / BASELINE)
        keep = min(len(mute), held) if held else len(mute)
        (repo / BASELINE).write_text(
            json.dumps(
                {
                    "note": "`except X: pass` with no comment saying why. Only "
                    "goes down. A defensible silent handler and a real swallow "
                    "look identical from outside; the difference is a sentence.",
                    "silent_handlers": len(every),
                    "unexplained": keep,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {keep} unexplained of {len(every)} silent handlers")
        return 0

    held = load_baseline(repo / BASELINE)
    print(f"{len(every)} silent handlers, {len(mute)} with no reason (baseline {held})")
    if len(mute) > held:
        for one in mute[:20]:
            print("  ", one)
        print(f"\n{len(mute) - held} more than the baseline. Say why, or handle it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
