#!/usr/bin/env python3
"""tools/check_raw_skill_execute.py — skills are entered through safe_execute.

`BaseSkill.safe_execute` held the active-runtime governance requirement and
described itself as the public entry point. Four live call sites called
`execute()` instead — `DesktopPlanner`'s adapter, three drivers in
`core/tools/computer_use.py`, and two capability modules — so the claim that no
consequential action runs outside the governed lane was false for the lane that
drives the mouse and keyboard.

`BaseSkill.__init_subclass__` now wraps every subclass's `execute` so a direct
call performs the same governance check. That closes the hole at runtime. This
gate closes it in the source, for two reasons the wrapper cannot cover:

  - The wrapper makes a raw call *governed*, not *correct*. It still skips
    input validation, the timeout, the circuit breaker, the retry policy and
    the result normalisation, so a raw call site is a latent defect even when
    it is safe.
  - A skill reached as a duck-typed object rather than a `BaseSkill` subclass
    never passes through `__init_subclass__` at all.

The baseline records the call sites that exist and may only shrink.

    python tools/check_raw_skill_execute.py             # check
    python tools/check_raw_skill_execute.py --baseline  # rewrite (shrink only)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "config" / "raw_skill_execute_baseline.json"
SEARCH_ROOTS = ("core", "interface", "tools")

#: Receiver names that denote a skill object. A bare `.execute(` matches
#: database cursors, thread pools, state machines and the capability engine, so
#: the receiver has to look like a skill for this to be about skills at all.
SKILL_RECEIVERS = ("skill", "skills")
SKILL_SUFFIXES = ("Skill", "_skill", "skill")


@dataclass(frozen=True)
class RawCall:
    path: str
    line: int
    receiver: str

    def key(self) -> str:
        return f"{self.path}:{self.receiver}"

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.receiver}.execute(...)"


def _receiver_name(node: ast.Attribute) -> str:
    value = node.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return ""


def looks_like_skill(receiver: str) -> bool:
    if not receiver:
        return False
    lowered = receiver.lower()
    if lowered in SKILL_RECEIVERS:
        return True
    return receiver.endswith(SKILL_SUFFIXES) or lowered.endswith("skill")


def scan_file(path: Path) -> list[RawCall]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    found: list[RawCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "execute":
            continue
        receiver = _receiver_name(func)
        if receiver == "self":
            continue  # BaseSkill.safe_execute calling its own body
        if not looks_like_skill(receiver):
            continue
        found.append(RawCall(rel, node.lineno, receiver))
    return found


def collect() -> list[RawCall]:
    found: list[RawCall] = []
    for root in SEARCH_ROOTS:
        base = PROJECT_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            found.extend(scan_file(path))
    return found


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.is_file():
        return {}
    try:
        data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("call_sites", {})
    return {str(k): int(v) for k, v in entries.items()} if isinstance(entries, dict) else {}


def write_baseline(found: list[RawCall]) -> int:
    counts: dict[str, int] = {}
    for call in found:
        counts[call.key()] = counts.get(call.key(), 0) + 1
    previous = load_baseline()
    grew = {
        key: (count, previous.get(key, 0))
        for key, count in counts.items()
        if count > previous.get(key, 0)
    }
    if previous and grew:
        print(
            "refusing to write a baseline that grows; this ratchet only shrinks:",
            file=sys.stderr,
        )
        for key, (now, before) in sorted(grew.items()):
            print(f"   {key}: {before} → {now}", file=sys.stderr)
        return 1
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": (
                    "Call sites entering a skill through raw execute() instead of "
                    "safe_execute. Only shrinks."
                ),
                "call_sites": dict(sorted(counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"baseline written: {sum(counts.values())} raw call site(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", action="store_true", help="rewrite the baseline (shrink only)"
    )
    args = parser.parse_args()

    found = collect()
    if args.baseline:
        return write_baseline(found)

    baseline = load_baseline()
    counts: dict[str, int] = {}
    for call in found:
        counts[call.key()] = counts.get(call.key(), 0) + 1

    regressions = [
        (key, count, baseline.get(key, 0))
        for key, count in sorted(counts.items())
        if count > baseline.get(key, 0)
    ]
    if regressions:
        print(
            f"{len(regressions)} new raw skill execute() call site(s). "
            "Enter skills through safe_execute:",
            file=sys.stderr,
        )
        for key, now, before in regressions:
            lines = [c.line for c in found if c.key() == key]
            print(f"   {key} (was {before}, now {now}) at line(s) {lines}", file=sys.stderr)
        return 1

    stale = sorted(set(baseline) - set(counts))
    if stale:
        print(
            "baseline lists call sites that no longer exist. "
            "Refresh it so the ratchet keeps ratcheting:",
            file=sys.stderr,
        )
        for key in stale:
            print(f"   {key}", file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"raw skill execute(): {total} call site(s), none new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
