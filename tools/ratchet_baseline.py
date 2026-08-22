#!/usr/bin/env python3
"""One refusal, shared by every ratchet that records a baseline.

Three tools here wrote `json.dumps(current)` when handed `--write-baseline`.
That is not a ratchet. It is a record of the last time somebody ran the
refresh, and this repository has already had one refresh re-record 37 grown
entries as the new normal — after which nobody could tell which numbers were
earned and which were levelled up.

The rule this enforces is not "the number may never rise". Sometimes it must:
work lands that a gate cannot judge, or debt arrives from a direction the gate
did not anticipate, and a permanently red gate is a gate nobody can require of
a branch. The rule is that a rise is never SILENT. It needs
``--accept-growth --reason "..."``, and the reason is written into the
baseline beside the number it excuses, dated, where the next reader finds it.

    from tools.ratchet_baseline import guard_growth

    return guard_growth(
        current, previous, BASELINE, argv,
        counts=lambda payload: payload["by_file"],
        tool="tools/lint_lexical_debt.py",
    )
"""

from __future__ import annotations

import datetime as _datetime
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _today() -> str:
    return _datetime.datetime.now(tz=_datetime.UTC).date().isoformat()


def accepted_growth(argv: list[str]) -> tuple[bool, str]:
    """(did the caller ask to record growth, the reason they gave)."""
    accept = "--accept-growth" in argv
    reason = ""
    if "--reason" in argv:
        index = argv.index("--reason")
        if index + 1 < len(argv):
            reason = argv[index + 1].strip()
    return accept, reason


def guard_growth(
    current: dict[str, Any],
    previous: dict[str, Any],
    baseline_path: Path,
    argv: list[str],
    *,
    counts: Callable[[dict[str, Any]], dict[str, int]],
    tool: str,
    limit: int = 15,
) -> int:
    """Write the baseline, refusing to record growth nobody explained.

    ``counts`` pulls the per-entry numbers out of a measurement payload, so
    each tool keeps its own shape and this keeps the one rule.
    """
    now = counts(current)
    before = counts(previous) if previous else {}
    grew = {
        name: (before[name], value)
        for name, value in now.items()
        if name in before and value > before[name]
    }
    appeared = {
        name: value for name, value in now.items() if name not in before and before
    }

    accept, reason = accepted_growth(argv)
    rising = {**grew, **{k: (0, v) for k, v in appeared.items()}}

    if rising and not accept:
        print(f"❌ refusing to record growth in {len(rising)} entr(ies):")
        for name, (was, is_now) in sorted(
            rising.items(), key=lambda kv: kv[1][0] - kv[1][1]
        )[:limit]:
            print(f"    {name}: {was} -> {is_now}")
        print(
            f"\nPay it down, or record it deliberately:\n"
            f'    python {tool} --write-baseline --accept-growth --reason "why"'
        )
        return 1
    if rising and accept and not reason:
        print("❌ --accept-growth needs --reason; an unexplained reset is a reset")
        return 1

    payload = dict(current)
    notes = dict((previous or {}).get("growth_notes") or {})
    for name in rising:
        notes[name] = f"{_today()}: {reason}"
    for name in list(notes):
        if name not in now:
            del notes[name]
    if notes:
        payload["growth_notes"] = dict(sorted(notes.items()))

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written: {baseline_path.name}")
    if rising:
        print(f"   {len(rising)} entr(ies) recorded as grown — {reason}")
    return 0


def load(baseline_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":  # pragma: no cover - a library, not a command
    print(__doc__, file=sys.stderr)
    raise SystemExit(2)
