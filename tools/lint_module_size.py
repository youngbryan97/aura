#!/usr/bin/env python3
"""Module and class size, as a ratchet that only tightens.

`interface/routes/chat.py` is 29,481 lines with 457 module-level functions.
`core/brain/llm/mlx_client.py` is 15,423 with a 165-method class.
`core/brain/inference_gate.py` is 13,416 with a 193-method class that handles
worker processes, cloud fallback, health probing, warm-up, desktop resource
guards, background deferral, PII scrubbing, PBKDF2 offloading, RAM diagnostics
and UI prompt strings. Thirty-two files are over three thousand lines.

None of that is fixable in one commit, and pretending otherwise is how it stays
unfixed. What IS fixable in one commit is the direction of travel: nothing stops
chat.py reaching forty thousand lines, and nothing stops the next God object
being created from scratch. This is the same shape as the layering, effect-
ownership, async-write and bounded-await ratchets already in this repo — a
checked-in baseline that may shrink and may not grow.

Three rules:

1. A file NOT in the baseline may not exceed the thresholds at all. A new God
   object is never grandfathered.
2. A file that has shrunk must have its baseline refreshed. A stale entry is
   headroom nobody earned, and it is how a ratchet quietly stops ratcheting.
3. The TOTAL oversize — every baselined line above the threshold, summed — may
   never grow. Individual files may move within that total.

Rule 3 is a correction to this tool's own first design, which pinned every file
individually and failed the moment a legitimate feature touched one. That is how
a gate gets deleted: it blocks work it was never meant to block, someone removes
it, and the debt it was holding resumes growing unobserved. A per-file pin also
cannot express the trade this gate exists to encourage — moving four hundred
lines out of `chat.py` into three new modules should PASS, and under a per-file
rule it fails on the new modules.

The budget is what actually matters and it cannot be gamed quietly: a file may
grow only if another shrinks by more, and `--write-baseline` refuses to record a
larger total than the one already checked in.

The thresholds come from this repository's own distribution rather than from
taste: 2,000 lines is just under the 98th percentile of file length (2,115) and
30 methods is just above the 98th percentile of class size (26). A new file
above either is an outlier by the standard of the code around it.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "config" / "module_size_baseline.json"
SCANNED = ("core", "interface")

#: p98 of file length across the scanned tree (2,115 lines) rounded down. A new
#: file longer than 98% of everything already here is an outlier by the
#: codebase's own measure, not by an opinion about file length.
MAX_NEW_MODULE_LINES = 2_000

#: Just above p98 of class size (26 methods). Same reasoning.
MAX_NEW_CLASS_METHODS = 30

BASELINE_SCHEMA = "aura.module_size_baseline.v1"


@dataclass(frozen=True)
class Measurement:
    path: str
    lines: int
    max_class_methods: int
    largest_class: str


def measure(path: Path) -> Measurement | None:
    try:
        source = path.read_text("utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError):
        return None
    worst_name = ""
    worst = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        count = sum(
            1 for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if count > worst:
            worst = count
            worst_name = node.name
    return Measurement(
        path=str(path.relative_to(ROOT)),
        lines=len(source.splitlines()),
        max_class_methods=worst,
        largest_class=worst_name,
    )


def measure_tree(roots: tuple[str, ...] = SCANNED) -> dict[str, Measurement]:
    found: dict[str, Measurement] = {}
    for root in roots:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            measurement = measure(path)
            if measurement is not None:
                found[measurement.path] = measurement
    return found


def load_baseline(path: Path) -> dict[str, dict[str, int]]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    entries = payload.get("modules")
    return entries if isinstance(entries, dict) else {}


def write_baseline(path: Path, measurements: dict[str, Measurement]) -> int:
    """Record only what exceeds a threshold. A baseline of everything is noise.

    A refresh may lower an entry and may never raise one. The first version of
    this function recorded whatever it measured, so running it after a
    legitimate shrink somewhere else quietly re-recorded every file that had
    grown — 29 line baselines and 8 class-method baselines in one run. A
    refresh command that can loosen the gate is not a ratchet, it is a reset
    button with a ratchet's name on it.
    """
    modules = {
        m.path: {"lines": m.lines, "max_class_methods": m.max_class_methods}
        for m in measurements.values()
        if m.lines > MAX_NEW_MODULE_LINES or m.max_class_methods > MAX_NEW_CLASS_METHODS
    }
    total = oversize_total(measurements)

    existing = load_baseline(path)
    raised = []
    admitted = []
    for name, entry in list(modules.items()):
        previous_entry = existing.get(name)
        if previous_entry is None:
            # A module that was not in the baseline is a NEW God object, and
            # the rule this gate states is that a new one is never
            # grandfathered. Recording it here would grant it exactly the
            # headroom that rule refuses — a refresh run to bank an unrelated
            # shrink took seven of them out of the zero-tolerance class in one
            # command, which made "never grandfathered" one keystroke from
            # false. It stays out, and the gate keeps failing on it.
            if existing:
                admitted.append(name)
                del modules[name]
            continue
        for field in ("lines", "max_class_methods"):
            was = int(previous_entry.get(field, 0))
            if entry[field] > was:
                raised.append(f"{name}: {field} {was} -> {entry[field]}")
            entry[field] = min(entry[field], was)
    if admitted:
        print(
            f"refused to baseline {len(admitted)} module(s) that were never in "
            "it; a new God object is never grandfathered:"
        )
        for name in sorted(admitted)[:20]:
            print(f"   {name}")
    if raised:
        # Clamped, not refused: a real shrink somewhere else still deserves to
        # be banked. What must never happen is the growth being written down
        # as the new normal — the gate below still fails on every one of these.
        print(
            f"clamped {len(raised)} entry/entries that had grown; the baseline "
            "only shrinks and the gate still fails on them:"
        )
        for line in sorted(raised)[:20]:
            print(f"   {line}")

    # A refresh records the tightest thing ever seen and nothing looser. It
    # never blocks either: refusing to write means a real shrink somewhere
    # else can never be banked while any file anywhere has grown, and a gate
    # that cannot be satisfied is a gate that gets deleted. Growth is the
    # main() check's job to report, not this one's job to hide.
    previous = load_budget(path)
    if previous is not None:
        total = min(total, previous)

    path.write_text(
        json.dumps(
            {
                "schema": BASELINE_SCHEMA,
                "description": (
                    "Modules already above the size thresholds, and the total "
                    "oversize budget. Individual entries may move; the total may "
                    "only shrink. A file that has shrunk must be re-recorded, "
                    "because a stale entry is headroom nobody earned."
                ),
                "max_new_module_lines": MAX_NEW_MODULE_LINES,
                "max_new_class_methods": MAX_NEW_CLASS_METHODS,
                "oversize_budget_lines": total,
                "modules": dict(sorted(modules.items())),
            },
            indent=2,
        )
        + "\n"
    )
    return len(modules)


def oversize_total(measurements: dict[str, Measurement]) -> int:
    """Every line above the threshold, summed across the tree.

    The quantity the ratchet actually holds. Moving four hundred lines out of a
    God object into three new modules leaves this unchanged or lower, which is
    the trade the gate exists to allow; growing one file without shrinking
    another raises it, which is the trade it exists to refuse.
    """
    return sum(
        max(0, m.lines - MAX_NEW_MODULE_LINES) for m in measurements.values()
    )


def load_budget(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("oversize_budget_lines")
    return int(value) if isinstance(value, int) else None


def check(
    measurements: dict[str, Measurement],
    baseline: dict[str, dict[str, int]],
    *,
    budget: int | None = None,
) -> tuple[list[str], list[str]]:
    """Returns (failures, stale_entries)."""
    failures: list[str] = []
    stale: list[str] = []

    if budget is not None:
        total = oversize_total(measurements)
        if total > budget:
            failures.append(
                f"BUDGET  total oversize is {total} lines against a budget of "
                f"{budget} (+{total - budget}). A file may grow only if another "
                "shrinks by more; this is the one number the ratchet holds"
            )

    for path, measurement in sorted(measurements.items()):
        recorded = baseline.get(path)
        if recorded is None:
            if measurement.lines > MAX_NEW_MODULE_LINES:
                failures.append(
                    f"NEW     {path}: {measurement.lines} lines exceeds the "
                    f"{MAX_NEW_MODULE_LINES}-line ceiling for a module not already "
                    "in the baseline — a new God object is never grandfathered"
                )
            if measurement.max_class_methods > MAX_NEW_CLASS_METHODS:
                failures.append(
                    f"NEW     {path}: class {measurement.largest_class} has "
                    f"{measurement.max_class_methods} methods, over the "
                    f"{MAX_NEW_CLASS_METHODS}-method ceiling for a new class"
                )
            continue

        allowed_methods = int(recorded.get("max_class_methods", 0))
        if measurement.max_class_methods > allowed_methods:
            # Method count is pinned per class rather than budgeted. Splitting a
            # God class is the point; growing one is never the trade.
            failures.append(
                f"GREW +{measurement.max_class_methods - allowed_methods} "
                f"{path}: class {measurement.largest_class} grew to "
                f"{measurement.max_class_methods} methods from a baseline of "
                f"{allowed_methods}"
            )
        allowed_lines = int(recorded.get("lines", 0))
        if measurement.lines < allowed_lines or measurement.max_class_methods < allowed_methods:
            stale.append(
                f"{path}: now {measurement.lines} lines / "
                f"{measurement.max_class_methods} methods, baseline still says "
                f"{allowed_lines} / {allowed_methods}"
            )

    for path in sorted(baseline):
        if path not in measurements:
            stale.append(f"{path}: recorded in the baseline but no longer exists")

    return failures, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    measurements = measure_tree()

    if args.write_baseline:
        try:
            count = write_baseline(baseline_path, measurements)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"module size baseline written: {count} oversized module(s), "
            f"budget {oversize_total(measurements)} lines"
        )
        return 0

    baseline = load_baseline(baseline_path)
    if not baseline:
        print(
            f"error: no usable baseline at {baseline_path}; "
            "run with --write-baseline to record the current state",
            file=sys.stderr,
        )
        return 2

    failures, stale = check(measurements, baseline, budget=load_budget(baseline_path))

    oversized = sum(
        1
        for m in measurements.values()
        if m.lines > MAX_NEW_MODULE_LINES or m.max_class_methods > MAX_NEW_CLASS_METHODS
    )
    budget = load_budget(baseline_path)
    print(
        f"modules scanned: {len(measurements)}; over threshold: {oversized}; "
        f"oversize {oversize_total(measurements)} lines"
        + (f" against a budget of {budget}" if budget is not None else "")
    )

    if stale:
        print(f"\n📉 {len(stale)} baseline entry/entries are stale — refresh with")
        print("   python tools/lint_module_size.py --write-baseline")
        for entry in stale[:20]:
            print(f"   {entry}")
        if len(stale) > 20:
            print(f"   … and {len(stale) - 20} more")

    if failures:
        # Grouped and worst-first. The gate has carried an inherited pile for
        # weeks, and a flat list of thirty complaints is one a reader stops
        # reading — a module that went over the ceiling TODAY was indexed
        # somewhere in the middle of it and looked like everything else.
        # Nothing is forgiven here; the same failures fail. What changes is
        # that the two kinds are told apart and the pile is named as a pile.
        new = [f for f in failures if f.startswith("NEW ")]
        grew = [f for f in failures if f.startswith("GREW")]
        budgets = [f for f in failures if f.startswith("BUDGET")]
        other = [f for f in failures if f not in set(new) | set(grew) | set(budgets)]

        def _amount(entry: str) -> int:
            head = entry.split()[1]
            return int(head) if head.lstrip("+").isdigit() else 0

        print(f"\n❌ {len(failures)} size regression(s):")
        if new:
            print(
                f"\n   NEW — over the ceiling and never baselined ({len(new)}). "
                "A new God object is never grandfathered:"
            )
            for failure in sorted(new):
                print(f"     {failure[8:]}")
        if grew:
            print(
                f"\n   GREW — past a recorded baseline ({len(grew)}), worst first:"
            )
            for failure in sorted(grew, key=lambda f: (-_amount(f), f)):
                print(f"     {failure.split(None, 2)[2]}")
        for failure in budgets + other:
            print(f"\n   {failure.split(None, 1)[-1]}")
        return 1

    if stale:
        # A stale entry is headroom nobody earned. Refusing it is what keeps the
        # ratchet ratcheting rather than slowly becoming a record of history.
        return 1

    print("✅ no module grew past its baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
