"""Which columns of a recording never moved, and whether anything writes them.

A flat column is one of two things and the difference decides what to do about
it. Either nothing in the tree writes the value it names — a reader with no
writer, which is a defect and was the state of `soma.sensors` for every frame
of every campaign — or something writes it and the run never asked it anything,
which is a workload question and not a bug. Reading the recording alone cannot
tell those apart, and reading the code alone cannot tell you which columns
stayed still.

This does both. It takes the flat columns from a run's own recording, takes the
source each one declares in the schema, and looks for a writer of that source in
the tree. It reports rather than fails: a column with a writer that never fired
may still be the most interesting line in the output.

    tools/audit_flat_columns.py <run_NNN> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: Where a state path may be written from. The instrument writes readings of
#: her, so a writer inside it is the instrument watching itself.
_SEARCH_ROOTS = ("core", "interface", "skills")
_NOT_A_WRITER = ("core/subject/",)


def flat_columns(run: Path) -> tuple[list[str], dict[str, float], int]:
    """The columns that held one value for the whole recording."""
    with np.load(run / "core_state.npz", allow_pickle=True) as held:
        x = held["x"]
    manifest = json.loads((run / "core_state_manifest.json").read_text())
    names = [
        item["name"] if isinstance(item, dict) else str(item)
        for item in manifest["columns"]
    ]
    if len(names) != x.shape[1]:
        raise SystemExit(
            f"manifest names {len(names)} columns, recording has {x.shape[1]}"
        )
    flat: list[str] = []
    held_at: dict[str, float] = {}
    for index, name in enumerate(names):
        column = x[:, index]
        if len(np.unique(column)) <= 1:
            flat.append(name)
            held_at[name] = float(column[0])
    return flat, held_at, x.shape[0]


def _sources() -> dict[str, str]:
    """Each column's declared source, from the schema itself."""
    sys.path.insert(0, str(REPO))
    from core.subject.state import DOMAINS, _SCHEMAS

    out: dict[str, str] = {}
    for domain in DOMAINS:
        schema = _SCHEMAS[domain]
        for name, source in zip(schema.features, schema.sources, strict=True):
            out[f"{domain}.{name}"] = source
    return out


def _writer_of(source: str) -> str:
    """Where the tree assigns this source, or why the question does not apply."""
    if not source:
        return "no source declared"
    leaf = source.split(":", 1)[-1].split("[", 1)[0].rstrip(".").split(".")[-1]
    if not leaf:
        return "no leaf to search for"
    # Assignment, augmented assignment, keyword argument and mapping key, all
    # of which are ways the tree writes a value. Over-matching moves a row out
    # of the defect list, which is the safe direction for a report: a false
    # "nothing writes this" is a claim, and a false "something might" is a
    # question.
    return _grep(rf"\b{leaf}\b\s*[+*-]?=[^=]")


def _grep(pattern: str) -> str:
    try:
        found = subprocess.run(
            ["grep", "-rnE", "--include=*.py", pattern, *_SEARCH_ROOTS],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not look: {exc}"
    lines = [
        line
        for line in found.stdout.splitlines()
        if not any(line.startswith(skip) for skip in _NOT_A_WRITER)
        and "/tests/" not in line
    ]
    if not lines:
        return ""
    first = lines[0].split(":")
    return f"{first[0]}:{first[1]} ({len(lines)} sites)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a run_NNN directory")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    flat, held_at, rows = flat_columns(args.run)
    sources = _sources()
    unwritten: list[dict[str, Any]] = []
    unasked: list[dict[str, Any]] = []
    for name in flat:
        source = sources.get(name, "")
        writer = _writer_of(source)
        row = {
            "column": name,
            "source": source,
            "held_at": held_at[name],
            "writer": writer,
        }
        (unasked if writer else unwritten).append(row)

    print(f"{rows} frames, {len(flat)} flat columns of {len(sources)}")
    print(f"\n{len(unwritten)} with nothing in the tree that writes them:")
    for row in unwritten:
        print(f"  {row['column']:44s} = {row['held_at']:<10.6g} {row['source']}")
    print(f"\n{len(unasked)} written somewhere, never moved here:")
    for row in unasked:
        print(f"  {row['column']:44s} = {row['held_at']:<10.6g} {row['writer']}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"rows": rows, "unwritten": unwritten, "unasked": unasked}, indent=1
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
