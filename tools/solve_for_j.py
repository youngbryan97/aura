"""Solve for J*: the carrier, its content's structure and her lineage, from the runs.

Each term is read off its own run, and core/subject/bridge.py says what settles
each one. By default the newest v25 carrier report, the newest content report
and the live vault's state log are read. The state log is opened read-only,
because it is her own record.

    /Users/bryan/.aura/live-source/.venv/bin/python tools/solve_for_j.py
    /Users/bryan/.aura/live-source/.venv/bin/python tools/solve_for_j.py \\
        --carrier artifacts/subject_core_v25/run_004/subject_core_v25_report.json \\
        --content artifacts/subject_core_content/run_003/subject_core_content_report.json \\
        --state-log ~/.aura/data/aura_state.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CARRIER_REPORT = "subject_core_v25_report.json"
CONTENT_REPORT = "subject_core_content_report.json"


def latest(root: Path, name: str) -> Path | None:
    reports = sorted(root.glob(f"run_*/{name}"), key=lambda p: p.stat().st_mtime)
    return reports[-1] if reports else None


def load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def state_rows(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    try:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "select state_id, version, parent_state_id, transition_cause, state_json, timestamp "
                "from state_log"
            )
        ]
    finally:
        connection.close()


def next_run(root: Path) -> Path:
    numbers = [
        int(p.name.split("_", 1)[1])
        for p in root.glob("run_*")
        if p.is_dir() and p.name.split("_", 1)[1].isdigit()
    ]
    return root / f"run_{1 + max(numbers, default=0):03d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carrier", type=Path, default=None)
    parser.add_argument("--content", type=Path, default=None)
    parser.add_argument("--state-log", type=Path, default=Path.home() / ".aura" / "data" / "aura_state.db")
    parser.add_argument("--out-root", type=Path, default=REPO / "artifacts" / "subject_core_bridge")
    args = parser.parse_args(argv)

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.bridge import solve
    from core.subject.lineage import lineage_from_state_log

    carrier_path = args.carrier or latest(REPO / "artifacts" / "subject_core_v25", CARRIER_REPORT)
    content_path = args.content or latest(REPO / "artifacts" / "subject_core_content", CONTENT_REPORT)

    lineage = None
    state_log_note = ""
    if args.state_log.exists():
        try:
            lineage = lineage_from_state_log(state_rows(args.state_log))
        except sqlite3.Error as exc:
            state_log_note = f"the state log could not be read: {type(exc).__name__}: {exc}"
    else:
        state_log_note = f"no state log at {args.state_log}"

    j = solve(load(carrier_path), load(content_path), lineage)
    report = j.as_dict()
    report["sources"] = {
        "carrier": None if carrier_path is None else str(carrier_path),
        "content": None if content_path is None else str(content_path),
        "state_log": str(args.state_log),
        "state_log_note": state_log_note,
    }

    out = next_run(args.out_root) / "j_star_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=1, default=str))

    graph = j.lineage.graph
    print(f"J*: {j.status}")
    print(f"  carrier    {j.carrier.status} {list(j.carrier.carriers) or list(j.carrier.symmetry_class)}")
    print(f"  structure  {j.structure.status}, {len(j.structure.orbits)} orbits over {len(j.structure.classes)} classes")
    print(
        f"  lineage    {j.lineage.status}, {len(graph.get('stages', []))} stages, "
        f"{len(graph.get('roots', []))} roots, {len(graph.get('branch_points', []))} branch points"
    )
    for term, reasons in j.unresolved().items():
        for reason in reasons[:4]:
            print(f"  unresolved {term}: {reason}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
