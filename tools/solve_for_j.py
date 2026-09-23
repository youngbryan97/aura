"""Solve for J*: the carrier, its content's structure and her lineage, from the runs.

Each term is read off its own run, and core/subject/bridge.py says what settles
each one. By default the newest v25 carrier report and the newest content
report are read, and her lineage comes from the newest state log among every
place a vault has written one. The state logs are opened read-only, because
they are her own record.

The bridge is judged beside J*, at parity (docs/BRIDGE_PARITY.md): the three
terms plus a campaign's battery (`--campaign`) and a report-grounding result
(`--reports`). A ground no run was given for reads NOT_MEASURED.

    /Users/bryan/.aura/live-source/.venv/bin/python tools/solve_for_j.py
    /Users/bryan/.aura/live-source/.venv/bin/python tools/solve_for_j.py \\
        --carrier artifacts/subject_core_v25/run_004/subject_core_v25_report.json \\
        --content artifacts/subject_core_content/run_003/subject_core_content_report.json \\
        --state-log ~/.aura/live-source/data/aura_state.db --state-log ~/.aura/data/aura_state.db
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


def default_state_logs() -> list[Path]:
    """Every place a vault has been started on.

    Which file the vault writes depends on the route that started it. The
    supervisor passes the configured data directory, resilient boot passes the
    proxy's relative default, which lands under whatever directory the process
    started in, and the container registration names a state subdirectory. So
    the newest log is found by reading all of them rather than trusting one.
    """
    from core.config import config

    data = Path(config.paths.data_dir)
    candidates = [
        data / "aura_state.db",
        data / "state" / "aura_state.db",
        Path.cwd() / "data" / "aura_state.db",
        REPO / "data" / "aura_state.db",
    ]
    return list(dict.fromkeys(candidates))


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)


def survey(path: Path) -> dict[str, Any]:
    """How many stages a state log holds and when it was last written."""
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    try:
        connection = _connect(path)
        try:
            rows, newest, highest = connection.execute(
                "select count(*), max(timestamp), max(version) from state_log"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out.update({"rows": rows, "newest": newest, "highest_version": highest})
    return out


def _person_part(data: Any) -> dict[str, Any]:
    """The part of a logged state the person reader looks at, and nothing else."""
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    if "identity" in data:
        out["identity"] = data["identity"]
    world = data.get("world")
    if isinstance(world, dict) and "relationship_graph" in world:
        out["world"] = {"relationship_graph": world["relationship_graph"]}
    return out


def state_rows(path: Path) -> list[dict[str, Any]]:
    """The log's rows, each carrying only the person part of its state.

    A live log runs to a gigabyte of state JSON. Keeping the person part of
    each row as it is read holds a fraction of that.
    """
    connection = _connect(path)
    try:
        rows: list[dict[str, Any]] = []
        for state_id, version, parent, cause, payload, stamp in connection.execute(
            "select state_id, version, parent_state_id, transition_cause, state_json, timestamp "
            "from state_log"
        ):
            try:
                data = json.loads(payload)
            except (TypeError, ValueError):
                data = {}
            rows.append(
                {
                    "state_id": state_id,
                    "version": version,
                    "parent_state_id": parent,
                    "transition_cause": cause,
                    "timestamp": stamp,
                    "state": _person_part(data),
                }
            )
        return rows
    finally:
        connection.close()


def next_run(root: Path) -> Path:
    numbers = [
        int(p.name.split("_", 1)[1])
        for p in root.glob("run_*")
        if p.is_dir() and p.name.split("_", 1)[1].isdigit()
    ]
    return root / f"run_{1 + max(numbers, default=0):03d}"


def battery_lines(report: dict[str, Any] | None) -> dict[str, bool] | None:
    """A campaign's lines by key under the newest version it was scored on, or None."""
    if not report:
        return None
    from core.subject.battery import assemble

    verdict = assemble(report)
    if verdict.v5_criteria:
        lines = verdict.v5_lines()
    elif verdict.v3_criteria:
        lines = verdict.v3_lines()
    elif verdict.v2_criteria:
        lines = verdict.v2_lines()
    else:
        lines = verdict.criteria
    return {line.key: bool(line.passed) for line in lines}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carrier", type=Path, default=None)
    parser.add_argument("--content", type=Path, default=None)
    parser.add_argument("--state-log", type=Path, action="append", default=None)
    parser.add_argument(
        "--campaign", type=Path, default=None,
        help="a campaign's subject_core_report.json: the battery lines the markers ground reads",
    )
    parser.add_argument(
        "--reports", type=Path, default=None,
        help="a report-grounding result, {measured, holds, why}: the reports ground",
    )
    parser.add_argument("--out-root", type=Path, default=REPO / "artifacts" / "subject_core_bridge")
    args = parser.parse_args(argv)

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.bridge import solve
    from core.subject.lineage import lineage_from_state_log

    carrier_path = args.carrier or latest(REPO / "artifacts" / "subject_core_v25", CARRIER_REPORT)
    content_path = args.content or latest(REPO / "artifacts" / "subject_core_content", CONTENT_REPORT)

    surveyed = [survey(path) for path in (args.state_log or default_state_logs())]
    readable = [entry for entry in surveyed if entry.get("rows")]
    chosen = max(readable, key=lambda entry: float(entry.get("newest") or 0.0), default=None)

    lineage = None
    state_log_note = ""
    if chosen is None:
        state_log_note = "no readable state log among " + ", ".join(entry["path"] for entry in surveyed)
    else:
        try:
            lineage = lineage_from_state_log(state_rows(Path(chosen["path"])))
        except sqlite3.Error as exc:
            state_log_note = f"the state log could not be read: {type(exc).__name__}: {exc}"

    j = solve(
        load(carrier_path),
        load(content_path),
        lineage,
        battery=battery_lines(load(args.campaign)),
        reports=load(args.reports),
    )
    report = j.as_dict()
    report["sources"] = {
        "carrier": None if carrier_path is None else str(carrier_path),
        "content": None if content_path is None else str(content_path),
        "state_log": None if chosen is None else chosen["path"],
        "state_logs_surveyed": surveyed,
        "state_log_note": state_log_note,
        "campaign": None if args.campaign is None else str(args.campaign),
        "reports": None if args.reports is None else str(args.reports),
    }

    out = next_run(args.out_root) / "j_star_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=1, default=str))

    graph = j.lineage.graph
    print(f"J*: {j.status}")
    print(f"  carrier    {j.carrier.status} {list(j.carrier.carriers) or list(j.carrier.symmetry_class)}")
    print(f"  structure  {j.structure.status}, {len(j.structure.orbits)} orbits over {len(j.structure.classes)} classes")
    print(
        f"  lineage    {j.lineage.status} from {report['sources']['state_log']}, "
        f"{len(graph.get('stages', []))} stages, {len(graph.get('roots', []))} roots, "
        f"{len(graph.get('branch_points', []))} branch points"
    )
    for term, reasons in j.unresolved().items():
        for reason in reasons[:4]:
            print(f"  unresolved {term}: {reason}")
    print(f"bridge: {j.bridge}")
    for reading in j.parity():
        print(f"  {reading.key:10s} {reading.status:12s} {reading.why}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
