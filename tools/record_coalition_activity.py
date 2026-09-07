#!/usr/bin/env python3
"""Drive the consciousness stack so a recording can say what it does.

The coalition test asks whether influence runs around the ring from
interoception through affect, the workspace, higher-order monitoring, the self
model and planning to action. Answering it from a recording needs a recording in
which those stations fired, and the nine workloads used so far — test slices and
compute loops — barely touch them. Planning fired in none of them.

This drives each station directly and then drives them in ring order, so both
questions can be asked: what each station does on its own, and what happens when
they run together.

The probes are deliberately shallow. Every module in a station is imported, and
every module-level function that takes no required argument and reads rather
than writes — a getter, a status, a snapshot, a report — is called. That fires
real cells without asking the stations to do anything they were not already
willing to do when something asked them how they are.

Anything that raises is counted and skipped. A station that cannot be driven is
reported as one, because a station that could not be exercised and a station
that was exercised and did nothing are different findings.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

#: Names that read rather than change. A probe is only called when its name
#: starts with one of these and it needs no arguments.
_READ_PREFIXES: tuple[str, ...] = (
    "get_",
    "current_",
    "latest_",
    "read_",
    "snapshot",
    "status",
    "report",
    "describe",
    "summary",
    "state",
    "is_",
    "has_",
)

#: Method names on a returned object that are safe to call for the same reason.
_READ_METHODS: tuple[str, ...] = (
    "get_status",
    "status",
    "snapshot",
    "report",
    "summary",
    "describe",
    "to_dict",
    "as_json",
    "get_state",
)


def _probe_module(name: str, budget: float) -> dict[str, Any]:
    """Import one module and call its readers until the budget runs out."""
    import importlib

    outcome = {"module": name, "imported": False, "called": 0, "failed": 0}
    deadline = time.monotonic() + budget
    try:
        module = importlib.import_module(name)
    except BaseException as exc:  # noqa: BLE001 - a station may not import here
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        return outcome
    outcome["imported"] = True

    # Find the readers once, then call them until the budget is spent. Calling
    # each one once takes no measurable time, and a recording of a station that
    # fired for a hundredth of a second cannot be asked what the station does.
    probes = []
    for attribute in sorted(dir(module)):
        if not attribute.startswith(_READ_PREFIXES):
            continue
        value = getattr(module, attribute, None)
        if not callable(value):
            continue
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            continue
        if any(
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        ):
            continue
        probes.append(value)
    if not probes:
        return outcome
    outcome["probes"] = len(probes)
    while time.monotonic() < deadline:
        for probe in probes:
            try:
                result = probe()
                outcome["called"] += 1
            except BaseException:  # noqa: BLE001 - a probe must never end the run
                outcome["failed"] += 1
                continue
            for method_name in _READ_METHODS:
                method = getattr(result, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    outcome["called"] += 1
                except BaseException:  # noqa: BLE001
                    outcome["failed"] += 1
            if time.monotonic() > deadline:
                break
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=45.0, help="seconds per station")
    parser.add_argument("--rounds", type=int, default=3, help="passes around the ring")
    parser.add_argument("--frame-seconds", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "connectome" / "coalition")
    args = parser.parse_args()

    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))
    args.out.mkdir(parents=True, exist_ok=True)

    from core.connectome.activity import ActivityRecorder, RecorderConfig
    from core.connectome.coalition import COALITION_ORDER, STATION_TABLE
    from core.connectome.volume import VolumeReconstructor

    reconstructor = VolumeReconstructor(REPO)
    reconstructor.scan()
    snapshot = reconstructor.build()
    modules = sorted({unit.neuropil for unit in snapshot.units.values()})
    by_station = {
        station: [m for m in modules if any(p in m for p in patterns)]
        for station, patterns in STATION_TABLE.items()
    }
    print(json.dumps({k: len(v) for k, v in by_station.items()}), flush=True)

    recorder = ActivityRecorder(
        REPO,
        RecorderConfig(
            frame_seconds=args.frame_seconds,
            capture_edges=True,
            max_wall_seconds=args.budget * len(COALITION_ORDER) * args.rounds + 600,
            max_frames=32_768,
        ),
    )
    log: list[dict[str, Any]] = []
    recorder.start(COALITION_ORDER[0])
    for round_index in range(args.rounds):
        for station in COALITION_ORDER:
            recorder.set_condition(station)
            started = time.monotonic()
            per_module = max(0.05, args.budget / max(1, len(by_station.get(station, []))))
            results = [
                _probe_module(name, per_module) for name in by_station.get(station, [])
            ]
            entry = {
                "round": round_index,
                "station": station,
                "seconds": round(time.monotonic() - started, 1),
                "modules": len(results),
                "imported": sum(1 for r in results if r["imported"]),
                "calls": sum(r["called"] for r in results),
                "failures": sum(r["failed"] for r in results),
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)
    trace = recorder.stop()

    import numpy as np

    np.savez_compressed(args.out / "activity.npz", spikes=trace.matrix())
    (args.out / "activity_manifest.json").write_text(
        json.dumps(
            {
                "uids": list(trace.uids),
                "conditions": list(trace.conditions),
                "frame_seconds": trace.frame_seconds,
                "summary": trace.summary(),
                "attrs": trace.attrs,
                "log": log,
            },
            indent=2,
        )
    )
    observed = {f"{pre}>{post}": count for (pre, post), count in recorder.observed.counts.items()}
    (args.out / "observed_edges.json").write_text(
        json.dumps({"counts": observed, "summary": recorder.observed.summary()}, indent=2)
    )
    print(json.dumps({"trace": trace.summary(), "observed": recorder.observed.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
