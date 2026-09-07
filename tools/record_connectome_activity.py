#!/usr/bin/env python3
"""Record Aura's own activity under a set of conditions, ZAPBench style.

Nine workloads stand in for ZAPBench's nine visual stimuli. Four are bounded
compute loops that exercise one subsystem hard; five run a slice of the offline
test suite, which is the broadest way to make a large amount of Aura's tissue
fire without booting a runtime beside the live one.

Each condition gets a wall-clock budget and stops at it, so the whole recording
is bounded whatever the suite does. The trace is written as compressed numpy
alongside a JSON manifest naming every cell, so the benchmark can be rerun
without recording again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")


def _bounded_pytest(selection: str, budget: float) -> str:
    import pytest

    class _Budget:
        def __init__(self, seconds: float) -> None:
            self.deadline = time.monotonic() + seconds

        def pytest_runtest_logstart(self, nodeid, location):  # noqa: ANN001, ARG002
            if time.monotonic() > self.deadline:
                pytest.exit("condition budget reached", returncode=0)

    code = pytest.main(
        [
            str(REPO / "tests"),
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-k",
            selection,
            "-m",
            "not live and not network and not external",
        ],
        plugins=[_Budget(budget)],
    )
    return f"pytest({selection}) -> {code}"


def _compute_reconstruction(budget: float) -> str:
    from core.connectome.volume import ReconstructionConfig, VolumeReconstructor

    deadline = time.monotonic() + budget
    rounds = 0
    while time.monotonic() < deadline:
        reconstructor = VolumeReconstructor(
            REPO, ReconstructionConfig(roots=("core",), max_files=900)
        )
        reconstructor.scan()
        reconstructor.build()
        rounds += 1
    return f"reconstruction rounds={rounds}"


def _compute_topology(budget: float) -> str:
    from core.connectome.topology import (
        DiGraphView,
        degree_preserving_rewire,
        reciprocity,
        rich_club,
    )
    from core.connectome.volume import ReconstructionConfig, VolumeReconstructor

    reconstructor = VolumeReconstructor(REPO, ReconstructionConfig(roots=("core",), max_files=700))
    reconstructor.scan()
    snapshot = reconstructor.build()
    graph = DiGraphView.from_snapshot(snapshot)
    deadline = time.monotonic() + budget
    rounds = 0
    while time.monotonic() < deadline:
        rewired = degree_preserving_rewire(graph, swaps_per_edge=1, seed=rounds)
        reciprocity(rewired)
        rich_club(rewired)
        rounds += 1
    return f"topology rounds={rounds}"


def _compute_typing(budget: float) -> str:
    from core.connectome.celltypes import adjusted_rand_index, refine_types
    from core.connectome.volume import ReconstructionConfig, VolumeReconstructor

    reconstructor = VolumeReconstructor(REPO, ReconstructionConfig(roots=("core",), max_files=700))
    reconstructor.scan()
    snapshot = reconstructor.build()
    deadline = time.monotonic() + budget
    rounds = 0
    previous = None
    while time.monotonic() < deadline:
        typing = refine_types(snapshot, rounds=1, drop_edges=0.05, seed=rounds)
        if previous is not None:
            adjusted_rand_index(previous.labels, typing.labels)
        previous = typing
        rounds += 1
    return f"typing rounds={rounds}"


def _compute_serialisation(budget: float) -> str:
    from core.connectome.synaptology import ei_report, measure_multiplicity
    from core.connectome.volume import ReconstructionConfig, VolumeReconstructor

    reconstructor = VolumeReconstructor(REPO, ReconstructionConfig(roots=("core",), max_files=700))
    reconstructor.scan()
    snapshot = reconstructor.build()
    deadline = time.monotonic() + budget
    rounds = 0
    while time.monotonic() < deadline:
        json.dumps(snapshot.summary(), sort_keys=True)
        json.dumps(measure_multiplicity(snapshot).as_json(), sort_keys=True)
        json.dumps(ei_report(snapshot), sort_keys=True)
        snapshot.digest()
        rounds += 1
    return f"serialisation rounds={rounds}"


CONDITIONS: tuple[tuple[str, object], ...] = (
    ("reconstruction", _compute_reconstruction),
    ("topology", _compute_topology),
    ("typing", _compute_typing),
    ("serialisation", _compute_serialisation),
    ("tests_memory", lambda b: _bounded_pytest("memory and not live", b)),
    ("tests_runtime", lambda b: _bounded_pytest("runtime and not live", b)),
    ("tests_verify", lambda b: _bounded_pytest("verify or invariant", b)),
    ("tests_governance", lambda b: _bounded_pytest("governance or policy", b)),
    ("tests_conversation", lambda b: _bounded_pytest("conversation or dialogue", b)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=240.0, help="seconds per condition")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "connectome")
    parser.add_argument("--frame-seconds", type=float, default=0.914)
    parser.add_argument("--only", type=str, default="", help="comma-separated condition names")
    args = parser.parse_args()

    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))
    args.out.mkdir(parents=True, exist_ok=True)

    from core.connectome.activity import ActivityRecorder, RecorderConfig

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    conditions = [c for c in CONDITIONS if not wanted or c[0] in wanted]
    total_budget = args.budget * len(conditions) + 600

    recorder = ActivityRecorder(
        REPO,
        RecorderConfig(
            frame_seconds=args.frame_seconds,
            capture_edges=True,
            max_wall_seconds=total_budget,
            max_frames=32_768,
        ),
    )
    log: list[dict[str, object]] = []
    recorder.start(conditions[0][0])
    for name, workload in conditions:
        recorder.set_condition(name)
        started = time.monotonic()
        try:
            detail = workload(args.budget)
        except BaseException as exc:  # noqa: BLE001 - one condition must not end the run
            detail = f"{type(exc).__name__}: {exc}"
        entry = {
            "condition": name,
            "seconds": round(time.monotonic() - started, 1),
            "detail": str(detail),
        }
        log.append(entry)
        print(json.dumps(entry), flush=True)
    trace = recorder.stop()

    import numpy as np

    matrix = trace.matrix()
    np.savez_compressed(args.out / "activity.npz", spikes=matrix)
    manifest = {
        "uids": list(trace.uids),
        "conditions": list(trace.conditions),
        "frame_seconds": trace.frame_seconds,
        "summary": trace.summary(),
        "attrs": trace.attrs,
        "log": log,
    }
    (args.out / "activity_manifest.json").write_text(json.dumps(manifest, indent=2))
    observed = {f"{pre}>{post}": count for (pre, post), count in recorder.observed.counts.items()}
    (args.out / "observed_edges.json").write_text(
        json.dumps({"counts": observed, "summary": recorder.observed.summary()}, indent=2)
    )
    print(json.dumps({"trace": trace.summary(), "observed": recorder.observed.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
