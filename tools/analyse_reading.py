#!/usr/bin/env python3
"""Ask her recording the five questions a meta-analysis of 163 studies answered.

Each is written down in ``core.connectome.reading.HUMAN_READING`` before the
recording is made, with what would refuse it. What comes out is her answer, not
a resemblance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording", type=Path, default=REPO / "artifacts" / "connectome" / "reading"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.recording / "reading_analysis.json")

    import numpy as np

    from core.connectome.activity import ActivityTrace
    from core.connectome.reading import (
        HUMAN_READING,
        LEVELS,
        ReadingReport,
        compare_levels,
        core_and_specific,
        dual_route,
        profile_condition,
        task_over_stimulus,
    )
    from core.connectome.volume import VolumeReconstructor

    manifest = json.loads((args.recording / "activity_manifest.json").read_text())
    trace = ActivityTrace(
        uids=tuple(manifest["uids"]),
        conditions=tuple(manifest["conditions"]),
        spikes=[],
        array=np.load(args.recording / "activity.npz")["spikes"],
        frame_seconds=float(manifest.get("frame_seconds", 0.002)),
    )
    reconstructor = VolumeReconstructor(REPO)
    reconstructor.scan()
    snapshot = reconstructor.build()

    counts: dict[str, int] = {}
    for name in trace.conditions:
        counts[name] = counts.get(name, 0) + 1
    wanted = [*LEVELS, "pseudoword", "judge_word"]
    present = [name for name in wanted if counts.get(name, 0) >= 20]
    # Every condition measured over the same number of frames, so a profile's
    # size is about what fired rather than about how long it ran.
    cap = min(counts[name] for name in present) if present else 0
    profiles = {
        name: profile_condition(trace, snapshot, name, frames_cap=cap)
        for name in present
    }

    report = ReadingReport()
    report.profiles = {name: profile.as_json() for name, profile in profiles.items()}
    report.profiles["frames_per_condition"] = cap

    shape = core_and_specific(profiles)
    peak = compare_levels(profiles)
    routes = (
        dual_route(profiles["word"], profiles["pseudoword"])
        if "word" in profiles and "pseudoword" in profiles
        else {"skipped": "word or pseudoword did not fire"}
    )
    task = task_over_stimulus(
        profiles,
        same_task_pairs=[("word", "sentence"), ("sentence", "text"), ("letter", "word")],
        same_stimulus_pairs=[("word", "judge_word")],
    )

    core_share = shape.get("core_share_of_each") or {}
    report.findings = {
        "common_core": {
            "holds": bool(core_share) and min(core_share.values()) >= 0.5,
            "measured": shape,
        },
        "level_specificity": {
            "holds": bool(shape.get("specific"))
            and all(count > 0 for count in (shape.get("specific") or {}).values()),
            "measured": {
                "specific": shape.get("specific"),
                "specific_share": shape.get("specific_share"),
                "concentration": {
                    name: profile.as_json()["concentration"]
                    for name, profile in profiles.items()
                },
            },
        },
        "the_middle_recruits_most": {
            "holds": bool(peak.get("peaks_in_the_middle")),
            "measured": peak,
        },
        "dual_route": {"holds": bool(routes.get("two_routes")), "measured": routes},
        "task_beats_stimulus": {
            "holds": bool(task.get("task_beats_stimulus")),
            "measured": task,
        },
    }
    for finding in HUMAN_READING:
        row = report.findings.get(finding.name)
        if row is None:
            continue
        row["in_humans"] = finding.in_humans
        row["predicted_here"] = finding.predicted_here
        row["falsifier"] = finding.falsifier

    payload = report.as_json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for name, row in payload["findings"].items():
        print(f'{"HOLDS " if row["holds"] else "REFUSED"}  {name}')
        print("   " + json.dumps(row["measured"])[:220])
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
