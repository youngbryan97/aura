#!/usr/bin/env python3
"""ISC-v2's persistence line for a run recorded before the line existed.

    python tools/subject_core_rescore_persistence.py artifacts/subject_core/run_033 [...]

The third amendment to docs/ISC_V2_PREREGISTRATION.md replaced the persistence
line in the v2 verdict before any v2 result existed, and says how a v2 run
recorded before the amendment was implemented is read: by rescoring its own
saved recording with the measure as committed, with the rescored reading marked
as added after the run.

This does that and nothing more. It never touches the run's own report. It
writes `persistence_v2_rescored.json` beside it, carrying the reading, the
commit it was computed at, and the v2 lines the battery assembles from the
report with the reading added. A run that already recorded `persistence_v2` is
left alone, because its own reading is the one that counts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SIDECAR = "persistence_v2_rescored.json"


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rescore(run: Path, *, seed: int | None = None) -> dict[str, Any]:
    """The rescored reading and the v2 lines it gives, for one run directory."""
    from core.subject.battery import assemble
    from core.subject.intrinsic import persistence
    from core.subject.recording import load_recording

    report = json.loads((run / "subject_core_report.json").read_text(encoding="utf-8"))
    if report.get("persistence_v2"):
        return {"run": str(run), "skipped": "the run recorded persistence_v2 itself"}
    campaign_seed = (report.get("campaign") or {}).get("seed")
    use_seed = int(seed if seed is not None else (campaign_seed if campaign_seed is not None else 0))
    turns = load_recording(run).by_turn()
    reading = persistence(turns, seed=use_seed).as_dict()
    reading["without_memory"] = persistence(turns, seed=use_seed, exclude=("M",)).as_dict()
    reading["rescored_after_the_run"] = True
    reading["rescored_at_commit"] = _commit()
    reading["seed"] = use_seed

    verdict = assemble({**report, "persistence_v2": reading})
    v2 = [item.as_dict() for item in verdict.v2_criteria]
    return {
        "run": str(run),
        "persistence_v2": reading,
        "v2_criteria": v2,
        "v2_lines_passed": sum(1 for item in verdict.v2_lines() if item.passed) if v2 else None,
        "v2_lines": len(verdict.v2_lines()) if v2 else None,
        "note": (
            "rescored from the run's own saved recording under the third amendment; "
            "the run's report is unchanged"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=None, help="defaults to the campaign's own seed")
    args = parser.parse_args()
    for run in args.runs:
        result = rescore(run, seed=args.seed)
        if "skipped" in result:
            print(f"{run}: {result['skipped']}")
            continue
        (run / SIDECAR).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        reading = result["persistence_v2"]
        print(
            f"{run}: persistence lower bound {reading['gain_lower_bound']:+.4f}, over shuffle "
            f"{reading['over_shuffle_lower_bound']:+.4f}, passes {reading['passes']}; "
            f"v2 lines {result['v2_lines_passed']}/{result['v2_lines']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
