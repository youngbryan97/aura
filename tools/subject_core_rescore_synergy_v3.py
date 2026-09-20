#!/usr/bin/env python3
"""ISC-v3's lines for the one campaign recorded before v3, and its instrument check for any v3 run.

    python tools/subject_core_rescore_synergy_v3.py RUN_DIR [...]

docs/ISC_V3_PREREGISTRATION.md names one campaign whose runs are rescored: the
ISC-v2 campaign recording at 1a9ebe561 on seeds 7, 11 and 13, none of which had
written a report when v3 was committed. For such a run this scores Aura's
declared triples on the run's own saved recording, and each null architecture's
on the toy recording its architecture and the run's seed produce, with the code
as committed. It first checks that the v2 reading it recomputes is the one the
run recorded, for Aura and for every null, and reads nothing under v3 where it
is not.

For every run read under v3, rescored or recorded that way, it also runs the
check the preregistration asks for: the preregistered coupling added to each
triple's target at one and two spreads on the run's own recording, and the
line scored.

It never touches the run's report. It writes `isc_v3_rescored.json` beside it,
and refuses any other run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SIDECAR = "isc_v3_rescored.json"
#: The campaign the preregistration lets this rescore, by its fingerprint.
RESCORED_CAMPAIGN = "39493ebf6330169637c46feb8febc62f"
#: A report rounds a synergy fraction to four places.
ROUNDING = 5e-5
#: The couplings the instrument check adds, in the target component's own spread.
CHECK_SCALES = (1.0, 2.0)
#: The length each null's toy recording is scored at, as the runner scores it.
TOY_STEPS = 2500


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def coupled(recording: Any, source_a: str, source_b: str, target: str, scale: float) -> Any:
    """The preregistered coupling of two sources added to a target's first change component.

    The coupling is (a-b)+a*b of the sources' first components, standardised,
    and sized in that change component's own spread.
    """
    from core.subject.synergy import _components

    where = recording.slices[target]
    block = recording.x[:, where]
    a = _components(recording.domain(source_a)[:-1])[:, 0]
    b = _components(recording.domain(source_b)[:-1])[:, 0]
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    change = np.diff(block, axis=0)
    centred = change - change.mean(axis=0)
    column_spread = np.maximum(centred.std(axis=0), 1e-9)
    _, _, vectors = np.linalg.svd(centred / column_spread, full_matrices=False)
    unit = vectors[0] * column_spread
    unit /= np.linalg.norm(unit)
    spread = float(np.std(centred @ unit))
    term = (a - b) + a * b
    term = (term - term.mean()) / term.std()
    pushed = block.copy()
    pushed[1:] += np.cumsum(np.outer(scale * spread * term, unit), axis=0)
    x = recording.x.copy()
    x[:, where] = pushed
    return replace(recording, x=x)


def instrument_check(recording: Any, seed: int) -> list[dict[str, Any]]:
    """Whether the v3 line registers the preregistered coupling on this recording, per triple."""
    from core.subject.synergy import TRIPLES, synergy

    rows = []
    for source_a, source_b, target in TRIPLES:
        entry: dict[str, Any] = {"triple": f"{source_a}+{source_b}->{target}"}
        for scale in CHECK_SCALES:
            report = synergy(coupled(recording, source_a, source_b, target, scale), source_a, source_b, target,
                             seed=seed, of="change")
            entry[f"passes_v3_at_{scale:g}_spreads"] = report.passes_v3
        entry["the_instrument_can_see_it"] = bool(entry["passes_v3_at_2_spreads"])
        rows.append(entry)
    return rows


def _aura_mismatches(again: list[dict[str, Any]], recorded: list[dict[str, Any]]) -> list[str]:
    found = []
    if len(again) != len(recorded):
        return [f"{len(again)} triples recomputed against {len(recorded)} recorded"]
    for new, old in zip(again, recorded, strict=True):
        name = f"{'+'.join(new['sources'])}->{new['target']}"
        if new["sources"] != old.get("sources") or new["target"] != old.get("target"):
            found.append(f"{name}: recorded as {old.get('sources')}->{old.get('target')}")
        elif bool(new["passes"]) != bool(old.get("passes")):
            found.append(f"{name}: v2 passes {new['passes']} against recorded {old.get('passes')}")
        elif abs(float(new["synergy_fraction"]) - float(old.get("synergy_fraction", 0.0))) > ROUNDING:
            found.append(f"{name}: fraction {new['synergy_fraction']} against recorded {old.get('synergy_fraction')}")
    return found


def rescore(run: Path) -> dict[str, Any]:
    """The v3 reading, the verdict it gives and the instrument check, for one run directory."""
    from core.subject.battery import assemble
    from core.subject.intrinsic import persistence
    from core.subject.nulls import architecture, toy_recording
    from core.subject.recording import load_recording
    from core.subject.synergy import synergy_suite

    report = json.loads((run / "subject_core_report.json").read_text(encoding="utf-8"))
    campaign = report.get("campaign") or {}
    recorded_v2 = report.get("synergy_v2") or []
    recorded_v3 = bool(recorded_v2) and all("passes_v3" in row for row in recorded_v2)
    out: dict[str, Any] = {"run": str(run), "campaign": campaign.get("fingerprint"), "computed_at_commit": _commit()}
    if not recorded_v3 and campaign.get("fingerprint") != RESCORED_CAMPAIGN:
        return {**out, "refused": "neither a v3 run nor a run of the campaign the preregistration rescores"}
    seed = (campaign.get("frozen") or {}).get("seed")
    if seed is None:
        return {**out, "refused": "the run recorded no seed"}
    seed = int(seed)
    out["seed"] = seed
    turns = load_recording(run).by_turn()
    evidence = dict(report)

    if not recorded_v3:
        rows = [item.as_dict() for item in synergy_suite(turns, seed=seed, of="change")]
        mismatches = _aura_mismatches(rows, recorded_v2)
        nulls = report.get("nulls") or {}
        table = {name: dict(row) for name, row in (nulls.get("detail") or {}).items()}
        for name, row in table.items():
            if row.get("kind") != "architecture":
                continue
            toy = toy_recording(architecture(name, seed=seed), steps=TOY_STEPS, seed=seed)
            reports = synergy_suite(toy, seed=seed, of="change")
            again = [bool(item.passes) for item in reports]
            if again != row.get("synergy_v2_passes"):
                mismatches.append(f"null {name}: v2 passes {again} against recorded {row.get('synergy_v2_passes')}")
            row["synergy_v3_passes"] = [bool(item.passes_v3) for item in reports]
        out["reproduced"] = not mismatches
        out["mismatches"] = mismatches
        if mismatches:
            return {**out, "refused": "the recomputed v2 reading is not the one the run recorded"}
        for row in rows:
            row["rescored_after_the_run"] = True
        evidence["synergy_v2"] = rows
        evidence["nulls"] = {**nulls, "detail": table}
        out["synergy_v2"] = rows
        out["nulls_detail"] = table
    else:
        out["reproduced"] = None

    if not evidence.get("persistence_v2"):
        reading = persistence(turns, seed=seed).as_dict()
        reading["without_memory"] = persistence(turns, seed=seed, exclude=("M",)).as_dict()
        reading["rescored_after_the_run"] = True
        evidence["persistence_v2"] = reading
        out["persistence_v2"] = reading

    verdict = assemble(evidence)
    out["v2_criteria"] = [item.as_dict() for item in verdict.v2_criteria]
    out["v3_criteria"] = [item.as_dict() for item in verdict.v3_criteria]
    out["isc_v3_on_this_seed"] = verdict.isc_v3_on_this_seed
    out["instrument_check"] = instrument_check(turns, seed)
    out["note"] = (
        "rescored under docs/ISC_V3_PREREGISTRATION.md from the run's own saved recording and the "
        "null architectures its seed produces; the run's report is unchanged"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        result = rescore(run)
        if "refused" in result and "reproduced" not in result:
            print(f"{run}: refused, {result['refused']}")
            continue
        (run / SIDECAR).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        if "refused" in result:
            print(f"{run}: {result['refused']}: {'; '.join(result['mismatches'])}")
            continue
        seen = sum(1 for row in result["instrument_check"] if row["the_instrument_can_see_it"])
        v3 = {row["criterion"]: row["passed"] for row in result["v3_criteria"]}
        print(f"{run}: seed {result['seed']}, v3 lines {v3}, the line registers two spreads on {seen} of "
              f"{len(result['instrument_check'])} triples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
