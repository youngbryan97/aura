#!/usr/bin/env python3
"""Everything somebody else needs to run this battery and disagree with it.

A result the group that built the system is the only group to have produced is
not yet a result. What turns "Aura's own test says Aura passes" into something
arguable is a second party running the same protocol on their own machine and
either reproducing it or not.

This writes the package that makes that possible: the commit and tree the runs
came from, the exact command, the preregistered thresholds, a hash of every
artifact file, the verdict each criterion reached, and the protocol — including
what counts as a reproduction and what would falsify the result.

    python tools/subject_core_replication_package.py --latest 3
    python tools/subject_core_replication_package.py --verify path/to/theirs

Identical numbers are the wrong bar. The campaign steps a live organism with a
seeded generator, and a different machine has a different clock, a different
thread interleaving and a different floating-point library. What has to agree is
the verdict on each criterion, and each measure being inside the uncertainty the
original run reported for it. Two runs whose irreducibility differs by less than
its own standard error agree; two whose verdicts differ do not, whatever the
numbers look like.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS = REPO / "artifacts" / "subject_core"

#: The files a run must have written for its evidence to be checkable. A run
#: missing one of these is not a run somebody else can verify.
REQUIRED: tuple[str, ...] = (
    "subject_core_report.json",
    "campaign.json",
    "nulls.json",
    "edges.csv",
    "manifest.json",
    "core_state_manifest.json",
)


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _runs(latest: int) -> list[Path]:
    found = sorted(
        (p for p in RUNS.glob("run_*") if (p / "subject_core_report.json").exists()),
        key=lambda p: p.name,
    )
    return found[-latest:] if latest > 0 else found


def _verdicts(run: Path) -> dict[str, Any]:
    report = json.loads((run / "subject_core_report.json").read_text())
    verdict = report.get("verdict", {})
    rows: dict[str, Any] = {}
    for item in verdict.get("criteria", []) or []:
        if not isinstance(item, dict):
            continue
        rows[str(item.get("criterion"))] = {
            "passed": bool(item.get("passed")),
            "value": item.get("value"),
            "bar": item.get("bar"),
        }
    return {
        "run": run.name,
        "passed": verdict.get("passed"),
        "total": verdict.get("total"),
        "isc": verdict.get("isc"),
        "failed": verdict.get("failed", []),
        "criteria": rows,
        "phi_do": (report.get("phi", {}) or {}).get("phi_do"),
        "phi_standard_error": (report.get("phi", {}) or {}).get("standard_error"),
        "phi_held_out": (report.get("phi", {}) or {}).get("held_out"),
    }


def _protocol(pins: dict[str, Any], runs: list[Path]) -> str:
    command = pins.get("command", "")
    return f"""# Reproducing the Intrinsic Subject Core result

## What this is

The battery is a conjunction of twenty-four preregistered criteria over ten
domains of a live offline organism. Every threshold was written down before the
first run and none of them moved. The claim under test is that the ten domains
form one causally closed, recurrent, irreducible, differentiated process, and
the conjunction fails if any single line fails.

## What to run

Check out the pinned commit, build the environment, and run:

    {command}

The commit is `{pins.get('commit')}` and the tree hash is `{pins.get('tree')}`.
The state schema hash is `{pins.get('schema_hash')}`. If your schema hash
differs, you are measuring a different set of columns and the comparison below
does not apply.

Python was `{pins.get('python')}` on `{pins.get('platform')}`. The campaign
fingerprint covers every preregistered value; it is in `PINS.json` and each
run's own `campaign.json`.

## What counts as a reproduction

Not identical numbers. The campaign steps a live organism with a seeded
generator, and a different machine has a different clock, a different thread
interleaving and a different floating-point library. Two things have to agree:

1. **The verdict on each criterion.** `EXPECTED.json` carries the pass or fail
   this run reached for all twenty-four, per seed. A criterion that passes here
   and fails there is a disagreement worth reporting whatever the numbers are.

2. **Each measure inside the uncertainty the original reported.** The
   irreducibility score carries its own standard error and its five held-out
   fold readings. Two runs whose scores differ by less than that error agree
   about the measurement; a difference several errors wide does not, even if
   both land the same side of the bar.

`--verify` does both and prints the criteria that disagree.

## What would falsify it

- A null architecture passing the conjunction. Nineteen are built and each is
  built to fail for a stated reason; one of them passing means the battery
  cannot separate a mind from the thing it was told to reject.
- The recurrent reference failing the conjunction. It is the architecture the
  battery exists to say yes to, and an instrument that cannot say yes is not
  measuring.
- The irreducibility score falling below its own shuffled and replayed
  surrogates. Those keep every marginal and destroy only the alignment, so a
  real system scoring under them is scoring on the recording rather than on
  itself.
- Any criterion that passes on one seed and fails on another with no account of
  which is right. Three seeds are run for that reason.

## Attacking it rather than repeating it

Repeating the protocol checks the arithmetic. These attack the claim:

- Re-derive the metrics. `core/subject/irreducibility.py` states the
  forward-chaining fold scheme and why k-fold was wrong here; disagree with it.
- Score the same recordings with a different estimator. The reports carry the
  recording (`core_state.npz`) and its manifest, so the arrays are there.
- Build a stronger null. The nineteen in `core/subject/nulls.py` are the ones
  that were thought of.
- Check the thresholds against `THRESHOLDS.json` and the git history of
  `core/subject/battery.py`. A threshold that moved after a result is the thing
  most worth finding, and the history is the record.

## The runs in this package

{chr(10).join(f'- `{run.name}`' for run in runs)}

Every artifact file is hashed in `HASHES.txt`.
"""


def build(latest: int, out: Path) -> int:
    runs = _runs(latest)
    if not runs:
        print("no runs with a report to package", file=sys.stderr)
        return 1
    from core.subject.battery import THRESHOLDS

    missing: list[str] = []
    hashes: list[str] = []
    for run in runs:
        for name in REQUIRED:
            path = run / name
            if not path.exists():
                missing.append(f"{run.name}/{name}")
        for path in sorted(run.rglob("*")):
            if path.is_file() and "logs" not in path.parts and "state" not in path.parts:
                hashes.append(f"{_digest(path)}  {path.relative_to(RUNS)}")

    # From the runs' own recorded fingerprint rather than recomputed here. The
    # pins describe the runs in the package, and recomputing would describe
    # this moment instead — which is the tree as it is now, not the tree the
    # measurements came from.
    newest = json.loads((runs[-1] / "campaign.json").read_text())
    frozen = newest.get("frozen", {}) or {}
    pins = {
        "commit": newest.get("commit"),
        "commit_subject": newest.get("commit_subject"),
        "tree": newest.get("tree_hash"),
        "dirty": newest.get("dirty"),
        "schema_hash": (frozen.get("schema", {}) or {}).get("hash"),
        "schema_width": (frozen.get("schema", {}) or {}).get("width"),
        "campaign_fingerprint": newest.get("fingerprint"),
        "authoritative": newest.get("authoritative"),
        "authority_blockers": newest.get("authority_blockers", []),
        "python": newest.get("python") or platform.python_version(),
        "platform": newest.get("platform")
        or f"{platform.system()} {platform.release()} {platform.machine()}",
        # Everything the fingerprint covers, so a replicator can see what was
        # fixed before the first run rather than taking the hash on trust.
        "frozen": frozen,
        "command": (
            f"python tools/run_subject_core.py"
            f" --rounds {(frozen.get('recording', {}) or {}).get('rounds', '?')}"
            f" --trials {(frozen.get('intervention', {}) or {}).get('trials', '?')}"
            f" --lesion-rounds {(frozen.get('lesion', {}) or {}).get('rounds', '?')}"
            f" --seed SEED --out artifacts/subject_core"
            "   # once per declared seed; `make subject-core-frozen` does all three"
        ),
        "seeds": sorted(
            {
                (json.loads((run / "campaign.json").read_text()).get("frozen", {}) or {}).get("seed")
                for run in runs
            }
            - {None}
        ),
        "runs": [run.name for run in runs],
        "artifacts_missing": missing,
    }
    expected = [_verdicts(run) for run in runs]

    out.mkdir(parents=True, exist_ok=True)
    (out / "PINS.json").write_text(json.dumps(pins, indent=1, default=str) + "\n")
    (out / "EXPECTED.json").write_text(json.dumps(expected, indent=1, default=str) + "\n")
    (out / "THRESHOLDS.json").write_text(json.dumps(THRESHOLDS, indent=1, default=str) + "\n")
    (out / "HASHES.txt").write_text("\n".join(hashes) + "\n")
    (out / "PROTOCOL.md").write_text(_protocol(pins, runs))
    print(f"wrote {out} covering {len(runs)} run(s), {len(hashes)} hashed files")
    if missing:
        print(f"  {len(missing)} required artifact(s) missing: {missing[:4]}")
    return 0


def verify(theirs: Path, latest: int) -> int:
    """Compare somebody else's runs against what this machine reported."""
    mine = [_verdicts(run) for run in _runs(latest)]
    if not mine:
        print("nothing here to compare against", file=sys.stderr)
        return 1
    expected_path = theirs / "EXPECTED.json"
    if expected_path.exists():
        rows = json.loads(expected_path.read_text())
    else:
        found = sorted(
            (p for p in theirs.glob("run_*") if (p / "subject_core_report.json").exists()),
            key=lambda p: p.name,
        )
        if not found:
            print(f"no EXPECTED.json and no runs under {theirs}", file=sys.stderr)
            return 1
        rows = [_verdicts(run) for run in found]

    names = sorted({name for row in mine for name in row["criteria"]})
    print(f"{'criterion':30s} {'here':>12s} {'theirs':>12s}")
    disagreements: list[str] = []
    for name in names:
        ours = [r["criteria"].get(name, {}).get("passed") for r in mine]
        yours = [r["criteria"].get(name, {}).get("passed") for r in rows]
        here = f"{sum(1 for v in ours if v)}/{len(ours)}"
        there = f"{sum(1 for v in yours if v)}/{len(yours)}"
        mark = "" if here == there else "   <- differs"
        if mark:
            disagreements.append(name)
        print(f"{name:30s} {here:>12s} {there:>12s}{mark}")

    # And the measurement, not just the verdict: a score several standard
    # errors away is a different reading even when both land the same side.
    for row, other in zip(mine, rows, strict=False):
        phi_a, phi_b = row.get("phi_do"), other.get("phi_do")
        error = row.get("phi_standard_error") or 0.0
        if phi_a is None or phi_b is None:
            continue
        gap = abs(float(phi_a) - float(phi_b))
        inside = error > 0 and gap <= float(error)
        print(
            f"\n{row['run']} vs {other.get('run', '?')}: phi_do {float(phi_a):+.5f} "
            f"against {float(phi_b):+.5f}, gap {gap:.5f}, standard error "
            f"{float(error):.5f} -> {'inside' if inside else 'outside'} it"
        )
    print(f"\n{len(disagreements)} criterion(s) disagree" + (f": {disagreements}" if disagreements else ""))
    return 0 if not disagreements else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=int, default=3)
    parser.add_argument("--out", type=Path, default=RUNS / "replication")
    parser.add_argument("--verify", type=Path, default=None)
    args = parser.parse_args()
    if args.verify is not None:
        return verify(args.verify, args.latest)
    return build(args.latest, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
