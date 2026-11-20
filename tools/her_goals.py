"""Her goals, and Bryan's answer to a draft she offers toward one.

    python tools/her_goals.py list
    python tools/her_goals.py show <slug>
    python tools/her_goals.py approve <slug>
    python tools/her_goals.py decline <slug> "what to change"
    python tools/her_goals.py improved

``improved`` reads each change of hers that landed as what its records show it did: a failure repaired, a cost
lowered, a drill, kept with nothing measured (core/agency/what_her_changes_improved.py).

``approve`` fast-forwards the source she runs on to her draft, only when that source has not moved past it; the change
reaches her when she is next started. ``decline`` hands the goal back to her with the words given.
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _when(at: float) -> str:
    return datetime.datetime.fromtimestamp(float(at or 0)).strftime("%d %b %H:%M") if at else "never"


def main() -> int:
    from core.agency.her_goal_at_work import all_records, approve, decline, load

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["list", "show", "approve", "decline", "improved"])
    parser.add_argument("slug", nargs="?")
    parser.add_argument("words", nargs="?", default="")
    asked = parser.parse_args()
    if asked.action == "improved":
        from core.agency.what_her_changes_improved import what_her_changes_improved
        from core.self_modification.faults_she_can_reproduce import faults_heard, recall

        recall()  # the faults kept from her last run, read; nothing is written
        told = what_her_changes_improved(all_records(), faults_heard())
        print("  ".join(f"{kind}: {count}" for kind, count in told["counts"].items() if count) or "nothing of hers has landed")
        for entry in told["changes"]:
            print(f"{_when(entry['landed_at'])}  {entry['kind']}: {entry['goal'][:80]}\n    {entry['evidence']}"
                  f"  ({entry['commit'][:9]}; {entry['undo'] or 'reverted'})")
        return 0
    if asked.action == "list":
        for record in all_records():
            print(f"{record.status:9} {record.slug}  {len(record.steps)} calls, {record.passes} passes, "
                  f"last {_when(record.last_worked_at)}  {record.goal[:90]}")
        return 0
    record = load(asked.slug or "")
    if record is None:
        print(f"no goal of hers named {asked.slug}")
        return 1
    if asked.action == "show":
        print(f"{record.goal}\nstatus: {record.status}; {len(record.steps)} calls over {record.passes} passes; "
              f"workspace {record.workspace}")
        for step in record.steps[-12:]:
            print(f"  {_when(step.get('at'))} {step.get('tool')}({', '.join(sorted(step.get('args') or {}))}) "
                  f"-> {str(step.get('result') or '')[:140]!r}")
        for said in record.said[-3:]:
            print(f"  she said: {said.get('words', '')[:300]}")
        if record.offer:
            print(f"\noffered {_when(record.offer.get('at'))}: {record.offer.get('summary')}")
            diff = subprocess.run(["git", "-c", "core.fsmonitor=false", "show", "--stat", record.offer.get("commit", "")],
                                  cwd=record.workspace, capture_output=True, text=True, check=False)
            print(diff.stdout[-3000:])
            for name, gate in (record.offer.get("gates") or {}).items():
                print(f"  gate {name}: {'passed' if gate.get('passed') else 'FAILED'} in {gate.get('seconds')} s")
            if record.offer.get("taken_by"):
                print(f"  taken by {record.offer['taken_by']} {_when(record.offer.get('merged_at'))}")
        return 0
    result = approve(record.slug) if asked.action == "approve" else decline(record.slug, asked.words)
    print(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
