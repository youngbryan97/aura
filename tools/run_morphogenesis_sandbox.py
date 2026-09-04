#!/usr/bin/env python3
"""Aura morphogenesis sandbox — offline, deterministic, model-free.

Runs the developmental scenarios against a population whose topology is real
state: work moves along declared bindings and nowhere else, so cutting one
removes a path and the computation notices.

    python tools/run_morphogenesis_sandbox.py --scenario task_shift --seed 42 --steps 20
    python tools/run_morphogenesis_sandbox.py --scenario all --json out.json
    python tools/run_morphogenesis_sandbox.py --ablations

Nothing here loads a model, opens a socket, reads live runtime state, or
touches the running instance. Set AURA_LOG_DIR before running so no test
artifact lands in the live log directory.

Exit code 0 when every executed scenario passes. A scenario that fails prints
what it measured and why the rule was not met; it is never hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.morphogenesis.scenarios import (  # noqa: E402
    SCENARIO_RUNNERS,
    run_ablation_matrix,
    run_scenario,
)


def _human(result: Any) -> str:
    lines = [
        f"  {result.scenario}",
        f"    verdict : {result.verdict}",
        f"    rule    : {result.verdict_rule}",
    ]
    for name, arm in sorted(result.arms.items()):
        metrics = arm.metrics
        lines.append(
            f"    {name:<17} score={arm.score:+.4f} "
            f"completion={metrics.get('completion_rate', 0.0):.3f} "
            f"sojourn={metrics.get('mean_sojourn', 0.0):6.2f} "
            f"cells={arm.final_cells:<3d} edges={arm.final_edges:<3d} "
            f"applied={arm.applied:<3d} refused={arm.rejected + arm.deferred:<3d} "
            f"rolled_back={arm.rolled_back}"
        )
    lines.append(f"    took    : {result.duration_s:.2f}s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline morphogenesis scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="scenario to run, or 'all' (default). One of: " + ", ".join(SCENARIO_RUNNERS),
    )
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed (default 42)")
    parser.add_argument(
        "--steps", type=int, default=20,
        help="scale of the run; each scenario derives its phases from this (default 20)",
    )
    parser.add_argument("--ablations", action="store_true", help="also run the ablation matrix")
    parser.add_argument("--only-ablations", action="store_true", help="run only the ablation matrix")
    parser.add_argument("--json", type=Path, default=None, help="write the full result to this path")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument(
        "--audit", action="store_true",
        help="print every seam to the rest of Aura, live or not, and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in SCENARIO_RUNNERS:
            print(name)
        return 0

    if args.audit:
        from core.morphogenesis.bridge import audit as bridge_audit

        report = bridge_audit()
        print("connections")
        for name, where in sorted(report["connections"].items()):
            print(f"  {name:<16} {where}")
        print("\nnot connected, and why")
        for name, why in sorted(report["not_connected"].items()):
            print(f"  {name:<40} {why}")
        print("\ngenotype")
        print(f"  {report['genotype']['composition']}")
        print("\nsubstrate roadmap")
        for candidate in report["substrate_roadmap"]["candidates"]:
            existing = ", ".join(candidate["existing"]) or "nothing in this tree"
            print(f"  phase {candidate['phase']}: {candidate['name']}")
            print(f"      existing   : {existing}")
            print(f"      blocked on : {candidate['blocked_on']}")
        topology = report["topology"]
        print("\nlive topology")
        if topology.get("online"):
            print(
                f"  v{topology['version']} {topology['cells']} cell(s), "
                f"{topology['bindings']} binding(s), {topology['components']} component(s)"
            )
        else:
            print("  no morphogenetic runtime in this process")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(f"\nwrote {args.json}")
        return 0

    if not os.environ.get("AURA_LOG_DIR"):
        # Never write a sandbox artifact into the live log directory.
        os.environ["AURA_LOG_DIR"] = "/tmp/aura-morph"

    started = time.monotonic()
    payload: dict[str, Any] = {
        "seed": args.seed,
        "steps": args.steps,
        "scenarios": {},
        "ablations": None,
    }
    failures: list[str] = []

    if not args.only_ablations:
        names = list(SCENARIO_RUNNERS) if args.scenario == "all" else [args.scenario]
        unknown = [n for n in names if n not in SCENARIO_RUNNERS]
        if unknown:
            parser.error(f"unknown scenario(s): {', '.join(unknown)}")
        print(f"morphogenesis sandbox — seed {args.seed}, steps {args.steps}\n")
        for name in names:
            result = run_scenario(name, seed=args.seed, steps=args.steps)
            payload["scenarios"][name] = result.to_dict()
            if not result.passed:
                failures.append(name)
            print(_human(result))
            print()

    if args.ablations or args.only_ablations:
        print("ablation matrix\n")
        matrix = run_ablation_matrix(seed=args.seed, steps=args.steps)
        payload["ablations"] = matrix
        rows = sorted(matrix["rows"].items(), key=lambda kv: -kv[1]["score"])
        for name, row in rows:
            print(
                f"  {name:<19} score={row['score']:+.4f} "
                f"delta={row['delta_vs_morphology_off']:+.4f} "
                f"completion={row['metrics']['completion_rate']:.3f} "
                f"sojourn={row['metrics']['mean_sojourn']:6.2f} "
                f"cells={row['final_cells']:<3d} applied={row['applied']}"
            )
        print(f"\n  baseline: {matrix['baseline']}   best: {matrix['best']}   spread: {matrix['spread']:+.4f}")
        if not matrix["discriminating"]:
            print(
                "  the arms did not separate; the load was inside what the seed population "
                "absorbs, so this matrix says nothing about morphology"
            )
        print()

    payload["duration_s"] = round(time.monotonic() - started, 3)
    payload["failures"] = failures

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")

    if payload["scenarios"]:
        passed = len(payload["scenarios"]) - len(failures)
        print(f"{passed}/{len(payload['scenarios'])} scenarios passed in {payload['duration_s']:.1f}s")
        if failures:
            print("failing: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
