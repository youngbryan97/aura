#!/usr/bin/env python3
"""Whether the same campaign on another machine gives the same answer.

A result that survives only on the machine it was measured on is a result about
that machine. The way to know is to run the frozen campaign somewhere else and
compare — but comparing two numbers is not enough, because two runs on one
machine already differ. What has to be separated is how much of the gap is the
machine and how much is the run.

So this takes run directories, groups them by the host they ran on, and reports
each measure two ways: the spread between runs on one machine, and the spread
between machines. A machine effect that is smaller than the within-machine
spread is not a machine effect. One that is larger is the scientifically
important case the specification names — an apparent integrated subject that
disappears under small timing differences.

It refuses to compare runs that are not the same experiment. Two runs of
different campaigns, or of the same campaign on different code, are two
experiments; reading across them would be the thing the fingerprint exists to
prevent.

    python tools/compare_across_machines.py artifacts/subject_core/run_0*
    python tools/compare_across_machines.py here/run_021 there/run_004 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: What is compared, and where it lives in a report. Named here so a measure
#: cannot be dropped from the comparison after seeing that it disagreed.
COMPARED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("phi_do", ("phi", "phi_do")),
    ("phi_lower_bound", ("phi", "lower_bound")),
    ("d_eff", ("differentiation", "d_eff")),
    ("d_eff_normalised", ("differentiation", "d_eff_normalised")),
    ("spread", ("perturbation", "mean_spread")),
    ("pci", ("perturbation", "mean_pci")),
    ("intrinsic", ("intrinsic", "delta_intrinsic")),
    ("vertex_connectivity", ("graph", "vertex_connectivity")),
    ("edges_kept", ("graph", "edge_count")),
    ("lesion_deficit_phi", ("lesion", "deltas", "phi_do")),
    ("lesion_deficit_spread", ("lesion", "deltas", "spread")),
    ("rescue_phi", ("lesion", "rescue", "phi_do")),
    ("ownership", ("agency", "ownership_divergence")),
    ("closure_leak", ("closure", "leak")),
    ("dwell_mean", ("metastability", "dwell_mean")),
    ("largest_regime_share", ("metastability", "largest_regime_share")),
    ("broadcast_consumers", ("global_access", "consumers")),
)


def _synergy(report: dict[str, Any]) -> dict[str, float]:
    """Each triple's synergy fraction, keyed by the triple.

    Synergy is a list rather than a number, and the list is the interesting
    part: two machines that agree on the best triple and disagree on which
    triples clear their null have not replicated the measure. So every triple
    is carried through by name, and the summary lines under it are derived
    rather than read.
    """
    rows = report.get("synergy", []) or []
    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sources = "+".join(str(s) for s in row.get("sources", []))
        target = str(row.get("target", ""))
        fraction = row.get("synergy_fraction")
        if sources and target and isinstance(fraction, (int, float)):
            out[f"{sources}->{target}"] = float(fraction)
    return out


def _synergy_passing(report: dict[str, Any]) -> set[str]:
    return {
        f"{'+'.join(str(s) for s in row.get('sources', []))}->{row.get('target', '')}"
        for row in (report.get("synergy", []) or [])
        if isinstance(row, dict) and row.get("passes")
    }


def _dig(blob: Any, path: tuple[str, ...]) -> Any:
    node = blob
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _host(report: dict[str, Any]) -> str:
    """What machine this ran on, as the run itself recorded it."""
    environment = report.get("environment", {}) or {}
    hardware = environment.get("hardware", {}) or {}
    # Falls back to the campaign's platform string, because reports written
    # before the runner recorded an environment carry only that. A run that
    # says nothing about its machine groups under one name rather than
    # silently counting as a separate host.
    parts = [
        str(environment.get("os") or (report.get("campaign", {}) or {}).get("platform", "")),
        str(hardware.get("cpus", "")),
        str(hardware.get("memory_gb", "")),
    ]
    return " / ".join(p for p in parts if p) or "unrecorded"


def _topology(report: dict[str, Any]) -> set[str]:
    return {
        f"{edge['source']}->{edge['target']}"
        for edge in report.get("edges", []) or []
        if isinstance(edge, dict) and edge.get("kept")
    }


def _comparable(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether these are the same experiment at all."""
    campaigns = {
        str((r.get("campaign", {}) or {}).get("fingerprint", "unrecorded")) for r in reports
    }
    trees = {str((r.get("campaign", {}) or {}).get("tree_hash", "unrecorded")) for r in reports}
    schemas = {
        str(((r.get("campaign", {}) or {}).get("frozen", {}) or {}).get("schema", {}).get("hash", ""))
        for r in reports
    }
    return {
        "campaigns": sorted(campaigns),
        "tree_hashes": sorted(trees),
        "schema_hashes": sorted(s for s in schemas if s),
        "same_campaign": len(campaigns) == 1,
        "same_code": len(trees) == 1,
        "same_schema": len({s for s in schemas if s}) <= 1,
    }


def compare(runs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    reports = [report for _name, report in runs]
    by_host: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_host.setdefault(_host(report), []).append(report)

    # Every triple that any run measured, so a synergy comparison is a
    # comparison of the same triples rather than of each run's own best one.
    triples = sorted({name for report in reports for name in _synergy(report)})
    readings: list[tuple[str, Any]] = [
        *COMPARED,
        *((f"synergy[{name}]", ("synergy", name)) for name in triples),
    ]

    measures: dict[str, Any] = {}
    for label, path in readings:
        per_host: dict[str, list[float]] = {}
        for host, group in by_host.items():
            if path[0] == "synergy" and len(path) == 2 and label.startswith("synergy["):
                found = [_synergy(r).get(path[1]) for r in group]
            else:
                found = [_dig(r, path) for r in group]
            values = [float(v) for v in found if isinstance(v, (int, float))]
            if values:
                per_host[host] = values
        if not per_host:
            continue
        within = [
            max(values) - min(values) for values in per_host.values() if len(values) > 1
        ]
        centres = [statistics.fmean(values) for values in per_host.values()]
        between = max(centres) - min(centres) if len(centres) > 1 else None
        worst_within = max(within) if within else None
        measures[label] = {
            "by_host": {h: [round(v, 5) for v in vs] for h, vs in per_host.items()},
            "within_machine_spread": None if worst_within is None else round(worst_within, 5),
            "between_machine_spread": None if between is None else round(between, 5),
            # A gap smaller than what one machine already produces on its own is
            # not a machine effect.
            "machine_effect": (
                None
                if between is None or worst_within is None
                else bool(between > worst_within)
            ),
        }

    topologies = {name: _topology(report) for name, report in runs}
    shared = set.intersection(*topologies.values()) if topologies else set()
    union = set.union(*topologies.values()) if topologies else set()
    verdicts = [
        {row["criterion"]: bool(row["passed"]) for row in (r.get("verdict", {}) or {}).get("criteria", [])}
        for r in reports
    ]
    disagreed = sorted(
        name
        for name in {k for v in verdicts for k in v}
        if len({v.get(name) for v in verdicts if name in v}) > 1
    )

    # Which triples cleared their null, per run. A machine that agrees on the
    # numbers and disagrees on the verdicts has not replicated the criterion.
    passing = [_synergy_passing(report) for report in reports]
    synergy_agreed = sorted(set.intersection(*passing)) if passing else []
    synergy_somewhere = sorted(set.union(*passing)) if passing else []

    moved = [label for label, row in measures.items() if row.get("machine_effect")]
    return {
        "synergy": {
            "triples_compared": triples,
            "passing_everywhere": synergy_agreed,
            "passing_somewhere": synergy_somewhere,
            "passing_only_on_some_runs": sorted(set(synergy_somewhere) - set(synergy_agreed)),
        },
        "runs": [name for name, _ in runs],
        "hosts": sorted(by_host),
        "comparable": _comparable(reports),
        "measures": measures,
        "topology": {
            "kept_everywhere": sorted(shared),
            "kept_somewhere": sorted(union),
            "agreement": round(len(shared) / len(union), 4) if union else None,
            "differing": sorted(union - shared),
        },
        "criteria_that_disagreed": disagreed,
        "measures_with_a_machine_effect": sorted(moved),
        "note": (
            "A gap between machines smaller than the gap one machine already "
            "produces on its own is not a machine effect. If the apparent "
            "integrated subject disappears under small timing differences, that "
            "is the result rather than a nuisance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--allow-different-campaigns", action="store_true",
        help="compare anyway. Two campaigns are two experiments and the "
             "fingerprint exists to stop this, so it has to be asked for",
    )
    args = parser.parse_args()

    runs: list[tuple[str, dict[str, Any]]] = []
    for directory in args.runs:
        path = directory / "subject_core_report.json"
        if not path.is_file():
            print(f"skipping {directory}: no report")
            continue
        runs.append((directory.name, json.loads(path.read_text(encoding="utf-8"))))
    if len(runs) < 2:
        raise SystemExit("need at least two runs with reports")

    out = compare(runs)
    same = out["comparable"]
    print(f"runs:   {', '.join(out['runs'])}")
    print(f"hosts:  {len(out['hosts'])}")
    for host in out["hosts"]:
        print(f"  {host}")
    print(
        f"same campaign: {same['same_campaign']}   same code: {same['same_code']}"
        f"   same schema: {same['same_schema']}"
    )
    if not (same["same_campaign"] and same["same_code"]) and not args.allow_different_campaigns:
        print("\nthese are different experiments; rerun with --allow-different-campaigns to look anyway")
        return 1

    print(f"\n{'measure':24} {'within':>9} {'between':>9}  machine effect")
    for label, row in out["measures"].items():
        within = "-" if row["within_machine_spread"] is None else f"{row['within_machine_spread']:.5f}"
        between = "-" if row["between_machine_spread"] is None else f"{row['between_machine_spread']:.5f}"
        print(f"{label:24} {within:>9} {between:>9}  {row['machine_effect']}")

    topology = out["topology"]
    print(f"\nedges kept everywhere: {len(topology['kept_everywhere'])} of {len(topology['kept_somewhere'])}")
    if topology["differing"]:
        print("  differ: " + ", ".join(topology["differing"][:12]))
    synergy = out["synergy"]
    print(
        f"synergy triples clearing their null everywhere: "
        f"{len(synergy['passing_everywhere'])} of {len(synergy['triples_compared'])}"
    )
    if synergy["passing_only_on_some_runs"]:
        print("  only on some runs: " + ", ".join(synergy["passing_only_on_some_runs"]))
    if out["criteria_that_disagreed"]:
        print("criteria that disagreed: " + ", ".join(out["criteria_that_disagreed"]))
    if out["measures_with_a_machine_effect"]:
        print("machine effect on: " + ", ".join(out["measures_with_a_machine_effect"]))

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
