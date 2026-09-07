#!/usr/bin/env python3
"""Run the Intrinsic Subject Core battery and write down what happened.

One command, one artefact directory, one verdict. The order matters: the
observational recording comes first because every scale the interventions are
measured in comes from it, then the interventions, then the graph they imply,
then the lesion of the cheapest cut and its rescue, then the same measurements
over every null.

Nothing here decides anything. It collects evidence and hands it to
`core.subject.battery`, which holds the thresholds, so the run cannot quietly
grade itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

import numpy as np  # noqa: E402


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def _record(runtime: Any, conditions: Any, rounds: int) -> tuple[list[Any], list[dict[str, float]]]:
    from core.subject.closure import read_periphery

    frames: list[Any] = []
    periphery: list[dict[str, float]] = []
    for _ in range(rounds):
        for condition in conditions:
            for reading in await runtime.turn_once(condition):
                frames.append(reading)
                periphery.append(read_periphery(runtime.kernel))
    return frames, periphery


def _periphery_matrix(rows: list[dict[str, float]]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(sorted({key for row in rows for key in row}))
    if not names:
        return np.zeros((len(rows), 0)), ()
    matrix = np.array(
        [[float(row.get(name, 0.0)) for name in names] for row in rows], dtype=np.float64
    )
    return matrix, names


def _scales(recording: Any) -> dict[str, np.ndarray]:
    """Per-column spread from ordinary operation. Zero means unmeasurable."""
    from core.subject.state import DOMAINS

    spread = recording.x.std(axis=0)
    return {key: spread[recording.slices[key]] for key in DOMAINS}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=24, help="baseline turns per condition")
    parser.add_argument("--trials", type=int, default=6, help="paired interventions per source per condition")
    parser.add_argument("--turns", type=int, default=2, help="turns each intervention arm runs")
    parser.add_argument("--agency-trials", type=int, default=5)
    parser.add_argument("--lesion-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core")
    parser.add_argument("--skip-nulls", action="store_true")
    parser.add_argument("--quick", action="store_true", help="a short run for wiring checks")
    args = parser.parse_args()

    if args.quick:
        args.rounds, args.trials, args.turns = 6, 2, 1
        args.agency_trials, args.lesion_rounds = 2, 4

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.agency import run_agency
    from core.subject.battery import assemble
    from core.subject.causal import build_edges, power_note, run_interventions
    from core.subject.clamp import clamped
    from core.subject.closure import closure_gain
    from core.subject.differentiation import effective_dimension
    from core.subject.driver import CONDITIONS, build_runtime, start_organism
    from core.subject.graph import analyse_graph
    from core.subject.intrinsic import intrinsic_gain
    from core.subject.irreducibility import phi_do
    from core.subject.metastability import regimes
    from core.subject.nulls import (
        ARCHITECTURES,
        architecture,
        replay_surrogate,
        shuffle_surrogate,
        toy_edges,
        toy_recording,
    )
    from core.subject.pci import perturbational_complexity
    from core.subject.recording import build_recording
    from core.subject.state import DOMAINS, FAST_DOMAINS, SLOW_DOMAINS
    from core.subject.synergy import synergy_suite

    started = time.monotonic()
    evidence: dict[str, Any] = {"notes": {}}

    _log(f"building the offline organism in {args.out}")
    runtime = build_runtime(args.out / "runtime", seed=args.seed)
    organism = await start_organism(runtime)
    evidence["organism"] = organism
    _log(f"organism up: {len(organism['up'])} layers, {len(organism['down'])} down")
    if organism["down"]:
        _log(f"  did not come up: {organism['down']}")

    _log(f"recording {args.rounds} rounds over {len(CONDITIONS)} conditions")
    frames, periphery_rows = await _record(runtime, CONDITIONS, args.rounds)
    recording = build_recording(
        frames,
        notes={
            "rounds": args.rounds,
            "conditions": [c.name for c in CONDITIONS],
            "phase_failures": dict(runtime.failures),
        },
    )
    recording.save(args.out)
    scale = _scales(recording)
    _log(
        f"{recording.frames} frames, {len(recording.live_domains())} live domains, "
        f"{len(recording.flat_columns())} flat columns"
    )
    evidence["recording"] = recording.summary()

    _log("observational measures")
    phi = phi_do(recording)
    evidence["phi"] = phi.as_dict()
    evidence["differentiation"] = effective_dimension(recording).as_dict()
    evidence["intrinsic"] = intrinsic_gain(recording, seed=args.seed).as_dict()
    evidence["metastability"] = regimes(recording, seed=args.seed).as_dict()
    evidence["synergy"] = [item.as_dict() for item in synergy_suite(recording, seed=args.seed)]
    matrix, names = _periphery_matrix(periphery_rows)
    evidence["closure"] = closure_gain(recording, matrix, names, seed=args.seed).as_dict()
    _log(
        f"phi_do={evidence['phi']['phi_do']} cut={evidence['phi']['best_cut']} "
        f"D_eff={evidence['differentiation']['d_eff_normalised']} "
        f"intrinsic={evidence['intrinsic']['delta_intrinsic']} "
        f"closed={evidence['closure']['closed']}"
    )

    _log(f"interventions: {len(DOMAINS)} domains x {len(CONDITIONS)} conditions x {args.trials} trials")
    results = await run_interventions(
        runtime,
        CONDITIONS,
        scale=scale,
        trials=args.trials,
        turns=args.turns,
        seed=args.seed,
        on_progress=_log,
    )
    edges, tested = build_edges(results, seed=args.seed)
    evidence["edges"] = tested
    _log(
        "coupling gain: "
        + ", ".join(
            f"{k}:{v['attenuation']}" for k, v in sorted(results.attenuation().items())
        )
    )
    evidence["notes"]["unwritable_domains"] = list(results.unwritable)
    evidence["notes"]["intervention_seconds"] = round(results.seconds, 1)
    kept = [(e.source, e.target) for e in edges]
    graph = analyse_graph(list(DOMAINS), kept)
    evidence["graph"] = graph.as_dict()
    _log(f"{len(kept)} edges kept of {len(tested)} tested; components={graph.components}")

    per_condition: dict[str, Any] = {}
    with_scc = 0
    for condition in CONDITIONS:
        local = [
            (e.source, e.target) for e in edges if condition.name in e.conditions
        ]
        report = analyse_graph(list(DOMAINS), local)
        per_condition[condition.name] = {
            "edges": len(local),
            "one_component": report.one_component,
            "vertex_connectivity": report.connectivity,
        }
        with_scc += int(report.one_component)
    evidence["per_condition"] = {
        "per_condition": per_condition,
        "conditions_with_scc": with_scc,
    }

    _log("perturbational complexity")
    spreads: dict[str, float] = {}
    pcis: dict[str, float] = {}
    null_pcis: dict[str, float] = {}
    null_spreads: dict[str, float] = {}
    rows: dict[str, Any] = {}
    for source in DOMAINS:
        report = perturbational_complexity(results, source=source)
        null = perturbational_complexity(results, source=source, arm="floor")
        spreads[source] = report.spread
        pcis[source] = report.pci
        null_pcis[source] = null.pci
        null_spreads[source] = null.spread
        rows[source] = report.as_dict()
    null_values = list(null_pcis.values())
    null_bar = float(np.quantile(null_values, 0.99)) if null_values else 0.0
    mean_pci = float(np.mean(list(pcis.values()))) if pcis else 0.0
    evidence["perturbation"] = {
        "mean_spread": float(np.mean(list(spreads.values()))) if spreads else 0.0,
        "mean_pci": mean_pci,
        "null_q99": round(null_bar, 4),
        "beats_null": mean_pci > null_bar,
        "mean_null_spread": float(np.mean(list(null_spreads.values()))) if null_spreads else 0.0,
        "spread_by_source": {k: round(v, 3) for k, v in spreads.items()},
        "pci_by_source": {k: round(v, 3) for k, v in pcis.items()},
        "null_pci_by_source": {k: round(v, 3) for k, v in null_pcis.items()},
        "matrices": rows,
    }
    evidence["notes"]["intervention_power"] = power_note(results)
    evidence["notes"]["attenuation"] = results.attenuation()

    consumers = sorted({t for s, t in kept if s == "G"})
    returns = [
        "->".join(cycle) for cycle in graph.cycles if "G" in cycle and len(cycle) >= 3
    ]
    evidence["global_access"] = {
        "consumers": len(consumers),
        "consumer_list": consumers,
        "returns": bool(returns),
        "return_paths": returns[:10],
    }
    fast_to_slow = [f"{s}->{t}" for s, t in kept if s in FAST_DOMAINS and t in SLOW_DOMAINS]
    slow_to_fast = [f"{s}->{t}" for s, t in kept if s in SLOW_DOMAINS and t in FAST_DOMAINS]
    evidence["timescale"] = {
        "fast_to_slow": bool(fast_to_slow),
        "fast_to_slow_edges": fast_to_slow,
        "slow_to_fast": bool(slow_to_fast),
        "slow_to_fast_edges": slow_to_fast,
    }

    _log("agency and ownership")
    act_condition = next(c for c in CONDITIONS if c.after == "act")
    evidence["agency"] = (
        await run_agency(runtime, act_condition, scale=scale, trials=args.agency_trials)
    ).as_dict()

    _log(f"lesion of the cheapest cut {phi.best_cut} and rescue")
    evidence["lesion"] = await _lesion(
        runtime, CONDITIONS, phi, args, build_recording, clamped, phi_do,
        perturbational_complexity, synergy_suite, run_interventions, scale
    )

    if not args.skip_nulls:
        _log("nulls")
        evidence["nulls"] = _nulls(
            recording,
            evidence,
            args,
            ARCHITECTURES,
            architecture,
            toy_recording,
            toy_edges,
            replay_surrogate,
            shuffle_surrogate,
            phi_do,
            analyse_graph,
            list(DOMAINS),
        )
    else:
        evidence["nulls"] = {"phi_beats_all": False, "all_nulls_fail": False, "note": "skipped"}

    verdict = assemble(evidence)
    evidence["verdict"] = verdict.as_dict()
    evidence["notes"]["seconds"] = round(time.monotonic() - started, 1)

    (args.out / "subject_core_report.json").write_text(json.dumps(evidence, indent=2, default=str))
    print()
    print(verdict.table())
    print()
    _log(f"wrote {args.out / 'subject_core_report.json'} in {evidence['notes']['seconds']}s")
    return 0


async def _lesion(
    runtime: Any,
    conditions: Any,
    phi: Any,
    args: Any,
    build_recording: Any,
    clamped: Any,
    phi_do: Any,
    perturbational_complexity: Any,
    synergy_suite: Any,
    run_interventions: Any,
    scale: Any,
) -> dict[str, Any]:
    """Clamp the cheapest side, measure, release, measure again.

    The cut is the one the irreducibility search found, not one chosen here,
    and the rescue arm runs after the clamp is released on the same runtime, so
    a recovery is a recovery of this life rather than of a fresh one.
    """
    smaller = min(phi.best_cut, key=len)

    async def measure(label: str) -> dict[str, Any]:
        frames: list[Any] = []
        for _ in range(args.lesion_rounds):
            for condition in conditions:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames, notes={"arm": label})
        results = await run_interventions(
            runtime,
            conditions[:3],
            scale=scale,
            trials=max(2, args.trials // 3),
            turns=1,
            seed=args.seed + 11,
        )
        spread = float(
            np.mean(
                [perturbational_complexity(results, source=key).spread for key in ("A", "G", "S")]
            )
        )
        synergies = [item.normalised for item in synergy_suite(recording, seed=args.seed)]
        return {
            "phi_do": phi_do(recording).phi,
            "spread": spread,
            "synergy": float(np.mean(synergies)) if synergies else 0.0,
        }

    intact = await measure("intact")
    with clamped(runtime, smaller):
        cut = await measure("cut")
    rescued = await measure("rescued")

    deltas = {key: round(intact[key] - cut[key], 5) for key in intact}
    deficit = all(cut[key] < intact[key] for key in ("phi_do", "spread", "synergy"))
    recovery = {
        key: round(rescued[key] - cut[key], 5) for key in intact
    }
    rescued_ok = all(rescued[key] > cut[key] for key in ("phi_do", "spread", "synergy"))
    return {
        "cut": list(smaller),
        "intact": {k: round(v, 5) for k, v in intact.items()},
        "lesioned": {k: round(v, 5) for k, v in cut.items()},
        "rescued": {k: round(v, 5) for k, v in rescued.items()},
        "deltas": deltas,
        "rescue": recovery,
        "deficit": deficit,
        "rescued_ok": rescued_ok,
    }


def _nulls(
    recording: Any,
    evidence: dict[str, Any],
    args: Any,
    architectures: Any,
    architecture: Any,
    toy_recording: Any,
    toy_edges: Any,
    replay_surrogate: Any,
    shuffle_surrogate: Any,
    phi_do: Any,
    analyse_graph: Any,
    domains: list[str],
) -> dict[str, Any]:
    """Every null, through the measures that are supposed to tell it apart."""
    table: dict[str, Any] = {}
    real_phi = float(evidence["phi"]["phi_do"])

    for name in ("replay", "time_shuffle"):
        maker = replay_surrogate if name == "replay" else shuffle_surrogate
        surrogate = maker(recording, seed=args.seed)
        table[name] = {"phi_do": round(phi_do(surrogate).phi, 5), "kind": "surrogate"}

    for name in architectures:
        system = architecture(name, seed=args.seed)
        toy = toy_recording(system, steps=2500, seed=args.seed)
        edges = toy_edges(system, trials=16, seed=args.seed)
        graph = analyse_graph(domains, edges)
        table[name] = {
            "phi_do": round(phi_do(toy).phi, 5),
            "kind": "architecture",
            "one_component": graph.one_component,
            "vertex_connectivity": graph.connectivity,
            "reentry": graph.every_node_reenters,
        }

    beaten = {
        name: real_phi > float(row["phi_do"])
        for name, row in table.items()
        if name != "recurrent"
    }
    reference = table.get("recurrent", {})
    # The instrument has to be able to say yes to something. A reference
    # architecture that is genuinely recurrent must clear the same bar the
    # nulls fail, or the battery is only capable of returning no.
    reference_passes = float(reference.get("phi_do", 0.0)) > 0.05
    nulls_fail = all(
        float(row["phi_do"]) <= 0.05 for name, row in table.items() if name != "recurrent"
    )
    return {
        "phi_table": {k: v["phi_do"] for k, v in table.items()},
        "detail": table,
        "phi_beats_all": all(beaten.values()) if beaten else False,
        "all_nulls_fail": bool(nulls_fail and reference_passes),
        "nulls_fail_the_bar": nulls_fail,
        "reference_architecture_passes": reference_passes,
        "reference_recurrent_phi": reference.get("phi_do"),
        "summary": {
            "reference_recurrent": reference.get("phi_do"),
            "worst_null": max(
                (float(row["phi_do"]) for name, row in table.items() if name != "recurrent"),
                default=0.0,
            ),
        },
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
