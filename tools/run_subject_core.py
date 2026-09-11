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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

import numpy as np  # noqa: E402

from core.subject.clock import real_time  # noqa: E402


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
    """Per-column spread during ordinary operation, pooled within condition.

    Pooled within rather than measured across, because that is the comparison
    the number is used for. An intervention is compared against a sham in the
    same condition, so the scale it should be read in is how much that column
    varies inside a condition — not how much it differs between an idle turn
    and a turn under load, which is the environment changing and is variance no
    displacement was ever going to produce. Measured across conditions the
    denominator is inflated by exactly the part of the spread the experiment
    holds fixed.
    """
    from core.subject.state import DOMAINS

    conditions = sorted(set(recording.conditions))
    groups = []
    weights = []
    for name in conditions:
        rows = recording.condition_rows(name)
        if rows.size < 8:
            continue
        groups.append(recording.x[rows].var(axis=0))
        weights.append(rows.size - 1)
    if not groups:
        spread = recording.x.std(axis=0)
    else:
        pooled = np.average(np.vstack(groups), axis=0, weights=weights)
        spread = np.sqrt(pooled)
    return {key: spread[recording.slices[key]] for key in DOMAINS}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=120, help="baseline turns per condition")
    parser.add_argument("--trials", type=int, default=6, help="paired interventions per source per condition")
    parser.add_argument("--turns", type=int, default=2, help="turns each intervention arm runs")
    # At least one trial per action kind, so ownership is asked about acting
    # rather than about whichever pathway the state happened to pick.
    parser.add_argument("--agency-trials", type=int, default=8)
    parser.add_argument("--lesion-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--null-draws",
        type=int,
        default=8,
        help="instantiations of each synthetic architecture, so a null is a "
        "distribution rather than one draw of a weight matrix",
    )
    parser.add_argument(
        "--surrogate-draws",
        type=int,
        default=64,
        help=(
            "surrogate recordings of the real run. Each one costs a phi_do over "
            "the same recording rather than a whole simulated system, so the "
            "floor the score is read against can be estimated far more finely "
            "than the architectures can"
        ),
    )
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core")
    parser.add_argument("--skip-nulls", action="store_true")
    parser.add_argument(
        "--here",
        action="store_true",
        help="write into --out itself rather than a fresh run_NNN beneath it",
    )
    parser.add_argument(
        "--skip-lesion",
        action="store_true",
        help="leave the lesion and rescue unmeasured; they read as failures, which is what an unmeasured criterion is",
    )
    parser.add_argument("--quick", action="store_true", help="a short run for wiring checks")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "record the run even when a cognitive loop could not be advanced by a "
            "count or a required phase raised; the run is marked unauthoritative"
        ),
    )
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
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
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

    from core.subject.archive import save_arms, save_edge_table, write_json
    from core.subject.provenance import (
        campaign,
        environment,
        manifest,
        mind_identity,
        next_run_directory,
    )

    started = time.monotonic()
    # A run never overwrites the one before it. The run that did not come out
    # well is the one a reader most needs, and `--out` pointing at a fixed
    # directory quietly destroyed it every time.
    root = args.out
    args.out = next_run_directory(root) if not args.here else root
    evidence: dict[str, Any] = {
        "notes": {},
        "campaign": campaign(
            seed=args.seed,
            rounds=args.rounds,
            trials=args.trials,
            turns=args.turns,
            lesion_rounds=0 if args.skip_lesion else args.lesion_rounds,
        ),
    }
    _log(
        f"run {args.out.name} on {evidence['campaign']['commit'][:12]}"
        f"{' (dirty tree)' if evidence['campaign']['dirty'] else ''}, "
        f"campaign {evidence['campaign']['fingerprint']}"
    )

    _log(f"building the offline organism in {args.out}")
    runtime = build_runtime(args.out / "runtime", seed=args.seed)
    organism = await start_organism(runtime)
    evidence["organism"] = organism
    _log(f"organism up: {len(organism['up'])} layers, {len(organism['down'])} down")
    if organism["down"]:
        _log(f"  did not come up: {organism['down']}")

    # Before anything is recorded: time this machine's frames and put the run
    # on a clock it advances itself. Two arms already saw the same state, the
    # same organs and the same host, and still saw different amounts of time —
    # which is half a standard deviation of deliberation's spread and one and a
    # half of the workspace's, from nothing but how long the machine took.
    evidence["notes"]["clock"] = await calibrate_clock(runtime, CONDITIONS)
    reading = evidence["notes"]["clock"]
    _log(
        f"experiment clock at {reading['step']:.4f}s a frame, "
        f"{reading['frames_per_turn']} frames a turn "
        f"(the machine took {reading['real_seconds_per_frame']:.4f}s a frame)"
    )

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
    # Everything that fits K_{t+1} from K_t runs on one row per turn. A
    # frame-to-frame step inside a turn is one line of the transition function,
    # not a transition: the phases run in a fixed order and most domains do not
    # move between two of them, so the frame series makes the prediction task
    # "the same as last time" and the comparison a comparison of noise.
    turns = recording.by_turn()
    _log(f"{turns.frames} turns from {recording.frames} frames")
    phi = phi_do(turns)
    evidence["phi"] = phi.as_dict()
    evidence["differentiation"] = effective_dimension(recording).as_dict()
    evidence["intrinsic"] = intrinsic_gain(turns, seed=args.seed).as_dict()
    evidence["metastability"] = regimes(turns, seed=args.seed).as_dict()
    evidence["synergy"] = [item.as_dict() for item in synergy_suite(turns, seed=args.seed)]
    matrix, names = _periphery_matrix(periphery_rows)
    turn_rows = recording.turn_rows()
    from core.subject.closure import coverage as periphery_coverage

    evidence["closure"] = closure_gain(
        turns, matrix[turn_rows] if matrix.size else matrix, names, seed=args.seed
    ).as_dict()
    # What the walk could and could not see. A closure result is a claim about
    # everything outside the core, and a walk that stopped at four hundred
    # numbers or two levels down has not seen everything outside the core.
    evidence["closure"]["coverage"] = periphery_coverage()
    _log(
        f"phi_do={evidence['phi']['phi_do']} cut={evidence['phi']['best_cut']} "
        f"D_eff={evidence['differentiation']['d_eff_normalised']} "
        f"intrinsic={evidence['intrinsic']['delta_intrinsic']} "
        f"closed={evidence['closure']['closed']}"
    )

    # The recording is taken with the organism running as it runs. The
    # interventions are not: two arms have to see the same computation, and a
    # loop ticking at whatever rate the machine allows makes them incomparable.
    stopped = await quiesce_organism(runtime)
    evidence["organism"] = runtime.organism.summary() if runtime.organism else evidence["organism"]
    # What the free-running layers did, counted rather than timed. An empty
    # `unsteppable` is the bar the specification asks for: a live cognitive
    # loop that cannot be advanced by a count cannot be inside a paired
    # measurement, so a run that has one says so in the report rather than
    # reporting numbers taken while it was stopped.
    evidence["notes"]["layers"] = runtime.layer_steps.summary()
    # Who bid for attention and who ever got it. A bid type that never wins is
    # a channel into the workspace that cannot fire, and the winner alone
    # cannot show it: every source that lost looks the same as one that never
    # spoke.
    try:
        workspace = runtime.organs.workspace
        snapshot = workspace.get_snapshot() if workspace is not None else {}
        evidence["notes"]["attention"] = {
            "bids_by_source": dict(snapshot.get("bids_by_source", {}) or {}),
            "wins_by_source": dict(snapshot.get("wins_by_source", {}) or {}),
            "sources_that_never_won": list(snapshot.get("sources_that_never_won", []) or []),
            "tie_impasses": snapshot.get("tie_impasses", 0),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _log(f"  the workspace could not say who won: {exc}")
    _log(f"stopped {len(stopped)} background loops for the paired arms")

    # An authoritative run refuses rather than reports.
    #
    # A live cognitive loop that cannot be advanced by a count cannot be inside
    # a paired measurement, and a required phase that raised was not run in
    # either arm — both leave the numbers below describing a different organism
    # from the one that lives here. Saying so in the report is not enough: a
    # reader who takes a scorecard at face value has no reason to look, and an
    # unauthoritative run is exactly the one somebody quotes.
    #
    # `--allow-degraded` exists because a diagnostic run during repair work is
    # a legitimate thing to want, and it is recorded on the run so a reader can
    # tell which kind they are holding.
    blocking = _authority_blockers(
        runtime, evidence, turns=args.rounds * len(CONDITIONS)
    )
    evidence["campaign"]["authoritative"] = not blocking
    evidence["campaign"]["authority_blockers"] = blocking
    if blocking and not args.allow_degraded:
        for line in blocking:
            _log(f"  REFUSED: {line}")
        raise SystemExit(
            "this run is not authoritative: "
            + "; ".join(blocking)
            + " — rerun with --allow-degraded to record it anyway"
        )
    if blocking:
        _log(f"  running degraded on purpose: {'; '.join(blocking)}")
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
    # The arms, not only the conclusion drawn from them. An edge that missed
    # the bar by one condition and an edge that carried in none read the same
    # from the summary, and neither can be rechecked from it.
    save_arms(args.out, results)
    save_edge_table(args.out, tested)
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
    # Whether the two-turn horizon is binding. An effect that peaks at the last
    # frame recorded is an effect the horizon cut off, and that is a fact about
    # the measurement rather than about the organism. The specification asks
    # for the horizon to be preregistered and for the question to be asked.
    at_horizon = [row for row in tested if row.get("at_the_horizon")]
    evidence["notes"]["horizon"] = {
        "turns_per_arm": args.turns,
        "pairs_peaking_at_the_last_frame": len(at_horizon),
        "of_pairs_tested": len(tested),
        "kept_edges_peaking_there": sum(1 for row in at_horizon if row.get("kept")),
        "binding": len(at_horizon) > len(tested) // 4 if tested else False,
    }
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

    if args.skip_lesion:
        _log("lesion skipped")
        evidence["lesion"] = {"deficit": False, "rescued_ok": False, "note": "not measured"}
    else:
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

    if not args.skip_nulls:
        write_json(args.out, "nulls.json", evidence["nulls"])
    write_json(args.out, "lesion.json", evidence["lesion"])
    write_json(args.out, "campaign.json", evidence["campaign"])

    verdict = assemble(evidence)
    evidence["verdict"] = verdict.as_dict()
    evidence["notes"]["seconds"] = round(time.monotonic() - started, 1)

    evidence["campaign"]["finished_at"] = real_time()
    evidence["campaign"]["environment"] = environment()
    evidence["campaign"]["mind"] = mind_identity(
        getattr(getattr(runtime.kernel, "organs", {}).get("llm", None), "instance", None)
        if hasattr(runtime.kernel, "organs")
        else None
    )
    evidence["campaign"]["command"] = " ".join([sys.executable, *sys.argv])
    write_json(args.out, "subject_core_report.json", evidence)
    # Last, because it hashes what the run wrote and the report is one of them.
    write_json(args.out, "manifest.json", manifest(args.out))
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
    """Cut the channels across the cheapest partition, measure, restore, measure.

    The cut is the one the irreducibility search found, not one chosen here,
    and the rescue arm runs on the same runtime afterwards, so a recovery is a
    recovery of this life rather than of a fresh one.

    The equation says to remove `E(A*, B*)` and `E(B*, A*)` and leave each
    side's internal dynamics alone. Holding one side still does not do that: it
    severs every edge out of those domains and destroys their own dynamics
    too, which is a node lesion and reads as a much larger intervention than
    the one written down.

    Each side is run once with the other held at the cut, so it evolves with no
    information crossing and its own pipeline intact, and the two recordings
    are composed column-wise. Both halves start from the same snapshot, and the
    experiment clock makes the two runs the same length of life.
    """
    smaller = min(phi.best_cut, key=len)
    larger = max(phi.best_cut, key=len)
    left, right = phi.best_cut[0], phi.best_cut[1]

    #: The sources perturbational spread is read from in a lesion arm. Three
    #: rather than ten, because each arm pays for its own intervention sweep
    #: and the lesion is run five times now.
    watched = ("A", "G", "S")

    async def _live() -> tuple[list[Any], Any]:
        """One arm's life, and the interventions run inside whatever holds it."""
        frames: list[Any] = []
        for _ in range(args.lesion_rounds):
            for condition in conditions:
                frames.extend(await runtime.turn_once(condition))
        results = await run_interventions(
            runtime,
            conditions[:3],
            scale=scale,
            sources=watched,
            trials=max(2, args.trials // 3),
            turns=1,
            seed=args.seed + 11,
        )
        return frames, results

    def _read(label: str, frames: list[Any], spread: float) -> dict[str, Any]:
        recording = build_recording(frames, notes={"arm": label}).by_turn()
        synergies = [item.normalised for item in synergy_suite(recording, seed=args.seed)]
        # Scored at the partition the lesion cuts, in every arm.
        #
        # Each arm searched for its own cheapest cut, so the intact score and
        # the cut score were two minima taken over different partitions and the
        # difference between them was not a comparison of anything. Measured
        # that way, spread and synergy both fell when the system was cut and
        # both returned when it was restored, while irreducibility moved the
        # wrong way twice — which is what two independent minima over five
        # hundred and eleven noisy estimates will do. Severing a partition is
        # evaluated at that partition.
        return {
            "phi_do": phi_do(recording, at=(tuple(smaller), tuple(larger))).phi,
            "spread": spread,
            "synergy": float(np.mean(synergies)) if synergies else 0.0,
        }

    def _spread(results: Any, only: Sequence[str] = watched) -> float:
        return float(
            np.mean([perturbational_complexity(results, source=key).spread for key in only])
        )

    async def measure(label: str) -> dict[str, Any]:
        frames, results = await _live()
        return _read(label, frames, _spread(results))

    from core.subject.clamp import compose

    intact = await measure("intact")

    # Each side once, with the other held at the cut. Both start from the same
    # place, so the two recordings are the same life with the crossing removed
    # — and each side's spread is read from the arm in which that side was
    # free, because a displacement of a held domain is a displacement of
    # nothing.
    start = runtime.snapshot()
    with clamped(runtime, right):
        left_frames, left_results = await _live()
    runtime.restore(start)
    with clamped(runtime, left):
        right_frames, right_results = await _live()
    cut_spread = []
    for key in watched:
        source = left_results if key in set(left) else right_results
        cut_spread.append(perturbational_complexity(source, source=key).spread)
    cut = _read("cut", compose(left_frames, right_frames, left), float(np.mean(cut_spread)))

    # And the node clamp beside it, kept because it is a useful ablation and
    # reported as what it is rather than as the partition lesion.
    runtime.restore(start)
    with clamped(runtime, smaller):
        clamped_side = await measure("clamped_side")

    rescued = await measure("rescued")

    deltas = {key: round(intact[key] - cut[key], 5) for key in intact}
    deficit = all(cut[key] < intact[key] for key in ("phi_do", "spread", "synergy"))
    recovery = {
        key: round(rescued[key] - cut[key], 5) for key in intact
    }
    rescued_ok = all(rescued[key] > cut[key] for key in ("phi_do", "spread", "synergy"))
    return {
        "cut": list(smaller),
        "severed": {"left": list(left), "right": list(right)},
        "node_clamp": {k: round(v, 5) for k, v in clamped_side.items()},
        "intact": {k: round(v, 5) for k, v in intact.items()},
        "lesioned": {k: round(v, 5) for k, v in cut.items()},
        "rescued": {k: round(v, 5) for k, v in rescued.items()},
        "deltas": deltas,
        "rescue": recovery,
        "deficit": deficit,
        "rescued_ok": rescued_ok,
    }


#: Phases whose failure makes a run unauthoritative. Not every phase: the
#: response phase needs a cortex this run deliberately does not have, and a
#: phase that is absent by design is a different fact from one that raised.
#: These are the ones the domains are read from.
REQUIRED_PHASES: tuple[str, ...] = (
    "ProprioceptiveLoop",
    "SensoryIngestionPhase",
    "MemoryRetrievalPhase",
    "AffectUpdatePhase",
    "MotivationUpdatePhase",
    "CognitiveIntegrationPhase",
    "ExecutiveClosurePhase",
    "ConsciousnessPhase",
)


def _authority_blockers(
    runtime: Any, evidence: dict[str, Any], *, turns: int = 0
) -> list[str]:
    """Everything that makes the numbers below describe a different organism.

    A required phase is judged on the share of turns it failed, not on whether
    it ever did. A phase that raised once in four hundred and eighty turns lost
    one turn's worth of that domain; a phase that raised on a fifth of them was
    not running. The share is the one the battery already uses to decide that a
    reader was absent rather than unlucky, so there is no second number.
    """
    blocking: list[str] = []

    layers = evidence.get("notes", {}).get("layers", {}) or {}
    unsteppable = layers.get("unsteppable") or layers.get("missing") or {}
    if unsteppable:
        blocking.append(
            f"a live cognitive loop cannot be advanced by a count: {sorted(unsteppable)}"
        )
    failed_layers = layers.get("failures") or {}
    if failed_layers:
        blocking.append(f"a layer raised while being stepped: {sorted(failed_layers)}")

    down = (evidence.get("organism", {}) or {}).get("down") or {}
    if down:
        blocking.append(f"a declared layer did not come up: {sorted(down)}")

    # A source that could not be read for most of the run is a subsystem that
    # was not there, and every column declaring it was a default. One
    # criterion at a time is already invalidated by the battery; a reader that
    # failed across the run is a fact about the whole of it.
    from core.subject.battery import MISSING_SHARE

    misses = (evidence.get("recording", {}) or {}).get("misses") or {}
    persistent = sorted(
        source
        for source, row in misses.items()
        if isinstance(row, dict) and float(row.get("share", 0.0)) >= MISSING_SHARE
    )
    if persistent:
        blocking.append(f"a reader failed for most of the run: {persistent}")

    failures = dict(getattr(runtime, "failures", {}) or {})
    notes = dict(getattr(runtime, "failure_notes", {}) or {})
    bound = max(1, int(turns)) * MISSING_SHARE if turns else 0.0
    hurt = {
        name: count
        for name, count in failures.items()
        if name in REQUIRED_PHASES and float(count) > bound
    }
    if hurt:
        blocking.append(
            "a required phase raised on more than "
            f"{MISSING_SHARE:.0%} of {turns or 'the'} turns: "
            + ", ".join(
                f"{name} x{count} ({notes.get(name, '')})"
                for name, count in sorted(hurt.items())
            )
        )
    evidence.setdefault("notes", {})["phase_failures"] = failures
    evidence["notes"]["phase_failure_notes"] = notes
    return blocking


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
    from core.subject.battery import THRESHOLDS

    table: dict[str, Any] = {}
    real_phi = float(evidence["phi"]["phi_do"])

    # Surrogates of the series the score is actually computed on. Building them
    # from the frame-level recording would compare a number measured on one
    # sampling against a floor measured on another.
    turns = recording.by_turn()
    for name in ("replay", "time_shuffle"):
        maker = replay_surrogate if name == "replay" else shuffle_surrogate
        draws = [
            round(phi_do(maker(turns, seed=args.seed + draw)).phi, 5)
            for draw in range(args.surrogate_draws)
        ]
        # The bar the real score has to clear is the top of the surrogate's
        # distribution, not its middle — but a quantile read off a handful of
        # draws is an unstable bar, so the distribution goes in the report and
        # the comparison is against a stated quantile of it.
        #
        # A surrogate costs one phi_do over the recording the score is computed
        # on, where an architecture costs a whole simulated system, so the
        # floor is estimated an order of magnitude more finely than the
        # architectures are — which is the right place to spend it, because the
        # floor is what the margin is measured against.
        table[name] = {
            "phi_do": round(float(np.quantile(draws, 0.95)), 5),
            "draws": draws,
            "max": max(draws),
            "median": round(float(np.median(draws)), 5),
            "q99": round(float(np.quantile(draws, 0.99)), 5),
            "spread": round(float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0, 5),
            "quantile": 0.95,
            "kind": "surrogate",
        }

    for name in architectures:
        # Several instantiations of each, so a null is a distribution rather
        # than one draw of a random weight matrix. A single instantiation can
        # be lucky in either direction and the comparison is with its tail.
        values: list[float] = []
        graphs: list[Any] = []
        for draw in range(args.null_draws):
            system = architecture(name, seed=args.seed + draw)
            toy = toy_recording(system, steps=2500, seed=args.seed + draw)
            values.append(round(phi_do(toy).phi, 5))
            if draw == 0:
                edges = toy_edges(system, trials=16, seed=args.seed)
                graphs.append(analyse_graph(domains, edges))
        graph = graphs[0]
        # And whether the null's own core is closed. A broker outside K makes
        # every domain depend on every other one, so the graph of a hidden
        # broker is indistinguishable from a mind's — measured, not assumed:
        # one component, vertex connectivity three, every node re-entering.
        # What separates them is that K's future depends on a variable no
        # reading of K contains, which is what this measures and what the
        # battery keeps causal closure for.
        closed = True
        leak = 0.0
        try:
            from core.subject.closure import closure_gain
            from core.subject.nulls import toy_periphery

            system = architecture(name, seed=args.seed)
            recording_for_closure = toy_recording(system, steps=2500, seed=args.seed)
            outside = toy_periphery(system, steps=2500, seed=args.seed)
            report = closure_gain(
                recording_for_closure,
                outside,
                tuple(f"broker.{index}" for index in range(outside.shape[1])),
                seed=args.seed,
            )
            closed = bool(report.closed)
            leak = float(report.leak)
        except (ImportError, ValueError, RuntimeError, AttributeError) as exc:
            _log(f"  closure unavailable for the {name} null: {exc}")
        table[name] = {
            "phi_do": round(float(np.quantile(values, 0.95)), 5),
            "draws": values,
            "max": max(values),
            "median": round(float(np.median(values)), 5),
            "quantile": 0.95,
            "kind": "architecture",
            "one_component": graph.one_component,
            "vertex_connectivity": graph.connectivity,
            "reentry": graph.every_node_reenters,
            "closed": closed,
            "leak": round(leak, 5),
        }

    beaten = {
        name: real_phi > float(row["phi_do"])
        for name, row in table.items()
        if name != "recurrent"
    }
    reference = table.get("recurrent", {})

    def _passes_the_conjunction(row: dict[str, Any]) -> bool:
        """The criterion's own words: does this system pass the battery?

        `beats_every_null` says "no null passes the conjunction" and the test
        behind it asked one question — whether irreducibility cleared its bar.
        That is not the conjunction, and it gave the wrong answer: the hub
        null, a broker that carries its own state across steps, scores 0.070
        against a bar of 0.05 and is counted as passing, though it fails the
        graph on vertex connectivity exactly as the null was designed to. A
        stateful broker is genuinely hard to partition; the battery separates
        it from a mind on the shape of its graph, not on phi, and the criterion
        has to ask the same question the battery asks.
        """
        if row.get("kind") != "architecture":
            return float(row.get("phi_do", 0.0)) > THRESHOLDS["phi_do"]
        return (
            float(row.get("phi_do", 0.0)) > THRESHOLDS["phi_do"]
            and bool(row.get("one_component"))
            and float(row.get("vertex_connectivity", 0.0)) >= THRESHOLDS["vertex_connectivity"]
            and bool(row.get("reentry"))
            # A core whose future depends on a variable no reading of it
            # contains is not the core. The hidden-broker null passes every
            # graph measure the reference passes and fails here, which is what
            # causal closure is in the conjunction for.
            and bool(row.get("closed", True))
        )

    # The instrument has to be able to say yes to something. A reference
    # architecture that is genuinely recurrent must pass everything the nulls
    # fail — not one line of it — or the battery is only capable of returning
    # no and nobody can tell a hard organism from a blunt instrument.
    reference_passes = _passes_the_conjunction(reference)
    nulls_fail = not any(
        _passes_the_conjunction(row) for name, row in table.items() if name != "recurrent"
    )
    # The floor the real score has to clear. A minimum over five hundred and
    # eleven noisy estimates is biased downward by the width of its own search,
    # and the matched surrogates are the only thing that measures how far: same
    # dimensionality, same cuts, same estimator, coupling removed.
    floor = max(
        (float(row["phi_do"]) for name, row in table.items() if row.get("kind") == "surrogate"),
        default=None,
    )
    return {
        "phi_table": {k: v["phi_do"] for k, v in table.items()},
        "detail": table,
        "surrogate_floor": floor,
        "phi_above_floor": None if floor is None else round(real_phi - floor, 5),
        # And the floor's own uncertainty, so a margin can be read as a margin.
        # A score a hundredth above a floor estimated to within two hundredths
        # has not cleared it, and reporting only the difference hides that.
        "surrogate_floor_detail": {
            name: {
                "draws": len(row["draws"]),
                "q95": row["phi_do"],
                "q99": row.get("q99"),
                "median": row["median"],
                "spread": row.get("spread"),
            }
            for name, row in table.items()
            if row.get("kind") == "surrogate"
        },
        "phi_beats_all": all(beaten.values()) if beaten else False,
        "all_nulls_fail": bool(nulls_fail and reference_passes),
        "nulls_fail_the_bar": nulls_fail,
        "reference_architecture_passes": reference_passes,
        "reference_recurrent_phi": reference.get("phi_do"),
        "conjunction": {
            name: _passes_the_conjunction(row) for name, row in table.items()
        },
        "summary": {
            "reference_recurrent": reference.get("phi_do"),
            "reference_passes_the_conjunction": reference_passes,
            "nulls_that_pass": [
                name
                for name, row in table.items()
                if name != "recurrent" and _passes_the_conjunction(row)
            ],
            "worst_null": max(
                (float(row["phi_do"]) for name, row in table.items() if name != "recurrent"),
                default=0.0,
            ),
        },
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
