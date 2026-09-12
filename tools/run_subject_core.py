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
import math
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
    # At least two trials per action kind, so ownership is asked about acting
    # rather than about whichever pathway the state happened to pick, and so
    # each kind is asked twice rather than once.
    parser.add_argument("--agency-trials", type=int, default=16)
    parser.add_argument(
        "--agency-seeds",
        type=int,
        default=2,
        help="times the whole ownership experiment is repeated on a fresh "
        "generator. One seed cannot say whether the divergence survives one",
    )
    parser.add_argument("--lesion-rounds", type=int, default=30)
    parser.add_argument(
        "--lesion-cycles",
        type=int,
        default=3,
        help="times the cut is made and released, sharing --lesion-rounds "
        "between them. One cycle decides both criteria on a single reading "
        "of each arm, with no scale to read the difference against",
    )
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
    parser.add_argument(
        "--sensory-tape",
        type=Path,
        help="a tape cut by tools/record_sensory_tape.py. Perception then comes "
        "from the channels that recorded it instead of from the conditions' "
        "scripted percepts, and the report says which channels carried it",
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
        args.agency_trials, args.lesion_rounds = 8, 4
        args.lesion_cycles = 2
        args.agency_seeds = 1

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    # The run's own directory, and its state root inside it, before any core
    # module is imported. A run never overwrites the one before it: the run
    # that did not come out well is the one a reader most needs, and `--out`
    # pointing at a fixed directory quietly destroyed it every time. And a run
    # never shares state with another. Every tool that built the organism left
    # the state root unset, so each campaign started from whatever the runs
    # before it had trained, and two runs with one fingerprint were two
    # experiments.
    from core.subject.isolation import isolate_state
    from core.subject.provenance import next_run_directory

    root = args.out
    args.out = next_run_directory(root) if not args.here else root
    state_root_path = isolate_state(args.out)

    from core.subject.agency import run_agency
    from core.subject.archive import save_arms, save_edge_table, write_json
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
    from core.subject.provenance import (
        campaign,
        environment,
        manifest,
        mind_identity,
    )
    from core.subject.recording import build_recording
    from core.subject.state import DOMAINS, FAST_DOMAINS, SLOW_DOMAINS
    from core.subject.synergy import synergy_suite

    started = time.monotonic()
    evidence: dict[str, Any] = {
        "notes": {},
        "campaign": campaign(
            seed=args.seed,
            rounds=args.rounds,
            trials=args.trials,
            turns=args.turns,
            lesion_rounds=0 if args.skip_lesion else args.lesion_rounds,
            lesion_cycles=0 if args.skip_lesion else args.lesion_cycles,
        ),
        # What it ran on, beyond the commit. `environment()` was imported and
        # never called, so every report carried nothing about its machine — and
        # a result cannot be compared across machines when a run does not say
        # which one it was. None of this enters the fingerprint: two machines
        # running one campaign are one campaign, and whether their numbers
        # agree is the question a second machine is run to answer.
        "environment": environment(),
    }
    _log(
        f"run {args.out.name} on {evidence['campaign']['commit'][:12]}"
        f"{' (dirty tree)' if evidence['campaign']['dirty'] else ''}, "
        f"campaign {evidence['campaign']['fingerprint']}"
    )

    _log(f"building the offline organism in {args.out}")
    runtime = build_runtime(args.out / "runtime", seed=args.seed)
    # Perception, before the organism lives a frame. A tape has to be on the
    # runtime for the calibration turns too, or the clock is calibrated against
    # a different world from the one the run records.
    evidence["perception"] = _attach_tape(runtime, args)
    organism = await start_organism(runtime)
    evidence["organism"] = organism
    # Whether anything kept a path into the root the process would otherwise
    # have used. Fifty-eight modules build such a path when they are imported,
    # and the organism imports most of them on the way up.
    from core.subject.isolation import state_leaks

    evidence["notes"]["state"] = {
        "root": str(state_root_path),
        "policy": "per_run",
        "leaks": state_leaks(),
    }
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
    # And which broadcast consumers ever did anything. A consumer that returns
    # early every time is a registered processor with no effect, and the list
    # of what is wired looks the same either way.
    try:
        from core.consciousness.broadcast_consumers import consumer_activity

        evidence["notes"]["broadcast_consumers"] = consumer_activity()
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        _log(f"  the broadcast consumers could not say what they did: {exc}")
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
    # And the instrument's own noise, broken out every way it can be read. An
    # effect bar a floor approaches is a bar measuring restoration rather than
    # coupling, and one pooled number cannot say which domain that happened in.
    evidence["sham_floor"] = results.floor_report()

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
    # More than one seed. A divergence that only survives the generator it was
    # found on is a property of that draw, and the report has to be able to say
    # which of the two it is.
    seeds = max(1, int(args.agency_seeds))
    runs: list[dict[str, Any]] = []
    for index in range(seeds):
        if index:
            runtime.rng.seed(args.seed + 1000 * index)
        runs.append(
            (
                await run_agency(
                    runtime, act_condition, scale=scale, trials=args.agency_trials
                )
            ).as_dict()
        )
        _log(
            f"  seed {index + 1}/{seeds}: ownership {runs[-1]['ownership_divergence']} "
            f"over floor {runs[-1]['ownership_floor']}, "
            f"generalises={runs[-1]['ownership_generalises']}"
        )
    runtime.rng.seed(args.seed)
    evidence["agency"] = _agency_across_seeds(runs)

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


def _agency_across_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """One report from several, with what each seed said kept beside it.

    The headline numbers are the means, which is what a criterion reads. What
    is added is whether every seed agreed: a divergence that cleared its floor
    on one generator and not the next is a property of that draw, and a mean
    hides which of the two it was.
    """
    if len(runs) == 1:
        out = dict(runs[0])
        out["seeds"] = 1
        out["agreed_across_seeds"] = True
        return out

    numbers = [
        "self_to_action", "self_to_action_floor",
        "ownership_divergence", "ownership_floor",
    ]
    out = dict(runs[-1])
    for key in numbers:
        out[key] = round(float(np.mean([float(r.get(key, 0.0)) for r in runs])), 5)
    out["self_drives_action"] = all(bool(r.get("self_drives_action")) for r in runs)
    out["outcome_updates_self"] = all(bool(r.get("outcome_updates_self")) for r in runs)
    out["ownership_generalises"] = all(bool(r.get("ownership_generalises")) for r in runs)
    out["worlds_identical"] = all(bool(r.get("worlds_identical")) for r in runs)
    out["seeds"] = len(runs)
    out["per_seed"] = [
        {
            "ownership_divergence": r.get("ownership_divergence"),
            "ownership_floor": r.get("ownership_floor"),
            "ownership_generalises": r.get("ownership_generalises"),
            "kinds_that_cleared": r.get("kinds_that_cleared", []),
            "outcomes_that_cleared": r.get("outcomes_that_cleared", []),
            "worlds_identical": r.get("worlds_identical"),
        }
        for r in runs
    ]
    # Which kinds and which outcome shapes cleared on every seed. A claim that
    # ownership generalises is a claim about the ones that held throughout.
    out["kinds_that_cleared"] = sorted(
        set.intersection(*(set(r.get("kinds_that_cleared", [])) for r in runs))
    )
    out["outcomes_that_cleared"] = sorted(
        set.intersection(*(set(r.get("outcomes_that_cleared", [])) for r in runs))
    )
    out["agreed_across_seeds"] = (
        len({bool(r.get("ownership_generalises")) for r in runs}) == 1
        and len({bool(r.get("outcome_updates_self")) for r in runs}) == 1
    )
    return out


def _attach_tape(runtime: Any, args: Any) -> dict[str, Any]:
    """Put a recorded sensory stream on the runtime, or say it is scripted.

    A run without a tape is not a run with an empty one. The conditions write
    percepts of their own, and reporting that as real perception with no
    channels on it would be the scripted stream under another name. So the
    absence says so, in the same field a tape's coverage would occupy.
    """
    if not getattr(args, "sensory_tape", None):
        return {
            "source": "scripted",
            "carried": [],
            "note": "the conditions wrote the percepts; no sensory channel was read",
        }
    from core.subject.perception_replay import SensoryTape

    tape = SensoryTape.load(args.sensory_tape)
    runtime.tape = tape
    coverage = tape.coverage()
    coverage["path"] = str(args.sensory_tape)
    _log(
        f"perception from {args.sensory_tape.name}: {coverage['frames']} frames, "
        f"{coverage['percepts']} percepts, carried by {coverage['carried'] or 'nothing'}"
    )
    return coverage


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

    The whole thing happens more than once. One cycle gives one reading of each
    arm, and three single readings decided both criteria: whether the cut cost
    anything, and whether putting it back brought anything home. A difference
    between two single readings has no scale to be read against. So the budget
    that used to buy one long cycle buys several shorter ones, which costs the
    same life and returns a spread — and with a spread the run can say whether
    it could have detected the effect it is claiming.

    Each cycle also measures a plain life twice before cutting anything. Two
    consecutive uncut arms differ by whatever ordinary drift there is over that
    stretch of life, and a deficit smaller than that is drift.
    """
    from core.subject.battery import DEFICIT_SHARE, RECOVERY_TOLERANCE

    smaller = min(phi.best_cut, key=len)
    larger = max(phi.best_cut, key=len)
    left, right = phi.best_cut[0], phi.best_cut[1]

    #: The sources perturbational spread is read from in a lesion arm, taken
    #: from both sides of the cut. The fixed three this used, A, G and S, all
    #: sat on the large side of run_023's P|C cut, so removing the crossing
    #: could not shrink their reach and spread read 0.444 in the intact, cut
    #: and rescued arms alike: a measure that could not move, scored as one
    #: that did not.
    watched = _watched_across_the_cut(left, right)

    cycles = max(1, int(getattr(args, "lesion_cycles", 1)))
    # The same life, divided. A cycle shorter than two rounds measures nothing,
    # so the count comes down rather than the rounds going to one.
    rounds = max(2, args.lesion_rounds // cycles)
    cycles = max(1, min(cycles, max(1, args.lesion_rounds // 2)))

    async def _live(workload: Sequence[Any], seed: int) -> tuple[list[Any], Any]:
        """One arm's life, and the interventions run inside whatever holds it."""
        frames: list[Any] = []
        for _ in range(rounds):
            for condition in workload:
                frames.extend(await runtime.turn_once(condition))
        results = await run_interventions(
            runtime,
            workload[:3],
            scale=scale,
            sources=watched,
            trials=max(2, args.trials // 3),
            turns=1,
            seed=seed,
        )
        return frames, results

    def _read(label: str, frames: list[Any], spread: float) -> dict[str, float]:
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

    async def measure(label: str, workload: Sequence[Any], seed: int) -> dict[str, float]:
        frames, results = await _live(workload, seed)
        return _read(label, frames, _spread(results))

    from core.subject.clamp import compose

    measures = ("phi_do", "spread", "synergy")
    every: list[dict[str, Any]] = []

    for index in range(cycles):
        # Each cycle drives the life from a different rotation of the workload,
        # so across the run the lesion is measured under more than the first
        # three conditions. What is read is the same in every cycle — spread at
        # the same sources, irreducibility at the same partition — so the
        # spread across cycles carries workload variation, which is variation
        # the criterion should have to survive.
        offset = index % max(1, len(conditions))
        workload = list(conditions[offset:]) + list(conditions[:offset])
        seed = args.seed + 11 + 101 * index

        # Two plain lives, back to back, with nothing cut between them. What
        # they differ by is ordinary drift over a stretch this long, and it is
        # the floor the deficit has to clear.
        intact = await measure("intact", workload, seed)
        drift = await measure("drift", workload, seed + 1)

        start = runtime.snapshot()
        # Each side once, with the other held at the cut. Both start from the
        # same place, so the two recordings are the same life with the crossing
        # removed — and each side's spread is read from the arm in which that
        # side was free, because a displacement of a held domain is a
        # displacement of nothing.
        #
        # Which side runs first alternates between cycles. Running one side
        # always first puts the whole of the other side's life later in the
        # cycle, and anything that changes with time is then confounded with
        # which half of the partition it belongs to.
        first_is_left = index % 2 == 0
        held_first, held_second = (right, left) if first_is_left else (left, right)
        with clamped(runtime, held_first):
            frames_first, results_first = await _live(workload, seed + 2)
        runtime.restore(start)
        with clamped(runtime, held_second):
            frames_second, results_second = await _live(workload, seed + 3)
        left_frames, left_results = (
            (frames_first, results_first) if first_is_left else (frames_second, results_second)
        )
        right_frames, right_results = (
            (frames_second, results_second) if first_is_left else (frames_first, results_first)
        )
        cut_spread = []
        for key in watched:
            source = left_results if key in set(left) else right_results
            cut_spread.append(perturbational_complexity(source, source=key).spread)
        cut = _read("cut", compose(left_frames, right_frames, left), float(np.mean(cut_spread)))

        # And the node clamp beside it, kept because it is a useful ablation and
        # reported as what it is rather than as the partition lesion.
        runtime.restore(start)
        with clamped(runtime, smaller):
            clamped_side = await measure("clamped_side", workload, seed + 4)

        # The rescue continues from the individual that was lesioned. Restoring
        # a snapshot here would measure a fresh baseline and call it a recovery.
        rescued = await measure("rescued", workload, seed + 5)

        baseline = {key: (intact[key] + drift[key]) / 2.0 for key in measures}
        every.append({
            "cycle": index,
            "left_ran_first": first_is_left,
            "conditions": [c.name for c in workload],
            "intact": intact,
            "drift_arm": drift,
            "baseline": baseline,
            "lesioned": cut,
            "rescued": rescued,
            "node_clamp": clamped_side,
            "deltas": {key: baseline[key] - cut[key] for key in measures},
            "drift": {key: abs(intact[key] - drift[key]) for key in measures},
            "recovery": {key: rescued[key] - cut[key] for key in measures},
        })

    def _mean(field: str, key: str) -> float:
        return float(np.mean([cycle[field][key] for cycle in every]))

    def _spread_of(field: str, key: str) -> float:
        values = [cycle[field][key] for cycle in every]
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    intact_mean = {key: _mean("baseline", key) for key in measures}
    cut_mean = {key: _mean("lesioned", key) for key in measures}
    rescued_mean = {key: _mean("rescued", key) for key in measures}
    deltas = {key: round(intact_mean[key] - cut_mean[key], 5) for key in measures}
    recovery = {key: round(rescued_mean[key] - cut_mean[key], 5) for key in measures}
    drift_mean = {key: _mean("drift", key) for key in measures}

    # A measure whose deficit was a rounding error has nothing to rescue, and
    # judging the rescue on it is judging noise. run_019's synergy fell by
    # 0.0166 from 0.245 — under seven per cent — and its recovery came out at
    # minus a quarter of that, which failed the whole criterion on a channel
    # the lesion barely touched. So a measure enters the rescue verdict only
    # when its own deficit was worth rescuing, and the share it has to lose to
    # count is fixed before the run rather than read off the result.
    real = {
        key: abs(deltas[key]) >= DEFICIT_SHARE * max(abs(intact_mean[key]), 1e-9)
        for key in measures
    }
    # And larger than the drift between two uncut arms. A deficit inside that
    # band is what a stretch of life does on its own.
    over_drift = {key: abs(deltas[key]) > drift_mean[key] for key in measures}
    fractions = {
        key: round(recovery[key] / deltas[key], 4) if abs(deltas[key]) > 1e-9 else None
        for key in measures
    }
    # And a trivial improvement is not a rescue. Half the deficit has to come
    # back, which is a tolerance set before the experiment and not "rescued is
    # larger than cut", a comparison two noisy readings pass half the time.
    judged = [key for key in measures if real[key] and over_drift[key]]
    deficit = bool(judged) and all(deltas[key] > 0 for key in judged)
    rescued_ok = bool(judged) and all(
        (fractions[key] or 0.0) >= RECOVERY_TOLERANCE for key in judged
    )
    # How often each measure fell when cut and came back when released. A
    # criterion carried by one cycle out of three is a different fact from one
    # that held every time, and the verdict above cannot show the difference.
    held = {
        key: {
            "fell_when_cut": sum(1 for c in every if c["deltas"][key] > 0),
            "returned_when_released": sum(1 for c in every if c["recovery"][key] > 0),
            "of": len(every),
        }
        for key in measures
    }
    return {
        "cut": list(smaller),
        "severed": {"left": list(left), "right": list(right)},
        "cycles": len(every),
        "rounds_per_cycle": rounds,
        "per_cycle": [
            {
                "cycle": c["cycle"],
                "left_ran_first": c["left_ran_first"],
                "conditions": c["conditions"],
                **{
                    field: {k: round(v, 5) for k, v in c[field].items()}
                    for field in ("intact", "drift_arm", "lesioned", "rescued", "node_clamp",
                                  "deltas", "drift", "recovery")
                },
            }
            for c in every
        ],
        "node_clamp": {k: round(_mean("node_clamp", k), 5) for k in measures},
        "intact": {k: round(v, 5) for k, v in intact_mean.items()},
        "lesioned": {k: round(v, 5) for k, v in cut_mean.items()},
        "rescued": {k: round(v, 5) for k, v in rescued_mean.items()},
        "deltas": deltas,
        "rescue": recovery,
        "drift_between_uncut_arms": {k: round(v, 5) for k, v in drift_mean.items()},
        "spread_across_cycles": {
            field: {k: round(_spread_of(field, k), 5) for k in measures}
            for field in ("baseline", "lesioned", "rescued")
        },
        "power": _lesion_power(every, measures),
        "held_across_cycles": held,
        "recovery_fraction": fractions,
        "deficit_worth_rescuing": real,
        "deficit_over_drift": over_drift,
        "judged_on": judged,
        "recovery_tolerance": RECOVERY_TOLERANCE,
        "deficit_share": DEFICIT_SHARE,
        "deficit": deficit,
        "rescued_ok": rescued_ok,
    }


def _watched_across_the_cut(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    """Sources for a lesion arm's spread, alternating between the two sides.

    The smaller side goes first, because it is the part the cut isolates and a
    source there is the one whose reach the cut should shrink most. Within a
    side the order is the domain order, so two runs with the same cut read the
    same sources.
    """
    from core.subject.battery import LESION_SOURCES
    from core.subject.state import DOMAINS

    small, large = (left, right) if len(left) <= len(right) else (right, left)
    queues = (
        [key for key in DOMAINS if key in set(small)],
        [key for key in DOMAINS if key in set(large)],
    )
    picked: list[str] = []
    while len(picked) < LESION_SOURCES and any(queues):
        for queue in queues:
            if queue and len(picked) < LESION_SOURCES:
                picked.append(queue.pop(0))
    return tuple(picked)


def _lesion_power(every: list[dict[str, Any]], measures: Sequence[str]) -> dict[str, Any]:
    """Whether the experiment could have found what it is reporting.

    A lesion that comes back "no deficit" from an experiment too small to find
    one has reported the size of its own budget. With a reading per cycle there
    is a standard error, and from that a smallest deficit this many cycles
    could have separated from zero — and how many cycles the observed deficit
    would need.

    One cycle gives no spread and therefore no power statement. It says so
    rather than returning a number, because an unmeasurable power reported as
    zero reads as a refusal the experiment never made.
    """
    count = len(every)
    if count < 2:
        return {
            "measured": False,
            "why": "one cycle gives no spread, so nothing here can be estimated",
            "cycles": count,
        }
    # Two-sided, alpha 0.05, and a conventional 80 per cent. Both fixed here
    # rather than chosen after seeing which measure missed.
    z_alpha, z_power = 1.959964, 0.8416212
    out: dict[str, Any] = {"measured": True, "cycles": count, "alpha": 0.05, "power": 0.80}
    for field, against in (("deltas", "lesion"), ("recovery", "rescue")):
        rows: dict[str, Any] = {}
        for key in measures:
            values = [float(c[field][key]) for c in every]
            observed = float(np.mean(values))
            sigma = float(np.std(values, ddof=1))
            if sigma <= 0.0:
                rows[key] = {
                    "observed": round(observed, 6),
                    "sigma": 0.0,
                    "detectable": 0.0,
                    "cycles_needed": 1,
                    "powered": True,
                }
                continue
            detectable = (z_alpha + z_power) * sigma / math.sqrt(count)
            needed = math.ceil(((z_alpha + z_power) * sigma / max(abs(observed), 1e-12)) ** 2)
            rows[key] = {
                "observed": round(observed, 6),
                "sigma": round(sigma, 6),
                # The smallest effect this many cycles could have separated
                # from zero at these conventions.
                "detectable": round(detectable, 6),
                "cycles_needed": int(needed),
                "powered": bool(abs(observed) >= detectable),
            }
        out[against] = rows
    return out


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

    leaks = ((evidence.get("notes", {}) or {}).get("state", {}) or {}).get("leaks") or []
    if leaks:
        blocking.append(f"a module kept a path into the shared state root: {leaks[:6]}")

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
    # The bound, to match what the criterion asks of the same number. Comparing
    # a point estimate whose interval contains zero against a null's upper
    # quantile is comparing noise against a ceiling, and it can come out either
    # way for reasons that are about the estimator.
    real_phi = float(evidence["phi"].get("lower_bound", evidence["phi"]["phi_do"]))
    real_point = float(evidence["phi"]["phi_do"])

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
        # And the rest of the suite, on the same toy recording. A null suite
        # that only measures irreducibility answers one line of a conjunction
        # and leaves the other twenty-three untested on exactly the systems
        # built to fake them: independent noise is what a differentiation bar
        # read one way rewards, a common driver is what an observational
        # spread measure cannot tell from coupling, and a memory-only system
        # is what intrinsic persistence was written to catch.
        extra: dict[str, Any] = {}
        try:
            from core.subject.differentiation import effective_dimension
            from core.subject.intrinsic import intrinsic_gain
            from core.subject.synergy import synergy_suite

            reference_system = architecture(name, seed=args.seed)
            scored = toy_recording(reference_system, steps=2500, seed=args.seed)
            spectrum = effective_dimension(scored)
            extra["d_eff"] = round(float(spectrum.d_eff), 4)
            extra["d_eff_normalised"] = round(float(spectrum.normalised), 4)
            extra["largest_component_share"] = round(float(spectrum.top_share), 4)
            extra["intrinsic_gain"] = round(float(intrinsic_gain(scored, seed=args.seed).gain), 5)
            # `normalised`, the report's own field. This read `.fraction`, which
            # the report has never had: every run raised here, the except below
            # logged it nineteen times a run, and no null was ever scored on
            # synergy or on spread, because spread came after this line in the
            # same block.
            reports = synergy_suite(scored, seed=args.seed)
            extra["synergy"] = [round(float(r.normalised), 4) for r in reports]
            extra["synergy_passes"] = [bool(r.passes) for r in reports]
            extra["synergy_min"] = (
                round(min(float(r.normalised) for r in reports), 4) if reports else 0.0
            )
        except (ImportError, ValueError, RuntimeError, AttributeError, TypeError) as exc:
            _log(f"  the wider suite was unavailable for the {name} null: {exc}")
        # Spread in its own block. It needs nothing the lines above produce,
        # and sharing their failure is how it went unmeasured for every null.
        # Spread is how far a displacement travels: the share of the other
        # domains a source reaches, taken at the source that reaches most.
        reached: dict[str, set[str]] = {}
        for source, target in edges:
            reached.setdefault(source, set()).add(target)
        extra["spread"] = round(
            max((len(v) for v in reached.values()), default=0) / max(1, len(domains) - 1), 4
        )

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
            **extra,
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
            # And differentiation, which is the line that separates a system
            # from a broadcast. All-to-all scores 0.44 on irreducibility with
            # one component and vertex connectivity three — it is genuinely
            # integrated, and that is not the objection to it. Its effective
            # dimension is 1.4 with 84% of the variance in one component: every
            # domain hears the same signal and adds nothing. A conjunction that
            # left this line out would count a broadcast as a mind.
            and _differentiated_enough(row)
            # And synergy, which the conjunction never asked about. The null
            # suite computed it for the report and then read a field the report
            # does not have, so it was never measured and never judged.
            and _synergy_holds(row)
        )

    def _differentiated_enough(row: dict[str, Any]) -> bool:
        """The differentiation criterion, on a null's own numbers.

        A row with no differentiation measured cannot be judged on it, and an
        unmeasured line is not a passed line — but it is not a failed one
        either, so the conjunction reads it as undecided and the row falls
        through on whatever else it failed. `nulls_fail_the_bar` is what would
        catch a null that passed everything measured, and it says so.
        """
        if "d_eff" not in row:
            return True
        # The two scale-free halves, and not the absolute count. `d_eff >= 3`
        # is a bar on a number that grows with the width of the system, and a
        # toy of forty columns is not comparable to an organism of two hundred
        # and eight on it: the recurrent reference scores 2.93 and would fail
        # the line it exists to pass, which would leave the instrument unable
        # to say yes to anything. What does compare is how much of the variance
        # sits in one component and how much of the width is live, and those
        # are what separate a broadcast from a system at any size.
        return (
            float(row.get("largest_component_share", 1.0)) < THRESHOLDS["component_share"]
            and float(row.get("d_eff_normalised", 1.0)) < THRESHOLDS["d_eff_normalised"]
        )

    def _synergy_holds(row: dict[str, Any]) -> bool:
        """The synergy criterion, on a null's own numbers.

        The battery passes synergy only when every triple clears its bar and
        its shifted null, so a null is held to all four as well. Nothing
        measured reads as undecided, in the way `_differentiated_enough` does,
        and the row falls through on whatever else it failed.

        Spread is reported beside it and not judged here. The null's spread is
        the reach of its single furthest-reaching source, and the battery's is
        the mean share reached over displaced sources: two quantities with one
        name, and holding a null to a bar set on the other would be a
        comparison of nothing.
        """
        passes = row.get("synergy_passes")
        if not passes:
            return True
        return all(bool(item) for item in passes)

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
        "compared_on": "lower_bound",
        "real_lower_bound": round(real_phi, 5),
        "real_point_estimate": round(real_point, 5),
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
