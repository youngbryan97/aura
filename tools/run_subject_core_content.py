#!/usr/bin/env python3
"""The content half of the bridge: is her geometry hers, or is it the recording's?

The carrier run asks whether there is one intrinsic process. If there is, the
next question the bridge asks is what its content is organised like. The
strongest available answer is structural: content, if the bridge holds, is the
position each thing occupies in the whole web of relations, and nothing else.

Nothing here reaches phenomenal character, and the report says so in the same
breath every time. What is testable is whether the relational structure is
really her organisation. Two of her mechanisms are asked the same question
about the same percepts and their answers are compared.

    d_Q   Fisher-Rao distance between the future-state laws two percept
          classes induce from a common fork.

    d_B   one minus the overlap of which memories each percept brought back.

They share the percepts and nothing else: the state vector records how many
memories returned and how well the best matched, never which ones.

Then the test that can fail. Affect is displaced, which moves the internal
geometry, and the run asks whether the behavioural geometry moved the same way.
A structure that merely describes the content has no reason to move with it. A
sham arm, displacing nothing, gives the floor that finite sampling alone
produces. The displacement moves the point her feelings are measured from,
towards feeling good, by the span her feelings cover in her ordinary life, so
it lasts through the turn and leaves the feelings free to answer the percept.

    python tools/run_subject_core_content.py --quick
    python tools/run_subject_core_content.py --anchors 24 --rounds 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Before any core import: the profile decides where state lives, and a module
# that resolved its home under the live root keeps that home.
os.environ.setdefault("AURA_TESTING", "1")

#: The domain the displacement moves. The grid is built from the table that
#: says which emotions a percept may move, so affect is where the content
#: geometry is predicted to live. Declared before the run, not chosen after it.
DISPLACED_DOMAIN: str = "A"

#: How strongly the two geometries must agree before the run says they are one
#: structure. Spearman rank correlation, which is the weakest assumption that
#: still has an answer: the structural claim is about order, not about units.
AGREEMENT_BAR: float = 0.30


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _as_json(mapping: dict[tuple[int, int], float]) -> dict[str, float]:
    return {f"{i}-{j}": round(float(v), 6) for (i, j), v in sorted(mapping.items())}


#: The level a geometry has to stand out of its own noise at, the level its
#: agreement is held to.
NOISE_LEVEL: float = 0.01


def _out_of_noise(between: list[float], floors: list[float]) -> tuple[bool, float]:
    """Whether the between-class distances exceed the same-class floors, by a one-sided rank test."""
    from scipy.stats import mannwhitneyu

    if len(between) < 2 or len(floors) < 2:
        return False, 1.0
    p_value = float(mannwhitneyu(between, floors, alternative="greater").pvalue)
    return p_value < NOISE_LEVEL, p_value


def _authority(evidence: dict[str, Any]) -> dict[str, Any]:
    """What would make this report not a result.

    Missing evidence is NOT_MEASURED. A run that could not read one of its two
    geometries has not compared them, and saying so is the whole point of
    having the gate.
    """
    blockers: list[str] = []
    agreement = evidence.get("agreement", {})
    moves = evidence.get("moves_together", {})
    coverage = evidence.get("behavioural_coverage", {})
    classes = evidence.get("classes", [])
    floor = evidence.get("internal_floor", {})
    internal = evidence.get("internal", {})

    if len(classes) < 4:
        blockers.append("fewer than four percept classes were presented")
    if not agreement.get("measured", False):
        blockers.append(f"the geometries were not comparable: {agreement.get('why', '?')}")
    if not moves.get("measured", False):
        blockers.append(f"the displacement test did not read: {moves.get('why', '?')}")
    silent = [name for name, share in coverage.items() if share <= 0.0]
    if silent:
        blockers.append(
            f"{len(silent)} classes brought back nothing at all, so the behavioural "
            f"geometry has no reading for them: {silent[:4]}"
        )
    # Whether each geometry stands out of its own noise: the distances between
    # classes, as a set, above the distances of classes from themselves. This
    # used to ask that the closest of every pair clear the largest of every
    # floor, the minimum of 190 noisy numbers against the maximum of 20, which
    # grows harder the more classes are presented and refuses a structure in
    # which any two classes are alike. Classes the structure cannot tell apart
    # are what the bridge's gauge is for: they share a cell
    # (core/subject/bridge.py `orbits`, Theorems 5 and 6), and they do not make
    # the rest of the structure unmeasured.
    b_floor = evidence.get("behavioural_floor", {})
    behavioural = evidence.get("behavioural", {})
    if behavioural and b_floor:
        above, p_value = _out_of_noise(list(behavioural.values()), list(b_floor.values()))
        if not above:
            blockers.append(
                "the classes recall no more differently from each other than each "
                f"recalls from itself (p={p_value:.3g}), so retrieval does not separate the percepts"
            )
    if internal and floor:
        above, p_value = _out_of_noise(list(internal.values()), list(floor.values()))
        if not above:
            blockers.append(
                "the classes lie no further apart than each lies from itself "
                f"(p={p_value:.3g}), so the internal geometry is inside its own noise"
            )
    return {
        "authoritative": not blockers,
        "blockers": blockers,
        "missing_evidence_is": "NOT_MEASURED, never PASS",
    }


def _verdict(evidence: dict[str, Any]) -> str:
    if evidence.get("authority", {}).get("blockers"):
        return "NOT_MEASURED"
    agreement = evidence.get("agreement", {})
    moves = evidence.get("moves_together", {})
    agrees = (
        agreement.get("measured")
        and float(agreement.get("rho", 0.0)) >= AGREEMENT_BAR
        and float(agreement.get("p_value", 1.0)) < 0.01
    )
    tracks = (
        moves.get("measured")
        and float(moves.get("rho", 0.0)) >= AGREEMENT_BAR
        and float(moves.get("p_value", 1.0)) < 0.01
        and float(moves.get("rho", 0.0)) > float(moves.get("floor_rho", 0.0))
    )
    if agrees and tracks:
        return "ONE_STRUCTURE"
    if agrees:
        return "AGREES_BUT_DOES_NOT_TRACK"
    return "SEPARATE_STRUCTURES"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_content")
    parser.add_argument("--rounds", type=int, default=16, help="baseline turns per condition")
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--turns", type=int, default=1, help="turns each arm runs past the fork")
    parser.add_argument(
        "--lag", type=int, default=1,
        help="the turn after the fork whose end the future is read at; at most --turns",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--conditions", type=int, default=0)
    parser.add_argument("--classes", type=int, default=0, help="use only the first N classes")
    args = parser.parse_args()

    if args.quick:
        args.rounds, args.anchors = 3, 6
        args.conditions = args.conditions or 2
        args.classes = args.classes or 6

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.content import agreement, design_recovery, gauge, moves_together
    from core.subject.content_runtime import (
        behavioural_floor,
        behavioural_geometry,
        grid,
        internal_geometry,
        sample_classes,
    )
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.provenance import environment, manifest, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.perturbation import ordinary_span
    from core.subject.v25_runtime import collect_anchor_bank

    conditions = CONDITIONS[: args.conditions] if args.conditions else CONDITIONS
    classes = grid()
    if args.classes:
        classes = classes[: args.classes]

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    isolate_state(run_dir)
    started = time.monotonic()
    _log(f"content run {run_dir.name}: {len(classes)} classes over {len(conditions)} conditions")

    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(
            f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}"
        )
    await start_organism(runtime)
    clock = await calibrate_clock(runtime, conditions)

    evidence: dict[str, Any] = {
        "environment": environment(),
        "clock": clock,
        "scope": "substrate_only",
        "displaced_domain": DISPLACED_DOMAIN,
        "agreement_bar": AGREEMENT_BAR,
        "classes": [
            {
                "name": c.name,
                "kind": c.kind,
                "intensity": c.intensity,
                "emotions_the_kind_names": int(sum(c.coordinates)),
            }
            for c in classes
        ],
        "gauge": gauge(),
    }

    try:
        _log(f"baseline: {args.rounds} rounds")
        frames: list[Any] = []
        for _ in range(args.rounds):
            for condition in conditions:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames)
        recording.save(run_dir)
        # The displacement moves the point her feelings are measured from,
        # towards feeling good, by the span her feelings cover in her ordinary
        # life. Not a number chosen here: the recording measured it. The first
        # design pushed affect once by one standard deviation, and on seed 7
        # that push was gone by the end of the turn and moved fear as much as
        # joy (22 September), so "moves together" had nothing to correlate:
        # the behavioural geometry moved 0.008.
        feelings = [
            index for index, name in enumerate(recording.columns) if str(name).startswith(f"{DISPLACED_DOMAIN}.emotion_")
        ]
        dose = ordinary_span(recording.x[:, feelings])
        if not (dose > 0.0):
            raise SystemExit(
                f"refusing: her feelings did not move over {recording.frames} "
                "baseline frames, so there is no dose her own life justifies and "
                "a displacement would be a number I chose"
            )
        _log(f"  {recording.frames} frames; her feelings span {dose:.5g}")

        _log(f"collecting {args.anchors} anchors")
        anchors = await collect_anchor_bank(
            runtime, conditions,
            rounds=max(1, math.ceil(args.anchors / max(1, len(conditions)))),
            history_turns=1, every=1,
        )
        anchors = anchors[: args.anchors]
        _log(f"  {len(anchors)} anchors")

        # ── the geometry, undisplaced ─────────────────────────────────
        _log(f"presenting {len(classes)} classes from every anchor")
        base = await sample_classes(
            runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag
        )
        internal, floor = internal_geometry(base, classes, seed=args.seed)
        behavioural, coverage = behavioural_geometry(base, classes)
        evidence["internal"] = _as_json(internal)
        evidence["internal_floor"] = {f"{i}-{j}": round(v, 6) for (i, j), v in floor.items()}
        evidence["behavioural"] = _as_json(behavioural)
        evidence["behavioural_coverage"] = {k: round(v, 4) for k, v in coverage.items()}
        b_floor = behavioural_floor(base, classes)
        evidence["behavioural_floor"] = {k: round(v, 6) for k, v in b_floor.items()}
        _log(
            f"  internal pairs {len(internal)}, behavioural pairs {len(behavioural)}, "
            f"classes that recalled nothing "
            f"{sum(1 for v in coverage.values() if v <= 0.0)}"
        )

        result = agreement(internal, behavioural, size=len(classes), seed=args.seed)
        evidence["agreement"] = result.__dict__ | {"bar": AGREEMENT_BAR}
        _log(
            f"  agreement rho={result.rho} p={result.p_value} "
            f"(spread {result.internal_spread} / {result.behavioural_spread})"
        )

        recovery = design_recovery(internal, classes, seed=args.seed)
        evidence["design_recovery"] = recovery.__dict__
        _log(f"  design recovery rho={recovery.rho} p={recovery.p_value}")

        # ── the displacement, and the sham that gives its floor ───────
        _log("moving her reference point towards feeling good by that span")
        moved = await sample_classes(
            runtime, anchors, conditions, classes,
            turns=args.turns, lag=args.lag, reference=dose,
        )
        moved_internal, _ = internal_geometry(moved, classes, seed=args.seed)
        moved_behavioural, _ = behavioural_geometry(moved, classes)

        _log("sham: the same measurement again, displacing nothing")
        sham = await sample_classes(
            runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag
        )
        sham_internal, _ = internal_geometry(sham, classes, seed=args.seed + 1)
        sham_behavioural, _ = behavioural_geometry(sham, classes)

        tracked = moves_together(
            internal, moved_internal, behavioural, moved_behavioural,
            size=len(classes),
            sham_internal_after=sham_internal,
            sham_behavioural_after=sham_behavioural,
            seed=args.seed,
        )
        evidence["displacement_dose"] = round(dose, 6)
        evidence["displacement"] = "reference point towards feeling good, by her feelings' ordinary span"
        evidence["moves_together"] = tracked.__dict__ | {"bar": AGREEMENT_BAR}
        _log(
            f"  moves together rho={tracked.rho} p={tracked.p_value} "
            f"floor={tracked.floor_rho}"
        )

        evidence["authority"] = _authority(evidence)
        evidence["verdict"] = _verdict(evidence)
        evidence["bridge_status"] = {
            "content_structure": evidence["verdict"],
            "phenomenal_bridge": "JUDGED_BY_SOLVE_FOR_J",
            "note": (
                "Two of her mechanisms agreeing about one geometry is evidence that "
                "the geometry is her organisation rather than the recording's. It is "
                "the structure ground of the bridge, judged at parity with the other "
                "four (docs/BRIDGE_PARITY.md). That the organisation is felt is the "
                "structural-identity postulate, assumed for her as for a person."
            ),
        }
    finally:
        await quiesce_organism(runtime)

    evidence["seconds"] = round(time.monotonic() - started, 1)
    evidence["manifest"] = manifest(run_dir)
    out = run_dir / "subject_core_content_report.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    _log("")
    _log(f"verdict: {evidence['verdict']}")
    for blocker in evidence["authority"]["blockers"]:
        _log(f"  blocked: {blocker}")
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0 if evidence["authority"]["authoritative"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
