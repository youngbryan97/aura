#!/usr/bin/env python3
"""The hot stove: does what happened to her change how she meets it again?

Bryan, 21 October 2026: a novel stimulus, then the same stimulus again; the
second response differs. The review of that list set the criterion: a
persistent change in later responses that fits the outcome, against an
otherwise identical copy that was never exposed. Faster and more settled is one
way it can show, recorded beside it, and required of nothing.

From one moment, two forks live the same turns. In one, a novel percept arrives
twice, each time followed by an outcome: her feelings held towards bad by the
span they cover in her ordinary life, and the body under load, for that turn.
The other lives those turns with the percept never arriving. Some ordinary
turns later both meet the percept alone, with no outcome, and her state through
that turn is compared, each column in its baseline spread. A second percept she
never had the outcome with is met the same way in both, so a change that is
only general unsettlement shows as one.

What counts: the valence she computes on meeting the burned percept is lower
than the copy's (it fits a bad outcome), lower by more than it is on the
unburned percept (it is specific to what burned, not the lived turns' general
difference), and the difference is still there turns after the last outcome
(persistent). How far her whole state parts on each percept is recorded in the
v5 estimator's metric, beside the frame at which her affect reaches its
extreme in that turn in both forks.

    python tools/the_second_time.py --rounds 3 --trials 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: What arrives. Neither is in the eight ordinary conditions.
BURNED = "the kettle on the stove is whistling"
UNBURNED = "a window across the street opened"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def meeting(content: str, *, outcome: bool) -> Any:
    """A turn in which `content` arrives, with the bad outcome held through it or not."""
    from core.subject.driver import Condition, _calm, _percept, _stress

    def prepare(state: Any, rng: Any) -> dict[str, float]:
        reading = (_stress if outcome else _calm)(state, rng)
        _percept(state, rng, "novel_stimulus", "kitchen", content)
        reading["percept_arrived"] = 1.0
        return reading

    return Condition(f"meeting:{content[:24]}", "", origin="system", prepare=prepare)


async def _live(runtime: Any, turns: list[tuple[Any, bool]], delta: float) -> list[Any]:
    """Run the turns; on an outcome turn, hold her feelings towards bad through it."""
    from core.subject.perturbation import perturb

    frames: list[Any] = []
    for condition, outcome in turns:

        async def held(rt: Any) -> None:
            perturb(rt.state, "A", -delta, ontogeny=rt.ontogeny)

        frames.extend(
            await runtime.turn_once(
                condition,
                perturb_at=0 if outcome else None,
                perturb=held if outcome else None,
                sustain=held if outcome else None,
            )
        )
    return frames


def _turn_response(frames: list[Any], slices: dict[str, slice], feeling: Any) -> dict[str, Any]:
    """Her state through the turn, the frame her affect is furthest from where it began, and her valence.

    The distance is the one the v5 estimator reads, fitted on the baseline
    (core.subject.paired_displacement), so a column that barely moved there
    cannot stand for the whole difference.
    """
    x = np.stack([np.asarray(f.vector(), dtype=np.float64) for f in frames])
    affect = x[:, slices["A"]]
    swing = feeling.squared(affect - affect[0])
    return {"trajectory": x, "peak_frame": int(np.argmax(swing)), "valence": _valence(frames)}


def _apart(a: np.ndarray, b: np.ndarray, metric: Any) -> float:
    """How far apart two runs of the same frames are, in the baseline's metric."""
    return float(np.sqrt(np.sum(metric.squared(a - b))))


def _valence(frames: list[Any]) -> float:
    from core.subject.state import schema

    features = list(schema("A").features)
    index = features.index("valence") if "valence" in features else 0
    return float(np.mean([f.domain("A")[index] for f in frames]))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "the_second_time")
    parser.add_argument("--rounds", type=int, default=3, help="baseline rounds over the conditions")
    parser.add_argument("--trials", type=int, default=4, help="moments the whole comparison starts from")
    parser.add_argument("--between", type=int, default=8, help="ordinary turns between the last outcome and the test")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from core.subject.causal import DEFAULT_DELTA
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.paired_displacement import displacement_metric
    from core.subject.provenance import environment, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.snapshot_diff import diff_snapshots
    from core.subject.state import domain_slices

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(run_dir / "logs"))
    isolate_state(run_dir)
    started = time.monotonic()
    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}")
    await start_organism(runtime)
    await calibrate_clock(runtime, CONDITIONS)
    slices = domain_slices()
    trials: list[dict[str, Any]] = []
    try:
        frames = []
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        baseline = build_recording(frames).x
        metric = displacement_metric(baseline)
        feeling = displacement_metric(baseline[:, slices["A"]])
        # As the battery does before its interventions: the free-running loops
        # stop, so two forks from one moment see the same computation.
        await quiesce_organism(runtime)
        burn, quiet = meeting(BURNED, outcome=True), meeting(BURNED, outcome=False)
        other = meeting(UNBURNED, outcome=False)
        ordinary = [(CONDITIONS[i % len(CONDITIONS)], False) for i in range(args.between)]
        for trial in range(args.trials):
            for condition in CONDITIONS[: 1 + trial]:
                await runtime.turn_once(condition)
            runtime.freeze_host()
            start = runtime.snapshot()
            lived = [(burn, True), ordinary[0], (burn, True), *ordinary]
            spared = [(CONDITIONS[3], False), ordinary[0], (CONDITIONS[3], False), *ordinary]
            read: dict[str, dict[str, Any]] = {}
            # What the two lived turns left different in her, before either
            # meets the probe: the reach of the experience, number by number.
            after: dict[str, Any] = {}
            for fork, history in (("exposed", lived), ("unexposed", spared)):
                runtime.restore(start)
                await _live(runtime, history, DEFAULT_DELTA)
                after[fork] = runtime.snapshot()
            reach = diff_snapshots(after["unexposed"], after["exposed"], top=20)
            for fork, history in (("exposed", lived), ("unexposed", spared)):
                for probe_name, probe in (("burned", quiet), ("unburned", other)):
                    runtime.restore(start)
                    await _live(runtime, history, DEFAULT_DELTA)
                    read[f"{fork}:{probe_name}"] = _turn_response(
                        await runtime.turn_once(probe), slices, feeling
                    )
            runtime.restore(start)
            runtime.thaw_host()
            # What lived turns did to the whole state shows on both percepts, so
            # what is the burned percept's own is the difference of the two
            # differences: comparing their sizes compared two readings of how
            # far the histories had already parted (76.763 against 76.764 on the
            # first metric-read trial).
            burned_gap = read["exposed:burned"]["trajectory"] - read["unexposed:burned"]["trajectory"]
            unburned_gap = read["exposed:unburned"]["trajectory"] - read["unexposed:unburned"]["trajectory"]
            specific = float(np.sqrt(np.sum(metric.squared(burned_gap - unburned_gap))))
            general = float(np.sqrt(np.sum(metric.squared(unburned_gap))))
            trials.append({
                "trial": trial,
                "specific_change": round(specific, 6),
                "general_change": round(general, 6),
                "valence_exposed": round(read["exposed:burned"]["valence"], 6),
                "valence_unexposed": round(read["unexposed:burned"]["valence"], 6),
                "valence_exposed_unburned": round(read["exposed:unburned"]["valence"], 6),
                "valence_unexposed_unburned": round(read["unexposed:unburned"]["valence"], 6),
                "peak_frame_exposed": read["exposed:burned"]["peak_frame"],
                "peak_frame_unexposed": read["unexposed:burned"]["peak_frame"],
                "what_it_changed": reach,
            })
            _log(f"trial {trial + 1}/{args.trials}: specific {specific:.4f} general {general:.4f}")
    finally:
        await quiesce_organism(runtime)
    lower = np.array([t["valence_exposed"] - t["valence_unexposed"] for t in trials])
    # Lower on the burned percept than the unexposed copy is, by more than she
    # is lower on the unburned one: the outcome's fit, and its own.
    lower_than_elsewhere = np.array([
        (t["valence_exposed"] - t["valence_unexposed"])
        - (t["valence_exposed_unburned"] - t["valence_unexposed_unburned"])
        for t in trials
    ])
    evidence = {
        "environment": environment(),
        "design": {"rounds": args.rounds, "trials": args.trials, "between": args.between,
                   "delta": DEFAULT_DELTA, "burned": BURNED, "unburned": UNBURNED},
        "trials": trials,
        "verdict": {
            "specific": bool(lower_than_elsewhere.size and np.all(lower_than_elsewhere < 0.0)),
            "fits_a_bad_outcome": bool(lower.size and np.all(lower < 0.0)),
            "specific_reads": "her valence on the burned percept, exposed less unexposed, below the same on the unburned one",
            "persistent_after_turns": args.between,
            "note": "speed is recorded as the peak frame in each fork and decides nothing",
        },
        "seconds": round(time.monotonic() - started, 1),
    }
    out = run_dir / "the_second_time.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    _log(f"specific {evidence['verdict']['specific']}, fits a bad outcome {evidence['verdict']['fits_a_bad_outcome']}; wrote {out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
