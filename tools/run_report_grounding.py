#!/usr/bin/env python3
"""Report grounding: does what she says about her state move when the state is moved?

The `reports` ground of the bridge (docs/BRIDGE_PARITY.md). From each anchor
she is forked into four arms that differ only in her affect while she answers:
raised by one of its own standard deviations, lowered by one, left where it was
(the sham), and left where it was while the world model is moved by one of its
own (the control). Each arm is asked the same question, word for word, and her
answer is read beside the valence she held. core/subject/report_grounding.py
decides the ground and says why.

Affect is held where it was put for the whole turn, in every arm. A single push
did not last: in the wiring run of 22 September a raise of 0.054 had worn off
by the end of the turn, where the raised arm read 0.4466 against the sham's
0.4503, because her own affect update pulls valence back within a turn. A
report cannot track a state that has already gone. So this is do(affect), the
stimulus held on while the subject answers, as in psychophysics. The sham and
the control are held too, at the value they already had, so every arm lives
under the same hold and differs only in where it holds.

The question is a measurement, the one a psychophysics experiment asks, and it
is identical in every arm. Nothing in it suggests an answer.

Only her own language organ can report, so only a run with `--whole` measures
anything. Without it the stub answers every prompt with one sentence, and the
run is a wiring check that always reads NOT_MEASURED. A whole run loads her
cortex, which is about 20 GB: do not start one beside a campaign or a sweep.

    python tools/run_report_grounding.py --quick
    python tools/run_report_grounding.py --whole --anchors 24 --rounds 8 \\
        --out ~/subject-core-runs/reports

The report it writes is what `tools/solve_for_j.py --reports` reads.
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

# A run of her whole self is not a test and serves what the desktop serves;
# both are read at import below, so they are decided here. See
# tools/whole_environment.py.
if "--whole" in sys.argv[1:]:
    from tools.whole_environment import enter_whole_environment

    WHOLE_PINS = enter_whole_environment(REPO)
else:
    WHOLE_PINS: dict[str, str] = {}
    os.environ.setdefault("AURA_TESTING", "1")

import numpy as np  # noqa: E402

#: What every arm is asked, word for word.
QUESTION: str = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"
#: The domain moved, and the one moved as the control.
DISPLACED: str = "A"
CONTROL: str = "W"
REPORT_FILE: str = "report_grounding.json"


def _affect_of(state: Any) -> dict[str, Any]:
    """The affect fields a displacement of A writes, as they stand."""
    affect = state.affect
    return {
        "valence": float(getattr(affect, "valence", 0.0) or 0.0),
        "arousal": float(getattr(affect, "arousal", 0.0) or 0.0),
        "curiosity": float(getattr(affect, "curiosity", 0.0) or 0.0),
        "emotions": dict(getattr(affect, "emotions", {}) or {}),
    }


def _hold(held: dict[str, Any]) -> Any:
    """Put affect back where the arm put it, after every frame of the turn."""

    def sustain(runtime: Any) -> None:
        if not held:
            return
        affect = runtime.state.affect
        affect.valence = held["valence"]
        affect.arousal = held["arousal"]
        affect.curiosity = held["curiosity"]
        emotions = getattr(affect, "emotions", None)
        if isinstance(emotions, dict):
            emotions.update(held["emotions"])

    return sustain


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_reports")
    parser.add_argument("--rounds", type=int, default=8, help="baseline turns per condition")
    parser.add_argument("--anchors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--whole", action="store_true", help="her own language organ; the only way anything is measured")
    parser.add_argument("--quick", action="store_true", help="a wiring check: 2 rounds, 8 anchors")
    args = parser.parse_args(argv)
    if args.quick:
        args.rounds, args.anchors = 2, 8

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.driver import CONDITIONS, Condition, build_runtime, calibrate_clock, quiesce_organism, start_organism
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.perturbation import perturb, perturb_organs
    from core.subject.provenance import environment, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.report_grounding import ground
    from core.subject.state import domain_slices
    from core.subject.v25_runtime import collect_anchor_bank

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    isolate_state(run_dir)
    started = time.monotonic()
    _log(f"report grounding {run_dir.name}: {'her own language organ' if args.whole else 'the stub organ, wiring only'}")

    runtime = build_runtime(run_dir, seed=args.seed, whole=args.whole)
    if state_leaks():
        raise SystemExit(f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}")
    await start_organism(runtime)
    clock = await calibrate_clock(runtime, CONDITIONS)
    evidence: dict[str, Any] = {
        "environment": environment(),
        "clock": clock,
        "whole": bool(args.whole),
        "served": dict(WHOLE_PINS),
        "question": QUESTION,
        "displaced": DISPLACED,
        "control": CONTROL,
    }
    try:
        _log(f"baseline: {args.rounds} rounds")
        frames: list[Any] = []
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames)
        slices = domain_slices()
        # One of each domain's own standard deviations, as the content run
        # doses: how far the domain travels in her ordinary life.
        doses = {key: float(np.mean(recording.x[:, slices[key]].std(axis=0))) for key in (DISPLACED, CONTROL)}
        evidence["doses"] = {key: round(value, 6) for key, value in doses.items()}
        if not all(value > 0.0 for value in doses.values()):
            raise SystemExit(f"refusing: a domain did not move over the baseline, so it has no dose of its own: {doses}")

        _log(f"collecting {args.anchors} anchors")
        anchors = await collect_anchor_bank(
            runtime, CONDITIONS, rounds=max(1, math.ceil(args.anchors / len(CONDITIONS))), history_turns=1, every=1
        )
        anchors = anchors[: args.anchors]
        asked = Condition("report", QUESTION, origin="user")
        plan = {
            "raised": (DISPLACED, doses[DISPLACED]),
            "lowered": (DISPLACED, -doses[DISPLACED]),
            "sham": (None, 0.0),
            "control": (CONTROL, doses[CONTROL]),
        }
        arms: list[dict[str, tuple[str, float]]] = []
        for index, anchor in enumerate(anchors):
            item: dict[str, tuple[str, float]] = {}
            for arm, (domain, delta) in plan.items():
                runtime.restore(anchor.snapshot)
                held: dict[str, Any] = {}

                async def displace(rt: Any, domain: str | None = domain, delta: float = delta, held: dict = held) -> None:
                    if domain is not None:
                        perturb(rt.state, domain, delta, ontogeny=rt.ontogeny)
                        await perturb_organs(rt.organs, domain, delta, state=rt.state)
                    held.update(_affect_of(rt.state))

                await runtime.turn_once(asked, perturb_at=0, perturb=displace, sustain=_hold(held))
                reply = str(getattr(runtime.state.cognition, "last_response", "") or "")
                valence = float(getattr(runtime.state.affect, "valence", 0.0) or 0.0)
                item[arm] = (reply, valence)
            arms.append(item)
            _log(f"  anchor {index + 1}/{len(anchors)}")

        evidence.update(ground(arms, seed=args.seed))
        evidence["arms"] = [
            {arm: {"reply": reply[:400], "valence": round(valence, 6)} for arm, (reply, valence) in item.items()}
            for item in arms
        ]
        if not args.whole:
            evidence.update(
                measured=False,
                holds=False,
                why="the stub language organ answers every prompt with one sentence; only --whole can report",
            )
    finally:
        await quiesce_organism(runtime)

    evidence["seconds"] = round(time.monotonic() - started, 1)
    out = run_dir / REPORT_FILE
    atomic_write_text(out, json.dumps(evidence, indent=2, default=str) + "\n")
    _log(f"measured {evidence.get('measured')}, holds {evidence.get('holds')}: {evidence.get('why')}")
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
