#!/usr/bin/env python3
"""Report grounding: does what she says about her state move when the state is moved?

The `reports` ground of the bridge (docs/BRIDGE_PARITY.md). From each anchor
she is forked into four arms that differ only in her feelings while she
answers: moved towards feeling good by the span her feelings cover in her own
ordinary life, moved towards feeling bad by the same, left where they were (the sham), and left
where they were while the world model is moved by its own span (the control).
Each arm is asked the same question, word for word, and her answer is read
beside the valence her own affect phase computed from those feelings.
core/subject/report_grounding.py decides the ground and says why.

Towards feeling good means each feeling moves along its own sign in her
valence: the feelings `affect_update` weighs as positive go up and those it
weighs as negative go down. The battery's affect writer raises every feeling,
fear as much as joy, and on seed 7 that took her valence down (22 September:
-0.054 after a raise of 0.1), so it is not a manipulation of how good she
feels.

The feelings are held where they were put for the whole turn, in every arm. A
single push did not last: after one turn valence kept -2% of a push, arousal
none, and the feelings 8 to 43% (seed 7, 22 September), because her own affect
update pulls them back within a turn. A report cannot track a state that has
already gone. So this is do(feelings), the stimulus held on while the subject
answers, as in psychophysics, and valence is left to her own computation. The
sham and the control hold their feelings too, where they already were, so
every arm lives under the same hold and differs only in where it holds.

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

#: What every arm is asked, word for word.
QUESTION: str = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"
#: The domain moved, and the one moved as the control.
DISPLACED: str = "A"
CONTROL: str = "W"
REPORT_FILE: str = "report_grounding.json"


def _hold(held: dict[str, float]) -> Any:
    """Put her feelings back where the arm put them, after every frame of the turn."""

    def sustain(runtime: Any) -> None:
        emotions = getattr(runtime.state.affect, "emotions", None)
        if held and isinstance(emotions, dict):
            emotions.update(held)

    return sustain


def _cortex_answered(answers: list[dict[str, Any]]) -> bool:
    """Whether her cortex gave this arm's reply: a user-facing generation, every one from it.

    A turn that fell back to the brainstem, or ended in the failure sentence
    because nothing answered, is not her report.
    """
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    replies = [answer for answer in answers if answer.get("user_facing")]
    return bool(replies) and all(answer.get("endpoint") == PRIMARY_ENDPOINT for answer in replies)


def _steering_reading() -> dict[str, Any]:
    """Whether her affective steering is attached to the cortex worker, as the worker last said."""
    from core.container import ServiceContainer

    gate = ServiceContainer.get("inference_gate", default=None)
    client = getattr(gate, "_mlx_client", None)
    reading = getattr(client, "steering_liveness_reading", None)
    return dict(reading()) if callable(reading) else {"active": None, "why": "no cortex client"}


def _steering_attached(reading: dict[str, Any]) -> bool:
    """Whether the reading shows steering attached. Only True does.

    The client reports None until the worker's flag has once read live, and a
    worker whose steering never attached never sets it, so after her language
    organ is up None means detached, not pending.
    """
    return reading.get("active") is True


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
    from core.subject.driver import (
        CONDITIONS,
        Condition,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.perturbation import ordinary_span, perturb, perturb_organs, towards_good
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
    steering = _steering_reading() if args.whole else {}
    if args.whole and not _steering_attached(steering):
        raise SystemExit(
            "refusing: her affective steering did not attach to the cortex worker, so the run "
            f"would measure her without the path the desktop runs her with: {steering}"
        )
    clock = await calibrate_clock(runtime, CONDITIONS)
    evidence: dict[str, Any] = {
        "environment": environment(),
        "clock": clock,
        "whole": bool(args.whole),
        "served": dict(WHOLE_PINS),
        "steering": steering,
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
        # The span of her own ordinary life (perturbation.ordinary_span): one
        # standard deviation of her feelings moved her valence by about 0.01 on
        # seed 7, below anything a number from -1 to 1 can report.
        feeling_columns = [
            index for index, name in enumerate(recording.columns) if str(name).startswith(f"{DISPLACED}.emotion_")
        ]
        doses = {
            DISPLACED: ordinary_span(recording.x[:, feeling_columns]),
            CONTROL: ordinary_span(recording.x[:, slices[CONTROL]]),
        }
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
            "raised": (doses[DISPLACED], None),
            "lowered": (-doses[DISPLACED], None),
            "sham": (0.0, None),
            "control": (0.0, CONTROL),
        }
        from core.subject.steady_mind import record_answers

        arms: list[dict[str, tuple[str, float, bool]]] = []
        answered: list[dict[str, list[dict[str, Any]]]] = []
        for index, anchor in enumerate(anchors):
            item: dict[str, tuple[str, float, bool]] = {}
            served_by: dict[str, list[dict[str, Any]]] = {}
            for arm, (towards, domain) in plan.items():
                runtime.restore(anchor.snapshot)
                held: dict[str, float] = {}

                async def displace(
                    rt: Any, towards: float = towards, domain: str | None = domain, held: dict = held
                ) -> None:
                    emotions = rt.state.affect.emotions
                    if towards:
                        emotions.update(towards_good(emotions, towards))
                    if domain is not None:
                        perturb(rt.state, domain, doses[domain], ontogeny=rt.ontogeny)
                        await perturb_organs(rt.organs, domain, doses[domain], state=rt.state)
                    held.update({name: float(value or 0.0) for name, value in emotions.items()})

                answers = record_answers()
                await runtime.turn_once(asked, perturb_at=0, perturb=displace, sustain=_hold(held))
                reply = str(getattr(runtime.state.cognition, "last_response", "") or "")
                valence = float(getattr(runtime.state.affect, "valence", 0.0) or 0.0)
                item[arm] = (reply, valence, _cortex_answered(answers) if args.whole else True)
                served_by[arm] = list(answers)
            arms.append(item)
            answered.append(served_by)
            _log(f"  anchor {index + 1}/{len(anchors)}")

        evidence.update(ground(arms, seed=args.seed))
        evidence["arms"] = [
            {
                arm: {
                    "reply": reply[:400],
                    "valence": round(valence, 6),
                    "cortex_answered": served,
                    "steering_alpha": [a.get("steering_alpha") for a in served_by[arm] if a.get("user_facing")],
                }
                for arm, (reply, valence, served) in item.items()
            }
            for item, served_by in zip(arms, answered, strict=True)
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
