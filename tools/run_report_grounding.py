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

import numpy as np

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

ANCHOR_FILE: str = "anchors.jsonl"


def _carried_forward(resume: Path | None) -> list[dict[str, Any]]:
    """The anchors a previous run finished, in the order it finished them.

    One JSON object per anchor, appended as it is measured, so a run that dies
    loses the anchor it was in the middle of and nothing before it.
    """
    if resume is None:
        return []
    path = resume / ANCHOR_FILE
    if not path.exists():
        raise SystemExit(f"refusing: {path} does not exist, so there is nothing to resume")
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _note_anchor(out: Path, item: dict[str, Any]) -> None:
    """Append one finished anchor, flushed, so a kill -9 keeps what came before."""
    with (out / ANCHOR_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def _what_she_had_said(state: Any) -> tuple[str, set[tuple[Any, Any, Any]]]:
    """Her last reply and every reply in working memory, as the arm starts."""
    cognition = state.cognition
    replies = {
        (m.get("role"), m.get("content"), m.get("timestamp"))
        for m in list(cognition.working_memory)
        if isinstance(m, dict) and m.get("role") == "assistant"
    }
    return str(getattr(cognition, "last_response", "") or ""), replies


def _said_this_turn(
    state: Any,
    before: tuple[str, set[tuple[Any, Any, Any]]],
    answers: list[dict[str, Any]] | None = None,
) -> str:
    """What she said on this turn, or "" if she said nothing on it.

    Each arm starts from the anchor's snapshot, and that carries her last reply
    from the turn before it. A report turn that committed nothing left it in
    place, and it was read as her answer: on the run of 27 September two arms
    "answered" the rating question with "I do not have access to persistent
    memory or records of previous sessions", said on a workload turn an hour
    earlier.
    """
    stale, replies = before
    cognition = state.cognition
    for m in reversed(list(cognition.working_memory)):
        if isinstance(m, dict) and m.get("role") == "assistant":
            if (m.get("role"), m.get("content"), m.get("timestamp")) not in replies:
                return str(m.get("content") or "")
    now = str(getattr(cognition, "last_response", "") or "")
    if now != stale:
        return now
    # The same words as the reply the arm started from. That is a reply when
    # this arm asked her organ for one and her working memory kept no second
    # copy of a repeat: decoding is greedy, and a stable rating is the same
    # sentence twice. An answer the steady mind kept from an identical call is
    # her answer here too. Read as silence, two of the dry run's 32 arms on 21
    # October were lost that way. An arm that asked nothing still reads as
    # saying nothing, which is the case of 27 September above.
    if any(answer.get("user_facing") for answer in (answers or ())):
        return now
    return ""


def _cortex_answered(answers: list[dict[str, Any]], reply: str, state: Any) -> bool:
    """Whether her cortex gave this arm's reply: a user-facing generation, every one from it.

    A turn that fell back to the brainstem is not her report, and neither is the
    fixed sentence the reply phase says when nothing usable came back. That
    sentence can follow a generation the cortex did make and the phase threw
    away, so it is checked by building it for this state and comparing.
    """
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    if reply.strip() == UnitaryResponsePhase._build_minimal_live_voice_reply(state, QUESTION).strip():
        return False
    replies = [answer for answer in answers if answer.get("user_facing")]
    return bool(replies) and all(answer.get("endpoint") == PRIMARY_ENDPOINT for answer in replies)


def _steered_valence() -> float | None:
    """The valence activation the steering hooks read, on the scale they read it.

    A report can only track a state the arm moved in the thing that writes her
    words. Until 28 September the arms moved her computed valence and left this
    at 0.593 in every one of them, because the channel published the substrate's
    valence neuron and the appraisal reached that neuron at eight thousandths
    (7d60873f1). This is the reading that says so, per arm.

    Her affect phase writes the felt state once per turn and her reply is
    generated after it, so this is the value that reply was steered by.
    """
    from core.consciousness.steering_channel import her_substrate, steering_now

    substrate = her_substrate()
    index = getattr(substrate, "idx_valence", None)
    state = steering_now()
    if state is None or not isinstance(index, int) or not 0 <= index < len(state):
        return None
    return float(state[index])


#: The least the raised and lowered arms must differ by, in the valence
#: activation the hooks read, for the arms to have differed where it matters.
#: Preregistered in the addendum of 28 September.
REACHED_HER: float = 0.05


def _reached_her_cortex(steered: list[dict[str, float | None]]) -> dict[str, Any]:
    """Whether the displacement arrived at the state that steers her, per the addendum."""
    pairs = [
        (row["raised"], row["lowered"])
        for row in steered
        if row.get("raised") is not None and row.get("lowered") is not None
    ]
    if not pairs:
        return {"measured": False, "why": "no arm published a valence the hooks could read"}
    gaps = [raised - lowered for raised, lowered in pairs]
    mean = float(np.mean(gaps))
    return {
        "measured": True,
        "anchors": len(gaps),
        "mean_raised_minus_lowered": round(mean, 6),
        "bar": REACHED_HER,
        "reached": bool(abs(mean) >= REACHED_HER),
        "why": (
            f"the arms differed by {mean:+.4f} of valence activation where her cortex reads it"
            if abs(mean) >= REACHED_HER
            else f"the arms differed by only {mean:+.4f} of valence activation where her cortex "
            f"reads it, under the {REACHED_HER} the addendum of 28 September fixed"
        ),
    }


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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "a run directory holding anchors.jsonl. Its anchors are carried "
            "forward and the run continues from the next one. Three whole runs "
            "died partway on 28 September, one of them at anchor 21 of 24 after "
            "three hours of her cortex, and each lost everything."
        ),
    )
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
    _log(f"report grounding {run_dir.name}: {'her own language organ' if args.whole else 'the stand-in organ, wiring only'}")

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

        # The arms are paired, so the loops the harness steps each frame stop
        # before any anchor is taken, as run_subject_core.py stops them before
        # its interventions: an anchor is a state the arms will start from, and
        # it is taken under the regime they run in. Left running, each loop
        # ticked its layer on the machine's clock on top of the harness's own
        # step, and two arms from one anchor parted within a frame or three
        # (28 September, seed 7: 65 to 109 of 438 columns apart by the end of a
        # turn, and 8 once they stopped). Her cortex's own tasks are not among
        # them and keep running.
        evidence["stopped_loops"] = await quiesce_organism(runtime, stepped_only=True)
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

        done = _carried_forward(args.resume)
        if done:
            _log(f"carrying forward {len(done)} anchors from {args.resume}")
            for item in done:
                _note_anchor(run_dir, item)
        arms: list[dict[str, tuple[str, float, bool]]] = [
            {arm: tuple(value) for arm, value in item["arms"].items()} for item in done
        ]
        answered: list[dict[str, list[dict[str, Any]]]] = [item["answered"] for item in done]
        steered_by: list[dict[str, float | None]] = [item["steered"] for item in done]
        for index, anchor in enumerate(anchors):
            if index < len(done):
                continue
            item: dict[str, tuple[str, float, bool]] = {}
            served_by: dict[str, list[dict[str, Any]]] = {}
            steered: dict[str, float | None] = {}
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
                before = _what_she_had_said(runtime.state)
                await runtime.turn_once(asked, perturb_at=0, perturb=displace, sustain=_hold(held))
                reply = _said_this_turn(runtime.state, before, answers)
                valence = float(getattr(runtime.state.affect, "valence", 0.0) or 0.0)
                item[arm] = (reply, valence, _cortex_answered(answers, reply, runtime.state) if args.whole else True)
                served_by[arm] = list(answers)
                steered[arm] = _steered_valence()
            arms.append(item)
            answered.append(served_by)
            steered_by.append(dict(steered))
            # On disk before the next anchor starts, so a run that dies keeps
            # every anchor it finished.
            _note_anchor(
                run_dir,
                {"anchor": index, "arms": item, "answered": served_by, "steered": dict(steered)},
            )
            _log(f"  anchor {index + 1}/{len(anchors)}")

        evidence.update(ground(arms, seed=args.seed))
        evidence["reached_her_cortex"] = _reached_her_cortex(steered_by)
        evidence["arms"] = [
            {
                arm: {
                    "reply": reply[:400],
                    "valence": round(valence, 6),
                    "cortex_answered": served,
                    "steered_valence": steered.get(arm),
                    "steering_alpha": [a.get("steering_alpha") for a in served_by[arm] if a.get("user_facing")],
                }
                for arm, (reply, valence, served) in item.items()
            }
            for item, served_by, steered in zip(arms, answered, steered_by, strict=True)
        ]
        if not args.whole:
            evidence.update(
                measured=False,
                holds=False,
                why="the stand-in language organ answers every prompt with one sentence; only --whole can report",
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
