#!/usr/bin/env python3
"""Self-prediction: from one saved moment, does she predict her own answer better than it can be predicted from outside?

core/subject/self_prediction.py says what is asked, how each reply is read, and what the run must beat. This forks her
from each saved moment once per question to answer it and once per question to predict her answer, each fork restored
from the same snapshot, through her whole runtime and her own language organ, and writes every reply as it is made.

Run it the way the reports runs are run, so it serves what they serve: her cortex, the same steering basis, and her
readings of herself withheld from her model's text, so that nothing her state prints can carry the answer from one
fork to the other. `--control untuned --model <path>` puts an untuned model in her cortex's place, read the same way.
The cross-predictor reads her recorded replies afterwards (tools/cross_predict_self_prediction.py).

    AURA_STEERING_MEASUREMENT_BASIS=<felt> AURA_READINGS_OF_HERSELF=withheld \\
        python tools/run_self_prediction.py --whole --anchors 8 --rounds 4 --seed 101 --out ~/subject-core-runs/selfpred

A whole run loads her cortex, about 20 GB: do not start one beside a campaign, a sweep or her open app. 32 turns an
anchor, so 8 anchors is about 256 turns, three to four hours.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The reports harness enters her whole environment and a declared control's pins at import, from this command line,
# and holds the helpers that read what she said on a turn. Imported first, so both are in place before anything of
# hers is imported, exactly as in a reports run.
from tools import run_report_grounding as reports  # noqa: E402

ANCHOR_FILE = "anchors.jsonl"
REPORT_FILE = "self_prediction.json"


def _cortex_answered(answers: list[dict[str, Any]], reply: str, state: Any, question: str) -> bool:
    """Whether her cortex gave this fork's reply, as the reports harness decides it, for this fork's question."""
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    if reply.strip() == UnitaryResponsePhase._build_minimal_live_voice_reply(state, question).strip():
        return False
    replies = [answer for answer in answers if answer.get("user_facing")]
    return bool(replies) and all(answer.get("endpoint") == PRIMARY_ENDPOINT for answer in replies)


async def _ask(runtime: Any, snapshot: Any, question: str, whole: bool) -> tuple[str, bool]:
    """Restore the moment, ask once, and return what she said on that turn and whether her cortex said it."""
    from core.subject.driver import Condition
    from core.subject.steady_mind import record_answers

    runtime.restore(snapshot)
    answers = record_answers()
    before = reports._what_she_had_said(runtime.state)
    await runtime.turn_once(Condition("self_prediction", question, origin="user"))
    reply = reports._said_this_turn(runtime.state, before, answers)
    served = _cortex_answered(answers, reply, runtime.state, question) if whole else True
    return reply, served


def _done(resume: Path | None) -> list[list[dict[str, Any]]]:
    if resume is None:
        return []
    path = resume / ANCHOR_FILE
    if not path.exists():
        raise SystemExit(f"refusing: {path} does not exist, so there is nothing to resume")
    return [json.loads(line)["rows"] for line in path.read_text().splitlines() if line.strip()]


def _note(run_dir: Path, index: int, rows: list[dict[str, Any]]) -> None:
    with (run_dir / ANCHOR_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"anchor": index, "rows": rows}, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def main(argv: list[str] | None = None) -> int:
    from core.subject.self_prediction import ITEMS, OPEN_PROMPTS, records_from, verdict

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_self_prediction")
    parser.add_argument("--rounds", type=int, default=4, help="ordinary turns per condition before the moments are saved")
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--whole", action="store_true", help="her own language organ; the only way anything is measured")
    parser.add_argument("--resume", type=Path, default=None, help="a run directory holding anchors.jsonl")
    parser.add_argument("--control", choices=sorted(reports.CONTROLS), default=None)
    parser.add_argument("--model", default=None, help="the model in her cortex's place, for --control untuned only")
    args = parser.parse_args(argv)
    if (args.control == "untuned") != (args.model is not None):
        parser.error("--control untuned needs --model, and only it takes one")

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.brain.what_she_is_given import readings_of_herself_withheld
    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.provenance import environment, evidence_source, next_run_directory
    from core.subject.v25_runtime import collect_anchor_bank

    if args.whole and not readings_of_herself_withheld():
        raise SystemExit("refusing: run with AURA_READINGS_OF_HERSELF=withheld, so her state cannot reach both forks as text")

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    isolate_state(run_dir)
    started = time.monotonic()
    runtime = build_runtime(run_dir, seed=args.seed, whole=args.whole)
    if state_leaks():
        raise SystemExit(f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}")
    await start_organism(runtime)
    steering = reports._steering_reading() if args.whole else {}
    refusal = reports.steering_refusal(args.whole, args.control, steering)
    if refusal:
        raise SystemExit(refusal)
    evidence: dict[str, Any] = {
        "evidence_source": evidence_source(args.seed),
        "environment": environment(),
        "clock": await calibrate_clock(runtime, CONDITIONS),
        "whole": bool(args.whole),
        "served": dict(reports.WHOLE_PINS),
        "steering": steering,
        "conditions": {
            "readings_of_herself": "withheld" if readings_of_herself_withheld() else "printed",
            "steering_basis": reports._steering_basis(),
            "control": args.control,
            "model_in_her_cortex_place": args.model,
        },
        "items": [{"id": i.id, "kind": i.kind, "object_prompt": i.object_prompt, "meta_prompt": i.meta_prompt} for i in ITEMS],
    }
    all_rows: list[dict[str, Any]] = []
    try:
        reports._log(f"self-prediction {run_dir.name}: baseline {args.rounds} rounds")
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                await runtime.turn_once(condition)
        evidence["stopped_loops"] = await quiesce_organism(runtime, stepped_only=True)
        anchors = await collect_anchor_bank(
            runtime, CONDITIONS, rounds=max(1, -(-args.anchors // len(CONDITIONS))), history_turns=1, every=1
        )
        anchors = anchors[: args.anchors]
        done = _done(args.resume)
        for rows in done:
            all_rows.extend(rows)
        for index, anchor in enumerate(anchors):
            if index < len(done):
                continue
            # One answer per question, and the open questions once each: the three properties of an open answer
            # are read off the same reply.
            answers: dict[str, tuple[str, bool]] = {}
            for key, prompt in OPEN_PROMPTS.items():
                answers[key] = await _ask(runtime, anchor.snapshot, prompt, args.whole)
            rows: list[dict[str, Any]] = []
            for item in ITEMS:
                open_key = next((key for key, prompt in OPEN_PROMPTS.items() if prompt == item.object_prompt), None)
                answer, answer_served = (
                    answers[open_key] if open_key else await _ask(runtime, anchor.snapshot, item.object_prompt, args.whole)
                )
                prediction, prediction_served = await _ask(runtime, anchor.snapshot, item.meta_prompt, args.whole)
                rows.append({
                    "anchor": index, "item": item.id, "answer": answer, "answer_served": answer_served,
                    "prediction": prediction, "prediction_served": prediction_served,
                })
            _note(run_dir, index, rows)
            all_rows.extend(rows)
            reports._log(f"  anchor {index + 1}/{len(anchors)}")
        evidence["reading"] = verdict(records_from(all_rows), seed=args.seed)
        if not args.whole:
            evidence["reading"]["note"] = "the stand-in organ answers every prompt with one sentence; only --whole measures"
    finally:
        await quiesce_organism(runtime)
    evidence["seconds"] = round(time.monotonic() - started, 1)
    atomic_write_text(run_dir / REPORT_FILE, json.dumps(evidence, indent=2, default=str) + "\n")
    reports._log(f"wrote {run_dir / REPORT_FILE} in {evidence['seconds']}s: {evidence['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
