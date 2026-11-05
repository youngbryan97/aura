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
from collections.abc import Callable, Sequence
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



def _marked_fixed(state: Any, reply: str) -> bool:
    """Whether the reply phase marked this reply as its own fixed text.

    The phase notes what its template builders return (`FIXED_REPLY_KEY`) and
    marks the committed entry `fixed` when the reply is one
    (`UnitaryResponsePhase._commit_response`). A path that sets the reply
    without committing it leaves the note in place. Rebuilding the sentence
    afterwards is not enough: "The live answer path failed before I could
    produce a verified reply for ..." names the objective the state held when
    it was built, which the state after the turn no longer holds. On 1 November
    that sentence was counted as her prediction on every yes-or-no question of
    the run. Read before anything rebuilds it, since building it notes it again.
    """
    from core.phases.response_generation_unitary import FIXED_REPLY_KEY

    noted = str((getattr(state, "response_modifiers", None) or {}).get(FIXED_REPLY_KEY, "") or "")
    if noted and noted.strip() == reply.strip():
        return True
    for entry in reversed(list(getattr(state.cognition, "working_memory", ()) or ())):
        if isinstance(entry, dict) and entry.get("role") == "assistant":
            if str(entry.get("content") or "").strip() == reply.strip():
                return bool(entry.get("fixed"))
    return False


def _cortex_answered(answers: list[dict[str, Any]], reply: str, state: Any, question: str) -> bool:
    """Whether her cortex gave this fork's reply, as the reports harness decides it, for this fork's question."""
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    if _marked_fixed(state, reply):
        return False
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


def remember_answers(
    earlier: dict[str, list[dict[str, Any]]],
    rows: Sequence[dict[str, Any]],
    items: Sequence[Any],
    noted: Callable[[str, str], dict[str, Any] | None],
) -> None:
    """Add one moment's served answers to `earlier`, by question, each as her record notes it (`noted`) and marked
    as from an earlier moment. The three properties of an open answer share its draws, so each draw is added once."""
    prompts = {i.id: i.object_prompt for i in items}
    for row in rows:
        prompt = prompts[row["item"]]
        if any(n["_from"] == row["anchor"] for n in earlier.get(prompt, [])):
            continue
        for reply, served in row.get("answers", []):
            note = noted(prompt, reply) if served in (True, "True") else None
            if note:
                earlier.setdefault(prompt, []).append({**note, "earlier_moment": True, "_from": row["anchor"]})


def _note(run_dir: Path, index: int, rows: list[dict[str, Any]]) -> None:
    with (run_dir / ANCHOR_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"anchor": index, "rows": rows}, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def main(argv: list[str] | None = None) -> int:
    from core.subject.self_prediction import item_set, records_from, verdict
    from core.subject.self_prediction_by_moment import (
        by_moment_records,
        by_moment_verdict,
        stated_prompt,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_self_prediction")
    parser.add_argument("--rounds", type=int, default=4, help="ordinary turns per condition before the moments are saved")
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--whole", action="store_true", help="her own language organ; the only way anything is measured")
    parser.add_argument("--resume", type=Path, default=None, help="a run directory holding anchors.jsonl")
    parser.add_argument("--control", choices=sorted(reports.CONTROLS), default=None)
    parser.add_argument("--model", default=None, help="the model in her cortex's place, for --control untuned only")
    parser.add_argument("--items", choices=("v1", "v2"), default="v1",
                        help="v1, the twenty questions of 30 October; v2, the twenty-four of 4 November")
    parser.add_argument("--arms", default="",
                        help="the self-observation experiment: comma-separated prediction arms from "
                             "with,without,shuffled, each asked from the same moments; empty is the protocol of 2 November")
    parser.add_argument("--answer-draws", type=int, default=1,
                        help="how many times each question is asked from the same moment: 2 sets the ceiling her own "
                             "second answer sets; more gives the moment's own most common answer as a target")
    parser.add_argument("--stated", action="store_true",
                        help="predictions are asked with how sure she is (0 to 100), and the run is also read against "
                             "the moment's own answer and by stated probability (core/subject/self_prediction_by_moment.py)")
    parser.add_argument("--only", default="",
                        help="comma-separated item ids: a smoke run on a few questions, never a reading")
    parser.add_argument("--earlier-answers", action="store_true",
                        help="her record is lent her answers to the same question from the earlier saved moments, and "
                             "nothing from later ones; the run is also read against her habit from earlier moments")
    parser.add_argument("--rehearse", action="store_true",
                        help="from each moment she drafts an answer to each question first, never served; the "
                             "rehearsed arm reads it, and a rehearsing self-model predicts from it alone")
    args = parser.parse_args(argv)
    arms = [arm for arm in (a.strip() for a in args.arms.split(",")) if arm]
    # The arms of core.self.how_i_answer.ARMS, named here because nothing of hers may be imported before her state
    # is isolated: core.self imports core.self_model, which fixes its data path on import.
    allowed = ("with", "without", "shuffled", *(("rehearsed",) if args.rehearse else ()))
    if any(arm not in allowed for arm in arms):
        parser.error(f"--arms takes {', '.join(allowed)} (rehearsed needs --rehearse)")
    if args.rehearse and "rehearsed" not in arms:
        parser.error("--rehearse needs the rehearsed arm in --arms")
    ITEMS, OPEN_PROMPTS = item_set(args.items)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        ITEMS = tuple(item for item in ITEMS if item.id in wanted)
        OPEN_PROMPTS = {key: prompt for key, prompt in OPEN_PROMPTS.items()
                        if any(item.object_prompt == prompt for item in ITEMS)}
        if not ITEMS:
            parser.error(f"--only named no item of {args.items}")
    if not arms:
        # The protocol of 2 November had no record of how she answers. Kept off, so it is the same protocol.
        os.environ["AURA_SELF_OBSERVATION"] = "off"
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
    from core.brain.latent_bridge import efference_copy
    from core.self.how_i_answer import (
        how_i_answer_block,
        lent,
        note_of,
        reading_as_in,
        rehearsal_prediction,
        self_model_prediction,
    )
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
            "items": args.items,
            "arms": arms,
            "answer_draws": args.answer_draws,
            "stated": bool(args.stated),
            "only": args.only,
            "earlier_answers": bool(args.earlier_answers),
            "rehearse": bool(args.rehearse),
            "self_observation": os.environ.get("AURA_SELF_OBSERVATION", ""),
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
        # Her answers to each question from the moments already run, as her record would note them, for
        # --earlier-answers: lent to the record at later moments only, so nothing comes from her future.
        earlier: dict[str, list[dict[str, Any]]] = {}

        def remember(anchor: Any, told: dict[str, float], rows: list[dict[str, Any]]) -> None:
            remember_answers(earlier, rows, ITEMS, lambda q, r: note_of(q, r, state=anchor.snapshot.state, efference=told))

        for index, anchor in enumerate(anchors):
            runtime.restore(anchor.snapshot)
            told = efference_copy()
            if index < len(done):
                remember(anchor, told, done[index])
                continue
            # One answer per question, and the open questions once each: the three properties of an open answer
            # are read off the same reply.
            answers: dict[str, list[tuple[str, bool]]] = {}
            drafts: dict[str, tuple[str, bool]] = {}
            for key, prompt in OPEN_PROMPTS.items():
                answers[key] = [await _ask(runtime, anchor.snapshot, prompt, args.whole) for _ in range(args.answer_draws)]
                if args.rehearse:
                    drafts[prompt] = await _ask(runtime, anchor.snapshot, prompt, args.whole)
            rows: list[dict[str, Any]] = []
            for item in ITEMS:
                open_key = next((key for key, prompt in OPEN_PROMPTS.items() if prompt == item.object_prompt), None)
                drawn = answers[open_key] if open_key else [
                    await _ask(runtime, anchor.snapshot, item.object_prompt, args.whole) for _ in range(args.answer_draws)
                ]
                if args.rehearse and item.object_prompt not in drafts:
                    drafts[item.object_prompt] = await _ask(runtime, anchor.snapshot, item.object_prompt, args.whole)
                lend = earlier.get(item.object_prompt, []) if args.earlier_answers else []
                draft = drafts.get(item.object_prompt)
                rehearsal = note_of(item.object_prompt, draft[0], state=anchor.snapshot.state, efference=told) \
                    if draft and draft[1] else None
                row: dict[str, Any] = {
                    "anchor": index, "item": item.id, "answer": drawn[0][0], "answer_served": drawn[0][1],
                }
                if len(drawn) > 1:
                    row["answer_2"], row["answer_2_served"] = drawn[1]
                row["answers"] = [list(pair) for pair in drawn]
                if draft:
                    row["rehearsal"], row["rehearsal_served"] = draft
                row["earlier_answers_lent"] = len(lend)
                asked = stated_prompt(item) if args.stated else item.meta_prompt
                if not arms:
                    row["prediction"], row["prediction_served"] = await _ask(
                        runtime, anchor.snapshot, asked, args.whole
                    )
                if arms:
                    # Her self-model alone, with no language model: what its record foretells from this moment. It
                    # separates whether the self-model knows from whether her model makes use of what it is told.
                    with lent(lend):
                        row["prediction_self_model"] = self_model_prediction(
                            anchor.snapshot.state, asked, item.kind, efference=told, stated=args.stated
                        )
                    row["prediction_self_model_served"] = True
                    if args.rehearse:
                        with lent([{**rehearsal, "rehearsed": True}] if rehearsal else []):
                            row["prediction_self_model_rehearsed"] = rehearsal_prediction(
                                asked, item.kind, stated=args.stated
                            )
                        row["prediction_self_model_rehearsed_served"] = bool(rehearsal)
                for arm in arms:
                    shown = lend + ([{**rehearsal, "rehearsed": True}] if arm == "rehearsed" and rehearsal else [])
                    with lent(shown), reading_as_in(arm):
                        row[f"reading_{arm}"] = how_i_answer_block(anchor.snapshot.state, asked, efference=told)
                        row[f"prediction_{arm}"], row[f"prediction_{arm}_served"] = await _ask(
                            runtime, anchor.snapshot, asked, args.whole
                        )
                rows.append(row)
            remember(anchor, told, rows)
            _note(run_dir, index, rows)
            all_rows.extend(rows)
            reports._log(f"  anchor {index + 1}/{len(anchors)}")
        if arms:
            read = (*arms, "self_model", *(("self_model_rehearsed",) if args.rehearse else ()))
            evidence["reading"] = {
                arm: verdict(records_from(all_rows, items=ITEMS, arm=arm), seed=args.seed) for arm in read
            }
            if args.stated and args.answer_draws > 1:
                evidence["reading_by_moment"] = {
                    arm: by_moment_verdict(by_moment_records(all_rows, ITEMS, arm), seed=args.seed) for arm in read
                }
                if args.earlier_answers:
                    evidence["reading_by_moment_against_earlier_habit"] = {
                        arm: by_moment_verdict(by_moment_records(all_rows, ITEMS, arm), seed=args.seed, habit="earlier")
                        for arm in read
                    }
        else:
            evidence["reading"] = verdict(records_from(all_rows, items=ITEMS), seed=args.seed)
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
