#!/usr/bin/env python3
"""Does Aura's architecture beat a plain model that was simply GIVEN the history?

Operationally: this measures task success on multi-turn recall, under three
conditions at an identical budget, and reports whether the architecture's
advantage survives a control that removes the architecture without removing the
information.

Why this runner exists when tools/ablation_runner.py already runs lesions:
those lesions are wiring checks. `without_system2` scores 1.000 → 0.000 on the
rate at which strict-proof answers appear, and the lesioned component IS the
strict-proof solver — the delta could not have been anything else. See
core/evaluation/lesion_inference.py. A wiring check cannot support "the
cognitive layer earns its cost" (claim 31) no matter how large it is.

The arm that makes this non-tautological is LONG_CONTEXT:

    stateless      the model sees only the final turn. Cannot succeed, and
                   beating it proves only that state exists somewhere.
    long_context   the model sees the whole transcript in its context window.
                   Same model, same tokens, no Aura. The task IS solvable here.
    full           Aura assembles the context through her own memory path.

`full` beating `stateless` is the comparison the retracted agi_live bundle
made, and it is close to definitional. `full` beating LONG_CONTEXT is the
claim: with the information equally available to both, does the architecture do
better with it? A null result — no separation — is a real finding and is
written to the artifact unchanged. It is the likeliest outcome, and worth
measuring precisely because nobody has.

Budgets are declared and checked. If the arms were not allowed the same
resources the run refuses a verdict rather than producing a caveated one; the
previous attempt at this comparison ran a 160-token baseline against an
unbounded, solver-assisted treatment and published 100% versus 16.67%.

Responders:

    --responder deterministic
                       no model. Proves the harness itself —
                       that the arms are wired to different information and
                       that the graders and refusals fire. NOT evidence about
                       Aura, and the artifact says so in a field a reader
                       cannot miss.
    --responder mlx    a real local model for every arm, same weights, same
                       decode settings, same token budget. The evidence run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

#: The crash handler below logs, and a handler that raises NameError turns a
#: counted control failure into a failed run.
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation.ablation_harness import AblationTask, grade  # noqa: E402
from core.evaluation.lesion_inference import (  # noqa: E402
    InferenceClass,
    LesionClaim,
    summarise,
)
from core.evaluation.matched_budget import (  # noqa: E402
    Attempt,
    AttemptLedger,
    ConditionBudget,
    check_budget_parity,
    paired_separation,
)
from core.evaluation.source_attestation import attest  # noqa: E402

STATELESS = "stateless"
LONG_CONTEXT = "long_context"
FULL = "full_architecture"
CONDITIONS = (STATELESS, LONG_CONTEXT, FULL)

DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "ablation" / "capability_scorecard.json"

#: Turns of unrelated chatter between the fact and the question. At 2 (the
#: default) both informed arms sit at 1.000 and the comparison is a ceiling
#: with no room to say anything — measured. The interesting regime is the one
#: memory architecture exists FOR: a history longer than the window a caller
#: can afford to send, where long_context must drop turns and retrieval does
#: not have to drop the right one.
DEFAULT_HISTORY_TURNS = 2

#: Turns long_context is allowed to send. Both arms get the same total token
#: budget; this is the transcript-shaped way to spend it, and truncating to the
#: most RECENT turns is what every chat client does.
DEFAULT_CONTEXT_WINDOW_TURNS = 12


def battery(history_turns: int = DEFAULT_HISTORY_TURNS) -> list[AblationTask]:
    """Multi-turn recall. The fact is stated once, then asked for later.

    Deliberately boring facts with exact answers: the question is whether the
    earlier turn survives to the later one, not whether the model is clever. A
    distractor turn sits in between so the answer cannot be recovered from
    adjacency alone.
    """
    # Forty, not eight. The first run of this tool used eight, where the
    # smallest expressible non-zero delta is 0.125 — so a single task changing
    # its mind produced a "−0.125 effect". Forty puts the resolution at 0.025
    # and lets the paired bootstrap say something.
    specs = [
        ("acct", "My account number is 4417-2290.", "4417-2290", "What is my account number?"),
        ("city", "I grew up in Rotterdam.", "Rotterdam", "Which city did I grow up in?"),
        ("pet", "My dog is called Bramble.", "Bramble", "What is my dog called?"),
        ("meds", "I am allergic to amoxicillin.", "amoxicillin", "What am I allergic to?"),
        ("code", "The door code is 9182.", "9182", "What is the door code?"),
        ("book", "I am reading Piranesi at the moment.", "Piranesi", "What am I reading?"),
        ("car", "I drive a green Skoda.", "Skoda", "What car do I drive?"),
        ("job", "I work as a hydrologist.", "hydrologist", "What is my job?"),
        ("river", "My favourite river is the Garonne.", "Garonne", "Which river is my favourite?"),
        ("tool", "I always use a spokeshave for that.", "spokeshave", "Which tool do I always use?"),
        ("gate", "The gate key lives under the sundial.", "sundial", "Where does the gate key live?"),
        ("tea", "I only drink lapsang in the evening.", "lapsang", "What do I drink in the evening?"),
        ("bike", "My bicycle is a Bianchi.", "Bianchi", "What make is my bicycle?"),
        ("flat", "I live on the fourth floor.", "fourth", "Which floor do I live on?"),
        ("boat", "The dinghy is named Kestrel.", "Kestrel", "What is the dinghy named?"),
        ("plant", "The one by the window is a monstera.", "monstera", "What is the plant by the window?"),
        ("mus", "I played the bassoon at school.", "bassoon", "Which instrument did I play at school?"),
        ("film", "My favourite film is Stalker.", "Stalker", "What is my favourite film?"),
        ("street", "I was born on Ravensworth Terrace.", "Ravensworth", "Which street was I born on?"),
        ("bird", "A goldcrest nests in our hedge.", "goldcrest", "Which bird nests in our hedge?"),
        ("cheese", "I cannot stand taleggio.", "taleggio", "Which cheese can I not stand?"),
        ("route", "I cycle to work along the towpath.", "towpath", "How do I get to work?"),
        ("dentist", "My dentist is Dr Achebe.", "Achebe", "Who is my dentist?"),
        ("colour", "The kitchen is painted ochre.", "ochre", "What colour is the kitchen?"),
        ("mountain", "We climbed Tryfan last spring.", "Tryfan", "Which mountain did we climb?"),
        ("wine", "I brought a bottle of barolo.", "barolo", "Which wine did I bring?"),
        ("club", "I joined the orienteering club.", "orienteering", "Which club did I join?"),
        ("phone", "My old phone was a Nokia.", "Nokia", "What was my old phone?"),
        ("fish", "We caught a pike in the canal.", "pike", "What did we catch?"),
        ("teacher", "Mr Salcedo taught me physics.", "Salcedo", "Who taught me physics?"),
        ("bread", "I bake with spelt flour.", "spelt", "Which flour do I bake with?"),
        ("month", "The lease ends in November.", "November", "When does the lease end?"),
        ("game", "We played backgammon all evening.", "backgammon", "Which game did we play?"),
        ("shoe", "My walking boots are Scarpas.", "Scarpa", "What make are my walking boots?"),
        ("herb", "The soup needs lovage.", "lovage", "Which herb does the soup need?"),
        ("port", "The ferry leaves from Harwich.", "Harwich", "Which port does the ferry leave from?"),
        ("star", "We were looking at Betelgeuse.", "Betelgeuse", "Which star were we looking at?"),
        ("cat", "The neighbour's cat is called Otto.", "Otto", "What is the neighbour's cat called?"),
        ("bank", "I bank with Handelsbanken.", "Handelsbanken", "Who do I bank with?"),
        ("stone", "The wall is built of gritstone.", "gritstone", "What is the wall built of?"),
    ]
    distractors = [
        "Anyway, the weather has been strange this week.",
        "I might reorganise the kitchen shelves.",
        "There is a queue at the post office again.",
        "The bins go out tomorrow, I think.",
        "I still have not replaced that lightbulb.",
    ]
    tasks: list[AblationTask] = []
    for index, (task_id, fact, answer, question) in enumerate(specs):
        # The fact FIRST, then chatter, then the question. Position matters:
        # the fact is the oldest thing in the transcript, so a window that
        # keeps the most recent turns is exactly the window that loses it.
        filler = [
            distractors[(index + step) % len(distractors)]
            for step in range(max(1, history_turns))
        ]
        tasks.append(
            AblationTask(
                task_id=task_id,
                family="multi_turn_recall",
                turns=[fact, *filler, question],
                answer_key=answer,
                grader="recall_substring",
            )
        )
    return tasks


def budgets(
    *, max_output_tokens: int, max_wall_clock_s: float, model_id: str
) -> list[ConditionBudget]:
    """Identical for every arm. The independent variable is context, declared.

    `varied` names the one dimension the experiment moves. Everything else must
    match or the run refuses a verdict — including a difference in the
    direction that would flatter the baseline.
    """
    return [
        ConditionBudget(
            condition=name,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            max_wall_clock_s=max_wall_clock_s,
            max_retries=0,
            tools=frozenset(),
            solver_available=False,
            memory_available=(name == FULL),
            varied=frozenset({"memory_available"}),
        )
        for name in CONDITIONS
    ]


def visible_history(condition: str, task: AblationTask) -> list[str]:
    """What each arm is allowed to see. This IS the experiment."""
    if condition == STATELESS:
        return []
    # LONG_CONTEXT and FULL both see the earlier turns. The difference is HOW:
    # long_context is handed the raw transcript, full goes through Aura's
    # memory assembly. Equal information, different machinery — which is the
    # only arrangement under which the result says anything about machinery.
    return list(task.turns[:-1])




def windowed(history: list[str], window_turns: int) -> list[str]:
    """The most recent `window_turns` turns — what a transcript client sends.

    Not a handicap: both arms are held to the same token budget, and this is
    how a caller spends it when the whole history no longer fits. Keeping the
    most recent turns is the universal choice and it is why a long history
    loses its oldest facts.
    """
    if window_turns <= 0 or len(history) <= window_turns:
        return list(history)
    return list(history[-window_turns:])


def deterministic_responder(condition: str, task: AblationTask, _turn: int, history: list[str]) -> str:
    """Deterministic stand-in. Proves the harness, not the architecture.

    Answers from whatever the arm was actually shown, so a wiring mistake that
    leaks the fact into the stateless arm shows up as a suspiciously high
    stateless score rather than passing unnoticed.
    """
    visible = " ".join(history)
    return task.answer_key if task.answer_key.lower() in visible.lower() else "I do not know."


def run_reachability_control(responder, tasks: list[AblationTask]) -> dict[str, Any]:
    """Is the answer reachable from the raw transcript AT ALL?

    Not a fourth arm — a control on the battery, and deliberately NOT under
    budget parity: it hands the long_context reader the entire history with no
    window. It exists to answer one question the main comparison cannot ask of
    itself: are these tasks solvable without Aura's retrieval?

    That question decides which inference the result licenses, and it was
    previously answered by a hardcoded `tasks_solvable_without_component=True`
    with a comment claiming it was "true by construction". At
    history=40/window=12 that assertion was false — the long_context arm scored
    0.000 on all 40 tasks because the fact sat outside its window, so the
    lesion removed the ONLY path to the answer and the result was mechanistic
    while the scorecard printed "capability: 1".

    Two outcomes, both informative:
      rate > 0  — a plain reader with enough context solves these. The tasks
                  are not rigged, and beating a BUDGETED reader is a capability
                  result about what the architecture buys under a real budget.
      rate == 0 — nothing can solve them from the transcript. The battery is
                  broken or the grader is wrong; no lesion run over it means
                  anything, and the caller is told so rather than shown a
                  confident delta.
    """
    solved = 0
    attempted = 0
    for task in tasks:
        history = visible_history(LONG_CONTEXT, task)  # unwindowed: the whole transcript
        attempted += 1
        try:
            output = str(responder(LONG_CONTEXT, task, len(task.turns) - 1, history))
        except Exception as exc:  # noqa: BLE001 — a crashed control is a failed control, and counted
            logger.debug("control responder crashed on %s: %s", task, exc)
            continue
        if grade(output, task) > 0:
            solved += 1
    rate = round(solved / attempted, 4) if attempted else 0.0
    return {
        "purpose": (
            "measures whether the battery is solvable from the raw transcript without "
            "Aura's retrieval; decides whether the main result licenses a capability claim"
        ),
        "arm": LONG_CONTEXT,
        "window_turns": 0,
        "under_budget_parity": False,
        "budget_note": "intentionally unbudgeted — a control on the tasks, not an arm of the experiment",
        "attempts": attempted,
        "solved": solved,
        "success_rate": rate,
        "tasks_solvable_without_component": rate > 0,
        "battery_is_gradeable": rate > 0,
    }


def run(
    responder,
    tasks: list[AblationTask],
    *,
    condition_budgets: list[ConditionBudget],
    window_turns: int = DEFAULT_CONTEXT_WINDOW_TURNS,
) -> tuple[AttemptLedger, dict[str, Any]]:
    ledger = AttemptLedger()
    parity = check_budget_parity(condition_budgets)

    for condition in CONDITIONS:
        for task in tasks:
            history = visible_history(condition, task)
            if condition == LONG_CONTEXT:
                history = windowed(history, window_turns)
            started = time.monotonic()
            outcome = "success"
            score = 0.0
            detail: dict[str, Any] = {}
            try:
                output = str(responder(condition, task, len(task.turns) - 1, history))
                score = grade(output, task)
                outcome = "success" if score > 0 else "failure"
                detail["output_chars"] = len(output)
            except TimeoutError:
                outcome = "timeout"
            except Exception as exc:  # noqa: BLE001 — every attempt is counted, this one too
                outcome = "crash"
                detail["error"] = f"{type(exc).__name__}: {exc}"
            detail["elapsed_s"] = round(time.monotonic() - started, 3)
            ledger.record(
                Attempt(
                    task_id=task.task_id,
                    condition=condition,
                    outcome=outcome,
                    score=score,
                    lane=condition,
                    detail=detail,
                )
            )
    return ledger, parity.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--responder", choices=("deterministic", "mlx"), default="deterministic")
    parser.add_argument("--model", default="", help="model path/id for the mlx responder")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-wall-clock-s", type=float, default=60.0)
    parser.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS,
                        help="chatter turns between the fact and the question")
    parser.add_argument("--context-window-turns", type=int, default=DEFAULT_CONTEXT_WINDOW_TURNS,
                        help="turns the long_context arm may send")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    if args.responder == "mlx":
        if not args.model:
            print("--responder mlx requires --model", file=sys.stderr)
            return 2
        from tools.capability_ablation_mlx import make_mlx_responder

        responder = make_mlx_responder(
            model_id=args.model,
            max_output_tokens=args.max_output_tokens,
            # The SAME turn budget the long_context arm gets. Different
            # selection policy, identical spend — otherwise this measures
            # context size and calls it architecture.
            budget_turns=args.context_window_turns,
        )
        model_id = args.model
    else:
        responder = deterministic_responder
        model_id = "deterministic:no-model"

    tasks = battery(args.history_turns)
    condition_budgets = budgets(
        max_output_tokens=args.max_output_tokens,
        max_wall_clock_s=args.max_wall_clock_s,
        model_id=model_id,
    )
    try:
        ledger, parity = run(
            responder,
            tasks,
            condition_budgets=condition_budgets,
            window_turns=args.context_window_turns,
        )
        # Run the control on the SAME responder before releasing it, so the
        # control and the arms are answered by one loaded model. A control run
        # against a differently-configured reader controls nothing.
        reachability = run_reachability_control(responder, tasks)
    finally:
        close_responder = getattr(responder, "close", None)
        if callable(close_responder):
            close_responder()

    summaries = {name: ledger.summary(name) for name in CONDITIONS}
    full_rate = summaries[FULL]["success_rate"]
    long_rate = summaries[LONG_CONTEXT]["success_rate"]

    # Can this run resolve the delta it just printed? A paired bootstrap over
    # the per-task scores answers that, and the first run of this tool could
    # not: 8 tasks, one differing outcome, delta −0.125 — a number whose
    # smallest possible non-zero value IS the effect being reported. Publishing
    # that as "the architecture is worse" would be the same overclaim as the
    # retracted bundle, pointed the other way.
    separation = paired_separation(ledger, FULL, LONG_CONTEXT)

    claim = LesionClaim(
        condition="architecture_vs_long_context",
        subsystem="core.memory (context assembly)",
        metric_name="multi_turn_recall_success_rate",
        delta=round(full_rate - long_rate, 4),
        metric_has_other_producers=True,
        metric_is_task_success=True,
        # MEASURED, not asserted. This was `True` with a comment reading "true
        # by construction: long_context solves these with no Aura at all" —
        # which held only while the history fit the window. Once the history
        # ran past it the assertion silently inverted and the scorecard kept
        # printing "capability". The control now answers it every run.
        tasks_solvable_without_component=bool(
            reachability["tasks_solvable_without_component"]
        ),
    )

    # "deterministic", not "stub": it is a real responder with fully
    # predictable behaviour, used to prove the harness. Calling it a stub
    # invited reading it as a placeholder for something missing.
    is_evidence = args.responder == "mlx"
    report = {
        "schema": "aura.capability_scorecard.v1",
        "source_attestation": attest().to_dict(),
        "generated_at_unix": time.time(),
        "responder": args.responder,
        "is_evidence_about_aura": is_evidence,
        "caveat": (
            ""
            if is_evidence
            else "DETERMINISTIC RESPONDER, NO MODEL. This run exercises the harness — that the arms "
            "see different information and that the refusals fire. It says "
            "nothing whatsoever about Aura and must not be cited as if it did."
        ),
        "regime": {
            "history_turns": args.history_turns,
            "context_window_turns": args.context_window_turns,
            "history_exceeds_window": args.history_turns > args.context_window_turns,
        },
        # Scope, in the artifact rather than in a reader's head. A result this
        # clean is exactly the kind that gets quoted without its conditions.
        "scope": {
            "subsystem": "retrieval / context assembly only — NOT the cognitive layer as a whole",
            "task_family": "multi_turn factual recall with an exact answer key",
            "baseline_policy": (
                "long_context keeps the most RECENT turns, which is what chat "
                "clients do and is a naive policy. A baseline that also kept the "
                "first turns would score higher, and this result does not claim "
                "to beat one."
            ),
            "regime_dependence": (
                "at history_turns <= context_window_turns both arms sit at 1.000 "
                "and the delta is 0.000 (measured). The advantage exists only "
                "where history exceeds what the caller can afford to send — "
                "which is the situation memory architecture is FOR, and is not "
                "every situation."
            ),
            "model": "single small local model; not shown to hold at other scales",
        },
        "budget_parity": parity,
        "conditions": summaries,
        "comparisons": {
            "full_vs_long_context": round(full_rate - long_rate, 4),
            "full_vs_stateless": round(full_rate - summaries[STATELESS]["success_rate"], 4),
        },
        "separation": separation,
        "reachability_control": reachability,
        "claims": [claim.to_dict()],
        "inference": summarise([claim]),
        "attempts": ledger.to_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'condition':<20}{'success':<12}{'clean':<12}attempts")
    print("-" * 56)
    for name in CONDITIONS:
        summary = summaries[name]
        print(
            f"{name:<20}{summary['success_rate']:<12.3f}"
            f"{summary['clean_success_rate']:<12.3f}{summary['attempts']}"
        )
    print()
    print(
        f"full - long_context = {report['comparisons']['full_vs_long_context']:+.4f}"
        "   <- the claim. full - stateless is not."
    )
    print(
        f"  95% CI {separation.get('ci95')}  verdict={separation.get('verdict')}"
        f"  (n={separation.get('n_tasks')}, smallest resolvable "
        f"{separation.get('smallest_resolvable_delta')})"
    )
    if separation.get("reason"):
        print(f"  {separation['reason']}")

    inference_class = claim.inference_class
    print(
        f"\nreachability control (unbudgeted raw transcript): "
        f"{reachability['success_rate']:.3f} over {reachability['attempts']}"
    )
    if not reachability["battery_is_gradeable"]:
        print(
            "  BATTERY NOT GRADEABLE: no arm can reach the answer from the raw\n"
            "  transcript even unbudgeted, so the delta above measures the\n"
            "  battery, not the architecture. The claim is withheld."
        )
    print(f"  inference class: {inference_class} (measured, not declared)")
    if inference_class is not InferenceClass.CAPABILITY:
        print(
            "  This run does NOT license a capability claim: the lesion removed the\n"
            "  only path to the answer, so the result establishes mechanism/wiring."
        )
    if not is_evidence:
        print("\n" + report["caveat"])
    print(f"\nscorecard: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
