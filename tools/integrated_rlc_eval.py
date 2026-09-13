"""Integrated RLC evaluation: does slot ingress convert organ content into
verified answers?

Every capability number before Jul 20 was measured with zero organs engaged
on tasks where retrieval could not help. This harness measures the actual
bet: paired arms on retrieval-DEPENDENT tasks —

    context-on   episode receives the facts as typed cognitive-context
                 items (the live organ wire format; distractors included)
    context-off  identical episode, distractors only

The answer codes exist ONLY in the context items, so the context-off arm is
structurally capped near guessing (1 in ~467,000 per code): if it scores
above chance, the harness is leaking and the run flags itself instead of
reporting capability. The context-on arm's per-episode receipts must show
the seeded cognitive slots (ingress FIRED, not merely existed) or the run
voids itself — the mechanism-present-without-firing hazard, tested at run
time.

Operator-launched, bounded, small checkpoints only beside a resident 32B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INTEGRATED_EVAL_SCHEMA = "aura.integrated_rlc_eval.v1"


def _build_engine(model, tokenizer, *, n_slots: int, max_steps: int):
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    config = CortexConfig(
        workspace=WorkspaceConfig(n_slots=n_slots, seed=11),
        recurrence=RecurrenceConfig(max_steps=max_steps, min_steps=1),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=48,
        decode_temperature=0.0,
        telemetry_enabled=False,
    )
    return LatentCortexEngine(model, tokenizer, config)


def _tally(outcomes: list[str]) -> dict[str, Any]:
    total = len(outcomes)
    correct = sum(1 for o in outcomes if o == "correct")
    lenient = sum(1 for o in outcomes if o == "correct_lenient")
    return {
        "n": total,
        "correct": correct,
        "correct_lenient": lenient,
        "incorrect": sum(1 for o in outcomes if o == "incorrect"),
        "incorrect_lenient": sum(1 for o in outcomes if o == "incorrect_lenient"),
        "unparseable": sum(1 for o in outcomes if o == "unparseable"),
        "accuracy": (correct / total) if total else 0.0,
        "reasoning_accuracy": ((correct + lenient) / total) if total else 0.0,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    # The clock starts here, not at import.
    #
    # `--max-seconds` used to be measured from a module-level `_START`, which
    # is set when the module is first imported. Run as a script that is the
    # same instant and the bound means what it says. Imported into a process
    # that goes on doing other things — a test chunk that loaded this module
    # fifteen minutes before it called it — the budget was already spent
    # before the first task, and the run raised TimeoutError having taken 29
    # seconds.
    started = time.time()

    from mlx_lm import load

    from core.brain.llm.latent_cortex.types import ComputeBudget
    from core.learning.integrated_eval_tasks import (
        context_for_arm,
        generate_tasks,
        grade,
    )
    from core.runtime.model_lane_control import standalone_model_lane

    tasks = generate_tasks(count=args.count, seed=args.seed, hops=args.hops)
    model_path = str(Path(args.model).expanduser().resolve())
    with standalone_model_lane(
        owner_id=f"integrated-rlc-eval:{Path(args.out).name}",
        model_path=model_path,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "integrated_rlc_eval", "operator_launched": True},
    ):
        model, tokenizer = load(model_path)
        engine = _build_engine(
            model, tokenizer, n_slots=args.n_slots, max_steps=args.max_steps
        )

        arms: dict[str, list[str]] = {"context_on": [], "context_off": []}
        ingress_failures = 0
        rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            for arm, with_facts in (("context_on", True), ("context_off", False)):
                budget = ComputeBudget(
                    max_layer_apps=args.budget_layer_apps, wall_clock_s=180.0
                )
                result = engine.reason(
                    messages=[{"role": "user", "content": task.prompt}],
                    budget=budget,
                    cognitive_context=context_for_arm(task, with_facts=with_facts),
                )
                outcome = grade(task, result.text or "")
                arms[arm].append(outcome)
                seeded_sources = [
                    row.get("source") for row in result.receipt.cognitive_slots
                ]
                if with_facts and "memory" not in seeded_sources:
                    # The facts were supplied and never became slots: the
                    # mechanism did not fire. One such episode voids the arm's
                    # claim to measure ingress.
                    ingress_failures += 1
                rows.append(
                    {
                        "task_id": task.task_id(),
                        "arm": arm,
                        "outcome": outcome,
                        "seeded_sources": seeded_sources,
                    }
                )
            if args.max_seconds and index >= 0 and time.time() - started > args.max_seconds:
                raise TimeoutError(
                    f"integrated eval exceeded --max-seconds {args.max_seconds}"
                )

    on = _tally(arms["context_on"])
    off = _tally(arms["context_off"])
    # Chance for these codes is ~1/467,000 per task; anything materially
    # above zero without context is leakage, not ability.
    leakage_suspected = off["reasoning_accuracy"] > max(
        0.02, 2.0 / max(1, off["n"])
    )
    return {
        "schema": INTEGRATED_EVAL_SCHEMA,
        "seed": args.seed,
        "hops": args.hops,
        "tasks": len(tasks),
        "context_on": on,
        "context_off": off,
        "integration_delta": round(
            on["reasoning_accuracy"] - off["reasoning_accuracy"], 4
        ),
        "ingress_failures": ingress_failures,
        "ingress_fired_everywhere": ingress_failures == 0,
        "leakage_suspected": leakage_suspected,
        "valid": ingress_failures == 0 and not leakage_suspected,
        "rows": rows,
        "finished_at": time.time(),
    }



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--budget-layer-apps", type=int, default=2_000_000)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        receipt = run_eval(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: every failure becomes exit code + stderr
        print(f"integrated_rlc_eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(receipt, indent=1, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).expanduser().write_text(rendered + "\n")
    return 0 if receipt["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
