#!/usr/bin/env python3
"""Run G04's preregistered transfer comparison, exactly as the plan froze it.

Refuses to start unless every frozen identity still holds: the candidate,
incumbent, model, generator, scorer and runtime files hash to what the plan
committed, and the feature bundle holds exactly the committed tasks. Then each
task is decoded by each arm in the plan's counterbalanced order, one row
written once per task and arm, so an interrupted run resumes on the same
tasks without replacement.

Grading reads neither model. The decoded program and the reference program
are compared by meaning (``compare_program_meanings`` over the public inputs
and counterfactual probes); anything short of "equivalent" counts against the
arm, a refusal or a runtime failure included. Every decode runs with program
execution made unavailable, so a program that came back was chosen without
executing anything; execution happens only afterwards, for the answer.

Usage:
    run_g04_transfer.py --protocol DIR --plan FILE --features DIR --output DIR
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ExecutionDuringDecode(RuntimeError):
    """Raised if anything tries to execute a program while one is being chosen."""


@contextlib.contextmanager
def execution_unavailable() -> Iterator[None]:
    import core.learning.semantic_program_execution as execution
    import core.learning.semantic_program_floor as floor
    from core.learning.procedure_induction import Program

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ExecutionDuringDecode("a program was executed while the decoder chose one")

    with (mock.patch.object(execution, "execute_semantic_program", refuse),
          mock.patch.object(floor, "execute_semantic_floor_program", refuse),
          mock.patch.object(Program, "run", refuse)):
        yield


def grade(item: Any, outcome: Any) -> dict[str, Any]:
    """The decoded program against the reference, by meaning; no model is read."""
    from core.learning.semantic_graph_counterexamples import (
        compare_program_meanings,
        counterfactual_inputs,
    )
    from core.learning.semantic_program_execution import execute_semantic_program

    reference = item.ir.to_program()
    expected = reference.run(item.public_inputs)
    if outcome is None or outcome.ir is None:
        return {"equivalent": False, "meaning": "no_program", "answer_correct": False}
    decoded = outcome.ir.to_program()
    try:
        meaning = compare_program_meanings(
            reference, decoded, counterfactual_inputs(item.public_inputs)
        )["status"]
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        meaning = f"comparison_failed:{type(exc).__name__}"
    try:
        answer = execute_semantic_program(outcome.ir, item.public_inputs).result == expected
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        answer = False
    return {"equivalent": meaning == "equivalent", "meaning": meaning, "answer_correct": answer,
            "decoded_program_sha256": decoded.sha()}


def decode(model: Any, item: Any) -> tuple[Any, str]:
    try:
        with execution_unavailable():
            outcome = model.decode(
                source_token_ids=item.ir.source_token_ids,
                hidden_states=item.hidden_states,
                public_inputs=item.public_inputs,
                source_text_sha256=item.ir.source_text_sha256,
                model_basis_sha256=item.ir.model_basis_receipt_sha256,
                search_time_limit_s=20.0,
            )
        return outcome, ""
    except ExecutionDuringDecode as exc:
        return None, f"execution_during_decode:{exc}"
    except Exception as exc:  # noqa: BLE001 - a runtime failure counts against the arm
        return None, f"runtime_failure:{type(exc).__name__}:{exc}"


def mcnemar(rows: Mapping[str, Mapping[str, Any]], treatment: str, control: str) -> dict:
    from scipy.stats import binomtest

    wins = sum(row[treatment]["equivalent"] and not row[control]["equivalent"] for row in rows.values())
    losses = sum(row[control]["equivalent"] and not row[treatment]["equivalent"] for row in rows.values())
    discordant = wins + losses
    p_value = float(binomtest(wins, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    return {"tasks": len(rows), "treatment_wins": wins, "control_wins": losses,
            "treatment_correct": sum(row[treatment]["equivalent"] for row in rows.values()),
            "control_correct": sum(row[control]["equivalent"] for row in rows.values()),
            "exact_one_sided_p": p_value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="the protocol output directory")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.learning.semantic_operation_peaks import peak_recognition_transducer_from_dict
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict as restore,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from tools import g04_transfer_protocol as protocol
    from tools.run_semantic_peak_recognition import _write_once

    spec = json.loads((args.protocol / "spec.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = json.loads((args.protocol / "plan_summary.json").read_text(encoding="utf-8"))
    if Path(summary["plan_path"]).name != args.plan.name:
        raise SystemExit("the plan given is not the one the protocol published")
    frozen = spec["artifacts"]
    observed = {
        "candidate": protocol._files_sha256([protocol.CANDIDATE]),
        "incumbent": protocol._files_sha256([protocol.INCUMBENT]),
        "resident_model": protocol._files_sha256(
            sorted(path for path in protocol.MODEL.iterdir() if path.is_file())),
        "task_generator": protocol._files_sha256([ROOT / p for p in protocol.GENERATOR_FILES]),
        "scorer": protocol._files_sha256([ROOT / p for p in protocol.SCORER_FILES]),
        "runtime": protocol._files_sha256([ROOT / p for p in protocol.RUNTIME_FILES]),
    }
    drift = sorted(name for name in frozen if frozen[name] != observed[name])
    if drift:
        raise SystemExit(f"frozen artifacts changed since the plan: {drift}")
    order = plan["parameters"]["arm_order_by_task"]

    incumbent = restore(json.loads(protocol.INCUMBENT.read_text(encoding="utf-8")))
    candidate = peak_recognition_transducer_from_dict(
        json.loads(protocol.CANDIDATE.read_text(encoding="utf-8")), restore_base=restore)
    if candidate.base.receipt_sha256 != incumbent.receipt_sha256:
        raise SystemExit("the candidate is not built on the incumbent the plan compares it with")
    arms = {"incumbent": incumbent, "candidate": candidate}

    bundle = load_standard_semantic_feature_bundle(args.features.expanduser())
    items = training_examples_from_feature_bundle(bundle, required_splits=frozenset({"test"}))
    items = tuple(replace(item, ir=replace(item.ir, model_basis_receipt_sha256=incumbent.model_basis_sha256))
                  for item in items)
    by_source = {item.ir.source_text_sha256: item for item in items}
    committed = {task["source_sha256"]: (stratum, task["id"])
                 for stratum, tasks in spec["domains"].items() for task in tasks}
    if set(by_source) != set(committed):
        raise SystemExit("the feature bundle does not hold exactly the committed tasks")

    output = args.output.expanduser()
    results: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in spec["domains"]}
    started = time.time()
    for count, (source, (stratum, task_id)) in enumerate(sorted(committed.items(), key=lambda kv: kv[1]), 1):
        item = by_source[source]
        row: dict[str, Any] = {}
        for arm in order[task_id]:
            path = output / "rows" / stratum / arm / f"{source}.json"
            if path.exists():
                row[arm] = json.loads(path.read_text(encoding="utf-8"))
                continue
            began = time.monotonic()
            outcome, failure = decode(arms[arm], item)
            graded = grade(item, outcome)
            row[arm] = _write_once(path, {
                "task_id": task_id, "stratum": stratum, "arm": arm, "source_sha256": source,
                "construction_id": item.construction_id,
                "refusal": getattr(outcome, "refusal", "") or "",
                "failure": failure, "decoded_without_execution": not failure.startswith("execution"),
                "seconds": round(time.monotonic() - began, 3), **graded,
            })
        results[stratum][task_id] = row
        if count % 32 == 0:
            print(f"  {count} of {len(committed)} tasks, {time.time() - started:.0f}s", flush=True)

    alpha = plan["parameters"]["per_comparison_alpha"]
    report = {"plan_hash": plan.get("plan_hash") or summary["plan_hash"], "per_comparison_alpha": alpha,
              "strata": {}}
    for stratum, rows in results.items():
        test = mcnemar(rows, "candidate", "incumbent")
        # A Python bool: numpy's would not serialise, which lost the report of
        # the first G04 run after every row was written (2026-10-06).
        test["rejects"] = bool(test["exact_one_sided_p"] <= alpha)
        test["answers_correct"] = {arm: sum(row[arm]["answer_correct"] for row in rows.values())
                                   for arm in arms}
        test["decoded_without_execution"] = {
            arm: sum(row[arm]["decoded_without_execution"] for row in rows.values()) for arm in arms}
        report["strata"][stratum] = test
    report["all_declared_comparisons_reject"] = all(s["rejects"] for s in report["strata"].values())
    _write_once(output / "report.json", report)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
