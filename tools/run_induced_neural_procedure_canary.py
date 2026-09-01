#!/usr/bin/env python3
"""Test family-blind procedure acquisition through learned neural tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.induced_neural_procedure import execute_induced_program  # noqa: E402
from core.learning.procedure_induction import (  # noqa: E402
    PRIMITIVE_SET_SHA,
    ProcedureInducer,
    Program,
    TaskInstance,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

SCHEMA = "aura.rlc.induced_neural_procedure_canary.v1"
DEFAULT_SEED = 2026083102
DEFAULT_SUPPORT = 16
DEFAULT_TASKS = 96
DEFAULT_NULL_RUNS = 15
SOURCE_FILES = (
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/learning/induced_neural_procedure.py",
    "core/learning/procedure_induction.py",
    "core/learning/semantic_neural_composition.py",
    "core/learning/semantic_neural_machine.py",
    "tools/run_induced_neural_procedure_canary.py",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_task(rng: random.Random) -> TaskInstance:
    """Generate public inputs whose hidden procedure has an exact quotient."""

    denominator = rng.randint(2, 9)
    quotient = rng.randint(3, 40)
    left = rng.randint(1, quotient * denominator - 1)
    right = quotient * denominator - left
    return TaskInstance((left, right, denominator), quotient)


def task_set(count: int, *, seed: int) -> list[TaskInstance]:
    rng = random.Random(seed)
    return [generate_task(rng) for _index in range(count)]


def _lesion() -> SemanticNeuralMachine:
    tissue = SemanticNeuralMachine().tissue
    original = tissue.raw_coefficients[0, 1]
    tissue.raw_coefficients = tissue.raw_coefficients.at[0, 1].add(-original)
    return SemanticNeuralMachine(tissue)


def _observed(
    program: Program,
    inputs: tuple[Any, ...],
    machine: SemanticNeuralMachine,
) -> tuple[int | None, str, dict[str, Any] | None]:
    try:
        execution = execute_induced_program(program, inputs, machine=machine)
        register = execution.lowered.output_register
        value = execution.composition.semantic_result[register]
        return int(value), execution.lowered.public_workflow, execution.lowered.receipt
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None, "", None


def _null_found(
    support: list[TaskInstance],
    *,
    seed: int,
    runs: int,
) -> int:
    found = 0
    for run in range(runs):
        outputs = [task.output for task in support]
        random.Random(seed + 9000 + run).shuffle(outputs)
        permuted = [
            TaskInstance(task.inputs, output)
            for task, output in zip(support, outputs, strict=True)
        ]
        found += int(ProcedureInducer(max_depth=3).induce(permuted).found)
    return found


def run_canary(
    *,
    seed: int = DEFAULT_SEED,
    support_count: int = DEFAULT_SUPPORT,
    task_count: int = DEFAULT_TASKS,
    null_runs: int = DEFAULT_NULL_RUNS,
) -> dict[str, Any]:
    if (
        type(seed) is not int
        or type(support_count) is not int
        or support_count < 12
        or type(task_count) is not int
        or task_count < 24
        or type(null_runs) is not int
        or null_runs < 5
    ):
        raise ValueError("induced neural procedure canary configuration is invalid")

    support = task_set(support_count, seed=seed + 500)
    outcome = ProcedureInducer(max_depth=3).induce(support)
    if outcome.program is None:
        raise RuntimeError(f"procedure induction failed: {outcome.refusal}")
    program = outcome.program
    depth_one = ProcedureInducer(max_depth=1).induce(support)
    null_found = _null_found(support, seed=seed, runs=null_runs)
    common_output = Counter(task.output for task in support).most_common(1)[0][0]

    treatment = SemanticNeuralMachine()
    lesion = _lesion()
    counts = {
        "treatment_exact": 0,
        "coefficient_lesion_disrupted": 0,
        "wrong_input_disrupted": 0,
        "no_procedure_exact": 0,
    }
    rows: list[dict[str, Any]] = []
    fresh = task_set(task_count, seed=seed + 777)
    for ordinal, task in enumerate(fresh):
        expected = int(task.output)
        observed, workflow, lowering_receipt = _observed(
            program, task.inputs, treatment
        )
        lesion_observed, _lesion_workflow, _lesion_receipt = _observed(
            program, task.inputs, lesion
        )
        left, right, denominator = task.inputs
        wrong_inputs = (left + denominator, right, denominator)
        wrong_observed, _wrong_workflow, _wrong_receipt = _observed(
            program, wrong_inputs, treatment
        )
        exact = observed == expected
        lesion_disrupted = lesion_observed != expected
        wrong_disrupted = wrong_observed != expected
        no_procedure_exact = common_output == expected
        counts["treatment_exact"] += int(exact)
        counts["coefficient_lesion_disrupted"] += int(lesion_disrupted)
        counts["wrong_input_disrupted"] += int(wrong_disrupted)
        counts["no_procedure_exact"] += int(no_procedure_exact)
        rows.append(
            {
                "ordinal": ordinal,
                "inputs": list(task.inputs),
                "expected": expected,
                "public_workflow": workflow,
                "lowering_receipt_sha256": (
                    "" if lowering_receipt is None else lowering_receipt["receipt_sha256"]
                ),
                "observed": observed,
                "coefficient_lesion_observed": lesion_observed,
                "wrong_inputs": list(wrong_inputs),
                "wrong_input_observed": wrong_observed,
                "no_procedure_observed": common_output,
                "treatment_exact": exact,
                "coefficient_lesion_disrupted": lesion_disrupted,
                "wrong_input_disrupted": wrong_disrupted,
                "no_procedure_exact": no_procedure_exact,
            }
        )

    admitted = bool(
        program.depth >= 2
        and not depth_one.found
        and null_found == 0
        and counts["treatment_exact"] == task_count
        and counts["coefficient_lesion_disrupted"] >= int(0.9 * task_count)
        and counts["wrong_input_disrupted"] == task_count
        and counts["no_procedure_exact"] <= int(0.1 * task_count)
    )
    body = {
        "schema": SCHEMA,
        "admitted": admitted,
        "verdict": (
            "SUPPORTED_INDUCED_NEURAL_PROCEDURE"
            if admitted
            else "REFUTED_OR_INCONCLUSIVE"
        ),
        "seed": seed,
        "support_count": support_count,
        "task_count": task_count,
        "null_runs": null_runs,
        "null_found": null_found,
        "single_primitive_shortcut": depth_one.found,
        "program": program.to_dict(),
        "programs_considered": outcome.programs_considered,
        "primitive_set_sha": PRIMITIVE_SET_SHA,
        "support": [
            {"inputs": list(task.inputs), "output": task.output} for task in support
        ],
        "counts": counts,
        "rows": rows,
        "task_set_sha256": _sha(rows),
        "tissue_sha256": treatment.tissue_sha256,
        "source_sha256s": {
            path: _file_sha256(REPO_ROOT / path) for path in SOURCE_FILES
        },
        "family_label_available_to_inducer": False,
        "family_solver_available_to_inducer": False,
        "support_outputs_available_to_inducer": True,
        "evaluation_outputs_available_to_treatment": False,
        "claim_boundary": (
            "a family-blind procedure induced from support examples transfers on fresh "
            "inputs through learned arithmetic tissue under composition, coefficient, "
            "wrong-input, and no-procedure controls; not natural-language compilation, "
            "open-domain reasoning, resident decode, unrestricted serving, or frontier performance"
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--support", type=int, default=DEFAULT_SUPPORT)
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS)
    parser.add_argument("--null-runs", type=int, default=DEFAULT_NULL_RUNS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run_canary(
        seed=args.seed,
        support_count=args.support,
        task_count=args.tasks,
        null_runs=args.null_runs,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        atomic_write_text(args.out.expanduser().resolve(), encoded, power_safe=True)
    print(encoded, end="")
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
