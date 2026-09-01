#!/usr/bin/env python3
"""Measure fresh typed-operation composition through the learned semantic tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.semantic_neural_composition import (  # noqa: E402
    PUBLIC_TYPED_WORKFLOW_SCHEMA,
    execute_public_typed_workflow,
    render_public_typed_workflow,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

SCHEMA = "aura.rlc.semantic_neural_composition_canary.v1"
DEFAULT_SEED = 2026083101
DEFAULT_TASKS = 96
SOURCE_FILES = (
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/learning/semantic_neural_composition.py",
    "core/learning/semantic_neural_machine.py",
    "tools/run_semantic_neural_composition_canary.py",
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


def _task_document(rng: random.Random) -> dict[str, Any]:
    left = rng.randint(3, 20)
    right = rng.randint(2, 20)
    factor = rng.randint(2, 6)
    divisor = rng.randint(2, 9)
    combined = left * factor + right
    remainder = combined % divisor
    quotient = (combined - remainder) // divisor
    denominator = quotient + rng.randint(1, max(2, quotient + 1))
    euclid_left = rng.randint(20, 120)
    euclid_right = rng.randint(2, euclid_left - 1)
    return {
        "schema": PUBLIC_TYPED_WORKFLOW_SCHEMA,
        "initial": {"r0": left, "r1": right, "r2": divisor, "r3": 0},
        "steps": [
            {"op": "copy", "dst": "r3", "src": "r0"},
            {"op": "mul", "dst": "r3", "factor": factor},
            {"op": "add", "dst": "r3", "left": "r3", "right": "r1"},
            {"op": "sub", "dst": "r3", "amount": remainder},
            {"op": "div_exact", "dst": "r0", "numerator": "r3", "denominator": "r2"},
            {"op": "set", "dst": "r1", "value": denominator},
            {"op": "ratio_choice", "dst": "s0", "numerator": "r0", "denominator": "r1"},
            {"op": "ratio_band", "dst": "s0", "numerator": "r0", "denominator": "r1"},
            {"op": "set", "dst": "r3", "value": euclid_left},
            {"op": "set", "dst": "r2", "value": euclid_right},
            *(
                {"op": "euclid_step", "left": "r3", "right": "r2"}
                for _step in range(12)
            ),
        ],
        "report": ["r0", "r1", "r2", "r3", "s0"],
    }


def _reference(document: dict[str, Any]) -> dict[str, int]:
    registers = dict(document["initial"])
    scalar = 0
    for step in document["steps"]:
        operation = step["op"]
        if operation == "set":
            registers[step["dst"]] = step["value"]
        elif operation == "copy":
            registers[step["dst"]] = registers[step["src"]]
        elif operation == "mul":
            registers[step["dst"]] *= step["factor"]
        elif operation == "add":
            registers[step["dst"]] = registers[step["left"]] + registers[step["right"]]
        elif operation == "sub":
            registers[step["dst"]] -= step["amount"]
        elif operation == "div_exact":
            numerator = registers[step["numerator"]]
            denominator = registers[step["denominator"]]
            if denominator < 1 or numerator % denominator:
                raise ValueError("reference division is not exact")
            registers[step["dst"]] = numerator // denominator
        elif operation in {"ratio_choice", "ratio_band"}:
            numerator = registers[step["numerator"]]
            denominator = registers[step["denominator"]]
            if denominator < 1:
                raise ValueError("reference ratio denominator is invalid")
            scalar = (
                int(2 * numerator >= denominator) + 1
                if operation == "ratio_choice"
                else 1
                if 2 * numerator < denominator
                else 2
                if 10 * numerator < 7 * denominator
                else 3
                if 10 * numerator < 9 * denominator
                else 4
            )
        elif operation == "euclid_step":
            left = registers[step["left"]]
            right = registers[step["right"]]
            registers[step["left"]] = right if right else left
            registers[step["right"]] = left % right if right else 0
        else:  # pragma: no cover - generator emits the closed operation set.
            raise ValueError("reference operation is unsupported")
    values = {**registers, "s0": scalar}
    return {name: values[name] for name in document["report"]}


def _lesion(operation: int, coefficient: int) -> SemanticNeuralMachine:
    tissue = SemanticNeuralMachine().tissue
    original = tissue.raw_coefficients[operation, coefficient]
    tissue.raw_coefficients = tissue.raw_coefficients.at[operation, coefficient].add(-original)
    return SemanticNeuralMachine(tissue)


def _execute(prompt: str, machine: SemanticNeuralMachine) -> dict[str, Any] | None:
    try:
        return execute_public_typed_workflow(prompt, machine=machine).semantic_result
    except (RuntimeError, ValueError):
        return None


def _wrong_operand(document: dict[str, Any]) -> dict[str, Any]:
    altered = json.loads(json.dumps(document))
    for step in altered["steps"]:
        if step["op"] == "mul":
            step["factor"] += 1
            return altered
    raise RuntimeError("generated workflow has no mutable operand")


def run_canary(*, seed: int = DEFAULT_SEED, task_count: int = DEFAULT_TASKS) -> dict[str, Any]:
    if type(seed) is not int or type(task_count) is not int or task_count < 24:
        raise ValueError("composition canary configuration is invalid")
    rng = random.Random(seed)
    treatment = SemanticNeuralMachine()
    additive_lesion = _lesion(0, 1)
    multiplicative_lesion = _lesion(1, 2)
    counts = {
        "treatment_exact": 0,
        "additive_lesion_disrupted": 0,
        "multiplicative_lesion_disrupted": 0,
        "wrong_operand_disrupted": 0,
    }
    rows = []
    for index in range(task_count):
        document = _task_document(rng)
        prompt = render_public_typed_workflow(document)
        expected = _reference(document)
        observed = _execute(prompt, treatment)
        additive = _execute(prompt, additive_lesion)
        multiplicative = _execute(prompt, multiplicative_lesion)
        wrong_prompt = render_public_typed_workflow(_wrong_operand(document))
        wrong = _execute(wrong_prompt, treatment)
        exact = observed == expected
        add_disrupted = additive != expected
        mul_disrupted = multiplicative != expected
        wrong_disrupted = wrong != expected
        counts["treatment_exact"] += int(exact)
        counts["additive_lesion_disrupted"] += int(add_disrupted)
        counts["multiplicative_lesion_disrupted"] += int(mul_disrupted)
        counts["wrong_operand_disrupted"] += int(wrong_disrupted)
        rows.append(
            {
                "ordinal": index,
                "public_prompt": prompt,
                "public_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "expected_sha256": _sha(expected),
                "observed_sha256": "" if observed is None else _sha(observed),
                "treatment_exact": exact,
                "additive_lesion_disrupted": add_disrupted,
                "multiplicative_lesion_disrupted": mul_disrupted,
                "wrong_operand_disrupted": wrong_disrupted,
            }
        )
    passed = (
        counts["treatment_exact"] == task_count
        and counts["additive_lesion_disrupted"] >= int(0.9 * task_count)
        and counts["multiplicative_lesion_disrupted"] >= int(0.9 * task_count)
        and counts["wrong_operand_disrupted"] >= int(0.9 * task_count)
    )
    body = {
        "schema": SCHEMA,
        "passed": passed,
        "verdict": "SUPPORTED_OPERATION_COMPOSITION" if passed else "REFUTED_OR_INCONCLUSIVE",
        "seed": seed,
        "task_count": task_count,
        "counts": counts,
        "task_set_sha256": _sha(rows),
        "rows": rows,
        "tissue_sha256": treatment.tissue_sha256,
        "source_sha256s": {path: _file_sha256(REPO_ROOT / path) for path in SOURCE_FILES},
        "teacher_available_to_treatment": False,
        "private_trace_available_to_treatment": False,
        "verifier_answer_available_to_treatment": False,
        "claim_boundary": (
            "fresh family-neutral typed-operation recombination through existing learned "
            "arithmetic tissue; this does not establish natural-language transfer, "
            "open-domain reasoning gain, resident decoded-answer superiority, or broader serving"
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run_canary(seed=args.seed, task_count=args.tasks)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        atomic_write_text(args.out.expanduser().resolve(), encoded, power_safe=True)
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
