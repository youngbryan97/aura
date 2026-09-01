#!/usr/bin/env python3
"""Independently replay a typed-operation composition canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.semantic_neural_composition import (  # noqa: E402
    execute_public_typed_workflow,
    public_typed_workflow_document,
    render_public_typed_workflow,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

INPUT_SCHEMA = "aura.rlc.semantic_neural_composition_canary.v1"
SCHEMA = "aura.rlc.semantic_neural_composition_verification.v1"
PRODUCER_SOURCE_FILES = (
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
            registers[step["dst"]] = registers[step["dst"]] * step["factor"]
        elif operation == "add":
            registers[step["dst"]] = registers[step["left"]] + registers[step["right"]]
        elif operation == "sub":
            registers[step["dst"]] = registers[step["dst"]] - step["amount"]
        elif operation == "div_exact":
            numerator = registers[step["numerator"]]
            denominator = registers[step["denominator"]]
            if denominator < 1 or numerator % denominator:
                raise ValueError("verification reference division is not exact")
            registers[step["dst"]] = numerator // denominator
        elif operation == "ratio_choice":
            numerator = registers[step["numerator"]]
            denominator = registers[step["denominator"]]
            scalar = int(2 * numerator >= denominator) + 1
        elif operation == "ratio_band":
            numerator = registers[step["numerator"]]
            denominator = registers[step["denominator"]]
            scalar = (
                1
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
        else:
            raise ValueError("verification reference operation is unsupported")
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


def _wrong_prompt(document: dict[str, Any]) -> str:
    altered = json.loads(json.dumps(document))
    for step in altered["steps"]:
        if step["op"] == "mul":
            step["factor"] += 1
            return render_public_typed_workflow(altered)
    raise ValueError("verification workflow has no multiplicative operand")


def verify(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != INPUT_SCHEMA:
        raise ValueError("composition canary report schema is invalid")
    receipt = report.get("receipt_sha256")
    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if receipt != _sha(body):
        raise ValueError("composition canary receipt is invalid")
    expected_sources = {
        path: _file_sha256(REPO_ROOT / path) for path in PRODUCER_SOURCE_FILES
    }
    if report.get("source_sha256s") != expected_sources:
        raise ValueError("composition canary source identity is invalid")
    rows = report.get("rows")
    task_count = report.get("task_count")
    if (
        type(task_count) is not int
        or task_count < 24
        or not isinstance(rows, list)
        or len(rows) != task_count
    ):
        raise ValueError("composition canary row set is invalid")
    treatment = SemanticNeuralMachine()
    additive = _lesion(0, 1)
    multiplicative = _lesion(1, 2)
    counts = {
        "treatment_exact": 0,
        "additive_lesion_disrupted": 0,
        "multiplicative_lesion_disrupted": 0,
        "wrong_operand_disrupted": 0,
    }
    replay_rows = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("ordinal") != ordinal:
            raise ValueError("composition canary row order is invalid")
        prompt = row.get("public_prompt")
        if (
            not isinstance(prompt, str)
            or row.get("public_prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
        ):
            raise ValueError("composition canary prompt binding is invalid")
        document = public_typed_workflow_document(prompt)
        expected = _reference(document)
        observed = _execute(prompt, treatment)
        additive_observed = _execute(prompt, additive)
        multiplicative_observed = _execute(prompt, multiplicative)
        wrong_observed = _execute(_wrong_prompt(document), treatment)
        replay = {
            "ordinal": ordinal,
            "public_prompt": prompt,
            "public_prompt_sha256": row["public_prompt_sha256"],
            "expected_sha256": _sha(expected),
            "observed_sha256": "" if observed is None else _sha(observed),
            "treatment_exact": observed == expected,
            "additive_lesion_disrupted": additive_observed != expected,
            "multiplicative_lesion_disrupted": multiplicative_observed != expected,
            "wrong_operand_disrupted": wrong_observed != expected,
        }
        if replay != row:
            raise ValueError("composition canary replay differs from producer row")
        for key in counts:
            counts[key] += int(replay[key])
        replay_rows.append(replay)
    passed = (
        counts == report.get("counts")
        and report.get("task_set_sha256") == _sha(replay_rows)
        and counts["treatment_exact"] == task_count
        and counts["additive_lesion_disrupted"] >= int(0.9 * task_count)
        and counts["multiplicative_lesion_disrupted"] >= int(0.9 * task_count)
        and counts["wrong_operand_disrupted"] >= int(0.9 * task_count)
        and report.get("passed") is True
        and report.get("verdict") == "SUPPORTED_OPERATION_COMPOSITION"
    )
    result = {
        "schema": SCHEMA,
        "verified": passed,
        "input_receipt_sha256": receipt,
        "task_count": task_count,
        "counts": counts,
        "task_set_sha256": _sha(replay_rows),
        "producer_source_sha256s": expected_sources,
        "verifier_source_sha256": _file_sha256(Path(__file__).resolve()),
        "claim_boundary": report.get("claim_boundary"),
    }
    return {**result, "verification_receipt_sha256": _sha(result)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))
    verification = verify(report)
    encoded = json.dumps(verification, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        atomic_write_text(args.out.expanduser().resolve(), encoded, power_safe=True)
    print(encoded, end="")
    return 0 if verification["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
