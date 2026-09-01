#!/usr/bin/env python3
"""Independently replay an induced neural procedure canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.procedure_induction import ProcedureInducer  # noqa: E402
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from tools.run_induced_neural_procedure_canary import (  # noqa: E402
    SCHEMA as INPUT_SCHEMA,
)
from tools.run_induced_neural_procedure_canary import (  # noqa: E402
    SOURCE_FILES,
    _null_found,
    _observed,
    _sha,
    task_set,
)

SCHEMA = "aura.rlc.induced_neural_procedure_verification.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lesion() -> SemanticNeuralMachine:
    tissue = SemanticNeuralMachine().tissue
    original = tissue.raw_coefficients[0, 1]
    tissue.raw_coefficients = tissue.raw_coefficients.at[0, 1].add(-original)
    return SemanticNeuralMachine(tissue)


def verify(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != INPUT_SCHEMA:
        raise ValueError("induced neural procedure report schema is invalid")
    receipt = report.get("receipt_sha256")
    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if receipt != _sha(body):
        raise ValueError("induced neural procedure receipt is invalid")
    expected_sources = {
        path: _file_sha256(REPO_ROOT / path) for path in SOURCE_FILES
    }
    if report.get("source_sha256s") != expected_sources:
        raise ValueError("induced neural procedure source identity is invalid")

    seed = report.get("seed")
    support_count = report.get("support_count")
    task_count = report.get("task_count")
    null_runs = report.get("null_runs")
    if not all(type(value) is int for value in (seed, support_count, task_count, null_runs)):
        raise ValueError("induced neural procedure dimensions are invalid")
    support = task_set(support_count, seed=seed + 500)
    expected_support = [
        {"inputs": list(task.inputs), "output": task.output} for task in support
    ]
    if report.get("support") != expected_support:
        raise ValueError("induced neural procedure support set is invalid")
    outcome = ProcedureInducer(max_depth=3).induce(support)
    if outcome.program is None or outcome.program.to_dict() != report.get("program"):
        raise ValueError("induced neural procedure identity is invalid")
    program = outcome.program
    depth_one = ProcedureInducer(max_depth=1).induce(support)
    null_found = _null_found(support, seed=seed, runs=null_runs)
    common_output = Counter(task.output for task in support).most_common(1)[0][0]

    rows = report.get("rows")
    fresh = task_set(task_count, seed=seed + 777)
    if not isinstance(rows, list) or len(rows) != len(fresh):
        raise ValueError("induced neural procedure rows are invalid")
    treatment = SemanticNeuralMachine()
    lesion = _lesion()
    counts = {
        "treatment_exact": 0,
        "coefficient_lesion_disrupted": 0,
        "wrong_input_disrupted": 0,
        "no_procedure_exact": 0,
    }
    replay_rows: list[dict[str, Any]] = []
    for ordinal, task in enumerate(fresh):
        expected = int(task.output)
        observed, workflow, lowering_receipt = _observed(program, task.inputs, treatment)
        lesion_observed, _unused, _unused_receipt = _observed(
            program, task.inputs, lesion
        )
        left, right, denominator = task.inputs
        wrong_inputs = (left + denominator, right, denominator)
        wrong_observed, _unused, _unused_receipt = _observed(
            program, wrong_inputs, treatment
        )
        replay = {
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
            "treatment_exact": observed == expected,
            "coefficient_lesion_disrupted": lesion_observed != expected,
            "wrong_input_disrupted": wrong_observed != expected,
            "no_procedure_exact": common_output == expected,
        }
        if replay != rows[ordinal]:
            raise ValueError("induced neural procedure replay differs from producer")
        for key in counts:
            counts[key] += int(replay[key])
        replay_rows.append(replay)

    verified = bool(
        report.get("admitted") is True
        and report.get("verdict") == "SUPPORTED_INDUCED_NEURAL_PROCEDURE"
        and not depth_one.found
        and report.get("single_primitive_shortcut") is False
        and null_found == report.get("null_found") == 0
        and counts == report.get("counts")
        and counts["treatment_exact"] == task_count
        and counts["coefficient_lesion_disrupted"] >= int(0.9 * task_count)
        and counts["wrong_input_disrupted"] == task_count
        and counts["no_procedure_exact"] <= int(0.1 * task_count)
        and report.get("task_set_sha256") == _sha(replay_rows)
        and report.get("family_label_available_to_inducer") is False
        and report.get("family_solver_available_to_inducer") is False
        and report.get("support_outputs_available_to_inducer") is True
        and report.get("evaluation_outputs_available_to_treatment") is False
    )
    body = {
        "schema": SCHEMA,
        "verified": verified,
        "input_receipt_sha256": receipt,
        "program_sha": program.sha(),
        "task_count": task_count,
        "counts": counts,
        "null_found": null_found,
        "task_set_sha256": _sha(replay_rows),
        "producer_source_sha256s": expected_sources,
        "verifier_source_sha256": _file_sha256(Path(__file__).resolve()),
        "claim_boundary": report.get("claim_boundary"),
    }
    return {**body, "verification_receipt_sha256": _sha(body)}


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
