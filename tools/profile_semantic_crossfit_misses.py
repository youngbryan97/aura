#!/usr/bin/env python3
"""Profile verified source-fold proposal misses before another broad RLC run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _load_verified(directory: Path) -> tuple[dict, list[dict]]:
    report = json.loads((directory / "report.json").read_bytes())
    plan = json.loads((directory / "plan.json").read_bytes())
    if (report.get("schema") != "aura.semantic_proposer_crossfit.v1"
            or plan.get("schema") != "aura.semantic_proposer_crossfit_plan.v1"
            or report.get("receipt_sha256") != _digest({
                key: value for key, value in report.items() if key != "receipt_sha256"})
            or plan.get("plan_sha256") != _digest({
                key: value for key, value in plan.items() if key != "plan_sha256"})
            or report.get("plan_sha256") != plan["plan_sha256"]
            or set(report.get("row_receipts", {})) != set(plan["held_ids"])):
        raise ValueError("crossfit report, plan, or held population is not verified")
    if (set(plan["held_ids"]) & (set(plan["fit_ids"]) | set(plan["calibration_ids"]))
            or set(plan["fit_ids"]) & set(plan["calibration_ids"])):
        raise ValueError("source-fold fit, calibration, and held sets overlap")
    rows = []
    for source, receipt in sorted(report["row_receipts"].items()):
        row = json.loads((directory / "rows" / f"{source}.json").read_bytes())
        if (row.get("source") != source or row.get("plan_sha256") != plan["plan_sha256"]
                or row.get("receipt_sha256") != receipt
                or receipt != _digest({key: value for key, value in row.items()
                                       if key != "receipt_sha256"})):
            raise ValueError(f"crossfit row does not match its receipt: {source}")
        rows.append(row)
    return plan, rows


def _operators(candidate: dict) -> tuple[str, ...]:
    return tuple(str(instruction[0]) for instruction in candidate["program"]["instructions"])


def classify_row(row: dict) -> dict[str, Any]:
    """Use only independently compared programs; incomplete reach stays unknown."""
    diagnosis = row["diagnosis"]
    candidates = row["bank"]["candidates"]
    status = {item["program_sha256"]: item["status"]
              for item in diagnosis["comparisons"]}
    identities = {item["program_sha256"] for item in candidates}
    if not identities <= set(status):
        raise ValueError("generated proposal lacks an independent comparison")
    correct = [item for item in candidates if status[item["program_sha256"]] == "equivalent"]
    selected = row["bank"]["selected_program_sha256"]
    selected_program = next((item for item in candidates
                             if item["program_sha256"] == selected), None)
    if diagnosis["correct_reachable"] is True and not correct:
        raise ValueError("reachable row has no verified equivalent proposal")
    if diagnosis["correct_reachable"] is None and correct:
        raise ValueError("unresolved row already has an equivalent proposal")
    if correct and status.get(selected) == "equivalent":
        stage = "correct_selection"
    elif correct and selected is None and row["bank"].get("selected_search_interrupted") is True:
        stage = "public_selection_interrupted"
    elif correct:
        stage = "selection_miss"
    elif diagnosis["correct_reachable"] is None and not row["bank"]["search_complete"]:
        stage = "reach_unobserved"
    elif diagnosis["correct_reachable"] is False and row["bank"]["search_complete"]:
        stage = "verified_unreachable_at_declared_search"
    else:
        raise ValueError("reach status conflicts with search completeness")
    mechanism = None
    if stage == "public_selection_interrupted":
        mechanism = "search_interrupted_after_correct_candidate"
    elif stage == "selection_miss":
        if selected_program is None:
            mechanism = "no_selected_candidate"
        else:
            selected_ops = Counter(_operators(selected_program))
            correct_ops = {tuple(sorted(Counter(_operators(item)).items()))
                           for item in correct}
            mechanism = ("same_operations_wrong_structure_or_arguments"
                         if tuple(sorted(selected_ops.items())) in correct_ops else
                         "operation_inventory_mismatch")
    return {"source": row["source"], "stage": stage, "mechanism": mechanism,
            "selected_program_sha256": selected,
            "correct_program_sha256s": sorted({item["program_sha256"] for item in correct}),
            "observed_correct_operation_signatures": sorted({
                "+".join(sorted(_operators(item))) for item in correct}),
            "search_complete": row["bank"]["search_complete"]}


def profile(directories: list[Path]) -> dict[str, Any]:
    if not directories:
        raise ValueError("at least one source-fold report is required")
    source_basis = None
    seen = set()
    rows = []
    folds = []
    for directory in directories:
        plan, observations = _load_verified(directory)
        if source_basis is None:
            source_basis = plan["source_report_sha256"]
        elif plan["source_report_sha256"] != source_basis:
            raise ValueError("source-fold reports use different source corpora")
        if seen & set(plan["held_ids"]):
            raise ValueError("held source appears in more than one report")
        seen.update(plan["held_ids"])
        folds.append(plan["fold"])
        rows.extend({"fold": plan["fold"], **classify_row(row)} for row in observations)
    if len(folds) != len(set(folds)):
        raise ValueError("source-fold report is repeated")
    by_stage = Counter(row["stage"] for row in rows)
    by_mechanism = Counter(row["mechanism"] for row in rows if row["mechanism"])
    by_signature: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for signature in row["observed_correct_operation_signatures"]:
            by_signature[signature][row["stage"]] += 1
    return {"schema": "aura.semantic_crossfit_miss_profile.v1",
            "source_report_sha256": source_basis,
            "input_report_sha256s": {str(directory): hashlib.sha256(
                (directory / "report.json").read_bytes()).hexdigest()
                for directory in directories},
            "population": len(rows), "folds": sorted(folds),
            "by_stage": dict(sorted(by_stage.items())),
            "noncorrect_by_mechanism": dict(sorted(by_mechanism.items())),
            "by_observed_correct_operation_signature": {
                key: dict(sorted(counts.items())) for key, counts in sorted(by_signature.items())},
            "rows": sorted(rows, key=lambda row: row["source"]),
            "development_only": True, "fresh_transfer": False,
            "serving_authority": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = profile(args.directories)
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("miss profile already exists with different content")
    args.output.write_bytes(payload)
    print(json.dumps({key: body[key] for key in (
        "population", "by_stage", "noncorrect_by_mechanism")}, sort_keys=True))


if __name__ == "__main__":
    main()
