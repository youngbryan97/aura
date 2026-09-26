#!/usr/bin/env python3
"""Test source-only ordinary/joint bank arbitration on nested construction folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bank_pair(row: dict) -> tuple[dict, dict, bool, bool] | None:
    """Extract two target-blind views; labels remain outside their features."""
    bank = row["bank"]
    incumbent = bank["selected_program_sha256"]
    scored = [candidate for candidate in bank["candidates"]
              if candidate["joint_score"] is not None]
    if incumbent is None or not scored:
        return None
    best = max(scored, key=lambda candidate: candidate["joint_score"])
    challenger = best["program_sha256"]
    if challenger == incumbent:
        return None
    by_program = {}
    for candidate in scored:
        key = candidate["program_sha256"]
        if key not in by_program or candidate["joint_score"] > by_program[key]["joint_score"]:
            by_program[key] = candidate
    statuses = {record["program_sha256"]: record["status"] == "equivalent"
                for record in row["diagnosis"]["comparisons"]}
    if incumbent not in statuses or challenger not in statuses:
        raise ValueError("bank choice lacks an independent comparison")
    unique = len({candidate["program_sha256"] for candidate in bank["candidates"]})
    maximum = best["joint_score"]

    def features(key: str) -> dict[str, float]:
        candidate = by_program.get(key)
        return {"executable_program": 1.0,
                "joint_evidence_available": float(candidate is not None),
                "joint_gap": float(maximum - candidate["joint_score"])
                if candidate is not None else 0.0,
                "program_depth": float(len((candidate or next(
                    item for item in bank["candidates"]
                    if item["program_sha256"] == key))["program"]["instructions"])),
                "candidate_count": float(unique)}

    left, right = features(incumbent), features(challenger)
    if any(not math.isfinite(value) for view in (left, right) for value in view.values()):
        raise ValueError("bank choice feature is nonfinite")
    return left, right, statuses[incumbent], statuses[challenger]


def construction_support(rows: list, constructions: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source, _pair in rows:
        if source not in constructions:
            raise ValueError("pairwise bank source is absent from construction ledger")
        group = constructions[source]
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-bank", type=Path, required=True)
    parser.add_argument("--inner-bank", type=Path, action="append", required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inner_bank) != 3:
        parser.error("three nested inner construction banks are required")

    from core.evidence.calibrated_binary import (
        VerifiedBinaryObservation, fit_calibrated_binary_scorer)
    from core.evidence.calibrated_candidate_selector import (
        VerifiedPairwiseObservation, build_calibrated_candidate_selector)
    from core.evidence.necessary_condition_selector import (
        NecessaryEvidenceCondition, build_necessary_condition_selector)
    from core.evidence.packet import observe
    from core.evidence.necessary_condition_selector import PairwiseSelectionEvidence
    from core.learning.semantic_program_campaign import _sha as semantic_sha
    from tools.evaluate_semantic_candidate_ranker import _read_bank
    from tools.train_nested_semantic_ranker import _verified_pair, validate_nested_provenance

    outer = _verified_pair(args.outer_bank)
    inners = [_verified_pair(path) for path in args.inner_bank]
    validate_nested_provenance(*outer, inners)
    folds_raw = args.folds.read_bytes()
    folds = json.loads(folds_raw)
    if (hashlib.sha256(folds_raw).hexdigest() != outer[0]["folds_sha256"]
            or folds.get("schema") != "aura.semantic_construction_folds.v1"
            or folds.get("receipt_sha256") != semantic_sha({
                key: value for key, value in folds.items() if key != "receipt_sha256"})):
        raise ValueError("construction ledger differs from signed nested bank")
    constructions = {source: group for source, group, _contrast in folds["population"]}
    grouped = {}
    for directory, (plan, report) in zip(
            (args.outer_bank, *args.inner_bank), (outer, *inners), strict=True):
        if plan["fold"] in grouped and plan.get("outer_fold") is not None:
            raise ValueError("nested construction fold is duplicated")
        rows = []
        for source, receipt in sorted(report["row_receipts"].items()):
            row = _read_bank(directory / "rows" / f"{source}.json", source=source,
                             plan_sha=plan["plan_sha256"],
                             model_receipt=report["candidate_receipt_sha256"],
                             expected_receipt=receipt)
            pair = bank_pair(row)
            if pair is not None:
                rows.append((source, pair))
        grouped[(plan.get("outer_fold"), plan["fold"])] = rows
    folds = sorted((fold for outer_fold, fold in grouped if outer_fold is not None))
    if folds != [0, 1, 2]:
        raise ValueError("nested fit, calibration and admission folds are incomplete")

    def binary(rows):
        return [VerifiedBinaryObservation.from_mapping(view,
                 verified_correct=correct, source_ref=f"{source}:{side}")
                for source, (left, right, a, b) in rows
                for side, view, correct in (("ordinary", left, a), ("joint", right, b))]

    def pairwise(rows):
        return [VerifiedPairwiseObservation.from_mappings(
                incumbent=left, challenger=right, incumbent_correct=a,
                challenger_correct=b, source_ref=source)
                for source, (left, right, a, b) in rows]

    fit, tune, admission = (grouped[(outer[0]["fold"], fold)] for fold in folds)
    support = [construction_support(rows, constructions)
               for rows in (fit, tune, admission, grouped[(None, outer[0]["fold"])])]
    if min(len(groups) for groups in support[:3]) < 2:
        scorer, scorer_report = None, {
            "admitted": False, "reason": "insufficient_independent_constructions",
            "construction_counts": [len(groups) for groups in support[:3]],
            "binary_calibration_observations": len(binary(tune)),
            "minimum_binary_calibration_observations": 24}
    else:
        scorer, scorer_report = fit_calibrated_binary_scorer(binary(fit), binary(tune))
    selector, selector_report = None, {"admitted": False, "reason": "scorer_not_admitted"}
    if scorer is not None:
        necessary = build_necessary_condition_selector((NecessaryEvidenceCondition(
            "executable_program", 1.0, "exact_program_requires_executable_ir"),))
        selector, selector_report = build_calibrated_candidate_selector(
            necessary=necessary, scorer=scorer, calibration_rows=pairwise(tune),
            admission_rows=pairwise(admission), maximum_regressions=0)
    outer_rows = grouped[(None, outer[0]["fold"])]
    outer_result = None
    if selector is not None:
        outcomes = []
        for source, (left, right, a, b) in outer_rows:
            evidence = PairwiseSelectionEvidence.from_mappings(
                incumbent=left, challenger=right,
                packet=observe(1.0, origin="semantic_bank", ref=source))
            selected = selector.select(incumbent="ordinary", challenger="joint",
                                       evidence=evidence).selected
            outcomes.append((a, b, selected))
        outer_result = {"pairs": len(outcomes),
                        "ordinary_correct": sum(a for a, _b, _s in outcomes),
                        "joint_correct": sum(b for _a, b, _s in outcomes),
                        "selected_correct": sum(b if selected == "joint" else a
                                                for a, b, selected in outcomes),
                        "gains": sum(not a and b and s == "joint" for a, b, s in outcomes),
                        "regressions": sum(a and not b and s == "joint" for a, b, s in outcomes)}
    body = {"schema": "aura.semantic_nested_bank_pairwise_calibration.v1",
            "outer_bank_receipt_sha256": outer[1]["receipt_sha256"],
            "inner_bank_receipt_sha256s": [pair[1]["receipt_sha256"] for pair in inners],
            "split_pair_counts": [len(fit), len(tune), len(admission), len(outer_rows)],
            "split_construction_support": support,
            "scorer_report": scorer_report, "selector_report": selector_report,
            "outer_result": outer_result, "development_only": True,
            "qualification_evidence": False, "serving_authority": False}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("nested calibration result already exists with different content")
    args.output.write_bytes(payload)
    print(json.dumps({"scorer_admitted": scorer is not None,
                      "selector_admitted": selector is not None,
                      "outer_result": outer_result}, sort_keys=True))


if __name__ == "__main__":
    main()
