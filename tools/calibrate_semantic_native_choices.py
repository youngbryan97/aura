#!/usr/bin/env python3
"""Calibrate combined native-program choices using source-only observations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_semantic_native_checkpoint import (  # noqa: E402
    digest,
    verified_document,
    verify_replay_row,
)


def partition_source_rows(rows, *, excluded_ids, seed=0):
    """Keep construction groups intact; exclude checkpoint-selection instances."""
    if type(seed) is not int:
        raise ValueError("native arbitration partition requires an integer seed")
    identities = [row["source"] for row in rows]
    if len(set(identities)) != len(identities) or not set(excluded_ids) <= set(identities):
        raise ValueError("native arbitration sources are duplicated or exclusions are absent")
    eligible = [row for row in rows if row["source"] not in set(excluded_ids)]
    groups = {row["construction"] for row in eligible}
    if len(groups) < 6 or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("native arbitration needs six separate construction groups")
    ordered = sorted(groups, key=lambda group: digest({"seed": seed, "construction": group}))
    assignments = {group: index % 3 for index, group in enumerate(ordered)}
    return tuple(tuple(sorted((row for row in eligible
                               if assignments[row["construction"]] == split),
                              key=lambda row: row["source"])) for split in range(3))


def combined_views(bank, native_rows):
    """Features contain computation evidence, never target labels or domain names."""
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_floor import semantic_program_structural_key

    if not native_rows or any(row.get("scored") is False for row in native_rows):
        return None
    keys = native_rows[0]["program_sha256s"]
    if not keys or any(row["program_sha256s"] != keys for row in native_rows):
        raise ValueError("native methods must score the identical proposal inventory")
    candidates = {}
    paths = {}
    for candidate in bank["bank"]["candidates"]:
        key = candidate["program_sha256"]
        payload = candidate["program"]
        program = Program(len(bank["bank"]["input_spans"]), tuple(
            Instruction(op, tuple(refs)) for op, refs in payload["instructions"]))
        if (program.sha() != key or payload["sha"] != key
                or semantic_program_structural_key(program) is None):
            raise ValueError("combined native choice is not an executable typed program")
        paths.setdefault(key, []).append(candidate)
        if key not in candidates or (candidate["joint_score"] is not None and (
                candidates[key]["joint_score"] is None
                or candidate["joint_score"] > candidates[key]["joint_score"])):
            candidates[key] = candidate
    if set(keys) != set(candidates):
        raise ValueError("native scored program is absent from its source bank")
    joint = [candidates[key]["joint_score"] for key in keys
             if candidates[key]["joint_score"] is not None]
    maximum = max(joint) if joint else None
    joint_winner = (max((key for key in keys if candidates[key]["joint_score"] is not None),
                        key=lambda key: (candidates[key]["joint_score"], key))
                    if maximum is not None else None)

    def midpoint(span):
        if (not isinstance(span, dict) or type(span.get("start")) is not int
                or type(span.get("end")) is not int
                or not 0 <= span["start"] < span["end"]):
            raise ValueError("candidate source evidence span is invalid")
        return (span["start"] + span["end"]) / 2

    source_spans = []
    for variant in bank["bank"]["candidates"]:
        for field in ("operation_spans", "argument_spans", "definition_spans"):
            rows = variant.get(field)
            if rows is not None:
                source_spans.extend(rows if field == "operation_spans"
                                    else (span for row in rows for span in row))
    for span in source_spans:
        midpoint(span)
    source_scale = max((span["end"] for span in source_spans), default=1)

    def path_evidence(variants, depth):
        anchors = []
        distances = []
        overlaps = []
        selected_definitions = 0
        for variant in variants:
            operations = variant.get("operation_spans")
            mentions = variant.get("argument_spans")
            definitions = variant.get("definition_spans")
            if operations is None:
                continue
            if len(operations) != depth:
                raise ValueError("candidate operation evidence differs from its graph")
            anchors.extend(operations)
            selected_definitions += variant.get("definition_provenance") == "optimizer_selected"
            if mentions is None or definitions is None:
                continue
            if len(mentions) != len(definitions) or len(mentions) != len(operations):
                raise ValueError("candidate source evidence differs from its graph")
            for mentioned, defined in zip(mentions, definitions, strict=True):
                if len(mentioned) != len(defined):
                    raise ValueError("candidate source roles differ")
                for left, right in zip(mentioned, defined, strict=True):
                    distances.append(abs(midpoint(left) - midpoint(right)))
                    overlaps.append(float(min(left["end"], right["end"])
                                          > max(left["start"], right["start"])))
                    anchors.extend((left, right))
        return {"source_evidence_paths": float(len(variants)),
                "source_span_available": float(bool(anchors)),
                "source_relation_available": float(bool(distances)),
                "source_relation_distance": (sum(distances) / len(distances) / source_scale
                                             if distances else 0.),
                "source_relation_overlap": (sum(overlaps) / len(overlaps)
                                            if overlaps else 0.),
                "selected_definition_paths": float(selected_definitions)}

    views = {}
    for index, key in enumerate(keys):
        candidate = candidates[key]
        score = candidate["joint_score"]
        values = {"executable_program": 1., "candidate_count": float(len(keys)),
                  "program_depth": float(len(candidate["program"]["instructions"])),
                  "joint_evidence_available": float(score is not None),
                  "joint_gap": float(maximum - score) if score is not None else 0.,
                  "joint_winner": float(key == joint_winner),
                  **path_evidence(paths[key], len(candidate["program"]["instructions"]))}
        for method, row in enumerate(native_rows):
            for field in ("scores", "pretrained_scores"):
                scores = row[field]
                if len(scores) != len(keys):
                    raise ValueError("native score vector differs from its program inventory")
                values[f"method_{method}_{field}"] = float(scores[index])
                values[f"method_{method}_{field}_gap"] = float(max(scores) - scores[index])
                values[f"method_{method}_{field}_winner"] = float(
                    key == keys[max(range(len(scores)), key=scores.__getitem__)])
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("combined native evidence is nonfinite")
        views[key] = values
    incumbent = bank["bank"]["selected_program_sha256"]
    if incumbent is not None and keys[0] != incumbent:
        raise ValueError("native score inventory lost its ordinary incumbent")
    choices = tuple(keys)
    if incumbent not in views or any(key not in views for key in choices):
        return None
    return incumbent, choices, views


def independently_graded(bank):
    from tools.evaluate_semantic_native_checkpoint import source_calibration_labels

    return source_calibration_labels(bank)


def preferred_challenger(scorer, choices, views):
    return max(choices, key=lambda key: (scorer.predict(views[key]), key))


def native_method_basis(plan):
    return {key: plan[key] for key in (
        "training_plan_sha256", "checkpoint_receipt_sha256", "weights_sha256", "selected_step",
        "semantic_decision_basis", "loss_scope", "model_descriptor_sha256", "pointer_sha256")}


def select_combined(selector, bank, native_rows, *, source_ref):
    """Apply the admitted evidence policy without reading comparison outcomes."""
    from core.evidence.necessary_condition_selector import PairwiseSelectionEvidence
    from core.evidence.packet import observe

    case = combined_views(bank, native_rows)
    if case is None:
        return bank["bank"]["selected_program_sha256"]
    incumbent, choices, views = case
    challenger = preferred_challenger(selector.scorer, choices, views)
    if challenger == incumbent:
        return incumbent
    evidence = PairwiseSelectionEvidence.from_mappings(
        incumbent=views[incumbent], challenger=views[challenger],
        packet=observe(1., origin="semantic_native_program_scores", ref=source_ref))
    decision = selector.select(incumbent=incumbent, challenger=challenger, evidence=evidence)
    return decision.selected


def source_observations(bank_directory, native_directories, *, origin_directory):
    """Bind every scored row to the complete source-only bank it consumed."""
    from tools.evaluate_semantic_candidate_ranker import _read_bank
    from tools.evaluate_semantic_native_checkpoint import verified_calibration_replay_basis

    origin_plan = verified_document(origin_directory / "plan.json", "plan_sha256")
    origin_report = verified_document(origin_directory / "report.json")
    bank_plan, bank_report, _identities = verified_calibration_replay_basis(
        bank_directory, origin_directory=origin_directory,
        origin_plan=origin_plan, origin_report=origin_report)
    if (bank_plan.get("schema") != "aura.semantic_proposer_source_calibration_plan.v1"
            or bank_report.get("schema") != "aura.semantic_proposer_source_calibration.v1"
            or bank_report["plan_sha256"] != bank_plan["plan_sha256"]
            or bank_report.get("held_rows_evaluated") is not False
            or set(bank_report["row_receipts"]) != set(bank_plan["evaluated_ids"])
            or bank_report["source_calibration_population"] != len(bank_plan["evaluated_ids"])):
        raise ValueError("native arbitration requires a complete source-only bank")
    methods, excluded = [], set()
    for directory in native_directories:
        plan = verified_document(directory / "plan.json", "plan_sha256")
        report = verified_document(directory / "report.json")
        if (plan.get("schema") != "aura.semantic_native_source_calibration_plan.v1"
                or report.get("schema") != "aura.semantic_native_source_calibration.v1"
                or report["plan_sha256"] != plan["plan_sha256"]
                or plan["source_calibration_plan_sha256"] != bank_plan["plan_sha256"]
                or plan["source_calibration_receipt_sha256"] != bank_report["receipt_sha256"]
                or plan["evaluated_ids"] != bank_plan["evaluated_ids"]
                or plan.get("held_rows_evaluated") is not False
                or report.get("held_rows_evaluated") is not False
                or type(plan.get("fit_updates")) is not int or plan["fit_updates"] != 0
                or type(report.get("fit_updates")) is not int or report["fit_updates"] != 0
                or report["population"] != len(bank_plan["evaluated_ids"])
                or len(report["rows"]) != report["population"]
                or any(document.get("serving_authority") is not False
                       or document.get("qualification_evidence") is not False
                       for document in (plan, report))):
            raise ValueError("native arbitration scoring provenance differs")
        by_source = {row["source"]: row for row in report["rows"]}
        if set(by_source) != set(bank_plan["evaluated_ids"]) or len(by_source) != len(report["rows"]):
            raise ValueError("native arbitration score population differs")
        if methods and any(plan[key] != methods[0][1][key]
                           for key in ("model_descriptor_sha256", "pointer_sha256")):
            raise ValueError("native arbitration methods use different resident identities")
        excluded.update(plan["checkpoint_calibration_ids"])
        methods.append((directory, plan, report, by_source))
    if not methods or len({method[2]["receipt_sha256"] for method in methods}) != len(methods):
        raise ValueError("native arbitration methods are empty or duplicated")
    result = []
    for source in bank_plan["evaluated_ids"]:
        bank = _read_bank(bank_directory / "rows" / f"{source}.json", source=source,
            plan_sha=bank_plan["plan_sha256"], model_receipt=bank_report["candidate_receipt_sha256"],
            expected_receipt=bank_report["row_receipts"][source])
        labels = independently_graded(bank)
        rows = []
        for directory, plan, _report, by_source in methods:
            row = by_source[source]
            if verified_document(directory / "rows" / f"{source}.json") != row:
                raise ValueError("native arbitration durable score row differs")
            kwargs = {} if row.get("scored") is False else dict(
                programs=row["program_sha256s"],
                labels=tuple(labels[key] for key in row["program_sha256s"]),
                incumbent_available=bank["bank"]["selected_program_sha256"] is not None)
            verify_replay_row(row, source=source, plan_sha256=plan["plan_sha256"], **kwargs)
            rows.append(row)
        if len({row["construction"] for row in rows}) != 1:
            raise ValueError("native arbitration construction identity differs")
        result.append({"source": source, "construction": rows[0]["construction"],
                       "views": combined_views(bank, rows), "labels": labels})
    return result, excluded, bank_report, tuple(method[2] for method in methods)


def calibrate(rows, *, excluded_ids, seed=0):
    """Use the existing scorer and selector with separate fit, tune and admission."""
    from core.evidence.calibrated_binary import (
        VerifiedBinaryObservation,
        fit_calibrated_binary_scorer,
    )
    from core.evidence.calibrated_candidate_selector import (
        VerifiedPairwiseObservation,
        build_calibrated_candidate_selector,
    )
    from core.evidence.necessary_condition_selector import (
        NecessaryEvidenceCondition,
        build_necessary_condition_selector,
    )

    fit, tune, admission = partition_source_rows(rows, excluded_ids=excluded_ids, seed=seed)
    all_groups = Counter(row["construction"] for row in rows)
    eligible_groups = Counter(row["construction"] for row in rows
                              if row["source"] not in excluded_ids)
    coverage = {
        "source_groups": dict(sorted(all_groups.items())),
        "eligible_groups": dict(sorted(eligible_groups.items())),
        "excluded_groups": sorted(set(all_groups) - set(eligible_groups)),
        "split_groups": [dict(sorted(Counter(row["construction"] for row in population).items()))
                         for population in (fit, tune, admission)],
    }
    def binary(population):
        observations = []
        for row in population:
            if row["views"] is None:
                continue
            incumbent, choices, views = row["views"]
            for key in dict.fromkeys((incumbent, *choices)):
                correct = row["labels"][key]
                if type(correct) is bool:
                    observations.append(VerifiedBinaryObservation.from_mapping(
                        views[key], verified_correct=correct, source_ref=f"{row['source']}:{key}"))
        return observations
    scorer, scorer_report = fit_calibrated_binary_scorer(binary(fit), binary(tune))
    selector, selector_report = None, {"admitted": False, "reason": "scorer_not_admitted"}
    if scorer is not None:
        def pairwise(population):
            observations = []
            for row in population:
                if row["views"] is None:
                    continue
                incumbent, choices, views = row["views"]
                challenger = preferred_challenger(scorer, choices, views)
                a, b = row["labels"][incumbent], row["labels"][challenger]
                if type(a) is bool and type(b) is bool:
                    observations.append(VerifiedPairwiseObservation.from_mappings(
                        incumbent=views[incumbent], challenger=views[challenger],
                        incumbent_correct=a, challenger_correct=b, source_ref=row["source"]))
            return observations
        necessary = build_necessary_condition_selector((NecessaryEvidenceCondition(
            "executable_program", 1., "exact_program_requires_executable_ir"),))
        selector, selector_report = build_calibrated_candidate_selector(
            necessary=necessary, scorer=scorer, calibration_rows=pairwise(tune),
            admission_rows=pairwise(admission), maximum_regressions=0)
    return {"split_ids": [[row["source"] for row in population]
                           for population in (fit, tune, admission)],
            "construction_coverage": coverage,
            "excluded_checkpoint_calibration_ids": sorted(excluded_ids),
            "scorer_report": scorer_report, "selector_report": selector_report,
            "selector": selector.to_dict() if selector is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-bank", type=Path, required=True)
    parser.add_argument("--origin-bank", type=Path, required=True)
    parser.add_argument("--native-calibration", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    paths = [Path(__file__), ROOT / "core/evidence/calibrated_binary.py",
             ROOT / "core/evidence/calibrated_candidate_selector.py",
             ROOT / "core/evidence/necessary_condition_selector.py",
             ROOT / "tools/evaluate_semantic_native_checkpoint.py",
             ROOT / "tools/probe_semantic_proposer_crossfit.py"]
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths}
    rows, excluded, bank_report, reports = source_observations(
        args.calibration_bank, args.native_calibration, origin_directory=args.origin_bank)
    result = calibrate(rows, excluded_ids=excluded, seed=args.seed)
    if (source_observations(args.calibration_bank, args.native_calibration,
                            origin_directory=args.origin_bank) != (rows, excluded, bank_report, reports)
            or any(hashlib.sha256(path.read_bytes()).hexdigest()
                   != implementation[str(path.relative_to(ROOT))] for path in paths)):
        raise ValueError("native arbitration inputs or implementation drifted")
    body = {"schema": "aura.semantic_native_combined_calibration.v1", **result,
        "source_bank_receipt_sha256": bank_report["receipt_sha256"],
        "origin_bank_receipt_sha256": verified_document(args.origin_bank / "report.json")["receipt_sha256"],
        "native_receipt_sha256s": [report["receipt_sha256"] for report in reports],
        "native_method_bases": [native_method_basis(verified_document(
            directory / "plan.json", "plan_sha256")) for directory in args.native_calibration],
        "population": len(rows), "seed": args.seed,
        "implementation": implementation,
        "held_rows_evaluated": False, "serving_authority": False, "qualification_evidence": False}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print({"selector_admitted": result["selector"] is not None,
           "split_counts": [len(population) for population in result["split_ids"]]})


if __name__ == "__main__":
    main()
