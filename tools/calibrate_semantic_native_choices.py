#!/usr/bin/env python3
"""Calibrate combined native-program choices using source-only observations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
import random
import statistics
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


def candidate_relation_keys(bank):
    """Identify typed computation structure without using answer or family labels."""
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_floor import semantic_program_structural_key

    result = {}
    for candidate in bank["bank"]["candidates"]:
        payload = candidate["program"]
        program = Program(len(bank["bank"]["input_spans"]), tuple(
            Instruction(operation, tuple(arguments))
            for operation, arguments in payload["instructions"]))
        structural = semantic_program_structural_key(program)
        if (structural is None or program.sha() != candidate["program_sha256"]
                or payload["sha"] != candidate["program_sha256"]):
            raise ValueError("source candidate has no executable typed relation")
        relation = digest({"structural_relation": structural})
        prior = result.setdefault(candidate["program_sha256"], relation)
        if prior != relation:
            raise ValueError("source candidate relation identity differs across paths")
    return result


def with_schema_support(case, candidate_relations, memory_counts):
    """Attach source-fit relation precedent to candidate views, never outcomes."""
    if case is None:
        return None
    incumbent, choices, views = case
    if set(choices) != set(candidate_relations):
        raise ValueError("schema memory and program inventory differ")
    if any(type(value) is not int or value < 0 for value in memory_counts.values()):
        raise ValueError("schema memory support counts are invalid")
    return incumbent, choices, {key: {**views[key],
        "schema_support_groups": float(memory_counts.get(candidate_relations[key], 0))}
        for key in choices}


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
    incumbent_values = views[incumbent]
    for key in choices:
        views[key].update({f"relative_{name}": value - incumbent_values[name]
                           for name, value in tuple(views[key].items())})
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


def source_invariance_audit(rows, *, excluded_ids, seed=0, permutations=100):
    """Probe nuisance recovery and relation geometry without fitting the selector."""
    eligible = [row for row in rows if row["source"] not in excluded_ids and row["views"] is not None]
    groups = Counter(row["construction"] for row in eligible)
    measured = [row for row in eligible if groups[row["construction"]] >= 2]
    if len(measured) < 2 or len({row["construction"] for row in measured}) < 2:
        return {"status": "insufficient_source_groups", "observations": len(measured)}
    names = tuple(sorted(measured[0]["views"][2][measured[0]["views"][0]]))
    if any(tuple(sorted(row["views"][2][row["views"][0]])) != names for row in measured):
        raise ValueError("nuisance probe feature schema differs")
    raw = [tuple(row["views"][2][row["views"][0]][name] for name in names) for row in measured]
    means = [statistics.fmean(column) for column in zip(*raw, strict=True)]
    scales = [max(statistics.pstdev(column), 1e-9) for column in zip(*raw, strict=True)]
    vectors = [tuple((value - mean) / scale for value, mean, scale
                     in zip(values, means, scales, strict=True)) for values in raw]
    labels = [row["construction"] for row in measured]
    classes = tuple(sorted(set(labels)))

    def balanced_nearest_centroid(assignments):
        hits = Counter()
        totals = Counter()
        for index, vector in enumerate(vectors):
            distances = []
            for group in classes:
                members = [candidate for offset, candidate in enumerate(vectors)
                           if offset != index and assignments[offset] == group]
                if not members:
                    continue
                centroid = tuple(statistics.fmean(column)
                                 for column in zip(*members, strict=True))
                distance = sum((left - right) ** 2 for left, right in zip(vector, centroid, strict=True))
                distances.append((distance, group))
            predicted = min(distances)[1]
            totals[assignments[index]] += 1
            hits[assignments[index]] += predicted == assignments[index]
        return statistics.fmean(hits[group] / totals[group] for group in classes)

    observed = balanced_nearest_centroid(labels)
    rng = random.Random(seed)
    null = []
    for _ in range(permutations):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        null.append(balanced_nearest_centroid(shuffled))

    # Correct-program labels enter this audit only to define matched relations.
    normalized = {row["source"]: vector for row, vector in zip(measured, vectors, strict=True)}
    relation_rows = []
    for row in measured:
        _incumbent, choices, _views = row["views"]
        relations = row.get("candidate_relation_keys", {})
        for relation in sorted({relations[key] for key in choices
                                if row["labels"][key] is True and key in relations}):
            relation_rows.append((row["source"], row["construction"], relation))
    same_cross, different_within = [], []
    for index, (left_source, left_group, left_relation) in enumerate(relation_rows):
        for right_source, right_group, right_relation in relation_rows[index + 1:]:
            if left_source == right_source:
                continue
            same = left_relation == right_relation and left_group != right_group
            different = left_relation != right_relation and left_group == right_group
            if not same and not different:
                continue
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(
                normalized[left_source], normalized[right_source], strict=True)))
            (same_cross if same else different_within).append(distance)
    return {"status": "measured", "observations": len(measured),
            "groups": {group: groups[group] for group in classes},
            "excluded_singleton_groups": sorted(set(groups) - set(classes)),
            "nuisance_probe": {"method": "leave_one_out_standardized_nearest_centroid",
                "balanced_accuracy": round(observed, 6),
                "permutation_mean": round(statistics.fmean(null), 6) if null else None,
                "permutation_p_upper_bound": ((1 + sum(value >= observed for value in null))
                                              / (len(null) + 1)) if null else None,
                "permutations": len(null)},
            "relation_geometry": {"same_relation_cross_group_pairs": len(same_cross),
                "different_relation_within_group_pairs": len(different_within),
                "same_relation_cross_group_median_distance": round(statistics.median(same_cross), 6)
                    if same_cross else None,
                "different_relation_within_group_median_distance": round(statistics.median(different_within), 6)
                    if different_within else None}}


def select_combined(selector, bank, native_rows, *, source_ref, schema_memory_counts=None):
    """Apply the admitted evidence policy without reading comparison outcomes."""
    from core.evidence.necessary_condition_selector import PairwiseSelectionEvidence
    from core.evidence.packet import observe

    case = combined_views(bank, native_rows)
    if schema_memory_counts is not None:
        case = with_schema_support(case, candidate_relation_keys(bank), schema_memory_counts)
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
        candidate_relations = candidate_relation_keys(bank)
        relation_keys = {relation for key, relation in candidate_relations.items()
                         if labels[key] is True}
        result.append({"source": source, "construction": rows[0]["construction"],
                       "views": combined_views(bank, rows), "labels": labels,
                       "verified_relation_keys": tuple(sorted(relation_keys)),
                       "candidate_relation_keys": candidate_relations})
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
    invariance_audit = source_invariance_audit(rows, excluded_ids=excluded_ids, seed=seed)
    relation_coverage = None
    if all("verified_relation_keys" in row for row in rows):
        relations = {}
        for row in rows:
            if row["source"] in excluded_ids:
                continue
            for key in row["verified_relation_keys"]:
                relations.setdefault(key, set()).add(row["construction"])
        relation_coverage = {
            "eligible_verified_relations": len(relations),
            "cross_group_verified_relations": sum(len(groups) > 1 for groups in relations.values()),
            "eligible_groups_with_cross_group_relation": sorted({group for groups in relations.values()
                                                                 if len(groups) > 1 for group in groups}),
        }
    fit_memory = {}
    for row in fit:
        for relation in row.get("verified_relation_keys", ()):
            fit_memory.setdefault(relation, set()).add(row["construction"])
    schema_memory_counts = {key: len(groups) for key, groups in sorted(fit_memory.items())}

    def enriched(population, *, leave_own_group):
        result = []
        for row in population:
            case = row["views"]
            relations = (row.get("candidate_relation_keys")
                         or ({key: key for key in case[1]} if case is not None else {}))
            memory = {key: len(groups - ({row["construction"]} if leave_own_group else set()))
                      for key, groups in fit_memory.items()}
            result.append({**row, "views": with_schema_support(case, relations, memory)})
        return tuple(result)

    fit = enriched(fit, leave_own_group=True)
    tune = enriched(tune, leave_own_group=False)
    admission = enriched(admission, leave_own_group=False)
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
    def rank_diagnostic(population):
        groups = {}
        for row in population:
            if row["views"] is None:
                continue
            incumbent, choices, views = row["views"]
            selected = preferred_challenger(scorer, choices, views)
            a, b = row["labels"][incumbent], row["labels"][selected]
            if type(a) is not bool or type(b) is not bool:
                continue
            counts = groups.setdefault(row["construction"], {"observations": 0,
                "incumbent_correct": 0, "ranked_correct": 0,
                "gains": 0, "regressions": 0, "switches": 0})
            counts["observations"] += 1
            counts["incumbent_correct"] += int(a)
            counts["ranked_correct"] += int(b)
            counts["gains"] += int(b and not a)
            counts["regressions"] += int(a and not b)
            counts["switches"] += int(selected != incumbent)
        return dict(sorted(groups.items()))
    rank_report = ([rank_diagnostic(population) for population in (fit, tune, admission)]
                   if scorer is not None else None)
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
            "invariance_audit": invariance_audit,
            "relation_coverage": relation_coverage,
            "schema_memory_counts": schema_memory_counts,
            "excluded_checkpoint_calibration_ids": sorted(excluded_ids),
            "scorer_report": scorer_report, "selector_report": selector_report,
            "raw_rank_groups": rank_report,
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
