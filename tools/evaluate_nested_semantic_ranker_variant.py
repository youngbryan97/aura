#!/usr/bin/env python3
"""Evaluate frozen source-only ranker weights on a signed proposal variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_variant(training: dict, outer_plan: dict, variant_plan: dict,
                   variant_report: dict, weights: bytes) -> tuple[str, ...]:
    if (training.get("schema") != "aura.semantic_nested_ranker_source_fold.v1"
            or training.get("receipt_sha256") != _digest({
                key: value for key, value in training.items() if key != "receipt_sha256"})
            or training.get("weights_sha256") != hashlib.sha256(weights).hexdigest()
            or training.get("outer_fold") != outer_plan.get("fold")
            or training.get("source_report_sha256") != outer_plan.get("source_report_sha256")
            or outer_plan.get("outer_fold") is not None
            or variant_plan.get("outer_fold") is not None
            or variant_plan.get("fold") != outer_plan.get("fold")
            or variant_plan.get("source_report_sha256") != outer_plan.get("source_report_sha256")
            or variant_plan.get("parent_receipt_sha256") != outer_plan.get("parent_receipt_sha256")
            or variant_plan.get("folds_sha256") != outer_plan.get("folds_sha256")
            or variant_plan.get("fit_ids") != outer_plan.get("fit_ids")
            or variant_plan.get("calibration_ids") != outer_plan.get("calibration_ids")
            or not set(variant_plan.get("held_ids", ())) <= set(outer_plan.get("held_ids", ()))
            or not variant_plan.get("held_ids")
            or set(variant_report.get("row_receipts", {})) != set(variant_plan["held_ids"])):
        raise ValueError("variant ranker evaluation crosses its source-only outer fold")
    return tuple(sorted(variant_plan["held_ids"]))


def score_margins(ranker, items: dict, rows: dict, sources: tuple[str, ...]) -> list[dict]:
    import torch

    from core.learning.semantic_candidate_ranker import aggregate_program_scores
    from core.learning.semantic_span_pointer import _hidden_array
    from tools.evaluate_semantic_candidate_ranker import _rankable_or_none

    ranker.eval()
    result = []
    with torch.no_grad():
        for source in sources:
            row = rows[source]
            case = _rankable_or_none(items[source], row,
                                     preserve_evidence=ranker.retain_evidence_variants)
            if case is None or row["bank"]["selected_program_sha256"] is None:
                result.append({"source": source, "margin": None,
                               "reason": "incumbent_or_bank_unavailable"})
                continue
            (programs, labels, keys), spans, kinds, anchors = case[:4]
            features = torch.from_numpy(_hidden_array(items[source].hidden_states)).float()
            kwargs = ({"argument_spans": case[4], "definition_spans": case[5]}
                      if ranker.argument_evidence else {})
            scores = ranker(features, spans, kinds, programs,
                            operation_spans=anchors, **kwargs)
            if ranker.retain_evidence_variants:
                distinct, grouped, _members = aggregate_program_scores(scores, keys)
                score_map = dict(zip(distinct, (float(value) for value in grouped), strict=True))
            else:
                score_map = dict(zip(keys, (float(value) for value in scores), strict=True))
            incumbent = row["bank"]["selected_program_sha256"]
            chosen = max(score_map, key=score_map.get)
            label_map = dict(zip(keys, labels, strict=True))
            result.append({"source": source, "margin": score_map[chosen] - score_map[incumbent],
                           "chosen_program_sha256": chosen,
                           "incumbent_correct": bool(label_map[incumbent]),
                           "chosen_correct": bool(label_map[chosen]),
                           "candidate_count": len(score_map)})
    return result


def choice_overlap(rows: dict, evaluation: dict) -> dict:
    from collections import Counter

    counts = Counter()
    for result in evaluation["rows"]:
        source = result["source"]
        bank = rows[source]["bank"]
        statuses = {record["program_sha256"]: record["status"] == "equivalent"
                    for record in rows[source]["diagnosis"]["comparisons"]}
        scored = [candidate for candidate in bank["candidates"]
                  if candidate["joint_score"] is not None]
        joint = max(scored, key=lambda candidate: candidate["joint_score"])[
            "program_sha256"] if scored else None
        if joint is not None and joint not in statuses:
            raise ValueError("joint-score candidate lacks an independent comparison")
        counts[(bool(result["incumbent_correct"]), bool(statuses.get(joint, False)),
                bool(result["selected_correct"]))] += 1
    if sum(counts.values()) != evaluation["population"]:
        raise ValueError("choice overlap population differs from paired evaluation")
    return {"ordinary_correct": evaluation["incumbent_correct"],
            "joint_correct": sum(number for (_, joint, _), number in counts.items() if joint),
            "ranker_correct": evaluation["ranker_correct"],
            "oracle_union": sum(number for choices, number in counts.items() if any(choices)),
            "joint_patterns": {"-".join(str(int(value)) for value in choices): number
                               for choices, number in sorted(counts.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--outer-bank", type=Path, required=True)
    parser.add_argument("--variant-bank", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from safetensors.torch import load_file

    from core.learning.semantic_candidate_ranker import ContextualProgramRanker
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_request_context import RequestContextConfig
    from tools.evaluate_semantic_candidate_ranker import _evaluate, _read_bank
    from tools.profile_semantic_crossfit_misses import _load_verified

    training = json.loads(args.training_report.read_bytes())
    outer_plan, outer_rows = _load_verified(args.outer_bank)
    variant_plan, variant_rows = _load_verified(args.variant_bank)
    variant_report_raw = (args.variant_bank / "report.json").read_bytes()
    variant_report = json.loads(variant_report_raw)
    weights = args.weights.read_bytes()
    wanted = verify_variant(training, outer_plan, variant_plan, variant_report, weights)
    outer_report = json.loads((args.outer_bank / "report.json").read_bytes())
    if (training["outer_bank_receipt_sha256"] != outer_report["receipt_sha256"]
            or set(row["source"] for row in outer_rows) != set(outer_plan["held_ids"])):
        raise ValueError("frozen ranker does not belong to the supplied outer bank")
    parent = compositional_semantic_program_transducer_from_dict(
        json.loads(args.parent.read_bytes()))
    source_raw = args.source_report.read_bytes()
    if (parent.receipt_sha256 != outer_plan["parent_receipt_sha256"]
            or hashlib.sha256(source_raw).hexdigest() != training["source_report_sha256"]):
        raise ValueError("ranker parent or source corpus differs")
    source = json.loads(source_raw)
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source["representation_compatibility"]["source_feature_manifest_sha256s"]]
    items = {item.ir.source_text_sha256: item for item in
             load_source_examples(parent, source, bundles) if item.split == "train"}
    if not set(wanted) <= set(items):
        raise ValueError("variant contains a source outside the retained feature corpus")
    variant_model = compositional_semantic_program_transducer_from_dict(
        json.loads((args.variant_bank / "candidate.json").read_bytes()))
    if variant_model.receipt_sha256 != variant_report["candidate_receipt_sha256"]:
        raise ValueError("variant model differs from its signed proposal bank")
    rows = {row["source"]: _read_bank(
        args.variant_bank / "rows" / f'{row["source"]}.json',
        source=row["source"], plan_sha=variant_plan["plan_sha256"],
        model_receipt=variant_model.receipt_sha256,
        expected_receipt=variant_report["row_receipts"][row["source"]])
        for row in variant_rows}
    config = RequestContextConfig(**training["config"])
    ranker = ContextualProgramRanker(
        config, identity_bindings=training["argument_evidence"],
        argument_evidence=training["argument_evidence"],
        retain_evidence_variants=training["argument_evidence"])
    ranker.load_state_dict(load_file(str(args.weights)), strict=True)
    result = _evaluate(ranker, items, rows, list(wanted))
    margins = score_margins(ranker, items, rows, wanted)
    paired = {row["source"]: row for row in result["rows"]}
    for margin in margins:
        row = paired[margin["source"]]
        if margin["margin"] is None:
            if row["chosen_program_sha256"] is not None:
                raise ValueError("margin and evaluation disagree on an unavailable bank")
            continue
        if (margin["chosen_program_sha256"] != row["chosen_program_sha256"]
                or margin["chosen_correct"] != row["selected_correct"]
                or margin["incumbent_correct"] != row["incumbent_correct"]):
            raise ValueError("margin and evaluation disagree on a paired source")
    overlap = choice_overlap(rows, result)
    body = {"schema": "aura.semantic_nested_ranker_variant_evaluation.v1",
            "training_receipt_sha256": training["receipt_sha256"],
            "variant_bank_report_sha256": hashlib.sha256(variant_report_raw).hexdigest(),
            "source_report_sha256": training["source_report_sha256"],
            "evaluation": result, "score_margins": margins, "choice_overlap": overlap,
            "development_only": True,
            "qualification_evidence": False, "serving_authority": False}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("variant ranker result already exists with different content")
    args.output.write_bytes(payload)
    print(json.dumps({key: result[key] for key in
                      ("population", "bank_reachable", "incumbent_correct", "ranker_correct",
                       "gains", "regressions")}, sort_keys=True))


if __name__ == "__main__":
    main()
