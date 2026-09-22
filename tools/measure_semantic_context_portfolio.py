#!/usr/bin/env python3
"""Replay retained method outputs through answer-blind portfolio selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--graph-report", type=Path, required=True, action="append")
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_portfolio import select_semantic_program_portfolio
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    root = args.evidence_root
    parent = compositional_semantic_program_transducer_from_dict(json.loads(
        (root / "semantic-source-fit-20260915/candidate.json").read_text()))
    source_report = json.loads((root / "semantic-source-fit-20260915/report.json").read_text())
    examples = load_source_examples(parent, source_report, [name + "=" + str(
        root / "semantic-source-reacquisition-20260915/features" / name)
        for name in source_report["representation_compatibility"]["source_feature_manifest_sha256s"]])
    reports = [json.loads(path.read_text()) for path in args.graph_report]
    evaluation_split = reports[0]["plan"].get("evaluation_split", "train")
    if evaluation_split not in {"train", "validation"}:
        raise ValueError("portfolio split is not source development data")
    source = {x.ir.source_text_sha256: x for x in examples if x.split == evaluation_split}
    arms = {}
    rows = []
    ids = reports[0]["plan"]["source_ids"]
    for path, report in zip(args.graph_report, reports, strict=True):
        graph_identity = hashlib.sha256(path.read_bytes()).hexdigest()
        if (report["plan"]["parent_receipt"] != parent.receipt_sha256 or report["plan"]["source_ids"] != ids
                or report["plan"].get("evaluation_split", "train") != evaluation_split):
            raise ValueError("portfolio graph observations or argument model differ")
        if report["plan"].get("input_coordinates") != "annotated_source_anchors_v1":
            raise ValueError("portfolio requires programs in common source coordinates")
        if any(row.get("coordinate_status") == "unaligned" for row in report["rows"]):
            raise ValueError("portfolio cannot select an unaligned program as if it refused")
        for arm in report["plan"]["arms"]:
            if arm["name"] in arms:
                raise ValueError("portfolio method identity is repeated")
            arms[arm["name"]] = hashlib.sha256(json.dumps(
                {"graph_report": graph_identity, "arm": arm}, sort_keys=True).encode()).hexdigest()
        rows.extend(report["rows"])
    if args.incumbent not in arms:
        raise ValueError("portfolio incumbent is absent")
    if len(set(ids)) != len(ids) or set(ids) - set(source):
        raise ValueError("portfolio population is not unique source development data")
    indexed = {(row["source"], row["arm"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {(i, a) for i in ids for a in arms}:
        raise ValueError("portfolio graph rows are not complete and paired")
    plan = {"schema": "aura.semantic_program_portfolio_probe.v1", "incumbent": args.incumbent,
            "method_order": list(arms), "source_ids": ids, "fuel_per_program": 2_000_000,
            "graph_reports": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in args.graph_report},
            "selection": "existing_necessary_condition_selector_floor_completion",
            "evaluation_split": evaluation_split,
            "input_coordinates": "annotated_source_anchors_v1",
            "argument_fit_includes_evaluation_examples": evaluation_split == "train",
            "argument_fit_includes_heldout_constructions": True if evaluation_split == "train" else None,
            "promotable": False,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).resolve(), ROOT / "core/learning/semantic_program_portfolio.py",
                          ROOT / "core/evidence/candidate_portfolio.py",
                          ROOT / "core/evidence/necessary_condition_selector.py")}}
    if not atomic_write_bytes_if_absent(args.output.with_suffix(".plan.json"), json.dumps(plan, sort_keys=True).encode()):
        raise RuntimeError("portfolio plan already exists")
    outcomes = []
    for identity in ids:
        inputs = source[identity].public_inputs
        proposals = {}
        for arm in arms:
            payload = indexed[identity, arm]["program"]
            program = (None if payload is None else Program(len(inputs), tuple(
                Instruction(op, tuple(arguments)) for op, arguments in payload["instructions"])))
            if program is not None and program.sha() != payload["sha"]:
                raise ValueError("retained program identity differs")
            proposals[arm] = program
        started = time.monotonic()
        decision = select_semantic_program_portfolio(
            proposals=proposals, provenance=arms, public_inputs=tuple(inputs),
            observation_sha256=identity, incumbent=args.incumbent,
            fuel=plan["fuel_per_program"])
        selection_s = time.monotonic() - started
        selected = decision.decision.selected
        # Evaluation labels enter only after the selector has returned.
        correct = {arm: indexed[identity, arm]["comparison"]["status"] == "equivalent" for arm in arms}
        outcomes.append({"source": identity, "selected": selected, "correct": correct[selected],
                         "incumbent_correct": correct[args.incumbent], "oracle_available": any(correct.values()),
                         "method_correctness": correct, "executions": dict(decision.executions),
                         "method_resolution_s": {arm: indexed[identity, arm]["resolution_s"] for arm in arms},
                         "portfolio_execution_and_comparison_s": selection_s,
                         "candidate_relations": decision.relations,
                         "selection_receipts": [x.receipt for x in decision.decision.comparisons]})
    summary = {"count": len(outcomes), "selected_correct": sum(x["correct"] for x in outcomes),
               "incumbent_correct": sum(x["incumbent_correct"] for x in outcomes),
               "oracle_available": sum(x["oracle_available"] for x in outcomes),
               "gains": sum(x["correct"] and not x["incumbent_correct"] for x in outcomes),
               "regressions": sum(x["incumbent_correct"] and not x["correct"] for x in outcomes),
               "selection_counts": dict(Counter(x["selected"] for x in outcomes))}
    summary["method_resolution_s"] = {arm: sum(x["method_resolution_s"][arm] for x in outcomes) for arm in arms}
    summary["portfolio_execution_and_comparison_s"] = sum(x["portfolio_execution_and_comparison_s"] for x in outcomes)
    summary["candidate_unavailable"] = sum(not x["oracle_available"] for x in outcomes)
    summary["available_but_not_selected"] = sum(x["oracle_available"] and not x["correct"] for x in outcomes)
    summary["candidate_relations"] = dict(Counter(
        relation["status"] for row in outcomes for _, _, relation in row["candidate_relations"]))
    if not atomic_write_bytes_if_absent(args.output, json.dumps(
            {"plan": plan, "summary": summary, "rows": outcomes}, sort_keys=True).encode()):
        raise RuntimeError("portfolio report already exists")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
