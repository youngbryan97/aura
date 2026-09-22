#!/usr/bin/env python3
"""Replay the frozen target-blind candidate bank on attributed G03 failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plan(gaps: dict, *, gap_sha: str, model, source_report_sha: str,
          candidate_report_sha: str, implementation: str, max_charts: int, max_graphs: int,
          solve_seconds: float, composition_budget: int,
          composition_examined_limit: int) -> dict:
    sources = tuple(sorted(row["source"] for row in gaps["failures"]))
    if (gaps.get("schema") not in {"aura.semantic_portfolio_gap_diagnosis.v1",
                                     "aura.semantic_portfolio_gap_diagnosis.v2"}
            or gaps.get("validation_labels_used_only_for_diagnosis") is not True
            or len(sources) != len(set(sources))
            or gaps["summary"]["candidate_unavailable"]
            + gaps["summary"]["selection_miss"] != len(sources)
            or any(type(value) is not int or value < 1 for value in (max_charts, max_graphs))
            or type(composition_budget) is not int or composition_budget < 0
            or type(composition_examined_limit) is not int or composition_examined_limit < 1
            or type(solve_seconds) not in (int, float) or not 0 < solve_seconds <= 60):
        raise ValueError("candidate bank audit requires frozen development failures and bounds")
    body = {"schema": "aura.semantic_gap_candidate_bank_plan.v1",
            "gap_report_sha256": gap_sha,
            "audit_tool_sha256": _sha(Path(__file__).read_bytes()),
            "source_report_sha256": source_report_sha,
            "candidate_report_sha256": candidate_report_sha,
            "model_receipt_sha256": model.receipt_sha256,
            "model_basis_sha256": model.model_basis_sha256,
            "implementation": implementation,
            "sources": sources, "max_charts": max_charts,
            "max_graphs_per_chart": max_graphs,
            "solve_seconds": float(solve_seconds),
            "composition_budget": composition_budget,
            "composition_examined_limit": composition_examined_limit,
            "validation_labels_not_sent_to_decoder": True,
            "serving_authority": False}
    return {**body, "plan_sha256": _sha(json.dumps(body, sort_keys=True).encode())}


def _frozen_methods(gaps: dict, items: dict) -> dict:
    from core.learning.semantic_candidate_union import GroundedProgram
    from core.learning.semantic_program_ir import TokenSpan
    from tools.diagnose_semantic_portfolio_gaps import _program_from_record

    wanted = {row["source"]: row for row in gaps["failures"]}
    methods = {source: {} for source in wanted}
    for name, expected_sha in gaps["graph_report_sha256s"].items():
        raw = Path(name).read_bytes()
        if _sha(raw) != expected_sha:
            raise ValueError("frozen graph report digest differs")
        for row in json.loads(raw)["rows"]:
            source = row["source"]
            if source not in wanted:
                continue
            arm = row["arm"]
            if (arm not in wanted[source]["method_programs"]
                    or row["program"] != wanted[source]["method_programs"][arm]
                    or arm in methods[source]):
                raise ValueError("frozen method differs from attributed proposal")
            original = row["original_program"]
            if original is None:
                continue
            spans = tuple(TokenSpan(*span) for span in row["runtime_input_spans"])
            methods[source][arm] = GroundedProgram(
                _program_from_record(original, len(items[source].public_inputs)),
                spans, source, expected_sha)
    if any(set(methods[source]) != {name for name, program in row["method_programs"].items()
                                    if program is not None}
           for source, row in wanted.items()):
        raise ValueError("frozen graph reports omit an attributed method")
    return methods


def _mixed_diagnosis(bank, item, frozen: dict, plan: dict) -> dict:
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_candidate_union import (
        UnalignableProposalError,
        unify_semantic_candidates,
    )
    from core.learning.semantic_graph_counterexamples import (
        ProgramObservationCache,
        compare_program_meanings,
        counterfactual_inputs,
    )
    from core.learning.semantic_joint_graph_learning import align_source_input_registers

    try:
        union = unify_semantic_candidates(
            banks={"candidate_bank": bank}, additional=frozen,
            public_inputs=tuple(item.public_inputs), source_sha256=item.ir.source_text_sha256,
            composition_budget=plan["composition_budget"],
            composition_examined_limit=plan["composition_examined_limit"])
    except UnalignableProposalError as exc:
        return {"correct_reachable": None, "stage": "coordinate_mismatch",
                "reason": str(exc), "serving_authority": False}
    try:
        instructions, _ = align_source_input_registers(item, union.input_spans)
    except ValueError as exc:
        return {"union_receipt_sha256": union.receipt["receipt_sha256"],
                "correct_reachable": None, "stage": "grounding", "reason": str(exc)}
    target = Program(len(item.public_inputs), tuple(
        Instruction(ins.op, ins.args) for ins in instructions))
    probes = counterfactual_inputs(item.public_inputs, count=16)
    cache = ProgramObservationCache()
    comparisons = tuple({"program_sha256": row.program.sha(), "origins": row.origins,
                         "comparison": compare_program_meanings(
                             target, row.program, probes, observation_cache=cache)}
                        for row in union.candidates)
    reached = any(row["comparison"]["status"] == "equivalent" for row in comparisons)
    unknown = any(row["comparison"]["status"] == "unknown" for row in comparisons)
    complete = bank.receipt["search_complete"]
    return {"union_receipt_sha256": union.receipt["receipt_sha256"],
            "candidate_count": len(comparisons),
            "composition": union.receipt["composition"],
            "search_complete": complete,
            "correct_reachable": True if reached else None if unknown or not complete else False,
            "comparisons": comparisons, "stage": "graded", "serving_authority": False}


def _audit_row(item, model, plan: dict, frozen: dict | None = None) -> dict:
    from core.learning.semantic_failure_diagnosis import diagnose_semantic_candidate_bank

    bank = model.decode_candidates(
        source_token_ids=item.ir.source_token_ids,
        hidden_states=item.hidden_states,
        public_inputs=item.public_inputs,
        source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
        max_charts=plan["max_charts"],
        max_graphs_per_chart=plan["max_graphs_per_chart"],
        solve_time_limit_s=plan["solve_seconds"],
    )
    bank.validate()
    diagnosis = diagnose_semantic_candidate_bank(bank, item)
    return {"source": item.ir.source_text_sha256,
            "plan_sha256": plan["plan_sha256"],
            "bank": bank.receipt, "diagnosis": diagnosis,
            "mixed": None if frozen is None else _mixed_diagnosis(bank, item, frozen, plan)}


def _read_or_run(item, model, plan: dict, directory: Path, frozen: dict) -> dict:
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    path = directory / "rows" / f"{item.ir.source_text_sha256}.json"
    if path.exists():
        row = json.loads(path.read_bytes())
    else:
        print(json.dumps({"stage": "started", "source": item.ir.source_text_sha256}), flush=True)
        body = _audit_row(item, model, plan, frozen)
        row = {**body, "receipt_sha256": _sha(json.dumps(body, sort_keys=True).encode())}
        if not atomic_write_bytes_if_absent(path, json.dumps(row, sort_keys=True).encode(), mode=0o400):
            raise FileExistsError(path)
    body = {key: value for key, value in row.items() if key != "receipt_sha256"}
    if (row.get("receipt_sha256") != _sha(json.dumps(body, sort_keys=True).encode())
            or row.get("plan_sha256") != plan["plan_sha256"]
            or row.get("source") != item.ir.source_text_sha256):
        raise ValueError("candidate bank row changed since acquisition")
    return row


def _summarize(rows: list[dict], plan: dict) -> dict:
    counts = Counter(row["diagnosis"]["failure_stage"] for row in rows)
    reachable = Counter(str(row["diagnosis"]["correct_reachable"]) for row in rows)
    mixed = Counter(str(row["mixed"]["correct_reachable"]) for row in rows)
    return {"schema": "aura.semantic_gap_candidate_bank_audit.v1",
            "plan": plan, "observed": len(rows), "population": len(plan["sources"]),
            "complete_population": len(rows) == len(plan["sources"]),
            "failure_stage_counts": dict(sorted(counts.items())),
            "correct_reachable_counts": dict(sorted(reachable.items())),
            "mixed_reachable_counts": dict(sorted(mixed.items())),
            "selected_equivalent": sum(row["diagnosis"]["selected_semantic_status"] == "equivalent"
                                       for row in rows),
            "selection_changed": False, "serving_authority": False,
            "rows": rows}


def _verify_inputs(gaps: dict, source_report: dict, candidate_report: dict,
                   model, items: dict) -> None:
    if candidate_report.get("candidate") != model.receipt_sha256:
        raise ValueError("candidate report does not identify this transducer")
    manifests = source_report["representation_compatibility"]["source_feature_manifest_sha256s"]
    if gaps.get("source_manifest_sha256s") != manifests:
        raise ValueError("gap cohort and source feature manifests differ")
    for failure in gaps["failures"]:
        item = items.get(failure["source"])
        if item is None or item.ir.to_program().to_dict() != failure["gold_program"]:
            raise ValueError("gap target differs from frozen source observation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True,
                        help="frozen candidate report naming the exact transducer receipt")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-charts", type=int, default=8)
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--solve-seconds", type=float, default=2.)
    parser.add_argument("--composition-budget", type=int, default=8)
    parser.add_argument("--composition-examined-limit", type=int, default=256)
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0:
        parser.error("stop-after must be nonnegative")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.directory / "report.json")
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_validation_checkpoint import validation_implementation_identity
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    gaps_raw = args.gap_report.read_bytes()
    gaps = json.loads(gaps_raw)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    source_raw = args.source_report.read_bytes()
    source_report = json.loads(source_raw)
    candidate_raw = args.candidate_report.read_bytes()
    candidate_report = json.loads(candidate_raw)
    bundles = [name + "=" + str(args.feature_root / name)
               for name in source_report["representation_compatibility"]
               ["source_feature_manifest_sha256s"]]
    examples = load_source_examples(model, source_report, bundles)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "validation"}
    _verify_inputs(gaps, source_report, candidate_report, model, items)
    frozen = _frozen_methods(gaps, items)
    implementation = validation_implementation_identity()
    plan = _plan(gaps, gap_sha=_sha(gaps_raw), model=model,
                 source_report_sha=_sha(source_raw), candidate_report_sha=_sha(candidate_raw),
                 implementation=implementation,
                 max_charts=args.max_charts, max_graphs=args.max_graphs,
                 solve_seconds=args.solve_seconds,
                 composition_budget=args.composition_budget,
                 composition_examined_limit=args.composition_examined_limit)
    (args.directory / "rows").mkdir(parents=True, exist_ok=True)
    rows = []
    for source in plan["sources"]:
        if args.stop_after and len(rows) == args.stop_after:
            break
        rows.append(_read_or_run(items[source], model, plan, args.directory, frozen[source]))
        print(json.dumps({"stage": "completed", "completed": len(rows),
                          "total": len(plan["sources"]), "source": source}), flush=True)
    if implementation != validation_implementation_identity():
        raise ValueError("candidate bank implementation changed during acquisition")
    if plan["audit_tool_sha256"] != _sha(Path(__file__).read_bytes()):
        raise ValueError("candidate bank audit tool changed during acquisition")
    report = _summarize(rows, plan)
    if not report["complete_population"]:
        print(json.dumps({"stage": "partial", "observed": len(rows),
                          "population": len(plan["sources"])}), flush=True)
        return
    body = {**report, "receipt_sha256": _sha(json.dumps(report, sort_keys=True).encode())}
    path = args.directory / "report.json"
    payload = json.dumps(body, sort_keys=True).encode()
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError("published bank audit differs")
    print(json.dumps({"stage": "complete", "summary": {
        key: value for key, value in body.items() if key in {
            "observed", "population", "failure_stage_counts", "correct_reachable_counts",
            "selected_equivalent", "mixed_reachable_counts"}}}), flush=True)


if __name__ == "__main__":
    main()
