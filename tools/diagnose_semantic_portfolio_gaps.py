#!/usr/bin/env python3
"""Attribute frozen portfolio misses to source-verified program differences."""

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


def _source_examples(feature_root: Path, wanted: set[str], split: str):
    from core.learning.semantic_program_feature_materialization import (
        rebuild_semantic_feature_selection,
    )

    examples, manifests = {}, {}
    for path in sorted(feature_root.glob("*/manifest.json")):
        manifest = json.loads(path.read_bytes())
        _, corpus = rebuild_semantic_feature_selection(manifest)
        manifests[path.parent.name] = manifest["manifest_sha256"]
        for example in corpus:
            identity = _sha(example.source_text.encode("utf-8"))
            if identity not in wanted:
                continue
            if identity in examples or example.split != split:
                raise ValueError("source identity is duplicated or outside the portfolio split")
            examples[identity] = example
    if set(examples) != wanted:
        raise ValueError(f"source corpus lacks {len(wanted - set(examples))} portfolio tasks")
    return examples, manifests


def _graph_rows(plan: dict):
    expected_ids = set(plan["source_ids"])
    rows, digests = {}, {}
    for name, declared in plan["graph_reports"].items():
        path = Path(name)
        data = path.read_bytes()
        digest = _sha(data)
        if digest != declared:
            raise ValueError("frozen graph report digest differs")
        report = json.loads(data)
        graph_plan = report["plan"]
        if (graph_plan["input_coordinates"] != "annotated_source_anchors_v1"
                or set(graph_plan["source_ids"]) != expected_ids
                or graph_plan.get("evaluation_split", "train") != plan["evaluation_split"]):
            raise ValueError("graph report population or coordinates differ")
        for row in report["rows"]:
            key = row["source"], row["arm"]
            if key in rows or key[0] not in expected_ids:
                raise ValueError("graph reports repeat or add a task")
            rows[key] = row
        digests[name] = digest
    if {source for source, _ in rows} != expected_ids:
        raise ValueError("graph reports omit portfolio tasks")
    return rows, digests


def classify_program_difference(gold, predicted: dict | None) -> str:
    """Localize an error without calling a syntactic match semantic proof."""
    if predicted is None:
        return "no_program"
    expected = tuple((row.instruction.op, row.instruction.args) for row in gold.instructions)
    actual = tuple((op, tuple(args)) for op, args in predicted["instructions"])
    if len(actual) != len(expected):
        return "depth"
    if tuple(op for op, _ in actual) != tuple(op for op, _ in expected):
        return "operation_or_order"
    if tuple(args for _, args in actual) != tuple(args for _, args in expected):
        return "argument_binding"
    return "syntactic_match_but_semantic_failure"


def _program_from_record(record: dict | None, n_inputs: int):
    from core.learning.procedure_induction import Instruction, Program

    if record is None:
        return None
    program = Program(n_inputs, tuple(Instruction(op, tuple(args))
                                      for op, args in record["instructions"]))
    if record.get("sha") != program.sha():
        raise ValueError("frozen proposal program identity differs")
    return program


def assess_composition(gaps: dict, examples: dict, *, budget: int,
                       examined_limit: int) -> dict:
    """Grade label-blind recombinations only after their programs are fixed."""
    from core.learning.semantic_graph_counterexamples import (
        compare_program_meanings, counterfactual_inputs,
    )
    from core.learning.semantic_program_composition import compose_semantic_programs

    if type(budget) is not int or budget < 1 or type(examined_limit) is not int or examined_limit < 1:
        raise ValueError("composition assessment needs finite positive allowances")
    rows = []
    for gap in gaps["failures"]:
        example = examples[gap["source"]]
        proposals = {name: _program_from_record(record, example.program.n_inputs)
                     for name, record in gap["method_programs"].items()}
        composed = compose_semantic_programs(
            proposals, tuple(example.inputs), max_candidates=budget,
            max_examined=examined_limit,
        )
        probes = counterfactual_inputs(tuple(example.inputs), count=16)
        candidates = []
        for candidate in composed.candidates:
            comparison = compare_program_meanings(example.program, candidate.program, probes)
            candidates.append({"program_sha256": candidate.program.sha(),
                               "instruction_sources": candidate.instruction_sources,
                               "comparison_status": comparison["status"]})
        rows.append({"source": gap["source"], "original_gap": gap["kind"],
                     "examined": composed.examined,
                     "search_exhausted": composed.search_exhausted,
                     "candidates": candidates})
    return {"schema": "aura.semantic_portfolio_composition_diagnosis.v1",
            "development_labels_used_only_after_generation": True,
            "serving_authority": False, "selection_changed": False,
            "budget": budget, "examined_limit": examined_limit,
            "summary": {
                "candidate_unavailable_proved_recovered": sum(
                    row["original_gap"] == "candidate_unavailable"
                    and any(candidate["comparison_status"] == "equivalent"
                            for candidate in row["candidates"])
                    for row in rows),
                "candidate_unavailable_total": sum(
                    row["original_gap"] == "candidate_unavailable" for row in rows),
                "incomplete_search_rows": sum(not row["search_exhausted"] for row in rows),
            }, "rows": rows}


def diagnose(portfolio: dict, examples: dict, graph_rows: dict, *, portfolio_sha: str,
             manifest_shas: dict, graph_shas: dict) -> dict:
    plan = portfolio["plan"]
    source_ids = plan["source_ids"]
    if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(examples):
        raise ValueError("portfolio population differs from rebuilt source corpus")
    if (len(portfolio["rows"]) != len(source_ids)
            or set(source_ids) != {row["source"] for row in portfolio["rows"]}):
        raise ValueError("portfolio rows differ from its declared population")
    expected_graph_rows = {(row["source"], arm) for row in portfolio["rows"]
                           for arm in row["method_correctness"]}
    if set(graph_rows) != expected_graph_rows:
        raise ValueError("graph methods differ from portfolio proposals")
    failures, counts = [], Counter()
    for row in portfolio["rows"]:
        if row["correct"] != row["method_correctness"][row["selected"]]:
            raise ValueError("selected correctness differs from retained method result")
        if row["correct"]:
            continue
        identity = row["source"]
        example = examples[identity]
        methods, proposals = {}, {}
        for arm, correct in row["method_correctness"].items():
            graph = graph_rows[identity, arm]
            if correct != (graph["comparison"]["status"] == "equivalent"):
                raise ValueError("portfolio and graph correctness disagree")
            proposals[arm] = graph["program"]
            if not correct:
                methods[arm] = classify_program_difference(example, graph["program"])
                counts[(arm, methods[arm])] += 1
        kind = "candidate_unavailable" if not row["oracle_available"] else "selection_miss"
        failures.append({
            "source": identity,
            "kind": kind,
            "construction": example.construction_id,
            "topology": example.topology_id,
            "source_text": example.source_text,
            "gold_program": example.program.to_dict(),
            "method_programs": proposals,
            "method_failures": methods,
        })
    summary = {
        "candidate_unavailable": sum(row["kind"] == "candidate_unavailable" for row in failures),
        "selection_miss": sum(row["kind"] == "selection_miss" for row in failures),
        "by_construction": dict(sorted(Counter(row["construction"] for row in failures).items())),
        "by_method_and_difference": {f"{arm}:{kind}": count
                                     for (arm, kind), count in sorted(counts.items())},
    }
    if (summary["candidate_unavailable"] != portfolio["summary"]["candidate_unavailable"]
            or summary["selection_miss"] != portfolio["summary"]["available_but_not_selected"]):
        raise ValueError("attribution does not reproduce portfolio summary")
    return {"schema": "aura.semantic_portfolio_gap_diagnosis.v1",
            "portfolio_sha256": portfolio_sha,
            "source_manifest_sha256s": manifest_shas, "graph_report_sha256s": graph_shas,
            "validation_labels_used_only_for_diagnosis": True, "serving_authority": False,
            "promotion_authorized": False, "summary": summary, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composition-budget", type=int, default=0)
    parser.add_argument("--composition-examined-limit", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output already exists")
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    data = args.portfolio.read_bytes()
    portfolio = json.loads(data)
    plan = portfolio["plan"]
    if plan.get("evaluation_split") not in {"train", "validation"}:
        raise ValueError("diagnosis accepts only development splits")
    examples, manifests = _source_examples(
        args.feature_root, set(plan["source_ids"]), plan["evaluation_split"])
    rows, digests = _graph_rows(plan)
    report = diagnose(portfolio, examples, rows, portfolio_sha=_sha(data),
                      manifest_shas=manifests, graph_shas=digests)
    if args.composition_budget:
        report["composition_assessment"] = assess_composition(
            report, examples, budget=args.composition_budget,
            examined_limit=args.composition_examined_limit)
        report["schema"] = "aura.semantic_portfolio_gap_diagnosis.v2"
    if not atomic_write_bytes_if_absent(args.output, json.dumps(report, sort_keys=True).encode()):
        raise FileExistsError(args.output)
    print(json.dumps(report["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
