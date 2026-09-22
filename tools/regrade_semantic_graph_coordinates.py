#!/usr/bin/env python3
"""Regrade frozen programs in common input coordinates without another decode."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--graph-report", type=Path, required=True)
    parser.add_argument("--baseline-candidate", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("regrading requires a new output")
    from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples

    configure_refit_environment(args.output)
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_graph_coordinates import reanchor_program_inputs
    from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    original = json.loads(args.graph_report.read_text())
    if original["plan"].get("input_coordinates"):
        raise ValueError("report already declares its input coordinates")
    original_identity = hashlib.sha256(args.graph_report.read_bytes()).hexdigest()
    for path, digest in original["plan"]["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"recorded inference source changed: {path}")
    root = args.evidence_root
    parent = compositional_semantic_program_transducer_from_dict(json.loads(
        (root / "semantic-source-fit-20260915/candidate.json").read_text()))
    if parent.receipt_sha256 != original["plan"]["parent_receipt"]:
        raise ValueError("parent identity changed")
    source_report = json.loads((root / "semantic-source-fit-20260915/report.json").read_text())
    examples = load_source_examples(parent, source_report, [name + "=" + str(
        root / "semantic-source-reacquisition-20260915/features" / name)
        for name in source_report["representation_compatibility"]["source_feature_manifest_sha256s"]])
    split = original["plan"].get("evaluation_split", "train")
    if split not in {"train", "validation"}:
        raise ValueError("regrading accepts development splits only")
    source = {x.ir.source_text_sha256: x for x in examples if x.split == split}
    arms = {arm["name"]: arm for arm in original["plan"]["arms"]}
    models = {}
    for path in args.baseline_candidate:
        name = path.parent.name + "-baseline"
        if name not in arms or hashlib.sha256(path.read_bytes()).hexdigest() != arms[name]["report_sha256"]:
            raise ValueError("baseline identity differs from measured arm")
        models[name] = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    if any(name.endswith("-baseline") and name not in models for name in arms):
        raise ValueError("every baseline requires its measured grounding model")
    expected = {(identity, name) for identity in original["plan"]["source_ids"] for name in arms}
    observed = [(row["source"], row["arm"]) for row in original["rows"]]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("original report is not complete and paired")
    files = [Path(__file__).resolve(), ROOT / "core/learning/semantic_graph_coordinates.py",
             ROOT / "core/learning/semantic_graph_counterexamples.py"]
    plan = {**original["plan"], "input_coordinates": "annotated_source_anchors_v1",
            "regrading": {"original_path": str(args.graph_report), "original_sha256": original_identity,
                "decode_repeated": False, "grounding_recomputed_from_bound_models": True,
                "historical_source_coverage_expanded_retroactively": False,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in files}}}
    if not atomic_write_bytes_if_absent(args.output.with_suffix(".plan.json"),
                                       json.dumps(plan, sort_keys=True).encode()):
        raise ValueError("regrading plan already exists")
    rows = []
    for row in original["rows"]:
        item = source[row["source"]]
        updated = {**row, "original_program": row["program"], "original_comparison": row["comparison"],
                   "coordinate_status": "no_program"}
        if row["program"] is not None:
            payload = row["program"]
            program = Program(len(item.public_inputs), tuple(Instruction(op, tuple(arguments))
                              for op, arguments in payload["instructions"]))
            if program.sha() != payload["sha"]:
                raise ValueError("frozen program identity differs")
            model = models.get(row["arm"], parent)
            try:
                spans, _, _ = model._runtime_input_grounding(
                    item.ir.source_token_ids, item.hidden_states, item.public_inputs)
                normalized = reanchor_program_inputs(program, from_spans=spans,
                    to_spans=item.ir.input_spans, from_inputs=item.public_inputs, to_inputs=item.public_inputs)
            except ValueError as exc:
                updated.update(coordinate_status="unaligned", program=None,
                    comparison={"status": "unknown", "reason": str(exc)})
            else:
                updated.update(coordinate_status="aligned", program=normalized.to_dict(),
                    comparison=compare_program_meanings(item.ir.to_program(), normalized,
                        counterfactual_inputs(item.public_inputs, count=16)),
                    runtime_input_spans=[[s.start, s.end] for s in spans],
                    source_input_spans=[[s.start, s.end] for s in item.ir.input_spans])
        rows.append(updated)
    for path, digest in {**original["plan"]["source_sha256"],
                         **plan["regrading"]["source_sha256"]}.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"regrading source changed during measurement: {path}")
    summary = {name: dict(Counter(row["comparison"]["status"] for row in rows if row["arm"] == name))
               for name in arms}
    changes = dict(Counter(row["arm"] for row in rows
                          if row["comparison"]["status"] != row["original_comparison"]["status"]))
    report = {"plan": plan, "summary": summary, "changed_grades": changes, "rows": rows}
    if not atomic_write_bytes_if_absent(args.output, json.dumps(report, sort_keys=True).encode()):
        raise ValueError("regrading output already exists")
    print(json.dumps({"summary": summary, "changed_grades": changes}), flush=True)


if __name__ == "__main__":
    main()
