#!/usr/bin/env python3
"""Regrade archived public programs without loading or rerunning the backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def adjudication_contract():
    paths = ("tools/adjudicate_semantic_grounded_development.py",
        "core/learning/procedure_induction.py", "core/learning/semantic_joint_graph_learning.py",
        "core/learning/semantic_program_floor.py", "core/learning/semantic_program_ir.py",
        "core/learning/semantic_program_compositional_campaign.py")
    return {"schema": "aura.grounded_development_adjudication_contract.v1",
        "sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        "execution_semantics": "python_reference_program_closed_value_algebra",
        "undefined_is_unmeasured": True, "numeric_equality_is_not_program_equivalence": True,
        "public_answer_generation_measured": False, "qualification_evidence": False}


def grade_archive(item, row):
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_joint_graph_learning import align_source_input_registers
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
    from core.learning.semantic_program_ir import normalize_semantic_value, semantic_program_ir_from_dict
    from tools.semantic_grounded_development_archive import digest

    identity = item.ir.source_text_sha256
    if (row["source_id"] != identity or tuple(row["source_token_ids"]) != item.ir.source_token_ids
            or digest(row["public_inputs"]) != digest(item.public_inputs)
            or row["model_basis_sha256"] != item.ir.model_basis_receipt_sha256):
        raise ValueError("adjudication source differs from its measured public request")
    record = {"source_id": identity, "construction": item.construction_id, "topology": item.topology_id,
        "archive_receipt_sha256": row["receipt_sha256"], "program_exact": False,
        "program_equivalent": False, "answer_correct": None, "refusal": row["decode"]["refusal"]}
    if (row["decoded_ir"] is None) != bool(row["decode"]["refusal"]):
        raise ValueError("adjudication refusal contradicts archived program")
    if row["decoded_ir"] is None:
        return {**record, "status": "decode_refused"}
    ir = semantic_program_ir_from_dict(row["decoded_ir"])
    if (ir.source_text_sha256 != identity or ir.source_token_ids != item.ir.source_token_ids
            or ir.model_basis_receipt_sha256 != item.ir.model_basis_receipt_sha256
            or ir.n_inputs != len(item.public_inputs)):
        raise ValueError("adjudication program differs from measured source")
    try:
        instructions, _mapping = align_source_input_registers(item, ir.input_spans)
    except ValueError:
        return {**record, "status": "input_anchors_unaligned"}
    target = Program(len(item.public_inputs), tuple(Instruction(value.op, value.args) for value in instructions))
    actual = ir.to_program()
    exact = actual == target
    equivalent = exact or semantic_programs_structurally_equivalent(actual, target)
    record.update(program_exact=exact, program_equivalent=equivalent,
        actual_program=[[value.op, list(value.args)] for value in actual.instructions],
        annotated_program=[[value.op, list(value.args)] for value in target.instructions])
    try:
        expected = normalize_semantic_value(target.run(item.public_inputs))
    except (ArithmeticError, ValueError, TypeError, IndexError, RuntimeError) as exc:
        return {**record, "status": "annotation_execution_unmeasured", "annotation_error": type(exc).__name__}
    try:
        produced = normalize_semantic_value(actual.run(item.public_inputs))
    except (ArithmeticError, ValueError, TypeError, IndexError, RuntimeError) as exc:
        return {**record, "status": "decoded_execution_failed", "answer_correct": False,
            "execution_error": type(exc).__name__, "annotated_value": expected}
    return {**record, "answer_correct": produced == expected, "decoded_value": produced,
        "annotated_value": expected, "status": "equivalent_interpretation" if equivalent else "program_mismatch"}


def adjudicate_rows(report, examples, rows):
    from tools.semantic_grounded_development_archive import ARMS, digest

    by_id = {item.ir.source_text_sha256: item for item in examples}
    identities = report["plan"]["held_ids"]
    if (not identities or len(set(identities)) != len(identities) or len(by_id) != len(examples)
            or any(identity not in by_id for identity in identities) or set(rows) != set(ARMS)):
        raise ValueError("adjudication population differs from measured report")
    measured, metrics, per_construction = {}, {}, {}
    for arm in ARMS:
        if set(rows[arm]) != set(identities):
            raise ValueError("adjudication archive population is incomplete")
        measured[arm] = [grade_archive(by_id[identity], rows[arm][identity]) for identity in identities]
        expected = report["comparison"]["candidates"][arm]
        if ([row["program_exact"] for row in measured[arm]] != expected["program_correct"]
                or [row["program_equivalent"] for row in measured[arm]] != expected["program_equivalent_correct"]):
            raise ValueError("independently regraded programs contradict reported outcomes")
        metrics[arm] = {"program_exact": sum(row["program_exact"] for row in measured[arm]),
            "program_equivalent": sum(row["program_equivalent"] for row in measured[arm]),
            "answer_measured": sum(row["answer_correct"] is not None for row in measured[arm]),
            "answer_correct": sum(row["answer_correct"] is True for row in measured[arm]),
            "refusals": sum(bool(row["refusal"]) for row in measured[arm])}
        groups = defaultdict(list)
        for row in measured[arm]:
            groups[row["construction"]].append(row)
        per_construction[arm] = {name: {"population": len(values),
            "program_equivalent": sum(value["program_equivalent"] for value in values),
            "answer_measured": sum(value["answer_correct"] is not None for value in values),
            "answer_correct": sum(value["answer_correct"] is True for value in values)}
            for name, values in sorted(groups.items())}
    body = {"schema": "aura.grounded_development_adjudication.v1",
        "adjudication_contract": adjudication_contract(),
        "evaluation_report_receipt_sha256": report["receipt_sha256"], "plan_sha256": report["plan"]["plan_sha256"],
        "population": len(identities), "metrics": metrics, "per_construction": per_construction,
        "source_outcomes": measured, "source_anchors_independently_regraded": True,
        "semantic_interpretations_independently_replayed": True, "backbone_loaded": False,
        "public_answer_generation_measured": False, "g03_complete": False,
        "fresh_transfer_proven": False, "qualification_evidence": False, "serving_authority": False}
    return {**body, "receipt_sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "directory", "parent", "source-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    args = parser.parse_args()
    from core.governance_context import local_internal_governed_scope
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.runtime.file_read_gateway import read_stable_bytes
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples
    from tools.semantic_grounded_development_archive import MAX_ROW_BYTES, verify_archived_decode
    from tools.verify_semantic_grounded_evaluation import verify

    configure_refit_environment(args.output)
    verified = verify(args.report, args.directory)
    report = json.loads(read_stable_bytes(args.report, max_bytes=MAX_ROW_BYTES))
    if report["plan"].get("archive_decodes") is not True or args.output.exists():
        raise ValueError("independent adjudication needs complete archives and a fresh output")
    native = json.loads(read_stable_bytes(args.directory / "report.json", max_bytes=MAX_ROW_BYTES))["native_contract"]
    raw = {name: read_stable_bytes(path, max_bytes=MAX_ROW_BYTES) for name, path in (
        ("parent", args.parent), ("source_report", args.source_report))}
    if any(hashlib.sha256(value).hexdigest() != native["source_basis"][name + "_sha256"] for name, value in raw.items()):
        raise ValueError("adjudication source basis differs from the measured fit")
    parent = compositional_semantic_program_transducer_from_dict(json.loads(raw["parent"]))
    examples = load_source_examples(parent, json.loads(raw["source_report"]), args.bundle)
    rows = {}
    for arm in report["plan"]["arms"]:
        rows[arm] = {}
        for identity in report["plan"]["held_ids"]:
            compact = report["decodes"][arm][identity]
            verify_archived_decode(args.report.with_suffix(".rows"), plan_sha256=report["plan"]["plan_sha256"],
                arm=arm, identity=identity, compact=compact)
            row = json.loads(read_stable_bytes(args.report.with_suffix(".rows") / compact["decode_archive"]["filename"],
                max_bytes=MAX_ROW_BYTES))
            # Retain public programs, not the potentially large learned edge arrays.
            rows[arm][identity] = {key: value for key, value in row.items() if key != "decode"}
            rows[arm][identity]["decode"] = {"refusal": row["decode"]["refusal"]}
    result = adjudicate_rows(report, examples, rows)
    result["artifact_verification_receipt_sha256"] = verified["receipt_sha256"]
    from tools.semantic_grounded_development_archive import digest
    result["receipt_sha256"] = digest({k: v for k, v in result.items() if k != "receipt_sha256"})
    with local_internal_governed_scope("grounded_development_adjudication", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(args.output, json.dumps(result, indent=2, allow_nan=False).encode(),
                source="grounded_development_adjudication", mode=0o400):
            raise FileExistsError(args.output)
    print(json.dumps({"receipt_sha256": result["receipt_sha256"], "population": result["population"],
        "metrics": result["metrics"], "g03_complete": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
