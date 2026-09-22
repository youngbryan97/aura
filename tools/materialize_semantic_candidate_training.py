#!/usr/bin/env python3
"""Retain target-blind candidate banks for source-training examples only."""

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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(body: dict) -> str:
    return _sha(json.dumps(body, sort_keys=True, allow_nan=False).encode())


def _plan(model, examples, source_raw: bytes, candidate_raw: bytes,
          manifests: dict, implementation: str, *, max_charts: int,
          max_graphs: int, solve_seconds: float, selected_ids: tuple[str, ...] | None = None,
          selection: dict | None = None) -> dict:
    full = tuple(sorted(item.ir.source_text_sha256 for item in examples if item.split == "train"))
    sources = full if selected_ids is None else selected_ids
    if (not full or len(full) != len(set(full))
            or len(full) != model.training_receipt["training_example_count"]
            or not sources or len(sources) != len(set(sources))
            or tuple(sorted(sources)) != sources or not set(sources) <= set(full)
            or (selection is None) != (sources == full)
            or any(type(value) is not int or value < 1 for value in (max_charts, max_graphs))
            or type(solve_seconds) not in (int, float) or not 0 < solve_seconds <= 60):
        raise ValueError("candidate training needs the complete frozen source cohort and bounds")
    body = {"schema": "aura.semantic_candidate_training_plan.v1",
            "source_ids": sources, "source_population": len(full),
            "source_selection": selection, "pilot_only": selection is not None,
            "split": "train", "validation_used": False,
            "test_used": False, "model_receipt_sha256": model.receipt_sha256,
            "model_basis_sha256": model.model_basis_sha256,
            "source_report_sha256": _sha(source_raw),
            "candidate_report_sha256": _sha(candidate_raw),
            "source_manifest_sha256s": manifests,
            "implementation": implementation,
            "tool_sha256": _sha(Path(__file__).read_bytes()),
            "max_charts": max_charts,
            "max_graphs_per_chart": max_graphs,
            "solve_seconds": float(solve_seconds),
            "labels_available_only_after_bank_generation": True,
            "serving_authority": False}
    return {**body, "plan_sha256": _digest(body)}


def _select_source_ids(examples, folds: dict, *, per_group: int) -> tuple[tuple[str, ...], dict]:
    from core.learning.semantic_program_campaign import _sha as semantic_sha

    body = {key: value for key, value in folds.items() if key != "receipt_sha256"}
    rows = sorted((item.ir.source_text_sha256, item.construction_id, item.contrast_id)
                  for item in examples if item.split == "train")
    if (type(per_group) is not int or per_group < 1
            or folds.get("schema") != "aura.semantic_construction_folds.v1"
            or folds.get("receipt_sha256") != semantic_sha(body)
            or rows != [tuple(row) for row in folds["population"]]
            or set(folds["assignments"]) != {row[0] for row in rows}
            or folds["validation_used"] is not False or folds["test_used"] is not False):
        raise ValueError("source-only selection must match the frozen construction folds")
    groups = {}
    for source, construction, _contrast in rows:
        groups.setdefault(construction, []).append(source)
    if len(groups) != folds["independent_groups"]:
        raise ValueError("construction strata do not match independent fold groups")
    selected = tuple(sorted(source for sources in groups.values()
                            for source in sources[:per_group]))
    selection = {"schema": "aura.semantic_candidate_source_subset.v1",
                 "method": "lowest_source_identity_per_construction",
                 "per_group": per_group, "groups": len(groups),
                 "folds_receipt_sha256": folds["receipt_sha256"],
                 "labels_used_for_selection": False}
    return selected, selection


def _read_or_run(item, model, plan: dict, directory: Path) -> dict:
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.diagnose_semantic_gap_candidate_bank import _audit_row

    source = item.ir.source_text_sha256
    path = directory / "rows" / f"{source}.json"
    if path.exists():
        row = json.loads(path.read_bytes())
    else:
        body = _audit_row(item, model, plan)
        row = {**body, "receipt_sha256": _digest(body)}
        if not atomic_write_bytes_if_absent(
                path, json.dumps(row, sort_keys=True).encode(), mode=0o400):
            raise FileExistsError(path)
    body = {key: value for key, value in row.items() if key != "receipt_sha256"}
    if (row.get("receipt_sha256") != _digest(body)
            or row.get("plan_sha256") != plan["plan_sha256"]
            or row.get("source") != source or row.get("mixed") is not None
            or row["diagnosis"]["split"] != "train"):
        raise ValueError("source-training candidate row identity differs")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-charts", type=int, default=8)
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--solve-seconds", type=float, default=2.)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--per-group", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.per_group < 0 or bool(args.folds) != bool(args.per_group):
        parser.error("stop-after is nonnegative; folds and positive per-group go together")
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

    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    candidate_raw = args.candidate_report.read_bytes()
    if json.loads(candidate_raw).get("candidate") != model.receipt_sha256:
        raise ValueError("candidate report names a different transducer")
    source_raw = args.source_report.read_bytes()
    source_report = json.loads(source_raw)
    manifests = source_report["representation_compatibility"]["source_feature_manifest_sha256s"]
    bundles = [name + "=" + str(args.feature_root / name) for name in manifests]
    examples = load_source_examples(model, source_report, bundles)
    training = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    selected, selection = (None, None)
    if args.folds:
        selected, selection = _select_source_ids(
            examples, json.loads(args.folds.read_bytes()), per_group=args.per_group)
    implementation = validation_implementation_identity()
    plan = _plan(model, examples, source_raw, candidate_raw, manifests, implementation,
                 max_charts=args.max_charts, max_graphs=args.max_graphs,
                 solve_seconds=args.solve_seconds, selected_ids=selected, selection=selection)
    (args.directory / "rows").mkdir(parents=True, exist_ok=True)
    rows = []
    for source in plan["source_ids"]:
        if args.stop_after and len(rows) == args.stop_after:
            break
        rows.append(_read_or_run(training[source], model, plan, args.directory))
        print(json.dumps({"stage": "source_bank", "completed": len(rows),
                          "total": len(plan["source_ids"]), "source": source,
                          "failure_stage": rows[-1]["diagnosis"]["failure_stage"]}), flush=True)
    if (implementation != validation_implementation_identity()
            or plan["tool_sha256"] != _sha(Path(__file__).read_bytes())):
        raise ValueError("candidate source implementation changed during acquisition")
    if len(rows) != len(plan["source_ids"]):
        print(json.dumps({"stage": "partial", "observed": len(rows),
                          "population": len(plan["source_ids"])}), flush=True)
        return
    body = {"schema": "aura.semantic_candidate_training.v1", "plan": plan,
            "pilot_only": plan["pilot_only"],
            "observed": len(rows), "population": len(plan["source_ids"]),
            "failure_stages": dict(sorted(Counter(
                row["diagnosis"]["failure_stage"] for row in rows).items())),
            "correct_reachable": sum(row["diagnosis"]["correct_reachable"] is True for row in rows),
            "selected_equivalent": sum(row["diagnosis"]["selected_semantic_status"] == "equivalent"
                                       for row in rows),
            "row_receipts": {row["source"]: row["receipt_sha256"] for row in rows},
            "serving_authority": False}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    path = args.directory / "report.json"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError("published source bank report differs")
    print(json.dumps({"stage": "complete", "observed": len(rows),
                      "correct_reachable": body["correct_reachable"],
                      "selected_equivalent": body["selected_equivalent"]}), flush=True)


if __name__ == "__main__":
    main()
