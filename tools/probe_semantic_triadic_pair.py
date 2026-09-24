#!/usr/bin/env python3
"""Compare a source-fold triadic candidate with its parent and lesion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def paired_accuracy(incumbent: dict, candidate: dict, lesion: dict) -> dict:
    """Count exact matched rows without changing either arm's grader."""
    names = ("incumbent", "candidate", "lesion")
    arms = (incumbent, candidate, lesion)
    source_orders = [tuple(row["source_text_sha256"] for row in arm["rows"])
                     for arm in arms]
    if (not source_orders[0] or any(order != source_orders[0] for order in source_orders[1:])
            or len(set(source_orders[0])) != len(source_orders[0])):
        raise ValueError("paired triadic decode arms differ in source identity or order")
    result = {"sources": len(source_orders[0])}
    for key in ("answer_exact", "program_exact", "argument_exact"):
        rows = [tuple(bool(row[key]) for row in arm["rows"]) for arm in arms]
        result[key] = {
            **{name: sum(values) for name, values in zip(names, rows, strict=True)},
            "candidate_only": sum(new and not old for old, new in zip(rows[0], rows[1], strict=True)),
            "incumbent_only": sum(old and not new for old, new in zip(rows[0], rows[1], strict=True)),
            "candidate_and_lesion": sum(new and cut for new, cut in zip(rows[1], rows[2], strict=True)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tools.probe_semantic_proposer_crossfit import crossfit_partition
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_shared_evaluation import (
        evaluate_shared_semantic_program_transducer,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    parent_raw = args.parent.read_bytes()
    candidate_raw = args.candidate.read_bytes()
    parent = compositional_semantic_program_transducer_from_dict(json.loads(parent_raw))
    candidate = compositional_semantic_program_transducer_from_dict(json.loads(candidate_raw))
    report = json.loads(args.candidate_report.read_bytes())
    source_report = json.loads(args.source_report.read_bytes())
    folds = json.loads(args.folds.read_bytes())
    if (report.get("parent_file_sha256") != hashlib.sha256(parent_raw).hexdigest()
            or report.get("source_fold_candidate_file_sha256")
            != hashlib.sha256(candidate_raw).hexdigest()
            or report.get("fold") != args.fold
            or candidate.model_basis_sha256 != parent.model_basis_sha256):
        raise ValueError("paired triadic candidate differs from source-fold proof")
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(parent, source_report, bundles)
    fit, calibration, held = crossfit_partition(examples, folds, args.fold, all_held=True)
    fit_receipt = candidate.training_receipt.get("triadic_binding_fit", {})
    if (fit_receipt.get("parent_transducer_receipt_sha256") != parent.receipt_sha256
            or fit_receipt.get("training_example_ids_sha256")
            != _sha(sorted(item.ir.source_text_sha256 for item in fit))
            or fit_receipt.get("validation_example_ids_sha256")
            != _sha(sorted(item.ir.source_text_sha256 for item in calibration))
            or fit_receipt.get("source_fold_provenance", {}).get("held_ids_sha256")
            != _sha(sorted(item.ir.source_text_sha256 for item in held))):
        raise ValueError("paired triadic fit overlaps or differs from held sources")
    if args.limit:
        if not 0 < args.limit <= len(held):
            raise ValueError("paired triadic limit exceeds held sources")
        held = sorted(held, key=lambda item: hashlib.sha256(
            item.ir.source_text_sha256.encode("ascii")).hexdigest())[:args.limit]
    arms = {
        "incumbent": evaluate_shared_semantic_program_transducer(
            parent, held, split="train", arm="incumbent").to_dict(),
        "candidate": evaluate_shared_semantic_program_transducer(
            candidate, held, split="train", arm="candidate").to_dict(),
        "lesion": evaluate_shared_semantic_program_transducer(
            candidate.triadic_binding_lesion(), held, split="train", arm="lesion").to_dict(),
    }
    body = {"schema": "aura.semantic_triadic_pair_probe.v1",
            "parent_file_sha256": hashlib.sha256(parent_raw).hexdigest(),
            "candidate_file_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "candidate_report_receipt_sha256": report["receipt_sha256"],
            "fold": args.fold, "held_ids_sha256": _sha(sorted(
                item.ir.source_text_sha256 for item in held)),
            "paired": paired_accuracy(*(arms[name] for name in ("incumbent", "candidate", "lesion"))),
            "arms": arms, "parent_model_trained_on_held_sources": True,
            "qualification_evidence": False, "serving_authority": False}
    payload = json.dumps({**body, "receipt_sha256": _sha(body)}, sort_keys=True,
                         separators=(",", ":")).encode("ascii") + b"\n"
    if not atomic_write_bytes_if_absent(args.output, payload, mode=0o400):
        if args.output.read_bytes() != payload:
            raise ValueError("paired triadic report already exists with different evidence")
    print(json.dumps({"output": str(args.output), "paired": body["paired"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
