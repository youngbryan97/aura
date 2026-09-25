#!/usr/bin/env python3
"""Train a selector on nested out-of-fold program banks, without held leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _verified_pair(directory: Path) -> tuple[dict, dict]:
    plan = json.loads((directory / "plan.json").read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    if (plan.get("schema") != "aura.semantic_proposer_crossfit_plan.v1"
            or report.get("schema") != "aura.semantic_proposer_crossfit.v1"
            or plan.get("plan_sha256") != _digest({
                key: value for key, value in plan.items() if key != "plan_sha256"})
            or report.get("receipt_sha256") != _digest({
                key: value for key, value in report.items() if key != "receipt_sha256"})
            or report.get("plan_sha256") != plan["plan_sha256"]
            or set(report.get("row_receipts", {})) != set(plan["held_ids"])):
        raise ValueError("nested ranker bank lacks a valid signed plan and report")
    return plan, report


def validate_nested_provenance(
    outer_plan: dict, outer_report: dict, inner_pairs: list[tuple[dict, dict]],
) -> dict[str, int]:
    """No proposal model or selector update may see an outer-held source."""
    outer_held = set(outer_plan["held_ids"])
    outer_train = set(outer_plan["fit_ids"]) | set(outer_plan["calibration_ids"])
    if (not outer_held or not outer_train or outer_held & outer_train
            or set(outer_report["row_receipts"]) != outer_held
            or outer_plan.get("outer_fold") is not None):
        raise ValueError("outer bank is not a disjoint source-construction holdout")
    assignments = {}
    inner_folds = set()
    inner_receipts = set()
    inner_seeds = set()
    for plan, report in inner_pairs:
        fit, calibration, held = (set(plan[key]) for key in
                                  ("fit_ids", "calibration_ids", "held_ids"))
        if (plan.get("outer_fold") != outer_plan["fold"]
                or plan.get("fold") in inner_folds
                or set(plan.get("outer_excluded_ids", ())) != outer_held
                or plan.get("source_report_sha256") != outer_plan["source_report_sha256"]
                or plan.get("parent_receipt_sha256") != outer_plan["parent_receipt_sha256"]
                or plan.get("folds_sha256") != outer_plan["folds_sha256"]
                or plan.get("inner_folds_receipt_sha256") is None
                or type(plan.get("inner_fold_seed")) is not int
                or not fit or not calibration or not held
                or fit & calibration or fit & held or calibration & held
                or fit | calibration | held != outer_train
                or set(report["row_receipts"]) != held
                or outer_held & (fit | calibration | held)
                or set(assignments) & held):
            raise ValueError("inner proposal bank crosses selector or proposer holdout")
        inner_folds.add(plan["fold"])
        inner_receipts.add(plan["inner_folds_receipt_sha256"])
        inner_seeds.add(plan["inner_fold_seed"])
        assignments.update({source: plan["fold"] for source in held})
    if (len(inner_folds) != 3 or len(inner_receipts) != 1 or len(inner_seeds) != 1
            or set(assignments) != outer_train):
        raise ValueError("nested proposal banks do not cover outer training sources")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--outer-bank", type=Path, required=True)
    parser.add_argument("--inner-bank", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--pilot-rows", type=int, default=0)
    parser.add_argument("--argument-evidence", action="store_true")
    args = parser.parse_args()
    if len(args.inner_bank) != 3 or args.epochs < 1 or args.pilot_rows < 0:
        parser.error("three inner banks and positive training epochs are required")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output_directory / "report.json")
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from core.learning.semantic_candidate_ranker import ContextualProgramRanker, candidate_set_loss
    from core.learning.semantic_construction_folds import construction_folds
    from core.learning.semantic_program_campaign import _sha as semantic_sha
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_request_context import RequestContextConfig
    from core.learning.semantic_span_pointer import _hidden_array
    from tools.evaluate_semantic_candidate_ranker import _evaluate, _rankable_or_none, _read_bank

    outer_plan, outer_report = _verified_pair(args.outer_bank)
    inner_pairs = [_verified_pair(path) for path in args.inner_bank]
    assignments = validate_nested_provenance(outer_plan, outer_report, inner_pairs)
    source_raw = args.source_report.read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != outer_plan["source_report_sha256"]:
        raise ValueError("source report differs from nested bank basis")
    parent = compositional_semantic_program_transducer_from_dict(
        json.loads(args.parent.read_bytes()))
    if parent.receipt_sha256 != outer_plan["parent_receipt_sha256"]:
        raise ValueError("parent transducer differs from nested bank basis")
    source = json.loads(source_raw)
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(parent, source, bundles)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    if not (set(assignments) | set(outer_plan["held_ids"])) <= set(items):
        raise ValueError("source examples do not cover nested evaluation")
    inner_seed = inner_pairs[0][0]["inner_fold_seed"]
    expected_inner = construction_folds(
        [items[identity] for identity in sorted(assignments)], count=3, seed=inner_seed)
    if (expected_inner["receipt_sha256"] != inner_pairs[0][0]["inner_folds_receipt_sha256"]
            or expected_inner["assignments"] != assignments):
        raise ValueError("nested proposal banks do not replay from source constructions")
    rows = {}
    for directory, (plan, report) in [
            (args.outer_bank, (outer_plan, outer_report)),
            *zip(args.inner_bank, inner_pairs, strict=True)]:
        candidate = compositional_semantic_program_transducer_from_dict(
            json.loads((directory / "candidate.json").read_bytes()))
        if candidate.receipt_sha256 != report["candidate_receipt_sha256"]:
            raise ValueError("proposal model differs from its frozen bank")
        if candidate.training_receipt["training_example_ids_sha256"] != semantic_sha(
                sorted(plan["fit_ids"])):
            raise ValueError("proposal model fit differs from its source partition")
        for identity, expected in report["row_receipts"].items():
            rows[identity] = _read_bank(
                directory / "rows" / f"{identity}.json", source=identity,
                plan_sha=plan["plan_sha256"], model_receipt=candidate.receipt_sha256,
                expected_receipt=expected)
    train = sorted(assignments)
    held = sorted(outer_plan["held_ids"])
    if args.pilot_rows:
        train, held = train[:args.pilot_rows], held[:args.pilot_rows]
    cases = {identity: case for identity in train if (case := _rankable_or_none(
        items[identity], rows[identity], preserve_evidence=args.argument_evidence)) is not None}
    train = sorted(cases)
    torch.manual_seed(20260925 + outer_plan["fold"])
    np.random.seed(20260925 + outer_plan["fold"])
    config = RequestContextConfig(parent.hidden_size, width=64, heads=4, layers=1,
                                  feature_scaling="unit_variance")
    ranker = ContextualProgramRanker(
        config, identity_bindings=args.argument_evidence,
        argument_evidence=args.argument_evidence,
        retain_evidence_variants=args.argument_evidence)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4, weight_decay=0.01)
    updates = 0
    for epoch in range(args.epochs):
        ranker.train()
        losses = []
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            20260925 + outer_plan["fold"] + epoch)).tolist()
        for index in order:
            identity = train[index]
            item = items[identity]
            (programs, labels, keys), spans, kinds, anchors, *evidence = cases[identity]
            if not any(labels):
                continue
            features = torch.from_numpy(_hidden_array(item.hidden_states)).float()
            kwargs = ({"argument_spans": evidence[0], "definition_spans": evidence[1]}
                      if args.argument_evidence else {})
            scores = ranker(features, spans, kinds, programs, operation_spans=anchors, **kwargs)
            loss = candidate_set_loss(scores, labels,
                                      program_keys=keys if args.argument_evidence else None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        print(json.dumps({"stage": "epoch", "epoch": epoch + 1, "updates": updates,
                          "mean_loss": sum(losses) / len(losses) if losses else None}), flush=True)
    if not updates:
        raise ValueError("nested source banks contain no witnessed positive training view")
    evaluation = _evaluate(ranker, items, rows, held)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    weights = args.output_directory / "ranker.safetensors"
    if weights.exists():
        raise FileExistsError(weights)
    save_file({key: value.detach().cpu().contiguous() for key, value in ranker.state_dict().items()},
              weights)
    body = {"schema": "aura.semantic_nested_ranker_source_fold.v1",
            "outer_fold": outer_plan["fold"], "pilot_only": bool(args.pilot_rows),
            "serving_authority": False, "qualification_evidence": False,
            "source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
            "outer_bank_receipt_sha256": outer_report["receipt_sha256"],
            "inner_bank_receipt_sha256s": sorted(report["receipt_sha256"]
                                                 for _plan, report in inner_pairs),
            "train_sources": train, "held_sources": held, "epochs": args.epochs,
            "updates": updates, "argument_evidence": args.argument_evidence,
            "config": vars(config), "evaluation": evaluation,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (args.output_directory / "report.json").write_text(json.dumps(
        {**body, "receipt_sha256": _digest(body)}, sort_keys=True))
    print(json.dumps({"stage": "complete", "outer_fold": outer_plan["fold"],
                      "population": evaluation["population"],
                      "ranker_correct": evaluation["ranker_correct"],
                      "incumbent_correct": evaluation["incumbent_correct"]}), flush=True)


if __name__ == "__main__":
    main()
