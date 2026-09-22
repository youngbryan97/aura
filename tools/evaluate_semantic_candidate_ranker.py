#!/usr/bin/env python3
"""Source-fold selection experiment over immutable, target-blind program banks."""

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


def _read_bank(path: Path, *, source: str, plan_sha: str, model_receipt: str,
               expected_receipt: str) -> dict:
    from core.learning.semantic_program_campaign import _sha

    row = json.loads(path.read_bytes())
    body = {key: value for key, value in row.items() if key != "receipt_sha256"}
    bank = row["bank"]
    bank_body = {key: value for key, value in bank.items() if key != "receipt_sha256"}
    diagnosis = row["diagnosis"]
    diagnosis_body = {key: value for key, value in diagnosis.items() if key != "receipt_sha256"}
    if (row["receipt_sha256"] != _digest(body) or row["receipt_sha256"] != expected_receipt
            or row["source"] != source
            or row["plan_sha256"] != plan_sha or row["mixed"] is not None
            or bank["receipt_sha256"] != _sha(bank_body)
            or bank["source_text_sha256"] != source
            or bank["transducer_receipt_sha256"] != model_receipt
            or diagnosis["receipt_sha256"] != _sha(diagnosis_body)
            or diagnosis["bank_receipt_sha256"] != bank["receipt_sha256"]
            or diagnosis["split"] != "train"):
        raise ValueError("candidate ranking bank identity differs")
    return row


def _rankable(item, row: dict) -> tuple[tuple, tuple[bool, ...], tuple]:
    """Return a target-free proposal view and separate post-generation labels."""
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_floor import semantic_program_structural_key
    from core.learning.semantic_program_ir import TokenSpan

    bank = row["bank"]
    statuses = {record["program_sha256"]: record["status"]
                for record in row["diagnosis"]["comparisons"]}
    spans = tuple(TokenSpan(**span) for span in bank["input_spans"])
    if len(spans) != len(item.public_inputs):
        raise ValueError("bank input grounding differs from source")
    programs, labels, keys, anchors = [], [], [], []
    seen = set()
    for candidate in bank["candidates"]:
        payload = candidate["program"]
        program = Program(len(spans), tuple(Instruction(op, tuple(args))
                                         for op, args in payload["instructions"]))
        key = program.sha()
        if (key != candidate["program_sha256"] or key != payload["sha"]
                or semantic_program_structural_key(program) is None
                or key not in statuses):
            raise ValueError("bank candidate is not a verified typed program")
        if key in seen:
            continue
        seen.add(key)
        programs.append(program)
        labels.append(statuses[key] == "equivalent")
        keys.append(key)
        anchors.append(tuple(TokenSpan(**span) for span in candidate["operation_spans"]))
    if not programs or (bank["selected_program_sha256"] is not None
                        and bank["selected_program_sha256"] != keys[0]):
        raise ValueError("candidate bank lacks its incumbent")
    return ((tuple(programs), tuple(labels), tuple(keys)), spans,
            tuple("integer_sequence" if isinstance(value, (tuple, list)) else "integer"
                  for value in item.public_inputs), tuple(anchors))


def _evaluate(model, items: dict, rows: dict, source_ids: list[str]) -> dict:
    import torch

    from core.learning.semantic_span_pointer import _hidden_array

    results = []
    model.eval()
    with torch.no_grad():
        for source in source_ids:
            item, row = items[source], rows[source]
            (programs, labels, keys), spans, kinds, anchors = _rankable(item, row)
            features = torch.from_numpy(_hidden_array(item.hidden_states)).float()
            scores = model(features, spans, kinds, programs, operation_spans=anchors)
            chosen = int(scores.argmax().item())
            results.append({"source": source, "candidate_count": len(keys),
                            "bank_reachable": any(labels),
                            "incumbent_correct": bool(row["bank"]["selected_program_sha256"]
                                                      is not None and labels[0]),
                            "selected_correct": labels[chosen], "chosen_program_sha256": keys[chosen],
                            "chosen_index": chosen})
    return {"population": len(results),
            "bank_reachable": sum(row["bank_reachable"] for row in results),
            "incumbent_correct": sum(row["incumbent_correct"] for row in results),
            "ranker_correct": sum(row["selected_correct"] for row in results),
            "gains": sum(row["selected_correct"] and not row["incumbent_correct"] for row in results),
            "regressions": sum(row["incumbent_correct"] and not row["selected_correct"] for row in results),
            "rows": results}


def _verify_sources(source_report: dict, candidate_report: dict, model, bank_report: dict,
                    folds: dict, examples: list) -> tuple[dict, list[str]]:
    from tools.materialize_semantic_candidate_training import _digest as training_digest

    plan = bank_report["plan"]
    full = sorted(item.ir.source_text_sha256 for item in examples if item.split == "train")
    ids = list(plan["source_ids"])
    checks = {
        "candidate_model": candidate_report.get("candidate") == model.receipt_sha256,
        "bank_schema": bank_report.get("schema") == "aura.semantic_candidate_training.v1",
        "bank_receipt": bank_report.get("receipt_sha256") == training_digest(
            {key: value for key, value in bank_report.items() if key != "receipt_sha256"}),
        "bank_population": bank_report["population"] == len(ids)
        and bank_report["observed"] == len(ids),
        "source_population": plan["source_population"] == len(full),
        "bank_source_membership": set(ids) <= set(full) and ids == sorted(ids),
        "bank_model": plan["model_receipt_sha256"] == model.receipt_sha256,
        "source_features": plan["source_manifest_sha256s"] == source_report[
            "representation_compatibility"]["source_feature_manifest_sha256s"],
        "fold_schema": folds.get("schema") == "aura.semantic_construction_folds.v1",
        "fold_membership": sorted(folds["assignments"]) == full,
        "fold_coverage": set(folds["assignments"].values()) == set(range(folds["count"])),
        "bank_row_receipts": sorted(bank_report["row_receipts"]) == ids,
        "fold_holdout": folds["validation_used"] is False and folds["test_used"] is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("source-only ranking inputs do not share a frozen cohort: "
                         + ",".join(failed))
    return plan, ids


def _contrast_cases(items: dict, train: list[str], held: list[str], *, limit: int) -> dict:
    from core.learning.semantic_candidate_contrasts import source_program_contrasts

    peers = tuple(items[source].ir.to_program() for source in train)
    cases = {}
    for source in train + held:
        item = items[source]
        target = item.ir.to_program()
        candidates = source_program_contrasts(
            target, item.public_inputs, peers, source_sha256=source, limit=limit)
        labels = tuple(program.sha() == target.sha() for program in candidates)
        kinds = tuple("integer_sequence" if isinstance(value, (tuple, list)) else "integer"
                      for value in item.public_inputs)
        anchors = tuple(tuple(instruction.operation_span for instruction in item.ir.instructions)
                        for _ in candidates)
        cases[source] = ((candidates, labels, tuple(program.sha() for program in candidates)),
                         item.ir.input_spans, kinds, anchors)
    return cases


def _training_views(source: str, banks: dict, contrasts: dict | None) -> tuple:
    """Keep real proposals and witnessed alternatives as separate evidence."""
    bank = banks[source]
    if contrasts is None:
        return (bank,)
    contrast = contrasts[source]
    if bank[1:3] != contrast[1:3]:
        raise ValueError("mixed candidate evidence changed source grounding")
    return bank, contrast


def _evaluate_contrasts(model, items: dict, cases: dict, held: list[str]) -> dict:
    import torch

    from core.learning.semantic_span_pointer import _hidden_array

    rows = []
    model.eval()
    with torch.no_grad():
        for source in held:
            (programs, labels, keys), spans, kinds, anchors = cases[source]
            features = torch.from_numpy(_hidden_array(items[source].hidden_states)).float()
            chosen = int(model(features, spans, kinds, programs,
                               operation_spans=anchors).argmax().item())
            rows.append({"source": source, "candidate_count": len(programs),
                         "selected_correct": labels[chosen],
                         "chosen_program_sha256": keys[chosen]})
    return {"population": len(rows), "synthetic_contrast_correct": sum(
        row["selected_correct"] for row in rows), "rows": rows,
        "qualification_evidence": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--bank-directory", type=Path)
    parser.add_argument("--training-mode", choices=("bank", "source_contrasts", "mixed"),
                        default="source_contrasts")
    parser.add_argument("--contrast-limit", type=int, default=24)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--pilot-rows", type=int, default=0,
                        help="source-only plumbing pilot; never a qualification result")
    args = parser.parse_args()
    if args.epochs < 1 or args.pilot_rows < 0 or args.contrast_limit < 2:
        parser.error("epochs and contrast-limit must be positive; pilot-rows nonnegative")
    if args.training_mode in {"bank", "mixed"} and args.bank_directory is None:
        parser.error("bank or mixed training needs a complete source bank directory")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output_directory / "report.json")
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from core.learning.semantic_candidate_ranker import ContextualProgramRanker, candidate_set_loss
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_request_context import RequestContextConfig
    from core.learning.semantic_span_pointer import _hidden_array

    model = compositional_semantic_program_transducer_from_dict(json.loads(args.transducer.read_bytes()))
    source_report = json.loads(args.source_report.read_bytes())
    candidate_report = json.loads(args.candidate_report.read_bytes())
    bank_report = (json.loads((args.bank_directory / "report.json").read_bytes())
                   if args.bank_directory else None)
    folds = json.loads(args.folds.read_bytes())
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(model, source_report, bundles)
    if bank_report is not None:
        plan, ids = _verify_sources(source_report, candidate_report, model, bank_report,
                                    folds, examples)
    else:
        ids = sorted(item.ir.source_text_sha256 for item in examples if item.split == "train")
        if (candidate_report.get("candidate") != model.receipt_sha256
                or folds.get("schema") != "aura.semantic_construction_folds.v1"
                or sorted(folds["assignments"]) != ids
                or folds["validation_used"] is not False or folds["test_used"] is not False):
            raise ValueError("source contrast inputs do not share a frozen cohort")
        plan = None
    if not 0 <= args.fold < folds["count"]:
        parser.error("fold lies outside the frozen source partition")
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    held = [source for source in ids if folds["assignments"][source] == args.fold]
    train = [source for source in ids if folds["assignments"][source] != args.fold]
    if args.pilot_rows:
        train, held = train[:args.pilot_rows], held[:args.pilot_rows]
    if args.training_mode in {"bank", "mixed"}:
        rows = {source: _read_bank(args.bank_directory / "rows" / f"{source}.json",
                                   source=source, plan_sha=plan["plan_sha256"],
                                   model_receipt=model.receipt_sha256,
                                   expected_receipt=bank_report["row_receipts"][source]) for source in ids}
        cases = {source: _rankable(items[source], rows[source]) for source in train + held}
        contrasts = (_contrast_cases(items, train, held, limit=args.contrast_limit)
                     if args.training_mode == "mixed" else None)
    else:
        cases = _contrast_cases(items, train, held, limit=args.contrast_limit)
        contrasts = None
    torch.manual_seed(20260922 + args.fold)
    np.random.seed(20260922 + args.fold)
    config = RequestContextConfig(model.hidden_size, width=64, heads=4, layers=1,
                                  feature_scaling="unit_variance")
    ranker = ContextualProgramRanker(config)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4, weight_decay=0.01)
    update_count = 0
    learning_curve = []
    for epoch in range(args.epochs):
        ranker.train()
        losses = []
        for index in torch.randperm(len(train), generator=torch.Generator().manual_seed(
                20260922 + args.fold + epoch)).tolist():
            source = train[index]
            item = items[source]
            features = torch.from_numpy(_hidden_array(item.hidden_states)).float()
            training_views = _training_views(source, cases, contrasts)
            for (programs, labels, _), spans, kinds, anchors in training_views:
                if not any(labels):
                    continue
                optimizer.zero_grad(set_to_none=True)
                scores = ranker(features, spans, kinds, programs, operation_spans=anchors)
                loss = candidate_set_loss(scores, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
                optimizer.step()
                update_count += 1
                losses.append(float(loss.detach()))
        learning_curve.append({"epoch": epoch + 1, "updates": update_count,
                               "mean_loss": sum(losses) / len(losses) if losses else None})
        print(json.dumps({"stage": "epoch", "fold": args.fold,
                          **learning_curve[-1]}), flush=True)
    if not update_count:
        raise ValueError("no source-training candidates had a verified solution")
    train_probe = (train if args.pilot_rows else train[:64])
    training_evaluation = (_evaluate(ranker, items, rows, train_probe)
                           if args.training_mode in {"bank", "mixed"} else
                           _evaluate_contrasts(ranker, items, cases, train_probe))
    evaluation = (_evaluate(ranker, items, rows, held)
                  if args.training_mode in {"bank", "mixed"}
                  else _evaluate_contrasts(ranker, items, cases, held))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    weights = args.output_directory / f"fold-{args.fold}.safetensors"
    if weights.exists():
        raise FileExistsError(weights)
    save_file({key: value.detach().cpu().contiguous() for key, value in ranker.state_dict().items()},
              weights)
    body = {"schema": "aura.semantic_candidate_ranker_source_fold.v1",
            "pilot_only": bool(args.pilot_rows or (bank_report and bank_report["pilot_only"])),
            "serving_authority": False,
            "fold": args.fold, "folds_sha256": hashlib.sha256(args.folds.read_bytes()).hexdigest(),
            "source_bank_receipt_sha256": (bank_report["receipt_sha256"]
                                           if bank_report else None),
            "training_mode": args.training_mode,
            "contrast_limit": args.contrast_limit if args.training_mode in {
                "source_contrasts", "mixed"} else None,
            "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
            "candidate_report_sha256": hashlib.sha256(args.candidate_report.read_bytes()).hexdigest(),
            "model_receipt_sha256": model.receipt_sha256,
            "train_sources": train, "held_sources": held,
            "epochs": args.epochs, "updates": update_count,
            "learning_curve": learning_curve,
            "training_evaluation": training_evaluation,
            "config": vars(config), "evaluation": evaluation,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    report = args.output_directory / f"fold-{args.fold}.json"
    report.write_text(json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True))
    print(json.dumps({"stage": "complete", "fold": args.fold,
                      "pilot_only": body["pilot_only"],
                      "training_mode": args.training_mode,
                      "population": evaluation["population"],
                      "selected_correct": evaluation.get("ranker_correct",
                                                         evaluation.get("synthetic_contrast_correct")),
                      "training_probe_correct": training_evaluation.get(
                          "ranker_correct", training_evaluation.get("synthetic_contrast_correct")),
                      "bank_reachable": evaluation.get("bank_reachable")}), flush=True)


if __name__ == "__main__":
    main()
