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


def _rankable(item, row: dict, *, preserve_evidence: bool = False) -> tuple:
    """Return a target-free proposal view and separate post-generation labels."""
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_floor import semantic_program_structural_key
    from core.learning.semantic_program_ir import TokenSpan

    bank = row["bank"]
    if preserve_evidence and bank.get("schema") != "aura.semantic_candidate_bank.v3":
        raise ValueError("argument evidence requires a source-evidence candidate bank")
    statuses = {record["program_sha256"]: record["status"]
                for record in row["diagnosis"]["comparisons"]}
    spans = tuple(TokenSpan(**span) for span in bank["input_spans"])
    if len(spans) != len(item.public_inputs):
        raise ValueError("bank input grounding differs from source")
    programs, labels, keys, anchors, mentions, definitions = [], [], [], [], [], []
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
        operation_anchors = tuple(TokenSpan(**span) for span in candidate["operation_spans"])
        argument_rows = candidate.get("argument_spans")
        definition_rows = candidate.get("definition_spans")
        if preserve_evidence and argument_rows is None:
            raise ValueError("candidate bank discarded argument mentions")
        if preserve_evidence and (candidate.get("definition_provenance") not in {
                "register_anchor", "optimizer_selected", "unavailable"}
                or (definition_rows is None) !=
                (candidate["definition_provenance"] == "unavailable")):
            raise ValueError("candidate bank definition evidence has no valid origin")
        evidence_key = (key, operation_anchors,
                        json.dumps(argument_rows, sort_keys=True),
                        json.dumps(definition_rows, sort_keys=True)) if preserve_evidence else key
        if evidence_key in seen:
            continue
        seen.add(evidence_key)
        programs.append(program)
        labels.append(statuses[key] == "equivalent")
        keys.append(key)
        anchors.append(operation_anchors)
        if preserve_evidence:
            mentions.append(tuple(tuple(TokenSpan(**span) for span in step)
                                  for step in argument_rows))
            definitions.append(tuple(tuple(TokenSpan(**span) for span in step)
                                     for step in definition_rows)
                               if definition_rows is not None else None)
    if not programs or (bank["selected_program_sha256"] is not None
                        and bank["selected_program_sha256"] != keys[0]):
        raise ValueError("candidate bank lacks its incumbent")
    result = ((tuple(programs), tuple(labels), tuple(keys)), spans,
              tuple("integer_sequence" if isinstance(value, (tuple, list)) else "integer"
                    for value in item.public_inputs), tuple(anchors))
    return (*result, tuple(mentions), tuple(definitions)) if preserve_evidence else result


def _rankable_or_none(item, row: dict, *, preserve_evidence: bool = False) -> tuple | None:
    """Only an explicitly empty ordinary decode may lack a ranking view."""
    bank = row["bank"]
    if (not bank["candidates"] and not bank["input_spans"]
            and bank["selected_program_sha256"] is None
            and bank["limit_reason"] == "ordinary_decode_unavailable"):
        return None
    return _rankable(item, row, preserve_evidence=preserve_evidence)


def _evaluate(model, items: dict, rows: dict, source_ids: list[str], *,
              cross_token: bool = True) -> dict:
    import torch

    from core.learning.semantic_span_pointer import _hidden_array

    results = []
    model.eval()
    with torch.no_grad():
        for source in source_ids:
            item, row = items[source], rows[source]
            case = _rankable_or_none(item, row,
                                     preserve_evidence=model.retain_evidence_variants)
            if case is None:
                results.append({"source": source, "candidate_count": 0,
                                "bank_reachable": None, "incumbent_correct": False,
                                "selected_correct": None, "chosen_program_sha256": None,
                                "chosen_index": None,
                                "unrankable_reason": "ordinary_decode_unavailable"})
                continue
            (programs, labels, keys), spans, kinds, anchors = case[:4]
            features = torch.from_numpy(_hidden_array(item.hidden_states)).float()
            kwargs = ({"argument_spans": case[4], "definition_spans": case[5]}
                      if model.argument_evidence else {})
            if not cross_token:
                kwargs["cross_token"] = False
            scores = model(features, spans, kinds, programs, operation_spans=anchors, **kwargs)
            if model.retain_evidence_variants:
                from core.learning.semantic_candidate_ranker import aggregate_program_scores

                _programs, grouped, members = aggregate_program_scores(scores, keys)
                winning = members[int(grouped.argmax().item())]
                chosen = max(winning, key=lambda index: float(scores[index]))
            else:
                chosen = int(scores.argmax().item())
            results.append({"source": source, "candidate_count": len(keys),
                            "bank_reachable": any(labels),
                            "incumbent_correct": bool(row["bank"]["selected_program_sha256"]
                                                      is not None and labels[0]),
                            "selected_correct": labels[chosen], "chosen_program_sha256": keys[chosen],
                            "chosen_index": chosen})
    return {"population": len(results),
            "bank_reachable": sum(row["bank_reachable"] is True for row in results),
            "ranker_evaluable": sum(row["selected_correct"] is not None for row in results),
            "incumbent_correct": sum(row["incumbent_correct"] for row in results),
            "ranker_correct": sum(row["selected_correct"] is True for row in results),
            "gains": sum(row["selected_correct"] is True and not row["incumbent_correct"]
                         for row in results),
            "regressions": sum(row["incumbent_correct"] and row["selected_correct"] is False
                               for row in results),
            "rows": results}


def _verify_sources(source_report: dict, candidate_report: dict, model, bank_report: dict,
                    folds: dict, examples: list) -> tuple[dict, list[str]]:
    from core.learning.semantic_program_campaign import _sha
    from tools.materialize_semantic_candidate_training import _digest as training_digest
    from tools.refit_semantic_argument_proposals import verify_candidate_report_identity

    verify_candidate_report_identity(candidate_report, model, source_report)
    plan = bank_report["plan"]
    full = sorted(item.ir.source_text_sha256 for item in examples if item.split == "train")
    withheld = {item.ir.source_text_sha256 for item in examples
                if item.split in {"validation", "test"}}
    ids = list(plan["source_ids"])
    checks = {
        "bank_schema": bank_report.get("schema") == "aura.semantic_candidate_training.v1",
        "bank_receipt": bank_report.get("receipt_sha256") == training_digest(
            {key: value for key, value in bank_report.items() if key != "receipt_sha256"}),
        "bank_population": bank_report["population"] == len(ids)
        and bank_report["observed"] == len(ids),
        "source_population": plan["source_population"] == len(full),
        "bank_source_membership": set(ids) <= set(full) and ids == sorted(ids),
        "source_split_disjoint": not (set(full) & withheld),
        "bank_model": plan["model_receipt_sha256"] == model.receipt_sha256,
        "source_features": plan["source_manifest_sha256s"] == source_report[
            "representation_compatibility"]["source_feature_manifest_sha256s"],
        "fold_schema": folds.get("schema") == "aura.semantic_construction_folds.v1",
        "fold_membership": sorted(folds["assignments"]) == full,
        "fold_coverage": set(folds["assignments"].values()) == set(range(folds["count"])),
        "bank_row_receipts": sorted(bank_report["row_receipts"]) == ids,
        "fold_holdout": folds["validation_used"] is False and folds["test_used"] is False,
        "proposal_training_cohort": model.training_receipt.get(
            "training_example_ids_sha256") == _sha(full),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("source-only ranking inputs do not share a frozen cohort: "
                         + ",".join(failed))
    return plan, ids


def _verify_augmentation(base_plan: dict, base_ids: list[str], wide_plan: dict,
                         wide_ids: list[str]) -> None:
    selection = wide_plan.get("source_selection") or {}
    checks = {
        "subset": bool(wide_ids) and set(wide_ids) <= set(base_ids),
        "model": wide_plan["model_receipt_sha256"] == base_plan["model_receipt_sha256"],
        "features": wide_plan["source_manifest_sha256s"] == base_plan[
            "source_manifest_sha256s"],
        "population": wide_plan["source_population"] == base_plan["source_population"],
        "search_width": (wide_plan["max_charts"] >= base_plan["max_charts"]
                         and wide_plan["max_graphs_per_chart"] >= base_plan[
                             "max_graphs_per_chart"]
                         and (wide_plan["max_charts"] > base_plan["max_charts"]
                              or wide_plan["max_graphs_per_chart"] > base_plan[
                                  "max_graphs_per_chart"])),
        "source_only_selection": (selection.get("method") ==
                                  "lowest_source_identity_per_construction"
                                  and selection.get("labels_used_for_selection") is False),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("augmented bank is not source-only and basis-compatible: "
                         + ",".join(failed))


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


def _factor_cases(items: dict, train: list[str]) -> dict:
    from core.learning.semantic_candidate_contrasts import source_program_factor_contrasts

    cases = {}
    for source in train:
        item = items[source]
        target = item.ir.to_program()
        programs = source_program_factor_contrasts(
            target, item.public_inputs, source_sha256=source)
        if len(programs) < 2:
            continue
        labels = tuple(program.sha() == target.sha() for program in programs)
        kinds = tuple("integer_sequence" if isinstance(value, (tuple, list)) else "integer"
                      for value in item.public_inputs)
        anchors = tuple(tuple(instruction.operation_span for instruction in item.ir.instructions)
                        for _ in programs)
        cases[source] = ((programs, labels, tuple(program.sha() for program in programs)),
                         item.ir.input_spans, kinds, anchors)
    return cases


def _runtime_factor_cases(items: dict, train: list[str], banks: dict,
                          *, max_charts: int = 4) -> dict:
    """Pair source-witnessed role rivals with answer-blind runtime chart views."""
    if type(max_charts) is not int or not 1 <= max_charts <= 64:
        raise ValueError("runtime factor chart limit is invalid")
    factors = _factor_cases(items, train)
    cases = {}
    for source, factor in factors.items():
        if source not in banks:
            continue
        bank = banks[source]
        if factor[1:3] != bank[1:3]:
            raise ValueError("runtime factor view changed source grounding")
        target_ops = tuple(ins.op for ins in factor[0][0][0].instructions)
        views = []
        for program, anchors in zip(bank[0][0], bank[3], strict=True):
            if tuple(ins.op for ins in program.instructions) != target_ops or anchors in views:
                continue
            views.append(anchors)
            if len(views) == max_charts:
                break
        if not views:
            views.append(factor[3][0])
        cases[source] = [
            (factor[0], factor[1], factor[2], tuple(anchors for _ in factor[0][0]))
            for anchors in views
        ]
    return cases


def _source_runtime_factor_cases(model, items: dict, train: list[str],
                                 *, max_charts: int) -> tuple[dict, dict]:
    """Build source-labelled factor sets on the decoder's own operation views."""
    from core.learning.semantic_runtime_argument_views import runtime_argument_training_views

    views, receipt = runtime_argument_training_views(
        model, tuple(items[source] for source in train), max_operation_charts=max_charts)
    cases = {}
    for item in views:
        source = item.ir.source_text_sha256
        case = _factor_cases({source: item}, [source]).get(source)
        if case is not None:
            cases.setdefault(source, []).append(case)
    return cases, receipt


def _training_views(source: str, banks: dict, contrasts: dict | None,
                    *, allow_grounding_permutation: bool = False) -> tuple:
    """Keep real proposals and witnessed alternatives as separate evidence."""
    bank = banks[source]
    if contrasts is None:
        return (bank,)
    contrast = contrasts.get(source)
    if contrast is None:
        return (bank,)
    views = contrast if isinstance(contrast, list) else [contrast]
    if any((bank[2] != view[2] or (
            set(bank[1]) != set(view[1]) if allow_grounding_permutation
            else bank[1] != view[1])) for view in views):
        raise ValueError("mixed candidate evidence changed source grounding")
    return (bank, *views)


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
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--bundle", action="append", metavar="NAME=PATH")
    parser.add_argument("--bank-directory", type=Path)
    parser.add_argument("--augmentation-bank-directory", type=Path,
                        help="source-only wider bank; one balanced training view per source")
    parser.add_argument("--training-mode", choices=("bank", "source_contrasts", "mixed",
                                                    "bank_factors", "bank_factors_runtime",
                                                    "source_factors_runtime"),
                        default="source_contrasts")
    parser.add_argument("--contrast-limit", type=int, default=24)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--identity-bindings", action="store_true",
                        help="train explicit operation-role to source-mention evidence")
    parser.add_argument("--argument-evidence", action="store_true",
                        help="train on exact runtime-selected argument and definition spans")
    parser.add_argument("--retain-evidence-variants", action="store_true",
                        help="keep and normalize all runtime evidence paths for each program")
    parser.add_argument("--runtime-charts", type=int, default=4,
                        help="bounded answer-blind operation views per source")
    parser.add_argument("--pilot-rows", type=int, default=0,
                        help="source-only plumbing pilot; never a qualification result")
    args = parser.parse_args()
    if (args.epochs < 1 or args.pilot_rows < 0 or args.contrast_limit < 2
            or not 1 <= args.runtime_charts <= 64):
        parser.error("epochs and contrast-limit must be positive; pilot-rows nonnegative")
    bank_modes = {"bank", "mixed", "bank_factors", "bank_factors_runtime",
                  "source_factors_runtime"}
    balanced_modes = {"bank_factors_runtime", "source_factors_runtime"}
    if args.training_mode in bank_modes and args.bank_directory is None:
        parser.error("bank training needs a complete source bank directory")
    if args.argument_evidence and (not args.identity_bindings or args.training_mode != "bank"):
        parser.error("argument evidence needs identity bindings and real bank training")
    if args.retain_evidence_variants and args.training_mode != "bank":
        parser.error("evidence-path retention needs real bank training")
    if args.augmentation_bank_directory and args.training_mode != "bank":
        parser.error("wide-bank augmentation needs real bank training")
    retain_evidence_variants = args.retain_evidence_variants or args.argument_evidence

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
        verify_candidate_report_identity,
    )

    configure_refit_environment(args.output_directory / "report.json")
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from core.learning.semantic_candidate_ranker import ContextualProgramRanker, candidate_set_loss
    from core.learning.semantic_program_campaign import _sha as semantic_sha
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
    bundles = source_bundle_arguments(source_report, feature_root=args.feature_root,
                                      bundles=args.bundle)
    examples = load_source_examples(model, source_report, bundles)
    if bank_report is not None:
        plan, ids = _verify_sources(source_report, candidate_report, model, bank_report,
                                    folds, examples)
    else:
        ids = sorted(item.ir.source_text_sha256 for item in examples if item.split == "train")
        withheld = {item.ir.source_text_sha256 for item in examples
                    if item.split in {"validation", "test"}}
        verify_candidate_report_identity(candidate_report, model, source_report)
        if (folds.get("schema") != "aura.semantic_construction_folds.v1"
                or sorted(folds["assignments"]) != ids
                or set(ids) & withheld
                or folds["validation_used"] is not False or folds["test_used"] is not False):
            raise ValueError("source contrast inputs do not share a frozen cohort")
        plan = None
    proposal_trained_on_source_folds = (model.training_receipt.get(
        "training_example_ids_sha256") == semantic_sha(sorted(
            item.ir.source_text_sha256 for item in examples if item.split == "train")))
    augmentation_report = None
    augmentation_ids = []
    if args.augmentation_bank_directory:
        augmentation_report = json.loads(
            (args.augmentation_bank_directory / "report.json").read_bytes())
        wide_plan, augmentation_ids = _verify_sources(
            source_report, candidate_report, model, augmentation_report, folds, examples)
        _verify_augmentation(plan, ids, wide_plan, augmentation_ids)
    if not 0 <= args.fold < folds["count"]:
        parser.error("fold lies outside the frozen source partition")
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    held = [source for source in ids if folds["assignments"][source] == args.fold]
    train = [source for source in ids if folds["assignments"][source] != args.fold]
    if args.training_mode == "source_factors_runtime":
        train = [source for source in sorted(items)
                 if folds["assignments"][source] != args.fold]
    if args.pilot_rows:
        train, held = train[:args.pilot_rows], held[:args.pilot_rows]
    planned_train = tuple(train)
    runtime_view_receipt = None
    if args.training_mode in bank_modes:
        rows = {source: _read_bank(args.bank_directory / "rows" / f"{source}.json",
                                   source=source, plan_sha=plan["plan_sha256"],
                                   model_receipt=model.receipt_sha256,
                                   expected_receipt=bank_report["row_receipts"][source]) for source in ids}
        cases = {source: case for source in ids if (case := _rankable_or_none(
            items[source], rows[source], preserve_evidence=retain_evidence_variants)) is not None}
        wide_rows = {}
        wide_cases = {}
        if augmentation_report is not None:
            wide_rows = {source: _read_bank(
                args.augmentation_bank_directory / "rows" / f"{source}.json",
                source=source, plan_sha=wide_plan["plan_sha256"],
                model_receipt=model.receipt_sha256,
                expected_receipt=augmentation_report["row_receipts"][source])
                for source in augmentation_ids}
            wide_cases = {source: case for source in augmentation_ids if (
                case := _rankable_or_none(items[source], wide_rows[source],
                                          preserve_evidence=retain_evidence_variants)) is not None}
        if args.training_mode == "source_factors_runtime":
            source_cases, runtime_view_receipt = _source_runtime_factor_cases(
                model, items, train, max_charts=args.runtime_charts)
            contrasts = {}
            for source in train:
                views = source_cases.get(source, [])
                if source not in cases:
                    if not views:
                        continue
                    cases[source] = views[0]
                    views = views[1:]
                contrasts[source] = views
            train = [source for source in train if source in cases]
        else:
            contrasts = ({source: wide_cases[source] for source in train
                          if source in wide_cases} if augmentation_report is not None else
                     _contrast_cases(items, train, held, limit=args.contrast_limit)
                     if args.training_mode == "mixed" else
                     _factor_cases(items, train) if args.training_mode == "bank_factors" else
                     _runtime_factor_cases(items, train, cases)
                     if args.training_mode == "bank_factors_runtime" else None)
    else:
        cases = _contrast_cases(items, train, held, limit=args.contrast_limit)
        contrasts = None
    if args.training_mode in bank_modes:
        train = [source for source in train if source in cases]
    balanced_views = args.training_mode in balanced_modes or augmentation_report is not None
    torch.manual_seed(20260922 + args.fold)
    np.random.seed(20260922 + args.fold)
    config = RequestContextConfig(model.hidden_size, width=64, heads=4, layers=1,
                                  feature_scaling="unit_variance")
    ranker = ContextualProgramRanker(config, identity_bindings=args.identity_bindings,
                                    argument_evidence=args.argument_evidence,
                                    retain_evidence_variants=retain_evidence_variants)
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
            training_views = _training_views(
                source, cases, contrasts,
                allow_grounding_permutation=args.training_mode == "source_factors_runtime")
            valid_views = [view for view in training_views if any(view[0][1])]
            if balanced_views and valid_views:
                optimizer.zero_grad(set_to_none=True)
            source_losses = []
            for view in valid_views:
                (programs, labels, keys), spans, kinds, anchors = view[:4]
                if not balanced_views:
                    optimizer.zero_grad(set_to_none=True)
                kwargs = ({"argument_spans": view[4], "definition_spans": view[5]}
                          if args.argument_evidence else {})
                scores = ranker(features, spans, kinds, programs,
                                operation_spans=anchors, **kwargs)
                loss = candidate_set_loss(scores, labels,
                                          program_keys=keys if retain_evidence_variants else None)
                if balanced_views:
                    (loss / len(valid_views)).backward()
                    source_losses.append(float(loss.detach()))
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
                optimizer.step()
                update_count += 1
                losses.append(float(loss.detach()))
            if source_losses:
                torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
                optimizer.step()
                update_count += 1
                losses.append(sum(source_losses) / len(source_losses))
        learning_curve.append({"epoch": epoch + 1, "updates": update_count,
                               "mean_loss": sum(losses) / len(losses) if losses else None})
        print(json.dumps({"stage": "epoch", "fold": args.fold,
                          **learning_curve[-1]}), flush=True)
    if not update_count:
        raise ValueError("no source-training candidates had a verified solution")
    train_probe = ([source for source in train if source in rows]
                   if args.training_mode == "source_factors_runtime" else
                   train if args.pilot_rows else train[:64])
    if args.pilot_rows:
        train_probe = train_probe[:args.pilot_rows]
    training_evaluation = (_evaluate(ranker, items, rows, train_probe)
                           if args.training_mode in bank_modes else
                           _evaluate_contrasts(ranker, items, cases, train_probe))
    evaluation = (_evaluate(ranker, items, rows, held)
                  if args.training_mode in bank_modes
                  else _evaluate_contrasts(ranker, items, cases, held))
    wide_held = [source for source in held if source in wide_rows] if augmentation_report else []
    wide_evaluation = (_evaluate(ranker, items, wide_rows, wide_held)
                       if augmentation_report is not None else None)
    evidence_lesion = None
    triadic_lesion = None
    if args.argument_evidence:
        from core.learning.semantic_candidate_ranker import triadic_evidence_lesion

        with triadic_evidence_lesion(ranker):
            triadic_lesion = _evaluate(ranker, items, rows, held)
        with torch.no_grad():
            original = ranker.evidence_key.weight.detach().clone()
            try:
                ranker.evidence_key.weight.zero_()
                evidence_lesion = _evaluate(ranker, items, rows, held)
            finally:
                ranker.evidence_key.weight.copy_(original)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    weights = args.output_directory / f"fold-{args.fold}.safetensors"
    if weights.exists():
        raise FileExistsError(weights)
    save_file({key: value.detach().cpu().contiguous() for key, value in ranker.state_dict().items()},
              weights)
    body = {"schema": "aura.semantic_candidate_ranker_source_fold.v1",
            "pilot_only": bool(args.pilot_rows or (bank_report and bank_report["pilot_only"])
                               or augmentation_report is not None),
            "serving_authority": False,
            "qualification_evidence": False,
            "proposal_model_trained_on_source_fold_held_rows": proposal_trained_on_source_folds,
            "fold": args.fold, "folds_sha256": hashlib.sha256(args.folds.read_bytes()).hexdigest(),
            "source_bank_receipt_sha256": (bank_report["receipt_sha256"]
                                           if bank_report else None),
            "augmentation_bank_receipt_sha256": (augmentation_report["receipt_sha256"]
                                                   if augmentation_report else None),
            "augmentation_source_ids": augmentation_ids,
            "augmentation_train_sources": sorted(set(train) & set(augmentation_ids)),
            "augmentation_held_sources": wide_held,
            "augmentation_held_evaluation": wide_evaluation,
            "training_mode": args.training_mode,
            "identity_bindings": args.identity_bindings,
            "argument_evidence": args.argument_evidence,
            "retain_evidence_variants": retain_evidence_variants,
            "source_balanced_runtime_factor_views": args.training_mode in balanced_modes,
            "source_balanced_training_views": balanced_views,
            "runtime_charts": args.runtime_charts if args.training_mode in balanced_modes else None,
            "runtime_view_receipt_sha256": (_digest(runtime_view_receipt)
                                             if runtime_view_receipt is not None else None),
            "runtime_view_coverage": (runtime_view_receipt["coverage"]
                                      if runtime_view_receipt is not None else None),
            "contrast_limit": args.contrast_limit if args.training_mode in {
                "source_contrasts", "mixed"} else None,
            "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
            "candidate_report_sha256": hashlib.sha256(args.candidate_report.read_bytes()).hexdigest(),
            "model_receipt_sha256": model.receipt_sha256,
            "train_sources": train, "held_sources": held,
            "source_training_coverage": {
                "eligible": len(train), "planned": len(planned_train),
                "without_witnessed_contrasts": sorted(set(planned_train) - set(train)),
            } if args.training_mode == "source_factors_runtime" else None,
            "bank_unrankable_sources": sorted(set(ids) - set(cases))
            if args.training_mode in bank_modes else None,
            "epochs": args.epochs, "updates": update_count,
            "learning_curve": learning_curve,
            "training_evaluation": training_evaluation,
            "config": vars(config), "evaluation": evaluation,
            "same_checkpoint_argument_evidence_lesion": evidence_lesion,
            "same_checkpoint_triadic_evidence_lesion": triadic_lesion,
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
