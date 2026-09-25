#!/usr/bin/env python3
"""Refit a source-fold direct decoder against its witnessed runtime-bank rivals.

The bank is generated without answer labels. Labels enter only this offline fit
and post-fit evaluation; neither held construction labels nor held outcomes
control the optimizer or checkpoint choice.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_refit(report: dict, checkpoint_path: Path, *, direct_checkpoint_sha256: str,
               folds: dict, source_bank_receipt_sha256: str, config):
    """Restore only a source-fold-bound, training-only direct-bank experiment."""
    import torch

    from core.learning.semantic_program_decoder import SemanticProgramDecoder

    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    plan = report["plan"]
    training = {source for source, fold in folds["assignments"].items()
                if fold != plan["fold"]}
    held = {source for source, fold in folds["assignments"].items()
            if fold == plan["fold"]}
    if (report.get("receipt_sha256") != _digest(body)
            or plan["schema"] != "aura.semantic_direct_bank_refit.v1"
            or plan["fold_receipt_sha256"] != folds["receipt_sha256"]
            or plan["parent_direct_checkpoint_sha256"] != direct_checkpoint_sha256
            or plan["source_bank_receipt_sha256"] != source_bank_receipt_sha256
            or not set(plan["training_ids"]) <= training
            or not set(plan["heldout_ids"]) <= held
            or set(plan["training_ids"]) & set(plan["heldout_ids"])
            or plan["optimizer_split"] != "train"
            or plan["heldout_controls_training"] is not False
            or report["validation_used"] is not False or report["test_used"] is not False
            or report["promotion_authorized"] is not False
            or report["checkpoint_sha256"] != _sha(checkpoint_path)):
        raise ValueError("direct-bank refit is not bound to its source-only fold")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["plan"] != plan or checkpoint["history"] != report["history"]:
        raise ValueError("direct-bank checkpoint differs from its fit receipt")
    model = SemanticProgramDecoder(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def _bank_view(item, row):
    from tools.evaluate_semantic_candidate_ranker import _rankable_or_none

    return _rankable_or_none(item, row)


def _generated_contrast(model, item, features, spans, kinds, target, *, width):
    """Generate wrong alternatives without a target, then witness them for training."""
    import torch

    from core.learning.semantic_graph_counterexamples import (
        compare_program_meanings,
        counterfactual_inputs,
    )

    with torch.no_grad():
        proposals = model.propose_beam(features, spans, kinds,
                                       beam_width=width, max_programs=width)
    probes = counterfactual_inputs(item.public_inputs, count=8,
                                  seed=int(item.ir.source_text_sha256[:8], 16))
    negatives = []
    for program, _receipt in proposals:
        if program.sha() == target.sha():
            continue
        if compare_program_meanings(target, program, probes)["status"] == "different":
            negatives.append(program)
    return (target, *negatives) if negatives else ()


def _evaluate(model, items, rows, ids):
    import torch

    from core.learning.semantic_span_pointer import _hidden_array

    results = []
    model.eval()
    with torch.no_grad():
        for source in ids:
            view = _bank_view(items[source], rows[source])
            if view is None:
                results.append({"source": source, "bank_reachable": False,
                                "incumbent_correct": False, "selected_correct": None})
                continue
            (programs, labels, keys), spans, kinds, _anchors = view
            features = torch.from_numpy(_hidden_array(items[source].hidden_states)).float()
            scores = model.score_many(features, spans, kinds, programs).tolist()
            chosen = max(range(len(scores)), key=scores.__getitem__)
            results.append({"source": source, "bank_reachable": any(labels),
                            "incumbent_correct": bool(labels[0]),
                            "selected_correct": labels[chosen],
                            "chosen_program_sha256": keys[chosen],
                            "chosen_index": chosen})
    return {"population": len(results),
            "bank_reachable": sum(row["bank_reachable"] for row in results),
            "incumbent_correct": sum(row["incumbent_correct"] for row in results),
            "selected_correct": sum(row["selected_correct"] is True for row in results),
            "gains": sum(row["selected_correct"] is True and not row["incumbent_correct"]
                         for row in results),
            "regressions": sum(row["incumbent_correct"] and row["selected_correct"] is False
                               for row in results), "rows": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transducer", "source-report", "candidate-report", "feature-root",
                 "bank-directory", "folds", "direct-report", "direct-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--held-limit", type=int, default=0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--generated-contrasts", type=int, default=0,
                        help="bounded target-blind beam rivals on training sources only")
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.epochs <= 100
            or args.train_limit < 0 or args.held_limit < 0
            or not 0 < args.rank_weight <= 10
            or not 0 <= args.generated_contrasts <= 8):
        parser.error("new output, bounded epochs/limits, and positive rank weight required")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    import torch

    from core.learning.semantic_graph_coordinates import reanchor_program_inputs
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_span_pointer import _hidden_array
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.compare_semantic_candidate_methods import _load_direct
    from tools.evaluate_semantic_candidate_ranker import _read_bank, _verify_sources

    torch.set_num_threads(2)
    torch.manual_seed(20260924 + args.fold)
    source_report = json.loads(args.source_report.read_bytes())
    candidate_report = json.loads(args.candidate_report.read_bytes())
    bank_report = json.loads((args.bank_directory / "report.json").read_bytes())
    folds = json.loads(args.folds.read_bytes())
    direct_report = json.loads(args.direct_report.read_bytes())
    transducer = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"][
                   "source_feature_manifest_sha256s"]]
    examples = load_source_examples(transducer, source_report, bundles)
    plan, ids = _verify_sources(source_report, candidate_report, transducer,
                                bank_report, folds, examples)
    if not 0 <= args.fold < folds["count"]:
        parser.error("fold outside frozen construction partition")
    model, parent_sha = _load_direct(direct_report, args.direct_checkpoint,
                                      fold=args.fold, folds=folds,
                                      source_report=source_report)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    train = [source for source in ids if folds["assignments"][source] != args.fold]
    held = [source for source in ids if folds["assignments"][source] == args.fold]
    if args.train_limit:
        train = train[:args.train_limit]
    if args.held_limit:
        held = held[:args.held_limit]
    if not train or not held:
        raise ValueError("source fold needs both training and held bank rows")
    selected = sorted(set(train) | set(held))
    rows = {source: _read_bank(args.bank_directory / "rows" / f"{source}.json",
                               source=source, plan_sha=plan["plan_sha256"],
                               model_receipt=transducer.receipt_sha256,
                               expected_receipt=bank_report["row_receipts"][source])
            for source in selected}
    paths = (Path(__file__).resolve(),
             ROOT / "core/learning/semantic_program_decoder.py",
             ROOT / "core/learning/semantic_candidate_ranker.py")
    fit_plan = {"schema": "aura.semantic_direct_bank_refit.v1", "fold": args.fold,
                "fold_receipt_sha256": folds["receipt_sha256"],
                "source_bank_receipt_sha256": bank_report["receipt_sha256"],
                "parent_direct_checkpoint_sha256": parent_sha,
                "training_ids": train, "heldout_ids": held,
                "epochs": args.epochs, "rank_weight": args.rank_weight,
                "generated_contrasts": args.generated_contrasts,
                "pilot_only": bool(bank_report["pilot_only"] or args.train_limit
                                   or args.held_limit),
                "optimizer_split": "train", "heldout_controls_training": False,
                "serving_authority": False,
                "implementation_sha256": {str(path.relative_to(ROOT)): _sha(path)
                                          for path in paths}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001)
    baseline = _evaluate(model, items, rows, held)
    history = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        updates = 0
        contrast_updates = 0
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            20260924 + args.fold + epoch)).tolist()
        for index in order:
            source = train[index]
            view = _bank_view(items[source], rows[source])
            if view is None:
                spans, _proof, _reason = transducer._runtime_input_grounding(
                    items[source].ir.source_token_ids,
                    items[source].hidden_states, items[source].public_inputs)
                kinds = tuple("integer_sequence" if isinstance(value, (tuple, list))
                              else "integer" for value in items[source].public_inputs)
            else:
                (programs, labels, _keys), spans, kinds, _anchors = view
            features = torch.from_numpy(_hidden_array(items[source].hidden_states)).float()
            target = reanchor_program_inputs(
                items[source].ir.to_program(),
                from_spans=items[source].ir.input_spans, to_spans=spans,
                from_inputs=items[source].public_inputs,
                to_inputs=items[source].public_inputs,
            )
            optimizer.zero_grad(set_to_none=True)
            source_loss = model.loss(features, spans, kinds, target)
            rank_loss = (model.rank_loss(features, spans, kinds, programs, labels)
                         if view is not None and any(labels) else None)
            loss = source_loss if rank_loss is None else source_loss + args.rank_weight * rank_loss
            if args.generated_contrasts:
                contrast = _generated_contrast(model, items[source], features, spans,
                                               kinds, target, width=args.generated_contrasts)
                if contrast:
                    loss = loss + args.rank_weight * model.rank_loss(
                        features, spans, kinds, contrast,
                        (True, *(False for _ in contrast[1:])))
                    contrast_updates += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
            losses.append((float(source_loss.detach()),
                           float(rank_loss.detach()) if rank_loss is not None else None))
        if not updates:
            raise ValueError("training fold has no verified correct bank contrast")
        history.append({"epoch": epoch + 1, "updates": updates,
                        "generated_contrast_updates": contrast_updates,
                        "mean_source_loss": sum(row[0] for row in losses) / len(losses),
                        "mean_rank_loss": (sum(row[1] for row in losses if row[1] is not None)
                                           / sum(row[1] is not None for row in losses)
                                           if any(row[1] is not None for row in losses) else None),
                        "elapsed_s": time.monotonic() - started})
        print(json.dumps({"stage": "epoch", **history[-1]}), flush=True)
    for path in paths:
        if _sha(path) != fit_plan["implementation_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError("direct-bank fit implementation changed")
    evaluation = _evaluate(model, items, rows, held)
    checkpoint = io.BytesIO()
    torch.save({"plan": fit_plan, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history}, checkpoint)
    checkpoint_path = args.output.with_suffix(".pt")
    if not atomic_write_bytes_if_absent(checkpoint_path, checkpoint.getvalue()):
        raise ValueError("direct-bank checkpoint already exists")
    result = {"plan": fit_plan, "history": history, "baseline": baseline,
              "evaluation": evaluation, "checkpoint_sha256": _sha(checkpoint_path),
              "promotion_authorized": False, "validation_used": False,
              "test_used": False}
    result["receipt_sha256"] = _digest(result)
    if not atomic_write_bytes_if_absent(args.output,
                                        json.dumps(result, sort_keys=True).encode()):
        raise ValueError("direct-bank report already exists")
    print(json.dumps({"baseline": baseline["selected_correct"],
                      "refit": evaluation["selected_correct"],
                      "held": len(held), "pilot_only": fit_plan["pilot_only"]}), flush=True)


if __name__ == "__main__":
    main()
