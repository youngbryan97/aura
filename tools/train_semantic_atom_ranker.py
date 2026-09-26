#!/usr/bin/env python3
"""Fit local semantic factors on source-only constructions and replay a frozen bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_atom_partition(examples, plan, folds):
    from tools.probe_semantic_proposer_crossfit import crossfit_partition

    fit, calibration, complete_held = crossfit_partition(
        examples, folds, plan["fold"], all_held=True)
    def identity(rows):
        return sorted(item.ir.source_text_sha256 for item in rows)
    if (plan.get("outer_fold") is not None or plan.get("input_order_policy")
            != "source_token_order_v1"
            or identity(fit) != plan["fit_ids"]
            or identity(calibration) != plan["calibration_ids"]
            or not plan["held_ids"] or len(plan["held_ids"]) != len(set(plan["held_ids"]))
            or not set(plan["held_ids"]) <= set(identity(complete_held))):
        raise ValueError("atom source partition differs from the frozen bank")
    return fit, calibration


def construction_weights(examples):
    counts = Counter(item.construction_id for item in examples)
    if not counts or any(not name for name in counts):
        raise ValueError("atom training needs source construction provenance")
    return {item.ir.source_text_sha256:
            len(examples) / (len(counts) * counts[item.construction_id]) for item in examples}


def _case(item):
    import torch

    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_span_pointer import _hidden_array

    return (torch.from_numpy(_hidden_array(item.hidden_states)).float(),
            item.ir.input_spans,
            tuple("integer_sequence" if isinstance(value, (tuple, list)) else "integer"
                  for value in item.public_inputs),
            Program(item.ir.n_inputs, tuple(Instruction(ins.op, ins.args)
                                            for ins in item.ir.instructions)),
            tuple(ins.operation_span for ins in item.ir.instructions))


def _measure_atoms(model, examples):
    import torch

    weights = construction_weights(examples)
    measured = Counter()
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for item in examples:
            features, spans, kinds, program, operations = _case(item)
            loss, counts = model.source_atom_loss(features, spans, kinds, program,
                                                  operation_spans=operations)
            total_loss += float(loss) * weights[item.ir.source_text_sha256]
            measured.update(counts)
    return {"population": len(examples), "construction_balanced_loss": total_loss / len(examples),
            **measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--bundle", action="append", metavar="NAME=PATH")
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if (args.epochs < 1 or args.width < 4 or args.width % 4
            or not 0 < args.max_seconds <= 14400):
        parser.error("positive epochs, a width divisible by four, and a bounded duration are required")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
    )

    configure_refit_environment(args.directory / "report.json")
    started = time.monotonic()
    import torch
    from safetensors.torch import load_file, save

    from core.learning.semantic_atom_ranker import AtomAlignedProgramRanker
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_request_context import RequestContextConfig
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.evaluate_semantic_candidate_ranker import _evaluate, _read_bank
    from tools.probe_semantic_proposer_crossfit import _digest, _save_if_absent
    from tools.train_nested_semantic_ranker import _verified_pair

    outer_plan, bank_report = _verified_pair(args.bank)
    raw = {name: path.read_bytes() for name, path in (
        ("parent", args.parent), ("source", args.source_report), ("folds", args.folds))}
    parent = compositional_semantic_program_transducer_from_dict(json.loads(raw["parent"]))
    source = json.loads(raw["source"])
    folds = json.loads(raw["folds"])
    if (parent.receipt_sha256 != outer_plan["parent_receipt_sha256"]
            or hashlib.sha256(raw["source"]).hexdigest() != outer_plan["source_report_sha256"]
            or hashlib.sha256(raw["folds"]).hexdigest() != outer_plan["folds_sha256"]):
        raise ValueError("atom source basis differs from the frozen bank")
    examples = load_source_examples(parent, source, source_bundle_arguments(
        source, feature_root=args.feature_root, bundles=args.bundle))
    fit, calibration = validate_atom_partition(examples, outer_plan, folds)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    implementation_paths = [ROOT / name for name in (
        "tools/train_semantic_atom_ranker.py", "tools/evaluate_semantic_candidate_ranker.py",
        "tools/probe_semantic_proposer_crossfit.py", "core/learning/semantic_atom_ranker.py",
        "core/learning/semantic_request_context.py")]
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in implementation_paths}
    config = RequestContextConfig(parent.hidden_size, width=args.width, heads=4, layers=2,
                                  position_mode="relative", feature_scaling="unit_variance")
    seed = 20260925 + outer_plan["fold"]
    plan = {"schema": "aura.semantic_atom_fit_plan.v1", "config": vars(config),
            "epochs": args.epochs, "max_seconds": args.max_seconds,
            "seed": seed, "learning_rate": 3e-4,
            "weight_decay": .01, "gradient_clip": 1.0,
            "selection": "minimum_source_calibration_atom_loss",
            "loss_weighting": "equal_source_construction_mass_v1",
            "bank_plan_sha256": outer_plan["plan_sha256"],
            "bank_receipt_sha256": bank_report["receipt_sha256"],
            "fit_ids": outer_plan["fit_ids"], "calibration_ids": outer_plan["calibration_ids"],
            "held_ids": outer_plan["held_ids"], "implementation": implementation,
            "source_report_sha256": outer_plan["source_report_sha256"],
            "model_basis_sha256": parent.model_basis_sha256,
            "heldout_axis": outer_plan["heldout_axis"],
            "held_labels_used_for_fit_or_selection": False, "serving_authority": False}
    plan = {**plan, "plan_sha256": _digest(plan)}
    _save_if_absent(args.directory / "plan.json", plan)
    if any(args.directory.glob("epoch-*.json")):
        raise FileExistsError("atom fit already has epoch evidence; use a fresh experiment directory")
    torch.manual_seed(seed)
    model = AtomAlignedProgramRanker(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"],
                                  weight_decay=plan["weight_decay"])
    weights = construction_weights(fit)
    best = None
    updates = 0
    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(seed + epoch))
        total_loss = 0.0
        for index in order.tolist():
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("bounded source fit ended; completed epochs remain durable")
            item = fit[index]
            features, spans, kinds, program, operations = _case(item)
            loss, _counts = model.source_atom_loss(features, spans, kinds, program,
                                                   operation_spans=operations)
            weighted = loss * weights[item.ir.source_text_sha256]
            optimizer.zero_grad(set_to_none=True)
            weighted.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), plan["gradient_clip"])
            optimizer.step()
            total_loss += float(weighted.detach())
            updates += 1
            if updates % 100 == 0:
                print(json.dumps({"stage": "fit", "epoch": epoch, "updates": updates}), flush=True)
        measured = _measure_atoms(model, calibration)
        weight_path = args.directory / f"epoch-{epoch}.safetensors"
        payload = save({key: value.detach().cpu().contiguous()
                        for key, value in model.state_dict().items()})
        if not atomic_write_bytes_if_absent(weight_path, payload, mode=0o400):
            raise FileExistsError(weight_path)
        row = {"epoch": epoch, "updates": updates, "plan_sha256": plan["plan_sha256"],
               "training_loss": total_loss / len(fit), "calibration": measured,
               "weights_sha256": hashlib.sha256(payload).hexdigest()}
        row = {**row, "receipt_sha256": _digest(row)}
        _save_if_absent(args.directory / f"epoch-{epoch}.json", row)
        epochs.append(row)
        key = (measured["construction_balanced_loss"], epoch)
        if best is None or key < best[0]:
            best = (key, weight_path, row)
        print(json.dumps({"stage": "epoch", **row}), flush=True)
    model.load_state_dict(load_file(best[1]), strict=True)
    rows = {identity: _read_bank(
        args.bank / "rows" / f"{identity}.json", source=identity,
        plan_sha=outer_plan["plan_sha256"], model_receipt=bank_report["candidate_receipt_sha256"],
        expected_receipt=receipt) for identity, receipt in bank_report["row_receipts"].items()}
    evaluation = _evaluate(model, items, rows, outer_plan["held_ids"])
    context_lesion = _evaluate(model, items, rows, outer_plan["held_ids"], cross_token=False)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != implementation[
            str(path.relative_to(ROOT))] for path in implementation_paths):
        raise ValueError("atom implementation changed during measurement")
    body = {"schema": "aura.semantic_atom_fit.v1", "plan_sha256": plan["plan_sha256"],
            "selected_epoch": best[2]["epoch"], "epochs": epochs, "evaluation": evaluation,
            "context_lesion": context_lesion, "source_fit_population": len(fit),
            "source_calibration_population": len(calibration), "updates": updates,
            "elapsed_seconds": time.monotonic() - started,
            "selected_weights_sha256": best[2]["weights_sha256"],
            "serving_authority": False, "qualification_evidence": False,
            "held_labels_used_for_fit_or_selection": False}
    _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": _digest(body)})
    print(json.dumps({"stage": "complete", "selected_epoch": best[2]["epoch"],
                      "incumbent_correct": evaluation["incumbent_correct"],
                      "atom_correct": evaluation["ranker_correct"],
                      "gains": evaluation["gains"], "regressions": evaluation["regressions"]}), flush=True)


if __name__ == "__main__":
    main()
