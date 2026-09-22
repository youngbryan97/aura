#!/usr/bin/env python3
"""Source-only contextual fit and explicit development evaluation."""

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


def select_populations(examples, folds, *, fold, full_source, limit):
    """Keep optimizer inputs separate from source folds or exposed development."""
    if type(limit) is not int or limit < 1 or fold not in range(3):
        raise ValueError("invalid contextual fit population settings")
    admitted = sorted((x for x in examples if x.split in {"train", "validation"}),
                      key=lambda x: x.ir.source_text_sha256)
    identities = [x.ir.source_text_sha256 for x in admitted]
    if len(set(identities)) != len(identities):
        raise ValueError("contextual populations contain repeated source identities")
    source = [x for x in admitted if x.split == "train"]
    if full_source:
        train = source[:limit]
        evaluation = [x for x in admitted if x.split == "validation"][:limit]
    else:
        train = [x for x in source if folds["assignments"][x.ir.source_text_sha256] != fold][:limit]
        evaluation = [x for x in source if folds["assignments"][x.ir.source_text_sha256] == fold][:limit]
    if not train or not evaluation:
        raise ValueError("contextual fit requires nonempty training and evaluation populations")
    return train, evaluation


def restore_fit(checkpoint_path, *, model, optimizer, plan):
    """Restore an epoch boundary without changing the source experiment."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    previous = checkpoint["plan"]
    for key in ("fold_receipt", "fold", "seed", "epochs", "training_ids", "heldout_ids",
                "config", "labels", "context", "trainable_parameters", "span_width", "capacity"):
        if json.dumps(previous.get(key), sort_keys=True) != json.dumps(plan.get(key), sort_keys=True):
            raise ValueError(f"fit recovery changes {key}")
    if previous.get("evaluation_split", "train") != plan.get("evaluation_split", "train"):
        raise ValueError("fit recovery changes evaluation_split")
    tool_path = str(Path(__file__).resolve().relative_to(ROOT))
    for path, digest in previous["source_sha256"].items():
        if path != tool_path and plan["source_sha256"].get(path) != digest:
            raise ValueError(f"fit recovery changes numerical source: {path}")
    epoch = checkpoint["epoch"]
    if type(epoch) is not int or not 0 < epoch < plan["epochs"]:
        raise ValueError("fit recovery has an invalid epoch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    # Legacy fits used fixed row order and zero dropout; no random draw occurs
    # after initialization. Their omitted timings remain unmeasured.
    history = checkpoint.get("history", [{"epoch": index + 1, "online_train_nll": None,
                                         "elapsed_s": None} for index in range(epoch)])
    if len(history) != epoch or [row["epoch"] for row in history] != list(range(1, epoch + 1)):
        raise ValueError("fit recovery history is incomplete")
    return epoch, history, {"checkpoint_path": str(checkpoint_path),
                           "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                           "epoch": epoch, "rng_state_retained": "rng_state" in checkpoint,
                           "legacy_deterministic_training": "rng_state" not in checkpoint}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--full-source", action="store_true",
                        help="fit source training only, then measure exposed development; never fresh transfer")
    parser.add_argument("--position-mode", choices=("absolute", "relative", "none"), default="absolute")
    parser.add_argument("--feature-scaling", choices=("none", "unit_variance"), default="none")
    parser.add_argument("--context", choices=("full", "local", "frozen"), default="full")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.examples < 1 or args.epochs < 1 or args.fold not in range(3):
        parser.error("positive examples/epochs and fold 0..2 required")
    if args.output.exists():
        parser.error("output already exists")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )
    configure_refit_environment(args.output)
    import numpy as np
    import torch

    from core.learning.semantic_construction_folds import construction_folds
    from core.learning.semantic_context_objective import (
        best_operation_set,
        operation_scores,
        operation_set_loss,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_floor import PRIMITIVES_BY_NAME
    from core.learning.semantic_request_context import (
        ContextualSpanRecognizer,
        RequestContextConfig,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    torch.set_num_threads(2)
    torch.manual_seed(73)
    root = args.evidence_root
    parent = compositional_semantic_program_transducer_from_dict(json.loads(
        (root / "semantic-source-fit-20260915/candidate.json").read_text()))
    report = json.loads((root / "semantic-source-fit-20260915/report.json").read_text())
    examples = load_source_examples(parent, report, [name + "=" + str(
        root / "semantic-source-reacquisition-20260915/features" / name)
        for name in report["representation_compatibility"]["source_feature_manifest_sha256s"]])
    source = sorted((x for x in examples if x.split == "train"),
                    key=lambda x: x.ir.source_text_sha256)
    folds = construction_folds(source)
    frozen = json.loads((root / "semantic-architecture-source-folds-20260921/folds.json").read_text())
    if folds != frozen:
        # JSON converts provenance tuples to lists; compare canonical encoding.
        if json.dumps(folds, sort_keys=True) != json.dumps(frozen, sort_keys=True):
            raise ValueError("source folds differ from the frozen plan")
    train, held = select_populations(examples, folds, fold=args.fold,
                                    full_source=args.full_source, limit=args.examples)
    evaluation_split = "validation" if args.full_source else "train"
    labels = tuple(sorted(PRIMITIVES_BY_NAME))
    model = ContextualSpanRecognizer(RequestContextConfig(source[0].hidden_states.shape[1],
                                     width=32, heads=4, layers=1, position_mode=args.position_mode,
                                     feature_scaling=args.feature_scaling), labels)
    if args.context == "frozen":
        model.context.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=0.001, weight_decay=0.001)
    cross_token = args.context != "local"
    span_width = 24
    capacity = 16

    def loss_for(example, cross_token=cross_token):
        hidden = torch.from_numpy(np.array(example.hidden_states, copy=True))
        scores = operation_scores(model, hidden, span_width=span_width, cross_token=cross_token)
        target = tuple((i.operation_span.start, i.operation_span.end, labels.index(i.op))
                       for i in example.ir.instructions)
        return operation_set_loss(scores, target, max_operations=capacity)

    def measure(rows, cross_token=cross_token):
        with torch.no_grad():
            return sum(float(loss_for(x, cross_token)) for x in rows) / len(rows)

    def exact(rows, cross_token=cross_token):
        outcomes = []
        with torch.no_grad():
            for example in rows:
                hidden = torch.from_numpy(np.array(example.hidden_states, copy=True))
                scores = operation_scores(model, hidden, span_width=span_width, cross_token=cross_token)
                selected = best_operation_set(scores, max_operations=capacity)
                target = tuple(sorted((i.operation_span.start, i.operation_span.end, labels.index(i.op))
                                      for i in example.ir.instructions))
                outcomes.append({"source": example.ir.source_text_sha256,
                                 "selected": selected, "target": target, "exact": selected == target})
        return {"correct": sum(x["exact"] for x in outcomes), "count": len(outcomes), "rows": outcomes}

    initial = {"train_nll": measure(train),
               "source_heldout_nll": None if args.full_source else measure(held)}
    plan = {"fold_receipt": folds["receipt_sha256"], "fold": None if args.full_source else args.fold, "seed": 73,
            "evaluation_split": evaluation_split,
            "optimizer_split": "train", "evaluation_controls_training": False,
            "epochs": args.epochs, "training_ids": [x.ir.source_text_sha256 for x in train],
            "heldout_ids": [x.ir.source_text_sha256 for x in held],
            "config": vars(model.context.config), "labels": labels,
            "context": args.context,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "span_width": span_width, "capacity": capacity,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(), ROOT / "core/learning/semantic_request_context.py",
                          ROOT / "core/learning/semantic_context_objective.py",
                          ROOT / "core/learning/semantic_span_autograd.py"]}}
    start_epoch, history = 0, []
    if args.resume_checkpoint is not None:
        start_epoch, history, plan["recovery"] = restore_fit(
            args.resume_checkpoint, model=model, optimizer=optimizer, plan=plan)
    if not atomic_write_bytes_if_absent(args.output.with_suffix(".plan.json"),
                                       (json.dumps(plan, sort_keys=True) + "\n").encode()):
        raise RuntimeError("probe plan already exists")
    print(json.dumps({"initial": initial, "feature_width": model.context.config.input_width}), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        total = 0.
        for example in train:
            optimizer.zero_grad()
            loss = loss_for(example)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        item = {"epoch": epoch + 1, "online_train_nll": total / len(train),
                "elapsed_s": time.monotonic() - started}
        history.append(item)
        checkpoint = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "plan": plan, "history": history,
                    "rng_state": torch.get_rng_state()}, checkpoint)
        checkpoint_path = args.output.with_suffix(f".epoch-{epoch + 1}.pt")
        if not atomic_write_bytes_if_absent(checkpoint_path, checkpoint.getvalue()):
            raise RuntimeError("checkpoint already exists")
        item["checkpoint_sha256"] = hashlib.sha256(checkpoint.getvalue()).hexdigest()
        print(json.dumps(item), flush=True)
    result = {"schema": "aura.semantic_context_feasibility_probe.v1",
              "scope": "source_fit_exposed_development" if args.full_source else "source_fold_structured_loss_only",
              "evaluation_split": evaluation_split,
              "promotion_authorized": False,
              "full_graph_evaluated": False, "fold_receipt": folds["receipt_sha256"],
              "parent_receipt": parent.receipt_sha256, "seed": 73,
              "span_width": span_width, "operation_capacity": capacity,
              "training_ids": [x.ir.source_text_sha256 for x in train],
              "heldout_ids": [x.ir.source_text_sha256 for x in held],
              "initial": initial, "history": history,
              "final_train_nll": measure(train), "final_source_heldout_nll": measure(held),
              "context_lesion_source_heldout_nll": measure(held, False),
              "train_operation_sets": exact(train), "heldout_operation_sets": exact(held),
              "context_lesion_operation_sets": exact(held, False),
              "validation_used": args.full_source, "validation_used_for_training": False,
              "test_used": False}
    if not atomic_write_bytes_if_absent(args.output, (json.dumps(result, sort_keys=True) + "\n").encode()):
        raise RuntimeError("probe output could not be published")
    print(json.dumps({key: ({k: v for k, v in value.items() if k != "rows"}
                           if key.endswith("operation_sets") else value)
                      for key, value in result.items() if key not in {"training_ids", "heldout_ids"}}), flush=True)


if __name__ == "__main__":
    main()
