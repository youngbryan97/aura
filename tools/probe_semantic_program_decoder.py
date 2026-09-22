#!/usr/bin/env python3
"""Fit whole programs on a frozen source-construction fold, then evaluate once."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--feature-scaling", choices=("none", "unit_variance"), default="none")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.epochs <= 100:
        parser.error("use a new output and 1..100 fixed epochs")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    import numpy as np
    import torch

    from core.learning.semantic_construction_folds import construction_folds
    from core.learning.semantic_graph_coordinates import reanchor_program_inputs
    from core.learning.semantic_graph_counterexamples import (
        compare_program_meanings,
        counterfactual_inputs,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_decoder import ProgramDecoderConfig, SemanticProgramDecoder
    from core.learning.semantic_validation_checkpoint import validation_implementation_identity
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.probe_semantic_request_context import restore_fit, select_populations

    torch.set_num_threads(2)
    torch.manual_seed(73)
    root = args.evidence_root
    parent = compositional_semantic_program_transducer_from_dict(
        json.loads((root / "semantic-source-fit-20260915/candidate.json").read_text())
    )
    source_report = json.loads((root / "semantic-source-fit-20260915/report.json").read_text())
    examples = load_source_examples(
        parent,
        source_report,
        [
            name + "=" + str(root / "semantic-source-reacquisition-20260915/features" / name)
            for name in source_report["representation_compatibility"][
                "source_feature_manifest_sha256s"
            ]
        ],
    )
    folds = construction_folds(tuple(x for x in examples if x.split == "train"))
    frozen = json.loads(
        (root / "semantic-architecture-source-folds-20260921/folds.json").read_text()
    )
    if json.dumps(folds, sort_keys=True) != json.dumps(frozen, sort_keys=True):
        raise ValueError("source construction folds differ from frozen plan")
    train, held = select_populations(
        examples, folds, fold=args.fold, full_source=False, limit=10000
    )
    model = SemanticProgramDecoder(
        ProgramDecoderConfig(
            parent.hidden_size, args.width, max_steps=16, feature_scaling=args.feature_scaling
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    plan = {
        "schema": "aura.semantic_direct_program_source_probe.v1",
        "fold": args.fold,
        "fold_receipt": folds["receipt_sha256"],
        "seed": 73,
        "epochs": args.epochs,
        "config": vars(model.config),
        "operations": model.operations,
        "parent_receipt": parent.receipt_sha256,
        "training_ids": [x.ir.source_text_sha256 for x in train],
        "heldout_ids": [x.ir.source_text_sha256 for x in held],
        "optimizer_split": "train",
        "evaluation_controls_training": False,
        "operation_span_labels_used": False,
        "frozen_grounding_saw_source_fold": True,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "implementation_identity": validation_implementation_identity(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "core/learning/semantic_program_decoder.py",
                ROOT / "core/learning/semantic_graph_coordinates.py",
            )
        },
    }
    start_epoch, history = 0, []
    if args.resume_checkpoint:
        start_epoch, history, plan["recovery"] = restore_fit(
            args.resume_checkpoint, model=model, optimizer=optimizer, plan=plan
        )
    if not atomic_write_bytes_if_absent(
        args.output.with_suffix(".plan.json"), json.dumps(plan, sort_keys=True).encode()
    ):
        raise ValueError("direct decoder plan already exists")
    grounded = {}

    def inputs(item):
        identity = item.ir.source_text_sha256
        if identity not in grounded:
            spans, _, _ = parent._runtime_input_grounding(
                item.ir.source_token_ids, item.hidden_states, item.public_inputs
            )
            grounded[identity] = (
                tuple(spans),
                tuple(
                    "integer" if type(x) is int else "integer_sequence" for x in item.public_inputs
                ),
            )
        spans, kinds = grounded[identity]
        return torch.from_numpy(np.array(item.hidden_states, copy=True)), spans, kinds

    print(
        json.dumps(
            {
                "stage": "start",
                "training": len(train),
                "heldout": len(held),
                "parameters": plan["trainable_parameters"],
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        for index in torch.randperm(len(train)).tolist():
            item = train[index]
            features, spans, kinds = inputs(item)
            target = reanchor_program_inputs(
                item.ir.to_program(),
                from_spans=item.ir.input_spans,
                to_spans=spans,
                from_inputs=item.public_inputs,
                to_inputs=item.public_inputs,
            )
            optimizer.zero_grad()
            loss = model.loss(features, spans, kinds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        row = {
            "epoch": epoch + 1,
            "online_train_nll": loss_sum / len(train),
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        buffer = io.BytesIO()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "plan": plan,
                "history": history,
                "rng_state": torch.get_rng_state(),
            },
            buffer,
        )
        if not atomic_write_bytes_if_absent(
            args.output.with_suffix(f".epoch-{epoch + 1}.pt"), buffer.getvalue()
        ):
            raise ValueError("direct decoder epoch already exists")
        row["checkpoint_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
        print(json.dumps(row), flush=True)
    model.eval()
    outcomes = {}
    for name, population in (("train", train), ("source_heldout", held)):
        rows = []
        for item in population:
            features, spans, kinds = inputs(item)
            before = time.monotonic()
            selected, receipt = model.decode(features, spans, kinds)
            elapsed = time.monotonic() - before
            normalized = reanchor_program_inputs(
                selected,
                from_spans=spans,
                to_spans=item.ir.input_spans,
                from_inputs=item.public_inputs,
                to_inputs=item.public_inputs,
            )
            comparison = compare_program_meanings(
                item.ir.to_program(),
                normalized,
                counterfactual_inputs(item.public_inputs, count=16),
            )
            row = {
                "source": item.ir.source_text_sha256,
                "program": normalized.to_dict(),
                "comparison": comparison,
                "decode": receipt,
                "decode_s": elapsed,
            }
            rows.append(row)
            if not atomic_write_bytes_if_absent(
                args.output.parent / name / (row["source"] + ".json"),
                json.dumps(row, sort_keys=True).encode(),
            ):
                raise ValueError("direct decoder observation already exists")
        outcomes[name] = {
            "count": len(rows),
            "correct": sum(r["comparison"]["status"] == "equivalent" for r in rows),
            "rows": rows,
        }
        print(
            json.dumps({"stage": name, "count": len(rows), "correct": outcomes[name]["correct"]}),
            flush=True,
        )
    if validation_implementation_identity() != plan["implementation_identity"]:
        raise ValueError("direct decoder implementation changed during measurement")
    if any(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest
        for p, digest in plan["source_sha256"].items()
    ):
        raise ValueError("direct decoder fit source changed")
    result = {
        "plan": plan,
        "history": history,
        "outcomes": outcomes,
        "promotion_authorized": False,
        "validation_used": False,
        "test_used": False,
    }
    if not atomic_write_bytes_if_absent(args.output, json.dumps(result, sort_keys=True).encode()):
        raise ValueError("direct decoder report already exists")


if __name__ == "__main__":
    main()
