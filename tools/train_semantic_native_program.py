#!/usr/bin/env python3
"""Source-only native adapter fit, with an exact cached decoder prefix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def construction_subset(examples, identities, *, per_construction):
    """Select by source identity before any label or model score is observed."""
    if type(per_construction) is not int or per_construction < 1:
        raise ValueError("native pilot needs a positive per-construction count")
    by_id = {item.ir.source_text_sha256: item for item in examples}
    if len(set(identities)) != len(identities) or not set(identities) <= set(by_id):
        raise ValueError("native pilot identities differ from the frozen source cohort")
    groups = {}
    for identity in sorted(identities):
        groups.setdefault(by_id[identity].construction_id, []).append(identity)
    return tuple(sorted(identity for group in groups.values()
                        for identity in group[:per_construction]))


def native_loss(suffix, hidden, sequence, *, summed=False):
    """Supervise only the unchanged template's continuation, without truncation."""
    import mlx.core as mx
    import mlx.nn as nn

    start = sequence.continuation_start
    if (type(start) is not int or start < 1 or start >= len(sequence.tokens)
            or hidden.ndim != 3 or hidden.shape[0] != 1
            or hidden.shape[1] != len(sequence.tokens) - 1):
        raise ValueError("native supervision lost its causal token boundary")
    logits = suffix(hidden)[:, start - 1:].astype(mx.float32)
    targets = mx.array([sequence.tokens[start:]], dtype=mx.int32)
    if logits.shape[:2] != targets.shape:
        raise ValueError("native supervision lost its causal token boundary")
    losses = nn.losses.cross_entropy(logits, targets)
    return mx.sum(losses) if summed else mx.mean(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "source-report", "folds", "bank", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--bundle", action="append", metavar="NAME=PATH")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--held-per-construction", type=int, default=1)
    parser.add_argument("--calibration-per-construction", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (any(type(value) is not int or value < 1 for value in (
            args.steps, args.save_every, args.rank, args.layers, args.max_sequence_tokens))
            or args.steps % args.save_every or not 0 < args.max_seconds <= 14400):
        parser.error("positive sizes, complete checkpoint intervals and a finite time bound required")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
    )
    configure_refit_environment(args.directory / "report.json")
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.probe_semantic_proposer_crossfit import _digest, _save_if_absent
    from tools.train_nested_semantic_ranker import _verified_pair
    from tools.train_semantic_atom_ranker import construction_weights, validate_atom_partition

    outer, bank_report = _verified_pair(args.bank)
    raw = {name: path.read_bytes() for name, path in (
        ("parent", args.parent), ("source", args.source_report), ("folds", args.folds))}
    parent = compositional_semantic_program_transducer_from_dict(json.loads(raw["parent"]))
    source, folds = json.loads(raw["source"]), json.loads(raw["folds"])
    if (parent.receipt_sha256 != outer["parent_receipt_sha256"]
            or hashlib.sha256(raw["source"]).hexdigest() != outer["source_report_sha256"]
            or hashlib.sha256(raw["folds"]).hexdigest() != outer["folds_sha256"]):
        raise ValueError("native fit source basis differs from the frozen bank")
    bundles = source_bundle_arguments(source, feature_root=args.feature_root, bundles=args.bundle)
    examples = load_source_examples(parent, source, bundles)
    fit, calibration = validate_atom_partition(examples, outer, folds)
    held_ids = construction_subset(examples, outer["held_ids"],
                                    per_construction=args.held_per_construction)
    calibration_ids = construction_subset(examples, outer["calibration_ids"],
                                           per_construction=args.calibration_per_construction)
    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None or not spec.exact_identity:
        raise ValueError("native fit needs the exact current resident descriptor")
    for value in bundles:
        manifest = json.loads((Path(value.partition("=")[2]) / "manifest.json").read_bytes())
        if Path(manifest["exact_model_path"]).resolve() != spec.model_path.resolve():
            raise ValueError("source features came from a different resident model")
    implementation_paths = [ROOT / name for name in (
        "tools/train_semantic_native_program.py", "core/learning/frozen_decoder_prefix.py",
        "core/learning/semantic_native_program.py", "core/brain/llm/decoder_topology.py",
        "core/learning/semantic_program_feature_materialization.py",
        "core/runtime/mlx_memory_guard.py", "tools/evaluate_semantic_candidate_ranker.py",
        "tools/train_semantic_atom_ranker.py")]
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in implementation_paths}
    plan = {"schema": "aura.semantic_native_fit_plan.v1", "steps": args.steps,
            "save_every": args.save_every, "rank": args.rank, "suffix_layers": args.layers,
            "adapter_keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.down_proj"],
            "learning_rate": 1e-4, "weight_decay": .01, "seed": 20260925,
            "max_seconds": args.max_seconds, "max_sequence_tokens": args.max_sequence_tokens,
            "model_descriptor_sha256": spec.descriptor_sha256,
            "model_path": str(spec.model_path), "pointer_sha256": spec.pointer_sha256,
            "bank_plan_sha256": outer["plan_sha256"],
            "bank_receipt_sha256": bank_report["receipt_sha256"],
            "source_report_sha256": outer["source_report_sha256"],
            "fit_ids": outer["fit_ids"], "calibration_ids": calibration_ids,
            "complete_calibration_population": len(calibration), "held_ids": held_ids,
            "heldout_axis": outer["heldout_axis"],
            "selection": "minimum_source_calibration_continuation_loss",
            "matched_control": "same_native_suffix_without_fitted_lora",
            "input": "unchanged_source_request_with_native_chat_template",
            "scoring": "summed_native_continuation_log_probability",
            "implementation": implementation, "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**plan, "plan_sha256": _digest(plan)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "plan_sha256": plan["plan_sha256"],
                          "fit": len(fit), "calibration": len(calibration_ids),
                          "held": len(held_ids), "model_weights_loaded": False}), flush=True)
        return
    if any(args.directory.glob("checkpoint-*.json")):
        raise FileExistsError("native fit already has evidence; use a fresh experiment directory")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_program import (
        native_program_sequence,
        source_text_from_tokens,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.evaluate_semantic_candidate_ranker import _rankable_or_none, _read_bank

    started = time.monotonic()
    def check_bound():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("native fit bound reached; durable checkpoints remain research-only")

    with (standalone_model_lane(owner_id=f"semantic-native:{args.directory.name}",
            model_path=str(spec.model_path), purpose="training", preemptible=False,
            metadata={"tool": "train_semantic_native_program", "production_effect": False}),
          mlx_memory_envelope(fraction=.80) as envelope):
        print(json.dumps({"stage": "load", "descriptor": spec.descriptor_sha256,
                          "memory_envelope": envelope.to_receipt()}), flush=True)
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - args.layers
        prefix = FrozenDecoderPrefix(model, split_at=split)
        suffix = NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(plan["seed"])
        linear_to_lora_layers(model, args.layers, {
            "rank": args.rank, "scale": 16., "dropout": 0., "keys": plan["adapter_keys"]})
        trainable = tree_flatten(suffix.trainable_parameters())
        if not trainable or any("lora_" not in name for name, _value in trainable):
            raise ValueError("native suffix adaptation escaped its declared LoRA sites")
        if any("lora_" not in name for name, _value in tree_flatten(model.trainable_parameters())):
            raise ValueError("native fit would update frozen resident parameters")
        baseline_weights = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(baseline_weights)
        for layer in suffix.layers:
            layer.train()
        items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
        texts = {identity: source_text_from_tokens(item, tokenizer) for identity, item in items.items()}
        public_by_id = {identity: item.public_inputs for identity, item in items.items()}
        sequences = {identity: native_program_sequence(texts[identity], item.ir.to_program(),
                      tokenizer, max_tokens=args.max_sequence_tokens) for identity, item in items.items()
                      if identity in set(plan["fit_ids"]) | set(calibration_ids)}
        weights = construction_weights(fit)
        construction_by_id = {identity: item.construction_id for identity, item in items.items()}
        del examples, fit, calibration, items, parent
        gc.collect()
        captured = {}
        for index, (identity, sequence) in enumerate(sequences.items(), 1):
            check_bound()
            tokens = mx.array([sequence.tokens[:-1]], dtype=mx.int32)
            hidden = prefix.capture(tokens)
            if index == 1:
                difference = mx.max(mx.abs(model(tokens) - suffix(hidden))).item()
                if not math.isfinite(difference) or difference > .01:
                    raise ValueError("cached native prefix does not reproduce model logits")
                _save_if_absent(args.directory / "prefix-equivalence.json", {
                    "plan_sha256": plan["plan_sha256"], "max_absolute_logit_difference": difference,
                    "tokens": len(sequence.tokens) - 1, "split_at": split,
                    "trainable_sites": [name for name, _value in trainable]})
            captured[identity] = hidden
            if index % 16 == 0 or index == len(sequences):
                print(json.dumps({"stage": "prefix", "captured": index,
                                  "population": len(sequences),
                                  "elapsed_seconds": time.monotonic() - started,
                                  "active_memory_bytes": mx.get_active_memory()}), flush=True)
        baseline = sum(native_loss(suffix, captured[identity], sequences[identity]).item()
                       for identity in calibration_ids) / len(calibration_ids)
        optimizer = optim.AdamW(learning_rate=plan["learning_rate"], weight_decay=plan["weight_decay"])
        order = list(plan["fit_ids"])
        rng = random.Random(plan["seed"])
        best, checkpoints, history = None, [], []
        for step in range(1, args.steps + 1):
            check_bound()
            if (step - 1) % len(order) == 0:
                rng.shuffle(order)
            identity = order[(step - 1) % len(order)]
            def weighted_loss(tail, hidden, sequence, weight=weights[identity]):
                return native_loss(tail, hidden, sequence) * weight
            loss, gradients = nn.value_and_grad(suffix, weighted_loss)(
                suffix, captured[identity], sequences[identity])
            norm = mx.sqrt(sum(mx.sum(value.astype(mx.float32) ** 2)
                               for _name, value in tree_flatten(gradients)))
            if not math.isfinite(norm.item()):
                raise ValueError("native semantic fit produced nonfinite gradients")
            gradients = tree_map(lambda value, scale=norm: value / mx.maximum(scale, 1.), gradients)
            optimizer.update(suffix, gradients)
            mx.eval(suffix.parameters(), optimizer.state, loss)
            if not math.isfinite(loss.item()):
                raise ValueError("native semantic fit produced nonfinite loss")
            history.append({"step": step, "loss": loss.item()})
            if step % 8 == 0:
                print(json.dumps({"stage": "fit", **history[-1]}), flush=True)
            if step % args.save_every == 0:
                calibration_loss = sum(native_loss(suffix, captured[identity], sequences[identity]).item()
                                       for identity in calibration_ids) / len(calibration_ids)
                tensors = dict(tree_flatten(model.trainable_parameters()))
                weight_path = args.directory / f"checkpoint-{step}.safetensors"
                stream = io.BytesIO()
                mx.save_safetensors(stream, tensors)
                payload = stream.getvalue()
                if not atomic_write_bytes_if_absent(weight_path, payload, mode=0o400):
                    raise FileExistsError(weight_path)
                row = {"plan_sha256": plan["plan_sha256"], "step": step,
                       "calibration_loss": calibration_loss,
                       "weights_sha256": hashlib.sha256(payload).hexdigest()}
                row = {**row, "receipt_sha256": _digest(row)}
                _save_if_absent(args.directory / f"checkpoint-{step}.json", row)
                checkpoints.append(row)
                if best is None or (calibration_loss, step) < (best[0], best[1]):
                    best = (calibration_loss, step, weight_path)
                print(json.dumps({"stage": "checkpoint", **row}), flush=True)
        model.load_weights(str(best[2]), strict=False)
        selected_weights = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(selected_weights)
        del captured, optimizer, gradients
        gc.collect()
        rows = []
        for identity in held_ids:
            check_bound()
            bank_row = _read_bank(args.bank / "rows" / f"{identity}.json", source=identity,
                plan_sha=outer["plan_sha256"], model_receipt=bank_report["candidate_receipt_sha256"],
                expected_receipt=bank_report["row_receipts"][identity])
            # Grounding types are public; comparison labels remain outside the scorer.
            from types import SimpleNamespace
            view = _rankable_or_none(SimpleNamespace(public_inputs=public_by_id[identity]), bank_row)
            if view is None:
                row = {"source": identity, "incumbent_correct": False,
                       "selected_correct": None, "bank_reachable": None,
                       "pretrained_correct": None,
                       "status": "bank_unrankable", "plan_sha256": plan["plan_sha256"]}
                row = {**row, "receipt_sha256": _digest(row)}
                rows.append(row)
                _save_if_absent(args.directory / "rows" / f"{identity}.json", row)
                continue
            (programs, labels, keys), _spans, _kinds, _anchors = view
            scores, pretrained_scores = [], []
            for program in programs:
                check_bound()
                sequence = native_program_sequence(texts[identity], program, tokenizer,
                                                    max_tokens=args.max_sequence_tokens)
                hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                score = -native_loss(suffix, hidden, sequence, summed=True).item()
                if not math.isfinite(score):
                    raise ValueError("native bank score is not finite")
                scores.append(score)
                suffix.update(baseline_weights)
                try:
                    pretrained_score = -native_loss(suffix, hidden, sequence, summed=True).item()
                    if not math.isfinite(pretrained_score):
                        raise ValueError("pretrained native bank score is not finite")
                    pretrained_scores.append(pretrained_score)
                finally:
                    suffix.update(selected_weights)
            chosen = max(range(len(scores)), key=scores.__getitem__)
            pretrained_chosen = max(range(len(pretrained_scores)), key=pretrained_scores.__getitem__)
            row = {"source": identity, "construction": construction_by_id[identity],
                   "incumbent_correct": labels[0], "selected_correct": labels[chosen],
                   "bank_reachable": any(labels), "chosen_program_sha256": keys[chosen],
                   "scores": scores, "program_sha256s": keys,
                   "pretrained_correct": labels[pretrained_chosen],
                   "pretrained_program_sha256": keys[pretrained_chosen],
                   "pretrained_scores": pretrained_scores,
                   "labels_available_to_scorer": False, "plan_sha256": plan["plan_sha256"]}
            row = {**row, "receipt_sha256": _digest(row)}
            rows.append(row)
            _save_if_absent(args.directory / "rows" / f"{identity}.json", row)
            print(json.dumps({"stage": "held", "observed": len(rows), "population": len(held_ids)}),
                  flush=True)
        current_spec = get_active_cortex_spec(force_refresh=True)
        if (current_spec is None or current_spec.descriptor_sha256 != spec.descriptor_sha256
                or current_spec.pointer_sha256 != spec.pointer_sha256
                or any(hashlib.sha256(path.read_bytes()).hexdigest() != implementation[
                       str(path.relative_to(ROOT))] for path in implementation_paths)
                or any(path.read_bytes() != raw[name] for name, path in (
                    ("parent", args.parent), ("source", args.source_report), ("folds", args.folds)))):
            raise ValueError("native fit identity changed during measurement")
        body = {"schema": "aura.semantic_native_fit.v1", "plan_sha256": plan["plan_sha256"],
                "selected_step": best[1], "baseline_calibration_loss": baseline,
                "memory_envelope": envelope.to_receipt(),
                "selected_calibration_loss": best[0], "checkpoints": checkpoints,
                "history": history, "population": len(rows), "rows": rows,
                "incumbent_correct": sum(row["incumbent_correct"] for row in rows),
                "native_correct": sum(row["selected_correct"] is True for row in rows),
                "pretrained_correct": sum(row["pretrained_correct"] is True for row in rows),
                "learning_gains": sum(row["selected_correct"] is True
                                      and row["pretrained_correct"] is False for row in rows),
                "learning_regressions": sum(row["pretrained_correct"] is True
                                            and row["selected_correct"] is False for row in rows),
                "bank_reachable": sum(row["bank_reachable"] is True for row in rows),
                "gains": sum(row["selected_correct"] is True and not row["incumbent_correct"]
                             for row in rows),
                "regressions": sum(row["incumbent_correct"] and row["selected_correct"] is False
                                   for row in rows),
                "elapsed_seconds": time.monotonic() - started,
                "serving_authority": False, "qualification_evidence": False,
                "held_labels_used_for_fit_or_selection": False}
        _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": _digest(body)})
        print(json.dumps({"stage": "complete", "native_correct": body["native_correct"],
                          "population": len(rows), "regressions": body["regressions"]}), flush=True)


if __name__ == "__main__":
    main()
