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


def exact_length_batches(sequences, *, batch_size):
    """Batch independent full sequences without adding padding or dropping tokens."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("native prefix batch size must be inside [1, 32]")
    groups = {}
    for identity, sequence in sorted(sequences.items()):
        groups.setdefault(len(sequence.tokens) - 1, []).append(identity)
    for length in sorted(groups):
        if length < 1:
            raise ValueError("native prefix batch contains an empty causal sequence")
        for start in range(0, len(groups[length]), batch_size):
            yield tuple(groups[length][start:start + batch_size])


def native_training_schedule(identities, *, steps, seed):
    """Freeze the original shuffled update order before capturing any states."""
    order = list(identities)
    if (not order or len(set(order)) != len(order) or type(steps) is not int
            or steps < 1 or type(seed) is not int):
        raise ValueError("native schedule needs unique eligible sources and positive steps")
    rng, schedule = random.Random(seed), []
    for index in range(steps):
        if index % len(order) == 0:
            rng.shuffle(order)
        schedule.append(order[index % len(order)])
    return tuple(schedule)


def native_prediction_positions(sequence, *, scope):
    """Identify targets before vocabulary projection, retaining every causal state."""
    start = sequence.continuation_start
    if type(start) is not int or not 1 <= start < len(sequence.tokens):
        raise ValueError("native supervision lost its causal token boundary")
    if scope == "continuation":
        return tuple(range(start, len(sequence.tokens)))
    if scope != "semantic_decisions":
        raise ValueError("unknown native supervision scope")
    positions = sequence.semantic_positions
    if (not positions or tuple(sorted(set(positions))) != positions
            or any(type(index) is not int or not start <= index < len(sequence.tokens)
                   for index in positions)):
        raise ValueError("native semantic loss needs complete offset-bound decision positions")
    return positions


def native_loss(suffix, hidden, sequence, *, summed=False, scope="continuation"):
    """Supervise only the unchanged template's continuation, without truncation."""
    import mlx.core as mx
    import mlx.nn as nn

    positions = native_prediction_positions(sequence, scope=scope)
    if (hidden.ndim != 3 or hidden.shape[0] != 1
            or hidden.shape[1] != len(sequence.tokens) - 1):
        raise ValueError("native supervision lost its causal token boundary")
    logits = suffix(hidden, logit_positions=tuple(index - 1 for index in positions)).astype(mx.float32)
    targets = mx.array([[sequence.tokens[index] for index in positions]], dtype=mx.int32)
    if logits.shape[:2] != targets.shape:
        raise ValueError("native supervision lost its causal token boundary")
    losses = nn.losses.cross_entropy(logits, targets)
    return mx.sum(losses) if summed else mx.mean(losses)


def native_source_loss(suffix, hidden_rows, sequences, *, scope, objective):
    """Train native likelihood and witnessed graph competition on the same source."""
    import mlx.core as mx

    from core.learning.semantic_native_program import native_choice_loss

    if (not sequences or len(sequences) != len(hidden_rows)
            or objective not in {"token", "contrastive"}
            or objective == "token" and len(sequences) != 1
            or objective == "contrastive" and (len(sequences) < 2 or scope != "semantic_decisions")):
        raise ValueError("native source objective differs from its frozen supervision set")
    scores = mx.stack([-native_loss(suffix, hidden, sequence, summed=True, scope=scope)
                       for hidden, sequence in zip(hidden_rows, sequences, strict=True)])
    loss = -scores[0] / len(native_prediction_positions(sequences[0], scope=scope))
    return loss if objective == "token" else loss + native_choice_loss(scores, (0,))


def native_relational_source_loss(suffix, left_hidden, left_sequences,
                                  right_hidden, right_sequences):
    """Optimize both source forms with extra weight on the weaker one."""
    import mlx.core as mx

    losses = mx.stack((
        native_source_loss(suffix, left_hidden, left_sequences,
                           scope="semantic_decisions", objective="contrastive"),
        native_source_loss(suffix, right_hidden, right_sequences,
                           scope="semantic_decisions", objective="contrastive")))
    return mx.logsumexp(losses) - math.log(2.)


def native_source_embedding(suffix, hidden, sequence):
    """Embed only the request prefix, excluding any answer-boundary token."""
    import mlx.core as mx

    cutoff = sequence.continuation_start - 1
    if (type(cutoff) is not int or cutoff < 1 or hidden.ndim != 3
            or hidden.shape[0] != 1 or hidden.shape[1] != len(sequence.tokens) - 1
            or cutoff > hidden.shape[1]):
        raise ValueError("relation embedding escaped its source-only token boundary")
    state = suffix.normalized_states(hidden[:, :cutoff])[:, -1, :].astype(mx.float32)
    return state / mx.maximum(mx.sqrt(mx.sum(state * state, axis=-1, keepdims=True)), 1e-6)


def native_relation_metric_loss(suffix, anchor, positive, negative):
    """Prefer a cross-form same-relation source over a same-form rival relation."""
    import mlx.core as mx

    vectors = [native_source_embedding(suffix, hidden, sequence)
               for hidden, sequence in (anchor, positive, negative)]
    same = mx.sum(vectors[0] * vectors[1], axis=-1)
    rival = mx.sum(vectors[0] * vectors[2], axis=-1)
    return mx.mean(mx.logaddexp(0., (rival - same) / .1))


def selected_projection_error(full_logits, selected_logits, sequence, positions):
    """Compare the supervised log probabilities, allowing one BF16 rounding step."""
    import mlx.core as mx

    if (not positions or any(type(index) is not int or
            not 0 <= index + 1 < len(sequence.tokens) for index in positions)):
        raise ValueError("selected projection lacks valid supervised positions")
    target_ids = [sequence.tokens[index + 1] for index in positions]
    reference = mx.take(full_logits, mx.array(positions, dtype=mx.int32), axis=1)
    if (reference.shape != selected_logits.shape or reference.shape[0] != 1
            or any(type(target) is not int or not 0 <= target < reference.shape[-1]
                   for target in target_ids)):
        raise ValueError("selected projection differs from the complete causal states")
    targets = mx.array(target_ids, dtype=mx.int32)
    def target_logprob(logits):
        rows = logits[0].astype(mx.float32)
        chosen = mx.take_along_axis(rows, targets[:, None], axis=1)[:, 0]
        return chosen - mx.logsumexp(rows, axis=-1)
    error = mx.max(mx.abs(target_logprob(reference) - target_logprob(selected_logits))).item()
    tolerance = 2 * max(mx.finfo(reference.dtype).eps, mx.finfo(selected_logits.dtype).eps)
    return float(error), float(tolerance)


def native_supervision_sets(items, texts, tokenizer, identities, *, peers=(),
                            contrast_limit=None, max_tokens=1024):
    """Reuse the floor's witnessed contrasts only for declared source supervision."""
    from core.learning.semantic_candidate_contrasts import source_program_contrasts
    from core.learning.semantic_native_program import native_program_sequence

    if (not identities or len(set(identities)) != len(identities)
            or not set(identities) <= set(items) or not set(identities) <= set(texts)):
        raise ValueError("native supervision source identities differ")
    sequences, groups = {}, {}
    for identity in sorted(identities):
        item = items[identity]
        if item.split != "train" or item.ir.source_text_sha256 != identity:
            raise ValueError("native supervision cannot consume validation or test targets")
        target = item.ir.to_program()
        programs = (target,) if contrast_limit is None else source_program_contrasts(
            target, item.public_inputs, peers, source_sha256=identity, limit=contrast_limit)
        programs = (target, *sorted((program for program in programs if program != target),
                                    key=lambda program: program.sha()))
        groups[identity] = tuple((identity, program.sha()) for program in programs)
        for key, program in zip(groups[identity], programs, strict=True):
            sequences[key] = native_program_sequence(texts[identity], program, tokenizer,
                                                      max_tokens=max_tokens)
    return sequences, groups


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
    parser.add_argument("--prefix-batch-size", type=int, default=1)
    parser.add_argument("--held-per-construction", type=int, default=1)
    parser.add_argument("--calibration-per-construction", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--loss-scope", choices=("continuation", "semantic_decisions"),
                        default="continuation")
    parser.add_argument("--objective", choices=("token", "contrastive", "relational", "relational_metric"), default="token")
    parser.add_argument("--contrast-limit", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (any(type(value) is not int or value < 1 for value in (
            args.steps, args.save_every, args.rank, args.layers, args.max_sequence_tokens))
            or args.steps % args.save_every or not 0 < args.max_seconds <= 14400
            or not 1 <= args.prefix_batch_size <= 32 or not 2 <= args.contrast_limit <= 32
            or args.objective in {"contrastive", "relational", "relational_metric"}
            and args.loss_scope != "semantic_decisions"):
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
    from core.learning.semantic_counterfactual_corpus import (
        cross_construction_relation_partners,
        cross_construction_relation_triplets,
    )

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
    schedule = native_training_schedule(outer["fit_ids"], steps=args.steps, seed=20260925)
    relational_partners = (cross_construction_relation_partners(tuple(fit))
                           if args.objective == "relational" else {})
    metric_triplets = (cross_construction_relation_triplets(tuple(fit))
                       if args.objective == "relational_metric" else {})
    scheduled_partners = {identity: relational_partners[identity] for identity in set(schedule)
                          if identity in relational_partners}
    scheduled_triplets = {identity: metric_triplets[identity] for identity in set(schedule)
                          if identity in metric_triplets}
    if args.objective == "relational" and not scheduled_partners:
        raise ValueError("relational objective has no scheduled cross-construction fit partners")
    if args.objective == "relational_metric" and not scheduled_triplets:
        raise ValueError("relational metric objective has no scheduled fit triplets")
    captured_fit_ids = sorted(set(schedule) | set(scheduled_partners.values())
                              | {source for pair in scheduled_triplets.values() for source in pair})
    peer_programs = {item.ir.to_program().sha(): item.ir.to_program() for item in fit}
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
        "core/learning/semantic_candidate_contrasts.py",
        "core/learning/semantic_counterfactual_corpus.py",
        "core/learning/semantic_graph_counterexamples.py",
        "core/learning/semantic_program_floor.py",
        "core/runtime/mlx_memory_guard.py", "tools/evaluate_semantic_candidate_ranker.py",
        "tools/train_semantic_atom_ranker.py")]
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in implementation_paths}
    plan = {"schema": "aura.semantic_native_fit_plan.v1", "steps": args.steps,
            "save_every": args.save_every, "rank": args.rank, "suffix_layers": args.layers,
            "prefix_batch_size": args.prefix_batch_size,
            "prefix_padding": False,
            "adapter_keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.down_proj"],
            "learning_rate": 1e-4, "weight_decay": .01, "seed": 20260925,
            "loss_scope": args.loss_scope,
            "semantic_decision_basis": "program_atoms_and_graph_termination_v1",
            "objective": args.objective, "contrast_limit": args.contrast_limit,
            "relational_fit_partners": dict(sorted(scheduled_partners.items())),
            "relational_paired_updates": sum(identity in scheduled_partners for identity in schedule),
            "relational_metric_fit_triplets": dict(sorted(scheduled_triplets.items())),
            "relational_metric_updates": sum(identity in scheduled_triplets for identity in schedule),
            "relational_metric_weight": .1 if args.objective == "relational_metric" else 0.,
            "relational_metric_temperature": .1 if args.objective == "relational_metric" else None,
            "contrast_policy": "source_floor_typed_witnessed_difference_v1",
            "contrast_weight": 1.0 if args.objective != "token" else 0.0,
            "supervision_peer_program_sha256s": sorted(peer_programs)
                if args.objective != "token" else [],
            "max_seconds": args.max_seconds, "max_sequence_tokens": args.max_sequence_tokens,
            "model_descriptor_sha256": spec.descriptor_sha256,
            "model_path": str(spec.model_path), "pointer_sha256": spec.pointer_sha256,
            "bank_plan_sha256": outer["plan_sha256"],
            "bank_receipt_sha256": bank_report["receipt_sha256"],
            "source_report_sha256": outer["source_report_sha256"],
            "fit_ids": outer["fit_ids"], "calibration_ids": calibration_ids,
            "scheduled_fit_ids": schedule,
            "captured_fit_ids": captured_fit_ids,
            "complete_calibration_population": len(calibration), "held_ids": held_ids,
            "heldout_axis": outer["heldout_axis"],
            "selection": "minimum_source_calibration_" + args.objective + "_" + args.loss_scope,
            "unfitted_checkpoint_eligible": True,
            "matched_control": "same_native_suffix_without_fitted_lora",
            "input": "unchanged_source_request_with_native_chat_template",
            "scoring": "summed_native_" + args.loss_scope + "_log_probability",
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
        supervised_ids = tuple(sorted(set(captured_fit_ids) | set(calibration_ids)))
        sequences, groups = native_supervision_sets(
            items, texts, tokenizer, supervised_ids,
            peers=tuple(peer_programs[key] for key in sorted(peer_programs)),
            contrast_limit=args.contrast_limit if args.objective != "token" else None,
            max_tokens=args.max_sequence_tokens)
        supervision = {"plan_sha256": plan["plan_sha256"],
            "rows": [{"source": key[0], "program_sha256": key[1],
                      "tokens": sequence.tokens, "continuation_start": sequence.continuation_start,
                      "semantic_positions": sequence.semantic_positions}
                     for key, sequence in sorted(sequences.items())]}
        supervision = {**supervision, "receipt_sha256": _digest(supervision)}
        _save_if_absent(args.directory / "supervision.json", supervision)
        weights = construction_weights(fit)
        calibration_weights = construction_weights([items[identity] for identity in calibration_ids])
        construction_by_id = {identity: item.construction_id for identity, item in items.items()}
        del examples, fit, calibration, items, parent
        gc.collect()
        captured = {}
        for batch in exact_length_batches(sequences, batch_size=args.prefix_batch_size):
            check_bound()
            tokens = mx.array([sequences[identity].tokens[:-1] for identity in batch], dtype=mx.int32)
            hidden = prefix.capture(tokens)
            if not captured:
                full = model(tokens)
                difference = mx.max(mx.abs(full - suffix(hidden))).item()
                positions = tuple(index - 1 for index in native_prediction_positions(
                    sequences[batch[0]], scope=args.loss_scope))
                selected_difference, selected_tolerance = selected_projection_error(
                    full, suffix(hidden, logit_positions=positions),
                    sequences[batch[0]], positions)
                if (not math.isfinite(difference) or difference > .01
                        or not math.isfinite(selected_difference)
                        or selected_difference > selected_tolerance):
                    raise ValueError("cached native prefix does not reproduce model logits")
                _save_if_absent(args.directory / "prefix-equivalence.json", {
                    "plan_sha256": plan["plan_sha256"], "max_absolute_logit_difference": difference,
                    "tokens": tokens.shape[1], "batch_size": tokens.shape[0], "split_at": split,
                    "selected_projection_max_target_logprob_difference": selected_difference,
                    "selected_projection_target_logprob_tolerance": selected_tolerance,
                    "projected_positions": len(positions),
                    "trainable_sites": [name for name, _value in trainable]})
            for index, identity in enumerate(batch):
                captured[identity] = hidden[index:index + 1]
            if len(captured) % 16 < len(batch) or len(captured) == len(sequences):
                print(json.dumps({"stage": "prefix", "captured": len(captured),
                                  "population": len(sequences),
                                  "elapsed_seconds": time.monotonic() - started,
                                  "active_memory_bytes": mx.get_active_memory()}), flush=True)
        def source_objective(tail, identity, states):
            keys = groups[identity]
            hidden = [states[key] for key in keys]
            rows = [sequences[key] for key in keys]
            partner = scheduled_partners.get(identity) if args.objective == "relational" else None
            if partner is not None:
                peer_keys = groups[partner]
                return native_relational_source_loss(tail, hidden, rows,
                    [states[key] for key in peer_keys], [sequences[key] for key in peer_keys])
            source_loss = native_source_loss(tail, hidden, rows, scope=args.loss_scope,
                objective="contrastive" if args.objective in {"relational", "relational_metric"}
                else args.objective)
            triplet = scheduled_triplets.get(identity) if args.objective == "relational_metric" else None
            if triplet is None:
                return source_loss
            def source_pair(source):
                key = groups[source][0]
                return states[key], sequences[key]
            return source_loss + plan["relational_metric_weight"] * native_relation_metric_loss(
                tail, source_pair(identity), source_pair(triplet[0]), source_pair(triplet[1]))

        def measure_calibration(states):
            return sum(source_objective(suffix, identity, states).item() * calibration_weights[identity]
                       for identity in calibration_ids) / len(calibration_ids)

        def save_checkpoint(step, calibration_loss):
            weight_path = args.directory / f"checkpoint-{step}.safetensors"
            stream = io.BytesIO()
            mx.save_safetensors(stream, dict(tree_flatten(model.trainable_parameters())))
            payload = stream.getvalue()
            if not atomic_write_bytes_if_absent(weight_path, payload, mode=0o400):
                raise FileExistsError(weight_path)
            row = {"plan_sha256": plan["plan_sha256"], "step": step,
                   "calibration_loss": calibration_loss,
                   "weights_sha256": hashlib.sha256(payload).hexdigest()}
            row = {**row, "receipt_sha256": _digest(row)}
            _save_if_absent(args.directory / f"checkpoint-{step}.json", row)
            print(json.dumps({"stage": "checkpoint", **row}), flush=True)
            return weight_path, row

        baseline = measure_calibration(captured)
        optimizer = optim.AdamW(learning_rate=plan["learning_rate"], weight_decay=plan["weight_decay"])
        zero_path, zero = save_checkpoint(0, baseline)
        best, checkpoints, history = (baseline, 0, zero_path), [zero], []
        for step, identity in enumerate(schedule, 1):
            check_bound()
            loss, gradients = nn.value_and_grad(suffix, lambda tail, source=identity, states=captured:
                source_objective(tail, source, states) * weights[source])(suffix)
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
                calibration_loss = measure_calibration(captured)
                weight_path, row = save_checkpoint(step, calibration_loss)
                checkpoints.append(row)
                if (calibration_loss, step) < (best[0], best[1]):
                    best = (calibration_loss, step, weight_path)
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
                score = -native_loss(suffix, hidden, sequence, summed=True,
                                     scope=args.loss_scope).item()
                if not math.isfinite(score):
                    raise ValueError("native bank score is not finite")
                scores.append(score)
                suffix.update(baseline_weights)
                try:
                    pretrained_score = -native_loss(suffix, hidden, sequence, summed=True,
                                                     scope=args.loss_scope).item()
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
                "supervision_receipt_sha256": supervision["receipt_sha256"],
                "prefix_sequence_population": len(sequences),
                "gradient_source_population": len(set(schedule)),
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
