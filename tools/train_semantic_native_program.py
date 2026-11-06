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


def _active_bytes() -> int:
    """What the accelerator holds now, read through the one resource observer."""
    from core.runtime.resource_observation import get_resource_observer

    return int(get_resource_observer().accelerator().active_bytes)


def native_fit_schema_version(*, objective, source_evidence, path_objective,
                              typed_pairs, joint_graph_contrasts):
    """Keep plan and report identities aligned when objectives are composed."""
    if joint_graph_contrasts:
        return 7
    if typed_pairs:
        return 6
    if path_objective:
        return 5
    if objective == "grammar_source_pairs":
        return 4
    if objective == "grammar_choices":
        return 3
    return 2 if source_evidence == "source_token_erasure" else 1


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


def projected_source_shard_bytes(sequences, *, hidden_size):
    """Bound lossless float32 prefix states before loading model weights."""
    if type(hidden_size) is not int or hidden_size < 1 or not sequences:
        raise ValueError("source shard projection needs declared hidden geometry")
    sizes = {}
    for key, sequence in sequences.items():
        if (not isinstance(key, tuple) or not key or not isinstance(key[0], str)
                or len(sequence.tokens) < 2):
            raise ValueError("source shard projection needs source-bound sequences")
        sizes[key[0]] = sizes.get(key[0], 0) + (len(sequence.tokens) - 1) * hidden_size * 4
    return sizes


def native_source_rankings(batch, grammar_groups, graph_groups, *, grammar_mode):
    """Check full/trie parity for every supervised branch of one source."""
    if not batch or any(key[0] != batch[0][0] for key in batch):
        raise ValueError("native ranking batch crosses source identities")
    source = batch[0][0]
    partitions = grammar_groups[source] if grammar_mode else ((batch, 0),)
    if graph_groups:
        partitions = (*partitions, (graph_groups[source], 0))
    ranked_keys = [key for keys, _correct in partitions for key in keys]
    if len(ranked_keys) != len(batch) or set(ranked_keys) != set(batch):
        raise ValueError("native ranking parity missed a supervised branch")
    return partitions


def graft_reused_grammar_states(prior_states, source, decisions, graph_states):
    """Bind old verified grammar rows and new graph rows into one source shard."""
    grammar_keys = [key for keys, _correct in decisions for key in keys]
    if (not graph_states or len(set(grammar_keys)) != len(grammar_keys)
            or any(key[0] != source or key[1] != -1 for key in graph_states)
            or any(key[0] != source or key[1] == -1 for key in grammar_keys)
            or set(grammar_keys) & set(graph_states)):
        raise ValueError("partial frozen capture crosses source or decision kinds")
    return {**{key: prior_states[key] for key in grammar_keys}, **graph_states}


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


def native_schedule_coverage(identities, schedule, captured_identities):
    """Separate eligible fit data, primary updates, and metric-donor capture."""
    from collections import Counter

    eligible = set(identities)
    primary = Counter(schedule)
    captured = set(captured_identities)
    if (not eligible or len(eligible) != len(identities) or not primary
            or not set(primary) <= captured <= eligible):
        raise ValueError("native coverage differs from the admitted fit partition")
    missing = sorted(eligible - set(primary))
    return {"eligible_sources": len(eligible), "primary_updates": len(schedule),
            "primary_sources": len(primary), "captured_sources": len(captured),
            "primary_unvisited_ids": missing,
            "minimum_primary_visits": min(primary[identity] for identity in eligible),
            "complete_primary_epoch": not missing}


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
                            contrast_limit=None, max_tokens=1024, register_encoding="absolute_v1",
                            source_erasure_ids=(), source_control_receipts=None):
    """Reuse the floor's witnessed contrasts only for declared source supervision."""
    from core.learning.semantic_candidate_contrasts import source_program_contrasts
    from core.learning.semantic_native_codec import (
        native_sequence_for_encoding,
        validate_register_encoding,
    )

    validate_register_encoding(register_encoding)
    erased_ids = set(source_erasure_ids)

    if (not identities or len(set(identities)) != len(identities)
            or not set(identities) <= set(items) or not set(identities) <= set(texts)
            or len(erased_ids) != len(source_erasure_ids) or not erased_ids <= set(identities)):
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
            sequences[key] = native_sequence_for_encoding(texts[identity], program, tokenizer,
                max_tokens=max_tokens, register_encoding=register_encoding)
            if identity in erased_ids:
                from core.learning.semantic_native_source_control import erase_native_source_tokens

                sequences[key], receipt = erase_native_source_tokens(sequences[key], texts[identity], tokenizer)
                if source_control_receipts is not None:
                    source_control_receipts[key] = receipt
    return sequences, groups


def native_grammar_supervision_sets(items, texts, tokenizer, identities, *, max_tokens=1024,
                                    register_encoding="absolute_v1", source_erasure_ids=()):
    """Bind each source's exact inference competition to unchanged native tokens."""
    from core.learning.semantic_native_decision_supervision import native_teacher_decisions
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_source_control import erase_native_source_tokens
    from core.learning.semantic_public_inputs import semantic_public_character_inputs

    erased = set(source_erasure_ids)
    if (not identities or len(set(identities)) != len(identities)
            or not set(identities) <= set(items) or not set(identities) <= set(texts)
            or len(erased) != len(source_erasure_ids) or not erased <= set(identities)):
        raise ValueError("native grammar supervision source identities differ")
    sequences, groups, receipts, rows = {}, {}, {}, []
    for identity in sorted(identities):
        item, source = items[identity], texts[identity]
        public = semantic_public_character_inputs(source)
        if (item.split != "train" or item.ir.source_text_sha256 != identity
                or hashlib.sha256(source.encode()).hexdigest() != identity
                or public.values != item.public_inputs):
            raise ValueError("native grammar targets differ from their public source-order inputs")
        kinds = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                      for value in public.values)
        decisions = native_teacher_decisions(item.ir.to_program(), kinds,
                                            register_encoding=register_encoding)
        groups[identity] = []
        for ordinal, decision in enumerate(decisions):
            keys = tuple((identity, ordinal, index) for index in range(len(decision.choices)))
            groups[identity].append((keys, decision.correct_index))
            for key, choice in zip(keys, decision.choices, strict=True):
                sequence = native_text_decision_sequence(source, choice.text, (choice.span,),
                                                        tokenizer, max_tokens=max_tokens)
                if identity in erased:
                    sequence, receipts[key] = erase_native_source_tokens(sequence, source, tokenizer)
                sequences[key] = sequence
                rows.append({"source": identity, "decision_index": ordinal, "choice_index": key[2],
                    "kind": decision.kind, "choice": choice.value,
                    "correct_index": decision.correct_index,
                    "target_program_sha256": item.ir.to_program().sha(),
                    "text": choice.text, "span": choice.span,
                    "tokens": sequence.tokens, "continuation_start": sequence.continuation_start,
                    "semantic_positions": sequence.semantic_positions,
                    **({"source_control_receipt": receipts.get(key)} if erased else {})})
        groups[identity] = tuple(groups[identity])
    return sequences, groups, rows


def native_grammar_source_loss(suffix, states, sequences, decisions):
    """Optimize conditional likelihood for every choice the decoder will score."""
    import mlx.core as mx

    from core.learning.semantic_native_decision_supervision import native_decision_choice_loss

    if not decisions:
        raise ValueError("native grammar source lacks supervised decisions")
    losses = []
    for keys, correct in decisions:
        scores = mx.stack([-native_loss(suffix, states[key], sequences[key], summed=True,
                                       scope="semantic_decisions") for key in keys])
        losses.append(native_decision_choice_loss(scores, correct))
    return mx.mean(mx.stack(losses))


def build_native_supervision(items, texts, tokenizer, identities, plan, peer_programs):
    """Materialize the plan's supervision before allocating the frozen decoder."""
    from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
    from core.learning.semantic_native_source_control import (
        SOURCE_ERASURE_CONTRACT,
        source_control_mode_from_plan,
    )
    from tools.probe_semantic_proposer_crossfit import _digest

    control = source_control_mode_from_plan(plan) == "source_token_erasure"
    erased = plan["captured_fit_ids"] if control else ()
    graph_groups, graph_programs = {}, {}
    if plan["objective"] in {"grammar_choices", "grammar_source_pairs"}:
        sequences, groups, rows = native_grammar_supervision_sets(items, texts, tokenizer, identities,
            max_tokens=plan["max_sequence_tokens"], register_encoding=plan["register_encoding"],
            source_erasure_ids=erased)
        if plan.get("joint_graph_contrast_limit", 0):
            graph_sequences, graph_sets = native_supervision_sets(items, texts, tokenizer, identities,
                peers=peer_programs, contrast_limit=plan["joint_graph_contrast_limit"],
                max_tokens=plan["max_sequence_tokens"], register_encoding=plan["register_encoding"])
            for identity, keys in graph_sets.items():
                graph_groups[identity] = tuple((identity, -1, index) for index in range(len(keys)))
                graph_programs[identity] = tuple(program_sha for _source, program_sha in keys)
                if (len(keys) < 2 or keys[0][1] != items[identity].ir.to_program().sha()
                        or len(set(keys)) != len(keys)):
                    raise ValueError("whole-graph supervision lacks its unique source-positive graph")
                for original, key in zip(keys, graph_groups[identity], strict=True):
                    sequences[key] = graph_sequences[original]
            if set(graph_groups) != set(identities):
                raise ValueError("whole-graph supervision missed a declared source")
    else:
        receipts = {}
        sequences, groups = native_supervision_sets(items, texts, tokenizer, identities,
            peers=peer_programs, contrast_limit=plan["contrast_limit"] if plan["objective"] != "token" else None,
            max_tokens=plan["max_sequence_tokens"], register_encoding=plan["register_encoding"],
            source_erasure_ids=erased, source_control_receipts=receipts)
        rows = [{"source": key[0], "program_sha256": key[1], "tokens": sequence.tokens,
                 "continuation_start": sequence.continuation_start,
                 "semantic_positions": sequence.semantic_positions,
                 **({"source_control_receipt": receipts.get(key)} if control else {})}
                for key, sequence in sorted(sequences.items())]
    supervision = {"plan_sha256": plan["plan_sha256"], "rows": rows}
    if plan["objective"] in {"grammar_choices", "grammar_source_pairs"}:
        supervision["grammar_choice_contract"] = dict(plan.get("grammar_choice_contract", GRAMMAR_CHOICE_CONTRACT))
    if graph_groups:
        supervision["graph_contrast_contract"] = plan["graph_contrast_contract"]
        supervision["graph_rows"] = [
            {"source": identity, "choice_index": index,
             "program_sha256": graph_programs[identity][index],
             "positive": index == 0, "tokens": sequences[key].tokens,
             "continuation_start": sequences[key].continuation_start,
             "semantic_positions": sequences[key].semantic_positions}
            for identity, keys in sorted(graph_groups.items()) for index, key in enumerate(keys)]
    if control:
        supervision["source_evidence_control"] = {
            **SOURCE_ERASURE_CONTRACT, "erased_fit_ids": sorted(plan["captured_fit_ids"]),
            "unchanged_calibration_ids": sorted(plan["calibration_ids"])}
    return sequences, groups, graph_groups, {**supervision, "receipt_sha256": _digest(supervision)}


def require_native_cortex_spec(authority_key_file=None):
    """Verify the signed descriptor with explicit custody under isolated state."""
    import os

    from core.brain.llm.model_registry import get_active_cortex_spec

    if authority_key_file is not None:
        path = str(Path(authority_key_file).expanduser().absolute())
        # The environment itself, not a declared flag: this checks the key
        # against the supervised environment, which is what it sets below.
        existing = os.environ.get("AURA_CORTEX_AUTHORITY_KEY_FILE", "").strip()
        if existing and Path(existing).expanduser().absolute() != Path(path):
            raise ValueError("native authority key differs from the supervised environment")
        os.environ["AURA_CORTEX_AUTHORITY_KEY_FILE"] = path
    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None or not spec.exact_identity:
        raise ValueError("native fit needs the exact signed resident descriptor; pin its existing authority key before isolation")
    return spec


def main():
    from core.learning.semantic_native_codec import REGISTER_ENCODINGS

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "source-report", "folds", "bank", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--authority-key-file", type=Path,
                        help="existing signed-cortex authority key; never created or copied by this tool")
    parser.add_argument("--bundle", action="append", metavar="NAME=PATH")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    from tools.semantic_native_adapters import (
        ADAPTER_IMPLEMENTATION_PATHS,
        ADAPTER_KINDS,
        ADAPTER_SITES,
        SCALING_POLICIES,
        adapter_contract,
        native_adapter_parameter_estimate,
    )

    parser.add_argument("--adapter-sites", choices=ADAPTER_SITES, default="binding_v1")
    parser.add_argument("--adapter-scaling", choices=SCALING_POLICIES, default="direct_v1")
    parser.add_argument("--adapter-alpha", type=float, default=16.)
    parser.add_argument("--adapter-kind", choices=ADAPTER_KINDS, default="lora")
    parser.add_argument("--adapter-layer-ranks", help="comma-separated ranks, earliest to latest suffix layer")
    parser.add_argument("--adapter-layer-kinds", help="comma-separated function classes, earliest to latest suffix layer")
    parser.add_argument("--adapter-experts", type=int, default=1)
    parser.add_argument("--adapter-b-learning-rate-ratio", type=float, default=1.)
    parser.add_argument("--adapter-router-balance-weight", type=float, default=0.)
    parser.add_argument("--prefix-batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("native", "float32"), default="native")
    parser.add_argument("--prefix-strategy", choices=("full", "trie"), default="full")
    parser.add_argument("--prefix-storage", choices=("memory", "source_shards"), default="memory")
    parser.add_argument("--prefix-resident-mib", type=int, default=512)
    parser.add_argument("--reuse-prefix-from", type=Path,
                        help="fresh optimizer run over a complete, bound lossless capture")
    parser.add_argument("--held-per-construction", type=int, default=1)
    parser.add_argument("--calibration-per-construction", type=int, default=1)
    parser.add_argument("--calibration-source-limit", type=int,
                        help="source-hash-only grammar-objective canary bound, never qualification evidence")
    parser.add_argument("--held-source-limit", type=int,
                        help="source-hash-only grammar-objective canary bound")
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--loss-scope", choices=("continuation", "semantic_decisions"),
                        default="continuation")
    parser.add_argument("--objective", choices=("token", "contrastive", "relational", "relational_metric",
                                               "grammar_choices", "grammar_source_pairs"), default="token")
    parser.add_argument("--contrast-limit", type=int, default=4)
    parser.add_argument("--register-encoding", choices=REGISTER_ENCODINGS, default="absolute_v1")
    parser.add_argument("--source-evidence", choices=("source_text", "source_token_erasure"),
                        default="source_text")
    parser.add_argument("--path-objective", action="store_true",
                        help="joint weak-choice risk and baseline-preserving source-path selection")
    parser.add_argument("--joint-graph-contrasts", type=int, default=0,
                        help="add witnessed whole-program competition to path-risk training")
    parser.add_argument("--source-pair-policy", choices=("shared_lineage_v1", "typed_choice_complete_v1"),
                        default="shared_lineage_v1",
                        help="fit-only source contrasts for each available grammar decision kind")
    parser.add_argument("--schedule-policy", choices=("shuffled_epoch_v1", "construction_depth_balanced_v1"),
                        default="shuffled_epoch_v1",
                        help="opt-in bounded pilot covering each fit construction/depth stratum")
    parser.add_argument("--reuse-annotation-sources", type=Path,
                        help="hash-bound archived source for annotation-only frozen implementation changes")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--supervision-only", action="store_true",
                        help="tokenize and audit the declared objective without loading model weights")
    parser.add_argument("--require-identifiable-supervision", action="store_true",
                        help="reject conflicting exact-path targets before loading model weights")
    args = parser.parse_args()
    adaptation = adapter_contract(rank=args.rank, layers=args.layers, sites=args.adapter_sites,
                                  scaling=args.adapter_scaling, alpha=args.adapter_alpha,
                                  kind=args.adapter_kind, experts=args.adapter_experts,
                                  layer_kinds=(args.adapter_layer_kinds.split(",") if args.adapter_layer_kinds else None),
                                  layer_ranks=([int(value) for value in args.adapter_layer_ranks.split(",")]
                                               if args.adapter_layer_ranks is not None else None))
    if not math.isfinite(args.adapter_b_learning_rate_ratio) or args.adapter_b_learning_rate_ratio <= 0:
        parser.error("adapter B learning-rate ratio must be positive and finite")
    if args.adapter_b_learning_rate_ratio != 1. and args.adapter_kind in {"square", "dense"}:
        parser.error("single-matrix adapters do not have a B-factor learning rate")
    if (not math.isfinite(args.adapter_router_balance_weight) or args.adapter_router_balance_weight < 0
            or args.adapter_router_balance_weight and
               "routed" not in adaptation.get("layer_kinds", [args.adapter_kind])):
        parser.error("router balancing needs a routed adapter and a finite nonnegative weight")
    from tools.semantic_native_execution import (
        EXECUTION_PATHS,
        execution_contract,
        execution_from_plan,
    )
    execution = execution_contract(precision=args.precision, prefix_strategy=args.prefix_strategy)
    if args.prefix_strategy == "trie" and args.prefix_batch_size != 1:
        parser.error("source trie capture does not batch independent requests")
    if (args.prefix_storage == "source_shards" and args.prefix_strategy != "trie"
            or not 1 <= args.prefix_resident_mib <= 4096):
        parser.error("source shards require trie capture and a resident bound inside [1, 4096] MiB")
    if args.reuse_prefix_from is not None and args.prefix_storage != "source_shards":
        parser.error("prefix reuse requires lossless source shards")
    if args.reuse_annotation_sources is not None and (args.reuse_prefix_from is None or not args.path_objective):
        parser.error("annotation reuse requires explicit path-objective frozen prefix reuse")
    if (any(type(value) is not int or value < 1 for value in (
            args.steps, args.save_every, args.rank, args.layers, args.max_sequence_tokens))
            or args.steps % args.save_every or not 0 < args.max_seconds <= 14400
            or not 1 <= args.prefix_batch_size <= 32 or not 2 <= args.contrast_limit <= 32
            or args.objective in {"contrastive", "relational", "relational_metric",
                                  "grammar_choices", "grammar_source_pairs"}
            and args.loss_scope != "semantic_decisions"):
        parser.error("positive sizes, complete checkpoint intervals and a finite time bound required")
    if (args.plan_only and args.supervision_only
            or any(value is not None and (value < 1 or args.objective not in
                   {"grammar_choices", "grammar_source_pairs"})
                   for value in (args.calibration_source_limit, args.held_source_limit))):
        parser.error("supervision inventory and source-hash canary bounds need explicit independent modes")
    if args.objective == "grammar_source_pairs" and args.source_evidence != "source_text":
        parser.error("paired-source training needs intact source evidence")
    if args.require_identifiable_supervision and args.objective not in {"grammar_choices", "grammar_source_pairs"}:
        parser.error("exact-path identifiability requires grammar supervision")
    if args.path_objective and (args.objective not in {"grammar_choices", "grammar_source_pairs"}
                               or args.source_evidence != "source_text"):
        parser.error("path-risk training needs intact source grammar supervision")
    if args.joint_graph_contrasts and (not args.path_objective
            or args.objective not in {"grammar_choices", "grammar_source_pairs"}
            or not 2 <= args.joint_graph_contrasts <= 32
            or args.prefix_storage != "source_shards" or args.prefix_strategy != "trie"):
        parser.error("joint graph contrast needs path risk, source shards and trie capture")
    typed_pairs = args.source_pair_policy == "typed_choice_complete_v1"
    if typed_pairs and (not args.path_objective or args.objective != "grammar_source_pairs"):
        parser.error("typed source contrasts require the paired-source path objective")
    bounded_schedule = args.schedule_policy == "construction_depth_balanced_v1"
    if bounded_schedule and not typed_pairs:
        parser.error("bounded native pilots require typed source contrasts")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
    )
    configure_refit_environment(args.directory / "report.json")
    spec = require_native_cortex_spec(args.authority_key_file)
    from core.learning.semantic_counterfactual_corpus import (
        cross_construction_relation_partners,
        cross_construction_relation_triplets,
    )
    from core.learning.semantic_native_codec import NATIVE_CODEC_IMPLEMENTATION_PATHS
    from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
    from core.learning.semantic_native_source_control import SOURCE_ERASURE_CONTRACT
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
    if bounded_schedule:
        from core.learning.semantic_native_fit_sampling import native_depth_calibration_subset

        calibration_ids = native_depth_calibration_subset(examples, outer["calibration_ids"],
            per_stratum=args.calibration_per_construction)
    complete_held_subset, complete_calibration_subset = len(held_ids), len(calibration_ids)
    if args.held_source_limit is not None:
        held_ids = tuple(sorted(held_ids)[:args.held_source_limit])
    if args.calibration_source_limit is not None:
        calibration_ids = tuple(sorted(calibration_ids)[:args.calibration_source_limit])
    sampling_receipt = None
    if bounded_schedule:
        from core.learning.semantic_native_fit_sampling import bounded_native_fit_schedule

        schedule, sampling_receipt = bounded_native_fit_schedule(
            fit, outer["fit_ids"], steps=args.steps, seed=20260925)
    else:
        schedule = native_training_schedule(outer["fit_ids"], steps=args.steps, seed=20260925)
    grammar_pairs = {}
    if args.objective == "grammar_source_pairs":
        from core.learning.semantic_native_source_pairs import native_source_pair_plan

        pair_planner = native_source_pair_plan
        if typed_pairs:
            from core.learning.semantic_native_typed_source_pairs import (
                native_typed_source_pair_plan,
            )

            pair_planner = native_typed_source_pair_plan
        grammar_pairs = pair_planner(fit, outer["fit_ids"],
            register_encoding=args.register_encoding)
        if bounded_schedule:
            from core.learning.semantic_native_fit_sampling import scheduled_native_source_pairs

            grammar_pairs = scheduled_native_source_pairs(grammar_pairs, schedule, outer["fit_ids"])
        if not grammar_pairs or not set(grammar_pairs) <= set(schedule):
            raise ValueError("paired-source objective lacks scheduled witnessed contrasts")
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
    pair_rows = [row for rows in grammar_pairs.values()
                for row in (rows if typed_pairs else [rows])]
    captured_fit_ids = sorted(set(schedule) | set(scheduled_partners.values())
                              | {row["partner"] for row in pair_rows}
                              | {source for pair in scheduled_triplets.values() for source in pair})
    peer_programs = {item.ir.to_program().sha(): item.ir.to_program() for item in fit}
    for value in bundles:
        manifest = json.loads((Path(value.partition("=")[2]) / "manifest.json").read_bytes())
        if Path(manifest["exact_model_path"]).resolve() != spec.model_path.resolve():
            raise ValueError("source features came from a different resident model")
    implementation_paths = [ROOT / name for name in (
        "tools/train_semantic_native_program.py", *ADAPTER_IMPLEMENTATION_PATHS,
        "core/learning/frozen_decoder_prefix.py",
        "core/learning/semantic_native_program.py", "core/brain/llm/decoder_topology.py",
        "core/learning/semantic_native_source_control.py",
        "core/learning/semantic_program_feature_materialization.py",
        "core/learning/semantic_candidate_contrasts.py",
        "core/learning/semantic_counterfactual_corpus.py",
        "core/learning/semantic_graph_counterexamples.py",
        "core/learning/semantic_native_decision_supervision.py",
        "core/learning/semantic_native_grammar.py",
        "core/learning/semantic_program_floor.py",
        "core/runtime/mlx_memory_guard.py", "tools/evaluate_semantic_candidate_ranker.py",
        "tools/train_semantic_atom_ranker.py", *NATIVE_CODEC_IMPLEMENTATION_PATHS)]
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in implementation_paths}
    if execution is not None:
        implementation_paths.extend(ROOT / name for name in EXECUTION_PATHS)
        implementation.update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in EXECUTION_PATHS})
    if args.prefix_storage == "source_shards":
        storage_path = ROOT / "core/learning/frozen_state_store.py"
        implementation_paths.append(storage_path)
        implementation[str(storage_path.relative_to(ROOT))] = hashlib.sha256(storage_path.read_bytes()).hexdigest()
    if args.reuse_prefix_from is not None:
        reuse_path = ROOT / "tools/semantic_native_prefix_reuse.py"
        implementation_paths.append(reuse_path)
        implementation[str(reuse_path.relative_to(ROOT))] = hashlib.sha256(reuse_path.read_bytes()).hexdigest()
    if grammar_pairs:
        pair_path = ROOT / "core/learning/semantic_native_source_pairs.py"
        implementation_paths.append(pair_path)
        implementation[str(pair_path.relative_to(ROOT))] = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    if args.path_objective:
        for name in ("semantic_native_path_objective", "semantic_native_path_calibration", "semantic_native_path_selection"):
            path = ROOT / "core" / "learning" / f"{name}.py"
            implementation_paths.append(path)
            implementation[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.joint_graph_contrasts:
        path = ROOT / "core/learning/semantic_native_search.py"
        implementation_paths.append(path)
        implementation[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if typed_pairs:
        path = ROOT / "core/learning/semantic_native_typed_source_pairs.py"
        implementation_paths.append(path)
        implementation[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if bounded_schedule:
        for name in ("core/learning/semantic_native_fit_sampling.py",
                     "core/learning/recurrent_sft_sampling.py"):
            path = ROOT / name
            implementation_paths.append(path)
            implementation[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    model_config = json.loads((spec.model_path / "config.json").read_text(encoding="utf-8"))
    adapter_memory = native_adapter_parameter_estimate(model_config, adaptation)
    plan = {"schema": "aura.semantic_native_fit_plan.v1", "steps": args.steps,
            "save_every": args.save_every, "rank": args.rank, "suffix_layers": args.layers,
            "prefix_batch_size": args.prefix_batch_size,
            "prefix_padding": False,
            "prefix_equivalence_policy": "complete_and_selected_logits_per_observed_batch_size",
            "adapter_keys": adaptation["keys"], "adapter_contract": adaptation,
            "adapter_b_learning_rate_ratio": args.adapter_b_learning_rate_ratio,
            "adapter_router_balance_weight": args.adapter_router_balance_weight,
            "adapter_memory_projection": adapter_memory,
            "learning_rate": 1e-4, "weight_decay": .01, "seed": 20260925,
            "loss_scope": args.loss_scope,
            "semantic_decision_basis": "program_atoms_and_graph_termination_v1",
            "register_encoding": args.register_encoding,
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
                if args.objective not in {"token", "grammar_choices", "grammar_source_pairs"} else [],
            "max_seconds": args.max_seconds, "max_sequence_tokens": args.max_sequence_tokens,
            "model_descriptor_sha256": spec.descriptor_sha256,
            "model_path": str(spec.model_path), "pointer_sha256": spec.pointer_sha256,
            "bank_plan_sha256": outer["plan_sha256"],
            "bank_receipt_sha256": bank_report["receipt_sha256"],
            "source_report_sha256": outer["source_report_sha256"],
            "fit_ids": outer["fit_ids"], "calibration_ids": calibration_ids,
            "scheduled_fit_ids": schedule,
            "captured_fit_ids": captured_fit_ids,
            "schedule_coverage": native_schedule_coverage(outer["fit_ids"], schedule,
                                                          captured_fit_ids),
            "complete_calibration_population": len(calibration), "held_ids": held_ids,
            "heldout_axis": outer["heldout_axis"],
            "selection": "minimum_source_calibration_" + args.objective + "_" + args.loss_scope,
            "unfitted_checkpoint_eligible": True,
            "matched_control": "same_native_suffix_without_fitted_lora",
            "input": "unchanged_source_request_with_native_chat_template",
            "scoring": "summed_native_" + args.loss_scope + "_log_probability",
            "implementation": implementation, "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False}
    if args.source_evidence == "source_token_erasure":
        if set(captured_fit_ids) & set(calibration_ids):
            raise ValueError("native source erasure cannot consume calibration source evidence")
        plan.update(schema="aura.semantic_native_fit_plan.v2",
                    input="native_chat_template_fit_source_erased_calibration_and_held_source_unchanged",
                    source_evidence_control=dict(SOURCE_ERASURE_CONTRACT))
    if args.prefix_storage == "source_shards":
        plan["prefix_storage_contract"] = {
            "schema": "aura.frozen_state_storage_contract.v1", "mode": "source_shards",
            "max_resident_bytes": args.prefix_resident_mib * 1024 * 1024,
            "lossy_compression": False, "all_alternatives_retained": True}
    if args.objective in {"grammar_choices", "grammar_source_pairs"}:
        plan.update(schema="aura.semantic_native_fit_plan.v3",
                    grammar_choice_contract=dict(GRAMMAR_CHOICE_CONTRACT),
                    contrast_policy="all_native_type_admitted_teacher_decisions_v1",
                    selection="minimum_source_calibration_conditional_grammar_choice_loss",
                    source_hash_canary_bounds={"calibration_limit": args.calibration_source_limit,
                        "held_limit": args.held_source_limit,
                        "eligible_calibration_subset": complete_calibration_subset,
                        "eligible_held_subset": complete_held_subset,
                        "selection_basis": "sorted_source_sha256_without_outcomes"})
    if args.objective == "grammar_source_pairs":
        from core.learning.semantic_native_source_pairs import SOURCE_PAIR_CONTRACT

        plan.update(schema="aura.semantic_native_fit_plan.v4",
                    grammar_source_pair_contract=dict(SOURCE_PAIR_CONTRACT),
                    grammar_source_pair_fit_partners=grammar_pairs,
                    grammar_source_pair_updates=sum(identity in grammar_pairs for identity in schedule))
    if args.path_objective:
        from core.learning.semantic_native_path_objective import (
            GRAMMAR_PATH_CONTRACT,
            path_choice_contract,
        )
        from core.learning.semantic_native_path_selection import PATH_SELECTION_CONTRACT

        plan.update(schema="aura.semantic_native_fit_plan.v5",
                    grammar_choice_contract=path_choice_contract(),
                    grammar_path_objective_contract=dict(GRAMMAR_PATH_CONTRACT),
                    path_checkpoint_selection_contract=dict(PATH_SELECTION_CONTRACT),
                    selection="baseline_preserving_complete_source_calibration_paths")
    if args.joint_graph_contrasts:
        from core.learning.semantic_native_path_objective import JOINT_GRAPH_CONTRAST_CONTRACT
        from core.learning.semantic_native_path_selection import JOINT_GRAPH_SELECTION_CONTRACT

        plan.update(schema="aura.semantic_native_fit_plan.v7",
                    joint_graph_contrast_limit=args.joint_graph_contrasts,
                    graph_contrast_contract=dict(JOINT_GRAPH_CONTRAST_CONTRACT),
                    path_checkpoint_selection_contract=dict(JOINT_GRAPH_SELECTION_CONTRACT),
                    selection="baseline_preserving_joint_source_calibration")
    if typed_pairs:
        from core.learning.semantic_native_typed_source_pairs import (
            TYPED_SOURCE_PAIR_CONTRACT,
            typed_source_pair_inventory,
        )

        plan.update(grammar_source_pair_contract=dict(TYPED_SOURCE_PAIR_CONTRACT),
                    grammar_source_pair_inventory=typed_source_pair_inventory(grammar_pairs, schedule))
    if args.require_identifiable_supervision:
        from tools.semantic_native_identifiability import IDENTIFIABILITY_CONTRACT

        plan["grammar_identifiability_contract"] = dict(IDENTIFIABILITY_CONTRACT)
        for name in ("tools/semantic_native_identifiability.py", "tools/verify_semantic_native_fit.py"):
            plan["implementation"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    schema_version = native_fit_schema_version(
        objective=args.objective, source_evidence=args.source_evidence,
        path_objective=args.path_objective, typed_pairs=typed_pairs,
        joint_graph_contrasts=args.joint_graph_contrasts)
    plan["schema"] = f"aura.semantic_native_fit_plan.v{schema_version}"
    if sampling_receipt is not None:
        plan["fit_sampling_contract"] = sampling_receipt
        plan["calibration_sampling_contract"] = {
            "policy": "construction_depth_lowest_source_sha256_v1",
            "per_stratum": args.calibration_per_construction,
            "source_limit": args.calibration_source_limit}
    plan = {**plan, "plan_sha256": _digest(plan)}
    if args.reuse_annotation_sources is not None:
        from tools.evaluate_semantic_native_checkpoint import verified_document

        archive = verified_document(args.reuse_annotation_sources)
        plan.pop("plan_sha256")
        plan["annotation_only_prefix_sources"] = {
            "path": str(args.reuse_annotation_sources.resolve()),
            "receipt_sha256": archive["receipt_sha256"]}
        plan["plan_sha256"] = _digest(plan)
    if execution is not None:
        plan.pop("plan_sha256")
        plan["execution_contract"] = execution
        plan["prefix_equivalence_policy"] = "all_first_source_choices_then_first_choice_per_source"
        plan["plan_sha256"] = _digest(plan)
    if args.reuse_prefix_from is not None:
        from tools.semantic_native_prefix_reuse import prefix_reuse_contract

        plan["reused_prefix_contract"] = prefix_reuse_contract(args.reuse_prefix_from, plan)
        plan.pop("plan_sha256")
        plan["plan_sha256"] = _digest(plan)
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "plan_sha256": plan["plan_sha256"],
                          "adapter_memory_projection": adapter_memory,
                          "fit": len(fit), "calibration": len(calibration_ids),
                          "schedule_coverage": plan["schedule_coverage"],
                          "held": len(held_ids), "model_weights_loaded": False}), flush=True)
        return
    if any(args.directory.glob("checkpoint-*.json")):
        raise FileExistsError("native fit already has evidence; use a fresh experiment directory")

    started = time.monotonic()
    from mlx_lm.utils import load_tokenizer

    from core.learning.semantic_native_program import source_text_from_tokens
    tokenizer = load_tokenizer(Path(spec.model_path))
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    texts = {identity: source_text_from_tokens(item, tokenizer) for identity, item in items.items()}
    public_by_id = {identity: item.public_inputs for identity, item in items.items()}
    supervised_ids = tuple(sorted(set(captured_fit_ids) | set(calibration_ids)))
    sequences, groups, graph_groups, supervision = build_native_supervision(items, texts, tokenizer, supervised_ids,
        plan, tuple(peer_programs[key] for key in sorted(peer_programs)))
    projected_shards = None
    if args.prefix_storage == "source_shards":
        geometry = model_config.get("text_config", model_config)
        if not isinstance(geometry, dict):
            raise ValueError("native source shard hidden geometry is undeclared")
        projected_shards = projected_source_shard_bytes(
            sequences, hidden_size=geometry.get("hidden_size"))
        if max(projected_shards.values()) > plan["prefix_storage_contract"]["max_resident_bytes"]:
            raise ValueError("one projected frozen source exceeds the resident shard bound")
    if args.reuse_prefix_from is not None:
        from tools.semantic_native_prefix_reuse import open_reused_prefix

        reused_states = open_reused_prefix(plan["reused_prefix_contract"], plan, supervision)
        del reused_states
    _save_if_absent(args.directory / "supervision.json", supervision)
    if args.require_identifiable_supervision:
        from tools.semantic_native_identifiability import save_identifiability_preflight

        proof = save_identifiability_preflight(args.directory, plan, supervision)
        print(json.dumps({"stage": "identifiability_preflight", "receipt_sha256": proof["receipt_sha256"],
                          "exact_targets_identifiable": proof["exact_targets_identifiable"],
                          "model_weights_loaded": False}), flush=True)
    preparation_seconds = time.monotonic() - started
    if args.supervision_only:
        from collections import Counter

        print(json.dumps({"stage": "supervision_only", "plan_sha256": plan["plan_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"],
            "prefix_sequences": len(sequences),
            "whole_graph_sequences": len(supervision.get("graph_rows", ())),
            "largest_projected_source_shard_bytes": (
                max(projected_shards.values()) if projected_shards is not None else None),
            "sequence_length_range": [min(len(row.tokens) for row in sequences.values()),
                                      max(len(row.tokens) for row in sequences.values())],
            "decision_counts": dict(Counter(kind for _source, _ordinal, kind in {
                    (row["source"], row["decision_index"], row["kind"]) for row in supervision["rows"]}))
                if args.objective in {"grammar_choices", "grammar_source_pairs"} else {},
            "preparation_seconds": preparation_seconds,
            "model_weights_loaded": False}), flush=True)
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_codec import native_sequence_for_encoding
    from core.learning.semantic_native_source_pairs import native_source_interaction_loss
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.evaluate_semantic_candidate_ranker import _rankable_or_none, _read_bank
    from tools.semantic_native_adapters import install_native_adapters

    def check_bound():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("native fit bound reached; durable checkpoints remain research-only")

    with (standalone_model_lane(owner_id=f"semantic-native:{args.directory.name}",
            model_path=str(spec.model_path), purpose="training", preemptible=False,
            require_exclusive=True, allow_owner_eviction=False,
            metadata={"tool": "train_semantic_native_program", "production_effect": False}),
          mlx_memory_envelope(fraction=.80) as envelope):
        print(json.dumps({"stage": "load", "descriptor": spec.descriptor_sha256,
                          "memory_envelope": envelope.to_receipt()}), flush=True)
        model, loaded_tokenizer = load(str(spec.model_path))
        if adapter_memory is not None and _active_bytes() + adapter_memory["five_float32_copies_bytes"] > envelope.memory_bytes:
            raise MemoryError("backbone residency plus adapter state alone exceeds the host envelope")
        if any(loaded_tokenizer.encode(texts[identity], add_special_tokens=False)
               != tokenizer.encode(texts[identity], add_special_tokens=False) for identity in supervised_ids):
            raise ValueError("native model load changed the preflight tokenizer")
        model.freeze()
        model.eval()
        split = len(model.layers) - args.layers
        prefix = FrozenDecoderPrefix(model, split_at=split)
        suffix = NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(plan["seed"])
        install_native_adapters(model, plan)
        from tools.semantic_native_execution import apply_execution, source_sequence_groups
        precision_receipt = apply_execution(model, plan)
        trainable = tree_flatten(suffix.trainable_parameters())
        observed_parameter_count = sum(value.size for _name, value in trainable)
        if adapter_memory is not None and observed_parameter_count != adapter_memory["trainable_parameters"]:
            raise ValueError("loaded native adapter geometry differs from its pre-load projection")
        print(json.dumps({"stage": "adapter_residency", "trainable_parameters": observed_parameter_count,
                          "projected_state": adapter_memory, "active_memory_bytes": _active_bytes(),
                          "activation_peak_measured": False}), flush=True)
        if not trainable or any("lora_" not in name for name, _value in trainable):
            raise ValueError("native suffix adaptation escaped its declared LoRA sites")
        if any("lora_" not in name for name, _value in tree_flatten(model.trainable_parameters())):
            raise ValueError("native fit would update frozen resident parameters")
        baseline_weights = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(baseline_weights)
        for layer in suffix.layers:
            layer.train()
        weights = construction_weights(fit)
        calibration_weights = construction_weights([items[identity] for identity in calibration_ids])
        construction_by_id = {identity: item.construction_id for identity, item in items.items()}
        del examples, fit, calibration, items, parent
        gc.collect()
        partial_reuse = (args.reuse_prefix_from is not None
                         and plan["reused_prefix_contract"]["schema"]
                         == "aura.native_frozen_prefix_reuse.v2")
        prior_states = None
        captured = {}
        if args.prefix_storage == "source_shards":
            from core.learning.frozen_state_store import FrozenStateStore
            if partial_reuse:
                prior_states = open_reused_prefix(plan["reused_prefix_contract"], plan, supervision)
            captured = (open_reused_prefix(plan["reused_prefix_contract"], plan, supervision)
                if args.reuse_prefix_from is not None and not partial_reuse else
                FrozenStateStore(args.directory / "prefix-states",
                    plan_sha256=plan["plan_sha256"],
                    max_resident_bytes=plan["prefix_storage_contract"]["max_resident_bytes"]))
        verified_batch_sizes = set()
        capture_receipts = []
        if args.reuse_prefix_from is not None and not partial_reuse:
            from tools.semantic_native_prefix_reuse import source_capture_receipts

            prior = plan["reused_prefix_contract"]
            capture_receipts = source_capture_receipts(
                Path(prior["source_directory"]), sources=sorted(set(supervised_ids)),
                plan_sha256=prior["source_plan_sha256"])
            if _digest(capture_receipts) != prior["capture_inventory_sha256"]:
                raise ValueError("reused prefix capture inventory changed")
        capture_sequences = ({key: row for key, row in sequences.items() if key[1] == -1}
                             if partial_reuse else sequences)
        batches = (() if args.reuse_prefix_from is not None and not partial_reuse else
                   source_sequence_groups(capture_sequences) if args.prefix_strategy == "trie" else
                   exact_length_batches(sequences, batch_size=args.prefix_batch_size))
        for batch in batches:
            check_bound()
            if args.prefix_strategy == "trie":
                from core.learning.frozen_prefix_branches import (
                    FrozenPrefixBranches,
                    native_source_anchor,
                )
                from tools.probe_semantic_native_prefix_branches import target_logprobs

                rows = tuple(sequences[key] for key in batch)
                branches = FrozenPrefixBranches(model, split_at=split,
                    anchor_tokens=native_source_anchor(rows), max_tokens=args.max_sequence_tokens)
                states = branches.capture_many(tuple(tuple(row.tokens[:-1]) for row in rows))
                errors = []
                reference_scores, branch_scores = [], []
                for key, row, state in zip(batch, rows, states, strict=True):
                    check_bound()
                    if args.prefix_storage == "memory":
                        captured[key] = state
                    if capture_receipts and key != batch[0] and not partial_reuse:
                        continue
                    positions = tuple(index - 1 for index in native_prediction_positions(row, scope=args.loss_scope))
                    expected = prefix.capture(mx.array([row.tokens[:-1]], dtype=mx.int32))
                    if not capture_receipts and key == batch[0]:
                        full_logits = model(mx.array([row.tokens[:-1]], dtype=mx.int32))
                        suffix_logits = suffix(expected)
                        difference = float(mx.max(mx.abs(full_logits - suffix_logits)).item())
                        if not math.isfinite(difference) or difference > .01:
                            raise ValueError("native trie suffix differs from the complete model")
                        _save_if_absent(args.directory / "prefix-equivalence.json", {
                            "plan_sha256": plan["plan_sha256"], "accepted": True,
                            "max_absolute_logit_difference": difference,
                            "execution_contract": execution, "batch_size": 1,
                            "split_at": split, "trainable_sites": [name for name, _value in trainable]})
                        del full_logits, suffix_logits
                    targets = mx.array([row.tokens[index + 1] for index in positions], dtype=mx.int32)
                    left = target_logprobs(suffix(expected, logit_positions=positions), targets)
                    right = target_logprobs(suffix(state, logit_positions=positions), targets)
                    error = float(mx.max(mx.abs(left - right)).item())
                    if not math.isfinite(error) or error > execution["precision_contract"]["target_logprob_tolerance"]:
                        raise ValueError("native training trie target probabilities differ")
                    errors.append(error)
                    reference_scores.append(float(mx.sum(left).item()))
                    branch_scores.append(float(mx.sum(right).item()))
                if not capture_receipts or partial_reuse:
                    from tools.probe_semantic_native_prefix_branches import ranked_score_equivalence
                    partitions = (((batch, 0),) if partial_reuse else native_source_rankings(
                        batch, groups, graph_groups,
                        grammar_mode=args.objective in {"grammar_choices", "grammar_source_pairs"}))
                    offsets = {key: index for index, key in enumerate(batch)}
                    for keys, _gold in partitions:
                        indices = [offsets[key] for key in keys]
                        comparison = ranked_score_equivalence(
                            [reference_scores[index] for index in indices],
                            [branch_scores[index] for index in indices],
                            tolerance=execution["precision_contract"]["target_logprob_tolerance"])
                        if not comparison["accepted"]:
                            raise ValueError("native training trie changed a choice ranking")
                capture_receipts.append({"source": batch[0][0], **branches.receipt(),
                    "checked_choices": len(errors), "max_target_error": max(errors),
                    "complete_rankings_checked": partial_reuse or len(capture_receipts) == 0})
                if args.prefix_storage == "source_shards":
                    source_states = dict(zip(batch, states, strict=True))
                    if partial_reuse:
                        source_states = graft_reused_grammar_states(
                            prior_states, batch[0][0], groups[batch[0][0]], source_states)
                    captured.write_source(batch[0][0], source_states,
                        sequence_digests={key: _digest(list(sequences[key].tokens))
                                          for key in source_states})
                _save_if_absent(args.directory / "prefix-receipts" / f"{batch[0][0]}.json",
                    {"plan_sha256": plan["plan_sha256"], **capture_receipts[-1]})
                del branches, states, expected, left, right
                print(json.dumps({"stage": "prefix", "captured": len(captured),
                    "population": len(sequences), "source_groups": len(capture_receipts),
                    "elapsed_seconds": time.monotonic() - started,
                    "active_memory_bytes": _active_bytes()}), flush=True)
                continue
            tokens = mx.array([sequences[identity].tokens[:-1] for identity in batch], dtype=mx.int32)
            hidden = prefix.capture(tokens)
            if len(batch) not in verified_batch_sizes:
                full = model(tokens)
                difference = mx.max(mx.abs(full - suffix(hidden))).item()
                positions = tuple(index - 1 for index in native_prediction_positions(
                    sequences[batch[0]], scope=args.loss_scope))
                selected_difference, selected_tolerance = selected_projection_error(
                    full[:1], suffix(hidden[:1], logit_positions=positions),
                    sequences[batch[0]], positions)
                proof = {
                    "plan_sha256": plan["plan_sha256"], "max_absolute_logit_difference": difference,
                    "tokens": tokens.shape[1], "batch_size": tokens.shape[0], "split_at": split,
                    "selected_projection_max_target_logprob_difference": selected_difference,
                    "selected_projection_target_logprob_tolerance": selected_tolerance,
                    "projected_positions": len(positions),
                    "trainable_sites": [name for name, _value in trainable],
                    "accepted": (math.isfinite(difference) and difference <= .01
                        and math.isfinite(selected_difference)
                        and selected_difference <= selected_tolerance)}
                _save_if_absent(args.directory / f"prefix-equivalence-batch-{len(batch)}.json", proof)
                if not proof["accepted"]:
                    raise ValueError("cached native prefix does not reproduce model logits: "
                        f"full={difference}, selected={selected_difference}, "
                        f"selected_tolerance={selected_tolerance}, batch={len(batch)}")
                if not captured:
                    _save_if_absent(args.directory / "prefix-equivalence.json", proof)
                verified_batch_sizes.add(len(batch))
            for index, identity in enumerate(batch):
                captured[identity] = hidden[index:index + 1]
            if len(captured) % 16 < len(batch) or len(captured) == len(sequences):
                print(json.dumps({"stage": "prefix", "captured": len(captured),
                                  "population": len(sequences),
                                  "elapsed_seconds": time.monotonic() - started,
                                  "active_memory_bytes": _active_bytes()}), flush=True)
        del prior_states

        def source_objective(tail, identity, states):
            if args.objective in {"grammar_choices", "grammar_source_pairs"}:
                if args.path_objective:
                    from core.learning.semantic_native_path_objective import (
                        native_grammar_path_objective,
                    )

                    pair = grammar_pairs.get(identity)
                    if typed_pairs:
                        return native_grammar_path_objective(
                            lambda key: -native_loss(tail, states[key], sequences[key], summed=True,
                                                    scope="semantic_decisions"), groups[identity],
                            typed_pairs=pair, partners=groups if pair else None,
                            graph_keys=graph_groups.get(identity))
                    return native_grammar_path_objective(
                        lambda key: -native_loss(tail, states[key], sequences[key], summed=True,
                                                scope="semantic_decisions"), groups[identity],
                        pair=pair, partner_decisions=groups[pair["partner"]] if pair else None,
                        graph_keys=graph_groups.get(identity))
                source_loss = native_grammar_source_loss(tail, states, sequences, groups[identity])
                pair = grammar_pairs.get(identity)
                if pair is None:
                    return source_loss
                partner = pair["partner"]
                ordinal = pair["decision_index"]
                own_keys, _own_correct = groups[identity][ordinal]
                partner_keys, _partner_correct = groups[partner][ordinal]
                i, j = pair["own_index"], pair["partner_index"]
                def score(key):
                    return -native_loss(tail, states[key], sequences[key], summed=True,
                                        scope="semantic_decisions")
                interaction = native_source_interaction_loss(
                    score(own_keys[i]), score(own_keys[j]),
                    score(partner_keys[i]), score(partner_keys[j]))
                return source_loss + plan["grammar_source_pair_contract"]["weight"] * interaction
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

        calibration_paths = []
        supervision_by_key = {(row["source"], row["decision_index"], row["choice_index"]): row
                              for row in supervision["rows"]} if args.path_objective else {}

        def measure_calibration(states):
            if args.path_objective:
                from core.learning.semantic_native_path_objective import (
                    native_grammar_path_objective,
                )

                calibration_paths.clear()
                total = 0.
                for identity in sorted(calibration_ids):
                    check_bound()
                    measured, measured_graph = [], []
                    loss = native_grammar_path_objective(
                        lambda key: -native_loss(suffix, states[key], sequences[key], summed=True,
                                                scope="semantic_decisions"), groups[identity],
                        measured_scores=measured, graph_keys=graph_groups.get(identity),
                        measured_graph_scores=measured_graph if graph_groups else None)
                    total += loss.item() * calibration_weights[identity]
                    decisions = [{"kind": supervision_by_key[keys[0]]["kind"],
                        "correct_index": correct, "choices": [supervision_by_key[key]["choice"] for key in keys],
                        "scores": scores.tolist()}
                        for (keys, correct), scores in zip(groups[identity], measured, strict=True)]
                    calibration_paths.append({"source": identity, "decisions": decisions,
                        **({"whole_graph": {"program_sha256s": [row["program_sha256"] for row in
                           supervision["graph_rows"] if row["source"] == identity],
                           "scores": measured_graph[0].tolist(), "positive_index": 0}}
                           if graph_groups else {})})
                return total / len(calibration_ids)
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
            if args.path_objective:
                from core.learning.semantic_native_path_calibration import native_path_totals

                body = {"schema": "aura.native_checkpoint_path_calibration.v1",
                        "plan_sha256": plan["plan_sha256"], "step": step,
                        "rows": list(calibration_paths),
                        "totals": native_path_totals(calibration_paths, sorted(calibration_ids))}
                receipt = {**body, "receipt_sha256": _digest(body)}
                _save_if_absent(args.directory / f"calibration-paths-{step}.json", receipt)
                row["calibration_path_receipt_sha256"] = receipt["receipt_sha256"]
            row = {**row, "receipt_sha256": _digest(row)}
            _save_if_absent(args.directory / f"checkpoint-{step}.json", row)
            print(json.dumps({"stage": "checkpoint", **row}), flush=True)
            return weight_path, row

        baseline = measure_calibration(captured)
        from tools.semantic_native_adapters import native_optimizer
        optimizer = native_optimizer(plan)
        zero_path, zero = save_checkpoint(0, baseline)
        best, checkpoints, history = (baseline, 0, zero_path), [zero], []
        for step, identity in enumerate(schedule, 1):
            check_bound()
            def training_objective(tail, source=identity, states=captured):
                from tools.semantic_native_adapters import native_router_regularized_objective

                return native_router_regularized_objective(tail,
                    lambda: source_objective(tail, source, states) * weights[source],
                    weight=plan["adapter_router_balance_weight"])

            loss, gradients = nn.value_and_grad(suffix, training_objective)(suffix)
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
        if args.path_objective:
            from tools.evaluate_semantic_native_checkpoint import selected_checkpoint

            _plan, selected = selected_checkpoint(args.directory)
            best = (selected["calibration_loss"], selected["step"],
                    args.directory / f"checkpoint-{selected['step']}.safetensors")
        model.load_weights(str(best[2]), strict=False)
        selected_weights = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(selected_weights)
        storage_receipt = captured.receipt() if args.prefix_storage == "source_shards" else None
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
                sequence = native_sequence_for_encoding(texts[identity], program, tokenizer,
                    max_tokens=args.max_sequence_tokens, register_encoding=args.register_encoding)
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
        current_spec = require_native_cortex_spec(args.authority_key_file)
        execution_from_plan(plan, check_installed=True)
        if (current_spec is None or current_spec.descriptor_sha256 != spec.descriptor_sha256
                or current_spec.pointer_sha256 != spec.pointer_sha256
                or any(hashlib.sha256(path.read_bytes()).hexdigest() != implementation[
                       str(path.relative_to(ROOT))] for path in implementation_paths)
                or any(path.read_bytes() != raw[name] for name, path in (
                    ("parent", args.parent), ("source", args.source_report), ("folds", args.folds)))):
            raise ValueError("native fit identity changed during measurement")
        body = {"schema": f"aura.semantic_native_fit.v{schema_version}",
                "plan_sha256": plan["plan_sha256"],
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
                "preparation_seconds": preparation_seconds,
                "serving_authority": False, "qualification_evidence": False,
                "held_labels_used_for_fit_or_selection": False}
        if execution is not None:
            body.update(execution_contract=execution, precision_receipt=precision_receipt,
                        prefix_capture_receipts=capture_receipts)
        if storage_receipt is not None:
            body.update(prefix_storage_receipt=storage_receipt)
        if args.reuse_prefix_from is not None:
            body.update(reused_prefix_contract=plan["reused_prefix_contract"])
        _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": _digest(body)})
        print(json.dumps({"stage": "complete", "native_correct": body["native_correct"],
                          "population": len(rows), "regressions": body["regressions"]}), flush=True)


if __name__ == "__main__":
    main()
