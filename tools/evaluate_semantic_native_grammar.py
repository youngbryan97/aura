#!/usr/bin/env python3
"""Generate typed graphs from native semantic choices, without a target bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def select_search_proposal(search, scorer):
    """Rank every admitted graph without receiving its target or correctness."""
    scores = tuple(scorer(candidate.result.program) for candidate in search.candidates)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in scores):
        raise ValueError("native proposal selection needs every finite graph score")
    return (max(range(len(scores)), key=scores.__getitem__) if scores else None), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--canary", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=3600.)
    parser.add_argument("--search-completions", type=int, default=0)
    parser.add_argument("--search-nodes", type=int, default=256)
    parser.add_argument("--search-score-mode", choices=("normalized_choices", "native_nonpositive"),
                        default="native_nonpositive")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 128 or not 1 <= args.canary <= 72 or not 0 < args.max_seconds <= 14400:
        parser.error("finite depth, population, and runtime bounds required")
    if not 0 <= args.search_completions <= 128 or not 1 <= args.search_nodes <= 100000:
        parser.error("finite search node and completion bounds required")
    from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    configure_refit_environment(args.directory / "report.json")
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )

    training, selected = selected_checkpoint(args.training_directory)
    from core.learning.semantic_native_codec import register_encoding_from_plan
    register_encoding = register_encoding_from_plan(training)
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("native grammar model identity differs from training")
    examples = build_semantic_program_natural_request_corpus(examples_per_schema_domain=3)
    # Interleave schema cells so the canary does not inspect only scalar chains.
    examples = tuple(examples[index] for sample in range(24) for index in
                     (sample, sample + 24, sample + 48))[:args.canary]
    sources = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in examples]
    forbidden = set(training["fit_ids"]) | set(training["calibration_ids"]) | set(training["held_ids"])
    if forbidden & set(sources) or len(set(sources)) != len(sources):
        raise ValueError("native grammar evaluation sources overlap training")
    paths = ("tools/evaluate_semantic_native_grammar.py",
             "core/learning/semantic_native_grammar.py",
             "core/learning/semantic_native_search.py",
             "core/learning/semantic_native_program.py",
             "core/learning/semantic_native_codec.py",
             "core/learning/semantic_native_relative_program.py",
             "core/learning/semantic_register_identity.py",
             "core/learning/frozen_decoder_prefix.py",
             "core/learning/semantic_program_floor.py",
             "core/learning/semantic_program_corpus_natural.py",
             "tools/train_semantic_native_program.py")
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    body = {"schema": "aura.semantic_native_grammar_plan.v1",
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "model_descriptor_sha256": spec.descriptor_sha256,
            "pointer_sha256": spec.pointer_sha256, "implementation": implementation,
            "sources": sources, "max_steps": args.max_steps, "max_seconds": args.max_seconds,
            "register_encoding": register_encoding,
            "search_mode": "best_first_then_complete_graph_score" if args.search_completions else "greedy",
            "search_nodes": args.search_nodes, "search_completions": args.search_completions,
            "search_score_mode": args.search_score_mode,
            "candidate_inventory": "none", "input_grounding": "declared_public_inputs",
            "target_available_to_scorer": False, "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "population": len(examples),
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_codec import native_sequence_for_encoding
    from core.learning.semantic_native_grammar import (
        NativeGrammarIncompleteError,
        decode_native_grammar,
    )
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_search import search_native_grammar
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.train_semantic_native_program import native_loss

    started, rows = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"semantic-native-grammar:{args.directory.name}",
                                model_path=str(spec.model_path), purpose="evaluation",
                                preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - training["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0., "keys": training["adapter_keys"]})
        model.load_weights(str(args.training_directory /
                               f"checkpoint-{selected['step']}.safetensors"), strict=False)
        for example, identity in zip(examples, sources, strict=True):
            scored = 0
            def score(choices, *, source=example.source_text, source_identity=identity):
                nonlocal scored
                scores = []
                for choice in choices:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("native grammar run reached its finite bound")
                    sequence = native_text_decision_sequence(
                        source, choice.text, (choice.span,), tokenizer,
                        max_tokens=training["max_sequence_tokens"])
                    hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                    scores.append(-native_loss(suffix, hidden, sequence, summed=True,
                                               scope="semantic_decisions").item())
                    scored += 1
                print(json.dumps({"stage": "decision", "source_sha256": source_identity,
                                  "scored_prefixes": scored, "choices": len(choices)}), flush=True)
                return tuple(scores)

            types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                          for value in example.inputs)
            search_evidence = None
            try:
                if args.search_completions:
                    searched = search_native_grammar(types, score, max_steps=args.max_steps,
                        max_nodes=args.search_nodes, completions=args.search_completions,
                        register_encoding=register_encoding, score_mode=args.search_score_mode)
                    def whole_graph_score(program, *, source=example.source_text):
                        if time.monotonic() - started > args.max_seconds:
                            raise TimeoutError("native grammar run reached its finite bound")
                        sequence = native_sequence_for_encoding(source, program, tokenizer,
                            max_tokens=training["max_sequence_tokens"], register_encoding=register_encoding,
                            decision_basis=training.get("semantic_decision_basis", "program_atoms_v1"))
                        hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                        return -native_loss(suffix, hidden, sequence, summed=True,
                                           scope="semantic_decisions").item()
                    chosen, graph_scores = select_search_proposal(searched, whole_graph_score)
                    search_evidence = {"expanded_nodes": searched.expanded_nodes,
                        "scored_decisions": searched.scored_decisions,
                        "scored_alternatives": searched.scored_alternatives,
                        "disconnected_leaves": searched.disconnected_leaves,
                        "frontier_nodes": searched.frontier_nodes,
                        "frontier_log_probability_bound": searched.frontier_log_probability_bound,
                        "halt_reason": searched.halt_reason,
                        "requested_top_k_proven": searched.requested_top_k_proven,
                        "selected_index": chosen, "complete_graph_scores": graph_scores,
                        "proposals": [{"program": candidate.result.program.to_dict(),
                            "log_probability": candidate.log_probability,
                            "decision_trace": candidate.result.trace,
                            "bound_forced_completion": candidate.result.bound_forced_completion}
                            for candidate in searched.candidates]}
                    generated = None if chosen is None else searched.candidates[chosen].result
                    # Grade reach only after target-blind generation and selection.
                    search_evidence["observed_program_reach"] = any(
                        semantic_programs_structurally_equivalent(candidate.result.program, example.program)
                        for candidate in searched.candidates)
                else:
                    generated = decode_native_grammar(types, score, max_steps=args.max_steps,
                                                       register_encoding=register_encoding)
                program = None if generated is None else generated.program
                trace = () if generated is None else generated.trace
                forced = False if generated is None else generated.bound_forced_completion
                status = "search_without_completion" if generated is None else "completed"
            except NativeGrammarIncompleteError as failure:
                program, trace = failure.program, failure.trace
                forced = False
                status = "disconnected_at_depth_bound"
            equivalent = status == "completed" and semantic_programs_structurally_equivalent(program, example.program)
            answer_correct = False
            if status == "completed":
                try:
                    answer_correct = program.run(example.inputs) == example.program.run(example.inputs)
                except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
                    answer_correct = False
            else:
                answer_correct = False
            row_body = {"plan_sha256": plan["plan_sha256"], "source_sha256": identity,
                        "construction": example.construction_id,
                        "program": None if program is None else program.to_dict(), "decode_status": status,
                        "program_equivalent": equivalent, "answer_correct": answer_correct,
                        "bound_forced_completion": forced,
                        "depth_bound_reached": forced or status == "disconnected_at_depth_bound",
                        "decision_trace": trace, "search": search_evidence,
                        "target_available_to_scorer": False}
            row = {**row_body, "receipt_sha256": digest(row_body)}
            _save_if_absent(args.directory / "rows" / f"{identity}.json", row)
            rows.append(row)
            print(json.dumps({"stage": "grammar", "observed": len(rows), "population": len(examples),
                              "program_equivalent": equivalent,
                              "depth": 0 if program is None else program.depth, "decode_status": status}), flush=True)
        current = get_active_cortex_spec(force_refresh=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                       for name, sha in implementation.items())
                or selected_checkpoint(args.training_directory) != (training, selected)):
            raise ValueError("native grammar implementation, model, or checkpoint drifted")
        result = {"schema": "aura.semantic_native_grammar.v1", "plan_sha256": plan["plan_sha256"],
                  "population": len(rows), "program_equivalent": sum(row["program_equivalent"] for row in rows),
                  "answer_correct": sum(row["answer_correct"] for row in rows),
                  "bound_forced_completion": sum(row["bound_forced_completion"] for row in rows),
                  "depth_bound_reached": sum(row["depth_bound_reached"] for row in rows),
                  "row_receipts": {row["source_sha256"]: row["receipt_sha256"] for row in rows},
                  "candidate_inventory": "none", "input_grounding": "declared_public_inputs",
                  "target_available_to_scorer": False, "serving_authority": False,
                  "qualification_evidence": False, "elapsed_seconds": time.monotonic() - started}
        _save_if_absent(args.directory / "report.json", {**result, "receipt_sha256": digest(result)})
        print(json.dumps({key: value for key, value in result.items() if key != "row_receipts"}), flush=True)


if __name__ == "__main__":
    main()
