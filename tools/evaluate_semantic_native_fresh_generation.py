#!/usr/bin/env python3
"""Probe target-blind native program generation on frozen fresh requests."""

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


def grade_generation(source: str, raw: str, *, prompt_thinking: bool,
                     target_sha256: str, inputs: tuple, target_value: object,
                     register_encoding: str = "absolute_v1") -> dict:
    """Interpret generated public text only after the native channel closes."""
    from core.brain.llm.chat_format import split_native_thinking_generation
    from core.learning.semantic_native_codec import (
        parse_native_for_encoding,
        validate_register_encoding,
    )
    validate_register_encoding(register_encoding)

    channels = split_native_thinking_generation(raw, native_thinking=prompt_thinking)
    result = {"raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
              "surface_sha256": hashlib.sha256(channels.surface.encode()).hexdigest(),
              "boundary_closed": channels.boundary_closed,
              "parse_status": "", "program_sha256": None,
              "program_exact": False, "answer_correct": False}
    if not channels.boundary_closed:
        result["parse_status"] = "thinking_incomplete"
        return result
    try:
        program = parse_native_for_encoding(channels.surface.strip(), register_encoding=register_encoding)
    except (ValueError, TypeError, json.JSONDecodeError):
        result["parse_status"] = "program_invalid"
        return result
    result["parse_status"] = "valid_program"
    result["program_sha256"] = program.sha()
    result["program_exact"] = program.sha() == target_sha256
    if program.n_inputs == len(inputs):
        try:
            result["answer_correct"] = program.run(inputs) == target_value
        except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
            result["parse_status"] = "program_execution_failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--selection-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=3600.)
    parser.add_argument("--canary", type=int, default=0,
                        help="score only the first N frozen requests; never a full result")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.max_tokens <= 1024 or not 0 < args.max_seconds <= 14400
            or not 0 <= args.canary <= 72):
        parser.error("finite decode bounds required")

    from tools.evaluate_semantic_native_checkpoint import (
        digest,
        selected_checkpoint,
        verified_document,
    )
    from tools.evaluate_semantic_native_fresh_schema import frozen_cases
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    configure_refit_environment(args.directory / "report.json")
    from core.brain.llm.model_registry import get_active_cortex_spec

    training, selected = selected_checkpoint(args.training_directory)
    from core.learning.semantic_native_codec import (
        NATIVE_CODEC_IMPLEMENTATION_PATHS,
        register_encoding_from_plan,
    )
    register_encoding = register_encoding_from_plan(training)
    selection = verified_document(args.selection_directory / "plan.json", "plan_sha256")
    if (selection["training_plan_sha256"] != training["plan_sha256"]
            or selection["checkpoint_receipt_sha256"] != selected["receipt_sha256"]
            or register_encoding_from_plan(selection) != register_encoding
            or selection["candidate_inventory"] != "target_derived_witnessed_contrasts"):
        raise ValueError("fresh-generation source selection basis differs")
    cases = frozen_cases(training, examples_per_cell=selection["examples_per_cell"],
                         contrast_limit=selection["contrast_limit"])
    if list(cases) != selection["cases"]:
        raise ValueError("fresh-generation corpus differs from frozen selection")
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("fresh-generation resident identity differs")
    paths = ("tools/evaluate_semantic_native_fresh_generation.py",
             "tools/evaluate_semantic_native_fresh_schema.py",
             "tools/evaluate_semantic_native_checkpoint.py",
             "core/learning/semantic_native_program.py",
             "core/learning/semantic_program_corpus_natural.py",
             "core/brain/llm/chat_format.py", *NATIVE_CODEC_IMPLEMENTATION_PATHS)
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in paths}
    body = {"schema": "aura.semantic_native_fresh_generation_plan.v1",
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "selection_plan_sha256": selection["plan_sha256"],
            "register_encoding": register_encoding,
            "model_descriptor_sha256": spec.descriptor_sha256,
            "pointer_sha256": spec.pointer_sha256, "implementation": implementation,
            "case_sources": [case["source_sha256"] for case in cases],
            "canary": args.canary, "max_tokens": args.max_tokens,
            "max_seconds": args.max_seconds, "candidate_inventory": "none",
            "target_available_to_generation": False,
            "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "population": args.canary or len(cases),
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    examples = {example.example_id: example for example in
                build_semantic_program_natural_request_corpus(
                    examples_per_schema_domain=selection["examples_per_cell"])}
    active_cases = cases[:args.canary] if args.canary else cases
    started, rows = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"semantic-native-generation:{args.directory.name}",
                                model_path=str(spec.model_path), purpose="evaluation",
                                preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0.,
            "keys": training["adapter_keys"]})
        model.load_weights(str(args.training_directory /
                               f"checkpoint-{selected['step']}.safetensors"), strict=False)
        for case in active_cases:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("fresh-generation run exceeded its bound; rows remain durable")
            path = args.directory / "rows" / f"{case['source_sha256']}.json"
            if path.exists():
                row = verified_document(path)
                if (row["plan_sha256"] != plan["plan_sha256"]
                        or row["source_sha256"] != case["source_sha256"]):
                    raise ValueError("fresh-generation resumed row differs")
                rows.append(row)
                continue
            example = examples[case["example_id"]]
            source = example.source_text
            if hashlib.sha256(source.encode()).hexdigest() != case["source_sha256"]:
                raise ValueError("fresh-generation source changed")
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": source}], add_generation_prompt=True,
                tokenize=True)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": source}], add_generation_prompt=True,
                tokenize=False)
            if not isinstance(prompt, list) or not prompt:
                raise ValueError("fresh-generation chat template produced no token prompt")
            if (not isinstance(prompt_text, str)
                    or tokenizer.encode(prompt_text, add_special_tokens=False) != prompt):
                raise ValueError("fresh-generation template text and tokens differ")
            prompt_thinking = prompt_text.rfind("<think>") > prompt_text.rfind("</think>")
            pieces = []
            finish_reason = "token_limit"
            generated = 0
            for frame in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens,
                                         sampler=lambda logits: mx.argmax(logits, axis=-1),
                                         prefill_step_size=512):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("fresh-generation decode exceeded its bound")
                pieces.append(str(frame.text or ""))
                if frame.finish_reason != "stop":
                    generated += 1
                if frame.finish_reason:
                    finish_reason = str(frame.finish_reason)
            raw = "".join(pieces)
            target_value = example.program.run(example.inputs)
            outcome = grade_generation(source, raw, prompt_thinking=prompt_thinking,
                                       target_sha256=case["target_sha256"],
                                       inputs=example.inputs, target_value=target_value,
                                       register_encoding=register_encoding)
            row_body = {"plan_sha256": plan["plan_sha256"],
                        "source_sha256": case["source_sha256"],
                        "example_id": case["example_id"],
                        "construction": case["construction"],
                        "prompt_token_count": len(prompt), "generated_tokens": generated,
                        "prompt_thinking": prompt_thinking,
                        "finish_reason": finish_reason, **outcome,
                        "candidate_inventory": "none", "target_available_to_generation": False}
            row = {**row_body, "receipt_sha256": digest(row_body)}
            _save_if_absent(path, row)
            rows.append(row)
            print(json.dumps({"stage": "generated", "observed": len(rows),
                              "population": len(active_cases),
                              "parse_status": outcome["parse_status"]}), flush=True)
        current = get_active_cortex_spec(force_refresh=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                       for name, sha in implementation.items())
                or selected_checkpoint(args.training_directory) != (training, selected)):
            raise ValueError("fresh-generation implementation, model, or checkpoint drifted")
        result = {"schema": "aura.semantic_native_fresh_generation.v1",
                  "plan_sha256": plan["plan_sha256"], "population": len(rows),
                  "valid_programs": sum(row["parse_status"] == "valid_program" for row in rows),
                  "program_exact": sum(row["program_exact"] for row in rows),
                  "answer_correct": sum(row["answer_correct"] for row in rows),
                  "parse_status_counts": {status: sum(row["parse_status"] == status for row in rows)
                                          for status in sorted({row["parse_status"] for row in rows})},
                  "row_receipts": {row["source_sha256"]: row["receipt_sha256"] for row in rows},
                  "candidate_inventory": "none", "target_available_to_generation": False,
                  "canary": args.canary, "serving_authority": False,
                  "qualification_evidence": False,
                  "elapsed_seconds": time.monotonic() - started}
        if not all(math.isfinite(value) for value in
                   (result["elapsed_seconds"], float(result["population"]))):
            raise ValueError("fresh-generation timing or population is not finite")
        _save_if_absent(args.directory / "report.json", {**result, "receipt_sha256": digest(result)})
        print(json.dumps({key: value for key, value in result.items() if key != "row_receipts"}),
              flush=True)


if __name__ == "__main__":
    main()
