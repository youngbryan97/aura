#!/usr/bin/env python3
"""Measure native candidate discrimination on source-disjoint three-step requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.learning.procedure_induction import Program

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "aura.semantic_native_fresh_schema_plan.v1"
IMPLEMENTATION = (
    "tools/evaluate_semantic_native_fresh_schema.py",
    "tools/evaluate_semantic_native_checkpoint.py",
    "tools/train_semantic_native_program.py",
    "core/learning/frozen_decoder_prefix.py",
    "core/learning/semantic_native_program.py",
    "core/learning/semantic_candidate_contrasts.py",
    "core/learning/semantic_program_corpus_natural.py",
    "core/learning/semantic_program_corpus.py",
    "core/learning/procedure_induction.py",
)


def _program_record(program: Program) -> dict:
    return {"sha256": program.sha(), "n_inputs": program.n_inputs,
            "instructions": [[step.op, list(step.args)] for step in program.instructions]}


def _program_from_record(record: dict) -> Program:
    from core.learning.procedure_induction import Instruction, Program

    program = Program(record["n_inputs"], tuple(
        Instruction(op, tuple(args)) for op, args in record["instructions"]
    ))
    if program.sha() != record["sha256"]:
        raise ValueError("fresh-schema program identity differs")
    return program


def frozen_cases(training: dict, *, examples_per_cell: int, contrast_limit: int) -> tuple[dict, ...]:
    from core.learning.semantic_candidate_contrasts import source_program_contrasts
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )

    if examples_per_cell < 1 or contrast_limit < 2:
        raise ValueError("fresh-schema cohort needs positive cells and contrasts")
    forbidden = set(training["fit_ids"]) | set(training["calibration_ids"]) | set(training["held_ids"])
    cases = []
    for example in build_semantic_program_natural_request_corpus(
            examples_per_schema_domain=examples_per_cell):
        source_sha = hashlib.sha256(example.source_text.encode()).hexdigest()
        if source_sha in forbidden or example.split == "train":
            raise ValueError("fresh-schema source overlaps training or development")
        candidates = source_program_contrasts(
            example.program, example.inputs, (), source_sha256=source_sha,
            limit=contrast_limit)
        if len(candidates) < 2 or len({p.sha() for p in candidates}) != len(candidates):
            raise ValueError("fresh-schema contrast inventory is incomplete")
        cases.append({"example_id": example.example_id,
                      "construction": example.construction_id,
                      "source_sha256": source_sha,
                      "target_sha256": example.program.sha(),
                      "candidates": [_program_record(program) for program in candidates]})
    if len({case["source_sha256"] for case in cases}) != len(cases):
        raise ValueError("fresh-schema cohort repeats a source")
    return tuple(cases)


def choose_by_scores(programs: tuple[Program, ...], scores: tuple[float, ...]) -> str:
    """The scorer sees candidate programs and model scores, never the gold program."""
    if not programs or len(programs) != len(scores) or any(not math.isfinite(x) for x in scores):
        raise ValueError("fresh-schema scorer needs complete finite candidate scores")
    return programs[max(range(len(scores)), key=scores.__getitem__)].sha()


def verified_row(row: dict, case: dict, plan_sha256: str) -> dict:
    from tools.evaluate_semantic_native_checkpoint import digest

    if row.get("receipt_sha256") != digest({k: v for k, v in row.items() if k != "receipt_sha256"}):
        raise ValueError("fresh-schema row digest differs")
    programs = tuple(_program_from_record(record) for record in case["candidates"])
    keys = [p.sha() for p in programs]
    if (row.get("plan_sha256") != plan_sha256 or row.get("source_sha256") != case["source_sha256"]
            or row.get("example_id") != case["example_id"]
            or row.get("program_sha256s") != keys
            or row.get("labels_available_to_scorer") is not False
            or row.get("selected_sha256") != choose_by_scores(programs, tuple(row["scores"]))
            or row.get("unfitted_sha256") != choose_by_scores(programs, tuple(row["unfitted_scores"]))
            or row.get("selected_correct") is not (row["selected_sha256"] == case["target_sha256"])
            or row.get("unfitted_correct") is not (row["unfitted_sha256"] == case["target_sha256"])):
        raise ValueError("fresh-schema row outcome differs from independent grading")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--examples-per-cell", type=int, default=3)
    parser.add_argument("--contrast-limit", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=3600.)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 14400:
        parser.error("finite positive evaluation bound required")

    from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    configure_refit_environment(args.directory / "report.json")
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )

    training, selected = selected_checkpoint(args.training_directory)
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("fresh-schema model identity differs from native training")
    cases = frozen_cases(training, examples_per_cell=args.examples_per_cell,
                         contrast_limit=args.contrast_limit)
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in IMPLEMENTATION}
    body = {"schema": SCHEMA, "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "weights_sha256": selected["weights_sha256"], "selected_step": selected["step"],
            "model_descriptor_sha256": spec.descriptor_sha256,
            "pointer_sha256": spec.pointer_sha256, "implementation": implementation,
            "examples_per_cell": args.examples_per_cell, "contrast_limit": args.contrast_limit,
            "cases": cases, "max_seconds": args.max_seconds,
            "candidate_inventory": "target_derived_witnessed_contrasts",
            "end_to_end_proposal_evidence": False, "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "population": len(cases),
                          "candidates": sum(len(row["candidates"]) for row in cases),
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
        return

    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_program import native_program_sequence
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.train_semantic_native_program import native_loss

    started = time.monotonic()
    rows = []
    with (standalone_model_lane(owner_id=f"semantic-native-fresh:{args.directory.name}",
                                model_path=str(spec.model_path), purpose="evaluation",
                                preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - training["suffix_layers"]
        prefix = FrozenDecoderPrefix(model, split_at=split)
        suffix = NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0.,
            "keys": training["adapter_keys"]})
        baseline = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(baseline)
        model.load_weights(str(args.training_directory /
                               f"checkpoint-{selected['step']}.safetensors"), strict=False)
        adapted = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
        mx.eval(adapted)
        examples = {example.example_id: example for example in
                    build_semantic_program_natural_request_corpus(
                        examples_per_schema_domain=args.examples_per_cell)}
        for case in cases:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("fresh-schema run exceeded its bound; rows remain resumable")
            path = args.directory / "rows" / f"{case['source_sha256']}.json"
            if path.exists():
                rows.append(verified_row(json.loads(path.read_bytes()), case,
                                         plan["plan_sha256"]))
                continue
            source = examples[case["example_id"]].source_text
            if hashlib.sha256(source.encode()).hexdigest() != case["source_sha256"]:
                raise ValueError("fresh-schema source changed after plan freeze")
            programs = tuple(_program_from_record(record) for record in case["candidates"])
            scores, controls = [], []
            for program in programs:
                sequence = native_program_sequence(
                    source, program, tokenizer, max_tokens=training["max_sequence_tokens"],
                    decision_basis=training["semantic_decision_basis"])
                hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                scores.append(-native_loss(suffix, hidden, sequence, summed=True,
                                           scope=training["loss_scope"]).item())
                suffix.update(baseline)
                try:
                    controls.append(-native_loss(suffix, hidden, sequence, summed=True,
                                                 scope=training["loss_scope"]).item())
                finally:
                    suffix.update(adapted)
            chosen = choose_by_scores(programs, tuple(scores))
            control = choose_by_scores(programs, tuple(controls))
            row_body = {"plan_sha256": plan["plan_sha256"],
                        "example_id": case["example_id"],
                        "source_sha256": case["source_sha256"],
                        "construction": case["construction"],
                        "program_sha256s": [p.sha() for p in programs],
                        "scores": scores, "unfitted_scores": controls,
                        "selected_sha256": chosen, "unfitted_sha256": control,
                        "selected_correct": chosen == case["target_sha256"],
                        "unfitted_correct": control == case["target_sha256"],
                        "labels_available_to_scorer": False}
            row = {**row_body, "receipt_sha256": digest(row_body)}
            verified_row(row, case, plan["plan_sha256"])
            _save_if_absent(path, row)
            rows.append(row)
            print(json.dumps({"stage": "fresh", "observed": len(rows),
                              "population": len(cases)}), flush=True)
        current = get_active_cortex_spec(force_refresh=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                       for name, sha in implementation.items())
                or selected_checkpoint(args.training_directory) != (training, selected)):
            raise ValueError("fresh-schema implementation, model, or weights drifted")
        result = {"schema": "aura.semantic_native_fresh_schema.v1",
                  "plan_sha256": plan["plan_sha256"], "population": len(rows),
                  "selected_correct": sum(row["selected_correct"] for row in rows),
                  "unfitted_correct": sum(row["unfitted_correct"] for row in rows),
                  "selected_only": sum(row["selected_correct"] and not row["unfitted_correct"]
                                       for row in rows),
                  "unfitted_only": sum(row["unfitted_correct"] and not row["selected_correct"]
                                       for row in rows),
                  "row_receipts": {row["source_sha256"]: row["receipt_sha256"] for row in rows},
                  "candidate_inventory": "target_derived_witnessed_contrasts",
                  "end_to_end_proposal_evidence": False,
                  "held_labels_used_for_fit_or_selection": False,
                  "serving_authority": False, "qualification_evidence": False,
                  "elapsed_seconds": time.monotonic() - started}
        _save_if_absent(args.directory / "report.json",
                        {**result, "receipt_sha256": digest(result)})
        print(json.dumps({key: value for key, value in result.items()
                          if key != "row_receipts"}), flush=True)


if __name__ == "__main__":
    main()
