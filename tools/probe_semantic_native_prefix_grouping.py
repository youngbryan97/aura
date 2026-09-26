#!/usr/bin/env python3
"""Measure native split, grouping, and single-row projection before any update."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def supervision_sequences(supervision):
    from core.learning.semantic_native_program import NativeProgramSequence

    rows = supervision["rows"]
    sequences = {(row["source"], row["program_sha256"]): NativeProgramSequence(
        tuple(row["tokens"]), row["continuation_start"], tuple(row["semantic_positions"]))
        for row in rows}
    if not rows or len(sequences) != len(rows):
        raise ValueError("prefix grouping probe needs unique supervision rows")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.train_semantic_native_program import exact_length_batches, native_prediction_positions, selected_projection_error

    configure_refit_environment(args.directory / "report.json")
    plan = verified_document(args.training_directory / "plan.json", "plan_sha256")
    supervision = verified_document(args.training_directory / "supervision.json")
    if supervision["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("prefix grouping probe supervision differs from its plan")
    sequences = supervision_sequences(supervision)
    batch = next((row for row in exact_length_batches(sequences, batch_size=4) if len(row) == 4), None)
    if batch is None:
        raise ValueError("prefix grouping probe lacks an exact-length batch of four")
    from core.brain.llm.model_registry import get_active_cortex_spec
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != plan["model_descriptor_sha256"]
            or spec.pointer_sha256 != plan["pointer_sha256"]):
        raise ValueError("prefix grouping probe model differs from training")
    paths = ("tools/probe_semantic_native_prefix_grouping.py", "tools/train_semantic_native_program.py",
             "core/learning/frozen_decoder_prefix.py", "core/brain/llm/decoder_topology.py")
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    started, rows = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"native-prefix-probe:{args.directory.name}",
            model_path=str(spec.model_path), purpose="evaluation", preemptible=False,
            metadata={"production_effect": False}), mlx_memory_envelope(fraction=.8)):
        model, _ = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - plan["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(plan["seed"])
        linear_to_lora_layers(model, plan["suffix_layers"], {
            "rank": plan["rank"], "scale": 16., "dropout": 0., "keys": plan["adapter_keys"]})
        sequence = sequences[batch[0]]
        positions = tuple(index - 1 for index in native_prediction_positions(sequence, scope=plan["loss_scope"]))
        individual = None
        for size in (1, 2, 4):
            tokens = mx.array([sequences[key].tokens[:-1] for key in batch[:size]], dtype=mx.int32)
            captured = prefix.capture(tokens)
            full = model(tokens)
            split_error = float(mx.max(mx.abs(full - suffix(captured))).item())
            selected_error, tolerance = selected_projection_error(full[:1],
                suffix(captured[:1], logit_positions=positions), sequence, positions)
            if individual is None:
                individual = mx.array(captured[:1])
                mx.eval(individual)
            grouping_error = float(mx.max(mx.abs(captured[:1] - individual)).item())
            individual_error, _ = selected_projection_error(model(tokens[:1]),
                suffix(captured[:1], logit_positions=positions), sequence, positions)
            row = {"batch_size": size, "full_split_max_logit_error": split_error,
                   "selected_target_logprob_error": selected_error, "selected_tolerance": tolerance,
                   "prefix_first_row_grouping_error": grouping_error,
                   "individual_inference_target_logprob_error": individual_error,
                   "accepted": split_error <= .01 and selected_error <= tolerance and individual_error <= tolerance}
            rows.append(row)
            print(row, flush=True)
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha for name, sha in implementation.items()):
            raise ValueError("prefix grouping probe implementation drifted")
        body = {"schema": "aura.native_prefix_grouping_probe.v1", "training_plan_sha256": plan["plan_sha256"],
                "supervision_sha256": supervision["receipt_sha256"], "implementation": implementation,
                "source_programs": batch, "model_descriptor_sha256": spec.descriptor_sha256,
                "pointer_sha256": spec.pointer_sha256, "rows": rows,
                "elapsed_seconds": time.monotonic() - started, "serving_authority": False}
        _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": digest(body)})


if __name__ == "__main__":
    main()
