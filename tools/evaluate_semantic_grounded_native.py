#!/usr/bin/env python3
"""Compare the joint native chart on a declared development bank, never serve it."""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import math
import pstats
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def profile_receipt(profiler):
    rows = []
    for (filename, line, function), (primitive, calls, elapsed, cumulative, _callers) in pstats.Stats(profiler).stats.items():
        rows.append({"file": filename, "line": line, "function": function, "primitive_calls": primitive,
            "calls": calls, "self_seconds": elapsed, "cumulative_seconds": cumulative})
    body = {"schema": "aura.grounded_decode_profile.v1",
        "rows": sorted(rows, key=lambda row: (-row["cumulative_seconds"], row["file"], row["line"]))}
    return {**body, "receipt_sha256": digest(body)}


class PublicDecodeArm:
    def __init__(self, decoder, *, receipt, search_seconds, name=None, progress=None, profile=False):
        self.decoder, self.search_seconds = decoder, search_seconds
        self.name, self.progress = name, progress
        self.profile = profile
        self.receipt_sha256 = receipt
        self.measured_decodes = {}

    def decode(self, *, source_token_ids, hidden_states, public_inputs, source_text_sha256, model_basis_sha256):
        started = time.monotonic()
        if self.progress is not None:
            self.progress({"stage": "grounded_decode_started", "arm": self.name, "source": source_text_sha256})
        profiler = cProfile.Profile() if self.profile else None
        if profiler is not None:
            profiler.enable()
        try:
            result = self.decoder.decode(source_token_ids=source_token_ids, hidden_states=hidden_states,
                public_inputs=public_inputs, source_text_sha256=source_text_sha256,
                model_basis_sha256=model_basis_sha256, search_time_limit_s=self.search_seconds)
        finally:
            if profiler is not None:
                profiler.disable()
        measured = getattr(self.decoder, "last_receipt", None)
        self.measured_decodes[source_text_sha256] = {"refusal": result.refusal,
            "program_sha256": result.ir.to_program().sha() if result.ir is not None else None,
            "joint_receipt": measured, "elapsed_seconds": time.monotonic() - started}
        if profiler is not None:
            self.measured_decodes[source_text_sha256]["profile"] = profile_receipt(profiler)
        if self.progress is not None:
            self.progress({"stage": "grounded_decode_completed", "arm": self.name, "source": source_text_sha256,
                **{key: value for key, value in self.measured_decodes[source_text_sha256].items() if key != "profile"}})
        return result


def compare(arms, items):
    from core.learning.semantic_program_compositional_campaign import (
        select_compositional_program_candidate,
    )

    if set(arms) != {"source_parent", "global_chart", "joint_native"}:
        raise ValueError("joint development screen needs all three declared arms")
    selected = tuple(replace(item, split="validation") for item in items)
    result = select_compositional_program_candidate(arms, selected, incumbent="source_parent", scoring="source_anchors_v2")
    metrics = result["candidates"]["joint_native"]
    candidate = metrics["program_equivalent_correct"]
    controls = {name: result["candidates"][name]["program_equivalent_correct"]
        for name in ("source_parent", "global_chart")}
    regressions = {name: sum(old and not new for old, new in zip(values, candidate, strict=True))
        for name, values in controls.items()}
    return {"comparison": result, "decodes": {name: arm.measured_decodes for name, arm in arms.items()},
        "paired_regressions": regressions, "advance_development": all(candidate)
        and metrics["equivalent_gains"] > 0 and not any(regressions.values()),
        "held_used_for_fit_or_checkpoint_selection": False, "qualification_evidence": False,
        "serving_authority": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "source-report", "folds", "bank", "directory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--authority-key-file", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=30.)
    parser.add_argument("--profile-source-id", help="One exposed development request; never advancement evidence")
    parser.add_argument("--chart-execution", choices=("individual", "batched"), default="individual")
    args = parser.parse_args(argv)
    if args.output.exists() or not math.isfinite(args.search_seconds) or not 0 < args.search_seconds <= 300:
        parser.error("evaluation needs a fresh output and bounded search")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )
    configure_refit_environment(args.output)
    import mlx.core as mx
    from mlx_lm import load

    from core.governance_context import local_internal_governed_scope
    from core.learning.semantic_grounded_binding_engine import implementation_receipt
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.file_read_gateway import read_stable_bytes
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.semantic_grounded_native_decode import GroundedNativeChartDecoder
    from tools.train_nested_semantic_ranker import _verified_pair
    from tools.train_semantic_atom_ranker import validate_atom_partition
    from tools.train_semantic_native_program import require_native_cortex_spec
    from tools.verify_semantic_grounded_fit import verify

    verified = verify(args.directory)
    if not verified["learned_checkpoint_selected"]:
        raise ValueError("joint development screen needs a positive-step selected checkpoint")
    spec = require_native_cortex_spec(args.authority_key_file)
    raw = {name: read_stable_bytes(path, max_bytes=64 * 1024 ** 2) for name, path in (
        ("parent", args.parent), ("source", args.source_report), ("folds", args.folds))}
    parent = compositional_semantic_program_transducer_from_dict(json.loads(raw["parent"]))
    outer, _bank_report = _verified_pair(args.bank)
    source, folds = json.loads(raw["source"]), json.loads(raw["folds"])
    if (outer["parent_receipt_sha256"] != parent.receipt_sha256
            or outer["source_report_sha256"] != hashlib.sha256(raw["source"]).hexdigest()
            or outer["folds_sha256"] != hashlib.sha256(raw["folds"]).hexdigest()):
        raise ValueError("joint development bank changed its source basis")
    examples = load_source_examples(parent, source, args.bundle)
    validate_atom_partition(examples, outer, folds)
    by_id = {item.ir.source_text_sha256: item for item in examples}
    held = tuple(by_id[identity] for identity in outer["held_ids"])
    if args.profile_source_id is not None:
        held = tuple(item for item in held if item.ir.source_text_sha256 == args.profile_source_id)
        if len(held) != 1:
            raise ValueError("profile source must belong to the declared exposed development bank")
    fit_report = json.loads(read_stable_bytes(args.directory / "report.json", max_bytes=64 * 1024 ** 2))
    native = fit_report["native_contract"]
    if (set(outer["held_ids"]) & set((*native["fit_ids"], *native["calibration_ids"]))
            or native["source_basis"]["parent_sha256"] != hashlib.sha256(raw["parent"]).hexdigest()
            or native["source_basis"]["bank_plan_sha256"] != outer["plan_sha256"]):
        raise ValueError("joint development holdout overlaps fitting or changes the frozen bank")
    del by_id, examples
    gc.collect()
    implementation = implementation_receipt()
    from tools.semantic_grounded_batched_chart import execution_contract
    execution = execution_contract() if args.chart_execution == "batched" else None
    evaluator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan = {"schema": "aura.grounded_native_development_plan.v1", "fit_verification": verified,
        "bank_plan_sha256": outer["plan_sha256"], "held_ids": [item.ir.source_text_sha256 for item in held],
        "profile_only": args.profile_source_id is not None,
        "chart_execution": args.chart_execution, "execution_contract": execution,
        "search_seconds": args.search_seconds, "arms": ["source_parent", "global_chart", "joint_native"],
        "implementation": implementation, "evaluator_sha256": evaluator_sha,
        "source_report_sha256": hashlib.sha256(raw["source"]).hexdigest(),
        "cohort": "previously_exposed_source_bank_development", "scoring": "source_anchors_v2",
        "held_controls_fit_or_checkpoint_selection": False, "serving_authority": False}
    plan = {**plan, "plan_sha256": digest(plan)}
    gateway = get_file_write_gateway()
    with local_internal_governed_scope("grounded_native_development", domain="file_write"):
        if not gateway.write_bytes_if_absent(args.output.with_suffix(".plan.json"),
                json.dumps(plan, indent=2).encode(), source="grounded_native_development", mode=0o400):
            raise FileExistsError("joint development plan already exists")

    def execute():
        model, _tokenizer = load(str(spec.model_path))
        decoder = GroundedNativeChartDecoder.from_fit(model, directory=args.directory, parent_bytes=raw["parent"], spec=spec)
        if args.chart_execution == "batched":
            from tools.semantic_grounded_batched_chart import BatchedNativeChartDecoder
            decoder = BatchedNativeChartDecoder(decoder)
        chart = parent.with_global_constraint_arguments().with_joint_operation_argument_scores()
        arms = {name: PublicDecodeArm(owner, receipt=digest({"arm": name, "plan": plan["plan_sha256"]}),
            search_seconds=args.search_seconds, name=name,
            profile=args.profile_source_id is not None,
            progress=lambda row: print(json.dumps(row), flush=True)) for name, owner in (
            ("source_parent", parent), ("global_chart", chart), ("joint_native", decoder))}
        return compare(arms, held)

    with (standalone_model_lane(owner_id=f"grounded-evaluation:{args.output.stem}", model_path=str(spec.model_path),
            purpose="evaluation", preemptible=False, require_exclusive=True, allow_owner_eviction=False,
            metadata={"tool": "evaluate_semantic_grounded_native", "production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        try:
            comparison = execute()
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            gc.collect()
            mx.clear_cache()
    if implementation_receipt() != implementation or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != evaluator_sha:
        raise ValueError("joint development implementation changed during evaluation")
    if execution is not None and execution_contract() != execution:
        raise ValueError("joint batched execution changed during evaluation")
    body = {"schema": "aura.grounded_native_development.v1", "plan": plan, **comparison,
        "g03_complete": False, "fresh_transfer_proven": False}
    if args.profile_source_id is not None:
        body["advance_development"] = False
    report = {**body, "receipt_sha256": digest(body)}
    with local_internal_governed_scope("grounded_native_development", domain="file_write"):
        if not gateway.write_bytes_if_absent(args.output, json.dumps(report, indent=2).encode(),
                source="grounded_native_development", mode=0o400):
            raise FileExistsError(args.output)
    print(json.dumps({"stage": "grounded_development_complete", "receipt_sha256": report["receipt_sha256"],
        "advance_development": report["advance_development"], "metrics": report["comparison"]["candidates"]}), flush=True)
    return 0 if report["advance_development"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
