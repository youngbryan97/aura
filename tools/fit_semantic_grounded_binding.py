#!/usr/bin/env python3
"""Fit grounded neural graph binding on verified acquired source folds only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class NativeSourceMetadata:
    ir: object
    split: str
    register_definition_spans: tuple


def compact_native_sources(examples):
    return tuple(NativeSourceMetadata(item.ir, item.split, item.register_definition_spans) for item in examples)


def resume_native_if_available(directory):
    if not Path(directory).exists():
        return False
    from tools.verify_semantic_grounded_restart import verify
    verify(directory)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "source-report", "folds", "bank", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--bundle", action="append", metavar="NAME=PATH")
    parser.add_argument("--channel", action="append", required=True)
    parser.add_argument("--relation-width", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--group-eta", type=float, default=0.)
    parser.add_argument("--domain-reversal", type=float, default=0.)
    parser.add_argument("--stationarity-weight", type=float, default=0.)
    parser.add_argument("--role-margin", type=float, default=0.)
    parser.add_argument("--graph-limit", type=int, default=32)
    parser.add_argument("--equivalence-supervision", action="store_true")
    parser.add_argument("--source-equivariance", action="store_true")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--authority-key-file", type=Path)
    parser.add_argument("--native-rank", type=int, default=32)
    parser.add_argument("--native-layers", type=int, default=8)
    parser.add_argument("--native-max-tokens", type=int, default=512)
    parser.add_argument("--native-cache-mib", type=int, default=512)
    parser.add_argument("--native-kind", choices=("lora", "silu", "product", "routed", "dora", "square", "dense"), default="lora")
    parser.add_argument("--native-layer-kinds", help="comma-separated suffix function classes")
    parser.add_argument("--native-layer-ranks", help="comma-separated suffix ranks")
    parser.add_argument("--native-experts", type=int, default=1)
    parser.add_argument("--native-operation-field", action="store_true")
    parser.add_argument("--operation-relation-width", type=int, default=128)
    parser.add_argument("--complete-program-objective", action="store_true")
    parser.add_argument("--program-charts", type=int, default=4)
    parser.add_argument("--program-graphs", type=int, default=4)
    parser.add_argument("--program-mining-seconds", type=float, default=10.)
    parser.add_argument("--program-pool-directory", type=Path)
    parser.add_argument("--program-stall-trace-seconds", type=float, default=60.)
    parser.add_argument("--revalidate-program-pools", type=Path,
                        help="rebuild factors and reprove existing source pools under current implementation")
    parser.add_argument("--calibration-per-stratum", type=int,
                        help="source-identity quota per eligible construction/depth calibration stratum")
    args = parser.parse_args()
    if args.prepare_only and (not args.native or args.resume or args.resume_if_available):
        parser.error("prepare-only requires a fresh native source fit")
    if args.resume_if_available and (not args.native or args.resume):
        parser.error("resume-if-available requires native fitting and cannot combine with resume")
    if not args.native and (args.native_kind != "lora" or args.native_layer_kinds
            or args.native_layer_ranks or args.native_experts != 1):
        parser.error("native adapter topology requires --native")
    if args.calibration_per_stratum is not None and args.calibration_per_stratum < 1:
        parser.error("calibration quota must be positive")
    if args.native_operation_field and not args.native:
        parser.error("native operation field requires --native")
    if args.complete_program_objective and not args.native_operation_field:
        parser.error("complete-program objective requires --native-operation-field")
    if args.revalidate_program_pools is not None and not args.complete_program_objective:
        parser.error("program pool revalidation requires --complete-program-objective")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
    )
    configure_refit_environment(args.directory.parent / (args.directory.name + "-runtime") / "report.json")
    if args.authority_key_file is not None and not args.native:
        raise ValueError("native authority key requires native source fitting")
    spec = None
    if args.native:
        from tools.train_semantic_native_program import require_native_cortex_spec
        spec = require_native_cortex_spec(args.authority_key_file)
        if args.resume_if_available:
            args.resume = resume_native_if_available(args.directory)
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.train_nested_semantic_ranker import _verified_pair
    from tools.train_semantic_atom_ranker import validate_atom_partition
    raw = {name: path.read_bytes() for name, path in (
        ("parent", args.parent), ("source", args.source_report), ("folds", args.folds))}
    outer, bank_report = _verified_pair(args.bank)
    parent = compositional_semantic_program_transducer_from_dict(json.loads(raw["parent"]))
    if (outer["parent_receipt_sha256"] != parent.receipt_sha256
            or outer["source_report_sha256"] != hashlib.sha256(raw["source"]).hexdigest()
            or outer["folds_sha256"] != hashlib.sha256(raw["folds"]).hexdigest()):
        raise ValueError("grounded source fit differs from the verified bank basis")
    source, folds = json.loads(raw["source"]), json.loads(raw["folds"])
    bundles = source_bundle_arguments(source, feature_root=args.feature_root, bundles=args.bundle)
    examples = load_source_examples(parent, source, bundles)
    fit, calibration_items = validate_atom_partition(examples, outer, folds)
    import mlx.core as mx

    from core.learning.semantic_grounded_binding_acquisition import (
        grounded_source_equivariance_pairs,
        grounded_supervision_from_source_example,
    )
    from core.learning.semantic_grounded_binding_engine import fit_grounded_binding
    from core.learning.semantic_native_fit_sampling import (
        bounded_native_fit_schedule,
        native_depth_calibration_subset,
    )
    from core.learning.semantic_relational_pointer import RelationalBindingPointer
    channels = tuple(args.channel)
    eligible_calibration_count = len(calibration_items)
    if args.calibration_per_stratum is not None:
        selected = set(native_depth_calibration_subset(calibration_items,
            tuple(item.ir.source_text_sha256 for item in calibration_items), per_stratum=args.calibration_per_stratum))
        calibration_items = tuple(item for item in calibration_items if item.ir.source_text_sha256 in selected)
    training = tuple(grounded_supervision_from_source_example(item, channels=channels, graph_limit=args.graph_limit,
                     equivalence_supervision=args.equivalence_supervision)
                     for item in fit)
    calibration = tuple(grounded_supervision_from_source_example(item, channels=channels, graph_limit=args.graph_limit,
                        equivalence_supervision=args.equivalence_supervision)
                        for item in calibration_items)
    pairs = grounded_source_equivariance_pairs(fit, training) if args.source_equivariance else ()
    schedule, sampling_receipt = bounded_native_fit_schedule(fit,
        tuple(item.evidence.source_id for item in training), steps=args.steps, seed=args.seed)
    if args.native_operation_field:
        from types import SimpleNamespace

        from core.learning.semantic_native_operation_field import OperationSetSupervision
        grammar = SimpleNamespace(labels=parent.operation_head.labels, max_steps=parent.max_steps,
            max_span_tokens=parent.max_span_tokens)
        for item in (*fit, *calibration_items):
            OperationSetSupervision(item.ir.source_text_sha256, len(item.ir.source_token_ids),
                item.ir.input_spans, tuple((instruction.op, instruction.operation_span)
                    for instruction in item.ir.instructions)).indices(grammar)
        print(json.dumps({"stage": "grounded_operation_grammar_preflight", "fit_sources": len(fit),
            "calibration_sources": len(calibration_items), "all_source_operations_reachable": True,
            "model_weights_loaded": False, "held_sources_scored": False}), flush=True)
    programs = None
    if args.complete_program_objective:
        from tools.semantic_grounded_program_pool import prepare_program_population
        pool_directory = args.program_pool_directory or args.directory.parent / (args.directory.name + "-program-pools")
        programs, _pool_report = prepare_program_population(parent, (*fit, *calibration_items), pool_directory,
            reuse_directory=args.revalidate_program_pools,
            stall_trace_seconds=args.program_stall_trace_seconds,
            max_charts=args.program_charts, max_graphs=args.program_graphs,
            max_seconds=args.program_mining_seconds)
    if not training:
        raise ValueError("grounded source fit has no observed roles")
    geometry = next(iter(training[0].evidence.candidates.values())).shape
    mx.random.seed(args.seed)
    fit_options = dict(
        steps=args.steps, learning_rate=args.learning_rate, save_every=args.save_every,
        max_seconds=args.max_seconds, group_eta=args.group_eta, domain_reversal=args.domain_reversal,
        stationarity_weight=args.stationarity_weight, equivariance_pairs=pairs, role_margin=args.role_margin,
        training_schedule=schedule, sampling_receipt=sampling_receipt)
    if args.native:
        from tools.semantic_grounded_native_fit import fit_native_grounded_sources
        native_items = compact_native_sources((*fit, *calibration_items))
        discarded = len(examples)
        del examples, fit, calibration_items
        gc.collect()
        print(json.dumps({"stage": "grounded_archived_features_released", "archived_source_count": discarded,
            "native_source_count": len(native_items), "supervision_retained": True}), flush=True)
        _engine, report = fit_native_grounded_sources(training, calibration, native_items, args.directory,
            spec=spec, rank=args.native_rank, layers=args.native_layers, max_tokens=args.native_max_tokens,
            cache_bytes=args.native_cache_mib * 1024 ** 2, seed=args.seed,
            relation_width=args.relation_width, rounds=args.rounds, fit_options=fit_options, resume=args.resume,
            program_supervision=programs,
            operation_options=({"labels": parent.operation_head.labels,
                "relation_width": args.operation_relation_width,
                "max_span_tokens": parent.max_span_tokens, "max_steps": parent.max_steps}
                if args.native_operation_field else None),
            source_basis={"parent_sha256": hashlib.sha256(raw["parent"]).hexdigest(),
                "source_report_sha256": hashlib.sha256(raw["source"]).hexdigest(),
                "folds_sha256": hashlib.sha256(raw["folds"]).hexdigest(), "bank_plan_sha256": outer["plan_sha256"]},
            prepare_only=args.prepare_only, adapter_options={"kind": args.native_kind, "experts": args.native_experts,
                **({"layer_kinds": args.native_layer_kinds.split(",")} if args.native_layer_kinds else {}),
                **({"layer_ranks": [int(value) for value in args.native_layer_ranks.split(",")]} if args.native_layer_ranks else {})})
        if args.prepare_only:
            print(json.dumps({"stage": "grounded_source_prepared", **report}), flush=True)
            return 0
    else:
        pointer = RelationalBindingPointer(geometry[1], depths=geometry[0], relation_width=args.relation_width,
                                          rounds=args.rounds)
        _engine, report = fit_grounded_binding(pointer, training, calibration, args.directory,
                                               resume=args.resume, **fit_options)
    acquisition = {"parent_sha256": hashlib.sha256(raw["parent"]).hexdigest(),
        "source_report_sha256": hashlib.sha256(raw["source"]).hexdigest(),
        "folds_sha256": hashlib.sha256(raw["folds"]).hexdigest(), "bank_plan_sha256": outer["plan_sha256"],
        "bank_candidate_receipt_sha256": bank_report["candidate_receipt_sha256"],
        "channels": channels, "seed": args.seed, "model_weights_loaded": args.native,
        "equivalence_supervision": args.equivalence_supervision, "source_equivariance_pairs": len(pairs),
        "eligible_calibration_count": eligible_calibration_count,
        "calibration_per_stratum": args.calibration_per_stratum,
        "sampling_receipt": sampling_receipt,
        "held_sources_scored": False, "fit_receipt_sha256": report["receipt_sha256"]}
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    with local_internal_governed_scope("grounded_binding_acquisition", domain="file_write"):
        get_file_write_gateway().write_text(args.directory / "acquisition.json", json.dumps(acquisition, indent=2) + "\n",
                                           source="grounded_binding_acquisition")
    print(json.dumps({"stage": "source_fit_complete", "selected_step": report["selected_step"],
                     "receipt_sha256": report["receipt_sha256"], "semantic_success": None}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
