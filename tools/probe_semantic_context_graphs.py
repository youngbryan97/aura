#!/usr/bin/env python3
"""Measure fixed operation predictions through the frozen argument resolver."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def recover_rows(previous_output: Path, plan: dict) -> tuple[dict, dict]:
    """Recover paired rows only when the computation and population match."""
    previous_plan_path = previous_output.with_suffix(".plan.json")
    previous_plan = json.loads(previous_plan_path.read_text())
    if previous_plan.get("evaluation_split", "train") != plan.get("evaluation_split", "train"):
        raise ValueError("graph recovery changes evaluation_split")
    for key in ("source_ids", "parent_receipt", "arms", "search_time_limit_s",
                "search_mode", "max_expansions", "legacy_context_override", "checkpoints",
                "input_coordinates", "implementation_identity"):
        if previous_plan.get(key) != plan.get(key):
            raise ValueError(f"graph recovery changes {key}")
    tool_path = str(Path(__file__).resolve().relative_to(ROOT))
    for path, digest in previous_plan["source_sha256"].items():
        if path != tool_path and plan["source_sha256"].get(path) != digest:
            raise ValueError(f"graph recovery changes inference source: {path}")
    recovered, identities = {}, {}
    allowed_arms = {arm["name"] for arm in plan["arms"]}
    allowed_sources = set(plan["source_ids"])
    for path in sorted((previous_output.parent / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        key = row["source"], row["arm"]
        if key in recovered or key[0] not in allowed_sources or key[1] not in allowed_arms:
            raise ValueError("recovered rows are duplicated or outside the planned population")
        recovered[key] = row
        identities[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not recovered:
        raise ValueError("graph recovery has no completed rows")
    return recovered, {"plan_path": str(previous_plan_path),
                       "plan_sha256": hashlib.sha256(previous_plan_path.read_bytes()).hexdigest(),
                       "rows_sha256": identities,
                       "newly_bound_source_paths": sorted(set(plan["source_sha256"])
                                                           - set(previous_plan["source_sha256"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, action="append", required=True)
    parser.add_argument("--baseline-candidate", type=Path, action="append", default=[],
                        help="retain and replay an existing transducer on the same observations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--search-mode", choices=("map", "typed"), default="map")
    parser.add_argument("--legacy-context", choices=("full", "local"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.output.exists():
        parser.error("nonnegative limit and new output required")
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from core.learning.semantic_context_graph_probe import resolve_operation_set
    from core.learning.semantic_graph_coordinates import reanchor_program_inputs
    from core.learning.semantic_graph_counterexamples import (
        compare_program_meanings,
        counterfactual_inputs,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from core.learning.semantic_validation_checkpoint import validation_implementation_identity

    root = args.evidence_root
    parent = compositional_semantic_program_transducer_from_dict(json.loads(
        (root / "semantic-source-fit-20260915/candidate.json").read_text()))
    source_report = json.loads((root / "semantic-source-fit-20260915/report.json").read_text())
    examples = load_source_examples(parent, source_report, [name + "=" + str(
        root / "semantic-source-reacquisition-20260915/features" / name)
        for name in source_report["representation_compatibility"]["source_feature_manifest_sha256s"]])
    arms = []
    evaluation_split = None
    for path in args.prediction_report:
        report = json.loads(path.read_text())
        plan = json.loads(path.with_suffix(".plan.json").read_text())
        split = report.get("evaluation_split", "train")
        if split not in {"train", "validation"} or plan.get("evaluation_split", "train") != split:
            raise ValueError("prediction evaluation split differs or is inadmissible")
        if evaluation_split is not None and evaluation_split != split:
            raise ValueError("paired predictions use different evaluation splits")
        evaluation_split = split
        source = {x.ir.source_text_sha256: x for x in examples if x.split == split}
        if report["parent_receipt"] != parent.receipt_sha256:
            raise ValueError("prediction argument parent differs")
        rows = report["heldout_operation_sets"]["rows"]
        ids = sorted(x["source"] for x in rows)
        if len(set(ids)) != len(ids) or set(ids) - set(source):
            raise ValueError("predictions are not unique source examples")
        if arms and ids != arms[0]["ids"]:
            raise ValueError("paired predictions cover different populations")
        arm = {"name": path.parent.name, "labels": plan["labels"], "ids": ids,
                     "rows": {x["source"]: x for x in rows},
                     "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if args.search_mode == "typed":
            import numpy as np
            import torch

            from core.learning.semantic_context_objective import operation_scores
            from core.learning.semantic_context_search_probe import resolve_operation_scores
            from core.learning.semantic_request_context import (
                ContextualSpanRecognizer,
                RequestContextConfig,
            )

            torch.set_num_threads(2)
            checkpoint_path = path.with_suffix(f".epoch-{plan['epochs']}.pt")
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if digest != report["history"][-1]["checkpoint_sha256"]:
                raise ValueError("context checkpoint identity differs from prediction report")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["plan"] != plan:
                # Checkpoint tuples and JSON lists have identical canonical meaning.
                if json.dumps(checkpoint["plan"], sort_keys=True) != json.dumps(plan, sort_keys=True):
                    raise ValueError("context checkpoint plan differs")
            recognizer = ContextualSpanRecognizer(RequestContextConfig(**plan["config"]), tuple(plan["labels"]))
            recognizer.load_state_dict(checkpoint["model"])
            recognizer.eval()
            context = plan.get("context", args.legacy_context)
            if context is None:
                raise ValueError("legacy checkpoint needs an explicit recorded context mode")
            arm.update(name=arm["name"] + "-typed", checkpoint_sha256=digest,
                       recognizer=recognizer, cross_token=context != "local", span_width=plan["span_width"])
        arms.append(arm)
    for path in args.baseline_candidate:
        baseline = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
        if (baseline.model_basis_sha256 != parent.model_basis_sha256
                or baseline.hidden_size != parent.hidden_size):
            raise ValueError("baseline candidate differs from the bound neural representation")
        arms.append({"name": path.parent.name + "-baseline", "ids": arms[0]["ids"],
                     "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "baseline_model": baseline})
    if len({arm["name"] for arm in arms}) != len(arms):
        raise ValueError("graph methods need distinct identities")
    ids = arms[0]["ids"][:args.limit or None]
    plan = {"schema": "aura.semantic_context_graph_probe.v1",
            "source_ids": ids, "parent_receipt": parent.receipt_sha256,
            "arms": [{k: x[k] for k in ("name", "report_sha256")} for x in arms],
            "evaluation_split": evaluation_split,
            "input_coordinates": "annotated_source_anchors_v1",
            "implementation_identity": validation_implementation_identity(),
            "argument_fit_includes_evaluation_examples": evaluation_split == "train",
            "argument_fit_includes_heldout_constructions": True if evaluation_split == "train" else None,
            "promotable": False, "search_time_limit_s": 10.,
            "search_mode": args.search_mode, "max_expansions": 100_000,
            "legacy_context_override": args.legacy_context,
            "checkpoints": {x["name"]: x.get("checkpoint_sha256") for x in arms},
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).resolve(), ROOT / "core/learning/semantic_context_graph_probe.py",
                          ROOT / "core/learning/semantic_program_compositional_transducer.py",
                          ROOT / "core/learning/semantic_context_search_probe.py",
                          ROOT / "core/learning/semantic_request_context.py",
                          ROOT / "core/learning/semantic_context_objective.py",
                          ROOT / "core/learning/semantic_span_autograd.py")}}
    recovered = {}
    if args.resume_from is not None:
        recovered, plan["recovery"] = recover_rows(args.resume_from, plan)
        print(json.dumps({"recovered_rows": len(recovered)}), flush=True)
    if not atomic_write_bytes_if_absent(args.output.with_suffix(".plan.json"),
                                       json.dumps(plan, sort_keys=True).encode()):
        raise RuntimeError("graph probe plan already exists")
    outcomes = []
    for index, identity in enumerate(ids):
        example = source[identity]
        # Rotate paired order; annotations are consulted only after resolution.
        for arm in arms[index % len(arms):] + arms[:index % len(arms)]:
            recovered_row = recovered.get((identity, arm["name"]))
            if recovered_row is not None:
                row_path = args.output.parent / "rows" / f"{index:04d}-{arm['name']}.json"
                if not atomic_write_bytes_if_absent(row_path, json.dumps(recovered_row, sort_keys=True).encode()):
                    raise RuntimeError("recovered graph row already exists")
                outcomes.append(recovered_row)
                continue
            prediction = arm.get("rows", {}).get(identity)
            started = time.monotonic()
            result = None
            spans = None
            search_trace = {}
            try:
                arguments = dict(source_token_ids=example.ir.source_token_ids,
                                 hidden_states=example.hidden_states, public_inputs=example.public_inputs)
                if "baseline_model" in arm:
                    baseline = arm["baseline_model"]
                    decoded = baseline.decode(**arguments, source_text_sha256=identity,
                                              model_basis_sha256=parent.model_basis_sha256,
                                              search_time_limit_s=plan["search_time_limit_s"])
                    result = None if decoded.ir is None else decoded.ir.to_program()
                    spans = None if decoded.ir is None else decoded.ir.input_spans
                    refusal = decoded.refusal
                    search_trace = {"transducer_receipt": baseline.receipt_sha256}
                elif args.search_mode == "typed":
                    with torch.no_grad():
                        scores = operation_scores(arm["recognizer"],
                            torch.from_numpy(np.array(example.hidden_states, copy=True)),
                            span_width=arm["span_width"], cross_token=arm["cross_token"]).numpy()
                    result, refusal, search_trace = resolve_operation_scores(
                        parent, **arguments, scores=scores, labels=arm["labels"],
                        time_limit_s=10., max_expansions=plan["max_expansions"])
                else:
                    operations = tuple((int(s), int(e), arm["labels"][int(label)])
                                       for s, e, label in prediction["selected"])
                    result, refusal = resolve_operation_set(
                        parent, **arguments, operations=operations, time_limit_s=10.)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                refusal = f"{type(exc).__name__}:{exc}"
            resolution_s = time.monotonic() - started
            original_program = None if result is None else result.to_dict()
            coordinate_status = "no_program"
            if result is None:
                comparison = {"status": "refused", "reason": refusal}
            else:
                try:
                    if spans is None:
                        spans, _, _ = parent._runtime_input_grounding(
                            example.ir.source_token_ids, example.hidden_states, example.public_inputs)
                    result = reanchor_program_inputs(result, from_spans=spans,
                        to_spans=example.ir.input_spans, from_inputs=example.public_inputs,
                        to_inputs=example.public_inputs)
                except ValueError as exc:
                    coordinate_status = "unaligned"
                    result = None
                    comparison = {"status": "unknown", "reason": str(exc)}
                else:
                    coordinate_status = "aligned"
                    comparison = compare_program_meanings(example.ir.to_program(), result,
                        counterfactual_inputs(example.public_inputs, count=16))
            row = {"source": identity, "arm": arm["name"], "comparison": comparison,
                   "resolution_s": resolution_s,
                   "coordinate_status": coordinate_status, "original_program": original_program,
                   "runtime_input_spans": None if spans is None else [[s.start, s.end] for s in spans],
                   "source_input_spans": [[s.start, s.end] for s in example.ir.input_spans],
                   "map_operation_exact": None if prediction is None else prediction["exact"],
                   "search_trace": search_trace,
                   "program": None if result is None else result.to_dict()}
            row_path = args.output.parent / "rows" / f"{index:04d}-{arm['name']}.json"
            if not atomic_write_bytes_if_absent(row_path, json.dumps(row, sort_keys=True).encode()):
                raise RuntimeError("graph row already exists")
            outcomes.append(row)
            print(json.dumps({"row": index, "arm": arm["name"],
                              "status": comparison["status"], "resolution_s": resolution_s}), flush=True)
    if validation_implementation_identity() != plan["implementation_identity"]:
        raise ValueError("inference implementation changed during graph measurement")
    for path, digest in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"graph source changed during measurement: {path}")
    summary = {arm["name"]: dict(Counter(row["comparison"]["status"] for row in outcomes
                                        if row["arm"] == arm["name"])) for arm in arms}
    result = {"plan": plan, "summary": summary, "rows": outcomes}
    if not atomic_write_bytes_if_absent(args.output, json.dumps(result, sort_keys=True).encode()):
        raise RuntimeError("graph report already exists")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
