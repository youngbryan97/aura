#!/usr/bin/env python3
"""Refit semantic evidence from the exact source cohort of a frozen parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def configure_refit_environment(output):
    """Keep standalone and supervised refits out of the live state and logs."""
    root = Path(output).expanduser().absolute().parent
    os.environ.setdefault("AURA_LOG_DIR", str(root / "logs"))
    os.environ.setdefault("AURA_STATE_ROOT", str(root / "state"))


def verify_source_splits(examples, receipt):
    """Reject both missing coverage and substituted examples before training."""
    for split, prefix in (("train", "training"), ("validation", "validation")):
        ids = sorted(x.ir.source_text_sha256 for x in examples if x.split == split)
        digest = hashlib.sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if (
            not ids
            or len(ids) != receipt.get(f"{prefix}_example_count")
            or digest != receipt.get(f"{prefix}_example_ids_sha256")
        ):
            raise ValueError(f"{split} source cohort differs from frozen parent")


def load_source_examples(model, report, bundles):
    """Share the exact parent-bound source admission across refits and diagnostics."""
    from core.learning.semantic_program_basis import bind_training_examples_to_shared_representation
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_feature_materialization import load_standard_semantic_feature_bundle

    compatibility = report["representation_compatibility"]
    expected = compatibility["source_feature_manifest_sha256s"]
    examples = {}
    for value in bundles:
        name, separator, path = value.partition("=")
        if not separator or name not in expected or name in examples:
            raise ValueError("source bundles must uniquely name every source family")
        bundle = load_standard_semantic_feature_bundle(Path(path).expanduser())
        if bundle.manifest["manifest_sha256"] != expected[name]:
            raise ValueError(f"source manifest differs: {name}")
        examples[name] = training_examples_from_feature_bundle(bundle, required_splits=frozenset({"train"}))
    if set(examples) != set(expected):
        raise ValueError("source bundles must include every source family")
    bound = bind_training_examples_to_shared_representation(examples, compatibility=compatibility)
    verify_source_splits(bound, model.training_receipt)
    return bound


def verify_fit_start(candidate, starting):
    """Recover a comparison arm only when the saved fit binds that exact parent."""
    receipt = candidate.training_receipt.get("joint_graph_refit", {})
    if receipt.get("parent_transducer_receipt_sha256") != starting.receipt_sha256:
        raise ValueError("comparison starting candidate differs from saved fit parent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--starting-candidate", type=Path,
                        help="fit this development candidate while retaining transducer as the comparison incumbent")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--validation-checkpoint", type=Path)
    parser.add_argument("--fit-checkpoint-dir", type=Path,
                        help="retained-constraint optimizer state; defaults beside the output candidate")
    parser.add_argument("--evaluate-existing", action="store_true",
                        help="evaluate the saved output candidate without fitting again")
    parser.add_argument("--runtime-operation-views", action="store_true")
    parser.add_argument("--runtime-mention-margin", action="store_true")
    parser.add_argument("--joint-operation-argument-scores", action="store_true")
    parser.add_argument("--objective", choices=("binary_proposals", "pairwise_arguments", "graph_factors", "graph_relations", "joint_graphs", "operation_pointer", "argument_pointer", "definition_pointer", "operation_views", "paired_operation_pointer", "ranked_operation_pointer"),
                        default="binary_proposals")
    parser.add_argument("--graph-rounds", type=int, default=3)
    parser.add_argument("--graph-update-steps", type=int, default=100)
    parser.add_argument("--source-operation-weight", type=float, default=1.,
                        help="source-label retention in joint_graphs; zero reproduces contrast-only fitting")
    parser.add_argument("--retain-semantic-constraints", action="store_true",
                        help="joint_graphs only: retain satisfied witnesses and source bindings during fitting")
    parser.add_argument("--learn-argument-heads", action="store_true",
                        help="retained joint_graphs only: differentiate runtime argument-mention heads too")
    parser.add_argument("--compare-fit-start", action="store_true",
                        help="also evaluate the pre-fit candidate to separate decoder changes from learning")
    args = parser.parse_args()
    configure_refit_environment(args.output)
    if args.evaluate_existing and args.validation_output is None:
        parser.error("evaluate-existing requires validation-output")
    if args.validation_output is not None:
        args.validation_checkpoint = args.validation_checkpoint or args.validation_output.with_suffix(".checkpoint.json")
    if args.validation_checkpoint is not None and args.validation_checkpoint.resolve() in {
        path.resolve() for path in (args.output, args.transducer, args.source_report, args.validation_output)
        if path is not None
    }:
        parser.error("validation checkpoint must not overwrite inputs or final outputs")
    if args.runtime_operation_views and args.objective != "pairwise_arguments":
        parser.error("runtime operation views require pairwise_arguments")
    if args.runtime_mention_margin and args.objective != "pairwise_arguments":
        parser.error("runtime mention margin requires pairwise_arguments")
    if args.joint_operation_argument_scores and args.objective != "joint_graphs":
        parser.error("joint operation-argument scoring requires joint_graphs")
    if args.retain_semantic_constraints and (args.objective != "joint_graphs" or args.evaluate_existing):
        parser.error("semantic constraints require a fresh joint_graphs fit")
    if args.learn_argument_heads and not args.retain_semantic_constraints:
        parser.error("argument-head learning requires retained semantic constraints")
    if args.fit_checkpoint_dir is not None and not args.retain_semantic_constraints:
        parser.error("fit checkpoints require a retained-constraint training run")
    if args.evaluate_existing and not args.compare_fit_start and (
        args.starting_candidate or args.joint_operation_argument_scores
    ):
        parser.error("evaluate-existing starting options require compare-fit-start")
    from core.learning.semantic_operation_view_refit import refit_compositional_operation_views
    from core.learning.semantic_graph_margin import refit_compositional_graph_scales
    from core.learning.semantic_relation_graph_learning import refit_compositional_graph_relations
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_paired_pointer_refit import (
        refit_compositional_paired_operation_pointer,
    )
    from core.learning.semantic_program_compositional_campaign import (
        select_compositional_program_candidate,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        refit_compositional_argument_proposals,
        refit_compositional_argument_rankings,
        refit_compositional_definition_pointer,
        refit_compositional_operation_pointer,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    if args.output.exists() and not args.evaluate_existing:
        raise FileExistsError(args.output)
    if args.validation_output is not None and (
        args.validation_output.exists()
        or args.validation_output.resolve() == args.output.resolve()
    ):
        raise FileExistsError(args.validation_output)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_text("ascii"))
    )
    report = json.loads(args.source_report.read_text("ascii"))
    bound = load_source_examples(model, report, args.bundle)
    starting = model
    if args.starting_candidate is not None:
        starting = compositional_semantic_program_transducer_from_dict(
            json.loads(args.starting_candidate.read_text("ascii")))
        verify_source_splits(bound, starting.training_receipt)
        if (starting.model_basis_sha256 != model.model_basis_sha256
                or starting.input_grounding != model.input_grounding
                or (starting.hidden_channels, starting.hidden_channel_widths)
                != (model.hidden_channels, model.hidden_channel_widths)):
            raise ValueError("starting candidate representation differs from incumbent")
    if args.joint_operation_argument_scores:
        starting = starting.with_joint_operation_argument_scores()
    refit = {
        "binary_proposals": refit_compositional_argument_proposals,
        "pairwise_arguments": refit_compositional_argument_rankings,
        "graph_factors": refit_compositional_graph_scales,
        "graph_relations": refit_compositional_graph_relations,
        "joint_graphs": refit_compositional_joint_graphs,
        "operation_pointer": refit_compositional_operation_pointer,
        "argument_pointer": refit_compositional_argument_proposals,
        "definition_pointer": refit_compositional_definition_pointer,
        "operation_views": refit_compositional_operation_views,
        "paired_operation_pointer": refit_compositional_paired_operation_pointer,
        "ranked_operation_pointer": refit_compositional_paired_operation_pointer,
    }[args.objective]
    options = {"refit_pointer": True} if args.objective == "argument_pointer" else {}
    if args.objective in {"graph_factors", "graph_relations", "joint_graphs"}:
        options["progress"] = lambda row: print(json.dumps(row, sort_keys=True), flush=True)
    if args.objective in {"graph_relations", "joint_graphs"}:
        options.update(rounds=args.graph_rounds, steps=args.graph_update_steps)
    if args.objective == "joint_graphs":
        options["source_weight"] = args.source_operation_weight
        options["constraint_learning"] = args.retain_semantic_constraints
        options["learn_arguments"] = args.learn_argument_heads
        if args.retain_semantic_constraints:
            options["checkpoint_dir"] = args.fit_checkpoint_dir or args.output.with_suffix(".fit-checkpoints")
    if args.runtime_mention_margin:
        options["runtime_mention_margin"] = True
    if args.runtime_operation_views:
        options["use_runtime_operation_views"] = True
        options["progress"] = lambda row: print(json.dumps(row, sort_keys=True), flush=True)
    if args.objective == "ranked_operation_pointer":
        options = {"ranking": True}
    if args.evaluate_existing:
        candidate = compositional_semantic_program_transducer_from_dict(
            json.loads(args.output.read_text("ascii"))
        )
        verify_source_splits(bound, candidate.training_receipt)
        if candidate.model_basis_sha256 != model.model_basis_sha256:
            raise ValueError("saved candidate representation differs from incumbent")
        if args.compare_fit_start:
            verify_fit_start(candidate, starting)
    else:
        candidate = refit(starting, bound, **options)
        payload = (json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
        if not atomic_write_bytes_if_absent(args.output, payload.encode("ascii"), mode=0o400):
            raise FileExistsError(args.output)
    if args.validation_output is not None:
        candidates = {"incumbent": model, "refit": candidate}
        if args.compare_fit_start:
            candidates["fit_start"] = starting
        selection = select_compositional_program_candidate(
            candidates, bound, incumbent="incumbent",
            checkpoint_path=args.validation_checkpoint,
            progress=lambda row: print(json.dumps(row, sort_keys=True), flush=True),
        )
        payload = json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n"
        if not atomic_write_bytes_if_absent(
            args.validation_output, payload.encode("ascii"), mode=0o400
        ):
            raise FileExistsError(args.validation_output)
    print(json.dumps({"receipt_sha256": candidate.receipt_sha256, "serving_authority": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
