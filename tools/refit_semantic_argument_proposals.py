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
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )

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


def load_evaluation_candidate(path, starting, *, round_index=None, numerical_checkpoint=None):
    """Read either a final model or a parent-bound accepted training round."""
    if (round_index is None) != (numerical_checkpoint is None):
        raise ValueError("round evaluation requires its numerical checkpoint")
    if round_index is not None:
        from core.learning.semantic_fit_checkpoint import load_round_candidate

        return load_round_candidate(path, expected_parent=starting.receipt_sha256,
            expected_round=round_index, numerical_checkpoint=numerical_checkpoint)
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )

    return compositional_semantic_program_transducer_from_dict(json.loads(Path(path).read_text("ascii")))


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
    parser.add_argument("--validation-scoring", choices=("register_indices_v1", "source_anchors_v2"),
                        default="source_anchors_v2")
    parser.add_argument("--fit-checkpoint-dir", type=Path,
                        help="retained-constraint optimizer state; defaults beside the output candidate")
    parser.add_argument("--evaluate-existing", action="store_true",
                        help="evaluate the saved output candidate without fitting again")
    parser.add_argument("--evaluate-round", type=int,
                        help="read output as a verified round snapshot; requires evaluate-existing")
    parser.add_argument("--round-checkpoint", type=Path,
                        help="numerical checkpoint bound to the requested round snapshot")
    parser.add_argument("--runtime-operation-views", action="store_true")
    parser.add_argument("--runtime-operation-view-charts", type=int, default=0)
    parser.add_argument("--triadic-feature-schema", choices=("triple_product_v1", "joint_source_v2",
                                                           "joint_representation_v3"),
                        default="triple_product_v1")
    parser.add_argument("--preserve-coreferent-mentions", action="store_true")
    parser.add_argument("--operation-view-mode", action="append",
                        help="operation_views only: explicitly select candidate feature modes")
    parser.add_argument("--conditional-operation-labels", action="store_true",
                        help="operation_views only: learn meanings on operation spans; retain pointer boundary scores")
    parser.add_argument("--learn-span-pairs", action="store_true",
                        help="span_set_pointer only: fit existing boundary interactions against complete span sets")
    parser.add_argument("--background-log-odds", action="store_true")
    parser.add_argument("--runtime-mention-margin", action="store_true")
    parser.add_argument("--joint-operation-argument-scores", action="store_true")
    parser.add_argument("--conditional-argument-choices", action="store_true",
                        help="compare complete graphs with source-learned local categorical evidence")
    parser.add_argument("--objective", choices=("binary_proposals", "pairwise_arguments", "triadic_bindings", "graph_factors", "graph_relations", "joint_graphs", "operation_pointer", "argument_pointer", "definition_pointer", "operation_views", "paired_operation_pointer", "ranked_operation_pointer", "operation_background", "span_set_pointer", "labeled_spans"),
                        default="binary_proposals")
    parser.add_argument("--graph-rounds", type=int, default=3)
    parser.add_argument("--graph-update-steps", type=int, default=100)
    parser.add_argument("--source-operation-weight", type=float, default=1.,
                        help="source-label retention in joint_graphs; zero reproduces contrast-only fitting")
    parser.add_argument("--retain-semantic-constraints", action="store_true",
                        help="joint_graphs only: retain satisfied witnesses and source bindings during fitting")
    parser.add_argument("--acquire-training-errors", type=int, metavar="CONTROLS",
                        help="scan every training row, mine unresolved rows plus this many source-selected controls")
    parser.add_argument("--replay-source-graphs", action="store_true",
                        help="recheck every source-training decision after updates and retain witnessed regressions")
    parser.add_argument("--retention-operation-charts", type=int, default=32,
                        help="runtime operation charts searched per source-training example for retention")
    parser.add_argument("--learn-argument-heads", action="store_true",
                        help="retained joint_graphs only: differentiate runtime argument-mention heads too")
    parser.add_argument("--learn-operation-pointer", action="store_true",
                        help="retained joint_graphs only: differentiate runtime operation boundary scores")
    parser.add_argument("--graph-update-rule", choices=("working_face", "minimum_change"), default="working_face")
    parser.add_argument("--freeze-operation-head", action="store_true")
    parser.add_argument("--operation-retention-policy", choices=("supervised", "retain_existing"),
                        default="supervised", help="retain auxiliary span evidence without retargeting its labels")
    parser.add_argument("--operation-metric", choices=("coefficient_euclidean", "source_function"),
                        default="coefficient_euclidean", help="measure classifier changes on retained source features")
    parser.add_argument("--relation-metric", choices=("coefficient_euclidean", "factor_function"),
                        default="coefficient_euclidean")
    parser.add_argument("--boundary-policy", choices=("supervised", "retain_existing"), default="supervised")
    parser.add_argument("--compare-fit-start", action="store_true",
                        help="also evaluate the pre-fit candidate to separate decoder changes from learning")
    parser.add_argument("--relation-rank", type=int,
                        help="opt-in larger relation rank before a joint-graph fit; does not authorize serving")
    parser.add_argument("--relation-rank-seed", type=int, default=0)
    args = parser.parse_args()
    configure_refit_environment(args.output)
    if args.evaluate_existing and args.validation_output is None:
        parser.error("evaluate-existing requires validation-output")
    if ((args.evaluate_round is None) != (args.round_checkpoint is None)
            or (args.evaluate_round is not None and (not args.evaluate_existing or args.evaluate_round < 1))):
        parser.error("round evaluation requires evaluate-existing, a positive round, and round-checkpoint")
    if args.validation_output is not None:
        args.validation_checkpoint = args.validation_checkpoint or args.validation_output.with_suffix(".checkpoint.json")
    if args.validation_checkpoint is not None and args.validation_checkpoint.resolve() in {
        path.resolve() for path in (args.output, args.transducer, args.source_report, args.validation_output,
                                   args.round_checkpoint)
        if path is not None
    }:
        parser.error("validation checkpoint must not overwrite inputs or final outputs")
    if (args.round_checkpoint is not None and args.validation_output is not None
            and args.round_checkpoint.resolve() == args.validation_output.resolve()):
        parser.error("validation output must not overwrite the numerical checkpoint")
    if args.runtime_operation_views and args.objective not in {"pairwise_arguments", "triadic_bindings"}:
        parser.error("runtime operation views require pairwise_arguments or triadic_bindings")
    if args.triadic_feature_schema != "triple_product_v1" and args.objective != "triadic_bindings":
        parser.error("triadic feature schema requires triadic_bindings")
    if args.preserve_coreferent_mentions and args.objective != "pairwise_arguments":
        parser.error("coreferent mention preservation requires pairwise_arguments")
    if args.operation_view_mode and args.objective != "operation_views":
        parser.error("operation view candidates require operation_views")
    if args.conditional_operation_labels and args.objective != "operation_views":
        parser.error("conditional operation labels require operation_views")
    if args.learn_span_pairs and args.objective != "span_set_pointer":
        parser.error("span pair learning requires span_set_pointer")
    if args.background_log_odds and args.objective != "operation_background":
        parser.error("background log odds require operation_background")
    if args.runtime_mention_margin and args.objective != "pairwise_arguments":
        parser.error("runtime mention margin requires pairwise_arguments")
    if args.joint_operation_argument_scores and args.objective != "joint_graphs":
        parser.error("joint operation-argument scoring requires joint_graphs")
    if args.retain_semantic_constraints and (args.objective != "joint_graphs" or args.evaluate_existing):
        parser.error("semantic constraints require a fresh joint_graphs fit")
    if args.acquire_training_errors is not None and (
        args.acquire_training_errors < 1 or not args.retain_semantic_constraints
    ):
        parser.error("training-error acquisition requires retained joint graphs and a positive control count")
    if args.replay_source_graphs and not args.retain_semantic_constraints:
        parser.error("source graph replay requires retained semantic constraints")
    if args.learn_argument_heads and not args.retain_semantic_constraints:
        parser.error("argument-head learning requires retained semantic constraints")
    if args.learn_operation_pointer and not args.retain_semantic_constraints:
        parser.error("operation pointer learning requires retained semantic constraints")
    if args.graph_update_rule != "working_face" and not args.retain_semantic_constraints:
        parser.error("minimum-change updates require retained semantic constraints")
    if args.freeze_operation_head and not args.retain_semantic_constraints:
        parser.error("frozen operation heads require retained semantic constraints")
    if args.relation_metric != "coefficient_euclidean" and not args.retain_semantic_constraints:
        parser.error("functional relation geometry requires retained semantic constraints")
    if args.operation_retention_policy != "supervised" and not args.retain_semantic_constraints:
        parser.error("operation retention policy requires retained semantic constraints")
    if args.operation_metric != "coefficient_euclidean" and not args.retain_semantic_constraints:
        parser.error("operation function geometry requires retained semantic constraints")
    if args.boundary_policy != "supervised" and not args.learn_operation_pointer:
        parser.error("boundary policy requires operation pointer learning")
    if args.relation_rank is not None and args.objective != "joint_graphs":
        parser.error("relation rank expansion requires joint_graphs")
    if args.evaluate_existing and args.relation_rank is not None and not args.compare_fit_start:
        parser.error("replaying a rank-expanded start requires compare-fit-start")
    if (args.fit_checkpoint_dir is not None and not args.retain_semantic_constraints
            and args.objective != "span_set_pointer"):
        parser.error("fit checkpoints require retained constraints or span_set_pointer")
    if args.evaluate_existing and not args.compare_fit_start and (
        args.starting_candidate or args.joint_operation_argument_scores or args.conditional_argument_choices
    ):
        parser.error("evaluate-existing starting options require compare-fit-start")
    from core.learning.semantic_graph_margin import refit_compositional_graph_scales
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_labeled_span_learning import refit_compositional_labeled_spans
    from core.learning.semantic_operation_background import refit_compositional_operation_background
    from core.learning.semantic_operation_view_refit import refit_compositional_operation_views
    from core.learning.semantic_paired_pointer_refit import (
        refit_compositional_paired_operation_pointer,
    )
    from core.learning.semantic_program_compositional_campaign import (
        select_compositional_program_candidate,
    )
    from core.learning.semantic_program_compositional_refits import (
        refit_compositional_triadic_bindings,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        refit_compositional_argument_proposals,
        refit_compositional_argument_rankings,
        refit_compositional_definition_pointer,
        refit_compositional_operation_pointer,
    )
    from core.learning.semantic_relation_graph_learning import refit_compositional_graph_relations
    from core.learning.semantic_span_set_learning import refit_compositional_span_set_pointer
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
    if args.conditional_argument_choices:
        starting = starting.with_conditional_argument_choices()
    if args.relation_rank is not None:
        starting = starting.with_expanded_relation_rank(args.relation_rank, seed=args.relation_rank_seed)
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
        "operation_background": refit_compositional_operation_background,
        "paired_operation_pointer": refit_compositional_paired_operation_pointer,
        "ranked_operation_pointer": refit_compositional_paired_operation_pointer,
        "span_set_pointer": refit_compositional_span_set_pointer,
        "labeled_spans": refit_compositional_labeled_spans,
        "triadic_bindings": refit_compositional_triadic_bindings,
    }[args.objective]
    options = {"refit_pointer": True} if args.objective == "argument_pointer" else {}
    if args.objective == "triadic_bindings":
        options["feature_schema"] = args.triadic_feature_schema
    if args.objective == "span_set_pointer":
        options["learn_pair"] = args.learn_span_pairs
        options["checkpoint_path"] = (
            args.fit_checkpoint_dir or args.output.with_suffix(".fit-checkpoints")
        ) / "span-set.npz"
    if args.objective == "operation_views":
        options["candidate_modes"] = args.operation_view_mode
        options["conditional_labels"] = args.conditional_operation_labels
        options["progress"] = lambda row: print(json.dumps(row, sort_keys=True), flush=True)
    if args.objective in {"graph_factors", "graph_relations", "joint_graphs", "operation_background", "span_set_pointer", "labeled_spans"}:
        options["progress"] = lambda row: print(json.dumps(row, sort_keys=True), flush=True)
    if args.objective in {"graph_relations", "joint_graphs"}:
        options.update(rounds=args.graph_rounds, steps=args.graph_update_steps)
    if args.objective == "joint_graphs":
        options["source_weight"] = args.source_operation_weight
        options["constraint_learning"] = args.retain_semantic_constraints
        options["source_graph_retention"] = args.replay_source_graphs
        options["learn_arguments"] = args.learn_argument_heads
        options["learn_operation_pointer"] = args.learn_operation_pointer
        options["update_rule"] = args.graph_update_rule
        options["learn_operations"] = not args.freeze_operation_head
        options["relation_metric"] = args.relation_metric
        options["boundary_policy"] = args.boundary_policy
        options["operation_policy"] = args.operation_retention_policy
        options["operation_metric"] = args.operation_metric
        if args.retain_semantic_constraints:
            options["checkpoint_dir"] = args.fit_checkpoint_dir or args.output.with_suffix(".fit-checkpoints")
            options["retention_operation_charts"] = args.retention_operation_charts
    if args.runtime_mention_margin:
        options["runtime_mention_margin"] = True
    if args.preserve_coreferent_mentions:
        options["preserve_coreferent_mentions"] = True
    if args.runtime_operation_views:
        if args.objective == "pairwise_arguments":
            options["use_runtime_operation_views"] = True
        options["runtime_operation_view_charts"] = args.runtime_operation_view_charts
        options["progress"] = lambda row: print(json.dumps(row, sort_keys=True), flush=True)
    elif args.runtime_operation_view_charts:
        parser.error("--runtime-operation-view-charts requires --runtime-operation-views")
    if args.objective == "ranked_operation_pointer":
        options = {"ranking": True}
    if args.objective == "operation_background":
        options["background_log_odds"] = args.background_log_odds
    if args.evaluate_existing:
        candidate = load_evaluation_candidate(args.output, starting,
            round_index=args.evaluate_round, numerical_checkpoint=args.round_checkpoint)
        verify_source_splits(bound, candidate.training_receipt)
        if candidate.model_basis_sha256 != model.model_basis_sha256:
            raise ValueError("saved candidate representation differs from incumbent")
        if args.compare_fit_start and args.evaluate_round is None:
            verify_fit_start(candidate, starting)
    else:
        mining = bound
        acquisition = None
        if args.acquire_training_errors is not None:
            from core.learning.semantic_graph_trial import acquire_semantic_training_errors

            selected, acquisition = acquire_semantic_training_errors(
                starting, bound, control_count=args.acquire_training_errors,
                progress=options["progress"],
            )
            mining = (*selected, *(item for item in bound if item.split == "validation"))
            options["source_retention_examples"] = tuple(item for item in bound if item.split == "train")
        candidate = refit(starting, mining, **options)
        if acquisition is not None:
            from dataclasses import replace

            from core.learning.semantic_program_campaign import _sha

            if candidate.training_receipt["joint_graph_refit"]["training_example_ids_sha256"] != acquisition["training_example_ids_sha256"]:
                raise ValueError("fit source identities differ from acquired training errors")
            receipt = {key: value for key, value in candidate.training_receipt.items() if key != "receipt_sha256"}
            receipt["training_error_acquisition"] = acquisition
            candidate = replace(candidate, training_receipt={**receipt, "receipt_sha256": _sha(receipt)})
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
            scoring=args.validation_scoring,
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
