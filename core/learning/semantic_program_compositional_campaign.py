"""Leave-family-out diagnosis for the compositional semantic transducer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from core.learning.semantic_input_grounding import SemanticInputGroundingContract
from core.learning.semantic_program_basis import (
    bind_examples_to_compatible_training_session,
    bind_training_examples_to_shared_representation,
    establish_semantic_representation_compatibility,
    establish_semantic_training_representation_compatibility,
)
from core.learning.semantic_program_campaign import (
    _sha,
    training_examples_from_feature_bundle,
)
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
    _register_definition_candidates,
    _register_definition_spans,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
from core.learning.semantic_program_shared_evaluation import (
    evaluate_shared_semantic_program_transducer,
)
from core.learning.semantic_program_shared_transducer import _relation_span_vector
from core.learning.semantic_program_transducer import SemanticTransducerTrainingExample
from core.learning.semantic_source_order import source_order_training_example

COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA: Final = (
    "aura.semantic_program_compositional_leave_family_out.v1"
)
COMPOSITIONAL_LESION_ARMS: Final = (
    "treatment",
    "chart_beam_lesion",
    "register_use_lesion",
    "relation_tissue_lesion",
    "argument_proposal_lesion",
    "relation_lesion",
    "dependency_lesion",
    "coefficient_lesion",
)


def select_compositional_program_candidate(
    candidates: Mapping[str, CompositionalSemanticProgramTransducer],
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    incumbent: str,
    checkpoint_path: Any=None,
    progress: Any=None,
    scoring: str='register_indices_v1',
) -> dict[str, Any]:
    """Select on autonomous validation programs, never gold answers or test tasks."""
    from core.learning.semantic_validation_checkpoint import (
        VALIDATION_SCORING,
        validation_implementation_identity,
    )
    if scoring not in VALIDATION_SCORING:
        raise ValueError("unknown semantic validation scoring")
    if incumbent not in candidates:
        raise ValueError("program selection needs a named incumbent")
    selected = tuple(item for item in examples if item.split == "validation")
    ids = [item.ir.source_text_sha256 for item in selected]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("program selection needs unique validation examples")
    if any(
        item.ir.source_text_sha256 in set(ids)
        for item in examples
        if item.split != "validation"
    ):
        raise ValueError("program selection validation overlaps another split")
    checkpoint = None
    implementation = validation_implementation_identity()
    if checkpoint_path is not None:
        from core.learning.semantic_validation_checkpoint import (
            SemanticValidationCheckpoint,
            validation_identity,
        )

        checkpoint = SemanticValidationCheckpoint(
            checkpoint_path, validation_identity(candidates, selected, scoring=scoring,
                implementation=implementation), candidates, ids,
        )
    outcomes = {}
    equivalents = {}
    for name, model in candidates.items():
        rows = []
        equivalent_rows = []
        for item in selected:
            source = item.ir.source_text_sha256
            cached = checkpoint.get(name, source) if checkpoint is not None else None
            if progress is not None:
                progress({"stage": "semantic_validation", "candidate": name,
                          "completed": len(rows), "total": len(selected),
                          "source": source, "cached": cached is not None})
            if cached is not None:
                rows.append(cached[0])
                equivalent_rows.append(cached[1])
                continue
            outcome = model.decode(
                source_token_ids=item.ir.source_token_ids,
                hidden_states=item.hidden_states,
                public_inputs=item.public_inputs,
                source_text_sha256=item.ir.source_text_sha256,
                model_basis_sha256=item.ir.model_basis_receipt_sha256,
            )
            target = item.ir.to_program()
            grounding_valid = True
            if outcome.ir is not None and scoring == "source_anchors_v2":
                from core.learning.procedure_induction import Instruction, Program
                from core.learning.semantic_joint_graph_learning import align_source_input_registers
                try:
                    instructions, _ = align_source_input_registers(item, outcome.ir.input_spans)
                    target = Program(len(item.public_inputs), tuple(
                        Instruction(instruction.op, instruction.args) for instruction in instructions))
                # not a failure: registers that will not align give no target program to compare.
                except ValueError:
                    grounding_valid = False
            exact = bool(outcome.ir is not None and grounding_valid and outcome.ir.to_program() == target)
            rows.append(exact)
            equivalent_rows.append(exact or bool(
                outcome.ir is not None and grounding_valid and semantic_programs_structurally_equivalent(
                    outcome.ir.to_program(), target
                )
            ))
            if checkpoint is not None:
                checkpoint.record(name, source, exact, equivalent_rows[-1])
        if progress is not None:
            progress({"stage": "semantic_validation", "candidate": name,
                      "completed": len(rows), "total": len(selected)})
        outcomes[name] = rows
        equivalents[name] = equivalent_rows
    baseline = outcomes[incumbent]
    summaries = {
        name: {
            "program_exact": sum(rows),
            "gains": sum(new and not old for new, old in zip(rows, baseline, strict=True)),
            "regressions": sum(old and not new for new, old in zip(rows, baseline, strict=True)),
            "program_correct": rows,
            "program_equivalent": sum(equivalents[name]),
            "program_equivalent_correct": equivalents[name],
            "equivalent_gains": sum(new and not old for new, old in zip(
                equivalents[name], equivalents[incumbent], strict=True
            )),
            "equivalent_regressions": sum(old and not new for new, old in zip(
                equivalents[name], equivalents[incumbent], strict=True
            )),
            "transducer_receipt_sha256": candidates[name].receipt_sha256,
        }
        for name, rows in outcomes.items()
    }
    improving = [
        name for name, result in summaries.items()
        if result["gains"] > 0 and result["regressions"] == 0
    ]
    winner = min(
        improving, key=lambda name: (-summaries[name]["program_exact"], name)
    ) if improving else incumbent
    if validation_implementation_identity() != implementation:
        raise ValueError("semantic validation implementation changed during evaluation")
    body = {
        "schema": "aura.semantic_program_validation_selection.v1",
        "objective": "exact_program_gain_without_validation_regression",
        "incumbent": incumbent,
        "selected": winner,
        "validation_examples": len(selected),
        "validation_example_ids": ids,
        "candidates": summaries,
        "expected_answers_available": False,
        "gold_program_available_to_decode": False,
        "test_examples_used": 0,
        "equivalence_rule": "connected_graph_schedule_and_integer_add_mul_exchange_v1",
        "equivalence_used_for_selection": False,
        "serving_authority": False,
        "implementation_source_sha256": implementation,
    }
    if scoring == "source_anchors_v2":
        body.update(schema="aura.semantic_program_validation_selection.v2",
                    scoring=scoring,
                    objective="source_anchored_exact_program_gain_without_validation_regression",
                    equivalence_rule="source_anchors_and_connected_graph_schedule_and_integer_add_mul_exchange_v2")
    return {**body, "report_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class CompositionalLeaveFamilyOutResult:
    model: CompositionalSemanticProgramTransducer
    report: dict[str, Any]


def prepare_compositional_source_training(
    bundles: Any, *, source_order_inputs: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Bind exact source cohorts, admitting training-only augmentation without test access."""
    manifests = {name: bundle.manifest for name, bundle in bundles.items()}
    compatibility = establish_semantic_training_representation_compatibility(manifests)
    examples = {name: training_examples_from_feature_bundle(bundle, required_splits=frozenset({"train"}))
                for name, bundle in bundles.items()}
    bound = bind_training_examples_to_shared_representation(examples, compatibility=compatibility)
    if source_order_inputs:
        bound = tuple(source_order_training_example(item) for item in bound)
    ids = [item.ir.source_text_sha256 for item in bound]
    if len(ids) != len(set(ids)):
        raise ValueError("compositional source examples repeat across cohorts or splits")
    selected = tuple(item for item in bound if item.split in {"train", "validation"})
    if {item.split for item in selected} != {"train", "validation"}:
        raise ValueError("compositional source training needs train and validation")
    body = {
        "schema": ("aura.compositional_source_training_plan.v2" if source_order_inputs
                   else "aura.compositional_source_training_plan.v1"),
        "representation_compatibility": compatibility,
        "cohort_split_counts": {name: {split: sum(item.split == split for item in items)
            for split in ("train", "validation", "test")} for name, items in examples.items()},
        "training_example_count": sum(item.split == "train" for item in selected),
        "validation_example_count": sum(item.split == "validation" for item in selected),
        "training_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in selected if item.split == "train")),
        "validation_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in selected if item.split == "validation")),
        "test_examples_available_to_fit": 0,
        "serving_authority": False,
    }
    if source_order_inputs:
        body["input_order_policy"] = "source_token_order_v1"
    return selected, {**body, "report_sha256": _sha(body)}


def fit_compositional_source_campaign(
    bundles: Any,
    *,
    input_grounding: Any,
    progress: Any=None,
    source_order_inputs: bool = False,
) -> Any:
    """Fit new coefficients on a measured representation, never relabel an older head."""
    examples, report = prepare_compositional_source_training(
        bundles, source_order_inputs=source_order_inputs)
    if progress:
        progress({"stage": "source_fit_start", "training_examples": report["training_example_count"],
                  "validation_examples": report["validation_example_count"]})
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=input_grounding)
    model = (model.with_global_constraint_arguments().with_conditional_argument_scores()
             .with_overlap_complete_mentions().with_atomic_literal_arguments()
             .with_feasible_operation_charts().with_order_invariant_argument_graph()
             .with_joint_definition_graph().with_categorical_relation_scores())
    if source_order_inputs:
        receipt_body = {key: value for key, value in model.training_receipt.items()
                        if key != "receipt_sha256"}
        receipt_body["input_order_policy"] = "source_token_order_v1"
        model = replace(model, training_receipt={**receipt_body,
                        "receipt_sha256": _sha(receipt_body)})
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    body.update(schema=("aura.compositional_source_training.v2" if source_order_inputs
                        else "aura.compositional_source_training.v1"),
                transducer_receipt_sha256=model.receipt_sha256,
                decoder_recipe="global_joint_categorical_atomic_v1", inherited_coefficients=False,
                fit_complete=True, evaluation_complete=False)
    return CompositionalLeaveFamilyOutResult(model, {**body, "report_sha256": _sha(body)})


def diagnose_compositional_definition_relations(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> dict[str, Any]:
    """Measure the relation head with gold references but no gold answers."""

    available_splits = tuple(
        split
        for split in ("train", "validation", "test")
        if any(item.split == split for item in examples)
    )
    if not available_splits:
        raise ValueError("compositional relation diagnostic has no observed split")
    by_split: dict[str, dict[str, Any]] = {}
    for split in available_splits:
        selected = tuple(item for item in examples if item.split == split)
        total = 0
        runtime_top1 = 0
        oracle_top1 = 0
        by_construction: dict[str, dict[str, int]] = {}
        by_slot: dict[str, dict[str, int]] = {}
        definition_origins: dict[str, int] = {}
        for item in selected:
            origin = item.register_definition_origin
            definition_origins[origin] = definition_origins.get(origin, 0) + 1
            oracle_definitions = _register_definition_spans(item)
            runtime_anchors = (
                *item.ir.input_spans,
                *(instruction.operation_span for instruction in item.ir.instructions),
            )
            definition_pointer_scores = model.definition_pointer.score_sequence(item.hidden_states)
            runtime_candidate_spans = _register_definition_candidates(
                runtime_anchors,
                input_count=item.ir.n_inputs,
                token_count=item.hidden_states.shape[0],
                max_span_tokens=model.max_definition_span_tokens,
                pointer_scores=definition_pointer_scores,
                strategy=model.definition_candidate_strategy,
            )
            runtime_definitions = tuple(
                tuple(
                    (
                        candidate,
                        _relation_span_vector(
                            item.hidden_states,
                            candidate,
                            hidden_channels=model.hidden_channels,
                            hidden_channel_widths=model.hidden_channel_widths,
                        ),
                    )
                    for candidate in candidates
                )
                for candidates in runtime_candidate_spans
            )
            oracle_vectors = tuple(
                (
                    span,
                    _relation_span_vector(
                        item.hidden_states,
                        span,
                        hidden_channels=model.hidden_channels,
                        hidden_channel_widths=model.hidden_channel_widths,
                    ),
                )
                for span in oracle_definitions
            )
            counts = by_construction.setdefault(
                item.construction_id,
                {"total": 0, "runtime_top1": 0, "oracle_top1": 0},
            )
            for step, instruction in enumerate(item.ir.instructions):
                available = item.ir.n_inputs + step
                for position, (reference_span, expected_register) in enumerate(
                    zip(
                        instruction.argument_spans,
                        instruction.args,
                        strict=True,
                    )
                ):
                    reference = _relation_span_vector(
                        item.hidden_states,
                        reference_span,
                        hidden_channels=model.hidden_channels,
                        hidden_channel_widths=model.hidden_channel_widths,
                    )
                    runtime_scores = tuple(
                        max(
                            model.definition_relation_head.score(reference, definition)
                            + model.definition_relation_head.pointer_scale
                            * definition_pointer_scores.score_span(span)
                            for span, definition in candidates
                        )
                        for candidates in runtime_definitions[:available]
                    )
                    oracle_scores = tuple(
                        model.definition_relation_head.score(reference, definition)
                        + model.definition_relation_head.pointer_scale
                        * definition_pointer_scores.score_span(span)
                        for span, definition in oracle_vectors[:available]
                    )
                    runtime_correct = int(
                        max(range(available), key=lambda index: runtime_scores[index])
                        == expected_register
                    )
                    oracle_correct = int(
                        max(range(available), key=lambda index: oracle_scores[index])
                        == expected_register
                    )
                    total += 1
                    runtime_top1 += runtime_correct
                    oracle_top1 += oracle_correct
                    counts["total"] += 1
                    counts["runtime_top1"] += runtime_correct
                    counts["oracle_top1"] += oracle_correct
                    slot = f"step:{step}|position:{position}|register:{expected_register}"
                    slot_counts = by_slot.setdefault(
                        slot,
                        {"total": 0, "runtime_top1": 0, "oracle_top1": 0},
                    )
                    slot_counts["total"] += 1
                    slot_counts["runtime_top1"] += runtime_correct
                    slot_counts["oracle_top1"] += oracle_correct
        by_split[split] = {
            "total": total,
            "runtime_top1": runtime_top1,
            "oracle_top1": oracle_top1,
            "by_construction": dict(sorted(by_construction.items())),
            "by_slot": dict(sorted(by_slot.items())),
            "definition_origin_examples": dict(sorted(definition_origins.items())),
        }
    body = {
        "schema": "aura.semantic_program_definition_relation_diagnostic.v2",
        "transducer_receipt_sha256": model.receipt_sha256,
        "gold_reference_spans_available": True,
        "gold_definition_spans_available_to_oracle_arm": all(
            item.register_definition_origin == "explicit_annotation" for item in examples
        ),
        "definition_targets_available_to_oracle_arm": True,
        "gold_definition_spans_available_to_runtime_arm": False,
        "expected_answers_available": False,
        "serving_authority": False,
        "evaluated_splits": list(available_splits),
        "splits": by_split,
    }
    return {**body, "report_sha256": _sha(body)}


def _family_report(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> dict[str, Any]:
    available_splits = tuple(
        split
        for split in ("train", "validation", "test")
        if any(item.split == split for item in examples)
    )
    if not {"validation", "test"} <= set(available_splits):
        raise ValueError("compositional family report needs validation and test examples")
    arms = {
        split: evaluate_shared_semantic_program_transducer(
            model,
            examples,
            split=split,
        ).to_dict()
        for split in available_splits
    }
    return {
        "example_count": len(examples),
        "evaluated_splits": list(available_splits),
        "splits": arms,
        "held_out_program_exact": sum(
            arms[split]["program_exact"] for split in ("validation", "test")
        ),
        "held_out_argument_exact": sum(
            arms[split]["argument_exact"] for split in ("validation", "test")
        ),
        "held_out_answer_exact": sum(
            arms[split]["answer_exact"] for split in ("validation", "test")
        ),
        "held_out_total": sum(arms[split]["total"] for split in ("validation", "test")),
    }


def diagnose_compositional_transfer_lesions(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    arm_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Replay frozen causal arms without fitting or changing the task set."""

    available_arms = {
        "treatment": model,
        "chart_beam_lesion": model.chart_beam_lesion(),
        "register_use_lesion": model.register_use_lesion(),
        "relation_tissue_lesion": model.relation_tissue_lesion(),
        "argument_proposal_lesion": model.argument_proposal_lesion(),
        "relation_lesion": model.relation_lesion(),
        "dependency_lesion": model.dependency_lesion(),
        "coefficient_lesion": model.coefficient_lesion(),
    }
    selected_arm_names = COMPOSITIONAL_LESION_ARMS if arm_names is None else tuple(arm_names)
    if not selected_arm_names:
        raise ValueError("compositional lesion arm selection is empty")
    if len(set(selected_arm_names)) != len(selected_arm_names):
        raise ValueError("compositional lesion arm selection contains duplicates")
    unsupported = sorted(set(selected_arm_names) - set(COMPOSITIONAL_LESION_ARMS))
    if unsupported:
        raise ValueError(f"unsupported compositional lesion arms: {unsupported}")
    if "treatment" not in selected_arm_names:
        raise ValueError("compositional lesion evaluation requires treatment")
    arms = {name: available_arms[name] for name in selected_arm_names}
    results = {
        name: {
            split: evaluate_shared_semantic_program_transducer(
                arm,
                examples,
                split=split,
                arm=name,
            ).to_dict()
            for split in ("validation", "test")
        }
        for name, arm in arms.items()
    }
    body = {
        "schema": "aura.semantic_program_compositional_lesions.v1",
        "transducer_receipt_sha256": model.receipt_sha256,
        "example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in examples)),
        "evaluated_arms": list(selected_arm_names),
        "arms": results,
        "fit_or_refit_calls": 0,
        "expected_answers_available_to_decode": False,
        "serving_authority": False,
    }
    return {**body, "report_sha256": _sha(body)}


def run_compositional_leave_family_out_campaign(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
    *,
    held_out_family: str,
    input_grounding: SemanticInputGroundingContract,
    evaluation_families: Sequence[str] | None = None,
) -> CompositionalLeaveFamilyOutResult:
    """Fit without one named family and measure transfer to every family."""

    if held_out_family not in bundles:
        raise ValueError("compositional held-out family is absent")
    if len(bundles) < 3:
        raise ValueError("compositional held-family diagnosis needs at least three families")
    examples_by_family = {
        family: training_examples_from_feature_bundle(
            bundle,
            required_splits=(
                frozenset({"validation", "test"})
                if family == held_out_family
                else frozenset({"train", "validation", "test"})
            ),
        )
        for family, bundle in bundles.items()
    }
    manifests = {family: bundle.manifest for family, bundle in bundles.items()}
    fit_families = sorted(set(bundles) - {held_out_family})
    fit_manifests = {family: manifests[family] for family in fit_families}
    fit_compatibility = establish_semantic_training_representation_compatibility(fit_manifests)
    fit_bound = bind_training_examples_to_shared_representation(
        {family: examples_by_family[family] for family in fit_families},
        compatibility=fit_compatibility,
    )
    fit_bound_by_family = {
        family: tuple(item for item in fit_bound if item.construction_id.startswith(f"{family}:"))
        for family in fit_families
    }
    if any(
        len(fit_bound_by_family[family]) != len(examples_by_family[family])
        for family in fit_families
    ):
        raise ValueError("compositional fit-family inventory changed during binding")
    fit_examples = tuple(item for family in fit_families for item in fit_bound_by_family[family])
    model = fit_compositional_semantic_program_transducer(
        fit_examples,
        input_grounding=input_grounding,
    )
    target_basis = fit_compatibility["target_training_session_basis_sha256"]
    anchor_families = [
        family
        for family in fit_families
        if target_basis in fit_compatibility["source_session_basis_sha256s"][family]
    ]
    if len(anchor_families) != 1 or model.model_basis_sha256 != target_basis:
        raise ValueError("compositional fit basis has no unique source cohort")
    held_out_compatibility = establish_semantic_representation_compatibility(
        model=model,
        training_manifest=manifests[anchor_families[0]],
        replication_manifest=manifests[held_out_family],
    )
    held_out_examples = bind_examples_to_compatible_training_session(
        examples_by_family[held_out_family],
        compatibility=held_out_compatibility,
    )
    bound_by_family = {
        **fit_bound_by_family,
        held_out_family: tuple(
            replace(
                item,
                construction_id=f"{held_out_family}:{item.construction_id}",
                topology_id=f"{held_out_family}:{item.topology_id}",
            )
            for item in held_out_examples
        ),
    }
    evaluated_families = (
        tuple(sorted(bound_by_family))
        if evaluation_families is None
        else tuple(dict.fromkeys(evaluation_families))
    )
    if (
        not evaluated_families
        or held_out_family not in evaluated_families
        or not set(evaluated_families) <= set(bound_by_family)
    ):
        raise ValueError(
            "compositional evaluation families must be known and include the held-out family"
        )
    families = {
        family: _family_report(model, bound_by_family[family]) for family in evaluated_families
    }
    body = {
        "schema": COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA,
        "held_out_family": held_out_family,
        "fit_families": fit_families,
        "evaluated_families": list(evaluated_families),
        "feature_manifest_sha256s": {
            family: manifests[family]["manifest_sha256"] for family in sorted(manifests)
        },
        "representation_compatibility": fit_compatibility,
        "held_out_representation_compatibility": held_out_compatibility,
        "model_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "fit_example_count": len(fit_examples),
        "families": families,
        "held_out_family_was_available_to_fit": False,
        "expected_answers_available_to_training": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
        "claim_boundary": (
            "diagnostic leave-family-out semantic-program transfer; no serving or "
            "broad-domain authority"
        ),
    }
    return CompositionalLeaveFamilyOutResult(
        model=model,
        report={**body, "report_sha256": _sha(body)},
    )


__all__ = [
    "COMPOSITIONAL_LESION_ARMS",
    "COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA",
    "CompositionalLeaveFamilyOutResult",
    "diagnose_compositional_definition_relations",
    "diagnose_compositional_transfer_lesions",
    "run_compositional_leave_family_out_campaign",
    "select_compositional_program_candidate",
]
