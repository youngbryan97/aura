"""Leave-family-out diagnosis for the compositional semantic transducer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.semantic_input_grounding import SemanticInputGroundingContract
from core.learning.semantic_program_basis import (
    bind_training_examples_to_shared_representation,
    establish_semantic_training_representation_compatibility,
)
from core.learning.semantic_program_campaign import (
    _sha,
    training_examples_from_feature_bundle,
)
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
    _definition_span_candidates,
    _register_definition_spans,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_shared_evaluation import (
    evaluate_shared_semantic_program_transducer,
)
from core.learning.semantic_program_shared_transducer import _relation_span_vector
from core.learning.semantic_program_transducer import SemanticTransducerTrainingExample

COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA: Final = (
    "aura.semantic_program_compositional_leave_family_out.v1"
)


@dataclass(frozen=True, slots=True)
class CompositionalLeaveFamilyOutResult:
    model: CompositionalSemanticProgramTransducer
    report: dict[str, Any]


def diagnose_compositional_definition_relations(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> dict[str, Any]:
    """Measure the relation head with gold references but no gold answers."""

    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        selected = tuple(item for item in examples if item.split == split)
        if not selected:
            raise ValueError(f"compositional relation diagnostic split is empty: {split}")
        total = 0
        runtime_top1 = 0
        oracle_top1 = 0
        by_construction: dict[str, dict[str, int]] = {}
        by_slot: dict[str, dict[str, int]] = {}
        for item in selected:
            oracle_definitions = _register_definition_spans(item)
            runtime_anchors = (
                *item.ir.input_spans,
                *(instruction.operation_span for instruction in item.ir.instructions),
            )
            definition_pointer_scores = model.definition_pointer.score_sequence(
                item.hidden_states
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
                    for candidate in _definition_span_candidates(
                        anchor,
                        token_count=item.hidden_states.shape[0],
                        max_span_tokens=model.max_definition_span_tokens,
                    )
                )
                for anchor in runtime_anchors
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
        }
    body = {
        "schema": "aura.semantic_program_definition_relation_diagnostic.v1",
        "transducer_receipt_sha256": model.receipt_sha256,
        "gold_reference_spans_available": True,
        "gold_definition_spans_available_to_oracle_arm": True,
        "gold_definition_spans_available_to_runtime_arm": False,
        "expected_answers_available": False,
        "serving_authority": False,
        "splits": by_split,
    }
    return {**body, "report_sha256": _sha(body)}


def _family_report(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> dict[str, Any]:
    arms = {
        split: evaluate_shared_semantic_program_transducer(
            model,
            examples,
            split=split,
        ).to_dict()
        for split in ("train", "validation", "test")
    }
    return {
        "example_count": len(examples),
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
) -> dict[str, Any]:
    """Replay frozen causal arms without fitting or changing the task set."""

    arms = {
        "treatment": model,
        "chart_beam_lesion": model.chart_beam_lesion(),
        "register_use_lesion": model.register_use_lesion(),
        "relation_lesion": model.relation_lesion(),
        "dependency_lesion": model.dependency_lesion(),
        "coefficient_lesion": model.coefficient_lesion(),
    }
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
        "example_ids_sha256": _sha(
            sorted(item.ir.source_text_sha256 for item in examples)
        ),
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
) -> CompositionalLeaveFamilyOutResult:
    """Fit without one named family and measure transfer to every family."""

    if held_out_family not in bundles:
        raise ValueError("compositional held-out family is absent")
    if len(bundles) < 3:
        raise ValueError("compositional held-family diagnosis needs at least three families")
    examples_by_family = {
        family: training_examples_from_feature_bundle(bundle)
        for family, bundle in bundles.items()
    }
    manifests = {family: bundle.manifest for family, bundle in bundles.items()}
    compatibility = establish_semantic_training_representation_compatibility(manifests)
    bound = bind_training_examples_to_shared_representation(
        examples_by_family,
        compatibility=compatibility,
    )
    bound_by_family = {
        family: tuple(
            item
            for item in bound
            if item.construction_id.startswith(f"{family}:")
        )
        for family in bundles
    }
    if any(
        len(bound_by_family[family]) != len(examples_by_family[family])
        for family in bundles
    ):
        raise ValueError("compositional held-family inventory changed during binding")
    fit_examples = tuple(
        item
        for family, examples in bound_by_family.items()
        if family != held_out_family
        for item in examples
    )
    model = fit_compositional_semantic_program_transducer(
        fit_examples,
        input_grounding=input_grounding,
    )
    families = {
        family: _family_report(model, examples)
        for family, examples in sorted(bound_by_family.items())
    }
    body = {
        "schema": COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA,
        "held_out_family": held_out_family,
        "fit_families": sorted(set(bundles) - {held_out_family}),
        "feature_manifest_sha256s": {
            family: manifests[family]["manifest_sha256"] for family in sorted(manifests)
        },
        "representation_compatibility": compatibility,
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
    "COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA",
    "CompositionalLeaveFamilyOutResult",
    "diagnose_compositional_definition_relations",
    "diagnose_compositional_transfer_lesions",
    "run_compositional_leave_family_out_campaign",
]
