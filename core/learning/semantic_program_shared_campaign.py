"""Auditable variable-geometry semantic transfer across named families."""

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
    _paired_control,
    _sha,
    training_examples_from_feature_bundle,
)
from core.learning.semantic_program_evaluation import shuffle_hidden_tokens
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_shared_evaluation import (
    evaluate_shared_semantic_program_transducer,
)
from core.learning.semantic_program_shared_transducer import (
    SharedSemanticProgramTransducer,
    fit_shared_semantic_program_transducer,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
)

SEMANTIC_PROGRAM_SHARED_CAMPAIGN_SCHEMA: Final = (
    "aura.semantic_program_shared_campaign.v1"
)
_EVALUATION_SPLITS: Final = ("validation", "test")
_CONTROL_ARMS: Final = ("hidden_token_shuffle", "coefficient_lesion")


@dataclass(frozen=True, slots=True)
class SharedSemanticProgramCampaignResult:
    model: SharedSemanticProgramTransducer
    report: dict[str, Any]


def _family_examples(
    examples: Sequence[SemanticTransducerTrainingExample],
    family: str,
) -> tuple[SemanticTransducerTrainingExample, ...]:
    prefix = f"{family}:"
    return tuple(item for item in examples if item.construction_id.startswith(prefix))


def _subset_arm(arm: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    prefix = f"{family}:"
    rows = tuple(
        row for row in arm["rows"] if str(row["construction_id"]).startswith(prefix)
    )
    if not rows:
        raise ValueError(f"shared semantic campaign arm has no family rows: {family}")
    metrics = (
        "accepted",
        "geometry_exact",
        "step_count_exact",
        "arity_exact",
        "program_exact",
        "operation_exact",
        "argument_exact",
        "input_span_exact",
        "answer_emitted",
        "answer_exact",
    )
    by_geometry: dict[str, dict[str, int]] = {}
    for row in rows:
        geometry = str(row["geometry"])
        counts = by_geometry.setdefault(
            geometry,
            {"total": 0, **{metric: 0 for metric in metrics}},
        )
        counts["total"] += 1
        for metric in metrics:
            counts[metric] += int(bool(row[metric]))
    return {
        "arm": arm["arm"],
        "split": arm["split"],
        "total": len(rows),
        **{
            metric: sum(int(bool(row[metric])) for row in rows)
            for metric in metrics
        },
        "by_geometry": dict(sorted(by_geometry.items())),
        "geometry_macro_program_accuracy": sum(
            counts["program_exact"] / counts["total"]
            for counts in by_geometry.values()
        )
        / len(by_geometry),
        "geometry_macro_answer_accuracy": sum(
            counts["answer_exact"] / counts["total"]
            for counts in by_geometry.values()
        )
        / len(by_geometry),
        "rows": list(rows),
    }


def _evaluate_arms(
    *,
    model: SharedSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> dict[str, dict[str, Any]]:
    lesion = model.coefficient_lesion()
    arms: dict[str, dict[str, Any]] = {}
    for split in ("train", *_EVALUATION_SPLITS):
        arms[f"treatment:{split}"] = evaluate_shared_semantic_program_transducer(
            model,
            examples,
            split=split,
        ).to_dict()
    for split in _EVALUATION_SPLITS:
        arms[f"hidden_token_shuffle:{split}"] = (
            evaluate_shared_semantic_program_transducer(
                model,
                examples,
                split=split,
                arm="hidden_token_shuffle",
                hidden_transform=shuffle_hidden_tokens,
            ).to_dict()
        )
        arms[f"coefficient_lesion:{split}"] = (
            evaluate_shared_semantic_program_transducer(
                lesion,
                examples,
                split=split,
                arm="coefficient_lesion",
            ).to_dict()
        )
    return arms


def _control_report(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    programs: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    for split in _EVALUATION_SPLITS:
        treatment = arms[f"treatment:{split}"]["rows"]
        for control in _CONTROL_ARMS:
            rows = arms[f"{control}:{split}"]["rows"]
            programs[f"{control}:{split}"] = _paired_control(
                treatment,
                rows,
                metric="program_exact",
            )
            answers[f"{control}:{split}"] = _paired_control(
                treatment,
                rows,
                metric="answer_exact",
            )
    return {
        "paired_program_controls": programs,
        "paired_answer_controls": answers,
    }


def run_shared_semantic_program_campaign_from_examples(
    examples_by_family: Mapping[str, Sequence[SemanticTransducerTrainingExample]],
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    input_grounding: SemanticInputGroundingContract,
) -> SharedSemanticProgramCampaignResult:
    """Fit one variable-geometry cortex and run matched held-out controls."""

    if set(examples_by_family) != set(manifests):
        raise ValueError("shared semantic campaign family manifests differ")
    compatibility = establish_semantic_training_representation_compatibility(manifests)
    examples = bind_training_examples_to_shared_representation(
        examples_by_family,
        compatibility=compatibility,
    )
    model = fit_shared_semantic_program_transducer(
        examples,
        input_grounding=input_grounding,
    )
    arms = _evaluate_arms(model=model, examples=examples)
    family_reports: dict[str, dict[str, Any]] = {}
    for family in sorted(examples_by_family):
        selected = _family_examples(examples, family)
        if len(selected) != len(examples_by_family[family]):
            raise ValueError(f"shared semantic campaign family inventory changed: {family}")
        family_arms = {
            name: _subset_arm(arm, family=family) for name, arm in arms.items()
        }
        family_reports[family] = {
            "example_count": len(selected),
            "arms": family_arms,
            **_control_report(family_arms),
        }
    body = {
        "schema": SEMANTIC_PROGRAM_SHARED_CAMPAIGN_SCHEMA,
        "feature_manifest_sha256s": {
            family: manifests[family]["manifest_sha256"] for family in sorted(manifests)
        },
        "representation_compatibility": compatibility,
        "model_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "shared_coefficient_sha256": model.training_receipt["coefficient_sha256"],
        "input_grounding_sha256": input_grounding.contract_sha256,
        "geometry_contract": model.geometry_contract,
        "relation_pointer_scale_selection": model.training_receipt[
            "relation_pointer_scale_selection"
        ],
        "shared_model_count": 1,
        "family_router_present": False,
        "family_count": len(family_reports),
        "families": family_reports,
        "arms": arms,
        **_control_report(arms),
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_evaluation": True,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
        "claim_boundary": (
            "bounded typed semantic-program families with learned variable geometry, "
            "operation semantics, and definition-reference relations; one transducer, "
            "no family router, and no broad natural-language domain claim"
        ),
    }
    return SharedSemanticProgramCampaignResult(
        model=model,
        report={**body, "report_sha256": _sha(body)},
    )


def run_shared_semantic_program_campaign(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
    *,
    input_grounding: SemanticInputGroundingContract,
) -> SharedSemanticProgramCampaignResult:
    """Convert verified bundles before running the shared campaign."""

    return run_shared_semantic_program_campaign_from_examples(
        {
            family: training_examples_from_feature_bundle(bundle)
            for family, bundle in bundles.items()
        },
        manifests={family: bundle.manifest for family, bundle in bundles.items()},
        input_grounding=input_grounding,
    )


__all__ = [
    "SEMANTIC_PROGRAM_SHARED_CAMPAIGN_SCHEMA",
    "SharedSemanticProgramCampaignResult",
    "run_shared_semantic_program_campaign",
    "run_shared_semantic_program_campaign_from_examples",
]
