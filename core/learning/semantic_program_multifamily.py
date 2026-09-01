"""One semantic transducer trained and measured across named task families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.semantic_program_basis import (
    bind_training_examples_to_shared_representation,
    establish_semantic_training_representation_compatibility,
)
from core.learning.semantic_program_campaign import (
    _paired_control,
    _sha,
    training_examples_from_feature_bundle,
)
from core.learning.semantic_program_evaluation import (
    coefficient_lesion,
    evaluate_semantic_program_transducer,
    label_permuted_training_examples,
    shuffle_hidden_tokens,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_transducer import (
    SemanticProgramTransducer,
    SemanticTransducerTrainingExample,
    fit_semantic_program_transducer,
)

SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA: Final = (
    "aura.semantic_program_multifamily_campaign.v1"
)
_EVALUATION_SPLITS: Final = ("validation", "test")
_CONTROL_ARMS: Final = (
    "hidden_token_shuffle",
    "coefficient_lesion",
    "label_permutation",
)


@dataclass(frozen=True, slots=True)
class SemanticProgramMultiFamilyCampaignResult:
    model: SemanticProgramTransducer
    report: dict[str, Any]


def _family_report(
    *,
    model: SemanticProgramTransducer,
    coefficient_control: SemanticProgramTransducer,
    permuted_model: SemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for split in ("train", *_EVALUATION_SPLITS):
        arms[f"treatment:{split}"] = evaluate_semantic_program_transducer(
            model,
            examples,
            split=split,
        ).to_dict()
    for split in _EVALUATION_SPLITS:
        arms[f"hidden_token_shuffle:{split}"] = evaluate_semantic_program_transducer(
            model,
            examples,
            split=split,
            arm="hidden_token_shuffle",
            hidden_transform=shuffle_hidden_tokens,
        ).to_dict()
        arms[f"coefficient_lesion:{split}"] = evaluate_semantic_program_transducer(
            coefficient_control,
            examples,
            split=split,
            arm="coefficient_lesion",
        ).to_dict()
        arms[f"label_permutation:{split}"] = evaluate_semantic_program_transducer(
            permuted_model,
            examples,
            split=split,
            arm="label_permutation",
        ).to_dict()
    paired_programs: dict[str, dict[str, Any]] = {}
    paired_answers: dict[str, dict[str, Any]] = {}
    for split in _EVALUATION_SPLITS:
        treatment_rows = arms[f"treatment:{split}"]["rows"]
        for control in _CONTROL_ARMS:
            control_rows = arms[f"{control}:{split}"]["rows"]
            paired_programs[f"{control}:{split}"] = _paired_control(
                treatment_rows,
                control_rows,
                metric="program_exact",
            )
            paired_answers[f"{control}:{split}"] = _paired_control(
                treatment_rows,
                control_rows,
                metric="answer_exact",
            )
    return {
        "example_count": len(examples),
        "held_out_treatment_program_exact": sum(
            arms[f"treatment:{split}"]["program_exact"]
            for split in _EVALUATION_SPLITS
        ),
        "held_out_treatment_answer_exact": sum(
            arms[f"treatment:{split}"]["answer_exact"]
            for split in _EVALUATION_SPLITS
        ),
        "held_out_total": sum(
            arms[f"treatment:{split}"]["total"] for split in _EVALUATION_SPLITS
        ),
        "arms": arms,
        "paired_program_controls": paired_programs,
        "paired_answer_controls": paired_answers,
    }


def run_semantic_program_multifamily_campaign_from_examples(
    examples_by_family: Mapping[str, Sequence[SemanticTransducerTrainingExample]],
    *,
    manifests: Mapping[str, Mapping[str, Any]],
) -> SemanticProgramMultiFamilyCampaignResult:
    """Fit one coefficient set and adjudicate every family independently."""

    compatibility = establish_semantic_training_representation_compatibility(manifests)
    shared_examples = bind_training_examples_to_shared_representation(
        examples_by_family,
        compatibility=compatibility,
    )
    model = fit_semantic_program_transducer(shared_examples)
    coefficient_control = coefficient_lesion(model)
    permuted_model = fit_semantic_program_transducer(
        label_permuted_training_examples(shared_examples)
    )
    family_reports: dict[str, dict[str, Any]] = {}
    for family in sorted(examples_by_family):
        prefix = f"{family}:"
        family_examples = tuple(
            item for item in shared_examples if item.construction_id.startswith(prefix)
        )
        if len(family_examples) != len(examples_by_family[family]):
            raise ValueError(f"semantic shared campaign family inventory changed: {family}")
        family_reports[family] = _family_report(
            model=model,
            coefficient_control=coefficient_control,
            permuted_model=permuted_model,
            examples=family_examples,
        )
    body = {
        "schema": SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA,
        "feature_manifest_sha256s": {
            family: manifests[family]["manifest_sha256"] for family in sorted(manifests)
        },
        "representation_compatibility": compatibility,
        "model_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "shared_coefficient_sha256": model.training_receipt["coefficient_sha256"],
        "shared_model_count": 1,
        "family_router_present": False,
        "family_count": len(family_reports),
        "families": family_reports,
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_evaluation": True,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
        "claim_boundary": (
            "bounded shared-geometry semantic-program families; one transducer and "
            "no family router; no variable-geometry or broad-domain claim"
        ),
    }
    return SemanticProgramMultiFamilyCampaignResult(
        model=model,
        report={**body, "report_sha256": _sha(body)},
    )


def run_semantic_program_multifamily_campaign(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
) -> SemanticProgramMultiFamilyCampaignResult:
    """Convert verified bundles and run one shared semantic campaign."""

    return run_semantic_program_multifamily_campaign_from_examples(
        {
            family: training_examples_from_feature_bundle(bundle)
            for family, bundle in bundles.items()
        },
        manifests={family: bundle.manifest for family, bundle in bundles.items()},
    )


__all__ = [
    "SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA",
    "SemanticProgramMultiFamilyCampaignResult",
    "run_semantic_program_multifamily_campaign",
    "run_semantic_program_multifamily_campaign_from_examples",
]
