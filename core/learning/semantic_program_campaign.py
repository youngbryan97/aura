"""CPU-only training and matched controls for semantic program transfer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.semantic_program_evaluation import (
    coefficient_lesion,
    evaluate_semantic_program_transducer,
    label_permuted_training_examples,
    shuffle_hidden_tokens,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_ir import semantic_program_ir_from_dict
from core.learning.semantic_program_transducer import (
    SemanticProgramTransducer,
    SemanticTransducerTrainingExample,
    fit_semantic_program_transducer,
)

SEMANTIC_PROGRAM_CAMPAIGN_SCHEMA: Final = "aura.semantic_program_campaign.v2"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def training_examples_from_feature_bundle(
    bundle: LoadedSemanticFeatureBundle,
) -> tuple[SemanticTransducerTrainingExample, ...]:
    """Convert a verified bundle without widening its evidence authority."""

    examples = tuple(
        SemanticTransducerTrainingExample(
            ir=semantic_program_ir_from_dict(item.metadata["gold_ir"]),
            hidden_states=item.hidden_states,
            split=str(item.metadata["split"]),
            construction_id=str(item.metadata["construction_id"]),
            topology_id=str(item.metadata["topology_id"]),
            public_inputs=tuple(int(value) for value in item.metadata["inputs"]),
        )
        for item in bundle.examples
    )
    if len(examples) != bundle.manifest["example_count"]:
        raise ValueError("semantic campaign bundle count changed during conversion")
    if {item.split for item in examples} != {"train", "validation", "test"}:
        raise ValueError("semantic campaign requires train, validation, and test splits")
    return examples


def _one_sided_exact_p(*, treatment_only: int, control_only: int) -> float:
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, successes)
        for successes in range(treatment_only, discordant + 1)
    ) / (2**discordant)


def _paired_control(
    treatment_rows: Sequence[dict[str, Any]],
    control_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    treatment = {
        str(row["source_text_sha256"]): bool(row["program_exact"])
        for row in treatment_rows
    }
    control = {
        str(row["source_text_sha256"]): bool(row["program_exact"])
        for row in control_rows
    }
    if treatment.keys() != control.keys():
        raise ValueError("semantic campaign paired arms contain different tasks")
    treatment_only = sum(treatment[key] and not control[key] for key in treatment)
    control_only = sum(control[key] and not treatment[key] for key in treatment)
    return {
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": treatment_only + control_only,
        "one_sided_exact_p": _one_sided_exact_p(
            treatment_only=treatment_only,
            control_only=control_only,
        ),
    }


@dataclass(frozen=True, slots=True)
class SemanticProgramCampaignResult:
    model: SemanticProgramTransducer
    report: dict[str, Any]


def run_semantic_program_campaign(
    bundle: LoadedSemanticFeatureBundle,
) -> SemanticProgramCampaignResult:
    """Fit once and measure unseen constructions against matched null arms."""

    examples = training_examples_from_feature_bundle(bundle)
    model = fit_semantic_program_transducer(examples)
    coefficient_control = coefficient_lesion(model)
    permuted_model = fit_semantic_program_transducer(
        label_permuted_training_examples(examples)
    )
    arms: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        arms[f"treatment:{split}"] = evaluate_semantic_program_transducer(
            model,
            examples,
            split=split,
        ).to_dict()
    for split in ("validation", "test"):
        arms[f"hidden_token_shuffle:{split}"] = (
            evaluate_semantic_program_transducer(
                model,
                examples,
                split=split,
                arm="hidden_token_shuffle",
                hidden_transform=shuffle_hidden_tokens,
            ).to_dict()
        )
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

    paired: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        treatment_rows = arms[f"treatment:{split}"]["rows"]
        for control in (
            "hidden_token_shuffle",
            "coefficient_lesion",
            "label_permutation",
        ):
            paired[f"{control}:{split}"] = _paired_control(
                treatment_rows,
                arms[f"{control}:{split}"]["rows"],
            )

    held_out_treatment = sum(
        arms[f"treatment:{split}"]["program_exact"]
        for split in ("validation", "test")
    )
    held_out_total = sum(
        arms[f"treatment:{split}"]["total"]
        for split in ("validation", "test")
    )
    body = {
        "schema": SEMANTIC_PROGRAM_CAMPAIGN_SCHEMA,
        "feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "model_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "example_count": len(examples),
        "held_out_treatment_program_exact": held_out_treatment,
        "held_out_total": held_out_total,
        "arms": arms,
        "paired_controls": paired,
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_evaluation": True,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
    }
    report = {**body, "report_sha256": _sha(body)}
    return SemanticProgramCampaignResult(model=model, report=report)


__all__ = [
    "SEMANTIC_PROGRAM_CAMPAIGN_SCHEMA",
    "SemanticProgramCampaignResult",
    "run_semantic_program_campaign",
    "training_examples_from_feature_bundle",
]
