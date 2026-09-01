"""Independent replay of a frozen variable-geometry semantic campaign."""

from __future__ import annotations

from collections.abc import Mapping
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
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_shared_campaign import (
    _CONTROL_ARMS,
    _EVALUATION_SPLITS,
    SEMANTIC_PROGRAM_SHARED_CAMPAIGN_SCHEMA,
    _control_report,
    _evaluate_arms,
    _family_examples,
    _subset_arm,
)
from core.learning.semantic_program_shared_transducer import (
    shared_semantic_program_transducer_from_dict,
)

SEMANTIC_PROGRAM_SHARED_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_program_shared_verification.v1"
)
SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES: Final = (
    "core/learning/semantic_input_grounding.py",
    "core/learning/semantic_program_basis.py",
    "core/learning/semantic_program_campaign.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_shared_campaign.py",
    "core/learning/semantic_program_shared_evaluation.py",
    "core/learning/semantic_program_shared_transducer.py",
    "core/learning/semantic_program_shared_verification.py",
    "core/learning/semantic_program_transducer.py",
    "tools/run_shared_semantic_program_campaign.py",
    "tools/verify_shared_semantic_program_campaign.py",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_shared_semantic_program_campaign(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
    *,
    stored_model_payload: Mapping[str, Any],
    stored_report: Mapping[str, Any],
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Replay every held-out arm without fitting or trusting stored counts."""

    if set(source_sha256s) != set(SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES) or any(
        not _is_sha256(value) for value in source_sha256s.values()
    ):
        raise ValueError("shared semantic verification source inventory differs")
    if set(bundles) != set(stored_report.get("feature_manifest_sha256s", {})):
        raise ValueError("shared semantic verification family inventory differs")
    report_body = {
        key: value for key, value in stored_report.items() if key != "report_sha256"
    }
    if (
        stored_report.get("schema") != SEMANTIC_PROGRAM_SHARED_CAMPAIGN_SCHEMA
        or stored_report.get("report_sha256") != _sha(report_body)
    ):
        raise ValueError("shared semantic campaign report identity differs")
    manifests = {family: bundle.manifest for family, bundle in bundles.items()}
    manifest_sha256s = {
        family: manifests[family]["manifest_sha256"] for family in sorted(manifests)
    }
    if stored_report["feature_manifest_sha256s"] != manifest_sha256s:
        raise ValueError("shared semantic campaign feature manifests differ")
    compatibility = establish_semantic_training_representation_compatibility(manifests)
    if stored_report.get("representation_compatibility") != compatibility:
        raise ValueError("shared semantic representation compatibility differs")
    grouped = {
        family: training_examples_from_feature_bundle(bundle)
        for family, bundle in bundles.items()
    }
    examples = bind_training_examples_to_shared_representation(
        grouped,
        compatibility=compatibility,
    )
    model = shared_semantic_program_transducer_from_dict(stored_model_payload)
    tokenizer_identities = {
        item.tokenizer_identity_sha256 for items in grouped.values() for item in items
    }
    if (
        model.to_dict() != dict(stored_model_payload)
        or model.model_basis_sha256
        != compatibility["target_training_session_basis_sha256"]
        or model.input_grounding.tokenizer_identity_sha256 not in tokenizer_identities
        or tokenizer_identities != {model.input_grounding.tokenizer_identity_sha256}
        or stored_report.get("model_basis_sha256") != model.model_basis_sha256
        or stored_report.get("transducer_receipt_sha256") != model.receipt_sha256
        or stored_report.get("shared_coefficient_sha256")
        != model.training_receipt["coefficient_sha256"]
        or stored_report.get("input_grounding_sha256")
        != model.input_grounding.contract_sha256
        or stored_report.get("geometry_contract") != model.geometry_contract
        or stored_report.get("relation_pointer_scale_selection")
        != model.training_receipt["relation_pointer_scale_selection"]
    ):
        raise ValueError("shared semantic frozen model binding differs")
    arms = _evaluate_arms(model=model, examples=examples)
    if stored_report.get("arms") != arms:
        raise ValueError("shared semantic replayed arms differ")
    controls = _control_report(arms)
    if any(stored_report.get(key) != value for key, value in controls.items()):
        raise ValueError("shared semantic paired controls differ")
    families = stored_report.get("families")
    if not isinstance(families, Mapping) or set(families) != set(grouped):
        raise ValueError("shared semantic family reports differ")
    for family in sorted(grouped):
        selected = _family_examples(examples, family)
        family_arms = {
            name: _subset_arm(arm, family=family) for name, arm in arms.items()
        }
        expected_family = {
            "example_count": len(selected),
            "arms": family_arms,
            **_control_report(family_arms),
        }
        if len(selected) != len(grouped[family]) or families[family] != expected_family:
            raise ValueError(f"shared semantic family replay differs: {family}")
    test = arms["treatment:test"]
    causal_program_controls = {
        control: arms[f"{control}:test"]["program_exact"]
        for control in _CONTROL_ARMS
    }
    if (
        stored_report.get("shared_model_count") != 1
        or stored_report.get("family_router_present") is not False
        or stored_report.get("family_count") != len(grouped)
        or stored_report.get("serving_authority") is not False
        or any(
            stored_report.get(field) is not False
            for field in (
                "expected_answers_available_to_training",
                "verifier_traces_available",
                "generated_compiler_text_available",
            )
        )
        or test["program_exact"] <= 0
        or any(test["program_exact"] <= value for value in causal_program_controls.values())
        or any(
            arms[f"treatment:{split}"]["total"]
            != arms[f"{control}:{split}"]["total"]
            for split in _EVALUATION_SPLITS
            for control in _CONTROL_ARMS
        )
    ):
        raise ValueError("shared semantic causal claim criteria are not met")
    paired = {
        control: _paired_control(
            arms["treatment:test"]["rows"],
            arms[f"{control}:test"]["rows"],
            metric="program_exact",
        )
        for control in _CONTROL_ARMS
    }
    body = {
        "schema": SEMANTIC_PROGRAM_SHARED_VERIFICATION_SCHEMA,
        "verified": True,
        "campaign_report_sha256": stored_report["report_sha256"],
        "transducer_receipt_sha256": model.receipt_sha256,
        "feature_manifest_sha256s": manifest_sha256s,
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "test_total": test["total"],
        "test_program_exact": test["program_exact"],
        "test_answer_exact": test["answer_exact"],
        "test_program_controls": causal_program_controls,
        "paired_test_program_controls": paired,
        "family_test_program_exact": {
            family: families[family]["arms"]["treatment:test"]["program_exact"]
            for family in sorted(families)
        },
        "family_test_answer_exact": {
            family: families[family]["arms"]["treatment:test"]["answer_exact"]
            for family in sorted(families)
        },
        "serving_authority": False,
        "claim_boundary": stored_report["claim_boundary"],
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PROGRAM_SHARED_VERIFICATION_SCHEMA",
    "SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES",
    "verify_shared_semantic_program_campaign",
]
