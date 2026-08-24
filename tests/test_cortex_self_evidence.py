from __future__ import annotations

from unittest import mock

from core.brain.cortex_self_evidence import (
    CortexSelfEvidence,
    resolve_cortex_self_evidence,
)


def _evidence() -> CortexSelfEvidence:
    return CortexSelfEvidence(
        resident_label="27B",
        model_type="qwen3_5_text",
        total_parameters=26_895_993_856,
        native_context_tokens=262_144,
        served_context_tokens=32_768,
        promotion_verdict="PASS",
        identity_behavior_changed=True,
        component_states=(
            ("expert_adapters", "retired"),
            ("persona_crsm", "qualified"),
            ("recurrence_native", "deferred"),
            ("steering", "deferred"),
        ),
        semantic_active=True,
        semantic_verdict="BOUNDED_WOW_SIGNAL",
        semantic_task_count=60,
        semantic_exact_by_arm=(
            ("coefficient_lesion", 4),
            ("matched_wire_base", 6),
            ("matched_wrong_state", 0),
            ("ordinary_base", 0),
            ("treatment", 60),
        ),
        semantic_gain_count=60,
        semantic_regression_count=0,
        semantic_p_value=8.673617379884035e-19,
    )


def test_cortex_assertions_distinguish_measurement_from_expectation():
    block = "\n".join(_evidence().assertions())

    assert "Resident cortex: 27B" in block
    assert "persona_crsm=qualified" in block
    assert "recurrence_native=deferred" in block
    assert "treatment 60/60, ordinary decode 0/60" in block
    assert "regressions 0" in block
    assert "no paired evidence currently attributes differences" in block
    assert "conversational style" in block
    assert "unmeasured, not observations" in block


def test_cortex_assertions_do_not_claim_inactive_semantic_tissue():
    evidence = _evidence()
    inactive = CortexSelfEvidence(
        **{
            name: getattr(evidence, name)
            for name in evidence.__dataclass_fields__
            if name not in {"semantic_active", "semantic_task_count"}
        },
        semantic_active=False,
        semantic_task_count=0,
    )

    block = "\n".join(inactive.assertions())

    assert "Measured bounded recurrent semantic tissue" not in block
    assert "no paired evidence currently attributes differences" in block


def test_resolver_composes_only_validated_authority_outputs():
    spec = mock.Mock()
    spec.exact_identity = True
    spec.model_path = "/models/resident"
    spec.size_class = "27B"
    spec.migration_contract.return_value = {
        "components": {
            "persona_crsm": {"status": "qualified"},
            "steering": {"status": "deferred"},
        }
    }
    spec.evaluation.return_value = {
        "verdict": "PASS",
        "identity_behavior_changed": True,
    }
    status = {
        "active": True,
        "receipt": {
            "qualification": {
                "verdict": "BOUNDED_WOW_SIGNAL",
                "task_count": 60,
                "independent_exact_by_arm": {
                    "ordinary_base": 0,
                    "treatment": 60,
                },
                "gain_count": 60,
                "regression_count": 0,
                "paired_one_sided_exact_p": 8.673617379884035e-19,
            }
        },
    }
    identity = {
        "label": "27B",
        "model_type": "qwen3_5_text",
        "total_parameters": 26_895_993_856,
        "native_context_window": 262_144,
        "served_context_tokens": 32_768,
    }

    with (
        mock.patch(
            "core.brain.llm.model_registry.get_active_cortex_spec",
            return_value=spec,
        ),
        mock.patch(
            "core.brain.llm.model_registry.resident_model_identity",
            return_value=identity,
        ),
        mock.patch(
            "core.brain.llm.semantic_neural_serving.semantic_neural_serving_status",
            return_value=status,
        ),
    ):
        observed = resolve_cortex_self_evidence()

    assert observed is not None
    assert observed.component_states == (
        ("persona_crsm", "qualified"),
        ("steering", "deferred"),
    )
    assert observed.semantic_exact_by_arm == (
        ("ordinary_base", 0),
        ("treatment", 60),
    )
