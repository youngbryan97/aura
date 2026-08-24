from __future__ import annotations

import json
from unittest import mock

from core.brain.cortex_self_evidence import (
    CortexCampaignEvidence,
    CortexEvidenceRequest,
    CortexSelfEvidence,
    _verified_campaign,
    classify_cortex_evidence_request,
    render_cortex_evidence_reply,
    render_cortex_evidence_response,
    resolve_cortex_self_evidence,
)


def _evidence() -> CortexSelfEvidence:
    old = CortexCampaignEvidence(
        cortex_label="32B",
        model_path="/models/32b",
        task_count=60,
        exact_by_arm=(("ordinary_base", 16), ("treatment", 60)),
        gain_count=44,
        regression_count=0,
        paired_p_value=5.684341886080802e-14,
        elapsed_seconds=4_814.533,
        artifact_receipt_sha256="a" * 64,
        verification_receipt_sha256="b" * 64,
    )
    current = CortexCampaignEvidence(
        cortex_label="27B",
        model_path="/models/27b",
        task_count=60,
        exact_by_arm=(("ordinary_base", 0), ("treatment", 60)),
        gain_count=60,
        regression_count=0,
        paired_p_value=8.673617379884035e-19,
        elapsed_seconds=3_283.718,
        artifact_receipt_sha256="c" * 64,
        verification_receipt_sha256="d" * 64,
    )
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
        semantic_activation_sha256="e" * 64,
        resident_descriptor_sha256="f" * 64,
        resident_model_path="/models/27b",
        campaigns=(old, current),
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
    assert "4,814.533s and 27B 3,283.718s" in block
    assert "31.8% faster" in block


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
        "descriptor_sha256": "f" * 64,
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
    assert observed.resident_model_path == "/models/resident"


def test_classifier_covers_closed_self_evidence_without_hijacking_open_questions():
    assert (
        classify_cortex_evidence_request(
            "What changed after replacing your former 32B model with the current "
            "27B that you can actually measure?"
        )
        is CortexEvidenceRequest.MEASURED_COMPARISON
    )
    assert (
        classify_cortex_evidence_request("Which cortex are you running now?")
        is CortexEvidenceRequest.IDENTITY
    )
    assert (
        classify_cortex_evidence_request("What evidence proves your recurrent tissue works?")
        is CortexEvidenceRequest.BOUNDED_MECHANISM
    )
    assert classify_cortex_evidence_request("Do you feel different today?") is None
    assert classify_cortex_evidence_request("Compare two language models for me") is None
    assert classify_cortex_evidence_request("Explain the cortex in biology") is None


def test_renderer_replaces_subjective_swap_story_with_verified_measurement():
    reply = render_cortex_evidence_reply(
        "What changed after replacing your former 32B model with the current 27B "
        "that you can actually measure?",
        evidence=_evidence(),
    )

    assert "4,814.533 seconds" in reply
    assert "3,283.718 seconds" in reply
    assert "31.8% faster" in reply
    assert "separately seeded" in reply
    assert "subjective experience" in reply
    assert "unmeasured" in reply
    assert "tighter" not in reply.casefold()
    assert "feel" not in reply.casefold()


def test_identity_renderer_retains_verified_assertion_authority():
    from core.epistemics.assertion import verified_assertion_response_matches

    response = render_cortex_evidence_response(
        "Which cortex are you running now?",
        evidence=_evidence(),
    )

    assert response is not None
    assert "resident cortex is 27B" in response.text
    authority = response.authority()
    assert verified_assertion_response_matches(response.text, authority)
    assert not verified_assertion_response_matches(
        response.text + "\n\nA contradictory correction.",
        authority,
    )


def test_mechanism_renderer_requires_an_activation_receipt():
    evidence = _evidence()
    unreceipted = CortexSelfEvidence(
        **{
            name: getattr(evidence, name)
            for name in evidence.__dataclass_fields__
            if name != "semantic_activation_sha256"
        },
        semantic_activation_sha256="",
    )

    assert (
        render_cortex_evidence_reply(
            "What evidence proves your recurrent tissue works?",
            evidence=unreceipted,
        )
        == ""
    )


def _write_campaign(root, *, verification_exact=60, adjudication_p=0.01):
    root.mkdir()
    identity = {"path": "/models/test", "config_sha256": "1", "weights_index_sha256": "2"}
    receipt = "a" * 64
    (root / "result.json").write_text(
        json.dumps(
            {
                "receipt_sha256": receipt,
                "model_identity": identity,
                "task_count": 1,
                "gain_count": 1,
                "regression_count": 0,
                "elapsed_seconds": 2.0,
                "arms": {
                    "ordinary_base": {"exact": 0},
                    "treatment": {"exact": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "verification.json").write_text(
        json.dumps(
            {
                "verified": True,
                "artifact_receipt_sha256": receipt,
                "verification_receipt_sha256": "b" * 64,
                "model_identity": identity,
                "task_count": 1,
                "gain_count": 1,
                "regression_count": 0,
                "paired_one_sided_exact_p": 0.01,
                "independent_exact_by_arm": {
                    "ordinary_base": 0,
                    "treatment": verification_exact,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "adjudication.json").write_text(
        json.dumps(
            {
                "passed": True,
                "verdict": "BOUNDED_WOW_SIGNAL",
                "task_count": 1,
                "gain_count": 1,
                "regression_count": 0,
                "paired_one_sided_exact_p": adjudication_p,
                "independent_exact_by_arm": {
                    "ordinary_base": 0,
                    "treatment": verification_exact,
                },
            }
        ),
        encoding="utf-8",
    )


def test_campaign_loader_requires_three_way_agreement(tmp_path):
    root = tmp_path / "campaign"
    _write_campaign(root, verification_exact=1)

    campaign = _verified_campaign("test", root)

    assert campaign is not None
    assert campaign.exact_by_arm == (("ordinary_base", 0), ("treatment", 1))
    assert campaign.paired_p_value == 0.01


def test_campaign_loader_rejects_score_or_p_value_disagreement(tmp_path):
    score_root = tmp_path / "score-mismatch"
    _write_campaign(score_root, verification_exact=0)
    p_root = tmp_path / "p-mismatch"
    _write_campaign(p_root, verification_exact=1, adjudication_p=0.02)

    assert _verified_campaign("test", score_root) is None
    assert _verified_campaign("test", p_root) is None
