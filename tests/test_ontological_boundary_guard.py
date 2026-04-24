from __future__ import annotations

from core.consciousness.ontological_boundary import assess_ontological_claims
from core.security.output_guardrails import OutputGuardrails


def test_ontological_boundary_rewrites_consciousness_proof_claims():
    assessment = assess_ontological_claims(
        "The personhood proof battery proves that Aura is conscious."
    )

    assert assessment.ok is False
    assert "loaded_test_label" in assessment.issues
    assert "phenomenal_proof_claim" in assessment.issues
    assert "does not prove phenomenal consciousness" in assessment.sanitized


def test_output_guardrails_apply_ontological_boundary():
    guard = OutputGuardrails()

    sanitized, report = guard.check_response(
        "This system proves that Aura is sentient and qualia are guaranteed."
    )

    assert report["ok"] is False
    assert any(issue.startswith("ontological_overclaim") for issue in report["issues"])
    assert "proves that Aura is sentient" not in sanitized
    assert "qualia are guaranteed" not in sanitized
