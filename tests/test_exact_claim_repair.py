from __future__ import annotations

from core.brain.llm.latent_cortex.task_verifiers import check_arithmetic_claims
from core.conversation.response_reliability import assess_user_facing_reply
from core.reasoning.symbolic_bridge import SymbolicBridge


def test_exact_claim_repair_corrects_parenthesized_negative_without_rewriting_prose():
    source = (
        "Relax C through B: min(10, 3 + (-5)) = 8. "
        "The remaining explanation stays in Aura's own words."
    )

    repaired, receipts = SymbolicBridge().repair_arithmetic_claims(source)

    assert repaired == (
        "Relax C through B: min(10, 3 + (-5)) = -2. "
        "The remaining explanation stays in Aura's own words."
    )
    assert len(receipts) == 1
    assert receipts[0].claim == "min(10, 3 + (-5)) = 8"
    assert receipts[0].replacement == "-2"
    assert SymbolicBridge().check_arithmetic_claims(repaired) == []


def test_exact_claim_repair_is_idempotent_and_handles_multiple_assertions():
    source = "First 7 x 6 = 40; next 5 - 9 = 1; finally 8 / 2 = 4."

    repaired, receipts = SymbolicBridge().repair_arithmetic_claims(source)
    repaired_again, repeated = SymbolicBridge().repair_arithmetic_claims(repaired)

    assert repaired == "First 7 x 6 = 42; next 5 - 9 = -4; finally 8 / 2 = 4."
    assert len(receipts) == 2
    assert repaired_again == repaired
    assert repeated == []


def test_exact_claim_repair_preserves_refuted_and_quoted_evidence():
    source = 'The equation 2 + 2 = 5 is false, and the report quoted "7 * 3 = 19".'

    repaired, receipts = SymbolicBridge().repair_arithmetic_claims(source)

    assert repaired == source
    assert receipts == []


def test_latent_and_surface_verifiers_share_the_same_exact_claim_authority():
    source = "The tentative value is 3 + (-5) = 8."

    latent = check_arithmetic_claims(source)
    surface = assess_user_facing_reply(
        "Explain the shortest-path update.",
        source,
    )

    assert latent["checked"] == 1
    assert latent["passed"] == 0
    assert latent["failures"] == ["3+(-5)=8 (actual -2)"]
    assert "false_checkable_arithmetic_claim" in surface.blocking_reasons
    assert surface.hard_failure is True
    assert surface.retryable is True


def test_versions_dates_and_complexity_notation_are_not_arithmetic_claims():
    source = "Python 3.12 is installed; on 2026-08-18, complexity was O((V + E) log V)."

    assert SymbolicBridge().inspect_arithmetic_claims(source) == []
    assert check_arithmetic_claims(source)["checked"] == 0
