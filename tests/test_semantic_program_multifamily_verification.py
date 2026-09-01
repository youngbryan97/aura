from __future__ import annotations

import copy

import pytest

from core.learning.semantic_program_multifamily_verification import (
    SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SOURCES,
    verify_semantic_program_multifamily_campaign,
)


def test_multifamily_verifier_rejects_source_inventory_before_refit() -> None:
    with pytest.raises(ValueError, match="source identity"):
        verify_semantic_program_multifamily_campaign(
            {},
            stored_model_payload={},
            stored_report={},
            source_sha256s={},
        )


def test_multifamily_verifier_rejects_tampered_report_before_refit(monkeypatch) -> None:
    report = {
        "schema": "aura.semantic_program_multifamily_campaign.v1",
        "report_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        "core.learning.semantic_program_multifamily_verification.run_semantic_program_multifamily_campaign",
        lambda _: (_ for _ in ()).throw(AssertionError("refit must not run")),
    )
    with pytest.raises(ValueError, match="envelope"):
        verify_semantic_program_multifamily_campaign(
            {"a": copy.copy(object()), "b": copy.copy(object())},
            stored_model_payload={},
            stored_report=report,
            source_sha256s={
                source: "1" * 64
                for source in SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SOURCES
            },
        )
