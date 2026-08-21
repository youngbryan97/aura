"""Repair governance decisions must not masquerade as pipeline failures."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.self_modification import code_repair
from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path
from core.self_modification.self_modification_engine import (
    AutonomousSelfModificationEngine,
    _autonomous_cycle_failed,
)


@pytest.mark.asyncio
async def test_code_repair_generates_evidence_before_governance_disposition(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

    generator = SimpleNamespace(
        code_base=tmp_path,
        generate_fix=AsyncMock(return_value=None),
    )
    repair = object.__new__(code_repair.AutonomousCodeRepair)
    repair.generator = generator
    repair._deep_repair_after_patch_failure = AsyncMock(
        return_value={"result": "deep_repair_deferred"}
    )
    gateway = SimpleNamespace(
        run_async=AsyncMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    )
    monkeypatch.setattr(code_repair, "get_subprocess_gateway", lambda: gateway)

    success, fix, result = await repair.repair_bug(
        "core/example.py",
        1,
        {"summary": "measured failure"},
    )

    assert success is False
    assert fix is None
    assert result["error"] == "Fix generation failed"
    generator.generate_fix.assert_awaited_once()


def test_ordinary_core_repair_uses_canonical_shadow_validated_tier():
    decision = classify_mutation_path("core/brain/cognitive/memory_management.py")

    assert decision.tier is MutationTier.SHADOW_VALIDATED_AUTO_FIX
    assert decision.auto_apply_allowed is True


@pytest.mark.parametrize(
    "disposition",
    [
        "proposal_quarantined",
        "proposal_refused_by_policy",
        "proposal_decision_reused",
    ],
)
def test_completed_governance_disposition_does_not_trip_circuit_breaker(disposition):
    assert (
        _autonomous_cycle_failed(
            {
                "success": disposition != "proposal_refused_by_policy",
                "bugs_found": 1,
                "fixes_applied": 0,
                "disposition": disposition,
            }
        )
        is False
    )


def test_real_generation_and_application_failures_stay_failures():
    assert _autonomous_cycle_failed(
        {"success": False, "disposition": "fix_generation_failed"}
    )
    assert _autonomous_cycle_failed(
        {"success": False, "disposition": "repair_application_failed"}
    )


def test_repair_decision_identity_changes_when_the_source_changes(tmp_path):
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    event = SimpleNamespace(
        file_path="core/example.py",
        line_number=1,
        error_type="RuntimeError",
        error_message="broken",
    )
    bug = {
        "pattern": SimpleNamespace(fingerprint="fault-1", events=[event]),
        "diagnosis": {"summary": "measured failure"},
    }
    engine = object.__new__(AutonomousSelfModificationEngine)
    engine.code_base = tmp_path

    first = engine._repair_decision_key(bug)
    assert engine._repair_decision_key(bug) == first

    target.write_text("value = 2\n", encoding="utf-8")
    assert engine._repair_decision_key(bug) != first
