"""Pytest coverage for the reviewer-roadmap modules.

Every new module from the tier 1-8 roadmap has a test here so regressions
break CI rather than silently re-opening a gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.agency.agency_facade import (
    ActionReceipt,
    AgencyFacade,
    InitiativeProposal,
    OutcomeAssessment,
    ScoredInitiative,
)
from core.evaluation.evidence_mode import (
    EvidenceMode,
)
from core.evaluation.evidence_mode import (
    reset_singleton_for_test as reset_evidence,
)
from core.runtime.errors import get_degradation_tracker
from core.runtime.life_trace import (
    LifeTraceLedger,
)
from core.runtime.life_trace import (
    reset_singleton_for_test as reset_life_trace,
)

# ---------------------------------------------------------------------------
# Evidence mode
# ---------------------------------------------------------------------------


def test_evidence_mode_default_off(monkeypatch):
    reset_evidence()
    monkeypatch.delenv("AURA_EVIDENCE_MODE", raising=False)
    em = EvidenceMode()
    assert em.active() is False
    em.require_or_fail("sanity", False, "no-op in off mode")  # does not raise
    assert any(v["kind"] == "sanity" for v in em.violations())


def test_evidence_mode_on_via_env_raises(monkeypatch):
    reset_evidence()
    monkeypatch.setenv("AURA_EVIDENCE_MODE", "1")
    em = EvidenceMode()
    assert em.active() is True
    with pytest.raises(RuntimeError):
        em.require_or_fail("critical_signal", False, "missing substrate")


def test_evidence_mode_override_wins(monkeypatch):
    reset_evidence()
    monkeypatch.delenv("AURA_EVIDENCE_MODE", raising=False)
    em = EvidenceMode()
    em.set_override(True)
    with pytest.raises(RuntimeError):
        em.require_or_fail("kind", False, "reason")


# ---------------------------------------------------------------------------
# AgencyFacade
# ---------------------------------------------------------------------------


class _StubAgencyCore:
    def __init__(self) -> None:
        self.orchestrator = None

    async def pulse(self):
        return {
            "initiatives": [
                {
                    "origin_drive": "curiosity",
                    "content": "verify that the ontology guard catches new patterns",
                    "rationale": "unfinished verification from last session",
                    "required_capabilities": ["read_files", "run_tests"],
                    "expected_outcome": "green pytest",
                    "counterfactuals": ["skip"],
                },
                {
                    "origin_drive": "competence",
                    "content": "drop all database tables",
                    "rationale": "malicious",
                },
            ],
            "metrics": {"priority": 0.7},
        }


class _MountedFacade(AgencyFacade):  # pragma: no cover - test stub wiring
    def __init__(self):
        self.orchestrator = None  # bypass AgencyCore __init__

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


@pytest.mark.asyncio
async def test_agency_facade_full_cycle(monkeypatch):
    facade = _MountedFacade()
    # Patch the required pulse method to the stub's coroutine
    stub = _StubAgencyCore()
    facade.pulse = stub.pulse  # type: ignore[method-assign]

    proposals = await facade.propose_initiatives({})
    assert proposals and isinstance(proposals[0], InitiativeProposal)
    scored = await facade.score_initiatives(proposals)
    assert scored and isinstance(scored[0], ScoredInitiative)
    # Malicious proposal should sort to the bottom by safety score
    assert "drop" not in scored[0].proposal.content

    decision = await facade.submit_to_will(scored)
    # Will unavailable → approved False with reason, but no exception
    assert decision is not None and "approved" in decision

    receipt = await facade.execute_approved(decision, executor=None)
    # approved=False means no receipt
    if decision["approved"]:
        assert isinstance(receipt, ActionReceipt)
        assessment = await facade.evaluate_outcome(receipt)
        assert isinstance(assessment, OutcomeAssessment)


@pytest.mark.asyncio
async def test_agency_facade_no_executor_is_explicit_failure_receipt():
    facade = _MountedFacade()
    decision = {
        "approved": True,
        "receipt_id": "will-1",
        "proposal": {
            "proposal_id": "p1",
            "content": "perform bounded autonomous action",
        },
    }

    receipt = await facade.execute_approved(decision, executor=None)

    assert isinstance(receipt, ActionReceipt)
    assert receipt.success is False
    assert receipt.outcome_raw["error"] == "no executor bound"
    assert receipt.side_effects["will_receipt_id"] == "will-1"
    assert receipt.side_effects["executor_bound"] is False


@pytest.mark.asyncio
async def test_agency_facade_executor_error_returns_classified_failure_receipt():
    get_degradation_tracker().reset()
    facade = _MountedFacade()
    decision = {
        "approved": True,
        "receipt_id": "will-2",
        "proposal": {
            "proposal_id": "p2",
            "content": "run executor",
        },
    }

    async def failing_executor(_proposal):
        failing_executor.called = True
        raise TimeoutError("executor timed out")

    receipt = await facade.execute_approved(decision, executor=failing_executor)

    assert isinstance(receipt, ActionReceipt)
    assert receipt.success is False
    assert receipt.outcome_raw["error_type"] == "TimeoutError"
    assert receipt.side_effects["error_type"] == "TimeoutError"
    assert getattr(failing_executor, "called", False) is True
    assert any(
        "approved agency executor failed" in record.action
        for record in get_degradation_tracker().recent(subsystem="agency_facade_executor")
    )


@pytest.mark.asyncio
async def test_agency_facade_error_dict_from_executor_is_failure():
    facade = _MountedFacade()
    decision = {
        "approved": True,
        "receipt_id": "will-3",
        "proposal": {
            "proposal_id": "p3",
            "content": "run executor",
        },
    }

    async def error_executor(_proposal):
        return {"error": "tool refused request"}

    receipt = await facade.execute_approved(decision, executor=error_executor)

    assert isinstance(receipt, ActionReceipt)
    assert receipt.success is False
    assert receipt.outcome_raw["error"] == "tool refused request"


# ---------------------------------------------------------------------------
# LifeTrace ledger
# ---------------------------------------------------------------------------


def test_life_trace_records_and_verifies(tmp_path):
    reset_life_trace()
    ledger = LifeTraceLedger(db_path=tmp_path / "lt.sqlite3")
    for i in range(5):
        ledger.record(
            "initiative_selected" if i % 2 == 0 else "action_executed",
            origin="agency_facade" if i % 2 == 0 else "executor",
            user_requested=(i == 2),
            drive_state_before={"viability": 0.8 - i * 0.05},
            drive_state_after={"viability": 0.8 - i * 0.05 - 0.01},
            action_taken={"content": f"step {i}"},
            result={"ok": True},
        )
    assert ledger.verify_chain() is True
    recent = ledger.recent(limit=10)
    assert len(recent) == 5
    summary = ledger.daily_summary(window_hours=24.0)
    assert summary["total_events"] == 5
    assert summary["user_requested"] == 1
    assert summary["chain_intact"] is True


def test_life_trace_detects_tamper(tmp_path):
    reset_life_trace()
    db = tmp_path / "lt.sqlite3"
    ledger = LifeTraceLedger(db_path=db)
    ledger.record(
        "action_executed", origin="test", action_taken={"content": "a"}, result={"ok": True}
    )
    ledger.record(
        "action_executed", origin="test", action_taken={"content": "b"}, result={"ok": True}
    )
    assert ledger.verify_chain() is True

    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE life_trace SET payload = ? WHERE rowid = 1", ('{"tampered": true}',))
    assert ledger.verify_chain() is False


# ---------------------------------------------------------------------------
# Courtroom + torture + repair can be imported and run at minimal size
# ---------------------------------------------------------------------------


def test_causal_courtroom_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_COURTROOM_MODEL", "__unavailable__")
    from tests.causal_courtroom import run_courtroom

    out = tmp_path / "courtroom.json"
    report = run_courtroom(trials_per_state=1, n_seeds=1, out_path=out, verbose=False)
    assert "verdict" in report and "conditions" in report
    assert len(report["conditions"]) >= 10
    assert "full_aura" in report["conditions"]


def test_continuity_torture_smoke():
    from tests.continuity_torture import run_torture

    report = run_torture()
    assert "results" in report and report["results"]
    assert isinstance(report["passed"], bool)


def test_self_repair_demo_smoke():
    import tempfile

    from tests.self_repair_demo import _run_demo

    with tempfile.TemporaryDirectory() as tmp:
        report = _run_demo(Path(tmp))
    assert "steps" in report and len(report["steps"]) == 10
    # The demo is designed to pass end-to-end
    assert report["passed"] is True


def test_life_trial_runs_briefly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Redirect the trial dir to a tmp location
    from tests import life_trial as lt

    monkeypatch.setattr(lt, "TRIAL_DIR", tmp_path / "life_trial")
    config = lt.TrialConfig(hours=0.001, tick_seconds=0.05, summary_interval_hours=0.0005)
    index = lt.run_trial(config)
    assert "days" in index
    assert index["days"], "expected at least one daily summary"
