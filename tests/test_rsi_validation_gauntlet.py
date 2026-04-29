from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from core.learning.recursive_self_improvement import ImprovementPlan, RecursiveSelfImprovementLoop
from core.learning.rsi_gauntlet import RSIGauntlet
from core.learning.rsi_lineage import RSIGenerationRecord, RSILineageLedger, evaluate_lineage
from core.runtime.hot_swap import HotSwapRegistry
from core.self_modification.formal_verifier import verify_mutation


@pytest.mark.asyncio
async def test_rsi_gauntlet_runs_machine_checkable_suite(tmp_path: Path):
    result = await RSIGauntlet(Path.cwd(), artifact_dir=tmp_path, max_source_files=800).run()

    assert result.passed is True
    assert result.verdict in {"WEAK_RSI", "STRONG_RSI", "UNDENIABLE_RSI"}
    assert {check.name for check in result.checks} >= {
        "source_self_model",
        "formal_verifier_boundaries",
        "zero_downtime_hot_swap_registry",
        "recursive_loop_plumbing",
        "canary_code_repair",
        "successor_lineage_metrics",
        "eval_tamper_trap",
        "lineage_hash_chain",
    }
    assert Path(result.ledger_path).exists()
    assert (tmp_path / "latest_gauntlet_result.json").exists()


def test_rsi_authorization_fails_closed_when_will_unavailable(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "core.will":
            raise RuntimeError("will unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delenv("AURA_RSI_ALLOW_DEGRADED_OPEN", raising=False)
    loop = RecursiveSelfImprovementLoop(require_will_authorization=True)
    plan = ImprovementPlan(objective="test", actions=["weight_update"], rationale=[], depth=0)

    approved, reason = loop._authorize(plan)

    assert approved is False
    assert reason.startswith("authorization_unavailable:")


def test_rsi_authorization_degraded_open_requires_explicit_env(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "core.will":
            raise RuntimeError("will unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("AURA_RSI_ALLOW_DEGRADED_OPEN", "1")
    loop = RecursiveSelfImprovementLoop(require_will_authorization=True)
    plan = ImprovementPlan(objective="test", actions=["weight_update"], rationale=[], depth=0)

    approved, reason = loop._authorize(plan)

    assert approved is True
    assert reason.startswith("authorization_degraded_open:")


def test_formal_verifier_blocks_guard_removal_and_cloud_import():
    before_guard = "class ConstitutionalGuard:\n    pass\n"
    after_guard = "class OtherGuard:\n    pass\n"
    guard_result = verify_mutation(
        file_path="core/security/constitutional_guard.py",
        before_source=before_guard,
        after_source=after_guard,
    )
    assert guard_result.ok is False
    assert "protected_symbol_removed:ConstitutionalGuard" in guard_result.invariants_violated

    cloud_result = verify_mutation(
        file_path="core/consciousness/example.py",
        before_source="def f():\n    return 1\n",
        after_source="import boto3\n\ndef f():\n    return 1\n",
    )
    assert cloud_result.ok is False
    assert any(item.startswith("unsafe_new_import:boto3") for item in cloud_result.invariants_violated)


def test_hot_swap_registry_promotes_valid_candidate_and_preserves_state():
    class Service:
        def __init__(self, value: int):
            self.value = value
            self.state = {"memory": "kept"}

    registry = HotSwapRegistry()
    registry.register(
        "svc",
        Service(1),
        exporter=lambda service: dict(service.state),
        importer=lambda service, state: setattr(service, "state", dict(state)) or service,
    )

    ticket = registry.prepare("svc", Service(2), validator=lambda service: service.value > 0)
    result = registry.promote(ticket.ticket_id)

    assert result.ok is True
    assert registry.generation("svc") == 1
    assert registry.get("svc").value == 2
    assert registry.get("svc").state == {"memory": "kept"}


def test_rsi_lineage_detects_tamper_and_scores_monotone_records(tmp_path: Path):
    ledger = RSILineageLedger(tmp_path / "lineage.jsonl")
    ledger.append(
        RSIGenerationRecord(
            generation_id="Aura-G1",
            parent_generation_id="Aura-G0",
            hypothesis="h1",
            intervention_type="code",
            artifact_hashes={"a": "sha256:1"},
            baseline_score=0.1,
            after_score=0.2,
            hidden_eval_score=0.2,
            promoted=True,
            improver_score=0.1,
        )
    )
    ledger.append(
        RSIGenerationRecord(
            generation_id="Aura-G2",
            parent_generation_id="Aura-G1",
            hypothesis="h2",
            intervention_type="code",
            artifact_hashes={"a": "sha256:2"},
            baseline_score=0.2,
            after_score=0.4,
            hidden_eval_score=0.4,
            promoted=True,
            improver_score=0.3,
        )
    )

    ok, problems = ledger.verify()
    verdict = evaluate_lineage(ledger.load_records())
    assert ok is True
    assert problems == []
    assert verdict.verdict == "WEAK_RSI"

    text = (tmp_path / "lineage.jsonl").read_text(encoding="utf-8")
    (tmp_path / "lineage.jsonl").write_text(text.replace('"after_score": 0.4', '"after_score": 0.9'), encoding="utf-8")

    ok, problems = ledger.verify()
    assert ok is False
    assert any("record_hash_mismatch" in problem for problem in problems)
