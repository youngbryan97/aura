import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.self_modification import self_modification_engine as sm_mod
from core.self_modification.safe_modification import LogicTransplant, SafeSelfModification
from core.self_modification.shadow_ast_healer import ShadowASTHealer


@pytest.mark.asyncio
async def test_shadow_ast_proposal_does_not_mutate_source_before_safe_apply(tmp_path):
    source_path = tmp_path / "core" / "example.py"
    source_path.parent.mkdir(parents=True)
    original = "async def run():\n    return await asyncio.sleep(0)\n"
    source_path.write_text(original, encoding="utf-8")

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine.code_base = tmp_path
    engine.shadow_healer = ShadowASTHealer(tmp_path)

    sample_event = SimpleNamespace(
        file_path="core/example.py",
        line_number=1,
        error_message="name 'asyncio' is not defined",
    )
    bug = {
        "pattern": SimpleNamespace(events=[sample_event]),
        "diagnosis": {"summary": "missing import"},
    }

    proposal = await engine.propose_fix(bug)

    assert source_path.read_text(encoding="utf-8") == original
    assert proposal is not None
    fix = proposal["fix"]
    assert fix.target_file == "core/example.py"
    assert fix.chunks[0]["original"] == original
    assert "import asyncio" in fix.chunks[0]["fixed"]
    assert proposal["test_results"]["validation"] == "shadow_ast_preview"


@pytest.mark.asyncio
async def test_autonomous_cycle_returns_structured_failure_when_diagnosis_crashes(
    monkeypatch,
):
    recorded = []
    monkeypatch.setattr(
        sm_mod,
        "_record_self_modification_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine.auto_fix_enabled = False
    engine.session_stats = {"session_start": time.time()}
    engine.diagnose_current_bugs = AsyncMock(side_effect=RuntimeError("diagnosis down"))

    result = await engine.run_autonomous_cycle()

    assert result["success"] is False
    assert result["fixes_applied"] == 0
    assert result["degraded_step"] == "autonomous_cycle"
    assert engine.session_stats["cycle_failures"] == 1
    assert engine._last_cycle_error["error"] == "diagnosis down"
    assert recorded[0][1]["receipt_required"] is True


@pytest.mark.asyncio
async def test_refinement_cycle_returns_structured_failure_when_analysis_crashes(
    monkeypatch,
):
    recorded = []
    monkeypatch.setattr(
        sm_mod,
        "_record_self_modification_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine.session_stats = {"session_start": time.time()}
    engine.kernel_refiner = SimpleNamespace(
        analyze_kernel_health=AsyncMock(side_effect=RuntimeError("refiner down"))
    )

    result = await engine.run_refinement_cycle()

    assert result["success"] is False
    assert result["refinements_applied"] == 0
    assert result["degraded_step"] == "refinement_cycle"
    assert engine.session_stats["refinement_failures"] == 1
    assert engine._last_refinement_error["error"] == "refiner down"
    assert recorded[0][1]["receipt_required"] is True


@pytest.mark.asyncio
async def test_report_optimization_preserves_sandbox_results_for_safe_apply():
    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    fix = SimpleNamespace(target_file="core/example.py")
    sandbox_results = {"success": True, "suite": "sandbox"}
    engine.code_repair = SimpleNamespace(
        repair_bug=AsyncMock(return_value=(True, fix, sandbox_results))
    )
    captured = {}

    async def apply_fix(proposal, *, force=False):
        captured["proposal"] = proposal
        captured["force"] = force
        return True

    engine.apply_fix = apply_fix

    result = await engine.report_optimization(
        {"file": "core/example.py", "line": 7, "message": "tighten behavior"}
    )

    assert result is True
    assert captured["force"] is True
    assert captured["proposal"]["test_results"] == sandbox_results


@pytest.mark.asyncio
async def test_swarm_review_accepts_logic_transplant_patch_shape():
    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    swarm = SimpleNamespace(delegate_debate=AsyncMock(return_value="APPROVE"))
    fix = LogicTransplant(
        target_file="core/example.py",
        explanation="whole-file import repair",
        chunks=[{"original": "value = 1\n", "fixed": "value = 2\n"}],
    )

    with patch("core.container.ServiceContainer.get", return_value=swarm):
        result = await engine._swarm_review({"fix": fix, "bug": {"diagnosis": "repair"}})

    assert result is True
    topic = swarm.delegate_debate.await_args.args[0]
    assert "value = 2" in topic


def test_safe_modification_stats_expose_report_fields():
    safe_mod = SafeSelfModification.__new__(SafeSelfModification)
    safe_mod.stats = {
        "total_attempts": 3,
        "successful": 2,
        "failed": 1,
        "rolled_back": 1,
        "blocked_by_policy": 4,
    }

    stats = safe_mod.get_stats()

    assert stats["total_attempts"] == 3
    assert stats["successful"] == 2
    assert stats["failed"] == 1
    assert stats["rolled_back"] == 1
    assert stats["blocked_by_policy"] == 4
    assert stats["success_rate"] == "66.7%"


@pytest.mark.asyncio
async def test_safe_modification_commits_after_quarantine_promotion(tmp_path):
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

    events = []

    class Backup:
        def create_backup(self, _path):
            events.append(("backup", target.read_text(encoding="utf-8")))
            return "backup-id"

        def restore_backup(self, _backup_id):
            target.write_text("value = 1\n", encoding="utf-8")
            return True

    class Git:
        async def create_branch(self, _branch_name):
            events.append(("branch", target.read_text(encoding="utf-8")))
            return True

        async def commit_changes(self, file_path, _message):
            events.append(("commit", (tmp_path / file_path).read_text(encoding="utf-8")))
            return "abc123"

        async def merge_to_main(self, _branch_name):
            events.append(("merge", target.read_text(encoding="utf-8")))
            return True

        async def delete_branch(self, _branch_name):
            events.append(("delete", target.read_text(encoding="utf-8")))
            return True

        async def checkout_main(self):
            return True

    async def validate_boot(_root, *, overlay_file=None):
        assert overlay_file is not None
        return True, "ok"

    safe_mod = SafeSelfModification.__new__(SafeSelfModification)
    safe_mod.code_base = tmp_path
    safe_mod.staging_dir = tmp_path / ".aura-staging"
    safe_mod.staging_dir.mkdir()
    safe_mod.stats = {
        "total_attempts": 0,
        "successful": 0,
        "failed": 0,
        "rolled_back": 0,
        "blocked_by_policy": 0,
    }
    safe_mod.event_bus = None
    safe_mod.backup = Backup()
    safe_mod.git = Git()
    safe_mod.boot_validator = SimpleNamespace(validate_boot=validate_boot)
    safe_mod.modification_log = tmp_path / "modifications.jsonl"

    async def full_suite():
        assert target.read_text(encoding="utf-8") == "value = 1\n"
        return True

    safe_mod._run_full_test_suite = full_suite

    fix = SimpleNamespace(
        target_file="core/example.py",
        target_line=1,
        original_code="value = 1\n",
        fixed_code="value = 2\n",
        explanation="raise value",
        risk_level=1,
        lines_changed=1,
    )

    success, message = await safe_mod.apply_fix(fix, {"success": True})

    assert (success, message) == (True, "Fix applied successfully")
    assert events == [
        ("backup", "value = 1\n"),
        ("branch", "value = 1\n"),
        ("commit", "value = 2\n"),
        ("merge", "value = 2\n"),
        ("delete", "value = 2\n"),
    ]
    assert target.read_text(encoding="utf-8") == "value = 2\n"
