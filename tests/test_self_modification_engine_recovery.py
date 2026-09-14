import inspect
import asyncio
import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.self_modification import self_modification_engine as sm_mod
from core.self_modification.safe_modification import LogicTransplant, SafeSelfModification
from core.self_modification.shadow_ast_healer import ShadowASTHealer


class _RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class _AsyncCallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.return_value = result
        self.side_effect = side_effect
        self.await_args_list = []
        self.await_args = None

    @property
    def await_count(self):
        return len(self.await_args_list)

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.await_args_list.append(call)
        self.await_args = call

        async def _complete():
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            if callable(self.side_effect):
                value = self.side_effect(*args, **kwargs)
            else:
                value = self.return_value
            if inspect.isawaitable(value):
                return await value
            return value

        return _complete()


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
async def test_runtime_self_modification_promotion_requires_operator_opt_in(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", raising=False)
    # (The cycle is deterministic now — the old random-gate patch died with
    # the stochastic path; the operator opt-in gate is the whole contract.)
    ServiceContainer.clear()
    ServiceContainer.register_instance("aura_kernel", SimpleNamespace(volition_level=3))

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine.auto_fix_enabled = True
    engine._auto_fix_requested = True
    engine.session_stats = {"session_start": time.time()}
    engine.diagnose_current_bugs = _AsyncCallRecorder(result=[])

    result = await engine.run_autonomous_cycle()

    assert result["success"] is True
    assert result["fixes_applied"] == 0
    assert engine.auto_fix_enabled is False
    ServiceContainer.clear()


def test_runtime_patch_promotion_requires_repair_lab_profile(monkeypatch):
    monkeypatch.setenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", "1")
    monkeypatch.setenv("AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION", "1")
    monkeypatch.delenv("AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION", raising=False)

    assert sm_mod._runtime_patch_promotion_enabled() is False

    monkeypatch.setenv("AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION", "1")

    assert sm_mod._runtime_patch_promotion_enabled() is True


def _enable_repair_lab_profile(monkeypatch):
    monkeypatch.setenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", "1")
    monkeypatch.setenv("AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION", "1")
    monkeypatch.setenv("AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION", "1")


@pytest.mark.asyncio
async def test_apply_fix_refuses_unsupervised_promotion_without_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", raising=False)
    ServiceContainer.clear()
    ServiceContainer.register_instance("aura_kernel", SimpleNamespace(volition_level=3))

    review_calls = []

    async def review(_proposal):
        review_calls.append(_proposal)
        raise AssertionError("policy should block before swarm review")

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine.auto_fix_enabled = True
    engine._auto_fix_requested = True
    engine.code_base = tmp_path
    engine._swarm_review = review

    proposal = {
        "fix": SimpleNamespace(target_file="core/example.py"),
        "test_results": {"success": True},
    }

    assert await engine.apply_fix(proposal, force=False) is False
    assert review_calls == []
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_apply_fix_allows_safe_autonomous_repair_for_safe_tier(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", raising=False)
    monkeypatch.setenv(
        "AURA_PENDING_PATCH_REGISTRY",
        str(tmp_path / "runtime" / "selfmod" / "pending_patch_registry.jsonl"),
    )
    ServiceContainer.clear()
    ServiceContainer.register_instance("aura_kernel", SimpleNamespace(volition_level=3))

    async def review(_proposal, **_kwargs):
        return True

    class _SafeApply:
        def __init__(self):
            self.calls = []

        async def apply_fix(self, *, fix, test_results, supervised=False, safe_autonomous=False):
            self.calls.append(
                {
                    "fix": fix,
                    "test_results": test_results,
                    "supervised": supervised,
                    "safe_autonomous": safe_autonomous,
                }
            )
            return True, "safe auto applied"

    class _Learning:
        def __init__(self):
            self.calls = []

        def record_fix_attempt(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine.auto_fix_enabled = False
    engine._auto_fix_requested = True
    engine.code_base = tmp_path
    engine._fix_lock = asyncio.Lock()
    engine._swarm_review = review
    engine.safe_modification = _SafeApply()
    engine.learning_system = _Learning()
    engine.session_stats = {"fixes_successful": 0, "fixes_attempted": 0}

    fix = SimpleNamespace(
        target_file="tests/test_generated_repair.py",
        target_line=1,
        original_code="assert False",
        fixed_code="assert True",
        explanation="repair failing generated test",
        hypothesis="test assertion is stale",
        confidence="high",
    )
    proposal = {
        "fix": fix,
        "test_results": {
            "success": True,
            "validation": "safe_modification_harness",
            "tests_run": ["tests/test_generated_repair.py"],
        },
    }

    assert await engine.apply_fix(proposal, safe_autonomous=True) is True
    assert engine.safe_modification.calls[0]["safe_autonomous"] is True
    assert engine.safe_modification.calls[0]["supervised"] is False
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_apply_fix_force_requires_supervised_operator_override(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_ALLOW_SUPERVISED_SELF_MODIFICATION", raising=False)
    ServiceContainer.clear()
    ServiceContainer.register_instance("aura_kernel", SimpleNamespace(volition_level=3))

    review_calls = []

    async def review(_proposal, **_kwargs):
        review_calls.append(_proposal)
        return True

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    engine._swarm_review = review
    engine.code_base = tmp_path

    proposal = {
        "fix": SimpleNamespace(target_file="core/example.py"),
        "test_results": {"success": True},
    }

    assert await engine.apply_fix(proposal, force=True) is False
    assert review_calls == []
    ServiceContainer.clear()


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
    engine.diagnose_current_bugs = _AsyncCallRecorder(
        side_effect=RuntimeError("diagnosis down")
    )

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
        analyze_kernel_health=_AsyncCallRecorder(
            side_effect=RuntimeError("refiner down")
        )
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
        repair_bug=_AsyncCallRecorder(result=(True, fix, sandbox_results))
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
    assert captured["force"] is False
    assert captured["proposal"]["test_results"] == sandbox_results


@pytest.mark.asyncio
async def test_swarm_review_accepts_logic_transplant_patch_shape(monkeypatch):
    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    swarm = SimpleNamespace(delegate_debate=_AsyncCallRecorder(result="APPROVE"))
    fix = LogicTransplant(
        target_file="core/example.py",
        explanation="whole-file import repair",
        chunks=[{"original": "value = 1\n", "fixed": "value = 2\n"}],
    )

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda *_args, **_kwargs: swarm))
    result = await engine._swarm_review({"fix": fix, "bug": {"diagnosis": "repair"}})

    assert result is True
    topic = swarm.delegate_debate.await_args.args[0]
    assert "value = 2" in topic


@pytest.mark.asyncio
async def test_swarm_review_fails_closed_when_delegator_missing(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        sm_mod,
        "_record_self_modification_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    fix = SimpleNamespace(target_file="core/example.py", fixed_code="value = 2\n")

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda *_args, **_kwargs: None))
    result = await engine._swarm_review({"fix": fix, "bug": {"diagnosis": "repair"}})

    assert result is False
    assert recorded
    assert recorded[0][1]["receipt_required"] is True
    assert "required swarm review was unavailable" in recorded[0][1]["action"]


@pytest.mark.asyncio
async def test_swarm_review_force_fails_closed_without_delegator(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        sm_mod,
        "_record_self_modification_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    engine = sm_mod.AutonomousSelfModificationEngine.__new__(
        sm_mod.AutonomousSelfModificationEngine
    )
    fix = SimpleNamespace(target_file="core/example.py", fixed_code="value = 2\n")

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda *_args, **_kwargs: None))
    result = await engine._swarm_review(
        {"fix": fix, "bug": {"diagnosis": "repair"}},
        force=True,
    )

    assert result is False
    assert recorded
    assert recorded[0][1]["receipt_required"] is True
    assert recorded[0][1]["extra"] == {"review": "swarm", "force": True}


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
async def test_safe_modification_commits_after_repair_lab_quarantine_promotion(
    tmp_path, monkeypatch
):
    _enable_repair_lab_profile(monkeypatch)
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
    safe_mod._run_promotion_harness = _AsyncCallRecorder(result=(True, "harness ok"))

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

    success, message = await safe_mod.apply_fix(
        fix,
        {
            "success": True,
            "validation": "safe_modification_harness",
            "command": "pytest tests/test_example.py",
            "validated_files": ["core/example.py"],
            "syntax_test": True,
            "import_test": True,
            "integrity_check": True,
            "unit_tests": True,
        },
    )

    assert (success, message) == (True, "Fix applied successfully")
    assert events == [
        ("backup", "value = 1\n"),
        ("branch", "value = 1\n"),
        ("commit", "value = 2\n"),
        ("merge", "value = 2\n"),
        ("delete", "value = 2\n"),
    ]
    assert target.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_safe_modification_blocks_branch_promotion_outside_repair_lab(tmp_path):
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

    events = []

    class Backup:
        def create_backup(self, _path):
            events.append("backup")
            return "backup-id"

    class Git:
        async def create_branch(self, _branch_name):
            events.append("branch")
            return True

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
    safe_mod.boot_validator = SimpleNamespace(
        validate_boot=_AsyncCallRecorder(result=(True, "ok"))
    )
    safe_mod.modification_log = tmp_path / "modifications.jsonl"
    safe_mod._run_full_test_suite = _AsyncCallRecorder(
        side_effect=AssertionError("promotion policy must block before suite execution")
    )

    fix = SimpleNamespace(
        target_file="core/example.py",
        target_line=1,
        original_code="value = 1\n",
        fixed_code="value = 2\n",
        explanation="raise value",
        risk_level=1,
        lines_changed=1,
    )

    success, message = await safe_mod.apply_fix(
        fix,
        {
            "success": True,
            "validation": "safe_modification_harness",
            "command": "pytest tests/test_example.py",
            "validated_files": ["core/example.py"],
            "syntax_test": True,
            "import_test": True,
            "integrity_check": True,
            "unit_tests": True,
        },
    )

    assert success is False
    assert "AURA_ALLOW_RUNTIME_SELF_MODIFICATION" in message
    assert "AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION" in message
    assert "AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION" in message
    assert events == []
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_safe_modification_refuses_preview_or_bare_success_evidence(tmp_path):
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

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
    safe_mod.backup = SimpleNamespace(create_backup=lambda *_args, **_kwargs: "backup-id")
    safe_mod.git = SimpleNamespace(create_branch=_AsyncCallRecorder(result=False))
    safe_mod.boot_validator = SimpleNamespace(
        validate_boot=_AsyncCallRecorder(result=(True, "ok"))
    )
    safe_mod.modification_log = tmp_path / "modifications.jsonl"

    full_suite_calls = []

    async def full_suite():
        full_suite_calls.append("attempted")
        raise AssertionError("bare or preview evidence must block before suite execution")

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
    assert success is False
    assert "validation evidence lacks command, artifact, receipt, or file proof" in message

    success, message = await safe_mod.apply_fix(
        fix,
        {"success": True, "validation": "shadow_ast_preview"},
    )
    assert success is False
    assert "shadow AST preview" in message
    assert full_suite_calls == []
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_safe_modification_blocks_no_branch_promotion_without_supervision(
    tmp_path, monkeypatch
):
    _enable_repair_lab_profile(monkeypatch)
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

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
    safe_mod.backup = SimpleNamespace(create_backup=lambda *_args, **_kwargs: "backup-id")
    safe_mod.git = SimpleNamespace(create_branch=_AsyncCallRecorder(result=False))
    safe_mod.boot_validator = SimpleNamespace(
        validate_boot=_AsyncCallRecorder(result=(True, "ok"))
    )
    safe_mod.modification_log = tmp_path / "modifications.jsonl"
    safe_mod._run_promotion_harness = _AsyncCallRecorder(result=(True, "harness ok"))

    safe_mod._run_full_test_suite = _AsyncCallRecorder(
        side_effect=AssertionError("branch policy must block before suite execution")
    )

    fix = SimpleNamespace(
        target_file="core/example.py",
        target_line=1,
        original_code="value = 1\n",
        fixed_code="value = 2\n",
        explanation="raise value",
        risk_level=1,
        lines_changed=1,
    )

    success, message = await safe_mod.apply_fix(
        fix,
        {
            "success": True,
            "validation": "safe_modification_harness",
            "command": "pytest tests/test_example.py",
            "validated_files": ["core/example.py"],
            "syntax_test": True,
            "import_test": True,
            "integrity_check": True,
            "unit_tests": True,
        },
    )

    assert success is False
    assert "clean git branch required" in message
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_safe_modification_allows_supervised_no_branch_promotion(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AURA_ALLOW_SUPERVISED_SELF_MODIFICATION", "1")
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

    class Backup:
        def create_backup(self, _path):
            return "backup-id"

        def restore_backup(self, _backup_id):
            target.write_text("value = 1\n", encoding="utf-8")
            return True

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
    safe_mod.git = SimpleNamespace(create_branch=_AsyncCallRecorder(result=False))
    safe_mod.boot_validator = SimpleNamespace(
        validate_boot=_AsyncCallRecorder(result=(True, "ok"))
    )
    safe_mod.modification_log = tmp_path / "modifications.jsonl"
    safe_mod._run_promotion_harness = _AsyncCallRecorder(result=(True, "harness ok"))
    safe_mod._run_full_test_suite = _AsyncCallRecorder(result=True)

    fix = SimpleNamespace(
        target_file="core/example.py",
        target_line=1,
        original_code="value = 1\n",
        fixed_code="value = 2\n",
        explanation="raise value",
        risk_level=1,
        lines_changed=1,
    )

    success, message = await safe_mod.apply_fix(
        fix,
        {
            "success": True,
            "validation": "safe_modification_harness",
            "command": "pytest tests/test_example.py",
            "validated_files": ["core/example.py"],
            "syntax_test": True,
            "import_test": True,
            "integrity_check": True,
            "unit_tests": True,
        },
        supervised=True,
    )

    assert (success, message) == (True, "Fix applied successfully")
    assert target.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_safe_modification_internal_harness_vetoes_caller_claimed_success(
    tmp_path, monkeypatch
):
    _enable_repair_lab_profile(monkeypatch)
    target = tmp_path / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")

    class Backup:
        def create_backup(self, _path):
            return "backup-id"

    class Git:
        async def create_branch(self, _branch_name):
            return True

        async def checkout_main(self):
            return True

        async def delete_branch(self, _branch_name):
            return True

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
    safe_mod.boot_validator = SimpleNamespace(
        validate_boot=_AsyncCallRecorder(result=(True, "ok"))
    )
    safe_mod.modification_log = tmp_path / "modifications.jsonl"
    safe_mod._run_promotion_harness = _AsyncCallRecorder(
        result=(False, "related tests missing")
    )
    safe_mod._run_full_test_suite = _AsyncCallRecorder(
        side_effect=AssertionError("internal harness must block before full suite")
    )

    fix = SimpleNamespace(
        target_file="core/example.py",
        target_line=1,
        original_code="value = 1\n",
        fixed_code="value = 2\n",
        explanation="raise value",
        risk_level=1,
        lines_changed=1,
    )

    success, message = await safe_mod.apply_fix(
        fix,
        {
            "success": True,
            "validation": "safe_modification_harness",
            "command": "pytest tests/test_example.py",
            "validated_files": ["core/example.py"],
            "syntax_test": True,
            "import_test": True,
            "integrity_check": True,
            "unit_tests": True,
        },
    )

    assert success is False
    assert "Safe modification harness failed" in message
    assert "related tests missing" in message
    assert target.read_text(encoding="utf-8") == "value = 1\n"


def test_safe_modification_static_validation_excludes_generated_artifacts(tmp_path):
    safe_mod = SafeSelfModification.__new__(SafeSelfModification)
    safe_mod.code_base = tmp_path

    runtime_source = tmp_path / "core" / "runtime.py"
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text("value = 1\n", encoding="utf-8")

    generated_artifact = tmp_path / "artifacts" / "current" / "battery" / "broken.py"
    generated_artifact.parent.mkdir(parents=True)
    generated_artifact.write_text("value = '\n", encoding="utf-8")

    assert safe_mod._validate_python_tree_parse() is True

    runtime_source.write_text("value = '\n", encoding="utf-8")

    assert safe_mod._validate_python_tree_parse() is False


def test_safe_modification_static_validation_does_not_enter_other_checkouts(tmp_path):
    """A worktree under .claude/ and a nested checkout are somebody else's tree.

    The live host carries thirty-six worktrees and a venv beside the source;
    a walk that listed them first and filtered after parsed 285,000 files per
    promotion. The walk must prune them, so a broken file there is never read.
    """
    safe_mod = SafeSelfModification.__new__(SafeSelfModification)
    safe_mod.code_base = tmp_path

    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "runtime.py").write_text("value = 1\n", encoding="utf-8")

    worktree = tmp_path / ".claude" / "worktrees" / "other" / "core"
    worktree.mkdir(parents=True)
    (worktree / "broken.py").write_text("value = '\n", encoding="utf-8")

    vendored = tmp_path / "vendor" / "checkout"
    vendored.mkdir(parents=True)
    (vendored / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    (vendored / "broken.py").write_text("value = '\n", encoding="utf-8")

    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "broken.py").write_text("value = '\n", encoding="utf-8")

    walked = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in SafeSelfModification._production_python_files(tmp_path)
    )
    assert walked == ["core/runtime.py"]
    assert safe_mod._validate_python_tree_parse() is True
