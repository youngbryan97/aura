"""Action-report honesty contracts.

An action's result must state what actually happened. These tests pin the
contracts fixed in the July 2026 depth pass:

- DesktopNotifier.send returns an honest DeliveryResult (delivered / disabled /
  suppressed_quiet_hours / failed) instead of silently returning None.
- notify_user reports delivered=False whenever the user was not reached.
- FileWriteGateway supports governed async delete/move/copy so destructive
  file operations flow through the same lane as writes.
- ActionExecutor FILE_WRITE accepts op=delete/move/copy.
- uplink_local performs a real persistence probe instead of returning a
  hardcoded "verified" string.
- environment_info detail=full returns real diagnostics, not a dead stub.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Action expectation depth contract
# ---------------------------------------------------------------------------

def test_action_expectation_downgrades_shallow_verified_result(tmp_path):
    from core.runtime.skill_contract import (
        ActionExpectation,
        SkillExecutionResult,
        SkillStatus,
        apply_action_expectation,
    )

    artifact_path = str(tmp_path / "report.pdf")
    result = SkillExecutionResult(
        skill="desktop.research_report",
        status=SkillStatus.SUCCESS_VERIFIED,
        output={"artifact_path": artifact_path},
        verification_evidence={
            "satisfied_criteria": ["artifact persisted"],
            "artifact_path": artifact_path,
        },
        expectation=ActionExpectation(
            objective="Create a sourced research report",
            acceptance_criteria=["artifact persisted", "credible sources cited"],
            required_evidence=["artifact_path"],
            user_visible_effect="report is visible to the user",
            repair_hint="return_to_browser_and_add_sources",
        ),
    )

    checked = apply_action_expectation(result)

    assert checked.status == SkillStatus.PARTIAL_SUCCESS
    assert checked.ok is False
    verdict = checked.verification_evidence["expectation_verdict"]
    assert verdict["passed"] is False
    assert "credible sources cited" in verdict["missing_criteria"]
    assert "user-visible effect: report is visible to the user" in verdict["missing_criteria"]
    assert verdict["next_step"] == "return_to_browser_and_add_sources"
    assert "expectation incomplete" in checked.failure_reason


def test_action_expectation_marks_missing_evidence_unverified(tmp_path):
    from core.runtime.skill_contract import (
        ActionExpectation,
        SkillExecutionResult,
        SkillStatus,
        apply_action_expectation,
    )

    result = SkillExecutionResult(
        skill="file.write",
        status=SkillStatus.SUCCESS_VERIFIED,
        output={"path": str(tmp_path / "a.txt")},
        verification_evidence={
            "criteria": {"file written": True},
        },
        expectation=ActionExpectation(
            objective="Write and verify a file",
            acceptance_criteria=["file written"],
            required_evidence=["sha256", "effect_verified"],
        ),
    )

    checked = apply_action_expectation(result)

    assert checked.status == SkillStatus.SUCCESS_UNVERIFIED
    verdict = checked.verification_evidence["expectation_verdict"]
    assert verdict["missing_criteria"] == []
    assert verdict["missing_evidence"] == ["sha256", "effect_verified"]
    assert verdict["next_step"] == "collect_missing_verification_evidence"


def test_action_expectation_can_require_evidence_presence_when_false_is_valid():
    from core.runtime.skill_contract import (
        ActionExpectation,
        SkillExecutionResult,
        SkillStatus,
        apply_action_expectation,
    )

    checked = apply_action_expectation(
        SkillExecutionResult(
            skill="file.exists",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"exists": False, "state": "missing"},
            expectation=ActionExpectation(
                objective="Determine whether the requested path exists",
                required_evidence_present=["exists"],
                rollback_hint="not_required_read_only",
                allow_partial=False,
            ),
        )
    )

    assert checked.ok is True
    assert checked.verification_evidence["expectation_verdict"]["passed"] is True
    assert checked.verification_evidence["action_expectation"]["rollback_hint"] == (
        "not_required_read_only"
    )


def test_action_expectation_payload_reuses_typed_verdict_semantics():
    from core.runtime.skill_contract import (
        ActionExpectation,
        apply_action_expectation_payload,
    )

    payload = apply_action_expectation_payload(
        "process_supervisor",
        {"ok": True, "updates": {"processes": []}},
        ActionExpectation(
            objective="List managed processes",
            required_evidence_present=["updates.processes"],
            rollback_hint="not_required_read_only",
            allow_partial=False,
        ),
    )

    assert payload["ok"] is True
    assert payload["status"] == "success_verified"
    assert payload["expectation_verdict"]["passed"] is True
    assert payload["action_expectation"]["required_evidence_present"] == [
        "updates.processes"
    ]


def test_skill_registry_enforces_expectation_after_verifier():
    from core.runtime.skill_contract import (
        ActionExpectation,
        SkillContract,
        SkillExecutionResult,
        SkillRegistry,
        SkillStatus,
    )

    registry = SkillRegistry()
    registry.register(SkillContract(name="browser.research", version="1.0", description=""))

    def verifier(result):
        return SkillExecutionResult(
            skill=result.skill,
            status=SkillStatus.SUCCESS_VERIFIED,
            output=result.output,
            verification_evidence={"criteria": {"browser opened": True}},
        )

    registry.register_verifier("browser.research", verifier)
    checked = registry.verify(
        SkillExecutionResult(
            skill="browser.research",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"url": "https://example.com"},
            expectation=ActionExpectation(
                objective="Research the topic and preserve sources",
                acceptance_criteria=["browser opened", "source notes preserved"],
                required_evidence=["url"],
            ),
        )
    )

    assert checked.status == SkillStatus.PARTIAL_SUCCESS
    assert checked.verification_evidence["expectation_verdict"]["missing_criteria"] == [
        "source notes preserved"
    ]


# ---------------------------------------------------------------------------
# DesktopNotifier delivery honesty
# ---------------------------------------------------------------------------

class _FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.args = ["osascript"]
        self.stdout = b""
        self.stderr = b"boom" if returncode else b""


class _FakeGateway:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return _FakeCompleted(self.returncode)


def _allow_notifications(monkeypatch, notif, enabled=True, quiet=False):
    def fake_setting(key, default=None):
        if key == "notify.enabled":
            return enabled
        if key == "notify.quiet_hours_start":
            return "00:00" if quiet else "22:00"
        if key == "notify.quiet_hours_end":
            return "23:59" if quiet else "08:00"
        return default

    monkeypatch.setattr(notif, "get_runtime_setting", fake_setting)
    if quiet:
        # Force "now" inside the window regardless of wall clock.
        monkeypatch.setattr(notif, "_within_quiet_hours", lambda *a, **k: True)
    else:
        monkeypatch.setattr(notif, "_within_quiet_hours", lambda *a, **k: False)


def test_notifier_reports_delivered_on_success(monkeypatch):
    from core.senses import notifications as notif

    _allow_notifications(monkeypatch, notif)
    gateway = _FakeGateway(returncode=0)
    monkeypatch.setattr(notif, "get_subprocess_gateway", lambda: gateway)

    result = notif.DesktopNotifier.send("Aura", "hello")
    assert result.delivered is True
    assert result.status == "delivered"
    assert bool(result) is True
    assert gateway.calls == 1


def test_notifier_reports_failed_on_subprocess_error(monkeypatch):
    from core.senses import notifications as notif

    _allow_notifications(monkeypatch, notif)
    gateway = _FakeGateway(returncode=1)
    monkeypatch.setattr(notif, "get_subprocess_gateway", lambda: gateway)

    result = notif.DesktopNotifier.send("Aura", "hello")
    assert result.delivered is False
    assert result.status == "failed"
    assert bool(result) is False


def test_notifier_reports_disabled(monkeypatch):
    from core.senses import notifications as notif

    _allow_notifications(monkeypatch, notif, enabled=False)
    gateway = _FakeGateway()
    monkeypatch.setattr(notif, "get_subprocess_gateway", lambda: gateway)

    result = notif.DesktopNotifier.send("Aura", "hello")
    assert result.delivered is False
    assert result.status == "disabled"
    assert gateway.calls == 0  # no OS call when disabled


def test_notifier_reports_quiet_hours(monkeypatch):
    from core.senses import notifications as notif

    _allow_notifications(monkeypatch, notif, quiet=True)
    gateway = _FakeGateway()
    monkeypatch.setattr(notif, "get_subprocess_gateway", lambda: gateway)

    result = notif.DesktopNotifier.send("Aura", "hello")
    assert result.delivered is False
    assert result.status == "suppressed_quiet_hours"
    assert gateway.calls == 0


# ---------------------------------------------------------------------------
# notify_user skill honesty
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_notify_user_reports_delivery(monkeypatch):
    import core.skills.notify_user as mod
    from core.senses.notifications import DeliveryResult

    monkeypatch.setattr(
        mod.DesktopNotifier,
        "send",
        staticmethod(lambda **_kw: DeliveryResult(delivered=True, status="delivered")),
    )
    result = _run(mod.NotifyUserSkill().execute({"message": "hi"}, {}))
    assert result["ok"] is True
    assert result["delivered"] is True


def test_notify_user_reports_suppression_honestly(monkeypatch):
    import core.skills.notify_user as mod
    from core.senses.notifications import DeliveryResult

    monkeypatch.setattr(
        mod.DesktopNotifier,
        "send",
        staticmethod(
            lambda **_kw: DeliveryResult(
                delivered=False,
                status="suppressed_quiet_hours",
                detail="quiet hours",
            )
        ),
    )
    result = _run(mod.NotifyUserSkill().execute({"message": "hi"}, {}))
    assert result["ok"] is True  # user preference, not a fault
    assert result["delivered"] is False
    assert "NOT delivered" in result["message"]


def test_notify_user_reports_failure(monkeypatch):
    import core.skills.notify_user as mod
    from core.senses.notifications import DeliveryResult

    monkeypatch.setattr(
        mod.DesktopNotifier,
        "send",
        staticmethod(
            lambda **_kw: DeliveryResult(delivered=False, status="failed", detail="osascript died")
        ),
    )
    result = _run(mod.NotifyUserSkill().execute({"message": "hi"}, {}))
    assert result["ok"] is False
    assert result["delivered"] is False
    assert "osascript died" in result["error"]


def test_notify_user_tolerates_legacy_none_return(monkeypatch):
    import core.skills.notify_user as mod

    monkeypatch.setattr(mod.DesktopNotifier, "send", staticmethod(lambda **_kw: None))
    result = _run(mod.NotifyUserSkill().execute({"message": "hi"}, {}))
    assert result["ok"] is True
    assert result["delivered"] is True


# ---------------------------------------------------------------------------
# Governed file delete/move/copy lane
# ---------------------------------------------------------------------------

def test_gateway_delete_move_copy_async(tmp_path):
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()

    async def scenario():
        src = tmp_path / "a.txt"
        src.write_text("payload", encoding="utf-8")

        # copy file
        copied = await gateway.copy_path_async(src, tmp_path / "b.txt", source="test")
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "payload"
        assert copied.endswith("b.txt")

        # move file
        moved = await gateway.move_path_async(src, tmp_path / "c.txt", source="test")
        assert not src.exists()
        assert (tmp_path / "c.txt").exists()
        assert moved.endswith("c.txt")

        # delete file
        assert await gateway.delete_path_async(tmp_path / "c.txt", source="test") is True
        assert not (tmp_path / "c.txt").exists()

        # delete of a missing path reports False, not an exception
        assert await gateway.delete_path_async(tmp_path / "missing.txt", source="test") is False

        # directory delete requires explicit recursive=True
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "x.txt").write_text("x", encoding="utf-8")
        with pytest.raises(IsADirectoryError):
            await gateway.delete_path_async(d, source="test")
        assert await gateway.delete_path_async(d, recursive=True, source="test") is True
        assert not d.exists()

        # move of a missing source raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            await gateway.move_path_async(tmp_path / "nope.txt", tmp_path / "z.txt", source="test")

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(scenario())


def test_gateway_copy_path_sync_is_governed_and_refuses_symlinks(
    tmp_path,
    monkeypatch,
):
    import core.runtime.file_write_gateway as gateway_module

    governed_calls = []
    monkeypatch.setattr(gateway_module, "governance_runtime_active", lambda: True)
    monkeypatch.setattr(
        gateway_module,
        "require_governance",
        lambda action, **kwargs: governed_calls.append((action, kwargs)),
    )
    gateway = gateway_module.FileWriteGateway()
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    destination = tmp_path / "destination.txt"

    copied = gateway.copy_path(source, destination, source="fusion-test")

    assert copied == str(destination)
    assert destination.read_text(encoding="utf-8") == "payload"
    assert governed_calls == [
        (
            "file_write_gateway.copy_path:fusion-test",
            {"strict": True, "allowed_domains": gateway._allowed_domains},
        )
    ]
    symlink = tmp_path / "source-link"
    symlink.symlink_to(source)
    with pytest.raises(OSError, match="symlink"):
        gateway.copy_path(symlink, tmp_path / "refused.txt", source="fusion-test")


def test_file_operation_write_returns_effect_evidence(tmp_path):
    import hashlib

    from core.skills.file_operation import FileOperationSkill

    skill = FileOperationSkill()
    skill.root_dir = str(tmp_path.resolve())

    result = _run(
        skill.execute(
            {"action": "write", "path": "verified.txt", "content": "real payload"},
            context={"origin": "unit_test"},
        )
    )

    expected_sha256 = hashlib.sha256(b"real payload").hexdigest()
    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["sha256"] == expected_sha256
    assert result["expected_sha256"] == expected_sha256
    assert result["criteria_results"]["file written"] is True
    assert (tmp_path / "verified.txt").read_text(encoding="utf-8") == "real payload"


def test_action_executor_file_ops(tmp_path):
    from core.runtime.action_executor import ActionExecutor

    async def scenario():
        src = tmp_path / "doc.txt"
        src.write_text("hello", encoding="utf-8")

        result = await ActionExecutor.execute(
            domain="file_write",
            action_name="test.copy",
            params={"op": "copy", "path": str(src), "destination": str(tmp_path / "doc2.txt")},
            source="test_suite",
        )
        assert result.get("ok") is True
        assert (tmp_path / "doc2.txt").exists()

        result = await ActionExecutor.execute(
            domain="file_write",
            action_name="test.move",
            params={"op": "move", "path": str(tmp_path / "doc2.txt"), "destination": str(tmp_path / "doc3.txt")},
            source="test_suite",
        )
        assert result.get("ok") is True
        assert not (tmp_path / "doc2.txt").exists()

        result = await ActionExecutor.execute(
            domain="file_write",
            action_name="test.delete",
            params={"op": "delete", "path": str(tmp_path / "doc3.txt")},
            source="test_suite",
        )
        assert result.get("ok") is True
        assert result.get("deleted") is True
        assert not (tmp_path / "doc3.txt").exists()

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(scenario())


# ---------------------------------------------------------------------------
# uplink_local: a real persistence probe
# ---------------------------------------------------------------------------

def test_uplink_fails_without_state_repository(monkeypatch, tmp_path):
    from core.container import ServiceContainer
    from core.skills.uplink_local import UplinkSkill

    previous = ServiceContainer.get("state_repository", default=None)
    if previous is not None:
        monkeypatch.setattr(
            UplinkSkill,
            "_check_state_repository",
            staticmethod(lambda checks: checks.update(
                {"state_repository": {"ok": False, "evidence": "forced for test"}}
            ) or False),
        )
        result = _run(UplinkSkill().execute({}, {}))
        assert result["ok"] is False
        return

    result = _run(UplinkSkill().execute({}, {}))
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["checks"]["state_repository"]["ok"] is False


def test_uplink_verifies_with_healthy_repo_and_disk(monkeypatch, tmp_path):
    import time as _time

    import core.config as config_mod
    from core.container import ServiceContainer
    from core.skills.uplink_local import UplinkSkill

    fake_repo = SimpleNamespace(
        get_runtime_status=lambda: {
            "state_available": True,
            "consumer_alive": True,
            "is_vault_owner": True,
            "db_connected": True,
            "dropped_commit_count": 0,
            "last_commit_at": _time.time(),
            "current_version": 42,
            "queue_depth": 0,
        }
    )
    ServiceContainer.register_instance("state_repository", fake_repo, required=False)
    try:
        monkeypatch.setattr(
            type(config_mod.config.paths), "data_dir", property(lambda self: tmp_path)
        )
        result = _run(UplinkSkill().execute({}, {}))
        assert result["ok"] is True, result
        assert result["status"] == "verified"
        assert result["checks"]["state_repository"]["ok"] is True
        assert result["checks"]["disk_round_trip"]["ok"] is True
        # probe file must not be left behind
        assert not (tmp_path / ".persistence_probe").exists()
    finally:
        ServiceContainer.register_instance("state_repository", None, required=False)


def test_uplink_reports_stale_commits(monkeypatch, tmp_path):
    from core.container import ServiceContainer
    from core.skills.uplink_local import UplinkSkill

    fake_repo = SimpleNamespace(
        get_runtime_status=lambda: {
            "state_available": True,
            "consumer_alive": True,
            "is_vault_owner": True,
            "db_connected": True,
            "dropped_commit_count": 0,
            "last_commit_at": 1.0,  # 1970 → hopelessly stale
        }
    )
    ServiceContainer.register_instance("state_repository", fake_repo, required=False)
    try:
        checks: dict = {}
        ok = UplinkSkill._check_state_repository(checks)
        assert ok is False
        assert "last commit" in checks["state_repository"]["evidence"]
    finally:
        ServiceContainer.register_instance("state_repository", None, required=False)


# ---------------------------------------------------------------------------
# environment_info full detail is real
# ---------------------------------------------------------------------------

def test_environment_info_full_detail_returns_real_diagnostics():
    from core.skills.environment_info import EnvironmentSkill

    result = _run(
        EnvironmentSkill().execute({"params": {"detail": "full"}}, {})
    )
    assert result["ok"] is True
    info = result["result"]
    assert "memory" in info, info
    assert info["memory"]["total_gb"] > 0
    assert "disk" in info
    assert info["disk"]["total_gb"] > 0
    assert "uptime_hours" in info


def test_semantic_predicates_distinguish_false_from_unmeasured() -> None:
    from core.runtime.skill_contract import (
        ActionExpectation,
        PredicateOperator,
        SemanticPredicate,
        SkillExecutionResult,
        SkillStatus,
        apply_action_expectation,
    )

    checked = apply_action_expectation(
        SkillExecutionResult(
            skill="research_report",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"evidence": {"read_count": 2}},
            expectation=ActionExpectation(
                objective="read three sources and write a synthesis",
                semantic_predicates=[
                    SemanticPredicate(
                        predicate_id="three_sources_read",
                        evidence_path="evidence.read_count",
                        operator=PredicateOperator.GREATER_THAN_OR_EQUAL,
                        expected=3,
                        repair_hint="read_one_more_source",
                    ),
                    SemanticPredicate(
                        predicate_id="synthesis_present",
                        evidence_path="evidence.synthesis",
                        operator=PredicateOperator.NONEMPTY_TEXT,
                        repair_hint="write_synthesis",
                    ),
                ],
            ),
        )
    )

    verdict = checked.verification_evidence["expectation_verdict"]
    assert checked.status == SkillStatus.PARTIAL_SUCCESS
    assert verdict["unsatisfied_predicates"] == ["three_sources_read"]
    assert verdict["unknown_predicates"] == ["synthesis_present"]
    assert verdict["repair_steps"] == ["read_one_more_source", "write_synthesis"]
    assert [item["state"] for item in verdict["predicate_results"]] == [
        "unsatisfied",
        "unknown",
    ]


def test_semantic_predicate_mapping_is_closed_and_serializable() -> None:
    from core.runtime.skill_contract import ActionExpectation, semantic_predicate_from_mapping

    predicate = semantic_predicate_from_mapping(
        {
            "id": "artifact_saved",
            "path": "artifacts.pdf_count",
            "operator": "gte",
            "expected": 1,
        }
    )
    payload = ActionExpectation(semantic_predicates=[predicate]).to_dict()

    assert payload["semantic_predicates"] == [
        {
            "predicate_id": "artifact_saved",
            "evidence_path": "artifacts.pdf_count",
            "operator": "gte",
            "expected": 1,
            "description": "",
            "repair_hint": "",
            "required": True,
        }
    ]

    with pytest.raises(ValueError, match="unsupported semantic predicate operator"):
        semantic_predicate_from_mapping(
            {"id": "unsafe", "path": "x", "operator": "python_eval"}
        )
