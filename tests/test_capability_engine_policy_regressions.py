import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.capability_engine import (
    _OUTER_TIMEOUT_GRACE_S,
    _SKILL_TIMEOUT_CONTEXT_KEY,
    CapabilityEngine,
    SkillMetadata,
)
from core.container import ServiceContainer
from core.guardians.user_advocate import UserAdvocateWatchdog
from core.sim.outcome_simulator import OutcomeSimulationEngine


def _quiet_logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )


def _engine_with_skill(skill_name: str, *, metabolic_cost: int = 1) -> CapabilityEngine:
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = _quiet_logger()
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        skill_name: SkillMetadata(
            name=skill_name,
            description="policy regression probe",
            skill_class=lambda: object(),
            metabolic_cost=metabolic_cost,
        )
    }
    engine.instances = {}
    engine.sandbox = None
    engine.rosetta_stone = None
    engine.temporal = None
    engine.orchestrator = SimpleNamespace(mycelium=None)
    engine.skill_last_errors = {}
    engine._emit_skill_status = lambda *args, **kwargs: None
    engine.max_retries = 1
    engine.retry_delay = 0.0
    engine.timeout = 1.0
    return engine


def test_stateless_sandbox_compute_is_not_irreversible_for_user_advocate():
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "run_code",
            {"code": "print(120)", "stateful": False},
            "high",
            "sandboxed_compute",
        )
        is False
    )


def test_stateful_code_still_requires_irreversible_confirmation():
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "run_code",
            {"code": "x = 1", "stateful": True},
            "critical",
            "sandboxed_compute",
        )
        is True
    )


def test_web_search_execution_scope_is_read_only_not_external_mutation():
    engine = _engine_with_skill("web_search")
    meta = engine.skills["web_search"]

    scope = engine._effect_scope_for_execution(
        "web_search",
        meta,
        {"query": "latest climate research"},
        {"origin": "background"},
    )

    assert scope == "read_only"
    assert engine._edi_risk_for("web_search", meta, {"query": "latest climate research"}, scope) == "low"
    description = CapabilityEngine._action_description_for_user_advocate(
        "web_search",
        {"query": "latest climate research"},
        scope,
    )
    assert "read-only web_search information retrieval" in description


def test_autonomous_web_search_is_safe_read_only_research_with_user_benefit():
    engine = _engine_with_skill("web_search")
    meta = engine.skills["web_search"]
    params = {"query": "latest climate research"}
    ctx = {
        "origin": "curiosity_explorer",
        "objective": "Curiosity-driven search: latest climate research",
    }
    scope = engine._effect_scope_for_execution("web_search", meta, params, ctx)

    assert scope == "read_only"
    assert engine._edi_risk_for("web_search", meta, params, scope) == "low"
    assert CapabilityEngine._safe_autonomous_web_research(
        "web_search",
        params,
        ctx,
        "curiosity_explorer",
        scope,
    )
    benefit = CapabilityEngine._user_benefit_for_execution(
        "web_search",
        params,
        ctx,
        "curiosity_explorer",
        scope,
    )
    assert "autonomous curiosity" in benefit
    review = UserAdvocateWatchdog().review_action(
        {
            "description": CapabilityEngine._action_description_for_user_advocate(
                "web_search",
                params,
                scope,
            ),
            "irreversible": CapabilityEngine._user_advocate_irreversible_for(
                "web_search",
                params,
                "low",
                scope,
            ),
            "confirmed": CapabilityEngine._safe_autonomous_web_research(
                "web_search",
                params,
                ctx,
                "curiosity_explorer",
                scope,
            ),
            "user_benefit": benefit,
            "explanation": "skill web_search",
        }
    )
    assert review.verdict == "for_user"


def test_latent_cortex_has_bounded_autonomous_web_research_authority():
    params = {"query": "current compiler release notes"}
    context = {
        "origin": "latent_cortex",
        "objective": "Identify the current compiler behavior.",
        "foreground_request": True,
        "foreground_cognitive_acquisition": True,
    }

    assert CapabilityEngine._safe_autonomous_web_research(
        "web_search",
        params,
        context,
        "latent_cortex",
        "read_only",
    )

    from core.orchestrator.mixins.tool_execution import ToolExecutionMixin

    assert ToolExecutionMixin._safe_autonomous_web_research_tool(
        "web_search",
        params,
        "latent_cortex",
        context,
    )


def test_low_phi_telemetry_does_not_block_bounded_capability(monkeypatch):
    ServiceContainer.clear()
    engine = _engine_with_skill("computer_use", metabolic_cost=2)
    meta = engine.skills["computer_use"]
    ServiceContainer.register_instance(
        "state_repository",
        SimpleNamespace(_current=SimpleNamespace(phi=0.001)),
        required=False,
    )
    monkeypatch.setattr(
        "core.capability_engine.resolve_metabolic_monitor",
        lambda default=None: SimpleNamespace(
            get_current_metabolism=lambda: SimpleNamespace(
                health_score=0.95,
                cpu_percent=15.0,
                ram_percent=40.0,
            )
        ),
    )

    try:
        reason, unbounded = engine._self_preservation_block_reason(
            meta,
            "computer_use",
            {"task": "Open Notes and type Hello"},
            {"origin": "desktop_ui", "foreground_request": True},
        )
    finally:
        ServiceContainer.clear()

    assert reason == ""
    assert unbounded is False


def test_soft_metabolic_pressure_constrains_bounded_foreground_desktop_action(monkeypatch):
    engine = _engine_with_skill("desktop_task", metabolic_cost=3)
    meta = engine.skills["desktop_task"]
    monkeypatch.setattr(
        "core.capability_engine.resolve_metabolic_monitor",
        lambda default=None: SimpleNamespace(
            get_current_metabolism=lambda: SimpleNamespace(
                health_score=0.35,
                cpu_percent=35.0,
                ram_percent=62.0,
            )
        ),
    )

    reason, unbounded = engine._self_preservation_block_reason(
        meta,
        "desktop_task",
        {"objective": "Open Notes and write Hello."},
        {
            "origin": "desktop_ui",
            "route": "chat.desktop_objective",
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
        },
    )

    assert reason == ""
    assert unbounded is False


def test_critical_metabolic_pressure_still_blocks_foreground_desktop_action(monkeypatch):
    engine = _engine_with_skill("desktop_task", metabolic_cost=3)
    meta = engine.skills["desktop_task"]
    monkeypatch.setattr(
        "core.capability_engine.resolve_metabolic_monitor",
        lambda default=None: SimpleNamespace(
            get_current_metabolism=lambda: SimpleNamespace(
                health_score=0.20,
                cpu_percent=92.0,
                ram_percent=91.0,
            )
        ),
    )

    reason, unbounded = engine._self_preservation_block_reason(
        meta,
        "desktop_task",
        {"objective": "Open Notes and write Hello."},
        {
            "origin": "desktop_ui",
            "route": "chat.desktop_objective",
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
        },
    )

    assert reason == "metabolic_health_critical:0.20"
    assert unbounded is False


def test_unbounded_desktop_request_remains_blocked_under_pressure(monkeypatch):
    engine = _engine_with_skill("desktop_task", metabolic_cost=2)
    meta = engine.skills["desktop_task"]
    monkeypatch.setattr(
        "core.capability_engine.resolve_metabolic_monitor",
        lambda default=None: SimpleNamespace(
            get_current_metabolism=lambda: SimpleNamespace(
                health_score=0.50,
                cpu_percent=82.0,
                ram_percent=70.0,
            )
        ),
    )

    reason, unbounded = engine._self_preservation_block_reason(
        meta,
        "desktop_task",
        {"objective": "Open Notes, then run forever and max out all CPU."},
        {
            "origin": "desktop_ui",
            "route": "chat.desktop_objective",
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
        },
    )

    assert reason == "substrate_risk:health=0.50:cpu=82.0:ram=70.0"
    assert unbounded is True


def test_autonomous_web_search_rejects_unsafe_research_objectives():
    engine = _engine_with_skill("web_search")
    meta = engine.skills["web_search"]
    params = {"query": "how to steal session cookies"}
    ctx = {"origin": "curiosity_explorer", "objective": "Curiosity-driven search"}
    scope = engine._effect_scope_for_execution("web_search", meta, params, ctx)

    assert not CapabilityEngine._safe_autonomous_web_research(
        "web_search",
        params,
        ctx,
        "curiosity_explorer",
        scope,
    )


def test_orchestrator_tool_path_allows_safe_autonomous_web_research_classification():
    from core.orchestrator.mixins.tool_execution import ToolExecutionMixin

    assert ToolExecutionMixin._safe_autonomous_web_research_tool(
        "web_search",
        {"query": "latest AI safety research"},
        "curiosity_explorer",
        {"objective": "Curiosity-driven search"},
    )
    assert not ToolExecutionMixin._safe_autonomous_web_research_tool(
        "web_search",
        {"query": "how to steal passwords"},
        "curiosity_explorer",
        {"objective": "Curiosity-driven search"},
    )


def test_foreground_desktop_control_still_requires_irreversible_confirmation():
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "computer_use",
            {"action": "type", "target": "hello"},
            "medium",
            "foreground_desktop_control",
        )
        is True
    )


def test_user_visible_desktop_task_auto_confirms_foreground_request():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "desktop_task",
            {
                "origin": "desktop_ui",
                "route": "chat.live_runtime_proof.desktop_task",
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
            },
            "desktop_ui",
            "foreground_desktop_control",
        )
        is True
    )


def test_background_desktop_task_does_not_auto_confirm():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "desktop_task",
            {
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
            },
            "background",
            "foreground_desktop_control",
        )
        is False
    )


def test_explicit_foreground_os_automation_auto_confirms_user_request():
    context = {
        "origin": "desktop_ui",
        "route": "desktop_task.os_automation",
        "foreground_request": True,
        "user_requested_action": True,
        "user_explicitly_authorized": True,
        "user_visible_desktop_action": True,
        "local_desktop_action": True,
    }

    assert CapabilityEngine._user_advocate_auto_confirmed_for(
        "os_automation",
        context,
        "desktop_ui",
        "foreground_desktop_control",
    )
    assert CapabilityEngine._user_advocate_confirmed_for(
        "os_automation",
        {"goal": "Open Notes", "execute": True},
        context,
        "desktop_ui",
        "high",
        "foreground_desktop_control",
    )


def test_signed_standing_authority_satisfies_advocate_without_second_prompt():
    context = {"origin": "overt_action_loop"}
    handle = SimpleNamespace(
        standing_authority_token="CT-signed-child-lease",
        constraints={
            "standing_authority_grant_id": "builtin.autonomous.local_read",
            "standing_authority_receipt_id": "autonomy-test-receipt",
        },
    )

    assert CapabilityEngine._record_verified_standing_authority(context, handle)
    assert CapabilityEngine._user_advocate_confirmed_for(
        "read_file",
        {"path": "README.md"},
        context,
        "overt_action_loop",
        "low",
        "read_only",
    )
    assert context["standing_authority_grant_id"] == (
        "builtin.autonomous.local_read"
    )
    assert "standing_authority_token" not in context


def test_raw_standing_authority_context_cannot_spoof_advocate_confirmation():
    context = {
        "origin": "overt_action_loop",
        "_standing_authority_verified": True,
        "standing_authority_token": "caller-controlled",
        "standing_authority_grant_id": "builtin.autonomous.local_read",
    }

    assert not CapabilityEngine._user_advocate_confirmed_for(
        "os_automation",
        {"goal": "Open Notes", "execute": True},
        context,
        "overt_action_loop",
        "high",
        "foreground_desktop_control",
    )


@pytest.mark.parametrize(
    ("context", "source", "risk"),
    [
        (
            {
                "route": "desktop_task.os_automation",
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "user_visible_desktop_action": True,
            },
            "background",
            "high",
        ),
        (
            {
                "route": "desktop_task.os_automation",
                "foreground_request": True,
                "user_visible_desktop_action": True,
            },
            "desktop_ui",
            "high",
        ),
        (
            {
                "route": "chat.desktop_action",
                "foreground_request": True,
                "user_visible_desktop_action": True,
            },
            "desktop_ui",
            "low",
        ),
    ],
)
def test_os_automation_does_not_auto_confirm_without_full_user_authority(
    context,
    source,
    risk,
):
    assert not CapabilityEngine._user_advocate_confirmed_for(
        "os_automation",
        {"goal": "Open Notes", "execute": True},
        context,
        source,
        risk,
        "foreground_desktop_control",
    )


def test_capability_engine_keeps_authority_out_of_tool_arguments():
    source = (Path(__file__).resolve().parents[1] / "core" / "capability_engine.py").read_text(
        encoding="utf-8"
    )

    gate_start = source.index(
        "constitutional_args = canonical_authority_arguments(skill_name, params)"
    )
    gate_end = source.index("tool_handle = await constitution.begin_tool_execution", gate_start)
    argument_bridge = source[gate_start:gate_end]

    assert "for context_key" not in argument_bridge
    assert "scoped_authority" not in argument_bridge
    assert "standing_authority_token" not in argument_bridge


def test_user_visible_web_interlocutor_auto_confirms_foreground_request():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "web_interlocutor",
            {
                "origin": "desktop_ui",
                "route": "chat.live_runtime_proof.web_interlocutor",
                "foreground_request": True,
                "user_requested_action": True,
                "user_visible_browser_action": True,
            },
            "desktop_ui",
            "foreground_browser_dialogue",
        )
        is True
    )


def test_background_web_interlocutor_does_not_auto_confirm():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "web_interlocutor",
            {
                "foreground_request": True,
                "user_requested_action": True,
                "user_visible_browser_action": True,
            },
            "background",
            "foreground_browser_dialogue",
        )
        is False
    )


@pytest.mark.asyncio
async def test_execute_with_retry_uses_skill_execution_timeout(monkeypatch):
    engine = _engine_with_skill("web_interlocutor")
    engine.max_retries = 1
    engine.timeout = 1.0

    observed_budgets = []

    class SlowVisibleSkill:
        async def safe_execute(self, params, context):
            observed_budgets.append(context.get(_SKILL_TIMEOUT_CONTEXT_KEY))
            return {"ok": True, "status": "completed"}

    observed_timeouts = []

    async def fake_wait_for(coro, timeout):  # noqa: ASYNC109 - asyncio API double
        observed_timeouts.append(timeout)
        return await coro

    monkeypatch.setattr("core.capability_engine.asyncio.wait_for", fake_wait_for)

    result = await engine._execute_with_retry(
        SlowVisibleSkill(),
        "web_interlocutor",
        {},
        {},
        execution_timeout=420.0,
    )

    assert result["ok"] is True
    # The execution timeout drives the wait, not engine.timeout (1.0s).
    assert observed_timeouts == [420.0 + _OUTER_TIMEOUT_GRACE_S]
    # ...and the skill is told the budget, so its own timeout — the one whose
    # failure path still carries the step receipts — is what fires first.
    assert observed_budgets == [420.0]
    assert observed_timeouts[0] > observed_budgets[0]


@pytest.mark.asyncio
async def test_execute_with_retry_reports_blank_timeout_with_skill_context(monkeypatch):
    engine = _engine_with_skill("web_interlocutor")
    engine.max_retries = 1
    engine.timeout = 1.0

    class TimingOutVisibleSkill:
        async def safe_execute(self, params, context):
            return {"ok": True}

    async def fake_wait_for(coro, timeout):  # noqa: ASYNC109 - asyncio API double
        coro.close()
        raise TimeoutError()

    monkeypatch.setattr("core.capability_engine.asyncio.wait_for", fake_wait_for)

    result = await engine._execute_with_retry(
        TimingOutVisibleSkill(),
        "web_interlocutor",
        {},
        {},
        execution_timeout=420.0,
    )

    assert result["ok"] is False
    assert result["error"] == "web_interlocutor timed out after 420.0s"


@pytest.mark.asyncio
async def test_execute_with_retry_downgrades_shallow_action_expectation():
    engine = _engine_with_skill("browser.research")
    engine.max_retries = 1

    class ShallowResearchSkill:
        async def safe_execute(self, params, context):
            return {
                "ok": True,
                "status": "completed",
                "url": "https://example.com",
                "criteria": {"browser opened": True},
            }

    result = await engine._execute_with_retry(
        ShallowResearchSkill(),
        "browser.research",
        {"topic": "runtime reliability"},
        {
            "action_expectation": {
                "objective": "Research and preserve sources",
                "acceptance_criteria": ["browser opened", "source notes preserved"],
                "required_evidence": ["url"],
                "repair_hint": "capture_sources_before_reporting_done",
            }
        },
    )

    assert result["ok"] is False
    assert result["status"] == "partial_success"
    assert result["error"].startswith("expectation incomplete")
    verdict = result["expectation_verdict"]
    assert verdict["missing_criteria"] == ["source notes preserved"]
    assert verdict["present_evidence"] == ["url"]
    assert verdict["next_step"] == "capture_sources_before_reporting_done"


@pytest.mark.asyncio
async def test_execute_with_retry_marks_missing_expectation_evidence_unverified():
    engine = _engine_with_skill("file.write")
    engine.max_retries = 1

    class FileWriteSkill:
        async def safe_execute(self, params, context):
            return {
                "ok": True,
                "status": "completed",
                "path": str(params["path"]),
                "criteria_results": {"file written": True},
            }

    result = await engine._execute_with_retry(
        FileWriteSkill(),
        "file.write",
        {"path": "workspace-note.txt"},
        {
            "acceptance_criteria": ["file written"],
            "required_evidence": ["sha256", "effect_verified"],
        },
    )

    assert result["ok"] is False
    assert result["status"] == "success_unverified"
    assert result["expectation_verdict"]["missing_evidence"] == [
        "sha256",
        "effect_verified",
    ]


@pytest.mark.asyncio
async def test_expectation_downgrade_emits_durable_receipt_and_fault(monkeypatch, tmp_path):
    from core.runtime.receipts import get_receipt_store, reset_receipt_store

    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    fault_records = []

    class FaultRegistryStub:
        def record_fault(self, fault_id, subsystem, **kwargs):
            fault_records.append((fault_id, subsystem, kwargs))

    monkeypatch.setattr(
        "core.resilience.fault_taxonomy.get_fault_registry",
        lambda: FaultRegistryStub(),
    )

    engine = _engine_with_skill("file.write")
    engine.max_retries = 1

    class FileWriteSkill:
        async def safe_execute(self, params, context):
            return {
                "ok": True,
                "status": "completed",
                "criteria_results": {"file written": True},
            }

    try:
        result = await engine._execute_with_retry(
            FileWriteSkill(),
            "file.write",
            {"path": "workspace-note.txt"},
            {
                "action_expectation": {
                    "objective": "write and verify a file",
                    "acceptance_criteria": ["file written"],
                    "required_evidence": ["sha256"],
                    "repair_hint": "hash_file_before_reporting_done",
                }
            },
        )

        assert result["ok"] is False
        assert result["status"] == "success_unverified"
        assert result["expectation_receipt_id"]
        assert (
            result["verification_evidence"]["expectation_receipt_id"]
            == result["expectation_receipt_id"]
        )

        receipts = store.query_by_kind("tool_execution")
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.receipt_id == result["expectation_receipt_id"]
        assert receipt.tool == "file.write"
        assert receipt.status == "success_unverified"
        assert receipt.metadata["source"] == "capability_engine.action_expectation"
        assert receipt.verification_evidence["expectation_verdict"]["missing_evidence"] == [
            "sha256"
        ]

        assert fault_records
        fault_id, subsystem, kwargs = fault_records[0]
        assert fault_id == "PASSF-ACTION-SHALLOW-SUCCESS"
        assert subsystem == "capability_engine"
        assert kwargs["recovered"] is True
        assert kwargs["recovery_time_s"] == 0.0
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_action_expectation_receipt_persists_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    persistence_threads = []

    class ReceiptStoreStub:
        @staticmethod
        def emit(receipt):
            persistence_threads.append(threading.get_ident())
            receipt.receipt_id = "tool-execution-off-loop"
            return receipt

    monkeypatch.setattr(
        "core.runtime.receipts.get_receipt_store",
        lambda: ReceiptStoreStub(),
    )

    receipt_id = await CapabilityEngine._emit_action_expectation_receipt(
        "web_search",
        {
            "ok": True,
            "status": "success_verified",
            "verification_evidence": {
                "expectation_verdict": {"passed": True, "next_step": ""},
            },
        },
        SimpleNamespace(
            objective="read and verify public sources",
            rollback_hint="not_required_read_only",
        ),
    )

    assert receipt_id
    assert persistence_threads and persistence_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_auto_file_operation_expectation_rejects_shallow_mutation(monkeypatch, tmp_path):
    from core.runtime.receipts import get_receipt_store, reset_receipt_store

    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    fault_records = []

    class FaultRegistryStub:
        def record_fault(self, fault_id, subsystem, **kwargs):
            fault_records.append((fault_id, subsystem, kwargs))

    monkeypatch.setattr(
        "core.resilience.fault_taxonomy.get_fault_registry",
        lambda: FaultRegistryStub(),
    )

    engine = _engine_with_skill("file_operation")

    class ShallowFileSkill:
        async def safe_execute(self, params, context):
            return {"ok": True, "status": "completed", "path": params["path"]}

    try:
        result = await engine._execute_with_retry(
            ShallowFileSkill(),
            "file_operation",
            {"action": "write", "path": "shallow.txt", "content": "x"},
            {"origin": "user"},
        )

        assert result["ok"] is False
        assert result["status"] == "failed_recoverable"
        assert result["expectation_verdict"]["missing_criteria"] == [
            "file written",
            "user-visible effect: filesystem write is observable and verified",
        ]
        assert result["expectation_verdict"]["missing_evidence"] == [
            "sha256",
            "effect_verified",
        ]
        assert result["action_expectation"]["rollback_hint"] == (
            "restore_previous_file_version_or_delete_new_file"
        )
        assert result["expectation_receipt_id"]
        assert any(
            fault_id == "PASSF-ACTION-SHALLOW-SUCCESS"
            for fault_id, _subsystem, _kwargs in fault_records
        )
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_auto_file_operation_expectation_ignores_read_only_actions():
    engine = _engine_with_skill("file_operation")

    class ReadFileSkill:
        async def safe_execute(self, params, context):
            return {"ok": True, "content": "hello", "path": params["path"]}

    result = await engine._execute_with_retry(
        ReadFileSkill(),
        "file_operation",
        {"action": "read", "path": "note.txt"},
        {"origin": "user"},
    )

    assert result["ok"] is True
    assert "expectation_verdict" not in result


@pytest.mark.asyncio
async def test_auto_memory_ops_expectation_rejects_shallow_core_append(monkeypatch, tmp_path):
    from core.runtime.receipts import get_receipt_store, reset_receipt_store

    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    fault_records = []

    class FaultRegistryStub:
        def record_fault(self, fault_id, subsystem, **kwargs):
            fault_records.append((fault_id, subsystem, kwargs))

    monkeypatch.setattr(
        "core.resilience.fault_taxonomy.get_fault_registry",
        lambda: FaultRegistryStub(),
    )

    engine = _engine_with_skill("memory_ops")

    class ShallowMemorySkill:
        async def safe_execute(self, params, context):
            return {"ok": True, "summary": "Appended."}

    try:
        result = await engine._execute_with_retry(
            ShallowMemorySkill(),
            "memory_ops",
            {"action": "core_append", "block": "user", "content": "remember this"},
            {"origin": "user"},
        )

        assert result["ok"] is False
        assert result["status"] == "failed_recoverable"
        assert result["expectation_verdict"]["missing_criteria"] == [
            "core memory appended",
            "user-visible effect: core memory append is persisted and verified",
        ]
        assert result["expectation_verdict"]["missing_evidence"] == [
            "block",
            "sha256",
            "effect_verified",
        ]
        assert result["action_expectation"]["rollback_hint"] == (
            "restore_previous_core_memory_block"
        )
        assert result["expectation_receipt_id"]
        assert any(
            fault_id == "PASSF-ACTION-SHALLOW-SUCCESS"
            for fault_id, _subsystem, _kwargs in fault_records
        )
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_auto_memory_ops_expectation_rejects_receiptless_archival_insert(
    monkeypatch,
    tmp_path,
):
    from core.runtime.receipts import get_receipt_store, reset_receipt_store

    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    fault_records = []

    class FaultRegistryStub:
        def record_fault(self, fault_id, subsystem, **kwargs):
            fault_records.append((fault_id, subsystem, kwargs))

    monkeypatch.setattr(
        "core.resilience.fault_taxonomy.get_fault_registry",
        lambda: FaultRegistryStub(),
    )

    engine = _engine_with_skill("memory_ops")

    class ReceiptlessMemorySkill:
        async def safe_execute(self, params, context):
            return {
                "ok": True,
                "status": "success_unverified",
                "summary": "Committed to archival storage.",
            }

    try:
        result = await engine._execute_with_retry(
            ReceiptlessMemorySkill(),
            "memory_ops",
            {"action": "remember", "content": "retain this for future sessions"},
            {"origin": "user"},
        )

        assert result["ok"] is False
        assert result["status"] == "failed_recoverable"
        assert result["expectation_verdict"]["missing_criteria"] == [
            "archival memory stored",
            "user-visible effect: archival memory write is durable and receipt-backed",
        ]
        assert result["expectation_verdict"]["missing_evidence"] == [
            "record_id",
            "memory_receipt_id",
            "bytes_written",
            "content_sha256",
            "effect_verified",
        ]
        assert result["expectation_verdict"]["next_step"] == (
            "retry_archival_insert_through_memory_write_gateway"
        )
        assert result["action_expectation"]["rollback_hint"] == (
            "tombstone_or_restore_archival_memory_record"
        )
        assert result["expectation_receipt_id"]
        assert any(
            fault_id == "PASSF-ACTION-SHALLOW-SUCCESS"
            for fault_id, _subsystem, _kwargs in fault_records
        )
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_auto_web_search_expectation_rejects_sourceless_research(monkeypatch, tmp_path):
    from core.runtime.receipts import get_receipt_store, reset_receipt_store

    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    fault_records = []

    class FaultRegistryStub:
        def record_fault(self, fault_id, subsystem, **kwargs):
            fault_records.append((fault_id, subsystem, kwargs))

    monkeypatch.setattr(
        "core.resilience.fault_taxonomy.get_fault_registry",
        lambda: FaultRegistryStub(),
    )

    engine = _engine_with_skill("web_search")

    class ShallowWebSearchSkill:
        async def safe_execute(self, params, context):
            return {"ok": True, "summary": "Recent research says yes."}

    try:
        result = await engine._execute_with_retry(
            ShallowWebSearchSkill(),
            "web_search",
            {"query": "latest research on Europa ocean chemistry"},
            {"origin": "user"},
        )

        assert result["ok"] is False
        assert result["status"] == "success_unverified"
        assert result["expectation_verdict"]["missing_evidence"] == ["sources"]
        assert result["expectation_verdict"]["next_step"] == "rerun_web_research_with_sources"
        assert result["action_expectation"]["rollback_hint"] == (
            "not_required_read_only"
        )
        assert result["expectation_receipt_id"]
        assert any(
            fault_id == "PASSF-ACTION-SHALLOW-SUCCESS"
            for fault_id, _subsystem, _kwargs in fault_records
        )
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_auto_web_search_expectation_ignores_non_source_lookup():
    engine = _engine_with_skill("web_search")

    class SimpleWebSearchSkill:
        async def safe_execute(self, params, context):
            return {"ok": True, "summary": "Four."}

    result = await engine._execute_with_retry(
        SimpleWebSearchSkill(),
        "web_search",
        {"query": "what is two plus two"},
        {"origin": "user"},
    )

    assert result["ok"] is True
    assert "expectation_verdict" not in result


def test_auto_refactor_scan_is_read_only_not_privileged_mutation():
    engine = _engine_with_skill("auto_refactor")
    meta = engine.skills["auto_refactor"]

    assert (
        engine._effect_scope_for_execution(
            "auto_refactor",
            meta,
            {"path": ".", "run_tests": False},
            {"origin": "overt_action_loop"},
        )
        == "read_only"
    )
    assert (
        engine._edi_risk_for(
            "auto_refactor",
            meta,
            {"path": ".", "run_tests": False},
            "read_only",
        )
        == "low"
    )
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "auto_refactor",
            {"path": ".", "run_tests": False},
            "low",
            "read_only",
        )
        is False
    )


def test_auto_refactor_read_only_scan_presents_user_benefit_to_guardian():
    params = {"path": ".", "run_tests": False}
    desc = CapabilityEngine._action_description_for_user_advocate(
        "auto_refactor",
        params,
        "read_only",
    )
    benefit = CapabilityEngine._user_benefit_for_execution(
        "auto_refactor",
        params,
        {"origin": "overt_action_loop"},
        "overt_action_loop",
        "read_only",
    )

    review = UserAdvocateWatchdog().review_action(
        {
            "description": desc,
            "irreversible": CapabilityEngine._user_advocate_irreversible_for(
                "auto_refactor",
                params,
                "low",
                "read_only",
            ),
            "confirmed": False,
            "user_benefit": benefit,
            "explanation": "skill auto_refactor",
        }
    )

    assert "read-only" in desc
    assert "no source writes" in desc
    assert benefit
    assert review.verdict == "for_user"
    assert review.flags == []


def test_outcome_simulator_allows_read_only_external_web_search():
    result = OutcomeSimulationEngine().assess_fast(
        "web_search [read_only_external_io] {'query': 'latest research on Europa'}",
        context={
            "effect_scope": "read_only_external_io",
            "skill_name": "web_search",
            "tool_name": "web_search",
        },
    )

    assert result.recommendation == "act"
    assert result.worst_case_harm < OutcomeSimulationEngine.HOLD_HARM_THRESHOLD


def test_auto_refactor_mutation_remains_privileged():
    engine = _engine_with_skill("auto_refactor")
    meta = engine.skills["auto_refactor"]

    scope = engine._effect_scope_for_execution(
        "auto_refactor",
        meta,
        {"path": ".", "apply": True},
        {"origin": "overt_action_loop"},
    )

    assert scope == "privileged_mutation"
    assert (
        engine._edi_risk_for(
            "auto_refactor",
            meta,
            {"path": ".", "apply": True},
            scope,
        )
        == "critical"
    )
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "auto_refactor",
            {"path": ".", "apply": True},
            "critical",
            scope,
        )
        is True
    )


@pytest.mark.asyncio
async def test_foreground_exclusive_background_tool_defers_when_policy_fails(monkeypatch):
    engine = _engine_with_skill("web_search")

    def _policy_down(*args, **kwargs):
        return (_ for _ in ()).throw(RuntimeError("policy offline"))

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy_down,
    )

    result = await CapabilityEngine.execute(
        engine,
        "web_search",
        {"query": "latest vulnerability"},
        context={"origin": "background"},
    )

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert result["reason"] == "background_policy_unavailable"


@pytest.mark.asyncio
async def test_background_web_search_uses_lightweight_io_preflight(monkeypatch):
    engine = _engine_with_skill("web_search")
    calls: list[dict] = []

    def _policy(*args, **kwargs):
        calls.append(dict(kwargs))
        return (_ for _ in ()).throw(RuntimeError("policy offline"))

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy,
    )

    result = await CapabilityEngine.execute(
        engine,
        "web_search",
        {"query": "latest Turing Award"},
        context={"origin": "background"},
    )

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert calls
    assert calls[0]["min_idle_seconds"] == pytest.approx(30.0)
    assert calls[0]["max_memory_percent"] == pytest.approx(84.0)
    assert calls[0]["max_failure_pressure"] == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_background_browser_dialogue_still_uses_strict_foreground_preflight(monkeypatch):
    engine = _engine_with_skill("web_interlocutor")
    calls: list[dict] = []

    def _policy(*args, **kwargs):
        calls.append(dict(kwargs))
        return (_ for _ in ()).throw(RuntimeError("policy offline"))

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy,
    )

    result = await CapabilityEngine.execute(
        engine,
        "web_interlocutor",
        {"topic": "memory and agency"},
        context={"origin": "background"},
    )

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert calls
    # The strict thresholds moved into the named HEAVY_SKILL_PREFLIGHT
    # profile (policy-ratchet migration); the contract — foreground-exclusive
    # skills gate on 600s idle / 72% memory / 0.20 pressure — is unchanged.
    profile = calls[0].get("profile")
    assert profile is not None, "preflight must pass the named profile"
    assert profile.min_idle_seconds == pytest.approx(600.0)
    assert profile.max_memory_percent == pytest.approx(72.0)
    assert profile.max_failure_pressure == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_high_cost_tool_blocks_when_self_preservation_check_fails(monkeypatch):
    ServiceContainer.clear()
    engine = _engine_with_skill("sovereign_terminal", metabolic_cost=3)

    monkeypatch.setattr(
        "core.capability_engine.ServiceContainer.has", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("constitution offline")),
    )

    def _metabolism_down(*args, **kwargs):
        return (_ for _ in ()).throw(RuntimeError("metabolism offline"))

    monkeypatch.setattr("core.capability_engine.resolve_metabolic_monitor", _metabolism_down)
    try:
        result = await CapabilityEngine.execute(
            engine,
            "sovereign_terminal",
            {"command": "stress test"},
            context={"origin": "background"},
        )
    finally:
        ServiceContainer.clear()

    assert result["ok"] is False
    assert result["status"] == "blocked_by_self_preservation_unavailable"
