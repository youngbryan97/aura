"""Causal contract tests for autonomous CRSM-to-LoRA closure."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning import crsm_closure_scheduler as mod
from core.learning.crsm_closure_scheduler import (
    AUTHORITY_SOURCE,
    PLASTIC_TARGET,
    CRSMClosureScheduler,
    reset_crsm_closure_scheduler_for_test,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_crsm_closure_scheduler_for_test()
    yield
    reset_crsm_closure_scheduler_for_test()


class _FakeMonitor:
    def __init__(
        self,
        state: str = "open",
        *,
        manifest_current: bool = True,
        close_on_train: bool = True,
        command_override: list[str] | None = None,
        closure_receipt: str = "will-crsm-1",
        closure_intent: str = "intent-crsm-1",
    ) -> None:
        self.state = state
        self.manifest_current = manifest_current
        self.close_on_train = close_on_train
        self.command_override = command_override
        self.closure_receipt = closure_receipt
        self.closure_intent = closure_intent
        self.next_action_calls = 0

    def next_action(self, *args, **kwargs) -> dict[str, object]:
        del args, kwargs
        self.next_action_calls += 1
        if self.state == "closed":
            return {"required": False, "reason": "already closed"}
        if not self.manifest_current:
            command = ["python", "training/build_dataset_v3.py"]
            phase = "prepare_dataset"
        else:
            command = [
                "python",
                "training/train_and_fuse.py",
                "--crsm-delta",
                "--tag",
                "crsm-closeout",
            ]
            phase = "crsm_delta_train_fuse_publish"
        if self.command_override is not None:
            command = list(self.command_override)
        return {"required": True, "phase": phase, "command": command}

    def loop_state(self) -> dict[str, object]:
        result = {
            "state": self.state,
            "reason": "trained in" if self.state == "closed" else "captures remain",
            "unconsumed": 0 if self.state == "closed" else 33,
            "next_action": self.next_action(),
        }
        if self.state == "closed":
            governance = {
                "will_receipt_id": self.closure_receipt,
                "executive_intent_id": self.closure_intent,
            }
            result.update(
                {
                    "marker_matches_dataset": True,
                    "active_model": "/models/Aura-32B-crsm",
                    "active_model_governance": governance,
                    "consumption_marker": {
                        "consumed_at": time.time(),
                        "model_path": "/models/Aura-32B-crsm",
                        "governance_receipt_id": self.closure_receipt,
                        "authority_intent_id": self.closure_intent,
                    },
                    "training_state": {
                        "crsm_delta": {
                            "status": "fused_published_marker_ready",
                            "governance": governance,
                        }
                    },
                }
            )
        return result

    def observe_success(self, phase: str) -> None:
        if phase == "prepare_dataset":
            self.manifest_current = True
        elif phase == "crsm_delta_train_fuse_publish" and self.close_on_train:
            self.state = "closed"


class _FakeAuthorityGateway:
    def __init__(self, *, approved: bool = True) -> None:
        self.approved = approved
        self.authorizations: list[dict[str, object]] = []
        self.finalizations: list[dict[str, object]] = []

    async def authorize_semantic_weight_update(self, origin, cause, **kwargs):
        self.authorizations.append({"origin": origin, "cause": cause, **kwargs})
        return SimpleNamespace(
            approved=self.approved,
            reason="approved" if self.approved else "test_authority_denial",
            will_receipt_id="will-crsm-1" if self.approved else "will-denied-1",
            executive_intent_id="intent-crsm-1" if self.approved else None,
            capability_token_id=None,
            standing_authority_token=None,
            domain="semantic_weight_update",
            source=AUTHORITY_SOURCE,
            constraints={"target_module": PLASTIC_TARGET},
        )

    def finalize_tool_execution(self, **kwargs):
        self.finalizations.append(dict(kwargs))
        return {
            "closed": True,
            "success": bool(kwargs.get("success")),
            "intent_closed": True,
            "token_revoked": True,
            "standing_authority_closed": True,
            "errors": [],
        }


def _install(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: CRSMClosureScheduler,
    monitor: _FakeMonitor,
    *,
    ram_gb: float = 64.0,
    authority_ok: bool = True,
    idle_ok: bool = True,
    rc_by_phase: dict[str, int] | None = None,
) -> tuple[_FakeAuthorityGateway, list[dict[str, object]]]:
    import core.consciousness.crsm_loop_monitor as clm
    import core.executive.authority_gateway as authority_module
    from core.governance_context import get_active_governance

    gateway = _FakeAuthorityGateway(approved=authority_ok)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(clm, "get_crsm_loop_monitor", lambda: monitor)
    monkeypatch.setattr(authority_module, "get_authority_gateway", lambda: gateway)
    monkeypatch.setattr(
        scheduler,
        "_ram_admits",
        lambda: (
            ram_gb >= scheduler.min_free_gb,
            f"free_ram:{ram_gb}GB"
            if ram_gb >= scheduler.min_free_gb
            else f"insufficient_free_ram:{ram_gb}GB",
        ),
    )
    monkeypatch.setattr(scheduler, "_idle_allows", lambda: idle_ok)
    monkeypatch.setattr(mod, "record_degradation", lambda *args, **kwargs: None)
    model_path = scheduler._state_path.parent / "test-cortex"
    model_path.mkdir(exist_ok=True)
    monkeypatch.setattr(scheduler, "_base_model_path", lambda: model_path)
    scheduler._resource_requirements_loaded = True
    scheduler._resource_model_path_cache = str(model_path)
    scheduler._required_free_gb_cache = scheduler.min_free_gb
    scheduler._model_request_gb_cache = scheduler.min_free_gb

    async def _fake_train(command, *, phase, decision):
        token = get_active_governance()
        calls.append(
            {
                "command": list(command),
                "phase": phase,
                "decision": decision,
                "token": token,
            }
        )
        rc = int((rc_by_phase or {}).get(phase, 0))
        if rc == 0:
            monitor.observe_success(phase)
        return {"returncode": rc, "stdout": "", "stderr": "boom" if rc else ""}

    monkeypatch.setattr(scheduler, "_run_training", _fake_train)
    return gateway, calls


def _scheduler(tmp_path: Path) -> CRSMClosureScheduler:
    scheduler = CRSMClosureScheduler()
    scheduler._state_path = tmp_path / "crsm-state.json"
    return scheduler


def test_enabled_by_default_and_explicit_kill_switch(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AURA_CRSM_AUTOCLOSE", raising=False)
    scheduler = _scheduler(tmp_path)
    assert scheduler.get_status()["enabled"] is True
    assert scheduler.train_timeout_s == 3 * 3600

    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "0")
    assert _scheduler(tmp_path).get_status()["enabled"] is False


@pytest.mark.asyncio
async def test_run_now_blocked_only_by_explicit_kill_switch(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AURA_CRSM_AUTOCLOSE", "0")
    out = await _scheduler(tmp_path).run_closure_now()
    assert out == {"status": "blocked", "reasons": ["disabled_by_env"]}


@pytest.mark.asyncio
async def test_noop_when_loop_not_open(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor("closed")
    gateway, calls = _install(monkeypatch, scheduler, monitor)

    out = await scheduler.run_closure_now()

    assert out["status"] == "noop"
    assert gateway.authorizations == []
    assert calls == []


@pytest.mark.asyncio
async def test_ram_deferral_does_not_consume_cooldown(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor()
    gateway, calls = _install(monkeypatch, scheduler, monitor, ram_gb=12.0)

    out = await scheduler.run_closure_now()

    assert out["status"] == "deferred"
    assert scheduler._last_attempt_at == 0.0
    assert not scheduler._state_path.exists()
    assert gateway.authorizations == []
    assert calls == []


@pytest.mark.asyncio
async def test_authority_denial_does_not_consume_cooldown(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor()
    gateway, calls = _install(monkeypatch, scheduler, monitor, authority_ok=False)

    out = await scheduler.run_closure_now()

    assert out["status"] == "will_declined"
    assert scheduler._last_attempt_at == 0.0
    assert not scheduler._state_path.exists()
    assert len(gateway.authorizations) == 1
    assert calls == []


@pytest.mark.asyncio
async def test_direct_train_closes_only_from_observed_monitor_proof(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor()
    gateway, calls = _install(monkeypatch, scheduler, monitor)

    out = await scheduler.run_closure_now(reason="unit")

    assert out["status"] == "closed"
    assert out["loop"]["state"] == "closed"
    assert [call["phase"] for call in calls] == ["crsm_delta_train_fuse_publish"]
    assert calls[0]["command"] == [
        sys.executable,
        str(mod._REPO_ROOT / "training/train_and_fuse.py"),
        "--crsm-delta",
        "--tag",
        "crsm-closeout",
    ]
    token = calls[0]["token"]
    assert token is not None
    assert token.receipt_id == "will-crsm-1"
    assert token.domain == "semantic_weight_update"
    assert gateway.authorizations[0]["target_module"] == PLASTIC_TARGET
    assert gateway.finalizations[0]["success"] is True
    assert out["authority_closure"]["closed"] is True
    persisted = json.loads(scheduler._state_path.read_text(encoding="utf-8"))
    assert persisted["last_status"] == "closed"
    assert persisted["last_attempt_at"] > 0.0


@pytest.mark.asyncio
async def test_stale_manifest_runs_prepare_then_train_under_one_receipt(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor(manifest_current=False)
    gateway, calls = _install(monkeypatch, scheduler, monitor)

    out = await scheduler.run_closure_now(reason="stale_manifest")

    assert out["status"] == "closed"
    assert [call["phase"] for call in calls] == [
        "prepare_dataset",
        "crsm_delta_train_fuse_publish",
    ]
    assert {call["token"].receipt_id for call in calls} == {"will-crsm-1"}
    assert len(gateway.authorizations) == 1
    assert len(gateway.finalizations) == 1


@pytest.mark.asyncio
async def test_zero_return_without_loop_closure_is_incomplete_not_closed(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor(close_on_train=False)
    gateway, calls = _install(monkeypatch, scheduler, monitor)

    out = await scheduler.run_closure_now()

    assert out["status"] == "incomplete"
    assert "successful_stage_did_not_advance" in out["reasons"][0]
    assert monitor.state == "open"
    assert len(calls) == 1, "a successful-but-unproven train must not be repeated in one cycle"
    assert gateway.finalizations[0]["success"] is False


@pytest.mark.asyncio
async def test_closed_marker_must_correlate_to_current_authority(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor(closure_receipt="forged-receipt")
    gateway, calls = _install(monkeypatch, scheduler, monitor)

    out = await scheduler.run_closure_now()

    assert out["status"] == "incomplete"
    assert "authority_closure_mismatch" in out["reasons"][0]
    assert "marker_receipt" in out["reasons"][0]
    assert len(calls) == 1
    assert gateway.finalizations[0]["success"] is False


@pytest.mark.asyncio
async def test_train_failure_preserves_open_loop_and_finalizes_failure(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor(close_on_train=False)
    gateway, _calls = _install(
        monkeypatch,
        scheduler,
        monitor,
        rc_by_phase={"crsm_delta_train_fuse_publish": 2},
    )

    out = await scheduler.run_closure_now()

    assert out["status"] == "train_failed"
    assert out["returncode"] == 2
    assert monitor.state == "open"
    assert gateway.finalizations[0]["success"] is False


@pytest.mark.asyncio
async def test_substituted_monitor_command_is_rejected_before_authority_or_spawn(
    monkeypatch,
    tmp_path,
) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor(command_override=["python", "training/train_and_fuse.py", "--skip-train"])
    gateway, calls = _install(monkeypatch, scheduler, monitor)

    out = await scheduler.run_closure_now()

    assert out["status"] == "blocked"
    assert "command substitution rejected" in out["reasons"][0]
    assert gateway.authorizations == []
    assert calls == []


@pytest.mark.asyncio
async def test_scheduled_path_respects_idle_and_durable_cooldown(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    scheduler.cooldown_s = 10_000.0
    monitor = _FakeMonitor()
    _gateway, calls = _install(monkeypatch, scheduler, monitor, idle_ok=False)

    await scheduler._maybe_close()
    assert calls == []

    scheduler._state_loaded = False
    scheduler._state_path.write_text(
        json.dumps({"last_attempt_at": time.time(), "last_status": "train_failed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(scheduler, "_idle_allows", lambda: True)
    await scheduler._maybe_close()
    assert calls == []


@pytest.mark.asyncio
async def test_single_flight_guard(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    scheduler._running_cycle = True
    out = await scheduler.run_closure_now()
    assert out == {"status": "blocked", "reasons": ["closure_already_running"]}


@pytest.mark.asyncio
async def test_cancelled_cycle_finalizes_authority_as_failure(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor(close_on_train=False)
    gateway, _calls = _install(monkeypatch, scheduler, monitor)
    started = mod.asyncio.Event()

    async def _blocking_train(command, *, phase, decision):
        del command, phase, decision
        started.set()
        await mod.asyncio.Event().wait()

    monkeypatch.setattr(scheduler, "_run_training", _blocking_train)
    task = mod.asyncio.create_task(scheduler.run_closure_now())
    await started.wait()
    task.cancel()

    with pytest.raises(mod.asyncio.CancelledError):
        await task
    assert scheduler._running_cycle is False
    assert gateway.finalizations[0]["success"] is False
    assert gateway.finalizations[0]["error"] == "scheduler_cancelled"
    persisted = json.loads(scheduler._state_path.read_text(encoding="utf-8"))
    assert persisted["last_status"] == "cancelled"


def test_health_status_uses_only_cached_state(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    scheduler._state_loaded = True
    scheduler._last_attempt_at = 123.0
    scheduler._loop_cache = {"state": "open", "unconsumed": 33}
    monkeypatch.setattr(
        scheduler,
        "_read_state_file",
        lambda: (_ for _ in ()).throw(AssertionError("status read disk")),
    )

    status = scheduler.get_status()

    assert status["last_attempt_at"] == 123.0
    assert status["loop"]["state"] == "open"


@pytest.mark.asyncio
async def test_state_and_monitor_reads_are_offloaded(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    monitor = _FakeMonitor()
    import core.consciousness.crsm_loop_monitor as clm

    monkeypatch.setattr(clm, "get_crsm_loop_monitor", lambda: monitor)
    called: list[str] = []

    async def _to_thread(func, *args, **kwargs):
        called.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(mod.asyncio, "to_thread", _to_thread)
    await scheduler._ensure_state_loaded()
    await scheduler._read_loop_state()

    assert "_read_state_file" in called
    assert "loop_state" in called


@pytest.mark.asyncio
async def test_real_training_gateway_receives_delegated_receipt_and_model_lane(
    monkeypatch,
    tmp_path,
) -> None:
    scheduler = _scheduler(tmp_path)
    model_path = tmp_path / "test-cortex"
    model_path.mkdir()
    monkeypatch.setattr(scheduler, "_base_model_path", lambda: model_path)
    scheduler._resource_requirements_loaded = True
    scheduler._resource_model_path_cache = str(model_path)
    scheduler._model_request_gb_cache = 51.0
    captured: dict[str, object] = {}

    class _Gateway:
        async def run_async(self, command, **kwargs):
            captured["command"] = list(command)
            captured.update(kwargs)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    import core.runtime.subprocess_gateway as subprocess_module

    monkeypatch.setattr(subprocess_module, "get_subprocess_gateway", lambda: _Gateway())
    decision = SimpleNamespace(
        will_receipt_id="will-real-1",
        executive_intent_id="intent-real-1",
    )
    command = [sys.executable, str(mod._REPO_ROOT / "training/train_and_fuse.py")]

    out = await scheduler._run_training(
        command,
        phase="crsm_delta_train_fuse_publish",
        decision=decision,
    )

    assert out["returncode"] == 0
    assert captured["offline_tooling"] is False
    assert captured["capture_output"] is False
    assert captured["cwd"] == mod._REPO_ROOT
    env = captured["env"]
    assert env["AURA_GOVERNANCE_MODE"] == "delegated_subprocess"
    assert env["AURA_DELEGATED_GOVERNANCE_RECEIPT_ID"] == "will-real-1"
    assert env["AURA_DELEGATED_AUTHORITY_INTENT_ID"] == "intent-real-1"
    assert env["AURA_DELEGATED_GOVERNANCE_PARENT_PID"] == str(os.getpid())
    assert env["AURA_TRAINING_ALLOW_LIVE_AURA"] == "1"
    claim = captured["model_lane_claim"]
    assert claim.request_gb == 51.0
    assert claim.purpose == "compound"
    assert claim.metadata["allow_inherited_model_children"] is True
    assert claim.metadata["allowed_inherited_model_purposes"] == [
        "train",
        "fuse",
        "benchmark",
    ]
    assert claim.metadata["allowed_inherited_model_roots"] == [
        str((mod._REPO_ROOT / "training" / "fused-model").resolve())
    ]


@pytest.mark.asyncio
async def test_prepare_stage_does_not_export_authority_secrets(monkeypatch, tmp_path) -> None:
    scheduler = _scheduler(tmp_path)
    captured: dict[str, object] = {}

    class _Gateway:
        async def run_async(self, command, **kwargs):
            captured["command"] = list(command)
            captured.update(kwargs)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    import core.runtime.subprocess_gateway as subprocess_module

    monkeypatch.setattr(subprocess_module, "get_subprocess_gateway", lambda: _Gateway())
    decision = SimpleNamespace(
        will_receipt_id="will-prepare-1",
        executive_intent_id="intent-prepare-1",
    )

    await scheduler._run_training(
        [sys.executable, str(mod._REPO_ROOT / "training/build_dataset_v3.py")],
        phase="prepare_dataset",
        decision=decision,
    )

    env = captured["env"]
    assert env["AURA_GOVERNANCE_MODE"] == "delegated_subprocess_child"
    assert "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID" not in env
    assert "AURA_DELEGATED_AUTHORITY_INTENT_ID" not in env
    assert captured["model_lane_claim"] is None
