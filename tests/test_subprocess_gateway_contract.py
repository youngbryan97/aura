from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime import shutdown_coordinator, subprocess_gateway
from core.runtime.model_lane_control import (
    LaneClaim,
    ModelLaneController,
    infer_model_process_claim,
)
from core.runtime.receipts import ReceiptStore
from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def model_lane_controller_factory(tmp_path: Path):
    stores: list[ReceiptStore] = []

    def _build(*, state_path: Path | None = None) -> ModelLaneController:
        store = ReceiptStore(tmp_path / f"receipts-{len(stores)}")
        stores.append(store)
        return ModelLaneController(
            state_path=state_path or tmp_path / "model_lanes.json",
            receipt_store=store,
            process_discovery=None,
        )

    try:
        yield _build
    finally:
        for store in reversed(stores):
            store.close()


def test_model_process_claim_inference_requires_identity_and_sizes_peak() -> None:
    claim = infer_model_process_claim(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            "/models/qwen-7b",
            "--train",
        ],
        source="training_tooling:unit",
        timeout_s=600.0,
    )

    assert claim is not None
    assert claim.model_path == "/models/qwen-7b"
    assert claim.purpose == "train"
    assert claim.request_gb >= 9.0
    assert claim.preemptible is True
    assert claim.runtime_assignment is not None
    assert claim.runtime_assignment.role == "trainer"
    assert claim.runtime_assignment.qos == "best_effort"


def test_model_process_claim_inference_fails_closed_without_model_path() -> None:
    with pytest.raises(RuntimeError, match="missing_model_path"):
        infer_model_process_claim(
            [sys.executable, "-m", "mlx_lm", "lora", "--train"],
            source="training_tooling:missing-model",
            timeout_s=600.0,
        )


def test_benchmark_model_under_training_directory_is_not_misclassified() -> None:
    claim = infer_model_process_claim(
        [
            sys.executable,
            "tools/heldout_eval.py",
            "--model",
            "/repo/training/fused-model/Aura-32B",
        ],
        source="training_tooling:heldout-regression",
        timeout_s=600.0,
    )

    assert claim is not None
    assert claim.purpose == "benchmark"


def test_sync_gateway_refuses_untrackable_model_process() -> None:
    with pytest.raises(RuntimeError, match="require run_async/spawn_async"):
        subprocess_gateway.SubprocessGateway().run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "lora",
                "--model",
                "/models/qwen-7b",
                "--train",
            ],
            source="training_tooling:sync-denied",
            accelerator_capability="model",
        )


def test_gateway_refuses_undeclared_accelerator_capability() -> None:
    with pytest.raises(
        subprocess_gateway.GovernanceViolation,
        match="accelerator_capability_undeclared",
    ):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('ordinary child')"],
            read_only=True,
            source="runtime_probe:undeclared-capability",
        )


def test_none_declaration_cannot_launder_renamed_model_loader(tmp_path: Path) -> None:
    loader = tmp_path / "innocent_name.py"
    loader.write_text("import mlx_lm\n", encoding="utf-8")

    with pytest.raises(
        subprocess_gateway.GovernanceViolation,
        match="accelerator_capability_contradiction",
    ):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, str(loader), "--model", "/models/qwen-7b"],
            read_only=True,
            source="runtime_probe:renamed-model-loader",
            accelerator_capability="none",
        )


def test_auto_declaration_attributes_renamed_model_loader(
    tmp_path: Path,
) -> None:
    loader = tmp_path / "innocent_name.py"
    loader.write_text("import mlx_lm\n", encoding="utf-8")

    claim = subprocess_gateway._resolve_accelerator_claim(
        [sys.executable, str(loader), "--model", "/models/qwen-7b"],
        source="runtime_probe:auto-renamed-model-loader",
        timeout_s=60.0,
        accelerator_capability="auto",
    )

    assert claim is not None
    assert claim.model_path == "/models/qwen-7b"
    assert claim.metadata["declared_model_process"] is True
    assert claim.runtime_assignment is not None
    assert claim.runtime_assignment.role == "auxiliary"


def test_auto_declaration_refuses_uninspectable_dynamic_python() -> None:
    with pytest.raises(
        subprocess_gateway.GovernanceViolation,
        match="accelerator_capability_unresolved",
    ):
        subprocess_gateway._resolve_accelerator_claim(
            [sys.executable, "missing_dynamic_program.py"],
            source="runtime_probe:missing-dynamic-program",
            timeout_s=60.0,
            accelerator_capability="auto",
        )


def test_blocking_model_gateway_routes_through_async_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = subprocess_gateway.SubprocessGateway()
    claim = LaneClaim(
        owner_id="subprocess:test:blocking-bridge",
        model_path="/models/qwen-7b",
        request_gb=0.1,
        purpose="benchmark",
        request_id="blocking-bridge",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    async def _run_async(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess_gateway.subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(gateway, "run_async", _run_async)
    result = gateway.run_model_blocking(
        [sys.executable, "-c", "raise SystemExit(0)"],
        timeout=12.0,
        source="training_tooling:blocking-bridge",
        model_lane_claim=claim,
    )

    assert result.returncode == 0
    assert calls[0][1]["model_lane_claim"] is claim
    assert calls[0][1]["timeout"] == 12.0


def test_delegated_governance_environment_must_match_active_scope() -> None:
    from core.governance_context import governed_scope_sync

    decision = SimpleNamespace(
        will_receipt_id="will-delegated-1",
        domain="semantic_weight_update",
        source="system_maintenance:crsm_closure",
        constraints={"executive_intent_id": "intent-delegated-1"},
    )
    env = {
        "AURA_GOVERNANCE_MODE": "delegated_subprocess",
        "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID": "will-delegated-1",
        "AURA_DELEGATED_GOVERNANCE_DOMAIN": "semantic_weight_update",
        "AURA_DELEGATED_GOVERNANCE_SOURCE": "system_maintenance:crsm_closure",
        "AURA_DELEGATED_AUTHORITY_INTENT_ID": "intent-delegated-1",
        "AURA_DELEGATED_GOVERNANCE_PARENT_PID": str(os.getpid()),
    }
    with governed_scope_sync(decision):
        subprocess_gateway._validate_delegated_governance_environment(
            env,
            source="system_maintenance:crsm_closure:crsm_delta_train_fuse_publish",
        )
        env["AURA_DELEGATED_AUTHORITY_INTENT_ID"] = "forged-intent"
        with pytest.raises(
            subprocess_gateway.GovernanceViolation,
            match="does not match active scope",
        ):
            subprocess_gateway._validate_delegated_governance_environment(
                env,
                source="system_maintenance:crsm_closure:crsm_delta_train_fuse_publish",
            )


def test_explicit_child_environment_does_not_rehydrate_parent_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "delegated_subprocess")
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_RECEIPT_ID", "parent-secret")
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_DOMAIN", "semantic_weight_update")
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_SOURCE", "parent-source")
    monkeypatch.setenv("AURA_DELEGATED_AUTHORITY_INTENT_ID", "parent-intent")
    monkeypatch.setenv("AURA_DELEGATED_GOVERNANCE_PARENT_PID", str(os.getpid()))

    child_env = {
        "AURA_GOVERNANCE_MODE": "delegated_subprocess_child",
        "AURA_REQUIRE_GOVERNANCE": "0",
    }
    subprocess_gateway._validate_delegated_governance_environment(
        child_env,
        source="training_tooling:delegated-child",
    )


def test_inherited_model_child_reuses_parent_group_and_strips_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import model_lane_control

    captured: dict[str, object] = {}

    class _Controller:
        def validate_inherited_child_claim(self, **kwargs):
            captured["validation"] = kwargs
            return True

        def release_inherited_child_claim(self, **kwargs):
            captured["release"] = kwargs
            return True

    def _run(command, **kwargs):
        captured["command"] = list(command)
        captured["run"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", _Controller)
    monkeypatch.setattr(subprocess_gateway.subprocess, "run", _run)
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    env = {
        "AURA_MODEL_LANE_INHERITED_OWNER_ID": "pipeline-owner",
        "AURA_MODEL_LANE_INHERITED_REQUEST_ID": "pipeline-request",
        "AURA_MODEL_LANE_INHERITED_MODEL_PATH": "/models/qwen-32b",
        "AURA_MODEL_LANE_INHERITED_PURPOSE": "compound",
        "AURA_MODEL_LANE_DELEGATION_TOKEN": "secret-token",
        "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID": "secret-receipt",
    }
    claim = LaneClaim(
        owner_id="nested-train",
        model_path="/models/qwen-32b",
        request_gb=12.0,
        purpose="train",
    )

    result = subprocess_gateway.SubprocessGateway().run_model_blocking(
        [sys.executable, "-c", "print('nested')"],
        env=env,
        offline_tooling=True,
        source="training_tooling:nested-model",
        model_lane_claim=claim,
    )

    assert result.returncode == 0
    assert captured["validation"]["requested_gb"] == 12.0
    assert captured["validation"]["child_request_id"] == claim.request_id
    assert captured["release"]["child_request_id"] == claim.request_id
    assert captured["validation"]["child_model_path"] == "/models/qwen-32b"
    assert captured["validation"]["child_purpose"] == "train"
    run = captured["run"]
    assert run["start_new_session"] is False
    assert run["env"]["AURA_MODEL_LANE_PARENT_ACCOUNTED"] == "1"
    assert "AURA_MODEL_LANE_DELEGATION_TOKEN" not in run["env"]
    assert "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID" not in run["env"]


@pytest.mark.asyncio
async def test_blocking_model_gateway_refuses_active_event_loop() -> None:
    claim = LaneClaim(
        owner_id="subprocess:test:blocking-loop-refusal",
        model_path="/models/qwen-7b",
        request_gb=0.1,
        purpose="benchmark",
        request_id="blocking-loop-refusal",
    )

    with pytest.raises(RuntimeError, match="cannot block an active event loop"):
        subprocess_gateway.SubprocessGateway().run_model_blocking(
            [sys.executable, "-c", "raise SystemExit(0)"],
            source="training_tooling:blocking-loop-refusal",
            model_lane_claim=claim,
        )


@pytest.mark.asyncio
async def test_cancelled_effectful_run_terminates_process_group_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = subprocess_gateway.SubprocessGateway()
    communicate_started = asyncio.Event()
    cleaned = asyncio.Event()

    class _Process:
        returncode = None

        async def communicate(self, _input=None):
            communicate_started.set()
            await asyncio.Event().wait()

    process = _Process()

    async def _spawn_async(*args, **kwargs):
        return process

    async def _cleanup(candidate, **kwargs):
        assert candidate is process
        cleaned.set()
        return b"", b""

    monkeypatch.setattr(gateway, "spawn_async", _spawn_async)
    monkeypatch.setattr(subprocess_gateway, "_terminate_async_process_group", _cleanup)
    task = asyncio.create_task(
        gateway.run_async(
            [sys.executable, "-c", "pass"],
            source="system_maintenance:cancellation-test",
            accelerator_capability="none",
        )
    )
    await communicate_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_cleanup_signals_isolated_group_after_root_already_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, signal.Signals]] = []

    class _ExitedRoot:
        pid = 11001
        returncode = 0
        _aura_process_group_id = 22002
        _aura_start_new_session = True

        async def communicate(self):
            return b"", b""

        def terminate(self):
            raise AssertionError("isolated process group should be signalled")

    monkeypatch.setattr(
        subprocess_gateway.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    assert await subprocess_gateway._terminate_async_process_group(_ExitedRoot()) == (b"", b"")
    assert signals == [(22002, signal.SIGTERM)]


@pytest.mark.host_observation
@pytest.mark.asyncio
async def test_spawn_async_commits_and_releases_model_process_owner(
    monkeypatch: pytest.MonkeyPatch,
    model_lane_controller_factory,
) -> None:
    from core.runtime import model_lane_control

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    controller = model_lane_controller_factory()
    monkeypatch.setattr(
        model_lane_control,
        "get_model_lane_controller",
        lambda: controller,
    )
    claim = LaneClaim(
        owner_id="subprocess:test:model-job",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="train",
        request_id="subprocess-model-job",
    )

    proc = await subprocess_gateway.SubprocessGateway().spawn_async(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        source="training_tooling:model-owner-test",
        model_lane_claim=claim,
        accelerator_capability="model",
    )
    snapshot = controller.snapshot()
    assert snapshot["reserved_gb"] == 0.0
    assert len(snapshot["owners"]) == 1
    assert snapshot["owners"][0]["process"]["pid"] == proc.pid
    assert snapshot["owners"][0]["metadata"]["managed_model_process"] is True
    assert snapshot["owners"][0]["metadata"]["process_group_id"] == proc.pid
    assert snapshot["owners"][0]["metadata"]["process_session_id"] == proc.pid
    assert proc._aura_model_lane_fencing_token > 0

    proc.terminate()
    await proc.wait()
    await proc._aura_model_lane_monitor

    assert controller.snapshot()["owners"] == []
    assert controller._receipt_store.coverage_stats()["resource_admission"] == 1


@pytest.mark.host_observation
@pytest.mark.asyncio
async def test_model_owner_survives_root_exit_until_process_group_drains(
    monkeypatch: pytest.MonkeyPatch,
    model_lane_controller_factory,
) -> None:
    from core.runtime import model_lane_control

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    controller = model_lane_controller_factory()
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    claim = LaneClaim(
        owner_id="subprocess:test:escaped-descendant",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="train",
        request_id="escaped-descendant-job",
    )
    root_code = "\n".join(
        (
            "import subprocess, sys",
            "subprocess.Popen(",
            "    [sys.executable, '-c', 'import time; time.sleep(1.5)'],",
            "    stdin=subprocess.DEVNULL,",
            "    stdout=subprocess.DEVNULL,",
            "    stderr=subprocess.DEVNULL,",
            ")",
        )
    )

    proc = await subprocess_gateway.SubprocessGateway().spawn_async(
        [sys.executable, "-c", root_code],
        source="training_tooling:descendant-accounting",
        model_lane_claim=claim,
        accelerator_capability="model",
    )
    await proc.wait()

    assert proc.returncode == 0
    assert proc._aura_model_lane_monitor.done() is False
    assert [owner["owner_id"] for owner in controller.snapshot()["owners"]] == [
        claim.owner_id
    ]

    await asyncio.wait_for(proc._aura_model_lane_monitor, timeout=5.0)
    assert controller.snapshot()["owners"] == []


@pytest.mark.asyncio
async def test_model_subprocess_requires_isolated_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    claim = LaneClaim(
        owner_id="subprocess:test:nonisolated",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="train",
        request_id="nonisolated-model-job",
    )

    with pytest.raises(RuntimeError, match="requires_isolated_process_group"):
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "raise SystemExit(0)"],
            source="training_tooling:nonisolated-model-job",
            model_lane_claim=claim,
            start_new_session=False,
            accelerator_capability="model",
        )


@pytest.mark.host_observation
@pytest.mark.asyncio
async def test_cancelled_monitor_retains_owner_until_child_process_dies(
    monkeypatch: pytest.MonkeyPatch,
    model_lane_controller_factory,
) -> None:
    from core.runtime import model_lane_control

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    controller = model_lane_controller_factory()
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    claim = LaneClaim(
        owner_id="subprocess:test:cancelled-monitor",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="benchmark",
        request_id="cancelled-monitor-job",
    )

    proc = await subprocess_gateway.SubprocessGateway().spawn_async(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        source="training_tooling:cancelled-monitor",
        model_lane_claim=claim,
        accelerator_capability="model",
    )
    proc._aura_model_lane_monitor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await proc._aura_model_lane_monitor

    assert proc.returncode is None
    assert [owner["owner_id"] for owner in controller.snapshot()["owners"]] == [
        claim.owner_id
    ]

    proc.terminate()
    await proc.wait()
    assert controller.snapshot()["owners"] == []


@pytest.mark.host_observation
@pytest.mark.asyncio
async def test_gateway_child_consumes_exact_inherited_model_lane_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_lane_controller_factory,
) -> None:
    from core.runtime import model_lane_control

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    home = tmp_path / "delegated-home"
    state_path = home / ".aura" / "run" / "model_lane_control.json"
    controller = model_lane_controller_factory(state_path=state_path)
    monkeypatch.setattr(
        model_lane_control,
        "get_model_lane_controller",
        lambda: controller,
    )
    claim = LaneClaim(
        owner_id="subprocess:test:delegated-child",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="benchmark",
        request_id="delegated-child-request",
    )
    child_code = "\n".join(
        (
            "import json",
            "from core.runtime.model_lane_control import acquire_standalone_model_lane",
            "lease = acquire_standalone_model_lane(owner_id='child-tool', "
            "model_path='/models/test-accelerator-job', purpose='benchmark', request_gb=0.1)",
            "print(json.dumps({'inherited': lease.inherited}))",
        )
    )
    child_env = {**os.environ, "HOME": str(home)}
    # The autouse resource_observer fixture pins AURA_MODEL_LANE_STATE_PATH
    # to a per-test file; inherited by the child it would point the child's
    # controller at a DIFFERENT state file than the parent controller above,
    # so the delegation could never match. The child derives its path from
    # HOME, exactly like a real delegated tool process.
    child_env.pop("AURA_MODEL_LANE_STATE_PATH", None)

    proc = await subprocess_gateway.SubprocessGateway().spawn_async(
        [sys.executable, "-c", child_code],
        cwd=PROJECT_ROOT,
        env=child_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        source="training_tooling:delegated-child-test",
        model_lane_claim=claim,
        accelerator_capability="model",
    )
    stdout, stderr = await proc.communicate()
    await proc._aura_model_lane_monitor

    assert proc.returncode == 0, stderr.decode()
    assert json.loads(stdout.decode().strip()) == {"inherited": True}
    assert controller.snapshot()["owners"] == []
    reservation = controller.snapshot()["reservations"][0]
    assert reservation["delegation"]["consumed_process"]["pid"] == proc.pid


@pytest.mark.host_observation
@pytest.mark.asyncio
async def test_run_async_tracks_model_owner_through_completion(
    monkeypatch: pytest.MonkeyPatch,
    model_lane_controller_factory,
) -> None:
    from core.runtime import model_lane_control

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    controller = model_lane_controller_factory()
    monkeypatch.setattr(
        model_lane_control,
        "get_model_lane_controller",
        lambda: controller,
    )
    claim = LaneClaim(
        owner_id="subprocess:test:run-model-job",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="benchmark",
        request_id="run-model-job",
    )

    completed = await subprocess_gateway.SubprocessGateway().run_async(
        [sys.executable, "-c", "print('model-job-complete')"],
        source="training_tooling:model-run-test",
        model_lane_claim=claim,
        accelerator_capability="model",
    )
    for _ in range(50):
        if not controller.snapshot()["owners"]:
            break
        await asyncio.sleep(0.01)

    assert completed.returncode == 0
    assert completed.stdout.strip() == "model-job-complete"
    assert controller.snapshot()["owners"] == []


@pytest.mark.host_observation
@pytest.mark.asyncio
async def test_monitor_registration_failure_reaps_committed_model_child(
    monkeypatch: pytest.MonkeyPatch,
    model_lane_controller_factory,
) -> None:
    from core.runtime import model_lane_control
    from core.utils import task_tracker

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    controller = model_lane_controller_factory()
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    actual_tracker = task_tracker.get_task_tracker()

    class _FailingMonitorTracker:
        def create_task(self, coroutine, *, name: str):
            if name.startswith("ModelProcessOwner:"):
                raise RuntimeError("injected monitor registration failure")
            return actual_tracker.create_task(coroutine, name=name)

    monkeypatch.setattr(task_tracker, "get_task_tracker", lambda: _FailingMonitorTracker())
    created: list[asyncio.subprocess.Process] = []
    create_process = asyncio.create_subprocess_exec

    async def _capture_process(*args, **kwargs):
        process = await create_process(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(subprocess_gateway.asyncio, "create_subprocess_exec", _capture_process)
    claim = LaneClaim(
        owner_id="subprocess:test:monitor-failure",
        model_path="/models/test-accelerator-job",
        request_gb=0.1,
        purpose="benchmark",
        request_id="monitor-failure-job",
    )

    with pytest.raises(RuntimeError, match="model_subprocess_monitor_registration_failed"):
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            source="training_tooling:model-monitor-failure",
            model_lane_claim=claim,
            accelerator_capability="model",
        )

    assert len(created) == 1
    assert created[0].returncode is not None
    assert controller.snapshot()["owners"] == []


def test_offline_tooling_run_requires_named_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="offline subprocess tooling requires"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('bad-source')"],
            timeout=5,
            offline_tooling=True,
            source="adhoc:test",
            accelerator_capability="none",
        )


def test_read_only_run_requires_attributable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="read-only subprocess probes require"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('anonymous')"],
            timeout=5,
            read_only=True,
            accelerator_capability="none",
        )


def test_read_only_run_rejects_multiline_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="single-line"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('bad-source')"],
            timeout=5,
            read_only=True,
            source="test\nspoof",
            accelerator_capability="none",
        )


def test_read_only_run_allows_named_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('named')"],
        timeout=5,
        read_only=True,
        source="test.subprocess_gateway.read_only",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "named"


def test_shutdown_latch_blocks_effectful_subprocess_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    clear_shutdown_request()
    try:
        request_shutdown("unit-test")
        with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
            subprocess_gateway.SubprocessGateway().run(
                [sys.executable, "-c", "print('must-not-run')"],
                timeout=5,
                source="test.subprocess_gateway.shutdown_effectful_run",
                accelerator_capability="none",
            )
    finally:
        clear_shutdown_request()


def test_shutdown_latch_blocks_implicit_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")
    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('must-not-run')"],
            timeout=5,
            read_only=True,
            source="test.subprocess_gateway.shutdown_implicit_read_only_probe",
            accelerator_capability="none",
        )


def test_shutdown_latch_allows_explicit_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")
    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('probe-ok')"],
        timeout=5,
        read_only=True,
        allow_during_shutdown=True,
        source="test.subprocess_gateway.shutdown_read_only_probe",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "probe-ok"


def test_shutdown_latch_never_allows_live_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        subprocess_gateway.SubprocessGateway().spawn(
            [sys.executable, "-c", "print('must-not-run')"],
            read_only=True,
            allow_during_shutdown=True,
            source="test.subprocess_gateway.shutdown_live_handle",
            accelerator_capability="none",
        )


@pytest.mark.asyncio
async def test_global_resource_fence_allows_only_bounded_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime.runtime_hygiene import RuntimeHygieneManager

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    request_shutdown("unit-test")

    result = await subprocess_gateway.SubprocessGateway().run_async(
        [sys.executable, "-c", "print('bounded-probe-ok')"],
        timeout=5,
        read_only=True,
        allow_during_shutdown=True,
        source="test.subprocess_gateway.shutdown_bounded_probe",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "bounded-probe-ok"
    assert shutdown_coordinator.shutdown_admission_snapshot()["counts"][
        "allowed_read_only"
    ] >= 1
    await hygiene.stop()


def test_shutdown_latch_never_allows_effectful_offline_tooling_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('must-not-run')"],
            timeout=5,
            offline_tooling=True,
            allow_during_shutdown=True,
            source="maintenance_tooling:test_shutdown_effectful_override",
            accelerator_capability="none",
        )


def test_shutdown_latch_blocks_shell_spawn_before_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _must_not_spawn(*_args, **_kwargs):
        raise AssertionError("shell subprocess creation reached after shutdown latch")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _must_not_spawn)
    request_shutdown("unit-test")

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            "printf must-not-run",
            source="test.subprocess_gateway.shutdown_shell_spawn",
            accelerator_capability="none",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        asyncio.run(_attempt())


def test_async_spawn_terminates_child_when_shutdown_crosses_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    class _Process:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            return 0

    process = _Process()

    async def _spawn(*_args, **_kwargs):
        request_shutdown("crossed-create-subprocess-exec")
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('must-not-survive')"],
            source="test.subprocess_gateway.crossed_async_spawn",
            accelerator_capability="none",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        asyncio.run(_attempt())
    assert process.terminated is True
    assert process.killed is False


def test_read_only_spawn_async_requires_attributable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('anonymous')"],
            read_only=True,
            accelerator_capability="none",
        )

    with pytest.raises(ValueError, match="read-only subprocess probes require"):
        asyncio.run(_attempt())


def test_read_only_spawn_async_allows_named_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> str:
        proc = await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('named-async')"],
            stdout=asyncio.subprocess.PIPE,
            read_only=True,
            source="test.subprocess_gateway.read_only_async",
            accelerator_capability="none",
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode("utf-8").strip()

    assert asyncio.run(_attempt()) == "named-async"


def test_spawn_async_registers_gateway_process_with_runtime_hygiene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    registrations: list[dict[str, object]] = []

    def _register(proc, *, kind, source, command) -> None:
        registrations.append(
            {
                "pid": getattr(proc, "pid", None),
                "kind": kind,
                "source": source,
                "command": tuple(command),
            }
        )

    monkeypatch.setattr(subprocess_gateway, "_register_runtime_hygiene_process", _register)

    async def _attempt() -> str:
        proc = await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('registered-async')"],
            stdout=asyncio.subprocess.PIPE,
            read_only=True,
            source="test.subprocess_gateway.register_async",
            accelerator_capability="none",
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode("utf-8").strip()

    assert asyncio.run(_attempt()) == "registered-async"
    assert registrations
    assert registrations[0]["pid"]
    assert registrations[0]["kind"] == "subprocess"
    assert registrations[0]["source"] == "test.subprocess_gateway.register_async"


def test_spawn_shell_async_registers_gateway_process_with_runtime_hygiene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    registrations: list[dict[str, object]] = []

    def _register(proc, *, kind, source, command) -> None:
        registrations.append(
            {
                "pid": getattr(proc, "pid", None),
                "kind": kind,
                "source": source,
                "command": command,
            }
        )

    monkeypatch.setattr(subprocess_gateway, "_register_runtime_hygiene_process", _register)

    async def _attempt() -> str:
        proc = await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            f"{sys.executable} -c \"print('registered-shell-async')\"",
            stdout=asyncio.subprocess.PIPE,
            source="test.subprocess_gateway.register_shell_async",
            accelerator_capability="auto",
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode("utf-8").strip()

    assert asyncio.run(_attempt()) == "registered-shell-async"
    assert registrations
    assert registrations[0]["pid"]
    assert registrations[0]["kind"] == "subprocess"
    assert registrations[0]["source"] == "test.subprocess_gateway.register_shell_async"


def test_read_only_run_async_requires_attributable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().run_async(
            [sys.executable, "-c", "print('anonymous')"],
            read_only=True,
            accelerator_capability="none",
        )

    with pytest.raises(ValueError, match="read-only subprocess probes require"):
        asyncio.run(_attempt())


def test_read_only_run_async_allows_named_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> str:
        result = await subprocess_gateway.SubprocessGateway().run_async(
            [sys.executable, "-c", "print('named-run-async')"],
            read_only=True,
            source="test.subprocess_gateway.read_only_run_async",
            accelerator_capability="none",
        )
        return result.stdout.strip()

    assert asyncio.run(_attempt()) == "named-run-async"


def test_spawn_shell_async_denied_when_live_governance_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            f"{sys.executable} -c \"print('strict-shell')\"",
            source="test.subprocess_gateway.shell",
            accelerator_capability="auto",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        asyncio.run(_attempt())


def test_offline_tooling_run_denied_when_live_governance_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('strict')"],
            timeout=5,
            offline_tooling=True,
            source="proof_tooling:test",
            accelerator_capability="none",
        )


def test_offline_tooling_run_allowed_for_approved_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('ok')"],
        timeout=5,
        offline_tooling=True,
        source="proof_tooling:test",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_low_trust_spawn_fails_closed_when_secret_classifier_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "core.security.structural_redaction", None)

    with pytest.raises(
        subprocess_gateway.GovernanceViolation,
        match="secret-key classification unavailable",
    ):
        subprocess_gateway._enforce_process_privilege(
            env={"API_KEY": "must-not-reach-child"},
            source="generated_code:test",
            operation="run",
        )


def test_desktop_safe_run_blocks_proof_scale_environment_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ALLOW_DESKTOP_LONGRUNS", raising=False)

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="desktop-safe long-run"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('nethack_challenge.py should-not-run')"],
            timeout=5,
            read_only=True,
            source="test.subprocess_gateway.desktop_guard",
            accelerator_capability="none",
        )


def test_desktop_safe_run_allows_explicit_operator_longrun_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_ALLOW_DESKTOP_LONGRUNS", "1")

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('nethack_challenge.py override-ok')"],
        timeout=5,
        read_only=True,
        source="test.subprocess_gateway.desktop_guard_override",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "nethack_challenge.py override-ok"


def test_desktop_safe_shell_spawn_blocks_proof_batteries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.delenv("AURA_ALLOW_DESKTOP_LONGRUNS", raising=False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            f"{sys.executable} -c \"print('run_dnu_agi_proof_battery.py should-not-run')\"",
            source="test.subprocess_gateway.desktop_shell_guard",
            accelerator_capability="auto",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="desktop-safe long-run"):
        asyncio.run(_attempt())


def test_nethack_runner_rejects_env_only_longrun_bypass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    env = os.environ.copy()
    env.update(
        {
            "AURA_ALLOW_DESKTOP_NETHACK": "1",
            "AURA_ALLOW_DESKTOP_LONGRUNS": "1",
            "AURA_ALLOW_LONG_NETHACK_RUN": "1",
            "AURA_NETHACK_STEPS": "100000",
            "AURA_NETHACK_LONG_RUN_CONFIRM_FILE": str(tmp_path / "missing-confirmation"),
        }
    )
    env.pop("AURA_NETHACK_UNSAFE_RAM_CONFIRM", None)

    result = subprocess_gateway.SubprocessGateway().run(
        ["bash", "scripts/nethack_runner.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        timeout=5,
        capture_output=True,
        source="test.subprocess_gateway.nethack_runner_guard",
        accelerator_capability="none",
    )

    assert result.returncode == 64
    runner_log = Path.home() / ".aura/logs/nethack/runner.log"
    assert "without one-shot confirmation file" in runner_log.read_text(encoding="utf-8")


def test_offline_tooling_spawn_denied_when_live_governance_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        subprocess_gateway.SubprocessGateway().spawn(
            [sys.executable, "-c", "print('strict-spawn')"],
            offline_tooling=True,
            source="training_tooling:test",
            accelerator_capability="none",
        )


def test_semantic_weight_receipt_can_launch_exact_live_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from core.governance_context import governed_scope_sync

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    decision = SimpleNamespace(
        will_receipt_id="will-semantic-subprocess-1",
        domain="semantic_weight_update",
        source="system_maintenance:crsm_closure",
        constraints={},
    )
    with governed_scope_sync(decision):
        result = subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('semantic-governed')"],
            timeout=5,
            source="system_maintenance:crsm_closure:test",
            accelerator_capability="none",
        )

    assert result.returncode == 0
    assert result.stdout.strip() == "semantic-governed"


def test_offline_tooling_spawn_async_denied_when_live_governance_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('strict-async-spawn')"],
            offline_tooling=True,
            source="maintenance_tooling:test",
            accelerator_capability="none",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        asyncio.run(_attempt())


def test_proof_tooling_run_allowed_in_test_mode_with_live_governance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.setenv("AURA_TEST_MODE", "1")

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('proof-ok')"],
        timeout=5,
        offline_tooling=True,
        source="proof_tooling:test",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "proof-ok"


def test_proof_tooling_run_allowed_with_explicit_child_test_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)
    child_env = os.environ.copy()
    child_env["AURA_TEST_MODE"] = "1"

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('proof-env-ok')"],
        timeout=5,
        env=child_env,
        offline_tooling=True,
        source="certification_tooling:test",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "proof-env-ok"


def test_non_proof_tooling_still_denied_in_test_mode_with_live_governance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.setenv("AURA_TEST_MODE", "1")

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('training-denied')"],
            timeout=5,
            offline_tooling=True,
            source="training_tooling:test",
            accelerator_capability="none",
        )


def test_spawn_routes_stdout_and_stderr_to_gateway_owned_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    stdout_path = tmp_path / "child.stdout"
    stderr_path = tmp_path / "child.stderr"

    proc = subprocess_gateway.SubprocessGateway().spawn(
        [
            sys.executable,
            "-c",
            "import sys; print('gateway-out'); print('gateway-err', file=sys.stderr)",
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        source="test.subprocess_gateway.path_streams",
        accelerator_capability="none",
    )
    assert proc.wait(timeout=5) == 0
    for stream in getattr(proc, "_aura_gateway_streams", ()):
        stream.close()

    assert stdout_path.read_text(encoding="utf-8").strip() == "gateway-out"
    assert stderr_path.read_text(encoding="utf-8").strip() == "gateway-err"


def test_spawn_accepts_preexec_fn_for_resource_fenced_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    proc = subprocess_gateway.SubprocessGateway().spawn(
        [sys.executable, "-c", "print('preexec-supported')"],
        preexec_fn=None,
        source="test.subprocess_gateway.preexec_none",
        accelerator_capability="none",
    )
    assert proc.wait(timeout=5) == 0
