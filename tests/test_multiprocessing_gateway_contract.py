from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from core.runtime import subprocess_gateway as gateway_module
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    GovernanceViolation,
    PythonProcessOwnershipError,
    PythonProcessSpec,
    SubprocessGateway,
    python_process_contract,
    python_process_role,
)


def _target() -> None:
    return None


def _report_secret_presence(result_queue) -> None:
    result_queue.put("AURA_TEST_API_KEY" in os.environ)


class _FakeProcess:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = kwargs.get("name") or "fake-process"
        self.pid = None
        self.started = False
        self.alive = False
        self.terminated = False
        self.killed = False
        self.joins: list[float] = []

    def start(self) -> None:
        self.started = True
        self.alive = True
        self.pid = 4242

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(float(timeout or 0.0))


class _FakeContext:
    def __init__(self, start_method: str = "spawn") -> None:
        self.start_method = start_method
        self.created: list[_FakeProcess] = []

    def get_start_method(self) -> str:
        return self.start_method

    def Process(self, **kwargs):  # noqa: N802 - multiprocessing API
        process = _FakeProcess(**kwargs)
        self.created.append(process)
        return process


def _spec(**overrides) -> PythonProcessSpec:
    values = {
        "target": _target,
        "source": "shadow_kernel.sandbox_validation",
        "name": "shadow-validator",
        "role": ProcessRole.UNTRUSTED_CODE,
        "accelerator_capability": AcceleratorCapability.NONE,
        "start_method": "spawn",
    }
    values.update(overrides)
    return PythonProcessSpec(**values)


def test_gateway_binds_role_and_registers_started_child(monkeypatch) -> None:
    context = _FakeContext()
    registrations = []
    monkeypatch.setattr(
        gateway_module,
        "_register_runtime_hygiene_process",
        lambda process, **metadata: registrations.append((process, metadata)) or True,
    )
    gateway = SubprocessGateway()

    process = gateway.spawn_python_process(_spec(), context=context)

    assert context.created == [process]
    assert process.started is True
    assert registrations[0][0] is process
    assert registrations[0][1]["source"] == "shadow_kernel.sandbox_validation"
    contract = getattr(process, gateway._PYTHON_PROCESS_CONTRACT_ATTRIBUTE)
    assert contract["role"] == "untrusted_code"
    assert contract["requested_privileges"] == ()
    assert contract["accelerator_capability"] == "none"
    assert python_process_contract(process) == contract
    assert python_process_role(process) is ProcessRole.UNTRUSTED_CODE


def test_gateway_rejects_privilege_escalation_before_factory() -> None:
    context = _FakeContext()

    with pytest.raises(GovernanceViolation, match="python_process_privilege_denied"):
        SubprocessGateway().spawn_python_process(
            _spec(requested_privileges=frozenset({Privilege.NETWORK})),
            context=context,
        )

    assert context.created == []


@pytest.mark.parametrize(
    ("role", "accelerator"),
    (
        (ProcessRole.MODEL_WORKER, AcceleratorCapability.NONE),
        (ProcessRole.COORDINATOR, AcceleratorCapability.MODEL),
    ),
)
def test_gateway_rejects_role_accelerator_mismatch(role, accelerator) -> None:
    with pytest.raises(GovernanceViolation, match="role_accelerator_mismatch"):
        SubprocessGateway().spawn_python_process(
            _spec(role=role, accelerator_capability=accelerator),
            context=_FakeContext(),
        )


def test_gateway_requires_explicit_accelerator_declaration() -> None:
    with pytest.raises(GovernanceViolation, match="must_be_explicit"):
        SubprocessGateway().spawn_python_process(
            _spec(accelerator_capability=AcceleratorCapability.AUTO),
            context=_FakeContext(),
        )


def test_sensitive_environment_override_requires_declared_secret_privilege() -> None:
    with pytest.raises(GovernanceViolation, match="secret_override_denied"):
        SubprocessGateway().spawn_python_process(
            _spec(
                role=ProcessRole.COORDINATOR,
                environment_overrides={"AURA_TEST_API_KEY": "not-a-real-secret"},
            ),
            context=_FakeContext(),
        )


def test_declared_and_admitted_secret_override_is_preserved(monkeypatch) -> None:
    context = _FakeContext()
    monkeypatch.setattr(
        gateway_module,
        "_register_runtime_hygiene_process",
        lambda *_args, **_kwargs: True,
    )

    process = SubprocessGateway().spawn_python_process(
        _spec(
            role=ProcessRole.COORDINATOR,
            requested_privileges=frozenset({Privilege.SECRETS}),
            environment_overrides={"AURA_TEST_API_KEY": "not-a-real-secret"},
        ),
        context=context,
    )

    assert process.kwargs["args"][-1] is False


def test_gateway_reaps_child_when_registration_is_not_observed(monkeypatch) -> None:
    context = _FakeContext()
    monkeypatch.setattr(
        gateway_module,
        "_register_runtime_hygiene_process",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(
        PythonProcessOwnershipError,
        match="python_process_registration_failed",
    ):
        SubprocessGateway().spawn_python_process(_spec(), context=context)

    process = context.created[0]
    assert process.terminated is True
    assert process.alive is False


def test_gateway_refuses_shutdown_before_factory(monkeypatch) -> None:
    context = _FakeContext()
    monkeypatch.setattr(gateway_module, "is_shutdown_requested", lambda: True)

    with pytest.raises(GovernanceViolation, match="runtime shutdown"):
        SubprocessGateway().spawn_python_process(_spec(), context=context)

    assert context.created == []


def test_gateway_reaps_child_when_shutdown_crosses_start(monkeypatch) -> None:
    context = _FakeContext()
    monkeypatch.setattr(
        gateway_module,
        "_register_runtime_hygiene_process",
        lambda *_args, **_kwargs: True,
    )
    checks = iter((False, False, True))
    monkeypatch.setattr(gateway_module, "is_shutdown_requested", lambda: next(checks))

    with pytest.raises(GovernanceViolation, match="runtime shutdown"):
        SubprocessGateway().spawn_python_process(_spec(), context=context)

    process = context.created[0]
    assert process.terminated is True
    assert process.alive is False


def test_low_trust_real_child_cannot_observe_parent_secret(monkeypatch) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    monkeypatch.setenv("AURA_TEST_API_KEY", "not-a-real-secret")
    monkeypatch.setattr(
        gateway_module,
        "_register_runtime_hygiene_process",
        lambda *_args, **_kwargs: True,
    )

    process = SubprocessGateway().spawn_python_process(
        _spec(target=_report_secret_presence, args=(result_queue,)),
        context=context,
    )
    process.join(timeout=10.0)
    try:
        assert process.exitcode == 0
        assert result_queue.get(timeout=2.0) is False
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        result_queue.close()
        result_queue.join_thread()


def test_coordinator_without_secret_declaration_cannot_observe_parent_secret(
    monkeypatch,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    monkeypatch.setenv("AURA_TEST_API_KEY", "not-a-real-secret")
    monkeypatch.setattr(
        gateway_module,
        "_register_runtime_hygiene_process",
        lambda *_args, **_kwargs: True,
    )

    process = SubprocessGateway().spawn_python_process(
        _spec(
            target=_report_secret_presence,
            args=(result_queue,),
            role=ProcessRole.COORDINATOR,
        ),
        context=context,
    )
    process.join(timeout=10.0)
    try:
        assert process.exitcode == 0
        assert result_queue.get(timeout=2.0) is False
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        result_queue.close()
        result_queue.join_thread()
