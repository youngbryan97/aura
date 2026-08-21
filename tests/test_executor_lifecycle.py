from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown
from core.utils import executor as executor_module


@pytest.fixture(autouse=True)
def _clean_executor_lifecycle() -> None:
    clear_shutdown_request()
    executor_module.shutdown_executors()
    yield
    clear_shutdown_request()
    executor_module.shutdown_executors()


def test_legacy_io_pool_is_named_and_registered_for_terminal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[dict[str, object]] = []

    class _Hygiene:
        def register_shutdown_resource(self, resource: object, **kwargs: object) -> None:
            registrations.append({"resource": resource, **kwargs})

    monkeypatch.setattr(
        "core.runtime.runtime_hygiene.get_runtime_hygiene",
        lambda: _Hygiene(),
    )

    pool = executor_module.get_io_executor()
    assert pool.submit(lambda: 7).result(timeout=1.0) == 7
    assert registrations[-1]["name"] == "legacy_io_thread_pool"
    closer = registrations[-1]["closer"]
    assert isinstance(closer, Callable)
    closer()


def test_new_executor_creation_is_refused_after_shutdown_latch() -> None:
    request_shutdown("unit-test")

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        executor_module.get_io_executor()


def test_runtime_pool_registers_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import executors

    names: list[str] = []

    class _Hygiene:
        def register_shutdown_resource(self, _resource: object, **kwargs: object) -> None:
            names.append(str(kwargs["name"]))

    monkeypatch.setattr(
        "core.runtime.runtime_hygiene.get_runtime_hygiene",
        lambda: _Hygiene(),
    )

    assert asyncio.run(executors.run_blocking_io(lambda: "ok")) == "ok"
    assert names == ["blocking_io_thread_pool"]


@pytest.mark.asyncio
async def test_pools_self_heal_after_external_shutdown():
    """The order-dependence family of 2026-07-12 (attention gates, lag
    budgets, this file): shutdown hygiene closed the module-level pools as
    registered resources, and every later client in the process died with
    'cannot schedule new futures after shutdown'. A shut-down pool in a
    process that continues serving earns a rebuilt one; real shutdown is
    still refused by the latch check in _register_pool."""
    from core.runtime import executors

    assert await executors.run_blocking_io(lambda: "io", timeout_s=3.0) == "io"
    executors.BLOCKING_IO_POOL.shutdown(wait=False, cancel_futures=True)
    assert await executors.run_blocking_io(lambda: "io2", timeout_s=3.0) == "io2"

    assert await executors.run_heavy_cpu(lambda: "cpu", timeout_s=3.0) == "cpu"
    executors.HEAVY_CPU_POOL.shutdown(wait=False, cancel_futures=True)
    assert await executors.run_heavy_cpu(lambda: "cpu2", timeout_s=3.0) == "cpu2"

    assert await executors.run_durable_receipt_io(
        lambda: "receipt", timeout_s=3.0
    ) == "receipt"
    executors.DURABLE_RECEIPT_POOL.shutdown(wait=False, cancel_futures=True)
    assert await executors.run_durable_receipt_io(
        lambda: "receipt2", timeout_s=3.0
    ) == "receipt2"
