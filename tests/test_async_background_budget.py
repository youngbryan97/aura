"""The shared resource budget must not stop the runtime event loop."""

from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_async_budget_keeps_event_loop_schedulable(monkeypatch) -> None:
    import core.runtime.background_policy as background_policy

    ticks = 0

    def slow_budget(*args, **kwargs):
        nonlocal ticks
        del args, kwargs
        time.sleep(0.12)
        return SimpleNamespace(interval_s=1.0, effective_hz=1.0, reason="nominal")

    async def observe_loop() -> None:
        nonlocal ticks
        for _ in range(4):
            await asyncio.sleep(0.02)
            ticks += 1

    monkeypatch.setattr(background_policy, "constitutive_compute_budget", slow_budget)
    observer = asyncio.create_task(observe_loop())
    result = await background_policy.constitutive_compute_budget_async("test", 1.0)
    await observer

    assert result.reason == "nominal"
    assert ticks == 4


def test_async_budget_preserves_sync_policy_defaults() -> None:
    from core.runtime.background_policy import (
        constitutive_compute_budget,
        constitutive_compute_budget_async,
    )

    sync_parameters = inspect.signature(constitutive_compute_budget).parameters
    async_parameters = inspect.signature(constitutive_compute_budget_async).parameters

    assert tuple(async_parameters) == tuple(sync_parameters)
    for name, parameter in sync_parameters.items():
        assert async_parameters[name].default == parameter.default
