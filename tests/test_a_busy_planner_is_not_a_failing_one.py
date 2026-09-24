"""A background planning call that runs out of time is backpressure.

From one boot's log on 2026-09-23: 54 planner timeouts, each a warning, a
logged error and a failure against her resilience, for a plan the
deterministic fallback then made anyway.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine


@pytest.mark.asyncio
async def test_a_planner_timeout_is_not_counted_against_her(monkeypatch):
    async def slow(*_args, **_kwargs):
        raise asyncio.TimeoutError()

    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: SimpleNamespace(think=slow))})
    recorded: list = []
    failures: list = []
    monkeypatch.setattr(
        "core.agency.autonomous_task_engine.record_degradation",
        lambda *args, **kwargs: recorded.append(args),
    )
    resilience = SimpleNamespace(
        record_failure=lambda *a, **k: failures.append(a),
        record_success=lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: resilience if name == "resilience_engine" else default),
    )
    plan = await AutonomousTaskEngine(kernel)._decompose_goal("Inspect runtime health", "p", context=None)
    assert plan.steps, "the deterministic fallback still plans"
    assert recorded == []
    assert failures == []
