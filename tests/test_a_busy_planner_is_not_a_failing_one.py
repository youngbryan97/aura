"""A background planning call that runs out of time is backpressure.

From one boot's log on 2026-09-23: 54 planner timeouts, each a warning, a
logged error and a failure against her resilience, for a plan the
deterministic fallback then made anyway.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine


@pytest.fixture(autouse=True)
def _planning_by_her_model_on(monkeypatch):
    """These tests exercise the worded plan request, which is off by default (AURA_TASK_PLANNING_BY_MODEL)."""
    monkeypatch.setenv("AURA_TASK_PLANNING_BY_MODEL", "1")


@pytest.mark.asyncio
async def test_a_planner_timeout_is_not_counted_against_her(monkeypatch):
    async def slow(*_args, **_kwargs):
        raise TimeoutError()

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
    # Not planned, and said so (core/agency/autonomous_task_engine.py `_not_planned`), without counting against her.
    assert plan.steps == [] and plan.context["not_planned"].startswith("TimeoutError")
    assert recorded == []
    assert failures == []
