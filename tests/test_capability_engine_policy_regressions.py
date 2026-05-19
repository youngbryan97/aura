from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata
from core.container import ServiceContainer


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
    monkeypatch.setattr(
        "core.capability_engine.resolve_state_repository", lambda default=None: None
    )

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
