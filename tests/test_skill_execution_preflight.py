from __future__ import annotations

import json
import threading
from typing import Any

import pytest

import core.capability_engine as capability_module
from core.capability_engine import CapabilityEngine, SkillMetadata, SkillRequirements
from core.runtime.lockdep import assert_no_locks_held
from core.skills.base_skill import BaseSkill


class _DependencySkill(BaseSkill):
    name = "preflight_dependency_fixture"
    description = "Exercises the real constructor dependency path."
    effect_scope = "pure_compute"
    constructor_calls = 0
    execution_calls = 0

    def __init__(self, dependency: Any):
        type(self).constructor_calls += 1
        self.dependency = dependency

    async def execute(self, params, context):
        type(self).execution_calls += 1
        return {"ok": True}


def _fixture_engine(*, schema: dict[str, Any] | None = None) -> CapabilityEngine:
    engine = CapabilityEngine()
    metadata = SkillMetadata(
        name=_DependencySkill.name,
        description=_DependencySkill.description,
        module_path=__name__,
        class_name=_DependencySkill.__name__,
        requirements=SkillRequirements(),
        effect_scope="pure_compute",
        authority_class="internal_compute",
        schema_override=(
            schema
            if schema is not None
            else {"additionalProperties": True, "properties": {}, "type": "object"}
        ),
        catalog_id="preflight-fixture",
        constructor_dependencies=["dependency"],
        validation_state="valid",
    )
    engine._catalog_loaded = True
    engine._catalog_digest = "fixture-catalog"
    engine._skills = {_DependencySkill.name: metadata}
    engine._instances = {}
    engine.active_skills = {_DependencySkill.name}
    engine.catalog_health = {"ready": True, "reason": "ready"}
    engine._catalog_preflight_summary = {
        "catalog_digest": "fixture-catalog",
        "complete": False,
        "entries": [],
        "failed": [],
        "live_count": 1,
        "ok": False,
        "reason": "not_run",
    }
    return engine


def test_preflight_resolves_real_service_constructs_once_and_never_executes(monkeypatch):
    dependency = object()
    _DependencySkill.constructor_calls = 0
    _DependencySkill.execution_calls = 0
    monkeypatch.setattr(
        capability_module,
        "optional_service",
        lambda name, default=None: dependency if name == "dependency" else default,
    )
    engine = _fixture_engine()

    first = engine.preflight_skill(_DependencySkill.name)
    second = engine.preflight_skill(_DependencySkill.name)

    assert first["ok"] is True
    assert first["stage"] == "ready"
    assert first["constructor_dependencies"] == ["dependency"]
    assert first["constructor_invoked"] is True
    assert first["skill_body_invoked"] is False
    assert second == first
    assert _DependencySkill.constructor_calls == 1
    assert _DependencySkill.execution_calls == 0
    assert engine._instances[_DependencySkill.name].dependency is dependency


def test_preflight_constructs_without_holding_catalog_locks(monkeypatch):
    dependency = object()
    observed: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        capability_module,
        "optional_service",
        lambda name, default=None: dependency if name == "dependency" else default,
    )
    engine = _fixture_engine()
    original_construct = engine._construct_skill_instance

    def construct(metadata, skill_class, *, dependencies=None):
        observed.append(
            tuple(assert_no_locks_held("skill constructor", strict=True))
        )
        return original_construct(
            metadata,
            skill_class,
            dependencies=dependencies,
        )

    monkeypatch.setattr(engine, "_construct_skill_instance", construct)

    receipt = engine.preflight_skill(_DependencySkill.name, refresh=True)

    assert receipt["ok"] is True
    assert observed == [()]


def test_concurrent_preflight_publishes_one_owner_and_retires_the_duplicate(monkeypatch):
    dependency = object()
    constructors_met = threading.Barrier(2, timeout=5.0)
    constructed: list[Any] = []
    retired: list[int] = []
    monkeypatch.setattr(
        capability_module,
        "optional_service",
        lambda name, default=None: dependency if name == "dependency" else default,
    )
    engine = _fixture_engine()

    class PreparedOwner:
        name = _DependencySkill.name

        def close(self):
            retired.append(id(self))

    def construct(_metadata, _skill_class, *, dependencies=None):
        assert dependencies == {"dependency": dependency}
        owner = PreparedOwner()
        constructed.append(owner)
        constructors_met.wait()
        return owner

    monkeypatch.setattr(engine, "_construct_skill_instance", construct)
    receipts: list[dict[str, Any]] = []
    instances: list[Any] = []

    def prepare() -> None:
        receipt, instance = engine._prepare_skill_instance(
            _DependencySkill.name,
            engine._skills[_DependencySkill.name],
            refresh=True,
        )
        receipts.append(receipt)
        instances.append(instance)

    threads = [threading.Thread(target=prepare) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(constructed) == 2
    assert len(receipts) == 2
    assert all(receipt["ok"] is True for receipt in receipts)
    assert instances[0] is instances[1]
    assert engine._instances[_DependencySkill.name] is instances[0]
    assert retired == [
        id(owner) for owner in constructed if owner is not instances[0]
    ]


def test_preflight_reports_exact_missing_dependency_stage_without_constructing(monkeypatch):
    _DependencySkill.constructor_calls = 0
    monkeypatch.setattr(capability_module, "optional_service", lambda _name, default=None: default)
    engine = _fixture_engine()

    receipt = engine.preflight_skill(_DependencySkill.name)

    assert receipt["ok"] is False
    assert receipt["stage"] == "dependency_resolution"
    assert "dependency" in receipt["error"]
    assert receipt["skill_body_invoked"] is False
    assert _DependencySkill.constructor_calls == 0
    item = engine.get_tool_catalog()[0]
    assert item["available"] is False
    assert item["preflight_state"] == "failed"


def test_preflight_rejects_malformed_schema_before_dependency_resolution(monkeypatch):
    service_lookups: list[str] = []

    def resolve(name, default=None):
        service_lookups.append(name)
        return object()

    monkeypatch.setattr(capability_module, "optional_service", resolve)
    engine = _fixture_engine(schema={"properties": {}})

    receipt = engine.preflight_skill(_DependencySkill.name)

    assert receipt["ok"] is False
    assert receipt["stage"] == "schema"
    assert service_lookups == []


def test_catalog_dry_run_is_complete_cached_and_effect_free(monkeypatch):
    dependency = object()
    _DependencySkill.constructor_calls = 0
    _DependencySkill.execution_calls = 0
    monkeypatch.setattr(
        capability_module,
        "optional_service",
        lambda name, default=None: dependency if name == "dependency" else default,
    )
    engine = _fixture_engine()

    first = engine.dry_run_catalog()
    second = engine.dry_run_catalog()

    assert first["ok"] is True
    assert first["complete"] is True
    assert first["failed"] == []
    assert first["entries"][0]["skill_body_invoked"] is False
    assert second == first
    assert _DependencySkill.constructor_calls == 1
    assert _DependencySkill.execution_calls == 0
    assert engine.get_catalog_health()["execution_preflight"]["ok"] is True


@pytest.mark.asyncio
async def test_engine_shutdown_retires_unique_sync_and_async_skill_owners():
    calls: list[str] = []

    class SyncOwner:
        def close(self):
            calls.append("sync")

    class AsyncOwner:
        async def shutdown(self):
            calls.append("async")

    engine = _fixture_engine()
    sync_owner = SyncOwner()
    engine._instances = {
        "sync": sync_owner,
        "sync_alias": sync_owner,
        "async": AsyncOwner(),
    }

    await engine.on_stop_async()

    assert sorted(calls) == ["async", "sync"]
    assert engine._instances == {}
    assert engine.active_skills == set()
    assert engine.get_catalog_preflight_status()["reason"] == "engine_stopped"


@pytest.mark.asyncio
async def test_engine_shutdown_drains_remaining_owners_before_reporting_failure():
    calls: list[str] = []

    class BrokenOwner:
        def close(self):
            calls.append("broken")
            raise RuntimeError("injected cleanup failure")

    class HealthyOwner:
        def close(self):
            calls.append("healthy")

    engine = _fixture_engine()
    engine._instances = {"broken": BrokenOwner(), "healthy": HealthyOwner()}

    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        await engine.on_stop_async()

    assert calls == ["broken", "healthy"]
    assert engine._instances == {}


@pytest.mark.asyncio
async def test_tools_catalog_endpoint_preserves_execution_preflight_health(monkeypatch):
    from interface.routes import system

    class FakeEngine:
        def get_catalog_health(self):
            return {
                "ready": False,
                "reason": "catalog_incomplete",
                "missing_live": ["z_fixture", "missing_fixture", "missing_fixture"],
                "quarantined_count": 1,
                "quarantined": [
                    {
                        "name": "quarantined_fixture",
                        "stage": "constructor",
                        "detail": "dependency unavailable",
                    }
                ],
                "execution_preflight": {
                    "complete": True,
                    "failed": ["z_broken", "broken_fixture", "broken_fixture"],
                    "ok": False,
                },
            }

    monkeypatch.setattr(system, "_collect_tool_catalog", lambda: [{"name": "clock"}])
    monkeypatch.setattr(
        system,
        "optional_service",
        lambda name, default=None: FakeEngine() if name == "capability_engine" else default,
    )

    response = await system.api_tools_catalog()
    payload = json.loads(response.body)

    assert payload["health"]["missing_live"] == ["missing_fixture", "z_fixture"]
    assert payload["health"]["quarantined"] == [
        {
            "catalog_id": "",
            "class_name": "",
            "error": "dependency unavailable",
            "module_path": "",
            "name": "quarantined_fixture",
            "stage": "constructor",
        }
    ]
    assert payload["health"]["execution_preflight"]["failed"] == [
        "broken_fixture",
        "z_broken",
    ]
    assert payload["health"]["execution_preflight"]["ok"] is False


def test_skills_ui_renders_catalog_and_preflight_truth():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    html = (root / "interface" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (root / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert 'id="tool-catalog-state"' in html
    assert 'id="tool-preflight-state"' in html
    assert 'id="tool-catalog-detail"' in html
    assert 'id="tool-catalog-issues"' in html
    assert "d.health && typeof d.health === 'object'" in javascript
    assert "payload.skill_catalog || null" in javascript
    assert "health.execution_preflight" in javascript
    assert "MISSING LIVE" in javascript
    assert "PREFLIGHT FAILED" in javascript
    assert "QUARANTINED" in javascript
    assert "preflight ${escHtml" in javascript
