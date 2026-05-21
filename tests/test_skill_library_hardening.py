from types import SimpleNamespace
from typing import Any

import pytest

import core.agency.skill_library as skill_library
from core.agency.skill_library import SkillLibrary
from core.container import ServiceContainer
from core.runtime.errors import get_degradation_tracker


@pytest.fixture
def isolated_skill_library(tmp_path, monkeypatch):
    ServiceContainer.clear()
    monkeypatch.setattr(skill_library.config, "paths", SimpleNamespace(data_dir=tmp_path))
    lib = SkillLibrary()
    lib.data_path = tmp_path / "skills.json"
    lib.skills.clear()
    return lib


class RecordingToolOrchestrator:
    def __init__(self, result: dict[str, Any] | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result = result or {"ok": True, "summary": "done"}

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]):
        self.calls.append((tool_name, dict(arguments)))
        return dict(self.result)


def _learn_probe_skill(lib: SkillLibrary) -> None:
    lib.learn_skill(
        "Probe Skill",
        "Run a bounded probe",
        ["target"],
        [{"tool_name": "web_search", "arguments": {"query": "{{target}}"}}],
    )


@pytest.mark.asyncio
async def test_skill_library_executes_real_tool_orchestrator_with_templates(isolated_skill_library):
    lib = isolated_skill_library
    _learn_probe_skill(lib)
    orchestrator = RecordingToolOrchestrator()
    ServiceContainer.register_instance("tool_orchestrator", orchestrator)

    result = await lib.execute_skill("probe_skill", {"target": "Aura runtime health"})

    assert result == [{"ok": True, "summary": "done"}]
    assert orchestrator.calls == [("web_search", {"query": "Aura runtime health"})]
    assert lib.skills["probe_skill"].successes == 1
    assert lib.skills["probe_skill"].failures == 0


@pytest.mark.asyncio
async def test_skill_library_fails_closed_without_tool_orchestrator(isolated_skill_library):
    get_degradation_tracker().reset()
    lib = isolated_skill_library
    _learn_probe_skill(lib)

    with pytest.raises(RuntimeError, match="tool_orchestrator"):
        await lib.execute_skill("probe_skill", {"target": "Aura"})

    assert lib.skills["probe_skill"].failures == 1
    assert any(
        "tool orchestrator was unavailable" in record.action
        for record in get_degradation_tracker().recent(subsystem="skill_library_execution")
    )


@pytest.mark.asyncio
async def test_skill_library_tool_failure_marks_skill_failure(isolated_skill_library):
    get_degradation_tracker().reset()
    lib = isolated_skill_library
    _learn_probe_skill(lib)
    ServiceContainer.register_instance(
        "tool_orchestrator",
        RecordingToolOrchestrator(result={"ok": False, "error": "search backend offline"}),
    )

    with pytest.raises(RuntimeError, match="search backend offline"):
        await lib.execute_skill("probe_skill", {"target": "Aura"})

    assert lib.skills["probe_skill"].successes == 0
    assert lib.skills["probe_skill"].failures == 1
    assert any(
        "failed macro skill execution" in record.action
        for record in get_degradation_tracker().recent(subsystem="skill_library_execution")
    )


@pytest.mark.asyncio
async def test_skill_library_rejects_orchestrator_without_execute_tool(isolated_skill_library):
    lib = isolated_skill_library
    _learn_probe_skill(lib)
    ServiceContainer.register_instance("tool_orchestrator", object())

    with pytest.raises(RuntimeError, match="lacks execute_tool"):
        await lib.execute_skill("probe_skill", {"target": "Aura"})

    assert lib.skills["probe_skill"].failures == 1
