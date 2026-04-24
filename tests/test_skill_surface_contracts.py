from __future__ import annotations

import asyncio
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.capability_engine import CapabilityEngine
from core.skills.install_package import InstallPackageSkill
from core.skills.self_evolution import SelfEvolutionSkill
from core.skills.self_improvement import SelfImprovementSkill
from core.skills.test_generator import TestGeneratorSkill
from core.skills.toggle_senses import ToggleSensesSkill


EXPECTED_REGISTERED_SKILLS = {
    "ManageAbilities",
    "add_belief",
    "auto_refactor",
    "clock",
    "coding_skill",
    "cognitive_trainer",
    "computer_use",
    "curiosity",
    "delegate_shard",
    "deploy_ghost_probe",
    "dream_sleep",
    "embodiment",
    "environment_info",
    "evolution_status",
    "file_operation",
    "force_dream_cycle",
    "free_search",
    "grounded_search",
    "install_package",
    "inter_agent_comm",
    "internal_sandbox",
    "listen",
    "malware_analysis",
    "manifest_to_device",
    "memory_ops",
    "memory_sync",
    "native_chat",
    "notify_user",
    "os_manipulation",
    "personality",
    "plan_mode",
    "propagation",
    "query_beliefs",
    "query_visual_context",
    "run_code",
    "search_web",
    "sec_ops",
    "self_evolution",
    "self_improvement",
    "self_repair",
    "social_lurker",
    "sovereign_browser",
    "sovereign_imagination",
    "sovereign_network",
    "sovereign_terminal",
    "sovereign_vision",
    "spawn_agent",
    "spawn_agents_parallel",
    "speak",
    "stealth_ops",
    "system_proprioception",
    "test_generator",
    "toggle_senses",
    "train_self",
    "uplink_local",
    "web_search",
}


class _MemoryFacadeStub:
    async def add_memory(self, *_args, **_kwargs):
        return None


class _BrainStub:
    async def think(self, *args, **kwargs):
        return SimpleNamespace(content="def test_generated_placeholder():\n    assert True\n")

    async def generate(self, *args, **kwargs):
        return {"response": "pass", "thought": "stubbed"}


@pytest.fixture(scope="module")
def skill_registry() -> dict[str, Any]:
    engine = CapabilityEngine()
    engine.reload_skills()
    return dict(engine.skills)


def _instantiate_skill(skill_name: str, meta: Any) -> Any:
    module = importlib.import_module(meta.module_path)
    cls = getattr(module, meta.class_name)
    if skill_name == "cognitive_trainer":
        return cls(memory_facade=_MemoryFacadeStub())
    if skill_name == "test_generator":
        return cls(brain=_BrainStub())

    sig = inspect.signature(cls)
    if "brain" in sig.parameters:
        return cls(brain=None)
    return cls()


def _params_for_skill(skill_name: str, tmp_path: Path) -> dict[str, Any]:
    test_file = tmp_path / "sample_module.py"
    test_file.write_text("def ok():\n    return 1\n", encoding="utf-8")

    overrides = {
        "ManageAbilities": {"action": "activate", "skill_name": "clock"},
        "add_belief": {"source": "Bryan", "relation": "prefers", "target": "Python"},
        "auto_refactor": {"path": str(tmp_path), "run_tests": False},
        "coding_skill": {"objective": "", "params": {"task": ""}},
        "cognitive_trainer": {"dataset_name": "unsupported", "limit": 1, "dry_run": True},
        "computer_use": {"action": "click", "x": 1, "y": 1},
        "curiosity": {"topic": ""},
        "delegate_shard": {"objective": "review this"},
        "deploy_ghost_probe": {"resource": "sample.txt"},
        "file_operation": {"action": "exists", "path": "."},
        "free_search": {"query": ""},
        "grounded_search": {"objective": ""},
        "install_package": {"package_name": "bad package!"},
        "inter_agent_comm": {"agent_name": "", "message": ""},
        "listen": {"duration": 0.01},
        "malware_analysis": {"path": str(tmp_path / "missing.bin")},
        "manifest_to_device": {"url": "notaurl"},
        "memory_ops": {"action": "unknown"},
        "notify_user": {"message": "Skill contract sweep complete."},
        "os_manipulation": {"action": "click", "x": 1, "y": 1},
        "personality": {"action": "list"},
        "plan_mode": {"objective": "enter"},
        "propagation": {"action": "connect", "target_ip": "10.0.0.7"},
        "query_beliefs": {"subject": "Bryan"},
        "query_visual_context": {"question": "what is on screen"},
        "run_code": {"code": "1 + 1"},
        "search_web": {"query": ""},
        "sec_ops": {"action": "bogus", "target": "localhost", "path": str(tmp_path)},
        "self_evolution": {
            "action": "propose",
            "objective": "Improve export stability.",
            "files": [str(test_file)],
        },
        "self_improvement": {"objective": "Improve resilience."},
        "social_lurker": {"source": "reddit"},
        "sovereign_browser": {"mode": "search"},
        "sovereign_imagination": {"prompt": "test"},
        "sovereign_network": {"mode": "status"},
        "sovereign_terminal": {"action": "execute"},
        "sovereign_vision": {"action": "look"},
        "spawn_agent": {"objective": "analyze this"},
        "spawn_agents_parallel": {"objectives": ["a", "b"]},
        "speak": {"text": "contract test"},
        "test_generator": {"target_file": str(tmp_path / "missing_target.py")},
        "toggle_senses": {"sense": "vision", "action": "off"},
        "web_search": {"query": ""},
    }
    return dict(overrides.get(skill_name, {}))


def _neutralize_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = RuntimeError("display access unavailable")

    import core.skills.computer_use as computer_use
    import core.skills.listen as listen
    import core.skills.notify_user as notify_user
    import core.skills.os_manipulation as os_manipulation
    import core.skills.social_lurker as social_lurker
    import core.skills.speak as speak
    import core.skills.sovereign_browser as sovereign_browser
    import core.skills.vision_actor as vision_actor
    from core.skills.auto_refactor import AutoRefactorSkill
    from core.skills.speak import SpeakSkill

    monkeypatch.setattr(computer_use, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(os_manipulation, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(vision_actor, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(listen, "_record_sync", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mic unavailable")))
    monkeypatch.setattr(notify_user.DesktopNotifier, "send", staticmethod(lambda **_kwargs: None))
    monkeypatch.setattr(AutoRefactorSkill, "_publish_proposals", lambda self, issues: None)
    monkeypatch.setattr(SpeakSkill, "_get_engine", lambda self: SimpleNamespace(synthesize_speech=lambda _text: asyncio.sleep(0)))
    monkeypatch.setattr(social_lurker, "PLAYWRIGHT", False)

    async def _raise_browser_unavailable(self):
        raise RuntimeError("browser unavailable during contract sweep")

    monkeypatch.setattr(
        sovereign_browser.SovereignBrowserSkill,
        "_create_browser",
        _raise_browser_unavailable,
    )
    monkeypatch.setattr(
        sovereign_browser.SovereignBrowserSkill,
        "_execute_fallback",
        lambda self, params: asyncio.sleep(0, result={"ok": False, "error": "fallback disabled in contract sweep"}),
    )


def _disable_governance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.governance_context.require_governance", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.governance_context.governance_runtime_active", lambda: False)


def test_registered_skill_surface_matches_expected_catalog(skill_registry):
    assert set(skill_registry) == EXPECTED_REGISTERED_SKILLS
    assert len(skill_registry) == 56


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_name", sorted(EXPECTED_REGISTERED_SKILLS))
async def test_registered_skills_support_safe_execute_contract(
    skill_name: str,
    skill_registry: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _neutralize_side_effects(monkeypatch)
    _disable_governance(monkeypatch)
    instance = _instantiate_skill(skill_name, skill_registry[skill_name])
    assert hasattr(instance, "safe_execute"), f"{skill_name} is missing safe_execute"

    result = await asyncio.wait_for(
        instance.safe_execute(_params_for_skill(skill_name, tmp_path), {}),
        timeout=4,
    )

    assert isinstance(result, dict)
    assert "ok" in result
    assert result.get("skill") == skill_name


@pytest.mark.asyncio
async def test_self_evolution_generates_fallback_proposal_without_brain(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    evolution_dir = tmp_path / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        SelfEvolutionSkill,
        "_evolution_dir",
        staticmethod(lambda: evolution_dir),
    )

    target = tmp_path / "export_source.py"
    target.write_text(
        "\n".join(
            [
                "def get_priority():",
                *["    value = 1" for _ in range(55)],
                "    return value",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    skill = SelfEvolutionSkill()
    skill.code_base = tmp_path

    result = await skill.safe_execute(
        {
            "action": "propose",
            "objective": "Refactor export priority planning.",
            "files": [str(target)],
        },
        {"proprioception": {"memory_percent": 42.0}},
    )

    assert result["ok"] is True
    assert result["fallback"] is True
    proposal_path = Path(result["proposal_path"])
    assert proposal_path.exists()
    proposal_text = proposal_path.read_text(encoding="utf-8")
    assert "deterministic fallback" in proposal_text
    assert "get_priority" in proposal_text


@pytest.mark.asyncio
async def test_self_improvement_reflects_without_brain(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    monkeypatch.setattr(SelfImprovementSkill, "_resolve_brain", staticmethod(lambda: None))

    skill = SelfImprovementSkill()
    skill.learning_log_path = tmp_path / "learning_history.json"

    result = await skill.safe_execute(
        {"objective": "Improve resilience."},
        {"stats": {"cycle_count": 7}},
    )

    assert result["ok"] is True
    plan = result["result"]["improvement_plan"]
    assert any("Improve resilience" in item for item in plan)


@pytest.mark.asyncio
async def test_install_package_awaits_async_sandbox_command(monkeypatch):
    _disable_governance(monkeypatch)
    skill = InstallPackageSkill()

    async def _run_command(*_args, **_kwargs):
        return SimpleNamespace(exit_code=0, stdout="installed", stderr="")

    monkeypatch.setattr(
        "core.skills.install_package.get_sandbox",
        lambda: SimpleNamespace(run_command=_run_command),
    )

    result = await skill.safe_execute({"package_name": "demo-package"}, {})

    assert result["ok"] is True
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_test_generator_falls_back_to_deterministic_smoke_without_brain(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "sample_module.py"
    target.write_text(
        "def square(value: int) -> int:\n"
        "    return value * value\n",
        encoding="utf-8",
    )

    skill = TestGeneratorSkill(brain=None)
    result = await skill.safe_execute({"target_file": str(target)}, {})

    assert result["ok"] is True
    assert Path(result["test_file"]).exists()
    assert "1 passed" in str(result.get("output") or "")


@pytest.mark.asyncio
async def test_test_generator_read_only_avoids_writing_into_repo(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "sample_module.py"
    target.write_text(
        "def square(value: int) -> int:\n"
        "    return value * value\n",
        encoding="utf-8",
    )

    skill = TestGeneratorSkill(brain=None)
    result = await skill.safe_execute({"target_file": str(target)}, {"read_only": True})

    assert result["ok"] is True
    assert Path(result["test_file"]).exists()
    assert not (target.parent / f"test_{target.name}").exists()


@pytest.mark.asyncio
async def test_test_generator_brain_uses_objective_keyword(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "export_source.py"
    target.write_text(
        "def export_source() -> str:\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    class _ObjectiveOnlyBrain:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def think(self, objective, **kwargs):
            self.calls.append({"objective": objective, **kwargs})
            return SimpleNamespace(content="def test_generated_placeholder():\n    assert True\n")

    class _SandboxStub:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.files: dict[str, str] = {}
            self.command = ""

        def start(self) -> None:
            self.started = True

        def write_file(self, name: str, content: str) -> None:
            self.files[name] = content

        async def run_command(self, command: str, timeout: int = 45):
            self.command = command
            return SimpleNamespace(exit_code=0, stdout="1 passed", stderr="")

        def stop(self) -> None:
            self.stopped = True

    brain = _ObjectiveOnlyBrain()
    sandbox = _SandboxStub()
    monkeypatch.setattr("core.sovereign.local_sandbox.LocalSandbox", lambda: sandbox)

    skill = TestGeneratorSkill(brain=brain)
    result = await skill.safe_execute({"target_file": str(target)}, {})

    assert result["ok"] is True
    assert Path(result["test_file"]).exists()
    assert brain.calls
    assert brain.calls[0]["origin"] == "test_generator"
    assert brain.calls[0]["context"]["target"] == str(target)
    assert "export_source.py" in brain.calls[0]["objective"]
    assert sandbox.started is True
    assert sandbox.stopped is True
    assert "pytest -q" in sandbox.command


@pytest.mark.asyncio
async def test_self_evolution_propose_read_only_skips_proposal_file(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    evolution_dir = tmp_path / "evolution"
    monkeypatch.setattr(
        SelfEvolutionSkill,
        "_evolution_dir",
        staticmethod(lambda: evolution_dir),
    )

    skill = SelfEvolutionSkill()
    result = await skill.safe_execute(
        {"action": "propose", "objective": "Draft a safe refactor plan."},
        {"read_only": True},
    )

    assert result["ok"] is True
    assert result["read_only"] is True
    assert result.get("proposal_path") in (None, "")
    assert not list(evolution_dir.glob("evolution_proposal_*.md"))


@pytest.mark.asyncio
async def test_toggle_senses_uses_subprocess_runner_without_local_sandbox(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)

    sense_dir = tmp_path / "senses"
    sense_dir.mkdir(parents=True, exist_ok=True)
    (sense_dir / "vision_service.py").write_text(
        "import time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    import core.skills.toggle_senses as toggle_senses

    monkeypatch.setattr(
        toggle_senses,
        "config",
        SimpleNamespace(paths=SimpleNamespace(project_root=tmp_path, data_dir=tmp_path / "data")),
    )

    skill = ToggleSensesSkill()
    on_result = await skill.safe_execute({"sense": "vision", "action": "on"}, {})
    assert on_result["ok"] is True
    assert isinstance(on_result.get("pid"), int)

    off_result = await skill.safe_execute(
        {"sense": "vision", "action": "off", "pid": on_result["pid"]},
        {},
    )
    assert off_result["ok"] is True
