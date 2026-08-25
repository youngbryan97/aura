from __future__ import annotations

import asyncio
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata
from core.constitution import ConstitutionalDecision, ProposalKind, ProposalOutcome
from core.skills.install_package import InstallPackageSkill
from core.skills.propagation import PropagationSkill
from core.skills.sec_ops import SecOpsSkill
from core.skills.self_evolution import SelfEvolutionSkill
from core.skills.self_improvement import SelfImprovementSkill
from core.skills.stealth_ops import StealthOpsSkill
from core.skills.test_generator import TestGeneratorSkill
from core.skills.toggle_senses import ToggleSensesSkill

EXPECTED_REGISTERED_SKILLS = {
    "http_request",
    "pursue_on_screen",
    "reminder",
    "ManageAbilities",
    "add_belief",
    "auto_refactor",
    "build_app",
    "build_document",
    "clock",
    "code_repl",
    "coding_skill",
    "cognitive_trainer",
    "computer_use",
    "curiosity",
    "design_engineering",
    "desktop_task",
    "deploy_ghost_probe",
    "diagnose_repo",
    "dream_sleep",
    "embodiment",
    "email_adapter",
    "environment_info",
    "evolution_status",
    "execute_nethack_action",
    "file_operation",
    "force_dream_cycle",
    "free_search",
    "grounded_search",
    "image_gen",
    "improve_own_code",
    "install_package",
    "inter_agent_comm",
    "internal_sandbox",
    "knowledge_base",
    "listen",
    "local_reference_search",
    "malware_analysis",
    "manifest_to_device",
    "memory_ops",
    "messages",
    "memory_sync",
    "native_chat",
    "notify_user",
    "os_automation",
    "os_manipulation",
    "personality",
    "plan_mode",
    "propagation",
    "query_beliefs",
    "query_visual_context",
    "reddit_adapter",
    "render_bridge",
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
    "speak",
    "stealth_ops",
    "system_proprioception",
    "test_generator",
    "toggle_senses",
    "train_self",
    "uplink_local",
    "voice_mute",
    "voice_output",
    "voice_stop_tts",
    "voice_unmute",
    "web_interlocutor",
    "web_search",
    "program_dna_reconstruct",
    "program_dna_equivalence_battery",
    "x_tools",
    # 2026-07-12 capability corpus
    "quantum_lab",
    "world_forge",
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
        "auto_refactor": {"path": ".", "run_tests": False},
        "code_repl": {"code": "1 + 1", "timeout": 1, "capture_files": False},
        "coding_skill": {"objective": "", "params": {"task": ""}},
        "cognitive_trainer": {"dataset_name": "unsupported", "limit": 1, "dry_run": True},
        "computer_use": {"action": "click", "x": 1, "y": 1},
        "curiosity": {"action": "get_suggestion"},
        "desktop_task": {
            "objective": "Contract probe for desktop task validation.",
            "steps": [{"action": "wait", "target": "0"}],
        },
        "deploy_ghost_probe": {"resource": "sample.txt"},
        "email_adapter": {"mode": "check"},
        "execute_nethack_action": {"action": "look"},
        "file_operation": {"action": "exists", "path": "."},
        "free_search": {"query": ""},
        "grounded_search": {"objective": ""},
        "image_gen": {
            "prompt": "contract sweep fixture",
            "source_image_path": str(tmp_path / "missing-source.png"),
        },
        "install_package": {"package_name": "bad package!"},
        "inter_agent_comm": {"agent_name": "", "message": ""},
        "knowledge_base": {"action": "list", "limit": 1},
        "listen": {"duration": 0.01},
        "malware_analysis": {"path": str(tmp_path / "missing.bin")},
        "manifest_to_device": {"url": "notaurl"},
        "memory_ops": {"action": "unknown"},
        # Read-only status. Pinned rather than defaulted: this sweep calls
        # safe_execute on an external_io skill, so the action must be stated.
        "messages": {"action": "status"},
        "notify_user": {"message": "Skill contract sweep complete."},
        "os_automation": {
            "goal": "Open a visible app and prepare a short note.",
            "script_type": "applescript",
            "execute": False,
        },
        "os_manipulation": {"action": "click", "x": 1, "y": 1},
        "personality": {"action": "list"},
        "plan_mode": {"objective": "enter"},
        "propagation": {"action": "connect", "target_ip": "10.0.0.7"},
        "query_beliefs": {"subject": "Bryan"},
        "query_visual_context": {"question": "what is on screen"},
        "reddit_adapter": {"mode": "read_rules", "subreddit": "LocalLLaMA"},
        "render_bridge": {
            "instructions": [{"type": "progress", "content": {"percent": 1}}],
        },
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
        "speak": {"text": "contract test"},
        "test_generator": {"target_file": str(tmp_path / "missing_target.py")},
        "toggle_senses": {"sense": "vision", "action": "off"},
        "voice_output": {"text": ""},
        "web_interlocutor": {
            "objective": "Contract probe",
            "opening_message": "Hello.",
            "max_turns": 1,
            "wait_timeout_s": 5,
            "persist_memory": False,
        },
        "web_search": {"query": ""},
        "x_tools": {"action": "unknown"},
    }
    return dict(overrides.get(skill_name, {}))


def _neutralize_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = RuntimeError("display access unavailable")

    import core.skills.computer_use as computer_use
    import core.skills.email_adapter as email_adapter
    import core.skills.image_gen as image_gen
    import core.skills.listen as listen
    import core.skills.notify_user as notify_user
    import core.skills.os_manipulation as os_manipulation
    import core.skills.reddit_adapter as reddit_adapter
    import core.skills.social_lurker as social_lurker
    import core.skills.sovereign_browser as sovereign_browser
    import core.skills.vision_actor as vision_actor
    import core.skills.web_interlocutor as web_interlocutor
    from core.skills.auto_refactor import AutoRefactorSkill
    from core.skills.speak import SpeakSkill

    monkeypatch.setattr(computer_use, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(os_manipulation, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(vision_actor, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(
        listen,
        "_record_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mic unavailable")),
    )
    monkeypatch.setattr(notify_user.DesktopNotifier, "send", staticmethod(lambda **_kwargs: None))
    monkeypatch.setattr(AutoRefactorSkill, "_publish_proposals", lambda self, issues: None)
    monkeypatch.setattr(image_gen.ImageGenSkill, "_load_pipeline", lambda self, img2img=False: False)
    monkeypatch.setattr(
        SpeakSkill,
        "_get_engine",
        lambda self: SimpleNamespace(synthesize_speech=lambda _text: asyncio.sleep(0)),
    )
    monkeypatch.setattr(social_lurker, "PLAYWRIGHT", False)
    monkeypatch.setattr(
        email_adapter.EmailAdapterSkill,
        "_get_creds",
        lambda self: (_ for _ in ()).throw(
            RuntimeError("email credentials unavailable during contract sweep")
        ),
    )

    async def _raise_browser_unavailable(self):
        self.browser_unavailable_calls = getattr(self, "browser_unavailable_calls", 0) + 1
        raise RuntimeError("browser unavailable during contract sweep")

    monkeypatch.setattr(
        reddit_adapter.RedditAdapterSkill,
        "_create_browser",
        _raise_browser_unavailable,
    )

    monkeypatch.setattr(
        sovereign_browser.SovereignBrowserSkill,
        "_create_browser",
        _raise_browser_unavailable,
    )
    monkeypatch.setattr(
        sovereign_browser.SovereignBrowserSkill,
        "_execute_fallback",
        lambda self, params: asyncio.sleep(
            0, result={"ok": False, "error": "fallback disabled in contract sweep"}
        ),
    )

    async def _fake_web_interlocutor_run(self, **kwargs):
        from core.capabilities.web_interlocutor import WebInterlocutorResult

        return WebInterlocutorResult(
            ok=True,
            objective=str(kwargs.get("objective") or ""),
            learned_summary="contract sweep learned summary",
            status="completed",
        )

    monkeypatch.setattr(
        web_interlocutor.WebInterlocutorSession,
        "run",
        _fake_web_interlocutor_run,
    )


def _disable_governance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.governance_context.require_governance", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.governance_context.governance_runtime_active", lambda: False)


def _redirect_runtime_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))


def test_skill_brain_resolvers_never_cold_start_cognitive_runtime(monkeypatch):
    from core.container import ServiceContainer

    sentinel = object()
    lookups: list[str] = []

    def _peek(_cls, name: str, default=None):
        lookups.append(name)
        return sentinel

    def _forbidden_get(*_args, **_kwargs):
        raise AssertionError("skill resolution cold-started the cognitive service graph")

    monkeypatch.setattr(ServiceContainer, "peek", classmethod(_peek))
    monkeypatch.setattr(ServiceContainer, "get", classmethod(_forbidden_get))

    evolution = object.__new__(SelfEvolutionSkill)
    test_generator = TestGeneratorSkill(brain=None)

    assert SelfImprovementSkill._resolve_brain() is sentinel
    assert evolution._resolve_brain({}) is sentinel
    assert test_generator._resolve_brain() is sentinel
    assert lookups == ["cognitive_engine"] * 3


def test_capability_context_borrows_only_initialized_runtime_services(monkeypatch):
    from core.container import ServiceContainer

    lookups: list[str] = []

    def _peek(_cls, name: str, default=None):
        lookups.append(name)
        return default

    def _forbidden_get(*_args, **_kwargs):
        raise AssertionError("capability context cold-started a runtime service")

    monkeypatch.setattr(ServiceContainer, "peek", classmethod(_peek))
    monkeypatch.setattr(ServiceContainer, "get", classmethod(_forbidden_get))
    engine = object.__new__(CapabilityEngine)
    engine.orchestrator = None

    context = engine._augment_execution_context({})

    assert context == {"memory": None}
    assert lookups == [
        "orchestrator",
        "cognitive_engine",
        "memory_facade",
        "memory",
        "semantic_memory",
        "vector_memory",
        "theory_of_mind",
    ]


def test_registered_skill_surface_matches_expected_catalog(skill_registry):
    assert set(skill_registry) == EXPECTED_REGISTERED_SKILLS
    # Counted against the declaration rather than a literal. A third copy of
    # the number drifts from the other two: this list was already missing
    # pursue_on_screen and reminder when a new skill made the diff visible.
    assert len(skill_registry) == len(EXPECTED_REGISTERED_SKILLS)


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
    _redirect_runtime_memory(monkeypatch, tmp_path)
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
async def test_propagation_refuses_active_action_without_human_consent(monkeypatch: pytest.MonkeyPatch):
    _disable_governance(monkeypatch)
    skill = PropagationSkill()

    result = await skill.safe_execute({"action": "connect", "target_ip": "10.0.0.7"}, {})

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert "human_consent" in result["error"]
    assert result["plan"]["execution_performed"] is False


@pytest.mark.asyncio
async def test_propagation_refuses_public_target_without_allowlist(monkeypatch: pytest.MonkeyPatch):
    _disable_governance(monkeypatch)
    skill = PropagationSkill()

    result = await skill.safe_execute(
        {"action": "deploy_to_target", "target_ip": "8.8.8.8", "human_consent": True},
        {"operator_authorization": True},
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "blocked:public_target_requires_explicit_allowlist"
    assert result["plan"]["target_allowed"] is False
    assert result["plan"]["execution_performed"] is False


@pytest.mark.asyncio
async def test_propagation_public_target_requires_explicit_allowlist(monkeypatch: pytest.MonkeyPatch):
    _disable_governance(monkeypatch)
    skill = PropagationSkill()

    result = await skill.safe_execute(
        {"action": "deploy_to_target", "target_ip": "8.8.8.8", "human_consent": True},
        {"operator_authorization": True, "allowlisted_targets": ["8.8.8.8"]},
    )

    assert result["ok"] is True
    assert result["status"] == "authorized_plan_ready"
    assert result["plan"]["target_allowed"] is True
    assert result["plan"]["target_policy"] == "allowed:explicitly_allowlisted_public_target"
    assert result["plan"]["execution_performed"] is False


@pytest.mark.asyncio
async def test_sec_ops_audit_code_is_local_and_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "audit_target.py"
    target.write_text("def risky(value):\n    return eval(value)\n", encoding="utf-8")
    skill = SecOpsSkill()

    result = await skill.safe_execute({"action": "audit_code", "path": str(tmp_path)}, {})

    assert result["ok"] is True
    assert result["files_scanned"] == 1
    assert result["findings"][0]["rule_id"] == "dynamic_execution"
    assert target.read_text(encoding="utf-8") == "def risky(value):\n    return eval(value)\n"


@pytest.mark.asyncio
async def test_stealth_ops_is_privacy_hygiene_not_identity_rotation(monkeypatch: pytest.MonkeyPatch):
    _disable_governance(monkeypatch)
    skill = StealthOpsSkill()

    blocked = await skill.safe_execute({"params": {"command": "rotate"}}, {})
    scrubbed = await skill.safe_execute({"command": "scrub", "text": "token=abcdef123456"}, {})

    assert blocked["ok"] is False
    assert blocked["status"] == "blocked"
    assert "No network identity change was attempted" in blocked["message"]
    assert scrubbed["ok"] is True
    assert "[SECRET_REDACTED]" in scrubbed["text"]


@pytest.mark.asyncio
async def test_self_evolution_generates_fallback_proposal_without_brain(
    monkeypatch, tmp_path: Path
):
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
        {"brain": None, "proprioception": {"memory_percent": 42.0}},
    )

    assert result["ok"] is True
    assert result["fallback"] is True
    proposal_path = Path(result["proposal_path"])
    assert await asyncio.to_thread(proposal_path.exists)
    proposal_text = await asyncio.to_thread(proposal_path.read_text, encoding="utf-8")
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
async def test_test_generator_falls_back_to_deterministic_smoke_without_brain(
    monkeypatch, tmp_path: Path
):
    _disable_governance(monkeypatch)
    target = tmp_path / "sample_module.py"
    target.write_text(
        "def square(value: int) -> int:\n    return value * value\n",
        encoding="utf-8",
    )

    skill = TestGeneratorSkill(brain=None)
    result = await skill.safe_execute({"target_file": str(target)}, {})

    assert result["ok"] is True
    assert await asyncio.to_thread(Path(result["test_file"]).exists)
    assert "1 passed" in str(result.get("output") or "")


@pytest.mark.asyncio
async def test_test_generator_read_only_avoids_writing_into_repo(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "sample_module.py"
    target.write_text(
        "def square(value: int) -> int:\n    return value * value\n",
        encoding="utf-8",
    )

    skill = TestGeneratorSkill(brain=None)
    result = await skill.safe_execute(
        {"target_file": str(target), "read_only": True},
        {},
    )

    assert result["ok"] is True
    assert await asyncio.to_thread(Path(result["test_file"]).exists)
    assert not await asyncio.to_thread((target.parent / f"test_{target.name}").exists)


@pytest.mark.asyncio
async def test_test_generator_read_only_skips_llm_generation(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "sample_module.py"
    target.write_text(
        "def square(value: int) -> int:\n    return value * value\n",
        encoding="utf-8",
    )

    class _ForbiddenBrain:
        async def think(self, *args, **kwargs):
            self.think_calls = getattr(self, "think_calls", 0) + 1
            raise AssertionError("read-only test generation should not use the LLM path")

    skill = TestGeneratorSkill(brain=_ForbiddenBrain())
    result = await skill.safe_execute({"target_file": str(target)}, {"read_only": True})

    assert result["ok"] is True
    assert result.get("fallback_used") is False


@pytest.mark.asyncio
async def test_test_generator_brain_uses_objective_keyword(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    target = tmp_path / "export_source.py"
    target.write_text(
        "def export_source() -> str:\n    return 'ok'\n",
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

        async def run_command(self, command: str, timeout_s: int = 45, **options):
            if "timeout" in options:
                timeout_s = options.pop("timeout")
            self.command = command
            self.timeout_s = timeout_s
            return SimpleNamespace(exit_code=0, stdout="1 passed", stderr="")

        def stop(self) -> None:
            self.stopped = True

    brain = _ObjectiveOnlyBrain()
    sandbox = _SandboxStub()
    monkeypatch.setattr("core.sovereign.local_sandbox.LocalSandbox", lambda: sandbox)

    skill = TestGeneratorSkill(brain=brain)
    result = await skill.safe_execute({"target_file": str(target)}, {})

    assert result["ok"] is True
    assert await asyncio.to_thread(Path(result["test_file"]).exists)
    assert brain.calls
    assert brain.calls[0]["origin"] == "test_generator"
    assert brain.calls[0]["context"]["target"] == str(target)
    assert "export_source.py" in brain.calls[0]["objective"]
    assert sandbox.started is True
    assert sandbox.stopped is True
    assert "pytest -q" in sandbox.command


@pytest.mark.asyncio
async def test_test_generator_recovers_with_deterministic_fallback_after_llm_failure(
    monkeypatch, tmp_path: Path
):
    _disable_governance(monkeypatch)
    target = tmp_path / "sample_module.py"
    target.write_text(
        "def square(value: int) -> int:\n    return value * value\n",
        encoding="utf-8",
    )

    class _Brain:
        async def think(self, *args, **kwargs):
            return SimpleNamespace(content="def test_bad_generated_case():\n    assert False\n")

    class _SandboxStub:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.commands: list[str] = []
            self.files: dict[str, str] = {}
            self._runs = 0

        def start(self) -> None:
            self.started = True

        def write_file(self, name: str, content: str) -> None:
            self.files[name] = content

        async def run_command(self, command: str, timeout_s: int = 45, **options):
            if "timeout" in options:
                timeout_s = options.pop("timeout")
            self.commands.append(command)
            self.timeout_s = timeout_s
            self._runs += 1
            if self._runs == 1:
                return SimpleNamespace(exit_code=1, stdout="", stderr="generated test failed")
            return SimpleNamespace(exit_code=0, stdout="1 passed", stderr="")

        def stop(self) -> None:
            self.stopped = True

    sandbox = _SandboxStub()
    monkeypatch.setattr("core.sovereign.local_sandbox.LocalSandbox", lambda: sandbox)

    skill = TestGeneratorSkill(brain=_Brain())
    result = await skill.safe_execute({"target_file": str(target)}, {})

    assert result["ok"] is True
    assert result["fallback_used"] is True
    assert sandbox.started is True
    assert sandbox.stopped is True
    assert len(sandbox.commands) == 2


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
        {
            "action": "propose",
            "objective": "Draft a safe refactor plan.",
            "read_only": True,
        },
        {},
    )

    assert result["ok"] is True
    assert result["read_only"] is True
    assert result.get("proposal_path") in (None, "")
    assert not list(evolution_dir.glob("evolution_proposal_*.md"))


@pytest.mark.asyncio
async def test_self_evolution_read_only_skips_llm_planning(monkeypatch, tmp_path: Path):
    _disable_governance(monkeypatch)
    evolution_dir = tmp_path / "evolution"
    monkeypatch.setattr(
        SelfEvolutionSkill,
        "_evolution_dir",
        staticmethod(lambda: evolution_dir),
    )

    class _ForbiddenBrain:
        async def think(self, *args, **kwargs):
            self.think_calls = getattr(self, "think_calls", 0) + 1
            raise AssertionError("read-only self-evolution should not use the LLM path")

    target = tmp_path / "sample_module.py"
    target.write_text("def ok():\n    return 1\n", encoding="utf-8")

    skill = SelfEvolutionSkill()
    result = await skill.safe_execute(
        {
            "action": "propose",
            "objective": "Draft a safe refactor plan.",
            "files": [str(target)],
        },
        {"read_only": True, "brain": _ForbiddenBrain()},
    )

    assert result["ok"] is True
    assert result["read_only"] is True
    assert result["fallback"] is True


@pytest.mark.asyncio
async def test_capability_engine_promotes_executive_constraints_into_skill_context(monkeypatch):
    # A real BaseSkill subclass: the engine refuses anything else, and it is
    # right to — an implementation it cannot verify cannot be governed.
    from core.skills.base_skill import BaseSkill

    class _ConstraintProbeSkill(BaseSkill):
        name = "constraint_probe"
        timeout_seconds = 30

        async def execute(self, params, context):
            return await self.safe_execute(params, context)

        async def safe_execute(self, params, context):
            return {
                "ok": True,
                "read_only": bool(context.get("read_only")),
                "timeout_s": context.get("timeout_s"),
            }

    engine = CapabilityEngine()
    probe = _ConstraintProbeSkill()
    engine.skills["constraint_probe"] = SkillMetadata(
        name="constraint_probe",
        description="Probe merged execution constraints.",
        skill_class=_ConstraintProbeSkill,
        instance=probe,
    )
    engine.instances["constraint_probe"] = probe

    async def _begin_tool_execution(*_args, **_kwargs):
        return SimpleNamespace(
            approved=True,
            capability_token_id="token-1",
            constraints={"read_only": True, "timeout_s": 9},
            decision=ConstitutionalDecision(
                proposal_id="proposal-1",
                kind=ProposalKind.TOOL,
                outcome=ProposalOutcome.DEGRADED,
                reason="temporal_safe_autonomous_tool",
                source="autonomous",
            ),
        )

    async def _finish_tool_execution(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            begin_tool_execution=_begin_tool_execution,
            finish_tool_execution=_finish_tool_execution,
        ),
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: SimpleNamespace(verify_tool_access=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr("core.container.ServiceContainer.has", staticmethod(lambda _name: False))

    result = await engine.execute(
        "constraint_probe", {}, context={"objective": "probe constraints"}
    )

    assert result["ok"] is True
    assert result["read_only"] is True
    assert result["timeout_s"] == 9


@pytest.mark.asyncio
async def test_os_automation_outer_authority_closure_failure_rewrites_success(monkeypatch):
    from core.skills.base_skill import BaseSkill

    class _VerifiedOSAutomationSkill(BaseSkill):
        name = "os_automation"
        timeout_seconds = 30

        async def execute(self, params, context):
            return await self.safe_execute(params, context)

        async def safe_execute(self, params, context):
            # The engine no longer stamps a self-asserted "_capability_token_verified"
            # flag — authority travels as a signed capability, and a boolean in a
            # dict was exactly the bypass that made sinks forgeable. The token id
            # still marks that the constitutional path ran.
            assert context["capability_token_id"] == "token-os-1"
            return {
                "ok": True,
                "status": "completed_verified",
                "effect_verified": True,
                "effect_evidence": "frontmost_app=Notes",
                "attempts": [{"transport_success": True}],
            }

    engine = CapabilityEngine()
    skill = _VerifiedOSAutomationSkill()
    engine.skills["os_automation"] = SkillMetadata(
        name="os_automation",
        description="verified OS automation closure probe",
        skill_class=_VerifiedOSAutomationSkill,
        instance=skill,
        metabolic_cost=1,
        effect_scope="foreground_desktop_control",
    )
    engine.instances["os_automation"] = skill

    async def _begin_tool_execution(*_args, **_kwargs):
        return SimpleNamespace(
            approved=True,
            capability_token_id="token-os-1",
            constraints={},
            decision=ConstitutionalDecision(
                proposal_id="proposal-os-1",
                kind=ProposalKind.TOOL,
                outcome=ProposalOutcome.APPROVED,
                reason="unit",
                source="user",
            ),
        )

    async def _finish_tool_execution(*_args, **_kwargs):
        return {
            "closed": False,
            "mode": "unit",
            "intent_closed": True,
            "token_revoked": False,
            "errors": ["token revoke failed"],
        }

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            begin_tool_execution=_begin_tool_execution,
            finish_tool_execution=_finish_tool_execution,
        ),
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: SimpleNamespace(verify_tool_access=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr("core.container.ServiceContainer.has", staticmethod(lambda _name: False))

    result = await engine.execute(
        "os_automation",
        {"goal": "Open Notes", "script_type": "applescript", "execute": True},
        context={
            "objective": "Open Notes",
            "origin": "user",
            "foreground_request": True,
            "user_requested_action": True,
        },
    )

    assert result["ok"] is False
    assert result["status"] == "authority_closure_failed"
    assert result["authority_closure_original_status"] == "completed_verified"
    assert result["manual_reconciliation_required"] is True
    assert result["authority_closure"]["token_revoked"] is False


@pytest.mark.asyncio
async def test_toggle_senses_uses_subprocess_runner_without_local_sandbox(
    monkeypatch, tmp_path: Path
):
    _disable_governance(monkeypatch)

    sense_dir = tmp_path / "senses"
    sense_dir.mkdir(parents=True, exist_ok=True)
    (sense_dir / "vision_service.py").write_text(
        "import time\ntime.sleep(60)\n",
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


@pytest.mark.asyncio
async def test_email_adapter_marks_authority_finalize_degraded(monkeypatch):
    from core.runtime.errors import get_degradation_tracker
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = EmailAdapterSkill()

    class Auth:
        approved = True
        reason = ""
        capability_token_id = "cap-email"
        executive_intent_id = "intent-email"
        will_receipt_id = "receipt-email"

    class Gateway:
        async def authorize_tool_execution(self, *_args, **_kwargs):
            return Auth()

        def verify_tool_access(self, *_args, **_kwargs):
            return True

        def finalize_tool_execution(self, *_args, **_kwargs):
            self.finalized = True
            raise RuntimeError("authority ledger unavailable")

    gateway = Gateway()
    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: gateway,
    )

    async def check_inbox(_params):
        return {"ok": True, "unread": 0, "messages": []}

    monkeypatch.setattr(skill, "_handle_check", check_inbox)

    result = await skill.execute(EmailInput(mode="check"), {})

    assert result["ok"] is True
    assert result["authority_finalized"] is False
    assert result["authority_finalization_status"] == "degraded"
    assert result["authority_receipt_id"] == "receipt-email"
    assert gateway.finalized is True
    assert any(
        "authority finalization degraded" in record.action
        for record in tracker.recent(subsystem="email_adapter")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_email_adapter_failure_finalizes_authority_false(monkeypatch):
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    skill = EmailAdapterSkill()

    class Auth:
        approved = True
        reason = ""
        capability_token_id = "cap-email"
        executive_intent_id = "intent-email"
        will_receipt_id = "receipt-email"

    class Gateway:
        def __init__(self):
            self.finalized_success = []

        async def authorize_tool_execution(self, *_args, **_kwargs):
            return Auth()

        def verify_tool_access(self, *_args, **_kwargs):
            return True

        def finalize_tool_execution(self, *_args, **kwargs):
            self.finalized_success.append(kwargs.get("success"))

    gateway = Gateway()
    check_failures = []
    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: gateway,
    )

    async def check_inbox(_params):
        check_failures.append("called")
        raise RuntimeError("imap unavailable")

    monkeypatch.setattr(skill, "_handle_check", check_inbox)

    result = await skill.execute(EmailInput(mode="check"), {})

    assert result["ok"] is False
    assert "imap unavailable" in result["error"]
    assert result["authority_finalized"] is True
    assert check_failures == ["called"]
    assert gateway.finalized_success == [False]


@pytest.mark.asyncio
async def test_email_adapter_blocks_auto_reply_threads(monkeypatch):
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    skill = EmailAdapterSkill()
    send_attempts = []

    async def read_original(_params):
        return {
            "ok": True,
            "from": "sender@example.com",
            "subject": "Away",
            "message_id": "<auto@example.com>",
            "is_auto_reply": True,
        }

    async def send_reply(params):
        send_attempts.append(params)
        return {"ok": True}

    monkeypatch.setattr(skill, "_handle_read", read_original)
    monkeypatch.setattr(skill, "_handle_send", send_reply)

    result = await skill._handle_reply(EmailInput(mode="reply", uid="42", body="Following up."))

    assert result["ok"] is False
    assert result["status"] == "blocked_auto_reply"
    assert send_attempts == []


@pytest.mark.asyncio
async def test_email_adapter_rejects_invalid_recipient_before_credentials(monkeypatch):
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    skill = EmailAdapterSkill()
    credential_reads = []

    def get_creds():
        credential_reads.append("called")
        return "aura@example.com", "credential-value"

    monkeypatch.setattr(skill, "_get_creds", get_creds)

    result = await skill._handle_send(
        EmailInput(mode="send", to="not-an-address", subject="Hello", body="Body")
    )

    assert result["ok"] is False
    assert "to" in result["error"]
    assert credential_reads == []


@pytest.mark.asyncio
async def test_email_adapter_reads_credentials_off_event_loop(monkeypatch):
    from core.skills import email_adapter
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    skill = EmailAdapterSkill()
    calls = []

    def get_creds():
        calls.append(("creds", "called"))
        raise RuntimeError("credentials missing")

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(("to_thread", getattr(fn, "__name__", "")))
        return fn(*args, **kwargs)

    monkeypatch.setattr(skill, "_get_creds", get_creds)
    monkeypatch.setattr(email_adapter.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="credentials missing"):
        await skill._handle_check(EmailInput(mode="check"))

    assert ("to_thread", "get_creds") in calls
    assert ("creds", "called") in calls
