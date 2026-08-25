################################################################################

"""
tests/test_skills.py
────────────────────
Verify skill loading and registry.
"""

import asyncio
import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata, SkillRequirements


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


def test_capability_engine_registered_skill_classes_are_importable():
    engine = CapabilityEngine()
    unresolved = []

    for name, meta in sorted(engine.skills.items()):
        module = importlib.import_module(meta.module_path)
        skill_class = getattr(module, meta.class_name, None)
        if skill_class is None:
            unresolved.append(name)

    assert not unresolved


def test_capability_engine_registered_skill_constructors_are_instantiable():
    engine = CapabilityEngine()
    failures = []

    for name, meta in sorted(engine.skills.items()):
        module = importlib.import_module(meta.module_path)
        skill_class = getattr(module, meta.class_name)
        try:
            skill_class()
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            failures.append((name, type(exc).__name__, str(exc)))

    assert not failures


def test_ui_control_skills_do_not_eagerly_import_pyautogui(monkeypatch):
    blocked = []
    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyautogui":
            blocked.append(name)
            raise AssertionError("pyautogui imported eagerly")
        return real_import(name, globals, locals, fromlist, level)

    for module_name in (
        "core.skills.computer_use",
        "core.skills.os_manipulation",
        "core.skills.vision_actor",
    ):
        sys.modules.pop(module_name, None)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    computer_use = importlib.import_module("core.skills.computer_use")
    os_manipulation = importlib.import_module("core.skills.os_manipulation")
    vision_actor = importlib.import_module("core.skills.vision_actor")

    computer_use.ComputerUseSkill()
    os_manipulation.DesktopControlSkill()
    vision_actor.VisionActorSkill()

    assert blocked == []


@pytest.mark.asyncio
async def test_ui_control_skills_fail_cleanly_when_pyautogui_is_unavailable(monkeypatch):
    import core.skills.computer_use as computer_use
    import core.skills.os_manipulation as os_manipulation
    import core.skills.vision_actor as vision_actor

    unavailable = RuntimeError("display access unavailable")
    monkeypatch.setattr(computer_use, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(os_manipulation, "get_pyautogui", lambda: (None, unavailable))
    monkeypatch.setattr(vision_actor, "get_pyautogui", lambda: (None, unavailable))

    computer_result = await computer_use.ComputerUseSkill().execute(
        {"action": "click", "x": 1, "y": 1},
        {},
    )
    assert computer_result["ok"] is False
    assert "PyAutoGUI unavailable" in computer_result["error"]

    os_result = await os_manipulation.DesktopControlSkill().execute(
        {"action": "click", "x": 1, "y": 1},
        {},
    )
    assert os_result["ok"] is False
    assert "PyAutoGUI unavailable" in os_result["error"]

    vision_result = await vision_actor.VisionActorSkill().execute(
        vision_actor.VisionActorInput(action="type", text_to_type="hello"),
        {},
    )
    assert vision_result["ok"] is False
    assert "Physical execution unavailable" in vision_result["summary"]


@pytest.mark.asyncio
async def test_os_manipulation_prefers_host_automation_without_pyautogui(monkeypatch):
    import core.skills.os_manipulation as os_manipulation
    from core.container import ServiceContainer

    class HostAutomation:
        async def type_text(self, text):
            return SimpleNamespace(
                success=True,
                result=f"typed {text}",
                error="",
                receipt_id="host-receipt-1",
            )

    async def _allowed_accessibility(self, capability):
        return None

    monkeypatch.setattr(os_manipulation, "get_pyautogui", lambda: (None, RuntimeError("no display")))
    monkeypatch.setattr(
        os_manipulation.DesktopControlSkill,
        "_require_accessibility",
        _allowed_accessibility,
    )
    ServiceContainer.register_instance("host_automation", HostAutomation(), required=False)
    try:
        result = await os_manipulation.DesktopControlSkill().execute(
            {"action": "type", "text": "hello"},
            {},
        )
    finally:
        ServiceContainer.clear()

    assert result == {
        "ok": True,
        "result": "typed hello",
        "receipt_id": "host-receipt-1",
    }


def test_legacy_skill_shims_resolve_to_core_implementations():
    from skills.computer_use import ComputerUseSkill as legacy_computer_use
    from skills.os_manipulation import DesktopControlSkill as legacy_os_manipulation
    from core.skills.computer_use import ComputerUseSkill as core_computer_use
    from core.skills.os_manipulation import DesktopControlSkill as core_os_manipulation

    assert legacy_computer_use is core_computer_use
    assert legacy_os_manipulation is core_os_manipulation


@pytest.mark.asyncio
async def test_ui_control_skills_fail_cleanly_when_accessibility_is_denied(monkeypatch):
    import core.skills.computer_use as computer_use
    import core.skills.os_manipulation as os_manipulation

    dummy_pyautogui = SimpleNamespace(
        click=lambda *args, **kwargs: None,
        write=lambda *args, **kwargs: None,
        typewrite=lambda *args, **kwargs: None,
        hotkey=lambda *args, **kwargs: None,
        scroll=lambda *args, **kwargs: None,
        press=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(computer_use, "get_pyautogui", lambda: (dummy_pyautogui, None))
    monkeypatch.setattr(os_manipulation, "get_pyautogui", lambda: (dummy_pyautogui, None))

    async def _blocked_permissions(self, capability, *permission_names):
        return {
            "ok": False,
            "status": "denied",
            "error": f"Accessibility permission is required for {capability}.",
            "permission": "accessibility",
            "guidance": "Enable Accessibility in System Settings.",
        }

    async def _blocked_accessibility(self, capability):
        return {
            "ok": False,
            "status": "denied",
            "error": f"Accessibility permission is required for {capability}.",
            "permission": "accessibility",
            "guidance": "Enable Accessibility in System Settings.",
        }

    monkeypatch.setattr(computer_use.ComputerUseSkill, "_require_permissions", _blocked_permissions)
    monkeypatch.setattr(os_manipulation.DesktopControlSkill, "_require_accessibility", _blocked_accessibility)

    computer_result = await computer_use.ComputerUseSkill().execute(
        {"action": "click", "x": 1, "y": 1},
        {},
    )
    assert computer_result["ok"] is False
    assert computer_result["permission"] == "accessibility"
    assert "Accessibility permission" in computer_result["error"]

    os_result = await os_manipulation.DesktopControlSkill().execute(
        {"action": "click", "x": 1, "y": 1},
        {},
    )
    assert os_result["ok"] is False
    assert os_result["permission"] == "accessibility"
    assert "Accessibility permission" in os_result["error"]


@pytest.mark.asyncio
async def test_computer_use_open_url_uses_default_browser_with_verified_permissions(monkeypatch):
    import core.skills.computer_use as computer_use

    calls = []

    async def allow_permissions(self, capability, *permission_names):
        return None

    class FakeSubprocessGateway:
        def run(self, args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(computer_use.shutil, "which", lambda name: "/usr/bin/open" if name == "open" else None)
    monkeypatch.setattr(
        computer_use,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    monkeypatch.setattr(computer_use.ComputerUseSkill, "_require_permissions", allow_permissions)
    monkeypatch.setattr(
        computer_use.ComputerUseSkill,
        "_frontmost_app_name",
        lambda self: "Safari",
    )
    monkeypatch.setattr(
        computer_use.ComputerUseSkill,
        "_wait_for_frontmost_app",
        lambda self, expected: asyncio.sleep(0, result=(True, "Safari")),
    )
    monkeypatch.setattr(
        computer_use.ComputerUseSkill,
        "_active_browser_location",
        lambda self, browser: ("https://duckduckgo.com/?q=aliens", "aliens at DuckDuckGo"),
    )

    result = await computer_use.ComputerUseSkill().execute(
        {"action": "open_url", "target": "aliens"},
        {},
    )

    assert result["ok"] is True
    assert result["action"] == "open_url"
    assert result["url"] == "https://duckduckgo.com/?q=aliens"
    assert calls[0][0] == ["open", "https://duckduckgo.com/?q=aliens"]
    assert calls[0][1]["accelerator_capability"] == "none"
    assert result["effect_verified"] is True


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_reports_accessibility_marker_as_failure(monkeypatch):
    import core.skills.computer_use as computer_use

    async def _allow_permissions(self, capability, *permission_names):
        return None

    monkeypatch.setattr(computer_use.ComputerUseSkill, "_require_permissions", _allow_permissions)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: type(
            "UnavailablePerception",
            (),
            {
                "capture": lambda self, *, save_screenshot=False: (_ for _ in ()).throw(
                    RuntimeError("structured screen perception unavailable")
                )
            },
        )(),
    )
    monkeypatch.setattr(
        computer_use.ComputerUseSkill,
        "_read_screen_text_macos",
        lambda self: "[Accessibility error or UI unresponsive]",
    )
    # Screen capture is admitted before any text is read, and on a host where
    # foreground privacy cannot be verified it refuses first — so this test
    # was asserting an accessibility branch it never reached. Admit the
    # capture explicitly; refusing it is a different test's subject.
    from core.security import screen_capture_policy

    class _Admitted:
        allowed = True
        public_error = ""

        def to_receipt(self):
            return {"schema": "aura.security.screen_capture_admission.v1", "allowed": True}

    async def _admit_async():
        return _Admitted()

    monkeypatch.setattr(
        screen_capture_policy, "evaluate_screen_capture_admission_async", _admit_async
    )

    result = await computer_use.ComputerUseSkill().execute(
        {"action": "read_screen_text", "target": ""},
        {},
    )

    assert result["ok"] is False
    assert result["status"] == "unavailable"
    assert "accessibility error" in result["error"].lower()


@pytest.mark.asyncio
async def test_web_search_skill_initializes_and_accepts_input():
    """Verify the web search skill initializes correctly and handles empty queries."""
    from core.skills.web_search import EnhancedWebSearchSkill

    skill = EnhancedWebSearchSkill()
    assert skill.name == "web_search"
    assert skill.pipeline is not None

    # Empty query should return an error, not crash
    result = await skill.execute({"query": "", "num_results": 5}, {})
    assert result["ok"] is False
    assert "error" in result

    async def _fake_search(query, **kwargs):
        return {"ok": True, "query": query, "answer": "stubbed", "summary": "stubbed"}

    skill.pipeline.search = _fake_search  # type: ignore[method-assign]

    # Valid query format should be accepted without touching the live network
    result = await skill.execute({"query": "test query", "num_results": 1}, {})
    assert result["ok"] is True
    assert result["query"] == "test query"


@pytest.mark.asyncio
async def test_capability_engine_uses_skill_timeout_budget_for_cognitive_governor(monkeypatch):
    captured = {}

    # A real BaseSkill subclass. The engine refuses an implementation that is
    # not one — correctly: an unverified callable on the tool path cannot be
    # governed, timed out or receipted. A bare `class _Skill` could only pass
    # if the engine abandoned that check, so the fixture is what changes.
    from tests.support.skill_doubles import make_test_skill

    _Skill = make_test_skill("slow_skill", timeout_seconds=57.0)

    class _Governor:
        async def execute_safely(self, task_name, coroutine, *args, timeout_seconds=30.0, **kwargs):
            captured["task_name"] = task_name
            captured["timeout_seconds"] = timeout_seconds
            return await coroutine(*args, **kwargs)

    # Same stale-assumption as the token-issuance tests (see
    # approval_overlay_off in tests/test_forensic_audit_regressions.py): these
    # used to pass by inheriting the developer's live
    # governance.approval_mode. Once test runs stopped reading live ~/.aura
    # state it resolves to its real default of "destructive", and a probe
    # skill's effect scope is "unknown", which is in the destructive set — so
    # the gateway correctly requires confirmation. This test is about the
    # cognitive-governor timeout budget, not the confirmation overlay, so it declares the mode it needs
    # instead of measuring the machine's configuration.
    monkeypatch.setattr(
        "core.runtime.runtime_settings.runtime_approval_mode", lambda: "none"
    )
    engine = CapabilityEngine()
    engine.skills = {
        "slow_skill": SkillMetadata(
            name="slow_skill",
            description="timeout budget probe",
            skill_class=_Skill,
            requirements=SkillRequirements(),
            timeout_seconds=12,
        )
    }
    engine.instances = {}
    engine._cognitive_governor = _Governor()
    execute_with_retry = AsyncCallRecorder(result={"ok": True})
    engine._execute_with_retry = execute_with_retry

    monkeypatch.setattr(
        "core.capability_engine.ServiceContainer.has",
        staticmethod(lambda *_args, **_kwargs: False),
    )
    monkeypatch.setattr("core.capability_engine.resolve_metabolic_monitor", lambda default=None: None)
    monkeypatch.setattr("core.capability_engine.resolve_edi", lambda default=None: None)

    result = await engine.execute("slow_skill", {}, {"origin": "desktop"})

    assert result["ok"] is True
    assert captured == {"task_name": "slow_skill", "timeout_seconds": 57.0}
    assert len(execute_with_retry.calls) == 1


@pytest.mark.asyncio
async def test_capability_engine_blocks_when_permission_model_check_fails(monkeypatch):
    class _Skill:
        timeout_seconds = 5.0

    class _Governor:
        async def execute_safely(self, task_name, coroutine, *args, timeout_seconds=30.0, **kwargs):
            return await coroutine(*args, **kwargs)

    class _FailingPermissionModel:
        def __init__(self) -> None:
            self.available = False

        def check_permission(self, *_args, **_kwargs):
            if not self.available:
                raise RuntimeError("permission service unavailable")
            return SimpleNamespace(approved=True)

    engine = CapabilityEngine()
    engine.skills = {
        "typed_action": SkillMetadata(
            name="typed_action",
            description="permission failure probe",
            skill_class=_Skill,
            requirements=SkillRequirements(),
            timeout_seconds=5,
        )
    }
    engine.instances = {}
    engine._cognitive_governor = _Governor()
    execute_with_retry = AsyncCallRecorder(result={"ok": True})
    engine._execute_with_retry = execute_with_retry

    def _get_service(name, default=None):
        if name == "permission_model":
            return _FailingPermissionModel()
        return default

    monkeypatch.setattr(
        "core.capability_engine.ServiceContainer.get",
        staticmethod(_get_service),
    )
    monkeypatch.setattr(
        "core.capability_engine.ServiceContainer.has",
        staticmethod(lambda *_args, **_kwargs: False),
    )
    monkeypatch.setattr("core.capability_engine.resolve_metabolic_monitor", lambda default=None: None)
    monkeypatch.setattr("core.capability_engine.resolve_edi", lambda default=None: None)

    result = await engine.execute("typed_action", {}, {"origin": "desktop"})

    assert result["ok"] is False
    assert result["status"] == "blocked_by_permission_model_failure"
    assert execute_with_retry.calls == []


@pytest.mark.asyncio
async def test_web_search_skill_uses_cognitive_engine_for_deep_research(monkeypatch):
    from core.skills.web_search import EnhancedWebSearchSkill

    engine_calls = []

    class _Engine:
        async def generate(self, prompt, **kwargs):
            engine_calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            return "deep research draft"

    async def _fake_run_deep_research(question, brain, search_fn, max_loops=3, on_phase=None, **_kwargs):
        generated = await brain.generate("expand query")
        assert generated == {"response": "deep research draft"}
        return {"answer": f"researched: {question}", "sources": []}

    monkeypatch.setattr(
        "core.skills.web_search.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Engine() if name == "cognitive_engine" else default),
    )
    monkeypatch.setattr("core.skills.web_search.run_deep_research", _fake_run_deep_research)

    skill = EnhancedWebSearchSkill()
    result = await skill.execute({"query": "history of basalt", "deep": True}, {})

    assert result["answer"] == "researched: history of basalt"
    assert result["summary"] == "researched: history of basalt"
    assert engine_calls
    assert engine_calls[0]["kwargs"]["purpose"] == "research"


@pytest.mark.asyncio
async def test_web_search_skill_deep_research_falls_back_when_synthesis_is_empty(monkeypatch):
    from core.skills.web_search import EnhancedWebSearchSkill

    class _Engine:
        async def generate(self, prompt, **kwargs):
            return "deep research draft"

    async def _fake_run_deep_research(question, brain, search_fn, max_loops=3, on_phase=None, **_kwargs):
        generated = await brain.generate("expand query")
        assert generated == {"response": "deep research draft"}
        return {"answer": "", "sources": [{"title": "Source", "url": "https://example.com"}]}

    async def _fake_search(query, **kwargs):
        return {
            "ok": True,
            "query": query,
            "answer": "fallback synthesized answer",
            "summary": "fallback synthesized answer",
            "citations": [{"title": "Source", "url": "https://example.com"}],
            "retained": True,
            "artifact_id": "artifact-1",
        }

    monkeypatch.setattr(
        "core.skills.web_search.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Engine() if name == "cognitive_engine" else default),
    )
    monkeypatch.setattr("core.skills.web_search.run_deep_research", _fake_run_deep_research)

    skill = EnhancedWebSearchSkill()
    skill.pipeline.search = _fake_search  # type: ignore[method-assign]
    result = await skill.execute({"query": "history of basalt", "deep": True, "retain": True}, {})

    assert result["answer"] == "fallback synthesized answer"
    assert result["retained"] is True
    assert result["artifact_id"] == "artifact-1"


@pytest.mark.asyncio
async def test_web_search_skill_retries_with_cached_artifact_when_force_refresh_fails(monkeypatch):
    from core.skills.web_search import EnhancedWebSearchSkill

    class _Engine:
        async def generate(self, prompt, **kwargs):
            return "deep research draft"

    async def _fake_run_deep_research(question, brain, search_fn, max_loops=3, on_phase=None, **_kwargs):
        return {"answer": "", "sources": [{"title": "Source", "url": "https://example.com"}]}

    calls = []

    async def _fake_search(query, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("force_refresh"):
            return {"ok": False, "error": "No results found for query."}
        return {
            "ok": True,
            "query": query,
            "answer": "cached retained answer",
            "summary": "cached retained answer",
            "cached": True,
            "retained": True,
            "artifact_id": "artifact-cached",
        }

    monkeypatch.setattr(
        "core.skills.web_search.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Engine() if name == "cognitive_engine" else default),
    )
    monkeypatch.setattr("core.skills.web_search.run_deep_research", _fake_run_deep_research)

    skill = EnhancedWebSearchSkill()
    skill.pipeline.search = _fake_search  # type: ignore[method-assign]

    result = await skill.execute(
        {"query": "python 3.12 release notes", "deep": True, "retain": True, "force_refresh": True},
        {},
    )

    assert result["ok"] is True
    assert result["cached"] is True
    assert result["retained"] is True
    assert result["artifact_id"] == "artifact-cached"
    assert calls[0]["force_refresh"] is True
    assert calls[1]["force_refresh"] is False


@pytest.mark.asyncio
async def test_web_search_skill_deep_research_success_retains_artifact(monkeypatch):
    from core.skills.web_search import EnhancedWebSearchSkill

    class _Engine:
        async def generate(self, prompt, **kwargs):
            return "deep research draft"

    async def _fake_run_deep_research(question, brain, search_fn, max_loops=3, on_phase=None, **_kwargs):
        return {
            "answer": "Deep retained answer",
            "sources": [{"title": "Source", "url": "https://example.com/source"}],
        }

    retained = []

    class _Artifact:
        artifact_id = "artifact-deep"

    class _Pipeline:
        def _format_message(self, query, result):
            return f"{query}: {result['answer']}"

        def _should_retain(self, query, *, deep, retain, context, result):
            assert query == "history of basalt"
            assert deep is True
            assert retain is True
            return True

        def _result_to_artifact(self, result, *, freshness_seconds):
            retained.append(("artifact", freshness_seconds, result))
            return _Artifact()

        async def _retain_artifact(self, artifact, context):
            retained.append(("retain", artifact.artifact_id, context))

    monkeypatch.setattr(
        "core.skills.web_search.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Engine() if name == "cognitive_engine" else default),
    )
    monkeypatch.setattr("core.skills.web_search.run_deep_research", _fake_run_deep_research)

    skill = EnhancedWebSearchSkill()
    skill.pipeline = _Pipeline()  # type: ignore[assignment]

    result = await skill.execute({"query": "history of basalt", "deep": True, "retain": True}, {})

    assert result["answer"] == "Deep retained answer"
    assert result["retained"] is True
    assert result["artifact_id"] == "artifact-deep"
    assert result["citations"] == [{"title": "Source", "url": "https://example.com/source"}]
    assert retained[0][0] == "artifact"
    assert retained[1] == ("retain", "artifact-deep", {})
