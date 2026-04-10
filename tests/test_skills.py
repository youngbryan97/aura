################################################################################

"""
tests/test_skills.py
────────────────────
Verify skill loading and registry.
"""

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine


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
        except Exception as exc:
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

    # Valid query format should be accepted (actual search depends on network)
    result = await skill.execute({"query": "test query", "num_results": 1}, {})
    # Should either succeed or fail gracefully — never crash
    assert "ok" in result
