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


@pytest.mark.asyncio
async def test_web_search_reranks_price_queries_away_from_promotional_results(monkeypatch):
    from core.skills.web_search import EnhancedWebSearchSkill

    skill = EnhancedWebSearchSkill()
    results = [
        {
            "title": "Trusted Bitcoin App - Most Trusted Crypto Exchange",
            "url": "https://www.coinbase.com/",
            "snippet": "The safe and trusted place to buy and sell Bitcoin, Ethereum, and more.",
        },
        {
            "title": "Bitcoin price today, BTC to USD live price",
            "url": "https://coinmarketcap.com/currencies/bitcoin/",
            "snippet": "Bitcoin price is $69,420.12 today with a live market cap and chart.",
        },
    ]
    reranked = skill._rerank_results("current Bitcoin price", list(results))
    assert reranked[0]["url"] == "https://coinmarketcap.com/currencies/bitcoin/"

    monkeypatch.setattr(skill, "_ddg_search", lambda query, num: list(reranked))

    payload = await skill.execute({"query": "current Bitcoin price", "num_results": 5}, {})

    assert payload["ok"] is True
    assert payload["results"][0]["url"] == "https://coinmarketcap.com/currencies/bitcoin/"
    assert "69,420.12" in payload["summary"]
