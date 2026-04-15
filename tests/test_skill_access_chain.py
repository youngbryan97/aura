from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine
from core.kernel.upgrades_10x import GodModeToolPhase
from interface.routes.chat import _check_response_consistency


def test_registered_search_skills_remain_distinct():
    engine = CapabilityEngine()

    assert engine.resolve_skill_name("web_search") == "web_search"
    assert engine.resolve_skill_name("search_web") == "search_web"
    assert engine.resolve_skill_name("free_search") == "free_search"
    assert engine.get("search_web").name == "search_web"
    assert engine.get("free_search").name == "free_search"


def test_explicit_skill_name_invocation_is_detectable():
    engine = CapabilityEngine()

    detected = set(
        engine.detect_intent(
            "Use ManageAbilities, then use query visual context and search_web for this."
        )
    )

    assert "ManageAbilities" in detected
    assert "query_visual_context" in detected
    assert "search_web" in detected


@pytest.mark.parametrize("skill_name", ["web_search", "search_web", "free_search"])
def test_search_skill_variants_normalize_query(skill_name: str):
    params = GodModeToolPhase._normalize_skill_params(
        skill_name,
        'Can you search for "aliens"?',
        {},
    )

    assert params["query"] == "aliens"


def test_grounded_search_normalizes_into_nested_query_payload():
    params = GodModeToolPhase._normalize_skill_params(
        "grounded_search",
        'Can you search for "aliens"?',
        {},
    )

    assert params["params"]["query"] == "aliens"


def test_false_desktop_inability_claim_is_flagged(monkeypatch):
    capability_engine = SimpleNamespace(
        get_catalog=lambda: {
            "computer_use": {"status": "ready"},
            "web_search": {"status": "ready"},
        }
    )

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: capability_engine if name == "capability_engine" else default),
    )

    ok, reason = _check_response_consistency(
        "I can't actually open tabs on your computer.",
        "Open a tab and search for aliens.",
    )

    assert ok is False
    assert reason == "false_inability_claim"
