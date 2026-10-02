"""A press that changed nothing on the page did nothing, so it is not a step of the task.

LIVE 2026-10-02: her personality test was answered, submitted and read, and her
verdict on it was hers. Under it the reply said "That much is verified. The rest
did not complete: step_16_effect_unverified (16/18 steps)" — one press on her
result page had changed nothing, and every round counted as a step whose effect
had to be proved. The rounds that changed the page are the task's steps; every
round is still in the narration.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.skills.desktop_task import DesktopTaskSkill
from interface.routes.chat_desktop_objective import _verified_desktop_task_result

pytestmark = pytest.mark.unit

_URL = "https://example.test/results"


def _delegated(monkeypatch, steps: list[dict[str, Any]]) -> dict[str, Any]:
    class _Engine:
        async def execute(self, name, params, context=None):
            return {"ok": True, "rounds": len(steps), "steps": steps, "final_url": _URL}

    monkeypatch.setattr(
        "core.conversation.page_interaction.page_interaction_target", lambda _text: _URL
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _Engine() if name == "capability_engine" else default,
    )
    skill = DesktopTaskSkill()
    return asyncio.run(
        skill._delegate_page_objective(SimpleNamespace(objective=f"take the test at {_URL}"), {})
    )


def test_a_press_that_changed_nothing_does_not_unverify_the_task(monkeypatch):
    result = _delegated(
        monkeypatch,
        [
            {"chose": ["Start"], "ok": True, "landed": 1, "moved": True, "url": _URL},
            {"chose": ["Submit"], "ok": True, "landed": 1, "moved": True, "url": _URL},
            {"chose": ["more"], "ok": True, "landed": 1, "moved": False, "url": _URL},
        ],
    )
    assert result["steps_requested"] == 2 and result["steps_completed"] == 2
    assert len(result["narration"]) == 3, "every round is still in what she did"
    assert _verified_desktop_task_result(result) == (True, "verified")


def test_a_run_in_which_nothing_changed_is_still_not_done(monkeypatch):
    result = _delegated(
        monkeypatch,
        [{"chose": ["more"], "ok": True, "landed": 1, "moved": False, "url": _URL}],
    )
    verified, why = _verified_desktop_task_result(result)
    assert not verified
    assert why == "no_steps_completed:0/1"
