"""A search she runs keeps what it read unless keeping it was ruled out.

LIVE 2026-10-03: "What's the latest Miami Heat news this week?" was
searched with retain=False because the person had not asked to save
anything, so nothing she looked up was learned.
"""

from __future__ import annotations

import asyncio

from interface.routes import chat_desktop_evidence


def _params_for(message: str, monkeypatch) -> dict:
    seen: dict = {}

    async def fake_skill(name, params, *, objective, extra_context):
        seen.update(params)
        return {"ok": True, "results": [], "result": {"ok": True, "results": []}}

    monkeypatch.setattr(chat_desktop_evidence._chat_capability_inventory, "_execute_governed_live_skill", fake_skill)
    asyncio.run(chat_desktop_evidence._collect_desktop_required_search_evidence(message, session_id="kept"))
    return seen


def test_an_unrequested_search_leaves_keeping_to_the_research_pipeline(monkeypatch) -> None:
    params = _params_for("Use web_search to check the latest Miami Heat news this week.", monkeypatch)
    assert params and params["retain"] is None


def test_a_search_asked_to_be_saved_is_kept(monkeypatch) -> None:
    params = _params_for(
        "Use web_search to check the latest Miami Heat news and save it as provisional research memory.",
        monkeypatch,
    )
    assert params and params["retain"] is True
