"""A skill that asks for authority of its own asks for it as the turn it belongs to.

Her chat's tool loop names itself the source of what it calls ("think_and_act"), and forwards the turn's origin beside it.
A lease asked for under the source is one nobody granted. LIVE 10 November 2026 that refused her mail, and LIVE 24
November 2026 her browser: "denied_by_default: network_call requires validated scoped authority
(signed_standing_authority_lease_missing)", three times, on a search a person had asked for.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.executive.standing_authority import authority_origin_of

ROOT = Path(__file__).resolve().parents[1]
#: A turn of a person's, as her tool loop hands it to a skill.
FROM_THE_LOOP = {"origin": "response_generation_user", "source": "think_and_act"}


def _asks_in(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "authorize_tool_execution"]


def test_the_turn_is_whose_authority_is_asked_for():
    assert authority_origin_of(FROM_THE_LOOP) == "response_generation_user"
    assert authority_origin_of({"authority_origin": "desktop_ui", **FROM_THE_LOOP}) == "desktop_ui"


@pytest.mark.parametrize("path", sorted(p for p in (ROOT / "core" / "skills").glob("*.py") if "authorize_tool_execution(" in p.read_text(encoding="utf-8")),
                         ids=lambda p: p.stem)
def test_no_skill_asks_under_the_name_of_the_loop_that_called_it(path):
    text = path.read_text(encoding="utf-8")
    assert "authority_origin_of(" in text, f"{path.name} asks for authority and does not ask as the turn it is of"
    for call in _asks_in(path):
        given = next((ast.unparse(k.value) for k in call.keywords if k.arg == "source"), "")
        assert given and given != "source", f"{path.name}: {given}"


def test_her_browser_asks_as_the_turn(monkeypatch):
    from core.executive import authority_gateway
    from core.skills import sovereign_browser

    asked: list[str] = []
    ran: list[dict] = []

    class Gateway:
        async def authorize_tool_execution(self, tool, arguments, *, source, **_how):
            asked.append(source)
            return SimpleNamespace(approved=True, source=source, constraints={"standing_authority_grant_id": "g"}, standing_authority_token="t",
                                   standing_authority_receipt_id="r", capability_token_id="c", executive_intent_id="i")

    async def execute(**how):
        ran.append(how["authority_context"])
        return {"ok": True}

    monkeypatch.setattr(authority_gateway, "get_authority_gateway", lambda: Gateway())
    monkeypatch.setattr(sovereign_browser.ActionExecutor, "execute", staticmethod(execute))
    skill = sovereign_browser.SovereignBrowserSkill()
    result = asyncio.run(skill.execute({"mode": "search", "query": "tide tables"}, dict(FROM_THE_LOOP)))
    assert result["ok"] and asked == ["response_generation_user"]
    assert ran[0]["authority_origin"] == "response_generation_user" and ran[0]["standing_authority_token"] == "t"


def test_her_messages_ask_as_the_turn(monkeypatch):
    from core.skills import messages

    asked: list[str] = []

    class Transport:
        def status(self):
            return {}

        async def set_paused_authorized(self, *, paused, source, context):
            asked.append(source)
            return {"ok": True}

    monkeypatch.setattr(messages.ServiceContainer, "get", staticmethod(lambda *a, **k: Transport()))
    skill = messages.MessagesSkill()
    asyncio.run(skill.execute({"action": "pause"}, dict(FROM_THE_LOOP)))
    assert asked == ["response_generation_user"]
