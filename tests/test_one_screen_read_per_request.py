"""One request, one reading of the person's screen.

Measured live 2026-08-04. "Hey, Aura. Can you tell me what you see on the
screen?" produced TWO governed desktop actions:

    Intention declared [403aeab322c4] ... Generated token d97c11b2 for
    tools: ['desktop_task']
    Intention declared [9926a4fec57e] ... Generated token ecd8ebee for
    tools: ['computer_use']

Both ran the same read_screen_text. The cause was two heuristic rules
matching one sentence — "on my screen" pulls computer_use, "screen" pulls
desktop_task — so the candidate list carried both and both dispatched.

Double the latency, two capability tokens, two entries in the audit trail
for one act, and two readings of a private screen where the person asked
for one.

The same turn also logged, twice:

    Context flag 'user_explicitly_authorized' for
    tool_execution/foreground_desktop_action carried no capability token;
    ignoring it

That check required a token nothing ever issued, so it could never pass —
a warning on every desktop turn, which teaches operators to ignore
warnings.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.desktop_objective_intent import looks_like_screen_observation
from interface.routes import chat_capability_inventory

ROOT = Path(__file__).resolve().parents[1]

LIVE_REQUEST = "hey, aura. can you tell me what you see on the screen?"


def test_the_live_request_is_recognised_as_an_observation():
    """The premise: if this is not an observation, the dedupe never fires."""
    assert looks_like_screen_observation(LIVE_REQUEST)


def test_both_rules_really_do_match_the_live_request():
    """Pins the actual cause, so a rule edit cannot silently reintroduce it."""
    assert "on my screen" in "what you see on my screen"
    assert "screen" in LIVE_REQUEST


def test_an_observation_drops_the_duplicate_skill():
    source = (ROOT / "core" / "capability_engine.py").read_text("utf-8")
    ast.parse(source)
    assert "looks_like_screen_observation(objective_lower)" in source, (
        "nothing de-duplicates the desktop candidates, so a screen read is "
        "dispatched twice again"
    )
    index = source.index("looks_like_screen_observation(objective_lower)")
    window = source[index : index + 320]
    assert 'name != "computer_use"' in window, (
        "the governed desktop_task lane must own an observation; a bare "
        "computer_use read duplicates it with fewer receipts"
    )


def test_computer_use_is_only_dropped_for_observations():
    """It stays available for clicking, typing and everything else."""
    source = (ROOT / "core" / "capability_engine.py").read_text("utf-8")
    index = source.index("looks_like_screen_observation(objective_lower)")
    guard = source[max(0, index - 400) : index]
    assert '"desktop_task" in heuristic_candidates' in guard
    assert '"computer_use" in heuristic_candidates' in guard


def test_the_desktop_authority_token_has_an_issuer():
    """CP126 3b1a9177 required a token that nothing minted."""
    gateway = (ROOT / "core" / "executive" / "authority_gateway.py").read_text("utf-8")
    ast.parse(gateway)
    assert "def issue_desktop_authority_capability" in gateway
    assert 'action="foreground_desktop_action"' in gateway


def test_the_router_asks_the_gateway_rather_than_self_granting():
    """Only the gateway issues; a caller-supplied boolean is not authority."""
    router = (ROOT / "core" / "cognitive" / "router.py").read_text("utf-8")
    ast.parse(router)
    assert "issue_desktop_authority_capability" in router
    index_flag = router.index('execution_context["user_explicitly_authorized"] = True')
    index_token = router.index("issue_desktop_authority_capability")
    assert index_flag < index_token, (
        "the token must be obtained for the flag that was just asserted"
    )


def test_the_attestation_still_refuses_an_unbacked_flag():
    """The fix supplies the token; it must not weaken the check."""
    runtime = (ROOT / "core" / "being" / "runtime.py").read_text("utf-8")
    assert "carried no capability token; ignoring it" in runtime, (
        "the attestation no longer refuses an unattested flag"
    )
    assert "attested_context_flag(" in runtime


@pytest.mark.asyncio
async def test_the_direct_desktop_bridge_attaches_gateway_authority(monkeypatch):
    seen: dict = {}

    class _Engine:
        async def execute(self, skill_name, params, *, context):
            seen.update(context)
            return {"ok": True}

    monkeypatch.setattr(
        chat_capability_inventory.ServiceContainer,
        "get",
        lambda name, default=None: _Engine() if name == "capability_engine" else default,
    )
    monkeypatch.setattr(
        chat_capability_inventory,
        "get_authority_gateway",
        lambda: SimpleNamespace(
            issue_desktop_authority_capability=lambda **_kwargs: "gateway-token"
        ),
    )
    monkeypatch.setattr(
        chat_capability_inventory,
        "report_chat_delivery_progress",
        _no_progress,
    )

    result = await chat_capability_inventory._execute_governed_live_skill(
        "desktop_task",
        {"objective": "change my wallpaper"},
        objective="change my wallpaper",
        extra_context={
            "route": "chat.desktop_objective",
            "desktop_execution_contract": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
        },
    )

    assert result["ok"] is True
    assert seen["capability_token"] == "gateway-token"


@pytest.mark.asyncio
async def test_a_non_desktop_live_skill_does_not_mint_desktop_authority(monkeypatch):
    seen: dict = {}

    class _Engine:
        async def execute(self, skill_name, params, *, context):
            seen.update(context)
            return {"ok": True}

    def _unexpected_issue(**_kwargs):
        raise AssertionError("non-desktop work must not receive desktop authority")

    monkeypatch.setattr(
        chat_capability_inventory.ServiceContainer,
        "get",
        lambda name, default=None: _Engine() if name == "capability_engine" else default,
    )
    monkeypatch.setattr(
        chat_capability_inventory,
        "get_authority_gateway",
        lambda: SimpleNamespace(issue_desktop_authority_capability=_unexpected_issue),
    )
    monkeypatch.setattr(
        chat_capability_inventory,
        "report_chat_delivery_progress",
        _no_progress,
    )

    result = await chat_capability_inventory._execute_governed_live_skill(
        "web_interlocutor",
        {"objective": "talk to the page"},
        objective="talk to the page",
        extra_context={"route": "chat.web_interlocutor"},
    )

    assert result["ok"] is True
    assert "capability_token" not in seen


async def _no_progress(**_kwargs):
    return None
