"""A named address is read, not described.

LIVE, 2026-08-22, typed into the window: "I'm reading
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7092803/ and I want a second
opinion. What was the primary endpoint..." The reply was "I can help you with
a structured approach to evaluating the study, but I won't have direct access
to the paper's content." Nothing was fetched. The grounding taken was "file
you were asked about" — the address had been handed to the filesystem reader.

The fetch exists and is wired into the kernel pipeline; chat is served by the
legacy one, where a named URL only suppressed the search. Naming an address
bought neither a search nor a read.
"""

from __future__ import annotations

import asyncio

import pytest

from interface.routes.chat import _collect_named_url_evidence


def test_no_address_means_nothing_to_read():
    assert asyncio.run(_collect_named_url_evidence("what is 2 + 2?")) is None


def test_the_address_is_read_and_carried(monkeypatch):
    seen = {}

    async def fake_skill(name, params, *, objective, extra_context):
        seen["skill"] = name
        seen["url"] = params.get("url")
        seen["route"] = extra_context.get("route")
        seen["scope"] = extra_context.get("effect_scope")
        return {"ok": True, "result": {"title": "A Study", "text": "The primary endpoint was X."}}

    import interface.routes.chat as chat

    monkeypatch.setattr(
        chat._chat_capability_inventory, "_execute_governed_live_skill", fake_skill
    )
    got = asyncio.run(
        _collect_named_url_evidence(
            "I'm reading https://example.invalid/study and want a second opinion."
        )
    )
    assert got is not None and got["ok"] is True
    assert got["url"] == "https://example.invalid/study"
    assert "primary endpoint was X" in got["text"]
    assert seen["skill"] == "http_request"
    assert seen["route"] == "chat.named_url_evidence"
    # Reading a page is a read.
    assert seen["scope"] == "read_only"


def test_a_page_that_cannot_be_read_says_so(monkeypatch):
    async def refuses(name, params, *, objective, extra_context):
        return {"ok": False, "error": "403 from the host"}

    import interface.routes.chat as chat

    monkeypatch.setattr(
        chat._chat_capability_inventory, "_execute_governed_live_skill", refuses
    )
    got = asyncio.run(_collect_named_url_evidence("read https://example.invalid/x please"))
    assert got is not None
    assert got["ok"] is False
    assert "403" in got["error"]


def test_an_empty_body_is_not_evidence(monkeypatch):
    async def empty(name, params, *, objective, extra_context):
        return {"ok": True, "result": {"text": "   "}}

    import interface.routes.chat as chat

    monkeypatch.setattr(chat._chat_capability_inventory, "_execute_governed_live_skill", empty)
    got = asyncio.run(_collect_named_url_evidence("read https://example.invalid/x"))
    assert got["ok"] is False
    assert "nothing readable" in got["error"]


def test_the_turn_attaches_the_page_it_read():
    """The evidence has to reach the message, or the fetch is decorative."""
    from pathlib import Path

    source = Path("interface/routes/chat.py").read_text(encoding="utf-8")
    assert "named_url_evidence = await _collect_named_url_evidence(" in source
    assert "[PAGE THE USER NAMED]" in source
    assert "could not be read:" in source
