"""Reply graphics retain their data; browser activity stays cosmetic and local."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.perception import browser_presence as presence

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_and_flow_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; the JS contract was not measured")
    result = subprocess.run([node, str(ROOT / "tests/js/conversation_visuals.mjs"), str(ROOT)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_activity_never_exports_page_secrets():
    payload = presence.activity_payload("reading", "started", "https://name:password@example.org/private/account?secret=123#message", False)
    assert payload["url"] == "https://example.org"
    assert not list(presence.activity_keeps_page_secrets_local())
    for url in ("file:///Users/bryan/private.txt", "about:blank", "https://[broken"):
        assert presence.activity_payload("reading", "completed", url, False)["url"] == ""
    with pytest.raises(ValueError):
        presence.activity_payload("run script", "started", "https://example.org", False)


class Page:
    url = "https://example.org/account?token=secret"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def evaluate(self, script, *args):
        self.calls.append((script, args))
        if self.fail:
            raise RuntimeError("page gone")
        return True


@pytest.mark.asyncio
async def test_cosmetic_failure_does_not_replace_the_browser_result(monkeypatch):
    from core.event_bus import get_event_bus

    async def unavailable(*_args):
        raise RuntimeError("transport gone")

    monkeypatch.setattr(get_event_bus(), "publish", unavailable)
    page = Page(fail=True)
    await presence.show_activity(SimpleNamespace(page=page, visible=False, _presence_available=True), "reading")


@pytest.mark.asyncio
async def test_capture_restores_the_mark_even_when_the_picture_fails():
    page = Page()
    presence._VISIBLE_UNTIL[page] = time.monotonic() + 5
    with pytest.raises(RuntimeError, match="capture failed"):
        async with presence.without_presence(page):
            raise RuntimeError("capture failed")
    assert "suspend" in page.calls[0][0] and "resume" in page.calls[1][0]


@pytest.mark.asyncio
async def test_an_expired_mark_costs_no_browser_round_trips():
    page = Page()
    async with presence.without_presence(page):
        await asyncio.sleep(0)
    assert not page.calls


def test_changed_source_has_no_current_measurement(tmp_path):
    from core.organism.model_validation_presentation import measured_presentation
    from core.organism.nothing_measured import NothingMeasured

    with pytest.raises(NothingMeasured):
        measured_presentation(tmp_path)
