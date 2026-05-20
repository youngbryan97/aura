import json
import urllib.error
from pathlib import Path

import pytest

from core.bus.sensory_gate import SensoryGateActor
from tools.audit_degradation import analyze_file


def _actor_without_bus() -> SensoryGateActor:
    actor = SensoryGateActor.__new__(SensoryGateActor)
    actor.browser = None
    actor._is_active = True
    actor._shutdown_event = None
    return actor


def test_sensory_gate_degradation_audit_is_clean():
    assert analyze_file(Path("core/bus/sensory_gate.py")) == []


def test_search_result_formatting_tolerates_mismatched_wikipedia_arrays():
    data = ["aura", ["Aura", "Aura 2"], ["first snippet"], ["https://example.com/aura"]]

    assert SensoryGateActor._format_search_results(data) == [
        "Aura: first snippet (https://example.com/aura)"
    ]


@pytest.mark.asyncio
async def test_browse_without_browser_fails_closed():
    actor = _actor_without_bus()

    result = await actor._handle_browse({"url": "https://example.com"}, "trace-1")

    assert result == {"error": "browser_unavailable"}


@pytest.mark.asyncio
async def test_search_handles_mismatched_response_without_crashing(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                ["aura", ["Aura", "Aura 2"], ["first snippet"], ["https://example.com/aura"]]
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    actor = _actor_without_bus()

    result = await actor._handle_search({"query": "aura"}, "trace-2")

    assert result["results"] == ["Aura: first snippet (https://example.com/aura)"]
    assert result["observation_only"] is True


@pytest.mark.asyncio
async def test_search_network_error_returns_structured_error(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    actor = _actor_without_bus()

    result = await actor._handle_search({"query": "aura"}, "trace-3")

    assert "offline" in result["error"]


@pytest.mark.asyncio
async def test_shutdown_handler_flips_actor_state_and_event():
    import asyncio

    actor = _actor_without_bus()
    actor._shutdown_event = asyncio.Event()

    result = await actor._handle_shutdown({}, "trace-4")

    assert result == "Acknowledged"
    assert actor._is_active is False
    assert actor._shutdown_event.is_set()
