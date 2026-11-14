"""While a person has her voice mode open, no background model is loaded into the memory her conversation needs.

LIVE 13 November 2026, one voice session: her memory watchdog reached its hard ceiling twice (managed 53,177 MB, then
53,574 MB, with 10 GB of swap). The only model worker it could stop was her cortex, so it stopped it both times: Bryan's
first question waited 41 s while it reloaded, and the second kill came in the middle of a reply. Her brainstem had
begun loading during the session. A resident background worker still serves between turns; only a load waits.
"""
from __future__ import annotations

import asyncio
import contextlib

import pytest

from core.brain import llm_health_router_gate_owners as owners


@pytest.fixture(autouse=True)
def _no_sessions_left_open():
    yield
    owners._VOICE_SESSIONS.clear()


def _asked_for_its_worker(monkeypatch, path: str, *, background: bool) -> tuple[bool, list[str]]:
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient(path)
    reached: list[str] = []

    @contextlib.asynccontextmanager
    async def _admitted(_client, *, foreground_request):
        reached.append("admission")
        yield None

    @contextlib.asynccontextmanager
    async def _gate(**_kw):
        yield None

    async def _inner(**_kw):
        reached.append("spawn")
        return True

    async def _no_cooldown(**_kw):
        return None

    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: False)
    monkeypatch.setattr(mlx_client, "_background_deferral_active", lambda *_a, **_k: None)
    monkeypatch.setattr(mlx_client, "_model_load_admission_context", _admitted)
    monkeypatch.setattr(mlx_client, "_spawn_gate_context", _gate)
    monkeypatch.setattr(client, "_ensure_worker_alive_inner", _inner)
    monkeypatch.setattr(client, "_await_swap_cooldown", _no_cooldown)
    result = asyncio.run(client._ensure_worker_alive(request_is_background=background,
                                                     foreground_request=not background))
    return result, reached


def test_a_background_model_is_not_loaded_while_her_voice_mode_is_open(monkeypatch):
    owners.a_voice_session("s1", True)
    loaded, reached = _asked_for_its_worker(monkeypatch, "/models/Ternary-Bonsai-2-27B-mlx-2bit", background=True)
    assert loaded is False and reached == []


def test_it_loads_once_the_voice_mode_is_closed(monkeypatch):
    owners.a_voice_session("s1", True)
    owners.a_voice_session("s1", False)
    loaded, reached = _asked_for_its_worker(monkeypatch, "/models/Ternary-Bonsai-2-27B-mlx-2bit", background=True)
    assert loaded is True and reached == ["admission", "spawn"]


def test_a_persons_turn_still_loads_what_it_needs(monkeypatch):
    owners.a_voice_session("s1", True)
    loaded, reached = _asked_for_its_worker(monkeypatch, "/models/Ternary-Bonsai-2-27B-mlx-2bit", background=False)
    assert loaded is True and reached == ["admission", "spawn"]


def test_two_sessions_hold_until_both_close():
    owners.a_voice_session("s1", True)
    owners.a_voice_session("s2", True)
    owners.a_voice_session("s1", False)
    assert owners.a_voice_session_is_open()
    owners.a_voice_session("s2", False)
    assert not owners.a_voice_session_is_open()


def test_only_her_voice_mode_holds_models_not_ambient_listening():
    """Auto-listen keeps an ambient session open for as long as her app runs (d194e9fc9); holding background models
    for it would hold them for ever."""
    from pathlib import Path

    route = Path(__file__).resolve().parents[1] / "interface" / "routes" / "voice_duplex.py"
    source = route.read_text(encoding="utf-8")
    assert 'held_for_voice = requested_mode == "focused"' in source
    assert "if held_for_voice:\n                a_voice_session(session_id, True)" in source
    assert "a_voice_session(session_id, False)" in source
