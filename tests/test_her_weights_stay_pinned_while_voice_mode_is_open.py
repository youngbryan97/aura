"""Her cortex's weights stay resident while voice mode is open, and are let go when it closes (Bryan, 11 November 2026).

R11: her prefill ran at 375 tokens a second a minute after a turn and at 144 after 25 idle minutes, because mlx_lm
wires a model's memory only while it generates and macOS paged her weights out between turns.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.brain.llm import her_pinned_weights as pinning
from core.brain.llm.mlx_worker_pinned_weights import her_weights_pinned

_ROOT = Path(__file__).resolve().parents[1]


class _Mx:
    def __init__(self):
        self.limit = 0
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.array = type("array", (), {})

    def device_info(self):
        return {"max_recommended_working_set_size": 48 * 2 ** 30}

    def set_wired_limit(self, limit):
        previous, self.limit = self.limit, limit
        return previous


def _replies(pinned, mx, device="gpu"):
    sent = []
    her_weights_pinned(pinned, device=device, ipc_writer=SimpleNamespace(put=sent.append), job={"id": "j"}, mx=mx, model={})
    return sent[-1]


def test_pinning_raises_the_wired_limit_and_releasing_puts_it_back_to_none():
    mx = _Mx()
    reply = _replies(True, mx)
    assert reply["status"] == "ok" and reply["pinned"] is True and mx.limit == 48 * 2 ** 30 and reply["id"] == "j"
    reply = _replies(False, mx)
    assert reply["pinned"] is False and mx.limit == 0 and reply["previous_wired_limit"] == 48 * 2 ** 30


def test_a_cpu_worker_has_nothing_to_pin():
    reply = _replies(True, _Mx(), device="cpu")
    assert reply["status"] == "ok" and reply["pinned"] is False


def test_two_sessions_pin_once_and_release_when_the_last_closes(monkeypatch):
    asked = []
    monkeypatch.setattr(pinning, "_SESSIONS_PINNING", 0)
    monkeypatch.setattr(pinning, "_ask", lambda pinned: asked.append(pinned) or {"pinned": pinned})
    pinning.a_voice_session_opened()
    pinning.a_voice_session_opened()
    pinning.a_voice_session_closed()
    assert asked == [True], "still one session open"
    pinning.a_voice_session_closed()
    assert asked == [True, False]


def test_a_worker_that_cannot_answer_never_fails_the_session(monkeypatch):
    def refuses(*_a, **_k):
        raise TimeoutError("no reply")

    monkeypatch.setattr(pinning, "_her_cortex_client", lambda: SimpleNamespace(_send_worker_control_action=refuses))
    assert pinning._ask(True) is None


def test_the_voice_route_pins_on_open_and_releases_on_close():
    route = (_ROOT / "interface/routes/voice_duplex.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread(a_voice_session_opened)" in route
    assert "await asyncio.to_thread(a_voice_session_closed)" in route
    # The release waits for the pin it undoes, or a pin answered late would hold her weights after the last session.
    closing = route[route.index("if pinning is not None:"):route.index("await asyncio.to_thread(a_voice_session_closed)")]
    assert "await pinning" in closing


def test_nothing_her_worker_answers_is_awaited_before_the_socket_is_read():
    """LIVE 11 November 2026: the pin and her live context's first read were awaited between the session's start and
    its receive loop. The browser streamed audio into a socket nobody read, the client's keepalive ping went
    unanswered for 20 s, and the session closed before its first turn."""
    route = (_ROOT / "interface/routes/voice_duplex.py").read_text(encoding="utf-8")
    start = route.index("await session.start()") + len("await session.start()")
    between = route[start:route.index("message = await ws.receive()", start)]
    assert "await " not in between, between
    worker = (_ROOT / "core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    assert 'elif action in ("pin_weights", "release_weights"):' in worker
