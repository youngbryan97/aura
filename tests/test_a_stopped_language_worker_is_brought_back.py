"""A whole run starts her language organ's worker again before a turn if it stopped.

The router's endpoint timeout stops a model worker to abort a generation. In the
whole report run of 23 September nothing started it again, and every turn after
the first ended in the failure sentence.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.subject import language_organ

pytestmark = pytest.mark.unit


class _Client:
    def __init__(self, alive: bool) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


class _Gate:
    def __init__(self, alive: bool, ready_after_checks: int = 1) -> None:
        self._mlx_client = _Client(alive)
        self.respawns = 0
        self.checks = 0
        self.ready_after_checks = ready_after_checks
        self.warmups = 0

    def get_conversation_status(self) -> dict:
        self.checks += 1
        return {"conversation_ready": self._mlx_client.alive and self.checks >= self.ready_after_checks}

    async def _respawn_cortex_if_needed(self) -> None:
        self.respawns += 1
        self._mlx_client.alive = True

    async def ensure_foreground_ready(self, timeout: float | None = None) -> dict:  # noqa: ASYNC109
        self.warmups += 1
        return {}


@pytest.fixture
def gate(monkeypatch):
    box: dict = {}
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: box.get(name, default)))
    monkeypatch.setattr(language_organ, "_RECHECK_S", 0.0)
    return box


def test_a_live_worker_is_left_alone(gate) -> None:
    gate["inference_gate"] = _Gate(alive=True)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out == {"checked": True, "recovered": False}
    assert gate["inference_gate"].respawns == 0


def test_a_stopped_worker_is_started_once_and_watched_until_ready(gate) -> None:
    gate["inference_gate"] = _Gate(alive=False, ready_after_checks=3)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["recovered"] is True
    assert gate["inference_gate"].respawns == 1
    # Asking the gate to warm the lane on every recheck kept its quiet window
    # open and the warmup refused; the recovery is asked once and watched.
    assert gate["inference_gate"].warmups == 0


def test_a_run_on_the_stub_organ_is_not_checked(gate) -> None:
    gate["inference_gate"] = _Gate(alive=False)
    assert asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=False))) == {"checked": False}
    assert gate["inference_gate"].respawns == 0


def test_every_turn_checks_before_its_first_frame() -> None:
    source = (Path(__file__).resolve().parents[1] / "core" / "subject" / "driver.py").read_text(encoding="utf-8")
    body = source[source.index("    async def turn_once("):]
    first_statement = body[body.index('"""', body.index('"""') + 3) + 3 :].lstrip()
    assert first_statement.startswith("await keep_language_ready(self)")
