"""A whole run brings her language organ back before a turn if its worker stopped.

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


class _Gate:
    def __init__(self, ready: bool, comes_back_after: int = 1, raises_first: bool = False) -> None:
        self.ready = ready
        self.calls = 0
        self.comes_back_after = comes_back_after
        self.raises_first = raises_first

    def get_conversation_status(self) -> dict:
        return {"conversation_ready": self.ready}

    async def ensure_foreground_ready(self, timeout: float | None = None) -> dict:  # noqa: ASYNC109
        self.calls += 1
        if self.raises_first and self.calls == 1:
            raise RuntimeError("foreground_warmup_timeout: recovery handed to the background")
        if self.calls >= self.comes_back_after:
            self.ready = True
        return {"state": "ready" if self.ready else "warming"}


@pytest.fixture
def gate(monkeypatch):
    box: dict = {}
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: box.get(name, default)))
    monkeypatch.setattr(language_organ, "_RECHECK_S", 0.0)
    return box


def test_a_ready_lane_is_left_alone(gate) -> None:
    gate["inference_gate"] = _Gate(ready=True)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out == {"checked": True, "recovered": False}
    assert gate["inference_gate"].calls == 0


def test_a_stopped_worker_is_warmed_until_the_lane_is_ready(gate) -> None:
    gate["inference_gate"] = _Gate(ready=False, comes_back_after=2, raises_first=True)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["recovered"] is True
    assert gate["inference_gate"].calls == 2


def test_a_run_on_the_stub_organ_is_not_checked(gate) -> None:
    gate["inference_gate"] = _Gate(ready=False)
    assert asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=False))) == {"checked": False}
    assert gate["inference_gate"].calls == 0


def test_every_turn_checks_before_its_first_frame() -> None:
    source = (Path(__file__).resolve().parents[1] / "core" / "subject" / "driver.py").read_text(encoding="utf-8")
    body = source[source.index("    async def turn_once("):]
    first_statement = body[body.index('"""', body.index('"""') + 3) + 3 :].lstrip()
    assert first_statement.startswith("await keep_language_ready(self)")
