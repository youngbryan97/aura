from __future__ import annotations

import asyncio
import types

import pytest

import core.continuous_cognition as continuous_cognition
from core.continuous_cognition import ContinuousCognitionLoop


def test_continuous_cognition_task_creation_failure_fails_closed(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def _record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    class _BrokenTracker:
        def create_task(self, coro, name=None):
            self.name = name
            raise RuntimeError("scheduler offline")

    monkeypatch.setattr(continuous_cognition, "record_degradation", _record)
    monkeypatch.setattr(
        continuous_cognition.ServiceContainer,
        "register_instance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(continuous_cognition, "get_task_tracker", lambda: _BrokenTracker())

    async def _scenario():
        loop = ContinuousCognitionLoop()
        with pytest.raises(RuntimeError):
            await loop.start()
        assert loop._running is False
        assert loop._task is None

    asyncio.run(_scenario())

    assert recorded
    assert recorded[0][0] == "continuous_cognition"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert recorded[0][2]["severity"] == "critical"


def test_continuous_cognition_run_loop_survives_step_failure(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def _record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    loop = ContinuousCognitionLoop()
    calls = {"count": 0}

    def _flaky_step():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("brainstem step failed once")
        loop._running = False

    monkeypatch.setattr(continuous_cognition, "record_degradation", _record)
    monkeypatch.setattr(loop, "_cognitive_step", _flaky_step)
    loop._running = True

    asyncio.run(loop._run())

    assert calls["count"] == 2
    assert loop._running is False
    assert loop._tick_count == 2
    assert recorded
    assert recorded[0][0] == "continuous_cognition"
    assert recorded[0][1] == "RuntimeError"
    assert "kept ContinuousCognition loop alive" in str(recorded[0][2]["action"])


def test_continuous_cognition_rejects_malformed_drive_vector(monkeypatch):
    recorded: list[tuple[str, str]] = []
    submitted: list[dict[str, object]] = []

    class _Drive:
        def get_drive_vector(self):
            return {"curiosity": "not-a-number"}

    loop = ContinuousCognitionLoop()
    loop._drive_engine = _Drive()
    loop._synthesizer = types.SimpleNamespace(
        submit=lambda **payload: submitted.append(payload),
    )
    loop._last_initiative_seed = 0.0

    monkeypatch.setattr(
        continuous_cognition,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    loop._seed_initiative_from_drives()

    assert submitted == []
    assert ("continuous_cognition", "ValueError") in recorded
