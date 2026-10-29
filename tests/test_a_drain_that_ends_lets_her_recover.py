"""A setpoint that adapts to a drain returns when the drain ends, and dread follows how hard she is pressed.

On the seed-89 carrier's recording and seed 97's campaign (29 October) her
prospective dread rose from 0.40 to 0.97 over the middle of the run and stayed
there. The homeostasis engine let each setpoint drift only toward the level its
drive reached, and its own pull never lifts a drive past its setpoint, so a
setpoint could only fall: any steady drain walked persistence's setpoint to its
floor and held persistence near zero, which dread reads as nearly the worst.
"""
from __future__ import annotations

import asyncio

import pytest

from core.consciousness import homeostasis as module
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.homeostatic_coupling import HomeostaticCoupling

pytestmark = pytest.mark.unit

#: The level below which HomeostaticCoupling counts a drive toward dread.
_DREAD_STARTS = HomeostaticCoupling._LOW_DRIVE / 100.0


class _Body:
    def __init__(self) -> None:
        self.anxiety = 0.0

    def get_status(self) -> dict:
        return {"soma": {"resource_anxiety": self.anxiety, "thermal_load": 0.0}}


@pytest.fixture
def body(monkeypatch) -> _Body:
    soma = _Body()
    monkeypatch.setattr(module.ServiceContainer, "get", lambda name, default=None: soma if name == "soma" else default)
    return soma


def _pulse(engine: HomeostasisEngine, times: int) -> None:
    async def _run() -> None:
        for _ in range(times):
            await engine.pulse()

    asyncio.run(_run())


def test_a_steady_drain_does_not_walk_her_setpoint_to_its_floor(body) -> None:
    engine = HomeostasisEngine()
    body.anxiety = 0.2
    _pulse(engine, 6000)
    # As it was: persistence 0.004 and its setpoint at the 0.2 floor, dread 0.99.
    assert engine.get_setpoints()["persistence"] > 0.6
    assert engine.persistence > _DREAD_STARTS


def test_her_setpoint_returns_when_the_drain_ends(body) -> None:
    engine = HomeostasisEngine()
    body.anxiety = 0.2
    _pulse(engine, 6000)
    body.anxiety = 0.0
    _pulse(engine, 8000)
    designed = HomeostasisEngine.DESIGNED_SETPOINTS["persistence"]
    # As it was: setpoint and persistence both held at 0.2 for good.
    assert engine.get_setpoints()["persistence"] == pytest.approx(designed, abs=0.01)
    assert engine.persistence == pytest.approx(designed, abs=0.01)


def test_a_drain_she_cannot_meet_still_registers(body) -> None:
    engine = HomeostasisEngine()
    body.anxiety = 0.4
    _pulse(engine, 6000)
    assert engine.persistence < _DREAD_STARTS

