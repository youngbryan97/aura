"""A whole run starts her language organ's worker again before a turn if it stopped.

The router's endpoint timeout stops a model worker to abort a generation. In the
whole report run of 23 September nothing started it again, and every turn after
the first ended in the failure sentence.
"""

from __future__ import annotations

import asyncio
import inspect
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
    assert out == {"checked": True, "recovered": False, "router_waited_s": 0.0}
    assert gate["inference_gate"].respawns == 0


class _Endpoint:
    """The router's endpoint for her cortex, eligible again after `closed_after` looks."""

    def __init__(self, closed_after: int) -> None:
        self.looks = 0
        self.closed_after = closed_after
        self.state = SimpleNamespace(value="open")
        self.last_failure_reason = "worker_not_alive,init_not_complete,lane_warming"

    def probe_eligible(self) -> bool:
        self.looks += 1
        return self.looks > self.closed_after


@pytest.fixture
def router(gate, monkeypatch):
    monkeypatch.setattr(language_organ, "_ROUTER_RECHECK_S", 0.0)

    def install(endpoint: _Endpoint) -> _Endpoint:
        gate["llm_router"] = SimpleNamespace(endpoints={"Cortex": endpoint})
        return endpoint

    return install


def test_a_turn_waits_until_the_router_sends_her_reply_to_her_cortex_again(gate, router) -> None:
    # Seed 101, 30 October, 21:04: the gate said her cortex was restored while
    # the router's circuit for it was open, and every arm fell to the failure
    # sentence within seconds.
    gate["inference_gate"] = _Gate(alive=True)
    endpoint = router(_Endpoint(closed_after=3))
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert endpoint.looks == 4
    assert "router_took_her" not in out


def test_after_a_restart_the_turn_also_waits_for_the_router(gate, router) -> None:
    gate["inference_gate"] = _Gate(alive=False, ready_after_checks=2)
    endpoint = router(_Endpoint(closed_after=2))
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["recovered"] is True
    assert endpoint.looks == 3


def test_a_router_that_never_takes_her_back_is_reported_not_waited_on_forever(gate, router, monkeypatch) -> None:
    monkeypatch.setattr(language_organ, "FOREGROUND_READY_S", 0.05)
    gate["inference_gate"] = _Gate(alive=True)
    router(_Endpoint(closed_after=10**9))
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["router_took_her"] is False


def test_the_wait_reads_the_circuit_without_moving_it() -> None:
    from core.brain.llm_health_router import CircuitState, EndpointHealth

    endpoint = EndpointHealth(name="Cortex", url="", model="m", is_local=True)
    endpoint.state = CircuitState.OPEN
    endpoint.cooldown_until_monotonic = 0.0
    assert endpoint.probe_eligible() is True
    # An observer that flipped OPEN to HALF_OPEN would spend the probe the
    # turn's own reply is meant to be.
    assert endpoint.state is CircuitState.OPEN
    source = inspect.getsource(language_organ._until_the_router_takes_her)
    assert "probe_eligible" in source and "is_available" not in source


def test_a_stopped_worker_is_started_once_and_watched_until_ready(gate) -> None:
    gate["inference_gate"] = _Gate(alive=False, ready_after_checks=3)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["recovered"] is True
    assert gate["inference_gate"].respawns == 1
    # Asking the gate to warm the lane on every recheck kept its quiet window
    # open and the warmup refused; the recovery is asked once and watched.
    assert gate["inference_gate"].warmups == 0


class _DecliningGate(_Gate):
    """A gate whose recovery declines to load the first few times it is asked."""

    def __init__(self, declines: int, recovering: bool = False) -> None:
        super().__init__(alive=False)
        self.declines = declines
        self._cortex_recovery_in_progress = recovering

    async def _respawn_cortex_if_needed(self) -> None:
        self.respawns += 1
        if self.respawns > self.declines:
            self._mlx_client.alive = True


def test_a_recovery_that_ends_with_no_worker_is_asked_again(gate) -> None:
    # Seed 97's report run, 29 October: the gate declined while the stopped
    # worker's memory was still coming back, and the turn waited out 900 s.
    gate["inference_gate"] = _DecliningGate(declines=2)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["recovered"] is True
    assert out["asked"] == gate["inference_gate"].respawns == 3


def test_a_recovery_still_working_is_not_asked_again(gate, monkeypatch) -> None:
    monkeypatch.setattr(language_organ, "FOREGROUND_READY_S", 0.05)
    gate["inference_gate"] = _DecliningGate(declines=10, recovering=True)
    out = asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=True)))
    assert out["recovered"] is False
    assert gate["inference_gate"].respawns == 1


def test_a_run_on_the_stub_organ_is_not_checked(gate) -> None:
    gate["inference_gate"] = _Gate(alive=False)
    assert asyncio.run(language_organ.keep_language_ready(SimpleNamespace(whole=False))) == {"checked": False}
    assert gate["inference_gate"].respawns == 0


def test_every_turn_checks_before_its_first_frame() -> None:
    """The check runs before the turn's body, and outside any turn it opens."""
    from core.runtime.turn_outcome import current_turn
    from core.subject.driver import SubjectRuntime

    assert inspect.unwrap(SubjectRuntime.turn_once) is not SubjectRuntime.turn_once
    order: list[str] = []

    async def fake_ready(runtime):
        order.append("ready" if current_turn() is None else "ready inside a turn")
        return {"checked": True}

    async def body(runtime, condition, **_):
        order.append("body")
        return []

    runtime = SimpleNamespace(whole=True, state=SimpleNamespace(cognition=SimpleNamespace(last_response="")))
    original = language_organ.keep_language_ready
    language_organ.keep_language_ready = fake_ready
    try:
        asyncio.run(language_organ.opens_the_turn(body)(runtime, SimpleNamespace(origin="user")))
    finally:
        language_organ.keep_language_ready = original
    assert order == ["ready", "body"]

