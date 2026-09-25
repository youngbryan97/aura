"""A person's turn in a whole run is an open turn, as it is on the desktop.

The subject driver runs her phases itself and never bound a turn, so in a
whole run the router took every generation a person waited for as unowned
work and armed its thread watchdog at the flat budget for a short reply. On
23 September it stopped her cortex's worker 105 s into a tool call that was
still decoding, and the reports run that followed measured nothing.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from core.brain.llm_health_router import _start_endpoint_wall_clock_watchdog
from core.runtime.turn_origin import a_person_is_waiting
from core.runtime.turn_outcome import current_turn
from core.subject.driver import Condition, SubjectRuntime
from core.subject.language_organ import person_turn

pytestmark = pytest.mark.unit


def _runtime(*, whole: bool, reply: str = "") -> SimpleNamespace:
    return SimpleNamespace(whole=whole, state=SimpleNamespace(cognition=SimpleNamespace(last_response=reply)))


def test_a_persons_turn_in_a_whole_run_is_bound_and_served() -> None:
    runtime = _runtime(whole=True)
    with person_turn(runtime, "user") as outcome:
        assert current_turn() is outcome
        assert a_person_is_waiting(outcome.origin)
        runtime.state.cognition.last_response = "Here is what I was thinking about."
    assert current_turn() is None
    assert outcome.is_finalized
    assert outcome.receipt.served_answer == "Here is what I was thinking about."


def test_a_turn_that_served_nothing_says_so() -> None:
    with person_turn(_runtime(whole=True), "user") as outcome:
        pass
    assert outcome.is_finalized
    assert not outcome.receipt.served_answer


@pytest.mark.parametrize(("whole", "origin"), [(False, "user"), (True, "autonomous"), (True, "idle")])
def test_nothing_is_bound_for_the_stub_organ_or_for_her_own_work(whole: bool, origin: str) -> None:
    with person_turn(_runtime(whole=whole), origin) as outcome:
        assert outcome is None
        assert current_turn() is None


def _watchdog_fires(*, bound: bool) -> bool:
    aborted = threading.Event()
    client = SimpleNamespace(force_abort_active_generation=lambda **_: aborted.set() or True)

    def arm() -> None:
        fired, _, handle = _start_endpoint_wall_clock_watchdog(
            client, reason="endpoint_timeout:Cortex:0.05s", timeout_s=0.05, user_facing=True, person_is_waiting=True
        )
        fired.wait(0.5)
        handle.cancel()
        result.append(fired.is_set())

    result: list[bool] = []
    if bound:
        with person_turn(_runtime(whole=True), "user"):
            arm()
    else:
        arm()
    return result[0]


def test_the_router_lets_a_persons_generation_run_on_inside_the_turn() -> None:
    assert _watchdog_fires(bound=False), "without a turn the thread watchdog aborts at its budget"
    assert not _watchdog_fires(bound=True), "inside a person's turn the endpoint's own liveness governs"


def test_turn_once_runs_the_turn_inside_it() -> None:
    import inspect

    from core.subject import language_organ

    assert inspect.unwrap(SubjectRuntime.turn_once).__name__ == "turn_once"
    assert SubjectRuntime.turn_once.__code__ is language_organ.opens_the_turn(lambda *a, **k: None).__code__
    seen: list[object] = []
    runtime = _runtime(whole=True)

    async def body(runtime_, condition, **_):
        seen.append(current_turn())
        runtime.state.cognition.last_response = "an answer"
        return []

    asyncio.run(language_organ.opens_the_turn(body)(runtime, Condition("report", "How are you feeling?", origin="user")))
    assert seen and seen[0] is not None and seen[0].is_finalized
    assert seen[0].receipt.served_answer == "an answer"


def test_the_turn_is_finalized_as_the_desktops_chat_route_finalizes_it(monkeypatch) -> None:
    """Under `chat`, not the fail-closed `cognitive_engine`, which raised on an unserved answer."""
    import core.runtime.turn_outcome as turn_outcome

    seen: list[str] = []
    original = turn_outcome.finalize_turn

    def finalize(outcome, *, subsystem="turn_outcome"):
        seen.append(subsystem)
        return original(outcome, subsystem=subsystem)

    monkeypatch.setattr(turn_outcome, "finalize_turn", finalize)
    with person_turn(_runtime(whole=True), "user"):
        pass
    assert seen == ["chat"]
