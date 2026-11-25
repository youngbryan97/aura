"""Work somebody asked her for holds back what she does for herself, and what works in Python beside it takes short turns."""
from __future__ import annotations

import sys
import threading

import pytest

from core.runtime import work_somebody_asked_for
from core.runtime.interpreter_turns import A_SHORT_TURN_S, short_turns


@pytest.fixture
def nothing_else_holds_it(monkeypatch):
    from core.runtime import background_policy as policy

    monkeypatch.setattr(policy, "foreground_only_runtime", lambda: False)
    monkeypatch.setattr(policy, "background_cognition_disabled_reason", lambda **_how: "")
    monkeypatch.setattr(policy, "_first_visible_conversation_probe_reason", lambda: "")
    monkeypatch.setattr(policy, "_foreground_activity_reason", lambda **_how: "")
    monkeypatch.setattr(policy, "_read_compute_pressure_reason", lambda: "a busy machine")
    return policy


def test_what_she_does_for_herself_waits_for_work_somebody_asked_for(nothing_else_holds_it):
    policy = nothing_else_holds_it
    assert policy.background_activity_reason(None, allow_no_user_anchor=True) != "work_somebody_asked_for"
    work_somebody_asked_for.begun("a")
    work_somebody_asked_for.begun("b")
    try:
        assert policy.background_activity_reason(None, allow_no_user_anchor=True) == "work_somebody_asked_for"
        work_somebody_asked_for.ended("a")
        assert policy.background_activity_reason(None, allow_no_user_anchor=True) == "work_somebody_asked_for"
    finally:
        work_somebody_asked_for.ended("a")
        work_somebody_asked_for.ended("b")
    assert policy.background_activity_reason(None, allow_no_user_anchor=True) != "work_somebody_asked_for"


async def test_a_request_counts_as_work_from_its_beginning_to_its_end(tmp_path, monkeypatch):
    import asyncio

    from core.understanding import carrying_out
    from core.understanding.requests_of_her import Part, RequestsOfHer

    async def nobody():
        return None

    monkeypatch.setattr(carrying_out, "nobody_waiting", nobody)
    ledger, seen, done = RequestsOfHer(tmp_path / "r.json"), [], asyncio.Event()
    request = ledger.asked("Make it.", effect="program", conversation="c")

    async def carrier(request, tell):
        seen.append(work_somebody_asked_for.under_way())
        return carrying_out.CameTo(how="built", parts=[Part("it runs", "met")])

    async def told(request):
        seen.append(work_somebody_asked_for.under_way())
        done.set()

    assert carrying_out.begin(request, tell=lambda line: None, told=told, carrier=carrier, ledger=ledger)
    await asyncio.wait_for(done.wait(), timeout=5)
    assert seen == [True, False]


def test_short_turns_are_asked_for_together_and_left_as_they_were():
    before = sys.getswitchinterval()
    with short_turns():
        assert sys.getswitchinterval() == pytest.approx(min(A_SHORT_TURN_S, before), rel=0.01)
        worker = threading.Thread(target=lambda: short_turns().__enter__())
        worker.start()
        worker.join()
    assert sys.getswitchinterval() < before or before <= A_SHORT_TURN_S
    short_turns().__exit__()
    assert sys.getswitchinterval() == pytest.approx(before, rel=0.01)


def test_her_practice_between_games_takes_short_turns(monkeypatch):
    from core.agency import practising_between_games as practice

    seen = []
    monkeypatch.setattr(practice, "_one_round_of", lambda name, written: seen.append(sys.getswitchinterval()))
    before = sys.getswitchinterval()
    practice._one_round("a-board", (0, 0))
    assert seen and seen[0] <= A_SHORT_TURN_S * 1.01 and sys.getswitchinterval() == pytest.approx(before, rel=0.01)


def test_her_whole_process_takes_short_turns():
    """LIVE 24 November 2026, 19:08: her event loop was starved for 14.4 s beside one thread working in Python."""
    import inspect

    import aura_main
    from core.brain.llm import mlx_worker
    from core.runtime.interpreter_turns import HER_TURN_S, take_short_turns

    was = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.005)
        assert take_short_turns() == pytest.approx(0.005)
        assert sys.getswitchinterval() == pytest.approx(HER_TURN_S)
        with short_turns():
            assert sys.getswitchinterval() <= HER_TURN_S * 1.01, "what asks for shorter turns is not given longer ones"
        assert sys.getswitchinterval() == pytest.approx(HER_TURN_S)
    finally:
        sys.setswitchinterval(was)
    assert "take_short_turns()" in inspect.getsource(aura_main.main) and "take_short_turns()" in inspect.getsource(mlx_worker._mlx_worker_loop)
