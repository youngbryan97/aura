"""A search over her draft keeps to the turn's clock, and a turn that ran out ends its kernel work.

LIVE 19 November 2026, four times on a quiet machine: "Can you look up what the latest stable release of Python is right
now?" was read as code. After her draft was written, a deep search of nine candidates began, ran past the turn's end
and kept the draft. The turn had timed out, so the person got nothing. The turn's kernel work was not cancelled either,
since the part of the route that started it kept the task to itself, and it held AuraKernel.StateLock while the next
turn waited for it.
"""
from __future__ import annotations

import asyncio
import time
import types

import pytest

from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.runtime.the_turn_clock import ROOM_KEPT_FOR_DELIVERY_S, close_the_turn_clock, open_a_turn_clock

#: A request the amplifier takes up, from tests/test_phase_response_amplifier_wiring.py.
PLANNING = "Schedule these jobs to minimize makespan: [{'name':'A','duration':2}, {'name':'B','duration':3}]"
DRAFT = "Her draft, written before any search."


class _NoModel:
    async def think(self, prompt: str, **kwargs) -> str:
        raise AssertionError("no candidate is generated in these tests")


def _asked(**changes):
    base = dict(objective=PLANNING, draft=DRAFT, llm=_NoModel(), state=types.SimpleNamespace(metadata={}),
                request_timeout=180.0, is_user_facing=True, is_background=False, proof_or_benchmark=False)
    return {**base, **changes}


@pytest.fixture
def her_draft_cost(monkeypatch):
    """What one candidate costs, as the search's own allowance prices it, and a receipt that says something else."""
    def cost(seconds: float, *, receipt_says: float = 0.0) -> None:
        monkeypatch.setattr("core.brain.reasoning_amplifier_v2.search_allowance",
                            lambda *a, **k: (seconds, max(seconds * 3, 1.0)))
        receipt = types.SimpleNamespace(latency_s={"prefill": receipt_says / 3, "decode": 2 * receipt_says / 3})
        monkeypatch.setattr("core.verify.turn_receipt.current_receipt", lambda: receipt)
    cost(0.0)
    yield cost
    close_the_turn_clock()


@pytest.fixture
def searches(monkeypatch):
    seen: list[dict] = []

    async def amplify_turn(objective, generate, **kwargs):
        seen.append(kwargs)
        await asyncio.sleep(float(kwargs.get("_takes", 0.0)) or 30.0)
        raise AssertionError("the search was meant to be cut off")

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.amplify_turn", amplify_turn)
    return seen


@pytest.mark.asyncio
async def test_with_less_left_than_one_candidate_costs_the_search_never_starts(her_draft_cost, searches):
    her_draft_cost(30.0)
    open_a_turn_clock(ROOM_KEPT_FOR_DELIVERY_S + 5.0)
    out = await UnitaryResponsePhase._maybe_amplify_response(types.SimpleNamespace(_last_reasoning_receipt=None),
                                                             **_asked())
    assert out == DRAFT and searches == []


@pytest.mark.asyncio
async def test_a_receipt_that_reads_more_than_a_whole_turn_does_not_stop_a_search_that_fits(her_draft_cost, searches):
    """LIVE 19 November 2026 the kernel turn's receipt read up to 601 s, and the search stood down on every math problem
    of the baseline with up to 245 s left: MATH-500 went from 9 of 12 to 2."""
    her_draft_cost(20.0, receipt_says=505.0)
    open_a_turn_clock(ROOM_KEPT_FOR_DELIVERY_S + 0.5 + 20.0)
    out = await UnitaryResponsePhase._maybe_amplify_response(types.SimpleNamespace(_last_reasoning_receipt=None),
                                                             **_asked())
    assert searches, "it searched" and out == DRAFT


@pytest.mark.asyncio
async def test_the_search_is_given_what_is_left_and_her_draft_stands_when_it_runs_out(her_draft_cost, searches):
    open_a_turn_clock(ROOM_KEPT_FOR_DELIVERY_S + 0.5)
    started = time.monotonic()
    out = await UnitaryResponsePhase._maybe_amplify_response(types.SimpleNamespace(_last_reasoning_receipt=None),
                                                             **_asked())
    assert out == DRAFT, "the search ran out, and her draft is what the turn has"
    assert searches and searches[0]["time_budget_s"] <= 0.5
    assert time.monotonic() - started < 3.0, "it stops at the turn's clock, not at the search's own"


@pytest.mark.asyncio
async def test_with_no_turn_clock_the_search_keeps_its_own_budget(her_draft_cost, monkeypatch):
    close_the_turn_clock()
    seen: list[dict] = []

    async def amplify_turn(objective, generate, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.amplify_turn", amplify_turn)
    with pytest.raises(RuntimeError, match="stop here"):
        await UnitaryResponsePhase._maybe_amplify_response(types.SimpleNamespace(_last_reasoning_receipt=None),
                                                           **_asked())
    assert seen[0]["time_budget_s"] == 180.0, "background work and tests have no turn to keep to"


@pytest.mark.asyncio
async def test_the_kernel_task_a_turn_starts_is_the_one_its_timeout_cancels(monkeypatch):
    from interface.routes import chat
    from interface.routes.chat_foreground_lane import _await_the_sovereign_kernel_reply

    monkeypatch.setattr(chat, "_KERNEL_SOFT_REPLY_SLA_SECONDS", 0.05)
    monkeypatch.setattr("interface.routes.chat_foreground_lane.resolve_inference_gate", lambda: None)
    held: list[asyncio.Task] = []
    cancelled: list[tuple[str, bool]] = []

    class _Kernel:
        def is_ready(self) -> bool:
            return True

        async def process(self, message, **kwargs):
            await asyncio.sleep(60)

    async def cancel(reason: str) -> None:
        task = held[-1] if held else None
        cancelled.append((reason, task is not None))
        if task is not None:
            task.cancel()

    async def no_protected_reply(_reason):
        return ""

    await _await_the_sovereign_kernel_reply(
        _attempt_protected_foreground_reply=no_protected_reply, _cancel_kernel_task_if_pending=cancel,
        _hold_kernel_task=held.append, _finalize_fastpath=None, _remaining_foreground_budget=lambda **_: 0.1,
        chat_origin="user", effective_user_message="hello", is_benchmark=False, kernel_timed_out=False,
        ki=_Kernel(), reply_text="")
    assert held and cancelled and cancelled[-1] == ("kernel_timeout", True), cancelled
    await asyncio.sleep(0)
    assert held[0].cancelled() or held[0].cancelling(), "the kernel work of a turn that ran out does not run on"


@pytest.mark.asyncio
async def test_the_conversational_amplifier_keeps_to_the_turns_clock_too(monkeypatch):
    """Its own budget only stops new generations from starting; one already started on her 27B ran its whole length."""
    from core.phases import response_generation_unitary as unitary

    monkeypatch.setattr(unitary._FLAG_CONVERSATIONAL_AMPLIFIER_LIVE, "value", lambda: "1")
    monkeypatch.setattr("core.brain.conversational_amplifier.is_conversationally_amplifiable", lambda objective, origin: True)
    monkeypatch.setattr(unitary, "_breath_in_words", lambda state, draft: 80)
    monkeypatch.setattr(unitary, "_taste_conversation_id", lambda state: "c")
    monkeypatch.setattr(unitary, "is_witnessing", lambda cognition: False)
    began: list = []

    async def slow(draft, **kwargs):
        began.append(kwargs["time_budget_s"])
        await asyncio.sleep(30)

    monkeypatch.setattr("core.brain.conversational_amplifier.amplify_conversation", slow)
    open_a_turn_clock(ROOM_KEPT_FOR_DELIVERY_S + 0.5)
    try:
        state = types.SimpleNamespace(cognition=types.SimpleNamespace(current_origin="user", rolling_summary=""),
                                      response_modifiers={"voice_word_budget": 50})
        phase = types.SimpleNamespace(_normalize_origin=lambda origin: origin)
        started = time.monotonic()
        out = await UnitaryResponsePhase._maybe_amplify_conversation(
            phase, objective="Tell me about your morning so far.", draft=DRAFT, llm=_NoModel(), state=state,
            request_timeout=40.0, is_user_facing=True, is_background=False, proof_or_benchmark=False)
    finally:
        close_the_turn_clock()
    assert began and out == DRAFT and time.monotonic() - started < 3.0
