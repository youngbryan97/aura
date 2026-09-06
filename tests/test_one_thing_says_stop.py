"""One token that says stop, carried through the model and the tool.

OpenHands threads a cancellation token through agent and tool calls so
cooperative cancellation composes. AutoGen puts it in the message API so a
handler cannot forget it. Aura had asyncio cancellation, deadlines and a token
inside the voice duplex — three mechanisms that do not compose — plus the
pattern this replaces, where a long step polls a module-level flag and cannot
say why it stopped, cannot be scoped to one turn, and cannot hand a narrower
stop to a subcall.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime.what_stops_it import (
    AnExecutionContext,
    Stopped,
    Stopping,
    current,
    stopping_with,
    under,
    what_is_not_threaded_yet,
)


def test_a_stop_says_why_and_says_it_once():
    token = Stopping("a turn")
    assert not token.stopped
    assert token.stop("the user pressed stop") is True
    assert token.stopped and token.why == "the user pressed stop"
    # Two subsystems noticing the same thing must not produce two reasons.
    assert token.stop("something else entirely") is False
    assert token.why == "the user pressed stop"


def test_a_child_dies_with_its_parent_and_not_the_other_way_round():
    """The direction is the whole design of composing."""

    turn = Stopping("turn")
    tool = turn.child("tool")
    tool.stop("the tool gave up")
    assert tool.stopped and not turn.stopped, "a subcall stopped its caller"

    other = Stopping("turn")
    under_it = other.child("tool")
    other.stop("the turn ended")
    assert under_it.stopped and under_it.why == "the turn ended"


def test_a_child_made_after_the_stop_is_born_stopped():
    turn = Stopping("turn")
    turn.stop("gone")
    assert turn.child("late").stopped, "a child made after the stop ran anyway"


def test_a_deadline_never_widens():
    with stopping_with("a turn", seconds=5) as turn:
        with stopping_with("a tool that wants longer", seconds=99) as tool:
            assert tool.seconds_left <= turn.seconds_left + 0.01


def test_check_is_the_one_call_a_loop_needs():
    with stopping_with("work", seconds=0) as ctx:
        ctx.check()
        ctx.stopping.stop("enough")
        with pytest.raises(Stopped) as raised:
            ctx.check()
        assert "enough" in str(raised.value)


def test_the_ambient_context_is_the_migration_path_and_is_counted():
    """Reading the ambient one is allowed and recorded, because it is not the end state."""

    before = dict(what_is_not_threaded_yet())
    with stopping_with("a turn"):
        current(whose="a caller nobody threaded")
    after = what_is_not_threaded_yet()
    assert after.get("a caller nobody threaded", 0) == before.get(
        "a caller nobody threaded", 0
    ) + 1


def test_the_tool_path_refuses_a_stopped_turn():
    from core.runtime.action_executor import ActionExecutor

    async def go() -> dict:
        token = Stopping("a turn")
        token.stop("the user pressed stop")
        return await ActionExecutor.execute(
            domain="tool_execution", action_name="anything", params={}, stopping=token
        )

    found = asyncio.run(go())
    assert found.get("ok") is False
    assert found.get("why") == "the user pressed stop"


def test_the_tool_path_reads_the_ambient_token_when_nobody_passed_one():
    from core.runtime.action_executor import ActionExecutor

    async def go() -> dict:
        with stopping_with("a turn") as ctx:
            ctx.stopping.stop("the turn ended")
            return await ActionExecutor.execute(
                domain="tool_execution", action_name="anything", params={}
            )

    found = asyncio.run(go())
    assert found.get("ok") is False and found.get("why") == "the turn ended"


def test_the_model_path_consults_it():
    """Checked by reading the gate, because generating needs a live model.

    What has to be true is that the model entry point asks before it starts.
    Whether it then generates is the model's business and not this test's.
    """

    import inspect

    from core.brain.inference_gate import InferenceGate

    source = inspect.getsource(InferenceGate.generate)
    assert "what_stops_it" in source, "the model path does not consult the token"
    assert "generation not started" in source


def test_an_asyncio_cancellation_becomes_a_reason():
    from core.runtime.what_stops_it import from_asyncio

    async def go() -> str:
        context = AnExecutionContext(stopping=Stopping("a turn"), doing="a turn")
        task = asyncio.create_task(asyncio.sleep(9))
        task.add_done_callback(from_asyncio(context))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        return context.stopping.why

    assert asyncio.run(go()) == "the task was cancelled"


def test_a_bound_context_is_what_current_returns():
    made = AnExecutionContext(stopping=Stopping("named"), doing="a thing")
    with under(made):
        assert current() is made
    assert current() is not made
