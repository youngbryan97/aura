"""A stopped owner ends its executing work, even when a waiter is shielded."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.runtime.what_stops_it import AnExecutionContext, Stopped, interruptible, under
from core.skills.fluid_executor import FluidExecutor, Step


def test_registered_lifetime_invariant_measures_detach_and_stop_direction():
    from core.runtime.what_stops_it import _stop_listener_lifetime_invariant

    assert _stop_listener_lifetime_invariant() == ()


@pytest.mark.asyncio
async def test_owner_stop_interrupts_a_shielded_effect_handler():
    from core.runtime.action_executor import _invoke_effect_handler

    context = AnExecutionContext(doing="document export")
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def effect(_context):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return {"ok": True}

    with under(context):
        operation = asyncio.create_task(_invoke_effect_handler(effect, {}, timeout_s=30))
    await entered.wait()
    context.stopping.stop("user pressed Stop")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 1.)
    assert cleaned.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["observe", "satisfied", "decide", "approve"])
async def test_stop_during_an_await_prevents_the_next_dispatch(phase):
    context = AnExecutionContext(doing="edit worksheet")
    entered = asyncio.Event()
    sent = []

    async def pause():
        entered.set()
        await asyncio.Event().wait()

    async def observe():
        if phase == "observe":
            await pause()
        return {}

    async def satisfied(_):
        if phase == "satisfied":
            await pause()
        return False

    async def act():
        sent.append("input")

    async def decide(_):
        if phase == "decide":
            await pause()
        return Step("place", act)

    executor = FluidExecutor(gateway=SimpleNamespace(approve=lambda _: True))
    if phase == "approve":
        async def approve(_):
            await pause()
            return True, ""
        executor._approved = approve
    operation = asyncio.create_task(executor.pursue("edit", observe=observe, decide=decide,
                                    is_satisfied=satisfied, execution_context=context))
    await entered.wait()
    context.stopping.stop("user pressed Stop")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 1.)
    assert sent == []


@pytest.mark.asyncio
async def test_swallowed_task_cancellation_cannot_retry_or_verify_a_stopped_action():
    context = AnExecutionContext(doing="simulation")
    entered = asyncio.Event()
    verifier = SimpleNamespace(verify=AsyncMock())
    recovery = AsyncMock()
    sent = []

    async def act():
        sent.append("input")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return None

    executor = FluidExecutor(gateway=SimpleNamespace(approve=lambda _: True), verifier=verifier)
    operation = asyncio.create_task(executor.run_step(Step("simulate", act, verify="file_exists", recovery=recovery),
                                                     execution_context=context))
    await entered.wait()
    context.stopping.stop("stop")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 1.)
    assert sent == ["input"]
    verifier.verify.assert_not_called()
    recovery.assert_not_called()


@pytest.mark.asyncio
async def test_completed_scope_detaches_and_late_stop_cannot_cancel_its_reused_task():
    context = AnExecutionContext()
    with interruptible(context):
        await asyncio.sleep(0)
    context.stopping.stop("after completion")
    await asyncio.sleep(0)
    assert not asyncio.current_task().cancelling()


@pytest.mark.asyncio
async def test_a_stop_from_another_thread_reaches_only_the_bound_operation():
    context = AnExecutionContext()
    child = context.under("owned child")
    sibling = context.under("sibling")
    entered = asyncio.Event()

    async def work():
        with interruptible(child):
            entered.set()
            await asyncio.Event().wait()

    operation = asyncio.create_task(work())
    await entered.wait()
    await asyncio.to_thread(child.stopping.stop, "child stopped")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 1.)
    assert not context.stopping.stopped and not sibling.stopping.stopped


@pytest.mark.asyncio
async def test_nested_scopes_cancel_once_so_cleanup_can_release_inputs():
    context = AnExecutionContext()
    entered = asyncio.Event()
    released = []

    async def work():
        with interruptible(context):
            with interruptible(context):
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    released.append("up")

    operation = asyncio.create_task(work())
    await entered.wait()
    context.stopping.stop("stop")
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert released == ["up"]


@pytest.mark.asyncio
async def test_stopped_page_inputs_are_refused_but_held_inputs_can_be_released():
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens
    from core.skills.screen_pursuit_on_a_page import OnAPage

    page = SimpleNamespace(mouse=SimpleNamespace(click=AsyncMock(), move=AsyncMock(), down=AsyncMock(), up=AsyncMock()),
                           keyboard=SimpleNamespace(press=AsyncMock(), up=AsyncMock()))
    context = AnExecutionContext()
    context.stopping.stop("stop")
    with under(context):
        surface = OnAPage(page, "editor")
        for operation in (surface.press("space"), surface.click(.5, .5, [0, 0, 400, 300]),
                          surface.carry((.1, .1), (.5, .5), [0, 0, 400, 300])):
            with pytest.raises(Stopped):
                await operation
        hands = PlayingAsItHappens(page, (0., 0., 1., 1.), "move", 100.)
        for operation in (hands.tap("space"), hands.down("right"), hands.click(.5, .5), hands.point(.5, .5), hands.press(.5, .5)):
            with pytest.raises(Stopped):
                await operation
        await hands.up("right")
    page.mouse.click.assert_not_called()
    page.mouse.down.assert_not_called()
    page.mouse.move.assert_not_called()
    page.keyboard.press.assert_not_called()
    page.keyboard.up.assert_awaited_once()


@pytest.mark.asyncio
async def test_realtime_play_is_interruptible_while_waiting_for_a_picture():
    from core.agency.playing_as_it_happens import play_as_it_happens

    context = AnExecutionContext()
    entered = asyncio.Event()
    hands = SimpleNamespace(tap=AsyncMock(), down=AsyncMock(), up=AsyncMock())

    async def look():
        entered.set()
        await asyncio.Event().wait()

    with under(context):
        operation = asyncio.create_task(play_as_it_happens(look, hands, keys=["right"], seconds=300))
    await entered.wait()
    context.stopping.stop("stop")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 1.)
    hands.down.assert_not_called()
    hands.tap.assert_not_called()
