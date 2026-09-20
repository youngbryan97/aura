"""`task.cancel()` then `await task`, and whose cancellation the `except` caught.

A hundred and fifty teardowns in this tree stop a background loop that way
and catch `asyncio.CancelledError` without re-raising. That is right for the
CHILD's cancellation and wrong for the caller's: while awaiting, this task
can be cancelled too, the same handler catches it, and the teardown reports
success after being told to stop. FAULT-001 and MATRIX-12 both name it —
"swallowed cancellation" — and nothing shows it, because the teardown looks
like it worked.
"""
from __future__ import annotations

import asyncio

import pytest

from core.utils.concurrency import cancel_and_join


async def _sleeper() -> None:
    await asyncio.sleep(60)


async def _raiser() -> None:
    raise ValueError("the loop died of something real")


async def _finished() -> int:
    return 1


async def _stubborn() -> None:
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        await asyncio.sleep(5)


@pytest.mark.asyncio
async def test_our_own_cancellation_is_not_eaten() -> None:
    """The defect, stated as the thing that used to happen."""
    child = asyncio.create_task(_sleeper())
    await asyncio.sleep(0)

    teardown = asyncio.create_task(cancel_and_join(child, owner="probe"))
    await asyncio.sleep(0)
    teardown.cancel()

    with pytest.raises(asyncio.CancelledError):
        await teardown


@pytest.mark.asyncio
async def test_the_old_pattern_would_have_swallowed_it() -> None:
    """The null: without the check, the caller runs on after being cancelled."""

    async def old(task: asyncio.Task) -> str:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return "teardown finished"

    child = asyncio.create_task(_sleeper())
    await asyncio.sleep(0)
    teardown = asyncio.create_task(old(child))
    await asyncio.sleep(0)
    teardown.cancel()

    assert await teardown == "teardown finished"


@pytest.mark.asyncio
async def test_the_childs_cancellation_is_absorbed() -> None:
    child = asyncio.create_task(_sleeper())
    await asyncio.sleep(0)

    await cancel_and_join(child, owner="probe")

    assert child.cancelled()


@pytest.mark.asyncio
async def test_a_child_that_died_of_something_real_is_recorded(monkeypatch) -> None:
    seen: list[tuple[str, BaseException]] = []
    import core.utils.concurrency as concurrency

    monkeypatch.setattr(
        concurrency,
        "record_degradation",
        lambda owner, exc, **kwargs: seen.append((owner, exc)),
    )
    child = asyncio.create_task(_raiser())
    await asyncio.sleep(0)

    await concurrency.cancel_and_join(child, owner="probe")

    assert [owner for owner, _exc in seen] == ["probe"]
    assert isinstance(seen[0][1], ValueError)


@pytest.mark.asyncio
async def test_a_finished_task_is_left_alone() -> None:
    child = asyncio.create_task(_finished())
    await child

    await cancel_and_join(child, owner="probe")

    assert child.result() == 1


@pytest.mark.asyncio
async def test_a_stubborn_child_is_bounded(monkeypatch) -> None:
    seen: list[BaseException] = []
    import core.utils.concurrency as concurrency

    monkeypatch.setattr(
        concurrency,
        "record_degradation",
        lambda owner, exc, **kwargs: seen.append(exc),
    )
    child = asyncio.create_task(_stubborn())
    await asyncio.sleep(0)

    await concurrency.cancel_and_join(child, owner="probe", timeout=0.05)

    assert [type(exc) for exc in seen] == [TimeoutError]
    child.cancel()


@pytest.mark.asyncio
async def test_none_is_not_an_error() -> None:
    await cancel_and_join(None, owner="probe")
