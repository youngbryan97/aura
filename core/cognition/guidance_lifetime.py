"""Own background questions and their publications for one operation.

Finishing an operation closes its question owner, including questions still
waiting for admission. A generation stamp rejects delayed publication even
when an observer catches cancellation. The parent execution owner remains open.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

from core.runtime.what_stops_it import AnExecutionContext, Stopped, current, interruptible
from core.verify.invariants import invariant

logger = logging.getLogger("Aura.GuidanceLifetime")
TEARDOWN_SECONDS = .5


class GuidanceLifetime:
    """A child execution owner shared by questions of the same operation."""

    def __init__(self, context: AnExecutionContext) -> None:
        self.context = context
        self.generation = 0
        self.closed = False
        self.tasks: set[asyncio.Task] = set()
        self.background: set[asyncio.Task] = set()
        self.retirements: list[tuple[asyncio.Task, Callable[[], None]]] = []

    def own(self, task: asyncio.Task) -> None:
        self.tasks.add(task)
        self.background.add(task)

        def finished(done: asyncio.Task) -> None:
            self.tasks.discard(done)
            self.background.discard(done)
            self.retirements[:] = [(held, release) for held, release in self.retirements if held is not done]

        task.add_done_callback(finished)

    async def close(self) -> None:
        """Retire publication authority immediately and bound cooperative teardown."""
        if self.closed:
            return
        self.closed = True
        self.generation += 1
        for task, release in self.retirements:
            if not task.done():
                release()
        self.retirements.clear()
        self.context.stopping.stop("guidance_operation_finished")
        own = asyncio.current_task()
        pending = {task for task in self.tasks if task is not own and not task.done()}
        for task in pending:
            if not task.cancelling():
                task.cancel("guidance_operation_finished")
        if pending:
            _done, still_running = await asyncio.wait(pending, timeout=TEARDOWN_SECONDS)
            if still_running:
                logger.info("%d retired guidance questions are still releasing their observers", len(still_running))


@dataclass(frozen=True)
class PublicationStamp:
    """The owner generation in which a result was requested."""

    lifetime: GuidanceLifetime | None
    generation: int = 0

    def check(self) -> None:
        if self.lifetime is None:
            return
        if self.lifetime.closed or self.generation != self.lifetime.generation:
            raise asyncio.CancelledError("guidance_operation_finished")
        try:
            self.lifetime.context.check()
        except Stopped as exc:
            if self.lifetime.context.stopping.stopped:
                raise asyncio.CancelledError(exc.why) from None
            raise


_CURRENT: contextvars.ContextVar[GuidanceLifetime | None] = contextvars.ContextVar("guidance_lifetime", default=None)


def publication_stamp() -> PublicationStamp:
    lifetime = _CURRENT.get()
    stamp = PublicationStamp(lifetime, lifetime.generation if lifetime is not None else 0)
    stamp.check()
    return stamp


def create_guidance_task(operation: Coroutine) -> asyncio.Task:
    """Register the task before it can wait, including a task not yet scheduled."""
    try:
        stamp = publication_stamp()
    except BaseException:
        operation.close()
        raise
    task = asyncio.get_running_loop().create_task(operation)
    if stamp.lifetime is not None:
        stamp.lifetime.own(task)
    return task


def retire_when_closed(task: asyncio.Task, release: Callable[[], None]) -> None:
    """Release a pending request marker while its old result loses authority."""
    stamp = publication_stamp()
    if stamp.lifetime is not None:
        stamp.lifetime.retirements.append((task, release))

        def cancelled(done: asyncio.Task) -> None:
            if done.cancelled():
                release()

        task.add_done_callback(cancelled)


@asynccontextmanager
async def guidance_work() -> AsyncIterator[GuidanceLifetime]:
    """Scope questions to this operation without stopping its caller on completion."""
    lifetime = GuidanceLifetime(current(whose="guidance_work").under("background guidance"))
    token = _CURRENT.set(lifetime)
    try:
        yield lifetime
    finally:
        try:
            await lifetime.close()
        finally:
            _CURRENT.reset(token)


@contextmanager
def question_work() -> Iterator[PublicationStamp]:
    """Own admission, retry and generation; reject a result from a retired owner."""
    stamp = publication_stamp()
    if stamp.lifetime is None:
        yield stamp
        return
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("guidance questions require an asyncio task")
    lifetime = stamp.lifetime
    lifetime.tasks.add(task)
    try:
        with interruptible(lifetime.context):
            yield stamp
            stamp.check()
    finally:
        if task not in lifetime.background:
            lifetime.tasks.discard(task)


@invariant("cognition.completed_guidance_cannot_publish", scope="cognition",
           owner="core/cognition/guidance_lifetime.py", observational=False)
def _completed_guidance_invariant() -> tuple:
    parent = AnExecutionContext()
    lifetime = GuidanceLifetime(parent.under("guidance"))
    stamp = PublicationStamp(lifetime, lifetime.generation)
    lifetime.closed = True
    lifetime.generation += 1
    try:
        stamp.check()
    except asyncio.CancelledError:
        pass  # not a failure: the retired stamp refusing is what this checks
    else:
        raise AssertionError("a retired guidance owner retained publication authority")
    assert not parent.stopping.stopped, "guidance retirement stopped its parent"
    return ()
