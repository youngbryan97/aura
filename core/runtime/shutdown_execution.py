"""Bounded execution primitives for synchronous shutdown callbacks."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Callable

from core.utils.task_tracker import (
    begin_shutdown_resource_creation_scope,
    end_shutdown_resource_creation_scope,
)

logger = logging.getLogger(__name__)


def _thread_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(name or "callback"))
    return f"aura-shutdown:{normalized[:48]}"


def run_sync_shutdown_callable_blocking[T](
    callback: Callable[[], T],
    *,
    timeout_s: float,
    name: str,
) -> T:
    """Bound a synchronous teardown callback without executor ownership."""

    completed = threading.Event()
    result_box: list[T] = []
    error_box: list[BaseException] = []

    def _worker() -> None:
        try:
            result_box.append(callback())
        except BaseException as exc:  # noqa: BLE001 - isolate teardown boundary
            error_box.append(exc)
        finally:
            completed.set()

    worker = threading.Thread(target=_worker, name=_thread_name(name), daemon=True)
    resource_token = begin_shutdown_resource_creation_scope()
    try:
        worker.start()
    finally:
        end_shutdown_resource_creation_scope(resource_token)

    budget = max(0.05, float(timeout_s))
    if not completed.wait(budget):
        raise TimeoutError(f"shutdown callback '{name}' exceeded {budget:.3f}s")
    if error_box:
        error = error_box[0]
        if isinstance(error, Exception):
            raise error
        raise RuntimeError(
            f"shutdown callback raised {type(error).__name__}: {error}"
        )
    if not result_box:
        raise RuntimeError(f"shutdown callback '{name}' completed without a result")
    return result_box[0]


async def run_sync_shutdown_callable[T](
    callback: Callable[[], T],
    *,
    timeout_s: float,
    name: str,
) -> T:
    """Run a synchronous teardown callback without risking executor shutdown.

    A timed-out ``ThreadPoolExecutor`` callback can keep Python alive because
    interpreter shutdown joins executor workers. Teardown callbacks instead run
    in a dedicated daemon thread that remains visible to runtime hygiene and can
    therefore degrade the final verdict without wedging process exit.
    """

    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[T] = loop.create_future()

    def _deliver_result(value: T) -> None:
        if not result_future.done():
            result_future.set_result(value)

    def _deliver_error(exc: BaseException) -> None:
        if result_future.done():
            return
        if isinstance(exc, Exception):
            result_future.set_exception(exc)
        else:
            result_future.set_exception(
                RuntimeError(f"shutdown callback raised {type(exc).__name__}: {exc}")
            )

    def _worker() -> None:
        try:
            result = callback()
        except BaseException as exc:  # noqa: BLE001 - isolate teardown boundary
            try:
                loop.call_soon_threadsafe(_deliver_error, exc)
            except RuntimeError:
                # The loop is already closed, so nobody is waiting on this
                # future — but the callback's failure would vanish with it.
                logger.warning(
                    "shutdown callback raised %s: %s, and the loop was gone "
                    "before it could be delivered",
                    type(exc).__name__,
                    exc,
                )
                return
        else:
            try:
                loop.call_soon_threadsafe(_deliver_result, result)
            # not a failure: a closed loop has nobody left waiting for this
            # result, and the callback already ran.
            except RuntimeError:
                return

    worker = threading.Thread(target=_worker, name=_thread_name(name), daemon=True)
    resource_token = begin_shutdown_resource_creation_scope()
    try:
        worker.start()
    finally:
        end_shutdown_resource_creation_scope(resource_token)

    try:
        async with asyncio.timeout(max(0.05, float(timeout_s))):
            return await asyncio.shield(result_future)
    except TimeoutError:
        # The daemon worker may finish later. Consume a late exception so the
        # loop does not emit an unhandled-future warning during finalization.
        def _consume_late_result(future: asyncio.Future[T]) -> None:
            try:
                future.exception()
            # not a failure: the comment above says why — this consumes a late
            # result so finalization does not warn about it, and whatever it
            # finds has already been reported through the timeout path.
            except (asyncio.CancelledError, Exception):
                return

        result_future.add_done_callback(_consume_late_result)
        raise
