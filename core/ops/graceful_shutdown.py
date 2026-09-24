from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, ClassVar, cast

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import (
    ShutdownReport,
    get_shutdown_coordinator,
    publish_shutdown_verdict,
    request_shutdown,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.task_tracker import (
    begin_shutdown_task_creation_scope,
    end_shutdown_task_creation_scope,
    get_task_tracker,
)

logger = logging.getLogger("Aura.Shutdown")

ShutdownHook = Callable[[], Any | Awaitable[Any]]


class GracefulShutdown:
    """Root signal owner and compatibility hook bridge for canonical teardown."""

    _hooks: ClassVar[list[ShutdownHook]] = []
    _is_shutting_down: ClassVar[bool] = False
    _shutdown_event: ClassVar[asyncio.Event | None] = None
    _shutdown_owner_task: ClassVar[asyncio.Task[Any] | None] = None

    @classmethod
    def register(cls, hook: ShutdownHook) -> None:
        """Register a compatibility cleanup hook in LIFO order."""

        if hook not in cls._hooks:
            cls._hooks.append(hook)
            logger.debug(
                "Registered shutdown hook: %s",
                getattr(hook, "__name__", str(hook)),
            )

    @classmethod
    def setup_signals(cls) -> None:
        """Bind OS signals without allowing inner services to replace them."""

        loop = asyncio.get_running_loop()
        if cls._shutdown_event is None:
            cls._shutdown_event = asyncio.Event()

        def _request_signal_shutdown(sig: signal.Signals) -> None:
            # Latch synchronously in the signal callback. Scheduling the cleanup
            # coroutine first leaves a window where recovery can create work.
            request_shutdown(f"signal:{sig.name}")
            shutdown_coro = cls.trigger_shutdown(sig)
            try:
                get_task_tracker().create_task(
                    shutdown_coro,
                    name=f"graceful_shutdown.{sig.name}",
                    allow_during_shutdown=True,
                )
            except TypeError:
                # Compatibility for minimal tracker implementations used by
                # embedded hosts; the production tracker accepts the flag.
                get_task_tracker().create_task(
                    shutdown_coro,
                    name=f"graceful_shutdown.{sig.name}",
                )

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    partial(_request_signal_shutdown, sig),
                )
            except NotImplementedError as exc:
                logger.debug("Signal handler registration unsupported for %s: %s", sig, exc)
            except (RuntimeError, AttributeError, TypeError) as exc:
                logger.error("Signal handler registration failed for %s: %s", sig, exc)
                raise

    @classmethod
    async def trigger_shutdown(cls, sig: signal.Signals | str | None = None) -> None:
        """Run compatibility hooks, canonical phases, and container teardown once."""

        if cls._shutdown_event is None:
            cls._shutdown_event = asyncio.Event()
        current_task = asyncio.current_task()
        if cls._is_shutting_down:
            if current_task is cls._shutdown_owner_task:
                return
            try:
                # Bounded: if the owning shutdown task wedges, a secondary
                # trigger must not wait forever — shutdown is already in
                # motion; proceeding after the grace window is harmless.
                await asyncio.wait_for(cls._shutdown_event.wait(), timeout=60.0)
            except TimeoutError:
                logger.warning(
                    "Secondary shutdown trigger stopped waiting after 60s; "
                    "primary shutdown task may be wedged."
                )
            return

        shutdown_scope_token = begin_shutdown_task_creation_scope()
        cls._is_shutting_down = True
        cls._shutdown_owner_task = current_task
        reason = f"signal:{getattr(sig, 'name', sig)}" if sig else "graceful_shutdown"
        request_shutdown(reason)

        prefix = f"Received signal {sig}: " if sig else ""
        logger.info("Shutdown: %sinitiating graceful teardown", prefix)
        coordinator_report: ShutdownReport | dict[str, object] | None = None
        container_report: dict[str, object] | None = None
        try:
            compatibility_deadline = time.monotonic() + 8.0
            while cls._hooks:
                hook = cls._hooks.pop()
                remaining = compatibility_deadline - time.monotonic()
                if remaining <= 0:
                    skipped = 1 + len(cls._hooks)
                    cls._hooks.clear()
                    exc = TimeoutError(
                        f"compatibility shutdown hook budget exhausted; skipped={skipped}"
                    )
                    record_degradation("graceful_shutdown", exc)
                    logger.error("%s", exc)
                    break
                try:
                    if inspect.iscoroutinefunction(hook):
                        result = hook()
                    else:
                        result = await run_sync_shutdown_callable(
                            cast(Callable[[], Any], hook),
                            timeout_s=min(8.0, remaining),
                            name=f"compatibility:{getattr(hook, '__name__', 'hook')}",
                        )
                    if inspect.isawaitable(result):
                        remaining = max(0.05, compatibility_deadline - time.monotonic())
                        await asyncio.wait_for(result, timeout=min(8.0, remaining))
                    logger.info("Shutdown compatibility hook completed: %s", hook)
                except Exception as exc:  # noqa: BLE001 - final teardown boundary
                    record_degradation("graceful_shutdown", exc)
                    logger.error("Shutdown compatibility hook failed: %s", exc)

            try:
                coordinator_report = await get_shutdown_coordinator().shutdown()
                if not coordinator_report.clean:
                    logger.warning(
                        "Shutdown coordinator completed with failures: phases=%s handlers=%s",
                        coordinator_report.failed_phases,
                        sorted(coordinator_report.handler_failures),
                    )
            except Exception as exc:  # noqa: BLE001 - final teardown boundary
                record_degradation("graceful_shutdown", exc)
                logger.error("Canonical shutdown coordinator failed: %s", exc)

            try:
                from core.container import get_container
                from core.runtime.lifecycle_probe import hold_shutdown_probe_async

                logger.info("Shutdown: container teardown started")
                await hold_shutdown_probe_async("container")
                container_result = await get_container().shutdown()
                container_report = (
                    dict(container_result)
                    if isinstance(container_result, dict)
                    else {"clean": True, "legacy_result": repr(container_result)}
                )
                logger.info(
                    "Shutdown: container teardown completed (clean=%s)",
                    container_report.get("clean"),
                )
            except Exception as exc:  # noqa: BLE001 - final teardown boundary
                record_degradation("graceful_shutdown", exc)
                logger.error("Container shutdown failed: %s", exc)

            runtime_hygiene_report = None
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                runtime_hygiene = get_runtime_hygiene()
                runtime_hygiene_report = runtime_hygiene.get_shutdown_report()
                if runtime_hygiene_report is None:
                    logger.info(
                        "Runtime hygiene had no coordinator report; running the final sweep inline."
                    )
                    runtime_hygiene_report = await runtime_hygiene.stop()
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("graceful_shutdown", exc)
                logger.error("Runtime hygiene shutdown report unavailable: %s", exc)
            try:
                verdict = dict(
                    coordinator_report=coordinator_report,
                    container_report=container_report,
                    runtime_hygiene_report=runtime_hygiene_report,
                    stage="graceful_shutdown_complete",
                    final=True,
                )
                # Off the loop: the verdict is written atomically, with an
                # fsync, and every task still finishing waited on the disk
                # for it (lockdep, every shutdown). Inline only where the
                # executor has already gone, which late in a shutdown it can.
                try:
                    await asyncio.to_thread(lambda: publish_shutdown_verdict(**verdict))
                except RuntimeError:
                    publish_shutdown_verdict(**verdict)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("graceful_shutdown", exc)
                logger.error("Final shutdown verdict persistence failed: %s", exc)

            logger.info("All Aura core services terminated")
            _arm_exit_stall_dump()
        finally:
            cls._shutdown_owner_task = None
            cls._shutdown_event.set()
            end_shutdown_task_creation_scope(shutdown_scope_token)

    @classmethod
    async def wait_for_shutdown(cls) -> None:
        """Block until the one canonical shutdown execution has completed."""

        if cls._shutdown_event is None:
            cls.setup_signals()
        shutdown_event = cls._shutdown_event
        if shutdown_event is None:  # pragma: no cover - setup_signals initializes it
            raise RuntimeError("shutdown event initialization failed")
        await shutdown_event.wait()


#: How long the process may take from "every service terminated" to the root
#: finalizer before whatever is holding it is written down. Measured
#: 2026-09-15: 14 seconds on one shutdown, 119 on the next, with nothing in
#: the log for the difference. asyncio.run joins the default executor on the
#: way out (up to 300s in 3.12), so a worker thread still busy at that
#: moment is a silent two-minute exit.
EXIT_STALL_DUMP_AFTER_S = 10.0

#: Frames that mean a thread is waiting for work, not doing it.
_IDLE_MARKERS = (
    'threading.py", line 355, in wait',
    "waiter.acquire()",
    'concurrent/futures/thread.py", line 90, in _worker',
    "work_queue.get(block=True)",
    'queue.py", line',
    'selectors.py", line',
)


def _arm_exit_stall_dump(after_s: float = EXIT_STALL_DUMP_AFTER_S) -> None:
    """Name the threads still working if the process has not exited in time."""
    import threading

    def _dump() -> None:
        import sys
        import traceback

        frames = sys._current_frames()
        busy: list[str] = []
        for thread in threading.enumerate():
            if thread is threading.main_thread() or thread.daemon and thread.ident not in frames:
                continue
            frame = frames.get(thread.ident or -1)
            if frame is None:
                continue
            stack = "".join(traceback.format_stack(frame)[-6:])
            if any(marker in stack for marker in _IDLE_MARKERS):
                continue
            busy.append(f"[{thread.name} daemon={thread.daemon}]\n{stack}")
        if busy:
            logger.warning(
                "Exit stalled %.0fs after every service terminated; %d thread(s) still "
                "working:\n%s",
                after_s,
                len(busy),
                "\n".join(busy)[:12000],
            )
        else:
            logger.warning(
                "Exit stalled %.0fs after every service terminated; no thread is "
                "working, so the wait is in the interpreter's own teardown.",
                after_s,
            )

    timer = threading.Timer(after_s, _dump)
    timer.daemon = True
    timer.name = "exit-stall-dump"
    timer.start()


def register_shutdown_hook(hook: ShutdownHook) -> None:
    GracefulShutdown.register(hook)
