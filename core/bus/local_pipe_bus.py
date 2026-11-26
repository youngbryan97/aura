import asyncio
import json
import logging
import multiprocessing
import multiprocessing.connection
import os
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from core.bus.shared_mem_bus import SharedMemoryTransport
from core.runtime.errors import record_degradation
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.LocalPipeBus")

_BUS_HANDLER_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    asyncio.InvalidStateError,
)


def _is_shutdown_commit_request(msg_type: str, payload: Any) -> bool:
    return (
        str(msg_type or "") == "commit"
        and isinstance(payload, dict)
        and str(payload.get("cause") or "").lower() == "shutdown"
    )


@dataclass(frozen=True)
class SendOutcome:
    """Whether a fire-and-forget send actually reached the transport.

    CP126 (critical): six drop paths in ``send`` returned None, exactly like
    a successful write, so "the await completed" meant neither admission nor
    delivery and drop counters lived only inside the bus.

    ``delivered`` is the honest answer; ``reason`` names which path was
    taken, because "the bus is not running" and "the writer is locked"
    demand different responses from a caller.
    """

    delivered: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.delivered


def _is_connection_pair(connection: Any) -> bool:
    return isinstance(connection, tuple) and len(connection) == 2


class LocalPipeBus:
    """
    High-performance, zero-config intra-process communication using multiprocessing.Pipe.
    Refactored to use unidirectional pipe pairs to eliminate bidirectional deadlocks.
    ZENITH LOCKDOWN: Dedicated ThreadPoolExecutor for Pipe I/O to prevent starvation.
    """
    _LIVE_BUSES: "weakref.WeakSet[LocalPipeBus]" = weakref.WeakSet()
    _SHM_OFFLOAD_THRESHOLD_BYTES = 8 * 1024
    _SHM_SEGMENT_RETENTION_SECONDS = 20.0
    _DEFAULT_MAX_PENDING_REQUESTS = 64
    # Generous safety-net deadline for a single async handler so a
    # non-terminating coroutine cannot block every later response and control
    # message forever. Deliberately large (env-overridable) — it exists to catch
    # a hung coroutine, not to bound legitimately slow work.
    try:
        _HANDLER_EXEC_TIMEOUT_S = float(os.getenv("AURA_BUS_HANDLER_TIMEOUT_S", "300") or 300)
    except (TypeError, ValueError):
        _HANDLER_EXEC_TIMEOUT_S = 300.0


    @classmethod
    def shutdown_executor(cls) -> None:
        for bus in list(cls._LIVE_BUSES):
            bus._shutdown_executor()

    @classmethod
    def cleanup_owned_shared_memory(cls) -> int:
        """Release every retained outbound segment at the process barrier."""
        cleaned = 0
        for bus in list(cls._LIVE_BUSES):
            cleaned += len(bus._outbound_shm_segments)
            bus._cleanup_expired_shm_segments(force=True)
        return cleaned

    def __init__(self, is_child: bool = False, 
                 read_conn: multiprocessing.connection.Connection | None = None,
                 write_conn: multiprocessing.connection.Connection | None = None,
                 start_reader: bool = True,
                 connection: Any = None):
        self.is_child = is_child
        self.start_reader = start_reader
        
        if _is_connection_pair(connection):
            self.read_conn, self.write_conn = connection
        elif connection is not None:
            raise ValueError(
                "LocalPipeBus requires an explicit (read_conn, write_conn) transport pair; "
                "shared single-connection compatibility is no longer supported."
            )
        elif read_conn is not None and write_conn is not None:
            self.read_conn = read_conn
            self.write_conn = write_conn
        else:
            # Create two unidirectional pipes
            # pipe1: Parent Reads, Child Writes
            p_read, c_write = multiprocessing.Pipe(duplex=False)
            # pipe2: Child Reads, Parent Writes
            c_read, p_write = multiprocessing.Pipe(duplex=False)
            
            if is_child:
                self.read_conn = c_read
                self.write_conn = c_write
            else:
                self.read_conn = p_read
                self.write_conn = p_write

        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._reader_task: asyncio.Task | None = None
        self._dispatcher_task: asyncio.Task | None = None
        self._dispatch_queue: asyncio.Queue | None = None
        self._handlers: dict[str, Callable] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._is_running = False
        self._activity_callback: Callable[[], None] | None = None
        self._pipe_broken = False
        self._degraded = False
        self._last_error: str | None = None
        self._last_error_at = 0.0
        self._write_timeout_count = 0
        self._write_suppressed_until = 0.0
        self._write_lock: asyncio.Lock | None = None
        self._write_backpressure_drops = 0
        self._max_pending_requests = self._pending_request_limit()
        self._outbound_shm_segments: dict[str, tuple[SharedMemoryTransport, float]] = {}
        self._LIVE_BUSES.add(self)

    @classmethod
    def _pending_request_limit(cls) -> int:
        try:
            value = int(os.getenv("AURA_PIPE_MAX_PENDING_REQUESTS", str(cls._DEFAULT_MAX_PENDING_REQUESTS)))
        except (TypeError, ValueError):
            value = cls._DEFAULT_MAX_PENDING_REQUESTS
        return min(1024, max(1, value))

    def _fire_and_forget_write_timeout_s(self) -> float:
        try:
            value = float(os.getenv("AURA_PIPE_FF_WRITE_TIMEOUT_S", "0.5") or 0.5)
        except (TypeError, ValueError):
            value = 0.5
        return min(5.0, max(0.05, value))

    def _pipe_suppression_window_s(self) -> float:
        try:
            value = float(os.getenv("AURA_PIPE_SUPPRESS_AFTER_TIMEOUT_S", "30.0") or 30.0)
        except (TypeError, ValueError):
            value = 30.0
        return min(300.0, max(1.0, value))

    def _get_write_lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    @staticmethod
    def _task_status(task: asyncio.Task | None) -> dict[str, Any]:
        if task is None:
            return {
                "present": False,
                "done": None,
                "cancelled": None,
                "failed": None,
                "exception": None,
            }
        done = bool(task.done())
        cancelled = bool(task.cancelled()) if done else False
        exception: str | None = None
        failed = False
        if done and not cancelled:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                cancelled = True
                exc = None
            except (RuntimeError, AttributeError, TypeError, ValueError) as status_exc:
                exc = status_exc
            if exc is not None:
                failed = True
                exception = f"{type(exc).__name__}: {exc}"
        return {
            "present": True,
            "done": done,
            "cancelled": cancelled,
            "failed": failed,
            "exception": exception,
        }

    @classmethod
    def _task_alive(cls, task: asyncio.Task | None) -> bool:
        status = cls._task_status(task)
        return bool(status["present"]) and not bool(status["done"])

    def _background_tasks_alive(self) -> bool:
        if not self.start_reader:
            return True
        return (
            self._dispatch_queue is not None
            and self._task_alive(self._reader_task)
            and self._task_alive(self._dispatcher_task)
        )

    def _dispatch_queue_saturated(self) -> bool:
        queue = self._dispatch_queue
        if queue is None:
            return False
        maxsize = int(getattr(queue, "maxsize", 0) or 0)
        return maxsize > 0 and queue.qsize() >= maxsize

    def _pending_requests_saturated(self) -> bool:
        return len(self._pending_requests) >= self._max_pending_requests

    def is_alive(self) -> bool:
        """Return true only when the pipe transport and its workers are live."""

        read_closed = bool(getattr(self.read_conn, "closed", False))
        write_closed = bool(getattr(self.write_conn, "closed", False))
        loop_closed = bool(self._loop is not None and self._loop.is_closed())
        executor_shutdown = bool(
            self._executor is not None and getattr(self._executor, "_shutdown", False)
        )
        return bool(
            self._is_running
            and not read_closed
            and not write_closed
            and not self._pipe_broken
            and not self._degraded
            and not loop_closed
            and not executor_shutdown
            and self._background_tasks_alive()
            and not self._dispatch_queue_saturated()
            and not self._pending_requests_saturated()
        )

    def get_status(self) -> dict[str, Any]:
        """Return a machine-readable health report for this transport."""

        now = time.monotonic()
        queue_size = self._dispatch_queue.qsize() if self._dispatch_queue is not None else 0
        queue_maxsize = int(getattr(self._dispatch_queue, "maxsize", 0) or 0)
        loop_closed = bool(self._loop is not None and self._loop.is_closed())
        executor_shutdown = bool(
            self._executor is not None and getattr(self._executor, "_shutdown", False)
        )
        return {
            "alive": self.is_alive(),
            "running": bool(self._is_running),
            "start_reader": bool(self.start_reader),
            "is_child": bool(self.is_child),
            "background_tasks_alive": self._background_tasks_alive(),
            "reader_task": self._task_status(self._reader_task),
            "dispatcher_task": self._task_status(self._dispatcher_task),
            "loop_closed": loop_closed,
            "executor_shutdown": executor_shutdown,
            "dispatch_queue_size": queue_size,
            "dispatch_queue_maxsize": queue_maxsize,
            "dispatch_queue_saturated": self._dispatch_queue_saturated(),
            "pipe_broken": bool(self._pipe_broken),
            "degraded": bool(self._degraded),
            "read_closed": bool(getattr(self.read_conn, "closed", False)),
            "write_closed": bool(getattr(self.write_conn, "closed", False)),
            "pending_requests": len(self._pending_requests),
            "pending_request_limit": int(self._max_pending_requests),
            "pending_requests_saturated": self._pending_requests_saturated(),
            "write_timeout_count": int(self._write_timeout_count),
            "write_backpressure_drops": int(self._write_backpressure_drops),
            "write_suppressed_for_s": round(max(0.0, self._write_suppressed_until - now), 3),
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
        }

    def _clear_transport_degradation(self) -> None:
        self._degraded = False
        self._last_error = None
        self._last_error_at = 0.0

    def _mark_transport_degraded(self, exc: BaseException, action: str) -> None:
        self._degraded = True
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._last_error_at = time.time()
        record_degradation(
            "local_pipe_bus",
            exc,
            severity="degraded",
            action=action,
            enforce_failure_policy=False,
            extra=self.get_status(),
        )

    def _record_transport_warning(self, exc: BaseException, action: str) -> None:
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._last_error_at = time.time()

    def _response_write_timeout_s(self) -> float:
        try:
            value = float(os.getenv("AURA_PIPE_RESPONSE_WRITE_TIMEOUT_S", "3.0") or 3.0)
        except (TypeError, ValueError):
            value = 3.0
        return min(30.0, max(0.25, value))

    async def _write_raw_message(
        self,
        raw_msg: str,
        *,
        timeout_s: float,
        context: str,
        lock_timeout_s: float | None = None,
    ) -> None:
        """Write one raw pipe message with bounded lock acquisition and send time.

        All request, response, and fire-and-forget paths share this lock so
        multiprocessing.Pipe never receives concurrent writes from heartbeat,
        response, and foreground request threads. Concurrent writes can corrupt
        frames or block the dispatch loop, which is worse than a bounded
        timeout because it starves every subsequent request.
        """
        if self.write_conn.closed or getattr(self, "_pipe_broken", False):
            raise BrokenPipeError("Connection is closed")

        loop = asyncio.get_running_loop()
        write_lock = self._get_write_lock()
        lock_timeout = (
            float(lock_timeout_s)
            if lock_timeout_s is not None
            else min(max(float(timeout_s) * 0.25, 0.25), 2.0)
        )
        await asyncio.wait_for(write_lock.acquire(), timeout=lock_timeout)
        try:
            if self.write_conn.closed or getattr(self, "_pipe_broken", False):
                raise BrokenPipeError("Connection is closed")
            await asyncio.wait_for(
                loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_msg),
                timeout=float(timeout_s),
            )
        except TimeoutError as exc:
            if context.startswith("send:"):
                logger.debug("Pipe write timed out in %s after %.1fs.", context, timeout_s)
            else:
                logger.warning("Pipe write timed out in %s after %.1fs.", context, timeout_s)
            raise TimeoutError(f"pipe write timed out in {context}") from exc
        finally:
            write_lock.release()

    def _should_log_backpressure_drop(self) -> bool:
        return self._write_backpressure_drops in {1, 10, 50, 100} or (
            self._write_backpressure_drops > 0
            and self._write_backpressure_drops % 250 == 0
        )

    def _get_executor(self) -> ThreadPoolExecutor:
        executor = self._executor
        # If the cached executor has already been shut down (e.g. a stray
        # LocalPipeBus.shutdown_executor() classmethod call from a sibling
        # subsystem), don't propagate the failure to every state commit
        # forever — recreate it. Bus shutdown is supposed to coincide with
        # process shutdown; if the bus is still being asked to do work, the
        # process is still alive and the executor must come back.
        if executor is not None:
            shutdown_flag = getattr(executor, "_shutdown", False)
            if shutdown_flag and self._is_running:
                logger.warning(
                    "LocalPipeBus: cached executor was shut down while bus is still running — recreating."
                )
                executor = None
                self._executor = None
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="AuraPipeIO",
            )
            self._executor = executor
        return executor

    def _shutdown_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        try:
            return asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "LocalPipeBus requires a running event loop. Start it from async boot/runtime code."
            ) from exc

    async def _run_on_transport_loop(
        self,
        operation: str,
        factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Bridge calls back onto the loop that owns this transport."""
        current_loop = asyncio.get_running_loop()
        target_loop = self._loop
        if target_loop is None or target_loop is current_loop:
            return await factory()
        if target_loop.is_closed() or not target_loop.is_running():
            raise RuntimeError(
                f"LocalPipeBus cannot {operation}: transport loop is unavailable"
            )
        bridged = asyncio.run_coroutine_threadsafe(factory(), target_loop)
        return await asyncio.wrap_future(bridged)

    def _safe_close_connection(self, conn: multiprocessing.connection.Connection | None) -> None:
        if conn is None:
            return
        try:
            conn.close()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('local_pipe_bus', exc)
            logger.debug("LocalPipeBus: connection close skipped: %s", exc)

    def start(self):
        """Start the background reader task."""
        if self._is_running and not self.start_reader:
            return
        if self._is_running and self._background_tasks_alive():
            return
        
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._is_running = True
        if self.start_reader:
            reader_alive = self._task_alive(self._reader_task)
            dispatcher_alive = self._task_alive(self._dispatcher_task)
            if self._reader_task is not None or self._dispatcher_task is not None:
                record_degradation(
                    "local_pipe_bus",
                    RuntimeError("LocalPipeBus background worker restart required"),
                    severity="warning",
                    action="restarting_dead_background_workers",
                    enforce_failure_policy=False,
                    extra={
                        "reader_task": self._task_status(self._reader_task),
                        "dispatcher_task": self._task_status(self._dispatcher_task),
                    },
                )
            if self._dispatch_queue is None:
                self._dispatch_queue = asyncio.Queue(maxsize=256)
            tracker = get_task_tracker()
            if not dispatcher_alive:
                self._dispatcher_task = tracker.create_task(
                    self._dispatch_loop(),
                    name="local_pipe_bus.dispatch",
                )
            if not reader_alive:
                self._reader_task = tracker.create_task(
                    self._read_loop(),
                    name="local_pipe_bus.read",
                )
            logger.info("LocalPipeBus reader ACTIVE (Child: %s)", self.is_child)
        else:
            logger.info("LocalPipeBus ACTIVE (Manual Polling mode)")

    async def stop(self):
        """Stop the reader task."""
        self._is_running = False
        self._cancel_pending_requests(cancel=True)
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await asyncio.wait_for(self._reader_task, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug("LocalPipeBus reader task cancelled during shutdown")
            except TimeoutError as e:
                record_degradation(
                    "local_pipe_bus",
                    e,
                    action="reader task did not stop before shutdown timeout",
                )
                logger.warning("LocalPipeBus reader did not stop before shutdown timeout")
            except (RuntimeError, AttributeError) as e:
                record_degradation('local_pipe_bus', e)
                logger.error("LocalPipeBus: Error during stop: %s", e)
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug("LocalPipeBus dispatcher task cancelled during shutdown")
            except TimeoutError as e:
                record_degradation(
                    "local_pipe_bus",
                    e,
                    action="dispatcher task did not stop before shutdown timeout",
                )
                logger.warning("LocalPipeBus dispatcher did not stop before shutdown timeout")
            except (RuntimeError, AttributeError) as e:
                record_degradation('local_pipe_bus', e)
                logger.error("LocalPipeBus: Dispatcher stop error: %s", e)
        self._cleanup_expired_shm_segments(force=True)
        if self.read_conn is self.write_conn:
            self._safe_close_connection(self.read_conn)
        else:
            self._safe_close_connection(self.read_conn)
            self._safe_close_connection(self.write_conn)
        executor, self._executor = self._executor, None
        if executor is not None:
            await run_sync_shutdown_callable(
                lambda: executor.shutdown(wait=True, cancel_futures=True),
                timeout_s=2.0,
                name="local-pipe-bus-executor",
            )

    def register_handler(self, msg_type: str, handler: Callable):
        """Register a handler for a specific message type."""
        self._handlers[msg_type] = handler

    def set_activity_callback(self, callback: Callable[[], None]):
        """Register a lightweight callback for inbound transport activity."""
        self._activity_callback = callback

    def _cleanup_expired_shm_segments(self, *, force: bool = False):
        if not self._outbound_shm_segments:
            return

        now = time.monotonic()
        expired = [
            name
            for name, (_, expires_at) in list(self._outbound_shm_segments.items())
            if force or expires_at <= now
        ]
        for name in expired:
            shm, _ = self._outbound_shm_segments.pop(name, (None, 0.0))
            if shm is None:
                continue
            try:
                shm.close()
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('local_pipe_bus', e)
                logger.debug("LocalPipeBus: SHM cleanup failed for %s: %s", name, e)

    def _retain_outbound_shm(self, shm: SharedMemoryTransport):
        self._cleanup_expired_shm_segments()
        self._outbound_shm_segments[shm.name] = (
            shm,
            time.monotonic() + self._SHM_SEGMENT_RETENTION_SECONDS,
        )

    async def _prepare_payload_for_transport(self, payload: Any) -> Any:
        """Serialize large payloads off-loop and offload them to SHM when needed."""
        if not isinstance(payload, (dict, list)):
            return payload

        serialized_payload = await asyncio.to_thread(json.dumps, payload)
        payload_bytes = serialized_payload.encode("utf-8")
        if len(payload_bytes) <= self._SHM_OFFLOAD_THRESHOLD_BYTES:
            return payload

        shm_name = f"shm_msg_{uuid.uuid4().hex[:8]}"
        shm = None
        try:
            shm = SharedMemoryTransport(shm_name, size=len(payload_bytes) + 1024)
            await shm.create()
            await asyncio.to_thread(shm.write_serialized, serialized_payload)
            self._retain_outbound_shm(shm)
            logger.debug("[SHM] Offloaded payload: %s (%d bytes)", shm_name, len(payload_bytes))
            return {"__shm__": shm_name}
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('local_pipe_bus', e)
            if shm is not None:
                try:
                    shm.close()
                except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    record_degradation('local_pipe_bus', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
            logger.warning("SHM offload failed, falling back to Pipe: %s", e)
            return payload

    async def send(
        self, msg_type: str, payload: Any, trace_id: str | None = None,
    ) -> "SendOutcome":
        """Fire-and-forget send that tells the caller what happened.

        CP126 (critical): "The public send API has no delivery result.
        Not-running, closed, broken, suppression-window, and locked-writer
        conditions all return normally without sending; RuntimeError is also
        swallowed. A normal await completion therefore means neither
        admission nor delivery."

        Six distinct drop paths all returned None, exactly like a successful
        write. Fire-and-forget is the right SEMANTIC — a caller should not
        block on the bus — but that is a statement about waiting, not about
        lying. A caller that wants to know can now ask; one that does not is
        unaffected, since the return value was previously unused.
        """
        try:
            outcome = await self._run_on_transport_loop(
                "send",
                lambda: self._send_local(msg_type, payload, trace_id=trace_id),
            )
        except RuntimeError as e:
            if self._is_running:
                logger.error("Unexpected error in bus send: %s", e)
            return SendOutcome(False, f"transport_loop_error:{type(e).__name__}")
        return outcome if isinstance(outcome, SendOutcome) else SendOutcome(True, "")

    async def _send_local(
        self, msg_type: str, payload: Any, trace_id: str | None = None,
    ) -> "SendOutcome":
        """Send a fire-and-forget message, reporting admission and delivery."""
        trace_id = trace_id or str(uuid.uuid4())
        msg = {
            "type": msg_type,
            "payload": payload,
            "trace_id": trace_id
        }
        try:
            # Pre-flight check to avoid BrokenPipeError hangs
            if (
                not self._is_running
                or
                self.write_conn.closed
                or getattr(self, '_pipe_broken', False)
                or time.monotonic() < getattr(self, "_write_suppressed_until", 0.0)
            ):
                # Was a bare `return` — indistinguishable from delivery.
                if not self._is_running:
                    return SendOutcome(False, "bus_not_running")
                if self.write_conn.closed:
                    return SendOutcome(False, "connection_closed")
                if getattr(self, "_pipe_broken", False):
                    return SendOutcome(False, "pipe_broken")
                return SendOutcome(False, "write_suppression_window")

            write_lock = self._get_write_lock()
            if write_lock.locked():
                self._write_backpressure_drops += 1
                if self._should_log_backpressure_drop():
                    self._record_transport_warning(
                        TimeoutError(f"fire-and-forget pipe write blocked for {msg_type}"),
                        "fire-and-forget send dropped by write backpressure",
                    )
                    logger.debug(
                        "Pipe write backpressure: dropped fire-and-forget message "
                        "(drops=%d, msg_type=%s).",
                        self._write_backpressure_drops,
                        msg_type,
                    )
                return SendOutcome(False, "write_backpressure")

            msg["payload"] = await self._prepare_payload_for_transport(payload)
            raw_msg = await asyncio.to_thread(json.dumps, msg)
            timeout_s = self._fire_and_forget_write_timeout_s()
            await self._write_raw_message(
                raw_msg,
                timeout_s=timeout_s,
                context=f"send:{msg_type}",
                lock_timeout_s=0.05,
            )
            self._write_timeout_count = 0
            self._write_backpressure_drops = 0
            self._clear_transport_degradation()
            return SendOutcome(True, "")
        except TimeoutError:
            self._write_timeout_count += 1
            suppress_for_s = self._pipe_suppression_window_s()
            self._write_suppressed_until = time.monotonic() + suppress_for_s
            self._record_transport_warning(
                TimeoutError(f"fire-and-forget pipe write timed out for {msg_type}"),
                (
                    f"pipe write timed out; suppressed fire-and-forget writes "
                    f"for {suppress_for_s:.1f}s"
                ),
            )
            logger.debug(
                "Pipe write TIMEOUT (%.1fs) — suppressing fire-and-forget writes "
                "for %.1fs (streak=%d).",
                self._fire_and_forget_write_timeout_s(),
                suppress_for_s,
                self._write_timeout_count,
            )
            if self._write_timeout_count >= 3:
                logger.error(
                    "Pipe write repeatedly timed out; transport is saturated."
                )
                try:
                    from core.resilience.omni_tracer import write_trace

                    write_trace(
                        "local_pipe_bus",
                        "PipeWriteSaturation",
                        (
                            f"write timeout streak={self._write_timeout_count}; "
                            f"suppressing writes for {suppress_for_s:.1f}s"
                        ),
                    )
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    logger.debug("Suppressed %s in core.bus.local_pipe_bus: %s", type(_exc).__name__, _exc)
            return SendOutcome(False, "write_timeout")
        except (BrokenPipeError, EOFError, OSError, ConnectionResetError) as e:
            if not getattr(self, '_pipe_broken', False):
                self._pipe_broken = True
                if self._is_running:
                    self._mark_transport_degraded(
                        e,
                        "pipe closed during fire-and-forget send; transport marked unhealthy",
                    )
                logger.info("Bus pipe closed (normal shutdown): %s", str(e)[:60])
            self._safe_close_connection(self.write_conn)
            return SendOutcome(False, f"pipe_closed:{type(e).__name__}")
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('local_pipe_bus', e)
            if self._is_running:
                logger.error("Unexpected error in bus send: %s", e)
            return SendOutcome(False, f"send_error:{type(e).__name__}")

    async def request(  # noqa: ASYNC109 - timeout is part of the public bus API.
        self,
        msg_type: str,
        payload: Any,
        timeout: float = 5.0,  # noqa: ASYNC109
    ) -> Any:
        return await self._run_on_transport_loop(
            "request",
            lambda: self._request_local(msg_type, payload, timeout=timeout),
        )

    async def _request_local(  # noqa: ASYNC109 - mirrors public request timeout API.
        self,
        msg_type: str,
        payload: Any,
        timeout: float = 5.0,  # noqa: ASYNC109
    ) -> Any:
        """Send a request and wait for a response."""
        if self.write_conn.closed or getattr(self, "_pipe_broken", False):
            raise BrokenPipeError("Connection is closed")
        pending_count = len(self._pending_requests)
        if pending_count >= self._max_pending_requests:
            admission_error = TimeoutError(
                f"LocalPipeBus pending request limit reached ({pending_count}/{self._max_pending_requests})"
            )
            self._mark_transport_degraded(
                admission_error,
                "request admission blocked by pending-request backpressure",
            )
            logger.warning(
                "LocalPipeBus request admission blocked: pending=%d max=%d msg_type=%s",
                pending_count,
                self._max_pending_requests,
                msg_type,
            )
            raise admission_error

        request_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[request_id] = future

        msg = {
            "type": msg_type,
            "payload": payload,
            "request_id": request_id,
            "trace_id": trace_id,
            "is_request": True
        }
        
        try:
            # Hardened connection check
            if self.write_conn.closed:
                 raise BrokenPipeError("Connection is closed")

            msg["payload"] = await self._prepare_payload_for_transport(payload)
            raw_msg = await asyncio.to_thread(json.dumps, msg)
            logger.debug("Sending request: %s (ID: %s)", msg_type, request_id)
            
            # ZENITH LOCKDOWN: Use isolated pipe executor and hard 10s timeout on write
            await self._write_raw_message(
                raw_msg,
                timeout_s=min(timeout, 10.0),
                context=f"request:{msg_type}",
            )
            result = await asyncio.wait_for(future, timeout=timeout)
            self._clear_transport_degradation()
            return result
        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            self._mark_transport_degraded(
                TimeoutError(f"pipe request timed out for {msg_type}"),
                f"pipe request timed out for {msg_type}",
            )
            logger.warning("Bus request timed out: %s", msg_type)
            raise
        except (BrokenPipeError, EOFError, OSError) as e:
            # [GENESIS FIX] Immediately resolve the specific future to avoid hanging the caller
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if not future.done():
                    future.cancel()
            
            quiet_shutdown_commit = _is_shutdown_commit_request(msg_type, payload)
            if not getattr(self, '_pipe_broken', False):
                self._pipe_broken = True
                if quiet_shutdown_commit:
                    logger.debug("Bus closed during shutdown commit; caller will handle snapshot fallback.")
                else:
                    self._mark_transport_degraded(
                        e,
                        "pipe request failed with broken transport",
                    )
                    logger.warning("Bus request failed (Broken Pipe): %s", e)
            
            try:
                self._safe_close_connection(self.write_conn)
            except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
                record_degradation('local_pipe_bus', _e)
                logger.debug("LocalPipeBus: Secondary error during request-failure close: %s", _e)
            raise

    async def _read_loop(self):
        """Internal reader loop using unidirectional read_conn."""
        while self._is_running:
            try:
                # ZENITH LOCKDOWN: Use isolated pipe executor for blocking recv()
                msg = await self.loop.run_in_executor(self._get_executor(), self.read_conn.recv)
                
                # Always handle potential JSON strings from other processes
                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse bus message: %s...", msg[:100])
                        continue

                if not msg or not isinstance(msg, dict):
                    continue

                # Reader resilience: a malformed envelope's field/hash operations
                # must not escape into the transport-level handlers below and
                # kill the reader while _is_running stays True (which would hang
                # every request until timeout). Drop the bad frame; keep looping.
                try:
                    await self._process_message(msg)
                except (KeyError, TypeError, ValueError, AttributeError,
                        IndexError, LookupError) as frame_err:
                    record_degradation('local_pipe_bus', frame_err)
                    logger.error(
                        "Dropped malformed bus frame (type=%r): %s",
                        msg.get("type") if isinstance(msg, dict) else None,
                        frame_err,
                    )
                    continue

            except EOFError:
                logger.info("Bus connection closed by peer.")
                self._is_running = False
                self._cancel_pending_requests(cancel=True)
                break
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.error("Bus read error: %s", e)
                self._is_running = False
                self._cancel_pending_requests(e)
                break
            except OSError as e:
                if not self._is_running or "handle is closed" in str(e).lower():
                    logger.info("Bus connection closed during shutdown.")
                    self._is_running = False
                    self._cancel_pending_requests(cancel=True)
                    break
                record_degradation('local_pipe_bus', e)
                logger.exception("Error in Bus read loop: %s", e)
                
                # Component Hardening: Active Self-Repair Invocation
                try:
                    from core.runtime.self_healing import get_healer
                    get_healer().schedule_deep_repair(
                        "core/bus/local_pipe_bus.py",
                        reason="read_loop_exception",
                        metadata={"error": str(e)}
                    )
                except (ImportError, AttributeError, RuntimeError) as _heal_e:
                    logger.debug("Deep repair scheduling failed: %s", _heal_e)

                await asyncio.sleep(1.0)

    async def _process_message(self, msg: dict) -> None:
        """Interpret one inbound envelope (SHM deref, response routing, dispatch).

        Extracted from _read_loop so a malformed frame's field/hash operations
        are contained by the caller's reader-resilience guard instead of killing
        the read loop. Field access is defensive (``.get`` and a hashable-type
        check) so a forged/partial envelope raises a caught, drop-the-frame
        error rather than an unbounded one.
        """
        if self._activity_callback:
            try:
                self._activity_callback()
            except (RuntimeError, AttributeError, TypeError, ValueError) as callback_err:
                record_degradation('local_pipe_bus', callback_err)
                logger.debug("LocalPipeBus activity callback failed: %s", callback_err)

        # SHM De-referencing
        payload = msg.get("payload")
        if isinstance(payload, dict) and "__shm__" in payload:
            shm_name = payload["__shm__"]
            try:
                shm = SharedMemoryTransport(shm_name)
                await asyncio.wait_for(shm.attach(), timeout=2.0)
                msg["payload"] = await shm.read()
                # Detach but don't unlink yet (let owner clean up or use a policy).
                shm.close()
                logger.debug("Resolved SHM payload: %s", shm_name)
            except (RuntimeError, TimeoutError, AttributeError) as e:
                record_degradation('local_pipe_bus', e)
                logger.error("Failed to resolve SHM payload %s: %s", shm_name, e)
                if msg.get("is_request") and "request_id" in msg:
                    err_resp = {
                        "response_to": msg["request_id"],
                        "payload": {"ok": False, "error": "shm_resolution_failed"},
                        "trace_id": msg.get("trace_id"),
                    }
                    raw_resp = json.dumps(err_resp)
                    try:
                        await self._write_raw_message(
                            raw_resp,
                            timeout_s=self._response_write_timeout_s(),
                            context="response:shm_resolution_failed",
                        )
                    except _BUS_HANDLER_ERRORS as send_exc:
                        self._mark_transport_degraded(
                            send_exc,
                            "failed to send SHM-resolution error response",
                        )
                return

        # Check if it's a response to a pending request. Use .get so a response
        # envelope missing its payload cannot raise KeyError after the future
        # has already been popped (which would leave the request unresolved).
        if "response_to" in msg:
            req_id = msg.get("response_to")
            future = self._pending_requests.pop(req_id, None)
            if future and not future.done():
                future.set_result(msg.get("payload"))
            return

        # Normal message or request. A forged non-hashable type must not raise
        # from the dict membership test.
        msg_type = msg.get("type")
        if not isinstance(msg_type, Hashable):
            logger.debug("Dropped bus message with non-hashable type: %r", type(msg_type))
            return
        if msg_type in self._handlers:
            handler = self._handlers[msg_type]
            if self._dispatch_queue is None:
                self._dispatch_queue = asyncio.Queue(maxsize=256)
            try:
                await asyncio.wait_for(
                    self._dispatch_queue.put((handler, msg)),
                    timeout=1.0,
                )
            except TimeoutError:
                logger.warning("Bus dispatch queue saturated. Dropping %s.", msg_type)
                if msg.get("is_request") and "request_id" in msg:
                    err_resp = {
                        "response_to": msg["request_id"],
                        "payload": {"ok": False, "error": "dispatch_queue_saturated"},
                        "trace_id": msg.get("trace_id"),
                    }
                    raw_resp = json.dumps(err_resp)
                    try:
                        await self._write_raw_message(
                            raw_resp,
                            timeout_s=self._response_write_timeout_s(),
                            context="response:dispatch_queue_saturated",
                        )
                    except _BUS_HANDLER_ERRORS as send_exc:
                        self._mark_transport_degraded(
                            send_exc,
                            "failed to send dispatch-saturation error response",
                        )
        else:
            logger.debug("Unhandled bus message type: %s", msg_type)

    async def _dispatch_loop(self):
        """Process inbound messages in arrival order with bounded backpressure."""
        while self._is_running:
            try:
                if self._dispatch_queue is None:
                    await asyncio.sleep(0.05)
                    continue
                handler, msg = await self._dispatch_queue.get()
                try:
                    await self._handle_message(handler, msg)
                finally:
                    self._dispatch_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — dispatcher must outlive any fault
                record_degradation('local_pipe_bus', e)
                logger.error("Error in Bus dispatch loop: %s", e)
                
                # Component Hardening: Active Self-Repair Invocation
                try:
                    from core.runtime.self_healing import get_healer
                    get_healer().schedule_deep_repair(
                        "core/bus/local_pipe_bus.py",
                        reason="dispatch_loop_exception",
                        metadata={"error": str(e)}
                    )
                except (ImportError, AttributeError, RuntimeError) as _heal_e:
                    logger.debug("Deep repair scheduling failed: %s", _heal_e)
                    
                await asyncio.sleep(1.0)

    def _cancel_pending_requests(self, exception: Exception | None = None, cancel: bool = False):
        """[GENESIS FIX] Ensure all awaiting requests are rejected immediately if the pipe dies."""
        for _req_id, future in list(self._pending_requests.items()):
            if future.done():
                continue
            if cancel or exception is None:
                future.cancel()
            else:
                future.set_exception(exception)
        self._pending_requests.clear()

    async def _handle_message(self, handler: Callable, msg: dict):
        """Wrap handler execution and handle responses."""
        try:
            result = handler(msg.get("payload"), msg.get("trace_id"))
            if asyncio.iscoroutine(result):
                # Bounded so a non-terminating coroutine handler cannot wedge the
                # single dispatcher and starve every later response.
                result = await asyncio.wait_for(result, timeout=self._HANDLER_EXEC_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            # Isolation boundary: a handler is effectively untrusted code. ANY
            # fault it raises (including types outside _BUS_HANDLER_ERRORS, which
            # previously escaped the dispatcher and terminated the worker) is
            # contained here and, for requests, returned as a typed failure.
            record_degradation('local_pipe_bus', e)
            logger.error("Bus handler error (%s): %s", msg.get("type"), e)

            # Component Hardening: Active Self-Repair Invocation
            try:
                from core.runtime.self_healing import get_healer
                get_healer().schedule_deep_repair(
                    "core/bus/local_pipe_bus.py",
                    reason="handler_exception",
                    metadata={"error": str(e), "msg_type": msg.get("type")}
                )
            except (ImportError, AttributeError, RuntimeError) as _heal_e:
                logger.debug("Deep repair scheduling failed: %s", _heal_e)

            if msg.get("is_request") and "request_id" in msg:
                err_resp = {
                    "response_to": msg["request_id"],
                    "payload": {"error": str(e)},
                    "trace_id": msg.get("trace_id"),
                    "failed": True
                }
                # Consistently use JSON
                raw_err = json.dumps(err_resp)
                try:
                    await self._write_raw_message(
                        raw_err,
                        timeout_s=self._response_write_timeout_s(),
                        context=f"error_response:{msg.get('type')}",
                    )
                except _BUS_HANDLER_ERRORS as send_exc:
                    record_degradation('local_pipe_bus', send_exc)
                    logger.error(
                        "Bus error-response write failed (%s): %s",
                        msg.get("type"),
                        send_exc,
                    )
            return

        # If it was a request, send back the result via write_conn.
        if msg.get("is_request") and "request_id" in msg:
            resp = {
                "response_to": msg["request_id"],
                "payload": result,
                "trace_id": msg.get("trace_id")
            }
            raw_resp = json.dumps(resp)
            try:
                await self._write_raw_message(
                    raw_resp,
                    timeout_s=self._response_write_timeout_s(),
                    context=f"response:{msg.get('type')}",
                )
            except _BUS_HANDLER_ERRORS as e:
                record_degradation('local_pipe_bus', e)
                logger.error("Bus response write failed (%s): %s", msg.get("type"), e)
