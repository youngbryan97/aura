from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import sys
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.shutdown_execution import run_sync_shutdown_callable_blocking
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    get_subprocess_gateway,
)

logger = logging.getLogger("core.senses.sensory_client")

class SensoryLocalClient:
    """
    Supervisor for the isolated Sensory Worker process.
    Manages Vision and Audio libraries (cv2, mss, sounddevice) in a sidecar PID.
    """
    def __init__(self) -> None:
        self._process: Any | None = None
        self._req_q: Any | None = None
        self._res_q: Any | None = None
        self._running = False
        # A multiprocessing request/response pair is one ordered transaction.
        # Camera calls originate in OS threads while screen/audio calls originate
        # in asyncio loops, so an asyncio.Lock is the wrong primitive: it becomes
        # bound to whichever loop touched it first and later sidecar camera reads
        # can fail with "bound to a different event loop". One checked thread
        # lock serializes both sync and async callers at the actual IPC boundary.
        from core.runtime.lockdep import checked_lock

        self._ipc_lock = checked_lock("sensory_client.ipc")
        self._lifecycle_lock = checked_lock("sensory_client.lifecycle")

    async def start(self) -> bool:
        """Start the isolated sensory worker."""
        return await asyncio.to_thread(self._start_blocking)

    def _start_blocking(self) -> bool:
        """Start and initialize the worker from any thread or event loop."""
        if is_shutdown_requested():
            logger.info("Sensory worker start refused: runtime shutdown requested")
            return False
        from .sensory_worker import sensory_worker_loop

        with self._lifecycle_lock:
            if is_shutdown_requested():
                return False
            if self.is_alive():
                logger.debug("👀 Sensory Client: Worker already alive.")
                return True

            ctx_name = "spawn" if sys.platform == "darwin" else "forkserver"
            ctx: Any = mp.get_context(ctx_name)
            self._replace_queues(ctx)
            try:
                process = get_subprocess_gateway().spawn_python_process(
                    PythonProcessSpec(
                        target=sensory_worker_loop,
                        args=(self._req_q, self._res_q),
                        source="sensory_client.worker_owner",
                        name="AuraSensoryWorker",
                        role=ProcessRole.TOOL_RUNNER,
                        requested_privileges=frozenset(
                            {Privilege.FILESYSTEM_READ, Privilege.USER_SURFACE}
                        ),
                        accelerator_capability=AcceleratorCapability.NONE,
                        daemon=True,
                        start_method=ctx_name,
                        environment_overrides={"AURA_MEDIA_SIDECAR_PROCESS": "1"},
                    ),
                    context=ctx,
                )
            except RuntimeError:
                if is_shutdown_requested():
                    self._process = None
                    self._close_queues()
                    return False
                raise
            self._process = process
            if is_shutdown_requested():
                logger.info("Sensory worker crossed shutdown during spawn; stopping it")
                self._stop_blocking(lock_held=True)
                return False
            self._running = True
            logger.info("👀 Sensory Client: Worker started via %s (PID: %d)", ctx_name, process.pid)

            if self._request_blocking("ping", None, 2.0).get("status") != "ok":
                logger.error("🛑 Sensory Client: Worker failed initial ping.")
                self._stop_blocking(lock_held=True)
                return False

            success = self._request_blocking("init_vision", None, 5.0).get("status") == "ok"
            if success:
                logger.info("   ✅ Vision isolated successfully")

            success = self._request_blocking("init_audio", None, 5.0).get("status") == "ok"
            if success:
                logger.info("   ✅ Audio isolated successfully")

            return True

    def _drain_queues(self) -> None:
        if self._req_q is None or self._res_q is None:
            return
        while not self._req_q.empty():
            try:
                self._req_q.get_nowait()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                break
        while not self._res_q.empty():
            try:
                self._res_q.get_nowait()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                break

    @staticmethod
    def _safe_close_queue(queue_obj: Any) -> None:
        if queue_obj is None:
            return

        def _close_and_join() -> None:
            close = getattr(queue_obj, "close", None)
            if callable(close):
                close()
            join_thread = getattr(queue_obj, "join_thread", None)
            if callable(join_thread):
                join_thread()

        try:
            run_sync_shutdown_callable_blocking(
                _close_and_join,
                timeout_s=1.0,
                name="sensory-queue-close",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            logger.debug("Suppressed queue close error in sensory client: %s", exc)
        else:
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().unregister_shutdown_resource(queue_obj)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                pass

    def _close_queues(self) -> None:
        self._safe_close_queue(self._req_q)
        self._safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None

    def _replace_queues(self, ctx: Any | None = None) -> None:
        self._close_queues()
        factory = ctx.Queue if ctx is not None and hasattr(ctx, "Queue") else mp.Queue
        self._req_q = factory()
        self._res_q = factory()
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            hygiene = get_runtime_hygiene()
            hygiene.register_shutdown_resource(
                self._req_q,
                kind="multiprocessing_queue",
                name="sensory.request_queue",
                source="core.senses.sensory_client",
                timeout_s=1.0,
            )
            hygiene.register_shutdown_resource(
                self._res_q,
                kind="multiprocessing_queue",
                name="sensory.response_queue",
                source="core.senses.sensory_client",
                timeout_s=1.0,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            self._close_queues()
            raise

    async def _send_command(  # noqa: ASYNC109 - timeout bounds blocking sidecar IPC.
        self,
        cmd: str,
        data: Any = None,
        *,
        timeout: float = 5.0,  # noqa: ASYNC109 - bounds blocking sidecar IPC.
        auto_restart: bool = True,
    ) -> bool:
        if is_shutdown_requested():
            logger.info("Sensory command %s refused during runtime shutdown", cmd)
            return False
        if not self.is_alive():
            if not auto_restart:
                logger.warning("👀 Sensory Client: Worker unavailable for command %s", cmd)
                return False
            logger.warning("♻️ Sensory Client: Worker offline before command %s. Restarting.", cmd)
            started = await self.start()
            if not started:
                return False

        from core.supervisor.registry import TaskStatus, get_task_registry

        registry = get_task_registry()
        task_id = registry.register_task(
            "sensory_gate",
            f"Sensory: {cmd}",
            {"command": cmd},
        )
        registry.update_task(task_id, status=TaskStatus.RUNNING)
        try:
            reply = await asyncio.to_thread(self._request_blocking, cmd, data, timeout)
        except Exception as exc:
            registry.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(exc),
            )
            raise
        succeeded = reply.get("status") == "ok"
        registry.update_task(
            task_id,
            status=TaskStatus.COMPLETED if succeeded else TaskStatus.FAILED,
            error="" if succeeded else str(reply.get("msg") or ""),
        )
        return succeeded

    def _request_blocking(
        self,
        cmd: str,
        data: Any,
        timeout: float,
    ) -> dict[str, Any]:
        """Perform one queue round-trip under the cross-thread IPC lock."""
        with self._ipc_lock:
            if self._req_q is None or self._res_q is None:
                return {"status": "error", "msg": "worker_queues_unavailable"}
            try:
                self._req_q.put({"command": cmd, "data": data})
                reply = self._res_q.get(timeout=timeout)
            except (OSError, ConnectionError, TimeoutError, queue.Empty) as exc:
                record_degradation("sensory_client", exc)
                logger.error("🛑 Sensory Client request [%s] failed: %s", cmd, exc)
                return {"status": "error", "msg": str(exc)}
        return reply if isinstance(reply, dict) else {"status": "error", "msg": "bad_reply"}

    async def request(  # noqa: ASYNC109 - timeout bounds blocking sidecar IPC.
        self,
        cmd: str,
        data: Any = None,
        *,
        timeout: float = 5.0,  # noqa: ASYNC109 - bounds blocking sidecar IPC.
        auto_restart: bool = True,
    ) -> dict[str, Any]:
        """Send a command and return the worker's FULL reply.

        `_send_command` reduces every reply to a bool, which is fine for
        ping and init but throws away the payload. A camera frame is a
        payload, so a bool-only channel cannot carry one — which is part of
        why the sidecar, documented as the production camera path, never
        grew a camera command.

        Returns a dict always: `{"status": "error", "msg": ...}` rather than
        raising, because every caller is a capture path that must turn a
        failure into a named reason.
        """
        if is_shutdown_requested():
            return {"status": "error", "msg": "runtime_shutdown"}
        if not self.is_alive():
            if not auto_restart:
                return {"status": "error", "msg": "worker_unavailable"}
            if not await self.start():
                return {"status": "error", "msg": "worker_start_failed"}

        return await asyncio.to_thread(self._request_blocking, cmd, data, timeout)

    def request_sync(
        self,
        cmd: str,
        data: Any = None,
        *,
        timeout: float = 5.0,
        auto_restart: bool = True,
    ) -> dict[str, Any]:
        """Thread-safe synchronous sidecar request for camera owners.

        The camera authority is deliberately synchronous because several
        holders are OS threads. Async callers must offload the authority call;
        silently nesting a new event loop here was both an event-loop stall and
        a cross-loop lock defect.
        """
        if is_shutdown_requested():
            return {"status": "error", "msg": "runtime_shutdown"}
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return {"status": "error", "msg": "sync_request_on_event_loop"}
        if not self.is_alive():
            if not auto_restart:
                return {"status": "error", "msg": "worker_unavailable"}
            if not self._start_blocking():
                return {"status": "error", "msg": "worker_start_failed"}
        return self._request_blocking(cmd, data, timeout)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_blocking)

    def _stop_blocking(self, *, lock_held: bool = False) -> None:
        if not lock_held:
            self._lifecycle_lock.acquire()
        try:
            self._stop_blocking_locked()
        finally:
            if not lock_held:
                self._lifecycle_lock.release()

    def _stop_blocking_locked(self) -> None:
        # Serialize exit and queue teardown with the same lock as requests.
        # Otherwise an in-flight camera read can consume the exit reply, while
        # shutdown consumes the JPEG as though it acknowledged exit.
        with self._ipc_lock:
            self._running = False
            process, self._process = self._process, None
            if process:
                try:
                    if self._req_q is not None:
                        self._req_q.put({"command": "exit"})
                except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    record_degradation('sensory_client', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
                # The async public wrapper runs this whole shutdown in a worker
                # thread, so process joins never block Aura's event loop.
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                logger.info("👀 Sensory Client: Worker stopped")
                self._drain_queues()
            self._close_queues()

    close = stop
    cleanup = stop
    on_stop = stop

_instance: SensoryLocalClient | None = None
_client_lock = checked_lock("core.senses.sensory_client._client_lock")

def get_sensory_client() -> SensoryLocalClient:
    global _instance
    with _client_lock:
        if _instance is None:
            _instance = SensoryLocalClient()
    return _instance
