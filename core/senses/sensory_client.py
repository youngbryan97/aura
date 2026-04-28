from core.runtime.errors import record_degradation
import asyncio
import logging
import multiprocessing as mp
import sys
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("core.senses.sensory_client")

class SensoryLocalClient:
    """
    Supervisor for the isolated Sensory Worker process.
    Manages Vision and Audio libraries (cv2, mss, sounddevice) in a sidecar PID.
    """
    def __init__(self):
        self._process = None
        self._req_q = mp.Queue()
        self._res_q = mp.Queue()
        self._running = False
        self._lock: Optional[asyncio.Lock] = None
        self._start_lock: Optional[asyncio.Lock] = None

    def _ensure_async_locks(self) -> tuple[asyncio.Lock, asyncio.Lock]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        return self._lock, self._start_lock

    async def start(self):
        """Start the isolated sensory worker."""
        from .sensory_worker import sensory_worker_loop
        _, start_lock = self._ensure_async_locks()

        async with start_lock:
            if self.is_alive():
                logger.debug("👀 Sensory Client: Worker already alive.")
                return True

            self._drain_queues()
            ctx_name = "spawn" if sys.platform == "darwin" else "forkserver"
            ctx = mp.get_context(ctx_name)
            self._process = ctx.Process(
                target=sensory_worker_loop,
                args=(self._req_q, self._res_q),
                name="AuraSensoryWorker",
                daemon=True
            )
            self._process.start()
            self._running = True
            logger.info("👀 Sensory Client: Worker started via %s (PID: %d)", ctx_name, self._process.pid)

            if not await self._send_command("ping", timeout=2.0, auto_restart=False):
                logger.error("🛑 Sensory Client: Worker failed initial ping.")
                await self.stop()
                return False

            success = await self._send_command("init_vision", auto_restart=False)
            if success:
                logger.info("   ✅ Vision isolated successfully")

            success = await self._send_command("init_audio", auto_restart=False)
            if success:
                logger.info("   ✅ Audio isolated successfully")

            return True

    def _drain_queues(self) -> None:
        while not self._req_q.empty():
            try:
                self._req_q.get_nowait()
            except Exception:
                break
        while not self._res_q.empty():
            try:
                self._res_q.get_nowait()
            except Exception:
                break

    async def _send_command(self, cmd: str, data: Any = None, *, timeout: float = 5.0, auto_restart: bool = True) -> bool:
        if not self.is_alive():
            if not auto_restart:
                logger.warning("👀 Sensory Client: Worker unavailable for command %s", cmd)
                return False
            logger.warning("♻️ Sensory Client: Worker offline before command %s. Restarting.", cmd)
            started = await self.start()
            if not started:
                return False

        command_lock, _ = self._ensure_async_locks()

        async with command_lock:

            # [STRUCTURAL UNIFICATION] Report sensory tasks to registry
            from core.supervisor.registry import get_task_registry, TaskStatus
            registry = get_task_registry()
            task_id = registry.register_task("sensory_gate", f"Sensory: {cmd}", {"data": str(data)})
            
            try:
                self._req_q.put({"command": cmd, "data": data})
                registry.update_task(task_id, status=TaskStatus.RUNNING)
                
                # Wait for response in a thread to non-block
                res = await asyncio.to_thread(self._res_q.get, timeout=timeout)
                
                if res.get("status") == "ok":
                    registry.update_task(task_id, status=TaskStatus.COMPLETED)
                    return True
                else:
                    registry.update_task(task_id, status=TaskStatus.FAILED, error=res.get("msg"))
                    return False
            except Exception as e:
                record_degradation('sensory_client', e)
                logger.error("🛑 Sensory Client Command [%s] failed: %s", cmd, e)
                registry.update_task(task_id, status=TaskStatus.FAILED, error=str(e))
                return False

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    async def stop(self):
        self._running = False
        if self._process:
            try:
                self._req_q.put({"command": "exit"})
            except Exception as _exc:
                record_degradation('sensory_client', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
            # Issue 26: Use asyncio.to_thread for blocking process join
            await asyncio.to_thread(self._process.join, timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                await asyncio.to_thread(self._process.join, timeout=1.0)
            logger.info("👀 Sensory Client: Worker stopped")
            self._process = None
            self._drain_queues()

_instance = None
_client_lock = threading.Lock()

def get_sensory_client():
    global _instance
    with _client_lock:
        if _instance is None:
            _instance = SensoryLocalClient()
    return _instance
