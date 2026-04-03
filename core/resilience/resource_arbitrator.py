import os
import logging
import asyncio
import contextlib
import fcntl
from pathlib import Path
from typing import Optional, AsyncGenerator

logger = logging.getLogger("Aura.ResourceArbitrator")

class ResourceArbitrator:
    """
    Arbitrates access to finite physical resources (GPU/VRAM) to prevent OOM-kills.
    In Apple Silicon, CPU and GPU share the same memory pool. 
    
    Tokens:
    - INFERENCE_TOKEN: High priority, granted immediately to primary inference requests.
    - EVOLUTION_TOKEN: Low priority, background tasks (LoRA training, heavy eval).
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResourceArbitrator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"): return
        self._initialized = True
        
        # Loop-Agnostic Synchronization (Fix #4)
        from core.utils.concurrency import RobustLock, get_robust_semaphore
        from core.utils.queues import LoopAgnosticQueue

        # Per-worker semaphores: each MLX worker process (32B Cortex, 7B Brainstem,
        # etc.) handles exactly ONE inference at a time via its IPC pipe.
        # Using per-worker semaphores means brainstem background inference does NOT
        # block cortex user inference — they run on separate processes in parallel.
        # A global fallback semaphore ("_default") serializes requests that don't
        # specify a worker label (e.g. Reflex-CPU or legacy callers).
        self._worker_sems: dict = {}
        # Legacy single semaphore kept for callers that don't pass a worker label.
        self._gpu_sem = get_robust_semaphore("ResourceArbitrator.GPUSem._default", 1)
        
        # For cross-process coordination (e.g. Inference Process vs Hephaestus Process)
        # We'll use a file-based lock for true cross-process synchronization on macOS.
        self._lock_path = str(Path.home() / ".aura" / "run" / "vram.lock")
        self._inference_active = False
        self._evolution_active = False
        self._mp_fd: Optional[int] = None
        self._mp_inference_fd: Optional[int] = None
        # FIX: _priority_queue was asyncio.Queue (loop-bound).
        # Replaced with LoopAgnosticQueue to prevent affinity crashes.
        self._priority_queue = LoopAgnosticQueue()

    def _get_mp_lock(self):
        """Standard file lock for cross-process VRAM arbitration."""
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            return fd
        except Exception as e:
            logger.error(f"Failed to open VRAM lock file: {e}")
            return None

    def _get_worker_sem(self, worker: Optional[str]):
        """Return (or create) the per-worker semaphore for the given label.

        Each MLX worker process has its own semaphore so that, e.g., brainstem
        background inference cannot block cortex user inference.  Callers that
        don't specify a worker fall back to the legacy global semaphore.
        """
        if not worker:
            return self._gpu_sem
        if worker not in self._worker_sems:
            from core.utils.concurrency import get_robust_semaphore
            self._worker_sems[worker] = get_robust_semaphore(
                f"ResourceArbitrator.GPUSem.{worker}", 1
            )
        return self._worker_sems[worker]

    async def acquire_inference(
        self,
        priority: bool = False,
        timeout: Optional[float] = None,
        worker: Optional[str] = None,
    ) -> bool:
        """Request VRAM access for inference.

        ``worker`` identifies the specific MLX worker process (e.g. "MLX-Cortex",
        "MLX-Brainstem").  Per-worker semaphores ensure that brainstem background
        inference never blocks cortex user inference.

        Priority (user-facing) requests get a longer timeout (90s).
        Background requests use the caller-supplied timeout.
        """
        sem = self._get_worker_sem(worker)
        if priority:
            logger.debug("🚀 [PRIORITY] User-facing inference token requested (worker=%s).", worker or "default")
            effective_timeout = max(0.25, float(timeout)) if timeout is not None else 90.0
        else:
            logger.debug("Attempting to acquire INFERENCE token (worker=%s)...", worker or "default")
            effective_timeout = max(0.25, float(timeout)) if timeout is not None else 30.0

        acquired = await sem.acquire(timeout=effective_timeout)
        if not acquired:
            logger.error("🕒 INFERENCE token TIMEOUT for worker=%s after %ss.", worker or "default", effective_timeout)
            return False

        self._inference_active = True
        logger.info("⚡ INFERENCE token acquired (worker=%s, priority=%s).", worker or "default", priority)
        return True

    async def release_inference(self, worker: Optional[str] = None):
        """Release VRAM access after inference."""
        self._inference_active = False
        sem = self._get_worker_sem(worker)
        try:
            sem.release()
        except ValueError:
            logger.warning("Attempted to over-release INFERENCE token for worker=%s.", worker or "default")
        logger.info("📥 INFERENCE token released (worker=%s).", worker or "default")

    async def acquire_evolution(self, timeout: float = 300.0) -> bool:
        """
        Request VRAM access for background evolution tasks (Low Priority).
        Blocks until inference is idle or timeout occurs.
        """
        logger.debug("Attempting to acquire EVOLUTION token...")
        
        start_time = asyncio.get_running_loop().time()
        while self._inference_active:
            if asyncio.get_running_loop().time() - start_time > timeout:
                logger.warning("🕒 EVOLUTION token request TIMEOUT (Inference busy).")
                return False
            await asyncio.sleep(0.1) # Check more frequently than 1s
            
        # We use a file lock to protect against other evolutionary cycles
        # or parallel worker initialization across processes.
        fd = self._get_mp_lock()
        if fd is None: return False
        
        try:
            # Non-blocking lock attempt first, then polling to avoid blocking the loop too long
            while self._inference_active:
                if asyncio.get_running_loop().time() - start_time > timeout:
                    logger.warning("🕒 EVOLUTION token request TIMEOUT (Inference busy).")
                    os.close(fd)
                    return False
                await asyncio.sleep(1.0)

            # Now try to acquire the process-wide lock
            acquired = False
            while not acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if asyncio.get_running_loop().time() - start_time > timeout:
                        logger.warning("🕒 EVOLUTION token request TIMEOUT (Lock busy).")
                        os.close(fd)
                        return False
                    await asyncio.sleep(0.1)
            
            self._evolution_active = True
            logger.info("🧬 EVOLUTION token acquired (Background tasks permitted).")
            # Store FD for release
            self._mp_fd = fd
            return True
            
        except Exception as e:
            logger.error(f"Error acquiring VRAM lock: {e}")
            os.close(fd)
            return False

    def release_evolution(self):
        """Release VRAM access after evolution task."""
        if hasattr(self, "_mp_fd") and self._mp_fd is not None:
            try:
                fcntl.flock(self._mp_fd, fcntl.LOCK_UN)
                os.close(self._mp_fd)
                self._mp_fd = None
                self._evolution_active = False
                logger.info("🧬 EVOLUTION token released.")
            except Exception as e:
                logger.error(f"Error releasing VRAM lock: {e}")

    def is_inference_busy(self) -> bool:
        """Check if primary inference is active."""
        return self._inference_active

    @contextlib.asynccontextmanager
    async def inference_context(
        self,
        priority: bool = False,
        worker: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[None, None]:
        """Context manager for inference-level VRAM locking.

        Pass ``worker`` (e.g. "MLX-Cortex", "MLX-Brainstem") to use a
        per-worker semaphore.  This lets the 32B cortex and 7B brainstem run
        their inferences concurrently without blocking each other.
        """
        acquired = await self.acquire_inference(priority=priority, timeout=timeout, worker=worker)
        if not acquired:
            raise asyncio.TimeoutError(f"inference_token_timeout:{worker or 'default'}")
        try:
            yield
        finally:
            await self.release_inference(worker=worker)

    @contextlib.asynccontextmanager
    async def evolution_context(self, timeout: float = 300.0) -> AsyncGenerator[bool, None]:
        """Context manager for background evolution/training VRAM arbitration."""
        acquired = await self.acquire_evolution(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release_evolution()

def get_resource_arbitrator() -> ResourceArbitrator:
    return ResourceArbitrator()
