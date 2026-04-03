import asyncio
import gc
import logging

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger("Aura.VRAMManager")

class VRAMManager:
    """Exclusive lock manager for M5 Pro Unified Memory.

    Ensures that heavy MLX models (LLM, Whisper, Vision) do not
    overlap in the 64GB Unified Memory, triggering MacOS SSD swap.
    """
    def __init__(self):
        from core.utils.concurrency import get_robust_lock
        self._lock = get_robust_lock("VRAMManager")
        self.active_model = None
        self._pre_purge_hook = None  # async callable → called before KV cache destruction

    def set_pre_purge_hook(self, hook) -> None:
        """Register an async callback invoked before Metal cache is cleared.

        Use this to trigger context summarization before a model swap destroys the
        KV cache.  When swapping from the 32B Cortex to the 72B DeepSolver the new
        model starts cold; compressing context here lets it ingest a dense abstract
        instead of re-computing attention for 40k tokens.

        The hook receives no arguments and should complete quickly (< 5s).
        Errors are suppressed so they never block the swap.
        """
        self._pre_purge_hook = hook
        logger.info("VRAMManager: pre-purge summarization hook registered.")

    def _fire_pre_purge_hook(self) -> None:
        if self._pre_purge_hook is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                task = loop.create_task(self._pre_purge_hook(), name="vram_manager.pre_purge")
                try:
                    from core.utils.task_tracker import get_task_tracker

                    get_task_tracker().track_task(task)
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                logger.info("🗜️ VRAMManager: pre-purge context summarization hook fired.")
            else:
                asyncio.run(self._pre_purge_hook())
                logger.info("🗜️ VRAMManager: pre-purge context summarization hook fired synchronously.")
        except Exception as exc:
            logger.debug("pre-purge hook error (non-fatal): %s", exc)

    async def acquire(self, model_name: str):
        """Acquire the Neural Engine lock for a specific model."""
        await self._lock.acquire()
        if self.active_model and self.active_model != model_name:
            logger.info(f"VRAM Swap: Unloading {self.active_model} for {model_name}")
            self.purge()
        self.active_model = model_name
        logger.debug(f"VRAM Lock Acquired: {model_name}")

    def release(self):
        """Release the Neural Engine lock."""
        self._lock.release()
        logger.debug(f"VRAM Lock Released: {self.active_model}")

    def acquire_session(self, model_name: str):
        """Returns an async context manager for VRAM usage."""
        return VRAMSession(self, model_name)

    def purge(self):
        """Forces Apple Silicon to dump VRAM back to the OS.

        Fires the pre-purge summarization hook synchronously (fire-and-forget
        coroutine) so the next model can boot with a compressed context digest
        instead of a cold 40k-token re-ingest.
        """
        self._fire_pre_purge_hook()
        gc.collect()
        if MLX_AVAILABLE:
            try:
                from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
                sentinel = get_gpu_sentinel()
                if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                    try:
                        if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                            mx.metal.clear_cache()
                        else:
                            mx.clear_cache()
                    finally:
                        sentinel.release()
            except Exception: pass
            logger.debug("VRAM purged and zeroed (MLX Metal Cache Cleared).")
        else:
            logger.debug("VRAM purged (GC only).")

class VRAMSession:
    def __init__(self, manager: VRAMManager, model_name: str):
        self.manager = manager
        self.model_name = model_name

    async def __aenter__(self):
        await self.manager.acquire(self.model_name)
        return self.manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.manager.release()

# Global Singleton
_vram_instance = None

def get_vram_manager() -> VRAMManager:
    """Get the active VRAM Manager instance."""
    global _vram_instance
    if _vram_instance is None:
        _vram_instance = VRAMManager()
    return _vram_instance
