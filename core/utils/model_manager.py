# core/utils/model_manager.py
"""
ModelManager: single point-of-truth for loading/unloading heavy model objects.
- serializes heavy loads with an asyncio.Semaphore
- tracks loaded models in LRU order (OrderedDict)
- evicts least-recently-used model when memory pressure or configured cap exceeded
- exposes async load_model / unload_model
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional
import psutil

logger = logging.getLogger("aura.model_manager")


class ModelLoadError(Exception):
    pass  # no-op: intentional


class ModelManager:
    def __init__(self, load_fn: Callable[[str, dict], Any], max_models: int = 2, semaphore_value: int = 1):
        """
        load_fn(name, opts) -> model_object
        """
        self._load_fn = load_fn
        self._models: "OrderedDict[str, Any]" = OrderedDict()
        self._meta: Dict[str, dict] = {}
        self._semaphore = asyncio.Semaphore(semaphore_value)
        self._max_models = max_models
        self._lock = asyncio.Lock()
        self._last_used: Dict[str, float] = {}

    def _pop_model_locked(self, name: str):
        """Internal helper to remove model from state tracking. MUST hold _lock."""
        if name not in self._models:
            return None, None
        obj = self._models.pop(name)
        meta = self._meta.pop(name, {})
        self._last_used.pop(name, None)
        return obj, meta

    async def _cleanup_model(self, obj: Any, name: str) -> None:
        """Internal helper to actually close/unload a model object. NO lock needed."""
        try:
            if hasattr(obj, "close"):
                maybe = obj.close()
                if asyncio.iscoroutine(maybe):
                    await maybe
            elif hasattr(obj, "unload"):
                maybe = obj.unload()
                if asyncio.iscoroutine(maybe):
                    await maybe
        except Exception:
            logger.exception("ModelManager: exception while unloading %s", name)

    async def load_model(self, name: str, opts: Optional[dict] = None) -> Any:
        opts = opts or {}
        async with self._lock:
            if name in self._models:
                # move to end (most-recently used)
                self._models.move_to_end(name)
                self._last_used[name] = time.time()
                logger.debug("ModelManager: model %s already loaded (touch)", name)
                return self._models[name]

        # serialize heavy model loads
        async with self._semaphore:
            cleanup_obj = None
            evicted_name = None
            
            async with self._lock:
                # double-check after obtaining lock
                if name in self._models:
                    self._models.move_to_end(name)
                    self._last_used[name] = time.time()
                    return self._models[name]

                # if we've hit capacity, evict LRU
                if len(self._models) >= self._max_models:
                    evicted_name = next(iter(self._models.keys()))
                    logger.info("ModelManager: capacity full (%d). Evicting LRU model: %s", self._max_models, evicted_name)
                    cleanup_obj, _ = self._pop_model_locked(evicted_name)

                # check memory pressure before loading
                vm = psutil.virtual_memory()
                if vm.percent > 85.0:
                    # If we popped but didn't clean up yet, we should probably re-add or just let it go.
                    # For safety if we are over 85%, we abort.
                    raise ModelLoadError(f"Refusing to load model {name} — host memory at {vm.percent:.1f}%")

            # Clean up evicted model WITHOUT holding self._lock to avoid deadlock
            if cleanup_obj:
                await self._cleanup_model(cleanup_obj, evicted_name)

            # perform actual load (synchronous or async support)
            logger.info("ModelManager: loading model %s", name)
            try:
                maybe_coro = self._load_fn(name, opts)
                if asyncio.iscoroutine(maybe_coro):
                    model_obj = await maybe_coro
                else:
                    model_obj = maybe_coro
            except Exception as e:
                record_degradation('model_manager', e)
                logger.error("ModelManager: failed to load %s: %s", name, e)
                raise ModelLoadError(f"Failed to load model {name}") from e

            async with self._lock:
                self._models[name] = model_obj
                self._meta[name] = {"loaded_at": time.time(), "opts": opts}
                self._last_used[name] = time.time()
                logger.info("ModelManager: loaded model %s", name)
                return model_obj

    async def unload_model(self, name: str) -> bool:
        async with self._lock:
            obj, _ = self._pop_model_locked(name)
        
        if obj is None:
            logger.debug("ModelManager: unload requested for unknown model %s", name)
            return False
            
        await self._cleanup_model(obj, name)
        return True

    async def evict_if_needed(self):
        """Evict LRU while memory pressure or over capacity."""
        while True:
            cleanup_obj = None
            evicted_name = None
            
            async with self._lock:
                vm = psutil.virtual_memory()
                if len(self._models) > 0 and (vm.percent > 80.0 or len(self._models) > self._max_models):
                    evicted_name = next(iter(self._models.keys()))
                    logger.warning("ModelManager: evicting %s due to memory/capacity", evicted_name)
                    cleanup_obj, _ = self._pop_model_locked(evicted_name)
                else:
                    break
            
            if cleanup_obj:
                await self._cleanup_model(cleanup_obj, evicted_name)

    def list_loaded(self):
        return list(self._models.keys())

    async def unload_all(self):
        async with self._lock:
            names = list(self._models.keys())
        for n in names:
            await self.unload_model(n)
