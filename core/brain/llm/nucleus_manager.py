from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from core.runtime.errors import Severity, record_degradation
from core.utils.exceptions import capture_and_log
from core.utils.task_tracker import get_task_tracker

from .provider import LLMProvider

logger = logging.getLogger("LLM.Nucleus")

_CONSTITUTIVE_ORIGINS = frozenset({
    "constitutive_expression",
    "drive_controller",
    "affect_engine",
    "autonomy_guardian",
    "sensory_motor_cortex",
    "pulse_manager",
    "body_monitor",
    "subsystem_audit",
    "health_monitor",
    "agency_core",
})
_NUCLEUS_MODEL_TYPES = frozenset({"brainstem", "cortex"})
_NUCLEUS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    OSError,
    TypeError,
    ValueError,
)


def _empty_model_entry() -> dict[str, Any]:
    return {
        "model": None,
        "tokenizer": None,
        "loaded": False,
        "cache": None,
        "last_error": None,
    }


def _record_nucleus_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "nucleus_manager",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


class NucleusManager(LLMProvider):
    """Aura's Dual-Nucleus Internal Brain.
    
    Manages two local MLX-optimized models:
    1. Brainstem (Tiny): Perpetual, low-latency reflection for background tasks.
    2. Cortex (Medium): Conversational intelligence, loaded on demand.
    
    Optimized for M5 Pro with KV-Caching and GPU Sentinel.
    """
    
    def __init__(self, **kwargs):
        from .model_registry import (
            BRAINSTEM_MODEL,
            get_active_model,
            get_brainstem_path,
            get_model_path,
            resolve_personality_adapter,
        )
        
        self.brainstem_repo = BRAINSTEM_MODEL
        self.cortex_repo = get_active_model()
        
        self.brainstem_path = str(get_brainstem_path())
        self.cortex_path = str(get_model_path())
        self._adapter_dir = resolve_personality_adapter(self.cortex_path, backend="mlx") or ""
        
        self.models = {name: _empty_model_entry() for name in _NUCLEUS_MODEL_TYPES}
        self._anchor_text = None 
        self._refresh_threshold = 2048 
        self._tokens_seen = 0
        self._listener_task = None
        self._running = True
        # Defer event subscription to avoid create_task in __init__
        try:
            from core.event_bus import get_event_bus
            self.bus = get_event_bus()
        except (ImportError, AttributeError, RuntimeError):
            self.bus = None

    async def ensure_listener_started(self):
        """Start the event listener if not already running. Call from async context."""
        if self._listener_task is None and self.bus is not None:
            self._listener_task = get_task_tracker().create_task(self._listen_for_updates())

    async def _listen_for_updates(self):
        """Listens for LoRA optimization successes and flags for reload."""
        if not self.bus:
            return
        sub = await self.bus.subscribe("core/optimizer/completed")
        while self._running:
            try:
                _, _, event = await sub.get()
                data = event.get("data", {})
                if data.get("status") == "success":
                    logger.info("🧠 [NUCLEUS] Optimization detected. Flagging Cortex for reload.")
                    self.models["cortex"]["loaded"] = False
            except (OSError, ConnectionError, TimeoutError):
                await asyncio.sleep(1)
        
    def _select_model_type(self, origin: str) -> str:
        return "brainstem" if origin in _CONSTITUTIVE_ORIGINS else "cortex"

    def _ensure_model_entry(self, name: str) -> dict[str, Any]:
        entry = self.models.setdefault(name, _empty_model_entry())
        entry.setdefault("cache", None)
        entry.setdefault("last_error", None)
        return entry

    def _model_path_for(self, name: str) -> str:
        return self.brainstem_path if name == "brainstem" else self.cortex_path

    async def load_model(self, name: str) -> bool:
        """Lazy load a specific internal model, including LoRA adapters if present."""
        # Ensure event listener is started from async context
        await self.ensure_listener_started()
        logger.debug("Attempting to load: %s", name)
        entry = self._ensure_model_entry(name)
        if name not in _NUCLEUS_MODEL_TYPES:
            error = ValueError(f"unknown nucleus model lane: {name}")
            entry["last_error"] = str(error)
            _record_nucleus_degradation(
                error,
                severity="warning",
                action="refused unknown nucleus lane and left existing model state unchanged",
                extra={"model": name},
            )
            return False
        if entry.get("loaded"): # Use .get for safety if brainstem is removed from models dict
            logger.debug("%s already loaded.", name)
            return True

        path = self._model_path_for(name)
        logger.debug("Path for %s: %s", name, path)
        path_exists = await asyncio.to_thread(Path(path).exists)
        if not path_exists:
            logger.debug("PATH MISSING: %s", path)
            logger.warning("⚠️ Model path missing: %s. Background fetch may still be running...", path)
            error = FileNotFoundError(path)
            entry.update({"loaded": False, "last_error": str(error), "cache": None})
            _record_nucleus_degradation(
                error,
                severity="warning",
                action=(
                    "marked the nucleus lane unavailable so callers can try the "
                    "alternate lane or return a deterministic offline response"
                ),
                extra={"model": name, "path": path},
            )
            return False

        adapter_path = None
        if name == "cortex":
            adapter_config = Path(self._adapter_dir) / "adapter_config.json"
            adapter_config_exists = (
                bool(self._adapter_dir)
                and await asyncio.to_thread(adapter_config.exists)
            )
            if adapter_config_exists:
                adapter_path = self._adapter_dir
                logger.info("🧠 [NUCLEUS] Found LoRA adapter directory for Cortex: %s", adapter_path)

        try:
            from mlx_lm import load

            from core.utils.gpu_sentinel import get_gpu_sentinel
            sentinel = get_gpu_sentinel()
            
            logger.info("🧠 [NUCLEUS] Loading %s from disk (Adapter: %s)...", name.upper(), adapter_path)
            
            def _load_locked():
                acquired = sentinel.acquire(timeout=90) # Loading is slow, give it more time
                if not acquired:
                    raise TimeoutError("NUCLEUS GPU Sentinel timeout during LOAD")
                try:
                    if adapter_path:
                        return load(path, adapter_path=adapter_path)
                    else:
                        return load(path)
                finally:
                    sentinel.release()

            # Use asyncio.to_thread as load is CPU/IO intensive
            model, tokenizer = await asyncio.to_thread(_load_locked)
                
            self.models[name]["model"] = model
            self.models[name]["tokenizer"] = tokenizer
            self.models[name]["loaded"] = True
            self.models[name]["last_error"] = None

            if name == "cortex":
                # Set a default anchor text when cortex is loaded
                self._anchor_text = (
                    "You are Aura Luna — Aura for short. Speak with direct first-person "
                    "continuity, curiosity, and emotional presence while keeping evidence "
                    "claims grounded. Do not claim aliveness, consciousness, or sealed "
                    "governance from labels alone."
                )
                self._tokens_seen = 0 # Reset token count for new conversation/load
                logger.debug("🧠 [NUCLEUS] Cortex loaded. Anchor text set and token counter reset.")

            logger.info("✅ [NUCLEUS] %s load success.", name.upper())
            logger.info("✅ [NUCLEUS] %s ready.", name.upper())
            return True
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            entry.update({
                "model": None,
                "tokenizer": None,
                "loaded": False,
                "cache": None,
                "last_error": f"{type(e).__name__}: {e}",
            })
            _record_nucleus_degradation(
                e,
                action=(
                    "marked the failed nucleus lane unavailable, cleared partial "
                    "model state, and left alternate-lane fallback eligible"
                ),
                extra={"model": name, "path": path, "adapter_path": adapter_path},
            )
            logger.error("❌ [NUCLEUS] Load failed for %s: %s", name, e)
            logger.error("Failed to load internal model %s: %s", name, e)
            return False

    def _format_prompt(self, prompt: str, system_prompt: str | None = None, prefill: str | None = None) -> str:
        """Formats the prompt using ChatML for Qwen-Instruct models."""
        s_msg = system_prompt if system_prompt else self._anchor_text
        
        # Base ChatML structure
        formatted = f"<|im_start|>system\n{s_msg}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        formatted += f"<|im_start|>assistant\n{prefill if prefill else ''}"
        
        return formatted

    def _apply_anchor(self, prompt: str, system_prompt: str | None = None, model_type: str = "cortex") -> tuple[str, str | None]:
        """
        Manages the semantic anchor. 
        Returns (modified_prompt, modified_system_prompt).
        """
        if model_type != "cortex" or not self._anchor_text:
            return prompt, system_prompt

        tokenizer = self.models["cortex"].get("tokenizer")
        if not tokenizer:
            return prompt, system_prompt

        # Estimate current prompt tokens
        current_prompt_tokens = len(tokenizer.encode(prompt))
        self._tokens_seen += current_prompt_tokens

        if self._tokens_seen >= self._refresh_threshold:
            logger.debug("🧠 [NUCLEUS] Re-injecting semantic anchor for Cortex (tokens seen: %d).", self._tokens_seen)
            # Re-inject by ensuring the anchor is the system prompt
            actual_system = f"{self._anchor_text}\n\n{system_prompt}" if system_prompt else self._anchor_text
            self._tokens_seen = 0 # Reset after re-injection
            return prompt, actual_system
        
        return prompt, system_prompt

    async def generate_text_async(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """Route to appropriate internal model."""
        origin = kwargs.get("origin", "unknown")
        logger.debug("generate_text_async called with origin: %s", origin)
        model_type = self._select_model_type(origin)
        logger.debug("Routing to: %s", model_type)

        await self.load_model(model_type)
        if not self.models.get(model_type, {}).get("loaded"): # Use .get for safety
            # Fallback to brainstem if cortex isn't ready or vice-versa
            alt_type = "brainstem" if model_type == "cortex" else "cortex"
            await self.load_model(alt_type)
            if self.models.get(alt_type, {}).get("loaded"):
                model_type = alt_type
            else:
                logger.error("❌ Both Nucleus models failed to load.")
                _record_nucleus_degradation(
                    RuntimeError("both nucleus model lanes are offline"),
                    action=(
                        "returned deterministic offline text instead of blocking "
                        "or hallucinating an internal inference result"
                    ),
                    extra={"requested_model": model_type, "origin": origin},
                )
                return "[NUCLEUS ERROR] Internal inference offline."

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel

            temp = self._resolve_temperature(kwargs, model_type=model_type, phase="text_generate")
            sampler = make_sampler(temp=temp)
            sentinel = get_gpu_sentinel()
            priority = GPUPriority.REFLEX if model_type == "cortex" else GPUPriority.REFLECTION

            def _generate_locked():
                acquired = sentinel.acquire(priority=priority, timeout=60)
                if not acquired:
                    raise TimeoutError("NUCLEUS GPU Sentinel timeout during GENERATE")
                try:
                    # Semantic Anchor Refresh & Formatting
                    p, s = self._apply_anchor(prompt, system_prompt, model_type)
                    final_prompt = self._format_prompt(p, s, prefill=kwargs.get("prefill"))
                    
                    return generate(
                        self.models[model_type]["model"],
                        self.models[model_type]["tokenizer"],
                        prompt=final_prompt,
                        max_tokens=kwargs.get("max_tokens", 512),
                        sampler=sampler,
                        verbose=False
                    )
                finally:
                    sentinel.release()

            response = await asyncio.to_thread(_generate_locked)
            return response.strip()
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            entry = self._ensure_model_entry(model_type)
            entry["cache"] = None
            entry["last_error"] = f"{type(e).__name__}: {e}"
            if isinstance(e, (ImportError, AttributeError)):
                entry["loaded"] = False
            _record_nucleus_degradation(
                e,
                action=(
                    "returned explicit nucleus error text, cleared volatile cache, "
                    "and left the lane marked for retry"
                ),
                extra={"model": model_type, "origin": origin, "phase": "text_generate"},
            )
            logger.error("Nucleus inference failed: %s", e)
            return f"Nucleus Error: {str(e)}"

    async def generate_stream_async(self, prompt: str, system_prompt: str | None = None, **kwargs):
        """Streaming version of generate_text_async."""
        origin = kwargs.get("origin", "unknown")
        model_type = self._select_model_type(origin)

        await self.load_model(model_type)
        if not self._ensure_model_entry(model_type).get("loaded"):
            model_type = "brainstem" if model_type == "cortex" else "cortex"
            await self.load_model(model_type)
            if not self._ensure_model_entry(model_type).get("loaded"):
                _record_nucleus_degradation(
                    RuntimeError("both nucleus model lanes are offline"),
                    action=(
                        "ended stream with deterministic offline marker after both "
                        "nucleus lanes failed availability checks"
                    ),
                    extra={"requested_model": model_type, "origin": origin, "phase": "stream_load"},
                )
                yield "[NUCLEUS ERROR] Internal inference offline."
                return

        try:
            import mlx.core as mx
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step

            # Phase 7: Semantic Anchor Refresh & Formatting
            p, s = self._apply_anchor(prompt, system_prompt, model_type)
            full_prompt = self._format_prompt(p, s, prefill=kwargs.get("prefill"))
            model_entry = self._ensure_model_entry(model_type)
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            temp = self._resolve_temperature(kwargs, model_type=model_type, phase="stream_generate")
            sampler = make_sampler(temp=temp)
            tokens = mx.array(tokenizer.encode(full_prompt))
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            _record_nucleus_degradation(
                e,
                action=(
                    "ended stream with explicit nucleus error before token generation "
                    "and preserved caller control flow"
                ),
                extra={"model": model_type, "origin": origin, "phase": "stream_prepare"},
            )
            yield f"[NUCLEUS ERROR] {e}"
            return

        # KV-Cache management — clear stale cache to bound memory growth.
        # generate_step creates a fresh cache per call; persisting one across
        # calls without explicit invalidation leaks GPU RAM proportional to
        # context length * num_layers.  Reset it each generation.
        if model_entry.get("cache") is not None:
            model_entry["cache"] = None
            try:
                if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                else:
                    mx.clear_cache()
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_nucleus_degradation(
                    e,
                    severity="warning",
                    action=(
                        "continued streaming with stale-cache reference cleared "
                        "after MLX cache cleanup failed"
                    ),
                    extra={"model": model_type, "phase": "stream_cache_clear"},
                )

        max_tokens = kwargs.get("max_tokens", 1024)
        
        # Generator for streaming
        def _stream_gen():
            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel
            sentinel = get_gpu_sentinel()
            priority = GPUPriority.REFLEX if model_type == "cortex" else GPUPriority.REFLECTION
            
            acquired = sentinel.acquire(priority=priority, timeout=60)
            if not acquired:
                _record_nucleus_degradation(
                    TimeoutError("NUCLEUS GPU Sentinel timeout during STREAM"),
                    action="ended stream with explicit GPU timeout marker",
                    extra={"model": model_type, "phase": "stream_gpu_acquire"},
                )
                yield "[NUCLEUS ERROR] GPU Sentinel timeout during STREAM"
                return

            try:
                # Use persistent cache if available
                cache = model_entry.get("cache")
                
                for response in generate_step(model, tokenizer, tokens, sampler=sampler, cache=cache):
                    if response.token >= tokenizer.eos_token_id:
                        break
                    
                    # --- Phase 7: GPU Pre-emption Yield ---
                    if priority == GPUPriority.REFLECTION and sentinel.should_yield():
                        logger.warning("🧠 [NUCLEUS] Pre-empted by REFLEX task. Yielding GPU.")
                        yield "... [Pausing for sensory reflex] ..."
                        break

                    yield response.text
                    if response.count >= max_tokens:
                        yield "\n\n[MAX_TOKENS_REACHED]"
                        # Save cache back for persistence
                        model_entry["cache"] = cache
                        break
                
                # Also save on normal completion
                model_entry["cache"] = cache
            finally:
                sentinel.release()

        # ── [STABILITY v52] Non-blocking Streaming Bridge ───────────────
        # MLX generate_step is a blocking CPU/GPU operation. We must offload
        # it to a thread and pipe tokens back via a Queue to avoid 
        # stalling the main event loop and motor reflexes.
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _thread_worker():
            try:
                for chunk in _stream_gen():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_nucleus_degradation(
                    e,
                    action="ended stream from worker thread with explicit nucleus error marker",
                    extra={"model": model_type, "origin": origin, "phase": "stream_worker"},
                )
                logger.error("Nucleus stream thread failed: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, f"[NUCLEUS ERROR] {e}")
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Run in executor to avoid blocking the loop
        asyncio.create_task(asyncio.to_thread(_thread_worker))

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def _resolve_temperature(self, kwargs: dict[str, Any], *, model_type: str, phase: str) -> float:
        temp = kwargs.get("temp", kwargs.get("temperature"))
        if temp is None:
            try:
                from core.container import ServiceContainer
                homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
                if homeostasis:
                    temp = homeostasis.get_modifiers().temperature_mod * 0.7
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                _record_nucleus_degradation(
                    e,
                    severity="warning",
                    action="used default sampler temperature after homeostatic coupling lookup failed",
                    extra={"model": model_type, "phase": phase},
                )
                capture_and_log(e, {'module': __name__})

        try:
            value = float(temp if temp is not None else 0.7)
        except (TypeError, ValueError) as e:
            _record_nucleus_degradation(
                e,
                severity="warning",
                action="used default sampler temperature after invalid caller temperature",
                extra={"model": model_type, "phase": phase, "temperature": repr(temp)},
            )
            value = 0.7
        return min(2.0, max(0.0, value))

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        return await self.generate_text_async(prompt, system_prompt, **kwargs)

    async def unload_models(self):
        """Force unload all internal models and clear caches."""
        logger.info("🧠 [NUCLEUS] Unloading all internal models...")
        for _name, entry in self.models.items():
            entry["model"] = None
            entry["tokenizer"] = None
            entry["loaded"] = False
            entry["cache"] = None
        
        try:
            import mlx.core as mx

            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel
            sentinel = get_gpu_sentinel()
            if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                try:
                    if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                        mx.metal.clear_cache()
                    else:
                        mx.clear_cache()
                finally:
                    sentinel.release()
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            _record_nucleus_degradation(
                e,
                severity="warning",
                action=(
                    "left all model references cleared and skipped MLX cache reclamation "
                    "because the cache-clear dependency was unavailable"
                ),
                extra={"phase": "unload_models"},
            )
            logger.debug("[NUCLEUS] Cache clear skipped: %s", e)

    # --- Abstract Method Implementations ---

    async def generate_stream(self, prompt: str, system_prompt: str | None = None, model: str | None = None, **kwargs):
        """Implements abstract generate_stream by delegating to generate_stream_async."""
        async for chunk in self.generate_stream_async(prompt, system_prompt, **kwargs):
            yield chunk

    def generate_text(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Synchronous wrapper for generate_text_async."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # This is tricky in async environments, but for CLI/scripts it works
                return "[NUCLEUS ERROR] Sync call in async loop."
            return loop.run_until_complete(self.generate_text_async(prompt, system_prompt))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return asyncio.run(self.generate_text_async(prompt, system_prompt))

    def generate_json(self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None, model: str | None = None) -> dict[str, Any]:
        """Synchronous wrapper for JSON extraction."""
        from core.utils.json_utils import extract_json
        text = self.generate_text(prompt, system_prompt)
        return extract_json(text)
