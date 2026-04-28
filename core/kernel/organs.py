from core.runtime.errors import record_degradation
import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

try:
    from core.container import ServiceContainer
except ImportError:
    ServiceContainer = None

from core.kernel.organ_fallbacks import (
    FallbackLLM,
    FallbackNeural,
    FallbackOrgan,
    FallbackVision,
    FallbackVoice,
)

logger = logging.getLogger(__name__)


@dataclass
class OrganStub:
    """Lazy-loading wrapper for high-latency hardware or external subsystems.

    Each organ resolves its real implementation from the service container
    or kernel registry. If that fails (timeout, missing dependency, crash),
    a minimal fallback keeps the kernel tick alive until the real subsystem
    comes online.
    """

    name: str
    kernel: "AuraKernel"
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    instance: Any = None

    async def load(self) -> None:
        """Resolve the real subsystem, falling back on timeout or error."""
        logger.info("Loading organ: %s...", self.name)
        try:
            async with asyncio.timeout(5.0):
                self.instance = await self._resolve()
        except asyncio.TimeoutError:
            logger.warning("Organ %s load TIMEOUT — using fallback.", self.name)
            self.instance = FallbackOrgan()
        except Exception as e:
            record_degradation('organs', e)
            logger.exception("Organ %s load failed: %s — using fallback.", self.name, e)
            self.instance = FallbackOrgan()

        self.ready.set()
        logger.debug("Organ %s load complete.", self.name)

    async def _resolve(self) -> Any:
        """Attempt to load the real subsystem for this organ."""
        name = self.name

        if name in ("brain", "llm"):
            return await self._resolve_llm()
        elif name == "vision":
            return self._resolve_from_container("vision_presence") or FallbackVision()
        elif name == "neural":
            return await self._resolve_neural()
        elif name == "cookie":
            return await self._resolve_module("core.autonomy.reflective_cookie", "ReflectiveCookie")
        elif name == "prober":
            return await self._resolve_module("core.brain.alignment_prober", "EmpathyProber")
        elif name == "tricorder":
            return await self._resolve_module("core.cybernetics.tricorder", "Tricorder")
        elif name == "ice_layer":
            return await self._resolve_module("core.cybernetics.ice_layer", "ICELayer")
        elif name == "omni_tool":
            return await self._resolve_module("core.cybernetics.omni_tool", "OmniTool")
        elif name == "memory":
            return self._resolve_from_container("memory_facade")
        elif name == "voice":
            return await self._resolve_voice()
        elif name == "metabolism":
            return self._resolve_from_container("metabolic_monitor")
        elif name == "continuity":
            return await self._resolve_continuity()
        else:
            logger.warning("Unknown organ '%s' — using generic fallback.", name)
            return FallbackOrgan()

    # ── Organ-specific resolvers ────────────────────────────────────────

    async def _resolve_llm(self) -> Any:
        try:
            from core.brain.llm.llm_router import IntelligentLLMRouter as LLMRouter
            instance = self.kernel.get(LLMRouter)
        except Exception:
            instance = None

        if instance:
            if not hasattr(instance, "think") and hasattr(instance, "generate"):
                instance.think = instance.generate
            return instance
        return FallbackLLM()

    async def _resolve_neural(self) -> Any:
        safe_boot = os.getenv("AURA_SAFE_BOOT_DESKTOP", "0") == "1"
        try:
            def _build():
                from core.senses.neural_bridge import NeuralBridge
                return NeuralBridge(lightweight_mode=safe_boot)

            instance = await asyncio.wait_for(asyncio.to_thread(_build), timeout=1.5)
            await asyncio.wait_for(instance.load(), timeout=2.5 if safe_boot else 4.0)
            return instance
        except Exception as e:
            record_degradation('organs', e)
            logger.warning("NeuralBridge load failed: %s", e)
            return FallbackNeural()

    async def _resolve_voice(self) -> Any:
        if ServiceContainer:
            try:
                instance = await asyncio.wait_for(
                    asyncio.to_thread(ServiceContainer.get, "voice_engine", default=None),
                    timeout=2.0,
                )
                if instance:
                    return instance
            except asyncio.TimeoutError:
                logger.warning("VoiceEngine resolution TIMEOUT.")
        return FallbackVoice()

    async def _resolve_continuity(self) -> Any:
        try:
            from core.cybernetics.knowledge_continuity import KnowledgeContinuity
            instance = KnowledgeContinuity(self.kernel)
            await asyncio.wait_for(instance.load(), timeout=3.0)
            return instance
        except Exception as e:
            record_degradation('organs', e)
            logger.warning("Continuity organ load failed: %s", e)
            return FallbackOrgan()

    async def _resolve_module(self, module_path: str, class_name: str) -> Any:
        """Generic resolver: import class, construct with kernel, call load()."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls(self.kernel)
            await instance.load()
            return instance
        except Exception:
            return FallbackOrgan()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_from_container(self, service_name: str) -> Any:
        if ServiceContainer:
            return ServiceContainer.get(service_name, default=None)
        return None

    def get_instance(self) -> Any:
        """Returns the organ instance. Raises if not yet loaded."""
        if not self.ready.is_set():
            raise RuntimeError(f"Attempted to access organ '{self.name}' before it was READY.")
        return self.instance

    async def shutdown(self) -> None:
        """Stop the backing organ instance if it exposes a shutdown hook."""
        inst = self.instance
        if inst is None:
            return
        hook = getattr(inst, "shutdown", None) or getattr(inst, "stop", None)
        if not callable(hook):
            return
        result = hook()
        if asyncio.iscoroutine(result):
            await result
