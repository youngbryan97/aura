from __future__ import annotations

import asyncio
import functools
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.progress_bound import (
    await_while_the_task_moves,
    run_on_a_thread_while_it_works,
)

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

_ORGAN_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)
_CRITICAL_ORGANS = frozenset({"brain", "llm", "memory", "continuity"})
_FALLBACK_TYPES = (FallbackLLM, FallbackNeural, FallbackOrgan, FallbackVision, FallbackVoice)


def _organ_severity(name: str) -> Severity:
    return "critical" if name in _CRITICAL_ORGANS else "degraded"


def _record_organ_degradation(
    error: BaseException,
    *,
    organ: str,
    action: str,
    severity: Severity | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "organs",
        error,
        severity=severity or _organ_severity(organ),
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=(severity or _organ_severity(organ)) in {"degraded", "critical"},
        extra={"organ": organ, **dict(extra or {})},
    )


@dataclass
class OrganStub:
    """Lazy-loading wrapper for high-latency hardware or external subsystems.

    Each organ resolves its real implementation from the service container
    or kernel registry. If that fails (timeout, missing dependency, crash),
    a minimal fallback keeps the kernel tick alive until the real subsystem
    comes online.
    """

    name: str
    kernel: AuraKernel
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    instance: Any = None
    fallback_used: bool = False
    resolved_kind: str = ""
    failure_reason: str = ""
    load_attempts: int = 0

    async def load(self) -> None:
        """Resolve the real subsystem, falling back on timeout or error."""
        logger.info("Loading organ: %s...", self.name)
        self.load_attempts += 1
        self.fallback_used = False
        self.failure_reason = ""
        try:
            async with asyncio.timeout(5.0):
                self.instance = await self._resolve()
        except TimeoutError as exc:
            logger.warning("Organ %s load TIMEOUT — using fallback.", self.name)
            self.instance = self._fallback_for_name()
            self._mark_fallback(
                exc,
                action="organ load timed out and resolved to bounded fallback",
            )
        except _ORGAN_RECOVERABLE_ERRORS as e:
            logger.exception("Organ %s load failed: %s — using fallback.", self.name, e)
            self.instance = self._fallback_for_name()
            self._mark_fallback(
                e,
                action="organ load failed and resolved to bounded fallback",
            )
        else:
            self._mark_resolved(self.instance)

        self.ready.set()
        logger.debug("Organ %s load complete.", self.name)

    def _fallback_for_name(self) -> Any:
        if self.name in {"brain", "llm"}:
            return FallbackLLM()
        if self.name == "vision":
            return FallbackVision()
        if self.name == "neural":
            return FallbackNeural()
        if self.name == "voice":
            return FallbackVoice()
        return FallbackOrgan()

    def _mark_resolved(self, instance: Any) -> None:
        self.resolved_kind = type(instance).__qualname__ if instance is not None else "None"
        self.fallback_used = isinstance(instance, _FALLBACK_TYPES)
        if self.fallback_used and not self.failure_reason:
            self.failure_reason = f"{self.name} resolved to {self.resolved_kind}"

    def _mark_fallback(self, error: BaseException, *, action: str) -> None:
        self._mark_resolved(self.instance)
        self.failure_reason = f"{type(error).__qualname__}: {str(error)[:200]}"
        _record_organ_degradation(
            error,
            organ=self.name,
            action=action,
            extra={
                "fallback_type": self.resolved_kind,
                "load_attempts": self.load_attempts,
            },
        )

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.ready.is_set(),
            "fallback_used": self.fallback_used,
            "resolved_kind": self.resolved_kind,
            "failure_reason": self.failure_reason,
            "load_attempts": self.load_attempts,
        }

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
        lookup_failed = False
        try:
            from core.brain.llm.llm_router import IntelligentLLMRouter as LLMRouter
            instance = self.kernel.get(LLMRouter)
        except _ORGAN_RECOVERABLE_ERRORS as exc:
            lookup_failed = True
            _record_organ_degradation(
                exc,
                organ=self.name,
                action="resolved LLM organ to fallback after router lookup failed",
                severity="critical",
            )
            instance = None

        if instance:
            if not hasattr(instance, "think") and hasattr(instance, "generate"):
                instance.think = instance.generate
            return instance
        if not lookup_failed:
            _record_organ_degradation(
                RuntimeError("LLM router unavailable"),
                organ=self.name,
                action="resolved LLM organ to fallback because no live router was available",
                severity="critical",
            )
        return FallbackLLM()

    async def _resolve_neural(self) -> Any:
        safe_boot = os.getenv("AURA_SAFE_BOOT_DESKTOP", "0") == "1"
        try:
            def _build():
                from core.senses.neural_bridge import NeuralBridge
                return NeuralBridge(lightweight_mode=safe_boot)

            instance = await run_on_a_thread_while_it_works(
                _build, stall_s=1.5, name="organs.neural_bridge.build"
            )
            await await_while_the_task_moves(
                instance.load(), stall_s=2.5 if safe_boot else 4.0, name="organs.neural_bridge.load"
            )
            return instance
        except _ORGAN_RECOVERABLE_ERRORS as e:
            _record_organ_degradation(
                e,
                organ=self.name,
                action="resolved neural organ to fallback after NeuralBridge load failed",
            )
            logger.warning("NeuralBridge load failed: %s", e)
            return FallbackNeural()

    async def _resolve_voice(self) -> Any:
        if ServiceContainer:
            try:
                instance = await run_on_a_thread_while_it_works(
                    functools.partial(ServiceContainer.get, "voice_engine", default=None),
                    stall_s=2.0,
                    name="organs.voice_engine",
                )
                if instance:
                    return instance
            except TimeoutError as exc:
                _record_organ_degradation(
                    exc,
                    organ=self.name,
                    action="resolved voice organ to fallback after voice engine lookup timed out",
                )
                logger.warning("VoiceEngine resolution TIMEOUT.")
            except _ORGAN_RECOVERABLE_ERRORS as exc:
                _record_organ_degradation(
                    exc,
                    organ=self.name,
                    action="resolved voice organ to fallback after voice engine lookup failed",
                )
                logger.warning("VoiceEngine resolution failed: %s", exc)
        return FallbackVoice()

    async def _resolve_continuity(self) -> Any:
        try:
            from core.cybernetics.knowledge_continuity import KnowledgeContinuity
            instance = KnowledgeContinuity(self.kernel)
            await asyncio.wait_for(instance.load(), timeout=3.0)
            return instance
        except _ORGAN_RECOVERABLE_ERRORS as e:
            _record_organ_degradation(
                e,
                organ=self.name,
                action="resolved continuity organ to fallback after continuity load failed",
            )
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
        except _ORGAN_RECOVERABLE_ERRORS as exc:
            _record_organ_degradation(
                exc,
                organ=self.name,
                action="resolved optional organ module to fallback after import/load failed",
                severity="warning",
                extra={"module_path": module_path, "class_name": class_name},
            )
            return FallbackOrgan()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_from_container(self, service_name: str) -> Any:
        if ServiceContainer:
            try:
                return ServiceContainer.get(service_name, default=None)
            except _ORGAN_RECOVERABLE_ERRORS as exc:
                _record_organ_degradation(
                    exc,
                    organ=self.name,
                    action="resolved organ container dependency to fallback after lookup failed",
                    extra={"service_name": service_name},
                )
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
        try:
            result = hook()
            if asyncio.iscoroutine(result):
                await result
        except _ORGAN_RECOVERABLE_ERRORS as exc:
            _record_organ_degradation(
                exc,
                organ=self.name,
                action="organ shutdown hook failed after kernel stop requested cleanup",
                severity="warning",
                extra={"resolved_kind": self.resolved_kind},
            )
            logger.warning("Organ %s shutdown hook failed: %s", self.name, exc)
