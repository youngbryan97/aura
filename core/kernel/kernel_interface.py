"""core/kernel/kernel_interface.py — The Unification Adapter.

This is the file that ends the two-pipeline problem.

Before this file, RobustOrchestrator._handle_incoming_message() went directly
to CognitiveIntegration, which called its own phase pipeline, which had no
connection to the Unitary Kernel. The kernel's phases were running in theory
but never receiving real user messages.

After this file, the orchestrator's cognitive call routes through the kernel.
The CognitiveIntegration path is preserved as a fallback — nothing breaks.

Usage in RobustOrchestrator._handle_incoming_message():

    # Replace the direct CognitiveIntegration call with:
    from core.kernel.kernel_interface import KernelInterface
    ki = KernelInterface.get_instance()
    if ki.is_ready():
        response = await ki.process(message, origin=origin)
    else:
        # Existing CognitiveIntegration fallback
        response = await cog.process_turn(message, payload_context)

Or, drop-in via the factory in orchestrator boot:

    await KernelInterface.attach_to_orchestrator(orchestrator)

The KernelInterface also exposes the feedback loop so you can query it from
anywhere in the codebase:

    from core.kernel.kernel_interface import KernelInterface
    ki = KernelInterface.get_instance()
    ki.print_loop()              # print last 5 ticks to stdout
    state = ki.loop_state()      # dict with phi, valence, mood, etc.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from core.runtime import service_access

if TYPE_CHECKING:
    from core.consciousness.unified_audit import AuditReport
    from core.kernel.aura_kernel import AuraKernel
    from core.phases.phi_consciousness import PhiConsciousnessPhase as PhiConsciousness


def get_audit_suite() -> Any:
    """Import and return the unified audit suite, or None if unavailable."""
    try:
        from core.consciousness.unified_audit import get_audit_suite as _get_suite
        return _get_suite()
    except ImportError:
        return None

logger = logging.getLogger("Aura.KernelInterface")


class KernelInterface:
    """
    Thin adapter between RobustOrchestrator and AuraKernel.

    Responsibilities:
      1. Hold a reference to the booted AuraKernel
      2. Route process() calls through kernel.tick()
      3. Return the response string to the orchestrator
      4. Expose feedback loop queries
    """

    _instance: Optional["KernelInterface"] = None

    def __init__(self):
        """Initialize the interface in an unbooted state."""
        self._kernel: Optional["AuraKernel"] = None
        self._ready:  bool = False

    @classmethod
    def get_instance(cls) -> "KernelInterface":
        """Return the process-wide KernelInterface singleton, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        assert cls._instance is not None
        return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the process-wide singleton, primarily for controlled test boots."""
        inst = cls._instance
        cls._instance = None
        if inst is not None:
            await inst.shutdown()

    # ── Lifecycle ────────────────────────────────────────────────────────────────

    async def boot(
        self,
        kernel=None,
        vault=None,
        config=None,
    ) -> "KernelInterface":
        """
        Boot the kernel and mark the interface as ready.

        Can be called with an already-constructed AuraKernel, or will
        create one from vault + config if not provided.
        """
        if self._ready:
            logger.warning("KernelInterface.boot() called on already-ready interface.")
            return self

        if kernel is not None:
            self._kernel = kernel
        else:
            if vault is None or config is None:
                raise ValueError(
                    "KernelInterface.boot() requires either a kernel or (vault, config)"
                )
            from core.kernel.aura_kernel import AuraKernel
            self._kernel = AuraKernel(config=config, vault=vault)

        if not self._kernel.state:
            await self._kernel.boot()

        self._ready = True
        logger.info("KernelInterface ready. LLM organ: %s",
                    self._kernel.organs.get("llm", {}).instance.__class__.__name__
                    if hasattr(self._kernel.organs.get("llm", {}), "instance") else "unknown")
        return self

    @classmethod
    async def attach_to_orchestrator(cls, orchestrator: Any) -> "KernelInterface":
        """
        Convenience factory: pulls vault + config from orchestrator,
        boots the kernel, registers it in ServiceContainer, and returns self.

        Call this once in orchestrator._async_init_subsystems().
        """
        ki = cls.get_instance()

        try:
            from core.container import ServiceContainer
            from core.kernel.aura_kernel import KernelConfig

            def _bind() -> None:
                orchestrator.kernel_interface = ki
                if ki._kernel is not None:
                    ServiceContainer.register_instance("aura_kernel", ki._kernel)
                ServiceContainer.register_instance("kernel_interface", ki)
                logger.info("KernelInterface attached to orchestrator.")

            if ki._ready:
                _bind()
                return ki

            vault = service_access.resolve_state_repository(orchestrator, default=None)

            if vault is None:
                logger.error("KernelInterface: no StateRepository found. Skipping boot.")
                return ki

            config = KernelConfig()
            await ki.boot(vault=vault, config=config)
            _bind()
        except Exception as e:
            record_degradation('kernel_interface', e)
            logger.error("KernelInterface.attach_to_orchestrator failed: %s", e)

        return ki

    def is_ready(self) -> bool:
        """Return True if the kernel has been successfully booted."""
        return self._ready and self._kernel is not None

    async def shutdown(self) -> None:
        """Shut down the attached kernel and mark the interface unready."""
        kernel = self._kernel
        self._ready = False
        if kernel is None:
            return
        shutdown = getattr(kernel, "shutdown", None)
        if callable(shutdown):
            await shutdown()
        self._kernel = None

    # ── Processing ───────────────────────────────────────────────────────────────

    async def process(
        self,
        message: str,
        origin: str = "user",
        inject_to_working_memory: bool = True,
        priority: bool = False,
    ) -> str:
        """
        Route a message through kernel.tick() and return Aura's response.

        This is the drop-in replacement for:
            response = await cog.process_turn(message, payload_context)

        Returns the response string, or a fallback if the kernel isn't ready.
        """
        if not self.is_ready():
            logger.warning("KernelInterface.process() called before boot.")
            return "I'm still initializing. Give me a moment."

        # Update the orchestrator's user-interaction timestamp so idle
        # detectors (substrate decay, sleep triggers, proactive presence)
        # know the user is present. Without this, the kernel path bypasses
        # the orchestrator entirely and the system thinks it's been idle
        # for the entire session.
        if origin in ("user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"):
            import time as _time
            try:
                from core.container import ServiceContainer
                orch = ServiceContainer.get("orchestrator", default=None)
                if orch is not None:
                    orch._last_user_interaction_time = _time.time()
            except Exception:
                pass  # no-op: intentional
            # Signal the subcortical core that a stimulus has arrived.
            # This raises arousal, opens the thalamic gate, and restores
            # full mesh/substrate gain for the duration of user interaction.
            try:
                from core.consciousness.subcortical_core import get_subcortical_core
                get_subcortical_core().receive_stimulus(intensity=1.0, source=origin)
            except Exception:
                pass  # no-op: intentional

        # Inject user turn into working memory before tick so history builds
        if inject_to_working_memory and self._kernel.state is not None:
            import time
            wm = self._kernel.state.cognition.working_memory
            # Avoid duplicating if routing phase or upstream already added it
            last = wm[-1] if wm else {}
            if last.get("role") != "user" or last.get("content") != message:
                wm.append({
                    "role":      "user",
                    "content":   message,
                    "timestamp": time.time(),
                    "origin":    origin,
                })

        # Propagate origin to kernel state so phases (especially ResponseGeneration)
        # can distinguish user-facing vs autonomous/background ticks and route
        # to the correct LLM tier (32B for user, 7B brainstem for background).
        if self._kernel.state is not None:
            self._kernel.state.cognition.current_origin = origin

        try:
            tick_entry = await self._kernel.tick(message, priority=priority)
            if tick_entry is not None:
                response = tick_entry.response_preview
                # response_preview is capped at 120 chars for the observer;
                # get the full response from state
                full_response = self._kernel.state.cognition.last_response
                return full_response or response or ""
        except asyncio.TimeoutError:
            logger.warning("KernelInterface.process() foreground timeout for origin=%s", origin)
            raise
        except Exception as e:
            record_degradation('kernel_interface', e)
            logger.error("KernelInterface.process() tick failed: %s", e, exc_info=True)

        return "Something went wrong in my thinking. Please try again."

    # ── Feedback loop queries ────────────────────────────────────────────────────

    def print_loop(self, n: int = 5) -> None:
        """Print the last N ticks of the causal chain to stdout."""
        if not self.is_ready():
            print("[KernelInterface] Not ready.")
            return
        self._kernel.print_loop(n)

    def loop_state(self) -> dict:
        """Return the current live state of the feedback loop."""
        if not self.is_ready():
            return {"status": "not_ready"}
        
        # Don't leak raw kernel state directly; create a sanitized copy
        raw_state = self._kernel.loop_state()
        state = {
            "phi":     float(raw_state.get("phi", 0.0)),
            "valence": float(raw_state.get("valence", 0.0)),
            "mood":    str(raw_state.get("mood", "neutral")),
            "arousal": float(raw_state.get("arousal", 0.0)),
            "status":  str(raw_state.get("status", "idle")),
        }
        
        # Inject stability health
        try:
            from core.container import ServiceContainer
            guardian = ServiceContainer.get("stability_guardian", default=None)
            if guardian:
                state["stability"] = guardian.get_health_summary()
        except Exception:
            logger.debug("StabilityGuardian: Failed to get health summary for loop state.")
            
        return state

    async def run_consciousness_audit(self) -> "AuditReport":
        """Triggers a full consciousness audit and returns the report."""
        audit = get_audit_suite()
        if audit is None:
            logger.warning("Consciousness audit suite not available.")
            return None
        return await audit.run()

    def get_kernel(self, trusted: bool = False) -> Optional["AuraKernel"]:
        """Access the kernel. Requires explicit trust flag to prevent accidental exposure."""
        if not trusted:
            raise PermissionError("Direct kernel access is restricted to trusted internal components.")
        return self._kernel

    @property
    def kernel(self) -> Optional["AuraKernel"]:
        """Direct kernel access for advanced use cases."""
        return self._kernel


# ── Singleton ─────────────────────────────────────────────────────────────────

_ki: Optional[KernelInterface] = None


def get_kernel_interface() -> KernelInterface:
    """Return the process-wide KernelInterface singleton."""
    global _ki
    if _ki is None:
        _ki = KernelInterface.get_instance()
    return _ki
