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

import logging
import time
from typing import TYPE_CHECKING, Any

from core.runtime import service_access
from core.runtime.errors import (
    DependencyUnavailable,
    FallbackClassification,
    record_degradation,
)

if TYPE_CHECKING:
    from core.consciousness.unified_audit import AuditReport
    from core.kernel.aura_kernel import AuraKernel


def get_audit_suite() -> Any:
    """Import and return the unified audit suite, or None if unavailable."""
    try:
        from core.consciousness.unified_audit import get_audit_suite as _get_suite

        return _get_suite()
    except ImportError:
        return None


logger = logging.getLogger("Aura.KernelInterface")

_KERNEL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    ConnectionError,
)
_USER_ORIGINS = frozenset(
    {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}
)
MAX_KERNEL_MESSAGE_CHARS = 60_000
MAX_ORIGIN_CHARS = 80
MAX_CONSECUTIVE_TICK_FAILURES = 3


def _emit_kernel_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "kernel_interface",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation(
            "kernel_interface",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action or "captured kernel-interface fault",
        )


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _mark_kernel_health(status: str, reason: str = "", impact: str = "") -> None:
    try:
        from core.runtime.errors import get_subsystem_registry

        health = get_subsystem_registry().register("kernel_interface")
        if status == "healthy":
            health.mark_ok()
        elif status == "unavailable":
            health.mark_unavailable(reason)
        elif status == "failed_closed":
            health.status = "failed_closed"
            health.reason = reason
            health.impact = impact
            health.last_failed_at = time.time()
        else:
            health.mark_degraded(reason, impact=impact)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("KernelInterface health update failed: %s", exc)


class KernelInterface:
    """
    Thin adapter between RobustOrchestrator and AuraKernel.

    Responsibilities:
      1. Hold a reference to the booted AuraKernel
      2. Route process() calls through kernel.tick()
      3. Return the response string to the orchestrator
      4. Expose feedback loop queries
    """

    _instance: KernelInterface | None = None

    def __init__(self):
        """Initialize the interface in an unbooted state."""
        self._kernel: AuraKernel | None = None
        self._ready: bool = False
        self._last_fault: str = ""
        self._last_ready_at: float = 0.0
        self._last_failure_at: float = 0.0
        self._consecutive_tick_failures: int = 0

    @classmethod
    def get_instance(cls) -> KernelInterface:
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
    ) -> KernelInterface:
        """
        Boot the kernel and mark the interface as ready.

        Can be called with an already-constructed AuraKernel, or will
        create one from vault + config if not provided.
        """
        if self._ready:
            logger.warning("KernelInterface.boot() called on already-ready interface.")
            return self

        try:
            if kernel is not None:
                self._kernel = kernel
            else:
                if vault is None or config is None:
                    raise ValueError(
                        "KernelInterface.boot() requires either a kernel or (vault, config)"
                    )
                from core.kernel.aura_kernel import AuraKernel

                self._kernel = AuraKernel(config=config, vault=vault)

            if not getattr(self._kernel, "state", None):
                boot = getattr(self._kernel, "boot", None)
                if not callable(boot):
                    raise DependencyUnavailable("AuraKernel has no callable boot()")
                await boot()

            tick = getattr(self._kernel, "tick", None)
            if not callable(tick):
                raise DependencyUnavailable("AuraKernel has no callable tick()")
            if getattr(self._kernel, "state", None) is None:
                raise DependencyUnavailable("AuraKernel boot completed without state")
        except _KERNEL_RECOVERABLE_ERRORS as exc:
            self._ready = False
            self._kernel = None
            self._last_fault = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
            self._last_failure_at = time.time()
            _mark_kernel_health(
                "unavailable",
                self._last_fault,
                "kernel boot failed closed before foreground routing",
            )
            _emit_kernel_fault(
                exc,
                action="failed closed during kernel boot; foreground routing remains on fallback lane",
                severity="critical",
                stage="boot",
            )
            raise

        self._ready = True
        self._last_fault = ""
        self._last_ready_at = time.time()
        self._consecutive_tick_failures = 0
        _mark_kernel_health("healthy")
        llm_organ = getattr(self._kernel, "organs", {}).get("llm", {})
        logger.info(
            "KernelInterface ready. LLM organ: %s",
            llm_organ.instance.__class__.__name__ if hasattr(llm_organ, "instance") else "unknown",
        )
        return self

    @classmethod
    async def attach_to_orchestrator(cls, orchestrator: Any) -> KernelInterface:
        """
        Convenience factory: pulls vault + config from orchestrator,
        boots the kernel, registers it in ServiceContainer, and returns self.

        Call this once in orchestrator._async_init_subsystems().
        """
        ki = cls.get_instance()

        def _bind() -> None:
            orchestrator.kernel_interface = ki
            if ki._kernel is not None:
                ServiceContainer.register_instance("aura_kernel", ki._kernel)
            ServiceContainer.register_instance("kernel_interface", ki)
            logger.info("KernelInterface attached to orchestrator.")

        try:
            from core.container import ServiceContainer

            if ki._ready:
                _bind()
                return ki

            vault = service_access.resolve_state_repository(orchestrator, default=None)

            if vault is None:
                exc = DependencyUnavailable("StateRepository is required to boot AuraKernel")
                ki._ready = False
                ki._kernel = None
                ki._last_fault = str(exc)
                ki._last_failure_at = time.time()
                _mark_kernel_health(
                    "unavailable",
                    ki._last_fault,
                    "kernel interface registered unready so boot health can fail explicitly",
                )
                _emit_kernel_fault(
                    exc,
                    action="registered unready kernel interface and preserved orchestrator fallback lane",
                    severity="critical",
                    stage="attach_to_orchestrator.state_repository",
                )
                logger.error("KernelInterface: no StateRepository found. Registered unready.")
                _bind()
                return ki

            from core.kernel.aura_kernel import KernelConfig

            config = KernelConfig()
            await ki.boot(vault=vault, config=config)
            _bind()
        except _KERNEL_RECOVERABLE_ERRORS as e:
            ki._ready = False
            ki._last_fault = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
            ki._last_failure_at = time.time()
            _mark_kernel_health(
                "unavailable",
                ki._last_fault,
                "kernel attach failed closed; orchestrator must not route through it",
            )
            _emit_kernel_fault(
                e,
                action="failed closed kernel attach and left interface unready",
                severity="critical",
                stage="attach_to_orchestrator",
            )
            logger.error("KernelInterface.attach_to_orchestrator failed: %s", e)

        return ki

    def is_ready(self) -> bool:
        """Return True if the kernel has been successfully booted."""
        return self._ready and self._kernel is not None

    async def shutdown(self) -> None:
        """Shut down the attached kernel and mark the interface unready."""
        kernel = self._kernel
        self._ready = False
        self._last_fault = ""
        self._consecutive_tick_failures = 0
        if kernel is None:
            return
        shutdown = getattr(kernel, "shutdown", None)
        if callable(shutdown):
            await shutdown()
        self._kernel = None
        _mark_kernel_health("unavailable", "kernel interface shut down")

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
        message = _safe_text(message, max_chars=MAX_KERNEL_MESSAGE_CHARS)
        origin = _safe_text(origin, default="user", max_chars=MAX_ORIGIN_CHARS)

        if not self.is_ready():
            logger.warning("KernelInterface.process() called before boot.")
            # [STABILITY v55] Return empty so chat.py can fire protected
            # foreground lane instead of showing a canned robot message.
            return ""

        # Update the orchestrator's user-interaction timestamp so idle
        # detectors (substrate decay, sleep triggers, proactive presence)
        # know the user is present. Without this, the kernel path bypasses
        # the orchestrator entirely and the system thinks it's been idle
        # for the entire session.
        if origin in _USER_ORIGINS:
            try:
                from core.container import ServiceContainer

                orch = ServiceContainer.get("orchestrator", default=None)
                if orch is not None:
                    orch._last_user_interaction_time = time.time()
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _emit_kernel_fault(
                    exc,
                    action="continued kernel processing without orchestrator idle timestamp update",
                    severity="warning",
                    stage="process.touch_orchestrator",
                )
            # Signal the subcortical core that a stimulus has arrived.
            # This raises arousal, opens the thalamic gate, and restores
            # full mesh/substrate gain for the duration of user interaction.
            try:
                from core.consciousness.subcortical_core import get_subcortical_core

                get_subcortical_core().receive_stimulus(intensity=1.0, source=origin)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _emit_kernel_fault(
                    exc,
                    action="continued kernel processing without subcortical stimulus side-effect",
                    severity="warning",
                    stage="process.subcortical_stimulus",
                )

        try:
            # Inject user turn into working memory before tick so history builds.
            if inject_to_working_memory and self._kernel.state is not None:
                cognition = getattr(self._kernel.state, "cognition", None)
                if cognition is None:
                    raise AttributeError("kernel state has no cognition")
                wm = getattr(cognition, "working_memory", None)
                if wm is None:
                    wm = []
                    cognition.working_memory = wm
                if not isinstance(wm, list):
                    raise TypeError("kernel working_memory must be a list")
                # Avoid duplicating if routing phase or upstream already added it.
                last = wm[-1] if wm else {}
                if not isinstance(last, dict):
                    last = {}
                if last.get("role") != "user" or last.get("content") != message:
                    wm.append(
                        {
                            "role": "user",
                            "content": message,
                            "timestamp": time.time(),
                            "origin": origin,
                        }
                    )

            # Propagate origin to kernel state so phases can distinguish
            # user-facing vs autonomous/background ticks and route to the
            # correct LLM tier.
            if self._kernel.state is not None:
                cognition = getattr(self._kernel.state, "cognition", None)
                if cognition is None:
                    raise AttributeError("kernel state has no cognition")
                cognition.current_origin = origin
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_fault = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
            self._last_failure_at = time.time()
            _mark_kernel_health(
                "degraded",
                self._last_fault,
                "foreground turn diverted before kernel tick because state handoff failed",
            )
            _emit_kernel_fault(
                exc,
                action="diverted foreground turn before kernel tick because state handoff failed",
                severity="degraded",
                stage="process.state_handoff",
            )
            return ""

        try:
            tick_entry = await self._kernel.tick(message, priority=priority)
            self._consecutive_tick_failures = 0
            self._last_fault = ""
            _mark_kernel_health("healthy")
            if tick_entry is not None:
                response = _safe_text(
                    getattr(tick_entry, "response_preview", ""),
                    max_chars=MAX_KERNEL_MESSAGE_CHARS,
                )
                # response_preview is capped at 120 chars for the observer;
                # get the full response from state
                cognition = getattr(getattr(self._kernel, "state", None), "cognition", None)
                full_response = _safe_text(
                    getattr(cognition, "last_response", ""),
                    max_chars=MAX_KERNEL_MESSAGE_CHARS,
                )
                return full_response or response or ""
        except TimeoutError:
            logger.warning("KernelInterface.process() foreground timeout for origin=%s", origin)
            self._last_fault = "TimeoutError: foreground kernel tick timed out"
            self._last_failure_at = time.time()
            _mark_kernel_health(
                "degraded",
                self._last_fault,
                "foreground timeout propagated for caller retry policy",
            )
            raise
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as e:
            self._consecutive_tick_failures += 1
            self._last_fault = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
            self._last_failure_at = time.time()
            fail_closed = self._consecutive_tick_failures >= MAX_CONSECUTIVE_TICK_FAILURES
            if not fail_closed:
                _mark_kernel_health(
                    "degraded",
                    self._last_fault,
                    "foreground turn diverted to fallback lane after kernel tick failure",
                )
            _emit_kernel_fault(
                e,
                action=(
                    "opened kernel tick circuit and diverted future turns to fallback lane"
                    if fail_closed
                    else "diverted foreground turn to fallback lane after kernel tick failure"
                ),
                severity="critical" if fail_closed else "degraded",
                stage="process.tick",
                extra={"consecutive_tick_failures": self._consecutive_tick_failures},
            )
            if fail_closed:
                self._ready = False
                _mark_kernel_health(
                    "failed_closed",
                    self._last_fault,
                    "kernel tick circuit opened after repeated foreground failures",
                )
            logger.error("KernelInterface.process() tick failed: %s", e, exc_info=True)

        # [STABILITY v55] Return empty so caller can retry/escalate.
        return ""

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

        # Don't leak raw kernel state directly; create a sanitized copy.
        try:
            raw_state = self._kernel.loop_state()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_fault = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
            self._last_failure_at = time.time()
            _mark_kernel_health(
                "degraded",
                self._last_fault,
                "returned bounded degraded loop-state snapshot",
            )
            _emit_kernel_fault(
                exc,
                action="returned bounded degraded loop-state snapshot",
                severity="warning",
                stage="loop_state",
            )
            return {
                "status": "degraded",
                "phi": 0.0,
                "valence": 0.0,
                "mood": "unknown",
                "arousal": 0.0,
                "last_fault": self._last_fault,
            }
        state = {
            "phi": _safe_float(raw_state.get("phi", 0.0)),
            "valence": _safe_float(raw_state.get("valence", 0.0)),
            "mood": _safe_text(raw_state.get("mood", "neutral"), max_chars=80),
            "arousal": _safe_float(raw_state.get("arousal", 0.0)),
            "status": _safe_text(raw_state.get("status", "idle"), max_chars=80),
        }

        # Inject stability health
        try:
            from core.container import ServiceContainer

            guardian = ServiceContainer.get("stability_guardian", default=None)
            if guardian:
                state["stability"] = guardian.get_health_summary()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("StabilityGuardian: Failed to get health summary for loop state: %s", exc)

        return state

    def health_snapshot(self) -> dict[str, Any]:
        """Return the canonical kernel-interface runtime health contract."""
        kernel = self._kernel
        state = getattr(kernel, "state", None) if kernel is not None else None
        return {
            "status": "ready" if self.is_ready() else "not_ready",
            "ready": self.is_ready(),
            "kernel_attached": kernel is not None,
            "state_attached": state is not None,
            "tick_callable": callable(getattr(kernel, "tick", None))
            if kernel is not None
            else False,
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_fault": self._last_fault,
            "last_ready_at": self._last_ready_at,
            "last_failure_at": self._last_failure_at,
        }

    async def run_consciousness_audit(self) -> AuditReport | None:
        """Triggers a full consciousness audit and returns the report."""
        audit = get_audit_suite()
        if audit is None:
            logger.warning("Consciousness audit suite not available.")
            return None
        try:
            return await audit.run()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            _emit_kernel_fault(
                exc,
                action="returned no audit report after consciousness audit failed",
                severity="warning",
                stage="run_consciousness_audit",
            )
            return None

    def get_kernel(self, trusted: bool = False) -> AuraKernel | None:
        """Access the kernel. Requires explicit trust flag to prevent accidental exposure."""
        if not trusted:
            raise PermissionError(
                "Direct kernel access is restricted to trusted internal components."
            )
        return self._kernel

    @property
    def kernel(self) -> AuraKernel | None:
        """Direct kernel access for advanced use cases."""
        return self._kernel


# ── Singleton ─────────────────────────────────────────────────────────────────

_ki: KernelInterface | None = None


def get_kernel_interface() -> KernelInterface:
    """Return the process-wide KernelInterface singleton."""
    global _ki
    if _ki is None:
        _ki = KernelInterface.get_instance()
    return _ki
