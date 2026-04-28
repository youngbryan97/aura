from __future__ import annotations
from core.runtime.errors import record_degradation

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.state.aura_state import AuraState
    from core.kernel.aura_kernel import AuraKernel

# Import of Legacy Orchestrator will be added here
# from core.orchestrator.main import RobustOrchestrator

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))

class Phase(ABC):
    """Base class for all Unitary Kernel phases."""
    def __init__(self, kernel: "AuraKernel" = None):
        """Store a reference to the owning kernel."""
        self.kernel = kernel

    @abstractmethod
    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """Run this phase against the given state and return the updated state."""
        raise NotImplementedError("Phases must implement execute()")

    async def __call__(self, state: AuraState) -> AuraState:
        """Execute the phase, pulling the current objective from state if available."""
        # Pull objective from state if not provided
        obj = getattr(state.cognition, "current_objective", None)
        return await self.execute(state, objective=obj)

class LegacyPhase(Phase):
    """
    The 'Kernel Bridge' Pattern.
    Wraps the existing modular chaos as a single Phase within the new Kernel.
    Allows for one-at-a-time migration of modules with zero downtime.
    """
    def __init__(self, kernel: "AuraKernel"):
        """Initialize the legacy bridge; the orchestrator is lazily constructed on first execute."""
        self.kernel = kernel
        self.legacy_orchestrator: Any = None # To be RobustOrchestrator()
        self._legacy_tasks: set[asyncio.Task] = set()
        logger.info("Bridge: LegacyPhase bridge established.")

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        normalized = cls._normalize_origin(origin)
        if not normalized:
            return False
        if normalized in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"})

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Delegates to the old world, but enforces the new state invariants.
        """
        priority = bool(kwargs.get("priority", False))
        current_origin = self._normalize_origin(getattr(getattr(state, "cognition", None), "current_origin", ""))

        # Constitutional migration hardening:
        # once the kernel is actively handling a foreground/user-facing turn,
        # the legacy bridge is no longer allowed to behave like a parallel
        # executive. It remains available only as a background compatibility organ.
        if priority or self._is_user_facing_origin(current_origin):
            logger.debug(
                "Bridge: bypassing legacy delegation for foreground/user-facing tick (origin=%s, priority=%s).",
                current_origin or "unknown",
                priority,
            )
            return state

        if self.legacy_orchestrator is None:
            # Lazy init legacy only once to avoid boot bloat
            from core.orchestrator.main import RobustOrchestrator
            
            # Instead of monkey-patching, we use the existing task_tracker
            # and ensure the Orchestrator is aware of the Kernel's hierarchy.
            from core.kernel.kernel_interface import get_kernel_interface
            self.legacy_orchestrator = RobustOrchestrator(kernel_interface=get_kernel_interface())
            self.legacy_orchestrator.state = state 
            
            # Force the legacy orchestrator to use the Kernel's Service Registry
            # This prevents it from hitting ServiceContainer.get("affect_engine")
            # and getting a 'rogue' engine.
            self.legacy_orchestrator._affect_engine_override = AffectBridge(self.kernel)
            self.legacy_orchestrator._motivation_engine_override = MotivationBridge(self.kernel)
            
            # Register bridges in ServiceContainer for other legacy components.
            # Wrap in try-except because the container locks after boot — if another
            # LegacyPhase instance or boot path already registered these, skip silently.
            from core.container import ServiceContainer
            try:
                ServiceContainer.register_instance("affect_engine", self.legacy_orchestrator._affect_engine_override)
            except Exception as _reg_err:
                record_degradation('bridge', _reg_err)
                logger.debug("Bridge: affect_engine already registered, skipping: %s", _reg_err)
            try:
                ServiceContainer.register_instance("motivation_engine", self.legacy_orchestrator._motivation_engine_override)
            except Exception as _reg_err:
                record_degradation('bridge', _reg_err)
                logger.debug("Bridge: motivation_engine already registered, skipping: %s", _reg_err)
            logger.info("Bridge: Legacy engines registered in ServiceContainer.")
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "kernel_bridge",
                    "legacy_bridge_active",
                    detail="legacy_orchestrator_initialized",
                    severity="warning",
                    classification="background_degraded",
                    context={"phase": "legacy_bridge"},
                )
            except Exception as exc:
                record_degradation('bridge', exc)
                logger.debug("Bridge: legacy activation degraded-event logging failed: %s", exc)
            
        logger.debug("Delegating objective '%s' to Legacy Bridge...", objective)
        
        # 1. Sync Kernel State -> Legacy Orchestrator (Conversation History)
        if hasattr(self.legacy_orchestrator, "conversation_history"):
            self.legacy_orchestrator.conversation_history = state.cognition.history
            
        # 2. Forward to legacy logic
        # process_user_input_priority is the most robust entry point for objective-driven thinking
        await self.legacy_orchestrator.process_user_input_priority(objective, origin="kernel")
        
        # 3. Sync Legacy Orchestrator -> Kernel State
        state.cognition.history = self.legacy_orchestrator.conversation_history
        
        return state

    async def cleanup(self):
        """
        [CF-4] Supervisor Reap: Ensures no orphaned legacy tasks escape.
        """
        if self._legacy_tasks:
            logger.info("Bridge: Reaping %d orphaned legacy tasks.", len(self._legacy_tasks))
            for task in self._legacy_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._legacy_tasks, return_exceptions=True)
            self._legacy_tasks.clear()
        
        if self.legacy_orchestrator and hasattr(self.legacy_orchestrator, "stop"):
            await self.legacy_orchestrator.stop()

class AffectBridge:
    """
    [PEER BRIDGE] Redirects legacy affect_engine lookups to the Kernel's Vault.
    Allows old modules to 'think' they are talking to a singleton engine,
    while they are actually reading/writing to the monolithic AuraState.
    """
    def __init__(self, kernel: "AuraKernel"):
        """Store a reference to the kernel for live state access."""
        self.kernel = kernel

    def get_status(self) -> dict:
        """Proxies to kernel state affect."""
        state = self.kernel.state # Use live kernel state
        if not state: return {}
        aff = state.affect
        return {
            "mood": aff.dominant_emotion.capitalize(),
            "energy": int(aff.physiology["heart_rate"]),
            "curiosity": int(aff.curiosity * 100),
            "valence": aff.valence
        }

    def get_state_sync(self) -> dict:
        """Compatibility shim for legacy callers expecting a sync affect snapshot."""
        return self.get_status()

    async def update(self, **kwargs):
        """
        [CF-4] FIX: Instead of mutating vault._current directly, we inject
        a percept into the live kernel.state. The AffectUpdatePhase will 
        process this on the next tick, ensuring the mutation is persisted.
        """
        state = self.kernel.state
        if not state: return
        
        # Inject as a 'virtual_percept' for the next tick
        state.world.recent_percepts.append({
            "type": "legacy_update",
            "intensity": kwargs.get("intensity", 0.5),
            "payload": kwargs
        })
        logger.debug("AffectBridge: Injected legacy update into percept stream.")

    async def apply_stimulus(self, stimulus_type: str, intensity: float):
        """Compatibility bridge for callers expecting affect stimulus injection."""
        await self.update(stimulus_type=stimulus_type, intensity=float(intensity or 0.0))

    async def decay_tick(self):
        """Legacy no-op decay hook for components that expect a coroutine."""
        return None

    def get_mood(self) -> str:
        """Return the current dominant emotion, or 'Stable' if state is unavailable."""
        state = self.kernel.state
        return state.affect.dominant_emotion if state else "Stable"

    def receive_qualia_echo(self, q_norm: float, pri: float, trend: float):
        """Kernel-safe compatibility bridge for qualia -> affect feedback."""
        state = self.kernel.state
        if not state:
            return

        affect = state.affect
        emotions = affect.emotions
        dominant = max(emotions.items(), key=lambda item: item[1])[0] if emotions else affect.dominant_emotion

        if q_norm > 0.5 and dominant in emotions:
            emotions[dominant] = _clamp(emotions.get(dominant, 0.0) + ((q_norm - 0.5) * 0.1), 0.0, 1.0)

        if pri > 0.7:
            emotions["awe"] = _clamp(emotions.get("awe", 0.0) + ((pri - 0.7) * 0.05), 0.0, 1.0)

        if trend > 0.02:
            emotions["anticipation"] = _clamp(emotions.get("anticipation", 0.0) + (trend * 0.5), 0.0, 1.0)
        elif trend < -0.02:
            emotions["sadness"] = _clamp(emotions.get("sadness", 0.0) + (abs(trend) * 0.3), 0.0, 1.0)

        affect.physiology["heart_rate"] = _clamp(
            affect.physiology.get("heart_rate", 72.0) + ((q_norm - 0.5) * 2.0),
            50.0,
            120.0,
        )
        affect.physiology["gsr"] = _clamp(
            affect.physiology.get("gsr", 2.1) + ((q_norm - 0.5) * 0.5),
            0.5,
            8.0,
        )
        affect.arousal = _clamp(affect.arousal + ((q_norm - 0.5) * 0.08), 0.0, 1.0)
        affect.updated_at = time.time()

        if emotions:
            affect.dominant_emotion = max(emotions.items(), key=lambda item: item[1])[0]

    @property
    def current(self):
        """Support for legacy .current property."""
        from types import SimpleNamespace
        state = self.kernel.state
        if not state: return SimpleNamespace(energy=0.5, curiosity=0.5, valence=0.0)
        aff = state.affect
        return SimpleNamespace(
            energy=float((aff.physiology["heart_rate"] - 60) / 40.0),
            curiosity=float(aff.curiosity),
            valence=float(aff.valence)
        )

class MotivationBridge:
    """
    [PEER BRIDGE] Redirects legacy motivation lookups to the Kernel's Vault.
    """
    def __init__(self, kernel: "AuraKernel"):
        """Store a reference to the kernel for live state access."""
        self.kernel = kernel

    async def get_status(self) -> dict:
        """Return the current motivation budget levels from kernel state."""
        state = self.kernel.state
        if not state: return {}
        return state.motivation.budgets

    async def satisfy(self, drive: str, amount: float):
        """Increase a motivation drive's level by the given amount, capped at its capacity."""
        state = self.kernel.state
        if not state: return
        # Strict Vault Routing: We derive a new state for the satisfaction
        # ensuring the mutation is versioned.
        b = state.motivation.budgets.get(drive)
        if b:
            b["level"] = min(b["capacity"], b["level"] + amount)
            logger.debug("MotivationBridge: Satisfied %s (+%s)", drive, amount)

    async def punish(self, drive: str, amount: float):
        """Decrease a motivation drive's level by the given amount, floored at zero."""
        state = self.kernel.state
        if not state: return
        b = state.motivation.budgets.get(drive)
        if b:
            b["level"] = max(0.0, b["level"] - amount)
            logger.debug("MotivationBridge: Punished %s (-%s)", drive, amount)
