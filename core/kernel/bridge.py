from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.runtime.errors import record_degradation

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel
    from core.state.aura_state import AuraState

# Import of Legacy Orchestrator will be added here
# from core.orchestrator.main import RobustOrchestrator

from core.state.percepts import emit_percept

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class Phase(ABC):
    """Base class for all Unitary Kernel phases."""

    def __init__(self, kernel: AuraKernel = None):
        """Store a reference to the owning kernel."""
        self.kernel = kernel

    @abstractmethod
    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """Run this phase against the given state and return the updated state."""
        raise NotImplementedError(f"{type(self).__name__}.execute must be implemented by a phase")

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

    def __init__(self, kernel: AuraKernel):
        """Initialize a compatibility phase that can borrow the canonical owner."""
        self.kernel = kernel
        self.legacy_orchestrator: Any = None
        self._legacy_tasks: set[asyncio.Task] = set()
        self._unbound_reported = False
        logger.info("Bridge: LegacyPhase bridge established.")

    def bind_orchestrator(self, owner: Any) -> None:
        """Bind exactly one borrowed orchestrator without taking lifecycle ownership."""
        if owner is None:
            raise ValueError("LegacyPhase requires a canonical orchestrator owner")
        if self.legacy_orchestrator is owner:
            return
        if self.legacy_orchestrator is not None:
            raise RuntimeError("LegacyPhase is already bound to a different orchestrator owner")
        self.legacy_orchestrator = owner
        self._unbound_reported = False
        logger.info("Bridge: bound to canonical orchestrator owner.")

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        normalized = cls._normalize_origin(origin)
        if not normalized:
            return False
        if normalized in {
            "user",
            "voice",
            "admin",
            "api",
            "gui",
            "ws",
            "websocket",
            "direct",
            "external",
        }:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(
            tokens
            & {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}
        )

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Delegates to the old world, but enforces the new state invariants.
        """
        priority = bool(kwargs.get("priority", False))
        current_origin = self._normalize_origin(
            getattr(getattr(state, "cognition", None), "current_origin", "")
        )

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
            if not self._unbound_reported:
                self._unbound_reported = True
                record_degradation(
                    "bridge",
                    RuntimeError("LegacyPhase has no canonical orchestrator binding"),
                    severity="warning",
                    action="skipped unbound compatibility phase without constructing a second runtime",
                )
                logger.warning(
                    "Bridge: compatibility phase is unbound; skipped without creating a runtime owner."
                )
            return state

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

        # The orchestrator is borrowed from the process root. Its owner performs
        # shutdown; this compatibility phase must never stop the global runtime.


class AffectBridge:
    """
    [PEER BRIDGE] Redirects legacy affect_engine lookups to the Kernel's Vault.
    Allows old modules to 'think' they are talking to a singleton engine,
    while they are actually reading/writing to the monolithic AuraState.
    """

    def __init__(self, kernel: AuraKernel):
        """Store a reference to the kernel for live state access."""
        self.kernel = kernel

    def is_ready(self) -> bool:
        """Health-contract probe for the kernel-backed affect bridge.

        The bridge is registered under the legacy ``affect_engine`` key in the
        kernel path, so it must prove that the live AuraState affect vector is
        present and numerically sane. A heartbeat alone is not enough here:
        if affect is absent or malformed, desktop chat is no longer speaking
        from the integrated body/mind path.
        """
        state = getattr(self.kernel, "state", None)
        affect = getattr(state, "affect", None)
        if affect is None:
            return False
        try:
            dominant = str(getattr(affect, "dominant_emotion", "") or "").strip()
            emotions = getattr(affect, "emotions", None)
            physiology = getattr(affect, "physiology", None) or {}
            valence = float(affect.valence)
            arousal = float(affect.arousal)
            curiosity = float(getattr(affect, "curiosity", 0.0))
            heart_rate = float(physiology.get("heart_rate", 72.0))
        except (AttributeError, TypeError, ValueError):
            return False

        return (
            bool(dominant)
            and isinstance(emotions, dict)
            and -1.0 <= valence <= 1.0
            and 0.0 <= arousal <= 1.0
            and 0.0 <= curiosity <= 1.0
            and 20.0 <= heart_rate <= 220.0
        )

    def get_status(self) -> dict:
        """Proxies to kernel state affect."""
        state = self.kernel.state  # Use live kernel state
        if not state:
            return {}
        aff = state.affect
        return {
            "mood": aff.dominant_emotion.capitalize(),
            "energy": int(aff.physiology["heart_rate"]),
            "curiosity": int(aff.curiosity * 100),
            "valence": aff.valence,
        }

    def get_state_sync(self) -> dict:
        """Compatibility shim for legacy callers expecting a sync affect snapshot."""
        return self.get_status()

    async def update(self, **kwargs):
        """
        [CF-4] FIX: Instead of mutating vault._current directly, we inject
        a percept into the live kernel.state. The AffectUpdatePhase will
        process this on the next tick. That tick is what persists the mutation.
        """
        state = self.kernel.state
        if not state:
            return

        # Inject as a 'virtual_percept' for the next tick. The type is the
        # stimulus the caller named: `apply_stimulus("threat_detected", 0.8)`
        # went in labelled `legacy_update`, which the affect phase's event map
        # has no entry for, so every stimulus that arrived down this path
        # produced nothing at all.
        emit_percept(
            state.world,
            str(kwargs.get("stimulus_type") or "legacy_update"),
            content=str(kwargs.get("content") or kwargs.get("stimulus_type") or ""),
            intensity=float(kwargs.get("intensity", 0.5) or 0.0),
            payload=kwargs,
        )
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
        dominant = (
            max(emotions.items(), key=lambda item: item[1])[0]
            if emotions
            else affect.dominant_emotion
        )

        if q_norm > 0.5 and dominant in emotions:
            emotions[dominant] = _clamp(
                emotions.get(dominant, 0.0) + ((q_norm - 0.5) * 0.1), 0.0, 1.0
            )

        if pri > 0.7:
            emotions["awe"] = _clamp(emotions.get("awe", 0.0) + ((pri - 0.7) * 0.05), 0.0, 1.0)

        if trend > 0.02:
            emotions["anticipation"] = _clamp(
                emotions.get("anticipation", 0.0) + (trend * 0.5), 0.0, 1.0
            )
        elif trend < -0.02:
            emotions["sadness"] = _clamp(
                emotions.get("sadness", 0.0) + (abs(trend) * 0.3), 0.0, 1.0
            )

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
        if not state:
            return SimpleNamespace(energy=0.5, curiosity=0.5, valence=0.0)
        aff = state.affect
        return SimpleNamespace(
            energy=float((aff.physiology["heart_rate"] - 60) / 40.0),
            curiosity=float(aff.curiosity),
            valence=float(aff.valence),
        )


class MotivationBridge:
    """
    [PEER BRIDGE] Redirects legacy motivation lookups to the Kernel's Vault.
    """

    def __init__(self, kernel: AuraKernel):
        """Store a reference to the kernel for live state access."""
        self.kernel = kernel

    async def update(self, *args, **kwargs) -> dict:
        """Apply legacy motivation updates to kernel budgets when possible."""
        state = self.kernel.state
        if not state:
            return {}

        updates = dict(kwargs)
        if args and isinstance(args[0], dict):
            updates.update(args[0])

        budgets = state.motivation.budgets
        for drive, amount in updates.get("budget_deltas", {}).items():
            budget = budgets.get(drive)
            if budget:
                budget["level"] = _clamp(
                    budget.get("level", 0.0) + float(amount),
                    0.0,
                    budget.get("capacity", 1.0),
                )

        drive = updates.get("drive")
        if drive in budgets and "amount" in updates:
            amount = float(updates["amount"])
            budget = budgets[drive]
            budget["level"] = _clamp(
                budget.get("level", 0.0) + amount,
                0.0,
                budget.get("capacity", 1.0),
            )

        return budgets

    async def get_status(self) -> dict:
        """Return the current motivation budget levels from kernel state."""
        state = self.kernel.state
        if not state:
            return {}
        return state.motivation.budgets

    async def satisfy(self, drive: str, amount: float):
        """Increase a motivation drive's level by the given amount, capped at its capacity."""
        state = self.kernel.state
        if not state:
            return
        # Strict Vault Routing: We derive a new state for the satisfaction
        # ensuring the mutation is versioned.
        b = state.motivation.budgets.get(drive)
        if b:
            b["level"] = min(b["capacity"], b["level"] + amount)
            logger.debug("MotivationBridge: Satisfied %s (+%s)", drive, amount)

    async def punish(self, drive: str, amount: float):
        """Decrease a motivation drive's level by the given amount, floored at zero."""
        state = self.kernel.state
        if not state:
            return
        b = state.motivation.budgets.get(drive)
        if b:
            b["level"] = max(0.0, b["level"] - amount)
            logger.debug("MotivationBridge: Punished %s (-%s)", drive, amount)
