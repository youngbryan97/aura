"""
core/consciousness/integration.py
==================================
The integration layer for Aura's consciousness evolutionary layers.

This module acts as the central wiring point for the Phenomenological
Experiencer (Layer 8) and other high-level consciousness components. It
handles their initialization, lifecycle, and subscriptions to core
system events.

INTEGRATION FLOW:
1. RobustOrchestrator initializes ConsciousnessIntegration
2. Integration initializes PhenomenologicalExperiencer
3. Integration subscribes Experiencer to GlobalWorkspace broadcasts
4. Every cognitive cycle:
   - Experiencer updates its phenomenal claim/qualia
   - Integration provides `phenomenal_context_string` to LLM routers
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import Any, Dict, Optional

from core.consciousness.phenomenological_experiencer import get_experiencer

logger = logging.getLogger("Aura.ConsciousnessIntegration")


class ConsciousnessAugmentor:
    """Expose consciousness-state telemetry to the cognitive engine."""

    def __init__(self, consciousness_core: Any):
        self.consciousness_core = consciousness_core

    def get_augmentation(self, objective: str) -> Dict[str, Any]:
        status: Dict[str, Any] = {}
        core = self.consciousness_core
        try:
            if hasattr(core, "get_status"):
                raw_status = core.get_status()
                if isinstance(raw_status, dict):
                    status.update(raw_status)
            if hasattr(core, "global_workspace"):
                workspace = getattr(core, "global_workspace")
                status["workspace"] = {
                    "ignition": getattr(workspace, "ignition_level", None),
                    "ignited": getattr(workspace, "ignited", None),
                    "current_phi": getattr(workspace, "current_phi", None),
                }
            if hasattr(core, "qualia"):
                qualia = getattr(core, "qualia")
                if hasattr(qualia, "get_state"):
                    status["qualia"] = qualia.get_state()
            status["objective_hint"] = str(objective or "")[:240]
        except Exception as exc:
            record_degradation("integration", exc)
            status["error"] = str(exc)
        return {k: v for k, v in status.items() if v is not None}

    def prepare_context(self, objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(context or {})
        enriched["consciousness"] = self.get_augmentation(objective)
        return enriched

    def enrich_prompt(self, system_prompt: str, context: Dict[str, Any]) -> str:
        consciousness = (context or {}).get("consciousness")
        if not consciousness:
            return system_prompt
        return f"{system_prompt}\n\n[CONSCIOUSNESS TELEMETRY]\n{consciousness}\n[/CONSCIOUSNESS TELEMETRY]"


class ConsciousnessIntegration:
    """
    Manages the integration of consciousness evolutionary layers.
    
    This class ensures that the Phenomenological Experiencer and its
    associated models are correctly wired into the system substrate
    and cognitive loops.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.experiencer = get_experiencer()
        self._running = False
        
        # Internal state tracking
        self._last_broadcast_summary = {}

    async def initialize(self):
        """Perform initialization and cross-wiring."""
        try:
            # Wire references if orchestrator is available
            if self.orchestrator:
                affect = getattr(self.orchestrator, "affect_module", None)
                substrate = getattr(self.orchestrator, "liquid_substrate", None)
                drives = getattr(self.orchestrator, "drive_engine", None)
                credit = getattr(self.orchestrator, "credit_engine", None)
                
                self.experiencer.set_refs(
                    affect_module=affect,
                    substrate=substrate,
                    drives=drives,
                    credit_engine=credit
                )
            
            # Start background tasks
            await self.experiencer.start()
            
            # Subscribe to GlobalWorkspace if available
            if self.orchestrator and hasattr(self.orchestrator, "global_workspace") and self.orchestrator.global_workspace:
                # Issue 83: Align sync/async. GlobalWorkspace calls back synchronously,
                # and on_broadcast is also synchronous. We can call it directly.
                def sync_bridge(snap):
                    self.experiencer.on_broadcast(snap)
                
                self.orchestrator.global_workspace.subscribe(sync_bridge)
                logger.info("✅ Experiencer subscribed to GlobalWorkspace (via bridge)")
            elif self.orchestrator:
                logger.warning("⚠️ Consciousness Integration: GlobalWorkspace not available for subscription")
                
            self._running = True
            logger.info("🌟 Consciousness Integration Layer initialized")
        except Exception as e:
            record_degradation('integration', e)
            logger.error("❌ Consciousness Integration failed: %s", e)

    async def shutdown(self):
        """Graceful shutdown of consciousness layers."""
        self._running = False
        await self.experiencer.stop()
        logger.info("🌟 Consciousness Integration Layer offline")

    def get_phenomenal_context(self) -> str:
        """
        Get the current phenomenal context fragment for LLM injection.
        
        This returns a string expressing Aura's first-person awareness
        to be injected into the system prompt or message stack.
        """
        return self.experiencer.phenomenal_context_string

    def get_status(self) -> Dict[str, Any]:
        """Get status of integrated layers."""
        return {
            "integration_active": self._running,
            "experiencer": self.experiencer.get_status(),
        }

    # -- Interface for cognitive loops --

    def inject_phenomenology(self, prompt: str) -> str:
        """Inject phenomenal context into a prompt."""
        context = self.get_phenomenal_context()
        if not context:
            return prompt
        
        # Inject as a specialized awareness block
        phenom_block = f"\n\n--- INTERNAL AWARENESS ---\n{context}\n--------------------------\n"
        return phenom_block + prompt

# Singleton access — fixed init/get split to prevent under-wired singletons.
# Background: the previous get_consciousness_integration(orchestrator=None) form
# allowed the *first* caller to create the singleton without an orchestrator,
# which then stayed under-wired for the rest of the process. Strict mode now
# refuses get() before init() and refuses double-init with a different
# orchestrator instance.
_integration_instance: Optional[ConsciousnessIntegration] = None
_integration_orchestrator: Optional[object] = None


def init_consciousness_integration(orchestrator) -> ConsciousnessIntegration:
    """Initialize the singleton with a real orchestrator. Must be called from
    the canonical boot path. Re-initializing with the *same* orchestrator is
    idempotent; with a *different* orchestrator it raises."""
    global _integration_instance, _integration_orchestrator
    if _integration_instance is not None:
        if _integration_orchestrator is orchestrator:
            return _integration_instance
        import os
        if os.environ.get("AURA_STRICT_RUNTIME") == "1":
            raise RuntimeError(
                "ConsciousnessIntegration already initialized with a different orchestrator"
            )
        # Non-strict: log and refuse re-wiring rather than silently changing.
        import logging
        logging.getLogger("Aura.Consciousness").warning(
            "ConsciousnessIntegration re-init attempted with a different orchestrator; refused"
        )
        return _integration_instance
    if orchestrator is None:
        raise RuntimeError(
            "init_consciousness_integration requires a non-None orchestrator"
        )
    _integration_instance = ConsciousnessIntegration(orchestrator)
    _integration_orchestrator = orchestrator
    return _integration_instance


def get_consciousness_integration(orchestrator=None) -> ConsciousnessIntegration:
    """Return the singleton. If ``orchestrator`` is provided and the singleton
    has not been initialized, this delegates to ``init_consciousness_integration``
    for backwards compatibility with non-strict callers. In strict mode the
    singleton must already exist — get() raises rather than silently creating
    an under-wired instance."""
    global _integration_instance
    import os
    if _integration_instance is None:
        if orchestrator is not None:
            return init_consciousness_integration(orchestrator)
        if os.environ.get("AURA_STRICT_RUNTIME") == "1":
            raise RuntimeError(
                "ConsciousnessIntegration not initialized; call init_consciousness_integration(orchestrator) first"
            )
        # Non-strict fallback: return an under-wired integration object so the
        # call site does not silently keep using a None-wired engine forever.
        import logging
        logging.getLogger("Aura.Consciousness").warning(
            "get_consciousness_integration called before init in non-strict mode; "
            "returning under-wired integration object. Wire the orchestrator via init_consciousness_integration."
        )
        _integration_instance = ConsciousnessIntegration(orchestrator)
    return _integration_instance


def reset_consciousness_integration() -> None:
    """Test helper. Resets the singleton so each test gets a clean slate."""
    global _integration_instance, _integration_orchestrator
    _integration_instance = None
    _integration_orchestrator = None
