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

import asyncio
import logging
from typing import Any, Dict, Optional

from core.consciousness.phenomenological_experiencer import get_experiencer

logger = logging.getLogger("Aura.ConsciousnessIntegration")

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

# Singleton access
_integration_instance: Optional[ConsciousnessIntegration] = None

def get_consciousness_integration(orchestrator=None) -> ConsciousnessIntegration:
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = ConsciousnessIntegration(orchestrator)
    return _integration_instance