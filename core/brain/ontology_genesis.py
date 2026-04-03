"""core/brain/ontology_genesis.py — Aura 3.0: Experimental Containment Mode
=======================================================================
Implements the "Hibernation Mode" protocol for high-compute autonomous 
scientific discovery. 

Refactored for ZENITH Protocol efficiency:
  - Background discovery loop is DISABLED by default.
  - Must be explicitly triggered by user or deep-research soul drive.
  - Mandatory resource_anxiety abort if system load exceeds thresholds.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.OntologyGenesis")


class OntologyGenesisEngine:
    """
    Manages autonomous discovery of new cognitive laws and heuristics.
    
    ZENITH Protocol:
      - Default state is inert.
      - Throttled by resource anxiety.
    """

    def __init__(self):
        self._active = False
        self._genesis_task: Optional[asyncio.Task] = None
        self._discovery_log: List[str] = []

    def _get_resource_anxiety(self) -> float:
        """Calculate system resource pressure (0.0 to 1.0)."""
        from core.container import ServiceContainer
        homeostasis = ServiceContainer.get("homeostasis", default=None)
        if homeostasis and hasattr(homeostasis, "anxiety"):
            return homeostasis.anxiety
        return 0.0  # Safe default if unavailable

    async def start_discovery(self, mode: str = "manual") -> bool:
        """
        Triggers a discovery cycle.
        Restored via Volition Level 3 or manual deep_research.
        """
        from core.container import ServiceContainer
        kernel = ServiceContainer.get("aura_kernel", default=None)
        volition = getattr(kernel, 'volition_level', 0) if kernel else 0

        # Level 3 grants autonomous deep_research access
        if mode != "deep_research" and volition < 3:
            logger.info("OntologyGenesis: Restricted to deep_research mode (Volition < 3).")
            return False
            
        anxiety = self._get_resource_anxiety()
        # Level 3 is more persistent under load
        anxiety_threshold = 0.2 if volition < 3 else 0.5
        
        if anxiety >= anxiety_threshold:
            logger.warning("OntologyGenesis: Abort. Resource pressure too high (%.2f > %.2f).", anxiety, anxiety_threshold)
            return False

        if self._active:
            return True

        self._active = True
        self._genesis_task = asyncio.create_task(self._discovery_loop(volition))
        logger.info("OntologyGenesis: Hibernation ended (Volition=%d). Discovery loop active.", volition)
        return True

    async def stop_discovery(self):
        self._active = False
        if self._genesis_task:
            self._genesis_task.cancel()
            try:
                await self._genesis_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("OntologyGenesis: Returning to hibernation.")

    async def _discovery_loop(self, volition: int = 0):
        """Main loop for high-compute cognitive law formation."""
        while self._active:
            # Check anxiety each cycle
            anxiety_threshold = 0.3 if volition < 3 else 0.6
            if self._get_resource_anxiety() >= anxiety_threshold:
                logger.warning("OntologyGenesis: Emergency hibernation due to load spikes (%.2f).", self._get_resource_anxiety())
                self._active = False
                break
                
            logger.debug("OntologyGenesis: Processing candidate heuristics...")
            # Simulate heavy discovery work
            await asyncio.sleep(60) 
            
            # Logic for calling LLM to synthesize new laws goes here
            # ...
            
    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "anxiety_threshold": 0.2,
            "current_anxiety": self._get_resource_anxiety()
        }


def register_ontology_genesis(orchestrator: Optional[Any] = None) -> OntologyGenesisEngine:
    """Legacy registration helper expected by boot code."""
    from core.container import ServiceContainer

    engine = OntologyGenesisEngine()
    ServiceContainer.register_instance("ontology_genesis", engine)
    if orchestrator is not None:
        try:
            setattr(orchestrator, "ontology_genesis", engine)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
    return engine
