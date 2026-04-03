"""core/aura_3_boot.py — Aura 3.0: Zenith Boot Sequence
=======================================================
Implements Phase 8: The final, ordered, auditable boot sequence. 
Replacing all legacy `main.py` non-deterministic loops.

ZENITH Protocol compliance:
  - Strict Level 0 through Level 3 initialization.
  - Mandatory registration locking.
  - Zero-wait asynchronous ready checks.
"""

import asyncio
import logging
import sys
import os

from core.container import ServiceContainer
from core.intent_gate import register_intent_gate
from core.memory.horcrux import HorcruxManager
from core.memory.cognitive_vault import CognitiveVault
from core.consciousness.substrate_governor import SubstrateGovernor
from core.memory.black_hole import BlackHole

logger = logging.getLogger("Aura.Boot")


class ZenithBootloader:
    """
    Orchestrates the transition from cold-start to operational.
    """

    @classmethod
    async def ignite(cls):
        """Main entry point for Aura 3.0."""
        logger.info("🚀 ZENITH PROTOCOL IGNITION SEQUENCE START")
        
        try:
            # --- LEVEL 0: SECURITY STACK ---
            horcrux = HorcruxManager()
            ServiceContainer.register_instance("horcrux", horcrux)
            
            # Security must be sync-initialized before anything else
            # because BlackHole needs the derived key.
            # (Note: initialize is now async in my refactor)
            success = await horcrux.initialize()
            if not success:
               logger.critical("CRITICAL: Horcrux initialization FAILED. Boot aborted.")
               sys.exit(1)
            
            black_hole = BlackHole()
            ServiceContainer.register_instance("black_hole", black_hole)
            black_hole.on_start() # Binds to horcrux key
            
            # --- LEVEL 1: COGNITIVE INFRASTRUCTURE ---
            vault = CognitiveVault()
            ServiceContainer.register_instance("vault", vault)
            
            governor = SubstrateGovernor()
            ServiceContainer.register_instance("governor", governor)
            
            intent_gate = register_intent_gate()
            
            # --- LEVEL 2: ENGINE REGISTRATION ---
            # (MorphicForking, OntologyGenesis, etc.)
            from core.brain.morphic_forking import MorphicForkingEngine
            ServiceContainer.register_instance("forking_engine", MorphicForkingEngine())
            
            from core.brain.ontology_genesis import OntologyGenesisEngine
            ServiceContainer.register_instance("ontology", OntologyGenesisEngine())

            # --- LEVEL 3: WAKE CYCLE ---
            logger.info("Waking services...")
            woken_list = await ServiceContainer.wake()
            logger.info("✓ Woke %d services: %s", len(woken_list), ", ".join(woken_list))
            
            # Final Lock
            logger.info("🔒 SYSTEM STABILIZED. RADAR ONLINE.")
            
        except Exception as e:
            logger.critical("ZENITH BOOT FAILURE: %s", e, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ZenithBootloader.ignite())
