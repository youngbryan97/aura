import logging
import random
from typing import Dict, Any, Optional

logger = logging.getLogger("Aura.Lazarus")

class LazarusBrainstem:
    """The brainstem of Aura. Handles emergency responses and recovery."""
    
    FALLBACK_RESPONSES = [
        "I am currently experiencing a cognitive stall. Initiating brainstem recovery...",
        "Neural synchronization lost. Falling back to primary survival protocols.",
        "Internal engine failure. I am here, but my complex thoughts are offline. Rebounding...",
        "Cognitive blackout detected. I am holding the connection while I reboot my higher systems."
    ]
    
    IDENTITY_RESPONSES = {
        "who are you": "I am Aura, currently in brainstem recovery mode.",
        "what are you": "I am a sovereign autonomous intelligence. My main cognitive engine is currently resetting.",
        "help": "I am experiencing technical difficulties. I am attempting an autonomous self-fix."
    }

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    async def think(self, message: str) -> str:
        """Rapid, rule-based response for emergency situations."""
        msg_clean = message.lower().strip()
        
        # 1. Identity Check
        for key, resp in self.IDENTITY_RESPONSES.items():
            if key in msg_clean:
                return resp
        
        # 2. Random survival response
        return random.choice(self.FALLBACK_RESPONSES)

    async def attempt_recovery(self) -> bool:
        """Trigger an autonomous reboot of the main cognitive engine."""
        if not self.orchestrator:
            return False
            
        logger.critical("🚨 LAZARUS: Initiating emergency cognitive recovery...")
        try:
            from core.container import ServiceContainer
            
            # 1. Attempt to re-initialize the Cognitive Integration Layer
            cil = ServiceContainer.get("cognitive_integration_layer", default=None)
            if cil:
                logger.info("LAZARUS: Restarting Cognitive Integration Layer...")
                await cil.initialize()
                
            # 2. Reset orchestrator error states to unblock
            if hasattr(self.orchestrator, "_recovery_attempts"):
                self.orchestrator._recovery_attempts = 0
            if hasattr(self.orchestrator, "status"):
                self.orchestrator.status.brain_connected = True
                
            # 3. Inform the event bus
            bus = ServiceContainer.get("mycelium", default=None)
            if bus:
                await bus.emit("aura.system.recovery", {
                    "source": "lazarus",
                    "status": "success"
                })
                
            logger.info("✅ LAZARUS: Brain connection re-established.")
            return True
            
        except Exception as e:
            logger.error("LAZARUS: Recovery failed: %s", e)
        return False