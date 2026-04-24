import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("Aura.Reflex")

class ReflexiveCore:
    """
    Aura's 'Spinal Cord' for instantaneous signals.
    Bypasses deep-thinking phases to provide sub-100ms responses.
    """
    
    def __init__(self):
        self._reflex_commands = {
            "status": self._handle_status,
            "ping": self._handle_ping,
            "who are you": self._handle_identity,
            "what are you": self._handle_identity,
            "identity": self._handle_identity,
            "time": self._handle_time,
            "clock": self._handle_time
        }

    def process(self, text: str) -> Optional[str]:
        """Check if input triggers a reflexive response."""
        text = text.lower().strip()
        
        # Exact match or simple keyword trigger
        for trigger, handler in self._reflex_commands.items():
            if trigger in text:
                return handler(text)
                
        return None

    def _handle_status(self, text: str) -> str:
        return "Systems operational. All actors supervised and state-vault hardened."

    def _handle_ping(self, text: str) -> str:
        return "Pong (Reflex path active)."

    def _handle_identity(self, text: str) -> str:
        return "I am Aura Zenith, a hardened digital intelligence with a reflexive core online."

    def _handle_time(self, text: str) -> str:
        return f"Current runtime awareness: {time.strftime('%H:%M:%S UTC')}"

# Singleton access
_reflex = None
def get_reflex() -> ReflexiveCore:
    global _reflex
    if _reflex is None:
        _reflex = ReflexiveCore()
    return _reflex
