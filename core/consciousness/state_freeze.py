"""core/consciousness/state_freeze.py

Global state freeze manager. Provides a context manager (sync/async)
to freeze continuous somatic/metabolic state updates during deep LLM inference.
"""

import threading
import logging

logger = logging.getLogger("Consciousness.StateFreeze")

_freeze_active = False
_freeze_lock = threading.Lock()

def is_state_frozen() -> bool:
    """Return True if the somatic state freeze is currently active."""
    global _freeze_active
    return _freeze_active

def set_state_freeze(active: bool):
    """Set the somatic state freeze state."""
    global _freeze_active
    with _freeze_lock:
        if _freeze_active != active:
            _freeze_active = active
            logger.debug("Somatic state freeze: %s", "ENABLED" if active else "DISABLED")

class state_freeze:
    """Context manager and async context manager to wrap LLM inference calls."""
    def __enter__(self):
        set_state_freeze(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_state_freeze(False)

    async def __aenter__(self):
        set_state_freeze(True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        set_state_freeze(False)
