import logging
import time

from core.container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Resource.TokenMonitor")

class TokenMonitor:
    """Tracks estimated token usage to prevent context overflow and reduce costs.
    """
    
    def __init__(self, limit: int = 120000): # Default context window (e.g. 128k)
        self.session_usage = 0
        self.context_limit = limit
        self.current_context_size = 0
        self.start_time = time.time()
        
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation: 1 token ~= 4 characters."""
        if not text:
            return 0
        return len(text) // 4
        
    def track_request(self, prompt: str):
        count = self.estimate_tokens(prompt)
        self.current_context_size = count
        self.session_usage += count
        
        if self.current_context_size > self.context_limit * 0.9:
            logger.warning("⚠️ Context Usage Warning: %d/%d tokens (90%%)", self.current_context_size, self.context_limit)
            
    def track_response(self, response: str):
        count = self.estimate_tokens(response)
        self.session_usage += count
        
    def get_stats(self):
        return {
            "session_total": self.session_usage,
            "current_context": self.current_context_size,
            "utilization": self.current_context_size / self.context_limit
        }

# Service Registration
def register_token_monitor():
    """Register the token monitor in the global container."""
    ServiceContainer.register_instance(
        "token_monitor",
        TokenMonitor()
    )

def get_token_monitor():
    """Resolve token monitor from container."""
    return ServiceContainer.get("token_monitor", default=None)

# Auto-register if not already present
try:
    if not ServiceContainer.get("token_monitor", default=None):
        register_token_monitor()
except Exception:
    register_token_monitor()

token_monitor = get_token_monitor()