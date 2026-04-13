import hashlib
import logging
from collections import deque

logger = logging.getLogger("Cognition.LoopDetector")

class LoopDetector:
    """detects recursive thought patterns or repetitive actions to prevent infinite loops.
    """
    
    def __init__(self, window_size: int = 10):
        self.history: deque[str] = deque(maxlen=window_size)
        self.threshold = 3 
        
    def add_event(self, content: str) -> None:
        """Add an event (thought content or action) to history."""
        # Normalize content to catch slight variations
        content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()
        self.history.append(content_hash)
        
    def detect_loop(self) -> bool:
        """Check if the most recent event has occurred too frequently in the window."""
        if not self.history:
            return False
            
        recent = self.history[-1]
        count = sum(1 for h in self.history if h == recent)
        
        if count >= self.threshold:
            logger.warning("⚠️ Loop Detected! Event '%s' repeated %d times in last %d cycles.", recent[:8], count, len(self.history))
            return True
            
        return False

    def clear(self) -> None:
        self.history.clear()

loop_detector = LoopDetector()
