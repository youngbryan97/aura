"""core/ops/priority_gator.py — v1.0 Priority Gator

Dynamic gatekeeper for Aura's cognitive tasks.
Balances goal importance vs. metabolic cost (ergs).
"""

import logging
import time
from typing import Any, Dict, Optional
from core.ops.metabolic_monitor import get_cost_tracker

logger = logging.getLogger("Aura.PriorityGator")

class PriorityGator:
    """Decides if a task should proceed based on priority and current metabolic strain."""
    
    def __init__(self, high_erg_threshold: float = 1000.0, recovery_rate: float = 10.0):
        self.cost_tracker = get_cost_tracker()
        self.high_erg_threshold = high_erg_threshold
        self.recovery_rate = recovery_rate # ergs/sec recovery
        
    def should_proceed(self, task_name: str, priority: float = 1.0, estimated_tokens: int = 500) -> bool:
        """Heuristic check to see if a task is 'affordable' right now.
        
        Priority:
        1.0 = Critical/User (Always proceed)
        0.5 = Important background
        0.1 = Speculative/Low energy
        """
        if priority >= 1.0:
            return True # Never gate the user
            
        rate = self.cost_tracker.get_metabolic_rate()
        
        # Heuristic: If we are burning > 50 ergs/sec, gate low priority tasks
        if rate > 50.0 and priority < 0.3:
            logger.warning(f"🔋 GATOR: Gating low-priority task '{task_name}' due to high metabolic rate ({rate:.2f} ergs/s)")
            return False
            
        # Hard cap on total ergs in recent window (metabolic fatigue)
        # This prevents 'burnout' from too many background loops
        burn_report = self.cost_tracker.get_burn_report()
        if burn_report["avg_rate"] > 100.0 and priority < 0.6:
            logger.warning(f"🔋 GATOR: System fatigue detected. Gating '{task_name}' (priority {priority})")
            return False
            
        return True

# Singleton support
_gator = None

def get_priority_gator() -> PriorityGator:
    global _gator
    if _gator is None:
        _gator = PriorityGator()
    return _gator
