"""core/meta/mirror_layer.py
Recursive Self-Modeling Layer: Auditing Aura's internal thought cycles.
"""
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("Core.Meta.MirrorLayer")

@dataclass
class ThoughtMirror:
    timestamp: float
    thought_id: str
    summary: str
    context_fingerprint: str
    mode: str
    affective_state: Dict[str, float]

class MirrorLayer:
    """The 'Hall of Mirrors' that observes and analyzes cognitive cycles."""
    
    def __init__(self, max_history: int = 50):
        self.history: List[ThoughtMirror] = []
        self.max_history = max_history
        self.acoustics_report: Dict[str, Any] = {
            "pollution_detected": False,
            "logic_loops": 0,
            "context_drift": 0.0
        }

    def audit_cycle(self, thought_data: Dict[str, Any]):
        """Analyze a single cognitive cycle for recursive health."""
        mirror = ThoughtMirror(
            timestamp=time.time(),
            thought_id=thought_data.get("id", "unknown"),
            summary=thought_data.get("content", "")[:100],
            context_fingerprint=self._generate_fingerprint(thought_data.get("context", {})),
            mode=thought_data.get("mode", "FAST"),
            affective_state=thought_data.get("affective_state", {})
        )
        
        self.history.append(mirror)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        self._detect_logic_loops()
        self._audit_acoustics()

    def _generate_fingerprint(self, context: Dict[str, Any]) -> str:
        """Create a stable hash/fingerprint of the input context."""
        # Simplified: key-based fingerprint
        keys = sorted(context.keys())
        return "|".join(keys)

    def _detect_logic_loops(self):
        """Identify if Aura is repeating sequences of thoughts."""
        if len(self.history) < 3: return
        
        last_three = [m.summary for m in self.history[-3:]]
        if len(set(last_three)) == 1:
            logger.warning("🔄 Recursive Logic Loop detected in Mirror Layer!")
            self.acoustics_report["logic_loops"] += 1
            self.acoustics_report["pollution_detected"] = True

    def _audit_acoustics(self):
        """Monitor context pollution and drift."""
        if len(self.history) < 2: return
        
        # Check if context fingerprint changed significantly
        if self.history[-1].context_fingerprint != self.history[-2].context_fingerprint:
            self.acoustics_report["context_drift"] += 0.1
            
    def get_audit_summary(self) -> Dict[str, Any]:
        """Return the current health of the 'Cathedral Acoustics'."""
        return self.acoustics_report