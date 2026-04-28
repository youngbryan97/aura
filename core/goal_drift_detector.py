from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("Cognition.GoalDrift")

class GoalDriftDetector:
    """Monitors distinct divergence from the original goal.
    Prevents the agent from "rabbit-holing" into irrelevant tasks.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        self.drift_count = 0
        self.max_drift_tolerance = 3
        
    async def check_drift(self, original_goal: str, current_thought: str, recent_history: str) -> bool:
        """Ask the LLM if the current thought aligns with the original goal.
        Returns True if drifting.
        """
        # Optimization: Don't check every single cycle, maybe every 3rd?
        # For now, we'll assume the caller manages frequency or we do it here.
        
        # Dynamic Context Windowing (v14 HARDENED)
        context_window = recent_history[-2000:] if len(recent_history) > 2000 else recent_history
        
        prompt = f"""
SYSTEM MONITOR: DRIFT DETECTION
Original Objective: "{original_goal}"
Current Cognitive Step: "{current_thought}"
Temporal Context Snapshot: "{context_window}"

Aura is designed for complex, agentic pursuits. Sometimes sub-steps look irrelevant but are vital (e.g., searching for a tool to fix a file).
Identify if the Current Cognitive Step is a constructive arc towards the Original Objective OR a distinct divergence into an unrelated tangent.

Output format:
DECISION: [ALIGNED / DRIFTING]
REASON: [Brief explanation]
"""
        try:
            # Use 'FAST' mode for this check to save cost/latency
            from core.brain.cognitive_engine import ThinkingMode
            thought = await self.brain.think(prompt, mode=ThinkingMode.FAST)
            response = thought.content
            
            if "DRIFTING" in response.upper():
                self.drift_count += 1
                logger.warning("⚠️ Goal Drift Detected (%d/%d): %s", self.drift_count, self.max_drift_tolerance, current_thought)
            else:
                self.drift_count = max(0, self.drift_count - 1)
                
            if self.drift_count >= self.max_drift_tolerance:
                return True
                
            return False
            
        except Exception as e:
            record_degradation('goal_drift_detector', e)
            logger.error("Drift check failed: %s", e)
            return False

    def reset(self):
        self.drift_count = 0