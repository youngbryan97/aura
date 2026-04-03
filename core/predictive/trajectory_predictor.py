import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("Aura.Trajectory")

@dataclass
class TrajectoryStep:
    phase_id: str
    description: str
    probability: float
    estimated_duration_s: float

@dataclass
class Trajectory:
    objective: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    confidence: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

class TrajectoryPredictor:
    """
    [MOTO TRANSIMAL] Predicts the future cognitive path for the current objective.
    Broadcasts 'mental trajectories' to the UI to show Aura's plan.
    """
    
    def __init__(self, container: Any):
        self.container = container
        self.last_trajectory: Optional[Trajectory] = None
        
    async def predict_path(self, objective: str, current_state: Any) -> Trajectory:
        """Analyze objective and predict next steps."""
        from core.brain.llm.llm_router import LLMTier
        router = self.container.get("llm_router", default=None)
        
        if not router or not objective:
            return Trajectory(objective=objective)
            
        prompt = (
            f"Given the objective: '{objective}'\n"
            f"State details: Affect={current_state.affect.valence:.2f}, Arousal={current_state.affect.arousal:.2f}\n"
            f"Predict the next 3 logical steps in your cognitive pipeline.\n"
            f"Return a JSON list of objects: [{{'phase': '...', 'task': '...', 'prob': 0.0-1.0}}]"
        )
        
        try:
            resp_obj = await router.generate(prompt, prefer_tier=LLMTier.TERTIARY, is_background=True)
            resp = resp_obj.get("text", "") if isinstance(resp_obj, dict) else str(resp_obj)
            
            import json, re, ast
            match = re.search(r"(\[.*?\])", resp, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(match.group(1))
                    except (SyntaxError, ValueError):
                        return Trajectory(objective=objective)
                
                if not isinstance(data, list):
                    return Trajectory(objective=objective)
                
                steps = [
                    TrajectoryStep(
                        phase_id=d.get('phase', 'thought'),
                        description=d.get('task', 'Processing...'),
                        probability=float(d.get('prob', 0.8)),
                        estimated_duration_s=2.0
                    ) for d in data[:3]
                ]
                traj = Trajectory(objective=objective, steps=steps, confidence=0.8)
                self.last_trajectory = traj
                
                # Broadcast to UI
                from core.event_bus import get_event_bus
                get_event_bus().publish_threadsafe("aura/ui/trajectory", {
                    "objective": objective,
                    "steps": [s.__dict__ for s in steps],
                    "confidence": traj.confidence
                })
                
                return traj
        except Exception as e:
            logger.warning(f"Trajectory prediction failed: {e}")
            
        return Trajectory(objective=objective)
