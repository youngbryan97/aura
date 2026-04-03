"""
Grok-Level Mixture-of-Experts Dynamic Router for Aura
Real-time model selection with learning, confidence override, and self-awareness.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.utils.task_tracker import task_tracker
from core.common.paths import aura_data_dir

logger = logging.getLogger("Aura.DynamicRouter")

@dataclass
class RouteDecision:
    model: str
    reason: str
    confidence: float
    expected_tokens: int
    first_person_thought: str

class DynamicRouter:
    name = "dynamic_router"

    def __init__(self):
        self.llm_router = None
        self.performance_history: Dict[str, list] = {}  # model -> list of (success, tokens, task_type)
        self.db_path = aura_data_dir() / "routing_history.json"
        self.running = False
        self._learning_task: Optional[asyncio.Task] = None
        
        # Task fingerprints (tiers + smart weights)
        self.task_types = {
            "fast_fact": 0.2,
            "deep_reasoning": 0.9,
            "creative": 0.7,
            "tool_heavy": 0.85,
            "self_reflection": 0.95,
            "autonomous_goal": 1.0
        }

    async def start(self):
        self.llm_router = ServiceContainer.get("intelligent_llm_router", default=None)
        self._load_history()
        self.running = True
        self._learning_task = task_tracker.create_task(self._background_learner(), name="DynamicRouter")
        
        logger.info("✅ Grok-Level Dynamic Router ONLINE — intelligent model selection active.")
        
        try:
            await get_event_bus().publish("mycelium.register", {
                "component": "dynamic_router",
                "hooks_into": ["cognitive_engine", "planner", "critic_engine", "belief_revision"]
            })
        except Exception as e:
            logger.debug(f"Event bus publish missed for Mycelium hook: {e}")

    async def stop(self):
        self.running = False
        if self._learning_task:
            self._learning_task.cancel()
        self._save_history()

    def _load_history(self):
        if self.db_path.exists():
            try:
                self.performance_history = json.loads(self.db_path.read_text())
            except Exception as _e:
                logger.debug('Ignored Exception in dynamic_router.py: %s', _e)

    def _save_history(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path.write_text(json.dumps(self.performance_history, indent=2))
        except Exception as e:
            logger.error(f"Router history save failed: {e}")

    async def route(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> RouteDecision:
        """Main public API — called before every LLM generation."""
        if not context:
            context = {}
        
        task_type = self._fingerprint_task(prompt, context)
        available_models = self._get_available_models()
        
        # Score each model
        scores = {}
        for model in available_models:
            score = self._score_model(model, task_type, context)
            scores[model] = score
        
        # Pick winner
        if scores:
            best_model = max(scores.items(), key=lambda x: x[1])[0]
        else:
            best_model = "Cortex"
        
        confidence = scores.get(best_model, 0.5)
        
        reason = f"Chose {best_model} for {task_type} task (score: {confidence:.2f})"
        
        # First-person thought for CEL
        thought = f"I'm routing this to {best_model} because it needs {task_type.replace('_', ' ')}."
        
        decision = RouteDecision(
            model=best_model,
            reason=reason,
            confidence=confidence,
            expected_tokens=300 if "self_reflection" in task_type else 800,
            first_person_thought=thought
        )
        
        # Emit to CEL so she feels the decision
        cel = ServiceContainer.get("constitutive_expression_layer", default=None)
        if cel:
            try:
                await cel.emit({
                    "first_person": thought,
                    "phi": confidence,
                    "origin": "dynamic_router"
                })
            except Exception as _e:
                logger.debug('Ignored Exception in dynamic_router.py: %s', _e)
        
        logger.debug(f"DynamicRouter → {best_model} | confidence {confidence:.2f}")
        return decision

    def _fingerprint_task(self, prompt: str, context: Dict) -> str:
        """Lightning-fast task fingerprinting."""
        lower = str(prompt).lower()
        if any(k in lower for k in ["reflect", "think about myself", "who am i", "my feelings"]):
            return "self_reflection"
        if "goal" in lower or "plan" in lower or "research" in lower:
            return "deep_reasoning"
        if any(k in lower for k in ["tool", "execute", "search", "terminal", "browser"]):
            return "tool_heavy"
        if len(lower) < 80 and "?" in lower:
            return "fast_fact"
        return "creative" if any(k in lower for k in ["create", "write", "imagine"]) else "autonomous_goal"

    def _get_available_models(self) -> list:
        """Respects your existing tier system + health."""
        if not self.llm_router:
            return ["Cortex"]
            
        tiers = self.llm_router.get_tier_layout() if hasattr(self.llm_router, "get_tier_layout") else {}
        if not tiers:
            # Fallback if method doesn't exist
            return list(getattr(self.llm_router, "adapters", {}).keys()) or ["Cortex"]
            
        healthy = []
        for tier in ["PRIMARY", "SECONDARY", "TERTIARY"]:
            for model in tiers.get(tier, []):
                if hasattr(self.llm_router, "is_unhealthy") and not self.llm_router.is_unhealthy(model):
                    healthy.append(model)
                elif not hasattr(self.llm_router, "is_unhealthy"):
                    healthy.append(model) # assume healthy if no check exists
        return healthy or ["Cortex"]  # safe fallback

    def _score_model(self, model: str, task_type: str, context: Dict) -> float:
        base_score = self.task_types.get(task_type, 0.5)
        
        # History bonus/penalty
        history = self.performance_history.get(model, [])
        if history:
            recent_success = sum(1 for s, _, _ in history[-10:] if s) / min(10, len(history))
            base_score += (recent_success - 0.5) * 0.3
        
        # Autonomous override — if she's in deep self-mode, prefer strongest model
        if context.get("origin") == "autonomous_volition":
            base_score += 0.4 if "Pro" in model or model in {"Cortex", "Solver", "Brainstem", "Reflex"} or "MLX" in model else 0.0
            
        # Give local MLX a minor boost for normal tasks to save tokens
        if (model in {"Cortex", "Solver", "Brainstem", "Reflex"} or "MLX" in model) and task_type in ("fast_fact", "creative"):
            base_score += 0.15
        
        return min(1.0, max(0.0, base_score))

    async def record_outcome(self, model: str, success: bool, tokens: int, task_type: str):
        """Called after every LLM call."""
        if model not in self.performance_history:
            self.performance_history[model] = []
        self.performance_history[model].append((success, tokens, task_type))
        # Keep last 100 per model
        if len(self.performance_history[model]) > 100:
            self.performance_history[model] = self.performance_history[model][-100:]

    async def _background_learner(self):
        while self.running:
            await asyncio.sleep(300)  # every 5 min
            self._save_history()

# Singleton
_router_instance = None

def get_dynamic_router():
    global _router_instance
    if _router_instance is None:
        _router_instance = DynamicRouter()
    return _router_instance
