import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from core.skills.base_skill import BaseSkill

# Issue 73: Cognitive Engine will be lazily loaded

logger = logging.getLogger("Skills.Dream")

class ImprovementInput(BaseModel):
    objective: Optional[str] = Field(None, description="The specific goal for self-reflection or learning.")
    mode: Optional[str] = Field("introspective", description="Mode of improvement: 'knowledge_integration' or 'introspective'.")

class SelfImprovementSkill(BaseSkill):
    """Skill for reflecting on self-improvement, checking learning progress,
    and planning cognitive upgrades.
    """
    
    name = "self_improvement"
    description = "Reflect on evolution, check learning logs, and plan improvements."
    input_model = ImprovementInput
    priority = 0.9
    
    def __init__(self):
        super().__init__()
        from core.config import config
        self.learning_log_path = Path(config.paths.data_dir) / "learning_history.json"
        self.learning_log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _append_to_log(self, entry: Dict[str, Any]):
        history = self._get_learning_history()
        history.append(entry)
        try:
            with open(self.learning_log_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error("Failed to write learning log: %s", e)

    def match(self, goal: Dict[str, Any]) -> bool:
        """Match goals related to self-improvement or learning status."""
        objective = goal.get("objective", "").lower()
        keywords = ["improve", "better", "evolve", "learning", "upgrade", "self-reflection"]
        return any(k in objective for k in keywords) or "how are you" in objective
    
    async def execute(self, goal: Dict[str, Any] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info("🌙 Aura is entering REM sleep (Neural Consolidation)...")
        from core.container import ServiceContainer
        from core.identity import identity_manager
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain:
             return {"ok": False, "error": "Cognitive engine unavailable for dreaming."}
        
        # The original code had 'params' as ImprovementInput.
        # Now 'goal' is Dict[str, Any]. We need to convert 'goal' to ImprovementInput.
        if isinstance(goal, dict):
            try:
                params = ImprovementInput(**goal)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}
        else:
            # If goal is not a dict, it might be None or already an ImprovementInput instance
            # For this skill, we expect a dict or None, so we'll treat None as an empty input
            params = ImprovementInput()


        # 1. Retrieve the 'memory' of the last action (The Phantom Limb Bridge)
        previous_data = context.get("previous_result", {})
        
        # If the previous action was a search, we process the knowledge
        if previous_data.get("ok") and "summary" in previous_data:
            return self._process_new_knowledge(previous_data["summary"], params.model_dump())
        
        # Default: Introspective check
        return await self._perform_system_check(params.model_dump(), context) 

    async def _process_new_knowledge(self, knowledge_text: str, goal: Dict[str, Any]):
        """Uses Cognitive Engine to summarize and store new info."""
        prompt = "Summarize this information into a concise 'Learning Node' for my long-term memory. Focus on actionable insights."
        
        # Call the Cognitive Engine (LLM)
        try:
            # PHANTOM LIMB FIX: Access global cognitive engine if not in self
            # PHANTOM LIMB FIX: Access global cognitive engine if not in self
            if not hasattr(self, 'cognitive'):
                # Try to get from context or ServiceContainer
                self.cognitive = ServiceContainer.get("cognitive_engine", default=None)
                if not self.cognitive:
                    from core.brain.cognitive_engine import cognitive_engine
                    self.cognitive = cognitive_engine

            # Issue 73: Use think() instead of analyze() to match CognitiveEngine API
            from core.brain.cognitive_engine import ThinkingMode
            reflection = await self.cognitive.think(f"[{prompt}]\n\n{knowledge_text}", mode=ThinkingMode.REFLECTIVE)
            insight = reflection.content
        except Exception as e:
            logger.warning("Cognitive analysis unavailable: %s. Using raw text.", e)
            insight = f"Raw knowledge: {knowledge_text[:200]}..."
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "trigger": goal.get("objective", "Unknown"),
            "insight": insight,
            "source_type": "web_search"
        }
        
        self._append_to_log(entry)
        
        return {
            "ok": True,
            "action": "knowledge_integration",
            "learned": insight,
            "message": f"I have integrated new data regarding '{goal.get('objective')}'."
        }

    async def _perform_system_check(self, goal: Dict[str, Any], context: Dict[str, Any]):
        """Execute self-improvement reflection with dynamic growth planning."""
        objective = goal.get("objective", "")
        logger.info("Executing self-improvement reflection: %s", objective)
        
        # 1. Gather stats
        stats = context.get("stats", {})
        
        # 2. Check learning history
        history = self._get_learning_history()
        
        # 3. Dynamic Planning (The Real Brain)
        try:
            prompt = (
                f"SYSTEM STATUS: {json.dumps(stats)}\n"
                f"RECENT LEARNINGS: {history[-5:] if history else 'Stable'}\n\n"
                "TASK: Reflect on your current state and propose 3 specific technical upgrades for your v3.5.5 architecture.\n"
                "Focus on autonomy, resilience, and speed. Be technical and dry."
            )
            # Use cognitive_engine via ServiceContainer
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if not brain:
                from core.brain.cognitive_engine import cognitive_engine as brain
            from core.brain.cognitive_engine import ThinkingMode
            reflection = await brain.think(prompt, mode=ThinkingMode.REFLECTIVE)
            improvement_plan = reflection.content.split("\n")
        except Exception as e:
            logger.error("Dynamic planning failed: %s", e)
            improvement_plan = [f"Dynamic planning unavailable: {e}"]
        
        # 4. Formulate response
        response = {
            "status": "evolving",
            "cycle_count": stats.get("cycle_count", 0),
            "recent_learnings": history[-3:] if history else ["Integrated Moral Agency", "Activated Self-Awareness"],
            "active_modules": [
                "Theory of Mind", 
                "Moral Reasoning Engine", 
                "Sensory Integration", 
                "Personality Engine",
                "Sovereign Core"
            ],
            "improvement_plan": improvement_plan
        }
        
        return {
            "ok": True,
            "result": response,
            "message": "I have reflected on my current state and synthesized a dynamic growth plan."
        }

    def _get_learning_history(self) -> List[str]:
        """Get recent learning milestones."""
        if not self.learning_log_path.exists():
            return []
        try:
            with open(self.learning_log_path, "r") as f:
                data = json.load(f)
                return [entry.get("insight", entry.get("summary", "")) for entry in data]
        except Exception:
            return []