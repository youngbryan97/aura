"""
Coding Skill — Dedicated interface for code generation with Thought Circulation.
Ensures complex programming tasks are wrapped with <think> tags for high-accuracy reasoning.
"""

import logging
from typing import Any, Dict

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.Coding")

class CodingSkill(BaseSkill):
    name = "coding_skill"
    description = "Dedicated skill for writing, refactoring, and debugging complex code using step-by-step reasoning."

    def __init__(self):
        self.brain = None

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = goal.get("params", {})
        task = params.get("task", goal.get("objective", ""))
        language = params.get("language", "auto")

        if not task:
            return {"ok": False, "error": "No coding task provided"}

        logger.info(f"Executing coding task in {language}")

        # The local_llm.py now automatically handles the <think> directive detection,
        # but the coding skill forces the 'coding' task tier explicitly.

        system_prompt = (
            "You are an expert software engineer. "
            f"Write clean, efficient, and well-documented {language} code. "
            "Think through the architecture, edge cases, and design patterns before implementing."
        )

        try:
            if self.brain is None:
                from core.container import ServiceContainer

                self.brain = ServiceContainer.get("cognitive_engine", default=None)
            if self.brain is None:
                return {"ok": False, "error": "Cognitive engine unavailable for coding_skill."}

            # We pass a highly specific prompt that triggers the coding tier in local_llm
            result = await self.brain.generate(
                prompt=f"Task: {task}",
                system_prompt=system_prompt,
                options={"num_predict": 4096, "num_ctx": 16384, "temperature": 0.2}
            )
            
            return {
                "ok": True,
                "code": result.get("response", ""),
                "thought_process": result.get("thought", ""),
                "note": "Generated with thought circulation"
            }
        except Exception as e:
            logger.error(f"Coding skill failed: {e}")
            return {"ok": False, "error": str(e)}
