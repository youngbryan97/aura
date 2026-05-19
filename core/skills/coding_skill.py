"""
Coding Skill — Dedicated interface for code generation with Thought Circulation.
Ensures complex programming tasks are wrapped with <think> tags for high-accuracy reasoning.
"""

from core.runtime.errors import record_degradation
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

        logger.info("Executing coding task in %s", language)

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
            raw_result = await self.brain.generate(
                prompt=f"Task: {task}",
                system_prompt=system_prompt,
                origin=str(context.get("origin") or "api"),
                purpose="coding",
                prefer_tier="primary",
                deep_handoff=bool(context.get("deep_handoff", False)),
                max_tokens=int(context.get("max_tokens", 4096) or 4096),
                temperature=float(context.get("temperature", 0.2) or 0.2),
                use_strategies=True,
            )
            if isinstance(raw_result, dict):
                code = str(raw_result.get("response") or raw_result.get("text") or "")
                thought = str(raw_result.get("thought") or raw_result.get("thought_process") or "")
            else:
                code = str(raw_result or "")
                thought = ""
            
            return {
                "ok": True,
                "code": code,
                "thought_process": thought,
                "note": "Generated through foreground coding reasoning",
            }
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('coding_skill', e)
            logger.error("Coding skill failed: %s", e)
            return {"ok": False, "error": str(e)}
