# core/brain/decomposition.py
"""
Task Decomposition Engine.
Breaks down complex objectives into smaller, manageable tasks.
"""

from typing import List
from core.brain.llm_interface import LLMInterface

class TaskDecomposer:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def decompose(self, objective: str) -> List[str]:
        """Break an objective into smaller tasks."""
        prompt = f"""
Break this objective into smaller tasks.

Objective:
{objective}
"""
        tasks_raw = await self.model.generate(prompt)
        # Split by newline and filter empty lines/bullet points
        tasks = [t.strip().lstrip("-*123456789. ") for t in tasks_raw.split("\n") if t.strip()]
        return tasks
