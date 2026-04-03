# core/brain/hierarchical.py
"""
Hierarchical Control Pattern.
Separates reasoning into Strategic, Tactical, and Execution layers.
"""

from core.brain.llm_interface import LLMInterface

class HierarchicalController:
    def __init__(self, strategic_model: LLMInterface, tactical_model: LLMInterface):
        self.strategic_model = strategic_model
        self.tactical_model = tactical_model

    async def determine_goal(self, user_input: str) -> str:
        """Strategic Layer: What is the real objective?"""
        prompt = f"""
User input:
{user_input}

What is the underlying goal?
Respond with a concise objective.
"""
        goal = await self.strategic_model.generate(prompt)
        return goal.strip()

    async def generate_plan(self, goal: str) -> str:
        """Tactical Layer: What plan achieves this?"""
        prompt = f"""
Goal:
{goal}

Create a step-by-step plan to accomplish this goal.
"""
        plan = await self.tactical_model.generate(prompt)
        return plan
