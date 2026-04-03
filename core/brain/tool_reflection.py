# core/brain/tool_reflection.py
"""
Tool Reflection Pattern.
Verifies whether a tool output actually solved the problem.
"""

from core.brain.llm_interface import LLMInterface

class ToolReflection:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def evaluate(self, task: str, tool_output: str) -> str:
        """Evaluate if the tool successfully solved the task."""
        prompt = f"""
Task:
{task}

Tool output:
{tool_output}

Did the tool successfully solve the task?

Answer YES or NO and explain briefly.
"""
        response = await self.model.generate(prompt)
        return response
