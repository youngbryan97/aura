# core/brain/debate.py
"""
Internal Debate Pattern.
Uses multiple perspectives to improve reasoning quality.
"""

from core.brain.llm_interface import LLMInterface

class InternalDebate:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def debate(self, question: str) -> str:
        """Run an internal debate between two perspectives and judge the result."""
        prompt1 = f"Argue for solution A to this problem:\n\n{question}"
        prompt2 = f"Argue against solution A and propose a better solution.\n\n{question}"

        argument1 = await self.model.generate(prompt1)
        argument2 = await self.model.generate(prompt2)

        judge_prompt = f"""
Argument 1:
{argument1}

Argument 2:
{argument2}

Which is better and why?
"""
        final = await self.model.generate(judge_prompt)
        return final
