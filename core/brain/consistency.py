# core/brain/consistency.py
"""
Self-Consistency Reasoning Pattern.
Generates multiple reasoning paths and chooses the most consistent result.
"""

import collections
from core.brain.llm_interface import LLMInterface

class SelfConsistency:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def solve(self, prompt: str, samples: int = 5) -> str:
        """Generate multiple answers and return the most common one."""
        answers = []
        for _ in range(samples):
            response = await self.model.generate(prompt, temperature=0.7)
            answers.append(response.strip())

        if not answers:
            return ""

        counter = collections.Counter(answers)
        best = counter.most_common(1)[0][0]
        return best
