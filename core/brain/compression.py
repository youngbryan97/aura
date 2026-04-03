# core/brain/compression.py
"""
Context Compression Pattern.
Summarizes long interaction history to fit within context limits.
"""

from core.brain.llm_interface import LLMInterface

class ContextCompressor:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def compress(self, history: str) -> str:
        """Summarize interaction history while preserving key facts."""
        prompt = f"""
Summarize the following interaction history
while preserving important facts.

{history}
"""
        response = await self.model.generate(prompt)
        if hasattr(response, 'content'):
            return response.content
        return str(response)
