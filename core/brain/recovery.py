# core/brain/recovery.py
"""
Error Recovery Pattern.
Proposes strategies to recover from failures automatically.
"""

from core.brain.llm_interface import LLMInterface

class RecoveryEngine:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def recover(self, error: str, context: str) -> str:
        """Propose a recovery strategy for a given error and context."""
        prompt = f"""
An error occurred:

{error}

Context:
{context}

Propose a strategy to recover.
"""
        response = await self.model.generate(prompt)
        return response
