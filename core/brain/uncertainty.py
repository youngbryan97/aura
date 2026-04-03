# core/brain/uncertainty.py
"""
Uncertainty Estimation Pattern.
Estimates the system's confidence in its own answers.
"""

from core.brain.llm_interface import LLMInterface

class ConfidenceEstimator:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def estimate(self, question: str, answer: str) -> float:
        """Estimate confidence between 0 and 1 for a given answer."""
        prompt = f"""
Question:
{question}

Answer:
{answer}

Estimate confidence between 0 and 1.
"""
        response = await self.model.generate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        try:
            return float(content.strip())
        except Exception:
            # Fallback for non-numeric responses
            return 0.5
