from core.runtime.errors import record_degradation
import logging

from ..brain.cognitive_engine import CognitiveEngine

logger = logging.getLogger("Motivation.Critic")

class AestheticCritic:
    """Judges the QUALITY and BEAUTY of work.
    (Code elegance, conversational tone, structural cleanliness).
    """
    
    def __init__(self, cognitive_engine: CognitiveEngine):
        self.brain = cognitive_engine
        
    async def critique_code(self, code: str, language: str = "python") -> dict:
        """Critique the aesthetics of code.
        Returns score (0-10) and feedback.
        """
        context = "Review the following code for ELEGANCE, READABILITY, and IDIOMATIC STYLE."
        if len(code) > 2000:
            code = code[:2000] + "\n... (truncated)"
            
        prompt = f"""
        You are an AESTHETIC CRITIC.
        Role: Judge the code not just on function, but on BEAUTY.
        
        Criteria:
        1. Cleanliness (Spacing, naming).
        2. Idiomatic usage (Pythonic).
        3. Documentation (Docstrings, comments).
        4. Simplicity (Avoid over-engineering).
        
        Code:
        ```{language}
        {code}
        ```
        
        Return JSON:
        {{
            "score": 0-10,
            "critique": "One sentence summary of style.",
            "suggestions": ["suggestion 1", "suggestion 2"]
        }}
        """
        
        try:
            response = await self.brain.think(
                objective=prompt, 
                context={"role": "critic"},
                mode="fast"
            )
            
            import json
            import re
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"score": 5, "critique": "Could not parse critique.", "suggestions": []}
        except Exception as e:
            record_degradation('aesthetic_critic', e)
            logger.error("Critique failed: %s", e)
            return {"score": 0, "error": str(e)}

    async def critique_thought(self, thought_content: str) -> dict:
        """Critique the reasoning process itself.
        """
        # Prose and logic coherence evaluation
        prompt = f"""
        You are an AESTHETIC CRITIC.
        Review the following reasoning for LOGIC, CLARITY, and ELEGANCE.

        Thought:
        \"\"\"{thought_content}\"\"\"

        Return JSON:
        {{
            "score": 0-10,
            "critique": "Short summary of reasoning clarity.",
            "suggestions": ["suggestion 1", "suggestion 2"]
        }}
        """

        try:
            response = await self.brain.think(
                objective=prompt, 
                context={"role": "critic"},
                mode="fast"
            )
            import json
            import re
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"score": 5, "critique": "Valid thought but could not parse explicit critique.", "suggestions": []}
        except Exception as e:
            record_degradation('aesthetic_critic', e)
            logger.error("Thought critique failed: %s", e)
            return {"score": 0, "error": str(e), "critique": "Failed to analyze reasoning."}