"""Prompt Scaffold — Structured Reasoning Template

Wraps user queries in a structured reasoning framework:
  Restate → Constraints → Assumptions → Solutions → Critique → Selection → Final Answer

This forces the LLM to think step-by-step, dramatically improving
response quality and consistency for complex queries.
"""
import logging
from typing import Optional

logger = logging.getLogger("Pipeline.PromptScaffold")


class PromptScaffold:
    """Builds structured prompts that guide LLM reasoning.
    Used by the cognitive loop for DEEP thinking mode.
    """

    # The scaffold template — each section forces a cognitive step
    SCAFFOLD_TEMPLATE = """## STRUCTURED REASONING

### 1. RESTATE THE PROBLEM
Restate the user's request in your own words to confirm understanding.

### 2. CONSTRAINTS
List any hard constraints, limitations, or requirements.

### 3. ASSUMPTIONS
State any assumptions you are making.

### 4. CANDIDATE SOLUTIONS
Propose 2-3 candidate approaches. For each, briefly note pros/cons.

### 5. CRITIQUE
Critically evaluate each candidate. Which is most robust? Which has hidden failure modes?

### 6. SELECTION
Choose the best approach and justify why.

### 7. FINAL ANSWER
Deliver the complete, polished answer.

---

**User Query:**
{query}

{context_block}

Begin your structured reasoning:"""

    LIGHT_TEMPLATE = """{query}

{context_block}

Respond concisely and directly."""

    def __init__(self):
        raise NotImplementedError("Aura Pass 2: Unimplemented Stub")

    def build_structured_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        mode: str = "deep",
    ) -> str:
        """Build a reasoning prompt appropriate to the thinking mode.

        Parameters
        ----------
        query : str
            The user's question or task.
        context : str, optional
            Additional context (memories, system state, etc.)
        mode : str
            "deep" → full scaffold, "light" → minimal wrapper.

        """
        context_block = f"**Context:**\n{context}" if context else ""

        # Use safe substitution to avoid KeyError on user queries containing { or }
        if mode == "deep":
            prompt = self.SCAFFOLD_TEMPLATE.replace("{query}", query).replace(
                "{context_block}", context_block
            )
            logger.debug("Built DEEP scaffold prompt (%d chars)", len(prompt))
        else:
            prompt = self.LIGHT_TEMPLATE.replace("{query}", query).replace(
                "{context_block}", context_block
            )
            logger.debug("Built LIGHT prompt (%d chars)", len(prompt))

        return prompt