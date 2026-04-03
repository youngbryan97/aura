"""Critic Pipeline — Self-Critique & Refinement Loop

Takes an initial LLM response, asks the LLM to critique it,
then refines based on the critique. Can run multiple rounds.

This implements a "generate → critique → refine" loop that
significantly improves output quality for complex tasks.
"""
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("Pipeline.Critic")


class CriticPipeline:
    """Critique-then-refine loop for LLM outputs.

    Parameters
    ----------
    think_fn : callable
        An async function that takes a prompt string and returns a response string.
        Typically: async def think(prompt) -> str
    max_rounds : int
        Maximum number of critique-refine iterations (default 1).

    """

    CRITIQUE_PROMPT = """## SELF-CRITIQUE

You are reviewing your own previous response for quality. Be ruthlessly honest.

**Original Query:** {query}

**Your Previous Response:**
{response}

**Evaluate on these axes:**
1. **Accuracy** — Are there any factual errors or unsupported claims?
2. **Completeness** — Did you miss any important aspects?
3. **Clarity** — Is the response easy to understand?
4. **Conciseness** — Is there unnecessary verbosity?
5. **Actionability** — Can the user act on this immediately?

Provide specific, constructive critique. If the response is already excellent, say "NO_ISSUES"."""

    REFINE_PROMPT = """## REFINEMENT

Improve your response based on the critique below.

**Original Query:** {query}

**Your Previous Response:**
{response}

**Critique:**
{critique}

Produce an improved, final response that addresses all valid critique points.
Do NOT mention the critique process — just deliver the refined answer."""

    def __init__(
        self,
        think_fn: Optional[Callable] = None,
        max_rounds: int = 1,
    ):
        self.think_fn = think_fn
        self.max_rounds = max_rounds

    async def critique_and_refine(
        self,
        query: str,
        initial_response: str,
    ) -> str:
        """Run the critique-refine loop.

        Returns the refined response (or original if critique finds no issues).
        """
        if not self.think_fn:
            logger.warning("No think_fn provided — returning original response.")
            return initial_response

        current = initial_response

        for round_num in range(self.max_rounds):
            logger.info("🔍 Critic round %d/%d", round_num + 1, self.max_rounds)

            # Step 1: Critique
            critique_prompt = self.CRITIQUE_PROMPT.format(
                query=query,
                response=current,
            )
            try:
                critique = await self.think_fn(critique_prompt)
            except Exception as exc:
                logger.warning("Critique failed (round %d): %s", round_num + 1, exc)
                break

            # Check if critique found no issues
            if "NO_ISSUES" in critique.upper():
                logger.info("🔍 Critic found no issues — keeping response.")
                break

            # Step 2: Refine
            refine_prompt = self.REFINE_PROMPT.format(
                query=query,
                response=current,
                critique=critique,
            )
            try:
                refined = await self.think_fn(refine_prompt)
                current = refined
                logger.info("🔍 Response refined (round %d, %d chars)", round_num + 1, len(current))
            except Exception as exc:
                logger.warning("Refinement failed (round %d): %s", round_num + 1, exc)
                break

        return current