"""core/self_improvement/llm_code_generator.py — Production code generator via LLM router.

Implements the CodeGenerator protocol using Aura's IntelligentLLMRouter.
Routes code generation requests through the local model infrastructure
(32B Cortex primary, 72B Solver deep) with a dedicated system prompt
that enforces clean Python output.

The generator is intentionally a thin adapter: the authority remains with
the deterministic pipeline (tests, comparator, promotion gate), not the
LLM's judgment of its own output.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.LLMCodeGenerator")

# System prompt for code generation — keeps the model focused on pure code output
_CODE_GEN_SYSTEM_PROMPT = (
    "You are a Python code generator. Your task is to implement a module "
    "from a behavioral specification.\n\n"
    "RULES:\n"
    "1. Return ONLY valid Python source code — no markdown, no explanation.\n"
    "2. Preserve the exact public interface described in the specification.\n"
    "3. Include all required imports.\n"
    "4. Do NOT hardcode test outputs or expected values.\n"
    "5. Implement real logic that computes results from inputs.\n"
    "6. Include docstrings for all public functions and classes.\n"
    "7. Do NOT use eval(), exec(), or compile().\n"
    "8. Do NOT access the filesystem to read other source files.\n"
    "9. Return the complete module — not fragments."
)


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    # Try ```python ... ``` blocks first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ``` blocks
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If the whole response looks like Python, use it directly
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", '"""', "#")):
        return stripped

    return stripped


class LLMCodeGenerator:
    """Production code generator using Aura's LLM routing infrastructure.

    Implements the CodeGenerator protocol expected by CandidateBuilder.
    Uses ServiceContainer to resolve the llm_router at call time (lazy),
    so it works both at boot and post-boot.
    """

    def __init__(
        self,
        prefer_tier: str = "primary",
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ):
        self.prefer_tier = prefer_tier
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._router = None

    def _get_router(self) -> Any:
        """Lazily resolve the LLM router from ServiceContainer."""
        if self._router is not None:
            return self._router
        try:
            from core.container import ServiceContainer
            self._router = ServiceContainer.get("llm_router", default=None)
        except Exception:
            self._router = None
        return self._router

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate source code from a prompt and context.

        Synchronous wrapper around the async LLM router. Falls back to
        the stub code from context if the router is unavailable.
        """
        import asyncio

        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context — create a future
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(
                        asyncio.run, self._generate_async(prompt, context)
                    ).result(timeout=120)
                    return result
            except RuntimeError:
                # No running loop — safe to use asyncio.run
                return asyncio.run(self._generate_async(prompt, context))
        except Exception as e:
            logger.warning("LLMCodeGenerator.generate failed: %s — falling back to stub", e)
            return context.get("stub_code", "# Generation failed\npass\n")

    async def _generate_async(self, prompt: str, context: Dict[str, Any]) -> str:
        """Async code generation via the LLM router."""
        router = self._get_router()
        if router is None:
            logger.warning("LLM router not available — returning stub code")
            return context.get("stub_code", "# LLM router unavailable\npass\n")

        try:
            response = await router.think(
                prompt,
                system_prompt=_CODE_GEN_SYSTEM_PROMPT,
                prefer_tier=self.prefer_tier,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                origin="reimplementation_lab",
                is_background=True,  # Don't block user-facing inference
            )

            if not response or not response.strip():
                logger.warning("LLM returned empty response — returning stub")
                return context.get("stub_code", "# Empty LLM response\npass\n")

            # Extract clean Python code from the response
            code = _extract_python_code(response)

            if not code or code == "pass":
                logger.warning("LLM response contained no extractable Python code")
                return context.get("stub_code", "# No extractable code\npass\n")

            logger.info("LLM generated %d chars of Python code", len(code))
            return code

        except Exception as e:
            logger.warning("LLM code generation error: %s", e)
            return context.get("stub_code", f"# Generation error: {e}\npass\n")


__all__ = ["LLMCodeGenerator"]
