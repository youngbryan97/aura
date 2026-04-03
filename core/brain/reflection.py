# core/brain/reflection.py
from typing import Any, Dict, Optional
import asyncio

from core.brain.llm_interface import LLMInterface
from core.brain.trace_logger import TraceLogger

class ReflectionEngine:
    """
    After an action finishes, ask the LLM to analyze the outcome, produce lessons and next-step suggestions.
    The reflection results are returned and optionally stored via a callback.
    """

    PROMPT = """
You are an analysis assistant. Given the CONTEXT, the ACTION taken, and the OUTCOME,
write a brief 'lesson' (1-2 sentences) and optionally a corrective next-step if the outcome was poor.

CONTEXT:
{context}

ACTION:
{action}

OUTCOME:
{outcome}

Respond as JSON:
{{ "lesson": "...", "next_step": "..." }}
"""

    def __init__(self, llm: LLMInterface, trace: Optional[TraceLogger] = None):
        self.llm = llm
        self.trace = trace

    async def reflect(self, context: str, action: str, outcome: str) -> Dict[str, str]:
        prompt = self.PROMPT.format(context=context[:1000], action=action, outcome=outcome[:1000])
        raw = await self.llm.generate(prompt, temperature=0.0)
        # try to parse json
        import json, re
        # crude JSON extraction
        m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        parsed = {"lesson": "", "next_step": ""}
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                # fallback: try to extract lines
                lines = raw.splitlines()
                parsed["lesson"] = lines[0] if lines else ""
        else:
            parsed["lesson"] = raw.strip().splitlines()[0][:500]
        if self.trace:
            self.trace.log({"type": "reflection", "context": context[:300], "action": action, "outcome": outcome, "raw": raw, "parsed": parsed})
        return parsed
