# core/brain/deliberation.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
import asyncio

from core.brain.llm_interface import LLMInterface
from core.brain.trace_logger import TraceLogger

@dataclass
class Decision:
    action: str
    reason: str
    raw: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None

class DeliberationController:
    """
    Given context and candidate actions, ask the LLM to deliberate and choose.
    Produces: Decision(action, reason, raw, confidence).
    """

    DEFAULT_PROMPT = """
You are an agent deliberator. Given the CONTEXT and a numbered list of ACTIONS,
explain briefly which action is best and why, then output:

Action: <number or action-name>
Reason: <short reasoning, 1-2 sentences>
Confidence: <0.0-1.0>

Keep answers concise.
"""

    def __init__(self, llm: LLMInterface, trace: Optional[TraceLogger] = None):
        self.llm = llm
        self.trace = trace

    async def deliberate(self, context: str, actions: List[str], temperature: float = 0.2, **opts) -> Decision:
        numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
        prompt = self.DEFAULT_PROMPT + "\n\nCONTEXT:\n" + context + "\n\nACTIONS:\n" + numbered + "\n\nAnswer:"
        # Use existing LLM router or the new interface
        raw = await self.llm.generate(prompt, temperature=temperature, **opts)
        dec = self._parse(raw, actions)
        if self.trace:
            self.trace.log({
                "type": "deliberation",
                "context": context[:300],
                "actions": actions,
                "raw": raw,
                "decision": {"action": dec.action, "reason": dec.reason, "confidence": dec.confidence}
            })
        return dec

    def _parse(self, raw: str, actions: List[str]) -> Decision:
        # Extract Action:
        action = None
        reason = ""
        confidence = None
        # common patterns:
        m = re.search(r"Action\s*:\s*(.+)", raw, flags=re.IGNORECASE)
        if m:
            a = m.group(1).strip()
            # if number, map to action
            if re.match(r"^\d+$", a):
                idx = int(a) - 1
                if 0 <= idx < len(actions):
                    action = actions[idx]
                else:
                    action = a
            else:
                # try exact match
                cand = next((x for x in actions if x.lower().startswith(a.lower())), None)
                action = cand or a
        m2 = re.search(r"Reason\s*:\s*(.+)", raw, flags=re.IGNORECASE)
        if m2:
            reason = m2.group(1).strip()
        m3 = re.search(r"Confidence\s*:\s*([0-9]*\.?[0-9]+)", raw, flags=re.IGNORECASE)
        if m3:
            try:
                confidence = float(m3.group(1))
            except Exception:
                confidence = None
        if action is None:
            # fallback: pick first action
            action = actions[0] if actions else ""
        return Decision(action=action, reason=reason, raw=raw, confidence=confidence, metadata={})
