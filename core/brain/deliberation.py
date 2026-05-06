# core/brain/deliberation.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import asyncio
import json

from core.brain.llm_interface import LLMInterface
from core.brain.trace_logger import TraceLogger
from core.runtime.errors import record_degradation

@dataclass
class Decision:
    action: str
    reason: str
    raw: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        if actions and opts.get("use_native_system2", True):
            system2_decision = await self._native_system2_deliberate(context, actions, **opts)
            if system2_decision is not None:
                if self.trace:
                    self.trace.log({
                        "type": "native_system2_deliberation",
                        "context": context[:300],
                        "actions": actions,
                        "decision": {
                            "action": system2_decision.action,
                            "reason": system2_decision.reason,
                            "confidence": system2_decision.confidence,
                            "metadata": system2_decision.metadata,
                        },
                    })
                return system2_decision

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

    async def _native_system2_deliberate(self, context: str, actions: List[str], **opts) -> Optional[Decision]:
        """Use Aura's native governed System 2 search to choose among actions.

        This is deliberately a commitment to a plan, not execution of the
        action. Any actual side effect still goes through the normal Will/tool
        governance path.
        """
        if len(actions) < 2:
            return None
        try:
            from core.container import ServiceContainer
            from core.reasoning.native_system2 import SearchAlgorithm, System2SearchConfig

            system2 = ServiceContainer.get("native_system2", default=None)
            if system2 is None:
                return None

            cfg = System2SearchConfig(
                algorithm=SearchAlgorithm.HYBRID,
                budget=int(opts.get("system2_budget", max(12, min(72, len(actions) * 12)))),
                max_depth=int(opts.get("system2_depth", 2)),
                branching_factor=max(1, len(actions)),
                beam_width=max(1, min(5, len(actions))),
                seed=opts.get("seed"),
                confidence_threshold=float(opts.get("system2_confidence_threshold", 0.56)),
            )
            ranked = await system2.rank_actions(
                context=context,
                actions=[
                    {
                        "name": action,
                        "prior": 1.0 / max(1, len(actions)),
                        "metadata": {"index": idx},
                    }
                    for idx, action in enumerate(actions)
                ],
                config=cfg,
                source="deliberation_controller",
            )
            selected = ranked.committed_action
            if selected is None:
                return None
            chosen = selected.metadata.get("verifies") or selected.name
            if str(chosen).startswith("verify:"):
                chosen = str(chosen)[len("verify:") :]
            action = next((candidate for candidate in actions if candidate == chosen), str(chosen))
            return Decision(
                action=action,
                reason=ranked.receipt.commitment_reason,
                raw=json.dumps(ranked.receipt.to_dict(), sort_keys=True),
                confidence=ranked.confidence,
                metadata={
                    "native_system2": True,
                    "system2_search_id": ranked.search_id,
                    "system2_algorithm": ranked.algorithm.value,
                    "system2_receipt": ranked.receipt.to_dict(),
                    "will_receipt_id": ranked.receipt.will_receipt_id,
                },
            )
        except Exception as exc:
            record_degradation("deliberation.native_system2", exc)
            return None
