# core/brain/deliberation.py
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm_interface import LLMInterface
from core.brain.trace_logger import TraceLogger
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service


@dataclass
class Decision:
    action: str
    reason: str
    raw: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def __init__(self, llm: LLMInterface, trace: TraceLogger | None = None):
        self.llm = llm
        self.trace = trace

    #: Ceiling for one ordinary deliberation call.
    #:
    #: CP126 (high): "Ordinary generation has no deadline or typed fallback.
    #: The model call can hang and exceptions escape without a decision
    #: receipt; native failures silently drop to the weaker ordinary route."
    #:
    #: A deliberation picks among actions the caller is about to take, so a
    #: hang here stalls the decision AND everything waiting on it, with no
    #: record that a decision was ever attempted.
    DELIBERATION_TIMEOUT_S = 45.0

    async def deliberate(self, context: str, actions: list[str], temperature: float = 0.2, **opts) -> Decision:
        native_declined = ""
        if actions and opts.get("use_native_system2", True):
            system2_decision = await self._native_system2_deliberate(context, actions, **opts)
            if system2_decision is None:
                # The downgrade to the weaker route was invisible. It is a
                # real change in decision quality and belongs in the receipt.
                native_declined = "native_system2_unavailable"
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
        timeout_s = float(opts.get("deliberation_timeout_s") or self.DELIBERATION_TIMEOUT_S)
        degraded = ""
        try:
            raw = await asyncio.wait_for(
                self.llm.generate(prompt, temperature=temperature, **opts),
                timeout=max(1.0, timeout_s),
            )
        except asyncio.CancelledError:
            # The caller's decision, not a failure to absorb.
            raise
        except TimeoutError:
            degraded = f"deliberation_timeout:{timeout_s:.0f}s"
            raw = ""
            record_degradation(
                "deliberation",
                TimeoutError(degraded),
                severity="warning",
                action="returned the first action with a degraded receipt rather than hanging",
                enforce_failure_policy=False,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            degraded = f"deliberation_failed:{type(exc).__name__}"
            raw = ""
            record_degradation(
                "deliberation",
                exc,
                severity="warning",
                action="returned a typed fallback decision rather than letting the error escape",
                enforce_failure_policy=False,
            )

        dec = self._parse(raw, actions)
        # The receipt is the point. A fallback that looks like a considered
        # choice is worse than no choice: downstream cannot tell that this
        # action was picked by position rather than by reasoning.
        if degraded or native_declined:
            dec.metadata = dict(dec.metadata or {})
            dec.metadata["degraded"] = degraded or ""
            dec.metadata["native_system2_declined"] = native_declined or ""
            dec.metadata["deliberated"] = not degraded
            if degraded:
                dec.confidence = 0.0
                dec.reason = dec.reason or "no deliberation was possible for this turn"
        if self.trace:
            self.trace.log({
                "type": "deliberation",
                "context": context[:300],
                "actions": actions,
                "raw": raw,
                "degraded": degraded,
                "native_system2_declined": native_declined,
                "decision": {"action": dec.action, "reason": dec.reason, "confidence": dec.confidence}
            })
        return dec

    def _parse(self, raw: str, actions: list[str]) -> Decision:
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
            except (RuntimeError, AttributeError, TypeError, ValueError):
                confidence = None
        if action is None:
            # fallback: pick first action
            action = actions[0] if actions else ""
        return Decision(action=action, reason=reason, raw=raw, confidence=confidence, metadata={})

    async def _native_system2_deliberate(self, context: str, actions: list[str], **opts) -> Decision | None:
        """Use Aura's native governed System 2 search to choose among actions.

        This is deliberately a commitment to a plan, not execution of the
        action. Any actual side effect still goes through the normal Will/tool
        governance path.
        """
        if len(actions) < 2:
            return None
        try:
            from core.reasoning.native_system2 import SearchAlgorithm, System2SearchConfig

            system2 = get_runtime_service("native_system2", default=None)
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
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("deliberation.native_system2", exc)
            return None
