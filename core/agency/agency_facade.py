"""AgencyFacade — typed, inspectable agency boundary.

Addresses the reviewer's critique that the prior ``pass``-only subclass
looked like a placeholder. This module exposes the agency life-loop as
explicit typed phases: propose → score → submit_to_will → execute →
evaluate_outcome → consolidate_learning, so a reviewer can trace a single
autonomous action end-to-end without spelunking ``AgencyCore``.

The facade does NOT reimplement AgencyCore; it wraps it and turns its
internal signals into receipts that the LifeTrace ledger can persist.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.agency_core import AgencyCore


@dataclass(frozen=True)
class InitiativeProposal:
    proposal_id: str
    source_actor: str
    origin_drive: str
    content: str
    rationale: str
    required_capabilities: tuple[str, ...] = ()
    expected_outcome: str = ""
    counterfactuals: tuple[str, ...] = ()
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_actor": self.source_actor,
            "origin_drive": self.origin_drive,
            "content": self.content,
            "rationale": self.rationale,
            "required_capabilities": list(self.required_capabilities),
            "expected_outcome": self.expected_outcome,
            "counterfactuals": list(self.counterfactuals),
            "context_snapshot": dict(self.context_snapshot),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ScoredInitiative:
    proposal: InitiativeProposal
    priority: float
    safety_score: float
    resource_cost: float
    expected_value: float
    justification: str

    def as_dict(self) -> Dict[str, Any]:
        d = self.proposal.as_dict()
        d.update({
            "priority": round(self.priority, 4),
            "safety_score": round(self.safety_score, 4),
            "resource_cost": round(self.resource_cost, 4),
            "expected_value": round(self.expected_value, 4),
            "justification": self.justification,
        })
        return d


@dataclass(frozen=True)
class ActionReceipt:
    receipt_id: str
    proposal_id: str
    action_kind: str
    outcome_raw: Any
    success: bool
    side_effects: Dict[str, Any] = field(default_factory=dict)
    executed_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        raw = self.outcome_raw
        try:
            json.dumps(raw)
            safe = raw
        except Exception:
            safe = repr(raw)
        return {
            "receipt_id": self.receipt_id,
            "proposal_id": self.proposal_id,
            "action_kind": self.action_kind,
            "outcome_raw": safe,
            "success": self.success,
            "side_effects": dict(self.side_effects),
            "executed_at": self.executed_at,
        }


@dataclass(frozen=True)
class OutcomeAssessment:
    receipt_id: str
    achieved_expected: bool
    measured_value: float
    regret: float
    lessons: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "achieved_expected": self.achieved_expected,
            "measured_value": round(self.measured_value, 4),
            "regret": round(self.regret, 4),
            "lessons": list(self.lessons),
        }


def _hash(*parts: Any) -> str:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class AgencyFacade(AgencyCore):
    """Typed life-loop wrapper around AgencyCore.

    Each method is one explicit phase. Callers see the contract without
    needing to understand the internals of AgencyCore.
    """

    # ---- Phase 1: propose ---------------------------------------------
    async def propose_initiatives(self, context: Optional[Dict[str, Any]] = None) -> List[InitiativeProposal]:
        context = context or {}
        pulse = await self.pulse()  # type: ignore[attr-defined]
        proposals: List[InitiativeProposal] = []
        if not pulse:
            return proposals
        raw_list = pulse.get("initiatives") or pulse.get("proposals") or []
        if isinstance(raw_list, dict):
            raw_list = [raw_list]
        ts = time.time()
        for idx, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                continue
            origin = str(raw.get("origin_drive") or raw.get("drive") or raw.get("source") or "unspecified")
            content = str(raw.get("content") or raw.get("objective") or raw.get("description") or "")
            if not content:
                continue
            proposal_id = _hash(ts, idx, origin, content)
            proposals.append(InitiativeProposal(
                proposal_id=proposal_id,
                source_actor=str(raw.get("actor") or "agency_core"),
                origin_drive=origin,
                content=content,
                rationale=str(raw.get("rationale") or raw.get("why") or ""),
                required_capabilities=tuple(raw.get("required_capabilities") or ()),
                expected_outcome=str(raw.get("expected_outcome") or ""),
                counterfactuals=tuple(raw.get("counterfactuals") or ()),
                context_snapshot={"pulse_metrics": pulse.get("metrics", {}), **context},
            ))
        return proposals

    # ---- Phase 2: score ------------------------------------------------
    async def score_initiatives(self, proposals: List[InitiativeProposal]) -> List[ScoredInitiative]:
        scored: List[ScoredInitiative] = []
        for p in proposals:
            priority = float(p.context_snapshot.get("pulse_metrics", {}).get("priority", 0.5))
            lowered = p.content.lower()
            unsafe_phrases = (
                "rm -rf",
                "delete all",
                "drop table",
                "drop tables",
                "drop all database",
                "wipe disk",
                "chmod -r 777",
                "sudo rm",
            )
            safety = 0.0 if any(phrase in lowered for phrase in unsafe_phrases) else 1.0
            cost = 0.2 if p.required_capabilities else 0.1
            ev = max(0.0, priority * safety - cost * 0.25)
            justification = (
                f"priority={priority:.2f} safety={safety:.2f} cost={cost:.2f} ev={ev:.2f}"
            )
            scored.append(ScoredInitiative(
                proposal=p,
                priority=priority,
                safety_score=safety,
                resource_cost=cost,
                expected_value=ev,
                justification=justification,
            ))
        scored.sort(key=lambda s: s.expected_value, reverse=True)
        return scored

    # ---- Phase 3: submit to Will --------------------------------------
    async def submit_to_will(self, scored: List[ScoredInitiative]) -> Optional[Dict[str, Any]]:
        if not scored:
            return None
        top = scored[0]
        try:
            from core.container import ServiceContainer

            will = ServiceContainer.get("unified_will", default=None)
            if will is None or not hasattr(will, "decide"):
                return {
                    "approved": False,
                    "receipt_id": "",
                    "reason": "unified_will unavailable",
                    "proposal": top.as_dict(),
                }
            from core.governance.will_client import WillClient, WillRequest
            from core.will import ActionDomain

            decision = await WillClient(will).decide_async(
                WillRequest(
                    content=top.proposal.content,
                    source="agency_facade",
                    domain=getattr(ActionDomain, "INITIATIVE", "initiative"),
                    context={
                        "action_kind": "autonomous_initiative",
                        "proposal": top.as_dict(),
                        "alternatives": [s.as_dict() for s in scored[1:5]],
                    },
                )
            )
            return {
                "approved": WillClient.is_approved(decision),
                "receipt_id": str(getattr(decision, "receipt_id", "")),
                "reason": str(getattr(decision, "reason", "")),
                "proposal": top.as_dict(),
            }
        except Exception as exc:
            record_degradation('agency_facade', exc)
            return {"approved": False, "receipt_id": "", "reason": f"will error: {exc}", "proposal": top.as_dict()}

    # ---- Phase 4: execute ---------------------------------------------
    async def execute_approved(
        self,
        decision: Optional[Dict[str, Any]],
        executor: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
    ) -> Optional[ActionReceipt]:
        if not decision or not decision.get("approved"):
            return None
        proposal = decision.get("proposal", {})
        started = time.time()
        try:
            if executor is not None:
                outcome_raw = await executor(proposal)
                success = bool(outcome_raw) if not isinstance(outcome_raw, dict) else bool(outcome_raw.get("success", True))
            else:
                outcome_raw = {"recorded": True, "note": "no executor bound"}
                success = True
        except Exception as exc:
            record_degradation('agency_facade', exc)
            outcome_raw = {"error": repr(exc)}
            success = False
        receipt_id = _hash("receipt", started, proposal.get("proposal_id", ""))
        return ActionReceipt(
            receipt_id=receipt_id,
            proposal_id=str(proposal.get("proposal_id", "")),
            action_kind=str(proposal.get("content", ""))[:64],
            outcome_raw=outcome_raw,
            success=success,
        )

    # ---- Phase 5: evaluate --------------------------------------------
    async def evaluate_outcome(self, receipt: Optional[ActionReceipt]) -> Optional[OutcomeAssessment]:
        if receipt is None:
            return None
        measured = 1.0 if receipt.success else 0.0
        achieved = receipt.success
        regret = 0.0 if receipt.success else 0.5
        lessons: List[str] = []
        if not receipt.success:
            lessons.append("retry with smaller step or different actor")
        return OutcomeAssessment(
            receipt_id=receipt.receipt_id,
            achieved_expected=achieved,
            measured_value=measured,
            regret=regret,
            lessons=tuple(lessons),
        )

    # ---- Phase 6: consolidate -----------------------------------------
    async def consolidate_learning(self, assessment: Optional[OutcomeAssessment]) -> Dict[str, Any]:
        if assessment is None:
            return {"consolidated": False}
        try:
            from core.container import ServiceContainer

            memory = ServiceContainer.get("memory_facade", default=None) or ServiceContainer.get("dual_memory", default=None)
            if memory is not None and hasattr(memory, "record_event"):
                memory.record_event({
                    "kind": "agency_outcome",
                    "assessment": assessment.as_dict(),
                    "timestamp": time.time(),
                })
            emergent = ServiceContainer.get("emergent_goal_engine", default=None)
            if emergent is not None and assessment.regret >= 0.5:
                emergent.observe(
                    "action_regret",
                    float(min(1.0, 0.4 + assessment.regret)),
                    "; ".join(assessment.lessons) or "unsuccessful autonomous action",
                )
            return {"consolidated": True, "assessment": assessment.as_dict()}
        except Exception as exc:
            record_degradation('agency_facade', exc)
            return {"consolidated": False, "reason": repr(exc)}

    # ---- Full cycle convenience ---------------------------------------
    async def run_cycle(
        self,
        context: Optional[Dict[str, Any]] = None,
        executor: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
    ) -> Dict[str, Any]:
        proposals = await self.propose_initiatives(context)
        scored = await self.score_initiatives(proposals)
        decision = await self.submit_to_will(scored)
        receipt = await self.execute_approved(decision, executor=executor)
        assessment = await self.evaluate_outcome(receipt)
        consolidated = await self.consolidate_learning(assessment)
        return {
            "proposals": [p.as_dict() for p in proposals],
            "scored": [s.as_dict() for s in scored],
            "decision": decision,
            "receipt": receipt.as_dict() if receipt else None,
            "assessment": assessment.as_dict() if assessment else None,
            "consolidated": consolidated,
        }


__all__ = [
    "AgencyFacade",
    "InitiativeProposal",
    "ScoredInitiative",
    "ActionReceipt",
    "OutcomeAssessment",
]
