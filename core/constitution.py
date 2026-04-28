"""Constitutional core for Aura's narrow-waist governance.

This module does not replace AuraState, the vault, or the existing executive
subsystems. It binds them into one auditable chain so that initiatives,
tool execution, belief mutation, state mutation, and continuity restoration
can all flow through one constitutional service.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.ConstitutionalCore")


class ProposalKind(str, Enum):
    INITIATIVE = "initiative"
    EXPRESSION = "expression"
    TOOL = "tool"
    STATE_MUTATION = "state_mutation"
    MEMORY_MUTATION = "memory_mutation"
    BELIEF_MUTATION = "belief_mutation"
    CONTINUITY = "continuity"


class ProposalOutcome(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEGRADED = "degraded"
    RECORDED = "recorded"


@dataclass
class ConstitutionalProposal:
    kind: ProposalKind
    source: str
    summary: str
    payload: Dict[str, Any] = field(default_factory=dict)
    urgency: float = 0.5
    confidence: float = 0.5
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)


@dataclass
class ConstitutionalDecision:
    proposal_id: str
    kind: ProposalKind
    outcome: ProposalOutcome
    reason: str
    source: str
    will_receipt_id: Optional[str] = None
    target: str = ""
    intent_id: Optional[str] = None
    commitment_id: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionHandle:
    proposal: ConstitutionalProposal
    decision: ConstitutionalDecision
    approved: bool
    constraints: Dict[str, Any] = field(default_factory=dict)
    executive_intent_id: Optional[str] = None
    intention_id: Optional[str] = None
    capability_token_id: Optional[str] = None
    authority_receipt_id: Optional[str] = None
    will_receipt_id: Optional[str] = None


@dataclass
class BeliefMutationRecord:
    namespace: str
    key: str
    value: Any
    reason: str
    allowed: bool = True
    status: str = "tentative"
    confidence: float = 0.35
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def unpack_governance_result(result: Any) -> tuple[bool, str, Optional[Any]]:
    """Support legacy 2-tuples and new 3-tuples from governance APIs."""
    if isinstance(result, tuple):
        if len(result) >= 3:
            return bool(result[0]), str(result[1] or ""), result[2]
        if len(result) >= 2:
            return bool(result[0]), str(result[1] or ""), None
        if len(result) == 1:
            return bool(result[0]), "", None
    return bool(result), "", None


class BeliefAuthority:
    """Single epistemic entry point for durable belief writes."""

    def __init__(self) -> None:
        self._history: Deque[BeliefMutationRecord] = deque(maxlen=300)
        self._beliefs: Dict[str, BeliefMutationRecord] = {}

    def review_update(
        self,
        namespace: str,
        key: str,
        value: Any,
        note: Optional[str] = None,
        evidence: Optional[List[str]] = None,
    ) -> BeliefMutationRecord:
        normalized_key = str(key or "").strip().lower().replace(" ", "_")
        normalized_value = value
        reason = "accepted"
        evidence_refs = list(evidence or [])
        if note:
            evidence_refs.append(str(note))
        status = "tentative"
        confidence = 0.35
        contradictions: List[str] = []

        try:
            state_authority = ServiceContainer.get("state_authority", default=None)
        except Exception:
            state_authority = None

        if state_authority is not None and normalized_key:
            try:
                authoritative, tier = state_authority.get_truth(normalized_key)
                if authoritative is not None and getattr(tier, "name", "") in {"IMMUTABLE", "HARD_FACT"}:
                    normalized_value = authoritative
                    reason = f"resolved_by_state_authority:{tier.name.lower()}"
                    status = "trusted"
                    confidence = 0.98
            except Exception as exc:
                record_degradation('constitution', exc)
                logger.debug("BeliefAuthority state-authority lookup skipped: %s", exc)

        belief_id = f"{namespace}:{normalized_key or key}"
        existing = self._beliefs.get(belief_id)
        if existing is not None:
            if existing.value == normalized_value:
                confidence = min(0.98, float(existing.confidence or 0.35) + 0.12)
                status = "trusted" if confidence >= 0.75 else "active"
                reason = f"{reason}|reinforced"
                contradictions = list(existing.contradictions or [])
            else:
                contradictions = list(existing.contradictions or [])
                contradictions.append(str(normalized_value)[:180])
                if float(existing.confidence or 0.35) >= 0.75:
                    normalized_value = existing.value
                    confidence = float(existing.confidence or 0.8)
                    status = "trusted"
                    reason = "contradicted_trusted_belief"
                else:
                    confidence = max(0.3, float(existing.confidence or 0.35) - 0.05)
                    status = "contested"
                    reason = "contested_update"

        record = BeliefMutationRecord(
            namespace=str(namespace or "unknown"),
            key=normalized_key or str(key),
            value=normalized_value,
            reason=reason,
            status=status,
            confidence=round(confidence, 4),
            evidence=evidence_refs[:10],
            contradictions=contradictions[:10],
            allowed=reason != "contradicted_trusted_belief",
        )
        self._beliefs[belief_id] = record
        self._history.append(record)
        return record

    def recent(self, limit: int = 25) -> List[Dict[str, Any]]:
        items = list(self._history)[-limit:]
        return [item.to_dict() for item in items]

    def summary(self) -> Dict[str, Any]:
        records = list(self._beliefs.values())
        contested = [record for record in records if record.status == "contested"]
        trusted = [record for record in records if record.status == "trusted"]
        active = [record for record in records if record.status in {"active", "trusted"}]
        coherence = 1.0
        if records:
            coherence = max(0.0, min(1.0, 1.0 - (len(contested) / max(1, len(records)))))
        return {
            "total": len(records),
            "trusted": len(trusted),
            "active": len(active),
            "contested": len(contested),
            "coherence_score": round(coherence, 4),
        }


class ConstitutionalCore:
    """Narrow-waist constitutional governor over Aura's existing primitives."""

    def __init__(self, orchestrator: Any = None) -> None:
        self.orchestrator = orchestrator
        self.belief_authority = BeliefAuthority()
        self._decision_history: Deque[ConstitutionalDecision] = deque(maxlen=500)
        self._lock = asyncio.Lock()

    def bind(self, orchestrator: Any) -> None:
        if orchestrator is not None:
            self.orchestrator = orchestrator

    def snapshot(self, state: Any = None) -> Dict[str, Any]:
        current_state = state
        if current_state is None:
            repo = self._get_state_repository()
            current_state = getattr(repo, "_current", None) if repo is not None else None

        if current_state is None:
            return {
                "state_version": None,
                "policy_mode": "unknown",
                "current_objective": "",
                "pending_initiatives": 0,
                "active_goals": 0,
                "health": {},
            }

        cognition = getattr(current_state, "cognition", None)
        belief_summary = self.belief_authority.summary()
        thermal_guard = bool(getattr(current_state, "response_modifiers", {}).get("thermal_guard", False))
        coherence_score = float(getattr(cognition, "coherence_score", 1.0) or 1.0)
        fragmentation_score = float(getattr(cognition, "fragmentation_score", 0.0) or 0.0)
        contradiction_count = int(getattr(cognition, "contradiction_count", 0) or 0)
        health_flags: List[str] = []
        if thermal_guard:
            health_flags.append("thermal_guard")
        if coherence_score < 0.72:
            health_flags.append("coherence_low")
        if fragmentation_score > 0.4:
            health_flags.append("fragmentation_high")
        if contradiction_count > 0:
            health_flags.append("contradictions_present")
        if int(belief_summary.get("contested", 0) or 0) > 0:
            health_flags.append("beliefs_contested")

        return {
            "state_version": getattr(current_state, "version", None),
            "policy_mode": getattr(getattr(cognition, "current_mode", None), "value", str(getattr(cognition, "current_mode", "unknown"))),
            "current_objective": getattr(cognition, "current_objective", "") or "",
            "pending_initiatives": len(getattr(cognition, "pending_initiatives", []) or []),
            "active_goals": len(getattr(cognition, "active_goals", []) or []),
            "health": dict(getattr(current_state, "health", {}) or {}),
            "rolling_summary": getattr(cognition, "rolling_summary", "") or "",
            "coherence_score": coherence_score,
            "fragmentation_score": fragmentation_score,
            "contradiction_count": contradiction_count,
            "phenomenal_state": str(getattr(cognition, "phenomenal_state", "") or ""),
            "thermal_guard": thermal_guard,
            "health_flags": health_flags,
            "epistemics": belief_summary,
        }

    def _emit_tool_event(
        self,
        stage: str,
        tool_name: str,
        *,
        source: str,
        args: Optional[Dict[str, Any]] = None,
        decision: Optional[ConstitutionalDecision] = None,
        handle: Optional[ToolExecutionHandle] = None,
        result: Any = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        try:
            from core.event_bus import get_event_bus

            payload = {
                "type": "tool_event",
                "stage": stage,
                "tool": tool_name,
                "source": source,
                "args": dict(args or {}),
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
            if decision is not None:
                payload["decision"] = {
                    "proposal_id": decision.proposal_id,
                    "outcome": decision.outcome.value,
                    "reason": decision.reason,
                    "constraints": dict(decision.constraints or {}),
                }
            if handle is not None:
                payload["handle"] = {
                    "approved": handle.approved,
                    "executive_intent_id": handle.executive_intent_id,
                    "intention_id": handle.intention_id,
                }
            if result is not None:
                payload["result"] = result
            get_event_bus().publish_threadsafe("telemetry", payload)
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore tool event emission skipped: %s", exc)

    async def begin_tool_execution(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        source: str = "unknown",
        objective: str = "",
    ) -> ToolExecutionHandle:
        async with self._lock:
            self._emit_tool_event("requested", tool_name, source=source, args=args)
            proposal = ConstitutionalProposal(
                kind=ProposalKind.TOOL,
                source=source,
                summary=f"execute_tool:{tool_name}",
                payload={"tool_name": tool_name, "args": dict(args or {}), "objective": objective},
                urgency=0.9 if source in {"user", "voice", "api", "admin"} else 0.5,
            )
            if self._strict_enforcement_active() and self._get_executive_core() is None:
                decision = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason="executive_core_required",
                        source=source,
                        snapshot=self.snapshot(),
                    )
                )
                handle = ToolExecutionHandle(
                    proposal=proposal,
                    decision=decision,
                    approved=False,
                    constraints={"blocked": True},
                )
                self._emit_tool_event("rejected", tool_name, source=source, args=args, decision=decision, handle=handle)
                return handle
            gateway = self._get_authority_gateway()
            if gateway is None:
                if self._strict_enforcement_active():
                    decision = self._record_decision(
                        ConstitutionalDecision(
                            proposal_id=proposal.proposal_id,
                            kind=proposal.kind,
                            outcome=ProposalOutcome.REJECTED,
                            reason="authority_gateway_required",
                            source=source,
                            snapshot=self.snapshot(),
                        )
                    )
                    handle = ToolExecutionHandle(
                        proposal=proposal,
                        decision=decision,
                        approved=False,
                        constraints={"blocked": True},
                    )
                    self._emit_tool_event("rejected", tool_name, source=source, args=args, decision=decision, handle=handle)
                    return handle
                decision = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.DEGRADED,
                        reason="authority_gateway_unavailable",
                        source=source,
                        snapshot=self.snapshot(),
                    )
                )
                handle = ToolExecutionHandle(
                    proposal=proposal,
                    decision=decision,
                    approved=True,
                    constraints={},
                )
                self._emit_tool_event("degraded", tool_name, source=source, args=args, decision=decision, handle=handle)
                return handle

            authority_decision = await gateway.authorize_tool_execution(
                tool_name,
                dict(args or {}),
                source=source,
                priority=proposal.urgency,
            )
            approved = authority_decision.approved

            intention_id = None
            if approved:
                intention_loop = self._get_intention_loop()
                if intention_loop is not None:
                    try:
                        intention_id = intention_loop.intend(
                            intention=objective or f"Use tool '{tool_name}'",
                            drive=source or "system",
                            expected_outcome=f"Successful execution of {tool_name}",
                            plan=[f"Invoke {tool_name}", "Observe result", "Revise if needed"],
                        )
                    except Exception as exc:
                        record_degradation('constitution', exc)
                        logger.debug("IntentionLoop begin skipped: %s", exc)

            outcome = {
                "approved": ProposalOutcome.APPROVED,
                "rejected": ProposalOutcome.REJECTED,
                "degraded": ProposalOutcome.DEGRADED,
                "deferred": ProposalOutcome.RECORDED,
            }.get(authority_decision.outcome, ProposalOutcome.RECORDED)
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=outcome,
                    reason=authority_decision.reason,
                    source=source,
                    will_receipt_id=authority_decision.will_receipt_id,
                    intent_id=authority_decision.executive_intent_id,
                    constraints=self._authority_constraints(authority_decision),
                    snapshot=self.snapshot(),
                )
            )
            handle = ToolExecutionHandle(
                proposal=proposal,
                decision=decision,
                approved=approved,
                constraints=dict(authority_decision.constraints or {}),
                executive_intent_id=authority_decision.executive_intent_id,
                intention_id=intention_id,
                capability_token_id=authority_decision.capability_token_id,
                authority_receipt_id=authority_decision.substrate_receipt_id,
                will_receipt_id=authority_decision.will_receipt_id,
            )
            self._emit_tool_event(
                "approved" if approved else "rejected",
                tool_name,
                source=source,
                args=args,
                decision=decision,
                handle=handle,
            )
            if approved:
                self._emit_tool_event("started", tool_name, source=source, args=args, decision=decision, handle=handle)
            return handle

    async def finish_tool_execution(
        self,
        handle: ToolExecutionHandle,
        *,
        result: Any,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        if handle is None:
            return

        intention_loop = self._get_intention_loop()
        if intention_loop is not None and handle.intention_id:
            try:
                tool_name = handle.proposal.payload.get("tool_name", "unknown")
                args = dict(handle.proposal.payload.get("args", {}) or {})
                intention_loop.record_action(
                    handle.intention_id,
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    success=success,
                    duration_ms=duration_ms,
                )
                observation = "tool_succeeded" if success else "tool_failed"
                actual_outcome = error or str(result)
                intention_loop.observe(
                    handle.intention_id,
                    observation=observation,
                    actual_outcome=actual_outcome[:500],
                )
                intention_loop.revise(
                    handle.intention_id,
                    belief_updates=[],
                    self_model_updates=[],
                    success=success,
                )
            except Exception as exc:
                record_degradation('constitution', exc)
                logger.debug("IntentionLoop completion skipped: %s", exc)

        gateway = self._get_authority_gateway()
        if gateway is not None:
            gateway.finalize_tool_execution(
                executive_intent_id=handle.executive_intent_id,
                capability_token_id=handle.capability_token_id,
                success=success,
            )
        elif handle.executive_intent_id:
            exec_core = self._get_executive_core()
            if exec_core is not None:
                try:
                    exec_core.complete_intent(handle.executive_intent_id, success=success)
                except Exception as exc:
                    record_degradation('constitution', exc)
                    logger.debug("Executive intent completion skipped: %s", exc)
        tool_name = str(handle.proposal.payload.get("tool_name", "unknown") or "unknown")
        self._emit_tool_event(
            "completed" if success else "failed",
            tool_name,
            source=handle.proposal.source,
            args=dict(handle.proposal.payload.get("args", {}) or {}),
            decision=handle.decision,
            handle=handle,
            result=result,
            success=success,
            error=error,
            duration_ms=duration_ms,
        )

    async def approve_state_mutation(
        self,
        origin: str,
        cause: str,
        state: Any = None,
        return_decision: bool = False,
    ) -> tuple[bool, str] | tuple[bool, str, ConstitutionalDecision]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.STATE_MUTATION,
            source=origin or "system",
            summary=f"state_mutation:{cause}",
            payload={"origin": origin, "cause": cause},
        )

        if self._strict_enforcement_active() and self._get_executive_core() is None:
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED,
                    reason="executive_core_required",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return False, "executive_core_required", recorded
            return False, "executive_core_required"

        gateway = self._get_authority_gateway()
        if gateway is None:
            if self._strict_enforcement_active():
                recorded = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason="authority_gateway_required",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                if return_decision:
                    return False, "authority_gateway_required", recorded
                return False, "authority_gateway_required"
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return True, "authority_gateway_unavailable", recorded
            return True, "authority_gateway_unavailable"

        authority_decision = await gateway.authorize_state_mutation(origin or "system", cause)
        recorded = self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                will_receipt_id=authority_decision.will_receipt_id,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        if return_decision:
            return authority_decision.approved, authority_decision.reason, recorded
        return authority_decision.approved, authority_decision.reason

    async def approve_memory_write(
        self,
        memory_type: str,
        content: str,
        *,
        source: str = "unknown",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        state: Any = None,
        return_decision: bool = False,
    ) -> tuple[bool, str] | tuple[bool, str, ConstitutionalDecision]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.MEMORY_MUTATION,
            source=source or "system",
            summary=f"memory_write:{memory_type}",
            payload={
                "memory_type": memory_type,
                "content": str(content or "")[:240],
                "importance": float(importance or 0.0),
                "metadata": dict(metadata or {}),
            },
            urgency=max(0.1, min(1.0, float(importance or 0.0))),
        )

        if self._strict_enforcement_active() and self._get_executive_core() is None:
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED,
                    reason="executive_core_required",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return False, "executive_core_required", recorded
            return False, "executive_core_required"

        gateway = self._get_authority_gateway()
        if gateway is None:
            if self._strict_enforcement_active():
                recorded = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason="authority_gateway_required",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                if return_decision:
                    return False, "authority_gateway_required", recorded
                return False, "authority_gateway_required"
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return True, "authority_gateway_unavailable", recorded
            return True, "authority_gateway_unavailable"

        authority_decision = await gateway.authorize_memory_write(
            memory_type,
            content,
            importance=max(0.0, min(1.0, float(importance or 0.0))),
            source=source or "unknown",
            metadata=metadata,
        )
        recorded = self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                will_receipt_id=authority_decision.will_receipt_id,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        if return_decision:
            return authority_decision.approved, authority_decision.reason, recorded
        return authority_decision.approved, authority_decision.reason

    def approve_memory_write_sync(
        self,
        memory_type: str,
        content: str,
        *,
        source: str = "unknown",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        state: Any = None,
        return_decision: bool = False,
    ) -> tuple[bool, str] | tuple[bool, str, ConstitutionalDecision]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.MEMORY_MUTATION,
            source=source or "system",
            summary=f"memory_write:{memory_type}",
            payload={
                "memory_type": memory_type,
                "content": str(content or "")[:240],
                "importance": float(importance or 0.0),
                "metadata": dict(metadata or {}),
            },
            urgency=max(0.1, min(1.0, float(importance or 0.0))),
        )

        if self._strict_enforcement_active() and self._get_executive_core() is None:
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED,
                    reason="executive_core_required",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return False, "executive_core_required", recorded
            return False, "executive_core_required"

        gateway = self._get_authority_gateway()
        if gateway is None:
            if self._strict_enforcement_active():
                recorded = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason="authority_gateway_required",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                if return_decision:
                    return False, "authority_gateway_required", recorded
                return False, "authority_gateway_required"
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return True, "authority_gateway_unavailable", recorded
            return True, "authority_gateway_unavailable"

        try:
            authority_decision = gateway.authorize_memory_write_sync(
                memory_type,
                content,
                source=source or "unknown",
                importance=max(0.0, min(1.0, float(importance or 0.0))),
                metadata=metadata,
            )
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore sync memory approval failed: %s", exc)
            if self._strict_enforcement_active():
                recorded = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason=f"sync_memory_gate_failed:{type(exc).__name__}",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                if return_decision:
                    return False, f"sync_memory_gate_failed:{type(exc).__name__}", recorded
                return False, f"sync_memory_gate_failed:{type(exc).__name__}"
            if return_decision:
                return True, f"sync_memory_gate_unavailable:{type(exc).__name__}", ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason=f"sync_memory_gate_unavailable:{type(exc).__name__}",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            return True, f"sync_memory_gate_unavailable:{type(exc).__name__}"

        recorded = self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                will_receipt_id=authority_decision.will_receipt_id,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        if return_decision:
            return authority_decision.approved, authority_decision.reason, recorded
        return authority_decision.approved, authority_decision.reason

    def approve_belief_update_sync(
        self,
        key: str,
        value: Any,
        *,
        note: Optional[str] = None,
        source: str = "unknown",
        importance: float = 0.7,
        state: Any = None,
        return_decision: bool = False,
    ) -> tuple[bool, str] | tuple[bool, str, ConstitutionalDecision]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.BELIEF_MUTATION,
            source=source or "system",
            summary=f"belief_update:{str(key or '')[:120]}",
            payload={
                "key": str(key or "")[:120],
                "value": value,
                "note": note,
                "importance": float(importance or 0.0),
            },
            urgency=max(0.1, min(1.0, float(importance or 0.0))),
        )

        if self._strict_enforcement_active() and self._get_executive_core() is None:
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED,
                    reason="executive_core_required",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return False, "executive_core_required", recorded
            return False, "executive_core_required"

        gateway = self._get_authority_gateway()
        if gateway is None:
            if self._strict_enforcement_active():
                recorded = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason="authority_gateway_required",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                if return_decision:
                    return False, "authority_gateway_required", recorded
                return False, "authority_gateway_required"
            recorded = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            if return_decision:
                return True, "authority_gateway_unavailable", recorded
            return True, "authority_gateway_unavailable"

        try:
            authority_decision = gateway.authorize_belief_update_sync(
                key,
                value,
                note=note,
                source=source or "unknown",
                priority=max(0.0, min(1.0, float(importance or 0.0))),
            )
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore sync belief approval failed: %s", exc)
            if self._strict_enforcement_active():
                recorded = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason=f"sync_belief_gate_failed:{type(exc).__name__}",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                if return_decision:
                    return False, f"sync_belief_gate_failed:{type(exc).__name__}", recorded
                return False, f"sync_belief_gate_failed:{type(exc).__name__}"
            if return_decision:
                return True, f"sync_belief_gate_unavailable:{type(exc).__name__}", ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason=f"sync_belief_gate_unavailable:{type(exc).__name__}",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            return True, f"sync_belief_gate_unavailable:{type(exc).__name__}"

        recorded = self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                will_receipt_id=authority_decision.will_receipt_id,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        if return_decision:
            return authority_decision.approved, authority_decision.reason, recorded
        return authority_decision.approved, authority_decision.reason

    def approve_state_mutation_sync(
        self,
        origin: str,
        cause: str,
        *,
        urgency: float = 0.5,
        state: Any = None,
    ) -> tuple[bool, str]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.STATE_MUTATION,
            source=origin or "system",
            summary=f"state_mutation:{str(cause or '')[:120]}",
            payload={"origin": origin, "cause": str(cause or "")[:240]},
            urgency=max(0.1, min(1.0, float(urgency or 0.0))),
        )

        if self._strict_enforcement_active() and self._get_executive_core() is None:
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED,
                    reason="executive_core_required",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            return False, "executive_core_required"

        gateway = self._get_authority_gateway()
        if gateway is None:
            if self._strict_enforcement_active():
                self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason="authority_gateway_required",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                return False, "authority_gateway_required"
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            return True, "authority_gateway_unavailable"

        try:
            authority_decision = gateway.authorize_state_mutation_sync(
                origin or "system",
                cause,
                priority=max(0.0, min(1.0, float(urgency or 0.0))),
            )
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore sync state approval failed: %s", exc)
            if self._strict_enforcement_active():
                self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason=f"sync_state_gate_failed:{type(exc).__name__}",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                return False, f"sync_state_gate_failed:{type(exc).__name__}"
            return True, f"sync_state_gate_unavailable:{type(exc).__name__}"

        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason

    def record_external_decision(
        self,
        *,
        kind: ProposalKind,
        source: str,
        summary: str,
        outcome: str,
        reason: str,
        target: str = "",
        payload: Optional[Dict[str, Any]] = None,
        state: Any = None,
    ) -> ConstitutionalDecision:
        decision_outcome = {
            "approved": ProposalOutcome.APPROVED,
            "released": ProposalOutcome.APPROVED,
            "queued": ProposalOutcome.APPROVED,
            "recorded": ProposalOutcome.RECORDED,
            "suppressed": ProposalOutcome.REJECTED,
            "rejected": ProposalOutcome.REJECTED,
            "degraded": ProposalOutcome.DEGRADED,
        }.get(str(outcome).lower(), ProposalOutcome.RECORDED)
        proposal = ConstitutionalProposal(
            kind=kind,
            source=source,
            summary=summary,
            payload=dict(payload or {}),
        )
        return self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=kind,
                outcome=decision_outcome,
                reason=reason,
                source=source,
                target=target,
                snapshot=self.snapshot(state),
            )
        )

    async def approve_initiative(
        self,
        summary: str,
        *,
        source: str = "unknown",
        urgency: float = 0.5,
        state: Any = None,
    ):
        proposal = ConstitutionalProposal(
            kind=ProposalKind.INITIATIVE,
            source=source or "autonomous",
            summary=f"initiative:{str(summary or '')[:120]}",
            payload={"summary": str(summary or "")[:240]},
            urgency=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        gateway = self._get_authority_gateway()
        if gateway is None:
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED if self._strict_enforcement_active() else ProposalOutcome.DEGRADED,
                    reason="authority_gateway_required" if self._strict_enforcement_active() else "authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            approved = decision.outcome != ProposalOutcome.REJECTED
            return approved, decision.reason, None

        authority_decision = await gateway.authorize_initiative(
            str(summary or ""),
            source=source or "autonomous",
            priority=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason, authority_decision

    def approve_initiative_sync(
        self,
        summary: str,
        *,
        source: str = "unknown",
        urgency: float = 0.5,
        state: Any = None,
    ) -> tuple[bool, str]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.INITIATIVE,
            source=source or "autonomous",
            summary=f"initiative:{str(summary or '')[:120]}",
            payload={"summary": str(summary or "")[:240]},
            urgency=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        if self._strict_enforcement_active() and self._get_executive_core() is None:
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED,
                    reason="executive_core_required",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            return False, "executive_core_required"

        gateway = self._get_authority_gateway()
        if gateway is None:
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED if self._strict_enforcement_active() else ProposalOutcome.DEGRADED,
                    reason="authority_gateway_required" if self._strict_enforcement_active() else "authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            approved = decision.outcome != ProposalOutcome.REJECTED
            return approved, decision.reason

        try:
            authority_decision = gateway.authorize_initiative_sync(
                str(summary or ""),
                source=source or "autonomous",
                priority=max(0.0, min(1.0, float(urgency or 0.0))),
            )
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore sync initiative approval failed: %s", exc)
            if self._strict_enforcement_active():
                self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason=f"sync_initiative_gate_failed:{type(exc).__name__}",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                return False, f"sync_initiative_gate_failed:{type(exc).__name__}"
            return True, f"sync_initiative_gate_unavailable:{type(exc).__name__}"

        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason

    async def approve_expression(
        self,
        content: str,
        *,
        source: str = "unknown",
        urgency: float = 0.5,
        state: Any = None,
    ):
        proposal = ConstitutionalProposal(
            kind=ProposalKind.EXPRESSION,
            source=source or "autonomous",
            summary=f"expression:{str(content or '')[:120]}",
            payload={"content": str(content or "")[:240]},
            urgency=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        gateway = self._get_authority_gateway()
        if gateway is None:
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED if self._strict_enforcement_active() else ProposalOutcome.DEGRADED,
                    reason="authority_gateway_required" if self._strict_enforcement_active() else "authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            approved = decision.outcome != ProposalOutcome.REJECTED
            return approved, decision.reason, None

        authority_decision = await gateway.authorize_expression(
            str(content or ""),
            source=source or "autonomous",
            urgency=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason, authority_decision

    def approve_expression_sync(
        self,
        content: str,
        *,
        source: str = "unknown",
        urgency: float = 0.5,
        state: Any = None,
    ) -> tuple[bool, str]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.EXPRESSION,
            source=source or "autonomous",
            summary=f"expression:{str(content or '')[:120]}",
            payload={"content": str(content or "")[:240]},
            urgency=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        gateway = self._get_authority_gateway()
        if gateway is None:
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED if self._strict_enforcement_active() else ProposalOutcome.DEGRADED,
                    reason="authority_gateway_required" if self._strict_enforcement_active() else "authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            approved = decision.outcome != ProposalOutcome.REJECTED
            return approved, decision.reason

        try:
            authority_decision = gateway.authorize_expression_sync(
                str(content or ""),
                source=source or "autonomous",
                urgency=max(0.0, min(1.0, float(urgency or 0.0))),
            )
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore sync expression approval failed: %s", exc)
            if self._strict_enforcement_active():
                self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason=f"sync_expression_gate_failed:{type(exc).__name__}",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                return False, f"sync_expression_gate_failed:{type(exc).__name__}"
            return True, f"sync_expression_gate_unavailable:{type(exc).__name__}"

        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason

    async def approve_response(
        self,
        content: str,
        *,
        source: str = "user",
        urgency: float = 0.4,
        state: Any = None,
    ):
        proposal = ConstitutionalProposal(
            kind=ProposalKind.EXPRESSION,
            source=source or "user",
            summary=f"response:{str(content or '')[:120]}",
            payload={"content": str(content or "")[:240]},
            urgency=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        gateway = self._get_authority_gateway()
        if gateway is None:
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.REJECTED if self._strict_enforcement_active() else ProposalOutcome.DEGRADED,
                    reason="authority_gateway_required" if self._strict_enforcement_active() else "authority_gateway_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            approved = decision.outcome != ProposalOutcome.REJECTED
            return approved, decision.reason, None

        authority_decision = await gateway.authorize_response(
            str(content or ""),
            source=source or "user",
            priority=max(0.0, min(1.0, float(urgency or 0.0))),
        )
        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if authority_decision.approved else ProposalOutcome.REJECTED,
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason, authority_decision

    def get_status(self) -> Dict[str, Any]:
        failure_state = {"pressure": 0.0, "count": 0, "critical": 0, "errors": 0, "warnings": 0, "top_subsystems": []}
        temporal_state: Dict[str, Any] = {}
        identity_integrity = True
        try:
            exec_core = self._get_executive_core()
            if exec_core is not None:
                failure_state = exec_core._get_failure_state()
                temporal_state = exec_core._get_temporal_identity_context()
                identity_integrity = bool(exec_core._identity_integrity_available())
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore status enrichment skipped: %s", exc)
        return {
            "recent_decisions": [decision.to_dict() for decision in list(self._decision_history)[-20:]],
            "belief_updates": self.belief_authority.recent(20),
            "belief_summary": self.belief_authority.summary(),
            "failure_state": failure_state,
            "temporal_identity": temporal_state,
            "identity_integrity": identity_integrity,
        }

    def _record_decision(self, decision: ConstitutionalDecision) -> ConstitutionalDecision:
        self._decision_history.append(decision)
        return decision

    def _authority_constraints(self, authority_decision: Any) -> Dict[str, Any]:
        constraints = dict(getattr(authority_decision, "constraints", {}) or {})
        receipt_id = getattr(authority_decision, "substrate_receipt_id", None)
        will_receipt_id = getattr(authority_decision, "will_receipt_id", None)
        governance_domain = getattr(authority_decision, "domain", None)
        capability_token_id = getattr(authority_decision, "capability_token_id", None)
        failure_pressure = getattr(authority_decision, "failure_pressure", None)
        canonical_self_version = getattr(authority_decision, "canonical_self_version", None)
        if receipt_id:
            constraints["substrate_receipt_id"] = receipt_id
        if will_receipt_id:
            constraints["will_receipt_id"] = will_receipt_id
        if governance_domain:
            constraints["governance_domain"] = governance_domain
        if capability_token_id:
            constraints["capability_token_id"] = capability_token_id
        if failure_pressure is not None:
            constraints["failure_pressure"] = float(failure_pressure)
        if canonical_self_version is not None:
            constraints["canonical_self_version"] = int(canonical_self_version)
        return constraints

    def _get_state_repository(self) -> Any:
        orch = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
        return getattr(orch, "state_repo", None) or ServiceContainer.get("state_repository", default=None)

    def _strict_enforcement_active(self) -> bool:
        try:
            return (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
        except Exception:
            return False

    def _get_executive_core(self) -> Any:
        try:
            from core.executive.executive_core import get_executive_core

            return get_executive_core()
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ExecutiveCore resolution failed: %s", exc)
            return None

    def _get_authority_gateway(self) -> Any:
        try:
            from core.executive.authority_gateway import get_authority_gateway

            return get_authority_gateway()
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("AuthorityGateway resolution failed: %s", exc)
            return None

    def _get_intention_loop(self) -> Any:
        try:
            from core.agency.intention_loop import get_intention_loop

            return ServiceContainer.get("intention_loop", default=None) or get_intention_loop()
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("IntentionLoop resolution failed: %s", exc)
            return None


_instance: Optional[ConstitutionalCore] = None


def get_constitutional_core(orchestrator: Any = None) -> ConstitutionalCore:
    global _instance

    existing = ServiceContainer.get("constitutional_core", default=None)
    if isinstance(existing, ConstitutionalCore):
        existing.bind(orchestrator)
        return existing

    if _instance is None:
        _instance = ConstitutionalCore(orchestrator=orchestrator)
        try:
            ServiceContainer.register_instance("constitutional_core", _instance, required=False)
            ServiceContainer.register_instance("belief_authority", _instance.belief_authority, required=False)
        except Exception as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore registration skipped: %s", exc)
    else:
        _instance.bind(orchestrator)

    return _instance
