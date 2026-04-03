"""Constitutional core for Aura's narrow-waist governance.

This module does not replace AuraState, the vault, or the existing executive
subsystems. It binds them into one auditable chain so that initiatives,
tool execution, belief mutation, state mutation, and continuity restoration
can all flow through one constitutional service.
"""
from __future__ import annotations

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

            exec_core = self._get_executive_core()
            if exec_core is None:
                if self._strict_enforcement_active():
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
                decision = self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.DEGRADED,
                        reason="executive_core_unavailable",
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

            intent, executive_record = await exec_core.prepare_tool_intent(tool_name, args, source=source)
            approved = executive_record.outcome.value in {"approved", "degraded"}

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
                        logger.debug("IntentionLoop begin skipped: %s", exc)

            outcome = {
                "approved": ProposalOutcome.APPROVED,
                "rejected": ProposalOutcome.REJECTED,
                "degraded": ProposalOutcome.DEGRADED,
                "deferred": ProposalOutcome.RECORDED,
            }.get(executive_record.outcome.value, ProposalOutcome.RECORDED)
            decision = self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=outcome,
                    reason=executive_record.reason,
                    source=source,
                    intent_id=intent.intent_id,
                    constraints=dict(executive_record.constraints or {}),
                    snapshot=self.snapshot(),
                )
            )
            handle = ToolExecutionHandle(
                proposal=proposal,
                decision=decision,
                approved=approved,
                constraints=dict(executive_record.constraints or {}),
                executive_intent_id=intent.intent_id,
                intention_id=intention_id,
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
                logger.debug("IntentionLoop completion skipped: %s", exc)

        exec_core = self._get_executive_core()
        if exec_core is not None and handle.executive_intent_id:
            try:
                exec_core.complete_intent(handle.executive_intent_id, success=success)
            except Exception as exc:
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
    ) -> tuple[bool, str]:
        proposal = ConstitutionalProposal(
            kind=ProposalKind.STATE_MUTATION,
            source=origin or "system",
            summary=f"state_mutation:{cause}",
            payload={"origin": origin, "cause": cause},
        )

        exec_core = self._get_executive_core()
        if exec_core is None:
            if self._strict_enforcement_active():
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
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="executive_core_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            return True, "executive_core_unavailable"

        approved, reason = await exec_core.approve_state_mutation(origin or "system", cause)
        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if approved else ProposalOutcome.REJECTED,
                reason=reason,
                source=proposal.source,
                snapshot=self.snapshot(state),
            )
        )
        return approved, reason

    async def approve_memory_write(
        self,
        memory_type: str,
        content: str,
        *,
        source: str = "unknown",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        state: Any = None,
    ) -> tuple[bool, str]:
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

        exec_core = self._get_executive_core()
        if exec_core is None:
            if self._strict_enforcement_active():
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
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="executive_core_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            return True, "executive_core_unavailable"

        approved, reason = await exec_core.approve_memory_write(
            memory_type,
            content,
            importance=max(0.0, min(1.0, float(importance or 0.0))),
            source=source or "unknown",
        )
        self._record_decision(
            ConstitutionalDecision(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                outcome=ProposalOutcome.APPROVED if approved else ProposalOutcome.REJECTED,
                reason=reason,
                source=proposal.source,
                snapshot=self.snapshot(state),
            )
        )
        return approved, reason

    def approve_memory_write_sync(
        self,
        memory_type: str,
        content: str,
        *,
        source: str = "unknown",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        state: Any = None,
    ) -> tuple[bool, str]:
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

        exec_core = self._get_executive_core()
        if exec_core is None:
            if self._strict_enforcement_active():
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
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.DEGRADED,
                    reason="executive_core_unavailable",
                    source=proposal.source,
                    snapshot=self.snapshot(state),
                )
            )
            return True, "executive_core_unavailable"

        try:
            from core.executive.executive_core import (
                ActionType,
                DecisionOutcome,
                Intent,
                _coerce_intent_source,
            )

            intent = Intent(
                source=_coerce_intent_source(source or "system"),
                goal=f"write_memory:{memory_type}",
                action_type=ActionType.WRITE_MEMORY,
                payload={
                    "type": memory_type,
                    "content": str(content or "")[:200],
                    "importance": max(0.0, min(1.0, float(importance or 0.0))),
                    "metadata": dict(metadata or {}),
                },
                priority=max(0.0, min(1.0, float(importance or 0.0))),
                requires_memory_commit=True,
            )
            record = exec_core.request_approval_sync(intent)
            approved = record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED)
            if approved:
                exec_core.complete_intent(intent.intent_id, success=True)
            self._record_decision(
                ConstitutionalDecision(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    outcome=ProposalOutcome.APPROVED if approved else ProposalOutcome.REJECTED,
                    reason=record.reason,
                    source=proposal.source,
                    intent_id=intent.intent_id,
                    constraints=dict(record.constraints or {}),
                    snapshot=self.snapshot(state),
                )
            )
            return approved, record.reason
        except Exception as exc:
            logger.debug("ConstitutionalCore sync memory approval failed: %s", exc)
            if self._strict_enforcement_active():
                self._record_decision(
                    ConstitutionalDecision(
                        proposal_id=proposal.proposal_id,
                        kind=proposal.kind,
                        outcome=ProposalOutcome.REJECTED,
                        reason=f"sync_memory_gate_failed:{type(exc).__name__}",
                        source=proposal.source,
                        snapshot=self.snapshot(state),
                    )
                )
                return False, f"sync_memory_gate_failed:{type(exc).__name__}"
            return True, f"sync_memory_gate_unavailable:{type(exc).__name__}"

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
            logger.debug("ExecutiveCore resolution failed: %s", exc)
            return None

    def _get_intention_loop(self) -> Any:
        try:
            from core.agency.intention_loop import get_intention_loop

            return ServiceContainer.get("intention_loop", default=None) or get_intention_loop()
        except Exception as exc:
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
            logger.debug("ConstitutionalCore registration skipped: %s", exc)
    else:
        _instance.bind(orchestrator)

    return _instance
