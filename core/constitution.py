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
from enum import StrEnum
from typing import Any

from core.container import ServiceContainer
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import record_degradation
from core.security.structural_redaction import (
    redact_mapping,
    redact_structure,
    redaction_marker,
)

logger = logging.getLogger("Aura.ConstitutionalCore")


class ProposalKind(StrEnum):
    INITIATIVE = "initiative"
    EXPRESSION = "expression"
    TOOL = "tool"
    STATE_MUTATION = "state_mutation"
    MEMORY_MUTATION = "memory_mutation"
    BELIEF_MUTATION = "belief_mutation"
    CONTINUITY = "continuity"


class ProposalOutcome(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEGRADED = "degraded"
    DEFERRED = "deferred"
    RECORDED = "recorded"


@dataclass
class ConstitutionalProposal:
    kind: ProposalKind
    source: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
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
    will_receipt_id: str | None = None
    target: str = ""
    intent_id: str | None = None
    commitment_id: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, Any] = field(default_factory=dict)
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionHandle:
    proposal: ConstitutionalProposal
    decision: ConstitutionalDecision
    approved: bool
    constraints: dict[str, Any] = field(default_factory=dict)
    executive_intent_id: str | None = None
    intention_id: str | None = None
    capability_token_id: str | None = None
    authority_receipt_id: str | None = None
    will_receipt_id: str | None = None
    standing_authority_token: str | None = None
    # The signed grant the sink authenticates. Unlike capability_token_id this
    # is unforgeable without the Will's private key.
    signed_capability: dict[str, Any] | None = None


#: How many DISTINCT evidence references a belief needs before repetition
#: can carry it to "trusted". One source repeating itself is one source.
BELIEF_MIN_SOURCES_FOR_TRUST = 2

#: Ceiling for a belief with no independent corroboration. Deliberately
#: below the 0.75 trusted threshold: an uncorroborated claim may become
#: "active" — worth acting on provisionally — but never trusted.
BELIEF_UNCORROBORATED_CEILING = 0.70


@dataclass
class BeliefMutationRecord:
    namespace: str
    key: str
    value: Any
    reason: str
    allowed: bool = True
    status: str = "tentative"
    confidence: float = 0.35
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unpack_governance_result(result: Any) -> tuple[bool, str, Any | None]:
    """Support legacy 2-tuples and new 3-tuples from governance APIs."""
    if isinstance(result, tuple):
        if len(result) >= 3:
            return bool(result[0]), str(result[1] or ""), result[2]
        if len(result) >= 2:
            return bool(result[0]), str(result[1] or ""), None
        if len(result) == 1:
            return bool(result[0]), "", None
    return bool(result), "", None


def _tool_result_is_deferred(result: Any) -> bool:
    if isinstance(result, dict):
        status = str(result.get("status", "") or "").strip().lower()
        error = str(result.get("error", "") or "").strip().lower()
        reason = str(result.get("reason", "") or "").strip().lower()
        return status == "deferred" or any(
            value.startswith("background_deferred:") for value in (error, reason)
        )
    return "background_deferred:" in str(result or "").lower()


# How long after start the constitution may run with degraded approval
# semantics because its enforcement services have not registered yet. Boot
# would deadlock against its own constitution without this; a window that
# outlives boot is a defect, so it is bounded and reported.
_BOOTSTRAP_LENIENCY_WINDOW_S = 180.0


class BeliefAuthority:
    """Single epistemic entry point for durable belief writes."""

    def __init__(self) -> None:
        self._history: deque[BeliefMutationRecord] = deque(maxlen=300)
        self._beliefs: dict[str, BeliefMutationRecord] = {}

    def review_update(
        self,
        namespace: str,
        key: str,
        value: Any,
        note: str | None = None,
        evidence: list[str] | None = None,
    ) -> BeliefMutationRecord:
        normalized_key = str(key or "").strip().lower().replace(" ", "_")
        normalized_value = value
        reason = "accepted"
        evidence_refs = list(evidence or [])
        if note:
            evidence_refs.append(str(note))
        status = "tentative"
        confidence = 0.35
        contradictions: list[str] = []

        try:
            state_authority = ServiceContainer.get("state_authority", default=None)
        except (ImportError, AttributeError, RuntimeError):
            state_authority = None

        if state_authority is not None and normalized_key:
            try:
                from core.state.state_authority import TruthTier

                authoritative, tier = state_authority.get_truth(
                    normalized_key, max_tier=TruthTier.HARD_FACT
                )
                if authoritative is not None and getattr(tier, "name", "") in {"IMMUTABLE", "HARD_FACT"}:
                    normalized_value = authoritative
                    reason = f"resolved_by_state_authority:{tier.name.lower()}"
                    status = "trusted"
                    confidence = 0.98
            except (RuntimeError, AttributeError, TypeError) as exc:
                record_degradation('constitution', exc)
                logger.debug("BeliefAuthority state-authority lookup skipped: %s", exc)

        belief_id = f"{namespace}:{normalized_key or key}"
        existing = self._beliefs.get(belief_id)
        if existing is not None:
            if existing.value == normalized_value:
                # CP126 (critical): "Repeated unverified claims self-promote
                # to trusted belief. Matching an existing value increments
                # confidence by a fixed amount until trusted status, without
                # requiring independent evidence, source diversity, or a
                # verifier receipt."
                #
                # Four repetitions of the same unsupported assertion walked
                # 0.35 -> 0.47 -> 0.59 -> 0.71 -> 0.83 and crossed the 0.75
                # trusted line. Saying a thing again is not evidence for it,
                # and a belief system in which it is will believe whatever it
                # is told most often.
                #
                # Two changes. Reinforcement now requires evidence this
                # belief has not already been credited with — repeating a
                # claim with the same citation, or none, holds confidence
                # where it is. And repetition alone cannot reach "trusted":
                # that requires either the state authority (handled above, at
                # 0.98) or independent corroboration, so an unsupported
                # belief tops out at "active" no matter how often it recurs.
                prior_evidence = set(existing.evidence or [])
                fresh_evidence = [ref for ref in evidence_refs if ref not in prior_evidence]
                distinct_sources = len(prior_evidence | set(evidence_refs))
                contradictions = list(existing.contradictions or [])

                if fresh_evidence:
                    confidence = min(0.98, float(existing.confidence or 0.35) + 0.12)
                    reason = f"{reason}|reinforced_by_new_evidence"
                else:
                    confidence = float(existing.confidence or 0.35)
                    reason = f"{reason}|repeated_without_new_evidence"

                if distinct_sources >= BELIEF_MIN_SOURCES_FOR_TRUST and confidence >= 0.75:
                    status = "trusted"
                else:
                    # Hold below the trusted line so repetition cannot cross
                    # it on its own.
                    confidence = min(confidence, BELIEF_UNCORROBORATED_CEILING)
                    status = "active"
                evidence_refs = list(prior_evidence | set(evidence_refs))
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

    # A contest that nothing re-asserts must not gate autonomy forever —
    # observed live: contested count only ever grew (1→2) and every
    # autonomous write deferred for the whole session because no
    # resolution API existed at all.
    CONTEST_FRESHNESS_S = 6 * 3600.0

    def reconcile(
        self,
        belief_id: str,
        *,
        resolution: str = "affirmed",
        evidence: str = "",
    ) -> bool:
        """Resolve a contested belief (research completed / adjudicated).

        resolution ∈ {affirmed, retired}: affirmed restores active status
        with the new evidence attached; retired removes the claim.
        """
        record = self._beliefs.get(belief_id)
        if record is None or record.status != "contested":
            return False
        if resolution == "retired":
            del self._beliefs[belief_id]
        else:
            record.status = "active"
            record.reason = "reconciled"
            if evidence:
                record.evidence = (record.evidence + [str(evidence)[:200]])[:10]
            record.recorded_at = time.time()
        return True

    def contested_records(self, *, fresh_only: bool = False) -> list[dict[str, Any]]:
        now = time.time()
        out = []
        for belief_id, record in self._beliefs.items():
            if record.status != "contested":
                continue
            age = now - float(record.recorded_at or now)
            if fresh_only and age > self.CONTEST_FRESHNESS_S:
                continue
            entry = record.to_dict()
            entry["belief_id"] = belief_id
            entry["age_s"] = round(age, 1)
            out.append(entry)
        return out

    def fresh_contested_count(self) -> int:
        """Contested records young enough to gate autonomy (age-out applies)."""
        return len(self.contested_records(fresh_only=True))

    def recent(self, limit: int = 25) -> list[dict[str, Any]]:
        items = list(self._history)[-limit:]
        return [item.to_dict() for item in items]

    def summary(self) -> dict[str, Any]:
        records = list(self._beliefs.values())
        contested = [record for record in records if record.status == "contested"]
        now = time.time()
        fresh_contested = [
            record for record in contested
            if (now - float(record.recorded_at or now)) <= self.CONTEST_FRESHNESS_S
        ]
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
            "fresh_contested": len(fresh_contested),
            # WHICH beliefs are contested, not just how many. A count can only
            # gate globally: two contested claims about anything blocked every
            # autonomous knowledge write about everything for a whole live hour
            # (2026-07-25: 71 blocked writes at contested=2). Relevance needs
            # the keys.
            "fresh_contested_keys": sorted(
                {
                    f"{record.namespace}:{record.key}"
                    for record in fresh_contested
                    if record.key
                }
            )[:32],
            "coherence_score": round(coherence, 4),
        }


class ConstitutionalCore:
    """Narrow-waist constitutional governor over Aura's existing primitives."""

    def __init__(self, orchestrator: Any = None) -> None:
        self.orchestrator = orchestrator
        # Anchors the bootstrap leniency window. A degraded constitution is
        # acceptable while the runtime comes up and never after.
        self._constitution_started_at = time.time()
        self._bootstrap_window_expired_reported = False
        self.belief_authority = BeliefAuthority()
        self._decision_history: deque[ConstitutionalDecision] = deque(
            maxlen=working_history_retention_policy("AURA_CONSTITUTION_DECISION_HISTORY_MAX").max_items
        )
        self._lock = asyncio.Lock()

    def bind(self, orchestrator: Any) -> None:
        if orchestrator is not None:
            self.orchestrator = orchestrator

    def snapshot(self, state: Any = None) -> dict[str, Any]:
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
        health_flags: list[str] = []
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
        args: dict[str, Any] | None = None,
        decision: ConstitutionalDecision | None = None,
        handle: ToolExecutionHandle | None = None,
        result: Any = None,
        success: bool | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        try:
            from core.event_bus import get_event_bus

            # CP126 (critical): "Tool telemetry publishes full arguments and
            # results. The event payload copies raw args and arbitrary
            # results into the telemetry bus. Credentials, message bodies,
            # file content, personal data, and large outputs have no
            # structural redaction or size limit."
            #
            # Every tool call in the system passes through here, so this was
            # the single widest exposure surface: a shell command's argv, a
            # web request's headers, a file write's content, an email body —
            # published verbatim to a bus with an unbounded payload.
            safe_args, args_report = redact_mapping(dict(args or {}))
            payload = {
                "type": "tool_event",
                "stage": stage,
                "tool": tool_name,
                "source": source,
                "args": safe_args,
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
            args_marker = redaction_marker(args_report)
            if args_marker:
                payload["args_redaction"] = args_marker
            if decision is not None:
                payload["decision"] = {
                    "proposal_id": decision.proposal_id,
                    "outcome": decision.outcome.value,
                    "reason": decision.reason,
                    # Constraints can carry caller-supplied values too.
                    "constraints": redact_mapping(dict(decision.constraints or {}))[0],
                }
            if handle is not None:
                payload["handle"] = {
                    "approved": handle.approved,
                    "executive_intent_id": handle.executive_intent_id,
                    "intention_id": handle.intention_id,
                }
            if result is not None:
                # Results are the larger half: a tool can return a whole
                # file, a page of HTML, or a directory listing.
                safe_result, result_report = redact_structure(result)
                payload["result"] = safe_result
                result_marker = redaction_marker(result_report)
                if result_marker:
                    payload["result_redaction"] = result_marker
            get_event_bus().publish_threadsafe("telemetry", payload)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore tool event emission skipped: %s", exc)

    async def begin_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        source: str = "unknown",
        objective: str = "",
        context: dict[str, Any] | None = None,
    ) -> ToolExecutionHandle:
        async with self._lock:
            from core.executive.execution_policy import (
                canonical_authority_arguments,
                canonical_authority_context,
            )

            authority_args = canonical_authority_arguments(tool_name, args)
            authority_context = canonical_authority_context(tool_name, context)
            authority_objective = (
                "Use Aura's private Messages channel"
                if str(tool_name or "").strip().lower() == "messages"
                else objective
            )
            self._emit_tool_event(
                "requested",
                tool_name,
                source=source,
                args=authority_args,
            )
            proposal = ConstitutionalProposal(
                kind=ProposalKind.TOOL,
                source=source,
                summary=f"execute_tool:{tool_name}",
                payload={
                    "tool_name": tool_name,
                    "args": authority_args,
                    "objective": authority_objective,
                    "context": authority_context,
                },
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
                self._emit_tool_event("rejected", tool_name, source=source, args=authority_args, decision=decision, handle=handle)
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
                    self._emit_tool_event("rejected", tool_name, source=source, args=authority_args, decision=decision, handle=handle)
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
                self._emit_tool_event("degraded", tool_name, source=source, args=authority_args, decision=decision, handle=handle)
                return handle

            authority_decision = await gateway.authorize_tool_execution(
                tool_name,
                authority_args,
                source=source,
                priority=proposal.urgency,
                context=authority_context,
            )
            approved = authority_decision.approved

            intention_id = None
            if approved:
                intention_loop = self._get_intention_loop()
                if intention_loop is not None:
                    try:
                        intention_id = intention_loop.intend(
                            intention=authority_objective or f"Use tool '{tool_name}'",
                            drive=source or "system",
                            expected_outcome=f"Successful execution of {tool_name}",
                            plan=[f"Invoke {tool_name}", "Observe result", "Revise if needed"],
                        )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation('constitution', exc)
                        logger.debug("IntentionLoop begin skipped: %s", exc)

            outcome = {
                "approved": ProposalOutcome.APPROVED,
                "rejected": ProposalOutcome.REJECTED,
                "degraded": ProposalOutcome.DEGRADED,
                "deferred": ProposalOutcome.DEFERRED,
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
                standing_authority_token=authority_decision.standing_authority_token,
                signed_capability=getattr(authority_decision, "signed_capability", None),
            )
            self._emit_tool_event(
                "approved" if approved else "rejected",
                tool_name,
                source=source,
                args=authority_args,
                decision=decision,
                handle=handle,
            )
            if approved:
                self._emit_tool_event("started", tool_name, source=source, args=authority_args, decision=decision, handle=handle)
            return handle

    async def finish_tool_execution(
        self,
        handle: ToolExecutionHandle,
        *,
        result: Any,
        success: bool,
        duration_ms: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        if handle is None:
            # CP126 (critical): "Missing execution handle is reported as
            # fully closed. finish_tool_execution(None) asserts that intent
            # closure and token revocation succeeded even though no handle
            # exists from which either fact can be established."
            #
            # With no handle there is genuinely nothing outstanding, so
            # `closed` stays True — a caller asking "is anything left to
            # reconcile?" gets the right answer. But `intent_closed` and
            # `token_revoked` asserted that two specific ACTIONS succeeded,
            # and neither was attempted. They are now None: not applicable,
            # rather than done.
            #
            # This matters downstream. capability_engine invalidates an
            # effectful result when a closure receipt reports failure, and
            # the authority gateway queues unreconciled grants — both read
            # these fields. A receipt that claims a revocation nobody
            # performed is exactly the kind of evidence those checks exist
            # to catch.
            return {
                "closed": True,
                "mode": "no_handle",
                "success": bool(success),
                "intent_closed": None,
                "token_revoked": None,
                "handle_present": False,
                "errors": [],
            }

        deferred_result = _tool_result_is_deferred(result)
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
                observation = "tool_deferred" if deferred_result else ("tool_succeeded" if success else "tool_failed")
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
                    status="deferred" if deferred_result else None,
                )
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation('constitution', exc)
                logger.warning(
                    "IntentionLoop completion recording failed; authority closure will continue: %s",
                    exc,
                )

        gateway = self._get_authority_gateway()
        standing_fallback = {
            "closed": not handle.standing_authority_token,
            "errors": [],
        }
        if gateway is None and handle.standing_authority_token:
            try:
                from core.executive.standing_authority import (
                    get_standing_authority_manager,
                )

                standing_fallback = get_standing_authority_manager().finalize_child_lease(
                    handle.standing_authority_token,
                    success=success,
                    result=result,
                    error=error or "",
                )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('constitution', exc)
                standing_fallback = {
                    "closed": False,
                    "errors": [f"standing_authority:{type(exc).__name__}:{exc}"],
                }
        if gateway is not None:
            closure = gateway.finalize_tool_execution(
                executive_intent_id=handle.executive_intent_id,
                capability_token_id=handle.capability_token_id,
                standing_authority_token=handle.standing_authority_token,
                success=success,
                result=result,
                error=error or "",
            )
        elif handle.executive_intent_id:
            errors: list[str] = []
            intent_closed = False
            exec_core = self._get_executive_core()
            if exec_core is not None:
                try:
                    exec_core.complete_intent(handle.executive_intent_id, success=success)
                    intent_closed = True
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('constitution', exc)
                    logger.debug("Executive intent completion skipped: %s", exc)
                    errors.append(f"executive_intent:{type(exc).__name__}:{exc}")
            else:
                errors.append("executive_core_unavailable")
            closure = {
                "closed": (
                    intent_closed
                    and not handle.capability_token_id
                    and bool(standing_fallback.get("closed"))
                ),
                "mode": "executive_core_fallback",
                "success": bool(success),
                "intent_closed": intent_closed,
                "token_revoked": not handle.capability_token_id,
                "standing_authority_closed": bool(standing_fallback.get("closed")),
                "errors": errors + list(standing_fallback.get("errors") or []),
            }
        else:
            closure = {
                "closed": not handle.capability_token_id and bool(standing_fallback.get("closed")),
                "mode": "no_gateway",
                "success": bool(success),
                "intent_closed": True,
                "token_revoked": not handle.capability_token_id,
                "standing_authority_closed": bool(standing_fallback.get("closed")),
                "errors": (
                    ([] if not handle.capability_token_id else ["authority_gateway_unavailable"])
                    + list(standing_fallback.get("errors") or [])
                ),
            }
        tool_name = str(handle.proposal.payload.get("tool_name", "unknown") or "unknown")
        self._emit_tool_event(
            "deferred" if deferred_result else ("completed" if success else "failed"),
            tool_name,
            source=handle.proposal.source,
            args=dict(handle.proposal.payload.get("args", {}) or {}),
            decision=handle.decision,
            handle=handle,
            result=result,
            success=None if deferred_result else success,
            error=None if deferred_result else error,
            duration_ms=duration_ms,
        )
        return closure

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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        metadata: dict[str, Any] | None = None,
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        metadata: dict[str, Any] | None = None,
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        note: str | None = None,
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        payload: dict[str, Any] | None = None,
        state: Any = None,
    ) -> ConstitutionalDecision:
        # CP126 (critical): "External callers can record approved
        # constitutional decisions. record_external_decision maps
        # caller-provided outcome strings to APPROVED and records the
        # caller-provided source without signed authority, verifier identity,
        # or proof that any adjudication occurred."
        #
        # This method exists to LOG a decision made elsewhere. Mapping
        # "approved"/"released"/"queued" onto ProposalOutcome.APPROVED let a
        # caller mint a verdict this core never reached, indistinguishable in
        # the history from one it did — an approval by assertion.
        #
        # An externally reported outcome is now RECORDED (which is precisely
        # what happened: something was recorded) with the caller's claim
        # preserved verbatim. Nothing is lost from the audit trail, and
        # nothing in it claims this core approved anything it did not.
        # REJECTED and DEGRADED are kept as-is: a caller reporting a refusal
        # or a degradation against itself is not claiming authority.
        claimed = str(outcome).lower()
        decision_outcome = {
            "recorded": ProposalOutcome.RECORDED,
            "suppressed": ProposalOutcome.REJECTED,
            "rejected": ProposalOutcome.REJECTED,
            "degraded": ProposalOutcome.DEGRADED,
            "deferred": ProposalOutcome.DEFERRED,
        }.get(claimed, ProposalOutcome.RECORDED)
        externally_claimed_approval = claimed in {"approved", "released", "queued"}
        if externally_claimed_approval:
            reason = f"{reason}|externally_claimed:{claimed}(not_adjudicated_here)"
        proposal = ConstitutionalProposal(
            kind=kind,
            source=source,
            summary=summary,
            payload={
                **dict(payload or {}),
                # Provenance, so a reader can tell a reported decision from
                # an adjudicated one without parsing the reason string.
                "adjudicated_by": "external_caller",
                "externally_claimed_outcome": claimed,
            },
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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
                outcome=self._authority_proposal_outcome(authority_decision),
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
                outcome=self._authority_proposal_outcome(authority_decision),
                reason=authority_decision.reason,
                source=proposal.source,
                intent_id=authority_decision.executive_intent_id,
                constraints=self._authority_constraints(authority_decision),
                snapshot=self.snapshot(state),
            )
        )
        return authority_decision.approved, authority_decision.reason, authority_decision

    def get_status(self) -> dict[str, Any]:
        failure_state = {"pressure": 0.0, "count": 0, "critical": 0, "errors": 0, "warnings": 0, "top_subsystems": []}
        temporal_state: dict[str, Any] = {}
        identity_integrity = True
        try:
            exec_core = self._get_executive_core()
            if exec_core is not None:
                failure_state = exec_core._get_failure_state()
                temporal_state = exec_core._get_temporal_identity_context()
                identity_integrity = bool(exec_core._identity_integrity_available())
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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

    @staticmethod
    def _authority_proposal_outcome(authority_decision: Any) -> ProposalOutcome:
        """Preserve an authority result without promoting inconsistent approval."""

        raw_outcome = getattr(authority_decision, "outcome", "")
        normalized = str(getattr(raw_outcome, "value", raw_outcome) or "").strip().lower()
        approved = bool(getattr(authority_decision, "approved", False))
        if normalized == ProposalOutcome.DEFERRED.value:
            return ProposalOutcome.DEFERRED
        if normalized == ProposalOutcome.DEGRADED.value and approved:
            return ProposalOutcome.DEGRADED
        if approved:
            return ProposalOutcome.APPROVED
        return ProposalOutcome.REJECTED

    def _authority_constraints(self, authority_decision: Any) -> dict[str, Any]:
        constraints = dict(getattr(authority_decision, "constraints", {}) or {})
        raw_outcome = getattr(authority_decision, "outcome", None)
        authority_outcome = str(getattr(raw_outcome, "value", raw_outcome) or "").strip().lower()
        receipt_id = getattr(authority_decision, "substrate_receipt_id", None)
        will_receipt_id = getattr(authority_decision, "will_receipt_id", None)
        governance_domain = getattr(authority_decision, "domain", None)
        capability_token_id = getattr(authority_decision, "capability_token_id", None)
        failure_pressure = getattr(authority_decision, "failure_pressure", None)
        canonical_self_version = getattr(authority_decision, "canonical_self_version", None)
        if authority_outcome:
            constraints["authority_outcome"] = authority_outcome
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
        """Whether constitutional decisions must be strictly enforced.

        True means a missing executive core or authority gateway REJECTS the
        proposal; False means degraded approval semantics apply.

        CP126 3da26028. Two ways this failed open. A lookup that raised
        returned False — so an error, which tells us nothing about whether
        enforcement applies, silently selected the lenient answer. And the
        pre-registration window was inferred from absence and unbounded, so
        constitutional paths could run degraded indefinitely while
        security-critical execution was already reachable.

        Now: an error means strict, because not knowing whether the
        constitution applies is not a licence to skip it. The startup window
        where services genuinely have not registered yet is preserved — boot
        would otherwise deadlock against its own constitution — but it is
        bounded in time and reported once, so a degraded window that
        outlives boot becomes visible instead of permanent.
        """
        try:
            if (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            ):
                return True
        except (RuntimeError, AttributeError, TypeError) as exc:
            record_degradation(
                "constitution",
                exc,
                severity="critical",
                action="enforced STRICT constitutional semantics after a failed strictness lookup",
            )
            return True

        # No enforcement signal. Legitimate only while the runtime is still
        # coming up.
        elapsed = time.time() - self._constitution_started_at
        if elapsed <= _BOOTSTRAP_LENIENCY_WINDOW_S:
            return False

        if not self._bootstrap_window_expired_reported:
            self._bootstrap_window_expired_reported = True
            record_degradation(
                "constitution",
                RuntimeError(
                    "no constitutional enforcement service registered "
                    f"{elapsed:.0f}s after start; enforcing strict semantics"
                ),
                severity="critical",
                action="closed the constitutional bootstrap window",
            )
        return True

    def _get_executive_core(self) -> Any:
        try:
            from core.executive.executive_core import get_executive_core

            return get_executive_core()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('constitution', exc)
            logger.debug("ExecutiveCore resolution failed: %s", exc)
            return None

    def _get_authority_gateway(self) -> Any:
        try:
            from core.executive.authority_gateway import get_authority_gateway

            return get_authority_gateway()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('constitution', exc)
            logger.debug("AuthorityGateway resolution failed: %s", exc)
            return None

    def _get_intention_loop(self) -> Any:
        try:
            from core.agency.intention_loop import get_intention_loop

            return ServiceContainer.get("intention_loop", default=None) or get_intention_loop()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('constitution', exc)
            logger.debug("IntentionLoop resolution failed: %s", exc)
            return None


_instance: ConstitutionalCore | None = None


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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('constitution', exc)
            logger.debug("ConstitutionalCore registration skipped: %s", exc)
    else:
        _instance.bind(orchestrator)

    return _instance
