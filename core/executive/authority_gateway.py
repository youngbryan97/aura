from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.agency.capability_system import get_capability_manager
from core.consciousness.substrate_authority import (
    ActionCategory,
    AuthorizationDecision,
)
from core.container import ServiceContainer
from core.executive.executive_core import (
    ActionType,
    DecisionOutcome,
    Intent,
    _coerce_intent_source,
)
from core.runtime.organism_status import get_organism_status
from core.runtime.service_access import optional_service, require_service

logger = logging.getLogger("Aura.AuthorityGateway")


@dataclass
class AuthorityDecision:
    approved: bool
    outcome: str
    reason: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    executive_intent_id: Optional[str] = None
    capability_token_id: Optional[str] = None
    substrate_receipt_id: Optional[str] = None
    will_receipt_id: Optional[str] = None
    domain: Optional[str] = None
    source: Optional[str] = None
    failure_pressure: float = 0.0
    canonical_self_version: Optional[int] = None


class AuthorityGateway:
    """Narrow-waist runtime gate over substrate, executive, and tokens.

    All consequential actions are first routed through the Unified Will
    (core/will.py) for a canonical WillDecision, then through the
    domain-specific checks (substrate, executive, capability tokens).
    This ensures a single audit trail for all authority decisions.
    """

    TOOL_TOKEN_TTL_S = 900

    def __init__(self) -> None:
        self._capabilities = get_capability_manager()

    @staticmethod
    def _will_gate(
        content: str,
        source: str,
        domain_str: str,
        priority: float,
        is_critical: bool = False,
    ) -> tuple[Optional["AuthorityDecision"], Optional[Any]]:
        """Route through UnifiedWill first.  Returns a blocking AuthorityDecision
        if the Will refuses, or None if the Will approves (let domain checks proceed).
        """
        try:
            from core.will import ActionDomain, get_will

            domain_map = {
                "tool_execution": ActionDomain.TOOL_EXECUTION,
                "state_mutation": ActionDomain.STATE_MUTATION,
                "memory_write": ActionDomain.MEMORY_WRITE,
                "initiative": ActionDomain.INITIATIVE,
                "expression": ActionDomain.EXPRESSION,
                "response": ActionDomain.RESPONSE,
            }
            domain = domain_map.get(domain_str, ActionDomain.STATE_MUTATION)
            will = get_will()
            decision = will.decide(
                content=content[:200],
                source=source,
                domain=domain,
                priority=priority,
                is_critical=is_critical,
            )
            if not decision.is_approved():
                return (
                    AuthorityDecision(
                        approved=False,
                        outcome=f"will_{decision.outcome.value}",
                        reason=decision.reason,
                        will_receipt_id=decision.receipt_id,
                        domain=domain.value,
                        source=source,
                    ),
                    decision,
                )
        except Exception as exc:
            record_degradation('authority_gateway', exc)
            logger.warning("UnifiedWill gate unavailable; failing closed: %s", exc)
            return (
                AuthorityDecision(
                    approved=False,
                    outcome="will_unavailable",
                    reason=f"UnifiedWill unavailable: {exc}",
                    domain=domain_str,
                    source=source,
                ),
                None,
            )
        return None, locals().get("decision")

    async def authorize_tool_execution(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        source: str = "unknown",
        priority: float = 0.7,
        is_critical: bool = False,
    ) -> AuthorityDecision:
        # ── Unified Will gate (canonical decision authority) ──
        will_block, will_decision = self._will_gate(
            f"tool:{tool_name}", source, "tool_execution", priority, is_critical
        )
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"tool:{tool_name} args:{str(args)[:100]}",
            source=source,
            category=ActionCategory.TOOL_EXECUTION,
            priority=priority,
            is_critical=is_critical,
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="tool_execution",
        )
        if blocked is not None:
            return blocked

        exec_core = self._get_executive_core()
        intent, record = await exec_core.prepare_tool_intent(tool_name, args, source=source)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="tool_execution",
            source=source,
        )
        if decision.approved:
            token = self._capabilities.generate_token(
                [tool_name],
                duration_s=self.TOOL_TOKEN_TTL_S,
            )
            token.metadata.update(
                {
                    "source": source,
                    "tool_name": tool_name,
                    "intent_id": intent.intent_id,
                    "substrate_receipt_id": receipt_id,
                }
            )
            decision.capability_token_id = token.token_id
        return decision

    async def authorize_state_mutation(
        self,
        origin: str,
        cause: str,
        *,
        priority: float = 0.5,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(f"state_mutation:{cause}", origin, "state_mutation", priority)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"state_mutation:{cause}",
            source=origin or "system",
            category=ActionCategory.STATE_MUTATION,
            priority=priority,
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="state_mutation",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(origin or "system"),
            goal=f"mutate_state:{origin}",
            action_type=ActionType.MUTATE_STATE,
            payload={"origin": origin, "cause": cause},
            priority=priority,
        )
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="state_mutation",
            source=origin or "system",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    def authorize_state_mutation_sync(
        self,
        origin: str,
        cause: str,
        *,
        priority: float = 0.5,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(f"state_mutation:{cause}", origin, "state_mutation", priority)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"state_mutation:{cause}",
            source=origin or "system",
            category=ActionCategory.STATE_MUTATION,
            priority=priority,
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="state_mutation",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(origin or "system"),
            goal=f"mutate_state:{origin}",
            action_type=ActionType.MUTATE_STATE,
            payload={"origin": origin, "cause": cause},
            priority=priority,
        )
        record = self._get_executive_core().request_approval_sync(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="state_mutation",
            source=origin or "system",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    async def authorize_memory_write(
        self,
        memory_type: str,
        content: str,
        *,
        source: str = "unknown",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(
            f"memory:{memory_type}:{str(content)[:80]}", source, "memory_write", importance
        )
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"memory:{memory_type}:{str(content)[:80]}",
            source=source or "system",
            category=ActionCategory.MEMORY_WRITE,
            priority=max(0.0, min(1.0, float(importance or 0.0))),
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="memory_write",
        )
        if blocked is not None:
            return blocked

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
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="memory_write",
            source=source or "system",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    def authorize_memory_write_sync(
        self,
        memory_type: str,
        content: str,
        *,
        source: str = "unknown",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(
            f"memory:{memory_type}:{str(content)[:80]}", source, "memory_write", importance
        )
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"memory:{memory_type}:{str(content)[:80]}",
            source=source or "system",
            category=ActionCategory.MEMORY_WRITE,
            priority=max(0.0, min(1.0, float(importance or 0.0))),
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="memory_write",
        )
        if blocked is not None:
            return blocked

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
        record = self._get_executive_core().request_approval_sync(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="memory_write",
            source=source or "system",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    def authorize_belief_update_sync(
        self,
        key: str,
        value: Any,
        *,
        note: Optional[str] = None,
        source: str = "unknown",
        priority: float = 0.7,
    ) -> AuthorityDecision:
        content = f"belief:{key}:{str(value)[:80]}"
        will_block, will_decision = self._will_gate(content, source, "memory_write", priority)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=content,
            source=source or "system",
            category=ActionCategory.MEMORY_WRITE,
            priority=max(0.0, min(1.0, float(priority or 0.0))),
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="memory_write",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "system"),
            goal=f"update_belief:{key}",
            action_type=ActionType.UPDATE_BELIEF,
            payload={
                "key": str(key or "")[:120],
                "value": value,
                "note": note,
            },
            priority=max(0.0, min(1.0, float(priority or 0.0))),
            requires_memory_commit=True,
        )
        record = self._get_executive_core().request_approval_sync(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="memory_write",
            source=source or "system",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    async def authorize_initiative(
        self,
        summary: str,
        *,
        source: str = "unknown",
        priority: float = 0.5,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(str(summary)[:200], source, "initiative", priority)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=str(summary or "")[:240],
            source=source or "autonomous",
            category=ActionCategory.INITIATIVE,
            priority=priority,
            require_substrate=True,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="initiative",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "autonomous"),
            goal=f"initiative:{str(summary or '')[:80]}",
            action_type=ActionType.SPAWN_TASK,
            payload={"summary": str(summary or "")[:240], "source": source},
            priority=priority,
        )
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="initiative",
            source=source or "autonomous",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    def authorize_initiative_sync(
        self,
        summary: str,
        *,
        source: str = "unknown",
        priority: float = 0.5,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(str(summary)[:200], source, "initiative", priority)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=str(summary or "")[:240],
            source=source or "autonomous",
            category=ActionCategory.INITIATIVE,
            priority=priority,
            require_substrate=True,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="initiative",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "autonomous"),
            goal=f"initiative:{str(summary or '')[:80]}",
            action_type=ActionType.SPAWN_TASK,
            payload={"summary": str(summary or "")[:240], "source": source},
            priority=priority,
        )
        record = self._get_executive_core().request_approval_sync(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="initiative",
            source=source or "autonomous",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    async def authorize_expression(
        self,
        content: str,
        *,
        source: str = "unknown",
        urgency: float = 0.5,
        is_critical: bool = False,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(str(content)[:200], source, "expression", urgency, is_critical)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=content[:240],
            source=source or "system",
            category=ActionCategory.EXPRESSION,
            priority=urgency,
            is_critical=is_critical,
            require_substrate=True,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="expression",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "autonomous"),
            goal=f"emit_message:{content[:40]}",
            action_type=ActionType.EMIT_MESSAGE,
            payload={"content": content[:240], "source": source},
            priority=urgency,
        )
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="expression",
            source=source or "autonomous",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    def authorize_expression_sync(
        self,
        content: str,
        *,
        source: str = "unknown",
        urgency: float = 0.5,
        is_critical: bool = False,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(str(content)[:200], source, "expression", urgency, is_critical)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=content[:240],
            source=source or "system",
            category=ActionCategory.EXPRESSION,
            priority=urgency,
            is_critical=is_critical,
            require_substrate=True,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="expression",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "system"),
            goal=f"emit_message:{content[:40]}",
            action_type=ActionType.EMIT_MESSAGE,
            payload={"content": content[:240], "source": source},
            priority=urgency,
        )
        record = self._get_executive_core().request_approval_sync(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="expression",
            source=source or "system",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    async def authorize_response(
        self,
        content: str,
        *,
        source: str = "user",
        priority: float = 0.4,
        is_critical: bool = False,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(str(content)[:200], source, "response", priority, is_critical)
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=content[:240],
            source=source or "user",
            category=ActionCategory.RESPONSE,
            priority=priority,
            is_critical=is_critical,
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="response",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "user"),
            goal=f"respond:{content[:40]}",
            action_type=ActionType.RESPOND,
            payload={"content": content[:240], "source": source},
            priority=priority,
        )
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="response",
            source=source or "user",
        )
        if decision.approved:
            self._complete_intent_safely(intent.intent_id, success=True)
        return decision

    def verify_tool_access(self, tool_name: str, token_id: Optional[str]) -> bool:
        return self._capabilities.verify_access(tool_name, token_id)

    def finalize_tool_execution(
        self,
        *,
        executive_intent_id: Optional[str] = None,
        capability_token_id: Optional[str] = None,
        success: bool = True,
    ) -> None:
        if executive_intent_id:
            try:
                self._get_executive_core().complete_intent(executive_intent_id, success=success)
            except Exception as exc:
                record_degradation('authority_gateway', exc)
                logger.debug("Executive intent completion skipped: %s", exc)
        if capability_token_id:
            try:
                self._capabilities.revoke_token(capability_token_id)
            except Exception as exc:
                record_degradation('authority_gateway', exc)
                logger.debug("Capability token revoke skipped: %s", exc)

    def _complete_intent_safely(self, intent_id: Optional[str], *, success: bool = True) -> None:
        if not intent_id:
            return
        try:
            self._get_executive_core().complete_intent(intent_id, success=success)
        except Exception as exc:
            record_degradation('authority_gateway', exc)
            logger.debug("Executive intent completion skipped: %s", exc)

    def _get_executive_core(self) -> Any:
        from core.executive import executive_core as executive_core_module

        return executive_core_module.get_executive_core()

    def _strict_runtime_active(self) -> bool:
        try:
            return (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
        except Exception:
            return False

    def _canonical_self_version(self) -> Optional[int]:
        organism = get_organism_status()
        version = organism.get("canonical_self_version")
        try:
            return int(version) if version is not None else None
        except (TypeError, ValueError):
            return None

    def _contextualize(
        self,
        *,
        approved: bool,
        outcome: str,
        reason: str,
        constraints: Optional[Dict[str, Any]] = None,
        executive_intent_id: Optional[str] = None,
        capability_token_id: Optional[str] = None,
        substrate_receipt_id: Optional[str] = None,
        will_receipt_id: Optional[str] = None,
        domain: Optional[str] = None,
        source: Optional[str] = None,
    ) -> AuthorityDecision:
        organism = get_organism_status()
        return AuthorityDecision(
            approved=approved,
            outcome=outcome,
            reason=reason,
            constraints=dict(constraints or {}),
            executive_intent_id=executive_intent_id,
            capability_token_id=capability_token_id,
            substrate_receipt_id=substrate_receipt_id,
            will_receipt_id=will_receipt_id,
            domain=domain,
            source=source,
            failure_pressure=float(organism.get("failure_pressure", 0.0) or 0.0),
            canonical_self_version=self._canonical_self_version(),
        )

    def _decision_from_record(
        self,
        record: Any,
        *,
        executive_intent_id: Optional[str] = None,
        substrate_constraints: Optional[Dict[str, Any]] = None,
        substrate_receipt_id: Optional[str] = None,
        will_receipt_id: Optional[str] = None,
        domain: Optional[str] = None,
        source: Optional[str] = None,
    ) -> AuthorityDecision:
        raw_outcome = getattr(record, "outcome", DecisionOutcome.REJECTED)
        outcome = getattr(raw_outcome, "value", str(raw_outcome or DecisionOutcome.REJECTED.value))
        approved = outcome in (DecisionOutcome.APPROVED.value, DecisionOutcome.DEGRADED.value)
        constraints = dict(getattr(record, "constraints", {}) or {})
        if substrate_constraints:
            constraints.update(substrate_constraints)
            if outcome == DecisionOutcome.APPROVED.value:
                outcome = DecisionOutcome.DEGRADED.value
        return self._contextualize(
            approved=approved,
            outcome=outcome,
            reason=str(getattr(record, "reason", "") or ""),
            constraints=constraints,
            executive_intent_id=executive_intent_id,
            substrate_receipt_id=substrate_receipt_id,
            will_receipt_id=will_receipt_id,
            domain=domain,
            source=source,
        )

    def _substrate_preflight(
        self,
        *,
        content: str,
        source: str,
        category: ActionCategory,
        priority: float,
        is_critical: bool = False,
        require_substrate: bool = False,
        will_receipt_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> tuple[Optional[AuthorityDecision], Dict[str, Any], Optional[str]]:
        authority = None
        require_substrate = bool(require_substrate and self._strict_runtime_active())
        try:
            if require_substrate:
                authority = require_service("substrate_authority")
            else:
                authority = optional_service("substrate_authority", default=None)
        except Exception as exc:
            record_degradation('authority_gateway', exc)
            return (
                self._contextualize(
                    approved=False,
                    outcome="rejected",
                    reason=f"substrate_authority_required:{type(exc).__name__}",
                    constraints={"blocked": True},
                    will_receipt_id=will_receipt_id,
                    domain=domain,
                    source=source,
                ),
                {},
                None,
            )

        if authority is None:
            return None, {}, None

        try:
            verdict = authority.authorize(
                content=content,
                source=source,
                category=category,
                priority=priority,
                is_critical=is_critical,
            )
        except Exception as exc:
            record_degradation('authority_gateway', exc)
            if require_substrate:
                return (
                    self._contextualize(
                        approved=False,
                        outcome="rejected",
                        reason=f"substrate_gate_failed:{type(exc).__name__}",
                        constraints={"blocked": True},
                        will_receipt_id=will_receipt_id,
                        domain=domain,
                        source=source,
                    ),
                    {},
                    None,
                )
            logger.debug("Substrate preflight skipped for %s: %s", category.name, exc)
            return None, {}, None

        if verdict.decision == AuthorizationDecision.BLOCK:
            return (
                self._contextualize(
                    approved=False,
                    outcome="rejected",
                    reason=f"substrate_blocked:{verdict.reason}",
                    constraints={
                        "blocked": True,
                        "substrate_constraints": list(verdict.constraints or []),
                    },
                    substrate_receipt_id=verdict.receipt_id,
                    will_receipt_id=will_receipt_id,
                    domain=domain,
                    source=source,
                ),
                {},
                verdict.receipt_id,
            )

        constraints: Dict[str, Any] = {}
        if verdict.decision == AuthorizationDecision.CONSTRAIN:
            constraints["substrate_constrained"] = True
            constraints["substrate_constraints"] = list(verdict.constraints or [])
        elif verdict.decision == AuthorizationDecision.CRITICAL_PASS:
            constraints["substrate_critical_pass"] = True

        return None, constraints, verdict.receipt_id


_instance: Optional[AuthorityGateway] = None


def get_authority_gateway() -> AuthorityGateway:
    global _instance
    if _instance is None:
        _instance = AuthorityGateway()
        try:
            ServiceContainer.register_instance("authority_gateway", _instance, required=False)
        except Exception as exc:
            record_degradation('authority_gateway', exc)
            logger.debug("AuthorityGateway registration skipped: %s", exc)
    return _instance
