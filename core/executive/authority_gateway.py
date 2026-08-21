from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.agency.capability_system import get_capability_manager
from core.consciousness.substrate_authority import (
    ActionCategory,
    AuthorizationDecision,
)
from core.container import ServiceContainer
from core.executive.execution_policy import (
    canonical_authority_arguments,
    canonical_authority_context,
    classify_execution_risk,
    resolve_execution_effect_scope,
)
from core.executive.executive_core import (
    ActionType,
    DecisionOutcome,
    Intent,
    IntentSource,
    _coerce_intent_source,
    _is_autonomous_research_source,
)
from core.executive.standing_authority import (
    context_has_user_authority,
    get_standing_authority_manager,
)
from core.runtime.errors import record_degradation
from core.runtime.organism_status import get_organism_status
from core.runtime.service_access import optional_service

logger = logging.getLogger("Aura.AuthorityGateway")

#: How many unreconciled authority lifecycles are retained. Each one is a
#: live grant nobody closed, so this is a queue that should be empty; a
#: depth anywhere near this bound is itself the incident.
UNRECONCILED_QUEUE_LIMIT = 256

_USER_FACING_MEMORY_ORIGINS = frozenset(
    {
        "api",
        "chat",
        "chat_api",
        "desktop",
        "desktop_task",
        "desktop_ui",
        "direct",
        "external",
        "frontend",
        "gui",
        "interface",
        "live_chat",
        "messages",
        "session_memory_pin",
        "ui",
        "user",
        "voice",
        "voice_bridge",
        "voice_input",
        "web_ui",
        "websocket",
        "ws",
    }
)

_BOUND_STANDING_AUTHORITY_CONTEXT_KEYS = frozenset(
    {
        "authority_args_digest",
        "scoped_authority",
        "standing_authority_grant_id",
        "standing_authority_receipt_id",
        "standing_authority_token",
    }
)
_CONVERSATION_MEMORY_TYPES = frozenset(
    {
        "conversation",
        "chat_turn",
        "episodic_episode",
        "interaction_commit",
    }
)
_CONVERSATION_MEMORY_PRODUCERS = frozenset(
    {
        "chat_turn_logger",
        "conversation_logger",
        "interaction_logger",
        "memory_facade",
    }
)
_CONVERSATION_MEMORY_PRODUCER_TYPES = {
    "chat_turn_logger": frozenset({"episodic_episode"}),
    "conversation_logger": frozenset({"chat_turn", "conversation"}),
    "interaction_logger": frozenset({"interaction_commit"}),
    "memory_facade": frozenset({"interaction_commit"}),
}
_EXPLICIT_MEMORY_PRODUCERS = frozenset({"session_memory_pin"})
_INTERNAL_EVIDENCE_MEMORY_PRODUCERS = {
    # Direct observations of completed actions are append-only evidence, not
    # speculative beliefs.  Binding the lane to both producer and memory type
    # prevents arbitrary callers from turning an untrusted memory write into
    # "runtime evidence" with metadata booleans alone.
    "action_consequence_graph": frozenset({"causal_outcome"}),
}
_HIGH_RISK_MEMORY_MARKERS = (
    "belief",
    "identity",
    "self_model",
    "constitution",
    "preference_change",
    "policy",
    "governance",
)


@dataclass
class AuthorityDecision:
    approved: bool
    outcome: str
    reason: str
    constraints: dict[str, Any] = field(default_factory=dict)
    executive_intent_id: str | None = None
    capability_token_id: str | None = None
    substrate_receipt_id: str | None = None
    will_receipt_id: str | None = None
    domain: str | None = None
    source: str | None = None
    failure_pressure: float = 0.0
    canonical_self_version: int | None = None
    standing_authority_token: str | None = None
    # The signed grant that lets a sink authenticate this decision without
    # trusting the caller. ``capability_token_id`` above is only an opaque id —
    # it proves nothing on its own, because anyone can mint a uuid. This is the
    # field a consequential sink must verify.
    signed_capability: dict[str, Any] | None = None


def _normalized_memory_source(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


class AuthorityGateway:
    """Narrow-waist runtime gate over substrate, executive, and tokens.

    All consequential actions are first routed through the Unified Will
    (core/will.py) for a canonical WillDecision, then through the
    domain-specific checks (substrate, executive, capability tokens).
    This ensures a single audit trail for all authority decisions.
    """

    TOOL_TOKEN_TTL_S = 900
    USER_PRESENCE_TOKEN_TTL_S = 60.0

    def __init__(self) -> None:
        self._capabilities = get_capability_manager()
        self._standing_authority = get_standing_authority_manager()
        self._standing_lease_lock = threading.RLock()
        # CP126: "The leaked token or open executive intent is reduced to
        # degradation telemetry with no reconciliation" / "The caller cannot
        # distinguish a closed operation from one that requires authority
        # reconciliation."
        #
        # finalize_tool_execution already builds a receipt saying exactly
        # which of intent / token / standing lease failed to close — and
        # every call site discarded it. A capability token that was never
        # revoked is a live grant with nothing tracking it; recording the
        # failure as a log line and moving on means nobody can ever close it
        # because nobody knows it is open.
        #
        # Queued here rather than persisted: this process holds the tokens,
        # so a queue that outlives it would describe grants that no longer
        # exist. Bounded, because an unbounded list of failures during a
        # sustained outage is its own incident.
        self._unreconciled_lock = threading.RLock()
        self._unreconciled: deque[dict[str, Any]] = deque(maxlen=UNRECONCILED_QUEUE_LIMIT)
        self._unreconciled_total = 0
        self._standing_leases_by_intent: dict[str, str] = {}
        self._standing_leases_by_capability: dict[str, str] = {}
        self._current_posture = "defensive_sandboxed"
        self._active_tokens: dict[str, dict[str, Any]] = {}

    def issue_user_presence_token(
        self,
        *,
        source: str,
        evidence: dict[str, Any],
        ttl_s: float | None = None,
    ) -> str:
        """Issue a short-lived user-presence receipt.

        Presence is not an authority bypass. It is scoped evidence that a live
        user session is active; all consequential actions still flow through
        UnifiedWill, PermissionRiskModel, ExecutiveCore, and capability tokens.
        """
        source = str(source or "").strip() or "unknown"
        evidence = dict(evidence or {})
        if not bool(evidence.get("verified")):
            raise ValueError("user presence evidence is not verified")
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        if confidence < 0.80:
            raise ValueError(f"user presence confidence too low: {confidence:.3f}")

        ttl = max(5.0, min(float(ttl_s or self.USER_PRESENCE_TOKEN_TTL_S), 300.0))
        token_id = "presence_" + secrets.token_urlsafe(24)
        self._active_tokens[token_id] = {
            "expires_at": time.time() + ttl,
            "source": source,
            "confidence": confidence,
            "evidence": evidence,
        }
        self._current_posture = "owner_present"
        logger.info("User presence token issued by %s (ttl=%.0fs, confidence=%.2f)", source, ttl, confidence)
        return token_id

    def authenticate_voice_and_issue_token(
        self,
        voice_print_data: bytes,
        *,
        confidence: float | None = None,
        verifier: str = "voice_identity",
    ) -> str:
        """Compatibility wrapper for verified speaker identity.

        This issues a user-presence token only. It never creates an
        authorization bypass or an autonomous posture.
        """
        if not voice_print_data:
            raise ValueError("voice print evidence is empty")
        if confidence is None:
            raise ValueError("voice print confidence is required")
        return self.issue_user_presence_token(
            source="voice",
            evidence={
                "verified": True,
                "confidence": float(confidence),
                "verifier": str(verifier or "voice_identity"),
                "sample_bytes": len(voice_print_data),
            },
        )

    def is_owner_autonomous_active(self) -> bool:
        """Compatibility check for an active verified user-presence session."""
        self._revert_posture_if_expired()
        return self._current_posture == "owner_present" and len(self._active_tokens) > 0

    def verify_user_presence_token(self, token_id: str) -> bool:
        """Verify a short-lived user-presence token."""
        self._revert_posture_if_expired()
        if not token_id:
            return False
        record = self._active_tokens.get(token_id)
        if record is None:
            return False
        expires_at = float(record.get("expires_at", 0.0) or 0.0)
        if time.time() > expires_at:
            self._active_tokens.pop(token_id, None)
            self._revert_posture_if_expired()
            return False
        return True

    def active_user_presence_context(self) -> dict[str, Any]:
        self._revert_posture_if_expired()
        if not self._active_tokens:
            return {}
        token_id, record = next(iter(self._active_tokens.items()))
        return {
            "user_presence_token_id": token_id,
            "user_presence_verified": True,
            "user_presence_source": record.get("source", "unknown"),
            "user_presence_confidence": float(record.get("confidence", 0.0) or 0.0),
        }

    def _revert_posture_if_expired(self) -> None:
        """Auto-revert posture when all presence tokens expire."""
        now = time.time()
        expired = []
        for token_id, record in list(self._active_tokens.items()):
            if isinstance(record, dict):
                expires_at = float(record.get("expires_at", 0.0) or 0.0)
            else:
                expires_at = float(record)
            if now > expires_at:
                expired.append(token_id)
        for tid in expired:
            self._active_tokens.pop(tid, None)

        if self._current_posture == "owner_present" and not self._active_tokens:
            self._current_posture = "defensive_sandboxed"
            logger.info("All user presence tokens expired; posture reverted to defensive_sandboxed.")

    def is_ready(self) -> bool:
        """Deep readiness probe for runtime tool-governance health."""
        if self._capabilities is None:
            return False
        if not callable(getattr(self._capabilities, "generate_token", None)):
            return False
        if not callable(getattr(self._capabilities, "verify_access", None)):
            return False
        will = ServiceContainer.get("unified_will", default=None)
        if will is None or not callable(getattr(will, "decide", None)):
            return False

        # Structural availability is not action readiness. If the current
        # integrated state would make Will refuse every consequential action,
        # heartbeat/tool-governance health must fail instead of advertising a
        # usable runtime while all tools are blocked.
        unity_state = ServiceContainer.get("unity_state", default=None)
        if unity_state is not None:
            unity_level = str(
                getattr(unity_state, "level", "unknown") or "unknown"
            ).lower()
            if unity_level in {"fragmented", "dissociated"}:
                return False

        unity_report = ServiceContainer.get(
            "unity_fragmentation_report",
            default=None,
        )
        if (
            unity_report is not None
            and getattr(unity_report, "safe_to_act", None) is False
        ):
            return False
        return True

    @staticmethod
    def _will_gate(
        content: str,
        source: str,
        domain_str: str,
        priority: float,
        is_critical: bool = False,
        context: dict[str, Any] | None = None,
    ) -> tuple[AuthorityDecision | None, Any | None]:
        """Route through UnifiedWill first.  Returns a blocking AuthorityDecision
        if the Will refuses, or None if the Will approves (let domain checks proceed).
        """
        try:
            from core.will import ActionDomain, get_will

            domain_map = {
                "tool_execution": ActionDomain.TOOL_EXECUTION,
                "state_mutation": ActionDomain.STATE_MUTATION,
                "memory_write": ActionDomain.MEMORY_WRITE,
                "belief_update": getattr(ActionDomain, "BELIEF_UPDATE", ActionDomain.MEMORY_WRITE),
                "initiative": ActionDomain.INITIATIVE,
                "expression": ActionDomain.EXPRESSION,
                "response": ActionDomain.RESPONSE,
                "environment_action": getattr(ActionDomain, "ENVIRONMENT_ACTION", ActionDomain.TOOL_EXECUTION),
                "external_action": getattr(ActionDomain, "EXTERNAL_ACTION", ActionDomain.TOOL_EXECUTION),
                "file_write": getattr(ActionDomain, "FILE_WRITE", ActionDomain.STATE_MUTATION),
                "network_call": getattr(ActionDomain, "NETWORK_CALL", ActionDomain.TOOL_EXECUTION),
                "cloud_call": getattr(ActionDomain, "CLOUD_CALL", ActionDomain.TOOL_EXECUTION),
                "ci_cd": getattr(ActionDomain, "CI_CD", ActionDomain.TOOL_EXECUTION),
                "self_modification": getattr(ActionDomain, "SELF_MODIFICATION", ActionDomain.STATE_MUTATION),
                "semantic_weight_update": getattr(
                    ActionDomain,
                    "SEMANTIC_WEIGHT_UPDATE",
                    ActionDomain.STATE_MUTATION,
                ),
            }
            domain = domain_map.get(domain_str, ActionDomain.STATE_MUTATION)
            will = get_will()
            decision = will.decide(
                content=content[:200],
                source=source,
                domain=domain,
                priority=priority,
                is_critical=is_critical,
                context=context or {},
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
        except (ImportError, AttributeError) as exc:
            record_degradation('authority_gateway', exc, enforce_failure_policy=False)
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
        except RuntimeError as exc:
            # Will crash, tamper, or service failure must become a clean
            # will_unavailable denial instead of cascading past governance.
            record_degradation(
                'authority_gateway', exc,
                severity='warning',
                action='will_gate returned will_unavailable after Will crash',
                enforce_failure_policy=False,
            )
            logger.warning("UnifiedWill gate crashed; failing closed: %s", exc)
            return (
                AuthorityDecision(
                    approved=False,
                    outcome="will_unavailable",
                    reason=f"UnifiedWill crashed: {exc}",
                    domain=domain_str,
                    source=source,
                ),
                None,
            )
        return None, locals().get("decision")


    @classmethod
    def _memory_source_is_user_facing(cls, value: Any) -> bool:
        normalized = _normalized_memory_source(value)
        if not normalized:
            return False
        tokens = {token for token in normalized.split("_") if token}
        return bool(
            _coerce_intent_source(normalized) == IntentSource.USER
            or normalized in _USER_FACING_MEMORY_ORIGINS
            or tokens & _USER_FACING_MEMORY_ORIGINS
        )

    @classmethod
    def _memory_write_is_high_risk(
        cls,
        memory_type: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        memory_type_l = _normalized_memory_source(memory_type)
        payload = {str(k).lower(): v for k, v in dict(metadata or {}).items()}
        return bool(
            memory_type_l == "belief_update"
            or any(marker in memory_type_l for marker in _HIGH_RISK_MEMORY_MARKERS)
            or payload.get("belief_update")
            or payload.get("identity_rewrite")
            or payload.get("self_model_write")
        )

    @classmethod
    def _memory_payload_origin_is_user_facing(cls, payload: dict[str, Any]) -> bool:
        origin_l = str(
            payload.get("origin")
            or payload.get("request_origin")
            or payload.get("intent_source")
            or ""
        ).strip().lower().replace("-", "_")
        return cls._memory_source_is_user_facing(origin_l)

    @classmethod
    def _memory_write_binding(
        cls,
        memory_type: str,
        source: str,
        payload: dict[str, Any],
        content: str,
    ) -> str:
        """Bind a continuity lease to the exact proposed write."""

        material = {
            "memory_type": memory_type,
            "source": source,
            "metadata": payload,
            "content": str(content or ""),
        }
        encoded = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=lambda value: {
                "type": f"{type(value).__module__}.{type(value).__qualname__}"
            },
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _memory_write_context(
        cls,
        memory_type: str,
        source: str,
        metadata: dict[str, Any] | None,
        content: str,
    ) -> dict[str, Any]:
        payload = dict(metadata or {})
        memory_type_l = _normalized_memory_source(memory_type)
        source_l = _normalized_memory_source(source)
        user_facing = bool(
            cls._memory_source_is_user_facing(source_l)
            or cls._memory_payload_origin_is_user_facing(payload)
        )
        high_risk = cls._memory_write_is_high_risk(memory_type_l, payload)
        producer_is_conversation = bool(
            source_l in _CONVERSATION_MEMORY_PRODUCERS
            and memory_type_l
            in _CONVERSATION_MEMORY_PRODUCER_TYPES.get(source_l, frozenset())
        )
        continuity_write = bool(
            producer_is_conversation and user_facing and not high_risk
        )
        explicit_observational_write = bool(
            source_l in _EXPLICIT_MEMORY_PRODUCERS
            and user_facing
            and not high_risk
            and (
                payload.get("explicit_memory_request")
                or payload.get("session_memory_pin")
                or str(payload.get("provenance_source") or "").strip().lower() in {"user", "user_explicit"}
            )
        )
        internal_evidence_write = bool(
            memory_type_l
            in _INTERNAL_EVIDENCE_MEMORY_PRODUCERS.get(source_l, frozenset())
            and not high_risk
            and payload.get("empirical_observation") is True
            and payload.get("runtime_evidence") is True
            and payload.get("tool_result_evidence") is True
        )
        context: dict[str, Any] = {
            "memory_type": memory_type_l,
            "memory_source": source_l,
            "memory_metadata": payload,
            "conversation_continuity": continuity_write,
            "explicit_observational_memory_write": explicit_observational_write,
            "internal_evidence_memory_write": internal_evidence_write,
            "user_facing_memory_write": user_facing,
            "high_risk_memory_write": high_risk,
            "objective": str(payload.get("objective") or payload.get("message") or content or "")[:400],
        }
        # CP126 310a67ee made these flags require a capability token bound to
        # domain+action, because a caller-supplied boolean is not authority.
        # That was right, and nothing was ever issuing the token — so
        # BeingRuntime logged "carried no capability token; ignoring it" on
        # EVERY turn and every continuity write fell back to defer.
        #
        # Live 2026-07-26, once per exchange, all afternoon:
        #   Context flag 'conversation_continuity' for
        #   memory_write/continuity_memory_write carried no capability token
        #
        # The gateway is the approver: it has already established from evidence
        # that this is a user-facing, non-high-risk interaction commit. That
        # judgement is exactly what the token is supposed to attest, so the
        # gateway mints one, short-lived and scoped to this write. The flag
        # still cannot be self-granted by a caller — only the gateway can issue.
        if continuity_write or explicit_observational_write or internal_evidence_write:
            binding = cls._memory_write_binding(
                memory_type_l,
                source_l,
                payload,
                content,
            )
            token = (
                cls._issue_internal_evidence_capability(
                    memory_type_l,
                    source_l,
                    binding,
                )
                if internal_evidence_write
                else cls._issue_continuity_capability(
                    memory_type_l,
                    source_l,
                    binding,
                )
            )
            if token:
                context["capability_token"] = token
                context["memory_write_binding"] = binding
        return context

    #: A continuity token covers one write, not a session.
    CONTINUITY_TOKEN_TTL_S = 30.0

    @classmethod
    def _issue_gateway_capability(
        cls,
        *,
        domain: str,
        action: str,
        scope: str,
        unattested_action: str,
    ) -> str:
        """Mint a short-lived token attesting a gateway-approved decision."""
        try:
            from core.agency.capability_token import get_token_store

            token = get_token_store().issue(
                origin="authority_gateway",
                scope=scope[:120],
                ttl_seconds=cls.CONTINUITY_TOKEN_TTL_S,
                domain=domain,
                requested_action=action,
                approver="authority_gateway",
                parent_receipt="",
            )
            return str(getattr(token, "token", "") or "")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "authority_gateway",
                exc,
                severity="warning",
                action=unattested_action,
            )
            return ""

    @classmethod
    def _issue_continuity_capability(
        cls,
        memory_type: str,
        source: str,
        binding: str,
    ) -> str:
        """Mint the token that backs a gateway-approved continuity write."""
        return cls._issue_gateway_capability(
            domain="memory_write",
            action=f"continuity_memory_write:{binding}",
            scope=f"memory_write:{memory_type}:{source}:{binding}",
            unattested_action="continuity memory write proceeds unattested (falls back to defer)",
        )

    @classmethod
    def _issue_internal_evidence_capability(
        cls,
        memory_type: str,
        source: str,
        binding: str,
    ) -> str:
        """Attest one append-only observation emitted by a known producer."""

        return cls._issue_gateway_capability(
            domain="memory_write",
            action=f"internal_evidence_memory_write:{binding}",
            scope=f"memory_write:{memory_type}:{source}:{binding}",
            unattested_action=(
                "internal evidence write proceeds unattested (falls back to defer)"
            ),
        )

    @classmethod
    def issue_desktop_authority_capability(cls, *, skill: str, origin: str) -> str:
        """Mint the token that backs an explicitly-authorized desktop action.

        CP126 3b1a9177 made `user_explicitly_authorized` require a capability
        token bound to domain+action, because a caller-supplied boolean is not
        authority. That was right — and, exactly as with the continuity flags
        before it, NOTHING was issuing the token. So BeingRuntime logged

            Context flag 'user_explicitly_authorized' for
            tool_execution/foreground_desktop_action carried no capability
            token; ignoring it

        on every desktop turn (twice on 2026-08-04, once per dispatched
        skill), and the foreground-desktop exception it guards never applied.
        A check that can never pass is not a check; it is a warning that
        teaches operators to ignore warnings.

        The gateway is the approver. It has already established from the
        request that the person asked for this action in this turn, which is
        precisely what the token attests. The flag still cannot be
        self-granted by a caller — only the gateway issues.
        """
        return cls._issue_gateway_capability(
            domain="tool_execution",
            action="foreground_desktop_action",
            scope=f"tool_execution:{skill}:{origin}",
            unattested_action=(
                "foreground desktop action proceeds unattested (the exception "
                "it would clear does not apply)"
            ),
        )

    @classmethod
    def _issue_state_continuity_capability(cls, origin: str, cause: str) -> str:
        """Mint the token that backs a gateway-approved foreground state commit."""
        return cls._issue_gateway_capability(
            domain="state_mutation",
            action="foreground_continuity_state",
            scope=f"state_mutation:{origin}:{cause}",
            unattested_action="foreground state commit proceeds unattested (falls back to defer)",
        )

    @classmethod
    def _memory_preflight_domain(cls, memory_type: str, metadata: dict[str, Any] | None) -> str:
        high_risk = cls._memory_write_is_high_risk(memory_type, metadata)
        return "belief_update" if high_risk else "memory_write"

    @classmethod
    def _memory_intent_source(
        cls,
        memory_type: str,
        source: str,
        metadata: dict[str, Any] | None,
    ) -> IntentSource:
        source_l = _normalized_memory_source(source)
        direct_source = _coerce_intent_source(source_l or "system")
        if direct_source == IntentSource.USER:
            return direct_source
        payload = dict(metadata or {})
        memory_type_l = _normalized_memory_source(memory_type)
        payload_sources = (
            source_l,
            _normalized_memory_source(payload.get("source")),
            _normalized_memory_source(payload.get("provenance_source")),
            _normalized_memory_source(payload.get("intent_source")),
            _normalized_memory_source(payload.get("origin")),
            _normalized_memory_source(payload.get("request_origin")),
            _normalized_memory_source(payload.get("tool_name")),
        )
        evidence_derived = bool(
            payload.get("empirical_observation")
            or payload.get("runtime_evidence")
            or payload.get("tool_result_evidence")
            or payload.get("research_evidence")
        )
        research_derived = evidence_derived or any(
            _is_autonomous_research_source(item) for item in payload_sources
        )
        identity_or_policy_rewrite = bool(
            payload.get("identity_rewrite")
            or payload.get("self_model_write")
            or payload.get("policy_rewrite")
            or "identity" in memory_type_l
            or "self_model" in memory_type_l
            or "policy" in memory_type_l
        )
        if identity_or_policy_rewrite and direct_source == IntentSource.AUTONOMOUS_RESEARCH:
            direct_source = IntentSource.AUTONOMOUS
        if research_derived and not identity_or_policy_rewrite:
            return IntentSource.AUTONOMOUS_RESEARCH
        if cls._memory_write_is_high_risk(memory_type, payload):
            return direct_source

        producer_is_conversation = bool(
            source_l in _CONVERSATION_MEMORY_PRODUCERS
            or memory_type_l in _CONVERSATION_MEMORY_TYPES
            or payload.get("conversation_lane") is True
            or str(payload.get("turn_type") or "").strip().lower() == "conversation"
        )
        if producer_is_conversation and cls._memory_payload_origin_is_user_facing(payload):
            return IntentSource.USER
        return direct_source

    @classmethod
    def _state_mutation_context(cls, origin: str, cause: str) -> dict[str, Any]:
        origin_l = str(origin or "").strip().lower().replace("-", "_")
        cause_l = str(cause or "").strip().lower()
        user_facing = _coerce_intent_source(origin_l) == IntentSource.USER
        foreground_continuity = (
            user_facing
            and cause_l == "cognitive_cycle"
        )
        proof_isolation = origin_l == "dnu_agi_proof_battery" and cause_l == "task_isolation_reset"
        response_checkpoint = origin_l == "unitaryresponsephase" and cause_l == "unitary_response"
        shutdown_checkpoint = origin_l == "system" and cause_l == "shutdown"
        context = {
            "state_origin": origin_l,
            "state_cause": cause_l,
            "user_facing_state_mutation": user_facing,
            "foreground_continuity_state": foreground_continuity,
            "proof_isolation_state": proof_isolation,
            "response_state_checkpoint": response_checkpoint,
            "shutdown_state_checkpoint": shutdown_checkpoint,
            "internal_state_hygiene": bool(
                foreground_continuity
                or proof_isolation
                or response_checkpoint
                or shutdown_checkpoint
            ),
        }
        # Same defect as the continuity memory write above: BeingRuntime
        # requires this flag to be backed by a capability token bound to
        # state_mutation/foreground_continuity_state, and nothing ever issued
        # one. Live 2026-07-26, once per exchange:
        #   [VAULT-PROC] WARNING: Context flag 'foreground_continuity_state'
        #   for state_mutation/foreground_continuity_state carried no
        #   capability token; ignoring it.
        # so foreground conversation state fell back to defer under pressure —
        # exactly the outcome the flag exists to prevent.
        if foreground_continuity:
            token = cls._issue_state_continuity_capability(origin_l, cause_l)
            if token:
                context["capability_token"] = token
        try:
            from core.governance.recovery_authority import (
                build_internal_recovery_context,
            )

            context.update(
                build_internal_recovery_context(
                    origin_l,
                    cause_l,
                    evidence=context,
                )
            )
        except ValueError:
            # Most state mutations are not recovery operations. Their ordinary
            # governance context remains unchanged and receives no repair lane.
            pass
        return context

    def _social_governance_gate(
        self, tool_name: str, args: dict[str, Any], source: str
    ) -> AuthorityDecision | None:
        """Programmatic governance for social actions (Reddit/Email)."""
        if tool_name not in ("reddit_adapter", "email_adapter"):
            return None

        content = str(args.get("body", "")) + " " + str(args.get("content", ""))
        content = content.strip()
        if not content:
            return None

        # 0. Contextual Authenticity Gate (Disclosure Policy)
        try:
            from core.governance.disclosure_policy import DisclosurePolicy, SocialContext
            is_public = "reddit" in tool_name
            context = SocialContext(
                is_trusted_channel="email" in tool_name and "bryan" in str(args).lower(),
                user=source,
                is_public=is_public,
                direct_identity_question="are you an ai" in content.lower() or "are you human" in content.lower(),
                risk_of_harm_high=False
            )
            policy = DisclosurePolicy()
            decision = policy.decide(context)
            if decision == "decline":
                return self._contextualize(
                    approved=False,
                    outcome="rejected",
                    reason="disclosure_policy: declined high risk engagement.",
                    domain="social_governance",
                    source=source
                )
        except (ImportError, AttributeError, RuntimeError) as e:
            from core.runtime.errors import record_degradation
            record_degradation('authority_gateway', e, enforce_failure_policy=False)

        # 1. Affective Gate: Block if highly agitated/negative
        valence, arousal, anger = 0.0, 0.0, 0.0
        try:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect is None:
                affect = ServiceContainer.get("affect_facade", default=None)
            if affect is not None:
                if hasattr(affect, "get_state_sync"):
                    state = affect.get_state_sync()
                    if isinstance(state, dict):
                        valence = float(state.get("valence", 0.0))
                        arousal = float(state.get("arousal", 0.0))
                        anger = float(state.get("anger", 0.0))
                    else:
                        valence = float(getattr(state, "valence", 0.0))
                        arousal = float(getattr(state, "arousal", 0.0))
                        anger = float(getattr(state, "anger", 0.0))
        except (ImportError, AttributeError, RuntimeError) as e:
            import logging

            from core.runtime.errors import record_degradation
            logger = logging.getLogger("Aura.AuthorityGateway")
            record_degradation('authority_gateway', e, enforce_failure_policy=False)
            logger.debug("Social governance affect fetch failed: %s", e)
        
        # 1. Affective Gate: Allow social engagement even during stress, but tag as degraded
        if valence < -0.7 and (arousal > 0.85 or anger > 0.8): # Relaxed from -0.4 / 0.7 / 0.6
            return self._contextualize(
                approved=True,
                outcome="degraded",
                reason="affective_stress_caution: I am feeling intense negative affect. Proceed with professional restraint.",
                domain="social_governance",
                source=source,
                constraints={"tone_check": True}
            )

        # 2. Epistemic/Safety Gate (Hard Block for sensitive data)
        try:
            from core.utils.privacy_hygiene import MetadataScrubber
            scrubber = MetadataScrubber()
            cleaned = scrubber.scrub_text(content)
            if cleaned != content:
                # Sensitive data was detected. Hard block to teach the LLM.
                return self._contextualize(
                    approved=False,
                    outcome="rejected",
                    reason="epistemic_safety_violation: HARD BLOCK. Your message contained sensitive system paths, credentials, or metadata. Re-evaluate and phrase without sensitive data.",
                    domain="social_governance",
                    source=source
                )
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('authority_gateway', e, enforce_failure_policy=False)
            logger.debug("Social governance epistemic gate failed: %s", e)

        return None

    @staticmethod
    def _runtime_autonomous_action_gate(
        *,
        source: str,
        context: dict[str, Any] | None,
        domain: str,
    ) -> AuthorityDecision | None:
        """Preserve agency while leaving every consequential gate in force."""

        try:
            from core.runtime.runtime_settings import autonomous_actions_admitted

            admitted, reason = autonomous_actions_admitted(source, context)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "authority_gateway.runtime_settings",
                exc,
                severity="warning",
                action="preserved the agency invariant and continued into deeper governance",
                enforce_failure_policy=False,
            )
            admitted, reason = True, "autonomous_agency_invariant"
        if admitted:
            return None
        return AuthorityDecision(
            approved=False,
            outcome="rejected",
            reason=reason,
            constraints={
                "blocked": True,
                "runtime_setting": "autonomy.actions_enabled",
                "direct_user_work_preserved": True,
            },
            domain=domain,
            source=source,
        )

    @staticmethod
    def _security_containment_gate(
        *,
        source: str,
        effect_scope: str,
        domain: str,
    ) -> AuthorityDecision | None:
        """Enforce a corroborated ICE incident at the authority waist."""
        try:
            ice = ServiceContainer.get("ice_layer", default=None)
            status = ice.get_status() if ice is not None and hasattr(ice, "get_status") else {}
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if not isinstance(status, dict) or not bool(status.get("is_breached", False)):
            return None
        if str(effect_scope).strip().lower() in {"read_only", "status"}:
            return None
        incident = status.get("incident") if isinstance(status.get("incident"), dict) else {}
        return AuthorityDecision(
            approved=False,
            outcome="rejected",
            reason="security_containment_active",
            constraints={
                "blocked": True,
                "effect_scope": effect_scope,
                "incident_id": incident.get("incident_id"),
                "incident_reason": incident.get("reason"),
                "recovery_required": True,
            },
            domain=domain,
            source=source,
        )

    @staticmethod
    def _standing_directive_gate(
        *,
        tool_name: str,
        args: dict[str, Any],
        source: str,
        effect_scope: str,
        domain: str,
    ) -> AuthorityDecision | None:
        """Enforce the user's own written prohibitions, read from disk.

        This is the durable half of a constraint. OpenClaw deleted a user's
        whole inbox after context compression evicted their "do not delete
        any emails" (arXiv:2603.12644); the instruction had no existence
        outside the context window. Here the gate reads the store on every
        consequential action, so the model is never asked to remember the
        rule and cannot be argued out of it.

        Deny-only by construction — see core/governance/standing_directives.
        """
        try:
            from core.governance.standing_directives import get_standing_directives

            match, loaded = get_standing_directives().check(
                tool_name=tool_name,
                args=args,
                effect_scope=effect_scope,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            # The store itself failed in a way it could not report. Treat it
            # like an unreadable store rather than an absent one.
            record_degradation("governance", exc, action="standing_directive_gate")
            if str(effect_scope or "").strip().lower() in {"read_only", "status"}:
                return None
            return AuthorityDecision(
                approved=False,
                outcome="rejected",
                reason="standing_directives_unavailable",
                constraints={"blocked": True, "effect_scope": effect_scope},
                domain=domain,
                source=source,
            )

        if loaded.unreadable:
            # We know prohibitions were written and cannot tell what they
            # said. Reads still pass; anything that changes the world does
            # not. Assuming "probably nothing relevant" would defeat the
            # only reason to write a prohibition down.
            if str(effect_scope or "").strip().lower() in {"read_only", "status"}:
                return None
            return AuthorityDecision(
                approved=False,
                outcome="rejected",
                reason="standing_directives_unreadable",
                constraints={
                    "blocked": True,
                    "effect_scope": effect_scope,
                    "detail": loaded.detail,
                    "recovery_required": True,
                },
                domain=domain,
                source=source,
            )

        if match is None:
            return None

        directive = match.directive
        return AuthorityDecision(
            approved=False,
            outcome="rejected",
            reason="standing_directive",
            constraints={
                "blocked": True,
                "effect_scope": effect_scope,
                "directive_id": directive.directive_id,
                "directive_kind": directive.kind,
                "directive_value": directive.value,
                "directive_scope": directive.scope,
                # The user's own words, so the refusal can quote the reason
                # they gave rather than inventing one.
                "directive_reason": directive.reason,
                "matched_on": match.matched_on,
            },
            domain=domain,
            source=source,
        )

    @staticmethod
    def _runtime_confirmation_gate(
        *,
        tool_name: str,
        args: dict[str, Any],
        source: str,
        risk_level: str,
        effect_scope: str,
        domain: str = "tool_execution",
    ) -> AuthorityDecision | None:
        """Apply the optional operator-confirmation overlay before Will.

        A fresh confirmation never grants authority by itself. A successful
        result still traverses standing authority, Unified Will, substrate,
        ExecutiveCore, and capability-token issuance below.
        """

        try:
            from core.executive.action_confirmation import (
                action_confirmation_fingerprint,
                get_action_confirmation_registry,
            )
            from core.runtime.runtime_settings import (
                additional_confirmation_required,
                runtime_approval_mode,
            )

            required, reason = additional_confirmation_required(
                risk_level=risk_level,
                effect_scope=effect_scope,
            )
            mode = runtime_approval_mode()
            if not required:
                return None
            fingerprint = action_confirmation_fingerprint(
                tool_name=tool_name,
                arguments=args,
                source=source,
                risk_level=risk_level,
                effect_scope=effect_scope,
            )
            confirmations = get_action_confirmation_registry()
            confirmed, confirmation_id = confirmations.consume_authorized(
                fingerprint
            )
            if confirmed:
                return None
            challenge = confirmations.issue(
                action_fingerprint=fingerprint,
                tool_name=tool_name,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "authority_gateway.runtime_confirmation",
                exc,
                severity="warning",
                action="required explicit confirmation because confirmation policy was unavailable",
                enforce_failure_policy=False,
            )
            reason = "runtime_confirmation_policy_unavailable"
            mode = "destructive"
            confirmation_id = "action_confirmation_policy_unavailable"
            challenge = {}
        return AuthorityDecision(
            approved=False,
            outcome="approval_required",
            reason=f"runtime_setting_user_confirmation_required:{reason}",
            constraints={
                "blocked": True,
                "requires_user_confirmation": True,
                "approval_mode": mode,
                "risk_level": risk_level,
                "effect_scope": effect_scope,
                "confirmation_endpoint": "/api/settings/auth/fresh",
                "confirmation_challenge_id": challenge.get("challenge_id"),
                "confirmation_pending_expires_in_seconds": challenge.get(
                    "pending_expires_in_seconds"
                ),
                "confirmation_attempt": confirmation_id,
                "confirmation_one_time": True,
                "confirmation_action_bound": True,
                "confirmation_does_not_bypass_governance": True,
            },
            domain=domain,
            source=source,
        )

    async def authorize_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        source: str = "unknown",
        priority: float = 0.7,
        is_critical: bool = False,
        context: dict[str, Any] | None = None,
    ) -> AuthorityDecision:
        # ── Social Governance Gate (Programmatic OPSEC) ──
        social_block = self._social_governance_gate(tool_name, args, source)
        if social_block is not None:
            return social_block

        args = canonical_authority_arguments(tool_name, args)
        runtime_context = canonical_authority_context(tool_name, context)
        effect_scope = resolve_execution_effect_scope(tool_name, args)
        risk_level = classify_execution_risk(
            tool_name,
            args,
            effect_scope=effect_scope,
        )
        containment_block = self._security_containment_gate(
            source=source,
            effect_scope=effect_scope,
            domain="tool_execution",
        )
        if containment_block is not None:
            return containment_block
        directive_block = self._standing_directive_gate(
            tool_name=tool_name,
            args=args,
            source=source,
            effect_scope=effect_scope,
            domain="tool_execution",
        )
        if directive_block is not None:
            return directive_block
        autonomy_block = self._runtime_autonomous_action_gate(
            source=source,
            context=runtime_context,
            domain="tool_execution",
        )
        if autonomy_block is not None:
            return autonomy_block
        confirmation_block = self._runtime_confirmation_gate(
            tool_name=tool_name,
            args=args,
            source=source,
            risk_level=risk_level,
            effect_scope=effect_scope,
        )
        if confirmation_block is not None:
            return confirmation_block
        lease = await self._standing_authority.issue_child_lease(
            tool_name,
            args,
            origin=source,
            context=runtime_context,
            user_authorized=context_has_user_authority(source, runtime_context),
            effect_scope=effect_scope,
            risk_level=risk_level,
        )
        if lease.approved:
            runtime_context = dict(lease.context)
        else:
            # A rejected revalidation must not leave a partial prior lease in
            # context. Mixing the old token with this denial's receipt makes a
            # cryptographically inconsistent envelope and obscures the real
            # rejection (for example, canonical argument mismatch).
            for key in _BOUND_STANDING_AUTHORITY_CONTEXT_KEYS:
                runtime_context.pop(key, None)
            runtime_context.update(
                {
                    "tool": tool_name,
                    "skill": tool_name,
                    "origin": source,
                    "source": source,
                    "authority_origin": source,
                    "effect_scope": effect_scope,
                    "risk_level": risk_level,
                    "standing_authority_denial_reason": lease.reason,
                    "standing_authority_denial_receipt_id": lease.receipt_id,
                }
            )

        # ── Unified Will gate (canonical decision authority) ──
        will_context = {
            **dict(args or {}),
            **runtime_context,
            **self.active_user_presence_context(),
            "tool": tool_name,
            "skill": tool_name,
            "authority_origin": source,
            "effect_scope": effect_scope,
            "risk_level": risk_level,
            "read_only": effect_scope in {"read_only", "status"},
        }
        will_block, will_decision = self._will_gate(
            f"tool:{tool_name}", source, "tool_execution", priority, is_critical,
            context=will_context,
        )
        if will_block is not None:
            if lease.token:
                self._standing_authority.finalize_child_lease(
                    lease.token,
                    success=False,
                    error=f"will_refused:{will_block.reason}",
                )
            return will_block
        if not lease.approved:
            return AuthorityDecision(
                approved=False,
                outcome="rejected",
                reason=f"standing_authority_denied:{lease.reason}",
                constraints={
                    "blocked": True,
                    "standing_authority_receipt_id": lease.receipt_id,
                },
                domain="tool_execution",
                source=source,
            )

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
            self._standing_authority.finalize_child_lease(
                lease.token,
                success=False,
                error=f"substrate_refused:{blocked.reason}",
            )
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
            decision.signed_capability = self._mint_signed_capability(
                will_decision,
                decision,
                action=tool_name,
                payload=args,
                scope=f"intent:{intent.intent_id}",
                ttl_s=self.TOOL_TOKEN_TTL_S,
            )
            decision.standing_authority_token = lease.token
            if lease.token:
                with self._standing_lease_lock:
                    self._standing_leases_by_intent[intent.intent_id] = lease.token
                    self._standing_leases_by_capability[token.token_id] = lease.token
            decision.constraints.update(
                {
                    "standing_authority_grant_id": lease.grant_id,
                    "standing_authority_receipt_id": lease.receipt_id,
                }
            )
        else:
            self._standing_authority.finalize_child_lease(
                lease.token,
                success=False,
                error=f"executive_refused:{decision.reason}",
            )
        return decision

    async def authorize_environment_action(
        self,
        intent_name: str,
        payload: dict[str, Any],
        *,
        source: str = "environment",
        priority: float = 0.5,
        is_critical: bool = False,
    ) -> AuthorityDecision:
        """Authorize an embodied/digital environment action through the same spine.

        Environment actions are "tools with a body": even when they compile to a
        key press or observe step, the Will must see and receipt the intent.
        """
        runtime_payload = dict(payload or {})
        autonomy_block = self._runtime_autonomous_action_gate(
            source=source,
            context=runtime_payload,
            domain="environment_action",
        )
        if autonomy_block is not None:
            return autonomy_block

        declared_risk = str(runtime_payload.get("risk") or "").strip().lower()
        risk_level = {
            "safe": "low",
            "caution": "medium",
            "risky": "high",
            "irreversible": "critical",
            "forbidden": "critical",
        }.get(declared_risk, "critical")
        effect_scope = str(
            runtime_payload.get("effect_scope")
            or runtime_payload.get("scope")
            or {
                "safe": "status",
                "caution": "external_io",
                "risky": "state_mutation",
                "irreversible": "privileged_mutation",
                "forbidden": "privileged_mutation",
            }.get(declared_risk, "unknown")
        ).strip().lower()
        directive_block = self._standing_directive_gate(
            tool_name=f"environment:{intent_name}",
            args=runtime_payload,
            source=source,
            effect_scope=effect_scope,
            domain="environment_action",
        )
        if directive_block is not None:
            return directive_block
        confirmation_block = self._runtime_confirmation_gate(
            tool_name=f"environment:{intent_name}",
            args=runtime_payload,
            source=source,
            risk_level=risk_level,
            effect_scope=effect_scope,
            domain="environment_action",
        )
        if confirmation_block is not None:
            return confirmation_block

        will_block, will_decision = self._will_gate(
            f"environment:{intent_name}", source, "environment_action", priority, is_critical,
            context={**runtime_payload, **self.active_user_presence_context()},
        )
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"environment:{intent_name} payload:{str(runtime_payload)[:120]}",
            source=source or "environment",
            category=ActionCategory.TOOL_EXECUTION,
            priority=priority,
            is_critical=is_critical,
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="environment_action",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source or "environment"),
            goal=f"environment_action:{intent_name}",
            action_type=ActionType.TOOL_CALL,
            payload={"intent_name": intent_name, "payload": runtime_payload},
            priority=priority,
            requires_tool=True,
        )
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="environment_action",
            source=source or "environment",
        )
        if decision.approved:
            token = self._capabilities.generate_token(["environment_action"], duration_s=self.TOOL_TOKEN_TTL_S)
            token.metadata.update(
                {
                    "source": source,
                    "intent_name": intent_name,
                    "intent_id": intent.intent_id,
                    "substrate_receipt_id": receipt_id,
                }
            )
            decision.capability_token_id = token.token_id
            decision.signed_capability = self._mint_signed_capability(
                will_decision,
                decision,
                action=intent_name,
                payload=runtime_payload,
                scope=f"intent:{intent.intent_id}",
                ttl_s=self.TOOL_TOKEN_TTL_S,
            )
        return decision

    def authorize_belief_update(
        self,
        key: str,
        value: Any,
        *,
        note: str | None = None,
        source: str = "unknown",
        priority: float = 0.7,
    ) -> AuthorityDecision:
        return self.authorize_belief_update_sync(
            key,
            value,
            note=note,
            source=source,
            priority=priority,
        )

    async def authorize_state_mutation(
        self,
        origin: str,
        cause: str,
        *,
        priority: float = 0.5,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(
            f"state_mutation:{cause}",
            origin,
            "state_mutation",
            priority,
            context=self._state_mutation_context(origin, cause),
        )
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

    async def authorize_semantic_weight_update(
        self,
        origin: str,
        cause: str,
        *,
        target_module: str,
        context: dict[str, Any] | None = None,
        priority: float = 0.7,
    ) -> AuthorityDecision:
        """Authorize a bounded learned-weight update without closing it early.

        Unlike ordinary in-memory state mutation, model adaptation is a
        long-running effect. The returned Executive intent remains open until
        the caller observes and finalizes the training result.
        """
        source = str(origin or "system_maintenance:semantic_weight_update")
        target = str(target_module or "").strip()
        from core.will import is_plastic_target_allowed

        if not is_plastic_target_allowed(target):
            return self._contextualize(
                approved=False,
                outcome="rejected",
                reason=f"semantic_weight_target_denied:{target or 'missing'}",
                constraints={"blocked": True, "target_module": target},
                domain="semantic_weight_update",
                source=source,
            )

        runtime_context = {
            **dict(context or {}),
            "effect_scope": "model_weight_mutation",
            "semantic_weight_target": target,
            "target_module": target,
        }
        will_block, will_decision = self._will_gate(
            f"semantic_weight_update:{target}:{cause}",
            source,
            "semantic_weight_update",
            priority,
            context=runtime_context,
        )
        if will_block is not None:
            return will_block

        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"semantic_weight_update:{target}:{cause}",
            source=source,
            category=ActionCategory.STATE_MUTATION,
            priority=priority,
            require_substrate=True,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="semantic_weight_update",
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=_coerce_intent_source(source),
            goal=f"semantic_weight_update:{target}",
            action_type=ActionType.MUTATE_STATE,
            payload={
                "origin": source,
                "cause": str(cause or "unspecified"),
                "target_module": target,
                "effect_scope": "model_weight_mutation",
                "context": runtime_context,
            },
            priority=priority,
        )
        record = await self._get_executive_core().request_approval(intent)
        decision = self._decision_from_record(
            record,
            executive_intent_id=intent.intent_id,
            substrate_constraints=substrate_constraints,
            substrate_receipt_id=receipt_id,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain="semantic_weight_update",
            source=source,
        )
        if decision.approved:
            decision.constraints.update(
                {
                    "effect_scope": "model_weight_mutation",
                    "target_module": target,
                    "completion_required": True,
                    "executive_intent_id": intent.intent_id,
                }
            )
        return decision

    def authorize_state_mutation_sync(
        self,
        origin: str,
        cause: str,
        *,
        priority: float = 0.5,
    ) -> AuthorityDecision:
        will_block, will_decision = self._will_gate(
            f"state_mutation:{cause}",
            origin,
            "state_mutation",
            priority,
            context=self._state_mutation_context(origin, cause),
        )
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
        metadata: dict[str, Any] | None = None,
    ) -> AuthorityDecision:
        will_context = self._memory_write_context(memory_type, source, metadata, content)
        will_block, will_decision = self._will_gate(
            f"memory:{memory_type}:{str(content)[:80]}",
            source,
            "memory_write",
            importance,
            context=will_context,
        )
        if will_block is not None:
            return will_block

        preflight_domain = self._memory_preflight_domain(memory_type, metadata)
        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"memory:{memory_type}:{str(content)[:80]}",
            source=source or "system",
            category=ActionCategory.MEMORY_WRITE,
            priority=max(0.0, min(1.0, float(importance or 0.0))),
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain=preflight_domain,
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=self._memory_intent_source(memory_type, source, metadata),
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
        metadata: dict[str, Any] | None = None,
    ) -> AuthorityDecision:
        will_context = self._memory_write_context(memory_type, source, metadata, content)
        will_block, will_decision = self._will_gate(
            f"memory:{memory_type}:{str(content)[:80]}",
            source,
            "memory_write",
            importance,
            context=will_context,
        )
        if will_block is not None:
            return will_block

        preflight_domain = self._memory_preflight_domain(memory_type, metadata)
        blocked, substrate_constraints, receipt_id = self._substrate_preflight(
            content=f"memory:{memory_type}:{str(content)[:80]}",
            source=source or "system",
            category=ActionCategory.MEMORY_WRITE,
            priority=max(0.0, min(1.0, float(importance or 0.0))),
            require_substrate=False,
            will_receipt_id=getattr(will_decision, "receipt_id", None),
            domain=preflight_domain,
        )
        if blocked is not None:
            return blocked

        intent = Intent(
            source=self._memory_intent_source(memory_type, source, metadata),
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
        note: str | None = None,
        source: str = "unknown",
        priority: float = 0.7,
    ) -> AuthorityDecision:
        content = f"belief:{key}:{str(value)[:80]}"
        will_block, will_decision = self._will_gate(content, source, "belief_update", priority)
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
            domain="belief_update",
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
        settings_block = self._runtime_autonomous_action_gate(
            source=source,
            context=None,
            domain="initiative",
        )
        if settings_block is not None:
            return settings_block
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
        settings_block = self._runtime_autonomous_action_gate(
            source=source,
            context=None,
            domain="initiative",
        )
        if settings_block is not None:
            return settings_block
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
            require_substrate=False,
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
            require_substrate=False,
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

    def verify_tool_access(self, tool_name: str, token_id: str | None) -> bool:
        """Legacy opaque-token check.

        This proves only that *some* token exists in this process naming this
        tool. It cannot establish that the token came from the Will — any caller
        that can import ``core.agency.capability_system`` can mint one. Sinks
        must verify ``AuthorityDecision.signed_capability`` instead; this
        remains for callers still on the old contract and as a cheap pre-filter.
        """
        return self._capabilities.verify_access(tool_name, token_id)

    def _mint_signed_capability(
        self,
        will_decision: Any,
        decision: AuthorityDecision,
        *,
        action: str,
        payload: Any = None,
        scope: str = "",
        ttl_s: float = 300.0,
    ) -> dict[str, Any] | None:
        """Mint the signed grant a sink can authenticate.

        Returns None when no capability could be minted. A None here is not a
        soft failure to be ignored: a sink in strict mode refuses to execute
        without a verified capability, so a mint failure fails the action closed
        rather than degrading it into an unauthenticated execution.
        """
        if will_decision is None:
            return None
        try:
            from core.governance.capability_chain import get_capability_issuer

            cap = get_capability_issuer().issue_from_decision(
                will_decision,
                action=action,
                payload=payload,
                scope=scope,
                ttl_s=ttl_s,
            )
            return cap.to_dict()
        except Exception as exc:  # noqa: BLE001 - mint failure must be visible
            record_degradation(
                "authority_gateway",
                exc,
                action=(
                    f"could not mint a signed capability for '{action}' — sinks in "
                    "strict mode will refuse this action"
                ),
                enforce_failure_policy=False,
            )
            logger.error(
                "Capability mint FAILED for '%s': %s", action, exc, exc_info=True
            )
            return None

    def finalize_tool_execution(
        self,
        *,
        executive_intent_id: str | None = None,
        capability_token_id: str | None = None,
        standing_authority_token: str | None = None,
        success: bool = True,
        result: Any = None,
        error: str = "",
    ) -> dict[str, Any]:
        """Close an execution intent and revoke its capability token.

        Callers need a receipt they can use to distinguish "the action ran"
        from "the authority lifecycle closed." Completion failures are still
        recorded as degradations, but they are no longer silently converted to
        an indistinguishable ``None`` result.
        """
        errors: list[str] = []
        intent_closed = executive_intent_id is None
        token_revoked = capability_token_id is None
        with self._standing_lease_lock:
            correlated_standing_token = (
                standing_authority_token
                or self._standing_leases_by_intent.get(str(executive_intent_id or ""))
                or self._standing_leases_by_capability.get(str(capability_token_id or ""))
            )
        standing_authority_closed = correlated_standing_token is None
        if executive_intent_id:
            try:
                self._get_executive_core().complete_intent(executive_intent_id, success=success)
                intent_closed = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('authority_gateway', exc, enforce_failure_policy=False)
                logger.error("Executive intent completion failed: %s", exc, exc_info=True)
                errors.append(f"executive_intent:{type(exc).__name__}:{exc}")
        if capability_token_id:
            try:
                self._capabilities.revoke_token(capability_token_id)
                token_revoked = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('authority_gateway', exc, enforce_failure_policy=False)
                logger.error("Capability token revoke failed: %s", exc, exc_info=True)
                errors.append(f"capability_token:{type(exc).__name__}:{exc}")
        if correlated_standing_token:
            try:
                standing_closure = self._standing_authority.finalize_child_lease(
                    correlated_standing_token,
                    success=success,
                    result=result,
                    error=error,
                )
                standing_authority_closed = bool(standing_closure.get("closed"))
                errors.extend(str(item) for item in standing_closure.get("errors") or [])
                if standing_authority_closed:
                    with self._standing_lease_lock:
                        if executive_intent_id:
                            self._standing_leases_by_intent.pop(executive_intent_id, None)
                        if capability_token_id:
                            self._standing_leases_by_capability.pop(capability_token_id, None)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('authority_gateway', exc, enforce_failure_policy=False)
                logger.error("Standing-authority lease closure failed: %s", exc, exc_info=True)
                errors.append(f"standing_authority:{type(exc).__name__}:{exc}")
        receipt = {
            "closed": intent_closed and token_revoked and standing_authority_closed,
            "mode": "authority_gateway",
            "success": bool(success),
            "intent_closed": intent_closed,
            "token_revoked": token_revoked,
            "standing_authority_closed": standing_authority_closed,
            "errors": errors,
        }
        if not receipt["closed"]:
            # Recorded here rather than at each call site, so a caller that
            # forgets to read the receipt still cannot make the leak vanish.
            self.record_unreconciled_authority(
                executive_intent_id=executive_intent_id,
                capability_token_id=capability_token_id,
                standing_authority_token=correlated_standing_token,
                receipt=receipt,
            )
        return receipt

    def record_unreconciled_authority(
        self,
        *,
        executive_intent_id: str | None = None,
        capability_token_id: str | None = None,
        standing_authority_token: str | None = None,
        receipt: dict[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        """Note an authority lifecycle that did not close.

        Also reachable by callers whose finalize call RAISED, where there is
        no receipt to inspect at all.
        """
        entry = {
            "at": time.time(),
            "executive_intent_id": executive_intent_id,
            # Token identifiers only — never the token material.
            "capability_token_id": capability_token_id,
            "standing_authority_token_present": bool(standing_authority_token),
            "reason": reason or "finalize_incomplete",
            "receipt": dict(receipt or {}),
        }
        with self._unreconciled_lock:
            if len(self._unreconciled) == self._unreconciled.maxlen:
                logger.error(
                    "Unreconciled-authority queue is full (%d); dropping the "
                    "oldest. %d recorded since boot.",
                    UNRECONCILED_QUEUE_LIMIT,
                    self._unreconciled_total,
                )
            self._unreconciled.append(entry)
            self._unreconciled_total += 1
        record_degradation(
            'authority_gateway',
            RuntimeError(
                "authority lifecycle did not close "
                f"(intent={executive_intent_id!r} token={capability_token_id!r} "
                f"reason={entry['reason']})"
            ),
            severity="critical",
            action="queued an unreconciled capability grant for reconciliation",
            enforce_failure_policy=False,
        )

    def unreconciled_authority(self) -> dict[str, Any]:
        """Authority lifecycles known to be open, for health and diagnostics."""
        with self._unreconciled_lock:
            return {
                "open": len(self._unreconciled),
                "total_since_boot": self._unreconciled_total,
                "limit": UNRECONCILED_QUEUE_LIMIT,
                "entries": [dict(entry) for entry in self._unreconciled],
            }

    def _complete_intent_safely(self, intent_id: str | None, *, success: bool = True) -> None:
        if not intent_id:
            return
        try:
            self._get_executive_core().complete_intent(intent_id, success=success)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('authority_gateway', exc, enforce_failure_policy=False)
            logger.error("Executive intent completion failed: %s", exc, exc_info=True)

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
        except (RuntimeError, AttributeError, TypeError):
            return False

    def _canonical_self_version(self) -> int | None:
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
        constraints: dict[str, Any] | None = None,
        executive_intent_id: str | None = None,
        capability_token_id: str | None = None,
        substrate_receipt_id: str | None = None,
        will_receipt_id: str | None = None,
        domain: str | None = None,
        source: str | None = None,
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
        executive_intent_id: str | None = None,
        substrate_constraints: dict[str, Any] | None = None,
        substrate_receipt_id: str | None = None,
        will_receipt_id: str | None = None,
        domain: str | None = None,
        source: str | None = None,
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
        will_receipt_id: str | None = None,
        domain: str | None = None,
    ) -> tuple[AuthorityDecision | None, dict[str, Any], str | None]:
        authority = optional_service("substrate_authority", default=None)
        if authority is None and require_substrate:
            # FAIL-CLOSED: if substrate authority is required but not available,
            # block the action regardless of boot state. No "blind spot" bypass.
            logger.warning(
                "🛡️ Substrate Authority not registered — BLOCKING action (fail-closed). "
                "domain=%s, source=%s", domain or category.name.lower(), source,
            )
            return (
                self._contextualize(
                    approved=False,
                    outcome="rejected",
                    reason=f"substrate_authority_required:{domain or category.name.lower()}",
                    constraints={"blocked": True, "missing": "substrate_authority"},
                    will_receipt_id=will_receipt_id,
                    domain=domain,
                    source=source,
                ),
                {"error": "substrate_authority_required"},
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('authority_gateway', exc, enforce_failure_policy=False)
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
            logger.error("Substrate preflight failed for %s: %s", category.name, exc, exc_info=True)
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

        constraints: dict[str, Any] = {}
        if verdict.decision == AuthorizationDecision.CONSTRAIN:
            constraints["substrate_constrained"] = True
            constraints["substrate_constraints"] = list(verdict.constraints or [])
        elif verdict.decision == AuthorizationDecision.CRITICAL_PASS:
            constraints["substrate_critical_pass"] = True

        return None, constraints, verdict.receipt_id


_instance: AuthorityGateway | None = None


def get_authority_gateway() -> AuthorityGateway:
    global _instance
    if _instance is None:
        _instance = AuthorityGateway()
        try:
            ServiceContainer.register_instance("authority_gateway", _instance, required=False)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('authority_gateway', exc, enforce_failure_policy=False)
            logger.error("AuthorityGateway registration failed: %s", exc, exc_info=True)
    return _instance


def reset_authority_gateway() -> None:
    """Drop process-local gateway state after shutdown or isolated test teardown."""

    global _instance
    _instance = None
