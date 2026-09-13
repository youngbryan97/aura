"""Whether a memory write may happen, and on whose word.

A memory write is the one action that changes what she will believe tomorrow,
so the source matters as much as the content: a write whose origin is her own
reasoning is not the same as one a person asked for, and a high-risk write is
not the same as a note. Each check names the binding it authorised under, which
is what makes an unreconciled write findable afterwards.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .authority_gateway import AuthorityDecision

import hashlib
import json
from typing import Any

from core.consciousness.substrate_authority import (
    ActionCategory,
)
from core.executive.executive_core import (
    ActionType,
    Intent,
    IntentSource,
    _coerce_intent_source,
    _is_autonomous_research_source,
)


class _AuthorisesAMemoryWrite:
    """Lifted whole from AuthorityGateway; see authority_gateway.py."""

    @classmethod
    def _memory_source_is_user_facing(cls, value: Any) -> bool:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .authority_gateway import (
            _USER_FACING_MEMORY_ORIGINS,
            _normalized_memory_source,
        )

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
        from .authority_gateway import (
            _HIGH_RISK_MEMORY_MARKERS,
            _normalized_memory_source,
        )

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
        from .authority_gateway import (
            _CONVERSATION_MEMORY_PRODUCER_TYPES,
            _CONVERSATION_MEMORY_PRODUCERS,
            _EXPLICIT_MEMORY_PRODUCERS,
            _INTERNAL_EVIDENCE_MEMORY_PRODUCERS,
            _normalized_memory_source,
        )

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
        from .authority_gateway import (
            _CONVERSATION_MEMORY_PRODUCERS,
            _CONVERSATION_MEMORY_TYPES,
            _normalized_memory_source,
        )

        source_l = _normalized_memory_source(source)
        direct_source = _coerce_intent_source(source_l or "system")
        payload = dict(metadata or {})
        memory_type_l = _normalized_memory_source(memory_type)
        # A high-risk write is not the person's just because a user-facing
        # producer logged it.
        #
        # `chat_turn_logger` coerces to USER, and this returned on that before
        # anything looked at what was being written, so a belief update — a
        # change to what she believes — was waved through as the person's own
        # intent on any conversation turn. The person said something; they did
        # not author the belief, and the governance that exists for these
        # writes is exactly the governance the shortcut skipped.
        if direct_source == IntentSource.USER and not cls._memory_write_is_high_risk(
            memory_type_l, payload
        ):
            return direct_source
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
            # Governed, never the person's. See the shortcut above.
            return (
                IntentSource.AUTONOMOUS
                if direct_source == IntentSource.USER
                else direct_source
            )

        producer_is_conversation = bool(
            source_l in _CONVERSATION_MEMORY_PRODUCERS
            or memory_type_l in _CONVERSATION_MEMORY_TYPES
            or payload.get("conversation_lane") is True
            or str(payload.get("turn_type") or "").strip().lower() == "conversation"
        )
        if producer_is_conversation and cls._memory_payload_origin_is_user_facing(payload):
            return IntentSource.USER
        return direct_source

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
