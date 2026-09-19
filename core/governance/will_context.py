"""Whether a context is observation only, hygiene, or consequential.

Lifted whole out of `will`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .will import (
        ActionDomain,
    )


class _ReadsTheContext:
    """Lifted whole out of UnifiedWill; see will.py."""

    @staticmethod
    def _is_consequential_domain(domain: ActionDomain) -> bool:
        from .will import (
            ActionDomain,
        )

        return domain in {
            ActionDomain.TOOL_EXECUTION,
            ActionDomain.MEMORY_WRITE,
            ActionDomain.STATE_MUTATION,
            ActionDomain.INITIATIVE,
            ActionDomain.EXPLORATION,
            ActionDomain.SEMANTIC_WEIGHT_UPDATE,
            ActionDomain.BELIEF_UPDATE,
            ActionDomain.ENVIRONMENT_ACTION,
            ActionDomain.EXTERNAL_ACTION,
            ActionDomain.FILE_WRITE,
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CLOUD_FALLBACK,
            ActionDomain.CI_CD,
            ActionDomain.SELF_MODIFICATION,
        }

    @staticmethod
    def _is_observation_only_tool_context(content: str, context: dict[str, Any]) -> bool:
        ctx = dict(context or {})
        tool_name = str(ctx.get("tool") or ctx.get("skill") or "").strip().lower()
        payload = str(content or "").strip().lower()
        if not tool_name and payload.startswith("tool:"):
            tool_name = payload.split(":", 1)[1].split()[0].strip()

        effect_scope = str(ctx.get("effect_scope") or "").strip().lower()
        read_only = bool(ctx.get("read_only")) or effect_scope == "read_only"
        observation_tools = {
            "clock",
            "environment_info",
            "system_proprioception",
            "query_beliefs",
        }
        prohibited_markers = {
            "external_action",
            "public_action",
            "social_action",
            "world_affecting",
            "file_write",
            "network_call",
            "desktop_control",
            "self_modification",
        }
        if any(bool(ctx.get(marker)) for marker in prohibited_markers):
            return False
        return bool(read_only or tool_name in observation_tools)

    @classmethod
    def _is_observation_only_action_context(
        cls,
        domain: ActionDomain,
        content: str,
        context: dict[str, Any],
    ) -> bool:
        from .will import (
            ActionDomain,
        )

        if domain == ActionDomain.TOOL_EXECUTION:
            return cls._is_observation_only_tool_context(content, context)
        if domain != ActionDomain.ENVIRONMENT_ACTION:
            return False
        try:
            from core.being.runtime import is_runtime_bound_passive_observation

            return is_runtime_bound_passive_observation(domain.value, context)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _is_observation_only_memory_context(content: str, context: dict[str, Any]) -> bool:
        """Allow bounded user-provided memory observations during present-state defer.

        This lane is for commitments like "remember this phrase" where the user
        supplied the fact and the runtime is only preserving it with provenance.
        It is not a bypass for belief, identity, policy, or self-model mutation.
        """

        ctx = dict(context or {})
        metadata = ctx.get("memory_metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        source = str(ctx.get("memory_source") or ctx.get("source") or "").strip().lower().replace("-", "_")
        provenance = str(metadata.get("provenance_source") or metadata.get("source") or "").strip().lower().replace("-", "_")
        user_facing = bool(ctx.get("user_facing_memory_write")) or source in {
            "api",
            "chat",
            "chat_api",
            "desktop",
            "desktop_ui",
            "live_chat",
            "session_memory_pin",
            "ui",
            "user",
            "voice",
            "web_ui",
        }
        explicit = bool(
            ctx.get("explicit_observational_memory_write")
            or metadata.get("explicit_memory_request")
            or metadata.get("session_memory_pin")
            or provenance in {"user", "user_explicit"}
        )
        high_risk = bool(ctx.get("high_risk_memory_write"))
        high_risk_markers = {
            "belief_update",
            "identity_rewrite",
            "self_model_write",
            "policy_change",
            "constitutional_change",
            "governance_change",
        }
        if high_risk or any(bool(metadata.get(marker)) for marker in high_risk_markers):
            return False

        content_len = len(str(content or ""))
        source_utterance_len = len(str(metadata.get("source_utterance") or metadata.get("objective") or ""))
        bounded = content_len <= 1200 and source_utterance_len <= 1200
        return bool(user_facing and explicit and bounded)

    @staticmethod
    def _is_internal_state_hygiene_context(context: dict[str, Any]) -> bool:
        """Allow bounded state bookkeeping during present-state recovery.

        This lane is deliberately narrower than general state mutation. It is
        for canonical internal continuity/proof/shutdown checkpoints that keep
        the runtime coherent and auditable; it does not authorize external
        effects, value edits, memory writes, tools, or self-modification.
        """

        ctx = dict(context or {})
        if not bool(ctx.get("internal_state_hygiene")):
            return False
        prohibited_markers = {
            "external_action",
            "public_action",
            "social_action",
            "world_affecting",
            "file_write",
            "network_call",
            "desktop_control",
            "memory_write",
            "belief_update",
            "identity_rewrite",
            "policy_change",
            "constitutional_change",
            "self_modification",
        }
        if any(bool(ctx.get(marker)) for marker in prohibited_markers):
            return False
        return bool(
            ctx.get("foreground_continuity_state")
            or ctx.get("proof_isolation_state")
            or ctx.get("response_state_checkpoint")
            or ctx.get("shutdown_state_checkpoint")
        )

