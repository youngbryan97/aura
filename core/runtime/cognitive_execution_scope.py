"""Per-objective authority to cross from cognition into external execution.

Tool effect scopes describe what a tool can change. This contract answers an
earlier question: whether the cognitive request being processed is an action
request at all. Keeping those concepts separate prevents generated prose from
acquiring authority merely because it happens to resemble a command.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from core.runtime.background_policy import is_user_facing_origin


class CognitiveExecutionScope(StrEnum):
    REASONING_ONLY = "reasoning_only"
    GOVERNED_ACTIONS = "governed_actions"


_BINDING_KEY = "cognitive_execution_binding"
_CONTEXT_KEY = "cognitive_execution_scope"
_EXPLICIT_ACTION_CONTRACTS = frozenset(
    {
        "desktop_execution_contract",
        "program_dna_execution_contract",
        "rsi_execution_contract",
        "web_interlocutor_execution_contract",
    }
)


def objective_digest(objective: Any) -> str:
    normalized = " ".join(str(objective or "").split())
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def resolve_cognitive_execution_scope(
    *,
    origin: Any,
    context: Mapping[str, Any] | None,
) -> CognitiveExecutionScope:
    """Resolve execution eligibility from typed provenance, never prompt text."""

    request_context = context if isinstance(context, Mapping) else {}
    explicit = str(request_context.get(_CONTEXT_KEY) or "").strip().lower()
    if explicit:
        try:
            return CognitiveExecutionScope(explicit)
        except ValueError:
            # An unknown authority value never widens execution.
            return CognitiveExecutionScope.REASONING_ONLY

    if is_user_facing_origin(origin):
        return CognitiveExecutionScope.GOVERNED_ACTIONS

    if any(bool(request_context.get(key)) for key in _EXPLICIT_ACTION_CONTRACTS):
        return CognitiveExecutionScope.GOVERNED_ACTIONS

    if (
        bool(request_context.get("autonomous"))
        and str(request_context.get("authorization") or "").strip()
        == "governed_autonomous_overt_action"
        and bool(str(request_context.get("requested_authority_scope") or "").strip())
    ):
        return CognitiveExecutionScope.GOVERNED_ACTIONS

    return CognitiveExecutionScope.REASONING_ONLY


def bind_cognitive_execution_scope(
    state: Any,
    objective: Any,
    scope: CognitiveExecutionScope | str,
    *,
    source: str,
) -> dict[str, str]:
    resolved = CognitiveExecutionScope(str(scope))
    binding = {
        "scope": resolved.value,
        "objective_digest": objective_digest(objective),
        "source": str(source or "unknown"),
    }
    modifiers = getattr(state, "response_modifiers", None)
    if not isinstance(modifiers, dict):
        modifiers = {}
        state.response_modifiers = modifiers
    modifiers[_BINDING_KEY] = binding
    return dict(binding)


def bound_cognitive_execution_scope(
    state: Any,
    objective: Any,
) -> CognitiveExecutionScope | None:
    modifiers = getattr(state, "response_modifiers", None)
    if not isinstance(modifiers, Mapping):
        return None
    binding = modifiers.get(_BINDING_KEY)
    if not isinstance(binding, Mapping):
        return None
    digest = objective_digest(objective)
    if not digest or str(binding.get("objective_digest") or "") != digest:
        return None
    try:
        return CognitiveExecutionScope(str(binding.get("scope") or ""))
    except ValueError:
        return CognitiveExecutionScope.REASONING_ONLY


def cognitive_request_allows_actions(state: Any, objective: Any) -> bool:
    scope = bound_cognitive_execution_scope(state, objective)
    return scope is not CognitiveExecutionScope.REASONING_ONLY


__all__ = [
    "CognitiveExecutionScope",
    "bind_cognitive_execution_scope",
    "bound_cognitive_execution_scope",
    "cognitive_request_allows_actions",
    "objective_digest",
    "resolve_cognitive_execution_scope",
]
