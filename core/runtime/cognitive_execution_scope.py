"""Per-objective authority to cross from cognition into external execution.

Tool effect scopes describe what a tool can change. This contract answers an
earlier question: whether the cognitive request being processed is an action
request at all. Keeping those concepts separate prevents generated prose from
acquiring authority merely because it happens to resemble a command.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from core.runtime.background_policy import is_user_facing_origin
from core.runtime.lockdep import checked_lock


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


#: Each recent objective bound reasoning-only, by its digest, so that scope travels with the objective's text.
#: LIVE 15-16 November 2026 her mind tick took the dream journal's prompt, left as her current objective, and ran it
#: through a kernel tick of its own whose state carried no binding. Her skill phase read no binding as no
#: objection and dispatched build_document and code_repl for a dream: 16 dispatches of skills nobody asked for,
#: 10 of them refused by their own parameter checks. The cap bounds memory only; an objective re-run is a recent one.
#: Only the narrower scope is remembered, so this can refuse what was allowed before and never allow what was not.
_BOUND_BY_DIGEST: OrderedDict[str, CognitiveExecutionScope] = OrderedDict()
_REMEMBERED = 256
_BOUND_LOCK = checked_lock("cognitive_execution_scope.bound_by_digest")


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
    digest = binding["objective_digest"]
    if digest:
        with _BOUND_LOCK:
            if resolved is CognitiveExecutionScope.REASONING_ONLY:
                _BOUND_BY_DIGEST[digest] = resolved
                _BOUND_BY_DIGEST.move_to_end(digest)
                while len(_BOUND_BY_DIGEST) > _REMEMBERED:
                    _BOUND_BY_DIGEST.popitem(last=False)
            else:
                _BOUND_BY_DIGEST.pop(digest, None)
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
    """The scope bound to this objective: on this state, or where its own cycle bound it."""
    digest = objective_digest(objective)
    modifiers = getattr(state, "response_modifiers", None)
    binding = modifiers.get(_BINDING_KEY) if isinstance(modifiers, Mapping) else None
    if not digest:
        return None
    if not isinstance(binding, Mapping) or str(binding.get("objective_digest") or "") != digest:
        with _BOUND_LOCK:
            return _BOUND_BY_DIGEST.get(digest)
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
