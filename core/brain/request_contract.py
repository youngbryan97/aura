"""A typed boundary for the inference request context.

CP126 raised three criticals against ``core/brain/inference_gate.py`` that
are one defect seen from three angles:

* "Unauthenticated context flags control proof, foreground, and model
  -tier policy. Plain context values select benchmark and proof status,
  protected foreground, deep handoff, cognitive-engine
  requirements, and requested tier."
* "Caller context is copied into provider policy and proof kwargs without
  validation. No per-key type schema, authority source, unknown-field
  rejection."
* "The public think interface forwards policy-sensitive kwargs without
  authority validation."

The most immediately dangerous half is not the missing authority model —
it is the missing *types*, because Python's truthiness makes the failure
silent and inverted::

    context["deep_handoff"] = "false"   # a string, from a config
    if context.get("deep_handoff"):     # → True
        ...                             # expensive lane enabled

Every policy flag in this context is read with a bare truthiness test, so
``"false"``, ``"no"``, ``"off"`` and ``"0"`` all turn the flag ON. A caller
disabling a deep handoff, or clearing a proof requirement, gets exactly the
opposite of what it asked for, and nothing anywhere reports it.

So this declares what each key is, coerces what can be coerced honestly,
and **rejects rather than guesses** — a value that cannot be read as its
declared type falls back to the field's default and is recorded as a
rejection. Silence is what made the original bug survivable; a rejection
list is what makes it findable.

Strict booleans, specifically. ``bool(value)`` is banned here. A string is
true only if it says so — ``1/true/yes/on`` — false only if it says so —
``0/false/no/off`` — and anything else is a rejection, not a coin flip.

This deliberately stops at types and domains. Binding a request to an
authenticated principal is a larger change to how callers reach the gate;
the field table below carries a ``policy`` marker on each key that steers
proof, tier or foreground behaviour, so the set that will need an
authority check is enumerated rather than rediscovered.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.runtime.numeric_guards import bounded_float, bounded_int, is_finite_number

logger = logging.getLogger(__name__)

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off", "n", "f"})

#: Tiers the gate actually understands, after ``_normalize_tier`` aliasing.
TIER_ALIASES: dict[str, str] = {
    "local": "primary",
    "local_deep": "secondary",
    "local_fast": "tertiary",
    "fast": "tertiary",
    "deep": "secondary",
}
KNOWN_TIERS: frozenset[str] = frozenset(
    {"primary", "secondary", "tertiary", "api_fast", "api_deep"}
)

# Both state-level CognitiveMode values and CognitiveEngine ThinkingMode values
# cross the inference boundary. They describe computation Aura already
# selected; they are never inferred again from prompt words at the model seam.
KNOWN_COGNITIVE_MODES: frozenset[str] = frozenset(
    {
        "reactive",
        "deliberate",
        "dreaming",
        "dormant",
        "fast",
        "quick",
        "slow",
        "deep",
        "reflective",
        "critical",
        "creative",
    }
)


class Kind:
    BOOL = "bool"
    UNIT_FLOAT = "unit_float"       # [0, 1]
    POSITIVE_FLOAT = "positive_float"
    FLOAT = "float"
    POSITIVE_INT = "positive_int"
    STRING = "string"
    TIER = "tier"
    COGNITIVE_MODE = "cognitive_mode"
    STRING_LIST = "string_list"
    MAPPING = "mapping"
    SEQUENCE = "sequence"
    OPAQUE = "opaque"               # structured payloads validated elsewhere


@dataclass(frozen=True)
class Field_:
    kind: str
    #: True when this key steers proof, tier, foreground or tool
    #: policy — i.e. the set that a future authority check must cover.
    policy: bool = False
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class ContextValidation:
    """The clean context, plus everything that did not survive."""

    context: dict[str, Any] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.rejected and not self.unknown

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejected": dict(self.rejected),
            "unknown": sorted(self.unknown),
        }


#: Every key the gate accepts, and what it is.
#:
#: Adding a key here is the point of contact for anyone extending the
#: request surface: a key that is not declared is reported as unknown rather
#: than quietly forwarded into provider kwargs.
REQUEST_FIELDS: dict[str, Field_] = {
    # ── conversation payload ────────────────────────────────────────────
    "messages": Field_(Kind.SEQUENCE),
    "history": Field_(Kind.SEQUENCE),
    "brief": Field_(Kind.STRING),
    "visible_user_message": Field_(Kind.STRING),
    "current_user_message": Field_(Kind.STRING),
    "recent_conversation_context": Field_(Kind.OPAQUE),
    "user_surface_validation_prompt": Field_(Kind.STRING),
    "user_surface_prompt_binding": Field_(Kind.OPAQUE),
    "user_surface_grounding_evidence": Field_(Kind.SEQUENCE),
    "turn_sensory_evidence": Field_(Kind.OPAQUE),
    # ── sampling ────────────────────────────────────────────────────────
    "max_tokens": Field_(Kind.POSITIVE_INT, minimum=1, maximum=1_000_000),
    # A user-surface completion floor is distinct from the requested cap.
    # Adaptive pressure may lower ordinary generation budgets, but it must not
    # silently turn a complete foreground answer into a clipped one.  Declaring
    # it here keeps the value typed across InferenceGate.think() rather than
    # dropping it as an unknown provider kwarg.
    "user_surface_completion_floor": Field_(
        Kind.POSITIVE_INT, minimum=1, maximum=1_000_000
    ),
    "temperature": Field_(Kind.FLOAT, minimum=0.0, maximum=2.0),
    "temp": Field_(Kind.FLOAT, minimum=0.0, maximum=2.0),
    "top_p": Field_(Kind.UNIT_FLOAT),
    "min_p": Field_(Kind.UNIT_FLOAT),
    "top_k": Field_(Kind.POSITIVE_INT, minimum=1, maximum=1_000_000),
    "repetition_penalty": Field_(Kind.POSITIVE_FLOAT, maximum=10.0),
    "repetition_context_size": Field_(Kind.POSITIVE_INT, minimum=1, maximum=1_000_000),
    "presence_penalty": Field_(Kind.FLOAT, minimum=-2.0, maximum=2.0),
    "stop_sequences": Field_(Kind.STRING_LIST),
    "schema": Field_(Kind.OPAQUE),
    "sampling_bias": Field_(Kind.OPAQUE),
    "imagination_sampling_bias": Field_(Kind.OPAQUE),
    "bicameral_sampling_bias": Field_(Kind.OPAQUE),
    "clean_user_surface_steering_alpha": Field_(Kind.UNIT_FLOAT),
    "clean_user_surface_recurrent_loops": Field_(Kind.POSITIVE_INT, minimum=1, maximum=64),
    # ── routing and lane policy (authority-relevant) ────────────────────
    "prefer_tier": Field_(Kind.TIER, policy=True),
    "proof_model_tier": Field_(Kind.TIER, policy=True),
    "deep_handoff": Field_(Kind.BOOL, policy=True),
    # Source compatibility for older callers. Validation forces this false;
    # no remote model-provider registration or dispatch path exists.
    "allow_cloud_fallback": Field_(Kind.BOOL),
    "allow_mesh_cognition": Field_(Kind.BOOL, policy=True),
    "allow_tools": Field_(Kind.BOOL, policy=True),
    "is_background": Field_(Kind.BOOL, policy=True),
    "foreground_request": Field_(Kind.BOOL, policy=True),
    # A planner that runs as part of the turn in progress. Unlike
    # foreground_request this is a request rather than an assertion: the gate
    # grants it only while the orchestrator reports a live foreground turn,
    # so outside a turn it buys nothing.
    "serves_current_turn": Field_(Kind.BOOL, policy=True),
    "protected_foreground_lane": Field_(Kind.BOOL, policy=True),
    "benchmark_request": Field_(Kind.BOOL, policy=True),
    "health_probe": Field_(Kind.BOOL, policy=True),
    "cognitive_engine_required": Field_(Kind.BOOL, policy=True),
    "desktop_cognitive_engine_required": Field_(Kind.BOOL, policy=True),
    "live_runtime_payload_required": Field_(Kind.BOOL, policy=True),
    "skip_runtime_payload": Field_(Kind.BOOL, policy=True),
    "disable_prompt_cache": Field_(Kind.BOOL, policy=True),
    "clear_prompt_cache": Field_(Kind.BOOL, policy=True),
    "recent_context_needed": Field_(Kind.BOOL),
    "origin": Field_(Kind.STRING),
    "purpose": Field_(Kind.STRING),
    "cognitive_mode": Field_(Kind.COGNITIVE_MODE),
    # ── proof and output contracts (authority-relevant) ─────────────────
    "proof_primary_lane_required": Field_(Kind.BOOL, policy=True),
    "strict_answer_contract": Field_(Kind.BOOL, policy=True),
    "strict_value_contract": Field_(Kind.BOOL, policy=True),
    "proof_evaluation_contract": Field_(Kind.BOOL, policy=True),
    "operator_evidence_contract": Field_(Kind.BOOL, policy=True),
    "web_interlocutor_contract": Field_(Kind.BOOL, policy=True),
    "desktop_execution_contract": Field_(Kind.BOOL, policy=True),
    "desktop_quick_reply_contract": Field_(Kind.BOOL, policy=True),
    "capability_inventory_contract": Field_(Kind.BOOL, policy=True),
    "memory_state_contract": Field_(Kind.BOOL, policy=True),
    "runtime_fact_status_contract": Field_(Kind.BOOL, policy=True),
    "grounded_runtime_status_contract": Field_(Kind.BOOL, policy=True),
    "self_condition_contract": Field_(Kind.BOOL, policy=True),
    "clean_user_surface_contract": Field_(Kind.BOOL, policy=True),
    "canonical_memory_state_evidence": Field_(Kind.OPAQUE),
    "response_style_contract": Field_(Kind.OPAQUE),
    "live_speech_grounding_frame": Field_(Kind.OPAQUE),
    "user_surface_completion_retry": Field_(Kind.BOOL),
    "user_surface_continuation_contract": Field_(Kind.BOOL),
    "user_surface_continuation_partial": Field_(Kind.STRING),
    "semantic_completion_contract": Field_(Kind.BOOL),
    # ── live-mind readiness claims (authority-relevant) ─────────────────
    "live_mind_controls_bound": Field_(Kind.BOOL, policy=True),
    "live_mind_generation_controls": Field_(Kind.OPAQUE),
    "live_mind_snapshot_ready": Field_(Kind.BOOL, policy=True),
    "live_mind_required_subsystems_ok": Field_(Kind.BOOL, policy=True),
    # ── misc ────────────────────────────────────────────────────────────
    "state": Field_(Kind.OPAQUE),
}

#: The keys that select proof, tier, foreground or tool behaviour.
#: Enumerated rather than rediscovered, so the authority work that follows
#: has a fixed target.
POLICY_FIELDS: frozenset[str] = frozenset(
    name for name, spec in REQUEST_FIELDS.items() if spec.policy
)


def strict_bool(value: Any) -> bool | None:
    """A boolean, or None when the value does not state one.

    ``bool("false")`` is True, which is how a caller disabling a policy
    option could enable it. Nothing here falls back to truthiness.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and is_finite_number(value):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    return None


def normalize_tier(value: Any) -> str | None:
    """Canonical tier name, or None when it is not one."""
    if not isinstance(value, str):
        return None
    tier = value.strip().lower()
    if not tier:
        return None
    tier = TIER_ALIASES.get(tier, tier)
    return tier if tier in KNOWN_TIERS else None


def validate_request_context(
    raw: Mapping[str, Any] | None,
    *,
    on_rejection: Callable[[str, Any, str], None] | None = None,
) -> ContextValidation:
    """Type-check an inference request context.

    Returns the subset that survived, plus what did not and why. A rejected
    key is DROPPED rather than defaulted: the gate's own defaults are
    engineered, and substituting a guess for a caller's malformed value is
    how the original bug behaved.
    """
    result = ContextValidation()
    for key, value in (raw or {}).items():
        spec = REQUEST_FIELDS.get(key)
        if spec is None:
            result.unknown.append(key)
            continue
        coerced, problem = _coerce(spec, value)
        if problem:
            result.rejected[key] = problem
            if on_rejection is not None:
                on_rejection(key, value, problem)
            continue
        if key == "allow_cloud_fallback":
            coerced = False
        result.context[key] = coerced
    if result.rejected or result.unknown:
        logger.warning(
            "inference request context: %d rejected (%s), %d unknown (%s)",
            len(result.rejected),
            ", ".join(sorted(result.rejected)) or "-",
            len(result.unknown),
            ", ".join(sorted(result.unknown)) or "-",
        )
    return result


def _coerce(spec: Field_, value: Any) -> tuple[Any, str]:
    kind = spec.kind
    if kind == Kind.BOOL:
        resolved = strict_bool(value)
        if resolved is None:
            return None, f"not a boolean: {type(value).__name__}={value!r:.40}"
        return resolved, ""
    if kind == Kind.TIER:
        resolved_tier = normalize_tier(value)
        if resolved_tier is None:
            return None, f"unknown tier: {value!r:.40}"
        return resolved_tier, ""
    if kind == Kind.COGNITIVE_MODE:
        if not isinstance(value, str):
            return None, f"not a cognitive mode string: {type(value).__name__}"
        resolved_mode = value.strip().lower()
        if resolved_mode not in KNOWN_COGNITIVE_MODES:
            return None, f"unknown cognitive mode: {value!r:.40}"
        return resolved_mode, ""
    if kind in (Kind.FLOAT, Kind.UNIT_FLOAT, Kind.POSITIVE_FLOAT):
        if not is_finite_number(value) and not _numeric_string(value):
            return None, f"not a finite number: {value!r:.40}"
        minimum = 0.0 if kind == Kind.UNIT_FLOAT else spec.minimum
        maximum = 1.0 if kind == Kind.UNIT_FLOAT else spec.maximum
        # bounded_float's contract is "give me a usable number"; here the
        # answer "this is not a number" must survive rather than become a
        # default, so finiteness is settled before clamping.
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None, f"not a finite number: {value!r:.40}"
        if not is_finite_number(candidate):
            return None, f"not a finite number: {value!r:.40}"
        resolved = bounded_float(
            candidate, default=candidate, minimum=minimum, maximum=maximum
        )
        if kind == Kind.POSITIVE_FLOAT and resolved <= 0.0:
            return None, f"must be positive: {value!r:.40}"
        return resolved, ""
    if kind == Kind.POSITIVE_INT:
        if isinstance(value, bool) or not (is_finite_number(value) or _numeric_string(value)):
            return None, f"not an integer: {value!r:.40}"
        # Clamping is wrong here. max_tokens=-5 clamped to 1 yields a
        # one-token answer; the gate's engineered default is a far better
        # response to a malformed value than a technically-in-range one.
        raw_int = bounded_int(value, default=0)
        if raw_int <= 0 or (spec.minimum is not None and raw_int < int(spec.minimum)):
            return None, f"below the allowed minimum: {value!r:.40}"
        if spec.maximum is not None and raw_int > int(spec.maximum):
            return None, f"above the allowed maximum: {value!r:.40}"
        return raw_int, ""
    if kind == Kind.STRING:
        if not isinstance(value, str):
            return None, f"not a string: {type(value).__name__}"
        return value, ""
    if kind == Kind.STRING_LIST:
        if isinstance(value, str):
            return [value], ""
        if not isinstance(value, (list, tuple)):
            return None, f"not a sequence: {type(value).__name__}"
        if not all(isinstance(item, str) for item in value):
            return None, "sequence contains non-string entries"
        return list(value), ""
    if kind == Kind.MAPPING:
        if not isinstance(value, Mapping):
            return None, f"not a mapping: {type(value).__name__}"
        return dict(value), ""
    if kind == Kind.SEQUENCE:
        if not isinstance(value, (list, tuple)):
            return None, f"not a sequence: {type(value).__name__}"
        return list(value), ""
    return value, ""


def _numeric_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "KNOWN_COGNITIVE_MODES",
    "KNOWN_TIERS",
    "POLICY_FIELDS",
    "REQUEST_FIELDS",
    "TIER_ALIASES",
    "ContextValidation",
    "Field_",
    "Kind",
    "normalize_tier",
    "strict_bool",
    "validate_request_context",
]
