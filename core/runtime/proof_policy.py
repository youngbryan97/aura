"""Runtime policy for proof/evaluation turns.

Proof runs must use the same live runtime path as normal Aura launches while
remaining auditable and isolated. This module centralizes the small set of
proof-specific knobs so individual phases do not quietly diverge.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

_PROOF_ACTIVE_ENV = ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS", "AURA_TESTING")
_PROOF_REPAIR_PREFIX = "Your previous proof/evaluation answer failed validation."
_PROOF_REPAIR_ORIGINAL_TASK_RE = re.compile(
    r"(?:^|\n)Original task:\s*\n(?P<task>.*?)(?:\n\s*\nValidation status:|\Z)",
    re.DOTALL,
)

TRANSIENT_RESPONSE_MODIFIER_KEYS = frozenset(
    {
        "adaptive_effector",
        "adaptive_immune_coverage",
        "adaptive_immune_verdict",
        "adaptive_immune_verification",
        "adaptive_immunity",
        "affective_reasoning_pressure",
        "agency_comparator",
        "anomaly_score",
        "anomaly_threat_level",
        "auto_browse_urls",
        "autonomous_resilience",
        "autonomous_resilience_immune",
        "conv_dynamics_state",
        "conversation_intelligence",
        "conversational_dynamics",
        "credit_assignment",
        "deep_handoff",
        "dialogue_validation",
        "evidence_turn_marker",
        "execution_report",
        "executive_closure",
        "executive_dominant_need",
        "executive_hysteresis",
        "executive_need_pressure",
        "executive_objective",
        "grounded_actions",
        "higher_order_thought",
        "humor_guidance",
        "intent_type",
        "interaction_signals",
        "last_skill_ok",
        "last_skill_objective_hash",
        "last_skill_result_payload",
        "last_skill_run",
        "last_skill_turn_marker",
        "last_task_id",
        "last_task_outcome",
        "last_task_result_payload",
        "matched_skills",
        "metacognitive_strategy",
        "model_tier",
        "multiple_drafts",
        "narrative_context",
        "narrative_gravity",
        "natural_followup",
        "pending_followup",
        "pre_linguistic_decision",
        "precomputed_grounded_reply",
        "prediction_error",
        "proof_model_tier",
        "proof_repair_turn",
        "proof_turn_objective",
        "queued_messages",
        "relational_intelligence",
        "response_contract",
        "semantic_intent",
        "strict_proof_answer_request",
        "structured_proof_solver",
        "system_failure_state",
        "thermal_guard",
        "user_sentiment",
    }
)

_PRIMARY_ALIASES = frozenset({"", "primary", "cortex", "32b", "live", "production"})
_TERTIARY_ALIASES = frozenset({"tertiary", "brainstem", "7b", "fast", "diagnostic"})


def proof_active_env_names() -> tuple[str, ...]:
    """Every environment variable that makes :func:`proof_run_active` true.

    Published because callers kept guessing. Tests that mean "this is not a
    proof run" were clearing AURA_PROOF_RUN alone and inheriting AURA_TESTING
    from the tooling that ran them, so the signal they thought they had
    neutralized was still on and they failed on a deferral that had nothing to
    do with what they were checking. Ask here instead of hardcoding a list that
    goes stale the next time one is added.
    """

    return tuple(_PROOF_ACTIVE_ENV)


def proof_run_active(origin: Any = None) -> bool:
    """Return True when the current turn is part of a proof/eval run."""

    normalized = str(origin or "").strip().lower()
    if normalized in {"test", "proof", "eval", "evaluation"}:
        return True
    return any(os.environ.get(name) for name in _PROOF_ACTIVE_ENV)


_HEADLESS_PROOF_ENV = ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS")
_TRUTHY_ENV = frozenset({"1", "true", "yes", "on"})


def proof_headless_run() -> bool:
    """Return True for headless proof/longevity/battery runs that have no
    user-facing UI surface attached.

    Distinct from :func:`proof_run_active`: this intentionally EXCLUDES
    ``AURA_TESTING`` so unit tests still exercise the live, UI-attached contracts
    (e.g. the visible-conversation zombie-lane guard, which is only meaningful
    when a conversation surface actually exists). Use this strictly to relax
    checks that would be permanent false positives when no UI is attached — never
    to weaken a live-path safety invariant.
    """

    return any(
        str(os.environ.get(name, "") or "").strip().lower() in _TRUTHY_ENV
        for name in _HEADLESS_PROOF_ENV
    )


def active_proof_ablation_services(*, origin: Any = None) -> tuple[str, ...]:
    """Return intentional service lesions that make a proof turn non-proving."""

    if not proof_run_active(origin=origin):
        return ()
    try:
        from core.runtime.ablation_policy import active_ablation_services
    except ImportError:
        return ()
    return tuple(sorted(active_ablation_services()))


def is_strict_proof_answer_prompt(prompt: Any, *, origin: Any = None) -> bool:
    """Detect sealed proof tasks that require a strict ``<answer>`` envelope."""

    return "<answer>" in str(prompt or "").lower() and proof_run_active(origin=origin)


def structured_proof_solver_enabled(*, origin: Any = None) -> bool:
    """Return whether the governed System2 proof reasoner may answer directly.

    This is intentionally separate from strict answer detection. Some validation
    runs need the exact same live proof path while forcing the requested model
    lane to answer without symbolic interception. The reasoner is prompt-derived:
    it may use the task text and internal symbolic procedures, but not task ids,
    fixture answer keys, grader salts, answer hashes, or benchmark lookup tables.
    """

    if not proof_run_active(origin=origin):
        return False
    try:
        from core.runtime.ablation_policy import service_intentionally_lesioned
    # not a failure: without an ablation policy nothing is lesioned, and the
    # else-branch below is the only thing this import feeds.
    except ImportError:
        pass
    else:
        if any(
            service_intentionally_lesioned(service)
            for service in (
                "native_system2",
                "system2_search",
                "structured_proof_solver",
                "proof_answer_solver",
            )
        ):
            return False
    raw = str(os.environ.get("AURA_ENABLE_STRUCTURED_PROOF_SOLVER", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def mlx_strict_answer_contract_enabled(*, origin: Any = None) -> bool:
    """Return whether the MLX worker should use its strict-answer micro-prompt.

    The response phase still enforces the final ``<answer>`` envelope when this
    is disabled; this only controls the low-level worker prompting strategy.
    """

    if not proof_run_active(origin=origin):
        return True
    raw = str(os.environ.get("AURA_DISABLE_MLX_STRICT_ANSWER_CONTRACT", "") or "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def is_proof_evaluation_purpose(purpose: Any) -> bool:
    """Return True for non-atomic proof/eval generation lanes."""

    normalized = str(purpose or "").strip().lower()
    return normalized in {"proof_evaluation", "proof_evaluation_repair"}


def is_proof_repair_prompt(prompt: Any, *, origin: Any = None) -> bool:
    """Detect internal proof repair prompts that must not become durable goals."""

    text = str(prompt or "").strip()
    return (
        bool(text)
        and proof_run_active(origin=origin)
        and text.startswith(_PROOF_REPAIR_PREFIX)
        and "Original task:" in text
        and "Validation status:" in text
    )


def extract_original_task_from_proof_repair_prompt(prompt: Any) -> str:
    """Extract the user-facing task from an internal proof repair prompt."""

    match = _PROOF_REPAIR_ORIGINAL_TASK_RE.search(str(prompt or ""))
    if not match:
        return ""
    return " ".join(match.group("task").strip().split())


def proof_persistent_objective(prompt: Any, *, origin: Any = None) -> str:
    """Return the durable objective represented by a proof/evaluation prompt.

    Repair prompts are runtime scaffolding. They should steer one generation
    attempt, but they must not be committed as Aura's active goal, memory focus,
    or future continuity anchor.
    """

    text = str(prompt or "")
    if is_proof_repair_prompt(text, origin=origin):
        original = extract_original_task_from_proof_repair_prompt(text)
        if original:
            return original
    return text


def proof_model_tier(default: str = "primary") -> str:
    """Resolve the LLM lane for proof tasks.

    The default is the production 32B Cortex lane. A fast 7B lane is still
    available for diagnostic isolation, but it must be requested explicitly via
    ``AURA_PROOF_MODEL_TIER=tertiary`` or the DNU runner's ``--model-tier`` flag.
    """

    raw = str(os.environ.get("AURA_PROOF_MODEL_TIER", default) or default).strip().lower()
    if raw in _TERTIARY_ALIASES:
        return "tertiary"
    if raw in _PRIMARY_ALIASES:
        return "primary"
    return "primary"


#: How deep a nested reading may be before it stops being one. Two levels
#: covers a metric and a mapping of metrics; below that a structure is carrying
#: something other than a measurement.
_READING_DEPTH = 2


def _is_a_reading(value: Any, depth: int = 0) -> bool:
    """A number she measured, rather than words that could be read as a directive.

    The separation a proof run needs is between what she found out about
    herself last turn and what somebody told her last turn. Text is the second
    kind: a rendered claim, a tool result, an open thread. A finite number is
    the first, and so is a mapping or sequence of them. Nothing else is either,
    so nothing else is carried.
    """
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if depth >= _READING_DEPTH:
        return False
    if isinstance(value, Mapping):
        return bool(value) and all(
            isinstance(key, str) and _is_a_reading(item, depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value) and all(_is_a_reading(item, depth + 1) for item in value)
    return False


def clear_transient_response_modifiers(modifiers: Any, *, strict: bool = False) -> None:
    """Remove per-turn prompt/runtime modifiers before a new turn starts.

    AuraState derives by copying ``response_modifiers`` so downstream phases can
    see the previous transition. That is correct for durable state summaries but
    unsafe for turn-local prompt directives: an old open conversational thread,
    tool result, or recovery flag can become an instruction on the next task.
    Normal live turns clear the known per-turn keys. Proof/eval turns clear
    those and then every value that is not a reading.

    ``strict`` used to empty the dict. Every campaign the battery has ever run
    is a proof run, so that cleared her measurements as well as her directives:
    the developmental novelty her body reads, phi and the three autonomy gates
    it sets, the workspace's ignition, the two voices' agreement. Each was
    written every turn and gone before the next one started, which left the
    stale snapshot a background task happened to be holding as the only thing
    that crossed a turn boundary at all.
    """

    if not isinstance(modifiers, dict):
        return
    for key in TRANSIENT_RESPONSE_MODIFIER_KEYS:
        modifiers.pop(key, None)
    if not strict:
        return
    for key in [name for name, value in modifiers.items() if not _is_a_reading(value)]:
        modifiers.pop(key, None)
