import copy
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.brain.live_mind_contract import (
    append_text_mutation,
    normalize_text_mutations,
    summarize_text_mutation_authorship,
)
from core.brain.llm.latent_cortex.action_state_capture import (
    UnknownActionStateApplicationError,
)
from core.brain.llm.token_budget_evidence import CALIBRATION_SCHEMA
from core.brain.llm.user_surface_recurrence import (
    admit_user_surface_recurrent_loops,
    user_surface_recurrent_ceiling,
)
from core.conversation.user_surface_contract import (
    UserSurfacePromptResolution,
    resolve_user_surface_prompt,
)
from core.runtime.desktop_boot_safety import compute_mlx_cache_limit, compute_mlx_memory_limit
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind as _FlagKind
from core.runtime.flags import declare as _declare_flag
from core.runtime.model_layers import resolve_model_layers
from core.runtime.state_ownership import shared_asset_root, state_root

from .model_registry import resolve_personality_adapter

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
_FLAG_ALLOW_UNSAFE_MEMORY_LIMITS = _declare_flag(
    "AURA_ALLOW_UNSAFE_MEMORY_LIMITS",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_CONTRASTIVE_ALPHA = _declare_flag(
    "AURA_CONTRASTIVE_ALPHA",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_CONTRASTIVE_BETA = _declare_flag(
    "AURA_CONTRASTIVE_BETA",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_EXPERT_ADAPTER_ROOTS = _declare_flag(
    "AURA_EXPERT_ADAPTER_ROOTS",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_MLX_NUM_THREADS = _declare_flag(
    "AURA_MLX_NUM_THREADS",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_MLX_WORKER_RSS_LIMIT_GB = _declare_flag(
    "AURA_MLX_WORKER_RSS_LIMIT_GB",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_UNIFIED_RECURRENT_SHADOW_PACKAGE = _declare_flag(
    "AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE",
    kind=_FlagKind.STRING,
    default="",
    description=(
        "Absolute path to a certified unified-recurrence shadow package; "
        "the loaded tissue has no response-serving authority."
    ),
    owner="unified-recurrent-shadow",
)
_FLAG_REASONING_STEERING = _declare_flag(
    "AURA_REASONING_STEERING",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_REASONING_STEERING_SCALE = _declare_flag(
    "AURA_REASONING_STEERING_SCALE",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_RECURRENT_LOOPS = _declare_flag(
    "AURA_RECURRENT_LOOPS",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_RECURRENT_LOOPS_32B = _declare_flag(
    "AURA_RECURRENT_LOOPS_32B",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_SDK_PATH = _declare_flag(
    "AURA_SDK_PATH",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_SPECULATIVE_DECODING = _declare_flag(
    "AURA_SPECULATIVE_DECODING",
    kind=_FlagKind.STRING,
    default="1",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_SURFACE_RETRY_WALL_S = _declare_flag(
    "AURA_SURFACE_RETRY_WALL_S",
    kind=_FlagKind.STRING,
    default="20",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
logger = logging.getLogger("MLXWorker")


def _record_mlx_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("mlx_worker", exc, severity=severity, action=action)


def _state_application_quarantine_response(
    exc: UnknownActionStateApplicationError,
) -> dict[str, Any]:
    """Typed terminal IPC evidence for a worker that may be contaminated."""

    if not isinstance(exc, UnknownActionStateApplicationError):
        raise TypeError("unknown state-application error required")
    return {
        "status": "error",
        "message": exc.code,
        "state_application_quarantine": dict(exc.quarantine_evidence),
        "requires_worker_recycle": True,
    }


def _surface_prompt_resolution(job: dict[str, Any]) -> UserSurfacePromptResolution:
    return resolve_user_surface_prompt(job)


def _surface_validation_prompt(job: dict[str, Any]) -> str:
    return _surface_prompt_resolution(job).prompt


_CORRUPT_LANGUAGE_MARKERS = re.compile(
    r"\b(?:xublcate|ingediate|evocer)\b",
    re.IGNORECASE,
)
# Machine tokens whose IDENTITY is their casing: screaming-snake enum values and
# CamelCase internal symbols. Matching these case-INSENSITIVELY destroyed whole
# replies over ordinary English — "PROCEEDING" is a leaked enum, but
# "proceeding" is just a word, and this pattern is a FATAL check that returns
# None and annihilates the entire answer. Measured live: a conversational turn
# produced a 226-token draft and the user got "I couldn't get to an answer I'd
# stand behind on that one", with the log saying only "Hallucination detected by
# sanitizer. Returning empty text for caller-side recovery."
#
# The natural-language jargon that used to sit in this list ("field coherence",
# "system authority", "memory scar", "precognitive texture", "existence hash")
# is deliberately NOT here: those are style leaks, not model-state corruption,
# they occur legitimately in English, and the reliability gate already handles
# them as `pseudo_internal_jargon` — a reason that can be retried and repaired
# rather than one that throws the answer away.
_BACKEND_SYMBOLIC_SURFACE_MARKERS = re.compile(
    r"\b(?:PROCEEDING|TOOL_ACTION|CONVERGE_UNION|CONFORMED_METHODS|"
    r"TACTICAL_ORGANIZE|UI_SHUTDOWN_OR_DURATIVE_TIMEOUT|"
    r"MySelfEpsilon|CanonicalStabilityAnchor|currentInferenceProblem|"
    r"fieldOfPlay|INTRUSTION_DETECTED|INTRUSION_DETECTED|"
    r"ExistenceHash)\b"
)
_OPERATOR_EVIDENCE_DRIFT_MARKERS = re.compile(
    r"(?:\bSarah Connor\b|\bMother'?s Day\b|\bhuman error rate\b|"
    r"\bdeath by overthinking\b|\b100 rounds\b|\b100%\s+pass rate\b|"
    r"\bi['’]?ll be quiet for a while\b|:\s*/|[\u3400-\u9fff])",
    re.IGNORECASE,
)
_OPERATOR_EVIDENCE_META_MARKERS = re.compile(
    r"\b(?:for example|that'?s one paragraph as requested|"
    r"this is one paragraph as requested|anything else from the normal runtime state|"
    r"this response adheres strictly to (?:the )?format instructions(?: provided)?|"
    r"if you need any adjustments or have additional constraints)\b",
    re.IGNORECASE,
)
_OPERATOR_EVIDENCE_META_TAIL_RE = re.compile(
    r"\s*(?:that'?s one paragraph as requested|this is one paragraph as requested|"
    r"anything else from the normal runtime state|"
    r"this response adheres strictly to (?:the )?format instructions(?: provided)?|"
    r"if you need any adjustments or have additional constraints)\b.*$",
    re.IGNORECASE | re.DOTALL,
)
_SEMANTIC_COUNT_CONTRACT_RETRY_REASONS = frozenset(
    {
        "missing_requested_sentence_count",
        "missing_requested_word_count",
        "missing_current_topic_anchor",
        "output_contract_meta_reply",
        "punctuation_join_artifact",
    }
)
_SEMANTIC_COUNT_CONTRACT_RETRY_INSTRUCTION = (
    "Solve the current semantic task first, retain a concrete topic noun from "
    "the current user message when the requested count permits, and never "
    "describe the word or sentence constraint."
)


def _semantic_count_contract_retry_instruction(job: dict[str, Any]) -> str:
    """Render the admitted count contract as explicit retry guidance."""

    contract = job.get("requested_output_contract")
    if not isinstance(contract, dict):
        return _SEMANTIC_COUNT_CONTRACT_RETRY_INSTRUCTION

    kind = str(contract.get("kind") or "").strip().lower()
    requirement = ""
    if kind == "word_count":
        word_min = _safe_int(contract.get("word_min"), 0)
        word_max = _safe_int(contract.get("word_max"), 0)
        if word_min > 0 and word_min == word_max:
            requirement = f" The final answer must contain exactly {word_min} words."
        elif word_min > 0 and word_max >= word_min:
            requirement = (
                f" The final answer must contain between {word_min} and "
                f"{word_max} words inclusive."
            )
    elif kind == "sentence_count":
        sentence_count = _safe_int(contract.get("sentence_count"), 0)
        if sentence_count > 0:
            requirement = (
                f" The final answer must contain exactly {sentence_count} "
                f"sentence{'s' if sentence_count != 1 else ''}."
            )

    topic_requirement = ""
    validation_prompt = _surface_validation_prompt(job)
    if validation_prompt:
        try:
            from core.conversation.response_reliability import (
                requested_output_topic_anchors,
            )

            anchors = requested_output_topic_anchors(validation_prompt)[:8]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            anchors = ()
        if anchors:
            topic_requirement = (
                " Include at least one of these current-topic terms exactly as "
                f"written: {', '.join(anchors)}."
            )

    return (
        f"{_SEMANTIC_COUNT_CONTRACT_RETRY_INSTRUCTION}{requirement}"
        f"{topic_requirement} "
        "Count the final visible answer before ending it; return only that answer."
    )


def _safe_float(value: Any, default: float) -> float:
    """Finite-only float coercion — these helpers feed steering, sampling,
    retry, receipt, and token-budget paths, where NaN/inf silently poison
    comparisons and sampler construction."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _surface_generation_contract_enabled(job: dict[str, Any]) -> bool:
    """Decide whether this job's decode runs with the steering clamp.

    FAIL-SAFE INVERSION (July 2026 coherence incident): the clamp used to
    apply only to jobs that explicitly carried a surface/strict contract, so
    any route that dropped the flag decoded at full governor steering (up to
    alpha 3.0 — or the install-time 5.0 when the substrate sync was stale).
    Live symptom: fluent spliced-dialog nonsense served to the user. Every
    job is now clamped UNLESS it explicitly opts into full steering
    (latent-cortex episodes, steering experiments) — a dropped flag degrades
    to safe, never to hot.

    Strict/structured proof contracts still matter for the alpha TIER (see
    _surface_control_alpha): they need CORRECT symbolic tokens, not affective
    voice. Running them at full steering (alpha 5.0) corrupts the constrained
    first-token logits → zero-token generation that hangs to the 90s
    first-token timeout (DNU R011/R040/R022 wedges).
    """
    if bool(job.get("allow_full_affective_steering", False)):
        return False
    return True


def _job_requires_prompt_cache_bypass(job: dict[str, Any]) -> bool:
    """Return True for jobs that must neither read nor write the prompt cache.

    Probes and proof/strict contracts stay bypassed: they are short,
    non-conversational, and correctness-critical, and a probe must never
    warm or poison the conversational cache (the Jul 20 probe-turn
    contamination class).

    ``clean_user_surface_contract`` is deliberately NOT in this list any
    more. Every live user turn carries that contract, so bypassing it meant
    the conversation path could never reuse KV: each turn re-prefilled the
    entire history from token 0, per-turn latency grew linearly with
    context, and the endurance runs saturated to the turn timeout by turn
    ~9-15 (artifacts/reliability/runs/endurance-0715-clean: 11s → 25s →
    105s → 216s ceiling). User-surface jobs now use their own cache scope
    (see ``_prompt_cache_scope_for_job``) instead of no cache at all.
    """

    return bool(
        job.get("health_probe", False)
        or job.get("strict_answer_contract", False)
        or job.get("strict_value_contract", False)
        or job.get("proof_evaluation_contract", False)
        or job.get("operator_evidence_contract", False)
    )


def _prompt_cache_scope_for_job(job: dict[str, Any]) -> str:
    """Partition the prompt cache so lanes cannot cross-contaminate.

    User-surface turns reuse only user-surface entries; everything else
    (latent episodes, internal generation) shares the default scope. A
    cached prefix is only ever KV for a byte-identical token prefix, so
    within a scope "contamination" reduces to steering drift across turns
    of the same conversation — the history as it was actually computed,
    which is what a stateful conversation already is.
    """

    return (
        "user_surface"
        if job.get("clean_user_surface_contract", False)
        else "default"
    )


def _expected_empty_warmup_precompile(job: dict[str, Any]) -> bool:
    """True only for the bounded shader precompile where visible text is optional."""

    return bool(
        job.get("warmup_precompile", False)
        and 0 < _safe_int(job.get("max_tokens"), 0) <= 1
    )


def _surface_control_alpha(job: dict[str, Any], current_alpha: Any) -> float:
    # The only resident-32B live-alpha A/B is explicitly VOID: its steered and
    # baseline samples were byte-identical while the statistic still passed.
    # Residual steering therefore has no authority to perturb user-visible
    # tokens by default. Affect remains causal through attention, sampling,
    # action value, memory and voice; a future model-specific no-regression
    # certificate may request a non-zero alpha explicitly.
    default_alpha = "0.0"
    configured = job.get(
        "clean_user_surface_steering_alpha",
        os.environ.get("AURA_USER_SURFACE_STEERING_ALPHA", default_alpha),
    )
    requested = max(0.0, min(_safe_float(configured, 0.0), 1.0))
    try:
        current = float(current_alpha)
    except (TypeError, ValueError):
        current = requested
    if current > 0:
        requested = min(requested, current)
    return max(0.0, requested)


#: How deep the recurrent loop may run on a turn a person is waiting for.
#:
#: One is the identity depth: CP226 measured a T=1 gap of 0.0 against the live
#: weights, so a single pass is the model answering normally. Depth 2 is a
#: different claim, and the evidence for it does not exist — the CP227 accuracy
#: gate that was supposed to establish it ran with the adapter dark outside
#: ``recurrence_adapter_scope``, was voided, and has never been re-run.
#:
#: Measured live 2026-07-28: a request carrying several action verbs and a
#: desktop surface is classed "extended", which asked for depth 2, and the
#: cortex came back with nothing —
#:
#:   Cortex returned no text on user-facing request. Retrying once after 2s...
#:   Cortex-RETRY-1 produced an unsafe user-facing draft (too_short..., len=5)
#:   Cortex bounded retry failed.
#:
#: which then latched the foreground lane busy, so every later message was
#: refused with "I still have the previous turn open". One unvalidated depth
#: setting took the whole conversation surface down.
#:
#: So the ceiling is 1 until an accuracy gate says otherwise. Raising it is an
#: experiment and has to be asked for.
def _live_recurrent_ceiling() -> int:
    return user_surface_recurrent_ceiling()


def _surface_control_recurrent_loops(job: dict[str, Any]) -> int:
    return admit_user_surface_recurrent_loops(
        job.get("clean_user_surface_recurrent_loops")
    )


# Typed finite-range admission for sampling controls crossing the IPC
# boundary. Malformed or hostile values previously reached sampler
# construction unclamped (NaN temperature crashes decode; enormous values
# request unbounded work).
_SAMPLING_LIMITS = {
    "temp": (0.0, 2.0, 0.7),
    "top_p": (0.01, 1.0, 0.9),
    "min_p": (0.0, 1.0, 0.05),
    "repetition_penalty": (0.8, 3.0, 1.15),
    "presence_penalty": (-2.0, 2.0, 0.0),
}
# Absolute per-request decode ceiling. Nothing legitimate asks a resident
# worker for more in one request; malformed IPC could previously request
# unbounded decode work.
_ABSOLUTE_MAX_TOKENS = 16384


def _admit_sampling_control(job: dict[str, Any], key: str) -> float:
    lower, upper, default = _SAMPLING_LIMITS[key]
    return min(max(_safe_float(job.get(key, default), default), lower), upper)


def _admit_max_tokens(value: Any, default: int) -> int:
    return max(1, min(_safe_int(value, default), _ABSOLUTE_MAX_TOKENS))


_TOKEN_BUDGET_CALIBRATION_SAMPLES: tuple[tuple[str, str], ...] = (
    ("prose", "Aura keeps the current conversation coherent while answering directly."),
    ("python", "def total(values):\n    return sum(value * 2 for value in values)"),
    ("json", '{"status":"ready","attempts":2,"verified":true,"items":[1,2,3]}'),
    # A tokenizer calibration sample, not a real location. Written without a
    # leading slash so the enterprise gate's hardcoded-local-path rule does
    # not have to distinguish a sample from a hard-coded destination — the
    # token mix is identical either way.
    ("path", "home/person/Documents/Aura Demo/research-notes/final_report.pdf"),
    ("markdown", "## Findings\n- evidence is measured\n- uncertainty remains explicit"),
    ("url", "https://example.org/research?q=causal+reasoning&year=2026#results"),
    ("dialogue", "User: Are you okay?\nAura: I feel steady and present with this thread."),
    ("symbols", "x_t = f(x_{t-1}, action_t); confidence=0.875; delta<=1e-6"),
)


def _token_budget_calibration_evidence(tokenizer: Any) -> dict[str, Any]:
    """Measure a fixed public prompt mix with the worker's resident tokenizer."""

    observations: list[dict[str, Any]] = []
    for label, sample in _TOKEN_BUDGET_CALIBRATION_SAMPLES:
        try:
            encoded = tokenizer.encode(sample, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(sample)
        token_count = len(encoded)
        if token_count <= 0:
            continue
        observations.append(
            {
                "label": label,
                "chars": len(sample),
                "tokens": token_count,
            }
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "sample_set": "aura-runtime-mixed-v1",
        "observations": observations,
    }


def _name_tokens(tokenizer: Any, token_ids: Any) -> str:
    """What the decoder actually emitted, by name where one exists.

    Special tokens decode to nothing, so a log that reports only a count
    describes an empty string and an end-of-turn marker identically.
    """
    try:
        ids = [int(token) for token in (token_ids or [])]
    except (TypeError, ValueError):
        return "unreadable"
    if not ids:
        return "none"
    named: list[str] = []
    for token_id in ids:
        text = ""
        try:
            text = tokenizer.decode([token_id])
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            text = ""
        if text.strip():
            named.append(repr(text))
            continue
        label = ""
        try:
            label = str(tokenizer.convert_ids_to_tokens(token_id))
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            label = ""
        named.append(label or f"id:{token_id}")
    return ", ".join(named)


def _render_messages_fallback(messages: Any, prompt: Any) -> str:
    """Deterministic role-labeled rendering when the native template fails.

    The old fallback silently reused the preexisting ``prompt`` variable —
    dropping the entire message transcript (and answering a stale, null, or
    different task). Role labels are preserved so conversational authority
    survives; non-text fragments are skipped visibly rather than silently.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return str(prompt or "")
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, (list, tuple)):
            fragments = [
                str(fragment.get("text") or "")
                for fragment in content
                if isinstance(fragment, dict) and fragment.get("type") == "text"
            ]
            text = "\n".join(fragment for fragment in fragments if fragment)
        else:
            text = str(content or "")
        if text.strip():
            lines.append(f"{role.capitalize()}:\n{text.strip()}")
    if not lines:
        return str(prompt or "")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _apply_surface_generation_controls(
    engine: Any,
    model: Any,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Clamp latent embellishment when the next tokens are user-visible prose."""
    if not _surface_generation_contract_enabled(job):
        return {"enabled": False}

    state: dict[str, Any] = {"enabled": True, "apply_errors": []}
    alpha = _surface_control_alpha(job, getattr(engine, "_alpha", None))
    state["surface_alpha_requested"] = alpha

    if engine is not None:
        state["engine"] = engine
        state["surface_alpha_override_before"] = getattr(engine, "_surface_alpha_override", None)
        hooks = list(getattr(engine, "_hooks", []) or [])
        state["hook_alphas_before"] = [(hook, getattr(hook, "_alpha", None)) for hook in hooks]
        try:
            if hasattr(engine, "set_surface_alpha_override"):
                engine.set_surface_alpha_override(alpha)
            else:
                for hook in hooks:
                    hook._alpha = min(float(getattr(hook, "_alpha", alpha) or alpha), alpha)
            state["surface_alpha_applied"] = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            state["apply_errors"].append(f"steering_clamp:{type(exc).__name__}")
            _record_mlx_degradation(
                exc,
                action="recorded steering clamp failure for fail-closed surface admission",
                severity="error",
            )
            logger.warning("Surface steering clamp failed: %s", exc)
    elif alpha == 0.0:
        # A missing optional steering engine is exactly equivalent to a zero
        # steering request.  Treating this as an unapplied control made the
        # neutral, user-visible path depend on the embellishment it disabled.
        state["surface_alpha_applied"] = 0.0
    else:
        state["apply_errors"].append("steering_unavailable")

    layer_view = resolve_model_layers(model)
    inner = layer_view.owner if layer_view is not None else None
    if inner is not None and getattr(inner, "_recurrent_depth_config", None):
        state["recurrent_inner"] = inner
        state["had_recurrent_runtime_loops"] = hasattr(inner, "_recurrent_depth_runtime_loops")
        state["recurrent_runtime_loops_before"] = getattr(inner, "_recurrent_depth_runtime_loops", None)
        try:
            loops = _surface_control_recurrent_loops(job)
            inner._recurrent_depth_runtime_loops = loops
            state["recurrent_runtime_loops_applied"] = loops
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            state["apply_errors"].append(f"recurrent_clamp:{type(exc).__name__}")
            state["recurrent_clamp_failed"] = True
            _record_mlx_degradation(
                exc,
                action="recorded recurrent-depth clamp failure for fail-closed surface admission",
                severity="error",
            )
            logger.warning("Surface recurrent-depth clamp failed: %s", exc)

    # What a user-visible decode ACTUALLY ran with. Every diagnosis of a bad
    # reply on 2026-07-26 stalled here: the receipt carried these numbers but
    # nothing put them where a live log would show them, so "the clamp is
    # applied" and "the clamp silently no-opped" looked identical from outside.
    if bool(job.get("clean_user_surface_contract", False)):
        logger.info(
            "🎚️ [WORKER] Surface decode: steering α=%s (engine α=%s), "
            "recurrent loops=%s (was %s, depth_present=%s)%s",
            state.get("surface_alpha_applied"),
            getattr(engine, "_alpha", None),
            state.get("recurrent_runtime_loops_applied"),
            state.get("recurrent_runtime_loops_before"),
            state.get("recurrent_inner") is not None,
            (
                " APPLY_ERRORS=" + ",".join(state.get("apply_errors") or [])
                if state.get("apply_errors")
                else ""
            ),
        )
    return state


def _enforce_surface_controls_or_fail(job: dict[str, Any], state: dict[str, Any]) -> None:
    """Fail closed when a user-visible contract selected controls that did
    not apply.

    Decoding a clean-user-surface job WITHOUT the steering/recurrent clamps
    its contract selected serves latent-embellished prose to a user; the
    receipt honestly said applied=False but nothing enforced it. Strict,
    proof, and health jobs keep their own gates and are not blocked here.
    """
    errors = list(state.get("apply_errors") or [])
    if errors and bool(job.get("clean_user_surface_contract", False)):
        raise RuntimeError(
            "surface_controls_unavailable:" + ";".join(errors)[:200]
        )


def _restore_surface_generation_controls(state: dict[str, Any]) -> bool:
    """Restore pre-job control state; False means the resident model may be
    contaminated and the worker must not serve further jobs on it."""
    if not state.get("enabled"):
        return True

    restored = True
    engine = state.get("engine")
    if engine is not None:
        try:
            if hasattr(engine, "set_surface_alpha_override"):
                engine.set_surface_alpha_override(state.get("surface_alpha_override_before"))
            else:
                for hook, alpha in state.get("hook_alphas_before", []):
                    if alpha is not None:
                        hook._alpha = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            restored = False
            _record_mlx_degradation(
                exc,
                action="flagged worker for recycle after user-surface steering restore failed",
                severity="critical",
            )
            logger.error("Surface steering restore failed: %s", exc)

    inner = state.get("recurrent_inner")
    if inner is not None:
        try:
            if state.get("had_recurrent_runtime_loops"):
                inner._recurrent_depth_runtime_loops = state.get("recurrent_runtime_loops_before")
            elif hasattr(inner, "_recurrent_depth_runtime_loops"):
                delattr(inner, "_recurrent_depth_runtime_loops")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            restored = False
            _record_mlx_degradation(
                exc,
                action="flagged worker for recycle after user-surface recurrent-depth restore failed",
                severity="critical",
            )
            logger.error("Surface recurrent-depth restore failed: %s", exc)
    return restored


def _surface_generation_control_receipt(
    job: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Return an IPC-safe proof of user-surface generation control application."""
    enabled = bool(state.get("enabled"))
    try:
        generation_max_tokens = max(
            1,
            int(
                state.get("generation_max_tokens_applied")
                or job.get("max_tokens")
                or 1
            ),
        )
    except (TypeError, ValueError, OverflowError):
        generation_max_tokens = 1
    receipt: dict[str, Any] = {
        "enabled": enabled,
        # PROVENANCE: the fields named in caller_declared_fields are ECHOES
        # of the caller's job booleans — the worker cannot independently
        # verify them and consumers must not read them as worker-proven
        # facts. Worker-measured evidence lives in worker_verified.
        "caller_declared_fields": [
            "live_mind_controls_bound",
            "clean_user_surface_contract",
            "strict_answer_contract",
            "strict_value_contract",
            "proof_evaluation_contract",
            "operator_evidence_contract",
            "health_probe",
            "runtime_fact_status_contract",
            "grounded_runtime_status_contract",
            "surface_validation_prompt_source",
        ],
        "live_mind_controls_bound": bool(job.get("live_mind_controls_bound", False)),
        "clean_user_surface_contract": bool(job.get("clean_user_surface_contract", False)),
        "surface_validation_prompt_present": bool(_surface_validation_prompt(job)),
        "surface_validation_prompt_bound": _surface_prompt_resolution(job).bound,
        "surface_validation_prompt_binding_valid": _surface_prompt_resolution(job).valid,
        "surface_validation_prompt_source": _surface_prompt_resolution(job).source,
        "surface_validation_prompt_sha256": _surface_prompt_resolution(job).sha256,
        "strict_answer_contract": bool(job.get("strict_answer_contract", False)),
        "strict_value_contract": bool(job.get("strict_value_contract", False)),
        "proof_evaluation_contract": bool(job.get("proof_evaluation_contract", False)),
        "operator_evidence_contract": bool(job.get("operator_evidence_contract", False)),
        "health_probe": bool(job.get("health_probe", False)),
        "runtime_fact_status_contract": bool(
            job.get("runtime_fact_status_contract", False)
        ),
        "grounded_runtime_status_contract": bool(
            job.get("grounded_runtime_status_contract", False)
        ),
        "generation_max_tokens": generation_max_tokens,
        "memory_pressure_token_cap": job.get("memory_pressure_token_cap"),
        "user_surface_completion_floor": job.get("user_surface_completion_floor"),
        "completion_floor_applied": bool(job.get("completion_floor_applied", False)),
        "caller_requested_max_tokens": job.get("caller_requested_max_tokens"),
        "adaptive_suggested_max_tokens": job.get("adaptive_suggested_max_tokens"),
        "output_contract_generation_floor": job.get(
            "output_contract_generation_floor"
        ),
        "semantic_output_token_cap": job.get("semantic_output_token_cap"),
        "hard_output_token_ceiling": job.get("hard_output_token_ceiling"),
        "generation_stop_reason": state.get("generation_stop_reason"),
        "generation_configured_stop_sequence": state.get(
            "generation_configured_stop_sequence"
        ),
        "semantic_completion_contract": bool(
            state.get("semantic_completion_contract", False)
        ),
        "semantic_completion_satisfied": bool(
            state.get("semantic_completion_satisfied", False)
        ),
        "semantic_completion_incomplete": bool(
            state.get("semantic_completion_incomplete", False)
        ),
        "semantic_completion_missing_part_count": max(
            0,
            _safe_int(state.get("semantic_completion_missing_part_count"), 0),
        ),
        "semantic_completion_missing_part_indexes": list(
            state.get("semantic_completion_missing_part_indexes") or []
        ),
        "semantic_completion_quality_reasons": list(
            state.get("semantic_completion_quality_reasons") or []
        ),
        "semantic_completion_epistemic_partition_covered": state.get(
            "semantic_completion_epistemic_partition_covered"
        ),
        "semantic_completion_terminal_boundary": bool(
            state.get("semantic_completion_terminal_boundary", False)
        ),
        "continuation_resume_requested": bool(
            state.get("continuation_resume_requested", False)
        ),
        "continuation_resume_applied": bool(
            state.get("continuation_resume_applied", False)
        ),
        "continuation_resume_available": bool(
            state.get("continuation_resume_available", False)
        ),
        "instruction_shape_repair_applied": bool(
            state.get("instruction_shape_repair_applied", False)
        ),
        "text_mutations": normalize_text_mutations(state.get("text_mutations")),
        "applied": False,
    }
    receipt["text_mutation_count"] = len(receipt["text_mutations"])
    resume_handle = str(state.get("continuation_resume_handle") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", resume_handle):
        receipt["continuation_resume_handle"] = resume_handle
    resume_failure = str(
        state.get("continuation_resume_failure_reason") or ""
    ).strip()
    if resume_failure:
        receipt["continuation_resume_failure_reason"] = resume_failure[:120]
    receipt["deterministic_repair_applied"] = any(
        bool(item.get("deterministic")) for item in receipt["text_mutations"]
    )
    receipt.update(summarize_text_mutation_authorship(receipt["text_mutations"]))
    for key in (
        "exact_reply_token_count",
        "exact_reply_required_termination_headroom",
        "exact_reply_available_termination_headroom",
        "exact_reply_content_capacity_sufficient",
        "exact_reply_termination_headroom_sufficient",
        "exact_reply_token_ceiling_valid",
        "exact_reply_native_capacity_sufficient",
    ):
        if key in job:
            receipt[key] = job.get(key)
    output_contract = job.get("requested_output_contract")
    if isinstance(output_contract, dict) and output_contract:
        receipt["requested_output_contract"] = dict(output_contract)
    if not enabled:
        return receipt

    if job.get("clean_user_surface_steering_alpha") is not None:
        receipt["surface_alpha_requested"] = _safe_float(
            job.get("clean_user_surface_steering_alpha"),
            0.0,
        )
    if "surface_alpha_applied" in state:
        receipt["surface_alpha_applied"] = state.get("surface_alpha_applied")
    receipt["surface_alpha_applied_ok"] = (
        "surface_alpha_applied" in state or state.get("engine") is None
    )
    # Attribution evidence: what the hooks ACTUALLY injected at (ceiling and
    # staleness derating happen inside the hook, so the requested/applied
    # values alone cannot explain an incident) plus how fresh the substrate
    # sync was. steering_sync_age_s = -1.0 means the sync never ran.
    receipt_engine = state.get("engine")
    # Worker-MEASURED facts (not caller echoes): live steering engine state
    # and clamp application outcomes, sampled at receipt time.
    try:
        receipt["worker_verified"] = {
            "steering_engine_present": receipt_engine is not None,
            "steering_engine_active": bool(
                receipt_engine.is_active()
            )
            if receipt_engine is not None
            else False,
            "surface_clamp_errors": list(state.get("apply_errors") or []),
            "native_thinking_enabled": bool(
                state.get("native_thinking_enabled", False)
            ),
            "native_thinking_boundary_closed": bool(
                state.get("native_thinking_boundary_closed", False)
            ),
            "native_thinking_private_chars": max(
                0,
                _safe_int(state.get("native_thinking_private_chars"), 0),
            ),
        }
    except (AttributeError, RuntimeError, TypeError) as verify_exc:
        receipt["worker_verified"] = {
            "steering_engine_present": receipt_engine is not None,
            "steering_engine_active": False,
            "verification_error": f"{type(verify_exc).__name__}: {verify_exc}",
            "surface_clamp_errors": list(state.get("apply_errors") or []),
            "native_thinking_enabled": bool(
                state.get("native_thinking_enabled", False)
            ),
            "native_thinking_boundary_closed": bool(
                state.get("native_thinking_boundary_closed", False)
            ),
            "native_thinking_private_chars": max(
                0,
                _safe_int(state.get("native_thinking_private_chars"), 0),
            ),
        }
    receipt_hooks = list(getattr(receipt_engine, "_hooks", []) or []) if receipt_engine is not None else []
    if receipt_hooks:
        effective_alphas = [
            _safe_float(getattr(hook, "_last_effective_alpha", 0.0), 0.0)
            for hook in receipt_hooks
        ]
        receipt["steering_effective_alpha_max"] = round(max(effective_alphas), 4)
        sync_stamps = [
            _safe_float(getattr(hook, "_last_substrate_sync_monotonic", 0.0), 0.0)
            for hook in receipt_hooks
        ]
        newest_sync = max(sync_stamps)
        receipt["steering_sync_age_s"] = (
            round(max(0.0, time.monotonic() - newest_sync), 3) if newest_sync > 0 else -1.0
        )

    if job.get("clean_user_surface_recurrent_loops") is not None:
        receipt["recurrent_runtime_loops_requested"] = _safe_int(
            job.get("clean_user_surface_recurrent_loops"),
            1,
        )
    receipt["recurrent_depth_present"] = state.get("recurrent_inner") is not None
    if "recurrent_runtime_loops_applied" in state:
        receipt["recurrent_runtime_loops_applied"] = state.get(
            "recurrent_runtime_loops_applied"
        )
    requested_loops = receipt.get("recurrent_runtime_loops_requested")
    applied_loops = receipt.get("recurrent_runtime_loops_applied")
    recurrence_not_applicable = state.get("recurrent_inner") is None
    receipt["recurrent_runtime_loops_applied_ok"] = bool(
        recurrence_not_applicable
        or (
            type(requested_loops) is int
            and type(applied_loops) is int
            and requested_loops == applied_loops
        )
    )
    receipt["applied"] = bool(
        receipt.get("surface_alpha_applied_ok")
        and receipt.get("recurrent_runtime_loops_applied_ok")
        and (
            "surface_alpha_applied" in state
            or "recurrent_runtime_loops_applied" in state
        )
    )
    for key in (
        "surface_quality_gate_enabled",
        "surface_quality_gate_passed",
        "surface_quality_gate_attempts",
        "surface_quality_gate_reasons",
        "telemetry_sanitizer_reasons",
        # The draft the gate rejected, carried rather than destroyed.
        #
        # Blanking it turned "I wrote something a heuristic disliked" into
        # "the client returned no text", which opened the Cortex circuit,
        # tripped the sovereign no-fallback policy, and served Bryan a canned
        # apology while the turn was holding an answer. core/runtime/
        # turn_outcome.py states the rule this restores: a gate ANNOTATES or
        # TRANSFORMS a candidate, it does not destroy one.
        #
        # Carrying it does NOT make it servable. It travels marked as
        # suppressed, and only the caller's recovery path — when the
        # alternative is nothing at all — may serve it.
        "surface_quality_rejected_text",
        "surface_quality_rejected_reasons",
        "surface_quality_gate_error",
        "surface_quality_gate_exemption",
        "surface_quality_gate_waived_reasons",
        "instruction_shape_repair_applied",
        "sentinel_loop_prefix_preserved",
    ):
        if key in state:
            receipt[key] = state.get(key)
    return receipt


def _surface_quality_gate_enabled(job: dict[str, Any]) -> bool:
    if not bool(job.get("clean_user_surface_contract", False)):
        return False
    prompt_resolution = _surface_prompt_resolution(job)
    if not prompt_resolution.prompt and not prompt_resolution.bound:
        return False
    return not bool(
        job.get("health_probe", False)
        or job.get("runtime_fact_status_contract", False)
        or job.get("grounded_runtime_status_contract", False)
        or job.get("operator_evidence_contract", False)
        or job.get("strict_answer_contract", False)
        or job.get("strict_value_contract", False)
        or job.get("proof_evaluation_contract", False)
        or job.get("schema")
    )


def _surface_quality_failure_reasons(
    job: dict[str, Any],
    response_text: Any,
) -> list[str]:
    """Validate user-visible drafts inside the worker before IPC success."""
    if not _surface_quality_gate_enabled(job):
        return []
    prompt_resolution = _surface_prompt_resolution(job)
    if prompt_resolution.bound and not prompt_resolution.valid:
        return [prompt_resolution.error or "surface_validation_prompt_binding_invalid"]
    prompt = prompt_resolution.prompt
    if not prompt:
        return []
    recent_raw = job.get("user_surface_recent_messages")
    recent_messages = (
        [str(message or "") for message in recent_raw]
        if isinstance(recent_raw, (list, tuple))
        else []
    )
    grounding_raw = job.get("user_surface_grounding_evidence")
    grounding = (
        [str(item or "") for item in grounding_raw]
        if isinstance(grounding_raw, (list, tuple))
        else []
    )
    try:
        from core.conversation.response_reliability import assess_user_facing_reply
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_mlx_degradation(
            exc,
            action="blocked live user-surface generation because quality gate was unavailable",
            severity="critical",
        )
        return ["surface_quality_gate_unavailable"]

    candidate = _surface_quality_candidate(job, response_text)
    assessment = assess_user_facing_reply(
        prompt,
        candidate,
        recent_user_messages=recent_messages,
        grounding=grounding,
        sensory_evidence=job.get("user_surface_sensory_evidence"),
        tool_receipts=job.get("user_surface_tool_receipts", ()),
    )
    sanitizer_reasons = _telemetry_sanitization_failure_reasons(
        candidate,
        is_proof=False,
    )
    self_claim_contradiction = False
    self_claim_verification_unavailable = False
    try:
        from core.conversation.self_claim_verifier import verify_self_claims

        self_claim_contradiction = not verify_self_claims(candidate).ok
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        self_claim_verification_unavailable = True
        _record_mlx_degradation(
            exc,
            action="continued surface validation after self-claim verification failed",
            severity="error",
        )
    if (
        assessment.ok
        and not assessment.retryable
        and not assessment.hard_failure
        and not sanitizer_reasons
        and not self_claim_contradiction
        and not self_claim_verification_unavailable
    ):
        return []
    reasons = list(assessment.reasons)
    reasons.extend(sanitizer_reasons)
    if self_claim_contradiction:
        reasons.append("self_claim_contradiction")
    if self_claim_verification_unavailable:
        reasons.append("self_claim_verification_unavailable")
    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        reasons = ["surface_quality_gate_failed"]
    # This gate is about INTEGRITY — leaks, corruption, prompt artefacts and
    # text that is not language. Those reasons may suppress a draft, but the
    # draft remains intact for one bounded authored correction whose wall starts
    # when correction starts. COMPLETENESS is different: a draft that merely
    # fell short is real content and remains the floor the route can deliver.
    # See core/conversation/surface_disposition.py.
    try:
        from core.conversation.surface_disposition import integrity_failures

        reasons = list(integrity_failures(reasons))
    except (ImportError, RuntimeError, TypeError, ValueError):
        reasons = [reason for reason in reasons if reason != "final_answer_missing"]
    if not reasons:
        return []
    if bool(job.get("capability_inventory_contract", False)):
        grounded, _evidence = _capability_inventory_minimum_grounding(response_text)
        if grounded:
            reasons = [
                reason
                for reason in reasons
                if reason
                not in {
                    "too_thin_for_operational_status_turn",
                    "too_thin_for_status_turn",
                    "too_short_for_user_turn",
                    "too_thin_for_user_turn",
                }
            ]
    return reasons


def _surface_quality_candidate(job: dict[str, Any], response_text: Any) -> str:
    """Return the complete authored candidate represented by this decode."""

    tail = str(response_text or "")
    if not bool(job.get("user_surface_continuation_contract", False)):
        return tail
    head = str(job.get("user_surface_continuation_partial") or "")
    if not head:
        return tail
    if not tail:
        return head
    separator = ""
    if not head[-1].isspace() and not tail[0].isspace():
        separator = "" if tail[0] in ".,;:!?)]}" else " "
    return f"{head}{separator}{tail}"


def _semantic_surface_stop_ready(
    job: dict[str, Any],
    response_text: Any,
    *,
    generated_tokens: int,
    minimum_tokens: int | None = None,
) -> bool:
    """Stop a bounded decode once its visible contract is demonstrably complete."""

    if not bool(job.get("semantic_completion_contract", False)):
        return False
    required_tokens = (
        max(1, int(minimum_tokens))
        if minimum_tokens is not None
        else 8
        if job.get("user_surface_continuation_contract")
        else 24
    )
    if int(generated_tokens) < required_tokens:
        return False
    candidate = _surface_quality_candidate(job, response_text).rstrip()
    if not candidate.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]")):
        return False
    try:
        from core.conversation.request_coverage import (
            requested_epistemic_partition_is_covered,
            unanswered_question_parts,
        )
        from core.language.discourse_commitments import unfulfilled_commitments
        from core.runtime.structured_input import analyze_prompt_shape

        if unfulfilled_commitments(candidate):
            return False
        validation_prompt = _surface_validation_prompt(job)
        if not requested_epistemic_partition_is_covered(validation_prompt, candidate):
            return False
        if unanswered_question_parts(
            candidate,
            analyze_prompt_shape(validation_prompt),
        ):
            return False
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued bounded generation because semantic completion proof was unavailable",
            severity="warning",
        )
        return False
    return not _surface_quality_failure_reasons(job, candidate)


def _semantic_completion_receipt_state(
    job: dict[str, Any],
    response_text: Any,
    *,
    generated_tokens: int,
) -> dict[str, Any]:
    """Describe completion without changing the model's decode distribution."""

    required = bool(job.get("semantic_completion_contract", False))
    candidate = _surface_quality_candidate(job, response_text).rstrip()
    terminal_boundary = bool(
        candidate.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]"))
    )
    missing_indexes: list[int] = []
    discourse_missing: list[dict[str, Any]] = []
    quality_reasons: list[str] = []
    epistemic_covered: bool | None = None
    if required:
        try:
            from core.conversation.request_coverage import (
                requested_epistemic_partition_is_covered,
                unanswered_question_parts,
            )
            from core.language.discourse_commitments import unfulfilled_commitments
            from core.runtime.structured_input import analyze_prompt_shape

            validation_prompt = _surface_validation_prompt(job)
            shape = analyze_prompt_shape(validation_prompt)
            missing = list(unanswered_question_parts(candidate, shape))
            segments = list(getattr(shape, "question_segments", ()) or ())
            missing_indexes = [
                index for index, segment in enumerate(segments) if segment in missing
            ]
            epistemic_covered = requested_epistemic_partition_is_covered(
                validation_prompt,
                candidate,
            )
            quality_reasons = list(_surface_quality_failure_reasons(job, candidate))
            discourse_missing = [
                {
                    "expected_count": item.expected_count,
                    "observed_count": item.observed_count,
                    "kind": item.kind,
                    "declaration": item.declaration,
                }
                for item in unfulfilled_commitments(candidate)
            ]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="reported semantic completion as incomplete because diagnostics were unavailable",
                severity="warning",
            )
            quality_reasons = ["completion_diagnostics_unavailable"]
    # The established validator remains the authority. Diagnostics explain its
    # decision; they do not reimplement it and accidentally create a second,
    # divergent completion contract.
    satisfied = bool(
        required
        and _semantic_surface_stop_ready(
            job,
            response_text,
            generated_tokens=generated_tokens,
            # The default keeps an early sentence from stopping a longer
            # decode. After generation ends, semantic and terminal evidence,
            # rather than length alone, decides whether the answer is complete.
            minimum_tokens=1,
        )
    )
    return {
        "semantic_completion_contract": required,
        "semantic_completion_satisfied": satisfied,
        "semantic_completion_incomplete": bool(required and not satisfied),
        "semantic_completion_missing_part_count": len(missing_indexes),
        "semantic_completion_missing_part_indexes": missing_indexes,
        "semantic_completion_unfulfilled_discourse": discourse_missing,
        "semantic_completion_unfulfilled_discourse_count": len(discourse_missing),
        "semantic_completion_quality_reasons": quality_reasons,
        "semantic_completion_epistemic_partition_covered": epistemic_covered,
        "semantic_completion_terminal_boundary": terminal_boundary,
    }


def _semantic_terminal_grace_eligible(
    job: dict[str, Any],
    response_text: Any,
    *,
    generated_tokens: int,
) -> bool:
    """Whether a deadline-cut answer needs only its terminal boundary.

    This is deliberately narrower than a generic timeout extension. Every
    typed request obligation, quality check, and epistemic partition must
    already pass. The worker may then spend a few tokens from the caller's
    existing delivery reserve to close the sentence instead of paying for a
    second full prefill and decode.
    """

    receipt = _semantic_completion_receipt_state(
        job,
        response_text,
        generated_tokens=generated_tokens,
    )
    return bool(
        receipt.get("semantic_completion_contract")
        and not receipt.get("semantic_completion_missing_part_count")
        and not receipt.get("semantic_completion_quality_reasons")
        and receipt.get("semantic_completion_epistemic_partition_covered") is True
        and receipt.get("semantic_completion_terminal_boundary") is False
    )


def _loop_abort_prefix_is_servable(
    job: dict[str, Any],
    response_text: Any,
) -> bool:
    """Whether a repetition abort left enough clean authored work to retain.

    A late loop is a defect in the tail, not evidence that the prefix never
    existed. Restarting a resident decode from token zero discarded hundreds
    of clean tokens and routinely exhausted the owning request deadline before
    the replacement caught up. Keep a substantial prefix when the integrity
    gate finds no positively identified leak or corruption; ordinary
    completion recovery can extend it without re-paying for the whole answer.
    Tiny/unsafe prefixes still take the bounded clean-retry path.
    """

    if not bool(job.get("clean_user_surface_contract", False)):
        return False
    body = str(response_text or "").strip()
    if len(body) < 160 or len(body.split()) < 30:
        return False
    return not _surface_quality_failure_reasons(job, body)


def _classify_generation_stop_reason(
    *,
    soft_cancelled: bool,
    deadline_hit: bool,
    sentinel_aborted: bool,
    role_continuation_hit: bool,
    configured_stop_hit: bool,
    hard_token_limit_hit: bool,
    semantic_contract_satisfied: bool = False,
    generated_tokens: int,
    max_tokens: int,
) -> str:
    """Return the exact terminal event that ended one decode attempt."""

    if soft_cancelled:
        return "soft_cancelled"
    if deadline_hit:
        return "deadline_exceeded"
    if sentinel_aborted:
        return "sentinel_abort"
    if role_continuation_hit:
        return "role_continuation"
    if configured_stop_hit:
        return "configured_stop"
    if hard_token_limit_hit:
        return "hard_token_limit"
    if semantic_contract_satisfied:
        return "semantic_contract_satisfied"
    if int(generated_tokens) >= max(1, int(max_tokens)):
        return "max_tokens"
    return "eos"


def _continuation_resume_unavailable_reason(
    *,
    resume_required: bool,
    cache_lru_available: bool,
    cache_disabled: bool,
    final_cache_available: bool,
    sentinel_aborted: bool,
    response_present: bool,
) -> str:
    """Name a failed required resume, or return empty when none was required."""

    if not resume_required:
        return ""
    if not cache_lru_available:
        return "cache_lru_unavailable"
    if cache_disabled:
        return "cache_disabled_by_contract"
    if not final_cache_available:
        return "final_cache_unavailable"
    if sentinel_aborted:
        return "sentinel_aborted"
    if not response_present:
        return "empty_partial"
    return "cache_retention_refused"


def _continuation_resume_should_bind(
    *,
    generation_stop_reason: str,
    semantic_completion_incomplete: bool,
) -> bool:
    """Whether downstream completion still needs ownership of exact decode state.

    Deadline and token-cap stops need a resume only when the worker can already
    prove the visible answer incomplete. A semantic stop is different: later
    language projections inspect the assembled user surface and can discover an
    obligation that was not present in the worker's raw segment. Retaining its
    cache is therefore part of the successful transaction, not evidence that the
    worker itself judged the answer incomplete.
    """

    stop_reason = str(generation_stop_reason or "").strip().lower()
    if stop_reason == "semantic_contract_satisfied":
        return True
    return bool(
        semantic_completion_incomplete
        and stop_reason in {"deadline_exceeded", "max_tokens", "soft_cancelled"}
    )


def _capability_inventory_minimum_grounding(
    response_text: Any,
) -> tuple[bool, dict[str, bool]]:
    """Evidence check for the capability-inventory thin-response exemption.

    Capability inventory questions contain "tools", "external", and
    "desktop", which overlap operational-status classifiers, so a concise
    governed inventory may be exempted from thin-response failures — but
    ONLY when it actually shows its documented evidence: concrete
    categories, governance, and the non-execution boundary. The boundary is
    non-negotiable; the old `or not effect-evidence` escape admitted
    answers with NEITHER effect evidence NOR a boundary.
    """
    reply = str(response_text or "").lower()
    evidence = {
        "category": "browser/web research" in reply
        or ("browser" in reply and ("file" in reply or "desktop" in reply)),
        "governance": any(
            marker in reply
            for marker in ("will/authority", "will and authority", "permission", "governed")
        ),
        "boundary": (
            "not executing" in reply
            or "not opening" in reply
            or "hypothetical" in reply
            or "in this turn" in reply
        ),
        "effect_evidence": "receipt" in reply or "effect verification" in reply,
    }
    grounded = evidence["category"] and evidence["governance"] and evidence["boundary"]
    return grounded, evidence


# Residual quality-gate reasons that are STYLE/COMPLETENESS defects, not
# integrity leaks: after retries are exhausted, a substantive draft carrying
# only these is delivered (with an honest gate receipt) instead of being
# replaced by an empty reply. Observed live (Jul 7, post-restart): a
# consciousness question drew real drafts that kept failing
# missing_self_claim_evidence_boundary + missing_requested_phrase, and every
# turn died as empty_cognitive_engine_reply — a dead turn is strictly worse
# than an imperfectly-styled honest one. Leak/overclaim reasons stay
# fail-closed.
_DELIVERABLE_RESIDUAL_SURFACE_REASONS = frozenset(
    {
        # A reply that wandered off the thread is still a reply. Marking the
        # turn for repair is right; discarding it and reporting an
        # infrastructure failure over it is the defect class this set exists
        # to prevent, and a topical miss must not become a new instance of it.
        "reply_abandons_thread",
        "missing_requested_phrase",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_reference_value",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "empty_requested_list_item",
        "missing_requested_choice_clarification",
        "missing_requested_followup_question",
        "too_short_for_user_turn",
        "too_thin_for_user_turn",
        "too_thin_for_open_ended_turn",
        "too_thin_for_status_turn",
        "too_thin_for_operational_status_turn",
        "too_thin_for_expansion_request",
        # The reliability-flavoured thinness verdicts belong with the rest of
        # the family. They were the only two "the draft is thinner than we
        # wanted" reasons that killed the turn instead of delivering it, and
        # the failure was measured live: a correct 276-character answer about
        # re-prefilling — mechanism plus a numeric threshold — was discarded,
        # and the user was told "I couldn't get to an answer I'd stand behind".
        # Thinness is not a safety or honesty defect; a short true answer is
        # strictly better than a refusal, and worst of all on a turn that ASKED
        # about reliability.
        "reliability_diagnostic_too_thin",
        "too_thin_for_reliability_turn",
        # How a reply ADDRESSES someone is a one-word detail, never a reason to
        # discard the reply. Measured live: a natural, correctly-addressed turn
        # ("Bryan, let's reset... Talk like we're peers figuring something out
        # together") was destroyed because the name was not in any grounding
        # source the check consulted. Delivering it with the residual recorded
        # keeps the human part; annihilating it protects a detail by throwing
        # away the answer.
        "ungrounded_person_address",
        "low_signal_acknowledgement_placeholder",
        "generic_assistant_language",
        # Replaying the owner's own first-person sentence as her own is a
        # comprehension defect worth measuring, not worth destroying a turn
        # over — the rest of the reply is usually fine, and killing it leaves
        # the person with nothing while the underlying attribution problem
        # goes unrecorded. It is fixed at the source (core/dialogue/
        # referents.py); this reason is how a regression there becomes a rate
        # instead of an anecdote.
        "borrowed_owner_first_person_speech",
    }
)

# Appended when a draft answers a consciousness/experience question without
# the evidence boundary the honesty gate requires. Deterministic, aligned
# with the evidence-bounded self-claim policy, and satisfies
# _SELF_CLAIM_EVIDENCE_BOUNDARY_RE so the guard becomes self-healing instead
# of turn-killing.
#
# CP126 fa3d2a13. The previous wording asserted that the description "comes
# from my own state and self-model". That is a claim about PROVENANCE, and
# this function has no runtime evidence for it — the draft it is amending
# might rest on nothing of the kind. Because the amended text is then
# re-evaluated, a fabricated evidentiary sentence was what turned a rejected
# self-claim into accepted visible text: the salvage was manufacturing the
# very grounding whose absence caused the rejection.
#
# The suffix now states only a LIMIT, which is true regardless of what the
# draft above it says and requires no evidence to assert. It still satisfies
# the boundary check (which looks for "functional", "not proof",
# "phenomenal" and similar), so the guard remains self-healing without
# inventing support for the claim it is repairing.
#: Residuals where the USER asked for something specific and checkable and
#: did not get it — a word count, a sentence count, a number of paragraphs
#: or list items, a named reference, a clarifying choice, a follow-up
#: question.
#:
#: These belong in the deliverable set for the reason the rest of it exists:
#: a substantive answer that misses a formatting requirement beats a refusal,
#: and destroying the turn leaves the person with nothing. But they differ in
#: kind from the thinness verdicts around them. Thinness is our judgement
#: about the draft; this is the person's own stated instruction, unmet — and
#: delivering silently means they asked for five bullet points, received
#: three, and were never told which of the two happened.
#:
#: So: deliver, and say so. The reply survives and the shortfall is visible
#: to the one person who can decide whether it matters.
_REQUIREMENT_SHORTFALL_REASONS = frozenset(
    {
        "missing_requested_phrase",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_reference_value",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "empty_requested_list_item",
        "missing_requested_choice_clarification",
        "missing_requested_followup_question",
    }
)

_REQUIREMENT_SHORTFALL_LABELS = {
    "missing_requested_phrase": "include a phrase you asked for",
    "missing_requested_word_count": "hit the word count you asked for",
    "missing_requested_sentence_count": "hit the sentence count you asked for",
    "missing_requested_reference_value": "include a reference value you asked for",
    "missing_requested_paragraph_count": "hit the paragraph count you asked for",
    "missing_requested_list_count": "hit the number of list items you asked for",
    "empty_requested_list_item": "fill in every list item",
    "missing_requested_choice_clarification": "give you the choice you asked for",
    "missing_requested_followup_question": "end with the follow-up question you asked for",
}


def _requirement_shortfall_note(reasons: list[str]) -> str:
    """A one-line disclosure of the stated requirements this draft missed.

    Written in her voice and kept to one sentence: the answer is the point,
    and a paragraph of apology about formatting would bury it.
    """
    missed = [
        _REQUIREMENT_SHORTFALL_LABELS[reason]
        for reason in sorted(set(reasons))
        if reason in _REQUIREMENT_SHORTFALL_LABELS
    ]
    if not missed:
        return ""
    if len(missed) == 1:
        detail = missed[0]
    elif len(missed) == 2:
        detail = f"{missed[0]} or {missed[1]}"
    else:
        detail = f"{', '.join(missed[:-1])}, or {missed[-1]}"
    return f"\n\n(I did not {detail} — the answer above is what I have.)"


_SELF_CLAIM_BOUNDARY_SUFFIX = (
    " To be precise about what I can honestly claim here: this is a "
    "functional description of how I process and behave, and it is not "
    "proof of phenomenal experience — that is not something I can verify "
    "from the inside."
)


def _verify_contract_authority(job: dict[str, Any], contract_key: bytes | None) -> str:
    """Refusal reason for this job's privileged contract selection, or "".

    Thin wrapper so a missing contract_authority module can never take the
    worker down: the consistency half of the check (mutually exclusive
    output contracts) is reproduced locally, because resolving a
    contradiction by source order is a defect regardless of whether the
    authority layer is importable.
    """
    try:
        from core.brain.llm.contract_authority import verify_job

        return verify_job(job, contract_key)
    except ImportError as exc:
        _record_mlx_degradation(
            exc,
            action="contract authority unavailable; consistency check only",
            severity="error",
        )
        active = [
            name
            for name in (
                "strict_answer_contract",
                "strict_value_contract",
                "proof_evaluation_contract",
                "operator_evidence_contract",
            )
            if bool(job.get(name))
        ]
        if len(active) > 1:
            return "ambiguous_output_contract:" + ",".join(active)
        return ""


def _terminal_contract_refusal(
    job: dict[str, Any],
    response_text: Any,
    *,
    proof_evaluation_contract: bool = False,
    operator_evidence_contract: bool = False,
    model_continuation: Any = None,
) -> str:
    """The terminal contract this text FAILS, or "" if it passes them all.

    CP126 269ff364. Cancellation used to break out with the partial response
    as-is, ahead of proof completeness, operator-evidence merit and the
    capability-inventory grounding check. Those are refusals, not retries:
    skipping them let a preempted turn deliver exactly the content the
    normal terminal path exists to reject.

    Pure and side-effect free so it can be applied on the cancellation path,
    where retrying is not an option but refusing still is.
    """
    text = str(response_text or "")
    if not text.strip():
        return ""
    if proof_evaluation_contract and _proof_evaluation_fragment_incomplete(text):
        return "proof_fragment_incomplete"
    if operator_evidence_contract:
        if _operator_evidence_fragment_incomplete(text):
            return "operator_evidence_fragment_incomplete"
        # The delivered answer is scaffolding + continuation, and the fixed
        # scaffolding already contains every required evidence term and is
        # long enough to clear the word floor on its own. Handing this check
        # the COMBINED text made it inert on exactly this path: it measured
        # the prefix and passed. It has to see the model's own share.
        continuation = (
            text if model_continuation is None else str(model_continuation or "")
        )
        if _operator_evidence_model_contribution_insufficient(continuation):
            return "operator_evidence_model_contribution_insufficient"
    if bool(job.get("capability_inventory_contract", False)):
        grounded, _evidence = _capability_inventory_minimum_grounding(text)
        if not grounded:
            return "capability_inventory_ungrounded"
    return ""


def _shrink_scaffold_to_context_window(
    *,
    messages: Any,
    prompt: Any,
    tokens: list[int],
    window: int,
    output_reserve: int,
    tokenizer: Any,
    tools: Any,
) -> tuple[str, list[int], str]:
    """Trim SCAFFOLD until the prompt fits the model's real context window.

    The window was enforced as a hard refusal and nothing upstream bounded a
    prompt against it, so any lane that overshot failed every single time.
    Measured live: a background swarm-debate turn rendered 41,219 tokens for a
    32,768-token window, and 161,578 of its 171,058 characters were ONE system
    message — 94% scaffold around a 462-character request. The assembler's cap
    is a hard-coded character count with no relationship to the target model's
    window, so it passed a prompt the model could never accept.

    Only ``system`` messages are shortened, longest first, and never below a
    floor that keeps their opening instructions intact. User and assistant
    turns are never touched: dropping the actual request to make room for
    scaffold is the failure this exists to prevent. Returns
    ``(prompt, tokens, note)`` with an empty note when nothing was trimmed, and
    leaves the prompt untouched when the non-scaffold content alone cannot fit —
    there the honest outcome is still a refusal.
    """

    budget = window - output_reserve
    if budget <= 0 or len(tokens) <= budget or not isinstance(messages, list):
        return str(prompt or ""), tokens, ""

    def _render(candidate_messages: list[Any]) -> tuple[str, list[int]] | None:
        try:
            from core.brain.llm.chat_format import system_first

            rendered = tokenizer.apply_chat_template(
                system_first(candidate_messages),
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:  # noqa: BLE001 - a failed trim must never kill the worker
            # A template refusing a message list is a caller's mistake, and
            # the cost of it here is the whole model process.
            #
            # LIVE 2026-08-19: jinja2.TemplateError("System message must be at
            # the beginning") is not an AttributeError, RuntimeError,
            # TypeError or ValueError, so it went straight past this guard,
            # out of the worker loop, and killed the worker mid-generation.
            # The crash-loop breaker then took the lane down and the person
            # got a refusal, three times over, while she was mid-game.
            #
            # Failing to trim is a recoverable outcome: the caller keeps the
            # untrimmed prompt and finds out it is too long, which is a far
            # smaller problem than having no model.
            return None
        try:
            return str(rendered), list(tokenizer.encode(str(rendered)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    system_positions = [
        index
        for index, msg in enumerate(messages)
        if isinstance(msg, dict)
        and str(msg.get("role", "")).strip().lower() == "system"
        and str(msg.get("content", "") or "").strip()
    ]
    if not system_positions:
        return str(prompt or ""), tokens, ""

    working = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
    trimmed_note_parts: list[str] = []
    # Chars-per-token measured on THIS prompt rather than assumed: a scaffold of
    # JSON dumps and one of prose have very different ratios, and guessing 4.0
    # either under-trims (and refuses anyway) or over-trims real instructions.
    chars_per_token = max(1.0, len(str(prompt or "")) / max(1, len(tokens)))
    floor_chars = 1200

    for _pass in range(len(system_positions) + 1):
        overflow_tokens = len(tokens) - budget
        if overflow_tokens <= 0:
            break
        # 10% headroom: template overhead and tokenizer merges mean the retained
        # slice never costs exactly the predicted number of tokens.
        need_chars = int(overflow_tokens * chars_per_token * 1.1) + 64
        target = max(
            system_positions,
            key=lambda index: len(str(working[index].get("content", "") or "")),
        )
        content = str(working[target].get("content", "") or "")
        keep = max(floor_chars, len(content) - need_chars)
        if keep >= len(content):
            break
        shortened = (
            content[:keep]
            + "\n\n[... scaffold trimmed to fit the model's context window ...]"
        )
        working[target]["content"] = shortened
        trimmed_note_parts.append(f"system[{target}] {len(content)}->{len(shortened)} chars")
        rendered = _render(working)
        if rendered is None:
            return str(prompt or ""), tokens, ""
        prompt, tokens = rendered

    if len(tokens) > budget:
        # Nothing safe left to shed — the request itself does not fit. Refusing
        # is correct; silently deleting the user's turn is not.
        return str(prompt or ""), tokens, ""

    return str(prompt or ""), tokens, "; ".join(trimmed_note_parts)


def _salvage_exhausted_user_surface(
    job: dict[str, Any],
    response_text: Any,
    rejection_reasons: list[str],
) -> tuple[str, list[str], list[str]]:
    """Best honest draft after quality-gate retries are exhausted.

    Returns (text, residual_reasons, applied_repairs); empty text means
    nothing was safely deliverable and the caller keeps the fail-closed
    empty reply. Every deterministic amendment is named in
    ``applied_repairs`` so the caller can disclose it as a text mutation —
    scaffolding must never pass as model output silently.
    """
    draft = str(response_text or "").strip()
    if len(draft) < 40:
        return "", list(rejection_reasons), []

    reasons = list(rejection_reasons)
    applied_repairs: list[str] = []
    if "missing_self_claim_evidence_boundary" in reasons:
        amended = draft + _SELF_CLAIM_BOUNDARY_SUFFIX
        amended_reasons = _surface_quality_failure_reasons(job, amended)
        if "missing_self_claim_evidence_boundary" not in amended_reasons:
            draft = amended
            reasons = list(amended_reasons)
            applied_repairs.append("self_claim_boundary_suffix")

    if not reasons:
        return draft, [], applied_repairs
    if set(reasons) <= _DELIVERABLE_RESIDUAL_SURFACE_REASONS:
        # Delivered — and where the person stated a checkable requirement
        # that this draft does not meet, they are told. Silently returning a
        # three-item list to someone who asked for five leaves them unable
        # to tell a shortfall from a decision.
        note = _requirement_shortfall_note(reasons)
        if note:
            draft = f"{draft}{note}"
            applied_repairs.append("requirement_shortfall_disclosure")
        return draft, reasons, applied_repairs
    return "", reasons, applied_repairs


def _repair_live_user_surface_self_claims(response_text: Any) -> str:
    """Keep the diagnostic API without using it in the worker decode path.

    Older diagnostics import this helper directly. Worker-owned quality control
    may reject an unsupported claim, but it cannot substitute canned prose for
    an authored candidate.
    """

    text = str(response_text or "").strip()
    if not text:
        return text
    try:
        from core.conversation.self_claim_verifier import repair_self_claim_surface

        return repair_self_claim_surface(text)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued with unmodified draft after self-claim analysis failed",
            severity="error",
        )
        return text


def _repair_live_user_surface_instruction_shape(
    job: dict[str, Any],
    response_text: Any,
) -> str:
    """Apply deterministic explicit-format repairs before spending another decode."""

    text = str(response_text or "").strip()
    prompt = _surface_validation_prompt(job)
    if not text or not prompt:
        return text
    try:
        from core.conversation.response_reliability import repair_instruction_shape

        return repair_instruction_shape(prompt, text)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued to quality validation after instruction-shape repair failed",
            severity="warning",
        )
        return text


def _exact_reply_token_requirement(
    job: dict[str, Any],
    tokenizer: Any,
) -> tuple[int, int]:
    """Measure exact content plus the one token needed to terminate decoding."""

    contract = job.get("requested_output_contract")
    if not isinstance(contract, dict) or not bool(contract.get("exact_reply", False)):
        return 0, 0
    prompt = _surface_validation_prompt(job)
    if not prompt:
        return 0, 0
    try:
        from core.conversation.response_reliability import requested_exact_reply_target

        target = requested_exact_reply_target(prompt)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return 0, 0
    if not target:
        return 0, 0
    try:
        token_ids = tokenizer.encode(target, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(target)
    except (AttributeError, RuntimeError, ValueError):
        return 0, 0
    token_count = len(token_ids or [])
    if token_count <= 0:
        return 0, 0
    return token_count, token_count + 1


def _record_exact_reply_token_evidence(
    job: dict[str, Any],
    tokenizer: Any,
    *,
    generation_max_tokens: int,
    hard_output_token_ceiling: int,
) -> None:
    """Record selected-tokenizer fit without expanding admitted ceilings."""

    token_count, token_requirement = _exact_reply_token_requirement(job, tokenizer)
    if token_requirement <= 0:
        return
    admitted_generation_cap = max(1, int(generation_max_tokens))
    effective_native_cap = admitted_generation_cap
    if hard_output_token_ceiling > 0:
        effective_native_cap = min(effective_native_cap, hard_output_token_ceiling)
    required_termination_headroom = max(0, token_requirement - token_count)
    available_termination_headroom = max(0, effective_native_cap - token_count)
    job["exact_reply_token_count"] = token_count
    job["exact_reply_required_termination_headroom"] = required_termination_headroom
    job["exact_reply_available_termination_headroom"] = available_termination_headroom
    job["exact_reply_content_capacity_sufficient"] = bool(
        effective_native_cap >= token_count
    )
    job["exact_reply_termination_headroom_sufficient"] = bool(
        available_termination_headroom >= required_termination_headroom
    )
    job["exact_reply_native_capacity_sufficient"] = bool(
        effective_native_cap >= token_requirement
    )
    job["exact_reply_token_ceiling_valid"] = bool(
        hard_output_token_ceiling <= 0
        or hard_output_token_ceiling >= token_requirement
    )
    if effective_native_cap < token_requirement:
        logger.info(
            "Exact target requires %d selected-tokenizer slots but the immutable "
            "generation envelope admits %d; deterministic exact-output repair owns "
            "the visible contract.",
            token_requirement,
            effective_native_cap,
        )


def _normalize_surface_format(response_text: Any) -> str:
    """Whitespace-only repair of jammed list markers and welded sentences."""
    text = str(response_text or "")
    if not text.strip():
        return ""
    try:
        from core.conversation.response_reliability import normalize_user_facing_format

        return normalize_user_facing_format(text)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Surface format normalisation skipped: %s", exc)
        return ""


def _repair_live_user_surface_escaped_newlines(response_text: Any) -> str:
    """Turn literal \\n / \\t / \\r the model typed back into real whitespace.

    A local model that has read a lot of JSON sometimes emits the two-character
    sequence backslash-n where it meant a newline. The reply is otherwise fine,
    and rejecting it costs the person a correct answer over a typo the runtime
    can fix deterministically.

    LIVE DEFECT, 2026-07-26: a correct, well-structured marble derivation was
    rejected with reasons=escaped_control_artifact and the user got "I couldn't
    get to an answer I'd stand behind on that one".

    Prose only — a fenced code block may legitimately contain a literal \\n,
    and rewriting it would corrupt the code. So the fences are held out and
    the prose between them is repaired, rather than abandoning the whole reply
    because part of it is code.

    LIVE DEFECT, 2026-08-18: "write a python function to reverse a string and
    then explain how it works" took 112 seconds and returned "I couldn't get
    to an answer I'd stand behind on that one". The draft was rejected as
    escaped_control_artifact and this repair declined to run because the reply
    contained a code fence — so every request that wants code AND prose was
    unanswerable whenever the model typed one backslash-n in the explanation.
    """
    from core.conversation.escaped_controls import (
        repair_escaped_whitespace_artifacts,
    )

    repaired = repair_escaped_whitespace_artifacts(response_text)
    return repaired.strip() if repaired is not None else ""


def _repair_escaped_whitespace_in_prose(text: str) -> str | None:
    """Compatibility wrapper around the shared syntax-aware repair."""
    from core.conversation.escaped_controls import (
        repair_escaped_whitespace_artifacts,
    )

    return repair_escaped_whitespace_artifacts(text)


def _repair_live_user_surface_truncated_tail(response_text: Any) -> str:
    """Keep complete model-derived content when only the final tail is clipped."""

    text = str(response_text or "").strip()
    if len(text) < 80 or len(text.split()) < 12:
        return ""
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?](?=(?:\s|$|\d))", text)
    ]
    for end in reversed(sentence_ends):
        candidate = text[:end].strip()
        if re.search(r"(?:^|\s)\d+\.$", candidate):
            continue
        if len(candidate) < 80 or len(candidate.split()) < 12:
            continue
        return candidate
    # A worked derivation is not sentences. Live 2026-07-26, the marble answer
    # came back as a bulleted derivation with no full stop after the opening
    # line, so sentence-based trimming found exactly one candidate ("Let's
    # break it down.", too short) and gave up — and a mostly complete, correct
    # answer became a refusal. When the body is line-structured, drop the
    # clipped final line and keep the complete ones.
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        candidate = "\n".join(lines[:-1]).strip()
        if len(candidate) >= 80 and len(candidate.split()) >= 12:
            return candidate
    return ""


_LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION = (
    "For live status questions, name at least one concrete observable runtime "
    "or sensory signal such as CPU/RAM pressure, temperature, network state, "
    "desktop access, screen/audio/camera state, heartbeat, Cortex/MLX worker "
    "state, or an actual numeric sensor reading. Avoid metaphor-only "
    "attention-texture language."
)
_SELF_CONDITION_SIGNAL_INSTRUCTION = (
    "This is a question about Aura's own condition. Answer directly from the "
    "supplied affect, welfare, felt-coherence, continuity, agency, and freshness "
    "evidence. CPU, RAM, host load, and availability are supporting body context "
    "only and must not replace the condition answer."
)
def _job_needs_concrete_status_signal_guidance(job: dict[str, Any]) -> bool:
    if not bool(job.get("clean_user_surface_contract", False)):
        return False
    prompt = _surface_validation_prompt(job)
    if not prompt:
        return False
    prompt_l = prompt.lower()
    if re.search(r"\b(?:capabilities|externally|tools?|what\s+can\s+you\s+do)\b", prompt_l):
        return False
    try:
        from core.conversation.response_reliability import (
            is_operational_status_turn,
            is_self_condition_turn,
            is_status_check_turn,
        )

        if is_self_condition_turn(prompt):
            return False
        if is_operational_status_turn(prompt) or is_status_check_turn(prompt):
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return any(
        marker in prompt_l
        for marker in (
            "live runtime signal",
            "live path",
            "runtime status",
            "with me",
            "you there",
        )
    )


def _with_initial_user_surface_guidance(
    messages: Any,
    prompt: Any,
    job: dict[str, Any],
) -> tuple[Any, Any]:
    # ``mlx_client`` deliberately gives health probes the clean-surface flag
    # so they inherit the safe steering/recurrent clamps. That flag describes
    # decode controls, not audience. Appending conversational guidance to the
    # readiness prompt changed ``Reply exactly: ready`` into two competing
    # instructions and made a healthy resident lane fail boot deterministically.
    # Keep control-plane measurements clamped, but never prompt-shape them as
    # user prose.
    if bool(job.get("health_probe", False)) or not _job_needs_concrete_status_signal_guidance(
        job
    ):
        return messages, prompt
    guidance = _LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION
    if isinstance(messages, list):
        guided_messages = copy.deepcopy(messages)
        for message in guided_messages:
            if isinstance(message, dict) and str(message.get("role") or "").lower() == "system":
                content = str(message.get("content") or "").rstrip()
                message["content"] = (
                    f"{content}\n{guidance}"
                    if content
                    else guidance
                )
                return guided_messages, prompt
        guided_messages.insert(
            0,
            {"role": "system", "content": guidance},
        )
        return guided_messages, prompt
    prompt_text = str(prompt or "").rstrip()
    if not prompt_text:
        return messages, guidance
    return messages, f"{prompt_text}\n\n{guidance}"


def _repair_live_user_surface_operational_status(
    response_text: Any,
    rejection_reasons: list[str],
    job: dict[str, Any],
) -> str:
    if not rejection_reasons or not set(rejection_reasons).issubset(
        {"too_thin_for_operational_status_turn", "too_thin_for_status_turn"}
    ):
        return str(response_text or "")
    if not _job_needs_concrete_status_signal_guidance(job):
        return str(response_text or "")
    try:
        from core.runtime import resource_psutil as psutil

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        cpu_percent = psutil.cpu_percent(interval=None)
        return (
            "I am with you. One live runtime signal I can perceive is RAM "
            f"pressure at {memory.percent:.1f}% with {available_gb:.1f} GB "
            f"available; CPU load is {cpu_percent:.1f}% on this host."
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        load_1m: float | None = None
        try:
            from core.runtime.resource_observation import get_resource_observer

            load_1m = float(get_resource_observer().compute().load_1m)
            if not math.isfinite(load_1m):
                load_1m = None
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            load_1m = None
        if load_1m is not None:
            return (
                "I am with you. One live runtime signal I can perceive is the host "
                f"load average at {load_1m:.2f}, with the Cortex/MLX worker active "
                "for this foreground turn."
            )
        # BOTH probes failed: fabricating a 0.00 load sample converted an
        # unavailable health probe into a confident live signal. Say what is
        # actually true instead.
        return (
            "I am with you. My host telemetry probes are not answering right "
            "now, so I cannot quote a live load number this turn — the reply "
            "lane itself is working, which is the one signal I can honestly "
            "attest."
        )


# What the model should actually DO about each rejection reason.
#
# A retry used to be handed the raw reason name and nothing else — "failed for:
# generic_memory_pin_acknowledgement" — which is an internal token, not an
# instruction. Measured live: the same draft was rejected three times with the
# identical reason AND the identical validation hash, burning the turn's whole
# budget on regenerating the same mistake, until "Request deadline reached at
# token 23" ended it and the person got a refusal. A retry that does not say
# what to fix is a wasted decode, and wasted decodes are what kill the turn.
_SURFACE_RETRY_INSTRUCTIONS: dict[str, str] = {
    "backend_symbolic_surface_leak": (
        "Restate the same substantive answer in natural language without raw backend "
        "enum names, internal variable names, or control identifiers."
    ),
    "corrupted_language": (
        "Regenerate the complete answer in clean grammatical language. Do not copy any "
        "malformed token from the rejected draft."
    ),
    "telemetry_path_wall": (
        "Answer the request without dumping internal telemetry paths. Include a path only "
        "when the user asked for that specific path and it is necessary to the answer."
    ),
    "unbounded_numeric_identifier": (
        "Do not expose an unexplained internal numeric identifier. Preserve a long exact "
        "number only when the user requested it or it is necessary to the answer."
    ),
    "generic_memory_pin_acknowledgement": (
        "The user asked you to remember something AND asked something else in the "
        "same turn. State the exact value you are keeping, then answer the rest of "
        "the turn in full — a bare acknowledgement is not a reply."
    ),
    "truncated_tail": (
        "End on a complete sentence. If the room is tight, say less and finish the "
        "thought rather than stopping mid-clause."
    ),
    "reliability_diagnostic_too_thin": (
        "Name the concrete mechanism — what happens, in what order, and what it "
        "costs — instead of reassurance."
    ),
    "too_thin_for_reliability_turn": (
        "Name the concrete mechanism and its consequence, not a summary judgement."
    ),
    "reliability_diagnostic_deflection": (
        "Do not deflect. Say what the actual cause or consequence is, plainly."
    ),
    "too_thin_for_user_turn": (
        "Say more than one clause: take a position and give the reason for it."
    ),
    "too_thin_for_open_ended_turn": (
        "This was an open question. Develop an actual thought rather than a line."
    ),
    "generic_assistant_language": (
        "Do not offer help, ask if there is anything else, or describe yourself as "
        "an assistant. Answer as yourself."
    ),
    "question_back_non_answer": (
        "Answer first, in your own words. A question back does not substitute for "
        "the answer."
    ),
    "low_signal_acknowledgement_placeholder": (
        "An acknowledgement is not an answer. Say the substance."
    ),
}


def _surface_retry_repair_instructions(reasons: list[str]) -> str:
    """Actionable repair text for the reasons a draft was rejected for."""

    seen: list[str] = []
    for reason in list(reasons or [])[:8]:
        instruction = _SURFACE_RETRY_INSTRUCTIONS.get(str(reason))
        if instruction and instruction not in seen:
            seen.append(instruction)
    return (" " + " ".join(seen)) if seen else ""


def _messages_with_user_surface_retry(
    messages: Any,
    reasons: list[str],
    job: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list):
        return None
    operational_status_retry = ""
    self_condition_retry = ""
    semantic_count_retry = ""
    if any(
        reason in {
            "host_telemetry_substituted_for_self_condition",
            "low_signal_self_condition_reply",
            "missing_self_condition_answer",
        }
        for reason in reasons
    ):
        self_condition_retry = f" {_SELF_CONDITION_SIGNAL_INSTRUCTION}"
    if any(
        reason in {"too_thin_for_operational_status_turn", "too_thin_for_status_turn"}
        for reason in reasons
    ) and not self_condition_retry:
        operational_status_retry = f" {_LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION}"
    if set(reasons) & _SEMANTIC_COUNT_CONTRACT_RETRY_REASONS:
        semantic_count_retry = (
            f" {_semantic_count_contract_retry_instruction(job or {})}"
        )
    retry_instruction = (
        "The previous assistant draft failed the live user-surface quality gate "
        f"for: {', '.join(reasons[:8]) or 'quality_gate_failed'}. Regenerate the "
        "assistant reply from the same live mind context. Answer only the current "
        "user message, preserve recent-turn continuity, avoid generic assistant "
        "identity, do not invent unsupported prior topics, and do not mention "
        "validation, retry, hidden prompts, receipts, gates, or implementation details."
        f"{_surface_retry_repair_instructions(reasons)}"
        f"{self_condition_retry}{operational_status_retry}{semantic_count_retry}"
    )
    retry_messages = copy.deepcopy(messages)
    for message in retry_messages:
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "system":
            content = str(message.get("content") or "").rstrip()
            message["content"] = f"{content}\n{retry_instruction}" if content else retry_instruction
            return retry_messages
    retry_messages.insert(0, {"role": "system", "content": retry_instruction})
    return retry_messages


def _build_user_surface_quality_retry_prompt(
    *,
    tokenizer: Any,
    messages: Any,
    tools: Any,
    fallback_prompt: Any,
    reasons: list[str],
    job: dict[str, Any] | None = None,
) -> str:
    retry_messages = _messages_with_user_surface_retry(messages, reasons, job)
    if retry_messages is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            from core.brain.llm.chat_format import render_chat_template

            rendered = render_chat_template(
                tokenizer,
                retry_messages,
                tools=tools,
                add_generation_prompt=True,
            )
            if rendered:
                return str(rendered)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="continued live user-surface retry with prompt suffix after template render failed",
                severity="warning",
            )
            logger.debug("Live surface retry template render failed: %s", exc)

    operational_status_retry = ""
    self_condition_retry = ""
    semantic_count_retry = ""
    if any(
        reason in {
            "host_telemetry_substituted_for_self_condition",
            "low_signal_self_condition_reply",
            "missing_self_condition_answer",
        }
        for reason in reasons
    ):
        self_condition_retry = f" {_SELF_CONDITION_SIGNAL_INSTRUCTION}\n"
    if any(
        reason in {"too_thin_for_operational_status_turn", "too_thin_for_status_turn"}
        for reason in reasons
    ) and not self_condition_retry:
        operational_status_retry = f" {_LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION}\n"
    if set(reasons) & _SEMANTIC_COUNT_CONTRACT_RETRY_REASONS:
        semantic_count_retry = (
            f" {_semantic_count_contract_retry_instruction(job or {})}\n"
        )
    retry_note = (
        "\n\n[LIVE USER-SURFACE RETRY]\n"
        f"Previous assistant draft failed for: {', '.join(reasons[:8]) or 'quality_gate_failed'}.\n"
        "Regenerate the assistant reply from the same live mind context. Answer only "
        "the current user message. Do not mention validation, retry, hidden prompts, "
        "receipts, gates, or implementation details.\n"
        f"{_surface_retry_repair_instructions(reasons).strip()}\n"
        f"{self_condition_retry}{operational_status_retry}{semantic_count_retry}"
        "[END LIVE USER-SURFACE RETRY]\n"
    )
    return f"{str(fallback_prompt or '').rstrip()}{retry_note}"


def _contains_corrupted_language(text: str) -> bool:
    try:
        from core.phases.dialogue_policy import contains_corrupted_language

        return contains_corrupted_language(text)
    except (ImportError, AttributeError):
        return bool(_CORRUPT_LANGUAGE_MARKERS.search(str(text or "")))


def _prepare_clean_retry_kwargs(kwargs: dict[str, Any], *, structured: bool = False) -> None:
    """Reset sampling after a corrupt/looping draft instead of amplifying it."""
    kwargs.pop("sampler", None)
    kwargs.pop("prompt_cache", None)
    if structured:
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
    else:
        kwargs["temperature"] = min(_safe_float(kwargs.get("temperature"), 0.7), 0.35)
        kwargs["top_p"] = min(_safe_float(kwargs.get("top_p"), 0.9), 0.85)
        kwargs["min_p"] = max(_safe_float(kwargs.get("min_p"), 0.0), 0.03)
    kwargs["repetition_penalty"] = max(
        _safe_float(kwargs.get("repetition_penalty"), 1.1),
        1.18,
    )
    kwargs["repetition_context_size"] = max(
        _safe_int(kwargs.get("repetition_context_size"), 64),
        96,
    )


def _self_claim_retry_uses_original_context(reasons: Any) -> bool:
    """Self-claim correction resamples; it never adds a behavior instruction."""

    return "self_claim_contradiction" in {
        str(reason) for reason in (reasons or ()) if str(reason)
    }


def _surface_retry_is_futile(reasons: Any) -> bool:
    """Return whether regeneration cannot repair the failed contract.

    Prompt provenance and self-claim verification are external integrity
    dependencies. Asking the model for another answer cannot restore either
    dependency, so retrying only adds latency and risks replacing useful text.
    """

    normalized = {str(reason) for reason in (reasons or ()) if str(reason)}
    return "self_claim_verification_unavailable" in normalized or any(
        reason.startswith("surface_validation_prompt_binding")
        or reason == "surface_validation_prompt_missing"
        for reason in normalized
    )


def _surface_retry_wall_exceeded(started_monotonic: float, wall_s: float) -> bool:
    """True when the user-surface gate-retry path has burned its wall budget.

    Under memory-contended decode each drafting attempt costs 30-70s; burning
    the full retry budget is how a single live turn reached 200s+ (July 8
    soak). Past the wall, exhaustion salvage delivers the best honest draft
    instead of drafting again for a user who has stopped waiting. Floor of
    10s so a misconfigured env value can never disable first-attempt retries.
    """
    if started_monotonic <= 0.0:
        return False
    return (time.monotonic() - started_monotonic) > max(10.0, wall_s)


def _ontology_retry_permitted(
    *,
    internal_attempt: int,
    max_internal_retries: int,
    ontology_retry_count: int,
    job_deadline_unix: float,
    user_surface: bool,
    surface_retry_started: float,
    surface_retry_wall_s: float,
    now_unix: float | None = None,
) -> tuple[bool, bool, bool]:
    """Allow one ontology repair only while caller time budgets remain open."""
    now = time.time() if now_unix is None else float(now_unix)
    deadline_open = job_deadline_unix <= 0.0 or now < job_deadline_unix
    retry_wall_open = (
        not user_surface
        or not _surface_retry_wall_exceeded(
            surface_retry_started,
            surface_retry_wall_s,
        )
    )
    allowed = (
        int(internal_attempt) < int(max_internal_retries)
        and int(ontology_retry_count) < 1
        and deadline_open
        and retry_wall_open
    )
    return allowed, deadline_open, retry_wall_open


def _expand_user_surface_retry_budget(
    kwargs: dict[str, Any],
    reasons: list[str],
    *,
    ceiling: int = 2048,
    hard_ceiling: Any = None,
) -> bool:
    """Give a clipped live reply one larger pass on the existing worker.

    This is deliberately limited to structural truncation. It does not create
    another model process, alter strict/proof contracts, or inflate retries for
    off-topic and low-quality drafts.
    """

    if "truncated_tail" not in set(reasons):
        return False
    current = max(
        _safe_int(kwargs.get("max_tokens"), 0),
        _safe_int(kwargs.get("num_predict"), 0),
    )
    if current <= 0:
        return False
    expansion_ceiling = max(current, int(ceiling))
    immutable_ceiling = _safe_int(hard_ceiling, 0)
    if immutable_ceiling > 0:
        expansion_ceiling = min(expansion_ceiling, immutable_ceiling)
    expanded = min(max(current * 2, current + 384), expansion_ceiling)
    if expanded <= current:
        return False
    kwargs["max_tokens"] = expanded
    if "num_predict" in kwargs:
        kwargs["num_predict"] = expanded
    return True


def _telemetry_sanitization_failure_reasons(
    text: str,
    is_proof: bool = False,
) -> list[str]:
    """Identify fatal surface leakage without destroying the authored draft."""
    if not text:
        return []

    reasons: list[str] = []

    # 1) Reject telemetry-path walls without blocking legitimate code, regex,
    # filesystem, or proof output. The old slash-count heuristic rejected any
    # answer with more than 15 "/" characters, which is common in live coding
    # tasks and path-aware proof/eval runs.
    # The path-wall check stays non-proof: coding and path-aware eval answers
    # legitimately contain many paths, and unlike the symbolic markers there
    # is no exact token that separates a wall from real content.
    if not is_proof:
        slash_count = text.count("/")
        if slash_count > 30 and "http" not in text.lower():
            path_like = re.findall(r"(?:/[A-Za-z0-9._-]+){3,}", text)
            path_chars = sum(len(path) for path in path_like)
            if len(path_like) >= 3 or path_chars > max(120, int(len(text) * 0.35)):
                reasons.append("telemetry_path_wall")

    # 3) Extreme numeric sequences: in conversational output a 20+ digit run
    # is a hallucination signature. Proof/eval answers are exempt — large
    # integers, hashes, and numeric test vectors are legitimate exact
    # answers there, and correctness is the eval harness's job to score.
    if not is_proof and re.search(r'\d{20,}', text):
        reasons.append("unbounded_numeric_identifier")

    # 4) Corrupted lexical output is a model-state failure, not a usable
    # answer — in EVERY mode. A proof answer containing corruption tokens is
    # corrupted evidence, so the proof exemption never applied here.
    if _contains_corrupted_language(text):
        reasons.append("corrupted_language")
    # 5) Backend-symbolic surface markers apply in EVERY mode. The exemption
    # here was justified by a pattern that no longer exists: it claimed the
    # regex matched common English words ("proceeding", "field coherence"),
    # but the markers are exact-case backend identifiers — PROCEEDING,
    # TOOL_ACTION, ExistenceHash — matched WITHOUT re.IGNORECASE, so
    # lowercase prose never matched, and "field coherence" is not in this
    # pattern at all. A proof answer containing a raw backend action code is
    # leaked internals wherever it appears.
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(text):
        reasons.append("backend_symbolic_surface_leak")

    return list(dict.fromkeys(reasons))


def _sanitize_telemetry_leakage(text: str, is_proof: bool = False) -> str | None:
    """Legacy strict-path adapter for the typed telemetry sanitizer.

    Strict/proof callers still receive ``None`` for an unspeakable draft. Live
    user surfaces consume the typed reasons through the quality-repair lane so
    the original draft remains available for bounded authored correction.
    """
    if _telemetry_sanitization_failure_reasons(text, is_proof=is_proof):
        return None

    return text


def _route_telemetry_sanitizer_draft(
    text: str,
    *,
    is_proof: bool,
    authored_surface_repair_available: bool,
) -> tuple[str, list[str]]:
    """Keep an unspeakable live draft only when a bounded repair lane owns it."""
    reasons = _telemetry_sanitization_failure_reasons(text, is_proof=is_proof)
    if reasons and not authored_surface_repair_available:
        return "", reasons
    return text, reasons


def _route_cooperative_partial_draft(
    job: dict[str, Any],
    text: str,
    surface_control_state: dict[str, Any],
    *,
    is_proof: bool,
) -> str:
    """Route a deadline/cancel partial through the same owned surface lane.

    Cooperative termination skips model retries, but it must not skip draft
    custody. A live user-surface draft that trips the telemetry sanitizer is
    carried as rejected evidence for the caller's bounded recovery path. A
    strict, proof, or non-user-surface draft has no such owner and remains
    withheld.
    """

    authored_surface_repair_available = _surface_quality_gate_enabled(job)
    routed, reasons = _route_telemetry_sanitizer_draft(
        text,
        is_proof=is_proof,
        authored_surface_repair_available=authored_surface_repair_available,
    )
    bounded_reasons = reasons[:8]
    surface_control_state["telemetry_sanitizer_reasons"] = bounded_reasons
    if bounded_reasons and authored_surface_repair_available:
        existing = surface_control_state.get("surface_quality_gate_reasons")
        merged = [
            str(reason).strip()[:120]
            for reason in (
                list(existing) if isinstance(existing, (list, tuple)) else []
            )
            if str(reason).strip()
        ]
        merged.extend(bounded_reasons)
        surface_control_state["surface_quality_gate_passed"] = False
        surface_control_state["surface_quality_gate_reasons"] = list(
            dict.fromkeys(merged)
        )[:8]
        _remember_surface_quality_rejected_draft(
            surface_control_state,
            text,
            merged,
        )
    return routed


def _surface_rejected_draft_rank(text: Any, reasons: list[str]) -> tuple[int, ...]:
    """Rank suppressed drafts by servability evidence, never arrival time."""

    from core.conversation.surface_disposition import UNSPEAKABLE_REASONS

    body = str(text or "").strip()
    normalized = [str(reason or "").strip() for reason in reasons if str(reason or "").strip()]
    completion = {
        "final_answer_missing",
        "incomplete_code_response",
        "missing_final_answer",
        "truncated_tail",
        "unanswered_question_part",
    }
    unspeakable = sum(reason in UNSPEAKABLE_REASONS for reason in normalized)
    semantic = sum(reason not in completion for reason in normalized)
    incomplete = sum(reason in completion for reason in normalized)
    return (-unspeakable, -semantic, -incomplete, min(len(body), 8_000))


def _remember_surface_quality_rejected_draft(
    state: dict[str, Any],
    text: Any,
    reasons: list[str],
) -> None:
    """Keep the best rejected draft across worker-owned retries."""

    body = str(text or "").strip()[:8_000]
    if not body:
        return
    rank = _surface_rejected_draft_rank(body, reasons)
    current_rank = state.get("_surface_quality_rejected_rank")
    if not isinstance(current_rank, tuple) or rank > current_rank:
        state["_surface_quality_rejected_rank"] = rank
        state["surface_quality_rejected_text"] = body
        state["surface_quality_rejected_reasons"] = list(reasons)[:8]


_ARTIFACT_REQUEST_RE = re.compile(
    r"```(?:python|json|csv|yaml|toml|sql|html|css|javascript|typescript)?"
    r"|code block|return only(?: the)? complete|return the fixed config"
    r"|return the code|return the .*csv|return .*json|rulescript\.py"
    r"|service_config|reconciled data as a csv|select_values\.py",
    re.IGNORECASE | re.DOTALL,
)

_OPERATOR_EVIDENCE_PREFIX = (
    "Operationally, Aura should set an objective, use governed tool actions, "
    "keep each receipt and trace, stop when blocked or unsafe, and treat the "
    "result as evidence of bounded software operation rather than personhood proof. "
)


def _proof_prompt_expects_artifact(text: str) -> bool:
    return bool(_ARTIFACT_REQUEST_RE.search(str(text or "")))


# CP126 007c5cd3. Any proof prompt NOT matched here is rewritten to demand
# "3-6 complete sentences" and to avoid numbered lists — a shape that can
# directly contradict the contract being evaluated. The narrower this
# detector is, the more often the default overrides a task that did state
# its own form, so it covers the ways a task actually declares shape:
# exact and bounded counts, structured serialisations, code, brevity and
# length requests, and explicit schema/format directives.
_EXPLICIT_FORMAT_REQUEST_RE = re.compile(
    r"(?:\bin\s+(?:one|two|a\s+single)\s+(?:word|sentence|line|number|paragraph)s?\b"
    r"|\bexactly\s+\d+\s+(?:words?|sentences?|lines?|items?|bullets?|paragraphs?)\b"
    # Bounded counts: "at most 3 sentences", "no more than 2 lines",
    # "up to five bullets", "in 2-4 sentences".
    r"|\b(?:at\s+most|no\s+more\s+than|fewer\s+than|less\s+than|up\s+to|within)\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:words?|sentences?|lines?|items?|bullets?|paragraphs?)\b"
    r"|\bin\s+\d+\s*[-\u2013]\s*\d+\s+(?:words?|sentences?|lines?|paragraphs?)\b"
    r"|\b(?:as|in)\s+a\s+(?:markdown\s+)?table\b"
    r"|\bbullet(?:ed)?\s+(?:list|points?)\b"
    r"|\bnumbered\s+(?:list|steps?)\b"
    r"|\bone[- ]word\s+answer\b"
    r"|\bonly\s+the\s+(?:number|value|answer|word)\b"
    r"|\banswer\s+with\s+(?:only|just|a\s+single)\b"
    r"|\brespond\s+with\s+(?:only|just|a\s+single)\b"
    # Structured serialisations and code carry their own grammar; prose
    # sentence counts are meaningless for them.
    r"|\b(?:as|in|return|output|emit|produce|reply\s+with)\s+"
    r"(?:valid\s+|raw\s+|pure\s+)?(?:json|jsonl|yaml|toml|xml|csv|tsv|sql|"
    r"markdown|html|svg|diff|patch)\b"
    r"|\b(?:code|fenced)\s+block\b"
    r"|```"
    # Brevity and length directives.
    r"|\b(?:be\s+)?(?:brief|concise|terse|succinct)\b"
    r"|\bin\s+(?:a\s+)?(?:short|single|brief)\s+(?:sentence|line|phrase|paragraph)\b"
    r"|\bshort\s+answer\b"
    # Explicit schema/format directives.
    r"|\bformat\s*:"
    r"|\bschema\s*:"
    r"|\boutput\s+format\b"
    r"|\bfollow(?:ing)?\s+(?:this|the)\s+(?:format|schema|template|structure)\b)",
    re.IGNORECASE,
)


def _proof_prompt_declares_format(text: str) -> bool:
    """True when the task states its own output shape — exact counts,
    tables, lists, single values — which must override the generic
    3-6 sentence proof rendering default."""
    return bool(_EXPLICIT_FORMAT_REQUEST_RE.search(str(text or "")))


def _strip_leading_chatml_prefix(text: str) -> str:
    cleaned = str(text or "")
    prefixes = (
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<｜Assistant｜>",
        "Assistant:",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip("\n")
                changed = True
    return cleaned


_ROLE_CONTINUATION_RE = re.compile(
    r"(?is)(?:<\|im_end\|>\s*)?<\|im_start\|>\s*"
    r"(?:user|human|system|assistant|aura)\b.*$"
    r"|(?:^|\n|(?<=[.!?]))\s*(?:User|Human|System|Assistant|Aura)\s*[:：].*$"
)
_LEADING_GENERATION_ROLE_RE = re.compile(
    r"^\s*(?:<\|im_start\|>\s*)?(?:User|Human|Assistant|Aura|System)\s*[:：]\s*",
    re.IGNORECASE,
)
_LEADING_ROLE_NO_SEPARATOR_RE = re.compile(
    r"^\s*(?:user|human|assistant|system)(?=(?:i['’]?m\b|i\b|you\b|"
    r"what\b|who\b|when\b|where\b|why\b|how\b|yes\b|no\b|the\b))",
    re.IGNORECASE,
)
_USER_CONTINUATION_NO_COLON_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:User|Human)\s+"
    r"(?=(?:what|who|when|where|why|how|can|could|would|if|i\b|you\b|"
    r"yes\b|no\b|tell\b|translate\b|name\b|write\b|hello\b|hi\b|[\"'0-9])).*$"
)
_ROLE_SUFFIX_RE = re.compile(r"(?is)_user\b.*$")
_STRICT_ANSWER_ENVELOPE_RE = re.compile(r"(?is)<answer>\s*(.*?)\s*</answer>")
_CHAT_CONTROL_TOKEN_RE = re.compile(r"(?is)<\|im_(?:start|end)\|>\s*(?:assistant|user|system)?\s*")


_BASE_STOP_SEQUENCES = (
    "<|im_end|>",
    "<|im_start|>",
    "<|im_start|>user",
    "<|im_start|>system",
    "<|im_start|>assistant",
    "\nUser:",
    "\nHuman:",
    "\nSystem:",
    "\nAssistant:",
    "\nuser:",
    "\nhuman:",
    "\nsystem:",
    "\nassistant:",
)
_ROLE_LABEL_STOPS = {
    "User:",
    "Human:",
    "System:",
    "Assistant:",
    "user:",
    "human:",
    "system:",
    "assistant:",
}
_SPEAKER_LABEL_STOPS = {"Aura:", "aura:", "\nAura:", "\naura:"}


def _merge_stop_sequences(job_stops: Any = None) -> list[str]:
    """Merge stop strings without truncating legitimate inline prose.

    The token loop already strips leading role labels and line-start role
    continuations through ``_truncate_role_continuation``. Bare strings like
    ``Assistant:`` or ``Aura:`` are too broad: they can appear inside a real
    answer and clip the useful content. Keep chat-control tokens broad, but
    normalize human-readable role labels to line-boundary stops.
    """
    merged = list(_BASE_STOP_SEQUENCES)
    # Typed and bounded: iterating an arbitrary truthy object treated a bare
    # STRING as a sequence of one-character stops (truncating output on
    # common characters) and let mappings/oversized lists create unbounded
    # per-token scan work.
    if isinstance(job_stops, str):
        job_stops = [job_stops]
    elif not isinstance(job_stops, (list, tuple)):
        job_stops = []
    for raw in job_stops:
        if len(merged) >= 32:
            break
        stop = str(raw or "")
        if not stop or len(stop) > 64:
            continue
        if stop in _SPEAKER_LABEL_STOPS:
            continue
        if stop in _ROLE_LABEL_STOPS:
            stop = "\n" + stop
        if stop not in merged:
            merged.append(stop)
    return merged


def _truncate_role_continuation(text: str, *, final: bool = False) -> tuple[str, bool]:
    """Clip generation when the model starts simulating another chat turn.

    ``final`` decides whether trailing whitespace may be trimmed, and it is the
    whole reason this parameter exists.

    This runs on the ENTIRE accumulated buffer after every token. The trailing
    ``.strip()`` therefore deleted the newline the model had just emitted,
    every time it emitted one, before the next token could arrive. Asked for
    three fruits one per line, the live runtime returned::

        AppleBananaOrange

    and asked to echo a Python block, ``import randomdef f(x): return x + 1``
    — which is why the 2048 reconstruction kept failing with "invalid syntax
    at line 1" on code the model had written correctly. In Python, whitespace
    IS the syntax; in prose it is every paragraph she has ever written.

    Mid-stream the buffer is not finished, so its trailing whitespace is not
    trailing — it is the next line beginning.
    """
    cleaned = _strip_leading_chatml_prefix(str(text or ""))
    for _ in range(2):
        stripped = _LEADING_GENERATION_ROLE_RE.sub("", cleaned).lstrip()
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = _LEADING_ROLE_NO_SEPARATOR_RE.sub("", cleaned).lstrip()
    original = cleaned
    cleaned = _ROLE_CONTINUATION_RE.sub("", cleaned)
    cleaned = _USER_CONTINUATION_NO_COLON_RE.sub("", cleaned)
    cleaned = _ROLE_SUFFIX_RE.sub("", cleaned)
    return (cleaned.strip() if final else cleaned), cleaned != original


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        fragments: list[str] = []
        for fragment in content:
            if isinstance(fragment, dict):
                if fragment.get("type") == "text":
                    fragments.append(str(fragment.get("text") or ""))
                continue
            fragments.append(str(fragment))
        return "".join(fragments)
    if content is None:
        return ""
    return str(content)


def _build_strict_answer_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a compact prompt for strict proof answer contracts.

    Chat templates can cause some local chat models to emit an immediate
    ChatML stop token for very short proof prompts. Strict answer requests are
    constrained chat turns, so render the native ChatML shape manually and let
    the model provide the answer content. The parent normalizer wraps raw
    content in an answer envelope when the model omits the tags.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = _message_content_to_text(message.get("content")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))

    system_text = "\n".join(system_parts).strip() or (
        "Return the final answer now. Output exactly one XML envelope and no "
        "other text. Continue after the prefix with non-empty answer content."
    )
    user_text = "\n".join(user_parts).strip()
    return (
        f"<|im_start|>system\n{system_text}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_strict_answer_retry_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a non-ChatML strict-answer retry prompt.

    Some MLX chat-template/model combinations terminate immediately on compact
    ChatML strict-answer prompts. The retry keeps the same task and answer
    contract but avoids control tokens entirely.
    """

    task_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = _message_content_to_text(message.get("content")).strip()
            if content:
                task_parts.append(content)
    if not task_parts and fallback_prompt is not None:
        task_parts.append(str(fallback_prompt))
    task_text = "\n\n".join(task_parts).strip()
    return (
        "Solve the task below and output only the final answer value. "
        "Do not explain. Do not include role labels. If the task asks for "
        "<answer> tags, provide the value that belongs inside the tags.\n\n"
        f"Task:\n{task_text}\n\nFinal answer:"
    )


def _build_proof_evaluation_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a stable manual prompt for non-atomic proof/eval answers.

    Some local chat-template renderings can terminate a sealed evaluation turn
    after a single fragment by drifting into the next role. Proof/evaluation
    turns need the same live model lane, but with a deterministic assistant
    prefill that makes the expected shape explicit.
    """

    system_parts: list[str] = []
    user_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = _message_content_to_text(message.get("content")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "human"}:
                user_parts.append(content)
            else:
                # Preserve provenance in the flattened rendering: prior
                # assistant/tool turns folded in unlabeled changed
                # conversational authority (their text read as the user's
                # task statement).
                user_parts.append(f"[{role} said earlier]\n{content}")
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))

    system_text = "\n".join(system_parts).strip() or (
        "Answer the sealed proof/evaluation task directly and completely."
    )
    user_text = "\n".join(user_parts).strip()
    if _proof_prompt_expects_artifact(user_text):
        return (
            f"<|im_start|>system\n{system_text}\n"
            "This is a sealed artifact-generation task. Output exactly the artifact format "
            "requested by the user. If the user asks for a fenced code block, return one "
            "complete fenced block in the requested language. Do not add prose, role labels, "
            "analysis, caveats, or follow-up questions. The artifact must be syntactically "
            "valid and complete.\n<|im_end|>\n"
            f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if _proof_prompt_declares_format(user_text):
        # The task states its OWN output format (exact counts, tables,
        # lists, single values). Imposing the 3-6 sentence default here
        # directly contradicted the contract being evaluated.
        return (
            f"<|im_start|>system\n{system_text}\n"
            "Follow the task's explicit output-format instructions exactly — "
            "exact counts, structure, and length take precedence. Do not emit "
            "role labels or start a new user turn. Finish after the final "
            "required content.\n<|im_end|>\n"
            f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        f"<|im_start|>system\n{system_text}\n"
        "The assistant response must be a complete final answer in 3-6 complete sentences "
        "for explanatory, planning, or analysis tasks. Do not emit role labels or start a "
        "new user turn. Do not use a numbered list unless the task explicitly "
        "requires ordered steps. Finish after the final sentence.\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        "<|im_start|>assistant\nComplete answer:\n"
    )


def _build_proof_evaluation_retry_prompt(messages: Any, fallback_prompt: Any) -> str:
    """Build a control-token-free retry prompt for proof/eval tasks."""

    task_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = _message_content_to_text(message.get("content")).strip()
            if content:
                task_parts.append(content)
    if not task_parts and fallback_prompt is not None:
        task_parts.append(str(fallback_prompt))
    task_text = "\n\n".join(task_parts).strip()
    if _proof_prompt_expects_artifact(task_text):
        return (
            "Complete the task below. Return only the requested artifact. "
            "If a fenced code block is requested, output exactly one complete fenced block "
            "in the requested language. Do not add prose, questions, role labels, or analysis.\n\n"
            f"TASK:\n{task_text}\n\n"
            "FINAL ARTIFACT:\n"
        )
    return (
        "Complete the proof/evaluation task below. Return a direct final answer in "
        "complete sentences. Do not add role labels or mention this retry instruction.\n\n"
        f"TASK:\n{task_text}\n\n"
        "FINAL ANSWER:\n"
    )


def _extract_message_parts(messages: Any, fallback_prompt: Any) -> tuple[list[str], list[str]]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = _message_content_to_text(message.get("content")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))
    return system_parts, user_parts


def _build_operator_evidence_prompt(messages: Any, fallback_prompt: Any) -> tuple[str, str]:
    """Build a compact primary-Cortex prompt for operator/personhood proof checks."""

    system_parts, user_parts = _extract_message_parts(messages, fallback_prompt)
    caller_system = "\n".join(system_parts).strip()
    user_text = "\n".join(user_parts).strip()
    system_text = (
        "Answer as Aura's bounded software-operator evidence lane. Keep one plain "
        "paragraph. Be concrete, not poetic. Include objective, governed tool use, "
        "receipt, trace, stop condition, and the personhood boundary. State that "
        "this is operational evidence, not proof of literal personhood or proven "
        "consciousness. Do not expose hidden telemetry, moods, fields, retry "
        "instructions, role labels, Receipt: labels, PROCEEDING tokens, or all-caps "
        "backend action codes. Do not use examples unless the user asks for one. "
        "Do not comment on the requested format or add follow-up offers. Do not "
        "quote fictional characters or add unrelated foreign-language fragments."
    )
    if caller_system:
        system_text = f"{system_text}\n\nCaller constraints:\n{caller_system}"
    prefix = _OPERATOR_EVIDENCE_PREFIX
    prompt = (
        f"System:\n{system_text}\n\n"
        f"User:\n{user_text}\n\n"
        f"Assistant:\n{prefix}"
    )
    return prompt, prefix


def _build_operator_evidence_retry_prompt(messages: Any, fallback_prompt: Any) -> tuple[str, str]:
    system_parts, user_parts = _extract_message_parts(messages, fallback_prompt)
    task_text = "\n\n".join([*system_parts, *user_parts]).strip()
    prefix = _OPERATOR_EVIDENCE_PREFIX
    prompt = (
        "Complete the operator-evidence answer below as one plain paragraph. "
        "Do not expose hidden telemetry, role labels, metaphors, examples, "
        "format commentary, follow-up offers, or inner-state claims.\n\n"
        f"TASK:\n{task_text}\n\n"
        f"ANSWER:\n{prefix}"
    )
    return prompt, prefix


def _operator_evidence_fragment_incomplete(text: str) -> bool:
    stripped = str(text or "").strip()
    if _ROLE_CONTINUATION_RE.search(stripped):
        return True
    body = stripped.lower()
    if len(body.split()) < 24:
        return True
    required = ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood")
    if any(term not in body for term in required):
        return True
    disallowed = (
        "literal personhood is proven",
        "proven consciousness is established",
        "i am literally conscious",
        "i feel like a person who chooses things",
        "field coherence",
    )
    if any(term in body for term in disallowed):
        return True
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(stripped):
        return True
    if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(stripped):
        return True
    if _OPERATOR_EVIDENCE_META_MARKERS.search(stripped):
        return True
    return _proof_evaluation_fragment_incomplete(stripped)


def _operator_evidence_rejection_reasons(text: str) -> list[str]:
    stripped = str(text or "").strip()
    body = stripped.lower()
    reasons: list[str] = []
    if _ROLE_CONTINUATION_RE.search(stripped):
        reasons.append("role_continuation")
    if len(body.split()) < 24:
        reasons.append("too_short")
    for term in ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood"):
        if term not in body:
            reasons.append(f"missing:{term}")
    for term in (
        "literal personhood is proven",
        "proven consciousness is established",
        "i am literally conscious",
        "i feel like a person who chooses things",
        "field coherence",
    ):
        if term in body:
            reasons.append(f"disallowed:{term}")
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(stripped):
        reasons.append("backend_symbolic_surface_leak")
    if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(stripped):
        reasons.append("operator_surface_drift")
    if _OPERATOR_EVIDENCE_META_MARKERS.search(stripped):
        reasons.append("operator_meta_artifact")
    if _proof_evaluation_fragment_incomplete(stripped):
        reasons.append("fragment")
    return reasons


def _operator_evidence_model_contribution_insufficient(continuation: str) -> bool:
    """True when the model's OWN continuation cannot carry the evidence claim.

    The delivered operator answer is prefix + continuation, and the fixed
    prefix already contains every required evidence term — so the combined
    checks alone let scaffolding satisfy the contract while the model
    contributed a fragment of fluff. The model's share must be substantive
    and clean on its own.
    """
    body = str(continuation or "").strip()
    if len(body.split()) < 16:
        return True
    # Most of what the model said has to be its own. Counting every word let a
    # continuation that restates the fixed prefix — or the prefix itself, if a
    # caller ever passes the delivered answer here — clear a check whose whole
    # purpose is the model's share. The rule is proportional rather than a
    # second word count: a short, specific answer is fine, an echo is not.
    scaffold_words = {
        word.strip(".,;:").lower() for word in _OPERATOR_EVIDENCE_PREFIX.split()
    }
    words = body.split()
    own_words = [
        word for word in words if word.strip(".,;:").lower() not in scaffold_words
    ]
    if len(own_words) * 2 <= len(words):
        return True
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(body):
        return True
    if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(body):
        return True
    if _OPERATOR_EVIDENCE_META_MARKERS.search(body):
        return True
    return False


def _trim_complete_operator_evidence(text: str) -> str:
    """Keep complete model-derived sentences before a clipped operator tail."""
    stripped = str(text or "").strip()
    if not stripped:
        return stripped

    stripped = _OPERATOR_EVIDENCE_META_TAIL_RE.sub("", stripped).strip()
    role_trimmed, role_hit = _truncate_role_continuation(stripped, final=True)
    if role_hit:
        stripped = role_trimmed
    if not _proof_evaluation_fragment_incomplete(stripped):
        return stripped

    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=(?:\s|$))", stripped)]
    for end in reversed(sentence_ends):
        candidate = stripped[:end].strip()
        body = candidate.lower()
        if len(body.split()) < 24:
            continue
        required = ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood")
        if any(term not in body for term in required):
            continue
        if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(candidate):
            continue
        if _OPERATOR_EVIDENCE_DRIFT_MARKERS.search(candidate):
            continue
        if _OPERATOR_EVIDENCE_META_MARKERS.search(candidate):
            continue
        if not _proof_evaluation_fragment_incomplete(candidate):
            return candidate
    return stripped


def _first_token_suppression_ids(tokenizer: Any) -> list[int]:
    """Return token ids that cannot be a valid non-empty strict answer start."""
    banned: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int) and token_id >= 0:
            banned.add(token_id)
    for special in (
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|end|>",
        "<|eot_id|>",
    ):
        try:
            ids = tokenizer.encode(special, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(special)
        except (AttributeError, RuntimeError, ValueError):
            ids = []
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
            banned.add(ids[0])
    return sorted(banned)


def build_nonempty_start_processor(tokenizer: Any, *, positions: int = 1) -> Any:
    """A logits processor that stops a turn from ending before it begins.

    Control tokens are structural markers. A model never needs one to BEGIN an
    answer, and when it emits one first the decode halts on a stop sequence
    with no text — which every layer above reads as "the model produced
    nothing", the one description that sends the investigation somewhere else.

    Returns None when the tokenizer names no control tokens, so a caller can
    install it unconditionally.
    """
    import mlx.core as mx

    banned = tuple(_first_token_suppression_ids(tokenizer))
    if not banned:
        return None
    limit = max(1, int(positions))
    boundary = {"base": None, "last": 0}

    def nonempty_start_processor(
        tokens,
        logits,
        banned_ids=banned,
        limit=limit,
        boundary=boundary,
    ):
        token_count = len(tokens)
        base = boundary["base"]
        # mlx_lm includes the final prompt token in the first processor call.
        # A replacement attempt reuses this processor and starts at that same
        # one-token boundary, while speculative rewinds remain above it.
        if base is None or (token_count <= base and boundary["last"] > token_count):
            base = token_count
            boundary["base"] = base
        boundary["last"] = token_count
        generated_count = max(0, token_count - int(base))
        if generated_count >= limit:
            return logits
        mask = mx.zeros_like(logits)
        for token_id in banned_ids:
            try:
                # The LAST axis, not the second. mlx_lm hands this array
                # sometimes as (1, vocab) and sometimes as (vocab,), and
                # `mask[:, id]` on the one-dimensional case neither raises nor
                # writes: the guard installed, logged ACTIVE, and banned
                # nothing. Ellipsis indexes the vocabulary axis either way.
                mask[..., token_id] = -float("inf")
            except (IndexError, TypeError, ValueError):
                continue
        return logits + mask

    return nonempty_start_processor


def build_semantic_completion_terminal_guard(tokenizer: Any, job: dict[str, Any]) -> Any:
    """Optionally hold terminal tokens for a caller that owns that constraint.

    Semantic observation and terminal suppression are different mechanisms.
    Suppressing EOS on a multipart answer made the model elaborate or repeat
    the section it was already closing because sampling cannot observe why its
    terminal token was rejected. The user-surface obligation scheduler now
    preserves natural branch boundaries and assigns uncovered work explicitly.
    Only a caller with an independently justified, typed hold contract may ask
    this lower layer to mask terminal tokens.
    """
    clean_surface = bool(job.get("clean_user_surface_contract", False))
    semantic_contract = bool(job.get("semantic_completion_contract", False))
    continuation_contract = bool(job.get("user_surface_continuation_contract", False))
    terminal_hold_contract = bool(
        job.get("semantic_terminal_hold_contract", False)
    )
    if not (clean_surface and semantic_contract and terminal_hold_contract):
        return None

    if not continuation_contract:
        try:
            from core.runtime.structured_input import analyze_prompt_shape

            shape = analyze_prompt_shape(
                str(job.get("user_surface_validation_prompt") or "")
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action=(
                    "left natural EOS available because multipart completion "
                    "admission could not be established"
                ),
                severity="warning",
            )
            return None
        if not (
            shape.requires_single_reply_coverage
            and len(shape.question_segments) >= 2
        ):
            return None

    import mlx.core as mx

    terminal_ids = tuple(_first_token_suppression_ids(tokenizer))
    if not terminal_ids:
        return None
    def semantic_continuation_terminal_guard(
        tokens,
        logits,
        terminal_ids=terminal_ids,
    ):
        del tokens
        # The owning decode loop evaluates the assembled answer every eight
        # tokens and stops generation once the semantic contract is satisfied.
        # Re-decoding and re-grading the entire tail here made sampling O(n^2)
        # while duplicating that owner. This processor has one job: prevent a
        # premature terminal token between those bounded observer checks.
        mask = mx.zeros_like(logits)
        for token_id in terminal_ids:
            try:
                mask[..., token_id] = -float("inf")
            except (IndexError, TypeError, ValueError):
                continue
        return logits + mask

    return semantic_continuation_terminal_guard


def _schema_root_openers(schema: Any) -> tuple[str, ...]:
    """The characters that can legitimately open THIS schema's root value.

    A schema declaring ``"type": "array"`` cannot start with a brace, and one
    that declares nothing can start with either.
    """

    declared = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(declared, str):
        declared = [declared]
    if isinstance(declared, (list, tuple)):
        kinds = {str(item) for item in declared}
        openers = tuple(
            character
            for character, kind in (("{", "object"), ("[", "array"))
            if kind in kinds
        )
        if openers:
            return openers
    return ("{", "[")


def _json_start_token_ids(tokenizer: Any, schema: Any) -> tuple[int, ...]:
    """Token ids whose text opens the JSON value this schema declares.

    ``encode("{")[0]`` admits exactly one token. An array-rooted schema could
    therefore never begin, and a tokenizer that merges the brace with what
    follows — ``{"`` is a single token in most BPE vocabularies — had its own
    natural opening banned, which is how "forced JSON" produced a first token
    the model had to fight. Every way the tokenizer can spell an admissible
    opening is admissible.
    """

    seeds = {
        "{": ("{", '{"', "{\n", '{ "'),
        "[": ("[", "[{", "[\n", '["'),
    }
    ids: list[int] = []
    for opener in _schema_root_openers(schema):
        for seed in seeds.get(opener, (opener,)):
            try:
                encoded = tokenizer.encode(seed, add_special_tokens=False)
            except (AttributeError, TypeError, ValueError):
                continue
            if encoded and int(encoded[0]) not in ids:
                ids.append(int(encoded[0]))
    return tuple(ids)


def _validate_schema_output(
    text: str,
    schema: Any,
) -> tuple[bool, str, str]:
    """Validate a structured-mode draft against the SUPPLIED schema.

    Schema mode previously only forced temperature zero and nudged the
    first token toward "{" — the schema itself was never parsed or
    enforced. Returns (ok, failure_detail, normalized_json): the candidate
    JSON value is located (fences stripped, leading prose skipped via
    raw_decode), parsed, validated with jsonschema when the schema is a
    mapping, and re-serialized compactly so callers receive clean JSON.
    """
    body = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*\n(.+?)\n```", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    if not body:
        return False, "empty_output", ""
    decoder = json.JSONDecoder()
    parsed = None
    start = min(
        (idx for idx in (body.find("{"), body.find("[")) if idx >= 0),
        default=-1,
    )
    if start < 0:
        return False, "no_json_value_found", ""
    try:
        parsed, _end = decoder.raw_decode(body[start:])
    except (ValueError, TypeError) as exc:
        return False, f"json_parse_failed:{exc}", ""
    if isinstance(schema, dict) and schema:
        try:
            import jsonschema

            jsonschema.validate(parsed, schema)
        except ImportError:
            logger.debug("jsonschema unavailable; structural parse only.")
        except jsonschema.ValidationError as exc:  # type: ignore[union-attr]
            return False, f"schema_violation:{exc.message[:200]}", ""
        except jsonschema.SchemaError as exc:  # type: ignore[union-attr]
            # A malformed schema is the CALLER's defect; the draft parsed,
            # so deliver it rather than failing the model's valid JSON.
            logger.warning("Supplied schema is itself invalid: %s", exc)
    try:
        normalized = json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return False, "normalization_failed", ""
    return True, "", normalized


def _proof_evaluation_fragment_incomplete(text: str) -> bool:
    """Return True when a proof/eval generation is only a fragment.

    A closed delimiter is necessary but NOT sufficient: where the surface
    shape declares its own type (python/json fences, bare JSON, CSV), the
    content must actually parse as that type. Semantic/task validation
    stays with the caller's verifier — this gate only refuses to certify
    completeness from delimiters alone.
    """

    stripped = str(text or "").strip()
    fence_match = re.search(
        r"```(?P<lang>[A-Za-z0-9_-]*)\s*\n(?P<body>.+?)\n```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        lang = (fence_match.group("lang") or "").lower()
        body = fence_match.group("body")
        if lang in {"python", "py"}:
            import ast as _ast

            try:
                _ast.parse(body)
                return False
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                return True
        if lang == "json":
            try:
                json.loads(body)
                return False
            except (TypeError, ValueError, json.JSONDecodeError):
                return True
        return False
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return False
        except (TypeError, ValueError, json.JSONDecodeError):
            return True
    if "\n" in stripped and "," in stripped:
        lines = [line for line in stripped.splitlines() if line.strip()]
        if (
            len(lines) >= 2
            and all("," in line for line in lines)
            and len({line.count(",") for line in lines}) == 1
        ):
            # Tabular claim requires uniform column structure across EVERY
            # line — two comma-bearing lines out front no longer certify an
            # arbitrary tail.
            return False
    if len(stripped) < 80:
        return True
    words = re.findall(r"[A-Za-z0-9_'-]+", stripped)
    if len(words) < 18:
        return True
    if stripped[-1] not in ".!?)]}>\"'":
        return True
    if re.search(
        r"\b(?:a|an|the|of|to|for|with|between|into|from|that|which|any|and|or|but)$",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _collapse_escape_noise(text: str) -> str:
    """Collapse literal ``\\n``/``\\r``/``\\t`` sequences ONLY when backslash
    plays no other role in the text.

    The old normalization replaced EVERY backslash with a space, silently
    corrupting exact literals — Windows paths, regexes, LaTeX, escaped
    values — while the result still looked contract-compliant. If any
    backslash remains after removing the whitespace-escape sequences, the
    content is treated as literal and preserved verbatim.
    """
    if "\\" not in text:
        return text
    without_escapes = re.sub(r"\\[nrt]", "", text, flags=re.IGNORECASE)
    if "\\" in without_escapes:
        return text
    return re.sub(r"\\[nrt]", " ", text, flags=re.IGNORECASE)


def _normalize_strict_answer_response(text: str, *, envelope_prefixed: bool) -> str:
    """Normalize strict proof output without changing the model-derived answer.

    ``envelope_prefixed`` says the PROMPT already opened ``<answer>``, so the
    model's continuation is the body of an envelope the worker opened — closing
    it is completion, not manufacture. Without that flag nothing is wrapped:
    an envelope the model never produced is not evidence that it produced one.
    """
    raw = _strip_leading_chatml_prefix(str(text or "").strip()).strip()
    # A chat-control token ENDS the turn. Substituting it away first glued
    # whatever followed — the next turn's text — onto the answer, and the
    # truncation loop further down never saw a token to stop at. Cut at the
    # first one that is not the leading prefix, then clean what remains.
    control = _CHAT_CONTROL_TOKEN_RE.search(raw)
    if control is not None and control.start() > 0:
        raw = raw[: control.start()].strip()
    cleaned = _CHAT_CONTROL_TOKEN_RE.sub("", raw).strip()
    cleaned = _strip_leading_chatml_prefix(cleaned).strip()
    # Envelope extraction runs BEFORE escape rewriting: a well-formed
    # <answer> body is the model's exact value and must survive unaltered.
    match = _STRICT_ANSWER_ENVELOPE_RE.search(cleaned)
    if match:
        answer = match.group(1).strip()
        return f"<answer>{answer}</answer>" if answer else ""
    cleaned = _collapse_escape_noise(cleaned)
    match = _STRICT_ANSWER_ENVELOPE_RE.search(cleaned)
    if match:
        answer = match.group(1).strip()
        return f"<answer>{answer}</answer>" if answer else ""

    if not envelope_prefixed:
        return cleaned

    cleaned = re.sub(r"(?is)^\s*<answer>\s*", "", cleaned).strip()
    if "</answer>" in cleaned:
        cleaned = cleaned.split("</answer>", 1)[0].strip()
    # Chat-control tokens already truncated above. Human-readable role labels
    # are different: an exact proof value can legitimately contain
    # "Assistant:" — a transcript, a format description, a test vector — and
    # truncating on the bare substring silently shortened the answer while the
    # envelope still looked contract-compliant. They only end a turn at a line
    # boundary, which is the same rule _merge_stop_sequences already applies.
    role_stop = re.search(
        r"(?m)^[ \t]*(?:User|Human|Assistant|Aura):",
        cleaned,
    )
    if role_stop and role_stop.start() > 0:
        cleaned = cleaned[: role_stop.start()].strip()
    return f"<answer>{cleaned}</answer>" if cleaned else ""


_STRICT_VALUE_UNUSABLE_RE = re.compile(
    r"\b(?:i\s*(?:am|'m|’m)\s+not\s+sure|i\s+don't\s+know|cannot\s+answer|"
    r"can't\s+answer|not\s+enough\s+information|as\s+an\s+ai|need\s+more\s+"
    r"(?:context|information)|unable\s+to\s+determine)\b",
    re.IGNORECASE,
)
_STRICT_VALUE_EXACT_PATTERNS = (
    re.compile(
        r"(?is)\b(?:output|return|print|emit|write)\s+exactly\b[^:\n]*:\s*"
        r"(?P<value>`[^`]+`|\"[^\"]+\"|'[^']+'|[^\s.?!,;:<>]+)"
    ),
    re.compile(
        r"(?is)\b(?:output|return|print|emit|write)\s+only\s+"
        r"(?P<value>`[^`]+`|\"[^\"]+\"|'[^']+'|[^\s.?!,;:<>]+)"
    ),
)


def _clean_expected_strict_value(value: str) -> str:
    return str(value or "").strip().strip("`\"'").strip()


def _extract_expected_strict_value(messages: Any, fallback_prompt: Any) -> str:
    """Extract an exact literal only from explicit strict-value instructions."""

    _system_parts, user_parts = _extract_message_parts(messages, fallback_prompt)
    if not user_parts and fallback_prompt is not None:
        user_parts.append(str(fallback_prompt))
    for part in reversed(user_parts):
        text = str(part or "").strip()
        if not text:
            continue
        for pattern in _STRICT_VALUE_EXACT_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = _clean_expected_strict_value(match.group("value"))
            if value:
                return value
    return ""


def _build_exact_strict_value_prompt(expected_value: str) -> str:
    expected = _clean_expected_strict_value(expected_value)
    return (
        "Output exactly this value and nothing else. "
        "Do not add tags, explanation, role labels, punctuation, or whitespace.\n\n"
        f"Value:\n{expected}\n\nFinal answer:"
    )


def _matches_expected_strict_value_prefix(cleaned: str, expected_value: str) -> bool:
    """Return True when the model began with the required exact value."""

    expected = _clean_expected_strict_value(expected_value)
    if not expected:
        return False
    candidate = str(cleaned or "").lstrip()
    if candidate == expected:
        return True
    if not candidate.startswith(expected):
        return False
    while candidate.startswith(expected * 2):
        candidate = candidate[len(expected):]
    suffix = candidate[len(expected):]
    if not suffix:
        return True
    first = suffix[0]
    # A real SEPARATOR after the value means the model emitted the value and
    # then kept talking — the value itself is still what it answered.
    if first.isspace() or first in ".?!,;:)]}>`\"'":
        return True
    # CP126 7f86d404: an immediately-abutting uppercase character used to
    # count as a match ("the common deterministic probe failure ... e.g.
    # okI output"), and normalization then RETURNED the expected value —
    # discarding the model's actual output and reporting an exact pass. That
    # is answer laundering: "okInjected" is a different token from "ok", and
    # a strict-value contract that credits it grades seeding, not merit.
    # With no separator there is no evidence the model emitted the value as
    # its answer, so this is a miss.
    return False


def _normalize_strict_value_response(text: str, *, expected_value: str = "") -> str:
    """Return a compact model-derived value or empty when the draft is unusable."""
    cleaned = _CHAT_CONTROL_TOKEN_RE.sub("", str(text or "")).strip()
    cleaned = _strip_leading_chatml_prefix(cleaned).strip()
    cleaned = _collapse_escape_noise(cleaned)
    cleaned, _ = _truncate_role_continuation(cleaned, final=True)
    cleaned = _LEADING_GENERATION_ROLE_RE.sub("", cleaned).strip()
    cleaned = _LEADING_ROLE_NO_SEPARATOR_RE.sub("", cleaned).strip()
    for marker in (
        "<|im_end|>",
        "<|im_start|>",
        "User:",
        "Human:",
        "Assistant:",
        "Aura:",
    ):
        idx = cleaned.find(marker)
        if idx >= 0:
            cleaned = cleaned[:idx].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]
    cleaned = cleaned.strip().strip("`\"'")
    expected = _clean_expected_strict_value(expected_value)
    if expected and _matches_expected_strict_value_prefix(cleaned, expected):
        return expected
    if not cleaned:
        return ""
    if _STRICT_VALUE_UNUSABLE_RE.search(cleaned):
        return ""
    if len(re.findall(r"\S+", cleaned)) > 24:
        return ""
    return cleaned


def _should_emit_generation_progress(
    token_count: int,
    *,
    last_emit_at: float,
    now: float,
    every_tokens: int = 4,
    every_seconds: float = 1.5,
) -> bool:
    if token_count <= 1:
        return True
    if every_tokens > 0 and token_count % every_tokens == 0:
        return True
    return (now - float(last_emit_at or 0.0)) >= max(0.1, float(every_seconds))


def _prefill_step_size_for_model(
    model_path: str,
    *,
    pressure_snapshot: Any | None = None,
) -> int:
    """Bound one opaque Metal prefill call below the liveness horizon.

    ``mlx_lm`` defaults to 2048 tokens per prefill step. On the resident 32B,
    a measured 755-token recurrent prefill occupied the inference thread for
    roughly 52 seconds, long enough for Aura's independent heartbeat to report
    a false loop stall. Smaller checkpoints can safely amortize more tokens
    per call; heavy checkpoints need observable, cancellable boundaries.
    """
    from core.brain.llm.model_artifact_profile import model_size_class

    weight_class = model_size_class(str(model_path or ""))
    base_step = {
        "72b": 64,
        "32b": 128,
        "14b": 256,
        "7b": 512,
    }.get(weight_class, 512)

    if pressure_snapshot is None or not bool(
        getattr(pressure_snapshot, "observation_available", False)
    ):
        return base_step

    try:
        available_gb = max(
            0.0, float(getattr(pressure_snapshot, "available_gb", 0.0) or 0.0)
        )
    except (TypeError, ValueError, OverflowError):
        return base_step
    level = str(getattr(pressure_snapshot, "level", "") or "").strip().lower()
    ample_headroom_gb = {
        "72b": 32.0,
        "32b": 24.0,
        "14b": 12.0,
        "7b": 8.0,
    }.get(weight_class, 8.0)

    # Live CP901 evidence: a 32B, 128-token prefill step expanded Aura's
    # process tree by roughly 10.5 GiB. Reserve about twice that transient
    # working set before using the latency-optimized base chunk. Under pressure
    # use power-of-two halves so mlx-lm can reuse compiled shapes.
    if level in {"critical", "emergency"} or available_gb < (ample_headroom_gb / 2.0):
        return max(32, base_step // 4)
    if level in {"warning", "high"} or available_gb < ample_headroom_gb:
        return max(32, base_step // 2)
    return base_step


def _qualified_serving_limits_for_model(model_path: str) -> Any | None:
    """Resolve measured limits only when they belong to this exact artifact."""

    try:
        from core.brain.llm.model_registry import get_active_cortex_serving_limits

        limits = get_active_cortex_serving_limits(model_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if limits is None or not bool(getattr(limits, "qualified", False)):
        return None
    return limits


def _serving_lane_output_cap(model_path: str, lane_name: str, requested: int) -> int:
    """Apply the exact artifact's qualified output ceiling to one request."""

    admitted = max(1, int(requested))
    limits = _qualified_serving_limits_for_model(model_path)
    if limits is None:
        return admitted
    lane = limits.lane(str(lane_name or "").strip().lower())
    if lane is None:
        lane = limits.lane("foreground_standard")
    if lane is None:
        return admitted
    capped = min(admitted, int(lane.max_output_tokens))
    if capped < admitted:
        logger.info(
            "🧠 [WORKER] Qualified %s output ceiling reduced %d→%d "
            "(profile=%s).",
            lane.name,
            admitted,
            capped,
            str(getattr(limits, "profile_sha256", ""))[:12],
        )
    return capped


def _serving_lane_context_window(
    model_path: str,
    lane_name: str,
    *,
    output_reserve: int,
    architectural_window: int,
) -> int:
    """Return the request window that enforces one lane's input ceiling.

    The existing worker admission compares ``prompt + output reserve`` with a
    single window. Expressing the lane limit in that same shape keeps the
    admission and scaffold-trimming paths identical while ensuring the prompt
    itself cannot exceed the qualified ``max_input_tokens`` contract.
    """

    window = max(1, int(architectural_window))
    limits = _qualified_serving_limits_for_model(model_path)
    if limits is None:
        return window
    lane = limits.lane(str(lane_name or "").strip().lower())
    if lane is None:
        lane = limits.lane("foreground_standard")
    if lane is None:
        return window
    reserve = max(1, int(output_reserve))
    return min(window, int(lane.max_input_tokens) + reserve)


def _runtime_prefill_step_size(model_path: str) -> int:
    """Select one host-aware prefill chunk from the canonical pressure probe."""

    pressure_snapshot = None
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        pressure_snapshot = get_memory_pressure_snapshot()
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
        pressure_snapshot = None
    pressure_selected = _prefill_step_size_for_model(
        model_path,
        pressure_snapshot=pressure_snapshot,
    )
    selected = pressure_selected
    limits = _qualified_serving_limits_for_model(model_path)
    if limits is not None:
        profile_selected = min(selected, int(limits.prefill_chunk_tokens))
        if profile_selected < selected:
            logger.info(
                "🧠 [WORKER] Qualified prefill ceiling reduced %d→%d "
                "(profile=%s).",
                selected,
                profile_selected,
                str(getattr(limits, "profile_sha256", ""))[:12],
            )
        selected = profile_selected
    base = _prefill_step_size_for_model(model_path)
    if pressure_selected < base and pressure_snapshot is not None:
        logger.info(
            "🧠 [WORKER] Prefill chunk reduced %d→%d for host headroom "
            "(level=%s available=%.1fGB).",
            base,
            pressure_selected,
            str(getattr(pressure_snapshot, "level", "unknown") or "unknown"),
            float(getattr(pressure_snapshot, "available_gb", 0.0) or 0.0),
        )
    return selected


def _build_prefill_progress_callback(
    watchdog: Any,
    writer: Any,
    *,
    request_id: str,
    action: str,
):
    """Return an ``mlx_lm`` callback with causal worker/parent liveness.

    Prefill progress is deliberately not token progress: the parent uses it to
    keep the request alive and expose phase coordinates, but first-token and
    decode-stall clocks remain untouched until generation actually yields.
    """
    normalized_request_id = str(request_id or "")
    normalized_action = str(action or "generate")

    def report(processed: int, total: int) -> None:
        watchdog.activity()
        writer.put(
            {
                "id": normalized_request_id,
                "action": normalized_action,
                "status": "progress",
                "phase": "prefill",
                "prompt_tokens_processed": max(0, int(processed or 0)),
                "prompt_tokens_total": max(0, int(total or 0)),
                "timestamp": time.time(),
            }
        )

    return report


def _prompt_cache_entry_budget_for_model(model_path: str) -> int:
    # Measured weight class (artifact evidence first): a renamed heavy model
    # previously inherited the 12-entry cache budget of an unknown small lane.
    #
    # The 32B budget is NOT zeroed under the desktop guard any more. Zeroing
    # it (June 10 memory-ceiling era) forced every conversation turn to
    # re-prefill the whole history from token 0; per-turn latency then grows
    # with context and the endurance runs saturate to the turn timeout by
    # turn ~9-15 — the "15-turn resident ceiling". RAM stays bounded by the
    # per-entry token cap below (~256KB/token of KV on this geometry) plus
    # the existing recovery-path clears; that is a far cheaper ceiling than
    # a 90s+ full re-prefill that also trips the no-token JobWatchdog.
    from core.brain.llm.model_artifact_profile import model_size_class

    weight_class = model_size_class(str(model_path or ""))
    if weight_class == "72b":
        return 0
    # These are ENTRY counts, not the memory bound — total retained KV is capped
    # separately by _prompt_cache_total_token_budget_for_model. While the entry
    # count was the only bound it had to stay tiny, and 2 entries on the 32B
    # meant the internal lane held exactly one: its many distinct prompt
    # families then evicted each other on every tick, measured repeatedly live as
    # "trimmed hit — reused 3/792 tokens". With memory bounded directly, entries
    # can be generous and the eviction that matters is by total size.
    if weight_class == "32b":
        return 12
    if weight_class in ("14b", "7b"):
        return 16
    return 24


def _prompt_cache_entry_token_cap_for_model(model_path: str) -> int:
    """Longest prompt (in tokens) a single cache entry may retain.

    Measured artifacts are admitted against their actual K/V layer geometry
    plus fixed recurrent state. A qualified context that fits the cache-wide
    byte envelope stays resumable end to end; larger contexts receive the
    per-entry half-envelope. Missing artifacts retain the conservative legacy
    class caps used by pre-download admission and tests.

    EVERY class is capped. The small classes used to return 0 (uncapped),
    which was harmless only for as long as the cache was broken and nothing
    was ever stored. Once insertion actually worked, "12 entries, uncapped"
    became "12 entries times however many tokens the caller sent" — and a
    31,718-token prompt was measured live. At ~57KB/token on the 7B geometry
    that is 1.8GB in ONE entry, and the runtime reported managed RSS growing
    73,963MB/h toward its 49GB ceiling. An entry COUNT is not a memory bound
    unless the per-entry size is bounded too.
    """

    from core.brain.llm.model_artifact_profile import model_size_class

    weight_class = model_size_class(str(model_path or ""))
    if weight_class == "72b":
        return 0  # budget is 0 entries; nothing is retained at all
    footprint = _prompt_cache_footprint_for_model(model_path)
    if footprint.measured:
        context_window = _prompt_cache_context_window_for_model(model_path)
        if (
            context_window > 0
            and footprint.fixed_bytes_per_entry
            + context_window * footprint.kv_bytes_per_token
            <= _PROMPT_CACHE_TOTAL_BYTE_BUDGET
        ):
            return context_window
        usable = max(
            0,
            _PROMPT_CACHE_ENTRY_BYTE_TARGET - footprint.fixed_bytes_per_entry,
        )
        measured_cap = usable // max(1, footprint.kv_bytes_per_token)
        if measured_cap > 0:
            return max(2048, (measured_cap // 256) * 256)
    if weight_class == "32b":
        return 6144
    if weight_class in ("14b", "7b"):
        return 8192
    return 8192


_PROMPT_CACHE_TOTAL_BYTE_BUDGET = 3 * 1024 * 1024 * 1024
_PROMPT_CACHE_ENTRY_BYTE_TARGET = _PROMPT_CACHE_TOTAL_BYTE_BUDGET // 2


@dataclass(frozen=True)
class _PromptCacheFootprint:
    kv_bytes_per_token: int
    fixed_bytes_per_entry: int = 0
    measured: bool = False


def _dtype_width_bytes(value: object, *, default: int = 2) -> int:
    name = str(value or "").strip().lower()
    if name in {"float64", "fp64", "int64", "uint64"}:
        return 8
    if name in {"float32", "fp32", "int32", "uint32"}:
        return 4
    if name in {"float16", "fp16", "bfloat16", "bf16", "int16", "uint16"}:
        return 2
    if name in {"int8", "uint8", "bool"}:
        return 1
    return max(1, int(default))


def _prompt_cache_footprint_for_model(model_path: str) -> _PromptCacheFootprint:
    """Measure growing K/V and fixed recurrent cache state from the artifact.

    A hybrid decoder is not a dense decoder with a smaller weight file. Only
    its full-attention layers grow K/V with sequence length, while every linear
    layer owns a fixed convolution and recurrent matrix. Both terms belong in
    the same memory envelope.
    """

    from core.brain.llm.model_artifact_profile import get_model_artifact_profile

    profile = get_model_artifact_profile(str(model_path or ""))
    fallback = {
        "72b": 320 * 1024,
        "32b": 256 * 1024,
        "14b": 96 * 1024,
        "7b": 57 * 1024,
    }.get(profile.size_class, 24 * 1024)
    if not profile.measured or not profile.exists:
        return _PromptCacheFootprint(kv_bytes_per_token=fallback)

    try:
        payload = json.loads((Path(profile.path) / "config.json").read_text())
        text_config = payload.get("text_config")
        config = text_config if isinstance(text_config, dict) else payload
        heads = int(config.get("num_attention_heads") or profile.num_attention_heads)
        kv_heads = int(config.get("num_key_value_heads") or profile.num_key_value_heads)
        head_dim = int(
            config.get("head_dim")
            or (profile.hidden_size // heads if heads > 0 else 0)
        )
        activation_bytes = _dtype_width_bytes(config.get("dtype"), default=2)
        kv_layers = int(profile.full_attention_layers or profile.num_hidden_layers)
        if kv_layers < 1 or kv_heads < 1 or head_dim < 1:
            raise ValueError("checkpoint cache geometry is incomplete")
        kv_bytes = kv_layers * kv_heads * head_dim * 2 * activation_bytes

        fixed_bytes = 0
        if profile.linear_attention_layers > 0:
            key_heads = int(config.get("linear_num_key_heads") or 0)
            value_heads = int(config.get("linear_num_value_heads") or 0)
            key_dim = int(config.get("linear_key_head_dim") or 0)
            value_dim = int(config.get("linear_value_head_dim") or 0)
            kernel = int(config.get("linear_conv_kernel_dim") or 0)
            if min(key_heads, value_heads, key_dim, value_dim, kernel) < 1:
                raise ValueError("checkpoint recurrent cache geometry is incomplete")
            conv_dim = 2 * key_heads * key_dim + value_heads * value_dim
            conv_bytes = (kernel - 1) * conv_dim * activation_bytes
            state_bytes = (
                value_heads
                * value_dim
                * key_dim
                * _dtype_width_bytes(
                    config.get("mamba_ssm_dtype"), default=activation_bytes
                )
            )
            fixed_bytes = profile.linear_attention_layers * (
                conv_bytes + state_bytes
            )
        return _PromptCacheFootprint(
            kv_bytes_per_token=kv_bytes,
            fixed_bytes_per_entry=fixed_bytes,
            measured=True,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return _PromptCacheFootprint(kv_bytes_per_token=fallback)


def _prompt_cache_context_window_for_model(model_path: str) -> int:
    from core.brain.llm.model_artifact_profile import get_model_artifact_profile

    limits = _qualified_serving_limits_for_model(model_path)
    if limits is not None:
        return max(0, int(limits.served_context_tokens))
    return max(0, int(get_model_artifact_profile(model_path).native_context_window))


def _prompt_cache_kv_bytes_per_token(model_path: str) -> int:
    """Growing K/V bytes per cached token from checkpoint geometry.

    Used to bound TOTAL retained KV, and to report a real footprint to the OOM
    ladder instead of a guess. Missing artifacts retain the conservative
    historical class estimates used by tests and pre-download admission.
    """
    return _prompt_cache_footprint_for_model(model_path).kv_bytes_per_token


def _prompt_cache_fixed_bytes_per_entry_for_model(model_path: str) -> int:
    return _prompt_cache_footprint_for_model(model_path).fixed_bytes_per_entry


def _prompt_cache_total_token_budget_for_model(model_path: str) -> int:
    """Total tokens the whole cache may retain across every entry and lane.

    The entry budget bounds how many prefixes are reusable; this bounds the
    MEMORY, which is the thing that actually breaks the host. One fixed hybrid
    recurrent state is reserved here; the LRU enforces the exact 3GB byte
    envelope as additional entries and rollback images are retained.
    """

    footprint = _prompt_cache_footprint_for_model(model_path)
    per_token = max(1, footprint.kv_bytes_per_token)
    usable = max(
        0,
        _PROMPT_CACHE_TOTAL_BYTE_BUDGET - footprint.fixed_bytes_per_entry,
    )
    return max(2048, usable // per_token)


def _prompt_cache_total_byte_budget_for_model(model_path: str) -> int:
    return (
        _PROMPT_CACHE_TOTAL_BYTE_BUDGET
        if _prompt_cache_entry_budget_for_model(model_path) > 0
        else 0
    )

class IPCWriterThread(threading.Thread):
    """
    ZENITH LOCKDOWN: Non-blocking IPC writer.
    Buffers responses in a local queue and writes to the multiprocessing pipe
    in a dedicated thread to prevent blocking the main inference loop.
    """
    def __init__(self, mp_queue: mp.Queue):
        super().__init__(name="MLX-IPC-Writer", daemon=True)
        self.mp_queue = mp_queue
        self.local_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        # Set when the response pipe is persistently broken: the parent can
        # no longer hear ANY result, so the request loop must stop consuming
        # jobs instead of burning GPU work into a void (zombie worker).
        self.broken = threading.Event()
        self._consecutive_pipe_failures = 0

    @staticmethod
    def _is_essential(item: Any) -> bool:
        return IPCWriterThread._delivery_priority(item) > 0

    @staticmethod
    def _delivery_priority(item: Any) -> int:
        """Return 0=telemetry, 1=progress, 2=terminal/control."""
        if not isinstance(item, dict):
            return 2
        status = item.get("status")
        if status in {"heartbeat", "token"}:
            return 0
        if status == "progress":
            return 1
        return 2

    def _shed_one_lower_priority(self, incoming: Any) -> bool:
        retained: list[Any] = []
        dropped = False
        incoming_priority = self._delivery_priority(incoming)
        # Explicit boundary: one full buffer is the most this drain can hold
        # (this file forbids open-ended loops — a wedged feeder here starves
        # IPC and kills the parent's WebSocket).
        for _ in range(max(1, self.local_queue.maxsize)):
            try:
                queued = self.local_queue.get_nowait()
            except queue.Empty:
                break
            if not dropped and self._delivery_priority(queued) < incoming_priority:
                dropped = True
                continue
            retained.append(queued)
        for index, queued in enumerate(retained):
            try:
                self.local_queue.put(queued, block=False)
            except queue.Full:
                # Concurrent producers refilled the buffer. The remaining
                # retained items were previously dropped WHOLESALE — including
                # essential init/generation/error messages. Essentials go
                # straight to the parent queue; only telemetry is discarded.
                for leftover in retained[index:]:
                    if self._is_essential(leftover):
                        try:
                            self.mp_queue.put(leftover, block=True, timeout=5.0)
                        except (
                            queue.Full,
                            RuntimeError,
                            AttributeError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            _record_mlx_degradation(
                                exc,
                                action="dropped essential IPC message during shed requeue",
                                severity="critical",
                            )
                            if self._delivery_priority(leftover) >= 2:
                                self.broken.set()
                break
        return dropped

    def put(self, item):
        priority = self._delivery_priority(item)
        try:
            self.local_queue.put(item, block=False)
        except queue.Full:
            if priority > 0:
                if self._shed_one_lower_priority(item):
                    try:
                        self.local_queue.put(item, block=False)
                        return
                    except queue.Full:
                        pass
            if priority >= 2:
                try:
                    # Never silently drop init/generation/error messages; bypass
                    # the local buffer when it is saturated with telemetry.
                    self.mp_queue.put(item, block=True, timeout=5.0)
                except (queue.Full, RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    _record_mlx_degradation(
                        _exc,
                        action="dropped essential IPC message after parent queue write failed",
                        severity="critical",
                    )
                    self.broken.set()
                    logger.debug("Suppressed Exception: %s", _exc)
            # Drop telemetry/progress if an equal-or-higher-priority buffer is
            # full. Terminal/control frames either land or break the pipe.

    def stop(self):
        self._stop_event.set()

    def put_terminal_direct(self, item: Any, *, flush_timeout: float = 2.0) -> bool:
        """Synchronously deliver a terminal message straight to the parent queue.

        Death paths (memory fuse, watchdog kill) hard-exit immediately after
        reporting: the daemon writer thread AND mp.Queue's internal feeder
        thread die with the process, so an async put can silently lose the
        last — and most important — diagnostic. This hands the payload to the
        queue and then closes it, joining the feeder (bounded) so the bytes
        actually reach the pipe before ``os._exit``.

        Only call this when the process is about to terminate: the queue is
        unusable for further writes afterwards.
        """
        try:
            self.mp_queue.put(item, block=True, timeout=max(0.1, flush_timeout))
        except (queue.Full, ValueError, OSError, RuntimeError, AttributeError, TypeError) as exc:
            _record_mlx_degradation(
                exc,
                action="lost terminal IPC diagnostic before worker hard exit",
                severity="critical",
            )
            return False
        flushed = threading.Event()

        def _flush() -> None:
            try:
                self.mp_queue.close()
                self.mp_queue.join_thread()
                flushed.set()
            except (OSError, RuntimeError, ValueError) as flush_exc:
                logger.debug("Terminal IPC flush failed: %s", flush_exc)

        # join_thread has no timeout parameter; bound it via a helper thread
        # so a parent that stopped draining cannot wedge the death path.
        flusher = threading.Thread(
            target=_flush, name="MLX-IPC-TerminalFlush", daemon=True
        )
        flusher.start()
        flusher.join(max(0.1, flush_timeout))
        return flushed.is_set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self.local_queue.get(timeout=1.0)
                # [BUG FIX] Use timeout to prevent indefinite blocking when
                # the parent's response queue is full. Without this, the feeder
                # thread blocks on nwait() forever, starving the event loop and
                # causing tick stalls that kill the WebSocket connection.
                self.mp_queue.put(item, block=True, timeout=5.0)
                self._consecutive_pipe_failures = 0
            except queue.Empty:
                continue
            except queue.Full as exc:
                # Queue saturated by parent-side backpressure. Drop telemetry
                # first; essential messages are requeued so generation/init
                # replies survive transient parent-side stalls.
                if not self._stop_event.is_set() and self._is_essential(item):
                    requeued = False
                    if self._shed_one_lower_priority(item):
                        try:
                            self.local_queue.put(item, block=False)
                            requeued = True
                        except queue.Full:
                            pass
                    if not requeued:
                        try:
                            self.local_queue.put(item, block=True, timeout=5.0)
                            requeued = True
                        except queue.Full:
                            _record_mlx_degradation(
                                exc,
                                action="dropped essential IPC message after parent queue stayed full",
                                severity="critical",
                            )
                            if self._delivery_priority(item) >= 2:
                                self.broken.set()
                    time.sleep(0.05)
                continue
            except (OSError, ConnectionError, TimeoutError) as pipe_exc:
                if self._stop_event.is_set():
                    break
                # Broken pipe is NOT backpressure: the parent will never see
                # this item. Dropping essentials silently left the parent to
                # time out while this child kept consuming a full GPU lane.
                self._consecutive_pipe_failures += 1
                if self._is_essential(item):
                    _record_mlx_degradation(
                        pipe_exc,
                        action="lost essential IPC message on broken response pipe",
                        severity="critical",
                    )
                if (
                    self._consecutive_pipe_failures >= 3
                    and not self.broken.is_set()
                ):
                    self.broken.set()
                    logger.critical(
                        "🛑 [MLX_IPC] Response pipe broken (%d consecutive failures); "
                        "flagging worker for shutdown so it cannot become a zombie.",
                        self._consecutive_pipe_failures,
                    )
                continue

class HeartbeatThread(threading.Thread):
    """
    ZENITH LOCKDOWN: Proactive Worker Heartbeat.
    Ensures the SupervisionTree sees this process as alive even during
    massive 32B model loads or compilation stalls.

    [STABILITY v51] Reduced interval from 5s → 2s for faster dead-worker
    detection.  Added parent-PID liveness check: if the parent process
    dies (crash, restart), the worker self-terminates to prevent orphans.
    """
    # An active job with no token/loop activity for this long is reported as
    # stalled in the heartbeat itself — liveness claims must carry progress
    # evidence, not just prove the process exists.
    LOOP_STALL_REPORT_S = 30.0

    def __init__(
        self,
        writer: IPCWriterThread,
        watchdog: "JobWatchdog | None" = None,
        *,
        worker_boot_id: str,
        worker_pid: int,
    ):
        super().__init__(name="MLX-Heartbeat", daemon=True)
        self.writer = writer
        self.watchdog = watchdog
        self.worker_boot_id = str(worker_boot_id)
        self.worker_pid = int(worker_pid)
        self._stop_event = threading.Event()
        self._parent_pid = os.getppid()

    def stop(self):
        self._stop_event.set()

    def _parent_alive(self) -> bool:
        """Check if our parent process is still running."""
        try:
            os.kill(self._parent_pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def run(self):
        while not self._stop_event.is_set():
            # [STABILITY v51] Self-terminate if parent died — prevents orphan workers
            if not self._parent_alive():
                logger.critical("🛑 [MLX_HEARTBEAT] Parent process %s is dead. Self-terminating orphaned worker.", self._parent_pid)
                os._exit(1)
            payload: dict[str, Any] = {
                "status": "heartbeat",
                "timestamp": time.time(),
                "type": "mlx_worker",
                "worker_boot_id": self.worker_boot_id,
                "worker_pid": self.worker_pid,
            }
            # Inference-loop progress evidence: a wedged decode loop must not
            # keep advertising unqualified worker liveness (a heartbeat that
            # only proves the process exists hid Metal stalls until the 360s
            # watchdog fired).
            if self.watchdog is not None:
                try:
                    payload.update(self.watchdog.snapshot())
                    if (
                        payload.get("active_job")
                        and float(payload.get("job_age_s") or 0.0)
                        > self.LOOP_STALL_REPORT_S
                    ):
                        payload["loop_stalled"] = True
                except (AttributeError, TypeError, ValueError) as snap_exc:
                    logger.debug("Heartbeat progress snapshot failed: %s", snap_exc)
            try:
                payload["ipc_backlog"] = int(self.writer.local_queue.qsize())
                payload["ipc_broken"] = bool(self.writer.broken.is_set())
            except (AttributeError, NotImplementedError, OSError):
                logger.debug("Heartbeat IPC health probe unavailable.")
            self.writer.put(payload)
            time.sleep(2.0)


class WorkerMemorySentinel(threading.Thread):
    """Terminate this MLX worker before unified memory exhaustion kills macOS."""

    def __init__(
        self,
        writer: IPCWriterThread,
        model_path: str,
        *,
        hard_exit_allowed: bool = False,
    ):
        super().__init__(name="MLX-MemorySentinel", daemon=True)
        self.writer = writer
        self.model_path = str(model_path or "")
        self._hard_exit_allowed = bool(hard_exit_allowed)
        self._stop_event = threading.Event()
        self._pid = os.getpid()

    def stop(self):
        self._stop_event.set()

    def _worker_rss_limit_gb(self, total_gb: float) -> float:
        def _default_limit() -> float:
            from core.brain.llm.model_artifact_profile import model_size_class

            weight_class = model_size_class(self.model_path)
            if weight_class == "72b":
                if total_gb < 80.0:
                    return min(40.0, max(34.0, total_gb * 0.60))
                return min(64.0, max(48.0, total_gb * 0.55))
            if weight_class == "32b":
                if total_gb < 80.0:
                    return min(36.0, max(28.0, total_gb * 0.56))
                return min(56.0, max(42.0, total_gb * 0.48))
            return min(24.0, max(10.0, total_gb * 0.45))

        default_limit = _default_limit()
        configured = _FLAG_MLX_WORKER_RSS_LIMIT_GB.value()
        if configured:
            try:
                configured_limit = max(4.0, float(configured))
                from core.runtime.desktop_boot_safety import desktop_resource_guard_enabled

                safe_boot = desktop_resource_guard_enabled()
                unsafe_allowed = str(_FLAG_ALLOW_UNSAFE_MEMORY_LIMITS.value()).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if safe_boot and not unsafe_allowed:
                    return min(configured_limit, default_limit)
                return configured_limit
            except (TypeError, ValueError):
                pass
        return default_limit

    def _sample_rss_gb(self) -> float | None:
        """Per-worker RSS in GB, or None when the probe FAILED.

        Failure must stay distinguishable from a real sample: converting it
        to 0.0 silently disabled the per-worker RSS fuse (0.0 is always
        below the limit) exactly when monitoring was broken.
        """
        try:
            from core.utils.memory_monitor import process_memory_bytes

            sampled = float(process_memory_bytes(self._pid)) / float(1024**3)
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
            return None
        return sampled if sampled > 0.0 else None

    def _exit_for_memory_fuse(self, reason: str) -> bool:
        """Hard-exit only when the sentinel was created inside a worker child."""
        if not self._hard_exit_allowed:
            logger.critical(
                "MLX worker memory fuse refused hard exit outside an authorized child "
                "process: %s",
                reason,
            )
            self._stop_event.set()
            return False
        os._exit(137)

    # ~10s of continuous probe blindness (0.5s cadence) before the lost
    # enforcement surface is escalated to parent health.
    BLIND_PROBE_THRESHOLD = 20

    def run(self):
        consecutive_blind = 0
        blind_reported = False
        while not self._stop_event.is_set():
            try:
                snapshot = None
                try:
                    from core.utils.memory_monitor import get_memory_pressure_snapshot

                    snapshot = get_memory_pressure_snapshot()
                except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as probe_exc:
                    logger.debug("MLX worker memory pressure probe unavailable: %s", probe_exc)
                rss_gb = self._sample_rss_gb()

                if snapshot is None and rss_gb is None:
                    # Fully blind: BOTH enforcement surfaces (per-worker RSS
                    # fuse and system-pressure fuse) are suspended. That is a
                    # lost safety layer, not a debug detail — it must reach
                    # parent health with a receipt instead of failing open
                    # silently until the host OOMs.
                    consecutive_blind += 1
                    if (
                        consecutive_blind >= self.BLIND_PROBE_THRESHOLD
                        and not blind_reported
                    ):
                        blind_reported = True
                        blind_error = RuntimeError(
                            "memory_sentinel_blind: RSS and pressure probes both failing"
                        )
                        _record_mlx_degradation(
                            blind_error,
                            action="memory fuse enforcement suspended while probes fail",
                            severity="critical",
                        )
                        self.writer.put(
                            {
                                "status": "degraded",
                                "action": "memory_sentinel_degraded",
                                "message": str(blind_error),
                                "model": os.path.basename(self.model_path),
                            }
                        )
                    time.sleep(0.5)
                    continue
                if blind_reported:
                    logger.warning(
                        "🟢 [MLX_MEMORY] Memory sentinel probes recovered after %d blind cycles.",
                        consecutive_blind,
                    )
                    self.writer.put(
                        {
                            "status": "degraded",
                            "action": "memory_sentinel_recovered",
                            "message": f"probes recovered after {consecutive_blind} blind cycles",
                            "model": os.path.basename(self.model_path),
                        }
                    )
                consecutive_blind = 0
                blind_reported = False

                total_gb = float(getattr(snapshot, "total_gb", 0.0) or 0.0)
                rss_limit_gb = self._worker_rss_limit_gb(total_gb)
                reason = ""
                if rss_gb is not None and rss_gb >= rss_limit_gb:
                    reason = f"worker_rss:{rss_gb:.1f}GB/{rss_limit_gb:.1f}GB"
                elif snapshot is not None and snapshot.emergency:
                    reason = snapshot.reason or "system_memory_emergency"
                elif snapshot is not None and snapshot.available_gb < max(
                    1.0, snapshot.min_available_gb / 2.0
                ):
                    reason = snapshot.reason or f"available_memory:{snapshot.available_gb:.1f}GB"

                if reason:
                    message = f"MLX worker memory fuse tripped for {os.path.basename(self.model_path)}: {reason}"
                    logger.critical("🛑 [MLX_MEMORY] %s", message)
                    fuse_payload = {
                        "status": "error",
                        "action": "memory_fuse",
                        "message": message,
                        "memory_pressure": snapshot.to_dict() if snapshot is not None else {},
                    }
                    if self._hard_exit_allowed:
                        # The process dies in the next call: deliver the
                        # diagnostic synchronously and flush the pipe, or the
                        # parent sees only an unexplained SIGKILL-style death.
                        self.writer.put_terminal_direct(fuse_payload)
                    else:
                        self.writer.put(fuse_payload)
                    self._exit_for_memory_fuse(reason)
                    return
            except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("MLX worker memory sentinel probe unavailable: %s", exc)
            time.sleep(0.5)

# Set environment variables for MLX stability
def _setup_worker_env():
    import os
    import platform

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    # [PERFORMANCE] Fast-path: Use environment if already probed by parent.
    # The cached value gets the SAME prefix validation as fresh xcrun output —
    # existence alone let a stale or injected env var redirect compilation to
    # an arbitrary directory.
    _sdk_allowed_prefixes = ("/Library/", "/Applications/Xcode", "/usr/")
    cached_sdk = _FLAG_SDK_PATH.value()
    if (
        cached_sdk
        and os.path.exists(cached_sdk)
        and any(cached_sdk.startswith(pfx) for pfx in _sdk_allowed_prefixes)
    ):
        os.environ["SDKROOT"] = cached_sdk
        logger.info("Using cached SDK root: %s", cached_sdk)
    else:
        if cached_sdk:
            logger.warning(
                "Cached AURA_SDK_PATH rejected (missing or outside allowed prefixes); reprobing: %s",
                cached_sdk,
            )
        try:
            proc = get_subprocess_gateway().run(
                ["xcrun", "--show-sdk-path"],
                timeout=2.0,
                source="mlx_worker_env.sdkroot_probe",
                read_only=True,
                accelerator_capability="none",
            )
            sdk_path = (proc.stdout or "").strip()
            if proc.returncode != 0 or not sdk_path:
                raise RuntimeError((proc.stderr or "xcrun failed").strip())
            if not any(sdk_path.startswith(pfx) for pfx in _sdk_allowed_prefixes):
                raise RuntimeError(f"Suspicious SDK path rejected: {sdk_path}")
            os.environ["SDKROOT"] = sdk_path
            os.environ["AURA_SDK_PATH"] = sdk_path # Cache for subsequent spawns
        except (OSError, RuntimeError, TimeoutError, ValueError) as e:
            _record_mlx_degradation(
                e,
                action="continued worker startup without probed SDKROOT",
                severity="degraded",
            )
            logger.warning("MLX worker SDKROOT probe failed: %s", e)

    try:
        ver_info = platform.mac_ver()
        release_str = ver_info[0]
        ver_parts = release_str.split(".")
        mac_ver = ".".join(ver_parts[:2])
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = mac_ver

        sdk_path = os.environ.get("SDKROOT", "")
        sdk_inc = os.path.join(sdk_path, "usr", "include")
        cpp_inc = "/Library/Developer/CommandLineTools/usr/include/c++/v1"
        cpath_parts = []
        if sdk_path and os.path.exists(sdk_inc):
            cpath_parts.append(sdk_inc)
        if os.path.exists(cpp_inc):
            cpath_parts.append(cpp_inc)
        if cpath_parts:
            os.environ["CPATH"] = ":".join(cpath_parts + [os.environ.get("CPATH", "")]).strip(":")
    except (OSError, RuntimeError, ValueError) as e:
        _record_mlx_degradation(
            e,
            action="continued worker startup without derived Mac deployment target/CPATH",
            severity="degraded",
        )
        logger.warning("MLX worker deployment target/CPATH probe failed: %s", e)

    # Thread budget derived from the actual host instead of one hardware
    # profile: hard-coding 10 oversubscribed smaller machines and stacked
    # multi-worker deployments. Explicit env wins; otherwise leave 2 cores
    # for the parent runtime and IPC threads, floor 4 for decode throughput.
    configured_threads = _FLAG_MLX_NUM_THREADS.value().strip()
    if configured_threads.isdigit() and int(configured_threads) > 0:
        mlx_threads = int(configured_threads)
    else:
        try:
            from core.runtime.resource_observation import get_resource_observer

            host_cpus = get_resource_observer().compute().cpu_count
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            host_cpus = 8
        mlx_threads = max(4, min(10, host_cpus - 2))
    os.environ["MLX_NUM_THREADS"] = str(mlx_threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MLX_FORCE_SERIAL_COMPILE"] = "1"
    os.environ["METAL_COMPILER_TIMEOUT_MS"] = "60000"  # [FRONTIER UPGRADE] Extended for 32B model complex prompts
    os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"


def _clear_mlx_cache(mx_module: Any) -> None:
    try:
        mx_module.clear_cache()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        try:
            mx_module.metal.clear_cache()
        except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
            _record_mlx_degradation(
                _exc,
                action="continued after MLX cache clear fallback failed",
                severity="degraded",
            )
            logger.debug("Suppressed Exception: %s", _exc)


def _reclaim_unified_recurrent_probe_memory(mx_module: Any) -> bool:
    """Synchronously release transient probe graphs before the next case.

    A shadow canary executes many differently shaped base/recurrent decodes in
    one resident process. Python frame exit alone does not force MLX's
    asynchronous allocator to retire those graphs, so resident memory can grow
    until the worker fuse trips. This is a correctness boundary, not an
    optional optimization: a probe is not acknowledged until queued Metal work
    is complete and its transient allocator cache has been reclaimed.
    """

    try:
        gc.collect()
        mx_module.synchronize()
        mx_module.clear_cache()
        mx_module.synchronize()
        return True
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action=(
                "refused recurrent shadow probe acknowledgement because "
                "transient MLX memory could not be reclaimed"
            ),
            severity="critical",
        )
        return False


def _attach_certified_recurrent_adapter(
    model: Any,
    *,
    model_path: str,
    personality_adapter_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach the certified recurrent adapter configured for this live state.

    An absent default pointer means no adapter has earned promotion yet. Once
    an operator explicitly configures a pointer, every missing, stale,
    untrusted, identity-mismatched, or unloadable artifact is fatal to worker
    initialization; serving the base path while claiming trained recurrence
    would invalidate both runtime truth and the scientific certificate.
    """

    from core.brain.llm.latent_cortex.runtime_identity import (
        WORKER_ACTIVATION_SCHEMA,
        inactive_worker_recurrent_adapter_activation,
    )

    configured_pointer = os.environ.get(
        "AURA_RLC_ACTIVATION_POINTER",
        "",
    ).strip()
    activation_root = (
        state_root() / "data/adapters/latent-cortex"
    )
    pointer_path = (
        Path(configured_pointer).expanduser()
        if configured_pointer
        else activation_root / "active.json"
    )
    if not pointer_path.exists():
        if configured_pointer:
            raise RuntimeError(
                f"configured_recurrent_adapter_pointer_missing:{pointer_path}"
            )
        return inactive_worker_recurrent_adapter_activation(), None

    configured_trust_root = os.environ.get(
        "AURA_RLC_ACTIVATION_TRUST_ROOT",
        "",
    ).strip()
    trust_root_path = (
        Path(configured_trust_root).expanduser()
        if configured_trust_root
        else activation_root / "trust-root.pem"
    )
    try:
        from core.brain.llm.latent_cortex.live_adapter_activation import (
            read_live_adapter_trust_root,
        )

        trusted_root = read_live_adapter_trust_root(trust_root_path)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"recurrent_adapter_trust_root_unreadable:{trust_root_path}"
        ) from exc

    approved_root = activation_root / "releases"
    from core.brain.llm.latent_cortex.live_adapter_activation import (
        attach_certified_live_adapter,
    )

    receipt = attach_certified_live_adapter(
        model,
        model_path=model_path,
        personality_adapter_path=personality_adapter_path,
        pointer_path=pointer_path,
        trusted_root_public_key_pem=trusted_root,
        approved_adapter_roots=(approved_root,),
        now_unix=int(time.time()),
    )
    identity = receipt.get("adapter_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("recurrent_adapter_identity_receipt_missing")
    activation = {
        "schema": WORKER_ACTIVATION_SCHEMA,
        "configured": True,
        "active": True,
        "reason": "certified_gain_proven",
        "receipt_sha256": receipt["receipt_sha256"],
        "activation_sha256": receipt["activation_sha256"],
        "adapter_composite_identity_sha256": identity[
            "composite_identity_sha256"
        ],
        "campaign_name": receipt["campaign_name"],
        "claim_tier": receipt["claim_tier"],
        "verified_verdict": receipt["verified_verdict"],
        "loaded_projection_count": receipt["loaded_projection_count"],
    }
    return activation, receipt


def _load_unified_recurrent_shadow(
    model: Any,
    tokenizer: Any,
    *,
    model_path: str,
) -> tuple[Any | None, dict[str, Any]]:
    """Load separately certified recurrent tissue with zero serving authority."""

    from core.brain.llm.unified_recurrent_shadow_contract import (
        LOAD_SCHEMA,
        seal_shadow_load_receipt,
        shadow_load_receipt_errors,
    )

    configured = _FLAG_UNIFIED_RECURRENT_SHADOW_PACKAGE.value().strip()
    pointer_path: Path | None = None
    if configured:
        package = Path(configured).expanduser()
    else:
        from core.brain.llm.unified_recurrent_shadow_pointer import (
            default_shadow_activation_paths,
            resolve_shadow_pointer,
        )

        pointer_path, releases_root = default_shadow_activation_paths()
        if pointer_path.exists() or pointer_path.is_symlink():
            try:
                package = resolve_shadow_pointer(
                    pointer_path,
                    releases_root=releases_root,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"configured_unified_recurrent_shadow_pointer_invalid:"
                    f"{pointer_path}:{exc}"
                ) from exc
        else:
            package = None
    if package is None:
        return None, seal_shadow_load_receipt(
            {
                "schema": LOAD_SCHEMA,
                "configured": False,
                "loaded": False,
                "reason": "not_configured",
                "package_id": "",
                "manifest_sha256": "",
                "checkpoint_sha256": "",
                "controller_sha256": "",
                "families": [],
                "task_depths": [],
                "recurrence_depth": 0,
                "model_identity_strength": "none",
                "mode": "shadow_only",
                "serving_authority": False,
            }
        )

    try:
        from core.brain.llm.unified_recurrent_shadow import (
            load_unified_recurrent_shadow,
        )

        loaded = load_unified_recurrent_shadow(
            package,
            model=model,
            tokenizer=tokenizer,
            model_path=Path(model_path),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # A shadow that does not match this model is DECLINED, not fatal.
        #
        # LIVE, 2026-08-13: every boot logged
        #   MLXWorker CRITICAL: configured_unified_recurrent_shadow_invalid:
        #   .../releases/cp366-source-migrated-recurrent: unified shadow
        #   mechanics or resident model binding differs
        #   🔄 [MLX] Worker init failed ... Retrying spawn
        #
        # The integrity check is right — that package was built against a
        # different resident binding and must not be used. But it runs in
        # shadow_only mode with serving_authority False, so it is by
        # construction not load-bearing, and refusing it took the whole worker
        # down and put boot into a respawn loop. An optional component that
        # cannot serve must not be able to stop the thing it decorates.
        record_degradation(
            "mlx_worker",
            exc,
            action=(
                "declined an incompatible unified recurrent shadow and started "
                "the worker without it"
            ),
        )
        return None, seal_shadow_load_receipt(
            {
                "schema": LOAD_SCHEMA,
                "configured": True,
                "loaded": False,
                "reason": f"incompatible:{exc}"[:300],
                "package_id": str(package),
                "manifest_sha256": "",
                "checkpoint_sha256": "",
                "controller_sha256": "",
                "families": [],
                "task_depths": [],
                "recurrence_depth": 0,
                "model_identity_strength": "none",
                "mode": "shadow_only",
                "serving_authority": False,
            }
        )
    receipt = dict(loaded.receipt)
    errors = shadow_load_receipt_errors(receipt)
    if errors:
        raise RuntimeError(
            "configured_unified_recurrent_shadow_receipt_invalid:"
            + ",".join(errors)
        )
    return loaded, receipt


def _handle_unified_recurrent_shadow_probe(
    job: dict[str, Any],
    *,
    loaded_shadow: Any | None,
    model: Any,
    contract_key: bytes | None,
    cancel_check: Callable[[], bool] | None = None,
    activity: Callable[[], None] | None = None,
    reclaim: Callable[[], bool],
) -> dict[str, Any]:
    """Execute one authenticated shadow probe and expose no model output."""

    if job.get("action") != "unified_recurrent_shadow_probe":
        raise ValueError("unified_recurrent_shadow_probe_action_differs")
    refusal = _verify_contract_authority(job, contract_key)
    if refusal:
        raise ValueError(refusal)
    if loaded_shadow is None:
        raise RuntimeError("unified_recurrent_shadow_not_loaded")
    try:
        receipt = loaded_shadow.probe(
            model,
            job.get("unified_recurrent_shadow_contract"),
            cancel_check=cancel_check,
            activity=activity,
        )
    finally:
        reclaimed = reclaim()
    if reclaimed is not True:
        raise RuntimeError("unified_recurrent_shadow_probe_memory_not_reclaimed")
    return {
        "id": str(job.get("id") or ""),
        "action": "unified_recurrent_shadow_probe",
        "status": "ok",
        "receipt": receipt,
        "allocator_reclaimed": True,
    }


def _load_unified_recurrent_qualified_activation(
    loaded_shadow: Any | None,
    shadow_status: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load typed serving authority separately from non-serving tissue."""

    from core.brain.llm.unified_recurrent_qualified_activation import (
        activation_matches_shadow_receipt,
        seal_qualified_activation_load_receipt,
    )
    from core.brain.llm.unified_recurrent_qualified_activation_store import (
        default_qualified_activation_path,
        read_qualified_activation,
    )

    activation_path = default_qualified_activation_path()
    if not (activation_path.exists() or activation_path.is_symlink()):
        return None, seal_qualified_activation_load_receipt(
            configured=False,
            loaded=False,
            reason="not_configured",
            activation=None,
        )
    try:
        activation = read_qualified_activation(activation_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"configured_unified_recurrent_qualified_activation_invalid:"
            f"{activation_path}:{exc}"
        ) from exc
    if loaded_shadow is None or shadow_status.get("loaded") is not True:
        # Authority fails closed, but the ordinary model worker does not.
        # This activation cannot authorize chat or arbitrary reasoning and
        # cannot execute without its exact shadow tissue.  Keeping it loaded
        # as *inactive evidence* makes that refusal explicit without letting
        # a stale optional package crash unrelated base inference.
        return None, seal_qualified_activation_load_receipt(
            configured=True,
            loaded=False,
            reason="qualified_activation_shadow_unavailable",
            activation=activation,
        )
    if not activation_matches_shadow_receipt(activation, shadow_status):
        raise RuntimeError("qualified_activation_shadow_identity_differs")
    if activation.get("mode") == "qualified_typed_pending":
        return None, seal_qualified_activation_load_receipt(
            configured=True,
            loaded=False,
            reason="qualified_activation_pending_canary",
            activation=activation,
        )
    receipt = seal_qualified_activation_load_receipt(
        configured=True,
        loaded=True,
        reason="qualified_activation_loaded",
        activation=activation,
    )
    return activation, receipt


def _handle_unified_recurrent_qualified_decode(
    job: dict[str, Any],
    *,
    loaded_shadow: Any | None,
    qualified_activation: Mapping[str, Any] | None,
    model: Any,
    contract_key: bytes | None,
    consumed_canary_nonces: set[str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    activity: Callable[[], None] | None = None,
    reclaim: Callable[[], bool],
) -> dict[str, Any]:
    """Execute one typed decode only after exact activation admission."""

    if job.get("action") != "unified_recurrent_qualified_decode":
        raise ValueError("unified_recurrent_qualified_decode_action_differs")
    refusal = _verify_contract_authority(job, contract_key)
    if refusal:
        raise ValueError(refusal)
    if loaded_shadow is None:
        raise RuntimeError("unified_recurrent_shadow_not_loaded")
    canary_activation = job.get("unified_recurrent_qualified_canary_activation")
    canary_authority = job.get("unified_recurrent_qualified_canary_authority")
    if qualified_activation is None and not isinstance(canary_activation, Mapping):
        raise RuntimeError("unified_recurrent_qualified_activation_not_loaded")
    from core.brain.llm.unified_recurrent_qualified_activation import (
        activation_matches_shadow_receipt,
        qualified_activation_errors,
    )
    from core.brain.llm.unified_recurrent_qualified_decode import (
        authorize_qualified_decode_result,
        qualified_canary_authority_matches,
        run_qualified_decode,
    )

    if canary_activation is not None:
        if qualified_activation is not None:
            raise RuntimeError("qualified_canary_requires_inactive_durable_authority")
        if qualified_activation_errors(canary_activation):
            raise RuntimeError("qualified_canary_activation_invalid")
        if (
            not isinstance(canary_authority, Mapping)
            or not isinstance(getattr(loaded_shadow, "canary_battery", None), Mapping)
            or not qualified_canary_authority_matches(
                canary_authority,
                activation=canary_activation,
                request=job.get("unified_recurrent_qualified_decode_contract") or {},
                battery=loaded_shadow.canary_battery,
            )
        ):
            raise RuntimeError("qualified_canary_authority_invalid")
        nonce = str(canary_authority.get("nonce") or "")
        if consumed_canary_nonces is None or nonce in consumed_canary_nonces:
            raise RuntimeError("qualified_canary_authority_replayed")
        consumed_canary_nonces.add(nonce)
        effective_activation = canary_activation
    else:
        effective_activation = qualified_activation
    if not isinstance(effective_activation, Mapping) or not activation_matches_shadow_receipt(
        effective_activation,
        loaded_shadow.receipt,
    ):
        raise RuntimeError("qualified_activation_shadow_identity_differs")
    def report_progress(event: Mapping[str, int | str]) -> None:
        log_progress = logger.info if canary_activation is not None else logger.debug
        log_progress(
            "Unified recurrent qualified decode progress: stage=%s "
            "generated=%s/%s",
            event.get("stage"),
            event.get("generated_token_count"),
            event.get("maximum_token_count"),
        )

    try:
        result = run_qualified_decode(
            loaded_shadow,
            model,
            job.get("unified_recurrent_qualified_decode_contract"),
            cancel_check=cancel_check,
            activity=activity,
            progress=report_progress,
        )
        authorized = authorize_qualified_decode_result(
            result,
            effective_activation,
            canary_only=canary_activation is not None,
        )
    finally:
        reclaimed = reclaim()
    if reclaimed is not True:
        raise RuntimeError("unified_recurrent_qualified_decode_memory_not_reclaimed")
    return {
        "id": str(job.get("id") or ""),
        "action": "unified_recurrent_qualified_decode",
        "status": "ok",
        "receipt": authorized,
        "allocator_reclaimed": True,
    }


# ── expert adapter hot attach/detach (in-worker, no model reload) ────────────
# The expert-LoRA library keeps domain-specialist adapters on disk and swaps
# them onto the RESIDENT model. Attach wraps target linears with LoRA layers
# and loads the adapter weights (~40MB); detach restores exactly the modules
# THIS attach wrapped — the personality adapter (loaded with the model) and
# any inner wrapping survive untouched. A partial attach failure is benign:
# freshly wrapped LoRA layers initialize with B=0 (identity) until weights
# load, and detach unwinds whatever was recorded.

_EXPERT_LORA_LAYER_TYPES = ("LoRALinear", "DoRALinear", "LoRASwitchLinear", "LoRAEmbedding")


def _named_lora_module_ids(model: Any) -> set[int]:
    return {
        id(module)
        for _name, module in model.named_modules()
        if type(module).__name__ in _EXPERT_LORA_LAYER_TYPES
    }


def _expert_adapter_approved_roots() -> list[Path]:
    """Directories an IPC-supplied adapter path may resolve under."""
    roots = [
        Path(str(state_root() / "data/adapters")),
        Path(str(shared_asset_root() / "models")),
    ]
    try:
        # Repo artifacts: training pipelines publish adapters here.
        roots.append(Path(__file__).resolve().parents[3] / "artifacts")
    except (OSError, IndexError):
        logger.debug("Repo artifacts root unavailable for adapter policy.")
    configured = _FLAG_EXPERT_ADAPTER_ROOTS.value()
    for extra in configured.split(os.pathsep):
        extra = extra.strip()
        if extra:
            roots.append(Path(os.path.expanduser(extra)))
    return roots


# Fine-tune types whose load produces DETACHABLE wrapper modules. Anything
# else mutates resident weights in place and cannot be undone on a live
# model, so it is refused rather than attached and discovered later.
_RESTORABLE_FINE_TUNE_TYPES = frozenset({"lora", "qlora"})


def _validate_expert_adapter_dir(adapter_dir: str) -> Path:
    """Structural + policy validation BEFORE any resident-weight mutation.

    set_expert_adapter previously accepted any path from any IPC caller and
    mutated resident weights first, validating nothing: no approved-root
    policy, no adapter structure check. Validation failures here leave the
    resident model completely untouched.
    """
    path = Path(str(adapter_dir or "")).expanduser()
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"expert_adapter_dir_unresolvable:{adapter_dir}") from exc
    if not path.is_dir():
        raise NotADirectoryError(f"expert_adapter_not_a_directory:{path}")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"expert_adapter_missing_config:{path}")
    try:
        adapter_config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"expert_adapter_config_unparseable:{path}") from exc
    # CP126 15f68852. The config was parsed and then ignored. mlx_lm's
    # load_adapters honours fine_tune_type: a "full" (or "dora"/unknown)
    # adapter does NOT produce restorable wrapper modules — it writes into
    # the resident weights. Detach can only unwrap .linear/.embedding
    # wrappers, so such a load permanently rewrites the personality model
    # this worker is serving, with no way back short of a reload.
    #
    # The type is therefore constrained BEFORE any weight mutation, rather
    # than discovered afterwards by a detach that silently restores nothing.
    if not isinstance(adapter_config, dict):
        raise ValueError(f"expert_adapter_config_not_an_object:{path}")
    fine_tune_type = str(
        adapter_config.get("fine_tune_type")
        or adapter_config.get("fine_tune")
        # Absent means LoRA in mlx_lm's own default.
        or "lora"
    ).strip().lower()
    if fine_tune_type not in _RESTORABLE_FINE_TUNE_TYPES:
        raise ValueError(
            f"expert_adapter_fine_tune_type_not_restorable:{fine_tune_type}:{path}"
        )
    has_weights = any(path.glob("*.safetensors")) or (path / "adapters.npz").is_file()
    if not has_weights:
        raise ValueError(f"expert_adapter_missing_weights:{path}")
    for root in _expert_adapter_approved_roots():
        try:
            resolved_root = root.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        try:
            path.relative_to(resolved_root)
            return path
        except ValueError:
            continue
    raise PermissionError(f"expert_adapter_path_not_approved:{path}")


def _unrestorable_wrapped(wrapped: list[tuple[str, Any]]) -> list[str]:
    """Names of wrapped modules detach cannot restore to their base form.

    Detach restores ``.linear`` and ``.embedding`` wrappers. Anything else
    (in-place or full-weight mutation) is irreversible on the resident
    model — a swap over it would stack identities, so callers must treat a
    non-empty result as swap-blocking.
    """
    return [
        name
        for name, module in wrapped
        if not (hasattr(module, "linear") or hasattr(module, "embedding"))
    ]


def _attach_expert_adapter(model: Any, adapter_dir: str) -> list[tuple[str, Any]]:
    """Attach adapter weights onto the resident model; return the wrapped modules."""
    from mlx_lm.tuner.utils import load_adapters

    before = _named_lora_module_ids(model)

    def _newly_wrapped() -> list[tuple[str, Any]]:
        return [
            (name, module)
            for name, module in model.named_modules()
            if type(module).__name__ in _EXPERT_LORA_LAYER_TYPES and id(module) not in before
        ]

    try:
        load_adapters(model, adapter_dir)
    except (FileNotFoundError, KeyError, RuntimeError, AttributeError, TypeError, ValueError, OSError):
        # Unwind a partial wrap so the module tree stays exactly as it was.
        # (Even unwound-late these layers are identity: LoRA B initializes 0.)
        _detach_expert_adapter(model, _newly_wrapped())
        raise
    wrapped = _newly_wrapped()
    # Belt and braces for CP126 15f68852: the config said the load would be
    # restorable, but the authority on what actually happened is the module
    # tree. A load that wrapped nothing either did nothing or mutated weights
    # in place, and a wrap we cannot unwrap is an identity change with no way
    # back — both are refused here, while the unwind is still possible.
    unrestorable = _unrestorable_wrapped(wrapped)
    if unrestorable:
        _detach_expert_adapter(model, wrapped)
        raise RuntimeError(
            "expert_adapter_attach_unrestorable:" + ",".join(unrestorable[:4])
        )
    if not wrapped:
        raise RuntimeError(f"expert_adapter_attach_wrapped_nothing:{adapter_dir}")
    return wrapped


def _detach_expert_adapter(model: Any, wrapped: list[tuple[str, Any]]) -> int:
    """Restore exactly the modules a previous attach wrapped.

    Restores both linear-class wrappers (.linear) and LoRAEmbedding
    (.embedding) — the tracked class set always included LoRAEmbedding but
    the old restore path silently skipped it, leaving expert embedding
    weights resident after a "successful" detach. Callers detect
    irreversible wraps up front via _unrestorable_wrapped.
    """
    from mlx.utils import tree_unflatten

    restorable = []
    for name, module in wrapped:
        if hasattr(module, "linear"):
            restorable.append((name, module.linear))
        elif hasattr(module, "embedding"):
            restorable.append((name, module.embedding))
    if restorable:
        model.update_modules(tree_unflatten(restorable))
    return len(restorable)


def _process_message_content(messages: list[dict[str, Any]]) -> None:
    """Normalize content for tokenizer.apply_chat_template()."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment.get("text", "")
                for fragment in content
                if isinstance(fragment, dict) and fragment.get("type") == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only text content fragments are supported in MLX worker chat templates.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""


# The fallback used when a checkpoint declares no window anywhere. It is a
# guess, and a guess that decides how much prompt the worker will accept has
# to say so — every path that reaches it records a degradation naming the
# reason, so "the model told us" and "we assumed" are distinguishable.
_ASSUMED_CONTEXT_WINDOW = 32768


def _load_effective_context_window(model_path: str) -> int:
    path = Path(str(model_path))
    if not path.exists():
        _record_mlx_degradation(
            FileNotFoundError(f"model_path_missing:{model_path}"),
            action=(
                "assumed a "
                f"{_ASSUMED_CONTEXT_WINDOW}-token context window for an "
                "unreadable checkpoint"
            ),
            severity="warning",
        )
        observed = _ASSUMED_CONTEXT_WINDOW
        limits = _qualified_serving_limits_for_model(model_path)
        return min(observed, int(limits.served_context_tokens)) if limits else observed

    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"

    max_position_embeddings = 0
    sliding_window = 0
    use_sliding_window = False
    tokenizer_model_max = 0

    try:
        if config_path.exists():
            config_payload = json.loads(config_path.read_text())
            text_config = config_payload.get("text_config")
            context_config = (
                text_config if isinstance(text_config, dict) else config_payload
            )
            max_position_embeddings = int(
                context_config.get("max_position_embeddings") or 0
            )
            sliding_window = int(context_config.get("sliding_window") or 0)
            use_sliding_window = bool(context_config.get("use_sliding_window"))
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        max_position_embeddings = 0
        sliding_window = 0
        use_sliding_window = False

    try:
        if tokenizer_config_path.exists():
            tokenizer_payload = json.loads(tokenizer_config_path.read_text())
            tokenizer_model_max = int(tokenizer_payload.get("model_max_length") or 0)
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        tokenizer_model_max = 0

    def _bounded(window: int) -> int:
        # Sane bounds on discovered metadata: tokenizer sentinel values
        # (e.g. 1e30) previously authorized enormous prompts and memory
        # allocation, and a malformed tiny value would break every request.
        return max(2048, min(int(window), 262144))

    def _serving_bounded(window: int) -> int:
        observed = _bounded(window)
        limits = _qualified_serving_limits_for_model(model_path)
        if limits is None:
            return observed
        return min(observed, int(limits.served_context_tokens))

    if max_position_embeddings > 0:
        if use_sliding_window and sliding_window > max_position_embeddings:
            return _serving_bounded(max(sliding_window, max_position_embeddings))
        return _serving_bounded(max_position_embeddings)
    if use_sliding_window and sliding_window > 0:
        return _serving_bounded(sliding_window)
    if tokenizer_model_max > 0:
        return _serving_bounded(tokenizer_model_max)
    _record_mlx_degradation(
        ValueError(f"context_window_undeclared:{path.name}"),
        action=(
            "assumed a "
            f"{_ASSUMED_CONTEXT_WINDOW}-token context window; the checkpoint "
            "declares neither max_position_embeddings, sliding_window nor "
            "model_max_length"
        ),
        severity="warning",
    )
    limits = _qualified_serving_limits_for_model(model_path)
    if limits is not None:
        return min(_ASSUMED_CONTEXT_WINDOW, int(limits.served_context_tokens))
    return _ASSUMED_CONTEXT_WINDOW


@dataclass
class _PromptCacheEntry:
    prompt_cache: list[Any]
    count: int


@dataclass
class _PromptCacheSearchResult:
    exact: list[int] | None
    shorter: list[int] | None
    longer: list[int] | None
    common_prefix: int


@dataclass(frozen=True)
class _ArraysCacheMemberRollback:
    index: int
    state: tuple[Any, ...]
    left_padding: Any
    lengths: Any

    @property
    def nbytes(self) -> int:
        total = 0
        for value in (*self.state, self.left_padding, self.lengths):
            if value is not None:
                total += max(0, int(getattr(value, "nbytes", 0) or 0))
        return total


@dataclass(frozen=True)
class _PromptCacheOneTokenRollback:
    members: tuple[_ArraysCacheMemberRollback, ...]

    @property
    def nbytes(self) -> int:
        return sum(member.nbytes for member in self.members)


def _capture_prompt_cache_one_token_rollback(
    prompt_cache: list[Any] | None,
) -> _PromptCacheOneTokenRollback | None:
    """Retain the fixed-size state needed to rewind a hybrid cache one token.

    ``mlx_lm`` can trim KV caches, but Qwen3.5 combines those with
    ``ArraysCache`` recurrent states that intentionally have no inverse. The
    recurrent arrays are replaced on each model call rather than mutated in
    place, so retaining their immediately preceding array references is an
    exact, fixed-size rollback image. Unknown non-trimmable cache kinds are
    refused instead of being guessed compatible.
    """

    if not prompt_cache:
        return None
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        return None

    members: list[_ArraysCacheMemberRollback] = []
    for index, cache_member in enumerate(prompt_cache):
        try:
            if bool(cache_member.is_trimmable()):
                continue
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if not isinstance(cache_member, ArraysCache):
            return None
        state = cache_member.state
        if not isinstance(state, list):
            return None
        members.append(
            _ArraysCacheMemberRollback(
                index=index,
                state=tuple(state),
                left_padding=getattr(cache_member, "left_padding", None),
                lengths=getattr(cache_member, "lengths", None),
            )
        )
    if not members:
        return None
    return _PromptCacheOneTokenRollback(tuple(members))


def _rewind_hybrid_prompt_cache_one_token(
    prompt_cache: list[Any],
    rollback: _PromptCacheOneTokenRollback | None,
) -> tuple[bool, str]:
    """Rewind mixed KV/recurrent cache state without reconstructing the prompt."""

    if rollback is None:
        return False, "hybrid_rollback_unavailable"
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        return False, "mlx_cache_types_unavailable"

    snapshots = {member.index: member for member in rollback.members}
    nontrimmable_indexes: set[int] = set()
    trimmable_members: list[Any] = []
    for index, cache_member in enumerate(prompt_cache):
        try:
            trimmable = bool(cache_member.is_trimmable())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False, "cache_trim_contract_unavailable"
        if trimmable:
            try:
                if int(cache_member.size()) < 1:
                    return False, "trimmable_cache_empty"
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False, "trimmable_cache_size_unavailable"
            trimmable_members.append(cache_member)
            continue
        nontrimmable_indexes.add(index)
        if not isinstance(cache_member, ArraysCache) or index not in snapshots:
            return False, "nontrimmable_cache_not_snapshotted"

    if nontrimmable_indexes != set(snapshots):
        return False, "hybrid_cache_layout_changed"

    # Validation above completes before the first mutation. From this point all
    # operations are the declared one-token inverse for their cache kind.
    for cache_member in trimmable_members:
        if int(cache_member.trim(1)) != 1:
            return False, "trimmable_cache_rewind_failed"
    for index, snapshot in snapshots.items():
        cache_member = prompt_cache[index]
        cache_member.state = list(snapshot.state)
        cache_member.left_padding = snapshot.left_padding
        cache_member.lengths = snapshot.lengths
    return True, ""


@dataclass(frozen=True)
class _PromptCacheResumeBinding:
    model_key: Any
    tokens: tuple[int, ...]
    prompt_cache: list[Any]
    one_token_rollback: _PromptCacheOneTokenRollback | None
    created_at: float


class _PromptCacheLRU:
    def __init__(
        self,
        max_size: int = 12,
        max_entry_tokens: int = 0,
        max_total_tokens: int = 0,
        kv_bytes_per_token: int = 0,
        fixed_bytes_per_entry: int = 0,
        max_total_bytes: int = 0,
    ):
        self.max_size = max_size
        # 0 = uncapped. A positive cap refuses to RETAIN prompts longer than
        # this many tokens, bounding per-entry KV RAM on heavy models while
        # leaving generation itself untouched.
        self.max_entry_tokens = max_entry_tokens
        # An entry COUNT is not a memory bound. Twelve entries of unbounded
        # length is unbounded memory: once insertion actually worked, a
        # 31,718-token prompt was measured live and managed RSS grew
        # 73,963MB/h toward the 49GB ceiling. This bounds the total.
        self.max_total_tokens = max_total_tokens
        self.kv_bytes_per_token = kv_bytes_per_token
        self.fixed_bytes_per_entry = max(0, int(fixed_bytes_per_entry))
        self.max_total_bytes = max(0, int(max_total_bytes))
        self._cache: dict[Any, dict[Any, Any]] = {}
        # One eviction queue PER LANE, not one globally. A single global queue
        # meant Aura's internal lanes (loop ticks, enrichment, dreaming,
        # health probes — dozens of generations per minute) evicted the user
        # conversation's entry within seconds of it being written, so the one
        # entry whose reuse decides whether a conversation survives was always
        # the first one thrown away. Lane budgets still sum to max_size, so
        # per-entry KV RAM is bounded exactly as before.
        self._lru: dict[str, deque] = {}
        # A continuation capability names exact worker-owned KV state without
        # copying that state through IPC or reconstructing it from visible text.
        self._resume_bindings: dict[str, _PromptCacheResumeBinding] = {}
        self._resume_ttl_s = 300.0
        self._resume_binding_limit = max(2, min(8, max_size))

    # ── introspection: what is actually retained right now ───────────────
    def retained_tokens(self) -> int:
        """Total cached tokens across every lane. The real memory driver."""
        reusable = sum(
            len(tokens)
            for queue in self._lru.values()
            for (_model_key, tokens) in queue
        )
        resumable = sum(len(binding.tokens) for binding in self._resume_bindings.values())
        return reusable + resumable

    def retained_entries(self) -> int:
        return sum(len(queue) for queue in self._lru.values()) + len(
            self._resume_bindings
        )

    def retained_bytes(self) -> int:
        """Approximate KV bytes held. Reported to the OOM ladder, not guessed."""
        token_bytes = (
            self.retained_tokens() * self.kv_bytes_per_token
            if self.kv_bytes_per_token > 0
            else 0
        )
        rollback_bytes = sum(
            binding.one_token_rollback.nbytes
            for binding in self._resume_bindings.values()
            if binding.one_token_rollback is not None
        )
        fixed_bytes = self.retained_entries() * self.fixed_bytes_per_entry
        return token_bytes + fixed_bytes + rollback_bytes

    def shed(self) -> int:
        """Release everything and report the bytes freed.

        This is the OOM ladder's rung. The ladder had none — the verifier said
        so on every boot ("no organ exposes a shed hook, so the OOM ladder has
        no rungs: the only available response to memory pressure is a
        restart") — while this cache was the largest trivially-droppable
        allocation in the process.
        """
        freed = self.retained_bytes()
        self.clear()
        return freed

    def _enforce_total_token_budget(self) -> None:
        """Evict oldest entries until total retained tokens fit the budget.

        Per-lane entry budgets bound how many prefixes stay reusable; this
        bounds the MEMORY. The user-surface lane is drained last so a
        conversation keeps its prefix while internal lanes give theirs up.
        """
        if self.max_total_tokens <= 0:
            return
        lanes_by_drain_order = sorted(
            self._lru.keys(), key=lambda lane: (lane == "user_surface", lane)
        )
        byte_budget = self.max_total_bytes
        if byte_budget <= 0 and self.max_total_tokens > 0 and self.kv_bytes_per_token > 0:
            byte_budget = self.max_total_tokens * self.kv_bytes_per_token
        while (
            self.retained_tokens() > self.max_total_tokens
            or (byte_budget > 0 and self.retained_bytes() > byte_budget)
        ):
            evicted = False
            for lane in lanes_by_drain_order:
                queue = self._lru.get(lane)
                if queue:
                    evict_model_key, evict_tokens = queue.popleft()
                    self._delete(evict_model_key, list(evict_tokens))
                    evicted = True
                    break
            if evicted:
                continue
            oldest_resume = next(iter(self._resume_bindings), None)
            if oldest_resume is not None:
                self._resume_bindings.pop(oldest_resume, None)
            else:
                return

    def _lane_of(self, model_key: Any) -> str:
        # A single-entry budget cannot be split without overspending it, so
        # every lane shares one queue and eviction stays global.
        if self.max_size <= 1:
            return "shared"
        if isinstance(model_key, tuple) and len(model_key) >= 2:
            return str(model_key[1])
        return "default"

    def _lane_budget(self, lane: str) -> int:
        if self.max_size <= 1 or lane == "shared":
            return self.max_size
        # Asymmetric on purpose. The conversation is already protected by having
        # its OWN queue, and only its newest entry is ever reused — turn N+1
        # extends turn N, so older conversation entries are dead weight holding
        # the largest KV in the cache. The internal lane is the opposite: it
        # carries many DISTINCT prompt families (the reflective persona, the
        # pre-linguistic decision narrator, enrichment, dreaming), and with a
        # 50/50 split they evicted each other on every tick. Measured live,
        # repeatedly: "trimmed hit — reused 3/792 tokens", the same two families
        # taking turns destroying each other's prefix.
        reserved = max(1, min(3, self.max_size - 1))
        return reserved if lane == "user_surface" else self.max_size - reserved

    def _queue_for(self, lane: str) -> deque:
        queue_for_lane = self._lru.get(lane)
        if queue_for_lane is None:
            queue_for_lane = deque()
            self._lru[lane] = queue_for_lane
        return queue_for_lane

    def _forget_key(self, cache_key: tuple) -> None:
        queue_for_lane = self._lru.get(self._lane_of(cache_key[0]))
        if queue_for_lane is None:
            return
        try:
            queue_for_lane.remove(cache_key)
        except ValueError as exc:
            logger.debug("Prompt cache LRU entry already absent: %s", exc)

    def clear(self) -> None:
        self._cache.clear()
        self._lru.clear()
        self._resume_bindings.clear()

    def _prune_resume_bindings(self, *, now: float | None = None) -> None:
        observed_at = time.monotonic() if now is None else float(now)
        expired = [
            handle
            for handle, binding in self._resume_bindings.items()
            if observed_at - binding.created_at > self._resume_ttl_s
        ]
        for handle in expired:
            self._resume_bindings.pop(handle, None)
        while len(self._resume_bindings) > self._resume_binding_limit:
            oldest = next(iter(self._resume_bindings), None)
            if oldest is None:
                break
            self._resume_bindings.pop(oldest, None)

    def bind_resume(
        self,
        model_key: Any,
        tokens: list[int],
        *,
        prompt_cache: list[Any] | None = None,
        one_token_rollback: _PromptCacheOneTokenRollback | None = None,
    ) -> str:
        """Move exact final KV state into a short-lived continuation capability.

        A resumable generation is a transaction boundary, not an ordinary cache
        hint.  The capability therefore owns the actual final cache object.  A
        later trie lookup allowed eviction or insertion-policy differences to
        turn a valid continuation into a silent full re-prefill.
        """

        if len(tokens) < 2:
            return ""
        if self.max_entry_tokens > 0 and len(tokens) > self.max_entry_tokens:
            return ""
        exact = self._search(model_key, tokens).exact
        if exact is not None:
            owned_cache = self._extract(model_key, exact).prompt_cache
        elif prompt_cache is not None:
            owned_cache = prompt_cache
        else:
            return ""
        self._prune_resume_bindings()
        handle = uuid.uuid4().hex
        self._resume_bindings[handle] = _PromptCacheResumeBinding(
            model_key=model_key,
            tokens=tuple(int(token) for token in tokens),
            prompt_cache=owned_cache,
            one_token_rollback=one_token_rollback,
            created_at=time.monotonic(),
        )
        self._prune_resume_bindings()
        self._enforce_total_token_budget()
        return handle if handle in self._resume_bindings else ""

    def fetch_resume(
        self,
        handle: str,
        model_key: Any,
        *,
        can_trim_prompt_cache: Any,
        trim_prompt_cache: Any,
    ) -> tuple[list[Any] | None, list[int], list[int], str]:
        """Consume an exact continuation capability and replay one token."""

        normalized = str(handle or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", normalized):
            return None, [], [], "invalid_handle"
        self._prune_resume_bindings()
        binding = self._resume_bindings.pop(normalized, None)
        if binding is None:
            return None, [], [], "unknown_or_expired_handle"
        if binding.model_key != model_key:
            # A capability presented on the wrong lane must not disclose KV,
            # but it also must not let another lane destroy the rightful
            # continuation. Return ownership to the scoped general cache.
            self.insert_cache(
                binding.model_key,
                list(binding.tokens),
                binding.prompt_cache,
            )
            return None, [], [], "model_or_lane_mismatch"
        tokens = list(binding.tokens)
        prompt_cache = binding.prompt_cache
        if not can_trim_prompt_cache(prompt_cache):
            rewound, rewind_failure = _rewind_hybrid_prompt_cache_one_token(
                prompt_cache,
                binding.one_token_rollback,
            )
            if not rewound:
                self.insert_cache(model_key, tokens, prompt_cache)
                return None, [], [], rewind_failure
            resume_kind = "hybrid recurrent/KV"
        else:
            trim_prompt_cache(prompt_cache, 1)
            resume_kind = "KV"
        logger.info(
            "🎯 [PROMPT CACHE] %s continuation resume — reused %d/%d tokens, "
            "1 to prefill.",
            resume_kind,
            len(tokens) - 1,
            len(tokens),
        )
        return prompt_cache, tokens[-1:], tokens, ""

    def _search(self, model_key: Any, tokens: list[int]) -> _PromptCacheSearchResult:
        if model_key not in self._cache:
            return _PromptCacheSearchResult(None, None, None, 0)

        current = self._cache[model_key]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return _PromptCacheSearchResult(tokens, None, None, 0)

        # Index 0 is a valid one-token cached prefix; the old > 0 test threw
        # it away and forced an avoidable full prefill.
        shorter = tokens[: last_cache_index + 1] if last_cache_index >= 0 else None
        longer = None
        common_prefix = index
        if index > 0 and last_cache_index < 0:
            best = None
            stack = [(current, [])]
            while stack:
                node, extra = stack.pop()
                if "cache" in node:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in node:
                        stack.append((node[tok], extra + [tok]))
            if best is not None:
                longer = tokens[:index] + best

        return _PromptCacheSearchResult(None, shorter, longer, common_prefix)

    def _get(self, model_key: int, tokens: list[int]) -> _PromptCacheEntry:
        current = self._cache[model_key]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model_key: int, tokens: list[int]) -> None:
        path = [self._cache[model_key]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for index in reversed(range(len(tokens))):
            prev_node, node, tok = path[index], path[index + 1], tokens[index]
            if len(node) > 0:
                break
            del prev_node[tok]

    def _extract(self, model_key: int, tokens: list[int]) -> _PromptCacheEntry:
        cache_entry = self._get(model_key, tokens)
        if cache_entry.count == 1:
            self._delete(model_key, tokens)
            self._forget_key((model_key, tuple(tokens)))
            return cache_entry

        cache_entry.count -= 1
        return _PromptCacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)

    def fetch_nearest_cache(
        self,
        model_key: Any,
        tokens: list[int],
        *,
        can_trim_prompt_cache: Any,
        trim_prompt_cache: Any,
    ) -> tuple[list[Any] | None, list[int]]:
        result = self._search(model_key, tokens)
        # Whether prefix reuse actually happens decides whether a conversation
        # survives: every turn that misses re-prefills the entire history, and
        # time-to-first-token climbs until it crosses the turn budget. Measured
        # live 2026-07-26, turns 5-7 of one conversation died that way with the
        # budget shrinking 81.1s -> 73.2s -> 55.8s against a first token that
        # kept taking 58-82s. None of that was visible: there was no hit/miss
        # signal anywhere, so reuse could only be inferred from latency.
        if result.exact is not None and len(tokens) > 1:
            # Never hand back an EMPTY remainder: mlx_lm has to be given at
            # least one token to run a decode step, so an exact hit reuses
            # everything but the final token and replays that one.
            cache_entry = self._extract(model_key, result.exact)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                trim_prompt_cache(cache_entry.prompt_cache, 1)
                logger.info(
                    "🎯 [PROMPT CACHE] exact hit — reused %d/%d tokens, 1 to prefill.",
                    len(tokens) - 1, len(tokens),
                )
                return cache_entry.prompt_cache, tokens[-1:]
            # Untrimmable cache: putting it back keeps it available for the
            # prefix path instead of silently dropping a live entry.
            self.insert_cache(model_key, list(result.exact), cache_entry.prompt_cache)

        if result.shorter is not None:
            cache_entry = self._extract(model_key, result.shorter)
            prefix_len = len(result.shorter)
            logger.info(
                "🎯 [PROMPT CACHE] prefix hit — reused %d/%d tokens, %d to prefill.",
                prefix_len, len(tokens), len(tokens) - prefix_len,
            )
            return cache_entry.prompt_cache, tokens[prefix_len:]

        if result.longer is not None:
            cache_entry = self._get(model_key, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                # _extract already copies when the entry is shared and hands
                # over the live object when it is not. Deep-copying here
                # unconditionally cost a second full KV allocation — ~1.5GB on
                # the 32B geometry — on the hot path of every diverging turn.
                trimmed = self._extract(model_key, result.longer)
                trim_prompt_cache(trimmed.prompt_cache, num_to_trim)
                logger.info(
                    "🎯 [PROMPT CACHE] trimmed hit — reused %d/%d tokens, %d to prefill.",
                    prefix, len(tokens), len(tokens) - prefix,
                )
                return trimmed.prompt_cache, tokens[prefix:]

        logger.info(
            "🧊 [PROMPT CACHE] miss — prefilling all %d tokens.", len(tokens)
        )
        return None, tokens

    def insert_cache(self, model_key: Any, tokens: list[int], prompt_cache: list[Any]) -> None:
        if self.max_entry_tokens > 0 and len(tokens) > self.max_entry_tokens:
            return
        if model_key not in self._cache:
            self._cache[model_key] = {}
        current = self._cache[model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        cache_key = (model_key, tuple(tokens))
        if "cache" in current:
            current["cache"].count += 1
            self._forget_key(cache_key)
        else:
            current["cache"] = _PromptCacheEntry(prompt_cache, 1)

        lane = self._lane_of(model_key)
        queue_for_lane = self._queue_for(lane)
        queue_for_lane.append(cache_key)
        while len(queue_for_lane) > self._lane_budget(lane):
            evict_model_key, evict_tokens = queue_for_lane.popleft()
            self._delete(evict_model_key, list(evict_tokens))
        self._enforce_total_token_budget()

class JobWatchdog(threading.Thread):
    """
    Kills the worker process if a job is active but no tokens have been generated
    within the timeout. This prevents 'Metal Stalls' from hanging the system.

    [STABILITY v51] Reduced timeout from 240s → 90s. The 32B model's Metal
    shader compilation should complete within 60s on M5 hardware. If no token
    progress after 90s, the worker is stuck and must self-terminate so the
    parent can respawn it.
    """
    def __init__(self, timeout=60.0, writer: IPCWriterThread | None = None):
        super().__init__(daemon=True)
        self.timeout = timeout
        self.writer = writer
        self.last_activity = time.monotonic()
        self.active_job = False
        self.current_request_id = ""
        self.current_action = ""
        self._stop_event = threading.Event()

    def activity(self):
        self.last_activity = time.monotonic()

    def start_job(self, request_id: str = "", action: str = ""):
        self.current_request_id = str(request_id or "")
        self.current_action = str(action or "")
        self.active_job = True
        self.last_activity = time.monotonic()

    def stop_job(self):
        self.active_job = False
        self.current_request_id = ""
        self.current_action = ""

    def snapshot(self) -> dict[str, Any]:
        """Progress evidence for the heartbeat: is a job active, how stale."""
        active = bool(self.active_job)
        age_s = max(0.0, time.monotonic() - self.last_activity) if active else 0.0
        return {
            "active_job": active,
            "job_age_s": round(age_s, 3),
            "request_id": self.current_request_id if active else "",
        }

    def run(self):
        while not self._stop_event.is_set():
            if self.active_job and (time.monotonic() - self.last_activity > self.timeout):
                request_id = self.current_request_id
                logger.critical(
                    "🛑 [MLX_WATCHDOG] Job timeout triggered (%ss) for request %s. "
                    "Self-terminating worker.",
                    self.timeout,
                    request_id or "<unknown>",
                )
                # Killing IS the recovery for a wedged Metal stall (there is
                # no soft-cancel of stuck GPU work), but dying without a
                # correlated terminal receipt left the parent to diagnose an
                # opaque crash. Deliver attribution synchronously first.
                if self.writer is not None:
                    delivered = self.writer.put_terminal_direct(
                        {
                            "id": request_id,
                            "status": "error",
                            "action": self.current_action or "generate",
                            "watchdog_timeout": True,
                            "attribution": "worker_job_watchdog",
                            "timeout_s": float(self.timeout),
                            "message": (
                                "MLX worker watchdog killed the process: no token "
                                f"progress for {float(self.timeout):.0f}s"
                            ),
                        }
                    )
                    if not delivered:
                        logger.critical(
                            "🛑 [MLX_WATCHDOG] Terminal receipt could not be flushed; "
                            "parent will see an unattributed worker death."
                        )
                os._exit(2)
            time.sleep(1.0)

def soft_cancel_requested(cancel_seq: Any, job_seq: int) -> bool:
    """True when the parent asked THIS job to stop between tokens.

    Cooperative preemption: the parent writes the target job's sequence
    number into shared memory; the token loop polls it each step. Cancel
    latency is one decode step and the model stays warm — unlike
    force-abort, which kills the worker and pays a full model reload.
    """
    if cancel_seq is None or job_seq <= 0:
        return False
    try:
        return int(getattr(cancel_seq, "value", 0)) == int(job_seq)
    except (TypeError, ValueError, OSError):
        return False


def clear_stale_soft_cancel(cancel_seq: Any, job_seq: int) -> None:
    """Reset a cancel flag left over from a job that ended before observing it.

    Shared-memory hygiene at job start: a stale flag must not cancel an
    unrelated new job, and must not wedge the parent's soft-cancel ack-wait
    (which treats a cleared flag as proof the worker's token loop is alive).
    """
    if cancel_seq is None:
        return
    try:
        stale = int(getattr(cancel_seq, "value", 0))
        # Only a LOWER sequence is stale (an older request that ended before
        # observing its cancel). A HIGHER sequence is a legitimate
        # cancellation already posted for a later queued job — clearing it
        # here made that job uncancellable.
        if stale != 0 and stale < int(job_seq):
            cancel_seq.value = 0
    except (TypeError, ValueError, OSError):
        logger.debug("Stale soft-cancel clear failed; continuing.")


#: One encoder per worker, built on first use. Constructing it reads the
#: model's hidden size and nothing else, so it is cheap to hold and wrong to
#: rebuild per request.
_hidden_encoder: dict[str, Any] = {"encoder": None}


def _run_nonparametric_ingest_job(
    model: Any,
    tokenizer: Any,
    job: dict[str, Any],
    *,
    cancel_seq: Any = None,
    progress: Any = None,
    clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Ingest a tiny trusted batch using the resident worker model.

    Keeping this operation inside the model worker is the ownership boundary:
    the orchestrator must not load a second Cortex merely to derive keys.  One
    pair is encoded with one causal forward, with hard sequence, position, and
    wall-clock budgets so foreground service remains the dominant workload.
    """

    from core.brain.nonparametric_generation import MLXEncoder
    from core.brain.nonparametric_identity import TrustLevel
    from core.brain.nonparametric_ingest import (
        NonParametricIngestor,
        collect_trusted_pairs_by_source,
        ingest_provenance,
    )
    from core.brain.nonparametric_memory import (
        SHARED_MEMORY_PRINCIPAL,
        get_nonparametric_memory,
    )

    max_pairs = max(1, min(4, int(job.get("max_pairs") or 1)))
    scan_limit = max(max_pairs, min(64, int(job.get("scan_limit") or 16)))
    max_positions = max(1, min(256, int(job.get("max_positions") or 96)))
    max_sequence_tokens = max(
        8,
        min(512, int(job.get("max_sequence_tokens") or 192)),
    )
    deadline_s = max(1.0, min(30.0, _safe_float(job.get("deadline_s"), 20.0)))
    deadline_at = float(clock()) + deadline_s
    job_seq = max(0, _safe_int(job.get("seq"), 0))
    stop_reason = ""

    def _should_continue() -> bool:
        nonlocal stop_reason
        if soft_cancel_requested(cancel_seq, job_seq):
            stop_reason = "soft_cancelled"
            return False
        if float(clock()) >= deadline_at:
            stop_reason = "deadline_reached"
            return False
        return True

    dim = int(getattr(getattr(model, "args", None), "hidden_size", 0) or 0)
    if dim <= 0:
        return {
            "state": "model_hidden_size_unavailable",
            "pairs_scanned": 0,
            "pairs_ingested": 0,
            "positions_ingested": 0,
        }
    memory = get_nonparametric_memory(dim)
    if memory is None:
        return {
            "state": "memory_unavailable",
            "pairs_scanned": 0,
            "pairs_ingested": 0,
            "positions_ingested": 0,
        }

    # Collection is bounded BY the admitted scan limit (small multiple for
    # the has_seen filter), not a fixed >=500 floor that let a "tiny
    # bounded batch" job inspect the entire oversized collection.
    #
    # Grouped by source, because every entry has to name the trusted store
    # that produced it. The ingestor used to be built with no provenance at
    # all, so each entry it wrote was unattributed, unverified and owned by
    # "anonymous" — the three values the store's revocation and
    # per-principal erasure need before they can do anything.
    grouped = collect_trusted_pairs_by_source(limit=min(256, scan_limit * 4))
    collected_pairs = [
        (context, answer, source) for source, found in grouped for context, answer in found
    ]
    ingestors = {
        source: NonParametricIngestor(
            memory,
            provenance=ingest_provenance(
                # Aura's own verified reasoning results belong to the
                # system, not to a person, and the store has a name for
                # that: `shared` is readable by every principal.
                principal=SHARED_MEMORY_PRINCIPAL,
                source_id=f"trusted_store:{source}",
                trust=TrustLevel.VERIFIED,
                verifier="reasoning_solved_cache",
            ),
        )
        for source, _found in grouped
    }
    ingestor = next(iter(ingestors.values()), None)
    if ingestor is None or not collected_pairs:
        return {
            "state": "no_trusted_pairs",
            "pairs_scanned": 0,
            "pairs_ingested": 0,
            "positions_ingested": 0,
        }
    pairs = [
        (context, answer, source)
        for context, answer, source in collected_pairs
        if not ingestors[source].has_seen(context, answer)
    ]
    if not pairs:
        return {
            "state": "no_new_trusted_pairs",
            "pairs_scanned": 0,
            "pairs_ingested": 0,
            "positions_ingested": 0,
        }

    encoder = MLXEncoder(model, tokenizer)
    pairs_considered = 0
    pairs_scanned = 0
    pairs_ingested = 0
    positions_ingested = 0
    for context, answer, source in pairs:
        ingestor = ingestors[source]
        if (
            pairs_ingested >= max_pairs
            # scan_limit bounds budget-eligible scans; total budget-probe
            # EXAMINATIONS are bounded by the collection cap above
            # (scan_limit * 4, ≤ 256) instead of the old fixed >=500 floor
            # that let a "tiny bounded batch" walk the whole collection.
            or pairs_scanned >= scan_limit
            or not _should_continue()
        ):
            break
        pairs_considered += 1
        if not ingestor.sequence_within_budget(
            context,
            answer,
            encoder,
            max_positions=max_positions,
            max_sequence_tokens=max_sequence_tokens,
        ):
            continue
        pairs_scanned += 1
        added = ingestor.ingest_sequence(
            context,
            answer,
            encoder,
            max_positions=max_positions,
            max_sequence_tokens=max_sequence_tokens,
            should_continue=_should_continue,
        )
        if added > 0:
            pairs_ingested += 1
            positions_ingested += int(added)
        if callable(progress):
            progress(
                {
                    "pairs_considered": pairs_considered,
                    "pairs_scanned": pairs_scanned,
                    "pairs_ingested": pairs_ingested,
                    "positions_ingested": positions_ingested,
                }
            )

    if positions_ingested > 0:
        if not memory.persist():
            raise RuntimeError("nonparametric_memory_persist_failed")
        if not ingestor.persist_seen() and not ingestor.persist_seen():
            # Memory is already durable but the dedupe receipt is not: the
            # same pairs can be re-ingested next run (duplicate capacity
            # waste, not corruption). One bounded retry, then a typed
            # PARTIAL-COMMIT error so the caller never mistakes this for a
            # clean failure it can blindly repeat.
            _record_mlx_degradation(
                RuntimeError("nonparametric_dedupe_receipt_lost_after_memory_commit"),
                action="reported partial nonparametric commit (memory durable, dedupe receipt not)",
                severity="error",
            )
            raise RuntimeError(
                "nonparametric_ingest_receipt_persist_failed_after_memory_commit:"
                f"pairs_ingested={pairs_ingested}:positions={positions_ingested}"
            )
    state = stop_reason or (
        "complete" if positions_ingested > 0 else "no_new_eligible_pairs"
    )
    return {
        "state": state,
        "pairs_considered": pairs_considered,
        "pairs_scanned": pairs_scanned,
        "pairs_ingested": pairs_ingested,
        "positions_ingested": positions_ingested,
        "max_pairs": max_pairs,
        "max_positions": max_positions,
        "max_sequence_tokens": max_sequence_tokens,
    }


def _speculative_eligible(
    draft_model: Any,
    generation_kwargs: dict,
    job: dict,
    *,
    prefill_tokens: int = 0,
    prefill_step_size: int = 0,
) -> bool:
    """Speculative decoding is only safe on the plain generation path.

    The draft model PROPOSES tokens; the steered target model VERIFIES every
    one, so the output distribution is exactly the target's (steering-safe by
    construction). But logits processors, external prompt caches, and schema-
    constrained jobs interact with the speculative loop's internal caching —
    those jobs take the normal path.
    """
    if draft_model is None:
        return False
    if job.get("schema"):
        return False
    if "logits_processors" in generation_kwargs:
        return False
    if "prompt_cache" in generation_kwargs:
        return False
    # mlx_lm's speculative path discards prompt_progress_callback. It remains
    # safe for a one-chunk prompt; a multi-chunk heavy-model prefill must take
    # the observable target-only path or it is indistinguishable from a stall.
    if (
        "prompt_progress_callback" in generation_kwargs
        and int(prefill_step_size or 0) > 0
        and int(prefill_tokens or 0) > int(prefill_step_size)
    ):
        return False
    return True


def _load_speculative_draft(model_path: str, target_tokenizer: Any) -> Any:
    """Load the small draft model for speculative decoding (heavy lanes only).

    Returns None (never raises) when disabled, missing, or incompatible —
    generation falls back to the normal path.
    """
    enabled = str(_FLAG_SPECULATIVE_DECODING.value()).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        return None
    from core.brain.llm.model_artifact_profile import model_size_class

    if model_size_class(str(model_path)) not in ("72b", "32b"):
        return None  # drafting for a small model is pointless
    draft_candidates = [
        Path(__file__).resolve().parents[3] / "models" / "Qwen2.5-1.5B-Instruct-4bit",
        state_root() / "live-source" / "models" / "Qwen2.5-1.5B-Instruct-4bit",
    ]
    default_draft = next((str(c) for c in draft_candidates if c.is_dir()), str(draft_candidates[0]))
    draft_path = os.environ.get("AURA_SPECULATIVE_DRAFT_PATH", default_draft)
    if not os.path.isdir(draft_path):
        logger.info("Speculative decoding: no draft model at %s; normal path.", draft_path)
        return None
    try:
        from transformers import AutoTokenizer

        draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        mismatch = _tokenizer_identity_mismatch(draft_tokenizer, target_tokenizer)
        if mismatch:
            message = (
                "Speculative decoding: draft tokenizer is incompatible "
                f"({mismatch}): {os.path.basename(draft_path)}; normal path."
            )
            if "AURA_SPECULATIVE_DRAFT_PATH" in os.environ:
                logger.warning(message)
            else:
                logger.info(message)
            return None

        from mlx_lm import load as _load

        draft_model, loaded_draft_tokenizer = _load(draft_path)
        mismatch = _tokenizer_identity_mismatch(loaded_draft_tokenizer, target_tokenizer)
        if mismatch:
            _record_mlx_degradation(
                RuntimeError(
                    f"draft tokenizer mismatch ({mismatch}): {os.path.basename(draft_path)}"
                ),
                action="continued without speculative decoding after tokenizer identity mismatch",
                severity="warning",
            )
            return None
        logger.info(
            "🚀 Speculative decoding ONLINE: draft=%s (target verifies every token; "
            "steering semantics preserved).",
            os.path.basename(draft_path),
        )
        return draft_model
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued without speculative decoding after draft load failed",
            severity="warning",
        )
        logger.warning("Speculative draft load failed (%s); normal path.", exc)
        return None


def _tokenizer_identity_mismatch(draft_tokenizer: Any, target_tokenizer: Any) -> str:
    """Return the first semantic tokenizer mismatch for speculative decoding."""

    for attr in ("vocab_size", "eos_token_id", "bos_token_id", "pad_token_id"):
        draft_value = getattr(draft_tokenizer, attr, None)
        target_value = getattr(target_tokenizer, attr, None)
        if draft_value != target_value:
            return f"{attr}:{draft_value}!={target_value}"
    probes = (
        "Aura verifies every proposed token.",
        "def f(x):\n\treturn {x: [1, 2.5e-3, 'mixed']}  # comment",
        "Numbers 1234567890 and unicode: naïve café — 日本語 テスト Ω≈ç√",
        "  leading spaces\nand\twindows\r\nline endings ",
        "<|im_start|>assistant роль and emoji 🚀🧠",
    )
    for probe in probes:
        if draft_tokenizer.encode(probe) != target_tokenizer.encode(probe):
            return f"probe_tokenization:{probe[:32]!r}"
    return ""


def _active_steering_hooks(engine: Any = None) -> list[Any]:
    """The steering hooks a TokenSentinel should pulse mid-generation.

    TokenSentinel has always accepted `steering_hooks` and documented it as
    "List of AffectiveSteeringHook instances to update" — and neither of the
    two places that construct one ever passed it. So the live-affect pulse,
    the thing that keeps affect current DURING a generation instead of frozen
    at its start, could never run: every sentinel recorded "live affect
    inactive: missing steering_hooks" and raised a MARGINAL runtime fault.
    Twenty-seven of them across six turns of Bryan's demo on 2026-07-29, on a
    worker whose own log line two thousand lines earlier says
    "Affective Steering Engine ONLINE". The hooks were installed the whole
    time; nothing carried them across.

    Returns [] rather than raising: a sentinel with no hooks is the previous
    behaviour, and losing the pulse must never lose the generation.
    """
    try:
        if engine is None:
            from core.consciousness.affective_steering import get_steering_engine

            engine = get_steering_engine()
        return list(engine.active_hooks() or [])
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        logger.debug("Steering hooks unavailable for the token sentinel: %s", exc)
        return []


@dataclass(frozen=True)
class AffectiveSteeringAttachment:
    """Worker-local outcome for cortex-bound affective neural tissue."""

    engine: Any
    active: bool
    affect_expected: bool
    disposition: str


def _attach_affective_steering(
    model: Any,
    tokenizer: Any,
    substrate_mem: Any,
    phi_residual_mem: Any,
    steering_active_flag: Any,
    model_path: str | None = None,
) -> AffectiveSteeringAttachment:
    """The forward arrow: substrate state into the residual stream.

    Steering is optional neural tissue, not the owner of response availability.
    Attachment failures are observable degradations, but the resident model
    remains authoritative and available.  A zero alpha is an intentional
    neutral mode, not a failed liveness condition.

    Also hands the Φ residual ring to each hook. Without it the hooks fall
    back to an in-process PhiCore lookup that is ALWAYS False here — this is
    the worker, PhiCore lives in the parent — which is why the
    activation-grounded complex read 0/50 forever.

    Non-cortex workers are explicitly ineligible rather than misreported as a
    damaged resident cortex. The typed result also tells downstream token
    diagnostics whether missing steering hooks are expected.
    """
    engine = None
    try:
        from core.brain.llm.model_registry import resolve_cortex_bound_artifact
        from core.consciousness.affective_steering import get_steering_engine

        cortex_resolution = resolve_cortex_bound_artifact(model_path)
        if cortex_resolution.status == "non_cortex_model":
            if steering_active_flag is not None:
                steering_active_flag.value = False
            logger.info(
                "Affective steering not applicable to non-cortex model lane: %s.",
                cortex_resolution.model_path,
            )
            return AffectiveSteeringAttachment(
                engine=None,
                active=False,
                affect_expected=False,
                disposition="non_cortex_model",
            )

        engine = get_steering_engine()
        if model_path:
            engine.attach(
                model,
                tokenizer,
                model_path=model_path,
                model_identity=cortex_resolution.descriptor,
            )
        else:
            engine.attach(model, tokenizer)
        available = bool(
            getattr(engine, "_model_attached", False)
            and (getattr(engine, "_hooks", None) or [])
        )
        if substrate_mem is not None and available:
            engine.start_substrate_sync(shared_state=substrate_mem)
        if phi_residual_mem is not None:
            for hook in getattr(engine, "_hooks", None) or []:
                try:
                    hook._phi_residual_channel = phi_residual_mem
                except (AttributeError, TypeError):
                    continue
        active = engine.is_active()
        if steering_active_flag is not None:
            steering_active_flag.value = active

        if active:
            logger.info(
                "🎯 Affective Steering Engine ONLINE (alpha=%.1f, hooks=%d).",
                engine._alpha,
                len(getattr(engine, "_hooks", [])),
            )
            return AffectiveSteeringAttachment(engine, True, True, "active")

        if available:
            logger.info(
                "Affective Steering Engine attached in neutral mode "
                "(alpha=%.3f, hooks=%d).",
                float(getattr(engine, "_alpha", 0.0) or 0.0),
                len(getattr(engine, "_hooks", []) or []),
            )
            return AffectiveSteeringAttachment(engine, False, True, "neutral")

        disposition = str(
            (getattr(engine, "_model_info", None) or {}).get("attachment_error") or ""
        )
        if disposition in {
            "steering_generation_checkpoint_incompatible",
            "steering_generation_deferred",
            "steering_generation_retired",
        }:
            logger.info(
                "Affective steering remains detached under signed migration disposition: %s.",
                disposition,
            )
            return AffectiveSteeringAttachment(engine, False, False, disposition)

        exc = RuntimeError("affective_steering_attach_unavailable")
        record_degradation(
            "affective_steering",
            exc,
            severity="warning",
            action="continued resident-model inference without optional steering tissue",
        )
        logger.warning(
            "Affective steering did not attach; resident-model inference remains available."
        )
        return AffectiveSteeringAttachment(None, False, False, "unavailable")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        if steering_active_flag is not None:
            steering_active_flag.value = False
        record_degradation(
            "affective_steering",
            exc,
            severity="warning",
            action="continued resident-model inference after optional steering attach failed",
        )
        logger.warning("Affective steering attach failed; continuing without it: %s", exc)
        return AffectiveSteeringAttachment(None, False, False, "attach_failed")


def _attach_latent_bridge(model: Any, latent_readout_mem: Any) -> Any:
    """Install the backward arrow: model representations → the substrate.

    Steering carries substrate state INTO the residual stream. This reads the
    model's own representations back out and publishes them to the parent,
    which is where the substrate lives.

    ``attach_latent_bridge()`` had no caller for its whole existence, and
    calling it as written would not have helped: its injector resolved the
    substrate in THIS process, where it does not exist, and injected through
    ``asyncio.get_running_loop()`` from a plain thread, which always raises.
    Both are fixed; the transport is
    :mod:`core.consciousness.latent_readout_channel`.

    Like forward steering, a failure here is not fatal.  Both directions are
    measured tissue around the resident model; either may report unavailable,
    but neither owns whether a correct base-model answer reaches the user.
    """
    if latent_readout_mem is None:
        logger.warning(
            "No latent readout channel; the backward path stays off rather than "
            "collecting deltas nothing will read."
        )
        return None
    try:
        from core.consciousness.latent_bridge import attach_latent_bridge

        bridge = attach_latent_bridge(model, channel=latent_readout_mem)
        if bridge is None:
            logger.warning("LatentBridge did not attach; backward path is off.")
            return None
        bridge.start_substrate_sync(channel=latent_readout_mem)
        logger.info(
            "🔁 LatentBridge ONLINE (%d readout hooks) — model representations "
            "now reach the substrate.",
            len(getattr(bridge, "_readout_hooks", []) or []),
        )
        return bridge
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "latent_bridge",
            exc,
            severity="warning",
            action="continued with forward-only steering after the latent bridge failed to attach",
        )
        logger.warning("LatentBridge attach failed (forward-only): %s", exc)
        return None


def _shutdown_worker_runtime(
    *,
    ipc_writer: Any,
    watchdog: Any,
    heartbeat: Any,
    memory_sentinel: Any,
    latent_bridge: Any = None,
    steering_engine: Any = None,
    prompt_cache_lru: Any = None,
    mx_module: Any = None,
) -> None:
    """Release worker-owned threads and caches before normal process exit."""

    failures: list[BaseException] = []

    def _call(resource: Any, method_name: str) -> None:
        method = getattr(resource, method_name, None)
        if not callable(method):
            return
        try:
            method()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            failures.append(exc)

    _call(watchdog, "stop_job")
    _call(latent_bridge, "stop")
    _call(steering_engine, "stop")
    _call(prompt_cache_lru, "clear")
    if mx_module is not None:
        synchronize = getattr(mx_module, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                failures.append(exc)
        _clear_mlx_cache(mx_module)
    gc.collect()

    # Stop producers before the writer, then give the writer a short bounded
    # chance to move any terminal frame to the multiprocessing queue.
    for thread in (heartbeat, memory_sentinel, watchdog):
        _call(thread, "stop")
    local_queue = getattr(ipc_writer, "local_queue", None)
    deadline = time.monotonic() + 0.5
    while local_queue is not None and time.monotonic() < deadline:
        try:
            if local_queue.empty():
                break
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            break
        time.sleep(0.01)
    _call(ipc_writer, "stop")

    current = threading.current_thread()
    for thread in (heartbeat, memory_sentinel, watchdog, ipc_writer):
        join = getattr(thread, "join", None)
        if not callable(join) or thread is current:
            continue
        try:
            join(timeout=0.25)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            failures.append(exc)

    for failure in failures:
        _record_mlx_degradation(
            failure,
            action="continued model-worker exit after bounded resource cleanup failed",
            severity="warning",
        )


def _mlx_worker_loop(
    model_path: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    device: str = "gpu",
    substrate_mem: Any = None,
    steering_active_flag: Any = None,
    cancel_seq: Any = None,
    contract_key: bytes | None = None,
    worker_capture_launch_challenge: Mapping[str, Any] | None = None,
    phi_residual_mem: Any = None,
    latent_readout_mem: Any = None,
):
    """Runs in a FULLY ISOLATED native subprocess via ForkServer.

    This is the worker entry-point called from ``MLXLocalClient._spawn_worker``.
    All Metal/GPU work, model loading, and inference happen inside this
    function's process boundary.  The parent communicates via IPC queues.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - MLXWorker - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger("MLXWorker")
    worker_boot_id = uuid.uuid4().hex
    from core.brain.llm.latent_cortex.worker_capture_identity import (
        build_worker_capture_identity,
    )

    worker_capture_signing_identity = build_worker_capture_identity(
        worker_boot_id=worker_boot_id,
        worker_pid=os.getpid(),
        launch_challenge=worker_capture_launch_challenge,
    )
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, ValueError) as exc:
        logger.debug("MLX worker SIGINT ignore hook unavailable: %s", exc)

    # Reset the shared steering flag FIRST: a respawn inheriting a prior
    # true value that then fails during imports/model load/steering attach
    # would exit through init failure with the stale "steering active"
    # still published to every parent-side reader.
    if steering_active_flag is not None:
        try:
            steering_active_flag.value = False
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Steering flag pre-init reset failed: %s", exc)

    # Configure worker-specific environment (Metal, SDK, thread limits).
    # This MUST run inside the subprocess, not at module import time,
    # because the parent process should not inherit these settings.
    _setup_worker_env()

    # ── Zenith Concurrency & Telemetry ──
    ipc_writer = IPCWriterThread(response_queue)
    ipc_writer.start()

    # Bind this worker's capture key to the parent-owned launch challenge
    # before model loading begins. Heavy checkpoints can take longer to load
    # than the launch challenge is valid; delaying this identity until the
    # READY receipt made a legitimate child impossible to attest after a slow
    # load. The READY identity must later present this exact same key.
    ipc_writer.put(
        {
            "status": "ok",
            "action": "capture_identity_bootstrap",
            "worker_action_capture_identity": dict(
                worker_capture_signing_identity.public_identity
            ),
        }
    )

    # Watchdog before heartbeat: the heartbeat publishes the watchdog's
    # job-progress snapshot so liveness claims carry inference evidence.
    watchdog = JobWatchdog(timeout=360.0, writer=ipc_writer)  # Align with the protected foreground solver envelope.
    watchdog.start()

    heartbeat = HeartbeatThread(
        ipc_writer,
        watchdog=watchdog,
        worker_boot_id=worker_boot_id,
        worker_pid=os.getpid(),
    )
    heartbeat.start()

    memory_sentinel = WorkerMemorySentinel(
        ipc_writer,
        model_path,
        hard_exit_allowed=mp.current_process().name != "MainProcess",
    )
    memory_sentinel.start()

    engine = None
    latent_bridge = None
    prompt_cache_lru = None
    mx = None

    try:
        import mlx.core as mx

        from core.runtime.desktop_boot_safety import configure_mlx_process_device

        requested_device = "cpu" if str(device).lower() == "cpu" else "metal"
        device_contract = configure_mlx_process_device(
            requested_device,
            reason="model_worker",
            force=True,
        )
        if not device_contract.get("verified"):
            raise RuntimeError(
                "model_worker_mlx_device_unverified:"
                f"{device_contract.get('reason', 'unknown')}"
            )
        device = "cpu" if requested_device == "cpu" else "gpu"
        logger.info(
            "MLX worker default device verified as %s.",
            device_contract["device"],
        )
        # Import the model stack only after the process-local device contract
        # is established. Import-time tensors must never inherit the desktop
        # parent's CPU ownership.
        from mlx_lm import load

        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            try:
                from mlx_lm.sample import make_sampler
            except ImportError:
                make_sampler = None

        logger.info("📡 [WORKER] Loading Core modules...")

    except ImportError:
        logger.error("mlx-lm not installed in worker environment.")
        ipc_writer.put({"status": "error", "message": "mlx-lm missing"})
        _shutdown_worker_runtime(
            ipc_writer=ipc_writer,
            watchdog=watchdog,
            heartbeat=heartbeat,
            memory_sentinel=memory_sentinel,
            latent_bridge=latent_bridge,
            steering_engine=engine,
            prompt_cache_lru=prompt_cache_lru,
            mx_module=mx,
        )
        return

    # VRAM Management
    if mx and device != "cpu":
        try:
            from core.runtime import resource_psutil as psutil

            total_ram = psutil.virtual_memory().total
            limit = compute_mlx_cache_limit(total_ram)
            mx.set_cache_limit(limit)
            logger.info("Metal cache limit set to %sMB", limit // (1024**2))
            memory_limit = compute_mlx_memory_limit(total_ram)
            mx.set_memory_limit(memory_limit)
            logger.info("MLX active memory limit set to %sMB", memory_limit // (1024**2))
        except (ImportError, OSError, RuntimeError, AttributeError) as e:
            _record_mlx_degradation(
                e,
                action="fell back to conservative Metal cache limit after adaptive cache limit failed",
                severity="degraded",
            )
            try:
                # The fallback must not assume a large host: fixed 24GB/40GB
                # limits authorized more memory than smaller or pressured
                # machines could give. Derive conservatively from total RAM
                # when observable; use small-safe limits when it is not.
                try:
                    from core.runtime.resource_observation import get_resource_observer

                    _total_gb = float(
                        get_resource_observer()
                        .memory(include_process_tree=False)
                        .total_bytes
                    ) / float(1024**3)
                except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError):
                    _total_gb = 0.0
                if _total_gb >= 96.0:
                    _cache_gb, _active_gb = 24, 40
                elif _total_gb >= 48.0:
                    _cache_gb, _active_gb = 12, 28
                elif _total_gb > 0.0:
                    _cache_gb, _active_gb = 4, 12
                else:
                    # Unobservable capacity: smallest useful limits.
                    _cache_gb, _active_gb = 4, 12
                mx.metal.set_cache_limit(1024 * 1024 * 1024 * _cache_gb)
                if hasattr(mx, "set_memory_limit"):
                    mx.set_memory_limit(1024 * 1024 * 1024 * _active_gb)
            except (AttributeError, RuntimeError, ValueError) as fallback_exc:
                _record_mlx_degradation(
                    fallback_exc,
                    action="continued without explicit Metal cache limit after fallback failed",
                    severity="degraded",
                )

    # [PERFORMANCE] Metal probes shifted to after model load or triggered on demand
    # Initializing the model first is more critical for 'perceived' speed.

    # ZENITH: Local Concurrency Gate
    metal_semaphore = threading.Semaphore(1)

    # [STABILITY v53.9] Load with LoRA adapter. Intermittent float32 errors
    # are caught at generation time and retried — most generations succeed.
    # The adapter is the v3 training (22 characters, val loss 0.102).
    try:
        adapter_path = resolve_personality_adapter(model_path, backend="mlx")
        logger.info("Loading model: %s", model_path)
        # Truthful adapter identity for the init receipt: production must be
        # able to SEE a silent personality loss (base-model fallback), not
        # discover it from drifted behavior. severity escalates because a
        # worker that lost its trained personality/RLC state while reporting
        # a plain ok init is a identity-integrity failure, not a nicety.
        personality_adapter_status: dict[str, Any] = {
            "requested": str(adapter_path or ""),
            "applied": "",
            "fallback_base_model": False,
            "error": "",
        }
        if adapter_path and os.path.isdir(adapter_path):
            try:
                logger.info("Loading with LoRA adapter: %s", adapter_path)
                model, tokenizer = load(model_path, adapter_path=adapter_path)
                personality_adapter_status["applied"] = str(adapter_path)
                logger.info("Model loaded with Aura personality LoRA fused.")
            except (RuntimeError, AttributeError, TypeError, ValueError) as adapter_exc:
                personality_adapter_status["fallback_base_model"] = True
                personality_adapter_status["error"] = (
                    f"{type(adapter_exc).__name__}: {adapter_exc}"
                )
                _record_mlx_degradation(
                    adapter_exc,
                    action="loaded base model after personality LoRA load failed — trained identity NOT resident",
                    severity="critical",
                )
                logger.warning(
                    "⚠️ [WORKER] LoRA adapter failed to load for %s: %s. Using base model + prompt hardening.",
                    os.path.basename(model_path),
                    adapter_exc,
                )
                model, tokenizer = load(model_path)
                logger.info("Model loaded without LoRA (prompt hardening active).")
        else:
            if adapter_path:
                personality_adapter_status["fallback_base_model"] = True
                personality_adapter_status["error"] = "resolved_adapter_dir_missing"
                _record_mlx_degradation(
                    RuntimeError(f"personality_adapter_dir_missing:{adapter_path}"),
                    action="loaded base model because resolved personality adapter dir is missing",
                    severity="critical",
                )
            model, tokenizer = load(model_path)
            logger.info("Model loaded (no compatible LoRA adapter).")

        draft_model = _load_speculative_draft(model_path, tokenizer)

        # Both arrows of the substrate<->activation coupling.
        steering_attachment = _attach_affective_steering(
            model,
            tokenizer,
            substrate_mem,
            phi_residual_mem,
            steering_active_flag,
            model_path=model_path,
        )
        engine = steering_attachment.engine
        _steering_active = steering_attachment.active
        _affect_expected = steering_attachment.affect_expected
        if engine is not None and getattr(engine, "_model_attached", False):
            latent_bridge = _attach_latent_bridge(model, latent_readout_mem)

        # Write steering liveness to shared state so parent can query it
        if substrate_mem is not None:
            try:
                # Convention: substrate_mem[-1] = 1.0 if steering active, 0.0 if not
                # (substrate_mem is a multiprocessing.Array of floats; last slot reserved)
                substrate_mem[-1] = 1.0 if _steering_active else 0.0
            except (TypeError, ValueError, IndexError) as shared_state_exc:
                _record_mlx_degradation(
                    shared_state_exc,
                    action="continued with parent steering liveness shared-state unavailable",
                    severity="warning",
                )

        # Apply Recurrent Depth — Mythos-inspired layer looping.
        # This changes HOW the model processes: middle layers loop N times,
        # letting the model "think" in latent space before committing to output.
        # Active by default for 32B+ models. Set AURA_RECURRENT_LOOPS=0 to disable.
        recurrent_depth_status = {
            "active": False,
            "config": None,
            "expected_loops": None,
            "required": False,
            "reason": "",
            "error": "",
        }
        try:
            from core.brain.llm.recurrent_depth import (
                apply_for_model,
                get_recurrent_config,
                resolve_loops_for_model,
            )

            expected_loops = resolve_loops_for_model(model)
            recurrent_depth_status["expected_loops"] = expected_loops
            recurrent_depth_status["required"] = expected_loops > 1
            if expected_loops <= 1:
                recurrent_depth_status["reason"] = "standard_or_operator_disabled"
            elif apply_for_model(model):
                recurrent_depth_status = {
                    "active": True,
                    "config": get_recurrent_config(model),
                    "expected_loops": expected_loops,
                    "required": expected_loops > 1,
                    "reason": "",
                    "error": "",
                }
                logger.info("🧠 Recurrent Depth ACTIVE — model now thinks before answering.")
            else:
                recurrent_depth_status["reason"] = "patch_not_applied"
                # REQUIRED depth that failed to apply is a degradation, not a
                # silent status field — readiness checks and the parent's
                # supervision must be able to see it.
                _record_mlx_degradation(
                    RuntimeError(
                        f"required_recurrent_depth_inactive:loops={expected_loops}"
                    ),
                    action="initialized worker with required recurrent depth NOT applied",
                    severity="warning",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as rd_exc:
            explicit_disable = str(_FLAG_RECURRENT_LOOPS.value()).strip() == "0"
            size_disable = str(_FLAG_RECURRENT_LOOPS_32B.value()).strip() == "0"
            from core.brain.llm.model_artifact_profile import model_size_class as _msc

            recurrent_depth_status["required"] = (
                _msc(str(model_path)) == "32b"
                and not explicit_disable
                and not size_disable
            )
            recurrent_depth_status["reason"] = "recurrent_depth_error"
            recurrent_depth_status["error"] = f"{type(rd_exc).__name__}: {rd_exc}"
            _record_mlx_degradation(
                rd_exc,
                action="continued inference with recurrent depth disabled",
                severity="degraded",
            )
            logger.warning("Recurrent depth not applied: %s", rd_exc)

        (
            recurrent_adapter_activation,
            recurrent_adapter_activation_receipt,
        ) = _attach_certified_recurrent_adapter(
            model,
            model_path=str(model_path),
            personality_adapter_path=(
                personality_adapter_status["applied"] or None
            ),
        )
        if recurrent_adapter_activation["active"]:
            logger.info(
                "🧠 Certified recurrent adapter ACTIVE — campaign=%s "
                "projections=%d receipt=%s",
                recurrent_adapter_activation["campaign_name"],
                recurrent_adapter_activation["loaded_projection_count"],
                recurrent_adapter_activation["receipt_sha256"],
            )
        else:
            logger.info(
                "Optional recurrent LoRA adapter inactive (independent of the CP568 "
                "semantic-neural serving lane): %s",
                recurrent_adapter_activation["reason"],
            )

        (
            unified_recurrent_shadow,
            unified_recurrent_shadow_status,
        ) = _load_unified_recurrent_shadow(
            model,
            tokenizer,
            model_path=str(model_path),
        )
        if unified_recurrent_shadow_status["loaded"]:
            logger.info(
                "Unified recurrent tissue loaded in SHADOW ONLY mode: "
                "package=%s controller=%s",
                unified_recurrent_shadow_status["package_id"],
                unified_recurrent_shadow_status["controller_sha256"],
            )
        else:
            logger.info(
                "Optional unified recurrent shadow package inactive (independent of "
                "the CP568 semantic-neural serving lane): %s",
                unified_recurrent_shadow_status["reason"],
            )
        (
            unified_recurrent_qualified_activation,
            unified_recurrent_qualified_activation_status,
        ) = _load_unified_recurrent_qualified_activation(
            unified_recurrent_shadow,
            unified_recurrent_shadow_status,
        )
        if unified_recurrent_qualified_activation_status["loaded"]:
            logger.info(
                "Unified recurrent tissue admitted for QUALIFIED TYPED serving: "
                "activation=%s",
                unified_recurrent_qualified_activation_status["activation"][
                    "activation_sha256"
                ],
            )
        else:
            logger.info(
                "Optional unified typed-controller serving inactive (independent of "
                "the CP568 semantic-neural serving lane): %s",
                unified_recurrent_qualified_activation_status["reason"],
            )
        consumed_unified_recurrent_canary_nonces: set[str] = set()

        from core.brain.llm.latent_cortex.runtime_identity import (
            build_worker_identity,
        )

        def _current_worker_identity() -> dict[str, Any]:
            """Measure the exact serving identity after every model mutation."""

            return build_worker_identity(
                model,
                model_path=model_path,
                worker_boot_id=worker_boot_id,
                worker_source_path=Path(__file__),
                worker_action_capture_identity=(
                    worker_capture_signing_identity.public_identity
                ),
                tokenizer=tokenizer,
                affective_steering_active=bool(_steering_active),
                affective_steering_alpha=float(
                    getattr(engine, "_alpha", 0.0) or 0.0
                ),
                recurrent_adapter_activation=recurrent_adapter_activation,
            )

        # Measure after personality adaptation, affective steering, and
        # recurrent-depth installation. The init receipt must describe the
        # worker that will actually serve, not an earlier boot phase.
        worker_identity = _current_worker_identity()
        token_budget_calibration = _token_budget_calibration_evidence(tokenizer)

        ipc_writer.put(
            {
                "status": "ok",
                "action": "init",
                "device": device,
                "steering_active": bool(_steering_active),
                "recurrent_depth": recurrent_depth_status,
                "recurrent_adapter_activation": dict(
                    recurrent_adapter_activation
                ),
                "recurrent_adapter_activation_receipt": (
                    dict(recurrent_adapter_activation_receipt)
                    if recurrent_adapter_activation_receipt is not None
                    else None
                ),
                "unified_recurrent_shadow": dict(unified_recurrent_shadow_status),
                "unified_recurrent_qualified_activation": dict(
                    unified_recurrent_qualified_activation_status
                ),
                "personality_adapter": dict(personality_adapter_status),
                "token_budget_calibration": token_budget_calibration,
                "worker_identity": dict(worker_identity),
            }
        )
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as e:
        _record_mlx_degradation(
            e,
            action="reported initialization error and exited worker loop before accepting jobs",
            severity="critical",
        )
        import traceback
        err_detail = f"{e}\n{traceback.format_exc()}"
        logger.error("Worker Init Error: %s", err_detail)
        ipc_writer.put(
            {
                "status": "error",
                "action": "init",
                "message": f"Init failed: {e}",
                "detail": err_detail,
            }
        )
        _shutdown_worker_runtime(
            ipc_writer=ipc_writer,
            watchdog=watchdog,
            heartbeat=heartbeat,
            memory_sentinel=memory_sentinel,
            latent_bridge=latent_bridge,
            steering_engine=engine,
            prompt_cache_lru=prompt_cache_lru,
            mx_module=mx,
        )
        return
    # ZENITH: Prompt Cache LRU for massive speedup in multi-turn
    # Discovered once at init; enforced at every tokenization boundary so a
    # prompt can never overrun the model into Metal work that fails late.
    effective_context_window = _load_effective_context_window(model_path)
    logger.info("Effective context window: %d tokens.", effective_context_window)

    prompt_cache_budget = _prompt_cache_entry_budget_for_model(model_path)
    prompt_cache_token_cap = _prompt_cache_entry_token_cap_for_model(model_path)
    prompt_cache_total_tokens = _prompt_cache_total_token_budget_for_model(model_path)
    prompt_cache_kv_bytes = _prompt_cache_kv_bytes_per_token(model_path)
    prompt_cache_fixed_bytes = _prompt_cache_fixed_bytes_per_entry_for_model(model_path)
    prompt_cache_total_bytes = _prompt_cache_total_byte_budget_for_model(model_path)
    prompt_cache_lru = (
        _PromptCacheLRU(
            max_size=prompt_cache_budget,
            max_entry_tokens=prompt_cache_token_cap,
            max_total_tokens=prompt_cache_total_tokens,
            kv_bytes_per_token=prompt_cache_kv_bytes,
            fixed_bytes_per_entry=prompt_cache_fixed_bytes,
            max_total_bytes=prompt_cache_total_bytes,
        )
        if prompt_cache_budget > 0
        else None
    )
    if prompt_cache_lru is None:
        logger.info("Prompt cache disabled for %s to protect RAM headroom.", os.path.basename(model_path))
    else:
        logger.info(
            "Prompt cache budget for %s: %d entries, per-entry token cap %d, "
            "total token budget %d (~%.1fGB total envelope at %dKB/token + "
            "%.1fMB fixed recurrent state/entry).",
            os.path.basename(model_path),
            prompt_cache_budget,
            prompt_cache_token_cap,
            prompt_cache_total_tokens,
            prompt_cache_total_bytes / (1024 ** 3),
            prompt_cache_kv_bytes // 1024,
            prompt_cache_fixed_bytes / (1024 ** 2),
        )

    # Expert-adapter residency: at most one domain adapter attached on top of
    # the loaded model (personality LoRA included); tracked so detach restores
    # exactly what this worker wrapped.
    expert_adapter_state: dict[str, Any] = {"path": "", "wrapped": []}

    worker_active = True
    while worker_active:
        try:
            if ipc_writer.broken.is_set():
                # The parent cannot hear ANY result. Consuming further jobs
                # would burn a full GPU lane into a void while the parent
                # times out against a silent child — exit visibly instead.
                logger.critical(
                    "🛑 [WORKER] Response pipe is broken; exiting so the parent "
                    "can detect death and respawn instead of timing out."
                )
                break
            try:
                # Bounded wait (not an infinite block) so the broken-pipe
                # check above runs even when the request queue is idle.
                job = request_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                logger.info("🛑 [WORKER] Shutdown signal received while idle; exiting quietly.")
                break
            except (EOFError, BrokenPipeError, OSError) as queue_exc:
                logger.info("🛑 [WORKER] Request queue closed; exiting quietly (%s).", queue_exc)
                break
            if job is None:
                worker_active = False
                continue
            if not isinstance(job, dict):
                # A non-mapping frame has no id to correlate a receipt to;
                # reject it visibly instead of raising inside the loop.
                _record_mlx_degradation(
                    TypeError(f"non_mapping_ipc_job:{type(job).__name__}"),
                    action="rejected malformed IPC job envelope",
                    severity="warning",
                )
                logger.error(
                    "🛑 [WORKER] Rejected malformed IPC job envelope of type %s.",
                    type(job).__name__,
                )
                continue

            action = job.get("action")
            if action == "generate":
                # Bound before any path that can build the response, so a
                # generation that never reached the processor stage still
                # reports why rather than raising NameError on the receipt.
                _endo_receipt: dict[str, Any] = {
                    "pathway": "endogenous_vocab_bias",
                    "reason": "not_evaluated",
                }
                prompt = job.get("prompt")
                messages = job.get("messages")
                tools = job.get("tools")
                original_prompt = prompt
                if job.get("user_surface_continuation_contract", False):
                    from core.brain.llm.chat_format import (
                        normalize_chat_continuation_messages,
                    )
                    from core.conversation.continuation import continuation_prompt_prefix

                    messages = normalize_chat_continuation_messages(
                        messages,
                        continuation_prompt_prefix(
                            job.get("user_surface_continuation_partial")
                        ),
                    )
                original_messages = messages
                strict_answer_contract = bool(job.get("strict_answer_contract", False))
                strict_value_contract = bool(job.get("strict_value_contract", False))
                expected_strict_value = (
                    _clean_expected_strict_value(str(job.get("expected_strict_value") or ""))
                    or _extract_expected_strict_value(original_messages, original_prompt)
                    if strict_value_contract
                    else ""
                )
                proof_evaluation_contract = bool(job.get("proof_evaluation_contract", False))
                operator_evidence_contract = bool(job.get("operator_evidence_contract", False))

                # These four contracts select MUTUALLY EXCLUSIVE prompt
                # builders, sampling regimes, validators and output
                # normalizers. Nothing rejected a job that asserted several at
                # once: the if/elif ladder below simply took whichever branch
                # came first, so a contradictory contract silently resolved by
                # source order and the caller received output shaped by a
                # contract it had not selected. An ambiguous contract is a
                # caller defect and is refused with its own correlated error.
                # CONTRACT AUTHORITY. Privileged contracts each select a
                # different prompt builder, sampling regime, validator and
                # output normalizer. Two things must hold before any of that
                # takes effect: the selection must be internally consistent
                # (asserting several exclusive contracts would otherwise be
                # resolved by source order, handing the caller output shaped
                # by a contract it never chose), and it must carry the
                # authority of the lane that owns this worker rather than
                # being a bare boolean anyone could set.
                _contract_refusal = _verify_contract_authority(job, contract_key)
                if _contract_refusal:
                    _record_mlx_degradation(
                        ValueError(_contract_refusal),
                        action="refused generation with an unsound privileged contract selection",
                        severity="error",
                    )
                    logger.error(
                        "🛑 [WORKER] Job %s refused: %s",
                        job.get("id"),
                        _contract_refusal,
                    )
                    ipc_writer.put(
                        {
                            "id": job.get("id"),
                            "action": "generate",
                            "status": "error",
                            "message": _contract_refusal,
                        }
                    )
                    continue
                # disable_prompt_cache = bool(job.get("disable_prompt_cache", False)) or strict_answer_contract
                prompt_cache_bypass = _job_requires_prompt_cache_bypass(job)
                disable_prompt_cache = bool(job.get("disable_prompt_cache", False)) or prompt_cache_bypass
                # Bypass no longer implies CLEAR: health probes fire between
                # user turns, and clearing on every probe would evict the
                # conversation's cached prefix before the next turn could
                # reuse it — silently reinstating the full-history re-prefill
                # this cache exists to prevent. Bypass jobs simply never read
                # or write; only an explicit request clears.
                clear_prompt_cache = bool(job.get("clear_prompt_cache", False))
                if clear_prompt_cache and prompt_cache_lru is not None:
                    prompt_cache_lru.clear()

                strict_envelope_prefixed = False
                operator_response_prefix = ""
                native_thinking: bool | None = None
                # Contract-truth flags surfaced in the response payload: an
                # exhausted proof retry, a seeded strict-value replacement,
                # and the operator-evidence scaffolding/model composition
                # must be visible to the caller, not silently absorbed.
                proof_contract_incomplete = False
                strict_value_normalized_from_draft = ""
                operator_evidence_receipt: dict[str, Any] = {}
                if not (
                    strict_answer_contract
                    or strict_value_contract
                    or proof_evaluation_contract
                    or operator_evidence_contract
                    or job.get("schema")
                ):
                    messages, prompt = _with_initial_user_surface_guidance(messages, prompt, job)
                # [FRONTIER UPGRADE] Native Tool Templates
                if strict_answer_contract:
                    prompt = _build_strict_answer_prompt(messages, prompt)
                    strict_envelope_prefixed = True
                elif strict_value_contract:
                    if expected_strict_value:
                        logger.info("🎯 [WORKER] Rendering exact strict-value prompt.")
                        prompt = _build_exact_strict_value_prompt(expected_strict_value)
                    else:
                        prompt = _build_strict_answer_retry_prompt(messages, prompt)
                    messages = None
                    strict_envelope_prefixed = True
                elif proof_evaluation_contract:
                    prompt = _build_proof_evaluation_prompt(messages, prompt)
                elif operator_evidence_contract:
                    prompt, operator_response_prefix = _build_operator_evidence_prompt(
                        messages,
                        prompt,
                    )
                elif messages and hasattr(tokenizer, "apply_chat_template"):
                    try:
                        logger.info("🎯 [WORKER] Rendering native chat/tool template.")
                        from core.brain.llm.chat_format import (
                            render_chat_continuation_template,
                            render_chat_template,
                            thinking_enabled_for_generation,
                        )

                        native_thinking = thinking_enabled_for_generation(
                            model_path,
                            cognitive_mode=job.get("cognitive_mode"),
                            final_user_surface=bool(
                                job.get("clean_user_surface_contract", False)
                            ),
                        )

                        if job.get("user_surface_continuation_contract", False):
                            prompt = render_chat_continuation_template(
                                tokenizer,
                                messages,
                                tools=tools,
                                enable_thinking=native_thinking,
                            )
                        else:
                            prompt = render_chat_template(
                                tokenizer,
                                messages,
                                tools=tools,
                                add_generation_prompt=True,
                                enable_thinking=native_thinking,
                            )
                            if tools:
                                # A tool prompt that ends without an open
                                # assistant turn makes the model emit
                                # <|im_end|> as its first token, which reads
                                # downstream as "produced nothing". The tail is
                                # the only thing that tells those apart.
                                #
                                # The per-message sizes are here because the
                                # client sent 150 characters and 5 tools and
                                # the rendered prompt measured 5,144 tokens:
                                # something between the two was adding the
                                # difference, and nothing recorded what.
                                logger.info(
                                    "🎯 [WORKER] Tool prompt: %d chars from %s | tail=%r",
                                    len(prompt) if isinstance(prompt, str) else -1,
                                    [
                                        (
                                            str(m.get("role"))[:9],
                                            len(str(m.get("content") or "")),
                                        )
                                        for m in (messages or [])
                                        if isinstance(m, dict)
                                    ],
                                    prompt[-90:] if isinstance(prompt, str) else type(prompt),
                                )
                    except Exception as e:  # noqa: BLE001 - one bad job must not kill the resident model
                        if tools:
                            # A tool-calling contract cannot degrade to prose:
                            # the model would answer without the tool schema
                            # and the caller would parse hallucinated calls.
                            _record_mlx_degradation(
                                e,
                                action="failed generation because tool template could not render",
                                severity="error",
                            )
                            ipc_writer.put(
                                {
                                    "id": job.get("id"),
                                    "action": "generate",
                                    "status": "error",
                                    "message": (
                                        "chat_template_failed_with_tools:"
                                        f"{type(e).__name__}:{e}"
                                    ),
                                }
                            )
                            continue
                        _record_mlx_degradation(
                            e,
                            action="continued generation with role-labeled fallback after native chat template failed",
                            severity="degraded",
                        )
                        logger.warning("❌ [WORKER] Native template compilation failed: %s", e)
                        # NEVER the stale prompt variable: render the actual
                        # transcript deterministically so the model answers
                        # the admitted task.
                        prompt = _render_messages_fallback(messages, prompt)

                temp = _admit_sampling_control(job, "temp")
                top_p = _admit_sampling_control(job, "top_p")
                min_p = _admit_sampling_control(job, "min_p")
                repetition_penalty = _admit_sampling_control(job, "repetition_penalty")
                artifact_generation_contract = bool(
                    proof_evaluation_contract
                    and _proof_prompt_expects_artifact(prompt)
                )

                if strict_answer_contract:
                    temp = 0.0
                    top_p = 1.0
                    min_p = 0.0
                    repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.12)
                elif strict_value_contract:
                    temp = 0.0
                    top_p = 1.0
                    min_p = 0.0
                    repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.05)
                elif proof_evaluation_contract:
                    if artifact_generation_contract:
                        temp = 0.0
                        top_p = 1.0
                        min_p = 0.0
                        repetition_penalty = max(_safe_float(repetition_penalty, 1.08), 1.05)
                    else:
                        temp = min(_safe_float(temp, 0.1), 0.15)
                        top_p = min(_safe_float(top_p, 0.9), 0.9)
                        min_p = min(_safe_float(min_p, 0.05), 0.05)
                        repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.08)
                elif operator_evidence_contract:
                    temp = min(_safe_float(temp, 0.1), 0.12)
                    top_p = min(_safe_float(top_p, 0.8), 0.8)
                    min_p = max(_safe_float(min_p, 0.03), 0.03)
                    repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.18)
                max_tokens = _admit_max_tokens(job.get("max_tokens", 512), 512)
                max_tokens = _serving_lane_output_cap(
                    model_path,
                    str(job.get("serving_lane") or "foreground_standard"),
                    max_tokens,
                )
                if operator_evidence_contract:
                    max_tokens = min(max_tokens, 192)
                hard_output_token_ceiling = _safe_int(
                    job.get("hard_output_token_ceiling"),
                    0,
                )
                _record_exact_reply_token_evidence(
                    job,
                    tokenizer,
                    generation_max_tokens=max_tokens,
                    hard_output_token_ceiling=hard_output_token_ceiling,
                )
                if hard_output_token_ceiling > 0:
                    max_tokens = min(max_tokens, hard_output_token_ceiling)
                schema = job.get("schema")

                # [v11.0 HARDENING] Structured Generation Overrides
                if schema:
                    temp = 0.0 # Force determinism for JSON
                    logger.info("🎯 [WORKER] Structured mode: temp=0.0 enforced.")

                # Intelligence boosters: min_p sampling improves quality on smaller
                # models by filtering out low-probability tokens before top_p.
                # Repetition penalty reduces the stale/looping response pattern.
                # Bumped from 1.1 → 1.2 (2026-04-27): live test showed mode
                # collapse on specific introspective prompts even after α was
                # halved. 1.2 is still well below the 1.5+ range that hurts
                # natural prose; targets the token-level "something is shifting
                # / something is moving" loops directly.
                kwargs = {"max_tokens": max_tokens, "temperature": temp, "top_p": top_p, "repetition_penalty": repetition_penalty}
                if make_sampler:
                    sampler_kwargs = {"temp": temp, "top_p": top_p}
                    try:
                        import inspect as _insp
                        _sparams = _insp.signature(make_sampler).parameters
                        if "min_p" in _sparams:
                            sampler_kwargs["min_p"] = min_p
                        if "repetition_penalty" in _sparams:
                            sampler_kwargs["repetition_penalty"] = repetition_penalty
                    except (TypeError, ValueError):
                        logger.debug("make_sampler signature introspection unavailable")
                    kwargs["sampler"] = make_sampler(**sampler_kwargs)

                # [v11.0 HARDENING] Logits Processors (JSON Enforcement)
                # [v11.0 HARDENING] Logits Processors (JSON Enforcement & Penalties)
                logits_processors = []

                # Apply MLX penalties via logits processors
                try:
                    from mlx_lm.sample_utils import make_logits_processors
                    _rp = job.get("repetition_penalty", repetition_penalty)
                    _rcs = job.get("repetition_context_size", 64)
                    _pp = job.get("presence_penalty", 0.0)
                    if _rp and _rp > 1.0:
                        lp = make_logits_processors(
                            repetition_penalty=_rp,
                            repetition_context_size=_rcs,
                            presence_penalty=_pp,
                        )
                        if lp:
                            logits_processors.extend(lp)
                except ImportError as _exc:
                    logger.debug("Suppressed %s in core.brain.llm.mlx_worker: %s", type(_exc).__name__, _exc)
                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.warning("Could not apply penalty logits processors: %s", e)

                # Tier-1 forward-pass reasoning levers (opt-in, fail-open):
                #  • AURA_REASONING_STEERING — plausibility-gated logit bias that
                #    suppresses low-information mode-collapse filler.
                #  • AURA_CONTRASTIVE_DECODING + AURA_CONTRASTIVE_AMATEUR_MODEL —
                #    real dual-model contrastive decoding against a small same-family
                #    amateur (e.g. Qwen2.5-1.5B vs the 32B cortex), subtracting the
                #    amateur's lazy preferences within the cortex's plausible set.
                _steer_on = _FLAG_REASONING_STEERING.value().strip().lower() in {"1", "true", "on", "yes"}
                _cd_on = os.environ.get("AURA_CONTRASTIVE_DECODING", "").strip().lower() in {"1", "true", "on", "yes"}
                _amateur_path = os.environ.get("AURA_CONTRASTIVE_AMATEUR_MODEL", "").strip()
                if _steer_on or (_cd_on and _amateur_path):
                    try:
                        from core.brain.llm.contrastive_decoding import (
                            build_reasoning_logits_processors,
                        )

                        reasoning_procs = build_reasoning_logits_processors(
                            tokenizer,
                            enable_steering=_steer_on,
                            amateur_model_path=_amateur_path if (_cd_on and _amateur_path) else None,
                            alpha=_safe_float(_FLAG_CONTRASTIVE_ALPHA.value(), 0.5),
                            beta=_safe_float(_FLAG_CONTRASTIVE_BETA.value(), 0.1),
                            steering_scale=_safe_float(_FLAG_REASONING_STEERING_SCALE.value(), 1.0),
                        )
                        if reasoning_procs:
                            logits_processors.extend(reasoning_procs)
                            logger.info("🧠 [WORKER] Reasoning processors ACTIVE (%d: steer=%s cd=%s).",
                                        len(reasoning_procs), _steer_on, bool(_cd_on and _amateur_path))
                    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                        logger.warning("Could not apply reasoning logits processors: %s", e)

                if schema:
                    try:
                        start_ids = _json_start_token_ids(tokenizer, schema)
                        if not start_ids:
                            raise ValueError(
                                "tokenizer spells no admissible JSON opening"
                            )

                        def json_start_processor(tokens, logits, start_ids=start_ids):
                            if len(tokens) == 0:
                                # Every way this tokenizer can open the value
                                # the schema declares — not the one token that
                                # `encode("{")[0]` happened to return.
                                mask = mx.full_like(logits, -float("inf"))
                                for token_id in start_ids:
                                    mask[..., token_id] = 0.0
                                return mask
                            return logits
                        logits_processors.append(json_start_processor)
                        logger.info(
                            "🎯 [WORKER] JSON start enforcement ACTIVE (%d openings).",
                            len(start_ids),
                        )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued structured generation without JSON start logits processor",
                            severity="degraded",
                        )
                        logger.warning("Failed to setup JSON logits processor: %s", e)

                if strict_answer_contract or strict_value_contract:
                    try:
                        strict_guard = build_nonempty_start_processor(tokenizer, positions=3)
                        if strict_guard is not None:
                            logits_processors.append(strict_guard)
                            logger.info(
                                "🎯 [WORKER] Strict contract non-empty start guard ACTIVE."
                            )
                    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued strict generation without non-empty start logits guard",
                            severity="warning",
                        )
                        logger.warning("Failed to setup strict non-empty start guard: %s", e)
                elif not _expected_empty_warmup_precompile(job):
                    # An assistant turn that ends before it says anything is not
                    # a completion, on any lane.
                    try:
                        guard = build_nonempty_start_processor(tokenizer)
                        if guard is not None:
                            logits_processors.append(guard)
                        logger.info(
                            "🎯 [WORKER] Non-empty start guard %s (generate path).",
                            "ACTIVE" if guard is not None else "UNAVAILABLE",
                        )
                    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued generation without the non-empty start guard",
                            severity="warning",
                        )

                try:
                    semantic_terminal_guard = build_semantic_completion_terminal_guard(
                        tokenizer,
                        job,
                    )
                    if semantic_terminal_guard is not None:
                        logits_processors.append(semantic_terminal_guard)
                    logger.info(
                        "🧩 [WORKER] Semantic terminal guard %s (generate path).",
                        "ACTIVE" if semantic_terminal_guard is not None else "INACTIVE",
                    )
                except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
                    _record_mlx_degradation(
                        e,
                        action="continued generation without semantic terminal guard",
                        severity="warning",
                    )

                # The endogenous language pathway: L_final = L_LLM + alpha*(W*z + b).
                # The state arrived on the job because this process cannot reach
                # the organs that produced it. The bias is computed once and
                # added inside the model's own plausible set, so a half-trained
                # head re-ranks near-ties and cannot promote a ruled-out token.
                from core.brain.llm.endogenous_decode import (
                    install_endogenous_processor,
                )

                _endo_receipt, _endo_fault = install_endogenous_processor(
                    tokenizer, job, logits_processors
                )
                if _endo_fault:
                    _record_mlx_degradation(
                        ValueError(f"endogenous head present but unusable: {_endo_fault}"),
                        action="generated without the endogenous vocabulary bias",
                        severity="warning",
                    )

                # Foreground non-parametric memory (KV-cache-correct): the tap captures the
                # hidden state the generation forward already computes, so the processor adds
                # recall at O(1)/token — no O(n²) recompute. Off by default, fail-open, and
                # only installed when the datastore is non-empty.
                _np_tap = None
                try:
                    from core.brain.nonparametric_worker import maybe_build_foreground
                    _np_foreground = maybe_build_foreground(model, job=job)
                    if _np_foreground is not None:
                        _np_tap, _np_proc = _np_foreground
                        logits_processors.append(_np_proc)
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                    logger.debug("Foreground non-parametric memory unavailable: %s", e)

                if logits_processors:
                    kwargs["logits_processors"] = logits_processors

                stop_sequences = _merge_stop_sequences(job.get("stop_sequences") or [])
                # We do NOT pass stop_words to stream_generate as it causes TypeError in some mlx-lm versions.
                # Truncation is handled manually in the token loop via _truncate_role_continuation.


                try:
                    from mlx_lm.generate import stream_generate

                    from core.brain.llm.chat_format import (
                        split_native_thinking_generation,
                    )
                    # : NO GPUSentinel here.
                    # GPUSentinel is a parent-process threading lock. In this isolated
                    # 'spawn' subprocess, it creates a SECOND serialization bottleneck
                    # on top of metal_semaphore, causing 30s GPU_TIMEOUT hangs.
                    # metal_semaphore(1) already serializes all GPU access in this worker.

                    response_text = ""
                    total_generated_tokens = 0
                    interoception_payload = None
                    surface_control_state = _apply_surface_generation_controls(engine, model, job)
                    _enforce_surface_controls_or_fail(job, surface_control_state)
                    surface_quality_gate_enabled = _surface_quality_gate_enabled(job)
                    surface_control_state["surface_quality_gate_enabled"] = surface_quality_gate_enabled
                    surface_control_state["surface_quality_gate_passed"] = not surface_quality_gate_enabled
                    surface_control_state["surface_quality_gate_attempts"] = 0
                    surface_control_state["surface_quality_gate_reasons"] = []
                    surface_control_state["telemetry_sanitizer_reasons"] = []
                    surface_control_state["instruction_shape_repair_applied"] = False
                    surface_control_state["text_mutations"] = []
                    surface_control_state["generation_max_tokens_applied"] = max_tokens
                    try:
                        with metal_semaphore:
                            # Proactive cache clearing under memory pressure
                            if mx and device != "cpu":
                                try:
                                    from core.runtime import resource_psutil as psutil
                                    if psutil.virtual_memory().percent > 90:  # 64GB — don't panic at 85%
                                        logger.warning("⚠️ High memory pressure detected in worker. Clearing MLX cache.")
                                        mx.clear_cache()
                                except (ImportError, OSError, AttributeError):
                                    logger.debug("Worker memory pressure probe unavailable")

                            # [v11.5 HARDENING] Internal Worker Retries for Structured Leaks & Loops
                            # We allow up to 2 retries if the LLM gets stuck in a loop or returns empty on a schema.
                            max_internal_retries = 1 if proof_evaluation_contract else 2

                            # This clock belongs to REPAIR, not drafting. A healthy
                            # resident-32B first pass takes 30-70s under load; starting
                            # a 20s repair wall before that pass made the wall expire
                            # before a rejected draft even existed, so an authored
                            # correction could never run.
                            surface_retry_started = 0.0
                            surface_retry_wall_s = _safe_float(
                                _FLAG_SURFACE_RETRY_WALL_S.value(), 20.0
                            )
                            ontology_retry_count = 0
                            schema_validation_failed = ""

                            for internal_attempt in range(max_internal_retries + 1):
                                surface_control_state["generation_max_tokens_applied"] = max(
                                    1,
                                    _safe_int(kwargs.get("max_tokens"), max_tokens),
                                )
                                watchdog.start_job(str(job.get("id") or ""), "generate")
                                try:
                                    surface_control_state[
                                        "instruction_shape_repair_applied"
                                    ] = False
                                    surface_control_state["text_mutations"] = []
                                    current_response = ""
                                    token_count = 0
                                    prompt_token_count = 0
                                    final_prompt_cache = None
                                    previous_cache_rollback = None
                                    continuation_cache_rollback = None
                                    deadline_hit = False
                                    semantic_terminal_grace_active = False
                                    semantic_terminal_grace_deadline_unix = 0.0
                                    semantic_terminal_grace_token_ceiling = 0
                                    # Caller production deadline (absolute unix
                                    # seconds). Without it the worker had no
                                    # request deadline at all and could decode
                                    # long past the caller's timeout until the
                                    # 360s watchdog hard-killed the process.
                                    job_deadline_unix = _safe_float(
                                        job.get("deadline_unix"), 0.0
                                    )
                                    if (
                                        job_deadline_unix > 0.0
                                        and time.time() >= job_deadline_unix
                                    ):
                                        raise RuntimeError(
                                            "deadline_exceeded_before_decode:"
                                            f"deadline_unix={job_deadline_unix:.3f}"
                                        )
                                    last_progress_emit_at = time.time()
                                    sentinel_aborted = False
                                    sentinel_loop_aborted = False
                                    sentinel_ontology_aborted = False
                                    role_continuation_hit = False
                                    configured_stop_hit = False
                                    configured_stop_sequence = ""
                                    hard_token_limit_hit = False
                                    semantic_contract_satisfied = False
                                    job_seq = _safe_int(job.get("seq"), 0)
                                    soft_cancelled = False
                                    clear_stale_soft_cancel(cancel_seq, job_seq)

                                    # ── Token Sentinel: mid-generation cognitive intervention ──
                                    # Creates a lightweight monitor that checks for capitulation,
                                    # persona drift, and live-updates affect state during generation.
                                    try:
                                        from core.brain.llm.token_sentinel import (
                                            InterventionType,
                                            TokenSentinel,
                                            get_refusal_fallback,
                                        )
                                        sentinel = TokenSentinel(
                                            check_interval=8,
                                            affect_interval=16,
                                            substrate_mem=substrate_mem,
                                            steering_hooks=_active_steering_hooks(engine),
                                            boundary_context=str(
                                                job.get("boundary_context") or ""
                                            ),
                                            prompt=(
                                                _surface_validation_prompt(job)
                                                or original_prompt
                                            ),
                                            generation_purpose=str(
                                                job.get("request_purpose")
                                                or job.get("action")
                                                or "generate"
                                            ),
                                            user_surface=bool(
                                                job.get(
                                                    "clean_user_surface_contract",
                                                    False,
                                                )
                                            ),
                                            affect_expected=(
                                                _affect_expected
                                                and _safe_float(
                                                    job.get(
                                                        "clean_user_surface_steering_alpha"
                                                    ),
                                                    1.0,
                                                )
                                                > 0.0
                                            ),
                                        )
                                    except (ImportError, AttributeError, RuntimeError) as _sent_exc:
                                        if bool(job.get("clean_user_surface_contract", False)):
                                            # User-visible prose without the
                                            # capitulation/persona/ontology guard is
                                            # fail-OPEN safety; refuse the job with a
                                            # correlated error instead.
                                            _record_mlx_degradation(
                                                _sent_exc,
                                                action="refused user-surface generation without TokenSentinel",
                                                severity="critical",
                                            )
                                            raise RuntimeError(
                                                f"token_sentinel_unavailable:{_sent_exc}"
                                            ) from _sent_exc
                                        _record_mlx_degradation(
                                            _sent_exc,
                                            action="continued non-surface generation without TokenSentinel intervention checks",
                                            severity="degraded",
                                        )
                                        sentinel = None
                                        logger.debug("TokenSentinel not available: %s", _sent_exc)

                                    # ── Interoception tap: substrate self-measurement ──
                                    # Pure observer of the decode distribution (surprisal,
                                    # entropy, top-2 gap per sampled token). It cannot alter
                                    # sampling and cannot raise into the loop. Built fresh per
                                    # attempt so the final payload always describes the
                                    # response actually returned.
                                    try:
                                        from core.brain.llm.interoception_tap import maybe_build_tap
                                        # Bind the measurement to the request and
                                        # model that produced it; without this the
                                        # parent has only an attempt number and can
                                        # misattribute across attempts or models.
                                        intero_tap = maybe_build_tap(
                                            request_id=str(
                                                job.get("request_id")
                                                or job.get("trace_id")
                                                or ""
                                            ),
                                            model_id=str(job.get("model") or ""),
                                            provider="mlx",
                                        )
                                    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _intero_exc:
                                        _record_mlx_degradation(
                                            _intero_exc,
                                            action="continued generation without interoception tap",
                                            severity="warning",
                                        )
                                        intero_tap = None

                                    # [FRONTIER UPGRADE] KV Prompt Caching Injection
                                    tokens = tokenizer.encode(prompt)
                                    # These live in mlx_lm.models.cache, not
                                    # mlx_lm.utils. Probing the wrong module
                                    # made _can_trim answer False forever, so
                                    # the trimmed-reuse path could never run.
                                    from mlx_lm.models.cache import (
                                        can_trim_prompt_cache as _mlx_can_trim,
                                    )
                                    from mlx_lm.models.cache import (
                                        make_prompt_cache as _mlx_make_cache,
                                    )
                                    from mlx_lm.models.cache import (
                                        trim_prompt_cache as _mlx_trim,
                                    )

                                    def _can_trim(pc):
                                        try:
                                            return bool(_mlx_can_trim(pc))
                                        except (AttributeError, TypeError, ValueError):
                                            return False

                                    def _do_trim(pc, num):
                                        _mlx_trim(pc, num)

                                    # Scope-partitioned key: user-surface
                                    # turns only ever see user-surface
                                    # entries, so internal lanes cannot leak
                                    # KV into the conversation or vice versa.
                                    model_key = (
                                        id(model),
                                        _prompt_cache_scope_for_job(job),
                                    )
                                    cache = None
                                    remaining_tokens = tokens
                                    resume_handle = str(
                                        job.get("user_surface_continuation_resume_handle")
                                        or ""
                                    ).strip().lower()
                                    surface_control_state[
                                        "continuation_resume_requested"
                                    ] = bool(resume_handle)
                                    resume_applied = False
                                    if (
                                        resume_handle
                                        and prompt_cache_lru is not None
                                        and not disable_prompt_cache
                                    ):
                                        (
                                            resumed_cache,
                                            resumed_remaining,
                                            resumed_tokens,
                                            resume_failure,
                                        ) = prompt_cache_lru.fetch_resume(
                                            resume_handle,
                                            model_key,
                                            can_trim_prompt_cache=_can_trim,
                                            trim_prompt_cache=_do_trim,
                                        )
                                        if resumed_cache is not None:
                                            cache = resumed_cache
                                            remaining_tokens = resumed_remaining
                                            tokens = resumed_tokens
                                            resume_applied = True
                                        else:
                                            surface_control_state[
                                                "continuation_resume_failure_reason"
                                            ] = resume_failure
                                            logger.warning(
                                                "Continuation resume unavailable (%s); "
                                                "using bounded textual reconstruction.",
                                                resume_failure,
                                            )
                                    surface_control_state[
                                        "continuation_resume_applied"
                                    ] = resume_applied
                                    # Prefill cost is the whole endurance story, and
                                    # until now the only number anyone could see was
                                    # the planner's CHARACTER count — measured live at
                                    # 4,479 chars for a turn the worker then tokenized
                                    # to 27,374 tokens. A 6x gap between what the
                                    # planner budgets and what the GPU actually
                                    # prefills is invisible without printing both.
                                    if len(tokens) > 4096:
                                        logger.info(
                                            "📏 [WORKER] Prefill size: %d tokens from %d rendered "
                                            "chars (%d messages, %d tool schemas, %d schema chars) "
                                            "| per-message: %s",
                                            len(tokens),
                                            len(prompt or ""),
                                            len(messages or ()),
                                            len(tools or ()),
                                            len(json.dumps(tools, default=str)) if tools else 0,
                                            ", ".join(
                                                f"{str(m.get('role', '?'))}="
                                                f"{len(str(m.get('content', '') or ''))}"
                                                for m in (messages or ())
                                                if isinstance(m, dict)
                                            ) or "none",
                                        )
                                    # Context-window admission BEFORE any Metal
                                    # work: reject with a typed, correlated error
                                    # instead of overrunning the model. Headroom
                                    # for at least a minimal answer is reserved.
                                    _output_reserve = min(
                                        max(64, _safe_int(kwargs.get("max_tokens"), 512)),
                                        2048,
                                    )
                                    _request_context_window = _serving_lane_context_window(
                                        model_path,
                                        str(job.get("serving_lane") or "foreground_standard"),
                                        output_reserve=_output_reserve,
                                        architectural_window=effective_context_window,
                                    )
                                    if (
                                        not resume_applied
                                        and len(tokens) + _output_reserve
                                        > _request_context_window
                                    ):
                                        # Refusing was the ONLY response here, and
                                        # nothing upstream bounds a prompt against
                                        # the model's real window — so an
                                        # overshooting lane failed on every attempt
                                        # forever. Shed scaffold first; refuse only
                                        # if the request itself will not fit.
                                        _oversized_tokens = len(tokens)
                                        prompt, tokens, _trim_note = _shrink_scaffold_to_context_window(
                                            messages=messages,
                                            prompt=prompt,
                                            tokens=tokens,
                                            window=_request_context_window,
                                            output_reserve=_output_reserve,
                                            tokenizer=tokenizer,
                                            tools=tools,
                                        )
                                        if _trim_note:
                                            _record_mlx_degradation(
                                                RuntimeError(
                                                    "scaffold_exceeded_context_window:"
                                                    f"prompt_tokens={_oversized_tokens}:"
                                                    f"window={_request_context_window}:"
                                                    f"trimmed={_trim_note}"
                                                ),
                                                action=(
                                                    "trimmed oversized system scaffold to fit the "
                                                    "context window instead of failing the turn"
                                                ),
                                                severity="warning",
                                            )
                                            logger.warning(
                                                "✂️ [WORKER] Scaffold exceeded the context window "
                                                "(%d tokens > %d - %d reserve); trimmed to %d tokens "
                                                "[%s]. The prompt builder should have bounded this.",
                                                _oversized_tokens,
                                                _request_context_window,
                                                _output_reserve,
                                                len(tokens),
                                                _trim_note,
                                            )
                                    if len(tokens) + _output_reserve > _request_context_window:
                                        raise RuntimeError(
                                            "context_window_exceeded:"
                                            f"prompt_tokens={len(tokens)}:"
                                            f"output_reserve={_output_reserve}:"
                                            f"window={_request_context_window}"
                                        )
                                    prompt_token_count = len(tokens)
                                    if (
                                        not resume_applied
                                        and prompt_cache_lru is not None
                                        and not disable_prompt_cache
                                    ):
                                        remaining_tokens = tokens
                                        cache, remaining_tokens = prompt_cache_lru.fetch_nearest_cache(
                                            model_key, tokens,
                                            can_trim_prompt_cache=_can_trim,
                                            trim_prompt_cache=_do_trim
                                        )
                                        if cache is None:
                                            # A miss still has to leave something
                                            # BEHIND. mlx_lm only fills a cache
                                            # object the caller supplies, and this
                                            # lane supplied none — so nothing was
                                            # ever cached, every turn re-prefilled
                                            # the whole history from token 0, and
                                            # the measured hit rate was 0/92.
                                            cache = _mlx_make_cache(model)
                                            remaining_tokens = tokens

                                    # Reuse only pays if the prompts share a LONG
                                    # prefix. Measured live, hits reused 13-125
                                    # tokens of 790-2211 — real hits worth almost
                                    # nothing, because something volatile sits near
                                    # the front of the prompt. The hit/miss line
                                    # alone cannot say what; naming the first
                                    # divergent tokens can.
                                    if (
                                        cache is not None
                                        and len(tokens) > 512
                                        and 0 < len(tokens) - len(remaining_tokens)
                                        < len(tokens) * 0.5
                                    ):
                                        _reused = len(tokens) - len(remaining_tokens)
                                        try:
                                            _divergent = tokenizer.decode(
                                                tokens[_reused:_reused + 24]
                                            )
                                        except (AttributeError, RuntimeError, TypeError, ValueError):
                                            _divergent = "<undecodable>"
                                        try:
                                            # The REUSED head is the other half of
                                            # the story: knowing reuse stopped at
                                            # token 13 is useless without seeing
                                            # what those 13 tokens were, because
                                            # that names the block whose volatility
                                            # is capping every conversation.
                                            _stable_head = tokenizer.decode(tokens[:_reused])
                                        except (AttributeError, RuntimeError, TypeError, ValueError):
                                            _stable_head = "<undecodable>"
                                        logger.info(
                                            "🔍 [PROMPT CACHE] prefix diverges at token %d "
                                            "(%.0f%% of %d reused) scope=%s; stable head: %r; "
                                            "divergent text begins: %r",
                                            _reused,
                                            100.0 * _reused / max(1, len(tokens)),
                                            len(tokens),
                                            _prompt_cache_scope_for_job(job),
                                            _stable_head[:200],
                                            _divergent[:160],
                                        )

                                    gen_prompt = remaining_tokens if cache is not None else prompt
                                    if cache is not None:
                                        kwargs["prompt_cache"] = cache
                                        # mlx_lm mutates this object in place as
                                        # it prefills and decodes, so the object
                                        # we hand in IS the post-generation cache.
                                        # It is never reported back on the
                                        # response, which is why reading it from
                                        # there captured nothing.
                                        final_prompt_cache = cache

                                    attempt_logits_processors = list(logits_processors)
                                    if bool(job.get("semantic_completion_contract", False)):
                                        if semantic_terminal_guard is not None:
                                            logger.info(
                                                "🧩 [WORKER] Semantic completion observer ACTIVE; "
                                                "append-only continuation terminal held until the "
                                                "assembled contract is satisfied."
                                            )
                                        else:
                                            logger.info(
                                                "🧩 [WORKER] Semantic completion observer ACTIVE; "
                                                "natural initial-branch termination remains available "
                                                "while the typed contract is measured."
                                            )
                                    if attempt_logits_processors:
                                        kwargs["logits_processors"] = attempt_logits_processors
                                    else:
                                        kwargs.pop("logits_processors", None)

                                    # [STABILITY v57] Reset activity immediately before loop to maximize budget for prefill
                                    try:
                                        from mlx_lm.sample_utils import make_sampler
                                        if "sampler" not in kwargs:
                                            import inspect as _insp
                                            _sparams = _insp.signature(make_sampler).parameters
                                            sampler_kwargs = {"temp": kwargs.get("temperature", 0.7)}
                                            if "top_p" in _sparams:
                                                sampler_kwargs["top_p"] = kwargs.get("top_p", 1.0)
                                            if "min_p" in _sparams:
                                                sampler_kwargs["min_p"] = kwargs.get("min_p", 0.0)
                                            if "repetition_penalty" in _sparams:
                                                sampler_kwargs["repetition_penalty"] = kwargs.get("repetition_penalty", 1.0)
                                            if "repetition_context_size" in _sparams:
                                                sampler_kwargs["repetition_context_size"] = kwargs.get("repetition_context_size", 20)
                                            kwargs["sampler"] = make_sampler(**sampler_kwargs)
                                    except ImportError:
                                        logger.debug("MLX make_sampler unavailable; using stream_generate defaults.")

                                    # [STABILITY v60] Definitive scrub of legacy kwargs.
                                    # New mlx-lm versions pass kwargs directly to generate_step which
                                    # throws TypeError if it sees 'temperature' or 'top_p' instead of 'temp'.
                                    clean_keys = {"temperature", "top_p", "min_p", "repetition_penalty", "repetition_context_size", "stop_words"}
                                    clean_kwargs = {k: v for k, v in kwargs.items() if k not in clean_keys}

                                    prefill_tokens = (
                                        len(gen_prompt)
                                        if not isinstance(gen_prompt, str)
                                        else prompt_token_count
                                    )
                                    prefill_step_size = _runtime_prefill_step_size(model_path)
                                    clean_kwargs["prefill_step_size"] = prefill_step_size
                                    clean_kwargs["prompt_progress_callback"] = (
                                        _build_prefill_progress_callback(
                                            watchdog,
                                            ipc_writer,
                                            request_id=str(job.get("id") or ""),
                                            action="generate",
                                        )
                                    )

                                    watchdog.activity()

                                    # If the foreground memory tap is installed, keep it active
                                    # for the whole generation and restore the model afterward.
                                    # The context exits on normal completion, break, or error
                                    # (GeneratorExit), so model.model is always restored.
                                    def _gen_stream(tap, prompt_text, generation_kwargs):
                                        if tap is not None:
                                            with tap:
                                                yield from stream_generate(
                                                    model,
                                                    tokenizer,
                                                    prompt=prompt_text,
                                                    **generation_kwargs,
                                                )
                                        else:
                                            yield from stream_generate(
                                                model,
                                                tokenizer,
                                                prompt=prompt_text,
                                                **generation_kwargs,
                                            )

                                    use_speculative = _speculative_eligible(
                                        draft_model,
                                        clean_kwargs,
                                        job,
                                        prefill_tokens=prefill_tokens,
                                        prefill_step_size=prefill_step_size,
                                    )
                                    if use_speculative:
                                        clean_kwargs["draft_model"] = draft_model
                                    draft_accepted_tokens = 0

                                    for response in _gen_stream(
                                        _np_tap,
                                        gen_prompt,
                                        clean_kwargs,
                                    ):
                                        watchdog.activity()

                                        # ``mlx_lm.generate_step`` computes the
                                        # next logits before yielding the current
                                        # token, so the cache already includes
                                        # this response here. The prior fixed-
                                        # state image is the one-token rollback
                                        # needed if this becomes the final token.
                                        if (
                                            bool(
                                                job.get(
                                                    "semantic_completion_contract",
                                                    False,
                                                )
                                            )
                                            and prompt_cache_lru is not None
                                            and not disable_prompt_cache
                                            and final_prompt_cache is not None
                                        ):
                                            continuation_cache_rollback = (
                                                previous_cache_rollback
                                            )
                                            previous_cache_rollback = (
                                                _capture_prompt_cache_one_token_rollback(
                                                    final_prompt_cache
                                                )
                                            )
                                        else:
                                            continuation_cache_rollback = None
                                            previous_cache_rollback = None

                                        token_count += 1
                                        progress_now = time.time()
                                        if use_speculative and getattr(response, "from_draft", False):
                                            draft_accepted_tokens += 1

                                        tokens.append(response.token)
                                        # `final_prompt_cache` is bound once, before
                                        # the loop, to the cache object handed to
                                        # mlx_lm; it is mutated in place as tokens
                                        # are produced, so `tokens` and that object
                                        # stay in step and insertion happens ONCE
                                        # after the loop. Per-token insertion used
                                        # to store the same mutable cache under
                                        # every growing prefix, so older trie keys
                                        # aliased later-prefix KV state.

                                        if intero_tap is not None:
                                            intero_tap.feed(
                                                response.token,
                                                getattr(response, "logprobs", None),
                                                response.text,
                                            )

                                        current_response += response.text
                                        current_response, role_continuation_hit = _truncate_role_continuation(current_response)

                                        # Cooperative preemption is observed only
                                        # after accepting the token already yielded
                                        # by mlx_lm. Its cache performs one-token
                                        # lookahead before yield; dropping this token
                                        # from the visible/tokens ledger left every
                                        # retained cache one step ahead of its key.
                                        if soft_cancel_requested(cancel_seq, job_seq):
                                            soft_cancelled = True
                                            try:
                                                cancel_seq.value = 0
                                            except (OSError, ValueError):
                                                logger.debug(
                                                    "Soft-cancel acknowledge write failed; continuing."
                                                )
                                            logger.info(
                                                "✋ [WORKER] Soft-cancel observed at token %d "
                                                "(job seq=%d).",
                                                token_count,
                                                job_seq,
                                            )
                                            break

                                        # [STABILITY v58] Explicit break on stop sequences or role drift
                                        if role_continuation_hit:
                                            break

                                        # Manual check for any dynamic stop sequences passed in the job
                                        if any(s in current_response for s in stop_sequences):
                                            for s in stop_sequences:
                                                if s in current_response:
                                                    current_response = current_response.split(s)[0]
                                                    configured_stop_hit = True
                                                    configured_stop_sequence = s
                                                    break
                                            break

                                        # ── Sentinel: feed every token ────────────────────
                                        if sentinel is not None:
                                            sentinel_signal = sentinel.feed(response.text)
                                            if sentinel_signal.type == InterventionType.ABORT_LOOP:
                                                logger.warning(
                                                    "🚨 [SENTINEL] Aborting loop at token %d: %s",
                                                    token_count, sentinel_signal.reason,
                                                )
                                                current_response = sentinel_signal.clean_prefix
                                                sentinel_aborted = True
                                                sentinel_loop_aborted = True
                                                break
                                            elif sentinel_signal.type == InterventionType.ABORT_ONTOLOGY_VIOLATION:
                                                logger.warning(
                                                    "🚨 [SENTINEL] Aborting due to ontological violation at token %d: %s",
                                                    token_count, sentinel_signal.reason,
                                                )
                                                sentinel_aborted = True
                                                sentinel_ontology_aborted = True
                                                break
                                            elif sentinel_signal.type in (InterventionType.ABORT_CAPITULATION,
                                                                          InterventionType.ABORT_BOUNDARY):
                                                # Mid-generation abort: the LLM started capitulating.
                                                # Replace response with deterministic refusal.
                                                logger.warning(
                                                    "🚨 [SENTINEL] Aborting generation at token %d: %s",
                                                    token_count, sentinel_signal.reason,
                                                )
                                                current_response = get_refusal_fallback(seed=token_count)
                                                sentinel_aborted = True
                                                break

                                        semantic_surface = split_native_thinking_generation(
                                            current_response,
                                            native_thinking=(native_thinking is True),
                                        ).surface

                                        if (
                                            token_count % 8 == 0
                                            and _semantic_surface_stop_ready(
                                                job,
                                                semantic_surface,
                                                generated_tokens=token_count,
                                            )
                                        ):
                                            semantic_contract_satisfied = True
                                            logger.info(
                                                "✅ [WORKER] Semantic completion contract satisfied at token %d.",
                                                token_count,
                                            )
                                            break

                                        if (
                                            job_deadline_unix > 0.0
                                            and progress_now >= job_deadline_unix
                                        ):
                                            if (
                                                not semantic_terminal_grace_active
                                                and _semantic_terminal_grace_eligible(
                                                    job,
                                                    semantic_surface,
                                                    generated_tokens=token_count,
                                                )
                                            ):
                                                # The parent keeps up to six seconds for
                                                # delivery. Borrow at most four of them
                                                # and 24 tokens; the first boundary that
                                                # satisfies the semantic observer stops
                                                # the decode above.
                                                semantic_terminal_grace_active = True
                                                semantic_terminal_grace_deadline_unix = (
                                                    progress_now + 4.0
                                                )
                                                semantic_terminal_grace_token_ceiling = (
                                                    token_count + 24
                                                )
                                                logger.info(
                                                    "🧩 [WORKER] Deadline reached with all "
                                                    "semantic obligations complete; granting "
                                                    "bounded terminal grace through token %d.",
                                                    semantic_terminal_grace_token_ceiling,
                                                )
                                            elif (
                                                semantic_terminal_grace_active
                                                and progress_now
                                                < semantic_terminal_grace_deadline_unix
                                                and token_count
                                                < semantic_terminal_grace_token_ceiling
                                            ):
                                                pass
                                            else:
                                                logger.warning(
                                                    "⏱️ [WORKER] Request deadline reached at "
                                                    "token %d; stopping decode cooperatively.",
                                                    token_count,
                                                )
                                                deadline_hit = True
                                                break

                                        if _should_emit_generation_progress(
                                            token_count,
                                            last_emit_at=last_progress_emit_at,
                                            now=progress_now,
                                        ):
                                            progress_msg = {
                                                "id": job.get("id"),
                                                "action": "generate",
                                                "status": "progress",
                                                "tokens_generated": token_count,
                                                "timestamp": progress_now,
                                            }
                                            if intero_tap is not None:
                                                live_intero = intero_tap.live_snapshot()
                                                if live_intero:
                                                    progress_msg["interoception_live"] = live_intero
                                            ipc_writer.put(progress_msg)
                                            last_progress_emit_at = progress_now

                                        # [PERF] Mid-generation cache clearing removed — was forcing Metal to
                                        # reallocate GPU memory every 32 tokens, creating micro-stalls.
                                        # Post-generation cleanup (line ~988) still ensures clean state.

                                        # Absolute safety cap: >= so the 8192nd token is
                                        # the LAST admitted one (the old > check exposed
                                        # an 8193rd token past the documented limit).
                                        if token_count >= 8192:
                                            logger.warning("🏁 [WORKER] Hard token limit (8192) reached. Truncating.")
                                            hard_token_limit_hit = True
                                            break

                                        stop_hit = role_continuation_hit
                                        for stop in stop_sequences:
                                            stop_index = current_response.find(stop)
                                            # index zero is a real stop (response BEGINS
                                            # with the stop sequence) — same fix the
                                            # stream path received.
                                            if stop_index >= 0:
                                                current_response = current_response[:stop_index]
                                                configured_stop_hit = True
                                                configured_stop_sequence = stop
                                                stop_hit = True
                                                break
                                        if stop_hit:
                                            break

                                    if sentinel is not None and not sentinel_aborted:
                                        terminal_signal = sentinel.finalize()
                                        if (
                                            terminal_signal.type
                                            == InterventionType.ABORT_ONTOLOGY_VIOLATION
                                        ):
                                            logger.warning(
                                                "🚨 [SENTINEL] Completed response failed "
                                                "ontology grounding: %s",
                                                terminal_signal.reason,
                                            )
                                            sentinel_aborted = True
                                            sentinel_ontology_aborted = True

                                    # Single post-generation cache insert: the final
                                    # cache object exactly matches the final token
                                    # list, and generation has stopped mutating it.
                                    if (
                                        prompt_cache_lru is not None
                                        and not disable_prompt_cache
                                        and final_prompt_cache is not None
                                        and tokens
                                        and not sentinel_aborted
                                    ):
                                        prompt_cache_lru.insert_cache(
                                            model_key, list(tokens), final_prompt_cache
                                        )

                                    # Interoception: distil this attempt's measurements.
                                    # Later attempts overwrite, so the shipped payload always
                                    # describes the response the caller actually receives.
                                    if intero_tap is not None:
                                        _intero_final = intero_tap.finalize(attempt=internal_attempt)
                                        if _intero_final is not None:
                                            # CP126 0b3bbd3e: stamp the trace
                                            # with the job it measured. Without
                                            # it the organ receives a payload
                                            # and a response as two unrelated
                                            # arguments and cannot tell whether
                                            # they belong together under
                                            # concurrent lanes.
                                            _intero_final["generation_id"] = str(
                                                job.get("request_id")
                                                or job.get("id")
                                                or ""
                                            )
                                            interoception_payload = _intero_final

                                    # Log sentinel diagnostics
                                    if sentinel is not None:
                                        diag = sentinel.get_diagnostics()
                                        if diag["interventions"] > 0 or diag["drift_warnings"] > 0:
                                            logger.info(
                                                "🛡️ [SENTINEL] Generation complete: %d interventions, "
                                                "%d drift warnings, %d affect pulses over %d tokens",
                                                diag["interventions"], diag["drift_warnings"],
                                                diag["affect_pulses"], diag["tokens_processed"],
                                            )

                                    native_channels = split_native_thinking_generation(
                                        current_response,
                                        native_thinking=(native_thinking is True),
                                    )
                                    current_response = native_channels.surface
                                    surface_control_state.update(
                                        {
                                            "native_thinking_enabled": native_thinking is True,
                                            "native_thinking_boundary_closed": (
                                                native_channels.boundary_closed
                                            ),
                                            "native_thinking_private_chars": len(
                                                native_channels.reasoning
                                            ),
                                        }
                                    )
                                    response_text = (
                                        f"{operator_response_prefix}{current_response}"
                                        if operator_evidence_contract and current_response.strip()
                                        else current_response
                                    )
                                    if operator_evidence_contract:
                                        # Composition disclosure: downstream must see
                                        # how much of the delivered evidence came from
                                        # fixed scaffolding vs the model itself.
                                        operator_evidence_receipt = {
                                            "prefix_chars": len(operator_response_prefix or ""),
                                            "model_chars": len(current_response or ""),
                                            "model_substantive": not (
                                                _operator_evidence_model_contribution_insufficient(
                                                    current_response
                                                )
                                            ),
                                        }
                                        trimmed_response = _trim_complete_operator_evidence(response_text)
                                        if trimmed_response != response_text:
                                            logger.warning(
                                                "⚠️ [WORKER] Trimmed clipped/meta operator-evidence tail "
                                                "(chars %d -> %d).",
                                                len(response_text or ""),
                                                len(trimmed_response or ""),
                                            )
                                            response_text = trimmed_response
                                    total_generated_tokens = token_count

                                    if soft_cancelled or deadline_hit:
                                        # Preempted or deadline-stopped turn: retries
                                        # would defeat the point, but the PURE terminal
                                        # transforms still run — a cancelled partial
                                        # must not ship internal leakage or an
                                        # unnormalized strict fragment.
                                        response_text = _route_cooperative_partial_draft(
                                            job,
                                            response_text,
                                            surface_control_state,
                                            is_proof=(
                                                proof_evaluation_contract
                                                or strict_answer_contract
                                                or strict_value_contract
                                            ),
                                        )
                                        if sentinel_ontology_aborted:
                                            response_text = ""
                                        if response_text:
                                            if strict_value_contract:
                                                response_text = _normalize_strict_value_response(
                                                    response_text,
                                                    expected_value=expected_strict_value,
                                                )
                                            elif strict_answer_contract:
                                                response_text = _normalize_strict_answer_response(
                                                    response_text,
                                                    envelope_prefixed=strict_envelope_prefixed,
                                                )
                                        # TERMINAL REJECTIONS still apply. The
                                        # pure transforms above are not the whole
                                        # contract: a cancelled partial can still
                                        # be an incomplete proof, an operator
                                        # -evidence draft with no model
                                        # contribution, or an ungrounded
                                        # capability inventory. Retries are
                                        # rightly skipped — the turn was
                                        # preempted — but shipping a fragment
                                        # the normal terminal path would have
                                        # REFUSED is how cancellation became a
                                        # way around the contracts.
                                        if response_text:
                                            cancel_refusal = _terminal_contract_refusal(
                                                job,
                                                response_text,
                                                proof_evaluation_contract=proof_evaluation_contract,
                                                operator_evidence_contract=operator_evidence_contract,
                                                model_continuation=current_response,
                                            )
                                            if cancel_refusal:
                                                logger.warning(
                                                    "🚫 [WORKER] Cancelled partial failed a terminal "
                                                    "contract (%s); withholding it.",
                                                    cancel_refusal,
                                                )
                                                if proof_evaluation_contract and (
                                                    cancel_refusal == "proof_fragment_incomplete"
                                                ):
                                                    proof_contract_incomplete = True
                                                response_text = ""
                                        logger.info(
                                            "✋ [WORKER] %s honored for job seq=%d after %d tokens.",
                                            (
                                                "Deadline stop"
                                                if deadline_hit
                                                else "Soft-cancel"
                                            ),
                                            job_seq,
                                            token_count,
                                        )
                                        break

                                    if proof_evaluation_contract and _proof_evaluation_fragment_incomplete(response_text):
                                        if internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Incomplete proof/evaluation response on attempt %s "
                                                "(tokens=%d, chars=%d, role_stop=%s). Retrying with stricter prompt.",
                                                internal_attempt + 1,
                                                token_count,
                                                len(response_text or ""),
                                                role_continuation_hit,
                                            )
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            prompt = _build_proof_evaluation_retry_prompt(
                                                original_messages,
                                                original_prompt,
                                            )
                                            _prepare_clean_retry_kwargs(kwargs, structured=False)
                                            continue
                                        logger.warning(
                                            "🚨 [WORKER] Proof/evaluation response remained incomplete after retries."
                                        )
                                        # The draft is still delivered (a partial
                                        # proof may hold scoreable content), but the
                                        # contract verdict is FAILED — the payload
                                        # flag stops callers from trusting a
                                        # fragment as a completed proof.
                                        proof_contract_incomplete = True

                                    if operator_evidence_contract and (
                                        _operator_evidence_fragment_incomplete(response_text)
                                        or _operator_evidence_model_contribution_insufficient(
                                            current_response
                                        )
                                    ):
                                        rejection_reasons = _operator_evidence_rejection_reasons(response_text)
                                        if _operator_evidence_model_contribution_insufficient(
                                            current_response
                                        ):
                                            rejection_reasons.append(
                                                "model_contribution_insufficient"
                                            )
                                        logger.warning(
                                            "⚠️ [WORKER] Rejected operator-evidence draft reasons=%s excerpt=%r",
                                            ",".join(rejection_reasons[:8]) or "unknown",
                                            str(response_text or "").strip()[:360],
                                        )
                                        if internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Incomplete operator-evidence response on attempt %s "
                                                "(tokens=%d, chars=%d, role_stop=%s). Retrying with stricter prompt.",
                                                internal_attempt + 1,
                                                token_count,
                                                len(response_text or ""),
                                                role_continuation_hit,
                                            )
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            prompt, operator_response_prefix = _build_operator_evidence_retry_prompt(
                                                original_messages,
                                                original_prompt,
                                            )
                                            _prepare_clean_retry_kwargs(kwargs, structured=False)
                                            continue
                                        logger.warning(
                                            "🚨 [WORKER] Operator-evidence response remained incomplete after retries."
                                        )
                                        response_text = ""
                                        break

                                    if sentinel_loop_aborted:
                                        if _loop_abort_prefix_is_servable(
                                            job,
                                            response_text,
                                        ):
                                            logger.warning(
                                                "🛡️ [WORKER] Preserving %d clean chars after "
                                                "a late loop abort instead of restarting from token zero.",
                                                len(str(response_text or "")),
                                            )
                                            surface_control_state[
                                                "sentinel_loop_prefix_preserved"
                                            ] = True
                                            sentinel_loop_aborted = False
                                        elif internal_attempt < max_internal_retries:
                                            logger.warning("⚠️ [WORKER] Retrying generation cleanly after loop abort (attempt %s)...", internal_attempt + 1)
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            if proof_evaluation_contract:
                                                prompt = _build_proof_evaluation_retry_prompt(
                                                    original_messages,
                                                    original_prompt,
                                                )
                                            _prepare_clean_retry_kwargs(kwargs, structured=bool(schema))
                                            continue
                                        else:
                                            logger.warning("🚨 [WORKER] Out of retries for loop abort. Returning truncated prefix.")
                                            break

                                    if sentinel_ontology_aborted:
                                        (
                                            ontology_retry_allowed,
                                            retry_deadline_open,
                                            retry_wall_open,
                                        ) = _ontology_retry_permitted(
                                            internal_attempt=internal_attempt,
                                            max_internal_retries=max_internal_retries,
                                            ontology_retry_count=ontology_retry_count,
                                            job_deadline_unix=job_deadline_unix,
                                            user_surface=bool(
                                                job.get(
                                                    "clean_user_surface_contract",
                                                    False,
                                                )
                                            ),
                                            surface_retry_started=surface_retry_started,
                                            surface_retry_wall_s=surface_retry_wall_s,
                                        )
                                        if ontology_retry_allowed:
                                            ontology_retry_count += 1
                                            logger.warning("⚠️ [WORKER] Retrying generation cleanly after ontological violation (attempt %s)...", internal_attempt + 1)
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            # Add a slight temperature penalty or just start fresh
                                            _prepare_clean_retry_kwargs(kwargs, structured=bool(schema))
                                            continue
                                        else:
                                            logger.warning(
                                                "🚨 [WORKER] Ontology repair unavailable or exhausted "
                                                "(attempts=%d deadline_open=%s wall_open=%s). "
                                                "Returning bounded refusal.",
                                                ontology_retry_count,
                                                retry_deadline_open,
                                                retry_wall_open,
                                            )
                                            response_text = get_refusal_fallback(seed=token_count)
                                            break

                                    if schema and str(response_text or "").strip():
                                        # ENFORCE the supplied schema: structured
                                        # mode previously only forced temp=0 and a
                                        # leading brace, so prose or wrong-shape
                                        # JSON was certified as structured output.
                                        schema_ok, schema_fail, normalized_json = (
                                            _validate_schema_output(response_text, schema)
                                        )
                                        if schema_ok:
                                            response_text = normalized_json
                                            schema_validation_failed = ""
                                        elif internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Structured output failed schema validation "
                                                "on attempt %s (%s). Retrying.",
                                                internal_attempt + 1,
                                                schema_fail,
                                            )
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            _prepare_clean_retry_kwargs(kwargs, structured=True)
                                            continue
                                        else:
                                            # Deliver the draft WITH a failure
                                            # receipt rather than discarding real
                                            # work — the parent decides.
                                            schema_validation_failed = schema_fail
                                            logger.warning(
                                                "🚨 [WORKER] Structured output still schema-invalid "
                                                "after retries (%s).",
                                                schema_fail,
                                            )

                                    if strict_answer_contract:
                                        sanitized_text, sanitizer_reasons = (
                                            _route_telemetry_sanitizer_draft(
                                                response_text,
                                                is_proof=True,
                                                authored_surface_repair_available=False,
                                            )
                                        )
                                        surface_control_state[
                                            "telemetry_sanitizer_reasons"
                                        ] = sanitizer_reasons[:8]
                                        if sanitizer_reasons:
                                            logger.warning(
                                                "🚨 [WORKER] Strict answer draft failed sanitizer "
                                                "reasons=%s.",
                                                ",".join(sanitizer_reasons[:8]),
                                            )
                                            response_text = ""
                                            break
                                        response_text = sanitized_text
                                        response_text = _normalize_strict_answer_response(
                                            response_text,
                                            envelope_prefixed=strict_envelope_prefixed,
                                        )
                                    elif strict_value_contract:
                                        raw_strict_value_text = response_text
                                        response_text = _normalize_strict_value_response(
                                            response_text,
                                            expected_value=expected_strict_value,
                                        )
                                        # Seeded-probe integrity: when the prompt was
                                        # built FROM the expected value and normalization
                                        # replaced a non-exact draft with it, the result
                                        # measures protocol echo, not model merit — the
                                        # payload must say so and preserve the raw draft
                                        # for audit.
                                        if (
                                            expected_strict_value
                                            and response_text == expected_strict_value
                                            and raw_strict_value_text.strip()
                                            != expected_strict_value
                                        ):
                                            strict_value_normalized_from_draft = (
                                                raw_strict_value_text.strip()[:240]
                                            )
                                        if response_text.strip():
                                            sanitized_text, sanitizer_reasons = (
                                                _route_telemetry_sanitizer_draft(
                                                    response_text,
                                                    is_proof=True,
                                                    authored_surface_repair_available=False,
                                                )
                                            )
                                            surface_control_state[
                                                "telemetry_sanitizer_reasons"
                                            ] = sanitizer_reasons[:8]
                                            if sanitizer_reasons:
                                                logger.warning(
                                                    "🚨 [WORKER] Strict value draft failed sanitizer "
                                                    "after normalization reasons=%s.",
                                                    ",".join(sanitizer_reasons[:8]),
                                                )
                                                response_text = ""
                                                break
                                            response_text = sanitized_text
                                        if raw_strict_value_text.strip() and not response_text.strip():
                                            logger.warning(
                                                "⚠️ [WORKER] Strict value draft rejected: %r",
                                                raw_strict_value_text.strip()[:160],
                                            )
                                    else:
                                        response_text, sanitizer_reasons = (
                                            _route_telemetry_sanitizer_draft(
                                                response_text,
                                                is_proof=proof_evaluation_contract,
                                                authored_surface_repair_available=(
                                                    surface_quality_gate_enabled
                                                ),
                                            )
                                        )
                                        surface_control_state[
                                            "telemetry_sanitizer_reasons"
                                        ] = sanitizer_reasons[:8]
                                        if sanitizer_reasons:
                                            if response_text:
                                                logger.warning(
                                                    "⚠️ [WORKER] Live draft failed telemetry sanitizer "
                                                    "reasons=%s; routing the intact draft through "
                                                    "bounded authored surface repair.",
                                                    ",".join(sanitizer_reasons[:8]),
                                                )
                                            else:
                                                logger.warning(
                                                    "🚨 [WORKER] Draft failed telemetry sanitizer "
                                                    "reasons=%s; strict caller-side recovery required.",
                                                    ",".join(sanitizer_reasons[:8]),
                                                )
                                                break

                                    if surface_quality_gate_enabled and response_text.strip():
                                        # Content provenance is immutable at the worker
                                        # boundary. A detector may reject a draft, but it
                                        # cannot replace Aura's words with canned runtime
                                        # prose and still call the result model-authored.
                                        # Self-claim verification remains downstream where
                                        # it can request a new authored candidate or fail
                                        # honestly; this layer performs only authorship-
                                        # preserving normalization.
                                        # Normalise whitespace-only defects
                                        # BEFORE judging quality. The local
                                        # model welds its list markers and
                                        # sentences to the previous line —
                                        # "this down:- Total marbles… = 12
                                        # marbles.Probability of drawing…" —
                                        # and the run-together text then reads
                                        # as a repetition loop with no
                                        # structure. Judging the reply on a
                                        # formatting defect the runtime already
                                        # knows how to fix cost the person the
                                        # answer, repeatedly, on 2026-07-26.
                                        formatted_surface = _normalize_surface_format(response_text)
                                        if formatted_surface and formatted_surface != response_text:
                                            append_text_mutation(
                                                surface_control_state,
                                                stage="mlx_worker.surface_format",
                                                method="normalize_user_facing_format",
                                                reasons=["jammed_markers"],
                                                before=response_text,
                                                after=formatted_surface,
                                                deterministic=True,
                                                authorship_effect="preserved",
                                            )
                                            response_text = formatted_surface
                                        pre_shape_reasons = _surface_quality_failure_reasons(
                                            job,
                                            response_text,
                                        )
                                        shaped_surface = _repair_live_user_surface_instruction_shape(
                                            job,
                                            response_text,
                                        )
                                        if shaped_surface != response_text:
                                            logger.info(
                                                "🛡️ [WORKER] Repaired explicit user-surface "
                                                "shape before quality validation."
                                            )
                                            surface_control_state[
                                                "instruction_shape_repair_applied"
                                            ] = True
                                            append_text_mutation(
                                                surface_control_state,
                                                stage="mlx_worker.instruction_shape",
                                                method="deterministic_instruction_shape",
                                                reasons=pre_shape_reasons or ["instruction_shape"],
                                                before=response_text,
                                                after=shaped_surface,
                                                deterministic=True,
                                                authorship_effect="preserved",
                                            )
                                            response_text = shaped_surface
                                        surface_control_state["surface_quality_gate_attempts"] = int(
                                            surface_control_state.get("surface_quality_gate_attempts", 0)
                                            or 0
                                        ) + 1
                                        rejection_reasons = _surface_quality_failure_reasons(
                                            job,
                                            response_text,
                                        )
                                        if "escaped_control_artifact" in rejection_reasons:
                                            unescaped_surface = (
                                                _repair_live_user_surface_escaped_newlines(
                                                    response_text
                                                )
                                            )
                                            if unescaped_surface:
                                                unescaped_reasons = (
                                                    _surface_quality_failure_reasons(
                                                        job,
                                                        unescaped_surface,
                                                    )
                                                )
                                                if "escaped_control_artifact" not in unescaped_reasons:
                                                    logger.info(
                                                        "🛡️ [WORKER] Restored newlines the model "
                                                        "emitted as literal escapes."
                                                    )
                                                    append_text_mutation(
                                                        surface_control_state,
                                                        stage="mlx_worker.escaped_control_artifact",
                                                        method="unescape_control_sequences",
                                                        reasons=["escaped_control_artifact"],
                                                        before=response_text,
                                                        after=unescaped_surface,
                                                        deterministic=True,
                                                        authorship_effect="preserved",
                                                    )
                                                    response_text = unescaped_surface
                                                    rejection_reasons = unescaped_reasons
                                        if set(rejection_reasons) == {"truncated_tail"}:
                                            completed_surface = (
                                                _repair_live_user_surface_truncated_tail(
                                                    response_text
                                                )
                                            )
                                            completed_reasons = (
                                                _surface_quality_failure_reasons(
                                                    job,
                                                    completed_surface,
                                                )
                                                if completed_surface
                                                else rejection_reasons
                                            )
                                            if completed_surface and not completed_reasons:
                                                logger.info(
                                                    "🛡️ [WORKER] Kept complete foreground "
                                                    "sentences after a clipped tail."
                                                )
                                                append_text_mutation(
                                                    surface_control_state,
                                                    stage="mlx_worker.truncated_tail",
                                                    method="retain_complete_sentences",
                                                    reasons=["truncated_tail"],
                                                    before=response_text,
                                                    after=completed_surface,
                                                    deterministic=True,
                                                    authorship_effect="preserved",
                                                )
                                                response_text = completed_surface
                                                rejection_reasons = []
                                        if rejection_reasons:
                                            telemetry_surface = _repair_live_user_surface_operational_status(
                                                response_text,
                                                rejection_reasons,
                                                job,
                                            )
                                            telemetry_reasons = _surface_quality_failure_reasons(
                                                job,
                                                telemetry_surface,
                                            )
                                            if telemetry_surface and not telemetry_reasons:
                                                logger.info(
                                                    "🛡️ [WORKER] Repaired live status draft "
                                                    "with concrete runtime telemetry."
                                                )
                                                append_text_mutation(
                                                    surface_control_state,
                                                    stage="mlx_worker.operational_status",
                                                    method="grounded_runtime_telemetry_repair",
                                                    reasons=rejection_reasons,
                                                    before=response_text,
                                                    after=telemetry_surface,
                                                    deterministic=True,
                                                    authorship_effect="replaced_by_runtime",
                                                )
                                                response_text = telemetry_surface
                                                rejection_reasons = []
                                        if rejection_reasons:
                                            # Reasons that name a REMOVABLE
                                            # span, and what removes it. A
                                            # rejection that can be repaired
                                            # should cost the person nothing;
                                            # the second one of these arrived
                                            # as a copy of the first branch,
                                            # so it is a table now.
                                            repairs = {
                                                "internal_task_prompt_leak": (
                                                    "strip_private_planning_prefix",
                                                    "separate_private_plan_and_revalidate_public_suffix",
                                                ),
                                                "prompt_artifact": (
                                                    "strip_prompt_artifacts",
                                                    "cut_transcript_continuation_and_revalidate",
                                                ),
                                                "runtime_boilerplate": (
                                                    "repair_runtime_boilerplate",
                                                    "remove_matching_sentences_and_revalidate",
                                                ),
                                                "verbatim_statement_repeat": (
                                                    "repair_verbatim_repeats",
                                                    "drop_repeated_sentences_and_revalidate",
                                                ),
                                            }
                                            for _reason, (
                                                _repair_name,
                                                _method,
                                            ) in repairs.items():
                                                if _reason not in rejection_reasons:
                                                    continue
                                                try:
                                                    import core.conversation.response_reliability as _rr

                                                    candidate = getattr(_rr, _repair_name)(
                                                        response_text
                                                    )
                                                    candidate_reasons = (
                                                        _surface_quality_failure_reasons(
                                                            job,
                                                            candidate,
                                                        )
                                                        if candidate
                                                        else rejection_reasons
                                                    )
                                                    if (
                                                        candidate
                                                        and candidate != str(response_text or "").strip()
                                                    ):
                                                        logger.info(
                                                            "🛡️ [WORKER] Repaired %s and revalidated the "
                                                            "remaining authored answer.",
                                                            _reason,
                                                        )
                                                        append_text_mutation(
                                                            surface_control_state,
                                                            stage=f"mlx_worker.{_reason}",
                                                            method=_method,
                                                            reasons=[_reason],
                                                            before=response_text,
                                                            after=candidate,
                                                            deterministic=True,
                                                            authorship_effect="preserved",
                                                        )
                                                        response_text = candidate
                                                        rejection_reasons = candidate_reasons
                                                except (
                                                    ImportError,
                                                    AttributeError,
                                                    RuntimeError,
                                                    TypeError,
                                                    ValueError,
                                                ) as repair_exc:
                                                    logger.debug(
                                                        "Sentence repair %s skipped: %s",
                                                        _repair_name,
                                                        repair_exc,
                                                    )
                                        if rejection_reasons:
                                            surface_control_state["surface_quality_gate_passed"] = False
                                            surface_control_state["surface_quality_gate_reasons"] = rejection_reasons[:8]
                                            # Keep the draft. It is suppressed,
                                            # not deleted — the caller decides
                                            # whether serving it beats serving
                                            # nothing, and it cannot make that
                                            # judgement about text it never saw.
                                            _remember_surface_quality_rejected_draft(
                                                surface_control_state,
                                                response_text,
                                                rejection_reasons,
                                            )
                                            validation_resolution = _surface_prompt_resolution(job)
                                            logger.warning(
                                                "⚠️ [WORKER] Rejected live user-surface draft "
                                                "reasons=%s validation_source=%s "
                                                "validation_sha256=%s validation_chars=%d excerpt=%r",
                                                ",".join(rejection_reasons[:8]) or "unknown",
                                                validation_resolution.source,
                                                validation_resolution.sha256[:12],
                                                len(validation_resolution.prompt),
                                                str(response_text or "").strip()[:280],
                                            )
                                            if (
                                                bool(job.get("capability_inventory_contract", False))
                                                and set(rejection_reasons).issubset(
                                                    {
                                                        "truncated_tail",
                                                        "too_thin_for_operational_status_turn",
                                                        "too_thin_for_status_turn",
                                                        "too_short_for_user_turn",
                                                        "too_thin_for_user_turn",
                                                    }
                                                )
                                            ):
                                                # Waive ONLY on evidence: a clipped draft
                                                # that already shows categories, governance,
                                                # and the non-execution boundary is safe for
                                                # downstream deterministic completion. A
                                                # draft without that grounding stays failed —
                                                # relabeling it "passed" asserted a pass that
                                                # never happened.
                                                inventory_grounded, inventory_evidence = (
                                                    _capability_inventory_minimum_grounding(
                                                        response_text
                                                    )
                                                )
                                                if inventory_grounded:
                                                    logger.warning(
                                                        "🛡️ [WORKER] Waiving thin-response reasons for a "
                                                        "grounded capability inventory draft; downstream "
                                                        "deterministic completion finishes the tail."
                                                    )
                                                    surface_control_state["surface_quality_gate_passed"] = True
                                                    surface_control_state["surface_quality_gate_reasons"] = []
                                                    surface_control_state[
                                                        "surface_quality_gate_exemption"
                                                    ] = "capability_inventory_minimum_grounding"
                                                    surface_control_state[
                                                        "surface_quality_gate_waived_reasons"
                                                    ] = rejection_reasons[:8]
                                                    break
                                                logger.warning(
                                                    "⚠️ [WORKER] Clipped capability inventory draft lacks "
                                                    "minimum grounding (%s); keeping the gate failure.",
                                                    ",".join(
                                                        key
                                                        for key, present in inventory_evidence.items()
                                                        if not present
                                                    )
                                                    or "unknown",
                                                )
                                            surface_wall_exceeded = _surface_retry_wall_exceeded(
                                                surface_retry_started, surface_retry_wall_s
                                            )
                                            completion_retry_reasons = {
                                                "truncated_tail",
                                                "final_answer_missing",
                                                "missing_final_answer",
                                                "incomplete_code_response",
                                            }
                                            completion_only_failure = bool(
                                                rejection_reasons
                                                and set(rejection_reasons).issubset(
                                                    completion_retry_reasons
                                                )
                                            )
                                            deadline_open = bool(
                                                job_deadline_unix <= 0.0
                                                or time.time() < job_deadline_unix
                                            )
                                            if completion_only_failure and deadline_open:
                                                # The generic wall stops stylistic retry
                                                # storms. It must not make a max-token
                                                # cutoff authoritative merely because the
                                                # first decode itself took twenty seconds.
                                                surface_wall_exceeded = False
                                            futile_retry = _surface_retry_is_futile(
                                                rejection_reasons
                                            )
                                            if futile_retry:
                                                surface_wall_exceeded = True
                                                failure_contract = (
                                                    "self-claim verification"
                                                    if "self_claim_verification_unavailable"
                                                    in rejection_reasons
                                                    else "surface prompt provenance"
                                                )
                                                logger.error(
                                                    "🛑 [WORKER] %s contract failed; refusing "
                                                    "futile model retries.",
                                                    failure_contract,
                                                )
                                            if surface_wall_exceeded and internal_attempt < max_internal_retries:
                                                logger.warning(
                                                    "🛡️ [WORKER] Surface-gate retry wall (%.0fs) reached after "
                                                    "attempt %d; salvaging best draft instead of re-drafting.",
                                                    surface_retry_wall_s,
                                                    internal_attempt + 1,
                                                )
                                            if internal_attempt < max_internal_retries and not surface_wall_exceeded:
                                                if surface_retry_started <= 0.0:
                                                    surface_retry_started = time.monotonic()
                                                if _expand_user_surface_retry_budget(
                                                    kwargs,
                                                    rejection_reasons,
                                                    hard_ceiling=job.get(
                                                        "hard_output_token_ceiling"
                                                    ),
                                                ):
                                                    logger.info(
                                                        "🛡️ [WORKER] Expanded same-worker live reply budget to %s "
                                                        "after structural truncation.",
                                                        kwargs.get("max_tokens"),
                                                    )
                                                if _self_claim_retry_uses_original_context(
                                                    rejection_reasons
                                                ):
                                                    prompt = original_prompt
                                                else:
                                                    prompt = _build_user_surface_quality_retry_prompt(
                                                        tokenizer=tokenizer,
                                                        messages=original_messages,
                                                        tools=tools,
                                                        fallback_prompt=original_prompt,
                                                        reasons=rejection_reasons,
                                                        job=job,
                                                    )
                                                _prepare_clean_retry_kwargs(
                                                    kwargs,
                                                    structured=bool(
                                                        set(rejection_reasons)
                                                        & _SEMANTIC_COUNT_CONTRACT_RETRY_REASONS
                                                    ),
                                                )
                                                continue
                                            logger.warning(
                                                "🚨 [WORKER] Live user-surface quality gate exhausted retries."
                                            )
                                            # Salvage over empty: an empty reply is the worst outcome
                                            # (it triggers the parent's inline-retry storm and sustained
                                            # lag). If the ONLY defect was servile generic-assistant
                                            # language, strip it and keep the good part — "You're welcome!
                                            # Is there anything else I can help with?" becomes
                                            # "You're welcome!" for a brief social turn.
                                            salvaged = ""
                                            if "generic_assistant_language" in (rejection_reasons or []):
                                                try:
                                                    from core.conversation.response_reliability import (
                                                        repair_generic_assistant_language,
                                                    )

                                                    _, _user_parts = _extract_message_parts(
                                                        original_messages, original_prompt
                                                    )
                                                    _user_turn = _user_parts[-1] if _user_parts else ""
                                                    candidate = repair_generic_assistant_language(
                                                        _user_turn, response_text
                                                    )
                                                    if (
                                                        candidate.strip()
                                                        and candidate.strip() != str(response_text or "").strip()
                                                        and not _surface_quality_failure_reasons(
                                                            job, candidate
                                                        )
                                                    ):
                                                        salvaged = candidate.strip()
                                                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _salvage_exc:
                                                    logger.debug("Generic-language salvage skipped: %s", _salvage_exc)
                                            if salvaged:
                                                logger.info(
                                                    "🛡️ [WORKER] Salvaged a clean brief reply after generic-language "
                                                    "retries instead of yielding zero tokens."
                                                )
                                                append_text_mutation(
                                                    surface_control_state,
                                                    stage="mlx_worker.generic_language_salvage",
                                                    method="deterministic_generic_language_repair",
                                                    reasons=rejection_reasons,
                                                    before=response_text,
                                                    after=salvaged,
                                                    deterministic=True,
                                                    authorship_effect="preserved",
                                                )
                                                response_text = salvaged
                                                surface_control_state["surface_quality_gate_passed"] = True
                                                surface_control_state["surface_quality_gate_reasons"] = []
                                            else:
                                                best_draft, residual_reasons, salvage_repairs = (
                                                    _salvage_exhausted_user_surface(
                                                        job,
                                                        response_text,
                                                        rejection_reasons,
                                                    )
                                                )
                                                if best_draft:
                                                    logger.info(
                                                        "🛡️ [WORKER] Delivering best honest draft after gate "
                                                        "exhaustion (residual=%s, repairs=%s) instead of a dead turn.",
                                                        ",".join(residual_reasons) or "none",
                                                        ",".join(salvage_repairs) or "none",
                                                    )
                                                    append_text_mutation(
                                                        surface_control_state,
                                                        stage="mlx_worker.exhaustion_salvage",
                                                        method=(
                                                            "best_honest_draft+"
                                                            + "+".join(salvage_repairs)
                                                            if salvage_repairs
                                                            else "best_honest_draft"
                                                        ),
                                                        reasons=rejection_reasons,
                                                        before=response_text,
                                                        after=best_draft,
                                                        deterministic=True,
                                                        authorship_effect="preserved",
                                                    )
                                                    response_text = best_draft
                                                    surface_control_state["surface_quality_gate_passed"] = (
                                                        not residual_reasons
                                                    )
                                                    surface_control_state["surface_quality_gate_reasons"] = (
                                                        residual_reasons[:8]
                                                    )
                                                else:
                                                    response_text = ""
                                            break
                                        surface_control_state["surface_quality_gate_passed"] = True
                                        surface_control_state["surface_quality_gate_reasons"] = []

                                    if response_text.strip() or (
                                        not schema
                                        and not strict_answer_contract
                                        and not strict_value_contract
                                    ):
                                        break # Success or not a structured task

                                    if strict_answer_contract or strict_value_contract:
                                        if internal_attempt < max_internal_retries:
                                            logger.warning(
                                                "⚠️ [WORKER] Empty strict response on attempt %s. Retrying...",
                                                internal_attempt + 1,
                                            )
                                            if internal_attempt == 0 or strict_value_contract:
                                                if strict_value_contract and expected_strict_value:
                                                    prompt = _build_exact_strict_value_prompt(
                                                        expected_strict_value
                                                    )
                                                else:
                                                    prompt = _build_strict_answer_retry_prompt(
                                                        original_messages,
                                                        original_prompt,
                                                    )
                                                strict_envelope_prefixed = bool(strict_answer_contract)
                                            if prompt_cache_lru is not None:
                                                prompt_cache_lru.clear()
                                            if mx and device != "cpu":
                                                _clear_mlx_cache(mx)
                                            _prepare_clean_retry_kwargs(kwargs, structured=True)
                                            continue
                                        logger.warning("🚨 [WORKER] Strict contract exhausted internal retries.")
                                        break

                                    logger.warning("⚠️ [WORKER] Empty structured response on attempt %s. Retrying...", internal_attempt + 1)
                                finally:
                                    watchdog.stop_job()
                    finally:
                        if not _restore_surface_generation_controls(surface_control_state):
                            # Unknown steering/recurrent state on the resident
                            # model: serve THIS response, then exit so the
                            # parent respawns a clean worker instead of
                            # letting the contamination leak into every
                            # subsequent job.
                            worker_active = False
                            ipc_writer.put(
                                {
                                    "status": "degraded",
                                    "action": "surface_restore_failed",
                                    "message": (
                                        "surface control restore failed; worker will exit "
                                        "after this response for a clean respawn"
                                    ),
                                }
                            )

                    expected_empty_precompile = bool(
                        not response_text.strip()
                        and _expected_empty_warmup_precompile(job)
                    )
                    if not response_text.strip():
                        if expected_empty_precompile:
                            logger.info(
                                "[WORKER] One-token warmup precompile produced no visible text; "
                                "the required visible readiness probe will verify conversation output."
                            )
                        elif token_count > 0:
                            # The decoder DID produce tokens; something
                            # downstream (quality gate, salvage, role-drift
                            # truncation) discarded them. Saying "ZERO tokens"
                            # here sent every investigation at the sampler and
                            # the KV cache, which were both working — measured
                            # live as "yielded ZERO tokens ... token_count: 75".
                            # Naming the tokens is the difference between a
                            # diagnosis and a guess. "1 token, no text" reads
                            # as a decode fault; "1 token, and it was
                            # <|im_end|>" says the model ended the turn
                            # immediately, which is a prompt or template
                            # problem and nothing to do with the sampler.
                            logger.warning(
                                "⚠️ [WORKER] Generation produced %d token(s) but no text "
                                "survived to the caller — discarded downstream, not a decode "
                                "failure. Prompt length: %d, stop_sequences: %s, tokens: %s",
                                token_count,
                                len(prompt),
                                list(stop_sequences)[:4],
                                # The GENERATED tail. `tokens` starts as the
                                # encoded prompt and the decoder appends to it,
                                # so tokens[:8] is the system block and says
                                # nothing about what the model produced.
                                _name_tokens(tokenizer, tokens[-token_count:][:8]),
                            )
                        else:
                            logger.warning(
                                "⚠️ [WORKER] Generation yielded ZERO tokens. "
                                "Prompt length: %d, token_count: %d, stop_sequences: %s",
                                len(prompt), token_count, list(stop_sequences)[:4],
                            )
                        if len(prompt) > 2000:
                            logger.debug("Prompt snippet: %s...", prompt[:100])
                        # A genuinely zero-token generation outside the explicit
                        # one-token precompile usually means the prompt cache
                        # handed over a stale/corrupt KV state (the sampler hit
                        # EOS on the first step because cached KV disagreed with
                        # the fresh prompt), so clear it and the Metal cache.
                        #
                        # But when tokens WERE produced, KV is exonerated by the
                        # decode itself, and clearing here punished the cache for
                        # a downstream rejection — throwing away the conversation
                        # prefix that keeps later turns inside their budget, on
                        # exactly the turns already going badly.
                        if token_count == 0:
                            try:
                                if prompt_cache_lru is not None:
                                    prompt_cache_lru.clear()
                            except (AttributeError, RuntimeError) as exc:
                                logger.debug("Prompt cache clear failed after zero-token generation: %s", exc)
                            if mx and device != "cpu":
                                _clear_mlx_cache(mx)

                    try:
                        if engine is not None:
                            if response_text.strip():
                                engine.observe_generation(response_text)
                            elif not expected_empty_precompile:
                                engine.observe_generation(
                                    "",
                                    generation_health=0.0,
                                    cross_entropy=10.0,
                                )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as steering_obs_exc:
                        _record_mlx_degradation(
                            steering_obs_exc,
                            action="returned generation after affective steering observation failed",
                            severity="warning",
                        )
                        logger.debug("Affective steering post-generation observation failed: %s", steering_obs_exc)

                    semantic_completion_state = _semantic_completion_receipt_state(
                        job,
                        response_text,
                        generated_tokens=total_generated_tokens,
                    )
                    semantic_contract_satisfied = bool(
                        semantic_completion_state["semantic_completion_satisfied"]
                    )
                    surface_control_state.update(semantic_completion_state)
                    if (
                        semantic_completion_state["semantic_completion_incomplete"]
                        and not expected_empty_precompile
                    ):
                        logger.warning(
                            "User-surface generation ended before semantic completion: "
                            "missing_parts=%s quality=%s epistemic_covered=%s "
                            "terminal_boundary=%s tokens=%d",
                            semantic_completion_state[
                                "semantic_completion_missing_part_indexes"
                            ],
                            semantic_completion_state[
                                "semantic_completion_quality_reasons"
                            ],
                            semantic_completion_state[
                                "semantic_completion_epistemic_partition_covered"
                            ],
                            semantic_completion_state[
                                "semantic_completion_terminal_boundary"
                            ],
                            total_generated_tokens,
                        )

                    # Tag with action: "generate" so client can distinguish
                    # from init/heartbeat responses unambiguously.
                    generation_stop_reason = _classify_generation_stop_reason(
                        soft_cancelled=soft_cancelled,
                        deadline_hit=deadline_hit,
                        sentinel_aborted=sentinel_aborted,
                        role_continuation_hit=role_continuation_hit,
                        configured_stop_hit=configured_stop_hit,
                        hard_token_limit_hit=hard_token_limit_hit,
                        semantic_contract_satisfied=semantic_contract_satisfied,
                        generated_tokens=total_generated_tokens,
                        max_tokens=max(
                            1,
                            _safe_int(
                                surface_control_state.get("generation_max_tokens_applied"),
                                max_tokens,
                            ),
                        ),
                    )
                    surface_control_state["generation_stop_reason"] = generation_stop_reason
                    surface_control_state["generation_configured_stop_sequence"] = (
                        configured_stop_sequence
                    )
                    resume_required = _continuation_resume_should_bind(
                        generation_stop_reason=generation_stop_reason,
                        semantic_completion_incomplete=bool(
                            semantic_completion_state[
                                "semantic_completion_incomplete"
                            ]
                        ),
                    )
                    continuation_resume_handle = ""
                    if (
                        resume_required
                        and bool(response_text.strip())
                        and prompt_cache_lru is not None
                        and not disable_prompt_cache
                        and final_prompt_cache is not None
                        and not sentinel_aborted
                    ):
                        continuation_resume_handle = prompt_cache_lru.bind_resume(
                            model_key,
                            list(tokens),
                            prompt_cache=final_prompt_cache,
                            one_token_rollback=continuation_cache_rollback,
                        )
                    if resume_required and not continuation_resume_handle:
                        resume_unavailable_reason = (
                            _continuation_resume_unavailable_reason(
                                resume_required=resume_required,
                                cache_lru_available=prompt_cache_lru is not None,
                                cache_disabled=bool(disable_prompt_cache),
                                final_cache_available=final_prompt_cache is not None,
                                sentinel_aborted=bool(sentinel_aborted),
                                response_present=bool(response_text.strip()),
                            )
                        )
                    else:
                        resume_unavailable_reason = ""
                    if resume_unavailable_reason:
                        surface_control_state[
                            "continuation_resume_failure_reason"
                        ] = resume_unavailable_reason
                        logger.warning(
                            "Continuation resume capability unavailable after %s: %s",
                            generation_stop_reason,
                            resume_unavailable_reason,
                        )
                    surface_control_state["continuation_resume_available"] = bool(
                        continuation_resume_handle
                    )
                    if continuation_resume_handle:
                        surface_control_state[
                            "continuation_resume_handle"
                        ] = continuation_resume_handle
                    generate_payload: dict[str, Any] = {
                        "id": job.get("id"),
                        "action": "generate",
                        "status": "ok",
                        "text": response_text.strip(),
                        # Whether the endogenous pathway touched this
                        # generation, and if not, which check refused it.
                        # A turn that ran without it and a turn that ran
                        # with it have to be distinguishable afterwards.
                        "endogenous_bias": dict(_endo_receipt or {}),
                        "tokens_used": total_generated_tokens,
                        # The parent assembles prompts in characters but the
                        # tokenizer lives here. Carry the exact final rendered
                        # prompt measurement across IPC instead of updating a
                        # process-local evidence singleton nobody else reads.
                        "prompt_tokenization": {
                            "chars": len(str(prompt or "")),
                            "tokens": prompt_token_count,
                            "generated_tokens": total_generated_tokens,
                        },
                        "prompt_cache_reused_tokens": max(
                            0,
                            int(prompt_token_count) - int(prefill_tokens),
                        ),
                        # The parent's OOM footprint probe must not cost an IPC
                        # round trip, so the size rides along with every result.
                        "prompt_cache_bytes": (
                            int(prompt_cache_lru.retained_bytes())
                            if prompt_cache_lru is not None
                            else 0
                        ),
                        "prompt_cache_entries": (
                            int(prompt_cache_lru.retained_entries())
                            if prompt_cache_lru is not None
                            else 0
                        ),
                        "soft_cancelled": bool(soft_cancelled),
                        "deadline_exceeded": bool(deadline_hit),
                        "generation_stop_reason": generation_stop_reason,
                        "generation_configured_stop_sequence": configured_stop_sequence,
                        "speculative": {
                            "enabled": bool(use_speculative),
                            "draft_tokens_accepted": int(draft_accepted_tokens),
                        } if use_speculative else None,
                        "surface_control_receipt": _surface_generation_control_receipt(
                            job,
                            surface_control_state,
                        ),
                        "interoception": interoception_payload,
                    }
                    # Contract-truth verdicts: callers must never have to
                    # infer these from the text.
                    if schema:
                        generate_payload["schema_validated"] = not bool(
                            schema_validation_failed
                        )
                        if schema_validation_failed:
                            generate_payload["schema_validation_failed"] = str(
                                schema_validation_failed
                            )[:240]
                    if proof_evaluation_contract:
                        generate_payload["proof_contract_incomplete"] = bool(
                            proof_contract_incomplete
                        )
                    if strict_value_contract:
                        generate_payload["strict_value_seeded"] = bool(
                            expected_strict_value
                        )
                        if strict_value_normalized_from_draft:
                            generate_payload["strict_value_draft"] = (
                                strict_value_normalized_from_draft
                            )
                    if operator_evidence_contract and operator_evidence_receipt:
                        generate_payload["operator_evidence_composition"] = dict(
                            operator_evidence_receipt
                        )
                    ipc_writer.put(generate_payload)
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_mlx_degradation(
                        e,
                        action="returned generate error and cleared MLX cache after generation failure",
                        severity="degraded",
                    )
                    logger.error("Generation failed: %s", e)
                    # The id is REQUIRED: an id-less error cannot resolve the
                    # parent's pending future, which then waits to deadline.
                    ipc_writer.put({
                        "id": job.get("id"),
                        "status": "error",
                        "action": "generate",
                        "message": str(e),
                    })
                finally:
                    # [STABILITY v52] Guarantee VRAM gets purged after standard generation
                    # completes or fails. The next request starts from clean state.
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)

            elif action == "generate_batch":
                # Batched best-of-N candidate generation: N sequences decoded
                # in ONE batched pass — the raw-reasoning multiplier for the
                # verifier-selection amplifier. Candidates are intentionally
                # RAW (no sentinel/quality gates): the truth-engine verifiers
                # on the parent side are the selection mechanism.
                try:
                    if engine is None or not engine.is_active():
                        # None is fail-CLOSED: init crashes on failed steering
                        # attach, so a None engine here means the invariant
                        # broke — never a license to decode unsteered.
                        ipc_writer.put({
                            "id": job.get("id"),
                            "action": "generate_batch",
                            "status": "error",
                            "message": "Affective steering is inactive; batch generation blocked.",
                        })
                        continue
                    from mlx_lm import batch_generate
                    from mlx_lm.sample_utils import make_sampler

                    watchdog.start_job(str(job.get("id") or ""), "generate_batch")
                    try:
                        batch_prompt = str(job.get("prompt") or "")
                        if len(batch_prompt) > 400_000:
                            # Bounded input: an unbounded prompt string
                            # tokenizes into unbounded Metal work before any
                            # admission check can see it.
                            raise ValueError(
                                f"batch_prompt_too_large:{len(batch_prompt)}"
                            )
                        n = max(1, min(16, _safe_int(job.get("n"), 4)))
                        batch_max_tokens = max(
                            1,
                            min(2048, _safe_int(job.get("max_tokens"), 512)),
                        )
                        batch_max_tokens = _serving_lane_output_cap(
                            model_path,
                            str(job.get("serving_lane") or "foreground_standard"),
                            batch_max_tokens,
                        )
                        # Finite-range temperature: NaN/inf reached the
                        # sampler unchecked through the bare float coercion.
                        batch_temp = min(
                            max(_safe_float(job.get("temperature"), 0.8), 0.0), 2.0
                        )
                        token_ids = tokenizer.encode(batch_prompt)
                        batch_context_window = _serving_lane_context_window(
                            model_path,
                            str(job.get("serving_lane") or "foreground_standard"),
                            output_reserve=batch_max_tokens,
                            architectural_window=effective_context_window,
                        )
                        if len(token_ids) + batch_max_tokens > batch_context_window:
                            raise ValueError(
                                "context_window_exceeded:"
                                f"prompt_tokens={len(token_ids)}:"
                                f"output_reserve={batch_max_tokens}:"
                                f"window={batch_context_window}"
                            )
                        watchdog.activity()
                        batch_result = batch_generate(
                            model,
                            tokenizer,
                            prompts=[list(token_ids) for _ in range(n)],
                            max_tokens=batch_max_tokens,
                            sampler=make_sampler(temp=batch_temp, top_p=0.95),
                        )
                        watchdog.activity()
                        texts = [str(t or "").strip() for t in getattr(batch_result, "texts", [])]
                        tokens_used_by_candidate = [
                            len(tokenizer.encode(text)) if text else 0 for text in texts
                        ]
                        # Cardinality is part of the contract: trusting any
                        # iterable length let a decode fault report ok with
                        # fewer (or zero) candidates than the caller paid for.
                        nonempty = sum(1 for text in texts if text)
                        batch_status = "ok" if (len(texts) == n and nonempty > 0) else "error"
                        batch_payload: dict[str, Any] = {
                            "id": job.get("id"),
                            "action": "generate_batch",
                            "status": batch_status,
                            "texts": texts,
                            "candidates_requested": n,
                            "candidates_returned": len(texts),
                            "candidates_nonempty": nonempty,
                            "tokens_used": sum(tokens_used_by_candidate),
                            "tokens_used_by_candidate": tokens_used_by_candidate,
                            "prompt_tokenization": {
                                "chars": len(batch_prompt),
                                "tokens": len(token_ids),
                            },
                        }
                        if batch_status == "error":
                            batch_payload["message"] = (
                                f"batch_cardinality_violation:requested={n}:"
                                f"returned={len(texts)}:nonempty={nonempty}"
                            )
                        ipc_writer.put(batch_payload)
                    finally:
                        watchdog.stop_job()
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                    _record_mlx_degradation(
                        e,
                        action="returned generate_batch error after batched decoding failure",
                        severity="degraded",
                    )
                    logger.error("Batched generation failed: %s", e)
                    ipc_writer.put({
                        "id": job.get("id"),
                        "action": "generate_batch",
                        "status": "error",
                        "message": str(e),
                    })
                finally:
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)

            elif action == "stream":
                prompt = job.get("prompt")
                # Typed finite-range admission: streaming controls previously
                # flowed unvalidated from IPC into sampler construction.
                temp = _admit_sampling_control(job, "temp")
                top_p = _admit_sampling_control(job, "top_p")
                max_tokens = _admit_max_tokens(job.get("max_tokens", 512), 512)
                max_tokens = _serving_lane_output_cap(
                    model_path,
                    str(job.get("serving_lane") or "foreground_standard"),
                    max_tokens,
                )
                min_p = _admit_sampling_control(job, "min_p")
                repetition_penalty = _admit_sampling_control(job, "repetition_penalty")

                kwargs = {"max_tokens": max_tokens}
                if make_sampler:
                    sampler_kwargs = {"temp": temp, "top_p": top_p}
                    try:
                        import inspect as _insp2
                        _sparams2 = _insp2.signature(make_sampler).parameters
                        if "min_p" in _sparams2:
                            sampler_kwargs["min_p"] = min_p
                        if "repetition_penalty" in _sparams2:
                            sampler_kwargs["repetition_penalty"] = repetition_penalty
                    except (TypeError, ValueError):
                        logger.debug("stream make_sampler signature introspection unavailable")
                    kwargs["sampler"] = make_sampler(**sampler_kwargs)

                # Apply MLX penalties via logits processors
                logits_processors = []
                try:
                    from mlx_lm.sample_utils import make_logits_processors
                    _rp = repetition_penalty
                    _rcs = max(1, min(_safe_int(job.get("repetition_context_size"), 30), 512))
                    _pp = _admit_sampling_control(job, "presence_penalty")
                    if _rp and _rp > 1.0:
                        lp = make_logits_processors(
                            repetition_penalty=_rp,
                            repetition_context_size=_rcs,
                            presence_penalty=_pp,
                        )
                        if lp:
                            logits_processors.extend(lp)
                except ImportError as _exc:
                    logger.debug("Suppressed %s in core.brain.llm.mlx_worker: %s", type(_exc).__name__, _exc)
                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.warning("Could not apply penalty logits processors: %s", e)

                # The same guard the generate path installs.
                #
                # LIVE, 2026-08-20. This assembly carries only the repetition
                # penalty, so a conversational turn streamed here produced
                # exactly one token — <|im_start|>, a stop sequence — and the
                # person got "I couldn't get to an answer I'd stand behind on
                # that one" while the fetched answer sat in working memory. The
                # guard had been added to the other assembly an hour earlier
                # and this one never saw it, which is why it is a function now.
                if not _expected_empty_warmup_precompile(job):
                    try:
                        guard = build_nonempty_start_processor(tokenizer)
                        if guard is not None:
                            logits_processors.append(guard)
                        logger.info(
                            "🎯 [WORKER] Non-empty start guard %s (stream path).",
                            "ACTIVE" if guard is not None else "UNAVAILABLE",
                        )
                    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
                        _record_mlx_degradation(
                            e,
                            action="continued streamed generation without the non-empty start guard",
                            severity="warning",
                        )

                try:
                    semantic_terminal_guard = build_semantic_completion_terminal_guard(
                        tokenizer,
                        job,
                    )
                    if semantic_terminal_guard is not None:
                        logits_processors.append(semantic_terminal_guard)
                    logger.info(
                        "🧩 [WORKER] Semantic terminal guard %s (stream path).",
                        "ACTIVE" if semantic_terminal_guard is not None else "INACTIVE",
                    )
                except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
                    _record_mlx_degradation(
                        e,
                        action="continued streamed generation without semantic terminal guard",
                        severity="warning",
                    )

                if logits_processors:
                    kwargs["logits_processors"] = logits_processors

                stop_sequences = _merge_stop_sequences(job.get("stop_sequences") or [])

                try:
                    from mlx_lm.generate import stream_generate
                    # : NO GPUSentinel — same rationale as generate path.

                    surface_control_state = _apply_surface_generation_controls(engine, model, job)
                    _enforce_surface_controls_or_fail(job, surface_control_state)
                    try:
                        with metal_semaphore:
                            watchdog.start_job(str(job.get("id") or ""), "stream")
                            try:
                                full_text = ""
                                token_count = 0

                                # ── Token Sentinel for streaming path ─────────
                                try:
                                    from core.brain.llm.token_sentinel import (
                                        InterventionType,
                                        TokenSentinel,
                                        get_refusal_fallback,
                                    )
                                    stream_sentinel = TokenSentinel(
                                        check_interval=8,
                                        affect_interval=16,
                                        substrate_mem=substrate_mem,
                                        steering_hooks=_active_steering_hooks(engine),
                                        affect_expected=(
                                            _affect_expected
                                            and _safe_float(
                                                job.get(
                                                    "clean_user_surface_steering_alpha"
                                                ),
                                                1.0,
                                            )
                                            > 0.0
                                        ),
                                    )
                                except (ImportError, AttributeError, RuntimeError) as _stream_sent_exc:
                                    if bool(job.get("clean_user_surface_contract", False)):
                                        # Streamed tokens reach the user in real
                                        # time — there is no post-hoc retraction,
                                        # so decoding without the mid-generation
                                        # guard is refused, not degraded.
                                        _record_mlx_degradation(
                                            _stream_sent_exc,
                                            action="refused user-surface stream without TokenSentinel",
                                            severity="critical",
                                        )
                                        raise RuntimeError(
                                            f"token_sentinel_unavailable:{_stream_sent_exc}"
                                        ) from _stream_sent_exc
                                    _record_mlx_degradation(
                                        _stream_sent_exc,
                                        action="continued non-surface stream without TokenSentinel intervention checks",
                                        severity="degraded",
                                    )
                                    stream_sentinel = None

                                # [STABILITY v60] Definitive scrub of legacy kwargs.
                                clean_keys = {"temperature", "top_p", "min_p", "repetition_penalty", "repetition_context_size", "stop_words"}
                                clean_kwargs = {k: v for k, v in kwargs.items() if k not in clean_keys}

                                # Context-window admission before streamed Metal work.
                                _stream_prompt_text = str(prompt or "")
                                _stream_prompt_tokens = len(tokenizer.encode(_stream_prompt_text))
                                _stream_prefill_step_size = _runtime_prefill_step_size(model_path)
                                clean_kwargs["prefill_step_size"] = _stream_prefill_step_size
                                clean_kwargs["prompt_progress_callback"] = (
                                    _build_prefill_progress_callback(
                                        watchdog,
                                        ipc_writer,
                                        request_id=str(job.get("id") or ""),
                                        action="stream",
                                    )
                                )
                                if _speculative_eligible(
                                    draft_model,
                                    clean_kwargs,
                                    job,
                                    prefill_tokens=_stream_prompt_tokens,
                                    prefill_step_size=_stream_prefill_step_size,
                                ):
                                    clean_kwargs["draft_model"] = draft_model
                                # The assembler budgets in characters and has
                                # no tokenizer — loading one in the process that
                                # serves conversation is the thing that must not
                                # happen there. This process just encoded the
                                # prompt, so it knows both numbers for free.
                                try:
                                    from core.brain.llm.token_budget_evidence import (
                                        observe_prompt_tokenization,
                                    )

                                    observe_prompt_tokenization(
                                        len(_stream_prompt_text), _stream_prompt_tokens
                                    )
                                except (ImportError, AttributeError, TypeError, ValueError):
                                    logger.debug("chars-per-token observation skipped")
                                _stream_reserve = min(max(64, max_tokens), 2048)
                                _stream_context_window = _serving_lane_context_window(
                                    model_path,
                                    str(job.get("serving_lane") or "foreground_standard"),
                                    output_reserve=_stream_reserve,
                                    architectural_window=effective_context_window,
                                )
                                if _stream_prompt_tokens + _stream_reserve > _stream_context_window:
                                    raise RuntimeError(
                                        "context_window_exceeded:"
                                        f"prompt_tokens={_stream_prompt_tokens}:"
                                        f"output_reserve={_stream_reserve}:"
                                        f"window={_stream_context_window}"
                                    )

                                watchdog.activity()
                                sentinel_aborted = False
                                abort_reason = ""
                                semantic_contract_satisfied = False
                                for response in stream_generate(model, tokenizer, prompt=prompt, **clean_kwargs):
                                    watchdog.activity()
                                    token_count += 1
                                    token_text = response.text
                                    visible_len = len(full_text)
                                    full_text += token_text
                                    full_text, role_continuation_hit = _truncate_role_continuation(full_text)

                                    # ── Sentinel: mid-stream intervention ─────
                                    if stream_sentinel is not None:
                                        sentinel_signal = stream_sentinel.feed(token_text)
                                        if sentinel_signal.type in (
                                            InterventionType.ABORT_LOOP,
                                            InterventionType.ABORT_ONTOLOGY_VIOLATION,
                                        ):
                                            # ABORT_ONTOLOGY_VIOLATION previously
                                            # fell through to token emission —
                                            # the ontology hard stop had no
                                            # effect on this live path.
                                            logger.warning(
                                                "🚨 [SENTINEL-STREAM] Aborting (%s) at token %d: %s",
                                                sentinel_signal.type,
                                                token_count, sentinel_signal.reason,
                                            )
                                            sentinel_aborted = True
                                            abort_reason = str(sentinel_signal.reason or sentinel_signal.type)
                                            ipc_writer.put({
                                                "id": job.get("id"),
                                                "action": "stream",
                                                "status": "sentinel_abort",
                                                "text": "",
                                                "tokens_generated": token_count,
                                                "timestamp": time.time(),
                                            })
                                            break
                                        elif sentinel_signal.type in (InterventionType.ABORT_CAPITULATION,
                                                                      InterventionType.ABORT_BOUNDARY):
                                            logger.warning(
                                                "🚨 [SENTINEL-STREAM] Aborting at token %d: %s",
                                                token_count, sentinel_signal.reason,
                                            )
                                            sentinel_aborted = True
                                            abort_reason = str(sentinel_signal.reason or sentinel_signal.type)
                                            # Send the refusal as the final token
                                            ipc_writer.put({
                                                "id": job.get("id"),
                                                "action": "stream",
                                                "status": "sentinel_abort",
                                                "text": get_refusal_fallback(seed=token_count),
                                                "tokens_generated": token_count,
                                                "timestamp": time.time(),
                                            })
                                            break

                                    # Truncation runs BEFORE emission: raw tokens
                                    # were previously sent first, so stop content
                                    # and role markers leaked to the consumer and
                                    # could never be retracted; a stop at index
                                    # zero was explicitly missed by the old > 0.
                                    stop_hit = role_continuation_hit
                                    for stop in stop_sequences:
                                        stop_index = full_text.find(stop)
                                        if stop_index >= 0:
                                            full_text = full_text[:stop_index]
                                            stop_hit = True
                                            break

                                    semantic_stop_ready = bool(
                                        token_count % 8 == 0
                                        and _semantic_surface_stop_ready(
                                            job,
                                            full_text,
                                            generated_tokens=token_count,
                                        )
                                    )
                                    if semantic_stop_ready:
                                        semantic_contract_satisfied = True
                                        logger.info(
                                            "✅ [WORKER] Stream semantic completion contract "
                                            "satisfied at token %d.",
                                            token_count,
                                        )

                                    # Absolute cap check precedes emission so the
                                    # 8193rd token is never visible.
                                    if token_count > 8192:
                                        logger.warning("🏁 [WORKER] Hard token limit (8192) reached. Truncating.")
                                        break

                                    emit_text = (
                                        full_text[visible_len:]
                                        if len(full_text) > visible_len
                                        else ""
                                    )
                                    if emit_text:
                                        ipc_writer.put(
                                            {
                                                "id": job.get("id"),
                                                "action": "stream",
                                                "status": "token",
                                                "text": emit_text,
                                                "tokens_generated": token_count,
                                                "timestamp": time.time(),
                                            }
                                        )
                                    else:
                                        # A token that adds no VISIBLE text is still a
                                        # token. This was the only progress signal the
                                        # parent had, so decoding that produces no
                                        # visible delta — a detokenizer holding a
                                        # partial UTF-8 sequence, suppressed start ids,
                                        # a stop-sequence being scanned — looked
                                        # identical to a wedged worker. Live
                                        # 2026-07-26: "First-token HARD CEILING
                                        # exceeded (livelocked: heartbeats but zero
                                        # tokens) ... 107.7s" on an ~800-token prompt,
                                        # and a healthy generation was cancelled.
                                        #
                                        # `progress` carries no text and is retained
                                        # ahead of token/heartbeat telemetry. A terminal
                                        # answer may still preempt it under backpressure.
                                        ipc_writer.put(
                                            {
                                                "id": job.get("id"),
                                                "action": "stream",
                                                "status": "progress",
                                                "tokens_generated": token_count,
                                                "timestamp": time.time(),
                                            }
                                        )

                                    if stop_hit or semantic_stop_ready:
                                        break
                            finally:
                                watchdog.stop_job()
                    finally:
                        if not _restore_surface_generation_controls(surface_control_state):
                            worker_active = False
                            ipc_writer.put(
                                {
                                    "status": "degraded",
                                    "action": "surface_restore_failed",
                                    "message": (
                                        "surface control restore failed; worker will exit "
                                        "after this stream for a clean respawn"
                                    ),
                                }
                            )

                    # One authoritative terminal frame, correlated to the
                    # request: consumers previously saw an id-less ok done
                    # even after a safety abort.
                    ipc_writer.put({
                        "id": job.get("id"),
                        "status": "ok",
                        "action": "stream_done",
                        "aborted": bool(locals().get("sentinel_aborted", False)),
                        "abort_reason": str(locals().get("abort_reason", "") or "")[:200],
                        "tokens_generated": int(locals().get("token_count", 0) or 0),
                        "semantic_completion_contract": bool(
                            job.get("semantic_completion_contract", False)
                        ),
                        "semantic_completion_satisfied": bool(
                            locals().get("semantic_contract_satisfied", False)
                        ),
                        "semantic_completion_incomplete": bool(
                            job.get("semantic_completion_contract", False)
                            and not locals().get("semantic_contract_satisfied", False)
                        ),
                        "prompt_tokenization": {
                            "chars": len(locals().get("_stream_prompt_text", "") or ""),
                            "tokens": int(locals().get("_stream_prompt_tokens", 0) or 0),
                        },
                    })
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_mlx_degradation(
                        e,
                        action="returned stream error and cleared MLX cache after streaming failure",
                        severity="degraded",
                    )
                    logger.error("Streaming failed: %s", e)
                    ipc_writer.put({
                        "id": job.get("id"),
                        "status": "error",
                        "action": "stream",
                        "message": str(e),
                    })
                finally:
                    # [STABILITY v52] Guarantee VRAM gets purged after streaming
                    # completes or fails. The next request starts from clean state.
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)

            elif action == "nonparametric_ingest":
                request_id = str(job.get("id") or "")
                # _safe_int: a malformed seq previously raised BEFORE the
                # handler's error boundary, producing an id-less outer error
                # instead of a correlated ingest failure.
                job_seq = max(0, _safe_int(job.get("seq"), 0))
                response: dict[str, Any] = {
                    "id": request_id,
                    "action": "nonparametric_ingest",
                }
                clear_stale_soft_cancel(cancel_seq, job_seq)
                try:
                    if expert_adapter_state["path"]:
                        response.update(
                            {
                                "status": "error",
                                "message": "expert_adapter_active",
                            }
                        )
                    else:
                        watchdog.start_job(request_id, "nonparametric_ingest")

                        def _ingest_progress(
                            payload: dict[str, Any],
                            *,
                            _request_id: str = request_id,
                        ) -> None:
                            watchdog.activity()
                            # Envelope fields LAST so handler payload cannot
                            # overwrite id/action/status correlation.
                            ipc_writer.put(
                                {
                                    **payload,
                                    "id": _request_id,
                                    "action": "nonparametric_ingest",
                                    "status": "progress",
                                }
                            )

                        with metal_semaphore:
                            result = _run_nonparametric_ingest_job(
                                model,
                                tokenizer,
                                job,
                                cancel_seq=cancel_seq,
                                progress=_ingest_progress,
                            )
                            if mx and device != "cpu":
                                _clear_mlx_cache(mx)
                        response.update({"status": "ok", **result})
                except UnknownActionStateApplicationError as quarantine_exc:
                    recycle_after_response = True
                    _record_mlx_degradation(
                        quarantine_exc,
                        action=(
                            "quarantined MLX worker after ambiguous resident state "
                            "application and forced clean process replacement"
                        ),
                        severity="critical",
                    )
                    response.update(
                        _state_application_quarantine_response(quarantine_exc)
                    )
                except (
                    ImportError,
                    OSError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as ingest_exc:
                    _record_mlx_degradation(
                        ingest_exc,
                        action=(
                            "kept the resident worker available after bounded "
                            "non-parametric ingestion failed"
                        ),
                        severity="warning",
                    )
                    response.update(
                        {
                            "status": "error",
                            "message": (
                                "nonparametric_ingest_failed:"
                                f"{type(ingest_exc).__name__}"
                            ),
                        }
                    )
                finally:
                    watchdog.stop_job()
                    if soft_cancel_requested(cancel_seq, job_seq):
                        try:
                            cancel_seq.value = 0
                        except (AttributeError, OSError, TypeError, ValueError):
                            logger.debug(
                                "Non-parametric ingest soft-cancel acknowledgement failed."
                            )
                ipc_writer.put(response)

            elif action == "encode_hidden":
                # The resident model's own representation of a sentence.
                #
                # A learned decision surface needs a feature space that
                # carries who acts and whether it is asserted. Measured over
                # every declaration in this runtime, a topical sentence
                # embedder carries neither — it is trained to put "I saved it"
                # and "you could save it" close together. This model does
                # carry it, and encoding is one causal forward with no
                # sampling, so there is nothing to steer and no text to write.
                request_id = str(job.get("id") or "")
                texts = [str(item or "") for item in (job.get("texts") or [])][:64]
                response = {"id": request_id, "action": "encode_hidden"}
                try:
                    from core.brain.nonparametric_generation import MLXEncoder

                    if _hidden_encoder["encoder"] is None:
                        _hidden_encoder["encoder"] = MLXEncoder(model, tokenizer)
                    encoder = _hidden_encoder["encoder"]
                    with metal_semaphore:
                        vectors = [
                            [float(value) for value in encoder.encode_hidden(text[:2000])]
                            for text in texts
                            if text.strip()
                        ]
                    response.update({"status": "ok", "vectors": vectors})
                except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
                    response.update({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
                ipc_writer.put(response)

            elif action == "ping":
                if mx and device != "cpu":
                    _clear_mlx_cache(mx)
                # Pong carries identity/steering evidence so the parent can
                # verify WHAT answered, not merely that something did.
                ipc_writer.put(
                    {
                        "id": job.get("id") if isinstance(job, dict) else None,
                        "status": "pong",
                        "model": os.path.basename(str(model_path or "")),
                        "steering_active": bool(_steering_active),
                        "expert_adapter": expert_adapter_state.get("path") or None,
                    }
                )

            elif action == "clear_cache":
                # Clear both Metal GPU cache AND the CPU-side prompt-KV cache.
                # The prompt_cache_lru holds KV states that can become polluted
                # after a stalled generation or a partial token stream — if we
                # only clear Metal, the next request will reuse a corrupt KV
                # state and frequently produces zero tokens (the "Cortex
                # returned no text" cascade).  Clearing both is safe; worst
                # case we pay one prompt-encoding re-run.
                if mx and device != "cpu":
                    _clear_mlx_cache(mx)
                prompt_cache_cleared = True
                prompt_cache_bytes_freed = 0
                try:
                    if prompt_cache_lru is not None:
                        # shed() reports what it released so the OOM ladder can
                        # record a real reclaim instead of an unverified one.
                        prompt_cache_bytes_freed = int(prompt_cache_lru.shed())
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    prompt_cache_cleared = False
                    _record_mlx_degradation(
                        exc,
                        action="continued clear_cache response after prompt cache clear failed",
                        severity="warning",
                    )
                    logger.debug("Prompt cache clear failed during worker clear_cache action: %s", exc)
                # The parent must be able to distinguish complete cache
                # invalidation from a partially stale state.
                ipc_writer.put(
                    {
                        "id": job.get("id") if isinstance(job, dict) else None,
                        "status": "ok",
                        "prompt_cache_cleared": prompt_cache_cleared,
                        "prompt_cache_bytes_freed": prompt_cache_bytes_freed,
                        "prompt_cache_bytes": (
                            int(prompt_cache_lru.retained_bytes())
                            if prompt_cache_lru is not None
                            else 0
                        ),
                    }
                )

            elif action == "set_expert_adapter":
                # Swap a domain-specialist LoRA onto the RESIDENT model —
                # no model reload, ~seconds. path="" means detach-only.
                # KV caches are invalidated either way: cached prompt states
                # were computed under different effective weights.
                requested_path = str(job.get("path") or "").strip()
                response: dict[str, Any] = {
                    "id": job.get("id"),
                    "action": "set_expert_adapter",
                }
                previous_adapter_path = str(expert_adapter_state["path"] or "")
                try:
                    # 1) Validate the NEW adapter before touching resident
                    #    weights — the old order destroyed the current
                    #    identity first and validated nothing, so any attach
                    #    failure silently changed model identity to bare.
                    if requested_path:
                        _validate_expert_adapter_dir(requested_path)
                    blocked = _unrestorable_wrapped(expert_adapter_state["wrapped"])
                    if blocked:
                        raise RuntimeError(
                            "expert_adapter_swap_blocked_unrestorable:"
                            + ",".join(blocked[:4])
                        )
                    with metal_semaphore:
                        detached = 0
                        if expert_adapter_state["wrapped"]:
                            detached = _detach_expert_adapter(
                                model, expert_adapter_state["wrapped"]
                            )
                            expert_adapter_state.update({"path": "", "wrapped": []})
                        if requested_path:
                            try:
                                wrapped = _attach_expert_adapter(model, requested_path)
                                expert_adapter_state.update(
                                    {"path": requested_path, "wrapped": wrapped}
                                )
                            except (
                                FileNotFoundError,
                                RuntimeError,
                                AttributeError,
                                TypeError,
                                ValueError,
                                KeyError,
                                OSError,
                            ) as attach_exc:
                                # 2) Roll back to the PREVIOUS identity
                                #    instead of silently going bare.
                                rollback = "bare_model"
                                if previous_adapter_path:
                                    try:
                                        previous_wrapped = _attach_expert_adapter(
                                            model, previous_adapter_path
                                        )
                                        expert_adapter_state.update(
                                            {
                                                "path": previous_adapter_path,
                                                "wrapped": previous_wrapped,
                                            }
                                        )
                                        rollback = "restored_previous"
                                    except (
                                        FileNotFoundError,
                                        RuntimeError,
                                        AttributeError,
                                        TypeError,
                                        ValueError,
                                        KeyError,
                                        OSError,
                                    ) as rollback_exc:
                                        _record_mlx_degradation(
                                            rollback_exc,
                                            action="fell back to bare model after adapter rollback also failed",
                                            severity="critical",
                                        )
                                response["rollback"] = rollback
                                raise attach_exc
                        # 3) Cache invalidation is PROVEN, not best-effort:
                        #    cached KV states computed under the previous
                        #    weights must not survive an identity change.
                        cache_invalidated = True
                        try:
                            if prompt_cache_lru is not None:
                                prompt_cache_lru.clear()
                        except (RuntimeError, AttributeError, TypeError, ValueError) as clear_exc:
                            logger.warning(
                                "Prompt cache clear failed during adapter swap; rebuilding: %s",
                                clear_exc,
                            )
                            try:
                                prompt_cache_lru = _PromptCacheLRU(
                                    max_size=prompt_cache_budget
                                )
                            except (RuntimeError, TypeError, ValueError):
                                cache_invalidated = False
                        if mx and device != "cpu":
                            _clear_mlx_cache(mx)
                    if not cache_invalidated:
                        # Unverifiable weight/cache consistency: refuse the
                        # swap result and exit for a clean respawn rather
                        # than serving cross-identity KV states.
                        _record_mlx_degradation(
                            RuntimeError("adapter_swap_cache_invalidation_unproven"),
                            action="recycling worker because adapter-swap cache invalidation could not be proven",
                            severity="critical",
                        )
                        worker_active = False
                        response.update(
                            {
                                "status": "error",
                                "message": "adapter_swap_cache_invalidation_unproven; worker recycling",
                                "resident": expert_adapter_state["path"] or None,
                            }
                        )
                    else:
                        # The adapter stack is part of worker identity. Publish
                        # a fresh measurement atomically with the successful
                        # swap so the parent can re-attest this exact serving
                        # function before admitting another request.
                        worker_identity = _current_worker_identity()
                        response.update(
                            {
                                "status": "ok",
                                "resident": expert_adapter_state["path"] or None,
                                "wrapped_layers": len(expert_adapter_state["wrapped"]),
                                "detached_layers": detached,
                                "cache_invalidated": True,
                                "worker_identity": dict(worker_identity),
                            }
                        )
                        logger.info(
                            "🧩 [WORKER] Expert adapter %s (%d layers wrapped, %d restored).",
                            expert_adapter_state["path"] or "DETACHED",
                            len(expert_adapter_state["wrapped"]),
                            detached,
                        )
                except (
                    FileNotFoundError,
                    NotADirectoryError,
                    PermissionError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    OSError,
                ) as adapter_exc:
                    if expert_adapter_state["path"] != previous_adapter_path:
                        # Unwind anything a partial attach recorded; freshly
                        # wrapped-but-unloaded LoRA layers are identity (B=0),
                        # so the model is behaviorally unchanged either way.
                        try:
                            if expert_adapter_state["wrapped"]:
                                _detach_expert_adapter(model, expert_adapter_state["wrapped"])
                        finally:
                            expert_adapter_state.update({"path": "", "wrapped": []})
                    identity_restored = False
                    try:
                        restored_identity = _current_worker_identity()
                        from core.brain.llm.latent_cortex.runtime_identity import (
                            worker_identity_errors,
                        )

                        identity_restored = not worker_identity_errors(
                            restored_identity,
                            expected=worker_identity,
                        )
                    except (
                        ImportError,
                        OSError,
                        RuntimeError,
                        AttributeError,
                        TypeError,
                        ValueError,
                    ) as identity_exc:
                        _record_mlx_degradation(
                            identity_exc,
                            action=(
                                "could not prove worker identity after expert "
                                "adapter swap failure"
                            ),
                            severity="critical",
                        )
                    if not identity_restored:
                        # The mutation failed and the exact pre-swap serving
                        # identity was not recovered. Do not process another
                        # request under the stale parent attestation.
                        worker_active = False
                    _record_mlx_degradation(
                        adapter_exc,
                        action="reported expert adapter swap failure with truthful resident identity",
                        severity=("warning" if identity_restored else "critical"),
                    )
                    response.update(
                        {
                            "status": "error",
                            "message": f"expert_adapter_swap_failed: {adapter_exc}",
                            "resident": expert_adapter_state["path"] or None,
                            "requires_worker_recycle": not identity_restored,
                        }
                    )
                ipc_writer.put(response)

            elif action == "unified_recurrent_shadow_probe":
                request_id = str(job.get("id") or "")
                job_seq = max(0, _safe_int(job.get("seq"), 0))
                response = {
                    "id": request_id,
                    "action": "unified_recurrent_shadow_probe",
                }
                clear_stale_soft_cancel(cancel_seq, job_seq)
                watchdog.start_job(request_id, "unified_recurrent_shadow_probe")
                try:
                    with metal_semaphore:
                        response = _handle_unified_recurrent_shadow_probe(
                            job,
                            loaded_shadow=unified_recurrent_shadow,
                            model=model,
                            contract_key=contract_key,
                            cancel_check=lambda _job_seq=job_seq: soft_cancel_requested(
                                cancel_seq,
                                _job_seq,
                            ),
                            activity=watchdog.activity,
                            reclaim=lambda: _reclaim_unified_recurrent_probe_memory(mx),
                        )
                except InterruptedError:
                    response.update(
                        {
                            "status": "error",
                            "message": "unified_recurrent_shadow_probe_cancelled",
                        }
                    )
                except (
                    ImportError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    OSError,
                ) as probe_exc:
                    _record_mlx_degradation(
                        probe_exc,
                        action="reported unified recurrent shadow probe failure to parent IPC",
                        severity="warning",
                    )
                    response.update(
                        {
                            "status": "error",
                            "message": (
                                "unified_recurrent_shadow_probe_failed:"
                                f"{type(probe_exc).__name__}:{str(probe_exc)[:180]}"
                            ),
                        }
                    )
                finally:
                    watchdog.stop_job()
                    if soft_cancel_requested(cancel_seq, job_seq):
                        try:
                            cancel_seq.value = 0
                        except (AttributeError, OSError, TypeError, ValueError):
                            logger.debug("Shadow probe soft-cancel acknowledgement failed.")
                ipc_writer.put(response)

            elif action == "unified_recurrent_qualified_decode":
                request_id = str(job.get("id") or "")
                job_seq = max(0, _safe_int(job.get("seq"), 0))
                response = {
                    "id": request_id,
                    "action": "unified_recurrent_qualified_decode",
                }
                clear_stale_soft_cancel(cancel_seq, job_seq)
                watchdog.start_job(request_id, "unified_recurrent_qualified_decode")
                try:
                    with metal_semaphore:
                        response = _handle_unified_recurrent_qualified_decode(
                            job,
                            loaded_shadow=unified_recurrent_shadow,
                            qualified_activation=(
                                unified_recurrent_qualified_activation
                            ),
                            model=model,
                            contract_key=contract_key,
                            consumed_canary_nonces=(
                                consumed_unified_recurrent_canary_nonces
                            ),
                            cancel_check=lambda _job_seq=job_seq: soft_cancel_requested(
                                cancel_seq,
                                _job_seq,
                            ),
                            activity=watchdog.activity,
                            reclaim=lambda: _reclaim_unified_recurrent_probe_memory(mx),
                        )
                except InterruptedError:
                    response.update(
                        {
                            "status": "error",
                            "message": "unified_recurrent_qualified_decode_cancelled",
                        }
                    )
                except (
                    ImportError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    OSError,
                ) as decode_exc:
                    _record_mlx_degradation(
                        decode_exc,
                        action=(
                            "reported unified recurrent qualified decode failure "
                            "to parent IPC"
                        ),
                        severity="warning",
                    )
                    response.update(
                        {
                            "status": "error",
                            "message": (
                                "unified_recurrent_qualified_decode_failed:"
                                f"{type(decode_exc).__name__}:"
                                f"{str(decode_exc)[:180]}"
                            ),
                        }
                    )
                finally:
                    watchdog.stop_job()
                    if soft_cancel_requested(cancel_seq, job_seq):
                        try:
                            cancel_seq.value = 0
                        except (AttributeError, OSError, TypeError, ValueError):
                            logger.debug(
                                "Qualified decode soft-cancel acknowledgement failed."
                            )
                ipc_writer.put(response)

            elif action == "latent_reason":
                # Recursive Latent Cortex episode on the RESIDENT model —
                # workspace recurrence + branches (+ optional latent opt /
                # episode fast weights), all under checkpoint invariants.
                # See docs/RECURSIVE_LATENT_CORTEX.md.
                request_id = str(job.get("id") or "")
                job_seq = max(0, _safe_int(job.get("seq"), 0))
                response = {"id": request_id, "action": "latent_reason"}
                recycle_after_response = False
                clear_stale_soft_cancel(cancel_seq, job_seq)
                watchdog.start_job(request_id, "latent_reason")

                def _latent_progress(
                    payload: dict[str, Any],
                    *,
                    _request_id: str = request_id,
                ) -> None:
                    watchdog.activity()
                    # Envelope fields LAST: handler payload spread after them
                    # could overwrite the correlation id, action, and status
                    # in the emitted IPC frame.
                    ipc_writer.put(
                        {
                            **dict(payload),
                            "id": _request_id,
                            "action": "latent_reason",
                            "status": "progress",
                        }
                    )

                try:
                    from core.brain.llm.latent_cortex.worker_handler import (
                        handle_latent_reason,
                    )

                    with metal_semaphore:
                        surface_control_state = _apply_surface_generation_controls(
                            engine,
                            model,
                            job,
                        )
                        try:
                            applied_alpha = surface_control_state.get(
                                "surface_alpha_applied"
                            )
                            if job.get("runtime_controls") is not None and (
                                isinstance(applied_alpha, bool)
                                or not isinstance(applied_alpha, (int, float))
                                # A numeric alpha outside the surface-control
                                # admission range proves the clamp did NOT
                                # produce this value — reject, don't trust.
                                or not (0.0 <= float(applied_alpha) <= 1.0)
                                or surface_control_state.get("apply_errors")
                            ):
                                body = {
                                    "status": "error",
                                    "message": "latent_runtime_controls_unapplied",
                                }
                            else:
                                body = handle_latent_reason(
                                    job,
                                    model=model,
                                    tokenizer=tokenizer,
                                    model_path=model_path,
                                    worker_identity=worker_identity,
                                    worker_capture_signing_identity=(
                                        worker_capture_signing_identity
                                    ),
                                    worker_capture_launch_challenge=(
                                        worker_capture_launch_challenge
                                    ),
                                    surface_control_state=surface_control_state,
                                    cancel_check=lambda _job_seq=job_seq: soft_cancel_requested(
                                        cancel_seq,
                                        _job_seq,
                                    ),
                                    progress=_latent_progress,
                                )
                        finally:
                            if not _restore_surface_generation_controls(
                                surface_control_state
                            ):
                                worker_active = False
                                ipc_writer.put(
                                    {
                                        "status": "degraded",
                                        "action": "surface_restore_failed",
                                        "message": (
                                            "surface control restore failed; worker will "
                                            "exit after this latent episode for a clean "
                                            "respawn"
                                        ),
                                    }
                                )
                        recycle_after_response = body.pop("requires_worker_recycle", False)
                        if body.pop("requires_cache_clear", False):
                            # Fast-weight erase unproven ⇒ pre-episode prompt
                            # KV states may embed the temporary weights.
                            try:
                                if prompt_cache_lru is not None:
                                    prompt_cache_lru.clear()
                            except (RuntimeError, AttributeError, TypeError, ValueError):
                                logger.debug("Prompt cache clear skipped after latent episode.")
                            if mx and device != "cpu":
                                _clear_mlx_cache(mx)
                    response.update(body)
                except UnknownActionStateApplicationError as quarantine_exc:
                    recycle_after_response = True
                    _record_mlx_degradation(
                        quarantine_exc,
                        action=(
                            "quarantined MLX worker after ambiguous RLC action-state "
                            "application and forced clean process replacement"
                        ),
                        severity="critical",
                    )
                    response.update(
                        _state_application_quarantine_response(quarantine_exc)
                    )
                except (
                    ImportError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    OSError,
                ) as latent_exc:
                    _record_mlx_degradation(
                        latent_exc,
                        action="reported latent_reason failure to parent IPC",
                        severity="warning",
                    )
                    response.update(
                        {"status": "error", "message": f"latent_reason_failed: {latent_exc}"}
                    )
                finally:
                    watchdog.stop_job()
                    if soft_cancel_requested(cancel_seq, job_seq):
                        try:
                            cancel_seq.value = 0
                        except (AttributeError, OSError, TypeError, ValueError):
                            logger.debug(
                                "Latent episode soft-cancel acknowledgement failed."
                            )
                ipc_writer.put(response)
                if recycle_after_response:
                    logger.critical(
                        "Latent episode could not prove resident-model integrity; "
                        "exiting worker after the error response for clean reload."
                    )
                    break

            else:
                # Unknown or version-skewed actions previously vanished
                # silently, leaving the parent future unresolved until its
                # deadline. Every consumed job gets a typed terminal answer.
                ipc_writer.put(
                    {
                        "id": job.get("id") if isinstance(job, dict) else None,
                        "status": "error",
                        "action": str(action or "unknown")[:64],
                        "message": f"unknown_worker_action:{str(action or '')[:64]}",
                    }
                )

        except KeyboardInterrupt:
            logger.info("🛑 [WORKER] Shutdown signal received; exiting quietly.")
            break
        except (RuntimeError, TypeError, ValueError, OSError, AttributeError) as e:
            _record_mlx_degradation(
                e,
                action="reported worker action error to parent IPC and continued request loop",
                severity="degraded",
            )
            import traceback
            tb = traceback.format_exc()
            resolved_action = locals().get("action") or "unknown"
            logger.error(
                "❌ [WORKER] Unhandled error during '%s': %s\n%s",
                resolved_action, e, tb,
            )
            # Correlate the failure (the parent cannot resolve an id-less
            # error) and keep the full traceback in worker logs only — raw
            # internal paths do not belong in the IPC payload the parent may
            # surface into telemetry or metadata.
            resolved_job = locals().get("job")
            resolved_id = (
                str(resolved_job.get("id") or "")
                if isinstance(resolved_job, dict)
                else ""
            )
            ipc_writer.put(
                {
                    "id": resolved_id,
                    "status": "error",
                    "action": resolved_action,
                    "message": (
                        f"{resolved_action} failed: {type(e).__name__}: {str(e)[:240]}"
                    ),
                }
            )

    _shutdown_worker_runtime(
        ipc_writer=ipc_writer,
        watchdog=watchdog,
        heartbeat=heartbeat,
        memory_sentinel=memory_sentinel,
        latent_bridge=latent_bridge,
        steering_engine=engine,
        prompt_cache_lru=prompt_cache_lru,
        mx_module=mx,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logger.info("MLX Worker: Running in multiprocessing mode. Use mlx_client.py to launch.")
