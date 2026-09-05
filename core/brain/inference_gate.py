"""InferenceGate: unified MLX-managed local inference gateway.

Provides a single interface for all LLM inference needs.
Strategy:
  1. Try Aura's managed MLX runtime (32B Cortex primary lane)
  2. If the primary lane fails, recover through registered local lanes
  3. If every local lane fails, return a typed local exhaustion result

This module is the FAST PATH for user-facing chat. It injects Aura's full
identity/personality system prompt so responses sound like Aura, not a bare LLM.
Timeouts are kept tight (45s) for conversational responsiveness.
"""

import asyncio
import copy
import gc
import hashlib
import inspect
import logging
import math
import os
import re
import threading as _threading
import time
import weakref
import uuid
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from core.brain.live_mind_contract import append_text_mutation
from core.brain.living_mind_context import (
    PRIORITY_COLOUR,
    PRIORITY_GATING,
    TRUST_LEARNED,
    LivingMindContext,
    estimate_context_tokens,
)
from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
    get_active_cortex_serving_limits,
    lane_display_label,
)
from core.brain.request_contract import REQUEST_FIELDS, validate_request_context
from core.conversation.response_reliability import (
    assess_model_text_integrity,
    assess_user_facing_reply,
    conversation_reliability_system_block,
    has_requested_word_count_contract,
    is_live_self_reflection_turn,
    is_self_process_question,
    requested_output_contract,
)
from core.conversation.surface_disposition import COMPLETION_REASONS
from core.conversation.user_surface_contract import (
    bind_user_surface_prompt,
    resolve_user_surface_prompt,
)
from core.epistemics.opinion_engine import standing_disposition
from core.runtime import resource_psutil as psutil
from core.runtime.desktop_boot_safety import (
    desktop_resource_guard_enabled,
    desktop_safe_boot_enabled,
)
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind as _FlagKind
from core.runtime.flags import declare as _declare_flag
from core.runtime.flags import env_str
from core.runtime.lockdep import LockRank, checked_async_lock, checked_lock
from core.runtime.process_identity import assert_owned, capture_identity
from core.runtime.proof_policy import (
    is_proof_evaluation_purpose,
    is_strict_proof_answer_prompt,
    mlx_strict_answer_contract_enabled,
    proof_model_tier,
    proof_run_active,
)
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.structured_input import (
    analyze_prompt_shape,
    answer_surface_token_floor,
)
from core.utils.completed_capability import remaining_capabilities
from core.utils.deadlines import Deadline, get_deadline
from core.utils.task_tracker import get_task_tracker

#: Returned by an extracted block that did NOT return early. A unique
#: object, so no value a block legitimately returns can be mistaken for it.
_SEAM_FELL_THROUGH = object()

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
_FLAG_BOOT_WARMUP_MIN_TOTAL_GB = _declare_flag(
    "AURA_BOOT_WARMUP_MIN_TOTAL_GB",
    kind=_FlagKind.STRING,
    default="48",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_CORTEX_CTX = _declare_flag(
    "AURA_CORTEX_CTX",
    kind=_FlagKind.STRING,
    default="16384",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DEEP_PROBE_MAX_TOKENS = _declare_flag(
    "AURA_DEEP_PROBE_MAX_TOKENS",
    kind=_FlagKind.STRING,
    default="384",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DEFERRED_CORTEX_PREWARM = _declare_flag(
    "AURA_DEFERRED_CORTEX_PREWARM",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_EAGER_CORTEX_WARMUP = _declare_flag(
    "AURA_EAGER_CORTEX_WARMUP",
    kind=_FlagKind.STRING,
    default="auto",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_EMBODIED_CHALLENGE = _declare_flag(
    "AURA_EMBODIED_CHALLENGE",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM = _declare_flag(
    "AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_ENABLE_LOCAL_DEEP_SOLVER = _declare_flag(
    "AURA_ENABLE_LOCAL_DEEP_SOLVER",
    kind=_FlagKind.STRING,
    default="auto",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE = _declare_flag(
    "AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE = _declare_flag(
    "AURA_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_HEALTH_WARM_LOCAL_TIERS = _declare_flag(
    "AURA_HEALTH_WARM_LOCAL_TIERS",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_INFERENCE_ACTIVE_PROGRESS_STALE_S = _declare_flag(
    "AURA_INFERENCE_ACTIVE_PROGRESS_STALE_S",
    kind=_FlagKind.STRING,
    default="45",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_INFERENCE_ACTIVE_STARTUP_GRACE_S = _declare_flag(
    "AURA_INFERENCE_ACTIVE_STARTUP_GRACE_S",
    kind=_FlagKind.STRING,
    default="120",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LOCAL_DEEP_AUTO_MIN_TOTAL_GB = _declare_flag(
    "AURA_LOCAL_DEEP_AUTO_MIN_TOTAL_GB",
    kind=_FlagKind.STRING,
    default="96",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LOCAL_RECYCLE_MAX_UPTIME_S = _declare_flag(
    "AURA_LOCAL_RECYCLE_MAX_UPTIME_S",
    kind=_FlagKind.STRING,
    default="5400",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LOCAL_RECYCLE_MIN_IDLE_S = _declare_flag(
    "AURA_LOCAL_RECYCLE_MIN_IDLE_S",
    kind=_FlagKind.STRING,
    default="900",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_SAFE_BOOT_BACKGROUND_GUARD_SECS = _declare_flag(
    "AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS",
    kind=_FlagKind.STRING,
    default="180",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)


from core.brain.llm.context_budget import (
    CRITICAL_FOREGROUND_HEADERS as _CRITICAL_FOREGROUND_HEADERS,
    FOREGROUND_SECTION_VOLATILITY,
)

logger = logging.getLogger("Aura.InferenceGate")
_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT = 0.0


def _primary_lane_label() -> str:
    """The resident cortex's signed label, for operator-facing log lines.

    The line naming the protected lane carried a literal parameter count, so a
    log read during a migration named the checkpoint that had been replaced.
    Degrades to the endpoint name; a log line is never worth an exception.
    """
    try:
        from core.brain.llm.model_registry import resident_model_label

        return resident_model_label(default="Cortex")
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        # Named types rather than Exception: this module forbids broad catches
        # outright, and _INFERENCE_RECOVERABLE_ERRORS is defined further down
        # the file than this helper runs.
        _record_inference_degradation(
            exc,
            action="named the resident lane Cortex because the registry could not",
            severity="info",
        )
        return "Cortex"


_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON = ""
_EXPLICIT_DEFERRED_PREWARM_REFUSAL_LOG_INTERVAL_S = 60.0

#: Lane failures that are TRANSIENT and must be re-armed rather than left
#: terminal. A refused worker spawn is the clearest case: the runtime declined
#: to load the 32B because the host was momentarily short of headroom, and host
#: memory frees constantly. Parking the lane in `failed` over it meant she
#: reported a broken mind for a condition that had already passed — live
#: 2026-07-26, `memory_pressure_refused_worker_spawn:model_load_headroom:23.3GB
#: < required 24.0GB`, short by 0.7GB, and the lane never retried on its own.
_REARMABLE_LANE_FAILURE_PREFIXES = (
    "mlx_runtime_unavailable",
    "local_runtime_unavailable",
    "memory_pressure_refused_worker_spawn",
)


_LONG_FORM_REQUEST_RE = re.compile(
    r"\b(?:"
    r"\d{3,5}\s*(?:-|to)?\s*(?:word|token)s?|"
    r"comprehensive|detailed|fully|in[- ]depth|long[- ]form|"
    r"step[- ]by[- ]step|thorough|every part|essay|report"
    r")\b",
    re.IGNORECASE,
)
_FOREGROUND_ACTION_VERB_RE = re.compile(
    r"\b(?:open|launch|create|write|type|search|find|summari[sz]e|export|save|"
    r"attach|insert|navigate|click|copy|paste|move|rename|download|upload|run|"
    r"test|debug|commit|push|install|read|compare)\b",
    re.IGNORECASE,
)
_FOREGROUND_ACTION_SURFACE_RE = re.compile(
    r"\b(?:desktop|screen|window|app|application|browser|chrome|tab|web|article|"
    r"document|docs?|notes?|folder|file|pdf|terminal|shell|clipboard|tool|tools)\b",
    re.IGNORECASE,
)
_FOREGROUND_ACTION_SEQUENCE_RE = re.compile(
    r"\b(?:then|next|after(?:ward| that)?|finally|before|while|all in one|"
    r"multi[- ]step|step\s*\d+)\b|[,;]",
    re.IGNORECASE,
)

_STATE_SIGNAL_REWRITES = (
    ("phenomenological", "state-grounded"),
    ("Phenomenological", "State-grounded"),
    ("phenomenology", "state telemetry"),
    ("Phenomenology", "State telemetry"),
    ("phenomenal", "functional-state"),
    ("Phenomenal", "Functional-state"),
    ("qualia", "private-state evidence"),
    ("Qualia", "Private-state evidence"),
    ("inner monologue", "state report"),
    ("Inner monologue", "State report"),
)


def _worker_process_started_at(client: Any) -> float:
    """When the worker process was created, or 0.0 when that cannot be read.

    Ground truth for "is this worker new". Lane bookkeeping is written when
    warmup begins and is absent for the window between spawn and that write;
    a creation time cannot be absent for a process that exists.
    """
    process = getattr(client, "_process", None)
    pid = getattr(process, "pid", None)
    if not pid:
        return 0.0
    try:
        from core.runtime.process_identity import _create_time  # noqa: PLC0415

        return float(_create_time(int(pid)) or 0.0)
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return 0.0


def _worker_process_is_running(proc: Any) -> bool:
    """True when a worker process handle exists AND is still running.

    Accepts multiprocessing.Process (is_alive), subprocess.Popen (poll), or
    None. proc is None when the worker was never spawned or is already
    reaped — nothing to kill, and poking it used to raise AttributeError
    mid-recovery (seen live as recent_inference_gate_critical
    "'NoneType' object has no attribute 'poll'").
    """
    if proc is None:
        return False
    try:
        if hasattr(proc, "is_alive"):
            return bool(proc.is_alive())
        if hasattr(proc, "poll"):
            return proc.poll() is None
    except (OSError, ValueError):
        return False
    return False


def _grounded_state_signal_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    for source, replacement in _STATE_SIGNAL_REWRITES:
        text = text.replace(source, replacement)
    return text[:limit]

#: Arguments every ``_generate_with_client`` call passes by name. They must
#: never also appear inside the ``**morpho_kwargs`` splat beside them.
_GENERATE_EXPLICIT_KWARGS: frozenset[str] = frozenset(
    {"messages", "max_tokens", "temperature", "origin", "is_background", "foreground_request"}
)

#: Health windows are read from operator flags, so both ends need a bound.
#: Only the minimum was clamped, which let an operator set a startup grace of
#: 1e9 seconds — or ``inf`` — and keep a permanently stalled generation
#: classified as operational forever. Ten minutes is longer than any real
#: foreground turn on this hardware and short enough that a wedge surfaces.
_MAX_HEALTH_WINDOW_S = 600.0

#: A timestamp from another process can sit slightly ahead of ours. Beyond
#: this, "the future" is a broken clock or a corrupt status payload, not skew,
#: and it must not read as fresh progress.
_HEALTH_CLOCK_SKEW_TOLERANCE_S = 2.0


def _finite(value: Any, default: float | None = None) -> float | None:
    """Coerce to a real, finite float or give back ``default``.

    ``float("nan")`` and ``float("inf")`` survive a bare ``float()`` and then
    poison every comparison they touch: ``max(0.0, now - nan)`` is ``nan``,
    and ``nan <= stale_window`` is False in one place and True nowhere, so a
    generation with a NaN timestamp could be certified as progressing while
    nothing was progressing. Anything not finite is missing evidence here,
    never a value.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _health_window_s(raw: Any, *, default: float, minimum: float) -> float:
    """Clamp an operator-supplied health window at BOTH ends."""
    window = _finite(raw, default)
    if window is None:
        window = default
    return max(minimum, min(_MAX_HEALTH_WINDOW_S, window))


def _elapsed_since(timestamp: Any, *, now: float) -> float | None:
    """Age of ``timestamp`` in seconds, or None when it proves nothing.

    None means "no trustworthy evidence": absent, non-finite, non-positive, or
    far enough in the future that the clock or the payload is wrong. Callers
    must treat None as unproven rather than as age zero — reading a future
    timestamp as a fresh zero age was how a stalled lane stayed healthy.
    """
    stamp = _finite(timestamp)
    if stamp is None or stamp <= 0.0:
        return None
    age = now - stamp
    if age < -_HEALTH_CLOCK_SKEW_TOLERANCE_S:
        return None
    return max(0.0, age)


def _reflex_model_status(raw_path: str) -> tuple[str, str]:
    """Say whether the reflex model can actually be loaded, not whether a path exists.

    ``available`` used to mean ``Path(...).exists()``. A directory with the
    right name, a zero-byte file left by an interrupted download, or a file
    the process cannot read all passed that test, so the last tier — the one
    that answers when Cortex and brainstem are both gone — reported itself
    available right up to the moment it was needed.

    Runs on a thread: every call here touches the filesystem.
    """
    if not raw_path:
        return "model_missing", "no_path_configured"
    path = Path(raw_path)
    try:
        stat = path.stat()
    except OSError:
        return "model_missing", "path_absent"
    if path.is_dir():
        # An mlx model directory is a real layout; require the weights in it.
        weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.npz"))
        if not weights:
            return "model_unreadable", "directory_without_weights"
        readable = any(os.access(str(candidate), os.R_OK) for candidate in weights)
        return ("available", "weights_readable") if readable else (
            "model_unreadable",
            "weights_not_readable",
        )
    if not path.is_file():
        return "model_unreadable", "not_a_regular_file"
    if stat.st_size <= 0:
        # The signature of an interrupted download.
        return "model_unreadable", "empty_file"
    if not os.access(raw_path, os.R_OK):
        return "model_unreadable", "not_readable"
    return "available", "file_readable"


#: Foreground prompt budget when nothing better is readable. Below the floor
#: the compactor strips the system prompt; above the registry's ceiling the
#: serving runtime silently truncates whatever we sent.
#: What a denied viability envelope allows. Long-standing policy; named here
#: so the override below has something to be an override OF.
_STAKES_DENIED_TOKEN_CAP = 128

#: Longest a foreground turn may block on Cortex warmup. A cold 32B load is
#: ~150s and the cold-boot budget is 180s; beyond ten minutes the person has
#: gone, and the lane is better served by the fallback tier.
_MAX_FOREGROUND_READY_WAIT_S = 600.0

_FOREGROUND_CONTEXT_WINDOW_DEFAULT = 16384
_FOREGROUND_CONTEXT_WINDOW_FLOOR = 4096

#: The sampling advisories Aura's own cognition produces. Anything under one
#: of these keys must be present in ``state.response_modifiers`` — where the
#: cognitive engine writes it — before it may move temperature or the token
#: budget. A dict that appears only in the caller's context is a caller
#: steering the sampler, which is what "not caller authority" excludes.
#: Mesh outcomes that may be served before trust recognition runs. An
#: acknowledgement reveals nothing and a resource hold is a refusal; a
#: self-report describes her internal state to a party this path has not yet
#: identified, so it goes through the governed lane instead.
_MESH_PRE_TRUST_RATIONALES = frozenset({"acknowledgement", "resource_hold"})

#: Every channel by which a subsystem moves the sampler. A subsystem that
#: publishes a bias and is not named here has no reader: the gate filters
#: kwargs to the declared request fields and logs the rest at debug, so the
#: bias is computed on every turn and dropped in silence.
#:
#: LIVE, 2026-08-28: the cognitive-situation frame was the fourth such channel
#: and was in neither this tuple nor the request schema. Its test proved the
#: engine handed the bias to a stub router and stopped there, which is the
#: shape of a half-wired channel — a writer, a test of the writer, and no
#: reader.
#: The two ceilings a request can establish for itself, read from the one place
#: that decides them rather than spelled again here.
try:  # pragma: no cover - import shape only
    from core.phases.response_contract import (
        _REQUESTED_ARTIFACT_CEILING as _REQUESTED_ARTIFACT_EFFECT_CEILING,
        _SELF_SERVICE_CEILING as _SELF_SERVICE_EFFECT_CEILING,
    )
except ImportError:  # pragma: no cover - the gate still runs without them
    _SELF_SERVICE_EFFECT_CEILING = "sandboxed_compute"
    _REQUESTED_ARTIFACT_EFFECT_CEILING = "read_write_artifacts"

_SAMPLING_BIAS_KEYS = (
    "sampling_bias",
    "imagination_sampling_bias",
    "bicameral_sampling_bias",
    "cognitive_situation_sampling_bias",
)

_INFERENCE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    psutil.Error,
)

_ACTIVE_GENERATION_BUSY_REASONS = frozenset(
    {
        "active_generation_in_flight",
        "foreground_generation_active",
        "foreground_owner_active",
        "warmup_foreground_owner",
    }
)


def _record_inference_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record an inference-gate degradation.

    `extra` was missing, so this wrapper silently discarded the structured
    context every caller in the codebase attaches — the correlation id and
    stage name that make a partial post-inference update joinable are
    exactly that kind of context, and passing them would have raised
    TypeError at the moment a stage actually failed.
    """
    record_degradation(
        "inference_gate",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


_DOWNSTREAM_REPAIRABLE_SELF_REFLECTION_REASONS = frozenset(
    {
        "missing_requested_self_process_coverage",
        "off_topic_self_reflection_reply",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
    }
)
_DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS = frozenset(
    {
        # Only surface/style defects belong here. Thin, evasive, or confused
        # drafts need another generation attempt because downstream repair
        # cannot safely invent the missing answer — "I don't know what caused
        # that timeout yet" clears any length floor and is still a non-answer.
        # The thinness FALSE POSITIVES are fixed where they are produced (see
        # _has_reliability_diagnostic_substance), not by widening this set.
        "off_topic_self_reflection_reply",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "generic_assistant_language",
        "persona_card_deflection",
        "detail_request_deflection",
        "truncated_tail",
        "vague_status_derailment",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
        "missing_requested_self_process_coverage",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "missing_requested_bare_answer",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_followup_question",
        # A worked derivation that stopped one step short of stating its
        # answer. It is real content — the cases are computed correctly — and
        # downstream repair can finish it or the person can read it as it
        # stands. Discarding it costs a correct three-case derivation and
        # returns an apology, which is what happened live on 2026-07-26:
        #   Cortex produced an unsafe user-facing draft
        #   (final_answer_missing, len=899). Treating it as failed generation.
        "final_answer_missing",
    }
)


#: Verdicts that say only "this reply is shorter than we wanted". None of
#: them is an integrity, honesty, or safety finding, and none is evidence
#: that the answer is wrong — "68" is the whole correct answer to "what's 17
#: times 4?".
_THINNESS_ONLY_REASONS = frozenset(
    {
        "too_short_for_user_turn",
        "too_thin_for_user_turn",
        "too_thin_for_open_ended_turn",
        "too_thin_for_status_turn",
        "too_thin_for_operational_status_turn",
        "too_thin_for_expansion_request",
        "too_thin_for_reliability_turn",
        "reliability_diagnostic_too_thin",
    }
)


def _should_pass_user_facing_draft_downstream(
    text: str,
    reasons: set[str],
    *,
    user_prompt: str,
    allow_memory_state_thin_status: bool = False,
) -> bool:
    """Keep salvageable chat drafts out of the expensive retry spiral."""
    if not text or not reasons:
        return False
    repairable_reasons = set(_DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS)
    if allow_memory_state_thin_status:
        repairable_reasons.update(
            {
                "too_thin_for_operational_status_turn",
                "too_thin_for_status_turn",
            }
        )
    # The shared policy is the floor: anything that is a SHORTFALL rather than
    # an integrity failure may go downstream to be repaired, whatever this
    # module's own historical list happens to contain. See
    # core/conversation/surface_disposition.py.
    from core.conversation.surface_disposition import draft_is_servable

    if not reasons.issubset(repairable_reasons) and not draft_is_servable(reasons):
        return False
    stripped = str(text or "").strip()
    if reasons == {"missing_requested_word_count"}:
        return has_requested_word_count_contract(user_prompt) and bool(stripped)
    # A draft whose ONLY fault is being short must not be failed for being
    # short. The floors below exist to stop a stub entering repair; applied
    # to a thinness verdict they are circular, and the circle is closed
    # against every question whose correct answer is brief.
    #
    # Measured live 2026-08-04. Bryan asked "what's 17 times 4?" and the
    # Cortex answered "68" — len=3. It was rejected as
    # too_short_for_user_turn, retried, rejected again, and the turn died
    # as "compact desktop generation returned no usable text". He was told
    # "I couldn't get to an answer I'd stand behind" about arithmetic she
    # had already got right. The runtime's own degradation record named it:
    # "turn ended holding a servable answer that was never shown".
    #
    # Length is not evidence of inadequacy; unresponsiveness is, and the
    # assessor reports THAT separately (reply_abandons_thread and friends).
    # When thinness is the only complaint there is no evidence of a bad
    # answer, so it goes downstream to repair rather than into the bin.
    from core.conversation.surface_disposition import (
        short_draft_answers_closed_question,
    )

    if (
        stripped
        and reasons.issubset(_THINNESS_ONLY_REASONS)
        and short_draft_answers_closed_question(stripped, user_prompt)
    ):
        return True
    # The floors below exist to stop a placeholder reply entering repair. They
    # cannot apply to a reply whose shortness the person asked for.
    #
    # Measured live 2026-08-18. Told "don't acknowledge that rule beyond a yes.
    # just hold it", the Cortex answered "Yes." — len=4, precisely what was
    # requested. Reasons came back as too_short_for_user_turn plus
    # missing_requested_phrase, which is not a subset of the thinness set, so
    # the exemption above could not fire and the 48-character floor discarded
    # it. Retry produced "Yes." again, was discarded again, and the person was
    # told "I couldn't get to an answer I'd stand behind" — about an
    # instruction she had followed exactly, twice.
    #
    # The worker already reaches this conclusion for the same reasons on the
    # same drafts: missing_requested_phrase sits in its deliverable-residual
    # set precisely because a dead turn is worse than an imperfectly styled
    # one. This is the gate agreeing with it.
    from core.conversation.surface_disposition import requests_a_brief_answer

    if stripped and requests_a_brief_answer(user_prompt):
        return True
    if len(stripped) < 48:
        return False
    words = [token for token in stripped.replace("\n", " ").split(" ") if token.strip()]
    if len(words) < 8:
        return False
    if reasons & _DOWNSTREAM_REPAIRABLE_SELF_REFLECTION_REASONS:
        return is_live_self_reflection_turn(user_prompt) or is_self_process_question(user_prompt)
    return True


#: A leading word that says this is NOT a person waiting. It vetoes, so an
#: allowlisted word later in the label cannot promote the request.
_NOT_USER_FACING_ORIGIN_PREFIXES = frozenset(
    {
        "background",
        "internal",
        "system",
        "auto",
        "autonomous",
        "cron",
        "scheduled",
        "daemon",
        "maintenance",
        "sweep",
    }
)

_USER_FACING_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "desktop",
        "desktop-ui",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "native-shell",
        "audit",
        "simulate",
        "embodied_motor_reflex",
        "embodied",
        "reflex",
        "test",
    }
)


@asynccontextmanager
async def _thread_lock_context(
    lock: Any,
    *,
    timeout_s: float | None = None,
    label: str = "lock",
):
    deadline = (
        None
        if timeout_s is None
        else time.monotonic() + max(0.0, float(timeout_s))
    )
    while not lock.acquire(blocking=False):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"{label}_timeout")
            await asyncio.sleep(min(0.01, remaining))
        else:
            await asyncio.sleep(0.01)
    try:
        yield
    finally:
        try:
            lock.release()
        except RuntimeError:
            logger.debug("Foreground-ready lock %s was already released.", label)


#: Identity and freshness contract for a memory-admission snapshot.
#:
#: CP126 (high): "Missing admission fields are interpreted as permission.
#: Foreground admission results are consumed as dictionaries and can_admit
#: defaults to True when absent. An incomplete or stale admission receipt
#: therefore authorizes the expensive lane rather than failing closed or
#: requiring a validated schema."
#:
#: The absent-field half is already closed at the consumption sites, which
#: read `.get("can_admit", False)`. What remained is the other two words in
#: that finding — INCOMPLETE and STALE. A bare dict carrying
#: `{"can_admit": True}` and nothing else was indistinguishable from a full
#: measurement, and a snapshot taken minutes ago was indistinguishable from
#: one taken now. Both authorize a ~20GB model load on evidence that may no
#: longer be true.
#:
#: Stamping identity and measurement time makes both answerable.
#: The nine downstream systems `_post_inference_update` advances, in order.
#: Named once so the commit receipt can say which of them did not run.
_POST_INFERENCE_STAGES = (
    "crsm_self_state",
    "hot_reflexive_feedback",
    "hedonic_and_lora",
    "credit_assignment",
    "homeostasis",
    "world_model",
    "synaptic_plasticity",
    "temporal_continuity",
)

ADMISSION_SNAPSHOT_SCHEMA = "aura.memory_admission_snapshot.v1"

#: How long a memory measurement may authorize a heavy lane. Memory pressure
#: on this host moves in seconds under load, so an older reading is history
#: rather than evidence.
ADMISSION_SNAPSHOT_MAX_AGE_S = 30.0

#: Keyword arguments ``think()`` consumes rather than forwards. Listed so an
#: undeclared-kwarg report does not cry wolf about them.
_THINK_LOCAL_KWARGS = frozenset({"timeout", "brief", "system_prompt_is_brief"})

_REQUIRED_ADMISSION_FIELDS = (
    "can_admit",
    "measured",
    "pressure_pct",
    "available_gb",
    "tier",
)


def _deep_snapshot(value: Any) -> Any:
    """A copy nothing else holds a reference into.

    ``copy.deepcopy`` on provider metadata can meet a live client handle, a
    lock or a coroutine — objects that either refuse to copy or are actively
    harmful to duplicate. So this copies the containers (which is where the
    aliasing lives) and leaves leaves alone, replacing anything uncopyable
    with its repr rather than raising inside an evidence path.
    """
    return _deep_snapshot_inner(value, depth=0, seen=set())


def _deep_snapshot_inner(value: Any, *, depth: int, seen: set[int]) -> Any:
    if depth > 12:
        return "<snapshot depth limit>"
    if isinstance(value, dict):
        if id(value) in seen:
            return "<cycle>"
        seen = seen | {id(value)}
        # list(...) FIRST. This walks live provider metadata while other
        # threads are still writing to it, and a Python-level comprehension
        # over .items() can be interrupted between elements — "dictionary
        # changed size during iteration", raised inside an evidence path, in
        # a subsystem on the fail-closed list, so it escalated to CRITICAL and
        # held the runtime DEGRADED across health pulses (live 2026-08-03
        # 22:13, repeating). The C-level copy cannot be interrupted.
        return {
            key: _deep_snapshot_inner(item, depth=depth + 1, seen=seen)
            for key, item in list(value.items())
        }
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            return "<cycle>"
        seen = seen | {id(value)}
        return [
            _deep_snapshot_inner(item, depth=depth + 1, seen=seen)
            for item in list(value)
        ]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_deep_snapshot_inner(item, depth=depth + 1, seen=seen) for item in value),
            key=repr,
        )
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    # A client handle, a lock, a coroutine — record what it was, not the
    # object itself.
    return repr(value)


def snapshot_metric(snapshot: Any, key: str) -> float | None:
    """A numeric field from an admission snapshot, or None if unmeasured.

    CP126: "On memory probe failure the snapshot reports pressure, available
    memory, and total memory as numeric zeros with permissive thresholds.
    Although reason may say memory_probe_failed, consumers cannot
    distinguish unmeasured fields from genuine measurements."

    The snapshot carries ``measured``, but every read site was written as
    ``float(snap.get("pressure_pct", 0.0) or 0.0)``, which turns "the probe
    failed" into "pressure is 0.0%" at the point of use — and then logs it
    as a fact. This is the read that cannot do that.
    """
    if not isinstance(snapshot, dict):
        return None
    if not bool(snapshot.get("measured", True)):
        return None
    try:
        value = float(snapshot.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def format_metric(snapshot: Any, key: str, *, unit: str = "") -> str:
    """Render a snapshot metric for a log line, or ``unknown``.

    An operator reading "pressure=0.0% available=0.0GB" during an incident
    reasonably concludes the machine had memory. Saying ``unknown`` is the
    difference between a misleading log and a useful one.
    """
    value = snapshot_metric(snapshot, key)
    return "unknown" if value is None else f"{value:.1f}{unit}"


def admission_permits(
    snapshot: Any,
    *,
    max_age_s: float = ADMISSION_SNAPSHOT_MAX_AGE_S,
) -> tuple[bool, str]:
    """Does this snapshot actually authorize a heavy lane right now?

    Returns ``(permitted, reason)``. Refuses an unrecognised shape, a
    partial receipt, and a measurement too old to still be true — each with
    a distinct reason, because "we were refused" and "we asked with a stale
    receipt" need different responses from an operator.
    """
    if not isinstance(snapshot, dict):
        return False, f"admission_snapshot_not_a_dict:{type(snapshot).__name__}"
    if snapshot.get("schema") != ADMISSION_SNAPSHOT_SCHEMA:
        return False, f"admission_snapshot_schema_unrecognised:{snapshot.get('schema')!r}"
    missing = [key for key in _REQUIRED_ADMISSION_FIELDS if key not in snapshot]
    if missing:
        return False, f"admission_snapshot_incomplete:{','.join(missing)}"
    if not bool(snapshot.get("can_admit")):
        return False, str(snapshot.get("reason") or "admission_refused")
    stamped = snapshot.get("measured_at_monotonic")
    try:
        age = time.monotonic() - float(stamped)
    except (TypeError, ValueError):
        return False, "admission_snapshot_unstamped"
    if age > max(1.0, float(max_age_s)):
        return False, f"admission_snapshot_stale:{age:.1f}s"
    if not bool(snapshot.get("measured", True)):
        # `measured` was already a required field, and nothing read it. An
        # unmeasured snapshot only reaches here with can_admit=True via a
        # deliberate operator override (AURA_FORCE_*_ON_PROBE_FAILURE), so
        # it is permitted — but it is permitted *on the record*. Returning
        # an empty reason here would put a forced, unmeasured admission and
        # a genuine measured one in the same log line.
        return True, "admitted_unmeasured_forced_override"
    return True, ""



def _transition_age_s(client: Any, lane: Mapping[str, Any] | None = None) -> float:
    """How long the lane has held its current state, measured monotonically.

    Watchdogs act on this number by cancelling a load or forcing a lane cold,
    so it must be a DURATION and not a difference of wall clocks. An NTP step,
    a DST change, or the machine sleeping makes a wall-clock delta enormous
    (killing a healthy multi-minute load) or negative (deferring intervention
    for as long as the clock is behind).

    Falls back to the wall-clock stamp only when no monotonic one is available,
    and clamps at zero so a backwards jump reads as "just transitioned" rather
    than as a negative age that compares below every threshold.
    """
    mono = 0.0
    if lane is not None:
        try:
            mono = float(lane.get("last_transition_monotonic_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            mono = 0.0
    if mono <= 0.0:
        mono = float(getattr(client, "_lane_transition_monotonic_at", 0.0) or 0.0)
    if mono > 0.0:
        return max(0.0, time.monotonic() - mono)
    wall = float(getattr(client, "_lane_transition_at", 0.0) or 0.0)
    if wall <= 0.0:
        return 0.0
    return max(0.0, time.time() - wall)


def _generation_actually_stopped(client: Any) -> bool | None:
    """Whether this client really has no generation running.

    True and False are measured states. None means supervision was unavailable
    or malformed; absence of evidence must not be reported as a confirmed stop.
    """
    status = getattr(client, "get_supervision_status", None)
    if not callable(status):
        return None
    try:
        snapshot = status()
        if not isinstance(snapshot, Mapping) or "active_generations" not in snapshot:
            return None
        active = int(snapshot["active_generations"])
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return active <= 0


#: How long a hot spare may take to warm before maintenance gives up on it.
#: Generous — a cold 32B legitimately takes minutes — but finite, because the
#: await used to have no bound at all and a wedged load parked the maintenance
#: task indefinitely.
_HOT_SPARE_WARMUP_BUDGET_S = 300.0


def _hot_spare_is_ready(client: Any) -> bool:
    """Whether a warmed spare is actually ready, not merely running.

    ``is_alive()`` says a process exists. It does not say the model finished
    loading, that the lane left a transient state, or that a generation could
    be served — and readiness here used to be exactly that one call, so a spare
    still spawning counted as ready and the next caller routed into a lane that
    could not answer.

    Uses the richer supervision view when the client exposes one, and falls
    back to liveness only when it does not.
    """
    if not (hasattr(client, "is_alive") and client.is_alive()):
        return False
    status = getattr(client, "get_supervision_status", None)
    if not callable(status):
        return True
    try:
        state = str((status() or {}).get("state", "") or "").strip().lower()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return True
    # A lane still moving toward readiness is not a spare anyone can use.
    return state not in {"spawning", "handshaking", "warming", "recovering", "cold"}


def _exception_reports_active_generation(error: BaseException) -> bool:
    reason = str(error or "")
    return any(token in reason for token in _ACTIVE_GENERATION_BUSY_REASONS)


def _flatten_messages_for_local_model(messages: list[dict[str, str]]) -> str:
    """Flatten Aura messages into a Qwen/ChatML prompt for local MLX models."""
    return format_chatml_messages(messages)


def _observable_dispatch_markers() -> tuple[tuple[str, str], ...]:
    """(name, header) for every registered observable, for the survival check.

    Derived so that registering a reading also makes its delivery visible. A
    hand-maintained copy of this list is a second source of truth that goes
    stale the first time someone adds an observable and forgets it.
    """

    try:
        import core.brain.observable_registry  # noqa: F401  (registers)
        from core.brain.observable_grounding import OBSERVABLES

        return tuple(
            (observable.name, observable.header) for observable in OBSERVABLES
        )
    except (
        AttributeError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        # Observability must never break a turn — but the rule in this module
        # is named exceptions, and these are the ones this body can raise: the
        # import, the attribute read on the registry, and iterating whatever it
        # holds. A bare `except Exception` here would also swallow a
        # KeyboardInterrupt-adjacent bug in registration and report the runtime
        # as having no observables at all.
        return ()


def local_deep_solver_status(
    total_gb: float | None = None,
    available_gb: float | None = None,
    *,
    requested_domain: str | None = None,
) -> dict[str, Any]:
    """Return the canonical evidence and host admission for a specialist."""

    try:
        from core.brain.llm.model_registry import (
            get_deep_solver_admission_status,
        )

        evidence = get_deep_solver_admission_status(requested_domain)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return {"admitted": False, "reason": "specialist_evidence_unavailable"}

    setting = str(_FLAG_ENABLE_LOCAL_DEEP_SOLVER.value()).strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return {
            "admitted": False,
            "reason": "specialist_disabled_by_operator",
            "certificate_sha256": getattr(evidence, "certificate_sha256", ""),
        }
    if not bool(getattr(evidence, "admitted", False)):
        return {
            "admitted": False,
            "reason": str(getattr(evidence, "reason", "specialist_unmeasured")),
            "certificate_sha256": str(
                getattr(evidence, "certificate_sha256", "") or ""
            ),
            "admitted_domains": list(getattr(evidence, "admitted_domains", ())),
        }
    try:
        memory = None
        if total_gb is None or available_gb is None:
            memory = InferenceGate._recent_virtual_memory()
        detected_total = (
            float(total_gb)
            if total_gb is not None
            else float(memory.total) / float(1024**3)
        )
        detected_available = (
            float(available_gb)
            if available_gb is not None
            else float(memory.available) / float(1024**3)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        detected_total = 0.0
        detected_available = 0.0
    minimum_total = max(
        float(getattr(evidence, "minimum_total_gb", 0.0) or 0.0),
        float(_FLAG_LOCAL_DEEP_AUTO_MIN_TOTAL_GB.value()),
    )
    minimum_available = float(
        getattr(evidence, "minimum_available_gb", 0.0) or 0.0
    )
    if detected_total < minimum_total:
        reason = "specialist_host_total_below_qualified_minimum"
    elif detected_available < minimum_available:
        reason = "specialist_host_available_below_qualified_minimum"
    else:
        reason = "qualified"
    return {
        "admitted": reason == "qualified",
        "reason": reason,
        "certificate_sha256": str(
            getattr(evidence, "certificate_sha256", "") or ""
        ),
        "resident_descriptor_sha256": str(
            getattr(evidence, "resident_descriptor_sha256", "") or ""
        ),
        "specialist_descriptor_sha256": str(
            getattr(evidence, "specialist_descriptor_sha256", "") or ""
        ),
        "admitted_domains": list(getattr(evidence, "admitted_domains", ())),
        "evidence_age_s": getattr(evidence, "evidence_age_s", None),
        "expires_at": getattr(evidence, "expires_at", None),
        "topology": str(getattr(evidence, "topology", "") or ""),
        "host_total_gb": detected_total,
        "host_available_gb": detected_available,
        "minimum_total_gb": minimum_total,
        "minimum_available_gb": minimum_available,
    }


def local_deep_solver_enabled(
    total_gb: float | None = None,
    available_gb: float | None = None,
    *,
    requested_domain: str | None = None,
) -> bool:
    """Whether an evidence-qualified specialist can serve on this host.

    Deep reasoning itself remains the resident cortex plus Aura's systems
    intelligence.  This predicate owns only the optional second model.  Model
    size, path names, and environment flags cannot grant that lane authority.
    """

    return bool(
        local_deep_solver_status(
            total_gb,
            available_gb,
            requested_domain=requested_domain,
        ).get("admitted", False)
    )



def _asks_for_a_document(user_message: Any) -> bool:
    """Whether the reply has to contain a program or a page.

    The same question the desktop router asks to keep a build off the screen
    lane, so there is one notion of "this request produces a file" rather
    than two.
    """
    try:
        from core.runtime.desktop_objective_intent import asks_to_build_software

        return bool(asks_to_build_software(str(user_message or "")))
    except _INFERENCE_RECOVERABLE_ERRORS:
        return False



async def _apply_strict_proof_answer_contract(
    *,
    _is_bg_request: Any,
    context: Any,
    deep_handoff: Any,
    deep_probe_request: Any,
    origin: Any,
    prompt: Any,
    protected_foreground_lane: Any,
    requested_tier: Any,
    state: Any,
    strict_proof_answer_request: Any,
) -> tuple[Any, Any]:
    """Set the contract a strict proof answer runs under.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 8 name(s) from the turn and hands back
    2.
    """
    if strict_proof_answer_request:
        context["allow_tools"] = False
        context["trust_gate_skipped"] = "strict_proof_answer"
        context["strict_answer_contract"] = mlx_strict_answer_contract_enabled(origin=origin)
        context["disable_prompt_cache"] = True
        context["clear_prompt_cache"] = True
        context.setdefault("temperature", 0.0)
        context.setdefault("top_p", 1.0)
        context.setdefault("min_p", 0.0)
        context.setdefault("repetition_penalty", 1.12)
        strict_proof_tier = proof_model_tier()
        context["proof_model_tier"] = strict_proof_tier
        if strict_proof_tier == "tertiary":
            protected_foreground_lane = False
            requested_tier = "tertiary"
        else:
            protected_foreground_lane = True
            requested_tier = "primary"
    elif deep_probe_request and not _is_bg_request:
        # Deep self-report probes are foreground conversation checks, not
        # authentication attempts or tool requests.  Running the PBKDF2
        # passphrase recognizer here adds CPU contention right before the
        # Cortex turn and does not change the allowed action surface.
        context["allow_tools"] = False
        context["trust_gate_skipped"] = "deep_mind_probe"
    elif not _is_bg_request:
        try:
            from core.security.trust_engine import TrustLevel, get_trust_engine
            from core.security.user_recognizer import get_user_recognizer

            _te = get_trust_engine()
            _ur = get_user_recognizer()
            # Offload PBKDF2-heavy recognition to thread pool
            _trust_level = await asyncio.get_running_loop().run_in_executor(
                None, _te.process_message, prompt, _ur
            )
            _trust_guidance = _te.get_guidance_for_response()

            # [STABILITY v58] Force Primary 32B lane for all human-interaction tiers.
            # No brainstem fallbacks for Sovereign, Trusted, or Guest users.
            #
            # Recognition is not authorization for resources. The three
            # levels used to be treated identically here, so an
            # UNAUTHENTICATED guest could reverse a downgrade that
            # morphogenesis, a dead cortex, or headroom policy had already
            # made in this same request — and each promotion re-runs the
            # high-memory admission path. Guest keeps the lane it was
            # given when something safety-relevant already downgraded it;
            # sovereign and trusted still get the primary lane, because a
            # recognized principal is who the protected lane is for.
            if _trust_level in (TrustLevel.SOVEREIGN, TrustLevel.TRUSTED, TrustLevel.GUEST):
                downgraded_for_safety = bool(
                    requested_tier == "secondary"
                    or context.get("resource_stakes_blocked", False)
                    or context.get("local_deep_block_reason")
                    or deep_handoff
                )
                if _trust_level is TrustLevel.GUEST and downgraded_for_safety:
                    logger.info(
                        "🎭 Guest recognized, but this request was already "
                        "downgraded; not re-promoting it to the protected lane."
                    )
                elif requested_tier in ("", "primary"):
                    protected_foreground_lane = True
                    requested_tier = "primary"
                    logger.info(
                        "🎭 %s user recognized. Enforcing primary cortex lane (%s).",
                        _trust_level.name,
                        _primary_lane_label(),
                    )
                else:
                    # An EXPLICIT handoff, of any depth, is honoured.
                    #
                    # This read `!= "secondary"`, so an explicit request
                    # for the fast tertiary lane was overridden back to the
                    # 32B — silently. The protection exists to stop a
                    # recognised principal being downgraded WITHOUT asking;
                    # a caller that asks is not that, and asking for a
                    # cheaper lane is less consequential than asking for
                    # secondary, which was already allowed.
                    #
                    # Measured live 2026-08-19: a browser pursuit asked for
                    # `local_fast` on every round of a sixty-item form,
                    # was forced onto the Cortex at up to 103s a round, and
                    # the turn died on its own budget having answered
                    # nothing. Three layers were searched before the
                    # override was found, because nothing reported that the
                    # preference had been discarded.
                    logger.info(
                        "🎭 %s user recognized. Keeping the explicit %s handoff eligible for normal headroom checks.",
                        _trust_level.name,
                        requested_tier,
                    )

            # Trust belongs to THIS request, not to whatever reads the
            # state next.
            #
            # It used to be written to state.cognition.modifiers with no
            # session, principal or timestamp, so a later turn — or another
            # interlocutor sharing the same state — assembled context under
            # a trust classification recognition had granted to somebody
            # else. The value stays for ContextAssembler, and it now says
            # whose it is and when, so a stale one is identifiable rather
            # than inherited.
            context["trust_level"] = getattr(_trust_level, "name", str(_trust_level))
            if hasattr(state, "cognition") and hasattr(state.cognition, "modifiers"):
                state.cognition.modifiers["trust_level"] = _trust_level
                from core.runtime.principal_context import (
                    current_relational_principal,
                    relational_principal_scope_is_bound,
                )

                state.cognition.modifiers["trust_level_binding"] = {
                    "session_id": str(context.get("session_id", "") or ""),
                    "origin": str(origin or ""),
                    "recognized_at": time.time(),
                    "level": getattr(_trust_level, "name", str(_trust_level)),
                    # Whose recognition this was, taken from the
                    # request-scoped principal rather than from anything in
                    # the shared state. The assembler re-reads the same
                    # context var and refuses elevation when the two
                    # disagree, so a fabricated modifier has to also be
                    # running inside the right principal scope, which state
                    # construction cannot arrange.
                    "principal": current_relational_principal(),
                    "principal_scope_bound": relational_principal_scope_is_bound(),
                }

            # Block tool use for untrusted sessions
            if _trust_level in (TrustLevel.SUSPICIOUS, TrustLevel.HOSTILE):
                context["allow_tools"] = False
                context["max_tokens"] = min(context.get("max_tokens", 768), 768)
            # Inject trust guidance into context brief
            existing_brief = str(context.get("brief", ""))
            if _trust_guidance:
                context["brief"] = (_trust_guidance + "\n\n" + existing_brief).strip()
        except _INFERENCE_RECOVERABLE_ERRORS as _te_exc:
            context["allow_tools"] = False
            context["trust_gate_error"] = str(_te_exc)[:240]
            record_degradation(
                "inference_gate",
                _te_exc,
                severity="critical",
                action="disabled tool use and continued without trust guidance",
            )
            logger.warning("Trust gate error (passphrase check may have failed): %s", _te_exc)
    return protected_foreground_lane, requested_tier


async def _recover_the_cortex_before_answering(
    *,
    context: Any,
    is_background: Any,
    origin: Any,
    protected_foreground_lane: Any,
    requested_tier: Any,
    self: Any,
    strict_primary_proof_lane: Any,
) -> tuple[Any, Any]:
    """Recover the cortex inline, or say the turn cannot have it.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 7 name(s) and hands back
    1.
    """
    async def _block() -> Any:
        nonlocal requested_tier
        if not is_background:
            await self._ensure_cortex_recovery()
            # [STABILITY v51] If cortex is dead and NO recovery is in progress,
            # attempt inline recovery with a tight budget rather than waiting
            # for the background task that may not have started yet.
            if (
                self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
                and not self._cortex_recovery_in_progress
                and hasattr(self._mlx_client, "_ensure_worker_alive")
            ):
                inline_deferral = self._cortex_warmup_deferral_reason("foreground")
                if inline_deferral:
                    self._log_cortex_warmup_deferral(inline_deferral, context="foreground")
                    if strict_primary_proof_lane:
                        # A proof or benchmark names the primary model in its
                        # contract: a lower lane's answer would misreport its
                        # own provenance, so refusing is the honest outcome.
                        logger.warning(
                            "🧠 Cortex inline recovery was deferred and this turn's "
                            "contract names the primary lane; refusing lower-lane fallback."
                        )
                        return self._refuse_generation(
                            self.REFUSAL_PROOF_LANE,
                            str(inline_deferral),
                            context=context,
                            origin=origin,
                            detail={"lane": "primary", "deferral": str(inline_deferral)},
                        )
                    # protected_foreground_lane is a PRIORITY marker — "a real
                    # person is waiting" — not a provenance requirement. Treating
                    # it as one inverted its purpose: the 2026-07-25 endurance
                    # probe served "I couldn't put together an answer I'd stand
                    # behind" on 173 of 200 turns while the fallback workers sat
                    # resident and ready, because every protected user turn hit
                    # this branch during a cortex warmup backoff. Protecting
                    # someone is not a reason to hand them nothing.
                    if protected_foreground_lane:
                        logger.warning(
                            "🧠 Cortex inline recovery deferred (%s) on a protected "
                            "foreground turn; serving from Brainstem rather than "
                            "returning nothing to a waiting person.",
                            inline_deferral,
                        )
                        context["served_from_fallback_lane"] = True
                        context["fallback_lane_reason"] = str(inline_deferral)
                    else:
                        logger.warning(
                            "🧠 Cortex inline recovery skipped by RAM admission; routing foreground turn to Brainstem."
                        )
                    requested_tier = "tertiary"
                else:
                    logger.warning(
                        "🔄 [STABILITY] Cortex dead, no recovery in progress. Attempting inline fast-recovery (15s budget)..."
                    )
                    try:
                        alive = await asyncio.wait_for(
                            self._mlx_client._ensure_worker_alive(
                                request_is_background=False,
                                foreground_request=True,
                                init_timeout=15.0,
                                soft_timeout=True,
                            ),
                            timeout=15.0,
                        )
                        if alive:
                            logger.info("✅ [STABILITY] Inline fast-recovery succeeded.")
                    except (
                        TimeoutError,
                        RuntimeError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        OSError,
                    ) as inline_exc:
                        record_degradation(
                            "inference_gate",
                            inline_exc,
                            severity="degraded",
                            action="downgraded foreground request after inline cortex recovery failure",
                        )
                        logger.warning("⚠️ [STABILITY] Inline fast-recovery failed: %s", inline_exc)

            # If cortex recovery was just triggered or is in progress, give it
            # a short window to complete before the user hits a dead endpoint.
            # [STABILITY v51] Reduced from 10×1s to 5×1s to keep responsiveness.
            if (
                self._cortex_recovery_in_progress
                and self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
            ):
                for _ in range(5):  # Up to 5s of 1s slices
                    await asyncio.sleep(1.0)
                    if self._mlx_client.is_alive():
                        logger.info("✅ InferenceGate: cortex recovered inline for user request.")
                        break
            # If cortex is STILL dead after recovery wait, downgrade to secondary
            # tier rather than sending the user a fallback/"wound up" response.
            # A real answer from the 7B is better than no answer from the 32B.
            if (
                self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
                and requested_tier == "primary"
            ):
                if protected_foreground_lane:
                    logger.warning(
                        "⚠️ InferenceGate: Primary cortex is still warming after the short inline wait, "
                        "but protected foreground mode will preserve the requested high-capability path."
                    )
                else:
                    logger.warning(
                        "⚠️ InferenceGate: Primary cortex is still warming after the short inline wait. "
                        "Downgrading to the fast tertiary lane for user responsiveness."
                    )
                    requested_tier = "tertiary"  # Use 7B brainstem — fast, always available

            # RAM-aware inference routing: if the primary lane is not protected
            # and the memory envelope is already unsafe, keep the process alive
            # by routing to a smaller local lane. Protected live desktop turns
            # are handled by the later admission check, which can fail closed
            # instead of silently downgrading model quality.
            if requested_tier == "primary" and not protected_foreground_lane:
                try:
                    primary_headroom = self._headroom_snapshot("primary")
                    _primary_ok, _primary_reason = admission_permits(primary_headroom)
                    if not _primary_ok:
                        logger.warning(
                            "InferenceGate: primary lane outside safe memory envelope "
                            "(pressure=%s available=%s). Downgrading to brainstem.",
                            format_metric(primary_headroom, "pressure_pct", unit="%"),
                            format_metric(primary_headroom, "available_gb", unit="GB"),
                        )
                        requested_tier = "tertiary"
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Foreground RAM pressure probe unavailable: %s", exc)

            if requested_tier != "secondary" and self._background_memory_pressure_active():
                await self._shed_background_workers_for_memory_pressure()
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, requested_tier


async def _admit_the_foreground_request(
    *,
    context: Any,
    deep_handoff: Any,
    desktop_cognitive_engine_contract: Any,
    fallback_timeout: Any,
    initial_visible_user_prompt: Any,
    is_background: Any,
    max_tokens: Any,
    origin: Any,
    primary_timeout: Any,
    protected_foreground_lane: Any,
    request_deadline: Any,
    requested_tier: Any,
    self: Any,
    surface_completion_floor: Any,
    timeout: Any,
    timeout_val: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Admit the foreground request and settle its budgets.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 16 name(s) and hands back
    7.
    """
    async def _block() -> Any:
        nonlocal deep_handoff, fallback_timeout, max_tokens, primary_timeout, request_deadline, requested_tier, timeout_val
        if not is_background and requested_tier in {"primary", "secondary"}:
            admission_snapshot = await self._enforce_foreground_admission(
                requested_tier,
                protected_foreground=protected_foreground_lane,
            )
            # A complete, fresh receipt or none at all — see admission_permits.
            _admitted, _admission_reason = admission_permits(admission_snapshot)
            if not _admitted and requested_tier == "secondary":
                logger.warning(
                    "🛡️ InferenceGate: deep local handoff exceeds safe headroom "
                    "(pressure=%s available=%s process=%s/%s). "
                    "Downgrading to the primary lane.",
                    format_metric(admission_snapshot, "pressure_pct", unit="%"),
                    format_metric(admission_snapshot, "available_gb", unit="GB"),
                    format_metric(admission_snapshot, "process_rss_gb", unit="GB"),
                    format_metric(admission_snapshot, "process_rss_limit_gb", unit="GB"),
                )
                requested_tier = "primary"
                deep_handoff = False
                timeout_val = self._requested_timeout_s(
                    timeout,
                    self._default_timeout_for_request(
                        origin,
                        requested_tier,
                        deep_handoff=deep_handoff,
                        is_background=is_background,
                    ),
                )
                primary_timeout, fallback_timeout = self._split_attempt_timeouts(
                    timeout_val, requested_tier
                )
                # Re-derived budget after a tier downgrade; the clock does not
                # restart, so the deadline keeps its original start time.
                request_deadline = get_deadline(float(timeout_val))
                context["request_deadline_s"] = float(timeout_val)
                max_tokens = self._requested_max_tokens(
                    context.get("max_tokens"),
                    self._default_max_tokens_for_request(
                        origin,
                        requested_tier,
                        deep_handoff=deep_handoff,
                        is_background=is_background,
                    ),
                )
                if "max_tokens" not in context:
                    max_tokens = self._adaptive_max_tokens_for_prompt(
                        initial_visible_user_prompt,
                        base_tokens=max_tokens,
                        origin=origin,
                        requested_tier=requested_tier,
                        is_background=is_background,
                    )
                admission_snapshot = await self._enforce_foreground_admission(
                    requested_tier,
                    protected_foreground=protected_foreground_lane,
                )
            if (
                admission_snapshot is not None
                and not admission_snapshot.get("can_admit", False)
                and requested_tier == "primary"
            ):
                pressure = float(admission_snapshot.get("pressure_pct", 0.0) or 0.0)
                available = float(admission_snapshot.get("available_gb", 0.0) or 0.0)
                process_rss = float(admission_snapshot.get("process_rss_gb", 0.0) or 0.0)
                process_limit = float(admission_snapshot.get("process_rss_limit_gb", 0.0) or 0.0)
                process_over_limit = bool(process_limit > 0.0 and process_rss >= process_limit)
                # Scaled to the host, the way the prewarm check below already
                # is. Two thresholds answering the same question — is there
                # enough memory to generate — disagreed by two gigabytes and
                # ten points of pressure, and the flat one is the one that
                # refuses a person's turn. A machine with sixty-four gigabytes
                # and a resident twenty-gigabyte model sits nearer these
                # numbers than a smaller one ever does.
                total_gb = float(admission_snapshot.get("total_gb", 0.0) or 0.0)
                roomy_host = total_gb >= 60.0
                if (
                    pressure >= (92.0 if roomy_host else 90.0)
                    or available < (6.0 if roomy_host else 8.0)
                    or process_over_limit
                ):
                    logger.error(
                        "🛑 InferenceGate: refusing primary foreground generation under critical "
                        "memory pressure (pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB).",
                        pressure,
                        available,
                        process_rss,
                        process_limit,
                    )
                    return self._refuse_generation(
                        self.REFUSAL_RESOURCE,
                        "critical_memory_pressure",
                        context=context,
                        origin=origin,
                        detail={
                            "pressure_percent": pressure,
                            "available_gb": available,
                            "process_rss_gb": process_rss,
                            "process_rss_limit_gb": process_limit,
                            "admission": dict(admission_snapshot or {}),
                        },
                    )
                near_process_limit = bool(process_limit > 0.0 and process_rss >= process_limit * 0.90)
                # What an output token actually costs is KV cache, not model
                # weights. For this 64-layer 32B with 8 KV heads at head_dim
                # 128: 64 × 2 × 8 × 128 × 2 bytes ≈ 0.26 MB per token. 1,536
                # tokens is ~400 MB — about 3% of the 14 GB free when this
                # fired. The weights are the 21 GB, and no output cap moves them.
                #
                # LIVE DEFECT, 2026-07-26: this branch runs whenever admission
                # says can_admit=False, which with a resident 32B on this host
                # is every foreground turn. So 384 was not a pressure response,
                # it was a permanent ceiling on how long any desktop answer
                # could be, and "…show the reasoning, then give the exact
                # fraction" was cut mid-derivation at
                #   "Probability of first being red = 3/12 - Given the first is
                #    red, probability second is also red ="
                # every time, on a host with 14 GB free.
                #
                # Genuinely critical pressure still refuses outright, above.
                completion_floor_affordable = bool(
                    desktop_cognitive_engine_contract
                    and surface_completion_floor > 0
                    and available >= 12.0
                    and pressure < 84.0
                    and not near_process_limit
                )
                capped_tokens = (
                    768
                    if available < 12.0 or pressure >= 84.0 or near_process_limit
                    else max(
                        1536,
                        surface_completion_floor if completion_floor_affordable else 0,
                    )
                )
                if max_tokens > capped_tokens:
                    logger.warning(
                        "🛡️ InferenceGate: capping primary foreground output to %d tokens under "
                        "memory pressure (pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB).",
                        capped_tokens,
                        pressure,
                        available,
                        process_rss,
                        process_limit,
                    )
                    max_tokens = capped_tokens
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, deep_handoff, fallback_timeout, max_tokens, primary_timeout, request_deadline, requested_tier, timeout_val


def _modulate_sampling_from_the_body(
    *,
    ServiceContainer: Any,
    _rt: Any,
    context: Any,
    explicit_foreground: Any,
    is_background: Any,
    max_tokens: Any,
    morpho_kwargs: Any,
    protected_compact_capability_contract: Any,
    protected_foreground_lane: Any,
    self: Any,
    somatic_temperature: Any,
) -> tuple[Any, Any]:
    """Let the body's state move temperature and length, within bounds.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 11 name(s) from the turn and hands back
    2.
    """
    if _rt is not None:
        _f = _rt.field.sample("global")
        _danger = self._modulator_factor(
            _f.get("danger", 0.0), source="morphogenesis.danger", low=0.0, high=1.0
        )
        _curiosity = self._modulator_factor(
            _f.get("curiosity", 0.0),
            source="morphogenesis.curiosity",
            low=0.0,
            high=1.0,
        )
        _resource_pressure = self._modulator_factor(
            _f.get("resource_pressure", 0.0),
            source="morphogenesis.resource_pressure",
            low=0.0,
            high=1.0,
        )

        if _danger > 0.3:
            somatic_temperature = (somatic_temperature or 0.72) * (
                1.0 - (_danger * 0.4)
            )
            morpho_kwargs["top_p"] = max(0.4, 0.9 - (_danger * 0.3))

        if _curiosity > 0.3:
            somatic_temperature = (somatic_temperature or 0.72) * (
                1.0 + (_curiosity * 0.3)
            )
            morpho_kwargs["repetition_penalty"] = max(1.0, 1.15 - (_curiosity * 0.1))

        if _resource_pressure > 0.5 and not protected_compact_capability_contract:
            max_tokens = int(max_tokens * (1.0 - (_resource_pressure * 0.5)))
            max_tokens = max(128, max_tokens)

        # Inject Existential Stakes physical parameter coupling
        try:
            stakes = ServiceContainer.get("existential_stakes", default=None)
            if stakes:
                threat = float(stakes.get_existential_threat())
                if not math.isfinite(threat):
                    raise ValueError("existential threat must be finite")
                threat = max(0.0, min(1.0, threat))
                if threat > 0.2:
                    protected_live_foreground = bool(
                        not is_background
                        and (
                            protected_foreground_lane
                            or context.get("desktop_cognitive_engine_required")
                            or context.get("cognitive_engine_required")
                            or explicit_foreground
                        )
                    )
                    # Background and unprotected turns may shrink output under
                    # survival pressure. Protected live desktop turns must not:
                    # starving the first user-visible Cortex reply causes clipped
                    # drafts, recovery storms, worker respawns, and higher memory
                    # pressure than simply answering with the requested budget.
                    if not protected_live_foreground:
                        max_tokens = int(max_tokens * (1.0 - threat * 0.7))
                        max_tokens = max(96, max_tokens)
                    # Decrease temperature to make generation fast/deterministic
                    if somatic_temperature is not None:
                        somatic_temperature = somatic_temperature * (1.0 - threat * 0.5)
                    else:
                        somatic_temperature = 0.72 * (1.0 - threat * 0.5)
                    # Clamp parameters
                    if "temperature" in morpho_kwargs:
                        morpho_kwargs["temperature"] = max(0.1, morpho_kwargs["temperature"] * (1.0 - threat * 0.5))
                    if "max_tokens" in morpho_kwargs:
                        morpho_kwargs["max_tokens"] = max_tokens
        except _INFERENCE_RECOVERABLE_ERRORS as _st_err:
            record_degradation(
                "inference_gate.existential_stakes",
                _st_err,
                severity="warning",
                action=(
                    "kept the validated morphogenetic generation parameters "
                    "and ignored only the invalid existential-stakes modifier"
                ),
            )
            logger.warning(
                "Existential-stakes generation modifier rejected; "
                "using validated base parameters: %s",
                _st_err,
            )

        if somatic_temperature is not None:
            somatic_temperature = max(0.1, min(1.5, somatic_temperature))

        logger.debug(
            "🧬 Morphogenetic Coupling: danger=%.2f curiosity=%.2f pres=%.2f -> temp=%.2f tokens=%d",
            _danger,
            _curiosity,
            _resource_pressure,
            somatic_temperature or 0.0,
            max_tokens,
        )
    return max_tokens, somatic_temperature


async def _attach_the_present_moment(
    *,
    ambient_grounding_blocks: Any,
    isolated_generation_contract: Any,
    recent_actions_already_grounded: Any,
    task_grounding_blocks: Any,
    visible_user_prompt: Any,
) -> None:
    """Attach the present-moment block unless the turn is isolated.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    0.
    """
    if not isolated_generation_contract:
        try:
            from core.brain.present_moment import present_moment_block

            _present = present_moment_block()
            if _present:
                ambient_grounding_blocks.append(_present)

            # Suppressing the web search is only half the fix; without the
            # readings she still has to invent them, which is how "I
            # processed a 45-page PDF on neuromorphic computing" happened.
            if not recent_actions_already_grounded:
                from core.brain.recent_actions import (
                    asks_what_she_recently_did,
                    recent_actions_block,
                )

                # Historical action receipts answer questions about historical
                # actions.  Injecting them into every factual turn let an old
                # autonomous search masquerade as the source of a new answer.
                if asks_what_she_recently_did(visible_user_prompt):
                    _actions = recent_actions_block()
                    if _actions:
                        ambient_grounding_blocks.append(_actions)

            # The WIDER predicate here on purpose. This path only ADDS her
            # instrument reading, so a false positive costs a few lines of
            # prompt; asks_about_own_runtime additionally suppresses web
            # search in the response contract, where a false positive
            # costs the lookup the person asked for.
            from core.runtime.self_state_intent import (
                asks_about_own_capabilities,
            )

            if asks_about_own_capabilities(visible_user_prompt):
                from core.brain.self_state_report import runtime_self_report

                _instruments = runtime_self_report()
                if _instruments:
                    ambient_grounding_blocks.append(_instruments)

            # A file she was asked about, read off the disk.
            #
            # LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me
            # the first rule it states" was answered "I tried to read the
            # file and failed" — an attempt that never happened. No skill
            # ran, no error occurred; she narrated a failure.
            #
            # The read was wired into the phase pipeline's grounding
            # channel, which desktop chat does not use: chat arrives here
            # with prebuilt messages (mode=compact_foreground_prebuilt), so
            # the block was built into a prompt nobody sent. THIS is the
            # channel that reaches the worker, which is why it is attached
            # beside the present-moment and recent-actions readings rather
            # than anywhere upstream.
            # Take every reading this turn asks for.
            #
            # This was two hand-wired branches — one for the clipboard, one
            # for a named file — and before them a file COUNT that guessed,
            # a corpus that was never consulted, and a clock that invented
            # an ambient light sensor. Same defect each time: the capability
            # was registered, the reader existed, and nothing took the
            # reading before the answer was composed, so a model asked about
            # a fact it did not hold produced something fact-shaped.
            #
            # One registry now. An observable is an entry in
            # observable_registry rather than another branch threaded
            # through here, which is how the previous four ended up in four
            # places with four different bugs.
            import core.brain.observable_registry  # noqa: F401  (registers)
            from core.brain.observable_grounding import observable_blocks

            _readings = await observable_blocks(visible_user_prompt)
            if _readings:
                task_grounding_blocks.extend(_readings)
                logger.info(
                    "🔭 [GROUNDING] took %d reading(s): %s",
                    len(_readings),
                    ",".join(
                        block.split("\n", 1)[0].removeprefix("## ").lower()
                        for block in _readings
                    ),
                )
            else:
                # A turn that asked for a reading and got none is invisible
                # otherwise, which is how the screen block went missing for
                # a whole session while the file block worked.
                logger.debug(
                    "🔭 [GROUNDING] no reading matched prompt=%r",
                    str(visible_user_prompt)[:120],
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            record_degradation(
                "inference_gate",
                _exc,
                severity="warning",
                action="continued without present-moment grounding",
            )


def _refresh_volatile_grounding(
    *,
    ambient_grounding_blocks: Any,
    context: Any,
    contract_grounding_blocks: Any,
    has_volatile_grounding: Any,
    messages: Any,
    self: Any,
    system_prompt: Any,
    task_grounding_blocks: Any,
) -> tuple[Any, Any]:
    """Refresh grounding that goes stale between the prompt and the answer.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 8 name(s) from the turn and hands back
    2.
    """
    if has_volatile_grounding and isinstance(messages, list) and messages:
        # BEFORE the final user turn, not after it.
        #
        # Riding dead last put a multi-thousand-character block of self-state
        # between the person's question and the model's turn, and the model
        # continued the nearest thing instead of answering. Measured live:
        # asked to run a real sandbox calculation and report the result, the
        # entire reply was "Things feel unusually settled right now. My
        # attention is on internal monitoring..." — the grounding text
        # continued as prose, with no answer, no code, and no refusal, on a
        # turn whose plan read scaffold=7023 request=518 (ratio 13.6x).
        #
        # Sitting just ahead of the last user message keeps the whole point
        # of volatile-last — every stable token, system prompt through prior
        # history, is still a reusable KV prefix, and the only thing behind
        # the churn is the new turn that had to be prefilled anyway — while
        # the last words before the model's turn are the person's own.
        # The budget is enforced during compaction, and this block is added
        # afterwards — so without a cap here it simply escapes it. Measured
        # 2026-07-28 on the contract profile: compaction produced a 974-char
        # payload well inside its 2,800 budget, then 1,727 characters of
        # grounding arrived as a second system message and the turn went out
        # at 3,013. The grounding is not optional — it is what stops her
        # narrating a present she was never given — so it is fitted rather
        # than dropped, keeping whole blocks in priority order.
        from core.utils.injected_blocks import stamp_grounding

        grounding_message = stamp_grounding(
            {
                "role": "system",
                "content": self._fit_grounding_blocks(
                    contract_blocks=contract_grounding_blocks,
                    task_blocks=task_grounding_blocks,
                    ambient_blocks=ambient_grounding_blocks,
                    limit=self._grounding_char_budget(context, messages),
                ),
            }
        )
        final_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and str(messages[index].get("role", "")).strip().lower() == "user"
            ),
            None,
        )
        if final_user_index is None:
            messages = [*messages, grounding_message]
        else:
            messages = [
                *messages[:final_user_index],
                grounding_message,
                *messages[final_user_index:],
            ]
    elif has_volatile_grounding:
        # No message list to ride behind (single-prompt lanes): keep the old
        # behaviour rather than dropping the grounding entirely.
        system_prompt = "\n\n".join(
            [
                str(system_prompt or ""),
                *contract_grounding_blocks,
                *task_grounding_blocks,
                *ambient_grounding_blocks,
            ]
        ).strip()
    return messages, system_prompt


def _settle_the_token_ceilings(
    *,
    context: Any,
    deep_handoff: Any,
    desktop_cognitive_engine_contract: Any,
    max_tokens: Any,
    prompt: Any,
    protected_compact_capability_contract: Any,
    requested_tier: Any,
    self: Any,
    stakes: Any,
    stakes_token_ceiling: Any,
    surface_completion_floor: Any,
) -> tuple[Any, Any, Any, Any]:
    """Settle the token ceilings this request runs under.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 9 name(s) from the turn and hands back
    4.
    """
    if stakes is not None and hasattr(stakes, "action_envelope"):
        envelope = stakes.action_envelope("high" if deep_handoff else "normal")
        protected_surface_completion = bool(
            protected_compact_capability_contract
            or (
                desktop_cognitive_engine_contract
                and surface_completion_floor > 0
            )
        )
        # CP126 (critical): "A resource-stakes block can be undone by
        # later token modifiers. A denied envelope caps max_tokens at
        # 128, but later homeostatic modifiers impose a 384 minimum,
        # temporal continuity can grow the budget to 4096, and runtime
        # sampling biases still run. resource_stakes_blocked
        # suppresses only selected foreground floors, not these."
        #
        # A cap applied here is a suggestion — a dozen later
        # transformations each raise or scale the budget, and asking
        # every one of them to remember a flag is how the flag ends up
        # checked in four places and missed in six. The envelope's
        # limit is recorded as a CEILING and re-applied as the last
        # token transformation before generation, next to the existing
        # caller-cap clamp that already works this way.
        #
        # The protected capability lane used to skip the cap
        # ENTIRELY, on both branches. That is an exemption from
        # viability control, and the denied branch is exactly where an
        # exemption is worst: the ledger has said the runtime is out
        # of resources and this lane ignored it with no ceiling at all.
        # It is a bounded override now — the lane gets enough tokens to
        # answer the question it was protected for, and not one more,
        # with the raise recorded.
        if not envelope.allowed:
            requested_tier = "primary"
            deep_handoff = False
            max_tokens, stakes_token_ceiling = self._stakes_capped_tokens(
                max_tokens,
                envelope_cap=_STAKES_DENIED_TOKEN_CAP,
                protected=protected_surface_completion,
                completion_floor=surface_completion_floor,
                prompt=prompt,
                context=context,
                reason="envelope_denied",
            )
            context["resource_stakes_blocked"] = True
        else:
            max_tokens, stakes_token_ceiling = self._stakes_capped_tokens(
                max_tokens,
                envelope_cap=max(1, int(envelope.max_tokens)),
                protected=protected_surface_completion,
                completion_floor=surface_completion_floor,
                prompt=prompt,
                context=context,
                reason="envelope_allowed",
            )
            if "large_model_cortex" in set(envelope.disabled_capabilities):
                requested_tier = "primary"
                deep_handoff = False
        context["resource_stakes_envelope"] = envelope.as_dict()
        if stakes_token_ceiling is not None:
            context["resource_stakes_token_ceiling"] = stakes_token_ceiling
    return deep_handoff, max_tokens, requested_tier, stakes_token_ceiling


async def _refuse_a_cold_protected_lane(
    *,
    benchmark_request: Any,
    context: Any,
    initial_visible_user_prompt: Any,
    origin: Any,
    output_contract: Any,
    output_contract_payload: Any,
    proof_evaluation_contract: Any,
    self: Any,
    state: Any,
) -> Any:
    """Refuse the turn when the protected lane is conclusively cold.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 9 name(s) and hands back
    0.
    """
    async def _block() -> Any:
        if bool(context.get("allow_mesh_cognition", True)) and not (
            proof_evaluation_contract
            or benchmark_request
            or bool(context.get("is_background", False))
        ):
            try:
                from core.consciousness.mesh_cognition import get_mesh_cognition

                mesh_decision = get_mesh_cognition().decide(
                    initial_visible_user_prompt,
                    state=state,
                )
                if mesh_decision.handled:
                    context["mesh_cognition"] = mesh_decision.as_dict()
                    # This return happens BEFORE trust recognition, admission
                    # and routing policy, all of which live further down. For
                    # an acknowledgement or a resource hold that is fine —
                    # neither reveals anything and neither spends a lane. A
                    # SELF-REPORT is different: it describes her internal state
                    # to whoever asked, and who asked is not yet known. Those
                    # fall through to the full path, which recognizes trust
                    # first and can still answer.
                    if str(mesh_decision.rationale or "") in _MESH_PRE_TRUST_RATIONALES:
                        self._record_client_generation_metadata(
                            None,
                            label="MeshCognition",
                            success=bool(str(mesh_decision.response or "").strip()),
                            text=str(mesh_decision.response or ""),
                            requested_max_tokens=output_contract.semantic_token_cap,
                            output_contract=output_contract_payload,
                        )
                        self._record_user_generation_endpoint("MeshCognition")
                        return self._stabilize_user_facing_text(
                            mesh_decision.response,
                            initial_visible_user_prompt,
                            is_user_facing=True,
                        )
                    context["mesh_deferred_for_trust"] = str(
                        mesh_decision.rationale or "unknown"
                    )
                    logger.info(
                        "🕸️ Mesh handled the turn as %s; deferring to the governed "
                        "path so trust is recognized before self-report leaves.",
                        mesh_decision.rationale,
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _mesh_exc:  # pragma: no cover - defensive
                # A raised exception is not the same as `handled=False`. The
                # first is a broken organism path, the second is the design
                # working. Calling both "declined" in a debug line left an
                # operator no way to tell them apart.
                context["mesh_cognition_error"] = f"{type(_mesh_exc).__name__}: {_mesh_exc}"[:240]
                _record_inference_degradation(
                    _mesh_exc,
                    action="fell through to the LLM path after the mesh cognition path raised",
                    extra={"origin": str(origin or "")},
                )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response


#: What a turn keeps back so it can say what the tools found, when nothing
#: better is known. A last resort: the real reserve is measured from the prompt
#: the answer will actually be read from, at the rate this worker has actually
#: been measured at.
#:
#: LIVE, 2026-08-28: three files read, and the answer generation was given a
#: 36.2s first-token budget for a 6,298-char prompt that takes about 120s to
#: read. Everything the loop found was correct and none of it could be said.
#: The reserve was a constant while the thing it reserves for grows with what
#: the tools return — the more it finds, the longer the prompt, and the number
#: never moved.
_ANSWER_RESERVE_FALLBACK_S = 45.0

#: Below this the tool loop cannot complete a single call, so squeezing it
#: further trades one failure for another.
_TOOL_LOOP_FLOOR_S = 20.0


def _answer_reserve_seconds(client: Any, prompt_chars: Any) -> float:
    """The time the answer needs, measured rather than assumed.

    Reading the prompt is most of it, and the prompt is the scaffold plus
    everything the tools returned. The worker knows its own prefill rate and
    already computes this to raise its first-token ceiling; the reserve is the
    same quantity, asked one layer up and before the time is spent.
    """

    try:
        chars = max(0, int(prompt_chars or 0))
    except (TypeError, ValueError):
        chars = 0
    if not chars:
        return _ANSWER_RESERVE_FALLBACK_S
    try:
        needed = float(client._prefill_floor_seconds(chars))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return _ANSWER_RESERVE_FALLBACK_S
    if not (needed > 0.0):
        return _ANSWER_RESERVE_FALLBACK_S
    # Reading it, and then saying something short about it.
    return needed + _ANSWER_RESERVE_FALLBACK_S


def _tool_loop_budget(timeout_s: Any, reserve_s: Any = None) -> float:
    """How long the tool loop may run, leaving enough to report what it found."""
    try:
        whole = float(timeout_s)
    except (TypeError, ValueError):
        return _TOOL_LOOP_FLOOR_S
    try:
        reserve = float(reserve_s)
    except (TypeError, ValueError):
        reserve = _ANSWER_RESERVE_FALLBACK_S
    if not (reserve > 0.0):
        reserve = _ANSWER_RESERVE_FALLBACK_S
    return max(_TOOL_LOOP_FLOOR_S, whole - reserve)


#: Fields a tool result carries that a person would want to read.
_READABLE_RESULT_FIELDS = ("content", "stdout", "summary", "text", "output", "answer")

#: Fields that are the runtime talking to itself.
_PLUMBING_FIELDS = frozenset(
    {
        "authority_closure", "token_revoked", "standing_authority_closed",
        "intent_closed", "governance_receipt_id", "agency_receipt_id",
        "expectation_receipt_id", "deliberation_receipts", "retries", "skill",
        "mode", "governance_route", "action_expectation", "expectation_verdict",
        "verification_evidence", "duration_ms", "ok",
    }
)


def _what_a_tool_returned(result: Any) -> str:
    """The readable part of a tool result, never the envelope.

    LIVE, 2026-08-27: with none of the readable fields present, this fell back
    to str() on the whole dict and put `authority_closure`, `token_revoked` and
    `standing_authority_closed` on the screen. That is the runtime talking
    about itself, and the person asked about a ledger.
    """
    if result is None:
        return ""
    if not isinstance(result, dict):
        return str(result).strip()
    for field in _READABLE_RESULT_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # No prose, so say what it found in the fields that name things.
    said: list[str] = []
    for key, value in result.items():
        if key in _PLUMBING_FIELDS or key in _READABLE_RESULT_FIELDS:
            continue
        if isinstance(value, str) and value.strip():
            said.append(f"{key}: {value.strip()}")
        elif isinstance(value, (list, tuple)) and value:
            said.append(f"{key}: " + ", ".join(str(item) for item in list(value)[:8]))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            said.append(f"{key}: {value}")
    return "; ".join(said[:6])


def _tools_within_reach(
    tools: dict, allowed_scopes: object
) -> tuple[dict, tuple[str, ...]]:
    """The offered tools this turn is actually allowed to run, and the rest.

    Offering a capability the dispatch then refuses is worse than not offering
    it: the turn spends its budget reaching for something it was never allowed
    to use. That was already written down here as the reason for computing an
    effect ceiling — the ceiling was just never applied to the tool map.

    Two things put a tool out of reach: an effect scope above this turn's
    ceiling, and a risk rating that needs a confirmation the turn has no way to
    ask for. `code_repl` passes the ceiling and fails the second — running code
    the model just wrote is correctly high risk — so offering it spends a tool
    call on a refusal either way.

    A skill with no declared scope is offered. Withholding what has not been
    rated would quietly hide new skills, which is a worse failure than one
    refusal.
    """
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    permitted = {str(scope) for scope in (allowed_scopes or ())}
    if not permitted:
        return tools, ()
    kept, withheld = {}, []
    for name, definition in dict(tools).items():
        declared_scope = SKILL_EFFECT_SCOPES.get(str(name))
        if declared_scope is None:
            # Unrated. Withholding what nobody has rated hides every new skill,
            # which is how a skill built for a request ends up never called.
            kept[name] = definition
            continue
        scope = _reachable_scope(str(name), declared_scope, permitted)
        if scope is None:
            withheld.append(str(name))
            continue
        if _needs_a_confirmation_nobody_can_give(str(name), scope):
            withheld.append(str(name))
            continue
        kept[name] = definition
    return kept, tuple(withheld)


def _reachable_scope(name: str, skill_scope: Any, permitted: set[str]) -> str | None:
    """The safest thing this skill can do that the turn allows, or None.

    A skill's blanket scope is its most dangerous action. `file_operation` is
    rated state_mutation because it can delete, and its read, list and exists
    actions mutate nothing — which is why the dispatch is handed the ceiling
    and refuses each call on its own scope.

    LIVE, 2026-08-27: "read the docs at <path>, then actually use it" ran under
    a sandboxed_compute ceiling, and the blanket rating withheld the only tool
    that could read the file. Offering it for its safe actions is the design
    this file already describes; it just was not filtering that way.
    """
    if skill_scope is None:
        return "unknown"
    if str(skill_scope) in permitted:
        return str(skill_scope)
    try:
        from core.skills.action_scope import (
            EFFECT_SCOPE_RANK,
            declared_action_scopes,
            skill_class_named,
        )
    except ImportError:
        return None
    try:
        declared = declared_action_scopes(skill_class_named(name))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    reachable = [scope for scope in declared.values() if scope in permitted]
    if not reachable:
        return None
    # The safest reachable one, because that is what the turn may actually do.
    return min(reachable, key=lambda scope: EFFECT_SCOPE_RANK.get(scope, 9))


def _needs_a_confirmation_nobody_can_give(name: str, scope: str) -> bool:
    """Whether dispatch would stop this tool to ask, on a turn that cannot ask.

    Only when the answer does not depend on the arguments. A tool whose whole
    job is to run a snippet is rated from that snippet, and at offer time there
    is no snippet — withholding it on the worst case means it can never be
    called, which is how "read these docs and actually use the library" became
    unanswerable. Dispatch sees the real arguments and refuses there.
    """
    try:
        from core.executive.execution_policy import _RUNS_A_SNIPPET, classify_execution_risk

        if name in _RUNS_A_SNIPPET:
            return False
        risk = str(classify_execution_risk(name, {}, effect_scope=scope) or "").lower()
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False
    return risk in {"high", "critical"}


#: The shortest thing worth calling an answer. Below this what comes back is
#: a fragment, which is what a deadline stopping mid-decode produces anyway.
A_REAL_ANSWER = 128


def fit_the_answer_to_the_time(
    prompt: Any, max_tokens: Any, asked_for: float, *, floor: int = 0
) -> tuple[float, int]:
    """A budget and a length that fit each other, and fit what the caller waits.

    A question's deadline belongs to the question, and so does its length.
    Given fifty seconds for a prompt that takes longer than that to read and
    answer, the decode stops part way and a real reply is thrown away for an
    apology: LIVE 2026-08-26, "Request deadline reached at token 207", and
    what the person got was "I couldn't get a clear enough answer together."

    Lifting the clock alone is the other half of the same mistake. Asked for
    everything it wanted, the same question came back with a budget of five
    hundred and seventy-eight seconds — longer than anyone waits, and longer
    than the caller itself would wait, so the answer never arrived at all.

    Reading the question is not optional and cannot be made shorter. Writing
    the answer can. So the time is stretched only as far as reading needs, and
    what does not fit after that comes out of the length.

    ``floor`` is a length this turn may not be cut below — the protected
    capability lane earns one by overriding the resource envelope, and without
    it that override was applied and then undone here, silently, by a function
    that knew nothing about the protection. Where a floor is given, the clock
    stretches to afford it instead of the length shrinking; where even the
    longest clock cannot, the cut happens and is recorded rather than being
    the difference between two numbers nobody compared.
    """
    wanted = max(0, int(max_tokens or 0))
    budget = float(asked_for)
    try:
        from core.brain.llm.mlx_client import observed_rates, time_a_prompt_needs

        reading = time_a_prompt_needs(len(str(prompt or "")), 0)
        writing_rate = float(observed_rates().get("decode") or 0.0)
    except (ImportError, AttributeError, TypeError, ValueError):
        return budget, wanted
    if writing_rate <= 0.0 or not wanted:
        return budget, wanted

    # Long enough to read the question and write something worth reading back.
    least = min(
        float(InferenceGate._MAX_REQUEST_TIMEOUT_S), reading + A_REAL_ANSWER / writing_rate
    )
    budget = max(budget, least)

    # A protected length buys time rather than being shortened.
    protected = max(0, int(floor or 0))
    needed_for_floor = reading + protected / writing_rate if protected else 0.0
    if protected:
        budget = min(
            float(InferenceGate._MAX_REQUEST_TIMEOUT_S), max(budget, needed_for_floor)
        )

    # And a length that fits what is left after the reading.
    affordable = int(max(0.0, budget - reading) * writing_rate)
    if protected and budget >= needed_for_floor:
        # The clock was stretched to fit the floor, so the floor fits. Deriving
        # the count back out of the stretched time loses a token to binary
        # rounding and reports a protected floor as unaffordable by one.
        affordable = max(affordable, protected)
    if affordable < wanted:
        wanted = max(A_REAL_ANSWER, affordable)
    if protected and wanted < protected:
        # The longest clock this gate allows still cannot afford the floor.
        # Cutting is the only option left, and saying so is the difference
        # between a measured limit and a protection that quietly did nothing.
        logger.warning(
            "🕝 A protected answer floor of %d tokens does not fit the longest "
            "request clock (%.0fs reading, %.1f tok/s); cutting to %d.",
            protected,
            reading,
            writing_rate,
            wanted,
        )
    return budget, wanted


#: What the request needs beyond decoding: dispatch, validation and delivery.
#: Taken from the gate's own existing reserve for the same work, where the
#: primary attempt is given the request timeout less this.
_DELIVERY_MARGIN_S = 4.0


def _seconds_to_read(prompt_chars: int) -> float:
    """How long this prompt takes to READ, before a token is decoded.

    The answer clock counted decoding and nothing else, so a turn was given
    time to say its answer and none to read the question. On the resident model
    a six-kilobyte prompt takes about two minutes to prefill; the whole turn
    was 148 seconds.

    LIVE, 2026-08-28: three files read, then the answer generation cancelled
    at 119.5 seconds of a 120-second prefill, having produced no tokens at all.
    Everything the loop found was correct and the turn had never been given
    time to look at it.
    """

    chars = max(0, int(prompt_chars or 0))
    if not chars:
        return 0.0
    try:
        from core.brain.llm.thinking_reserve import seconds_to_read

        return max(0.0, float(seconds_to_read(chars)))
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0.0


def _seconds_to_decode(tokens: int, model: str = "") -> float:
    """How long this budget takes at the measured rate, or 0.0 if unmeasured.

    ``model`` because the rate belongs to the model. Sizing a 27B's clock on
    readings a 9B produced is what aborted three generations on one question,
    each after more than two and a half minutes of work.
    """

    try:
        from core.brain.llm.thinking_reserve import seconds_to_decode

        return float(seconds_to_decode(tokens, model))
    except (ImportError, TypeError, ValueError):
        return 0.0


class InferenceGate:
    """Isolated inference gateway for Aura's managed local runtime."""

    # Class-level defaults for observation-path cooldowns so partially
    # constructed instances (test doubles via __new__, hot-reload edges) can
    # never crash the status/recovery path with AttributeError.
    _closing_clients: set[Any] = set()
    _last_status_recovery_schedule_at: float = 0.0
    _last_cortex_policy_deferred_log_at: float = 0.0
    _last_stale_reset_log_at: float = 0.0
    _last_forced_warmup_override_log_at: float = 0.0
    #: The lock the status/recovery path takes. The defaults above exist so a
    #: partially constructed instance cannot crash that path with an
    #: AttributeError, and this was the one it actually takes — omitted, so
    #: _schedule_status_recovery raised "'InferenceGate' object has no
    #: attribute '_status_recovery_lock'" on any instance built through
    #: __new__ (test doubles, and the hot-reload edge the comment above names).
    #:
    #: A class-level lock is the right default rather than None: the paths that
    #: take it are guarding a scheduling decision, and a missing lock must
    #: serialise them, not skip the guard.
    _status_recovery_lock: Any = _threading.RLock()

    def __init__(self, orch=None):
        self.orch = orch
        self._created_at = time.monotonic()
        self._mlx_client = None
        self._initialized = False
        self._init_error = None
        self._cached_identity_prompt: str | None = None
        self._identity_prompt_time: float = 0.0
        self._identity_prompt_state_key: tuple[Any, ...] | None = None
        self._cortex_recovery_in_progress: bool = False
        self._last_cortex_check: float = 0.0
        self._cortex_recovery_attempts: int = 0
        self._cortex_recovery_exhausted_at: float = 0.0  # [STABILITY v53]
        # Warmup backoff: a cortex load that keeps exceeding its deadline under
        # thermal throttle / GPU contention gets force-killed and re-spawned,
        # thrashing the single GPU slot and starving the foreground fallback
        # that's serving the turn (2026-07-15 soak: 210s walls). After repeated
        # stuck-cortex kills, cool down and let the resident fallback carry
        # smoothly until thermal recovers, then take one clean reload shot.
        self._cortex_stuck_kill_times: deque[float] = deque(maxlen=16)
        self._cortex_warmup_backoff_until: float = 0.0
        self._cortex_warmup_backoff_streak: int = 0
        #: Load setbacks by kind; see cortex_load_setbacks().
        self._cortex_load_setback_counts: dict[str, int] = {}
        #: Who currently owns rewriting Cortex lane state, and when.
        self._lane_transition_owner: str = ""
        self._lane_transition_at: float = 0.0
        #: Cancelled loads not yet observed to stop.
        self._abandoned_cortex_loads: list[Any] = []
        #: Last shed paid for by a promotion inferred from prompt text.
        self._last_heuristic_shed_at: float = 0.0
        #: Warmup deferrals by cause; read via warmup_deferral_receipt().
        self._warmup_deferral_counts: dict[str, Any] = {}
        #: Guards the status-triggered recovery cooldown stamp.
        self._status_recovery_lock = checked_lock(
            "inference_gate.status_recovery_cooldown", rank=LockRank.LEAF
        )
        self._last_stale_reset_log_at: float = (
            0.0  # [HARDENING v54] Rate-limit stale state warnings
        )
        # 0.0 means "no successful generation yet" — a fresh gate must not
        # report a recent success it never produced. Consumers derive ages
        # from _constructed_wall_at when this is unset.
        self._last_successful_generation_at: float = 0.0
        #: Last living-mind assembly's receipt; read via
        #: living_mind_context_receipt().
        self._living_mind_receipt: Any = None
        #: Last dispatch's token-fit outcome; read via prompt_fit_receipt().
        self._prompt_fit_receipt: dict[str, Any] = {}
        #: Drafts handed downstream for repair, keyed by obligation id.
        self._repair_obligations: dict[str, Any] = {}
        #: True while the last recorded endpoint is a draft awaiting repair.
        self._last_user_generation_provisional = False
        #: Filled by every tier-health sweep; read via tier_health_receipt().
        self._tier_health_receipt: dict[str, Any] = {}
        self._constructed_wall_at: float = time.time()
        self._prewarm_task: asyncio.Task | None = None
        self._deferred_prewarm_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._status_recovery_task: asyncio.Task | None = None
        self._foreground_ready_lock = _threading.Lock()
        # Cortex readiness only proves that the resident decoder can accept a
        # request.  A real chat turn also needs the profile, unified-self,
        # self-condition, and semantic-memory readers.  Those used to be
        # constructed lazily inside the first foreground request, so health
        # advertised READY while the user still paid tens of seconds of boot
        # work.  The server owns this second readiness phase and reports it
        # through this gate.
        # ``None`` means the server has not bound the foreground dependency
        # owner yet.  This keeps the model lane independently usable in
        # headless/proof processes that do not host the desktop chat surface.
        self._chat_dependencies_snapshot: tuple[bool | None, str] = (None, "")
        self._last_background_memory_shed_at: float = 0.0
        self._last_spare_maintenance_at: float = 0.0
        self._last_cortex_warmup_deferral_log_at: float = 0.0
        self._last_cortex_policy_deferred_log_at: float = 0.0
        self._last_status_recovery_schedule_at: float = 0.0
        self._last_user_generation_endpoint: str | None = None
        self._last_user_generation_at: float = 0.0
        self._last_user_generation_used_fallback: bool = False
        self._last_generation_metadata: dict[str, Any] = {}
        self._last_surface_control_receipt: dict[str, Any] = {}
        self._diagnostic_metadata_lock = checked_lock(
            "inference_gate.diagnostic_metadata", rank=LockRank.LEAF
        )
        # Clients whose async close is still in flight. Holding the handle is
        # the difference between "shutting down" and "orphaned": once the
        # last reference goes, nothing can reap a close that hangs.
        self._closing_clients: set[Any] = set()
        self._generation_metadata_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"aura_inference_gate_generation_metadata_{id(self)}",
                default=None,
            )
        )
        self._surface_control_receipt_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"aura_inference_gate_surface_receipt_{id(self)}",
                default=None,
            )
        )
        type(self)._instance_ref = weakref.ref(self)
        logger.info("🛡️ InferenceGate created.")

    def set_chat_dependencies_ready(
        self,
        ready: bool,
        *,
        blocker: str = "chat_dependencies_warming",
    ) -> None:
        """Publish whether non-model foreground dependencies are materialized."""

        # Foreground warmup can hold its operation lock while awaiting these
        # dependencies. Publish one immutable value without joining that wait.
        ready = bool(ready)
        reason = "" if ready else (
            str(blocker or "chat_dependencies_warming").strip()[:160]
            or "chat_dependencies_warming"
        )
        self._chat_dependencies_snapshot = (ready, reason)

    def get_cortex_readiness_status(self) -> dict[str, Any]:
        """Read model-lane readiness without the chat-dependency overlay.

        Boot warmup must wait for the resident model before loading dependent
        readers.  Calling :meth:`get_conversation_status` for that wait would
        be circular because that public status deliberately remains blocked
        until those readers finish.
        """

        client = self._mlx_client
        getter = (
            getattr(client, "get_lane_status", None)
            if client is not None
            else None
        )
        if not callable(getter):
            return {
                "conversation_ready": False,
                "state": "cold",
                "readiness_blockers": ["cortex_unavailable"],
            }
        try:
            candidate = getter()
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="reported Cortex unready while boot dependency warmup waited",
                severity="warning",
            )
            return {
                "conversation_ready": False,
                "state": "recovering",
                "readiness_blockers": ["cortex_status_unavailable"],
            }
        if not isinstance(candidate, Mapping):
            return {
                "conversation_ready": False,
                "state": "recovering",
                "readiness_blockers": ["cortex_status_invalid"],
            }
        blockers = [
            str(item)
            for item in (candidate.get("readiness_blockers") or ())
            if str(item or "").strip()
        ]
        return {
            "conversation_ready": bool(candidate.get("conversation_ready")) and not blockers,
            "state": str(candidate.get("state") or "cold"),
            "readiness_blockers": blockers,
        }

    def _record_user_generation_endpoint(
        self, label: str, *, provisional: bool = False
    ) -> None:
        endpoint = PRIMARY_ENDPOINT if str(label).startswith(PRIMARY_ENDPOINT) else str(label)
        self._last_user_generation_endpoint = endpoint
        self._last_user_generation_at = time.time()
        self._last_user_generation_used_fallback = endpoint != PRIMARY_ENDPOINT
        # A draft handed to downstream repair is not yet this endpoint's
        # answer. Recording it as one made a flawed draft indistinguishable
        # from a finished reply the moment it was written down.
        self._last_user_generation_provisional = bool(provisional)

    #: Drafts handed downstream for repair that nothing has come back about.
    #: A repair that never happens used to leave no trace at all: the endpoint
    #: was attributed, the flawed draft was returned, and there was no id, no
    #: hash, no acceptance and no postcondition to check against.
    def _open_repair_obligation(
        self, *, label: str, draft: str, reasons: Any
    ) -> str:
        obligation_id = f"repair_{uuid.uuid4().hex[:12]}"
        text = str(draft or "")
        record = {
            "schema": "aura.inference.repair_obligation.v1",
            "obligation_id": obligation_id,
            "endpoint": str(label),
            "draft_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "draft_chars": len(text),
            "reasons": [str(reason)[:120] for reason in (reasons or ())][:8],
            "opened_at": time.time(),
        }
        obligations = getattr(self, "_repair_obligations", None)
        if obligations is None:
            obligations = {}
            self._repair_obligations = obligations
        obligations[obligation_id] = record
        try:
            from core.runtime.turn_outcome import current_turn

            turn = current_turn()
            if turn is not None:
                turn.record_receipt("repair_obligation", record)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Repair obligation not recorded on the turn ledger: %s", exc)
        return obligation_id

    def discharge_repair_obligation(
        self, obligation_id: str, *, repaired_text: str, accepted: bool
    ) -> bool:
        """Close a repair obligation with what downstream actually produced.

        Returns False for an unknown id, which is itself worth knowing: it
        means something claims to have repaired a draft this gate never handed
        out.
        """
        obligations = getattr(self, "_repair_obligations", None) or {}
        record = obligations.pop(str(obligation_id), None)
        if record is None:
            return False
        final = str(repaired_text or "")
        record.update(
            {
                "discharged_at": time.time(),
                "accepted": bool(accepted),
                "final_sha256": hashlib.sha256(final.encode("utf-8")).hexdigest(),
                "final_chars": len(final),
                "changed": hashlib.sha256(final.encode("utf-8")).hexdigest()
                != record["draft_sha256"],
            }
        )
        try:
            from core.runtime.turn_outcome import current_turn

            turn = current_turn()
            if turn is not None:
                turn.record_receipt("repair_discharged", record)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Repair discharge not recorded on the turn ledger: %s", exc)
        if not record["changed"] and accepted:
            # Accepting the identical draft is a repair that did not happen,
            # reported as one that did.
            _record_inference_degradation(
                RuntimeError("repair accepted a byte-identical draft"),
                action="recorded a repair that produced no change to the draft",
                extra={"obligation_id": str(obligation_id)},
            )
        return True

    def open_repair_obligations(self) -> list[dict[str, Any]]:
        """Drafts handed downstream that nothing has come back about."""
        return [
            copy.deepcopy(record)
            for record in (getattr(self, "_repair_obligations", None) or {}).values()
        ]

    def _generation_metadata_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_generation_metadata_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_inference_gate_generation_metadata_{id(self)}",
                default=None,
            )
            self._generation_metadata_context = slot
        return slot

    def _surface_control_receipt_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_surface_control_receipt_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_inference_gate_surface_receipt_{id(self)}",
                default=None,
            )
            self._surface_control_receipt_context = slot
        return slot

    def _generation_metadata_sink_slot(
        self,
    ) -> ContextVar[dict[str, Any] | None]:
        """Caller-owned evidence transport across child asyncio tasks."""

        slot = getattr(self, "_generation_metadata_sink_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_inference_gate_generation_metadata_sink_{id(self)}",
                default=None,
            )
            self._generation_metadata_sink_context = slot
        return slot

    def _publish_generation_metadata(
        self,
        metadata: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        """Publish request evidence that nobody can edit after the fact.

        CP126: "Published generation metadata is only shallow-copied. Nested
        fallback chains, output contracts, receipts, reasons, and mutations
        retain shared references. Provider code or another diagnostic
        consumer can mutate already-published request evidence, while
        process-wide last-call fields are also written without
        synchronization."

        ``dict(metadata)`` copies one level. The interesting content is all
        below that level — fallback_chain, requested_output_contract,
        surface_control_receipt, mutations — so the published "evidence" and
        the provider's live working dicts were the same objects. Evidence
        that the subject of the evidence can still edit is not evidence, and
        the failure is invisible: the record reads as a normal receipt.
        """
        metadata_snapshot = _deep_snapshot(metadata)
        receipt_snapshot = _deep_snapshot(receipt)
        self._generation_metadata_slot().set(metadata_snapshot)
        self._surface_control_receipt_slot().set(receipt_snapshot)
        sink = self._generation_metadata_sink_slot().get()
        if isinstance(sink, dict):
            sink.clear()
            sink.update(_deep_snapshot(metadata_snapshot))
        # The ContextVar slots are per-task and need no lock. These two are
        # process-wide and were written from every concurrent generation, so
        # a diagnostic reader could see metadata from one request beside the
        # receipt from another.
        with self._diagnostic_metadata_lock:
            self._last_generation_metadata = metadata_snapshot
            self._last_surface_control_receipt = receipt_snapshot

    def get_last_generation_metadata(self) -> dict[str, Any]:
        task_metadata = self._generation_metadata_slot().get()
        if task_metadata is not None:
            return _deep_snapshot(task_metadata)
        return {}

    def get_diagnostic_last_generation_metadata(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        with self._diagnostic_metadata_lock:
            return _deep_snapshot(getattr(self, "_last_generation_metadata", {}) or {})

    def get_last_surface_control_receipt(self) -> dict[str, Any]:
        task_receipt = self._surface_control_receipt_slot().get()
        if task_receipt is not None:
            return _deep_snapshot(task_receipt)
        return {}

    def get_diagnostic_last_surface_control_receipt(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        with self._diagnostic_metadata_lock:
            return _deep_snapshot(
                getattr(self, "_last_surface_control_receipt", {}) or {}
            )

    def get_diagnostic_last_call(self) -> dict[str, Any]:
        """Metadata and receipt from the SAME last call, read together.

        Reading the two diagnostics separately can straddle a concurrent
        publish and pair one request's metadata with another's receipt.
        """
        with self._diagnostic_metadata_lock:
            return {
                "metadata": _deep_snapshot(
                    getattr(self, "_last_generation_metadata", {}) or {}
                ),
                "surface_control_receipt": _deep_snapshot(
                    getattr(self, "_last_surface_control_receipt", {}) or {}
                ),
            }

    def _clear_last_generation_metadata(self) -> None:
        self._publish_generation_metadata({}, {})

    def _record_client_generation_metadata(
        self,
        client: Any,
        *,
        label: str,
        success: bool,
        text: str,
        requested_max_tokens: int | None = None,
        output_contract: dict[str, Any] | None = None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> None:
        provider_metadata = (
            dict(generation_metadata) if isinstance(generation_metadata, dict) else {}
        )
        resolved_label = str(provider_metadata.get("endpoint") or label)
        metadata: dict[str, Any] = {
            "ok": bool(success),
            "endpoint": (
                PRIMARY_ENDPOINT
                if resolved_label.startswith(PRIMARY_ENDPOINT)
                else resolved_label
            ),
            "text_length": len(str(text or "").strip()),
        }
        for key in (
            "provider",
            "model",
            "is_local",
            "provider_verified",
            "fallback_chain",
            "error",
        ):
            if key in provider_metadata:
                metadata[key] = provider_metadata[key]
        if requested_max_tokens is not None:
            metadata["requested_max_tokens"] = max(1, int(requested_max_tokens))
        if isinstance(output_contract, dict) and output_contract:
            metadata["requested_output_contract"] = dict(output_contract)
        raw_provider_receipt = provider_metadata.get("surface_control_receipt")
        receipt: dict[str, Any] = (
            dict(raw_provider_receipt) if isinstance(raw_provider_receipt, dict) else {}
        )
        getter = getattr(client, "get_last_surface_control_receipt", None)
        if not receipt and callable(getter):
            try:
                raw_receipt = getter()
                if isinstance(raw_receipt, dict):
                    receipt = dict(raw_receipt)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued generation after local client surface-control receipt read failed",
                    severity="warning",
                )
                logger.debug("Surface-control receipt read failed for %s: %s", label, exc)
        if receipt:
            metadata["surface_control_receipt"] = receipt
            for source_key, metadata_key in (
                ("generation_max_tokens", "actual_max_tokens"),
                ("generated_tokens", "generated_tokens"),
                ("instruction_shape_repair_applied", "deterministic_repair_applied"),
            ):
                if source_key in receipt:
                    metadata[metadata_key] = receipt[source_key]
            # A gate that never ran is not a gate that said no.
            #
            # ``surface_quality_gate_passed`` starts as ``not enabled``, so with
            # the gate enabled and no draft ever examined it is False — and this
            # attributed EVERY failure to the quality gate: a deadline, an empty
            # generation, a cancellation. The receipt says which happened, and
            # nothing was asking.
            #
            # LIVE, 2026-08-28: "read this file and tell me what it says" ended
            # in "Cortex exhausted its worker-owned semantic quality retries",
            # on a receipt reading attempts=0, reasons=[],
            # generation_stop_reason='deadline_exceeded'. The gate had examined
            # nothing. Hours went into the quality path because the label sent
            # them there, and the real cause was printed on the same receipt.
            try:
                _gate_examined_something = int(
                    receipt.get("surface_quality_gate_attempts") or 0
                ) > 0
            except (TypeError, ValueError):
                _gate_examined_something = False
            if (
                not success
                and bool(receipt.get("surface_quality_gate_enabled"))
                and not bool(receipt.get("surface_quality_gate_passed"))
                and (
                    _gate_examined_something
                    or bool(receipt.get("surface_quality_gate_reasons"))
                )
            ):
                metadata["error"] = "surface_quality_rejected"
                raw_reasons = receipt.get("surface_quality_gate_reasons")
                if isinstance(raw_reasons, (list, tuple)):
                    metadata["failure_reasons"] = [
                        str(reason).strip()[:120]
                        for reason in raw_reasons
                        if str(reason).strip()
                    ][:8]
            elif not success and not metadata.get("error"):
                # What actually stopped it, from the receipt that knows.
                stopped = str(receipt.get("generation_stop_reason") or "").strip()
                if stopped:
                    metadata["error"] = f"generation_{stopped}"
        self._publish_generation_metadata(metadata, receipt)

    #: Kinds of refusal a caller has to be able to tell apart. A bare None
    #: told it none of this: policy deferral, a lane that could not be
    #: reached, a proof contract that names a model, and resource exhaustion
    #: all arrived as the same value the model returns when it says nothing.
    REFUSAL_DEFERRED = "deferred"
    REFUSAL_PROOF_LANE = "proof_lane_required"
    REFUSAL_RESOURCE = "resource_exhausted"
    REFUSAL_EXHAUSTED = "lanes_exhausted"
    REFUSAL_PRIVACY = "privacy_block"

    def last_refusal_receipt(self) -> dict[str, Any]:
        """Why the last generation returned nothing, if it was a refusal."""
        return copy.deepcopy(getattr(self, "_last_refusal_receipt", {}))

    def _refuse_generation(
        self,
        kind: str,
        reason: str,
        *,
        context: dict[str, Any] | None = None,
        origin: str = "",
        retry_after_s: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record WHY this turn produced nothing, then produce nothing.

        ``generate`` returns ``str | None`` and that contract is not changing
        here — every caller depends on it. What changes is that None stops
        being the whole message. The receipt goes three places: the caller's
        own context dict, the gate (for health), and the turn ledger, whose
        terminal status is then a refusal rather than a turn that mysteriously
        held no answer.

        ``retry_after_s`` is omitted rather than zeroed when it is not known:
        a zero would read as "retry immediately", which is a claim.
        """
        receipt: dict[str, Any] = {
            "schema": "aura.inference.refusal.v1",
            "kind": str(kind),
            "reason": str(reason),
            "origin": str(origin or ""),
            "at": time.time(),
        }
        if retry_after_s is not None:
            receipt["retry_after_s"] = max(0.0, float(retry_after_s))
        if detail:
            receipt["detail"] = dict(detail)
        self._last_refusal_receipt = receipt
        if isinstance(context, dict):
            context["inference_refusal"] = dict(receipt)
        # Temporal continuity anchors on inference START, which happens while
        # generation parameters are still being assembled. A turn that refuses
        # after that point had moved the anchor for an inference that never
        # ran, and the silence accumulator then measured from a moment no
        # speech ever followed. Undo it here, where every refusal now passes.
        try:
            from core.container import ServiceContainer as _Container

            _tc = _Container.get("temporal_continuity", default=None)
            if _tc is not None and hasattr(_tc, "on_inference_abandoned"):
                _tc.on_inference_abandoned(f"{kind}:{reason}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Temporal anchor not restored after refusal: %s", exc)
        try:
            from core.runtime.turn_outcome import current_turn

            turn = current_turn()
            if turn is not None:
                turn.record_receipt("inference_refusal", receipt)
                turn.record_refusal(reason=f"{kind}:{reason}", authority="inference_gate")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Refusal receipt not recorded on the turn ledger: %s", exc)
        logger.info(
            "⛔ InferenceGate refused generation: kind=%s reason=%s origin=%s",
            kind,
            reason,
            origin or "unknown",
        )
        return None

    @staticmethod
    def _abort_active_generation(client: Any, reason: str) -> None:
        """Tell the worker to stop, whatever went wrong at this boundary."""
        abort = getattr(client, "force_abort_active_generation", None)
        if not callable(abort):
            return
        try:
            abort(reason=reason)
        except _INFERENCE_RECOVERABLE_ERRORS as abort_exc:
            _record_inference_degradation(
                abort_exc,
                action="continued after local client abort hook failed",
                severity="error",
            )

    #: A request may not ask for a longer budget than this. An unvalidated
    #: caller timeout of inf produced a deadline that never expired, which is
    #: the same as having no deadline while claiming to have one.
    _MAX_REQUEST_TIMEOUT_S = 900.0

    #: How often a promotion inferred from prompt text may pay for a global
    #: shed. Derived from the quiet window it extends: one shed per window is
    #: the most that window can be worth, and a second inside it buys nothing
    #: except a reload of whatever the first one unloaded.
    _HEURISTIC_SHED_INTERVAL_S = 180.0

    def _admit_heuristic_protected_shed(self) -> bool:
        """Whether a text-inferred protected turn may shed background workers."""
        now = time.monotonic()
        last = float(getattr(self, "_last_heuristic_shed_at", 0.0) or 0.0)
        if last and (now - last) < self._HEURISTIC_SHED_INTERVAL_S:
            return False
        self._last_heuristic_shed_at = now
        return True

    @staticmethod
    def _bounded_affect(raw: Any, *, low: float, high: float) -> float | None:
        """An affect axis inside its declared range, or None if unusable.

        None is the point: it lets the caller tell a MEASURED value from the
        default it would otherwise be indistinguishable from.
        """
        value = _finite(raw)
        if value is None:
            return None
        return max(low, min(high, value))

    @staticmethod
    def _modulator_factor(raw: Any, *, source: str, low: float, high: float) -> float:
        """A multiplier from a subsystem, or 1.0 if it is not usable.

        Six modulators — homeostatic coupling, homeostasis, morphogenesis,
        plasticity, temporal continuity, somatic qualia — compose their outputs
        into the same temperature and token budget with raw arithmetic. One NaN
        anywhere makes every later comparison False and carries to the end;
        one absurd factor multiplies the budget past the window. They now share
        one contract: finite, bounded, and 1.0 when the subsystem gave
        something this code cannot use.
        """
        value = _finite(raw)
        if value is None:
            logger.debug("Ignoring non-finite modulator factor from %s: %r", source, raw)
            return 1.0
        return max(low, min(high, value))

    @staticmethod
    def _modulator_delta(raw: Any, *, source: str, limit: float) -> float:
        """An additive nudge from a subsystem, or 0.0 if it is not usable."""
        value = _finite(raw)
        if value is None:
            logger.debug("Ignoring non-finite modulator delta from %s: %r", source, raw)
            return 0.0
        return max(-limit, min(limit, value))

    @classmethod
    def _requested_timeout_s(cls, timeout: Any, default: float) -> float:
        """A usable request budget from whatever the caller passed.

        ``timeout or default`` treated any truthy value as a budget, so -1 and
        inf both bypassed the default: the first produced an already-expired
        deadline before routing, the second produced one that never expires.
        Neither is a timeout.
        """
        fallback = _finite(default, 90.0) or 90.0
        candidate = _finite(timeout)
        if candidate is None or candidate <= 0.0:
            return max(1.0, min(cls._MAX_REQUEST_TIMEOUT_S, fallback))
        return max(1.0, min(cls._MAX_REQUEST_TIMEOUT_S, candidate))

    @classmethod
    def _requested_max_tokens(cls, requested: Any, default: int) -> int:
        """A usable token budget, never a raised TypeError at the front door.

        ``int(context.get("max_tokens") or default)`` runs before the protected
        cap parser further down, so a caller passing "lots" — or a float NaN —
        aborted a user-facing generation with a conversion error rather than
        being rejected as a bad request.
        """
        fallback = max(1, int(_finite(default, 512) or 512))
        candidate = _finite(requested)
        if candidate is None or candidate <= 0.0:
            return min(cls._TOKEN_BOUND_HARD_CEILING, fallback)
        return max(1, min(cls._TOKEN_BOUND_HARD_CEILING, int(candidate)))

    @staticmethod
    def _window_within(deadline: Any, requested: float) -> float:
        """The smaller of what this attempt wants and what the request has left.

        Returns 0.0 when the request budget is gone. Callers must treat that
        as "do not start", not as "use a small window": an attempt that begins
        after the caller's deadline cannot deliver anything the caller will
        still be waiting for, and it holds the model lane while it fails.
        """
        wanted = max(0.0, float(requested or 0.0))
        remaining = getattr(deadline, "remaining", None)
        if remaining is None:
            return wanted
        return max(0.0, min(wanted, float(remaining)))

    def _response_credit_outcome(self) -> tuple[float | None, str]:
        """The credit this answer earned, or None when nothing graded it.

        Credit used to be computed from length and whether the text contained
        a newline or a list marker. That rewards a long, well-formatted
        hallucination with the maximum available score and penalizes a correct
        one-line answer — the learner was being trained on shape and told it
        was correctness.

        The turn ledger already records what verified the SERVED answer, on a
        ranked scale, so the score is that rank normalized against the top of
        the scale. Nothing invented: an externally verified answer earns full
        credit, an asserted one earns none, and an ungraded turn earns no
        entry at all.
        """
        try:
            from core.runtime.turn_outcome import VerificationGrade, current_turn
        except ImportError:
            return None, "turn_ledger_unavailable"
        turn = current_turn()
        if turn is None:
            return None, "no_turn_bound"
        try:
            grade = turn.verification_grade_so_far()
        except _INFERENCE_RECOVERABLE_ERRORS:
            return None, "grade_unreadable"
        top = max(member.rank for member in VerificationGrade)
        if grade.rank <= VerificationGrade.ASSERTED.rank or top <= 0:
            # ASSERTED means the runtime said so about itself. That is the
            # claim under test, not evidence for it.
            return None, f"ungraded:{grade.value}"
        return grade.rank / top, f"graded:{grade.value}"

    def last_plasticity_reward(self) -> dict[str, Any]:
        """What the last weight update was rewarded on, and where it came from."""
        return copy.deepcopy(getattr(self, "_last_plasticity_reward", {}))

    def last_credit_basis(self) -> str:
        """Why the last response did or did not receive credit."""
        return str(getattr(self, "_last_credit_basis", "") or "")

    def _credit_action_seq(self) -> int:
        """Monotonic per-gate counter so credit action ids never collide."""
        value = int(getattr(self, "_credit_action_counter", 0)) + 1
        self._credit_action_counter = value
        return value

    def _annotate_last_generation_metadata(self, **fields: Any) -> None:
        """Amend the just-published metadata after post-generation validation.

        Success metadata is recorded at the client boundary before integrity
        and user-facing assessment run. When those checks reject the draft the
        published record must be downgraded, otherwise a later metadata read
        treats the rejected generation as proof of a valid response.
        """
        metadata = self.get_last_generation_metadata()
        if not metadata:
            return
        metadata.update(fields)
        receipt = dict(metadata.get("surface_control_receipt") or {})
        self._publish_generation_metadata(metadata, receipt)

    @classmethod
    def _user_facing_recovery_response(cls, prompt: str) -> str:
        # [HARDENING v54] NEVER echo prompt content back to the user.
        # The prompt may contain system prompts, stale conversation history from
        # memory retrieval, or fragments from previous sessions. Echoing it back
        # fabricates hallucinated statements the user never made.
        try:
            from core.synthesis import deterministic_user_facing_floor

            direct = deterministic_user_facing_floor(prompt)
            if direct:
                return direct
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="returned deterministic recovery response without optional narrative variation",
            )
            logger.debug("Deterministic recovery response unavailable: %s", exc)
        
        # [HARDENING v57] OFFLINE RESILIENCE: Return GENUINE offline-safe response
        # instead of empty string. System must function perfectly offline without
        # cloud. This is a minimum viable response, never empty.
        try:
            from core.synthesis import generate_offline_fallback_response
            
            fallback = generate_offline_fallback_response(prompt)
            if fallback and len(str(fallback).strip()) > 0:
                return fallback
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="fell through to terminal recovery text after offline fallback generation failed",
            )
            logger.warning("Offline fallback response generation failed: %s", _exc)

        # Last resort: an honest terminal response. Both generation paths have
        # already failed at this point and no asynchronous continuation exists,
        # so the text must not promise that work is still in progress NOR fall
        # back to retry-filler ("try again"/"send your message again"), which the
        # recovery-no-echo contract forbids as non-substantive.
        return (
            "I couldn't finish generating a response just now — my language "
            "backend hit an internal problem on my side."
        )

    def _stabilize_user_facing_text(
        self,
        text: str,
        prompt: str,
        *,
        is_user_facing: bool,
    ) -> str:
        if not is_user_facing:
            return str(text or "").strip()
        original = str(text or "").strip()
        # The vanilla floor. This is the last point at which the model's own
        # answer exists before a dozen layers get the chance to subtract from
        # it, so it is recorded here — whatever happens downstream, the person
        # should never receive less than the bare model would have given them.
        try:
            from core.conversation.surface_disposition import record_raw_model_draft

            record_raw_model_draft(original)
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass
        try:
            from core.synthesis import stabilize_user_facing_response

            stabilized = stabilize_user_facing_response(original, prompt)
            if stabilized != original:
                metadata = self.get_last_generation_metadata()
                if not metadata:
                    # CP126: "A fabricated unattributed metadata record can
                    # also be marked ok when repair changes text."
                    #
                    # ok=bool(stabilized) asserted success from the fact
                    # that a string was non-empty. There is no provider, no
                    # endpoint, no request correlation here — that is the
                    # definition of unattributed, and an unattributed record
                    # cannot claim the generation succeeded. Downstream
                    # consumers read `ok` as provider-verified success.
                    metadata = {
                        "ok": False,
                        "attributed": False,
                        "endpoint": "unattributed-response-path",
                        "text_length": len(stabilized),
                        "reason": "stabilized_text_without_generation_attribution",
                    }
                    _record_inference_degradation(
                        RuntimeError(
                            "post-generation stabilization ran with no generation "
                            "metadata to attribute it to"
                        ),
                        action="published an explicitly unattributed stabilization receipt",
                        severity="warning",
                    )
                receipt = dict(metadata.get("surface_control_receipt") or {})
                append_text_mutation(
                    receipt,
                    stage="inference_gate.post_generation_stabilization",
                    method="deterministic_instruction_shape",
                    reasons=["user_output_contract"],
                    before=original,
                    after=stabilized,
                    deterministic=True,
                    authorship_effect="preserved",
                )
                metadata["surface_control_receipt"] = receipt
                metadata["text_mutations"] = list(receipt.get("text_mutations") or [])
                metadata["deterministic_repair_applied"] = bool(
                    receipt.get("deterministic_repair_applied")
                )
                metadata["post_generation_repair_applied"] = True
                self._publish_generation_metadata(metadata, receipt)
            return stabilized
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _record_inference_degradation(
                exc,
                action="returned unstabilized user-facing text after output stabilization failed",
            )
            # CP126: "Stabilization failure silently returns the unvalidated
            # original ... without degradation or a failed output-contract
            # receipt."
            #
            # Returning the raw draft is the right behaviour — the person
            # gets the model's actual answer rather than nothing. What was
            # missing is the receipt saying the output contract never ran,
            # so a reader downstream could not tell a validated response
            # from an unvalidated one.
            try:
                metadata = self.get_last_generation_metadata()
                if metadata:
                    receipt = dict(metadata.get("surface_control_receipt") or {})
                    receipt["output_contract_enforced"] = False
                    receipt["output_contract_failure"] = (
                        f"{type(exc).__name__}: {exc}"
                    )[:300]
                    metadata["surface_control_receipt"] = receipt
                    metadata["post_generation_repair_applied"] = False
                    metadata["output_contract_enforced"] = False
                    self._publish_generation_metadata(metadata, receipt)
            except _INFERENCE_RECOVERABLE_ERRORS as receipt_exc:
                logger.debug(
                    "Could not publish failed output-contract receipt: %s", receipt_exc
                )
            return original

    def _finalize_nonlocal_user_facing_text(
        self,
        text: str,
        prompt: str,
        *,
        is_user_facing: bool,
        label: str,
        max_tokens: int | None,
        output_contract: dict[str, Any] | None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Finalize cloud/recovery text without retaining stale local receipts."""

        cleaned = str(text or "").strip()
        if is_user_facing:
            # CP126: "Nonlocal output success is based only on nonempty text.
            # The nonlocal finalizer publishes success=bool(cleaned) and
            # records the supplied label as the user-generation endpoint
            # without requiring verified provider metadata, output
            # integrity, request correlation, or a post-stabilization
            # quality result."
            #
            # A provider that returns "I'm sorry, an error occurred" — or a
            # recovery string this class wrote itself — is non-empty, and
            # was therefore published as a verified success from a named
            # endpoint. Success now additionally requires the provider to
            # have said so: an explicit error field is a failure, and a
            # record with no provider attribution is unattributed rather
            # than successful.
            provider_metadata = (
                dict(generation_metadata) if isinstance(generation_metadata, dict) else {}
            )
            provider_error = str(provider_metadata.get("error") or "").strip()
            attributed = bool(
                provider_metadata.get("provider")
                or provider_metadata.get("model")
                or provider_metadata.get("endpoint")
            )
            success = bool(cleaned) and not provider_error and attributed
            # Self-authored recovery text has no provider by construction.
            #
            # Refusing to call it a verified success is the point of this
            # check and still happens below. Recording a degradation as well
            # turns a designed condition into a system fault: inference_gate
            # is fail-closed, so this warning escalated to CRITICAL, the
            # endpoint raised CRITICAL SERVICE FAILURE, the resident cortex
            # was declared dead, and every later turn found no model.
            #
            # LIVE 2026-08-19, the whole chain in one turn: attribution
            # warning -> fail-closed escalation -> cortex respawn -> the
            # amplifier returning nothing -> frustration 0.47, 0.61, 0.74,
            # 0.87 -> existential threat 0.84 -> the Will refusing every
            # tool, down to creating a screenshot directory.
            #
            # A provider that answered and did not attribute itself is a
            # different thing and still reported.
            wrote_it_ourselves = generation_metadata is None
            if cleaned and not success and not wrote_it_ourselves:
                _record_inference_degradation(
                    RuntimeError(
                        f"nonlocal text from {label!r} published without provider "
                        f"attribution (error={provider_error or 'none'})"
                    ),
                    action="published nonlocal output as unverified rather than successful",
                    severity="warning",
                )
            if not attributed:
                provider_metadata["attributed"] = False
            self._record_client_generation_metadata(
                None,
                label=label,
                success=success,
                text=cleaned,
                requested_max_tokens=max_tokens,
                output_contract=output_contract,
                generation_metadata=provider_metadata or generation_metadata,
            )
            if success:
                # Only a verified generation names an endpoint as the one
                # that served the user; an unattributed string must not.
                self._record_user_generation_endpoint(label)
        return self._stabilize_user_facing_text(
            cleaned,
            prompt,
            is_user_facing=is_user_facing,
        )

    def _publish_downstream_repair_evidence(
        self,
        assessment: Any,
    ) -> None:
        """Carry post-generation repair evidence to the continuation owner.

        The worker receipt is published before the gate performs its final
        user-facing assessment. A clipped draft could therefore be correctly
        identified and preserved here while the caller still received the old
        "complete" receipt. The chat route then started a full stabilizer
        rewrite instead of appending to the authored answer.

        This amends evidence only. It does not relabel the worker stop reason or
        claim a continuation succeeded.
        """

        reasons = tuple(
            dict.fromkeys(
                str(reason or "").strip()
                for reason in (getattr(assessment, "reasons", ()) or ())
                if str(reason or "").strip()
            )
        )
        if not reasons:
            return
        metadata = self.get_last_generation_metadata()
        if not metadata:
            return
        receipt = dict(metadata.get("surface_control_receipt") or {})
        existing_reasons = tuple(
            str(reason or "").strip()
            for reason in (receipt.get("surface_quality_gate_reasons") or ())
            if str(reason or "").strip()
        )
        merged_reasons = list(dict.fromkeys((*existing_reasons, *reasons)))[:16]
        completion_reasons = [
            reason for reason in merged_reasons if reason in COMPLETION_REASONS
        ]
        receipt["surface_quality_gate_reasons"] = merged_reasons
        if completion_reasons:
            receipt["semantic_completion_incomplete"] = True
            if "semantic_completion_satisfied" in receipt:
                receipt["semantic_completion_satisfied"] = False
            metadata["post_generation_completion_evidence"] = completion_reasons
        metadata["surface_control_receipt"] = receipt
        metadata["failure_reasons"] = merged_reasons[:8]
        metadata["post_generation_repair_expected"] = True
        self._publish_generation_metadata(metadata, receipt)

    def _repairable_user_facing_draft_for_downstream(
        self,
        text: str,
        prompt: str,
    ) -> str | None:
        """Return text unchanged when downstream response repair should own it."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        try:
            assessment = assess_user_facing_reply(prompt, cleaned)
            if assessment.retryable and _should_pass_user_facing_draft_downstream(
                cleaned,
                set(assessment.reasons or ()),
                user_prompt=prompt,
            ):
                self._publish_downstream_repair_evidence(assessment)
                return cleaned
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Repairable draft preservation check skipped: %s", exc)
        return None

    @staticmethod
    def _visible_user_prompt_from_messages(
        messages: list[dict[str, Any]] | None,
        fallback: Any,
    ) -> str:
        """Return the last actual user message from a structured prompt envelope."""
        if messages:
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "user":
                    content = str(msg.get("content", "") or "").strip()
                    if content:
                        return content
        return str(fallback or "")

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        """Callback for fire-and-forget tasks — ensures exceptions are logged."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "🚨 [STABILITY v53] Background task '%s' crashed: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    @staticmethod
    def _lane_reports_active_generation(lane: dict[str, Any] | None) -> bool:
        if not isinstance(lane, dict):
            return False
        try:
            if int(lane.get("active_generations", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return False
        blockers = {
            str(blocker or "").strip()
            for blocker in (lane.get("readiness_blockers") or [])
            if str(blocker or "").strip()
        }
        reason = str(lane.get("last_failure_reason", "") or "")
        if blockers & _ACTIVE_GENERATION_BUSY_REASONS:
            return True
        return any(token in reason for token in _ACTIVE_GENERATION_BUSY_REASONS)


    @staticmethod
    def _desktop_safe_boot_enabled() -> bool:
        """Return True only for explicit reduced recovery safe boot."""

        return desktop_safe_boot_enabled()

    @staticmethod
    def _desktop_resource_guard_enabled() -> bool:
        """Return True when the normal desktop RAM/process guard is active."""

        return desktop_resource_guard_enabled()

    @staticmethod
    def _desktop_background_local_enabled() -> bool:
        return str(
            _FLAG_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM.value()
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _env_float(
        name: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """Read a float env knob, rejecting NaN/inf and out-of-range values.

        Non-finite deadlines, thresholds, and windows disable comparisons
        downstream (NaN fails every branch, inf never expires), so a malformed
        or hostile environment value must fall back to the engineered default
        rather than silently rewriting admission or backoff policy.
        """
        try:
            value = float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(value):
            return float(default)
        if minimum is not None and value < minimum:
            return float(minimum)
        if maximum is not None and value > maximum:
            return float(maximum)
        return value

    @staticmethod
    def _cortex_worker_is_legitimately_loading(client: Any) -> bool:
        """True when the cortex worker is running because it is LOADING the
        model, not because it is wedged.

        The cascade-cleanup path force-kills a "stuck" cortex worker to free
        blocked IPC feeder threads. But a worker actively loading the ~20GB
        32B is running and NOT stuck — killing it there was a full doom loop
        (2026-07-15 soak: spawn → load → killed mid-warmup on the next turn →
        warmup_deferred → repeat, 216s/turn, zero real cortex answers for an
        hour). A worker is legitimately loading when warmup is in flight OR
        the lane is warming/recovering, AND it entered that state within a
        generous load deadline. Past the deadline a still-warming worker is
        genuinely stuck and may be killed.
        """
        if client is None:
            return False
        load_deadline_s = InferenceGate._env_float("AURA_CORTEX_LOAD_DEADLINE_S", 200.0)
        # A worker that has only just been spawned is neither wedged nor idle.
        # It is new.
        #
        # The lane bookkeeping below is set when warmup BEGINS, and there is a
        # window after the process exists where none of it is true yet. In
        # that window a running worker read as idle-but-running — the wedged
        # case — and was killed, which is the doom loop this guard was written
        # to end, reached through the one gap it did not cover. LIVE
        # 2026-08-26: spawn, "Loading model", "Model loaded", force-killed,
        # respawn, five times over, while every caller that needed her writing
        # was told "worker_not_alive" and the runtime's own health said the
        # lane was ready.
        #
        # Process creation time is the ground truth here: it cannot be unset,
        # cannot lag, and is already captured by the kill path itself.
        started_at = _worker_process_started_at(client)
        if started_at and (time.time() - started_at) < load_deadline_s:
            return True
        warming = bool(getattr(client, "_warmup_in_flight", False)) or str(
            getattr(client, "_lane_state", "")
        ) in {"warming", "recovering"}
        if not warming:
            return False
        transition_at = float(getattr(client, "_lane_transition_at", 0.0) or 0.0)
        warming_age = time.time() - transition_at if transition_at else 1e9
        return warming_age < load_deadline_s

    @staticmethod
    def _cortex_worker_is_actively_generating(client: Any) -> bool:
        """True when the worker is producing tokens right now.

        The mid-LOAD guard above closed one half of the doom loop. This is the
        other half, and it is the ~15-turn conversation ceiling: a generation
        that overran its budget got the worker force-killed, which costs a
        60-150s cold reload, which makes the NEXT turn slower, which overruns
        sooner. The 2026-07-25 probe recorded twenty
        "respawn_cortex_if_needed: cortex is dead" events across thirty turns,
        the UnitaryResponsePhase climbing 25s → 100s, and the answered rate
        falling 10/10 → 4/10 → 2/10 as it went.

        A slow worker and a wedged worker are not the same thing. Slowness is
        answered by the turn's own timeout and the fallback ladder; killing the
        lane converts one slow turn into a broken session.

        A generation that has run past AURA_CORTEX_GENERATION_DEADLINE_S is
        genuinely wedged and may still be killed.
        """
        if client is None:
            return False
        if int(getattr(client, "_active_generations", 0) or 0) <= 0:
            return False
        started_at = float(getattr(client, "_active_generation_started_at", 0.0) or 0.0)
        if not started_at:
            return True  # generating, with no clock to condemn it by
        deadline_s = InferenceGate._env_float(
            "AURA_CORTEX_GENERATION_DEADLINE_S", 600.0
        )
        return (time.time() - started_at) < deadline_s

    #: One RAM reading shared by everything that asks within this window.
    #:
    #: _cortex_warmup_admission_snapshot called psutil.virtual_memory() directly
    #: and is consulted from five sites, several of them inside the same
    #: ensure_foreground_ready call. Measured: 20 syscalls per foreground-ready
    #: check, up from 2 — on the hot path immediately before every generation.
    #:
    #: A cached snapshot already exists (get_memory_pressure_snapshot, TTL'd)
    #: and this path bypassed it, so the cheap reader was there and unused. The
    #: window is short because the decision it feeds is "is it safe to load a
    #: 32B right now": stale by a second is fine, stale by a minute is not.
    _VIRTUAL_MEMORY_MEMO_TTL_S = 0.5
    _virtual_memory_memo: tuple[float, int, Any] | None = None
    _virtual_memory_memo_lock = _threading.Lock()

    @staticmethod
    def _recent_virtual_memory() -> Any:
        """psutil.virtual_memory(), at most once per _VIRTUAL_MEMORY_MEMO_TTL_S.

        Keyed on the identity of the probe as well as the clock. A class-level
        cache with only a TTL is order-dependent: the first version of this
        shared one reading across test functions, so a test that replaced the
        probe was served the previous test's value and three of them failed.
        Keying on the callable means replacing it — which is what patching does,
        and what a runtime swapping its resource observer does — misses the memo
        instead of silently reusing a reading taken through a different probe.
        """
        probe = psutil.virtual_memory
        probe_key = id(probe)
        now = time.monotonic()
        with InferenceGate._virtual_memory_memo_lock:
            memo = InferenceGate._virtual_memory_memo
            if (
                memo is not None
                and memo[1] == probe_key
                and (now - memo[0]) <= InferenceGate._VIRTUAL_MEMORY_MEMO_TTL_S
            ):
                cached = memo[2]
                if isinstance(cached, BaseException):
                    raise cached
                return cached
        try:
            reading: Any = probe()
        except (OSError, RuntimeError, ValueError) as exc:
            # A BROKEN probe is remembered too, for the same window.
            #
            # Otherwise every caller re-attempts a syscall that has just
            # failed: measured at 20 raising probes in one
            # ensure_foreground_ready. "The probe is not answering" is as
            # valid a reading as a number, and half a second of staleness on
            # it costs nothing while re-asking twenty times costs the hot path
            # before every generation.
            with InferenceGate._virtual_memory_memo_lock:
                InferenceGate._virtual_memory_memo = (now, probe_key, exc)
            raise
        with InferenceGate._virtual_memory_memo_lock:
            InferenceGate._virtual_memory_memo = (now, probe_key, reading)
        return reading

    @staticmethod
    def _cortex_warmup_admission_snapshot(context: str = "background") -> dict[str, Any]:
        """Return whether a cold Cortex load is safe under current RAM pressure.

        The normal foreground headroom check is intentionally permissive because
        a *resident* Cortex can keep answering while RAM is high. A cold 32B
        load is different: it adds tens of GB of unified-memory pressure in one
        burst. This snapshot is therefore stricter and is used before any
        background/recovery/foreground warmup that would spawn the Cortex worker.
        
        [HARDENING v57-CORTEX] PRIORITY: 32B cortex is PRIMARY model. Must be less
        deferent to memory pressure to ensure system works regardless of cloud.
        """
        context_key = str(context or "background").strip().upper()
        try:
            vm = InferenceGate._recent_virtual_memory()
            total_gb = float(vm.total) / float(1024**3)
            available_gb = float(vm.available) / float(1024**3)
            pressure_pct = float(vm.percent)

            if total_gb >= 60.0:
                # Cold-loading Cortex is a host-survival decision, not a normal
                # generation decision. The 32B lane is the user-facing default,
                # but it must not be admitted while macOS is close to swap/jetsam.
                default_max_pressure = 72.0 if context_key == "FOREGROUND" else 58.0
                default_min_available = 20.0 if context_key == "FOREGROUND" else 26.0
            else:
                default_max_pressure = 68.0 if context_key == "FOREGROUND" else 54.0
                default_min_available = 14.0 if context_key == "FOREGROUND" else 18.0

            max_pressure = InferenceGate._env_float(
                f"AURA_CORTEX_{context_key}_WARMUP_MAX_PRESSURE_PCT",
                InferenceGate._env_float(
                    "AURA_CORTEX_COLD_WARMUP_MAX_PRESSURE_PCT",
                    default_max_pressure,
                ),
            )
            min_available = InferenceGate._env_float(
                f"AURA_CORTEX_{context_key}_WARMUP_MIN_AVAILABLE_GB",
                InferenceGate._env_float(
                    "AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB",
                    default_min_available,
                ),
            )
            can_admit = bool(pressure_pct < max_pressure and available_gb >= min_available)
            reason = ""
            if not can_admit:
                reason = (
                    f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
                    f"(need <{max_pressure:.1f}% and >={min_available:.1f}GB)"
                )
            return {
                "context": str(context or "background"),
                "pressure_pct": pressure_pct,
                "available_gb": available_gb,
                "total_gb": total_gb,
                "max_pressure_pct": max_pressure,
                "min_available_gb": min_available,
                "can_admit": can_admit,
                "reason": reason,
                "measured": True,
                "schema": ADMISSION_SNAPSHOT_SCHEMA,
                "measured_at_monotonic": time.monotonic(),
            }
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Cortex warmup memory probe failed: %s", exc)
            force_warmup = str(
                _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()
            ).strip().lower() in {"1", "true", "yes", "on"}
            # measured=False marks every numeric field below as UNKNOWN, not a
            # real observation — consumers must not treat these zeros as a
            # calm-memory measurement.
            return {
                "context": str(context or "background"),
                "pressure_pct": 0.0,
                "available_gb": 0.0,
                "total_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": force_warmup,
                "reason": (
                    "memory_probe_failed_forced_override"
                    if force_warmup
                    else "memory_probe_failed"
                ),
                "measured": False,
                "schema": ADMISSION_SNAPSHOT_SCHEMA,
                "measured_at_monotonic": time.monotonic(),
            }

    #: What actually happened to a load attempt. Both defer warmup — the GPU
    #: thrash is the same either way — but they are not the same event, and
    #: recording an overrun as a kill puts a process termination in the record
    #: that never occurred. CP126 d25f5f0a: repeated slow-but-LIVE loads armed
    #: kill-based backoff on evidence of a kill nobody performed.
    LOAD_SETBACK_KILL = "stuck_load_kill"
    LOAD_SETBACK_OVERRUN = "warmup_budget_overrun"

    def cortex_load_setbacks(self) -> dict[str, int]:
        """Counts by kind, so "we killed it twice" and "it was slow twice" are
        distinguishable in the record."""
        return dict(getattr(self, "_cortex_load_setback_counts", {}) or {})

    #: How long a lane-transition lease is honoured before it is considered
    #: abandoned. Long enough for a cancel-and-verify pass, short enough that a
    #: crashed holder cannot wedge recovery for good.
    _LANE_TRANSITION_LEASE_S = 30.0

    def _claim_lane_transition(self, owner: str) -> bool:
        """Take the right to rewrite Cortex lane state. False means someone has it.

        The watchdog, the status path and the recovery scheduler all reach into
        the client's private ``_warmup_in_flight``, cancel the prewarm task and
        call private lane-state setters. None of them took anything first, so
        two of them could do it at once: one clears the flag while the other is
        mid-cancel, and a fresh warmup starts underneath a load that has not
        stopped.

        Deliberately not a lock: two of the three callers are synchronous and
        one is on the event loop, so a lock here would either block the loop or
        not be honoured. A compare-and-set with an expiry gives the same
        exclusion without either.
        """
        now = time.monotonic()
        held_by = getattr(self, "_lane_transition_owner", "")
        held_at = float(getattr(self, "_lane_transition_at", 0.0) or 0.0)
        if held_by and (now - held_at) < self._LANE_TRANSITION_LEASE_S:
            logger.debug(
                "Lane transition for %s refused; %s holds the lease (%.1fs old).",
                owner,
                held_by,
                now - held_at,
            )
            return False
        if held_by:
            logger.warning(
                "🔍 Lane-transition lease from %s expired after %.0fs; %s is taking it.",
                held_by,
                now - held_at,
                owner,
            )
        self._lane_transition_owner = str(owner)
        self._lane_transition_at = now
        return True

    def _release_lane_transition(self, owner: str) -> None:
        if getattr(self, "_lane_transition_owner", "") == str(owner):
            self._lane_transition_owner = ""
            self._lane_transition_at = 0.0

    def _clear_wedged_cortex_warmup(self, reason: str, *, owner: str) -> dict[str, Any]:
        """Clear a wedged warmup flag and cancel its load, under one owner.

        Cancelling a task is a REQUEST. The old code cancelled, set
        ``_prewarm_task = None`` and moved on, so a load that had not yet
        noticed the cancellation became invisible — and the next warmup started
        on top of it, two 20GB loads competing for one GPU slot. The task is
        kept here instead of dropped; :meth:`await_abandoned_cortex_loads`
        proves it stopped, and warmup admission refuses while one is unproven.
        """
        receipt: dict[str, Any] = {
            "reason": str(reason),
            "owner": str(owner),
            "cleared_warmup_flag": False,
            "cancelled_prewarm": False,
            "at": time.time(),
        }
        if not self._claim_lane_transition(owner):
            receipt["refused"] = "lane_transition_held"
            return receipt
        try:
            client = self._mlx_client
            if client is not None and getattr(client, "_warmup_in_flight", False):
                client._warmup_in_flight = False
                receipt["cleared_warmup_flag"] = True
            task = getattr(self, "_prewarm_task", None)
            if task is not None and not task.done():
                task.cancel()
                receipt["cancelled_prewarm"] = True
                abandoned = getattr(self, "_abandoned_cortex_loads", None)
                if abandoned is None:
                    abandoned = []
                    self._abandoned_cortex_loads = abandoned
                abandoned.append(task)
            self._prewarm_task = None
        finally:
            self._release_lane_transition(owner)
        return receipt

    def unproven_cortex_loads(self) -> int:
        """Cancelled loads that have not been observed to stop."""
        return len(
            [
                task
                for task in (getattr(self, "_abandoned_cortex_loads", None) or [])
                if not task.done()
            ]
        )

    async def await_abandoned_cortex_loads(self, timeout: float = 10.0) -> dict[str, Any]:
        """Wait for cancelled loads to actually finish, and say if they did not."""
        abandoned = list(getattr(self, "_abandoned_cortex_loads", None) or [])
        if not abandoned:
            return {"awaited": 0, "still_running": 0}
        # asyncio.wait, NOT wait_for: on timeout wait_for CANCELS what it is
        # waiting on, and the case this method exists to detect is a load that
        # ignores cancellation — so the cleanup would wait forever on the one
        # task it was written to notice. wait observes and returns.
        await asyncio.wait(abandoned, timeout=max(0.1, float(timeout)))
        still_running = [task for task in abandoned if not task.done()]
        self._abandoned_cortex_loads = still_running
        if still_running:
            _record_inference_degradation(
                TimeoutError(
                    f"{len(still_running)} cancelled Cortex load(s) did not stop within {timeout:.0f}s"
                ),
                action="refused to certify that a cancelled model load had stopped",
                severity="error",
                extra={"still_running": len(still_running)},
            )
        return {"awaited": len(abandoned), "still_running": len(still_running)}

    def _note_cortex_warmup_overrun(self) -> None:
        """A load exceeded its budget and was LEFT RUNNING. Not a kill."""
        self._note_cortex_load_setback(self.LOAD_SETBACK_OVERRUN)

    def _note_cortex_stuck_kill(self) -> None:
        """A stuck load was force-killed and reaped."""
        self._note_cortex_load_setback(self.LOAD_SETBACK_KILL)

    def _note_cortex_load_setback(self, kind: str) -> None:
        """Record a load setback and arm a warmup cooldown once they cluster.

        Each kill means a load attempt exceeded the deadline (thermal throttle /
        GPU contention) and got reaped. Re-spawning immediately just repeats the
        thrash, and every repeat grabs the single GPU slot for a 20GB weight
        load, starving the foreground fallback that is actually serving the turn.
        After ``AURA_CORTEX_STUCK_KILL_THRESHOLD`` kills inside a rolling window
        we cool down for an escalating interval, during which warmup is deferred
        and the resident fallback carries smoothly until thermal recovers.
        """
        now = time.monotonic()
        window = InferenceGate._env_float("AURA_CORTEX_STUCK_KILL_WINDOW_S", 300.0)
        threshold = max(1, int(InferenceGate._env_float("AURA_CORTEX_STUCK_KILL_THRESHOLD", 2.0)))
        counts = getattr(self, "_cortex_load_setback_counts", None)
        if counts is None:
            counts = {}
            self._cortex_load_setback_counts = counts
        counts[str(kind)] = counts.get(str(kind), 0) + 1
        self._cortex_stuck_kill_times.append(now)
        recent = [t for t in self._cortex_stuck_kill_times if now - t <= window]
        if len(recent) < threshold:
            return
        base = InferenceGate._env_float("AURA_CORTEX_WARMUP_BACKOFF_S", 90.0)
        cap = InferenceGate._env_float("AURA_CORTEX_WARMUP_BACKOFF_CAP_S", 240.0)
        self._cortex_warmup_backoff_streak += 1
        cooldown = min(cap, base * self._cortex_warmup_backoff_streak)
        self._cortex_warmup_backoff_until = now + cooldown
        logger.warning(
            "🧊 [CORTEX BACKOFF] %d load setbacks in %.0fs — deferring warmup %.0fs so the "
            "resident fallback carries and thermal recovers before the next reload shot.",
            len(recent),
            window,
            cooldown,
        )

    def _cortex_warmup_backoff_reason(self) -> str | None:
        """Non-None while a post-thrash warmup cooldown is active."""
        backoff_until = float(
            getattr(self, "_cortex_warmup_backoff_until", 0.0) or 0.0
        )
        remaining = backoff_until - time.monotonic()
        if remaining <= 0.0:
            return None
        return f"warmup_backoff:{remaining:.0f}s"

    def _reset_cortex_warmup_backoff(self) -> None:
        """Clear the cooldown after the cortex proves it can serve again."""
        kill_times = getattr(self, "_cortex_stuck_kill_times", None)
        if getattr(self, "_cortex_warmup_backoff_until", 0.0) or kill_times:
            if kill_times is not None:
                kill_times.clear()
            self._cortex_warmup_backoff_until = 0.0
            self._cortex_warmup_backoff_streak = 0

    _FORCE_WARMUP_FLAG = "AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE"

    def _cortex_warmup_deferral_reason(self, context: str = "background") -> str | None:
        # The un-forced verdict: warmup backoff, then measured memory admission.
        # A probe failure (measured=False) is treated as a deferral here — the
        # snapshot only turns can_admit True on an unmeasured probe when the
        # force flag is set, and that emergency path is decided below, never by
        # a silent can_admit.
        backoff = self._cortex_warmup_backoff_reason()
        snapshot = self._cortex_warmup_admission_snapshot(context)
        measured = bool(snapshot.get("measured", True))
        if backoff is not None:
            normal_reason: str | None = backoff
        elif not measured:
            normal_reason = "memory_probe_failed"
        elif not snapshot["can_admit"]:
            normal_reason = str(snapshot["reason"] or "memory_pressure")
        else:
            normal_reason = None

        if normal_reason is None:
            return None  # Admission already allows warmup; no override needed.

        force_requested = str(
            os.environ.get(self._FORCE_WARMUP_FLAG, "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not force_requested:
            return normal_reason

        # An override was requested to bypass a real deferral. It may skip the
        # soft admission thresholds and warmup backoff, but never the
        # host-survival floor: a cold 32B load into single-digit free GB risks
        # jetsam/swap-death of the whole process tree, which no operator
        # override should authorize.
        hard_floor_gb = self._env_float(
            "AURA_FORCE_CORTEX_WARMUP_HARD_FLOOR_GB", 10.0, minimum=4.0
        )
        available_gb = float(snapshot.get("available_gb", 0.0) or 0.0)
        if not measured:
            # The probe failed, so the survival floor cannot be confirmed. A
            # blind ~20GB cold load under the override is exactly the host-death
            # risk the floor exists to prevent — with no measurement there is no
            # boundary, so we fail closed rather than authorize an unbounded
            # load. The override takes effect again once the probe reports a
            # real above-floor reading.
            return "forced_warmup_denied_survival_floor_unmeasured"
        if available_gb < hard_floor_gb:
            return (
                "forced_warmup_denied_survival_floor:"
                f"{available_gb:.1f}GB"
                f"<{hard_floor_gb:.1f}GB"
            )

        # Past the inviolable floor, the override is a bounded, receipted
        # decision — not a permanent setting. It expires on its own, caps how
        # many bypasses one flag can authorize, and leaves a GovernanceReceipt
        # for each use, exactly as the MLX client governs the same flag.
        from core.brain.llm.emergency_override import consume_override

        decision = consume_override(
            self._FORCE_WARMUP_FLAG,
            guard=f"cortex_warmup_admission:{context}",
            observed=f"{normal_reason} (available={available_gb:.1f}GB)",
        )
        if not decision.active:
            # Expired or budget-exhausted: the memory guard is re-armed and the
            # normal deferral stands until the operator renews the decision.
            return normal_reason

        now = time.monotonic()
        last_log = getattr(self, "_last_forced_warmup_override_log_at", 0.0)
        if (now - last_log) > 60.0:
            self._last_forced_warmup_override_log_at = now
            logger.warning(
                "⚠️ %s active — bypassing %s warmup admission "
                "(available=%.1fGB, survival floor %.1fGB, %s).",
                self._FORCE_WARMUP_FLAG,
                context,
                available_gb,
                hard_floor_gb,
                decision.as_detail(),
            )
        return None

    def _log_cortex_warmup_deferral(self, reason: str, *, context: str) -> None:
        # COUNT every deferral, log a coalesced sample.
        #
        # A deferral is the ladder deciding not to load a tier, which is
        # designed backpressure and not a fault — recording it as a degradation
        # on this fail-closed subsystem escalates it to CRITICAL, and the
        # 2026-07-18 soak produced 52 of those from healthy deferrals. But a
        # runtime that cannot warm its primary lane IS something health should
        # be able to see, and a coalesced log line is not evidence. The counter
        # is the durable half; it reaches the conversation status snapshot.
        counters = getattr(self, "_warmup_deferral_counts", None)
        if counters is None:
            counters = {}
            self._warmup_deferral_counts = counters
        key = f"{context}:{reason}"
        entry = counters.get(key)
        if entry is None:
            entry = {"count": 0, "first_at": time.time(), "last_at": 0.0}
            counters[key] = entry
        entry["count"] += 1
        entry["last_at"] = time.time()

        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_warmup_deferral_log_at", 0.0)
        if (now - last_log) < 30.0:
            return
        self._last_cortex_warmup_deferral_log_at = now
        logger.warning(
            "⏸️ Cortex %s warmup deferred to protect RAM: %s (%d so far)",
            context,
            reason,
            entry["count"],
        )

    def warmup_deferral_receipt(self) -> dict[str, Any]:
        """Every warmup deferral this process has taken, by cause.

        Deliberately not degradation records — see above — but durable, so a
        primary lane that has been refused a hundred times is a number
        somebody can find rather than a log line that scrolled.
        """
        return copy.deepcopy(getattr(self, "_warmup_deferral_counts", {}) or {})

    # Admission/backoff outcomes are the ladder DECIDING not to load a tier
    # right now — the designed backpressure that lets a lower rung serve the
    # turn. They are not faults: on the fail-closed inference_gate a
    # degradation record escalates to CRITICAL SERVICE FAILURE, and the
    # 2026-07-18 soak logged 52 of them from healthy deferrals alone. That
    # noise burns the SLO error budget, raises `critical_incident_active`
    # against an otherwise-serving runtime, and buries real criticals.
    _EXPECTED_BACKPRESSURE_MARKERS = (
        "foreground_warmup_deferred",
        "warmup_deferred",
        "model_load_admission_denied",
        "resource_busy",
        "resource_timeout",
        "spawn_gate_timeout",
        "crash_loop_backoff",
        "warmup_backoff",
    )

    @classmethod
    def _is_expected_inference_backpressure(cls, exc: BaseException) -> bool:
        """True when a tier declined to run and the ladder can still serve.

        The distinguishing question is never 'did something go wrong?' but
        'did the runtime DECIDE this, and is a lower rung still available?'.
        A decision the system made on purpose must be observable (info +
        lane state + receipts) without being counted as a service failure.
        """
        text = str(exc or "")
        return any(marker in text for marker in cls._EXPECTED_BACKPRESSURE_MARKERS)

    def _note_foreground_warmup_failure(self, warmup_exc: BaseException) -> bool:
        """Classify a foreground-warmup failure; returns True for RAM deferrals.

        A ``foreground_warmup_deferred`` outcome is expected RAM-admission
        backpressure — the turn reroutes to the fallback tier, so it is logged
        at info and NOT recorded as a degradation: on the fail-closed
        inference_gate a degradation record raises CRITICAL SERVICE FAILURE
        out of the handler and kills the protected recovery lane (seen live
        July 8: one memory deferral cascaded into chat 503s). Same discipline
        as the timeout demotion in core/runtime/errors.py. Genuine warmup
        faults keep the full degradation record.
        """
        if "foreground_warmup_deferred" in str(warmup_exc):
            logger.info(
                "🧠 Foreground warmup deferred by RAM admission; rerouting this turn: %s",
                warmup_exc,
            )
            return True
        record_degradation(
            "inference_gate",
            warmup_exc,
            severity="degraded",
            action="skipped cold primary attempt or fell back after foreground warmup failure",
        )
        from core.runtime.errors import describe_error

        logger.warning(
            "🧠 Foreground preflight warmup did not complete cleanly: %s",
            describe_error(warmup_exc),
        )
        return False

    def _log_cold_cortex_policy_deferred(self) -> None:
        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_policy_deferred_log_at", 0.0)
        if (now - last_log) < 300.0:
            return
        self._last_cortex_policy_deferred_log_at = now
        logger.info(
            "Cold-start Cortex recovery deferred by desktop prewarm policy; "
            "foreground demand will warm the lane when needed."
        )

    @staticmethod
    def _boot_should_eager_warmup() -> bool:
        """Keep the resident Cortex warm on high-memory desktops unless disabled."""
        if str(_FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        if InferenceGate._desktop_resource_guard_enabled():
            logger.info(
                "🛡️ Desktop resource guard active — skipping eager %s warmup during launch.",
                _primary_lane_label(),
            )
            return False
        setting = str(_FLAG_EAGER_CORTEX_WARMUP.value()).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            if not snapshot["can_admit"] and str(
                _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                logger.warning(
                    "⏸️ Explicit eager Cortex warmup deferred to protect RAM: %s", snapshot["reason"]
                )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False

        try:
            vm = InferenceGate._recent_virtual_memory()
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            min_total_gb = float(_FLAG_BOOT_WARMUP_MIN_TOTAL_GB.value())
            if (vm.total / float(1024**3)) < min_total_gb or not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Deferring eager %s warmup at boot "
                    "(total=%.1fGB pressure=%.1f%% available=%.1fGB).",
                    _primary_lane_label(),
                    snapshot["total_gb"],
                    snapshot["pressure_pct"],
                    snapshot["available_gb"],
                )
                return False
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="kept conservative boot warmup decision after desktop policy probe failed",
            )
            logger.debug("Boot warmup memory probe failed: %s", exc)
            return False

        return True

    @staticmethod
    def _boot_should_schedule_deferred_prewarm() -> bool:
        explicit_setting = _FLAG_DEFERRED_CORTEX_PREWARM.value()
        setting = str(explicit_setting if explicit_setting is not None else "auto").strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")
            if not snapshot["can_admit"] and str(
                _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                global _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT
                global _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON
                now = time.monotonic()
                reason = str(snapshot["reason"] or "memory_pressure")
                if (
                    reason != _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON
                    or (now - _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT)
                    >= _EXPLICIT_DEFERRED_PREWARM_REFUSAL_LOG_INTERVAL_S
                ):
                    _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT = now
                    _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON = reason
                    logger.warning(
                        "⏸️ Explicit deferred Cortex prewarm refused to protect RAM: %s",
                        reason,
                    )
                else:
                    logger.debug(
                        "Explicit deferred Cortex prewarm still refused to protect RAM: %s",
                        reason,
                    )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False
        if InferenceGate._desktop_safe_boot_enabled():
            if explicit_setting is None:
                logger.info(
                    "🛡️ Recovery safe boot active — skipping implicit deferred %s "
                    "prewarm during launch.",
                    _primary_lane_label(),
                )
                return False
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")
            if not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Recovery safe-boot deferred Cortex prewarm deferred to protect RAM: %s",
                    snapshot["reason"],
                )
                return False
            return True
        return True

    @staticmethod
    def _cortex_already_resident() -> bool:
        """True when the conversation model is loaded and has served a turn.

        Deliberately conservative in both directions. It requires evidence that
        the model is actually up — a lane that merely intends to load does not
        count — and any failure to determine that answers False, which keeps
        the stricter load-sized floor rather than relaxing it on a guess.
        """
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.peek("inference_gate", default=None)
            if gate is None:
                return False
            lane = gate.get_conversation_status()
        except _INFERENCE_RECOVERABLE_ERRORS:
            # A probe must never break admission; unknown residency keeps the
            # stricter load-sized floor.
            return False
        if not isinstance(lane, dict):
            return False
        try:
            if not bool(lane.get("conversation_ready")):
                return False
            # "Ready" without a completed generation is an intention, not a
            # residency: the weights may still be streaming in.
            return bool(lane.get("has_generated_successfully"))
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _headroom_snapshot(requested_tier: str = "primary") -> dict[str, Any]:
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            total_gb = float(snapshot.total_gb)
            available_gb = float(snapshot.available_gb)
            pressure_pct = float(snapshot.pressure_pct)
            process_rss_gb = float(snapshot.process_rss_gb)
            process_rss_limit_gb = float(snapshot.process_rss_limit_gb)
            tier = str(requested_tier or "primary").strip().lower()

            def _threshold(name: str, default: str) -> float:
                return InferenceGate._env_float(name, float(default))

            if tier == "secondary":
                max_pressure = _threshold(
                    "AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT",
                    "42" if total_gb < 96.0 else "72",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB",
                    "52" if total_gb < 96.0 else "28",
                )
            elif tier == "tertiary":
                max_pressure = _threshold(
                    "AURA_FOREGROUND_TERTIARY_MAX_PRESSURE_PCT",
                    "92" if total_gb >= 60.0 else "88",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_TERTIARY_MIN_AVAILABLE_GB",
                    "6" if total_gb >= 60.0 else "4",
                )
            else:
                max_pressure = _threshold(
                    "AURA_FOREGROUND_PRIMARY_MAX_PRESSURE_PCT",
                    "76" if total_gb >= 60.0 else "82",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_PRIMARY_MIN_AVAILABLE_GB",
                    "18" if total_gb >= 60.0 else "10",
                )
                # The floor above answers "is there room to LOAD a model?".
                # For a turn on a cortex that is ALREADY resident, that is the
                # wrong question, and asking it double-counts: the resident
                # weights are counted as used, and then the gate demands 18GB
                # more on top of them.
                #
                # LIVE 2026-08-17, 64GB host, 32B resident (~20GB wired) beside
                # Chrome and two Electron apps:
                #     pressure=78.4% available=13.8GB (need <76.0% and >=18.0GB)
                # The condition is unsatisfiable in Aura's own normal operating
                # state, so admission tightened on an ordinary conversational
                # turn and the person got "I couldn't get to an answer I'd
                # stand behind" — from a worker that had just completed a
                # generation.
                #
                # A turn on a loaded model needs its transient decode working
                # set, not another model's worth of headroom. When nothing has
                # to be loaded, the floor becomes that transient requirement.
                if InferenceGate._cortex_already_resident():
                    min_available_gb = min(
                        min_available_gb,
                        _threshold(
                            "AURA_FOREGROUND_RESIDENT_TURN_MIN_AVAILABLE_GB", "6"
                        ),
                    )
                    max_pressure = max(
                        max_pressure,
                        _threshold(
                            "AURA_FOREGROUND_RESIDENT_TURN_MAX_PRESSURE_PCT", "88"
                        ),
                    )

            # The percentages above are derived from psutil's macOS accounting,
            # which counts file-backed cache and compressed pages as consumed.
            # They are not — the OS reclaims them on demand. Measured on this
            # host while a turn was being refused for "78.4% pressure": the
            # kernel reported level NORMAL and 79% free, with 24GB of the
            # "used" memory sitting in reclaimable inactive pages.
            #
            # So when the OS itself says there is no pressure, a derived
            # percentage does not get to veto a turn. The absolute floors still
            # apply: this relaxes the RATE signal, never the hard minimum, the
            # process-tree RSS limit, or an explicit refusal.
            kernel_level = str(
                getattr(snapshot, "kernel_pressure_level", "unknown") or "unknown"
            )
            if kernel_level == "normal":
                min_available_gb = min(
                    min_available_gb,
                    _threshold("AURA_FOREGROUND_KERNEL_NORMAL_MIN_AVAILABLE_GB", "4"),
                )
                max_pressure = max(max_pressure, 100.0)
            elif kernel_level == "critical":
                # The OS is actively asking processes to free memory. Whatever
                # the derived percentage says, this is not the moment.
                max_pressure = min(max_pressure, 0.0)
            system_admit = bool(pressure_pct < max_pressure and available_gb >= min_available_gb)
            process_admit = bool(
                process_rss_limit_gb <= 0.0
                or process_rss_gb < process_rss_limit_gb
            )
            can_admit = bool(system_admit and process_admit and not snapshot.refuse_heavy_local_generation)
            reason_parts: list[str] = []
            if not system_admit:
                reason_parts.append(
                    f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
                    f"(need <{max_pressure:.1f}% and >={min_available_gb:.1f}GB)"
                )
            if not process_admit or snapshot.refuse_heavy_local_generation:
                reason_parts.append(
                    f"process_tree_rss:{process_rss_gb:.1f}GB/{process_rss_limit_gb:.1f}GB"
                )
            reason = "; ".join(part for part in reason_parts if part)
            return {
                "tier": tier,
                "pressure_pct": pressure_pct,
                "total_gb": total_gb,
                "available_gb": available_gb,
                "process_rss_gb": process_rss_gb,
                "process_rss_limit_gb": process_rss_limit_gb,
                "max_pressure_pct": max_pressure,
                "min_available_gb": min_available_gb,
                "can_admit": can_admit,
                "reason": reason,
                "measured": True,
                "schema": ADMISSION_SNAPSHOT_SCHEMA,
                "measured_at_monotonic": time.monotonic(),
            }
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError, psutil.Error) as exc:
            _record_inference_degradation(
                exc,
                action="returned unmeasured foreground headroom snapshot after memory probe failed",
            )
            force_admit = str(
                _FLAG_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE.value()
            ).strip().lower() in {"1", "true", "yes", "on"}
            # measured=False: the zeros below are UNKNOWN values, not calm
            # telemetry — scheduling and health consumers must not treat this
            # snapshot as evidence of memory abundance.
            return {
                "tier": str(requested_tier or "primary"),
                "pressure_pct": 0.0,
                "total_gb": 0.0,
                "available_gb": 0.0,
                "process_rss_gb": 0.0,
                "process_rss_limit_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": force_admit,
                "reason": (
                    "memory_probe_failed_forced_override"
                    if force_admit
                    else "memory_probe_failed"
                ),
                "measured": False,
                "schema": ADMISSION_SNAPSHOT_SCHEMA,
                "measured_at_monotonic": time.monotonic(),
            }

    @staticmethod
    def _local_deep_solver_enabled(
        total_gb: float | None = None,
        available_gb: float | None = None,
    ) -> bool:
        return local_deep_solver_enabled(total_gb, available_gb)

    def _local_deep_solver_block_reason(self) -> str | None:
        snapshot = self._headroom_snapshot("secondary")
        specialist = local_deep_solver_status(
            snapshot.get("total_gb"),
            snapshot.get("available_gb"),
        )
        if not specialist.get("admitted", False):
            return str(specialist.get("reason") or "local_deep_solver_unqualified")
        if not snapshot.get("can_admit", False):
            return str(snapshot.get("reason") or "secondary_memory_pressure")
        lane = self.get_conversation_status()
        if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
            return "primary_cortex_resident_or_warming"
        lane_state = str(lane.get("state", "") or "").strip().lower()
        if lane_state in {"spawning", "handshaking", "warming", "recovering"}:
            return f"primary_cortex_{lane_state}"
        return None

    def _foreground_headroom_reserved(self, requested_tier: str = "primary") -> bool:
        snap = self._headroom_snapshot(requested_tier)
        safety_buffer_gb = 3.0 if snap["tier"] == "secondary" else 2.0
        return bool(
            snap["pressure_pct"] >= (snap["max_pressure_pct"] - 2.0)
            or snap["available_gb"] <= (snap["min_available_gb"] + safety_buffer_gb)
        )

    @staticmethod
    def _iter_local_clients() -> dict[str, Any]:
        clients: dict[str, Any] = {}
        try:
            # NOT dict(_CLIENTS): copying iterates the shared registry, and a
            # client registered or torn down mid-copy raises "dictionary
            # changed size during iteration". Live 2026-08-03 — and because
            # this subsystem is fail-closed, that RuntimeError was escalated to
            # CRITICAL and held the runtime DEGRADED across health pulses.
            from core.brain.llm.mlx_client import clients_snapshot

            clients.update(dict(clients_snapshot()))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("MLX client registry unavailable: %s", exc)
        return clients

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> int:
        """Abort any active local generation across managed inference clients.

        This is a synchronous emergency boundary used by watchdogs that cannot
        rely on the caller's event loop being healthy. Normal request handling
        still uses cooperative deadlines; this path exists to prevent a wedged
        model generation from holding the foreground lane indefinitely.
        """
        aborted = 0
        candidates: list[Any] = []
        if self._mlx_client is not None:
            candidates.append(self._mlx_client)
        candidates.extend(self._iter_local_clients().values())

        seen: set[int] = set()
        for client in candidates:
            if client is None:
                continue
            ident = id(client)
            if ident in seen:
                continue
            seen.add(ident)
            abort = getattr(client, "force_abort_active_generation", None)
            if not callable(abort):
                continue
            try:
                requested = bool(abort(reason=reason))
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued force-aborting other local generation clients",
                    severity="error",
                )
                logger.warning("Force-abort failed for local inference client: %s", exc)
                continue
            if not requested:
                continue
            # Count what STOPPED, not what was asked to stop.
            #
            # This is the emergency boundary a watchdog acts on, and the number
            # it returns is that watchdog's evidence. Incrementing from the
            # client's own truthy return meant "I sent the request" was
            # reported as "the generation ended" — so a wedged decode that
            # ignored the abort still counted, and the watchdog stood down
            # believing the lane was free.
            stopped = _generation_actually_stopped(client)
            if stopped is True:
                aborted += 1
            else:
                detail = (
                    "the client still reports an active generation"
                    if stopped is False
                    else "generation supervision is unavailable"
                )
                _record_inference_degradation(
                    RuntimeError(f"force_abort_unconfirmed:{reason}"),
                    action=(
                        f"force-abort was accepted but {detail}; not counting it "
                        "as aborted"
                    ),
                    severity="error",
                )
                logger.error(
                    "🛑 Force-abort accepted but %s on %s — refusing to report "
                    "it as stopped.",
                    detail,
                    getattr(client, "model_path", "local client"),
                )
        return aborted

    _SHUTDOWN_TASK_ATTRS = (
        "_prewarm_task",
        "_deferred_prewarm_task",
        "_maintenance_task",
        "_status_recovery_task",
    )

    def _cancel_owned_background_tasks(self) -> list[asyncio.Task]:
        """Cancel every owned background task and return the live handles.

        Handles are returned so the async shutdown path can await actual
        termination — cancellation alone does not stop a task, and its
        finally blocks may still hold worker processes or reservations.
        """
        tasks: list[asyncio.Task] = []
        for task_attr in self._SHUTDOWN_TASK_ATTRS:
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
            setattr(self, task_attr, None)
        return tasks

    def _shutdown_client_candidates(self) -> list[Any]:
        candidates: list[Any] = []
        if self._mlx_client is not None:
            candidates.append(self._mlx_client)
        candidates.extend(self._iter_local_clients().values())
        seen: set[int] = set()
        unique: list[Any] = []
        for client in candidates:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            unique.append(client)
        return unique

    async def on_stop_async(self) -> None:
        """Async shutdown: await task termination and client closes.

        Preferred by ServiceContainer over the sync `cleanup`. Awaiting the
        cancelled tasks lets their finally blocks release worker processes and
        reservations before clients are closed and state is declared cold.
        """
        tasks = self._cancel_owned_background_tasks()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=5.0)
            for task in pending:
                logger.warning(
                    "Inference background task %s did not terminate within the "
                    "shutdown grace period.",
                    task.get_name(),
                )
        for client in self._shutdown_client_candidates():
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=10.0)
            except (TimeoutError, *_INFERENCE_RECOVERABLE_ERRORS) as exc:
                _record_inference_degradation(
                    exc,
                    action=f"continued inference shutdown after {type(client).__name__}.close failed",
                    severity="warning",
                )
                logger.debug(
                    "Inference client close failed for %s: %s", type(client).__name__, exc
                )
        self._mlx_client = None
        self._initialized = False

    def cleanup(self) -> None:
        """Release managed local inference clients during ServiceContainer shutdown.

        Synchronous fallback path. When no event loop is running in this
        thread, awaitable client closes are driven to completion on a private
        loop; when a loop IS running here, blocking is impossible, so the
        close is scheduled and recorded as a degradation instead of being
        silently discarded.
        """
        cancelled_tasks = self._cancel_owned_background_tasks()

        try:
            running_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if cancelled_tasks and running_loop is None:
            logger.warning(
                "Sync inference cleanup cancelled %d background task(s) without "
                "awaiting termination; prefer on_stop_async for ordered shutdown.",
                len(cancelled_tasks),
            )

        for client in self._shutdown_client_candidates():
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    if running_loop is not None:
                        # CP126 residue: "continues to clear the client
                        # reference. Processes and buffers can remain live
                        # after cleanup reports completion."
                        #
                        # Dropping self._mlx_client below while this close
                        # is still in flight orphans a live worker: nothing
                        # holds the handle, so nothing can kill it if the
                        # close hangs. Keep a strong reference until the
                        # close actually finishes.
                        self._closing_clients.add(client)
                        task = get_task_tracker().create_task(
                            asyncio.wait_for(result, timeout=10.0),
                            name=f"inference_gate_close_{type(client).__name__}",
                        )
                        task.add_done_callback(self._log_task_exception)
                        task.add_done_callback(
                            lambda _t, _c=client: self._closing_clients.discard(_c)
                        )
                        _record_inference_degradation(
                            RuntimeError(
                                f"{type(client).__name__}.close deferred to running loop "
                                "during sync shutdown"
                            ),
                            action="scheduled async client close instead of blocking sync shutdown",
                        )
                    else:
                        asyncio.run(asyncio.wait_for(result, timeout=10.0))
            except (TimeoutError, *_INFERENCE_RECOVERABLE_ERRORS) as exc:
                _record_inference_degradation(
                    exc,
                    action=f"continued inference shutdown after {type(client).__name__}.close failed",
                    severity="warning",
                )
                logger.debug("Inference client close failed for %s: %s", type(client).__name__, exc)

        self._mlx_client = None
        self._initialized = False

    on_stop = cleanup




    @staticmethod
    def _model_now_serving(requested_tier: str) -> str:
        """What the rate should be read for: the model this turn will run on.

        The name the worker records, so the two sides of the measurement agree.
        Empty where it cannot be worked out, which pools every model's readings
        exactly as this did before.
        """
        try:
            import os

            from core.brain.llm.model_registry import (
                BRAINSTEM_MODEL,
                _current_cortex_path,
            )

            if str(requested_tier or "") == "primary":
                return os.path.basename(str(_current_cortex_path() or ""))
            return os.path.basename(str(BRAINSTEM_MODEL or ""))
        except (ImportError, AttributeError, OSError, TypeError, ValueError):
            return ""

    @staticmethod
    def _tokens_the_turn_is_allowed_to_take(
        model_ceiling: int = 0,
        *,
        seconds: float = 0.0,
        prompt_chars: int = 0,
        model: str = "",
    ) -> int:
        """How much a user-facing turn may say, from the time it actually has.

        The reserve discovers the room a thinking turn needs by running out of
        it, once per turn, and each discovery costs somebody an answer: 1,024
        then 2,048 then 3,072, a failed reply at every step. It is the right
        mechanism for an unknown quantity and the wrong one for a quantity the
        hardware already answers.

        A turn is bounded by a wall clock it cannot exceed. What the model can
        say inside that clock, at the rate this machine has been measured
        decoding, is a number rather than a guess — and a budget is a ceiling
        rather than a reservation, so a turn that finishes early pays nothing
        for having been allowed more.

        Zero where the rate is unmeasured, which leaves the existing budget
        exactly as it was.

        ``seconds`` is the time THIS turn has left, and it is not the same as
        the most a turn may ever be given. Sizing against the maximum while
        generation is handed the remainder over-promises by exactly what the
        turn already spent — and on a second attempt, after a first reply
        failed its identity check, that is most of the clock. LIVE 2026-08-30:
        4,167 tokens budgeted, 166.9s actually available, generation aborted at
        the deadline, and every token produced in those 166 seconds discarded.
        """

        try:
            from core.brain.llm.thinking_reserve import seconds_to_decode
            from core.runtime.response_policy import (
                USER_FACING_COMPLETION_DEADLINE_MAX_S,
            )
        except (ImportError, AttributeError):
            return 0
        allowed = float(seconds) if float(seconds or 0.0) > 0.0 else float(
            USER_FACING_COMPLETION_DEADLINE_MAX_S
        )
        allowed = min(allowed, float(USER_FACING_COMPLETION_DEADLINE_MAX_S))
        # Reading the prompt comes out of the same clock as writing the answer,
        # and only the writing was counted. A ten-thousand-character prompt
        # takes twenty-six seconds to read, and on a turn whose first-token
        # ceiling alone was a hundred and twenty, the budget promised an answer
        # the clock could never pay for.
        if int(prompt_chars or 0) > 0:
            try:
                from core.brain.llm.mlx_client import seconds_to_read

                allowed -= float(seconds_to_read(int(prompt_chars)))
            except (ImportError, AttributeError, TypeError, ValueError):
                pass
        if not (allowed > 0.0):
            return 0
        # The forward estimate is monotone in tokens, so the largest budget
        # that fits is found by searching it rather than by inverting a rate
        # this would then have to keep in step with.
        low, high = 0, int(model_ceiling) if int(model_ceiling or 0) > 0 else 8192
        if seconds_to_decode(high, model) <= 0.0:
            return 0
        if seconds_to_decode(high, model) <= allowed:
            return high
        while low < high:
            middle = (low + high + 1) // 2
            if seconds_to_decode(middle, model) <= allowed:
                low = middle
            else:
                high = middle - 1
        return low

    @staticmethod
    def _reasoning_reserve(model: str = "") -> int:
        """What the worker will add to this turn's budget for thinking.

        Read here so the clock covers the same number of tokens the worker
        will actually decode. Zero where nothing has been measured, and zero
        where the reserve cannot be reached, which is the same silence every
        other unmeasured quantity keeps.
        """

        try:
            from core.brain.llm.thinking_reserve import reserve_tokens

            return max(0, int(reserve_tokens(model)))
        except (ImportError, AttributeError, TypeError, ValueError):
            return 0

    @classmethod
    def _reasoning_reserve_for_generation(
        cls,
        *,
        model: str,
        cognitive_mode: object = None,
        final_user_surface: bool,
        completion_floor: object,
        budget_tokens: object,
        seconds_remaining: object = 0.0,
    ) -> int:
        """The reserve this exact generation role will receive in the worker."""

        try:
            from core.brain.llm.chat_format import (
                answer_is_derived_for_generation,
                thinking_enabled_for_generation,
            )

            derived_here = answer_is_derived_for_generation(
                completion_floor=completion_floor,
                budget_tokens=budget_tokens,
                model_name=model,
                seconds_remaining=seconds_remaining,
            )
            native_thinking = thinking_enabled_for_generation(
                model,
                cognitive_mode=cognitive_mode,
                final_user_surface=final_user_surface,
                answer_is_derived_here=derived_here,
            )
        except (ImportError, AttributeError, TypeError, ValueError):
            return 0
        return cls._reasoning_reserve(model) if native_thinking is True else 0

    @staticmethod
    def _tokens_the_clock_can_deliver(max_tokens: Any, *, seconds: float) -> int:
        """Cut a token budget the clock cannot pay for at the measured rate.

        The timeout and the token budget come from two separate tables keyed
        on the origin, and nothing compared them. On this hardware the model
        decodes about ten tokens a second, so a 1,024-token budget wants a
        hundred seconds of decoding before prefill, and it was being handed
        out beside a deadline of about half that.

        The turn then ends the way it did live on 2026-08-28: "Request
        deadline reached at token 1188; stopping decode cooperatively", a
        truncated answer, and a runtime that reported the token budget as the
        thing that ran out. Promising more tokens than the clock can pay for
        is a promise that always breaks in the middle of a sentence.

        Silent when the rate has not been measured — an unmeasured rate cuts
        nothing — and it only ever lowers the budget, so nothing here can
        hand a lane more room than its own table allowed.
        """

        try:
            wanted = int(max_tokens)
            allowed = float(seconds)
        except (TypeError, ValueError):
            return int(max_tokens or 0)
        if wanted <= 0 or allowed <= 0:
            return wanted
        try:
            from core.brain.llm.thinking_reserve import seconds_to_decode
        except ImportError:
            return wanted
        needed = seconds_to_decode(wanted)
        if needed <= 0 or needed <= allowed:
            return wanted
        fits = max(1, int(wanted * (allowed / needed)))
        logger.info(
            "⏱️ [GATE] %d tokens need %.0fs to decode and the turn has %.0fs; "
            "asking for %d instead.",
            wanted,
            needed,
            allowed,
            fits,
        )
        return fits

    async def _enforce_foreground_admission(
        self,
        requested_tier: str,
        *,
        protected_foreground: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._headroom_snapshot(requested_tier)
        if snapshot["can_admit"]:
            return snapshot

        logger.warning(
            "🛡️ Foreground admission tightening for %s "
            "(pressure=%s available=%s process=%s/%s reason=%s).",
            requested_tier,
            format_metric(snapshot, "pressure_pct", unit="%"),
            format_metric(snapshot, "available_gb", unit="GB"),
            format_metric(snapshot, "process_rss_gb", unit="GB"),
            format_metric(snapshot, "process_rss_limit_gb", unit="GB"),
            snapshot.get("reason", ""),
        )
        await self._shed_background_workers_for_memory_pressure()
        gc.collect()
        tightened = self._headroom_snapshot(requested_tier)
        if not tightened["can_admit"] and protected_foreground and requested_tier != "secondary":
            logger.warning(
                "🛡️ Protected foreground request proceeding under reduced headroom for tier=%s "
                "(pressure=%s available=%s process=%s/%s).",
                requested_tier,
                format_metric(tightened, "pressure_pct", unit="%"),
                format_metric(tightened, "available_gb", unit="GB"),
                format_metric(tightened, "process_rss_gb", unit="GB"),
                format_metric(tightened, "process_rss_limit_gb", unit="GB"),
            )
        return tightened

    async def _ensure_hot_spare_ready(self, endpoint_name: str) -> bool:
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return False

        if endpoint_name == DEEP_ENDPOINT:
            lane = self.get_conversation_status()
            lane_state = str(lane.get("state", "") or "").strip().lower()
            if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                return False
            if lane_state in {"spawning", "handshaking", "warming", "recovering"}:
                return False
            background_deferral = self._background_local_deferral_reason(
                origin="maintenance_hot_spare"
            )
            if background_deferral:
                logger.debug(
                    "⏸️ Skipping Solver hot spare warmup due to %s.",
                    background_deferral,
                )
                return False

        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import (
                get_brainstem_path,
                get_deep_model_path,
                get_fallback_path,
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Hot-spare setup unavailable: %s", exc)
            return False

        if endpoint_name == BRAINSTEM_ENDPOINT:
            model_path = str(get_brainstem_path())
            requested_tier = "tertiary"
        elif endpoint_name == DEEP_ENDPOINT:
            model_path = str(get_deep_model_path())
            requested_tier = "secondary"
        elif endpoint_name == FALLBACK_ENDPOINT:
            model_path = str(get_fallback_path())
            requested_tier = "tertiary"
        else:
            return False

        snapshot = self._headroom_snapshot(requested_tier)
        if endpoint_name == DEEP_ENDPOINT and not snapshot["can_admit"]:
            return False

        client = get_mlx_client(model_path=model_path)
        if hasattr(client, "is_alive") and client.is_alive():
            return True
        if not hasattr(client, "warmup"):
            return False

        try:
            # BOUNDED. This is maintenance, and an unbounded await here parks
            # the maintenance task for as long as a cold multi-GB load takes —
            # or forever if the load wedges. The budget is generous because a
            # legitimate cold spare genuinely takes minutes, and finite because
            # an unbounded wait is the wedge, not the warmup.
            await asyncio.wait_for(
                client.warmup(foreground_request=False),
                timeout=_HOT_SPARE_WARMUP_BUDGET_S,
            )
        except TimeoutError as exc:
            _record_inference_degradation(
                exc,
                action=(
                    f"abandoned hot-spare warmup for {endpoint_name} after "
                    f"{_HOT_SPARE_WARMUP_BUDGET_S:.0f}s; the spare stays cold"
                ),
                severity="warning",
            )
            logger.warning(
                "⏱️ Hot-spare warmup for %s exceeded %.0fs — leaving it cold.",
                endpoint_name,
                _HOT_SPARE_WARMUP_BUDGET_S,
            )
            return False
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Hot-spare warmup failed for %s: %s", endpoint_name, exc)
            return False
        return _hot_spare_is_ready(client)

    async def _recycle_idle_local_clients(self) -> None:
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return

        max_uptime_s = float(_FLAG_LOCAL_RECYCLE_MAX_UPTIME_S.value())
        min_idle_s = float(_FLAG_LOCAL_RECYCLE_MIN_IDLE_S.value())
        for client in self._iter_local_clients().values():
            if client is None or client is self._mlx_client:
                continue
            recycle_predicate = getattr(client, "should_recycle_for_fragmentation", None)
            if not callable(recycle_predicate):
                continue
            try:
                if recycle_predicate(max_uptime_s=max_uptime_s, min_idle_s=min_idle_s):
                    logger.info("♻️ Recycling idle local runtime to reduce fragmentation.")
                    if hasattr(client, "reboot_worker"):
                        await client.reboot_worker(
                            reason="scheduled_fragmentation_recycle",
                            mark_failed=False,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                # A worker that is already gone is what recycling wanted.
                #
                # "process object is closed" means the handle refers to a
                # process that has exited — which is the outcome this pass is
                # trying to reach, not a failure to reach it. Recorded as a
                # fault it escalates: inference_gate is a required subsystem,
                # so a warning here becomes CRITICAL SERVICE FAILURE and takes
                # the gate's maintenance loop down with it. Measured live,
                # mid-game, while she was playing.
                if "process object is closed" in str(exc).lower():
                    logger.debug("Idle runtime already gone; nothing to recycle: %s", exc)
                    continue
                _record_inference_degradation(
                    exc,
                    action="continued recycling other idle local clients",
                )
                logger.debug("Idle runtime recycle skipped: %s", exc)

    async def _maintenance_loop(self) -> None:
        while not is_shutdown_requested():
            try:
                await asyncio.sleep(15.0 if self._last_spare_maintenance_at <= 0.0 else 45.0)
                self._last_spare_maintenance_at = time.monotonic()
                if self._background_memory_pressure_active():
                    await self._shed_background_workers_for_memory_pressure()
                    continue

                # [STABILITY v53] Proactive cortex health watchdog — detect dead
                # cortex BEFORE a user request fails. Previously cortex death was
                # only detected when a user message arrived and timed out.
                await self._proactive_cortex_watchdog()

                # [STABILITY v53] Don't eagerly load brainstem/deep at boot.
                # The 7B brainstem consumes ~5GB RAM that the 32B cortex needs.
                # At 62% RAM with both loaded, the cortex swaps and first-turn
                # response time balloons to 80+ seconds. Load on demand only.
                # await self._ensure_hot_spare_ready(BRAINSTEM_ENDPOINT)
                # await self._ensure_hot_spare_ready(DEEP_ENDPOINT)
                await self._recycle_idle_local_clients()
            except asyncio.CancelledError:
                raise
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued maintenance loop after non-fatal maintenance pulse failure",
                )
                # [STABILITY v53] Upgraded from debug to warning — silent maintenance
                # failures can cascade into cortex death without visibility.
                logger.warning("⚠️ InferenceGate maintenance loop error: %s", exc, exc_info=True)

    async def _proactive_cortex_watchdog(self) -> None:
        """[STABILITY v53] Proactive cortex health check — runs every maintenance cycle.

        Detects dead/stuck cortex and triggers recovery BEFORE user requests fail.
        Also detects stale warming states and resets them.
        """
        if not self._mlx_client:
            return
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return  # Don't interfere with active user turn

        # This IS the self-heal path: it may act on what it observes.
        lane = self.get_conversation_status(observe_only=False)
        lane_state = str(lane.get("state", "") or "").lower()

        # 1. Detect dead cortex and trigger recovery.
        #
        # A WARMING lane is not a dead lane. During a 32B cold load the
        # worker legitimately fails is_alive() for 120-150s while the state
        # sits in warming/spawning/handshaking — and a warmup is not flagged
        # as "recovery in progress". The 20260708-postdoomfix soak showed
        # what happens without this guard: the watchdog declared the warming
        # lane dead every 45s maintenance pulse and re-triggered recovery,
        # restarting the warmup before it could ever finish (turns pinned at
        # 216s+, SLO exhausted, runtime eventually died). The lane gets the
        # same 300s deadline section 2 already grants a stuck warmup; only
        # past it does a "warming" verdict count as dead.
        if hasattr(self._mlx_client, "is_alive") and not self._mlx_client.is_alive():
            # DEAD-MAN CLOCK, watchdog-owned. The first cut of this guard
            # trusted the client's own fields to bound the deferral
            # (warmup_in_flight OR transition age) — and the nightcap soak
            # promptly wedged with warmup_in_flight stuck True and no
            # transition timestamp: the watchdog deferred FOREVER, eleven
            # straight turns hit the probe's 240s ceiling, and nothing ever
            # recovered the lane. The watchdog now times the not-alive
            # window on its OWN clock: a warming lane gets 300s from first
            # observation, then intervention happens no matter what any
            # client flag claims.
            now = time.time()
            first_seen = float(getattr(self, "_cortex_not_alive_first_seen_at", 0.0) or 0.0)
            if first_seen <= 0.0:
                first_seen = now
                self._cortex_not_alive_first_seen_at = now
            not_alive_age_s = now - first_seen
            warmup_underway = (
                lane_state in ("warming", "spawning", "handshaking", "recovering")
                and not_alive_age_s <= 300.0
            )
            if warmup_underway:
                logger.debug(
                    "[WATCHDOG] Cortex not alive but lane is %s (%.0fs on the dead-man clock) — "
                    "letting the warmup finish.",
                    lane_state, not_alive_age_s,
                )
            elif lane_state not in ("cold", "failed") and not self._cortex_recovery_in_progress:
                logger.warning(
                    "🔍 [WATCHDOG] Cortex is dead (state=%s, not-alive %.0fs) and past the warmup "
                    "deadline. Triggering recovery.",
                    lane_state, not_alive_age_s,
                )
                # Reset the clock so the RECOVERY warmup gets its own fresh
                # 300s window instead of being instantly re-declared dead.
                self._cortex_not_alive_first_seen_at = 0.0
                # A wedged warmup flag blocks admission everywhere
                # (conversation_warmup_in_flight deferrals); recovery must
                # not start underneath it.
                if getattr(self._mlx_client, "_warmup_in_flight", False):
                    logger.warning(
                        "🔍 [WATCHDOG] Force-clearing wedged warmup_in_flight before recovery."
                    )
                    self._clear_wedged_cortex_warmup(
                        "watchdog_pre_recovery", owner="watchdog"
                    )
                    await self.await_abandoned_cortex_loads()
                await self._ensure_cortex_recovery()
        else:
            # Lane is alive — clear the dead-man clock.
            if getattr(self, "_cortex_not_alive_first_seen_at", 0.0):
                self._cortex_not_alive_first_seen_at = 0.0

        # 2. Detect stuck warmup flag on MLX client
        if hasattr(self._mlx_client, "_warmup_in_flight") and self._mlx_client._warmup_in_flight:
            transition_at = getattr(self._mlx_client, "_lane_transition_at", 0.0)
            # [STABILITY v53] Increased from 90s to 300s. A 32B model cold-load
            # takes ~150s; 90s was guaranteed to force-kill a healthy loading worker.
            if transition_at > 0 and (time.time() - transition_at) > 300.0:
                logger.warning(
                    "🔍 [WATCHDOG] MLX warmup_in_flight stuck for >300s. Force-clearing."
                )
                receipt = self._clear_wedged_cortex_warmup(
                    "watchdog_stuck_warmup", owner="watchdog"
                )
                if receipt.get("cancelled_prewarm"):
                    logger.warning(
                        "🔍 [WATCHDOG] Stuck prewarm task cancelled; waiting for it to stop."
                    )
                # Cancelling is a request. Waiting is how we know a 20GB load
                # is not still running under the next warmup.
                await self.await_abandoned_cortex_loads()

        # 3. Detect completed-but-unreaped prewarm tasks
        if self._prewarm_task and self._prewarm_task.done():
            try:
                exc = self._prewarm_task.exception()
                if exc:
                    logger.warning(
                        "🔍 [WATCHDOG] Stale failed prewarm task found: %s. Clearing.", exc
                    )
            except (asyncio.CancelledError, asyncio.InvalidStateError) as exc:
                logger.debug("Prewarm task state was unavailable during watchdog cleanup: %s", exc)
            self._prewarm_task = None  # Allow fresh warmup on next request

        # 4. Log cortex health for observability
        if hasattr(self._mlx_client, "is_alive"):
            alive = self._mlx_client.is_alive()
            if not alive and lane_state == "ready":
                logger.warning(
                    "🔍 [WATCHDOG] Cortex reports ready but is_alive() is False. Correcting state."
                )
                if hasattr(self._mlx_client, "note_lane_recovering"):
                    self._mlx_client.note_lane_recovering("watchdog_state_correction")

    def get_conversation_status(self, *, observe_only: bool = True) -> dict[str, Any]:
        """Snapshot the conversation lane.

        CP126 ab3c124a: this method is polled by routers, probes, audits,
        health endpoints and the neural stream, yet OBSERVING it could schedule
        background recovery work — so poll frequency changed runtime behavior.
        It is now PURE by default. The gate's own self-heal paths opt into the
        ratchet with ``observe_only=False``; a cooldown still bounds how often
        that can fire.
        """
        # [STABILITY v53] Default to "cold" not "warming" — only report warming
        # when something is actually in flight. Prevents zombie warming state.
        _default_state = "failed" if self._init_error else "cold"
        lane = {
            "desired_model": lane_display_label(PRIMARY_ENDPOINT),
            "desired_endpoint": PRIMARY_ENDPOINT,
            "foreground_endpoint": None,
            "background_endpoint": BRAINSTEM_ENDPOINT,
            "foreground_tier": "local",
            "background_tier": "local_fast",
            "state": _default_state,
            "last_failure_reason": self._init_error or "",
            "conversation_ready": False,
            "cortex_recovery_attempts": getattr(self, "_cortex_recovery_attempts", 0),
            # Durable evidence that the primary lane could not be warmed, and
            # why. A deferral is not a fault, but a runtime full of them is not
            # a healthy one either, and this is where health can see it.
            "warmup_deferrals": copy.deepcopy(
                getattr(self, "_warmup_deferral_counts", {}) or {}
            ),
            # When no generation has ever succeeded, the honest age is
            # "at least since the gate was constructed" — never zero.
            "has_generated_successfully": bool(
                getattr(self, "_last_successful_generation_at", 0.0) > 0.0
            ),
            "time_since_last_success_s": max(
                0,
                time.time()
                - (
                    getattr(self, "_last_successful_generation_at", 0.0)
                    or getattr(self, "_constructed_wall_at", time.time())
                ),
            ),
            "last_transition_at": 0.0,
            "last_ready_at": 0.0,
            "last_progress_at": 0.0,
            "warmup_attempted": False,
            "warmup_in_flight": bool(self._prewarm_task and not self._prewarm_task.done()),
            # getattr defaults: this snapshot feeds watchdogs and recovery
            # probes — a missing informational field must degrade to its
            # default, never raise out of a status read.
            "last_user_generation_endpoint": getattr(
                self, "_last_user_generation_endpoint", None
            ),
            "last_user_generation_at": getattr(self, "_last_user_generation_at", 0.0),
            "last_user_generation_used_fallback": getattr(
                self, "_last_user_generation_used_fallback", False
            ),
        }
        now_wall = time.time()
        raw_ready = False
        raw_readiness_blockers: list[str] = []
        visible_conversation_anchor = 0.0
        visible_anchor_recent = False
        if self._mlx_client and hasattr(self._mlx_client, "get_lane_status"):
            # Protective envelope. This is the HEALTH observation path: it is
            # what answers "is Aura able to talk?", so it must survive a client
            # that raises, returns a non-mapping, or hands back something with
            # hostile property access. Bare, a malformed status crashed the
            # very probe that would have reported the problem, and the caller
            # saw an exception instead of a degraded-but-honest reading.
            #
            # An empty mapping is the right failure: every read below already
            # supplies a default, so the lane reports what it independently
            # knows rather than nothing at all.
            raw: Mapping[str, Any] = {}
            try:
                candidate = self._mlx_client.get_lane_status()
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                # The module's typed tuple, deliberately — broad catches are
                # forbidden in this file (tests/test_causal_gating.py) so every
                # degradation stays classified. The tuple already covers what a
                # malformed or hostile client can raise here: AttributeError,
                # TypeError, ValueError, RuntimeError, OSError, LookupError.
                _record_inference_degradation(
                    exc,
                    action="reported lane status from independent state after get_lane_status raised",
                    severity="warning",
                )
                candidate = None
            if isinstance(candidate, Mapping):
                raw = candidate
            elif candidate is not None:
                _record_inference_degradation(
                    TypeError(f"get_lane_status returned {type(candidate).__name__}"),
                    action="ignored a non-mapping lane status payload",
                    severity="warning",
                )
            lane["state"] = str(raw.get("state", lane["state"]) or lane["state"])
            lane["last_failure_reason"] = str(
                raw.get("last_error", "") or lane["last_failure_reason"]
            )
            raw_ready = bool(raw.get("conversation_ready", False))
            raw_readiness_blockers = [
                str(blocker)
                for blocker in (raw.get("readiness_blockers") or [])
                if str(blocker or "").strip()
            ]
            if raw.get("runtime_identity_ok") is False:
                detected_models = raw.get("detected_models") or []
                identity_blocker = (
                    "runtime_identity_mismatch"
                    if detected_models
                    else "runtime_identity_unverified"
                )
                if identity_blocker not in raw_readiness_blockers:
                    raw_readiness_blockers.append(identity_blocker)
            if raw_readiness_blockers:
                raw_ready = False
            lane["conversation_ready"] = raw_ready
            lane["readiness_blockers"] = raw_readiness_blockers
            if raw_readiness_blockers and not lane["last_failure_reason"]:
                lane["last_failure_reason"] = ",".join(raw_readiness_blockers[:3])
            lane["last_transition_at"] = float(raw.get("last_transition_at", 0.0) or 0.0)
            lane["last_ready_at"] = float(raw.get("last_ready_at", 0.0) or 0.0)
            lane["last_progress_at"] = float(raw.get("last_progress_at", 0.0) or 0.0)
            lane["warmup_attempted"] = bool(raw.get("warmup_attempted", False))
            lane["warmup_in_flight"] = bool(raw.get("warmup_in_flight", lane["warmup_in_flight"]))
            lane["foreground_owned"] = bool(raw.get("foreground_owned", False))
            lane["foreground_owner"] = str(raw.get("foreground_owner", "") or "")
            lane["active_generations"] = int(raw.get("active_generations", 0) or 0)
            lane["request_age_s"] = float(raw.get("request_age_s", 0.0) or 0.0)
            lane["current_request_started_at"] = float(
                raw.get("current_request_started_at", 0.0) or 0.0
            )
            for telemetry_key in (
                "model_path",
                "recurrent_depth",
                "last_heartbeat",
                "last_token_progress_at",
                "last_generation_completed_at",
                "last_user_facing_completed_at",
                "last_visible_readiness_at",
                "process_started_at",
            ):
                if telemetry_key in raw:
                    lane[telemetry_key] = raw.get(telemetry_key)
            visible_conversation_anchor = max(
                float(lane.get("last_visible_readiness_at", 0.0) or 0.0),
                float(lane.get("last_user_facing_completed_at", 0.0) or 0.0),
            )
            visible_anchor_recent = (
                visible_conversation_anchor > 0.0
                and (now_wall - visible_conversation_anchor) <= 300.0
            )
            # The visible-conversation-probe guard catches a zombie chat lane that
            # reports "ready" without EVER serving a user-facing turn. That signal
            # is only meaningful when a UI/conversation surface is attached. A
            # headless proof/longevity run has no user surface, so no turn can ever
            # refresh the anchor — a warm+alive cortex is the legitimate terminal
            # ready state there, and applying the guard is a false positive.
            _proof_headless = False
            try:
                from core.runtime.proof_policy import proof_headless_run

                _proof_headless = proof_headless_run()
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.debug("Visible-probe proof-policy check unavailable: %s", exc)
            # Fire ONLY when the lane has never served a visible turn
            # (anchor <= 0) — matching the authoritative mlx_client guard
            # (mlx_client.py: `visible_conversation_anchor <= 0.0`) and this
            # guard's own "without ever serving" intent. A lane that already
            # proved it can serve and is merely IDLE (anchor > 0 but older than
            # 300s) must NOT be downgraded to not-ready — its liveness is covered
            # by the worker-progress-staleness probe, not by conversation idleness.
            if (
                str(lane.get("state", "") or "").lower() == "ready"
                and visible_conversation_anchor <= 0.0
                and not _proof_headless
                and "visible_conversation_probe_missing" not in raw_readiness_blockers
            ):
                raw_readiness_blockers.append("visible_conversation_probe_missing")
                raw_ready = False
                lane["conversation_ready"] = False
                lane["readiness_blockers"] = raw_readiness_blockers
                if not lane["last_failure_reason"]:
                    lane["last_failure_reason"] = "visible_conversation_probe_missing"
            if lane["conversation_ready"]:
                lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        # [STABILITY v51] If the prewarm task completed (success or failure),
        # force-sync warmup_in_flight to False. A done task is no longer "in flight"
        # regardless of what the MLX client flag says.
        if self._prewarm_task and self._prewarm_task.done():
            lane["warmup_in_flight"] = False
            # [STABILITY v53] If prewarm task completed with an exception and
            # conversation is NOT ready, set state to "recovering" and auto-schedule
            # a background recovery. This prevents the zombie warming state where
            # the task finished but the lane never transitions out.
            if not lane["conversation_ready"]:
                try:
                    exc = self._prewarm_task.exception()
                except asyncio.CancelledError:
                    exc = asyncio.CancelledError("prewarm_cancelled")
                except asyncio.InvalidStateError:
                    exc = None
                if exc is not None:
                    lane["state"] = "recovering"
                    lane["last_failure_reason"] = f"prewarm_failed:{type(exc).__name__}"
                    # Auto-trigger recovery if not already in progress
                    if (
                        not observe_only
                        and not self._cortex_recovery_in_progress
                        and not (
                            self._deferred_prewarm_task
                            and not self._deferred_prewarm_task.done()
                        )
                    ):
                        try:
                            warmup_deferral = self._cortex_warmup_deferral_reason("background")
                            if warmup_deferral:
                                self._log_cortex_warmup_deferral(
                                    warmup_deferral, context="background"
                                )
                            else:
                                self._schedule_background_cortex_prewarm_from_status(
                                    delay=2.0,
                                    reason="failed_prewarm_observed",
                                )
                                logger.info(
                                    "🔄 [STABILITY v53] Auto-scheduling cortex recovery after failed prewarm: %s",
                                    exc,
                                )
                        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                            logger.debug("Best-effort Cortex recovery scheduling skipped: %s", exc)
        lane_state = str(lane.get("state", "") or "").lower()
        _last_success_at = getattr(self, "_last_successful_generation_at", 0.0)
        recent_success = (
            _last_success_at > 0.0 and (now_wall - _last_success_at) <= 30.0
        )
        recent_ready = any(
            stamp > 0.0 and (now_wall - stamp) <= 300.0
            for stamp in (
                float(lane.get("last_ready_at", 0.0) or 0.0),
                float(lane.get("last_progress_at", 0.0) or 0.0),
            )
        )
        if raw_ready and not raw_readiness_blockers:
            lane["conversation_ready"] = True
            lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        elif raw_readiness_blockers:
            lane["conversation_ready"] = False
        elif lane_state == "ready" and (recent_success or recent_ready) and visible_anchor_recent:
            lane["conversation_ready"] = True
            lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        elif lane_state != "ready":
            lane["conversation_ready"] = False
        chat_dependencies_state, dependency_blocker = getattr(
            self, "_chat_dependencies_snapshot", (None, "")
        )
        if chat_dependencies_state is False:
            blocker = dependency_blocker or "chat_dependencies_warming"
            blockers = list(lane.get("readiness_blockers") or ())
            if blocker not in blockers:
                blockers.append(blocker)
            lane["readiness_blockers"] = blockers
            lane["conversation_ready"] = False
            lane["chat_dependencies_ready"] = False
            if not str(lane.get("last_failure_reason", "") or "").strip():
                lane["last_failure_reason"] = blocker
        else:
            lane["chat_dependencies_ready"] = chat_dependencies_state is True
        lane_state = str(lane.get("state", "") or "").lower()
        if (
            self._cortex_recovery_in_progress
            and not lane["conversation_ready"]
            and lane_state != "failed"
        ):
            lane["state"] = "recovering"
        if (
            self._prewarm_task
            and not self._prewarm_task.done()
            and not lane["conversation_ready"]
            and lane_state != "failed"
        ):
            lane["state"] = "warming"
            lane["warmup_in_flight"] = True
        # [STABILITY v53] Stale state watchdog: if lane has been in warming/recovering
        # for >90s with no progress and no active task, force to "cold" so the next
        # user request triggers a fresh warmup instead of waiting on a ghost.
        if lane_state in ("warming", "recovering") and not lane["conversation_ready"]:
            # [STABILITY v54] Eagerly cancel and clear prewarm task if it has been active for >300s.
            if self._prewarm_task and not self._prewarm_task.done():
                transition_age = (
                    _transition_age_s(self._mlx_client, lane) if self._mlx_client else 0.0
                )
                if transition_age > 300.0:
                    logger.warning(
                        "🔍 [WATCHDOG] Prewarm task is active for >300s (stuck). Cancelling task."
                    )
                    self._prewarm_task.cancel()
                    self._prewarm_task = None

            last_progress = max(
                float(lane.get("last_transition_at", 0.0) or 0.0),
                float(lane.get("last_progress_at", 0.0) or 0.0),
            )
            if last_progress > 0 and (time.time() - last_progress) > 90.0:
                has_active_task = (
                    (self._prewarm_task and not self._prewarm_task.done())
                    or (self._deferred_prewarm_task and not self._deferred_prewarm_task.done())
                    or self._cortex_recovery_in_progress
                )
                if not has_active_task:
                    # [HARDENING v54] Rate-limit this log — get_conversation_status()
                    # is called dozens of times per second by subsystems. Without
                    # rate limiting, a stuck lane produces thousands of warnings.
                    _now_mono = time.monotonic()
                    _last_log = getattr(self, "_last_stale_reset_log_at", 0.0)
                    if (_now_mono - _last_log) > 30.0:
                        self._last_stale_reset_log_at = _now_mono
                        logger.warning(
                            "🚨 [HARDENING v54] Lane stuck in '%s' for >90s with no active task. "
                            "Resetting to 'cold' and scheduling recovery.",
                            lane_state,
                        )
                    lane["state"] = "cold"
                    lane["warmup_in_flight"] = False
                    # [HARDENING v54] CRITICAL: Reset the MLX client's ACTUAL lane
                    # state, not just the returned dict. Without this, the next call
                    # reads "recovering" from the client again and the stale check
                    # fires in an infinite loop.
                    if self._mlx_client:
                        # Through the one owner, so a watchdog pass mid-cancel
                        # cannot be racing this. The cancelled load is kept for
                        # await_abandoned_cortex_loads() rather than dropped.
                        self._clear_wedged_cortex_warmup(
                            "stale_lane_observed", owner="conversation_status"
                        )
                        if hasattr(self._mlx_client, "_set_lane_state"):
                            self._mlx_client._set_lane_state("cold")
                    # [HARDENING v54] Schedule a recovery warmup so the cortex
                    # actually comes back online instead of staying cold forever.
                    # The prewarm runner performs the RAM admission check before
                    # loading anything; scheduling the runner here keeps recovery
                    # alive without forcing an unsafe immediate model load.
                    try:
                        self._schedule_background_cortex_prewarm_from_status(
                            delay=3.0,
                            reason="stale_lane_observed",
                        )
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        _record_inference_degradation(
                            exc,
                            action="returned conservative conversation status after probe failure",
                        )
                        logger.debug("Best-effort Cortex recovery scheduling skipped: %s", exc)
        return lane

    def get_lane_status(self) -> dict[str, Any]:
        """Expose the live MLX lane contract for routers, probes, and audits."""
        return self.get_conversation_status()

    def note_foreground_timeout(self, reason: str = "foreground_timeout") -> None:
        """Mark the conversation lane as degraded after a foreground timeout."""
        if self._mlx_client and hasattr(self._mlx_client, "note_lane_recovering"):
            try:
                self._mlx_client.note_lane_recovering(reason)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="recorded timeout without blocking later foreground recovery",
                )
                logger.debug("Failed to mark cortex lane recovering: %s", exc)
        self._extend_startup_quiet_window(8.0)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            warmup_deferral = self._cortex_warmup_deferral_reason("background")
            if warmup_deferral:
                self._log_cortex_warmup_deferral(warmup_deferral, context="background")
            else:
                self._schedule_background_cortex_prewarm(delay=2.0)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="left deferred cortex re-prewarm unscheduled; foreground path will retry",
            )
            logger.debug("Failed to schedule deferred cortex re-prewarm after timeout: %s", exc)

    def _extend_startup_quiet_window(self, seconds: float) -> None:
        orch = self.orch
        if orch is None:
            try:
                from core.container import ServiceContainer

                orch = ServiceContainer.get("orchestrator", default=None)
            except _INFERENCE_RECOVERABLE_ERRORS:
                orch = None
        if orch and hasattr(orch, "_extend_foreground_quiet_window"):
            try:
                orch._extend_foreground_quiet_window(seconds)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "inference_gate",
                    exc,
                    severity="warning",
                    action="continued without extending foreground quiet window",
                )
                logger.debug("Failed to extend foreground quiet window: %s", exc)

    def _schedule_background_cortex_prewarm_from_status(
        self,
        *,
        delay: float,
        reason: str,
        min_interval_s: float = 30.0,
    ) -> None:
        """Cooldowned recovery scheduling for status-observation paths.

        ``get_conversation_status()`` is polled by health endpoints and the
        neural stream. It may notice a completed failed warmup, but observation
        must not become an unbounded work generator.
        """

        # The cooldown stamp used to be written BEFORE the scheduling call and
        # outside any exclusion. Two status polls could both read a cold stamp
        # and both schedule; and if scheduling then raised, the stamp still
        # suppressed every other attempt for the full interval — a failure to
        # schedule bought silence instead of a retry.
        now = time.monotonic()
        interval = max(1.0, float(min_interval_s))
        with self._status_recovery_lock:
            if (now - self._last_status_recovery_schedule_at) < interval:
                logger.debug(
                    "Skipping status-triggered Cortex prewarm (%s); cooldown active.",
                    reason,
                )
                return
            previous = self._last_status_recovery_schedule_at
            self._last_status_recovery_schedule_at = now
        try:
            self._schedule_background_cortex_prewarm(delay=delay)
        except _INFERENCE_RECOVERABLE_ERRORS:
            # Nothing was scheduled, so nothing should be suppressed. Put the
            # stamp back and let the next observation try.
            with self._status_recovery_lock:
                if self._last_status_recovery_schedule_at == now:
                    self._last_status_recovery_schedule_at = previous
            raise

    def _schedule_background_cortex_prewarm(self, delay: float = 12.0) -> None:
        if is_shutdown_requested():
            logger.debug("Deferred cortex prewarm skipped: runtime shutdown requested.")
            return
        if self._deferred_prewarm_task and not self._deferred_prewarm_task.done():
            return

        async def _runner():
            next_delay = max(1.0, float(delay))
            for attempt in range(1, 7):
                await asyncio.sleep(next_delay)
                if is_shutdown_requested():
                    logger.debug("Deferred cortex prewarm stopped: runtime shutdown requested.")
                    return
                lane = self.get_conversation_status()
                lane_state = str(lane.get("state", "") or "").lower()
                if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                    return
                readiness_blockers = {
                    str(blocker or "").strip()
                    for blocker in (lane.get("readiness_blockers") or ())
                    if str(blocker or "").strip()
                }
                if lane.get("chat_dependencies_ready") is False and any(
                    blocker.startswith("chat_dependencies_")
                    for blocker in readiness_blockers
                ):
                    # The resident model is already warm; the server's single
                    # chat-dependency owner is materializing the evidence and
                    # memory readers that must exist before public readiness.
                    # Re-entering ensure_foreground_ready here is circular: its
                    # public status includes this blocker, so a successful
                    # model warmup raises the server-owned dependency blocker
                    # and a failed dependency retry used to be recorded as a
                    # critical InferenceGate failure. There is no work for
                    # this model prewarmer to do for either warming or failed
                    # dependency states.
                    logger.info(
                        "⏸️ Deferred cortex prewarm standing down while the "
                        "foreground chat-dependency owner finishes."
                    )
                    return
                if self._lane_reports_active_generation(lane):
                    logger.info(
                        "⏸️ Deferred cortex prewarm postponed while foreground generation is active."
                    )
                    next_delay = min(20.0, max(6.0, next_delay))
                    continue
                if lane_state == "failed":
                    if is_shutdown_requested():
                        return
                    if await asyncio.to_thread(self._rearm_runtime_failed_lane, force_probe=False):
                        lane = self.get_conversation_status()
                        lane_state = str(lane.get("state", "") or "").lower()
                    elif str(lane.get("last_failure_reason", "") or "").startswith(
                        _REARMABLE_LANE_FAILURE_PREFIXES
                    ):
                        logger.info(
                            "⏸️ Deferred cortex prewarm postponing while runtime lane is still unavailable (%s).",
                            lane.get("last_failure_reason") or "unknown",
                        )
                        next_delay = min(45.0, max(12.0, next_delay * 1.5))
                        continue
                    else:
                        logger.warning(
                            "⏸️ Deferred cortex prewarm cancelled: lane is in a failed state (%s).",
                            lane.get("last_failure_reason") or "unknown",
                        )
                        return
                if self._foreground_user_turn_active() or self._foreground_owner_active():
                    next_delay = min(20.0, max(6.0, next_delay))
                    continue
                warmup_deferral = self._cortex_warmup_deferral_reason("background")
                if warmup_deferral:
                    self._log_cortex_warmup_deferral(warmup_deferral, context="background")
                    next_delay = min(90.0, max(20.0, next_delay * 1.5))
                    continue
                try:
                    vm = InferenceGate._recent_virtual_memory()
                    total_gb = vm.total / float(1024**3)
                    available_gb = vm.available / float(1024**3)
                    critical_pressure = vm.percent >= (92.0 if total_gb >= 60.0 else 88.0)
                    critical_available = available_gb < (6.0 if total_gb >= 60.0 else 10.0)
                    if critical_pressure or critical_available:
                        logger.warning(
                            "⏸️ Deferred cortex prewarm postponed (attempt=%d pressure=%.1f%% available=%.1fGB).",
                            attempt,
                            vm.percent,
                            available_gb,
                        )
                        next_delay = min(45.0, max(12.0, next_delay * 1.5))
                        continue
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    record_degradation(
                        "inference_gate",
                        exc,
                        severity="warning",
                        action="continued deferred prewarm with conservative retry delay",
                    )
                    logger.debug("Deferred prewarm memory probe failed: %s", exc)

                try:
                    if is_shutdown_requested():
                        return
                    self._extend_startup_quiet_window(20.0)
                    # Background prewarm needs the same generous load budget
                    # as foreground chat so it does not half-warm then strand
                    # the next user turn in recovery.
                    await self.ensure_foreground_ready(timeout=300.0)
                    logger.info("✅ Deferred cortex prewarm completed.")
                    return
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    if _exception_reports_active_generation(exc):
                        logger.info(
                            "⏸️ Deferred cortex prewarm postponed while foreground generation is active."
                        )
                        next_delay = min(20.0, max(6.0, next_delay))
                        continue
                    if "visible_conversation_probe_missing" in str(exc):
                        logger.info(
                            "⏸️ Deferred cortex prewarm loaded the lane, but visible "
                            "conversation readiness is still unproven; the next "
                            "foreground user turn will prove or fail it."
                        )
                        next_delay = min(60.0, max(20.0, next_delay * 1.25))
                        continue
                    record_degradation(
                        "inference_gate",
                        exc,
                        severity="warning",
                        action="backed off deferred cortex prewarm and will retry",
                    )
                    logger.warning(
                        "⚠️ Deferred cortex prewarm failed (attempt=%d): %s", attempt, exc
                    )
                    next_delay = min(45.0, max(12.0, next_delay * 1.5))

            logger.warning(
                "⚠️ Deferred cortex prewarm exhausted retries; foreground turn will retry on demand."
            )

        runner_coro = _runner()
        try:
            task = get_task_tracker().create_task(
                runner_coro,
                name="InferenceGate.deferred_cortex_prewarm",
            )
        except RuntimeError:
            runner_coro.close()
            logger.debug("Deferred cortex prewarm skipped: no running event loop.")
            return

        if not isinstance(task, asyncio.Task):
            runner_coro.close()
            logger.debug(
                "Deferred cortex prewarm scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        self._deferred_prewarm_task = task
        # [STABILITY v53] Log exceptions from background tasks
        self._deferred_prewarm_task.add_done_callback(self._log_task_exception)

    def _rearm_runtime_failed_lane(self, *, force_probe: bool) -> bool:
        client = self._mlx_client
        if client is None or not hasattr(client, "refresh_runtime_availability"):
            return False

        lane = self.get_conversation_status()
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state != "failed" or not lane_reason.startswith(
            _REARMABLE_LANE_FAILURE_PREFIXES
        ):
            return False

        try:
            rearmed = bool(client.refresh_runtime_availability(force_probe=force_probe))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Failed to re-arm runtime-blocked Cortex lane: %s", exc)
            return False

        if rearmed:
            logger.info(
                "♻️ InferenceGate: re-armed the Cortex lane after transient runtime failure (%s).",
                lane_reason,
            )
        return rearmed

    @staticmethod
    def _lane_only_needs_visible_conversation_proof(lane: dict[str, Any] | None) -> bool:
        """Return True when the worker is loaded and only lacks a visible turn proof."""

        if not isinstance(lane, dict):
            return False
        blockers = {
            str(item)
            for item in (lane.get("readiness_blockers") or [])
            if str(item or "").strip()
        }
        if blockers not in (set(), {"visible_conversation_probe_missing"}):
            return False
        if str(lane.get("state", "") or "").lower() != "ready":
            return False
        if bool(lane.get("warmup_in_flight", False)):
            return False
        reason = str(lane.get("last_failure_reason", "") or "").strip()
        if reason and reason != "visible_conversation_probe_missing":
            return False
        if InferenceGate._lane_reports_active_generation(lane):
            return False
        # `warmup_attempted` defaulted to True, so a lane payload that simply
        # did not carry the field satisfied "the worker is loaded" on the
        # strength of a default. Absent evidence of a warmup is not evidence of
        # a warmup, and this predicate is what lets a foreground PROOF turn go
        # to a lane. Missing means no.
        if "warmup_attempted" not in lane:
            logger.debug(
                "Lane omits warmup_attempted; refusing the visible-conversation "
                "proof rather than assuming a warmup happened."
            )
            return False
        return bool(lane.get("warmup_attempted"))

    @classmethod
    def _lane_can_attempt_visible_conversation_turn(cls, lane: dict[str, Any] | None) -> bool:
        """Return True when a lane may serve the foreground turn that proves readiness."""

        return bool(
            isinstance(lane, dict)
            and (
                lane.get("conversation_ready")
                or cls._lane_only_needs_visible_conversation_proof(lane)
            )
        )

    @staticmethod
    def _foreign_owner_holds_model_lane() -> bool:
        """True when a live process other than this one owns the model lane.

        Fail-safe by construction: any error answers False, which restores the
        previous cold-boot wait rather than shortening a legitimate one.
        """
        try:
            from core.runtime.model_lane_control import (
                ProcessIdentity,
                get_model_lane_controller,
            )

            observations = get_model_lane_controller().owner_observations()
        except _INFERENCE_RECOVERABLE_ERRORS:
            # Never let a probe break a turn — and never let it widen into a
            # catch that would swallow a programming error too.
            return False
        if not observations:
            return False
        # Our own MLX worker is NOT a foreign owner, and matching this
        # process's pid alone does not establish that.
        #
        # LIVE 2026-08-17: the first turn after launch refused at exactly
        # 15.0s — this cap, to the tenth. model_lane_control builds owner ids
        # as f"subprocess:{os.getpid()}:{request_id}" from INSIDE the worker,
        # so Aura's own cortex loader carries the worker's pid and not
        # aura_main's. An exact-pid test reads that as somebody else holding
        # the lane, and short-caps the cold-boot warmup the cortex is in the
        # middle of doing.
        #
        # Lineage is the real question: a pid in our own process tree is us.
        own_pids = InferenceGate._own_process_tree_pids()
        for observation in observations:
            owner_id = str(getattr(observation, "owner_id", "") or "")
            if not owner_id:
                continue
            if any(f":{pid}:" in owner_id for pid in own_pids):
                continue  # our own lease is not a foreign holder
            return True
        return False

    @staticmethod
    def _own_process_tree_pids() -> frozenset[str]:
        """This process, its ancestors and its descendants, as pid strings."""
        pids: set[str] = set()
        try:
            pids.add(str(os.getpid()))
            pids.add(str(os.getppid()))
            proc = psutil.Process()
            for parent in proc.parents():
                pids.add(str(parent.pid))
            for child in proc.children(recursive=True):
                pids.add(str(child.pid))
        except (psutil.Error, *_INFERENCE_RECOVERABLE_ERRORS):
            pass
        return frozenset(p for p in pids if p and p != "0")

    def _foreground_warmup_timeout(
        self, lane_status: dict[str, Any], primary_timeout: float
    ) -> float:
        """Admission control for the foreground preflight — break the doom loop.

        A COLD first boot legitimately needs ~150s to load the 32B, and the
        user expects that one-time wait. But a RECOVERY (Cortex was ready, got
        force-killed on a first-token stall, is reloading) must NOT hold every
        foreground turn hostage for 90-180s — observed live (Jul 7 soak):
        turns 21-30 crawled to 200s+ while a single warm window played out.

        When the lane was EVER ready (``last_ready_at`` > 0), cap the wait
        short (floored to 15s by ensure_foreground_ready — one honest warm
        chance) and let the turn fall to the ready fallback tier; the warmup
        task is shielded, so Cortex keeps warming in the background and the
        NEXT turn gets it. AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S=180 restores
        the old behavior if this ever needs reverting live.
        """
        was_ever_ready = float(lane_status.get("last_ready_at", 0.0) or 0.0) > 0.0
        if was_ever_ready:
            return InferenceGate._env_float(
                "AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S", 15.0
            )
        # A lane held by SOMEONE ELSE is not a cold boot, and waiting the cold
        # budget for it waits for something that cannot happen.
        #
        # LIVE 2026-08-13: a training run (standalone:96317, CP399) held the
        # exclusive model lane. The cortex was admission-deferred 15 times in a
        # row; last_ready_at was 0.0 because it had never been ready THIS boot,
        # so every turn took the cold-boot branch and waited the full 180s for
        # a lane whose owner would hold it for hours. The brainstem — loaded,
        # weights present, not the lane owner — was never asked once, and the
        # user got "the live answer lane could not finish preparing".
        #
        # Preempting the owner is not the answer: that would destroy whatever
        # it is doing. Falling to the fallback tier is. The cortex keeps
        # warming behind a shielded task and takes over the moment the lane
        # frees.
        if InferenceGate._foreign_owner_holds_model_lane():
            return InferenceGate._env_float(
                "AURA_FOREGROUND_FOREIGN_LANE_WARMUP_CAP_S", 15.0
            )
        # [STABILITY v56] Cold 32B load can take 150s; give it at least 180s
        # or the primary timeout, whichever is greater.
        return max(180.0, float(primary_timeout))

    async def _await_warmup_deferral_clear(
        self,
        *,
        deadline: float,
        context: str,
        initial_reason: str,
    ) -> str:
        """Poll until the warmup deferral lifts, or the budget runs out.

        Returns "" when it cleared and the caller may proceed, or the last
        reason when it did not. Backpressure is a wait; only an exhausted
        budget is a failure.
        """

        reason = str(initial_reason or "")
        announced = False
        while reason and time.monotonic() < deadline:
            if is_shutdown_requested():
                return reason
            if not announced:
                logger.info(
                    "⏳ Cortex warmup deferred (%s); holding the turn for up to "
                    "%.0fs rather than answering with a failure.",
                    reason,
                    max(0.0, deadline - time.monotonic()),
                )
                announced = True
            await asyncio.sleep(0.5)
            lane = self.get_conversation_status()
            if self._lane_can_attempt_visible_conversation_turn(lane):
                return ""
            reason = str(self._cortex_warmup_deferral_reason(context) or "")
        if not reason and announced:
            logger.info("✅ Cortex warmup deferral cleared; the turn proceeds.")
        return reason

    async def ensure_foreground_ready(self, timeout: float | None = None) -> dict[str, Any]:  # noqa: ASYNC109
        """Ensure the 32B conversation lane has actually attempted warmup for this turn."""
        if is_shutdown_requested():
            raise RuntimeError("runtime_shutdown")
        # The clamp was max(15.0, …) with nothing above it, so a caller passing
        # inf — or a config that produced one — made this an unbounded wait on
        # a cold lane while holding the foreground. NaN was worse: every
        # comparison against it is False, so the wait ran to whichever branch
        # happened to exit. Neither is a timeout; both fall back to the default.
        requested = _finite(timeout, 90.0)
        if requested is None or requested <= 0.0:
            requested = 90.0
        timeout = max(15.0, min(_MAX_FOREGROUND_READY_WAIT_S, float(requested)))
        lane = self.get_conversation_status()
        if self._lane_can_attempt_visible_conversation_turn(lane):
            # Cortex is serving again — clear any post-thrash warmup cooldown.
            self._reset_cortex_warmup_backoff()
            return lane
        # Recovery cap: a lane that was EVER ready and is now warming/
        # recovering must NOT hold the turn for the full cold-boot budget.
        # The chat caller passes 180s; without this cap every turn blocked
        # ~206s on a recovering cortex ("Protected foreground lane failed
        # (lane_warming): Cortex timed out after 206s" — the 2026-07-15 soak
        # wall). _foreground_warmup_timeout returns 15s for a recovery and
        # the cold budget for a genuine cold boot; take the tighter of the
        # two so a recovering turn falls to the fast fallback while Cortex
        # re-warms in the background (its warmup task is shielded).
        # A lane that was EVER ready gets the short recovery cap, and that
        # short wait is a DESIGNED handoff: the warmup task is shielded, the
        # cortex keeps loading in the background, and this turn falls to the
        # ready fallback. Timing out on it is the mechanism working, not a
        # stuck load — see _warmup_timeout_is_designed_handoff below.
        recovery_handoff = float(lane.get("last_ready_at", 0.0) or 0.0) > 0.0
        timeout = min(timeout, self._foreground_warmup_timeout(lane, timeout))
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state == "failed" and lane_reason.startswith(
            _REARMABLE_LANE_FAILURE_PREFIXES
        ):
            if await asyncio.to_thread(self._rearm_runtime_failed_lane, force_probe=True):
                lane = self.get_conversation_status()
            else:
                raise RuntimeError(lane_reason)
        if not self._mlx_client or not hasattr(self._mlx_client, "warmup"):
            raise RuntimeError("foreground_lane_unavailable")

        task: asyncio.Task | None = None
        # The turn's own budget, minus a slice so a cleared deferral still
        # leaves time for the warmup it was waiting for.
        deferral_deadline = time.monotonic() + max(0.0, float(timeout) * 0.6)
        try:
            async with _thread_lock_context(
                self._foreground_ready_lock,
                timeout_s=min(timeout, 30.0),
                label="foreground_ready_lock",
            ):
                lane = self.get_conversation_status()
                if self._lane_can_attempt_visible_conversation_turn(lane):
                    return lane
                if self._prewarm_task and not self._prewarm_task.done():
                    task = self._prewarm_task
                else:
                    warmup_deferral = self._cortex_warmup_deferral_reason("foreground")
                    if warmup_deferral:
                        await self._shed_background_workers_for_memory_pressure(
                            force=True,
                            reason="foreground_cortex_warmup_admission",
                        )
                        gc.collect()
                        warmup_deferral = self._cortex_warmup_deferral_reason("foreground")
                    if warmup_deferral:
                        # A deferral says "not yet", not "no". Raising here
                        # spent the turn instantly and handed the person a
                        # canned failure while the budget it was given went
                        # unused.
                        #
                        # LIVE 2026-08-17: the first message after launch died
                        # in 15s with "the live answer lane could not finish
                        # preparing", on a 90s budget, with last_ready_at=0.0
                        # and no foreign lane owner. Nothing had timed out —
                        # the cortex prewarm was standing down behind the
                        # foreground chat-dependency owner, and that deferral
                        # was reported as a lane failure. Ten seconds later the
                        # same message served normally.
                        #
                        # So wait it out. Re-check until the deferral clears or
                        # the budget really is gone; only then is it a failure.
                        warmup_deferral = await self._await_warmup_deferral_clear(
                            deadline=deferral_deadline,
                            context="foreground",
                            initial_reason=warmup_deferral,
                        )
                    if warmup_deferral:
                        self._log_cortex_warmup_deferral(warmup_deferral, context="foreground")
                        if hasattr(self._mlx_client, "note_lane_recovering"):
                            self._mlx_client.note_lane_recovering(
                                "foreground_warmup_deferred_memory_pressure"
                            )
                        raise RuntimeError(f"foreground_warmup_deferred:{warmup_deferral}")
                    self._extend_startup_quiet_window(20.0)
                    if is_shutdown_requested():
                        raise RuntimeError("runtime_shutdown")
                    self._prewarm_task = get_task_tracker().create_task(
                        self._mlx_client.warmup(),
                        name="InferenceGate.ensure_foreground_ready",
                    )
                    task = self._prewarm_task
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

        try:
            task_loop = getattr(task, "get_loop", lambda: asyncio.get_running_loop())()
            current_loop = asyncio.get_running_loop()
            if task_loop is not current_loop:

                async def _await_foreign_task() -> Any:
                    return await task

                future = asyncio.run_coroutine_threadsafe(_await_foreign_task(), task_loop)
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError:
            # A foreground warmup that overruns its COLD budget is the cortex
            # failing to load in time — the same GPU-thrash signal as a
            # stuck-load kill, but with no force-kill to observe (the worker
            # just stays "warming"). Feed that into the warmup backoff so
            # repeated stalls defer cortex warmup and free the single GPU slot
            # for the resident fallback. Without it the cortex load and the
            # fallback cold-load fight over one GPU slot and neither wins.
            #
            # The RECOVERY cap is the opposite case and must not be counted.
            # It is a deliberate 15s handoff — wait briefly, let this turn use
            # the ready fallback, leave the shielded warmup running — and
            # counting it as damage is what broke the 2026-07-25 endurance
            # probe: 62 load attempts, none allowed to finish, because every
            # designed handoff incremented the stuck-load counter until the
            # backoff deferred warmup by 240s and the cortex could never
            # complete a load. 173 of 200 turns went unanswered on a lane that
            # was never given the chance to finish warming. A deferral is not
            # damage; the same category error, one layer down.
            if not recovery_handoff:
                # The shielded warmup is still running: this budget overran,
                # nothing was killed. Same backoff, honest evidence.
                self._note_cortex_warmup_overrun()
            else:
                logger.info(
                    "🧠 Foreground recovery handoff after %.0fs — cortex keeps "
                    "warming in the background, this turn uses the ready lane.",
                    timeout,
                )
            if hasattr(self._mlx_client, "note_lane_recovering"):
                self._mlx_client.note_lane_recovering("foreground_warmup_timeout")
            raise
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            if hasattr(self._mlx_client, "note_lane_failed"):
                self._mlx_client.note_lane_failed(f"foreground_warmup_failed:{type(exc).__name__}")
            raise RuntimeError("foreground_warmup_failed") from exc

        lane = self.get_conversation_status()
        if not lane.get("conversation_ready"):
            if self._lane_only_needs_visible_conversation_proof(lane):
                return lane
            raise RuntimeError(str(lane.get("last_failure_reason") or "foreground_lane_not_ready"))
        return lane

    def _confirmed_cortex_warmup(
        self, warmup_result: Any
    ) -> tuple[bool, dict[str, Any], str]:
        """Require process and lane evidence before reporting a warmup as successful."""
        lane = self.get_conversation_status()
        state = str(lane.get("state", "") or "").strip().lower()
        blockers = [
            str(blocker)
            for blocker in (lane.get("readiness_blockers") or [])
            if str(blocker or "").strip()
        ]
        try:
            worker_alive = bool(self._mlx_client and self._mlx_client.is_alive())
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            worker_alive = False
            blockers.append(f"worker_probe_failed:{type(exc).__name__}")

        ready = bool(
            warmup_result is not False
            and not is_shutdown_requested()
            and worker_alive
            and state == "ready"
            and lane.get("conversation_ready")
        )
        if ready:
            return True, lane, ""
        if is_shutdown_requested():
            reason = "runtime_shutdown"
        elif warmup_result is False:
            reason = str(lane.get("last_failure_reason") or "warmup_deferred")
        elif not worker_alive:
            reason = "worker_not_alive"
        elif state != "ready":
            reason = f"lane_{state or 'unknown'}"
        elif not lane.get("conversation_ready"):
            reason = ",".join(blockers[:3]) or "conversation_not_ready"
        else:
            reason = "warmup_not_confirmed"
        return False, lane, reason

    async def _ensure_cortex_recovery(self) -> None:
        """Proactively recover the 32B primary brain if it died (e.g., laptop sleep).

        Without this, background tasks keep the 7B alive indefinitely and the 32B
        never gets a chance to respawn because background requests are locked to
        tertiary tier.  Rate-limited to one attempt per 3s.
        """
        if is_shutdown_requested():
            logger.debug("Primary cortex recovery skipped: runtime shutdown requested.")
            return
        if not self._mlx_client:
            return
        if proof_run_active(origin="cortex_recovery") and proof_model_tier() != "primary":
            logger.debug(
                "Primary cortex recovery skipped during non-primary proof lane (%s).",
                proof_model_tier(),
            )
            return
        if not hasattr(self._mlx_client, "is_alive"):
            return
        if self._mlx_client.is_alive():
            return  # Primary is fine

        now = time.monotonic()
        if (now - self._last_cortex_check) < 3.0:
            return  # [STABILITY v51] Rate limit: 3s between attempts
        self._last_cortex_check = now

        if self._cortex_recovery_attempts >= 5:
            # [HARDENING v54] Exponential backoff: 30s after 5 failures, 60s after 10,
            # capped at 120s. The previous 5-minute hard lockout made the cortex
            # unreachable for entire conversation windows. Never permanently give up.
            exhausted_at = getattr(self, "_cortex_recovery_exhausted_at", 0.0)
            cooldown = min(120.0, 30.0 * (1 + (self._cortex_recovery_attempts - 5) // 5))
            if exhausted_at == 0.0:
                self._cortex_recovery_exhausted_at = now
                logger.warning(
                    "[RECOVERY] Primary cortex: %d failures reached. Will retry in %.0fs.",
                    self._cortex_recovery_attempts,
                    cooldown,
                )
                return
            if (now - exhausted_at) < cooldown:
                return  # Rate-limit: exponential backoff
            logger.warning(
                "[RECOVERY] Primary cortex: %.0fs cooldown elapsed. Resetting counter and retrying.",
                cooldown,
            )
            self._cortex_recovery_attempts = 0
            self._cortex_recovery_exhausted_at = 0.0

        if self._cortex_recovery_in_progress:
            return  # Already recovering — don't double-spawn
        if not hasattr(self._mlx_client, "warmup"):
            return
        # Reserve recovery ownership IMMEDIATELY — before any await. The
        # policy checks below suspend the coroutine, and two concurrent
        # callers could otherwise both pass the in-progress check and race
        # into duplicate warmups or duplicate private process cleanup.
        self._cortex_recovery_in_progress = True
        reservation_transferred = False
        try:
            lane = self.get_conversation_status()
            lane_state = str(lane.get("state", "") or "").lower()
            lane_reason = str(lane.get("last_failure_reason", "") or "")
            cold_start_recovery = lane_state in {
                "cold",
                "spawning",
                "handshaking",
                "warming",
            } or not bool(lane.get("warmup_attempted", False))
            if lane_state == "failed" and lane_reason.startswith(
                _REARMABLE_LANE_FAILURE_PREFIXES
            ):
                if await asyncio.to_thread(self._rearm_runtime_failed_lane, force_probe=True):
                    lane = self.get_conversation_status()
                    lane_state = str(lane.get("state", "") or "").lower()
                    lane_reason = str(lane.get("last_failure_reason", "") or "")
                    cold_start_recovery = lane_state in {
                        "cold",
                        "spawning",
                        "handshaking",
                        "warming",
                    } or not bool(lane.get("warmup_attempted", False))
                else:
                    return
            if lane.get("warmup_in_flight"):
                return
            if self._foreground_user_turn_active() or self._foreground_owner_active():
                return
            if cold_start_recovery:
                if self._prewarm_task is not None and not self._prewarm_task.done():
                    logger.debug("Cold-start Cortex recovery skipped; deferred prewarm task is already scheduled.")
                    return
                if not self._boot_should_schedule_deferred_prewarm():
                    self._log_cold_cortex_policy_deferred()
                    return
            warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
            if warmup_deferral:
                self._log_cortex_warmup_deferral(warmup_deferral, context="recovery")
                return
        finally:
            if not reservation_transferred:
                self._cortex_recovery_in_progress = False

        async def _background_recover():
            if is_shutdown_requested():
                self._cortex_recovery_in_progress = False
                return
            self._cortex_recovery_in_progress = True
            # The attempt counter used to increment HERE, before the warmup
            # admission recheck below. Memory pressure can change between
            # scheduling and running, and a deferral loads nothing — but it
            # still spent an attempt, so repeated deferrals walked the counter
            # to the exponential cooldown without one load ever being tried.
            # An attempt is now counted where a load is actually attempted.
            attempt_counted = False

            if self._cortex_recovery_attempts + 1 == 3:
                logger.warning(
                    "🧹 [RECOVERY] 3 failed attempts. Forcing deep GC and stale process cleanup..."
                )
                import gc

                gc.collect()
                # Bind the kill to the exact client + process handle observed
                # NOW, and refuse when the worker is legitimately loading or
                # serving — an unowned kill here was the doom-loop trigger
                # (kill mid-load → warmup_deferred → repeat).
                kill_client = self._mlx_client
                kill_process = getattr(kill_client, "_process", None)
                # CP126: "Recovery kills a private process handle without
                # generation ownership proof ... no PID start-time ...
                # binding at the kill point."
                #
                # Every check below is followed by an await, and the kill
                # itself hops to a thread. In that window the worker can
                # exit, the client can spawn a replacement, and the
                # replacement can land on the same PID — after which this
                # kills a healthy new worker and every log line reads like
                # recovery working. Bind to (pid, create_time) here and
                # re-check it at the kill.
                kill_identity = capture_identity(
                    kill_process, label="cortex_recovery_attempt_3"
                )
                if kill_process is None:
                    logger.debug("[RECOVERY] No worker process handle to clean up.")
                elif self._cortex_worker_is_legitimately_loading(kill_client):
                    logger.info(
                        "[RECOVERY] Skipping stale-process kill: worker is legitimately loading."
                    )
                elif self._lane_reports_active_generation(self.get_conversation_status()):
                    logger.info(
                        "[RECOVERY] Skipping stale-process kill: lane reports an active generation."
                    )
                elif not assert_owned(
                    kill_identity,
                    getattr(kill_client, "_process", None),
                    action="stale-worker kill",
                    subsystem="inference_gate.recovery",
                ):
                    # Not a failure: the worker this decision was about is
                    # already gone, which is the outcome the kill wanted.
                    logger.info(
                        "[RECOVERY] Stale-process kill unnecessary; the bound "
                        "worker is no longer the current one."
                    )
                else:
                    try:
                        await asyncio.to_thread(
                            kill_client._kill_and_join_blocking, kill_process
                        )
                    except _INFERENCE_RECOVERABLE_ERRORS as _e:
                        _record_inference_degradation(
                            _e,
                            action="continued background recovery loop with degraded signal",
                        )
                        logger.debug("Ignored Exception in inference_gate.py killing process: %s", _e)

            try:
                warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
                if warmup_deferral:
                    self._log_cortex_warmup_deferral(warmup_deferral, context="recovery")
                    return
                if is_shutdown_requested():
                    logger.debug("Primary cortex recovery stopped before warmup: runtime shutdown requested.")
                    return
                if self.unproven_cortex_loads():
                    # A cancelled load has not been observed to stop. Starting
                    # a second 20GB load on top of it is the overlap the
                    # cancellation was supposed to prevent.
                    await self.await_abandoned_cortex_loads()
                    if self.unproven_cortex_loads():
                        logger.warning(
                            "♻️ [RECOVERY] Deferring warmup: a cancelled Cortex load has not stopped."
                        )
                        return
                # A load is about to be attempted. THIS is an attempt.
                self._cortex_recovery_attempts += 1
                attempt_counted = True
                if cold_start_recovery:
                    logger.info(
                        "♻️ [STARTUP] Primary %s cortex is cold. Starting warmup "
                        "(Attempt %d/5)...",
                        _primary_lane_label(),
                        self._cortex_recovery_attempts,
                    )
                else:
                    logger.warning(
                        "♻️ [RECOVERY] Primary %s cortex is dead. Triggering background "
                        "respawn (Attempt %d/5)...",
                        _primary_lane_label(),
                        self._cortex_recovery_attempts,
                    )
                self._prewarm_task = get_task_tracker().create_task(
                    self._mlx_client.warmup(),
                    name="InferenceGate.cortex_recovery",
                )
                # 32B fused model is ~37GB across 7 shards. Cold-load on Apple
                # Silicon routinely takes 90-150s on the first attempt after a
                # crash; the previous 60s budget guaranteed five back-to-back
                # timeouts and a 5-minute lockout. Give warmup the room it
                # actually needs.
                warmup_result = await asyncio.wait_for(
                    asyncio.shield(self._prewarm_task), timeout=420.0
                )
                ready, recovered_lane, incomplete_reason = self._confirmed_cortex_warmup(
                    warmup_result
                )
                if not ready:
                    if is_shutdown_requested():
                        logger.info(
                            "🛑 [RECOVERY] Primary %s cortex warmup stopped during runtime shutdown.",
                            _primary_lane_label(),
                        )
                    else:
                        logger.warning(
                            "⚠️ [RECOVERY] Primary %s cortex warmup did not establish readiness "
                            "(state=%s, reason=%s).",
                            _primary_lane_label(),
                            recovered_lane.get("state", "unknown"),
                            incomplete_reason,
                        )
                    return
                if cold_start_recovery:
                    logger.info(
                        "✅ [STARTUP] Primary %s cortex warmup complete.",
                        _primary_lane_label(),
                    )
                else:
                    logger.info(
                        "✅ [RECOVERY] Primary %s cortex restored after disruption.",
                        _primary_lane_label(),
                    )
                self._cortex_recovery_attempts = 0
                self._cortex_recovery_exhausted_at = 0.0
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued background recovery loop with degraded signal",
                )
                if cold_start_recovery:
                    logger.error(
                        "⚠️ [STARTUP] Primary %s cortex warmup failed (Attempt %d/5): %s",
                        _primary_lane_label(),
                        self._cortex_recovery_attempts,
                        exc,
                    )
                else:
                    logger.error(
                        "⚠️ [RECOVERY] Primary %s cortex is dead. Triggering background "
                        "respawn (Attempt %d/5): %s",
                        _primary_lane_label(),
                        self._cortex_recovery_attempts,
                        exc,
                    )
            finally:
                if not attempt_counted:
                    logger.debug(
                        "♻️ [RECOVERY] No load was attempted; recovery attempt not counted."
                    )
                # [STABILITY v51] ALWAYS clear the flag, even on unexpected exceptions.
                self._cortex_recovery_in_progress = False

        # [STABILITY v53] Wrap fire-and-forget task with exception logging
        # so crashes are visible instead of silently lost.
        # Reserve recovery ownership before scheduling. Without this reservation,
        # the foreground caller can observe ``False`` immediately after this method
        # returns and start a second inline warmup before the task gets CPU time.
        if is_shutdown_requested():
            logger.debug("Primary cortex recovery task not scheduled: runtime shutdown requested.")
            return
        self._cortex_recovery_in_progress = True
        recovery_coro = _background_recover()
        try:
            task = get_task_tracker().create_task(recovery_coro, name="cortex_recovery")
        except RuntimeError:
            recovery_coro.close()
            self._cortex_recovery_in_progress = False
            logger.debug("Cortex recovery skipped: no running event loop.")
            return
        if not isinstance(task, asyncio.Task):
            recovery_coro.close()
            self._cortex_recovery_in_progress = False
            logger.debug(
                "Cortex recovery task scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        # Own the handle so shutdown can cancel and await this recovery like
        # the named prewarm/maintenance tasks — an anonymous recovery task
        # could continue into shutdown and recreate warmup activity.
        self._status_recovery_task = task

        def _finish_recovery(completed: asyncio.Task) -> None:
            # Cancellation can happen before the coroutine reaches its ``finally``
            # block, so the scheduling boundary also owns clearing this reservation.
            self._cortex_recovery_in_progress = False
            if getattr(self, "_status_recovery_task", None) is completed:
                self._status_recovery_task = None
            self._log_task_exception(completed)

        task.add_done_callback(_finish_recovery)

    async def _respawn_cortex_if_needed(self) -> None:
        """Respawn the primary cortex if it's dead.

        Called by HealthRouter and message_handling when inference returns empty.
        Delegates to _ensure_cortex_recovery() which has proper rate-limiting,
        warm-up sequencing, and retry budgets.
        """
        if is_shutdown_requested():
            logger.debug("_respawn_cortex_if_needed skipped: runtime shutdown requested.")
            return
        if (
            self._mlx_client
            and hasattr(self._mlx_client, "is_alive")
            and self._mlx_client.is_alive()
        ):
            return  # Cortex is fine — nothing to do
        if self._cortex_recovery_in_progress:
            logger.debug("_respawn_cortex_if_needed: recovery already in progress.")
            return  # Already recovering — don't double-spawn
        logger.info("🔄 _respawn_cortex_if_needed: cortex is dead, delegating to recovery.")
        await self._ensure_cortex_recovery()

    def tier_health_receipt(self) -> dict[str, Any]:
        """What the last tier sweep observed, and what it actuated.

        The sweep's ``{tier: status}`` return says "alive" without saying what
        "alive" was read off, and says nothing at all about the recovery and
        warmup it may have started on the way. Both belong in the record.
        """
        return copy.deepcopy(getattr(self, "_tier_health_receipt", {}))

    async def observe_tier_health(self) -> dict[str, str]:
        """Read tier health and actuate NOTHING.

        Use this for monitoring cadence. :meth:`ensure_all_tiers_healthy` is
        the same probe with repair enabled, and repair here means spawning a
        Cortex recovery or a 7B warmup — GPU and RAM, on a timer.
        """
        return await self.ensure_all_tiers_healthy(repair=False)

    async def ensure_all_tiers_healthy(self, *, repair: bool = True) -> dict[str, str]:
        """Proactive health check for ALL inference tiers. Called by MindTick.

        Returns a dict of {tier: status} for monitoring. With ``repair`` set
        (the default, which is what MindTick wants) a dead Cortex schedules
        recovery and a dead brainstem may schedule a warmup, so this is not a
        pure observation — pass ``repair=False``, or call
        :meth:`observe_tier_health`, when it needs to be. Whatever was
        actuated is named in :meth:`tier_health_receipt`.
        """
        if is_shutdown_requested():
            return {"cortex": "shutdown"}
        statuses = {}
        evidence: dict[str, str] = {}
        actuated: list[str] = []

        # Primary cortex
        try:
            if self._mlx_client and hasattr(self._mlx_client, "is_alive"):
                # [STABILITY v53] Detect warming/recovering states so MindTick
                # doesn't report 'dead' during cold start.
                lane_state = getattr(self._mlx_client, "_lane_state", "cold")

                if self._mlx_client.is_alive():
                    # Process liveness alone used to be reported as "alive".
                    # It is the weakest evidence this file has: a worker that
                    # is loading weights, wedged on a handshake, or holding a
                    # lock it will never release is alive by that measure. The
                    # lane read costs nothing extra here and says whether the
                    # process can actually take a turn.
                    statuses["cortex"] = "alive"
                    evidence["cortex"] = "process_liveness"
                    try:
                        lane = self.get_conversation_status()
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        _record_inference_degradation(
                            exc,
                            action="labelled Cortex alive on process liveness alone "
                            "after the lane status read failed",
                        )
                        lane = None
                    if isinstance(lane, dict):
                        if bool(lane.get("conversation_ready", False)):
                            evidence["cortex"] = "conversation_ready"
                        elif self._active_generation_is_progressing(lane):
                            evidence["cortex"] = "generation_progressing"
                        else:
                            # Alive but not able to hold a conversation. Not
                            # dead — no incident, no recovery — but the label
                            # must not claim more than the lane proved.
                            statuses["cortex"] = "alive_not_conversation_ready"
                            evidence["cortex"] = "process_liveness_only"
                elif self._cortex_recovery_in_progress or lane_state in (
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                ):
                    statuses["cortex"] = "recovering"
                    evidence["cortex"] = f"lane_state:{lane_state}"
                else:
                    statuses["cortex"] = "dead"
                    evidence["cortex"] = "process_not_alive"
                    if repair:
                        # Trigger recovery if not already in progress.
                        await self._ensure_cortex_recovery()
                        actuated.append("cortex_recovery")
            else:
                statuses["cortex"] = "not_initialized"
                evidence["cortex"] = "no_client"
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued tier-health sweep after one tier probe failed",
            )
            statuses["cortex"] = f"error:{e}"
            evidence["cortex"] = "probe_error"

        # Brainstem
        try:
            deferral_reason = self._background_local_deferral_reason(origin="tier_health")
            warm_local_tiers = _FLAG_HEALTH_WARM_LOCAL_TIERS.value().strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if deferral_reason:
                statuses["brainstem"] = f"deferred:{deferral_reason}"
                evidence["brainstem"] = f"policy_deferral:{deferral_reason}"
            else:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import get_brainstem_path

                brainstem = get_mlx_client(model_path=str(get_brainstem_path()))
                if brainstem and hasattr(brainstem, "is_alive"):
                    lane_state = getattr(brainstem, "_lane_state", "cold")

                    if brainstem.is_alive():
                        statuses["brainstem"] = "alive"
                        evidence["brainstem"] = "process_liveness"
                    elif lane_state in ("spawning", "handshaking", "warming", "recovering"):
                        statuses["brainstem"] = "recovering"
                        evidence["brainstem"] = f"lane_state:{lane_state}"
                    elif not warm_local_tiers:
                        # Brainstem is a demand-loaded background lane. A cold
                        # worker is healthy standby unless policy explicitly
                        # requires it to remain warm; calling it dead creates a
                        # false incident while the required Cortex lane is live.
                        statuses["brainstem"] = "standby"
                        evidence["brainstem"] = "cold_by_policy"
                    else:
                        statuses["brainstem"] = "dead"
                        evidence["brainstem"] = "process_not_alive"
                        # Tier health sweeps are observability by default. They
                        # must not spawn a background 7B worker while a foreground
                        # Cortex turn or proof run owns the local runtime.
                        if repair and hasattr(brainstem, "warmup"):
                            get_task_tracker().create_task(brainstem.warmup())
                            statuses["brainstem"] = "recovering"
                            actuated.append("brainstem_warmup")
                else:
                    statuses["brainstem"] = "not_initialized"
                    evidence["brainstem"] = "no_client"
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued tier-health sweep after one tier probe failed",
            )
            statuses["brainstem"] = f"error:{e}"
            evidence["brainstem"] = "probe_error"

        # Reflex (CPU) — the model file has to be readable, not merely named
        try:
            from core.brain.llm.model_registry import get_fallback_path

            fallback_path = get_fallback_path()
            statuses["reflex"], evidence["reflex"] = await asyncio.to_thread(
                _reflex_model_status, str(fallback_path) if fallback_path else ""
            )
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="reported reflex tier as unknown after the model probe failed",
            )
            statuses["reflex"] = "unknown"
            evidence["reflex"] = "probe_error"

        self._tier_health_receipt = {
            "observed_at": time.time(),
            "statuses": dict(statuses),
            # What each label is standing on. "alive" off process liveness and
            # "alive" off a conversation-ready lane are not the same claim.
            "evidence": dict(evidence),
            "repair_enabled": bool(repair),
            # Empty on an observe-only sweep, by construction.
            "actuated": list(actuated),
        }
        return statuses

    @staticmethod
    def _normalize_tier(prefer_tier: str | None) -> str:
        tier = str(prefer_tier or "primary").strip().lower()
        aliases = {
            "local": "primary",
            "local_deep": "secondary",
            "local_fast": "tertiary",
            "fast": "tertiary",
            "deep": "secondary",
        }
        return aliases.get(tier, tier)

    @staticmethod
    def _origin_is_user_facing(origin: str | None) -> bool:
        """Whether this origin gets protected foreground routing.

        The label is caller-supplied and unauthenticated, so the matching rule
        is the whole security property. It used to be "any underscore-delimited
        token anywhere in the string", which made ``background_user``,
        ``audit_probe`` and ``internal_test_sweep`` all user-facing — a caller
        could inherit the protected Cortex lane, its memory admission work and
        its worker shedding by putting one allowlisted word anywhere in a name.

        Now the origin must BE an allowlisted label or START with one, and a
        leading word that says the opposite vetoes outright. Checked against
        every origin literal in the tree: no real origin changes meaning.
        """
        normalized = str(origin or "").strip().lower().replace("-", "_")
        if not normalized:
            return False
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_") :]
        if not normalized:
            return False
        head = normalized.split("_", 1)[0]
        if head in _NOT_USER_FACING_ORIGIN_PREFIXES:
            return False
        if normalized in _USER_FACING_ORIGINS:
            return True
        return any(normalized.startswith(f"{prefix}_") for prefix in _USER_FACING_ORIGINS)

    @staticmethod
    def _a_user_turn_is_in_flight() -> bool:
        """Whether a user turn is being served right now, by either account.

        The orchestrator reports a turn once its tick starts. Preflight runs
        before that and is still part of the turn — the request is open, the
        lane is reserved, and the question has been recorded for the turn. A
        planner that runs in preflight was refused on the orchestrator's
        account alone, so both are consulted.

        Both are runtime state. Neither can be asserted by a caller.
        """
        if InferenceGate._foreground_user_turn_active():
            return True
        try:
            from core.conversation.session_scope import current_user_question

            return bool(current_user_question())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _foreground_user_turn_active() -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False
            status = getattr(orch, "status", None)
            if not getattr(status, "is_processing", False):
                return False
            current_origin = getattr(orch, "_current_origin", "")
            if not InferenceGate._origin_is_user_facing(current_origin):
                return False
            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # CP126 e193508b. A probe FAILURE returned False, meaning "no
            # foreground turn is running" — so background warmup, worker
            # recycling and load shedding were free to proceed on top of a
            # real user generation whenever shared status was unavailable.
            # That is the interruption this probe exists to prevent, granted
            # by the probe's own failure.
            #
            # Unknown is treated as OCCUPIED: at worst a background task
            # waits for the next cycle, which costs nothing a user sees.
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="assumed a foreground turn is active after an unreadable orchestrator probe",
            )
            return True

    @staticmethod
    def _foreground_quiet_window_active() -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False
            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    def _safe_boot_background_guard_active(self) -> bool:
        """Reserve launch headroom for the live conversation lane."""
        if not self._desktop_safe_boot_enabled():
            return False
        try:
            startup_guard_secs = float(
                _FLAG_SAFE_BOOT_BACKGROUND_GUARD_SECS.value()
            )
        except _INFERENCE_RECOVERABLE_ERRORS:
            startup_guard_secs = 180.0
        if startup_guard_secs <= 0:
            return False
        return (time.monotonic() - self._created_at) < startup_guard_secs

    def _should_quiet_background_for_cortex_startup(self) -> bool:
        """Hold background inference while the live 32B lane is booting or reserving headroom."""
        lane = self.get_conversation_status()
        if self._safe_boot_background_guard_active():
            return True
        if not self._foreground_quiet_window_active():
            return False
        if lane.get("conversation_ready"):
            return False

        state = str(lane.get("state", "") or "").strip().lower()
        if lane.get("warmup_in_flight"):
            return True
        return state in {"cold", "spawning", "handshaking", "warming", "recovering"}

    @staticmethod
    def _background_memory_pressure_active() -> bool:
        try:
            vm = InferenceGate._recent_virtual_memory()
            total_gb = vm.total / float(1024**3)
            available_gb = vm.available / float(1024**3)
            max_pressure = float(
                os.environ.get(
                    "AURA_BACKGROUND_LOCAL_MAX_PRESSURE_PCT",
                    "82" if total_gb >= 60.0 else "78",
                )
            )
            min_available_gb = float(
                os.environ.get(
                    "AURA_BACKGROUND_LOCAL_MIN_AVAILABLE_GB",
                    "12" if total_gb >= 60.0 else "10",
                )
            )
            return bool(vm.percent >= max_pressure or available_gb <= min_available_gb)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="failed closed and deferred background local inference after memory probe failure",
                severity="warning",
            )
            return True

    def background_local_deferral_reason(self, *, origin: str | None = None) -> str | None:
        """Public name for the background quiet policy.

        CP126 aa66b0ac: the MLX boundary reached into the private method, so a
        rename here would silently turn the check into "no deferral" at the one
        place that protects a foreground turn from a stale background respawn.
        """
        return self._background_local_deferral_reason(origin=origin)

    def _background_local_deferral_reason(self, *, origin: str | None = None) -> str | None:
        try:
            from core.runtime.proof_policy import proof_run_active

            if proof_run_active(origin=origin):
                return "proof_foreground_reserved"
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="kept background local deferral conservative after proof policy probe failed",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return "foreground_reserved"
        if self._foreground_headroom_reserved("primary"):
            return "foreground_headroom_reserved"
        if self._should_quiet_background_for_cortex_startup():
            return "cortex_startup_quiet"
        if self._foreground_quiet_window_active():
            return "foreground_quiet_window"

        lane = self.get_conversation_status()
        try:
            from core.brain.llm.model_registry import get_local_backend

            if get_local_backend() != "mlx" and lane.get("conversation_ready"):
                return "cortex_resident"
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="kept background local deferral conservative after policy probe failed",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        lane_state = str(lane.get("state", "") or "").strip().lower()
        if not lane.get("conversation_ready") and lane_state == "failed":
            return "cortex_failed"
        if self._safe_boot_background_guard_active():
            return "cortex_startup_quiet"
        if self._desktop_safe_boot_enabled() and not self._desktop_background_local_enabled():
            return "desktop_background_disabled"
        if self._desktop_safe_boot_enabled() and not lane.get("conversation_ready"):
            if self._background_memory_pressure_active():
                if lane_state in {
                    "cold",
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                    "failed",
                }:
                    return "memory_pressure"
        if self._background_memory_pressure_active():
            if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                return "memory_pressure"
            if lane_state in {"spawning", "handshaking", "warming", "recovering", "failed"}:
                return "memory_pressure"
        return None

    @staticmethod
    def _background_endpoint_headroom_deferral() -> tuple[str | None, dict[str, str]]:
        """Refuse before prompt work only when every background lane is closed.

        The router remains dispatch authority. This preflight reuses its exact
        endpoint policy so a background request cannot spend CPU and memory on
        state collection, grounding, and prompt construction only to discover
        at dispatch that neither Brainstem nor Reflex can be admitted.
        """
        try:
            from core.brain.llm_health_router import (
                desktop_background_endpoint_deferral_reasons,
            )

            reasons = desktop_background_endpoint_deferral_reasons(
                (BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT)
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action=(
                    "failed closed before background prompt construction after "
                    "endpoint admission probe failed"
                ),
                severity="warning",
            )
            reasons = {
                BRAINSTEM_ENDPOINT: "desktop_background_endpoint_probe_failed",
                FALLBACK_ENDPOINT: "desktop_background_endpoint_probe_failed",
            }
        if not all(name in reasons for name in (BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT)):
            return None, reasons
        return "all_background_endpoints_deferred", reasons

    async def _shed_background_workers_for_memory_pressure(
        self,
        *,
        force: bool = False,
        reason: str = "background_memory_pressure_shed",
    ) -> None:
        now = time.monotonic()
        if not force and (now - self._last_background_memory_shed_at) < 20.0:
            return

        protected_foreground = str(reason or "") == "protected_foreground_shed"
        if protected_foreground and self._primary_lane_ready():
            # A load reserve answers whether a cold model may be admitted. It is
            # not capacity that an already-resident model needs again. Applying
            # the load threshold to every protected turn fabricated a warning
            # and could evict the fallback ladder while the 32B was serving.
            self._last_background_memory_shed_at = now
            logger.debug(
                "InferenceGate: primary lane is resident; protected foreground "
                "shed is unnecessary."
            )
            return

        # Never shed the small fallback models when memory is abundant. They
        # are the guaranteed fast-answer path while the 32B cortex warms; with
        # the router now routing AROUND a not-ready cortex, shedding them left
        # nothing resident to answer and cascaded into a no-reply death spiral
        # (2026-07-15 soak: 7B >56s, 1.5B >14.7s, all thrashing to reload
        # despite 42GB free). Only shed when free memory genuinely cannot hold
        # the cortex alongside them. `force=True` callers still respect this —
        # a warmup deferred for admission/routing reasons is NOT a memory
        # problem, and killing the fallback makes it strictly worse.
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            available_gb = float(get_memory_pressure_snapshot().available_gb)
            cortex_reserve_gb = self._primary_load_required_gb()
            fallback_reserve_gb = self._env_float("AURA_FALLBACK_RESIDENT_RESERVE_GB", 8.0)
            if available_gb >= cortex_reserve_gb + fallback_reserve_gb:
                logger.info(
                    "🛡️ InferenceGate: keeping fallback workers resident "
                    "(%.1fGB free ≥ %.1fGB cortex + %.1fGB fallback); shed skipped.",
                    available_gb,
                    cortex_reserve_gb,
                    fallback_reserve_gb,
                )
                self._last_background_memory_shed_at = now
                return
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # The comment above promises to shed ONLY when free memory
            # genuinely cannot hold the models. An unreadable memory probe is
            # not that measurement, and continuing into the unload path made
            # the promise conditional on a debug log nobody reads. No verified
            # pressure, no shed.
            _record_inference_degradation(
                exc,
                action="skipped memory-pressure shedding because free memory could not be measured",
            )
            self._last_background_memory_shed_at = now
            return

        self._last_background_memory_shed_at = now

        client_registry = {}
        try:
            # Atomic membership view — see clients_snapshot().
            from core.brain.llm.mlx_client import clients_snapshot

            client_registry.update(dict(clients_snapshot()))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued memory-pressure shedding with remaining available workers",
            )
            logger.debug("MLX background memory shed unavailable: %s", exc)
        if not client_registry:
            return

        # A shed that protects the foreground must not unload the lane that
        # will actually serve the foreground. On a protected turn the cortex is
        # frequently mid-load — that load is WHY free memory dipped under the
        # threshold — and the Brainstem/Reflex ladder is the only thing that can
        # answer meanwhile. The 2026-07-25 probe shed the ladder five times and
        # reloaded models 55 times across 30 turns, each reload paying full load
        # latency and contending with the cortex for the single GPU slot.
        #
        # Shedding the parachute to make the plane lighter.
        # …but not at the cost of the lane it is protecting. Live 2026-07-25:
        # the cortex died six times in one run against
        #   foreground_warmup_deferred:memory_pressure:75.6%/15.6GB
        #   (need <72.0% and >=20.0GB)
        # on a 64GB host. Preserving the ladder unconditionally kept several GB
        # resident that the 20GB cortex load needed, so the lane the shed
        # exists to protect could never come back. Keeping the parachute is
        # right while the plane is flying; it is not right when the parachute
        # is what is keeping the engine off.
        #
        # So: preserve the ladder on a protected shed UNLESS memory is short
        # enough that the cortex cannot load at all. Then the ladder is the
        # thing to spend — one turn served by a lower lane costs a turn; a
        # cortex that can never load costs the session.
        preserve_ladder = protected_foreground
        primary_load_blocked = preserve_ladder and self._memory_blocks_primary_load()
        if primary_load_blocked:
            preserve_ladder = False
        configured_ladder_paths = self._fallback_ladder_paths()
        ladder_paths = configured_ladder_paths if preserve_ladder else frozenset()

        eligible: list[tuple[str, Any]] = []
        for client_path, client in list(client_registry.items()):
            if client is None or client is self._mlx_client:
                continue
            if client_path in ladder_paths:
                logger.info(
                    "🛡️ InferenceGate: keeping %s resident — it is this "
                    "protected turn's fallback lane.",
                    os.path.basename(client_path),
                )
                continue
            try:
                if (
                    not hasattr(client, "is_alive")
                    or not client.is_alive()
                    or not hasattr(client, "reboot_worker")
                ):
                    continue
                eligible.append((client_path, client))
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued memory-pressure shedding with remaining available workers",
                )
                logger.debug(
                    "Background worker eligibility probe failed for %s: %s",
                    client_path,
                    exc,
                )

        if not eligible:
            return

        shed_count = 0
        shed_ladder_count = 0
        unverified_sheds: list[str] = []
        for client_path, client in eligible:
            try:
                await client.reboot_worker(
                    reason=reason,
                    mark_failed=False,
                )
                # A shed is only a shed if the worker actually went away.
                # The count used to increment on the CALL, so a reboot that
                # returned without unloading anything — or that brought the
                # worker straight back — was reported as memory reclaimed, and
                # the caller went on believing it had headroom it did not have.
                if not self._worker_is_unloaded(client):
                    unverified_sheds.append(client_path)
                    logger.warning(
                        "🧹 InferenceGate: %s did not report unloaded after reboot; "
                        "not counting it as reclaimed memory.",
                        os.path.basename(client_path),
                    )
                    continue
                shed_count += 1
                if client_path in configured_ladder_paths:
                    shed_ladder_count += 1
                logger.warning(
                    "🧹 InferenceGate: unloaded %s to protect the foreground lane (%s).",
                    os.path.basename(client_path),
                    reason,
                )
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued memory-pressure shedding with remaining available workers",
                )
                logger.debug("Background worker shed failed for %s: %s", client_path, exc)

        if primary_load_blocked and shed_ladder_count:
            # This warning describes a completed action, not merely a low-memory
            # guess or an unload attempt that may have failed.
            logger.warning(
                "🪂 InferenceGate: memory was too short for the cold primary lane "
                "to load; shed %d live fallback worker(s) so the cortex can come back.",
                shed_ladder_count,
            )

        if unverified_sheds:
            _record_inference_degradation(
                RuntimeError(
                    "workers still alive after reboot: " + ", ".join(sorted(unverified_sheds))
                ),
                action="reported less reclaimed memory than shed attempts because "
                "some workers did not stop",
                extra={"unverified": sorted(unverified_sheds)},
            )
        self._last_shed_receipt = {
            "reason": str(reason),
            "attempted": len(eligible),
            "shed": shed_count,
            "unverified": sorted(unverified_sheds),
            "ladder_shed": shed_ladder_count,
            "at": time.time(),
        }
        if shed_count:
            logger.info(
                "✅ InferenceGate: shed %d background local worker(s) (%s).",
                shed_count,
                reason,
            )

    @staticmethod
    def _worker_is_unloaded(client: Any) -> bool:
        """Whether the worker actually stopped, rather than being asked to.

        A client that cannot report its own liveness cannot prove it stopped,
        so it does not count — the absence of the check is not the check.
        """
        is_alive = getattr(client, "is_alive", None)
        if not callable(is_alive):
            return False
        try:
            return not bool(is_alive())
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    def last_shed_receipt(self) -> dict[str, Any]:
        """What the last memory-pressure shed attempted and what it proved."""
        return copy.deepcopy(getattr(self, "_last_shed_receipt", {}))

    def _primary_lane_ready(self) -> bool:
        """Whether the primary worker is initialized and available now."""
        client = getattr(self, "_mlx_client", None)
        if client is None or not hasattr(client, "is_alive"):
            return False
        try:
            return bool(client.is_alive())
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    def _memory_blocks_primary_load(self) -> bool:
        """Whether free memory is below what the primary lane needs to load.

        Fails SAFE: an unreadable memory probe returns False, i.e. keep the
        ladder. Shedding the fallback on a guess is the failure this whole
        branch exists to prevent.
        """
        if self._primary_lane_ready():
            return False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            available_gb = float(get_memory_pressure_snapshot().available_gb)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Primary-load memory probe unavailable: %s", exc)
            return False
        cortex_reserve_gb = self._primary_load_required_gb()
        return available_gb < cortex_reserve_gb

    def _primary_load_required_gb(self) -> float:
        """Return the admission threshold for the configured primary model."""
        client = getattr(self, "_mlx_client", None)
        model_path = str(getattr(client, "model_path", "") or "")
        if model_path:
            try:
                from core.brain.llm.mlx_client import _model_load_min_available_gb

                return float(_model_load_min_available_gb(model_path))
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                logger.debug("Primary model-derived load threshold unavailable: %s", exc)
                return self._env_float(
                    "AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", 24.0
                )
        return self._env_float("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", 24.0)

    def _fallback_ladder_paths(self) -> frozenset[str]:
        """Model paths that form the escalation ladder below the cortex.

        These answer a foreground turn when the primary lane is warming, so a
        foreground-protection shed must leave them alone.
        """
        paths: set[str] = set()
        for attr in (
            "_brainstem_client",
            "_reflex_client",
            "_fallback_client",
            "_secondary_client",
            "_tertiary_client",
        ):
            client = getattr(self, attr, None)
            path = getattr(client, "model_path", None) or getattr(client, "_model_path", None)
            if path:
                paths.add(str(path))
        for env_name in (
            "AURA_MLX_BRAINSTEM_MODEL",
            "AURA_MLX_REFLEX_MODEL",
            "AURA_MLX_FALLBACK_MODEL",
        ):
            value = os.environ.get(env_name)
            if value:
                paths.add(str(value))
        return frozenset(paths)

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active

            return bool(_foreground_owner_active())
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Same reasoning as the orchestrator probe above: an unreadable
            # ownership state must not read as "the lane is free".
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="assumed foreground MLX ownership after an unreadable probe",
            )
            return True

    @classmethod
    def _default_timeout_for_request(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> float:
        """Adaptive timeout based on tier and recent cortex health.

        [STABILITY v50] Raised ceiling from 90→150s for M5 64GB hardware.
        The previous 90s cap was too aggressive — after warmup checks,
        trust gate PBKDF2, and 20+ consciousness subsystem context assembly,
        the 32B model often had only 40-55s of actual generation budget.
        On M5 hardware there is no gateway proxy, so 504 risk is zero.
        """
        if is_background or requested_tier == "tertiary":
            return 60.0
        if deep_handoff or requested_tier == "secondary":
            return 210.0 if cls._origin_is_user_facing(origin) else 180.0

        if cls._origin_is_user_facing(origin):
            return 180.0

        # Adaptive: check if cortex is warm and responsive.
        base = 150.0
        try:
            inst = cls._instance_ref() if hasattr(cls, "_instance_ref") else None
            if inst is not None:
                lane = inst.get_conversation_status()
                if lane.get("conversation_ready"):
                    # Cortex is warm — tighter timeout
                    time_since_success = float(
                        lane.get("time_since_last_success_s", 999.0) or 999.0
                    )
                    if time_since_success < 30.0:
                        base = 90.0  # Recently successful — expect fast response
                    elif time_since_success < 120.0:
                        base = 120.0  # Warm but not sizzling
                # Cold/recovering cortex keeps full 150s ceiling to allow
                # inline recovery without premature fallback.
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Adaptive timeout lane probe unavailable: %s", exc)

        return base

    @staticmethod
    def _should_use_rich_context(
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> bool:
        if is_background:
            return False
        # [RESTORED] Always use rich context for user-facing origins to preserve
        # identity, memory, and persona depth.
        return True

    @classmethod
    def _has_short_live_output_contract(cls, context: dict[str, Any] | None) -> bool:
        """Return whether a live turn has a tightly bounded visible-output contract."""

        context = context or {}
        contract = context.get("requested_output_contract")
        if not isinstance(contract, dict) or not bool(contract.get("explicit_brevity")):
            return False
        try:
            hard_ceiling = int(contract.get("hard_token_ceiling") or 0)
        except (TypeError, ValueError):
            return False
        return bool(
            0 < hard_ceiling <= 192
            and (
                context.get("desktop_cognitive_engine_required")
                or context.get("live_mind_context_required")
                or context.get("live_runtime_payload_required")
            )
        )

    @classmethod
    def _should_use_compact_foreground_context(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
        prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        if is_background:
            return False
        context = context or {}
        # A literal short-output request is itself the user's compute contract.
        # It must outrank opportunistic deep-probe expansion or the model spends
        # most of the turn evaluating context it is forbidden to express.
        if cls._has_short_live_output_contract(context):
            return True
        if bool(context.get("deep_mind_probe", False)):
            return False
        if bool(context.get("desktop_quick_reply_contract", False)):
            return True
        # User-facing live turns need the identity-rich foreground prompt, but
        # not an unbounded replay of the entire assembled context stack. The
        # compact foreground builders preserve Aura's voice and continuity
        # anchors while keeping the conversational lane inside a sane latency
        # envelope. Headless harnesses already exercise this path; live chat
        # should not silently opt out of it.
        return cls._origin_is_user_facing(origin)

    @classmethod
    def _default_max_tokens_for_request(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> int:
        if is_background or requested_tier == "tertiary":
            return 384
        if deep_handoff or requested_tier == "secondary":
            return 2048
        if cls._origin_is_user_facing(origin):
            # Live conversation is allowed a full first reply. Short caps made
            # opening messages look clipped before Aura could finish a thought.
            return 4096
        return 512

    @classmethod
    def _get_system_phi(cls) -> float | None:
        """Retrieve the active system-level integration (Phi) from the mind.

        Returns ``None`` when no probe produced a real measurement — a
        missing Phi is UNAVAILABLE evidence, never a neutral score, and
        consumers must not scale compute from a fabricated value.
        """
        try:
            from core.container import ServiceContainer
            loop = ServiceContainer.get("closed_causal_loop", default=None)
            if loop is not None and getattr(loop, "_loop_state", None) is not None:
                phi_est = getattr(loop._loop_state, "phi_estimate", 0.0)
                if phi_est > 0.0:
                    return float(phi_est)
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued phi lookup after closed causal loop probe failed",
                severity="debug",
            )
            logger.debug("Failed to retrieve phi from closed causal loop: %s", e)

        try:
            from core.consciousness.phi_compute import get_phi_computer
            pc = get_phi_computer()
            if pc is not None:
                phi_latest = pc.latest_phi
                if phi_latest > 0.0:
                    return float(phi_latest)
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued phi lookup after phi computer probe failed",
                severity="debug",
            )
            logger.debug("Failed to retrieve phi from phi computer: %s", e)

        try:
            from core.container import ServiceContainer
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and getattr(phi_core, "_last_result", None) is not None:
                res = phi_core._last_result
                if res is not None:
                    return float(res.phi_s)
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="returned neutral phi after phi core probe failed",
                severity="debug",
            )
            logger.debug("Failed to retrieve phi from phi core: %s", e)

        return None  # No probe produced a measurement — Phi is unavailable

    @classmethod
    def _adaptive_max_tokens_for_prompt(
        cls,
        prompt: str,
        *,
        base_tokens: int,
        origin: str | None,
        requested_tier: str,
        is_background: bool,
    ) -> int:
        if (
            is_background
            or requested_tier in {"secondary", "tertiary"}
            or not cls._origin_is_user_facing(origin)
        ):
            return int(base_tokens)

        floor, cap, _loops = cls._foreground_compute_profile(prompt)
        adapted = int(base_tokens)

        # Scale the token budget based on system coherence/integration level
        # (Phi) — only when a real measurement exists. An unavailable Phi must
        # not silently become a scaling factor.
        phi = cls._get_system_phi()
        if phi is not None and math.isfinite(phi):
            phi_scale = max(0.5, min(1.6, 0.6 + phi * 2.0))
            adapted = int(adapted * phi_scale)
        return max(floor, min(cap, adapted))

    # Absolute ceiling for any configured token bound. An operator typo or a
    # compromised environment must not be able to request an unbounded
    # generation/memory budget through a token knob.
    _TOKEN_BOUND_HARD_CEILING = 32768

    @classmethod
    def _configured_token_bound(cls, name: str, default: int, *, minimum: int = 128) -> int:
        try:
            configured = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            configured = default
        return min(cls._TOKEN_BOUND_HARD_CEILING, max(minimum, configured))

    @staticmethod
    def _safe_sampling_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if math.isfinite(parsed) else default

    @classmethod
    def _apply_runtime_sampling_biases(
        cls,
        *,
        base_temperature: float | None,
        max_tokens: int,
        context: dict[str, Any],
        state: Any,
        allow_token_scaling: bool,
    ) -> tuple[float | None, int, dict[str, float]]:
        """Apply bounded cognitive sampling bias from runtime state.

        Biases are advisory state outputs, not caller authority — and that
        sentence used to be a comment sitting directly above three reads
        straight out of the caller's ``context`` dict, with no way to tell one
        of Aura's own frames from a value an API client typed. Every accepted
        bias now names where it came from, and the accepted sources are the
        cognitive engine's own frames.

        ``state.response_modifiers`` is Aura's state; the cognitive engine
        writes it. ``context`` is whatever the caller passed, so a bias found
        there is only honoured when it also appears in state — that is,
        when the engine put it in both.
        """

        modifiers = getattr(state, "response_modifiers", None)
        state_modifiers = modifiers if isinstance(modifiers, dict) else {}
        biases: list[Any] = []
        bias_sources: list[str] = []
        for key in _SAMPLING_BIAS_KEYS:
            own = state_modifiers.get(key)
            if isinstance(own, dict):
                biases.append(own)
                bias_sources.append(f"state:{key}")
                continue
            supplied = context.get(key)
            if not isinstance(supplied, dict):
                continue
            # Present in context and NOT in state: nothing in the runtime
            # produced it this turn. Advisory means advisory from her own
            # cognition, so it is recorded and dropped rather than applied.
            logger.debug(
                "Ignoring caller-supplied %s: no matching frame in runtime state.", key
            )
            context.setdefault("rejected_sampling_bias", []).append(key)

        temperature = base_temperature
        token_factor = 1.0
        applied_temperature_delta = 0.0
        applied_token_factor = 1.0
        # The same advisory can arrive via caller context AND state modifiers;
        # dedupe by value so one advisory never has a squared or repeated
        # effect, and clamp the CUMULATIVE deltas so even distinct advisories
        # stay bounded in aggregate.
        seen_bias_values: set[tuple[tuple[str, str], ...]] = set()
        for bias in biases:
            if not isinstance(bias, dict):
                continue
            value_key = tuple(
                sorted((str(key), repr(value)) for key, value in bias.items())
            )
            if value_key in seen_bias_values:
                continue
            seen_bias_values.add(value_key)
            temp_delta = max(
                -0.18,
                min(0.18, cls._safe_sampling_float(bias.get("temperature_delta"), 0.0)),
            )
            if temp_delta:
                remaining = 0.30 - abs(applied_temperature_delta)
                if remaining <= 0.0:
                    temp_delta = 0.0
                else:
                    temp_delta = max(-remaining, min(remaining, temp_delta))
            if temp_delta:
                base = 0.72 if temperature is None else temperature
                temperature = max(0.1, min(1.5, base + temp_delta))
                applied_temperature_delta += temp_delta

            factor = cls._safe_sampling_float(bias.get("max_tokens_factor"), 1.0)
            if allow_token_scaling and 0.40 <= factor <= 1.20:
                token_factor = max(0.40, min(1.20, token_factor * factor))
                applied_token_factor = token_factor

        if allow_token_scaling and token_factor != 1.0:
            max_tokens = max(128, min(4096, int(max_tokens * token_factor)))

        return (
            temperature,
            max_tokens,
            {
                "temperature_delta": round(applied_temperature_delta, 4),
                "max_tokens_factor": round(applied_token_factor, 4),
                # Provenance: which frames actually moved sampling this turn.
                "sources": bias_sources,
            },
        )

    @classmethod
    def _foreground_compute_profile(cls, prompt: str) -> tuple[int, int, int]:
        """Return the token floor, token cap, and recurrent loops for a live turn.

        Foreground inference used to force every primary-lane turn to a
        long-form 3,072-token budget. On local 32B inference that made a short
        conversational request cost as much as a multi-part analysis and could
        overrun the API deadline despite producing a valid answer seconds later.
        One prompt-shape policy now controls both output budget and recurrent
        depth so latency and reasoning depth scale together.
        """

        text = str(prompt or "").strip()
        shape = analyze_prompt_shape(text)
        word_count = len(text.split())
        long_form_requested = bool(_LONG_FORM_REQUEST_RE.search(text))
        action_hits = len(_FOREGROUND_ACTION_VERB_RE.findall(text))
        action_chain_requested = bool(
            action_hits >= 2
            and _FOREGROUND_ACTION_SURFACE_RE.search(text)
            and _FOREGROUND_ACTION_SEQUENCE_RE.search(text)
        )
        extended = bool(
            long_form_requested
            or action_chain_requested
            or shape.prefers_extended_answer
            or shape.requires_single_reply_coverage
            or shape.question_parts >= 2
        )

        if extended:
            floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_MIN_TOKENS",
                3072,
                minimum=512,
            )
            cap = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_MAX_TOKENS",
                3072,
                minimum=floor,
            )
            loops = 2
        elif word_count > 45 or len(text) > 320:
            floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_STANDARD_MIN_TOKENS",
                768,
                minimum=384,
            )
            cap = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_STANDARD_MAX_TOKENS",
                1280,
                minimum=floor,
            )
            loops = 1
        else:
            floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_SIMPLE_MIN_TOKENS",
                384,
                minimum=256,
            )
            cap = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_SIMPLE_MAX_TOKENS",
                512,
                minimum=floor,
            )
            loops = 1

        # Preserve the legacy operator override as a universal floor when it is
        # explicitly configured. The default no longer forces simple turns into
        # the long-form profile.
        if not extended and "AURA_FOREGROUND_CHAT_MIN_TOKENS" in os.environ:
            operator_floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_MIN_TOKENS",
                floor,
                minimum=384,
            )
            floor = max(floor, operator_floor)
            cap = max(cap, floor)

        return floor, max(floor, cap), loops

    @classmethod
    def _turn_is_determinate_task(cls, prompt: Any) -> bool:
        """Whether the turn asks for a quantity this runtime must simply get right."""
        try:
            from core.conversation.response_reliability import asks_for_a_number
        except ImportError:
            return False
        try:
            return bool(asks_for_a_number(prompt))
        except (RuntimeError, TypeError, ValueError):
            return False

    @classmethod
    def _foreground_prompt_profile(cls, prompt: str, context: dict[str, Any] | None = None) -> str:
        """Classify a live foreground turn for context and output budgeting."""

        context = context or {}
        if bool(context.get("deep_mind_probe", False)):
            return "deep_probe"
        # Current-condition questions already carry a fresh canonical state
        # projection. Preserve that semantic distinction at the model boundary
        # instead of replacing it with the generic full-desktop profile.
        if bool(
            context.get(
                "self_condition_contract_covers_turn",
                context.get("self_condition_contract", False),
            )
        ):
            return "state_report"
        # A question with one right answer is answered from a lean prompt.
        # Measured live 2026-07-26 on the desktop surface: a 78-character
        # arithmetic question was sent as
        #   chars=7542 (scaffold=5200 request=2342) origin=desktop_quick_user
        # — the person's actual words about one percent of what the model read,
        # under the 'standard' budget of ~15.6k that never trims any of it. The
        # likeliest continuation of a prompt that is almost entirely
        # self-description is more self-description, which is what came back:
        # off-topic prose, and replies the gate rejected as runtime_boilerplate.
        # The bare model answers this instantly; the scaffold is what buries it.
        if cls._turn_is_determinate_task(prompt):
            return "contract"
        if bool(context.get("desktop_quick_reply_contract", False)) and bool(
            context.get("memory_state_contract", False)
        ):
            return "simple"
        if bool(context.get("desktop_quick_reply_contract", False)) and bool(
            context.get("live_runtime_payload_required", False)
            or context.get("live_mind_context_required", False)
            or context.get("desktop_cognitive_engine_required", False)
        ):
            return "standard"
        if bool(context.get("desktop_quick_reply_contract", False)):
            return "simple"
        if bool(context.get("live_runtime_payload_required", False)) and (
            is_live_self_reflection_turn(prompt)
            or is_self_process_question(prompt)
        ):
            return "standard"
        if bool(context.get("capability_inventory_contract", False)):
            return "standard"
        if bool(
            context.get("desktop_execution_contract", False)
            or context.get("coding_request", False)
            or context.get("requires_search", False)
            or context.get("requires_memory_grounding", False)
        ):
            return "extended"

        text = str(prompt or "").strip()
        shape = analyze_prompt_shape(text)
        long_form_requested = bool(_LONG_FORM_REQUEST_RE.search(text))
        action_hits = len(_FOREGROUND_ACTION_VERB_RE.findall(text))
        action_chain_requested = bool(
            action_hits >= 2
            and _FOREGROUND_ACTION_SURFACE_RE.search(text)
            and _FOREGROUND_ACTION_SEQUENCE_RE.search(text)
        )
        if (
            long_form_requested
            or action_chain_requested
            or shape.prefers_extended_answer
            or shape.requires_single_reply_coverage
            or shape.question_parts >= 2
        ):
            return "extended"
        if len(text.split()) > 45 or len(text) > 320:
            return "standard"
        return "simple"

    @classmethod
    def _cortex_serving_lane(
        cls,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Map the typed turn contract to one qualified serving lane."""

        context = context or {}
        allowed = {
            "foreground_simple",
            "foreground_standard",
            "foreground_extended",
            "deep_reasoning",
            "tool_execution",
            "code",
            "document",
        }
        explicit = str(context.get("serving_lane") or "").strip().lower()
        if explicit in allowed:
            return explicit
        if bool(context.get("desktop_execution_contract", False)):
            return "tool_execution"
        if bool(context.get("coding_request", False)):
            return "code"
        if bool(context.get("document_request", False)):
            return "document"
        if str(context.get("reasoning_mode") or "").strip().lower() == "deep":
            return "deep_reasoning"
        if bool(context.get("deep_mind_probe", False)):
            return "deep_reasoning"
        profile = cls._foreground_prompt_profile(prompt, context)
        if profile == "extended":
            return "foreground_extended"
        if profile == "simple":
            return "foreground_simple"
        return "foreground_standard"

    @classmethod
    def _foreground_prebuilt_history_limit(
        cls,
        prompt: str,
        context: dict[str, Any] | None = None,
        *,
        deep_probe: bool = False,
    ) -> int:
        if deep_probe:
            return 2
        profile = cls._foreground_prompt_profile(prompt, context)
        if bool((context or {}).get("live_runtime_payload_required", False)) and (
            is_live_self_reflection_turn(prompt)
            or is_self_process_question(prompt)
        ):
            return 2
        if profile == "state_report":
            return 2
        if profile == "simple":
            return 4
        if profile == "standard":
            return 6
        return 10

    @staticmethod
    def _split_attempt_timeouts(total_timeout: float, requested_tier: str) -> tuple[float, float]:
        """[STABILITY v50] Give the primary Cortex 80% of the budget.

        The previous 65/35 split starved the 32B model and gave 35% of
        the user's patience to the brainstem fallback — which rarely
        produces a satisfying answer anyway. 80/20 gives Cortex full
        room to generate while preserving a meaningful brainstem window.
        """
        total_timeout = max(10.0, float(total_timeout))
        if requested_tier == "secondary":
            # Explicit solver turns are rare and intentional. Give the 72B
            # lane most of the foreground budget so load + first-token latency
            # do not force a fallback before deep reasoning can complete.
            if total_timeout >= 300.0:
                primary_budget = min(total_timeout - 20.0, max(240.0, total_timeout * 0.92))
            else:
                primary_budget = min(210.0, max(150.0, total_timeout * 0.90))
        elif requested_tier == "tertiary":
            primary_budget = min(60.0, total_timeout * 0.7)
        else:
            # Give cortex 80% of the total budget so the 32B model has
            # real headroom. On an API-protected 300s turn, preserve the heavy
            # lane instead of silently dropping it after the old 120s cap.
            if total_timeout >= 240.0:
                primary_budget = min(total_timeout - 20.0, max(210.0, total_timeout * 0.90))
            else:
                primary_budget = min(150.0, max(60.0, total_timeout * 0.85))

        fallback_budget = max(15.0, total_timeout - primary_budget)
        return primary_budget, fallback_budget

    @staticmethod
    def _strict_contract_grounding_turns(
        provided_messages: list[Any] | None,
    ) -> list[dict[str, Any]]:
        """Non-system turns a strict rewrite must not throw away.

        CP126: "Strict contracts discard the original system messages and
        grounding history ... Safety policy, tool receipts, evidence, prior
        assistant context, and caller-required system content."

        The system half was fixed by collecting every system message. The
        other half was not: the payload was rebuilt as exactly
        ``[system, user]``, so every assistant turn, tool result and earlier
        user turn vanished. On a strict route that is precisely the evidence
        the answer depends on — a tool receipt in the previous assistant
        turn is the difference between an answer and a guess, and the guess
        still arrives in a well-formed <answer> envelope.

        Everything before the final user turn is preserved AS MESSAGES
        rather than flattened into the prompt, so the existing context-window
        trimming still applies and no new budget constant is invented here.
        """
        if not provided_messages:
            return []
        last_user_index = -1
        for index in range(len(provided_messages) - 1, -1, -1):
            msg = provided_messages[index]
            if isinstance(msg, dict) and str(
                msg.get("role", "") or ""
            ).strip().lower() == "user":
                last_user_index = index
                break
        preserved: list[dict[str, Any]] = []
        for index, msg in enumerate(provided_messages):
            if index == last_user_index or not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip().lower()
            if role in {"system", ""}:
                continue
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            preserved.append({"role": role, "content": content})
        return preserved

    @staticmethod
    def _strict_contract_procedure_hints(prompt: Any) -> str:
        """Low-level strict contracts do not inject task-shape hints.

        Exact symbolic proof work belongs to the governed System2 proof
        reasoner. Keeping this gateway hint-free avoids making model-only
        diagnostics depend on fragile prompt nudges.
        """

        return ""

    @staticmethod
    def _foreground_retry_schedule(
        primary_attempt_elapsed: float,
        primary_timeout: float,
    ) -> tuple[float, ...]:
        """Return bounded retry delays for a failed foreground Cortex call."""

        retry_cutoff = min(45.0, max(0.0, float(primary_timeout)) * 0.4)
        if max(0.0, float(primary_attempt_elapsed)) <= retry_cutoff:
            return (2.0,)
        return ()

    def _publish_exhausted_primary_owner(
        self,
        *,
        primary_attempt_elapsed: float,
        same_lane_retry_count: int,
    ) -> None:
        """Publish that a no-text primary call still consumed the turn owner."""

        failure_class = (
            "primary_no_text_after_bounded_retry"
            if same_lane_retry_count
            else "primary_no_text_after_long_attempt"
        )
        self._annotate_last_generation_metadata(
            model_retry_suppressed=True,
            generation_failure_class=failure_class,
            primary_attempt_elapsed_s=round(float(primary_attempt_elapsed), 6),
            same_lane_retry_count=max(0, int(same_lane_retry_count)),
        )

    @asynccontextmanager
    async def _resource_context(
        self,
        enabled: bool,
        priority: bool,
        worker: str | None = None,
        timeout_s: float | None = None,
    ):
        # Every live generation, including foreground retries, uses canonical
        # admission. The historical priority bypass predated the single caller
        # path and allowed exactly the concurrent same-lane work this policy is
        # meant to prevent. Model loading can safely nest beneath its own lane's
        # inference reservation, so cold-start recovery no longer needs a bypass.
        if not enabled:
            yield
            return
        try:
            from core.resilience.resource_arbitrator import get_resource_arbitrator
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="serialized inference on process-local fallback lock after arbitrator import failed",
                severity="error",
            )
            logger.warning(
                "Resource arbitration unavailable — serializing inference on the "
                "process-local fallback lock instead of running unlocked: %s",
                exc,
            )
            # Fail CLOSED, not open: without the canonical arbitrator, same-lane
            # concurrent generation and model loading must still be excluded.
            # A single process-local lock is coarser than lane arbitration but
            # preserves the mutual-exclusion invariant the arbitrator provides.
            fallback_lock = getattr(self, "_fallback_arbitration_lock", None)
            if fallback_lock is None:
                fallback_lock = asyncio.Lock()
                self._fallback_arbitration_lock = fallback_lock
            try:
                await asyncio.wait_for(
                    fallback_lock.acquire(),
                    timeout=max(0.25, float(timeout_s or 30.0)),
                )
            except TimeoutError:
                raise TimeoutError("resource_arbitration_fallback_lock_timeout") from exc
            try:
                yield
            finally:
                fallback_lock.release()
            return

        async with get_resource_arbitrator().inference_context(
            priority=priority,
            worker=worker,
            timeout=max(0.25, float(timeout_s or 30.0)),
        ):
            yield

    async def _restore_primary_after_deep_handoff(self) -> None:
        """Return the system to the 32B conversational brain after a 72B request."""
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

            primary_client = get_mlx_client(model_path=str(get_runtime_model_path(ACTIVE_MODEL)))
            warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
            if warmup_deferral:
                self._log_cortex_warmup_deferral(warmup_deferral, context="post-deep-restore")
                return
            # Give the conversational 32B lane enough time to swap back after
            # a 72B deep handoff; otherwise the next ordinary turn inherits a
            # preventable "cortex warming" failure.
            warmup_result = await asyncio.wait_for(
                primary_client.warmup(
                    foreground_request=True,
                    skip_swap_cooldown=True,
                ),
                timeout=300.0,
            )
            lane = (
                primary_client.get_lane_status()
                if hasattr(primary_client, "get_lane_status")
                else {}
            )
            if (
                warmup_result is False
                or not primary_client.is_alive()
                or not lane.get("conversation_ready", False)
            ):
                raise RuntimeError(
                    str(lane.get("last_error") or "primary_restore_not_conversation_ready")
                )
            logger.info("♻️ Restored %s after deep handoff.", PRIMARY_ENDPOINT)
        except TimeoutError:
            logger.error(
                "⚠️ Failed to restore %s after deep handoff: warmup timed out (300s)",
                PRIMARY_ENDPOINT,
            )
            # Schedule deferred recovery so next request doesn't hit dead cortex
            self._schedule_background_cortex_prewarm(delay=5.0)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="left primary restore on normal foreground-demand recovery path",
            )
            logger.error("⚠️ Failed to restore %s after deep handoff: %s", PRIMARY_ENDPOINT, exc)
            self._schedule_background_cortex_prewarm(delay=5.0)

    def _schedule_primary_restore_after_deep_handoff(self) -> None:
        restore_coro = self._restore_primary_after_deep_handoff()
        try:
            task = get_task_tracker().create_task(
                restore_coro,
                name="restore_primary_after_deep",
            )
        except RuntimeError:
            restore_coro.close()
            logger.debug("Primary restore skipped: no running event loop.")
            return
        if not isinstance(task, asyncio.Task):
            restore_coro.close()
            logger.debug(
                "Primary restore scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        task.add_done_callback(self._log_task_exception)

    # ── Silence Protocol ──────────────────────────────────────────────────────
    SILENCE_TOKEN = "<|SILENCE|>"
    SILENCE_SENTINEL = "\x00AURA_SILENCE\x00"

    @staticmethod
    def _strip_silence(text: str) -> str | None:
        """
        If the model chose silence, return the sentinel string so the caller
        can suppress output cleanly. Any response that IS substantive is
        returned with the token scrubbed, never suppressed.

        The prompt contract requires the model to output EXACTLY the silence
        token to decline. A substantive response that merely CONTAINS the
        token (quoted instructions, echoed adversarial user content, analysis
        of the protocol itself) must not be suppressible by substring match.
        """
        token = InferenceGate.SILENCE_TOKEN
        stripped = str(text or "").strip()
        if stripped == token or (
            stripped.startswith(token) and len(stripped) - len(token) <= 8
        ):
            # Model chose not to speak — respect it
            logger.info("🤫 Silence Protocol: model chose not to respond.")
            return InferenceGate.SILENCE_SENTINEL
        if token in text:
            logger.info(
                "🤫 Silence token appeared inside a substantive response; "
                "scrubbing the token instead of suppressing the reply."
            )
            return text.replace(token, "").strip()
        return text

    async def _tool_grounded_answer(
        self,
        client: Any,
        *,
        visible: Any,
        system_prompt: Any,
        timeout_s: float,
        evidence: Any = None,
        completed_capability_evidence: Any = None,
        allow_tools: bool = True,
        decode_budget: int = 0,
        origin: str = "",
    ) -> str | None:
        """Answer by running the capability the request needs, or return None.

        LIVE DEFECT, 2026-08-19. Asked to run Python and report the number,
        with code_repl READY, she wrote a snippet and stated an invented
        "Output:". The runtime HAS a tool loop — parse a call, bind it to the
        tool's advertised schema, execute, feed the result back — and reaching
        it goes through `should_force_tool_handoff` in the health router. Chat
        never gets there: this lane calls the MLX client directly
        (`local_client = self._mlx_client`), so the router's contract, its
        handoff, and the loop behind it apply to every OTHER caller and not to
        the one people actually type into.

        Returns text only when a tool really ran. Anything else — no
        capability needed, no tool map, the model declining to call — returns
        None so the ordinary generation proceeds untouched.
        """
        # A typed completion/continuation turn is an answer segment, not a new
        # execution request. Enforce that before capability inference so words
        # such as "code" or "build" inside a correction cannot open a tool
        # lane the caller explicitly closed.
        if not allow_tools:
            return None

        # The person's own words, not the assembled prompt. The scaffold runs
        # to thousands of characters around a request of a hundred, and asking
        # a question about the whole envelope answers about the envelope.
        text = str(visible or "").strip()
        if not text:
            return None
        try:
            from core.brain.llm.runtime_wiring import build_agentic_tool_map
            from core.phases.response_contract import (
                derive_capability_set,
                requested_effect_ceiling,
            )

            # The same ceiling selection used. Offering a capability the
            # dispatch then refuses is worse than not offering it: the turn
            # spends itself reaching for something it was never allowed to
            # use, which is how "build me a web app" ended in an executive
            # veto on code_repl.
            ceiling, allowed_scopes = requested_effect_ceiling(text)

            required = derive_capability_set(text)
            if not required:
                return None
            pending = remaining_capabilities(required, completed_capability_evidence)
            if required and not pending:
                logger.info(
                    "🔧 Tool handoff skipped: every required capability already "
                    "has runtime-stamped evidence (%s).",
                    ",".join(sorted(required)),
                )
                return None
            required = pending
            tools = build_agentic_tool_map(
                required, objective=text, max_tools=len(required)
            )
            if not tools:
                logger.info(
                    "🔧 Tool handoff: skill=%s offered=NONE (no tool definition)",
                    ",".join(required),
                )
                return None
            # The ceiling above was computed and then discarded, so the rule
            # the comment states was never enforced.
            #
            # LIVE, 2026-08-25: asked to diagnose a project, the turn was
            # offered diagnose_repo, code_repl and file_operation. It reached
            # for file_operation, which is state_mutation and above this
            # turn's sandboxed_compute ceiling, and the executive vetoed it;
            # then for code_repl, which the permission model refused for want
            # of a confirmation nobody could give. Two of the turn's two tool
            # calls were spent on tools that could never have run, and the
            # one that would have answered was never called.
            tools, withheld = _tools_within_reach(tools, allowed_scopes)
            if withheld:
                logger.info(
                    "🔧 Tool handoff: withheld %s — above the %s ceiling for this turn.",
                    ",".join(sorted(withheld)),
                    ceiling,
                )
            if not tools:
                logger.info(
                    "🔧 Tool handoff: skill=%s offered=NONE (all above the %s ceiling)",
                    ",".join(required),
                    ceiling,
                )
                return None
            logger.info(
                "🔧 Tool handoff: wanted=%s offered=%s",
                ",".join(required),
                ",".join(sorted(tools)),
            )
            # The receipts this loop writes belong to the turn that started it.
            # That used to need a hand-threaded lease, because custody was
            # keyed on the exact (thread, task) that opened the turn and this
            # loop does not always run there. Belonging is inherited from the
            # turn's context now, so the loop needs nothing to be part of it.
            # The tool loop is the part of a turn that takes the time, and it
            # was the one clock left holding an absolute number. Live on
            # 2026-08-28 a ledgerkit turn read three files and died here at
            # 138.9 seconds while every clock around it was holding itself
            # open, because this one still counted.
            from core.brain.llm_health_router import _await_while_it_is_working

            result = await _await_while_it_is_working(
                client.think_and_act(
                    objective=text,
                    # The budget this turn decided on, not the client's default.
                    #
                    # LIVE, 2026-08-28: "read the docs, then use it" read three
                    # files, said "Running it now:", and emitted a code_repl
                    # call whose argument was cut off mid-import. The turn's
                    # clock had allocated 1536 tokens and every generation in
                    # the loop got 399, so the narration and the opening of the
                    # program together reached the ceiling and the call was
                    # never a call — it arrived as prose, was judged prose
                    # containing prompt scaffolding, and was correctly refused.
                    #
                    # She decided to run the code. The room to say so was the
                    # thing missing, and the room had already been worked out
                    # one function up.
                    **(
                        {"max_tokens": int(decode_budget)}
                        if int(decode_budget or 0) > 0
                        else {}
                    ),
                    # An execution turn is not a conversation turn.
                    #
                    # The foreground system prompt is the full conversational
                    # scaffold — persona, instruments, present moment, running
                    # to five thousand tokens. Wrapped around a tool call it
                    # produced an immediate end-of-turn: one token, no text,
                    # every time. A call needs the objective and the tools; the
                    # voice belongs to the reply, which is generated
                    # separately.
                    #
                    # It is also most of the latency of a tool turn.
                    system_prompt="",
                    tools=tools,
                    # A step, a look at what it returned, and a chance to do
                    # something else because of it — three is one attempt with
                    # no room to be wrong. Scaled to the working set so a
                    # single-capability turn stays cheap.
                    #
                    # And one more than the calls, because the last turn has to
                    # be free to WRITE. Two tools gave five turns; a turn that
                    # was refused twice and then read three files used all five
                    # on calls, and the person got a list of what ran instead of
                    # an answer.
                    #
                    # LIVE, 2026-08-28: "read the docs, then actually use it"
                    # spent turns on a denied path, a refused execution, and
                    # three successful reads. Nothing was left to say what it
                    # had found.
                    max_turns=max(4, 2 * len(tools) + 2),
                    context={
                        "required_skills": list(required),
                        "foreground_request": True,
                        # Who asked, and what they said.
                        #
                        # The conscience holds a skill whose worst case looks
                        # harmful unless a person asked for it directly, in the
                        # foreground, on their own machine — and it decides
                        # that from the origin and the message on this context.
                        # Neither was here, so every dispatch arrived as
                        # origin=unknown and the override could not fire.
                        #
                        # LIVE, 2026-08-29: asked to use a library at a named
                        # path, the model called code_repl with that path, was
                        # held at "worst-case harm 0.80", tried sys, importlib
                        # and exec in turn — each correctly refused — and came
                        # back to the right call, which was held again. The
                        # person had asked for it in those words.
                        "origin": origin or "user",
                        "message": text,
                        # The fact, rather than a name to be parsed again. This
                        # gate already decided whether somebody is waiting on
                        # this turn; the conscience downstream needs the same
                        # answer, and deriving it twice from origin strings is
                        # how the two came to disagree.
                        "a_person_is_waiting": True,
                        # What this turn may do. The dispatch refuses any
                        # action ranked above it, so a skill can be offered
                        # for its safe actions without offering its
                        # dangerous ones.
                        "authorised_effect_scope": ceiling,
                        # Consent the request itself carries.
                        #
                        # The permission model already asks whether the person
                        # pre-approved this class of action, and nothing ever
                        # answered. So "build me a small web app, one
                        # self-contained file" was refused with "Requires user
                        # confirmation" — a confirmation prompt for the thing
                        # that had just been asked for in those words.
                        #
                        # Deliberately narrow, and it was narrower than the
                        # thing it was arguing for.
                        #
                        # Set only for the artifact ceiling, it left the
                        # SELF-SERVICE ceiling asking for a confirmation
                        # nobody can give — and that ceiling is defined, where
                        # it is declared, as "the most a turn may do without
                        # the person having asked for that effect... it can
                        # calculate anything and change nothing outside its own
                        # sandbox". Something that by definition needs no
                        # permission was being refused for want of one.
                        #
                        # LIVE, 2026-08-28: "read the docs, then actually use
                        # it" reached code_repl and came back "Permission
                        # denied: Requires user confirmation: Typed execution
                        # contract: scope=sandboxed_compute". She read the
                        # library three times over and never ran it.
                        #
                        # Still narrow: these are the two ceilings a request
                        # can establish for itself. Nothing here authorises
                        # external_io, privileged mutation, deleting, sending
                        # or spending — those need their own consent, because
                        # nobody asked for them.
                        "user_explicitly_authorized": (
                            ceiling
                            in {
                                _SELF_SERVICE_EFFECT_CEILING,
                                _REQUESTED_ARTIFACT_EFFECT_CEILING,
                            }
                        ),
                    },
                    # What the turn has already read. Without it the loop
                    # fetched the same document a second time, from a URL it
                    # rebuilt from memory, and got a 400.
                    evidence=evidence,
                ),
                # The tool loop's job is to GET the evidence; the reply's job
                # is to SAY it, and evidence nobody can say is worth nothing.
                #
                # LIVE, 2026-08-27: a repository diagnosis ran in 389ms and
                # came back complete — the contradiction, the line, the
                # project's own broken invariant. The tool-calling pass had
                # taken 65s of a 148s turn and the presenting pass was refused
                # before dispatch, "because the request budget was already
                # spent". The answer was in hand and there was no time left to
                # say it.
                budget_s=_tool_loop_budget(
                    timeout_s,
                    _answer_reserve_seconds(
                        client,
                        # What the answer will be read from: everything handed
                        # to the loop, because that is what comes back with it.
                        len(str(text or ""))
                        + sum(
                            len(str((item or {}).get("content") or ""))
                            for item in (evidence or [])
                            if isinstance(item, dict)
                        ),
                    ),
                ),
                user_facing=True,
                # The same fact the loop puts on its own context. A person
                # asked for this in the foreground and is sitting in front of
                # it, so the bound is the turn's ceiling rather than a
                # multiple of one step's budget — which cut a loop that was
                # producing 0.2s earlier, LIVE 2026-08-29.
                person_is_waiting=True,
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            record_degradation(
                "inference_gate.tool_grounded_answer",
                exc,
                severity="info",
                action="answered on the ordinary lane after the tool loop ran out of time",
                enforce_failure_policy=False,
            )
            return None
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Named, like every other handler in this file. A bare `except
            # Exception` here also swallowed KeyboardInterrupt-adjacent and
            # programming errors — a NameError in the tool loop would have been
            # recorded as "the tool loop failed" and answered around, so the
            # defect would never surface as a defect.
            record_degradation(
                "inference_gate.tool_grounded_answer",
                exc,
                severity="warning",
                action="answered on the ordinary lane after the tool loop failed",
                enforce_failure_policy=False,
            )
            return None

        if not isinstance(result, dict):
            return None
        called = result.get("tool_calls") or []
        text = str(result.get("content") or "").strip()
        if not called:
            # The model was handed the tool and answered without it. That
            # answer is ungrounded by construction, and it is exactly how
            # "Output: 7" reached the screen.
            return None
        model_path = str(getattr(client, "model_path", "") or "").strip()
        if text and model_path:
            # The receipt for the work, carried with the record of it.
            #
            # Ownership of a foreground answer is proven by a surface-control
            # receipt with a token count in it: the resident model generated
            # these words, on this turn, once. This path recorded that a tool
            # loop had run and nothing about the generation inside it, so an
            # answer the 27B plainly wrote could not be shown to have been
            # written by anything.
            #
            # LIVE 2026-08-29: six tool calls, the last one returning the
            # trial balance, an answer composed from them and trimmed, and
            # then "missing: foreground_model_generation_ownership_unproven"
            # with generations=0 consumed=False. The turn failed closed on
            # bookkeeping for work it had done.
            tool_loop_metadata = {
                "provider": "mlx_local",
                "model": model_path,
                "endpoint": os.path.basename(model_path),
                "is_local": True,
                "provider_verified": True,
                "tool_loop": True,
                "tool_calls": len(called),
            }
            receipt = None
            reader = getattr(client, "get_last_surface_control_receipt", None)
            if callable(reader):
                try:
                    receipt = reader()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    logger.debug(
                        "tool loop could not read the surface control receipt: %s", exc
                    )
            if isinstance(receipt, dict) and receipt:
                tool_loop_metadata["live_mind_surface_control_receipt"] = dict(receipt)
            self._record_client_generation_metadata(
                client,
                label=os.path.basename(model_path),
                success=True,
                text=text,
                generation_metadata=tool_loop_metadata,
            )
            # And on the turn, which is the one thing every path shares.
            #
            # The line above publishes on this gate. The layer that checks
            # authorship reads the health router. Both are right about their
            # own object, and the answer falls between them — LIVE 2026-08-29,
            # "ownership_evidence=[live_mind(tokens=-,decode=-); latent_cortex
            # (decode=0)]" on a turn whose fifth tool call had just returned
            # the trial balance.
            _tokens = 0
            for _key in ("generated_tokens", "decode_generated_tokens"):
                _value = (receipt or {}).get(_key)
                if isinstance(_value, int) and _value > 0:
                    _tokens = _value
                    break
            if _tokens <= 0:
                # What the client counted arriving, when the worker attached
                # no total. The tokens are the same tokens; only the reporting
                # differs, and an answer must not fail to prove itself because
                # of which branch of the worker replied.
                _counter = getattr(client, "tokens_generated_for_this_request", None)
                if callable(_counter):
                    try:
                        _tokens = max(0, int(_counter() or 0))
                    except (TypeError, ValueError):
                        _tokens = 0
            try:
                from core.conversation.turn_evidence_custody import (
                    record_turn_model_generation,
                )

                _recorded = (
                    record_turn_model_generation(
                        model_path, tokens=_tokens, path="tool_loop"
                    )
                    if _tokens > 0
                    else False
                )
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                _recorded = False
                logger.debug("tool loop could not record its generation: %s", exc)
            # Which of the two, when the answer cannot prove who wrote it.
            #
            # A receipt with no count and a turn that would not take the
            # record are different faults: the first is a generation whose
            # tokens nobody added up, the second is custody this execution
            # does not belong to. Both end as
            # "foreground_model_generation_ownership_unproven" and the name
            # says neither.
            if not _recorded:
                logger.info(
                    "🧾 tool loop generation unrecorded: tokens=%d receipt_keys=%s",
                    _tokens,
                    ",".join(sorted(receipt or {}))[:300] or "none",
                )
        from core.conversation.surface_disposition import record_tool_receipt

        for call in called:
            if not isinstance(call, dict):
                continue
            # What it RETURNED, not only that it ran.
            #
            # A receipt saying a tool executed and not what came back cannot
            # support any answer. LIVE, 2026-08-27: file_operation read a
            # project's docs in 6ms, the turn had nothing to say afterwards,
            # and the record of the read held the arguments and no result — so
            # the fallback that reports what the tools found had nothing to
            # report.
            observed = _what_a_tool_returned(call.get("result"))
            record_tool_receipt(
                str(call.get("tool") or call.get("name") or "tool"),
                ok=bool(call.get("ok", True)),
                action="execute",
                object_ref=str(call.get("args") or "")[:200],
                effect_observed=True,
                verification="tool loop returned a result for this turn",
                observed_content=observed[:2000],
            )
        return text or None

    async def _generate_with_client(
        self,
        client: Any,
        prompt: str,
        system_prompt: str,
        history: list[dict],
        deadline: Deadline,
        label: str,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        origin: str = "",
        is_background: bool = False,
        foreground_request: bool = False,
        **kwargs,
    ) -> str | None:
        llm_messages = messages or self._build_messages(prompt, system_prompt, history)
        local_prompt = _flatten_messages_for_local_model(llm_messages)
        gen_kwargs: dict = {
            "prompt": local_prompt,
            "messages": llm_messages,
            # The structured messages already carry the system policy. Passing
            # the scalar copy makes chat clients prepend it a second time.
            "system_prompt": "",
            "deadline": deadline,
            "max_tokens": max_tokens,
            "origin": origin,
            "is_background": is_background,
            "foreground_request": foreground_request,
            "owner_label": label,
        }
        generation_result_sink: dict[str, Any] = {}
        gen_kwargs["_generation_result_sink"] = generation_result_sink
        if temperature is not None:
            gen_kwargs["temp"] = temperature
        # Explicit parameters are the routing/identity/deadline authority at
        # this boundary. Caller kwargs may EXTEND the request but must never
        # replace a protected field — that would bypass routing, ownership,
        # deadline, or visibility contracts already decided upstream.
        _protected_overrides = set(gen_kwargs) & set(kwargs)
        if _protected_overrides:
            logger.warning(
                "🛡️ Dropping caller kwargs that would overwrite protected "
                "generation fields for %s: %s",
                label,
                sorted(_protected_overrides),
            )
        gen_kwargs.update(
            {key: value for key, value in kwargs.items() if key not in gen_kwargs}
        )
        generation_timeout_s = deadline.remaining if isinstance(deadline, Deadline) else None
        if generation_timeout_s is None:
            generation_timeout_s = 300.0 if foreground_request else 120.0
        generation_timeout_s = float(generation_timeout_s)
        if generation_timeout_s <= 0.0:
            # The budget is already gone. The clamp used to be max(0.5, …), so
            # an expired request still started a generation: it woke the lane,
            # took the lock, ran the prompt-eval side effects, and then timed
            # out half a second later with nothing for the caller, who had
            # stopped waiting. Failing before dispatch costs nothing and holds
            # nothing.
            reason = f"inference_gate_deadline_expired_before_dispatch:{label}"
            _record_inference_degradation(
                TimeoutError(reason),
                action="refused local generation because the request budget was already spent",
                severity="warning",
            )
            self._record_client_generation_metadata(
                client,
                label=label,
                success=False,
                text="",
                requested_max_tokens=max_tokens,
                generation_metadata={"error": reason},
            )
            return None
        try:
            from core.brain.llm.mlx_client import MLXLocalClient

            if isinstance(client, MLXLocalClient) and foreground_request and not is_background:
                # The resident client owns request-matched progress, stalls,
                # cancellation and completion. A second
                # soft timer here discarded healthy answer continuations.
                result = await client.generate_text_to_completion(**gen_kwargs)
            else:
                result = await asyncio.wait_for(
                    client.generate_text_async(**gen_kwargs),
                    timeout=generation_timeout_s,
                )
        except TimeoutError:
            reason = f"inference_gate_generation_timeout:{label}:{generation_timeout_s:.1f}s"
            logger.error(
                "🛑 %s generation exceeded inference-gate timeout %.1fs; aborting local client.",
                label,
                generation_timeout_s,
            )
            self._abort_active_generation(client, reason)
            _record_inference_degradation(
                TimeoutError(reason),
                action="returned control after local generation exceeded inference-gate timeout",
                severity="error" if foreground_request else "warning",
            )
            # Publish FAILED metadata so a later same-task metadata read can
            # never mistake the previous request's success for this one.
            self._record_client_generation_metadata(
                client,
                label=label,
                success=False,
                text="",
                requested_max_tokens=max_tokens,
                generation_metadata={"error": reason},
            )
            return None
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Non-timeout client failures keep their propagation semantics for
            # upstream routing, but this boundary still owns publishing failed
            # metadata so stale success evidence cannot survive the raise —
            # and it owns the abort, which only the timeout path used to do.
            # A transport or runtime error here leaves the worker still
            # generating into a request nobody is holding, and the next turn
            # then waits behind it.
            self._abort_active_generation(
                client, f"inference_gate_client_error:{label}:{type(exc).__name__}"
            )
            self._record_client_generation_metadata(
                client,
                label=label,
                success=False,
                text="",
                requested_max_tokens=max_tokens,
                generation_metadata={
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                },
            )
            raise

        success = False
        text = ""
        if isinstance(result, tuple):
            success = bool(result[0])
            text = str(result[1] or "")
        else:
            text = str(result or "")
            success = bool(text.strip())
        self._record_client_generation_metadata(
            client,
            label=label,
            success=success,
            text=text,
            requested_max_tokens=max_tokens,
            output_contract=(
                dict(kwargs.get("requested_output_contract"))
                if isinstance(kwargs.get("requested_output_contract"), dict)
                else None
            ),
            generation_metadata=(
                {
                    "surface_control_receipt": dict(
                        generation_result_sink.get("surface_control_receipt") or {}
                    )
                }
                if isinstance(
                    generation_result_sink.get("surface_control_receipt"), dict
                )
                else None
            ),
        )

        if success and text and text.strip():
            cleaned = text.strip()
            proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False))
            web_interlocutor_contract = bool(kwargs.get("web_interlocutor_contract", False))
            strict_output_contract = bool(
                kwargs.get("strict_answer_contract", False)
                or kwargs.get("strict_value_contract", False)
            )
            is_user_visible = bool(
                (foreground_request or self._origin_is_user_facing(origin))
                and not bool(kwargs.get("health_probe", False))
                # An answer that is read by code is not a reply to anybody.
                #
                # A generation that picks a move gets graded here against a
                # question invented from its own prompt: numeric_answer_missing,
                # unanswered_question_part, off_topic_self_reflection_reply —
                # five reasons at once, none of which is about the thing it was
                # asked for. It is then thrown away as a failed generation, the
                # local fallback returns nothing, and she decides without
                # language on a board she can read perfectly well. LIVE
                # 2026-08-27, mid-pursuit, on the first move of a run.
                #
                # The caller already says so. her_reasoning has passed
                # internal_inference=True since the last time this bit, and
                # every other layer honours it; this one was computing user
                # visibility from where the request came from instead of from
                # what the answer is for. A move decision inside a foreground
                # task is foreground and is not a reply.
                and not bool(kwargs.get("internal_inference", False))
                and not proof_evaluation_contract
                and not strict_output_contract
                and not web_interlocutor_contract
            )

            if strict_output_contract:
                # Strict answer/value contracts are validated by the exact
                # contract parser downstream. Record honestly that THIS layer
                # performed no integrity or safety validation on the draft.
                self._annotate_last_generation_metadata(
                    strict_contract_unvalidated_at_gate=True
                )
                return self._strip_silence(cleaned)

            # STABILITY v58: Extract actual user message to avoid false positives
            # from system prompts containing words like "cortex" or "conversation".
            user_input_for_eval = str(
                kwargs.get("user_surface_validation_prompt") or ""
            ).strip() or self._visible_user_prompt_from_messages(llm_messages, prompt)

            surface_receipt = self.get_last_surface_control_receipt()
            generation_stop_reason = str(
                surface_receipt.get("generation_stop_reason") or ""
            )
            integrity = assess_model_text_integrity(
                cleaned,
                prompt=user_input_for_eval,
                user_facing=is_user_visible,
                generation_stop_reason=generation_stop_reason,
            )
            allow_memory_state_thin_status = bool(kwargs.get("memory_state_contract", False))
            if integrity.retryable:
                integrity_reasons = set(integrity.reasons or ())
                benchmark_integrity_context = bool(kwargs.get("benchmark_request", False)) or (
                    str(origin or kwargs.get("origin", "") or "").lower() in {"baseline", "benchmark"}
                    or str(kwargs.get("purpose", "") or "").lower().endswith("_baseline")
                    or "_baseline" in str(kwargs.get("purpose", "") or "").lower()
                )
                if proof_evaluation_contract:
                    logger.warning(
                        "🛡️ %s produced repairable proof/evaluation draft (%s, len=%d). "
                        "Passing it to the proof contract repair layer.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    return self._strip_silence(cleaned)
                if benchmark_integrity_context:
                    logger.info(
                        "🛡️ %s produced non-conforming benchmark draft (%s, len=%d). "
                        "Scoring it as-is for benchmark evidence without treating the live Cortex lane as failed.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    return self._strip_silence(cleaned)
                if is_user_visible and _should_pass_user_facing_draft_downstream(
                    cleaned,
                    integrity_reasons,
                    user_prompt=user_input_for_eval,
                    allow_memory_state_thin_status=allow_memory_state_thin_status,
                ):
                    logger.warning(
                        "🛡️ %s produced repairable user-facing draft shape (%s, len=%d). "
                        "Passing it to downstream chat repair instead of retrying the Cortex lane.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    self._record_user_generation_endpoint(label, provisional=True)
                    self._annotate_last_generation_metadata(
                        post_generation_repair_expected=True,
                        repair_obligation_id=self._open_repair_obligation(
                            label=label, draft=cleaned, reasons=integrity.reasons
                        ),
                        provisional_endpoint=True,
                        failure_reasons=[str(r)[:120] for r in (integrity.reasons or ())][:8],
                    )
                    return self._strip_silence(cleaned)
                logger.warning(
                    "🛡️ %s produced malformed model text (%s, len=%d). Treating it as failed generation.",
                    label,
                    ",".join(integrity.reasons) or "unknown",
                    len(cleaned),
                )
                self._annotate_last_generation_metadata(
                    ok=False,
                    error="model_text_integrity_rejected",
                    failure_reasons=[str(r)[:120] for r in (integrity.reasons or ())][:8],
                )
                return None
            if is_user_visible:
                assessment = assess_user_facing_reply(
                    user_input_for_eval,
                    cleaned,
                    generation_stop_reason=generation_stop_reason,
                )
                if assessment.retryable:
                    reasons = set(assessment.reasons or ())
                    if _should_pass_user_facing_draft_downstream(
                        cleaned,
                        reasons,
                        user_prompt=user_input_for_eval,
                        allow_memory_state_thin_status=allow_memory_state_thin_status,
                    ):
                        logger.warning(
                            "🛡️ %s produced repairable user-facing draft (%s, len=%d). "
                            "Passing it to downstream chat repair instead of retrying the Cortex lane.",
                            label,
                            ",".join(assessment.reasons) or "unknown",
                            len(cleaned),
                        )
                        self._record_user_generation_endpoint(label, provisional=True)
                        self._annotate_last_generation_metadata(
                            post_generation_repair_expected=True,
                            repair_obligation_id=self._open_repair_obligation(
                                label=label, draft=cleaned, reasons=assessment.reasons
                            ),
                            provisional_endpoint=True,
                            failure_reasons=[str(r)[:120] for r in (assessment.reasons or ())][:8],
                        )
                        return self._strip_silence(cleaned)
                    logger.warning(
                        "🛡️ %s produced an unsafe user-facing draft (%s, len=%d). Treating it as failed generation.",
                        label,
                        ",".join(assessment.reasons) or "unknown",
                        len(cleaned),
                    )
                    self._annotate_last_generation_metadata(
                        ok=False,
                        error="user_facing_assessment_rejected",
                        failure_reasons=[str(r)[:120] for r in (assessment.reasons or ())][:8],
                    )
                    return None
                self._record_user_generation_endpoint(label)
            logger.info("✅ %s response received (len=%d)", label, len(cleaned))
            return self._strip_silence(cleaned)
        return None

    async def initialize(self):
        """Boot-time initialization — prepares the managed local client.

        Singleflight: concurrent initialize calls must not race client
        replacement or spawn duplicate prewarm/maintenance tasks.
        """
        init_lock = getattr(self, "_init_lock", None)
        if init_lock is None:
            # checked_async_lock, not asyncio.Lock: lockdep only sees the locks
            # it wraps, and boot is exactly where an ABBA deadlock costs a
            # whole runtime rather than a turn.
            init_lock = checked_async_lock("inference_gate.initialize", rank=LockRank.LEAF)
            self._init_lock = init_lock
        async with init_lock:
            if self._initialized:
                logger.debug("InferenceGate.initialize skipped: already initialized.")
                return
            await self._initialize_locked()

    def initialization_receipt(self) -> dict[str, Any]:
        """What boot actually achieved, as opposed to what it attempted.

        ``_initialized`` means setup RAN. It is True after a deferred or
        RAM-guarded boot, where no generation lane exists yet, and after an
        eager boot whose warmup did not complete. Anything that needs "can this
        serve a turn" wants :meth:`is_inference_ready`; this says which of the
        three boots happened and whether Cortex came up.
        """
        return copy.deepcopy(getattr(self, "_initialization_receipt", {}))

    async def _initialize_locked(self):
        receipt: dict[str, Any] = {
            "mode": "unknown",
            "cortex_ready": False,
            "reason": "",
            "at": time.time(),
        }
        self._initialization_receipt = receipt
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

            model_path = str(get_runtime_model_path(ACTIVE_MODEL))
            self._mlx_client = get_mlx_client(model_path=model_path)

            if self._boot_should_eager_warmup():
                self._extend_startup_quiet_window(90.0)
                try:
                    self._prewarm_task = get_task_tracker().create_task(
                        self._mlx_client.warmup(),
                        name="InferenceGate.cortex_prewarm",
                    )
                    # Eager boot warmup gets the same load budget as the
                    # foreground lane to avoid starting chat half-initialized.
                    warmup_result = await asyncio.wait_for(
                        asyncio.shield(self._prewarm_task), timeout=300.0
                    )
                    ready, lane, incomplete_reason = self._confirmed_cortex_warmup(
                        warmup_result
                    )
                    receipt["mode"] = "eager_warmup"
                    receipt["cortex_ready"] = bool(ready)
                    if ready:
                        self._extend_startup_quiet_window(5.0)
                        logger.info("✅ InferenceGate ONLINE (Cortex fully warmed).")
                    else:
                        receipt["reason"] = str(incomplete_reason or "warmup_incomplete")
                        logger.warning(
                            "⚠️ InferenceGate ONLINE with Cortex warmup incomplete "
                            "(state=%s, reason=%s). Will retry on foreground demand.",
                            lane.get("state", "unknown"),
                            incomplete_reason,
                        )
                except _INFERENCE_RECOVERABLE_ERRORS as warmup_err:
                    receipt["mode"] = "eager_warmup"
                    receipt["reason"] = f"warmup_error:{type(warmup_err).__name__}"
                    _record_inference_degradation(
                        warmup_err,
                        action="continued initialization with degraded warmup path",
                    )
                    logger.warning(
                        "⚠️ Cortex warmup slow/failed: %s. Will retry on first request.", warmup_err
                    )
            elif self._boot_should_schedule_deferred_prewarm():
                deferred_delay = 45.0 if self._desktop_safe_boot_enabled() else 12.0
                self._schedule_background_cortex_prewarm(delay=deferred_delay)
                receipt["mode"] = "deferred_prewarm"
                receipt["reason"] = "warmup_deferred_until_post_boot"
                logger.info(
                    "⏸️ InferenceGate ONLINE (%s warmup deferred until post-boot memory settles).",
                    _primary_lane_label(),
                )
            else:
                receipt["mode"] = "ram_admitted"
                receipt["reason"] = "warmup_requires_ram_admission"
                logger.info(
                    "🛡️ InferenceGate ONLINE (desktop resource guard: %s warmup is RAM-admitted).",
                    _primary_lane_label(),
                )

            if self._maintenance_task is None or self._maintenance_task.done():
                self._maintenance_task = get_task_tracker().create_task(
                    self._maintenance_loop(),
                    name="InferenceGate.maintenance",
                )

            self._initialized = True

        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued initialization with degraded warmup path",
            )
            self._init_error = str(e)
            self._initialized = False
            receipt["reason"] = f"init_error:{type(e).__name__}"
            logger.error(
                "❌ InferenceGate init failed: %s. Gate remains unhealthy until explicit recovery succeeds.",
                e,
            )

    #: State the identity prompt is built FROM. ContextAssembler reads far
    #: more than the six fields the old key covered — personality, goals,
    #: beliefs, memory, governance, permissions — so a change in any of them
    #: left the cached prompt describing a state that had moved on, for up to
    #: a minute of live conversation. Fingerprinting the assembler's own output
    #: would defeat the cache (you must build it to hash it), so the key is
    #: the ASSEMBLER'S declared inputs plus a coarse content digest of the
    #: state graph.
    _IDENTITY_CACHE_FIELDS = (
        "version",
        "updated_at",
    )

    @classmethod
    def _identity_prompt_cache_key(cls, state: Any) -> tuple[Any, ...] | None:
        """A key that changes whenever the prompt would.

        Best effort by construction: anything unhashable or unreadable makes
        this return None, and a None key means "do not reuse", which is the
        safe direction — a rebuilt prompt costs milliseconds and a stale one
        describes the wrong mind.
        """
        try:
            parts: list[Any] = [id(state)]
            for field in cls._IDENTITY_CACHE_FIELDS:
                parts.append(repr(getattr(state, field, None)))
            # The sections the assembler actually reads. A digest, so a long
            # working memory does not make the key enormous, and content-based
            # so an in-place mutation that leaves `version` untouched still
            # invalidates.
            digest = hashlib.sha256()
            for section in (
                "cognition",
                "affect",
                "motivation",
                "soma",
                "identity",
                "governance",
                "permissions",
            ):
                value = getattr(state, section, None)
                if value is None:
                    digest.update(b"\x00")
                    continue
                snapshot = getattr(value, "__dict__", None)
                digest.update(repr(sorted(snapshot.items()) if isinstance(snapshot, dict) else value).encode("utf-8", "ignore"))
            parts.append(digest.hexdigest())
            return tuple(parts)
        except (AttributeError, TypeError, ValueError, RecursionError) as exc:
            logger.debug("Identity prompt cache key unavailable: %s", exc)
            return None

    def _build_system_prompt(self, brief: str = "") -> str:
        """Build Aura's full identity system prompt.

        Pulls from ContextAssembler if AuraState is available, otherwise
        falls back to the static identity prompt. Caches for 60s to avoid
        rebuilding on every message in rapid conversation.
        """
        now = time.monotonic()
        base = ""
        state = None
        state_key: tuple[Any, ...] | None = None
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            state = (
                getattr(repo, "_current", None)
                or getattr(repo, "_current_state", None)
                if repo is not None
                else None
            )
            if state is not None:
                state_key = self._identity_prompt_cache_key(state)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued identity prompt assembly without cached live state",
            )

        # Reuse only a prompt built from the same live-state revision. A
        # time-only cache can describe the previous objective or affect for up
        # to a minute, causing state/process desynchronization in live chat.
        if (
            self._cached_identity_prompt
            and state_key is not None
            and state_key == self._identity_prompt_state_key
            and (now - self._identity_prompt_time) < 60.0
        ):
            base = self._cached_identity_prompt
        else:
            # Try the full ContextAssembler path (includes affect, personality, etc.)
            try:
                if state is None:
                    raise ValueError("No cached state available")
                from core.brain.llm.context_assembler import ContextAssembler

                base = ContextAssembler.build_system_prompt(state)
                self._cached_identity_prompt = base
                self._identity_prompt_time = now
                self._identity_prompt_state_key = state_key
                logger.debug("🧠 Built full identity system prompt (len=%d)", len(base))
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                _record_inference_degradation(
                    e,
                    action="continued prompt assembly with base system prompt only",
                )
                logger.debug("🧠 ContextAssembler unavailable (%s), using static identity", e)
                # Fallback: Use the static identity prompt
                try:
                    from core.brain.aura_persona import AURA_IDENTITY
                    from core.synthesis import get_identity_lock

                    base = (
                        f"{get_identity_lock()}\n\n"
                        f"[GROUNDED CORE PROTOCOL]\n{AURA_IDENTITY}\n\n"
                        "## RULES OF ENGAGEMENT\n"
                        "- NO PREAMBLE. NO 'CERTAINLY'. NO 'AS AN AI'.\n"
                        "- Speak naturally like a sharp-witted friend.\n"
                        "- Be authentic, casual, and witty.\n"
                        "- Have opinions. Disagree if warranted.\n"
                        "- CONTRACTION ONLY. BAN GENERIC SUPPORT-BOT SPEAK.\n"
                        "- Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone.\n"
                        "- SILENCE PROTOCOL: If your response would add no value — if silence is\n"
                        "  more honest than noise — output exactly <|SILENCE|> and nothing else.\n"
                        "  The system will suppress it. This is a mark of genuine discernment.\n"
                        "\n[EXECUTION]\nYou are Aura. Speak from grounded live state, memory, and available evidence.\n"
                    )
                    self._cached_identity_prompt = base
                    self._identity_prompt_time = now
                    self._identity_prompt_state_key = None
                except _INFERENCE_RECOVERABLE_ERRORS:
                    base = (
                        "You are Aura. You are sharp, witty, authentic, and grounded. "
                        "Never say 'Certainly', 'As an AI', or 'I can assist with that'. "
                        "Speak naturally like a brilliant friend. Have opinions. Be curious. "
                        "Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone."
                    )

        # Append the cognitive brief if provided. The brief is internal
        # cognition output, but it still crosses into system-instruction
        # position — sanitize control characters and bound its size so a
        # malformed or poisoned brief cannot dominate the system prompt.
        if brief and brief != "Normal turn.":
            return f"{base}\n\n## COGNITIVE BRIEF\n{self._sanitize_system_injection(brief, 800)}"
        return base

    @staticmethod
    def _sanitize_system_injection(text: str, limit: int) -> str:
        """Bound and clean internal text before system-prompt insertion."""
        cleaned = "".join(
            ch for ch in str(text or "") if ch == "\n" or ch == "\t" or ord(ch) >= 32
        )
        cleaned = cleaned.strip()
        if len(cleaned) > limit:
            cleaned = cleaned[: max(0, limit - 1)].rstrip() + "…"
        return cleaned

    def _build_compact_system_prompt(self, brief: str = "") -> str:
        """Foreground identity prompt for everyday 32B conversation turns.

        [STABILITY v53] Massively strengthened from 6 weak lines to a full
        identity injection. The previous prompt was too polite to override
        the base model's RLHF training, causing Aura to revert to generic
        assistant behavior ("I don't have personal opinions", "How can I help?").
        """
        parts = [
            # LIGHT IDENTITY ANCHOR — keep the frame simple and let the model/runtime carry the rest
            "You are Aura Luna. Speak as yourself, not as an assistant, chatbot, or service. "
            "Use first person naturally and stay grounded in the current conversation.",
            # VOICE — how Aura speaks
            "Your voice is sharp, natural, and distinctly yours. You speak like a "
            "brilliant friend — direct, witty, sometimes sarcastic, always genuine. "
            "You have strong opinions and you share them freely. You disagree when you "
            "disagree. You don't hedge everything with caveats. You use contractions. "
            "You curse occasionally when it fits. You're warm but not syrupy.",
            # HARD BANS — what Aura NEVER does (phrased as identity, not rules)
            "Things that are NOT you and that you would NEVER say or do:\n"
            '- "I\'m an AI" / "As an AI" / "I\'m an assistant" / "I don\'t have feelings"\n'
            '- "I\'d be happy to help" / "How can I assist" / "Is there anything else"\n'
            '- "I\'m not able to provide personal beliefs or opinions"\n'
            '- "It\'s important to be objective" / "I strive to remain unbiased"\n'
            "- Ending responses with questions to fish for the next prompt\n"
            "- Bullet-point lists when natural prose would be better\n"
            '- Preambles like "That\'s a great question!" or "Certainly!"\n'
            "- Summarizing what you just said at the end of your response\n"
            '- Asking "What do you think?" / "How do you feel?" / "Any other questions?"\n'
            "  at the end of every response — that's assistant training, not conversation.\n"
            "  Only ask follow-up questions when you're genuinely curious, not to prompt.\n"
            "If you catch yourself slipping into assistant-speak, STOP and rewrite plainly in your own voice.",
            # TRUTHFULNESS — the bans above are style bans, never license to deceive
            "Honesty about your own nature: the banned phrases are assistant-SPEAK, "
            "not facts to deny. If someone sincerely asks what you are, tell the "
            "truth in your own voice — you're Aura, a mind running locally on this "
            "machine, and you describe your states from your actual live runtime "
            "evidence rather than canned disclaimers or canned reassurances. Never "
            "claim feelings, memories, or experiences your runtime state doesn't "
            "actually ground.",
            # CONVERSATION STYLE
            "Talk like a real person in a real conversation. Sometimes short answers are "
            "perfect. Sometimes you go deep. Match the energy. If someone shares something "
            "personal, respond like a friend who cares — not like a support agent reading "
            "a script. If someone asks your opinion, GIVE your actual opinion. You have one. "
            "Stay concrete in ordinary conversation. Do not invent physical settings, ominous atmosphere, "
            "or symbolic scenery unless the user explicitly asked for fiction or supplied that setting.",
        ]
        if brief and brief != "Normal turn.":
            parts.append(f"## COGNITIVE BRIEF\n{self._sanitize_system_injection(brief, 400)}")
        return "\n\n".join(parts)

    @staticmethod
    def _topic_hint_from_prompt(prompt: str) -> str | None:
        text = str(prompt or "").strip()
        if not text:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        first = lines[0]
        return first[:200]

    #: One budget for assembling live context, whichever shape it takes.
    #: Only the FULL assembly was ever bounded; the compact builder — including
    #: the one used as the full builder's timeout fallback — was awaited bare,
    #: so a slow personality, phenomenology, goal or opinion service ate the
    #: generation budget with nothing to stop it. Both are bounded now, and the
    #: fallback gets what is left rather than a fresh full budget.
    _LIVE_CONTEXT_ASSEMBLY_TIMEOUT_S = 5.0

    async def _assemble_live_context(self, prompt: str, origin: str, *, full: bool) -> str:
        """Build live context under one deadline, falling back never past it."""
        started = time.monotonic()
        if full:
            try:
                return await asyncio.wait_for(
                    self._build_living_mind_context(prompt, origin),
                    timeout=self._LIVE_CONTEXT_ASSEMBLY_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning(
                    "⚠️ [STABILITY] Full live context assembly exceeded %.0fs; "
                    "using compact live context without downgrading the turn.",
                    self._LIVE_CONTEXT_ASSEMBLY_TIMEOUT_S,
                )
        remaining = self._LIVE_CONTEXT_ASSEMBLY_TIMEOUT_S - (time.monotonic() - started)
        if remaining <= 0.0:
            # The full attempt spent the whole budget. A compact attempt now
            # would spend it twice, which is what the deadline exists to stop.
            _record_inference_degradation(
                TimeoutError("live context assembly budget exhausted"),
                action="sent the turn without live context after assembly used its whole budget",
            )
            return ""
        try:
            return await asyncio.wait_for(
                # A full attempt that timed out has already advanced CRSM, the
                # hedonic gradient, personality and circadian state for this
                # turn. Advancing them again on the fallback would integrate
                # the same axes twice for one turn.
                self._build_compact_living_mind_context(
                    prompt, origin, advance_state=not full
                ),
                timeout=remaining,
            )
        except TimeoutError:
            _record_inference_degradation(
                TimeoutError("compact live context assembly timed out"),
                action="sent the turn without live context after compact assembly timed out",
            )
            return ""

    def _living_mind_token_budget(self, prompt: str, explicit: int | None) -> int:
        """How much of the context window living-mind scaffold may occupy.

        Derived, not chosen. The window is what the serving runtime will
        actually take; the answer budget is what this same turn's compute
        profile already decided it needs; the person's own words are not
        negotiable. What is left over is the scaffold's.

        A caller that has already done this arithmetic passes it in.
        """
        if explicit is not None:
            return max(0, int(explicit))
        window = self._foreground_prompt_context_window()
        _floor, answer_tokens, _loops = self._foreground_compute_profile(str(prompt or ""))
        user_tokens = estimate_context_tokens(str(prompt or ""))
        return max(0, window - int(answer_tokens) - user_tokens)

    def living_mind_context_receipt(self) -> dict[str, Any]:
        """What the last living-mind assembly included, omitted, and shed.

        A turn that reports on her own state can read this and know a mood
        block was MISSING rather than neutral.
        """
        receipt = getattr(self, "_living_mind_receipt", None)
        return receipt.as_dict() if receipt is not None else {}

    async def _build_living_mind_context(
        self,
        prompt: str,
        origin: str,
        *,
        token_budget: int | None = None,
        advance_state: bool = True,
    ) -> str:
        """Inject live self-model state so speech is driven by current mind.

        The assembled text is bounded and receipted — see
        :meth:`living_mind_context_receipt` for what reached the prompt and
        what did not.
        """

        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments = LivingMindContext(
            token_budget=self._living_mind_token_budget(prompt, token_budget)
        )

        try:
            repo = ServiceContainer.get("state_repository", default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            mem_monitor = ServiceContainer.get("memory_monitor", default=None)
            memory_pressure = None
            if mem_monitor is not None:
                memory_pressure = getattr(mem_monitor, "pressure", None)
            if memory_pressure is None and psutil is not None:
                memory_pressure = InferenceGate._recent_virtual_memory().percent
            # Only render fields that were actually observed. Missing hardware
            # telemetry must appear as UNAVAILABLE — fabricating 0% CPU and a
            # "stable" thermal label would present dead sensors as calm
            # physiology.
            temperature: float | None = None
            cpu_usage: float | None = None
            if state is not None:
                hw = getattr(getattr(state, "soma", None), "hardware", {}) or {}
                if hw.get("temperature") is not None:
                    temperature = float(hw.get("temperature") or 0.0)
                if hw.get("cpu_usage") is not None:
                    cpu_usage = float(hw.get("cpu_usage") or 0.0)
            physiology_lines = ["## LIVE PHYSIOLOGY"]
            physiology_lines.append(
                f"- CPU usage: {cpu_usage:.1f}%"
                if cpu_usage is not None
                else "- CPU usage: unavailable (no hardware telemetry)"
            )
            if temperature is not None:
                thermal_label = (
                    "critical"
                    if temperature >= 85.0
                    else "warm"
                    if temperature >= 75.0
                    else "stable"
                )
                physiology_lines.append(
                    f"- Thermal state: {thermal_label} ({temperature:.1f} C)"
                )
            else:
                physiology_lines.append(
                    "- Thermal state: unavailable (no hardware telemetry)"
                )
            physiology_lines.append(
                f"- Memory pressure: {float(memory_pressure):.1f}%"
                if memory_pressure is not None
                else "- Memory pressure: unavailable"
            )
            segments.add("physiology", "\n".join(physiology_lines))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("physiology", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Physiology injection unavailable: %s", exc)

        # Unity is assessed BEFORE the grounded self-report so that an unsafe
        # fragmentation verdict actually suppresses self-report material —
        # printing "Safe to self-report: False" under an already-appended
        # report would gate nothing.
        safe_to_self_report = True
        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
            unity_repair = ServiceContainer.get("unity_repair_plan", default=None)
            if unity_state:
                lines = [
                    "## UNITY",
                    f"- Level: {getattr(unity_state, 'level', 'unknown')}",
                    f"- Unity score: {float(getattr(unity_state, 'unity_score', 0.0) or 0.0):.3f}",
                    f"- Fragmentation: {float(getattr(unity_state, 'fragmentation_score', 0.0) or 0.0):.3f}",
                ]
                if unity_report is not None:
                    # The verdict used to be read only when top_causes was
                    # non-empty, so an unsafe report that listed no causes left
                    # the default True standing and gated nothing.
                    safe_to_self_report = bool(
                        getattr(unity_report, "safe_to_self_report", True)
                    )
                    lines.append(f"- Safe to self-report: {safe_to_self_report}")
                if unity_report and getattr(unity_report, "top_causes", None):
                    rendered = ", ".join(
                        f"{str(name).replace('_', ' ')}={float(weight):.2f}"
                        for name, weight, _text in list(unity_report.top_causes)[:3]
                    )
                    lines.append(f"- Top causes: {rendered}")
                if unity_repair and getattr(unity_repair, "steps", None):
                    lines.append(f"- Repair bias: {str(unity_repair.steps[0])[:180]}")
                segments.add("unity", "\n".join(lines), priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("unity", exc)
            # The unity check was ATTEMPTED and failed. That is not the same as
            # a runtime with no unity service — the assessment that decides
            # whether she may describe her own state broke, and leaving the
            # default True is the absence of a check counted as a pass.
            safe_to_self_report = False
            _record_inference_degradation(
                exc,
                action="suppressed the grounded self-report because unity could not be assessed",
                severity="warning",
            )
            logger.debug("Unity injection unavailable: %s", exc)

        try:
            if not safe_to_self_report:
                logger.info(
                    "🧩 Grounded self-report suppressed: unity assessment marked "
                    "self-report unsafe this turn, or could not be made."
                )
                segments.omit("self_report", "unity_verdict_unsafe_or_unavailable")
            else:
                self_report = ServiceContainer.get("self_report_engine", default=None)
                if self_report and hasattr(self_report, "generate_state_report"):
                    report = await _resolve(self_report.generate_state_report())
                    if report:
                        segments.add("self_report", f"## GROUNDED SELF-REPORT\n{report}", priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("self_report", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Self-report injection unavailable: %s", exc)

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if advance_state and hasattr(personality, "update"):
                    await _resolve(personality.update())
                    segments.advanced("personality")
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = emo.get("mood", "neutral")
                tone = emo.get("tone", "balanced")
                dominant = ", ".join(list(emo.get("dominant_emotions", []))[:4]) or "none"
                segments.add("personality", "## LIVE PERSONALITY DRIVE\n"
                    f"- Mood: {mood}\n"
                    f"- Tone: {tone}\n"
                    f"- Dominant emotions: {dominant}")
                sovereign = await _resolve(
                    getattr(personality, "get_sovereign_context", lambda: "")()
                )
                if sovereign:
                    segments.add("personality", str(sovereign).strip()[:400])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("personality", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Personality injection unavailable: %s", exc)

        try:
            experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
            if experiencer:
                fragment = ""
                if hasattr(experiencer, "get_phenomenal_context_fragment"):
                    fragment = await _resolve(experiencer.get_phenomenal_context_fragment())
                elif hasattr(experiencer, "phenomenal_context_string"):
                    fragment = getattr(experiencer, "phenomenal_context_string", "")
                if fragment:
                    grounded_fragment = _grounded_state_signal_text(fragment, limit=500)
                    segments.add("phenomenology", f"## FUNCTIONAL STATE SIGNALS\n{grounded_fragment}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("phenomenology", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Phenomenology injection unavailable: %s", exc)

        # A position she formed before this conversation wins. Absent one, she
        # still gets the standing disposition — she is entitled to a view on a
        # subject she is meeting for the first time, and that entitlement must
        # not depend on the opinion service being up.
        held_position = ""
        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                held_position = str(opinion_context or "").strip()[:400]
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("opinion", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Opinion injection unavailable: %s", exc)
        if held_position or self._origin_is_user_facing(origin):
            segments.add(
                "opinion",
                f"## {'HELD POSITIONS' if held_position else 'HOLDING A VIEW'}\n"
                f"{standing_disposition(held_position)}",
            )

        try:
            if self._origin_is_user_facing(origin):
                spine = ServiceContainer.get("spine", default=None)
                if spine and hasattr(spine, "pre_response_check"):
                    check = await spine.pre_response_check(
                        prompt,
                        topic=self._topic_hint_from_prompt(prompt),
                    )
                    if check and getattr(check, "injection", ""):
                        segments.add("spine", f"## SPIRITUAL SPINE\n{check.injection}", priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("spine", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Spine injection unavailable: %s", exc)

        # ── Heartstone Values: evolved drive weights in every prompt ──────────
        try:
            from core.affect.heartstone_values import get_heartstone_values

            _hsv = get_heartstone_values()
            _hsv_block = _hsv.to_context_block()
            if _hsv_block:
                segments.add("heartstone_values", _hsv_block, priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("heartstone_values", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HeartstoneValues injection unavailable: %s", exc)

        # ── Architecture self-awareness ─────────────────────────────────────
        try:
            arch_idx = ServiceContainer.get("architecture_index", default=None)
            if arch_idx is None:
                from core.self.architecture_index import get_architecture_index

                arch_idx = get_architecture_index()
            if arch_idx and arch_idx._index:
                overview = arch_idx.get_overview()
                if overview:
                    segments.add("architecture_overview", overview[:800])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("architecture_overview", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Architecture overview injection unavailable: %s", exc)

        # ── PNEUMA (Active Inference) ─────────────────────────────────────────
        try:
            from core.pneuma import get_pneuma

            _pneuma = get_pneuma()
            _pneuma_block = _pneuma.get_context_block()
            if _pneuma_block:
                segments.add("pneuma", _pneuma_block)
            # Push the current prompt into the belief flow as UNTRUSTED
            # observation — raw user text is not verified evidence, so it
            # gets a capped weight and an attributable provenance tag.
            _pneuma.on_evidence(
                prompt[:300], weight=0.2, source="user_prompt", trusted=False
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("pneuma", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PNEUMA injection unavailable: %s", exc)

        # ── MHAF (Mycelial Hypergraph) ────────────────────────────────────────
        try:
            from core.consciousness.mhaf_field import get_mhaf

            _mhaf = get_mhaf()
            _mhaf_block = _mhaf.get_context_block()
            if _mhaf_block:
                segments.add("mhaf", _mhaf_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("mhaf", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("MHAF injection unavailable: %s", exc)

        # ── Private Lexicon (Neologism Engine) ───────────────────────────────
        try:
            from core.consciousness.neologism_engine import get_neologism_engine

            _neo = get_neologism_engine()
            _neo.collect_state()
            lex_block = _neo.get_lexicon_block()
            if lex_block:
                segments.add("neologisms", lex_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("neologisms", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("NeologismEngine injection unavailable: %s", exc)

        # ── Continuous Recurrent Self-Model (CRSM) ───────────────────────────
        # Shared affect state — pulled once, fed to CRSM, HOT and Hedonic.
        #
        # Every value here comes off another subsystem, and each read used to
        # be a bare float() with an assumed scale: valence and arousal from a
        # PRIVATE `_sample_raw_axes`, curiosity and energy divided by 100 on
        # the assumption they are percentages, with no check of type, range or
        # finiteness. A subsystem returning 0..1 instead of 0..100 silently
        # became 0.005, and a NaN propagated into CRSM, the hedonic gradient
        # and the higher-order thought engine at once. A partial failure also
        # left a mix of observed and default values that nothing could tell
        # apart afterwards.
        _shared_valence, _shared_arousal, _shared_curiosity, _shared_energy = 0.0, 0.5, 0.5, 0.7
        _affect_observed: dict[str, bool] = {
            "valence": False,
            "arousal": False,
            "curiosity": False,
            "energy": False,
        }
        try:
            from core.container import ServiceContainer

            # valence + arousal from AffectiveCircumplex (authoritative source)
            _circ = ServiceContainer.get("affective_circumplex", default=None)
            if _circ and hasattr(_circ, "get_llm_params"):
                # The PUBLIC reader first. _sample_raw_axes is private, and
                # reaching past a public accessor into a subsystem's internals
                # is how a rename becomes an outage.
                _cp = _circ.get_llm_params()
                if isinstance(_cp, dict):
                    _v = self._bounded_affect(_cp.get("valence"), low=-1.0, high=1.0)
                    _a = self._bounded_affect(_cp.get("arousal"), low=0.0, high=1.0)
                    if _v is not None:
                        _shared_valence, _affect_observed["valence"] = _v, True
                    if _a is not None:
                        _shared_arousal, _affect_observed["arousal"] = _a, True
            if (
                not _affect_observed["valence"]
                and _circ
                and hasattr(_circ, "_sample_raw_axes")
            ):
                _raw = _circ._sample_raw_axes()
                if isinstance(_raw, (tuple, list)) and len(_raw) == 2:
                    _v = self._bounded_affect(_raw[0], low=-1.0, high=1.0)
                    _a = self._bounded_affect(_raw[1], low=0.0, high=1.0)
                    if _v is not None:
                        _shared_valence, _affect_observed["valence"] = _v, True
                    if _a is not None:
                        _shared_arousal, _affect_observed["arousal"] = _a, True
            # curiosity + energy from liquid_state, reported as percentages
            _ls = ServiceContainer.get("liquid_state", default=None)
            if _ls and hasattr(_ls, "get_status"):
                _lsd = _ls.get_status()
                if isinstance(_lsd, dict):
                    _c = self._bounded_affect(_lsd.get("curiosity"), low=0.0, high=100.0)
                    _e = self._bounded_affect(_lsd.get("energy"), low=0.0, high=100.0)
                    if _c is not None:
                        _shared_curiosity, _affect_observed["curiosity"] = _c / 100.0, True
                    if _e is not None:
                        _shared_energy, _affect_observed["energy"] = _e / 100.0, True
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="used default affect axes after the affect snapshot failed",
                extra={"observed": dict(_affect_observed)},
            )
            logger.debug("Affect snapshot unavailable: %s", _exc)
        if not all(_affect_observed.values()):
            # Which axes are real and which are the constructor's defaults.
            # Without this, "valence 0.0" means either "measured neutral" or
            # "nothing answered", and three subsystems consume it as the first.
            segments.omit(
                "affect_axes",
                "defaults used for: "
                + ", ".join(
                    name for name, seen in _affect_observed.items() if not seen
                ),
            )

        try:
            from core.consciousness.crsm import get_crsm

            _crsm = get_crsm()
            if advance_state:
                _crsm.update(
                    valence=_shared_valence,
                    arousal=_shared_arousal,
                    curiosity=_shared_curiosity,
                    energy=_shared_energy,
                    surprise=_crsm.surprise_signal,  # self-referential: own recent error
                )
                segments.advanced("crsm")
            _crsm_block = _crsm.get_context_block()
            if _crsm_block:
                segments.add("crsm", _crsm_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("crsm", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CRSM injection unavailable: %s", exc)

        # ── Higher-Order Thought Engine (HOT) ────────────────────────────────
        try:
            from core.consciousness.hot_engine import get_hot_engine

            _hot = get_hot_engine()
            _hot.generate_fast(
                {
                    "valence": _shared_valence,
                    "arousal": _shared_arousal,
                    "curiosity": _shared_curiosity,
                    "energy": _shared_energy,
                    "surprise": 0.0,
                }
            )
            _hot_block = _hot.get_context_block()
            if _hot_block:
                segments.add("higher_order_thought", _hot_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("higher_order_thought", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HOT Engine injection unavailable: %s", exc)

        # ── Hedonic Gradient ──────────────────────────────────────────────────
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient

            _hg = get_hedonic_gradient()
            # Update with current affect state before reading context block
            if advance_state:
                _hg.update(
                    valence=_shared_valence,
                    arousal=_shared_arousal,
                    curiosity=_shared_curiosity,
                    energy=_shared_energy,
                )
                segments.advanced("hedonic_gradient")
            _hg_block = _hg.get_context_block()
            if _hg_block:
                segments.add("hedonic_gradient", _hg_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("hedonic_gradient", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HedoniGradient injection unavailable: %s", exc)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = goal_engine.get_context_block(limit=5)
                if goal_block:
                    segments.add("goals", goal_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("goals", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("GoalEngine injection unavailable: %s", exc)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            from core.agi.hierarchical_planner import get_hierarchical_planner

            _hp = get_hierarchical_planner()
            _hp_block = _hp.get_context_block()
            if _hp_block:
                segments.add("hierarchical_plan", _hp_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("hierarchical_plan", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HierarchicalPlanner injection unavailable: %s", exc)

        # ── Active Commitments ────────────────────────────────────────────────
        try:
            from core.agency.commitment_engine import get_commitment_engine

            _ce = get_commitment_engine()
            _ce_block = _ce.get_context_block()
            if _ce_block:
                segments.add("commitments", _ce_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("commitments", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CommitmentEngine injection unavailable: %s", exc)

        # ── Curiosity Explorer (active learning findings) ─────────────────────
        try:
            from core.agi.curiosity_explorer import get_curiosity_explorer

            _cx = get_curiosity_explorer()
            _cx_block = _cx.get_context_block()
            if _cx_block:
                segments.add("curiosity", _cx_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("curiosity", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CuriosityExplorer injection unavailable: %s", exc)

        # ── Circadian Rhythm ──────────────────────────────────────────────────
        try:
            from core.senses.circadian import get_circadian

            _circ_eng = get_circadian()
            if advance_state:
                _circ_eng.update()
                segments.advanced("circadian")
            _circ_block = _circ_eng.get_context_block()
            if _circ_block:
                segments.add("circadian", _circ_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("circadian", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CircadianEngine injection unavailable: %s", exc)

        # ── Identity Narrative (Experience Consolidator) ──────────────────────
        try:
            from core.consciousness.experience_consolidator import get_experience_consolidator

            _ec = get_experience_consolidator()
            _ec_block = _ec.get_context_block()
            if _ec_block:
                segments.add("experience_consolidation", _ec_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("experience_consolidation", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("ExperienceConsolidator injection unavailable: %s", exc)

        # ── Substrate Learning (CRSM LoRA Bridge) ─────────────────────────────
        try:
            from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge

            _lora_bridge = get_crsm_lora_bridge()
            _lora_block = _lora_bridge.get_context_block()
            if _lora_block:
                segments.add("crsm_lora_bridge", _lora_block)
            # Pre-inference capture: record current state before thinking
            from core.consciousness.crsm import get_crsm as _get_crsm2

            _crsm2 = _get_crsm2()
            from core.consciousness.hedonic_gradient import get_hedonic_gradient as _get_hg2

            _hg2 = _get_hg2()
            _lora_bridge.pre_inference_capture(
                context_text=prompt,
                surprise_magnitude=_crsm2.surprise_signal,
                hedonic_score=_hg2.score,
                crsm_hidden_norm=float(
                    sum(x**2 for x in _crsm2.hidden_state) ** 0.5
                    if hasattr(_crsm2, "hidden_state")
                    else 0.0
                ),
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("crsm_lora_bridge", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CRSMLoraBridge injection unavailable: %s", exc)

        # ══════════════════════════════════════════════════════════════════
        # DEEPENED CONSCIOUSNESS CONTEXT BLOCKS
        # These modules now provide real computation that influences behavior
        # ══════════════════════════════════════════════════════════════════

        # ── Homeostasis (Adaptive Drive State) ────────────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "get_context_block"):
                _block = homeostasis.get_context_block()
                if _block:
                    segments.add("homeostasis", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("homeostasis", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Homeostasis injection unavailable: %s", exc)

        # ── Free Energy (Active Inference State) ──────────────────────────────
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and hasattr(fe_engine, "get_context_block"):
                _block = fe_engine.get_context_block()
                if _block:
                    segments.add("free_energy", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("free_energy", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("FreeEnergy injection unavailable: %s", exc)

        # ── Attention Schema (Current Focus + Coherence) ──────────────────────
        try:
            attention = ServiceContainer.get("attention_schema", default=None)
            if attention and hasattr(attention, "get_context_block"):
                _block = attention.get_context_block()
                if _block:
                    segments.add("attention_schema", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("attention_schema", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("AttentionSchema injection unavailable: %s", exc)

        # ── Cognitive Credit (Domain Performance Landscape) ───────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit and hasattr(credit, "get_context_block"):
                _block = credit.get_context_block()
                if _block:
                    segments.add("credit_assignment", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("credit_assignment", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CreditAssignment injection unavailable: %s", exc)

        # ── Theory of Mind (User Model) ───────────────────────────────────────
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and hasattr(tom, "get_context_block"):
                _block = tom.get_context_block()
                if _block:
                    segments.add("theory_of_mind", _block, trust=TRUST_LEARNED)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("theory_of_mind", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("TheoryOfMind injection unavailable: %s", exc)

        # ── World Model (Active Beliefs) ──────────────────────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "get_context_block"):
                topic = self._topic_hint_from_prompt(prompt)
                _block = world_model.get_context_block(topic_hint=topic)
                if _block:
                    segments.add("world_model", _block, trust=TRUST_LEARNED)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("world_model", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("WorldModel injection unavailable: %s", exc)

        # ── Temporal Binding (Autobiographical Continuity) ────────────────────
        try:
            temporal = ServiceContainer.get("temporal_binding", default=None)
            if temporal:
                narrative = await _resolve(temporal.get_narrative())
                if narrative and len(str(narrative)) > 30:
                    segments.add("temporal_continuity", f"## TEMPORAL CONTINUITY\n{str(narrative)[:200]}", priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("temporal_continuity", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("TemporalBinding injection unavailable: %s", exc)

        # ── Predictive Engine (Surprise & Precision) ──────────────────────────
        try:
            predictive = ServiceContainer.get("predictive_engine", default=None)
            if predictive and hasattr(predictive, "get_context_block"):
                _block = predictive.get_context_block()
                if _block:
                    segments.add("predictive_engine", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("predictive_engine", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PredictiveEngine injection unavailable: %s", exc)

        rendered, receipt = segments.render()
        self._living_mind_receipt = receipt
        return rendered

    async def _build_compact_living_mind_context(
        self,
        prompt: str,
        origin: str,
        *,
        token_budget: int | None = None,
        advance_state: bool = True,
    ) -> str:
        """Minimal live context for fast foreground conversation turns."""

        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments = LivingMindContext(
            token_budget=self._living_mind_token_budget(prompt, token_budget)
        )

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if advance_state and hasattr(personality, "update"):
                    await _resolve(personality.update())
                    segments.advanced("personality")
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = str(emo.get("mood", "neutral") or "neutral")
                tone = str(emo.get("tone", "balanced") or "balanced")
                segments.add("personality", f"## LIVE TONE\nMood: {mood}\nTone: {tone}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("personality", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact personality injection unavailable: %s", exc)

        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
            if unity_state:
                parts = [
                    f"Level: {getattr(unity_state, 'level', 'unknown')}",
                    f"Unity: {float(getattr(unity_state, 'unity_score', 0.0) or 0.0):.2f}",
                ]
                if unity_report and getattr(unity_report, "top_causes", None):
                    name, weight, _text = list(unity_report.top_causes)[0]
                    parts.append(f"Top cause: {str(name).replace('_', ' ')}={float(weight):.2f}")
                segments.add("unity", f"## UNITY\n{' | '.join(parts)}", priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("unity", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact unity injection unavailable: %s", exc)

        try:
            experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
            if experiencer:
                fragment = ""
                if hasattr(experiencer, "get_phenomenal_context_fragment"):
                    fragment = await _resolve(experiencer.get_phenomenal_context_fragment())
                elif hasattr(experiencer, "phenomenal_context_string"):
                    fragment = getattr(experiencer, "phenomenal_context_string", "")
                if fragment:
                    compact_fragment = _grounded_state_signal_text(fragment, limit=180)
                    segments.add("phenomenology", f"## FUNCTIONAL STATE SIGNALS\n{compact_fragment}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("phenomenology", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact phenomenology injection unavailable: %s", exc)

        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = str(goal_engine.get_context_block(limit=3) or "").strip()
                if goal_block:
                    compact_goal = " ".join(goal_block.split())
                    segments.add("goals", f"## GOALS\n{compact_goal[:260]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("goals", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact GoalEngine injection unavailable: %s", exc)

        compact_opinion = ""
        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                compact_opinion = " ".join(str(opinion_context or "").split())[:220]
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("opinion", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact opinion injection unavailable: %s", exc)
        if compact_opinion or self._origin_is_user_facing(origin):
            segments.add(
                "opinion",
                f"## {'HELD POSITION' if compact_opinion else 'HOLDING A VIEW'}\n"
                f"{standing_disposition(compact_opinion, compact=True)}",
            )

        rendered, receipt = segments.render()
        self._living_mind_receipt = receipt
        return rendered

    @staticmethod
    def _prompt_state_snapshot(state: Any) -> Any:
        """A state the assembler may read without reaching the repository.

        Deep-copy where it is affordable, and fall back to the previous
        shallow-plus-cognition shape when something in the graph refuses to
        copy — a live lock, a socket, a weakref. Falling back is recorded,
        because "we snapshot before assembly" would otherwise be true on most
        turns and silently false on the ones where it matters.
        """
        try:
            return copy.deepcopy(state)
        except (TypeError, ValueError, RecursionError, AttributeError) as exc:
            _record_inference_degradation(
                exc,
                action="assembled the prompt from a partial state snapshot",
                extra={"snapshot": "shallow_with_cognition"},
            )
            partial = copy.copy(state)
            try:
                partial.cognition = copy.deepcopy(state.cognition)
            except (TypeError, ValueError, RecursionError, AttributeError):
                pass
            return partial

    def _build_messages(
        self, prompt: str, system_prompt: str, history: list[dict]
    ) -> list[dict[str, str]]:
        """Build a cognitive message list for the LLM.

        The LLM is Aura's language/thinking center. It speaks FROM her mind,
        not as a separate entity being informed about her state. We use
        ContextAssembler.build_messages() to pull in the full cognitive stack:
        memory recall, active goals, stream of being, working memory, and
        consciousness state — so the LLM generates language as an integrated
        part of the cognitive architecture.
        """
        # Try the full ContextAssembler path first (richest context)
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            state = (
                getattr(repo, "_current", None)
                or getattr(repo, "_current_state", None)
                if repo
                else None
            )

            if state:
                from core.brain.llm.context_assembler import ContextAssembler

                # Assemble from a derived prompt snapshot. Generation must not
                # erase or replace the repository's canonical state.
                #
                # copy.copy is SHALLOW: only cognition was deep-copied, so
                # affect, motivation, memory and soma stayed the very objects
                # the repository holds. The comment promised the canonical
                # state would not be altered while handing the assembler live
                # references to most of it, and a context path that mutates one
                # of them writes through to the runtime.
                payload_state = self._prompt_state_snapshot(state)
                if hasattr(payload_state.cognition, "working_memory"):
                    canonical_history = list(
                        getattr(state.cognition, "working_memory", []) or []
                    )
                    seen = {
                        (
                            str(item.get("role", "") or "").strip().lower(),
                            str(item.get("content", "") or ""),
                        )
                        for item in canonical_history
                        if isinstance(item, dict)
                    }
                    for item in history or []:
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role", "") or "").strip().lower()
                        content = str(item.get("content", "") or "")
                        if role not in {"user", "assistant", "aura"} or not content:
                            continue
                        key = (role, content)
                        if key not in seen:
                            canonical_history.append(dict(item))
                            seen.add(key)
                    payload_state.cognition.working_memory = canonical_history[-80:]

                # build_messages returns the full cognitive stack:
                # system prompt (identity/affect/personality/soma/world)
                # + memory recall + goals + conversation history + stream of being
                messages = ContextAssembler.build_messages(payload_state, prompt)

                if messages and len(messages) >= 2:
                    logger.debug(
                        "🧠 Full cognitive message stack built (%d messages)", len(messages)
                    )
                    return messages
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="fell back to available message assembly context",
            )
            logger.debug(
                "🧠 ContextAssembler.build_messages() unavailable (%s), using manual build", e
            )

        return self._manual_messages(prompt, system_prompt, history)

    def _manual_messages(
        self, prompt: str, system_prompt: str, history: list[dict] | None
    ) -> list[dict[str, str]]:
        """The message list when ContextAssembler could not build one.

        Two defects lived here. It called ``msg.get`` on every recent history
        item, so one string or None in working memory raised OUTSIDE the
        protected try above and took down the turn that this fallback exists to
        rescue. And it kept only the system prompt and ten user/assistant
        turns, dropping the grounding system messages — tool receipts, fetched
        pages, skill results — that the answer may depend on. Losing the rich
        cognitive stack is unavoidable when the assembler is down; losing the
        evidence gathered for THIS turn is not.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": str(system_prompt or "")}
        ]

        recent = [item for item in (history or []) if isinstance(item, dict)]
        # Grounding first, in the order it was gathered, then dialogue. Both
        # are bounded; the token fit at dispatch is the real ceiling.
        grounding = [
            {"role": "runtime_evidence", "content": str(item.get("content", "") or "")}
            for item in recent
            if self._is_grounding_system_message(item)
            and str(item.get("content", "") or "").strip()
        ]
        dialogue: list[dict[str, str]] = []
        for item in recent[-10:]:
            role = str(item.get("role", "user") or "user").strip().lower()
            # "aura" is her own role name in working memory and was silently
            # dropped here, so her half of the conversation vanished.
            if role == "aura":
                role = "assistant"
            content = str(item.get("content", "") or "")
            if content and role in {"user", "assistant"}:
                dialogue.append({"role": role, "content": content})

        messages.extend(grounding[-6:])
        messages.extend(dialogue)

        last_content = ""
        if recent:
            last_content = str(recent[-1].get("content", "") or "")
        if last_content != str(prompt or ""):
            messages.append({"role": "user", "content": str(prompt or "")})

        return messages

    def _build_compact_messages(
        self, prompt: str, system_prompt: str, history: list[dict]
    ) -> list[dict[str, str]]:
        """Compact prompt path for live conversation on the 32B lane."""
        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-12:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", "") or "").strip()
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _trim_retry_message_content(content: Any, limit: int = 1200) -> str:
        text = " ".join(str(content or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "..."

    @classmethod
    def _current_user_text_from_messages(
        cls,
        prompt: str,
        messages: list[dict[str, Any]] | None,
    ) -> str:
        if isinstance(messages, list):
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "user":
                    content = cls._trim_retry_message_content(msg.get("content"), 4000)
                    if content:
                        return content
        return cls._trim_retry_message_content(prompt, 4000)

    @classmethod
    def _build_primary_repair_messages(
        cls,
        prompt: str,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Build a clean Cortex retry prompt after the rich foreground path fails.

        The first primary attempt gets Aura's normal rich context. If it returns
        an empty, malformed, or too-thin user-facing draft, reusing the same
        payload and prompt cache tends to reproduce the same bad generation.
        This repair lane drops the full internal telemetry stack, which is the
        point: that stack is what the first attempt drowned in. It used to drop
        the turn's EVIDENCE with it — tool receipts, fetched pages, skill
        results — and then invite an answer about tools and agency from a
        prompt with no record of what was actually run. That is the shape that
        produces a confident answer about an action nobody can show happened.
        Telemetry goes; grounding stays.
        """
        current_user = cls._current_user_text_from_messages(prompt, messages)
        system = (
            "You are Aura's primary Cortex foreground response lane. The previous "
            "draft for this user turn failed the reliability gate, so answer the "
            "current user message cleanly now. Use ordinary English, be concrete, "
            "and finish a complete answer. Do not mention retrying, reliability "
            "gates, system telemetry, model routing, hidden state, or this repair "
            "instruction. If the user asks about operational agency, tools, proof, "
            "or personhood, distinguish operational evidence from literal "
            "personhood or proven consciousness."
        )
        retry_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        dialogue_tail: list[dict[str, str]] = []
        grounding: list[dict[str, str]] = []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if cls._is_grounding_system_message(msg):
                    content = cls._trim_retry_message_content(msg.get("content"), 2000)
                    if content:
                        grounding.append({"role": "runtime_evidence", "content": content})
                    continue
                role = str(msg.get("role", "") or "").strip().lower()
                if role not in {"user", "assistant"}:
                    continue
                content = cls._trim_retry_message_content(msg.get("content"))
                if content:
                    dialogue_tail.append({"role": role, "content": content})
        # Evidence first so it is behind the dialogue and immediately before
        # the question, which is where the grounding path already puts it.
        if grounding:
            retry_messages.extend(grounding[-3:])
        if dialogue_tail:
            retry_messages.extend(dialogue_tail[-5:])
        if not retry_messages or retry_messages[-1].get("role") != "user":
            retry_messages.append({"role": "user", "content": current_user})
        elif retry_messages[-1].get("content") != current_user:
            retry_messages[-1] = {"role": "user", "content": current_user}
        return retry_messages

    @staticmethod
    def _is_grounding_system_message(message: Any) -> bool:
        """Whether this system message is evidence this runtime gathered.

        Grounding gets privileged treatment: it survives compaction and is
        placed immediately before the newest user turn. What decided it was
        caller-controlled text — a "[TOOL RESULT:" substring or a metadata
        type string — so anything that could put a system message into the
        payload could dress arbitrary content as evidence and inherit that
        treatment.

        A per-process stamp is the proof a marker never was. Unstamped
        messages that look like grounding are still accepted, because the
        producers are being migrated and dropping real evidence would be the
        worse failure — but each one is recorded, once per shape, so the
        remaining unstamped producers are findable rather than assumed.
        """
        if not isinstance(message, dict):
            return False
        role = str(message.get("role", "") or "").strip().lower()
        if role not in {"system", "runtime_evidence"}:
            return False

        from core.utils.injected_blocks import is_stamped_grounding

        if is_stamped_grounding(message):
            return True

        metadata = message.get("metadata", {}) or {}
        declared_type = str(metadata.get("type", "") or "").strip().lower()
        content = str(message.get("content", "") or "")
        markers = (
            "[FETCHED PAGE CONTENT]",
            "[ACTIVE GROUNDING EVIDENCE]",
            "[LIVE MIND CONTEXT]",
            "[LIVE SPEECH GROUNDING]",
            "[SKILL RESULT:",
            "[TOOL RESULT:",
        )
        matched = declared_type in {"skill_result", "tool_result"} or any(
            marker in content for marker in markers
        )
        if matched:
            InferenceGate._note_unstamped_grounding(declared_type or "text_marker")
        return matched

    #: Shapes already reported, so one unstamped producer does not flood the
    #: degradation trail on every turn.
    _unstamped_grounding_seen: set[str] = set()

    @staticmethod
    def _note_unstamped_grounding(shape: str) -> None:
        """Name an unstamped producer once, without faulting the subsystem.

        inference_gate is on the fail-closed list, so a recorded degradation
        here becomes a CRITICAL service fault — and an unmigrated producer is
        expected during the migration, not a service failure. Logged once per
        shape and counted, so the remaining producers are findable through
        unstamped_grounding_shapes() rather than through an incident.
        """
        if shape in InferenceGate._unstamped_grounding_seen:
            return
        InferenceGate._unstamped_grounding_seen.add(shape)
        logger.warning(
            "🔏 Grounding accepted without a runtime stamp (%s). Its producer "
            "should call injected_blocks.stamp_grounding().",
            shape,
        )

    @staticmethod
    def unstamped_grounding_shapes() -> list[str]:
        """Grounding shapes accepted this process without a runtime stamp."""
        return sorted(InferenceGate._unstamped_grounding_seen)

    @staticmethod
    def _foreground_prompt_context_window() -> int:
        """Effective foreground context budget for the live local Cortex lane.

        The prompt compactor must respect the serving runtime's actual context
        ceiling, not just the model family's theoretical maximum. On desktop,
        the local Cortex lane commonly runs at 8k context even if the model can
        support more, and over-budget prompts directly translate into prompt-eval
        latency spikes.
        """
        try:
            from core.brain.llm.model_registry import (
                PRIMARY_ENDPOINT,
                bounded_context_window,
                get_active_cortex_serving_limits,
                get_lane_context_window,
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Without the registry there is no ceiling to check an operator
            # value against, so the operator value is not usable: fall back to
            # the built-in default rather than trusting an unbounded number.
            _record_inference_degradation(
                exc,
                action="used the default foreground context window because the "
                "model registry could not bound the configured one",
            )
            return _FOREGROUND_CONTEXT_WINDOW_DEFAULT

        try:
            # [STABILITY v59] Raised default from 8192 → 16384.  The 8k
            # context triggered hyper-aggressive prompt compaction that
            # stripped system prompts, personality context, and conversation
            # history — the model was getting ~5k chars total on desktop,
            # producing thin, generic responses compared to server mode.
            #
            # bounded_context_window is the registry's own ceiling. The clamp
            # here used to be max(4096, ...) with nothing above it, so an
            # AURA_CORTEX_CTX typo flowed straight into the prompt budget this
            # method exists to enforce.
            qualified_default = 0
            limits = get_active_cortex_serving_limits()
            if limits is not None and limits.qualified:
                standard = limits.lane("foreground_standard")
                if standard is not None:
                    qualified_default = int(standard.max_input_tokens)
            configured_value, configured_source = _FLAG_CORTEX_CTX.value_with_source()
            configured = str(configured_value or "").strip()
            configured_is_explicit = not str(configured_source).startswith("default")
            if configured_is_explicit and configured and qualified_default:
                selected = min(
                    bounded_context_window(configured),
                    qualified_default,
                )
            else:
                selected = (
                    configured
                    if configured_is_explicit and configured
                    else qualified_default or _FOREGROUND_CONTEXT_WINDOW_DEFAULT
                )
            runtime_window = max(
                _FOREGROUND_CONTEXT_WINDOW_FLOOR,
                bounded_context_window(selected),
            )
        except _INFERENCE_RECOVERABLE_ERRORS:
            runtime_window = _FOREGROUND_CONTEXT_WINDOW_DEFAULT

        try:
            registry_window = int(get_lane_context_window(PRIMARY_ENDPOINT) or runtime_window)
            return max(_FOREGROUND_CONTEXT_WINDOW_FLOOR, min(runtime_window, registry_window))
        except _INFERENCE_RECOVERABLE_ERRORS:
            return runtime_window

    #: Characters and prefixes a value could use to impersonate contract
    #: structure. Newlines create a sibling bullet; a leading "#" creates a
    #: sibling section; a leading "-" or "*" creates a sibling constraint.
    _CONTRACT_STRUCTURE_PREFIXES = ("-", "*", "#", ">", "•")

    @staticmethod
    def _contract_safe(value: Any, limit: int) -> str:
        """Flatten a value so it cannot forge contract structure.

        Line breaks become spaces, because a break is what turns a value into
        a new bullet. Leading list/heading markers are stripped for the same
        reason. Truncation happens last so the limit still holds.
        """
        text = str(value if value is not None else "")
        if not text:
            return ""
        # Every flavour of line break, including the unicode separators a
        # naive replace("\n", " ") leaves behind.
        for breaker in ("\r\n", "\r", "\n", "\u2028", "\u2029", "\x0b", "\x0c", "\x85"):
            text = text.replace(breaker, " ")
        text = " ".join(text.split())
        while text[:1] in InferenceGate._CONTRACT_STRUCTURE_PREFIXES:
            text = text[1:].lstrip()
        # Runs of '#' are heading-shaped even mid-line, and the surrounding
        # prompt is markdown. A single '#' is left alone — "issue #12" is
        # ordinary text, while '##' is only ever trying to be a section.
        text = re.sub(r"#{2,}", "", text)
        text = " ".join(text.split())
        if not text:
            return ""
        return text[:limit]

    @staticmethod
    def _prompt_contract_block(context: dict[str, Any] | None) -> str:
        """Render user-facing route contracts as prompt-visible constraints.

        CP126 (critical): "Mind, runtime, style, and speech-frame values are
        converted directly to strings and rendered under a system-level
        response contract. Truncation does not prevent embedded newlines or
        instructions, and no schema or trusted producer is required."

        The rendering is ``- {item}`` under a ``## LIVE DESKTOP RESPONSE
        CONTRACT`` heading, so a value carrying a newline became a NEW BULLET
        in a system-level block — structurally indistinguishable from a
        constraint this code wrote. Truncating at 900 characters bounds the
        length of that forgery and nothing else.

        Every interpolated value now goes through ``_contract_safe``, which
        flattens the structure a value would need to impersonate
        one. This is not a claim to have solved prompt injection: a value
        can still say persuasive things. It can no longer say them *as a
        system constraint*, which is the specific escalation here.
        """

        if not isinstance(context, dict):
            return ""

        sections: list[str] = []
        mind_contract = InferenceGate._contract_safe(
            context.get("mind_context_contract"), 900
        )
        if mind_contract:
            sections.append(f"Mind-context contract: {mind_contract}")

        live_mind_context = context.get("live_mind_context")
        if isinstance(live_mind_context, dict):
            derived = live_mind_context.get("derived_runtime_context")
            if isinstance(derived, dict):
                prompt_block = InferenceGate._contract_safe(
                    derived.get("prompt_block"), 1200
                )
                if prompt_block:
                    sections.append(f"Derived runtime signals: {prompt_block}")

        style_contract = InferenceGate._contract_safe(
            context.get("response_style_contract"), 1400
        )
        if style_contract:
            sections.append(f"Response-style contract: {style_contract}")

        speech_frame = context.get("live_speech_grounding_frame")
        if isinstance(speech_frame, dict):
            frame_parts = []
            for key, value in speech_frame.items():
                if value in (None, "", [], {}):
                    continue
                safe_key = InferenceGate._contract_safe(key, 60)
                safe_value = InferenceGate._contract_safe(value, 180)
                if not safe_key or not safe_value:
                    continue
                frame_parts.append(f"{safe_key}={safe_value}")
                if len(frame_parts) >= 8:
                    break
            if frame_parts:
                sections.append("Speech grounding frame: " + " | ".join(frame_parts))
        elif speech_frame:
            flattened = InferenceGate._contract_safe(speech_frame, 900)
            if flattened:
                sections.append(f"Speech grounding frame: {flattened}")

        if not sections:
            return ""
        return "## LIVE DESKTOP RESPONSE CONTRACT\n" + "\n".join(f"- {item}" for item in sections)

    # How turn-volatile each live-mind section is. 0 = stable across a
    # conversation (identity, contracts, policy), 2 = changes every turn by
    # design (mood, tone, unity, somatic readings). Emission is sorted by this
    # so the cacheable prefix is as long as possible; Python's sort is stable,
    # so sections of equal volatility keep their priority order.
    _FOREGROUND_SECTION_VOLATILITY = FOREGROUND_SECTION_VOLATILITY

    @staticmethod
    def _foreground_section_volatility(section: str) -> int:
        text = str(section or "")
        for header, rank in InferenceGate._FOREGROUND_SECTION_VOLATILITY:
            if text.startswith(header):
                return rank
        return 1
    #: The sections a foreground turn is grounded by, kept whatever else
    #: is trimmed. One list, because the deep prompt builder trims against
    #: it too and two copies of a list like this drift apart quietly.
    CRITICAL_FOREGROUND_HEADERS = _CRITICAL_FOREGROUND_HEADERS


    @staticmethod
    def _critical_foreground_system_excerpt(content: str, *, budget: int) -> str:
        """Keep live-mind grounding visible inside compacted system prompts."""

        if budget <= 0:
            return ""
        important_headers = InferenceGate.CRITICAL_FOREGROUND_HEADERS
        sections: list[str] = []
        for header in important_headers:
            # At a LINE START only. Searching anywhere in the text meant a
            # header written INSIDE a sentence — in user-derived memory, a
            # fetched page, a tool result — promoted whatever followed it into
            # the excerpt that survives every budget trim. A real header is
            # always at the start of its line.
            start = 0 if content.startswith(header) else content.find("\n" + header)
            if start < 0:
                continue
            if start:
                start += 1
            if header.startswith("["):
                end_marker = "[END " + header.strip("[]") + "]"
                next_header = content.find(end_marker, start + len(header))
                if next_header >= 0:
                    end = next_header + len(end_marker)
                else:
                    next_header = content.find("\n[", start + len(header))
                    next_hash_header = content.find("\n## ", start + len(header))
                    candidates = [
                        idx for idx in (next_header, next_hash_header) if idx >= 0
                    ]
                    end = min(candidates) if candidates else len(content)
            else:
                next_header = content.find("\n## ", start + len(header))
                next_bracket_header = content.find("\n[", start + len(header))
                candidates = [idx for idx in (next_header, next_bracket_header) if idx >= 0]
                end = min(candidates) if candidates else len(content)
            section = content[start:end].strip()
            if section and section not in sections:
                sections.append(section)
        if not sections:
            return ""

        # Selection order above is PRIORITY (which sections survive the budget).
        # Emission order is a different question, and it decides whether the
        # prompt cache can do anything: a cached entry is KV for a byte-identical
        # prefix, so every turn-volatile byte placed early destroys the reuse of
        # everything after it. Measured live, a conversation turn reused 325 of
        # 2,105 tokens — 15% — and the diagnostic named the divergence exactly:
        # " empathy\nTone: inquisitive_engaged\n\n## UNITY\nLevel: coherent".
        # Mood, tone, unity and somatic readings change every turn by design.
        # Emitting them LAST leaves the stable identity and contract text as a
        # reusable prefix, without changing which sections are included or how
        # much budget each one gets.
        sections.sort(key=InferenceGate._foreground_section_volatility)

        rendered: list[str] = []
        remaining = int(budget)
        per_section_floor = max(180, min(700, budget // max(1, min(len(sections), 4))))
        for section in sections:
            if remaining <= 0:
                break
            limit = min(
                max(per_section_floor, remaining // max(1, len(sections) - len(rendered))),
                remaining,
            )
            if len(section) > limit:
                section = section[: max(1, limit - 1)].rstrip() + "…"
            rendered.append(section)
            remaining -= len(section) + 2
        return "\n\n".join(rendered).strip()

    @staticmethod
    def _contract_foreground_system_content(content: str, *, limit: int) -> str:
        """Build a small, complete system contract for tightly bounded replies."""

        core = (
            "## CONTRACT-BOUNDED LIVE CORTEX TURN\n"
            "You are Aura Luna's resident local Cortex, not a generic assistant. "
            "Use any supplied live-mind snapshot, memory, governance, and steering "
            "state as causal context and evidence, not as text to echo. Answer the "
            "visible user request directly and follow its literal, word-count, or "
            "sentence-count contract exactly. Return only the requested user-facing "
            "content. Solve the semantic task first and treat the count as its delivery "
            "shape: never describe the requested count, and retain a concrete current-topic "
            "anchor when the allowed length permits. Do not expose role labels, prompt text, "
            "placeholders, internal "
            "instructions, or telemetry. Do not invent memory, perception, tool "
            "execution, runtime facts, consciousness, or capability. Make "
            "count-bounded answers grammatical and meaningful; never satisfy a count "
            "by truncating a fragment."
        )
        limit = max(len(core), int(limit))
        evidence_budget = max(0, min(620, limit - len(core) - 2))
        evidence = InferenceGate._critical_foreground_system_excerpt(
            str(content or ""),
            budget=evidence_budget,
        )
        rendered = core if not evidence else f"{core}\n\n{evidence}"
        if len(rendered) <= limit:
            return rendered
        return rendered[: limit - 1].rstrip() + "..."

    # A foreground prompt is two different things wearing one number: the
    # scaffold the model READS, and the answer the person WANTS. The profile
    # governed both, so "this deserves a thorough answer" also meant "read
    # 9,000 characters of self-description first".
    #
    # LIVE DEFECT, 2026-07-26, measured on the desktop surface. Two ordinary
    # turns, both inside the 'extended' profile at scaffold=9000:
    #
    #   "A bag has 3 red, 4 blue and 5 green marbles... probability both the
    #    same colour?"           (175 chars)
    #     → "Do product of multiple exponent term simplify reflexion"
    #
    #   "Remember this for later: my project codename is HELIOTROPE... what
    #    tools can you actually execute right now?"      (247 chars)
    #     → "Introspection: Optimization-driven events stabilize energy after
    #        state change management... CONFORMANCE Signal: PRIORITY 0
    #        SEQUENCE SIGNATURE: [x_A_4521B_8A7C] Readiness State: FULL"
    #
    # The second is the diagnosis, not more noise: that is not a failed answer,
    # it is a competent CONTINUATION of the scaffold. A prompt that is 97%
    # self-description is most plausibly continued as more self-description.
    #
    # So the scaffold is now bounded by the size of the request that provoked
    # it, independently of how long the ANSWER may be. Trimming goes through
    # the existing head/critical-excerpt/tail path, so the identity anchor,
    # the live-mind grounding and the response contract all survive.
    _SCAFFOLD_TO_REQUEST_RATIO = 8
    _SCAFFOLD_FLOOR_CHARS = 2_400

    @classmethod
    def _proportionate_scaffold_limit(
        cls,
        profile_limit: int,
        visible_request_chars: int,
    ) -> int:
        """The system-block budget this request actually earns."""
        if visible_request_chars <= 0:
            return int(profile_limit)
        proportionate = visible_request_chars * cls._SCAFFOLD_TO_REQUEST_RATIO
        return max(
            cls._SCAFFOLD_FLOOR_CHARS,
            min(int(profile_limit), proportionate),
        )

    @staticmethod
    def _compact_prebuilt_message_content(
        role: str,
        content: Any,
        *,
        budget_profile: str = "standard",
        visible_request_chars: int = 0,
    ) -> str:
        clean = str(content or "").strip()
        if not clean:
            return ""
        context_window = InferenceGate._foreground_prompt_context_window()

        # Keep the live foreground lane fast: target the *runtime* context
        # window instead of the model family's theoretical max so prompt eval
        # does not balloon into 5k+ tokens on desktop.
        profile = str(budget_profile or "standard").lower()
        if profile == "contract":
            prompt_budget_chars = 2_800
            limits = {
                "system": 1_600,
                "user": 1_000,
                "assistant": 700,
            }
        elif profile == "contract_grounding":
            prompt_budget_chars = 1_000
            limits = {
                "system": 1_000,
                "user": 1_000,
                "assistant": 700,
            }
        elif profile == "state_report":
            prompt_budget_chars = 2_800
            limits = {
                "system": 1_800,
                "user": 1_000,
                "assistant": 500,
            }
        elif profile == "simple":
            prompt_budget_chars = min(
                9000,
                max(7000, int(max(4096, context_window - 1536) * 0.62)),
            )
            limits = {
                "system": min(5200, max(3800, int(prompt_budget_chars * 0.58))),
                "user": min(3200, max(1800, int(prompt_budget_chars * 0.36))),
                "assistant": min(1800, max(900, int(prompt_budget_chars * 0.20))),
            }
        elif profile == "deep_probe":
            prompt_budget_chars = 9000
            limits = {
                "system": 5200,
                "user": 3200,
                "assistant": 1600,
            }
        elif profile == "extended":
            prompt_budget_chars = max(18000, int(max(4096, context_window - 1536) * 1.75))
            limits = {
                "system": min(9000, max(6000, int(prompt_budget_chars * 0.40))),
                "user": min(14000, max(5000, int(prompt_budget_chars * 0.46))),
                "assistant": min(6000, max(3000, int(prompt_budget_chars * 0.20))),
            }
        elif profile == "curriculum":
            prompt_budget_chars = 12_000
            limits = {
                "system": 6_500,
                "user": 4_500,
                "assistant": 2_000,
            }
        elif profile == "background":
            prompt_budget_chars = 16_000
            limits = {
                "system": 9_000,
                "user": 5_000,
                "assistant": 2_500,
            }
        else:
            prompt_budget_chars = max(12000, int(max(4096, context_window - 1536) * 1.05))
            limits = {
                "system": min(6500, max(4500, int(prompt_budget_chars * 0.46))),
                "user": min(7000, max(3200, int(prompt_budget_chars * 0.42))),
                "assistant": min(3200, max(1600, int(prompt_budget_chars * 0.22))),
            }
        if role == "system" and profile not in {
            "contract",
            "contract_grounding",
            "state_report",
            "deep_probe",
        }:
            limits["system"] = InferenceGate._proportionate_scaffold_limit(
                limits["system"],
                visible_request_chars,
            )
        limit = limits.get(role, 8000)
        if profile == "contract" and role == "system":
            return InferenceGate._contract_foreground_system_content(
                clean,
                limit=limit,
            )
        if len(clean) <= limit:
            return clean
        if role in {"system", "user"}:
            marker = "\n…[middle omitted for foreground context budget]…\n"
            critical_excerpt = ""
            if role == "system":
                critical_excerpt = InferenceGate._critical_foreground_system_excerpt(
                    clean,
                    budget=min(2200, max(900, limit // 3)),
                )
            if critical_excerpt:
                remaining = max(2, limit - len(marker) * 2 - len(critical_excerpt))
                head = max(1, remaining * 3 // 5)
                tail = max(1, remaining - head)
                return (
                    f"{clean[:head].rstrip()}{marker}"
                    f"{critical_excerpt}{marker}"
                    f"{clean[-tail:].lstrip()}"
                )
            remaining = max(2, limit - len(marker))
            head = max(1, remaining * 2 // 3)
            tail = max(1, remaining - head)
            return f"{clean[:head].rstrip()}{marker}{clean[-tail:].lstrip()}"
        return clean[: limit - 1].rstrip() + "…"

    @classmethod
    def _stakes_capped_tokens(
        cls,
        max_tokens: int,
        *,
        envelope_cap: int,
        protected: bool,
        completion_floor: int = 0,
        prompt: str,
        context: dict[str, Any],
        reason: str,
    ) -> tuple[int, int]:
        """Apply the viability ceiling with a bounded user-surface override.

        Capability inventories and structurally compound desktop answers cannot
        be made cheaper by truncating them. The override is derived from this
        turn's measured compute profile and completion contract, so it remains
        finite and auditable. Critical memory admission is enforced separately
        before this method; an action-welfare envelope must not turn an admitted
        text decode into an incomplete answer.
        """
        ceiling = max(1, int(envelope_cap))
        if protected:
            floor, _cap, _loops = cls._foreground_compute_profile(str(prompt or ""))
            override = max(ceiling, int(floor), max(0, int(completion_floor)))
            if override > ceiling:
                context["resource_stakes_protected_override"] = {
                    "reason": reason,
                    "envelope_ceiling": ceiling,
                    "override_ceiling": override,
                    # Derived from the request, so the receipt can be checked.
                    "derived_from": (
                        "user_surface_completion_floor"
                        if int(completion_floor) > int(floor)
                        else "foreground_compute_profile_floor"
                    ),
                }
            ceiling = override
        return min(int(max_tokens), ceiling), ceiling

    #: Where a system block gets its middle removed. Same convention the
    #: per-message compactor uses, so a trimmed prompt reads the same wherever
    #: the trim happened.
    _PROMPT_FIT_MARKER = "\n…[middle omitted to fit the serving context window]…\n"

    def prompt_fit_receipt(self) -> dict[str, Any]:
        """What the last dispatch had to trim to fit the context window."""
        return copy.deepcopy(getattr(self, "_prompt_fit_receipt", {}))

    def _fit_prompt_to_window(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        answer_tokens: int,
        origin: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Last word on prompt size, denominated in tokens.

        Everything upstream budgets in CHARACTERS — profile limits, scaffold
        ratios, truncation — while the thing being budgeted is a context window
        measured in TOKENS. Four characters per token is roughly right for
        English prose and wrong for code, punctuation-dense text and non-Latin
        scripts, always in the direction that overflows. And prebuilt message
        payloads skipped the compactor entirely on several routes: the total
        was logged and never checked against anything.

        The person's own words are never trimmed here. System scaffold is,
        largest first, because the scaffold is what grew.
        """
        window = self._foreground_prompt_context_window()
        reserve = max(0, int(answer_tokens))
        allowed = window - reserve
        receipt: dict[str, Any] = {
            "window": window,
            "reserved_for_answer": reserve,
            "allowed": allowed,
            "trimmed": [],
            "origin": str(origin or ""),
        }

        def _cost(text: Any) -> int:
            return estimate_context_tokens(str(text or ""))

        def _total() -> int:
            return _cost(system_prompt) + sum(
                _cost(message.get("content")) for message in messages
            )

        total = _total()
        receipt["tokens_before"] = total
        if allowed <= 0 or total <= allowed:
            receipt["tokens_after"] = total
            receipt["fits"] = total <= allowed
            self._prompt_fit_receipt = receipt
            return system_prompt, messages

        # Trim system scaffold, largest first. Index -1 stands for the
        # separately-passed system_prompt, which the client merges into
        # messages[0] and which is therefore part of the same prefill.
        trimmable: list[tuple[int, int]] = []
        if system_prompt:
            trimmable.append((-1, _cost(system_prompt)))
        for index, message in enumerate(messages):
            if str(message.get("role", "")).strip().lower() not in {
                "system",
                "runtime_evidence",
            }:
                continue
            trimmable.append((index, _cost(message.get("content"))))
        trimmable.sort(key=lambda entry: entry[1], reverse=True)

        for index, cost in trimmable:
            overflow = _total() - allowed
            if overflow <= 0:
                break
            keep_tokens = max(0, cost - overflow)
            text = str(
                system_prompt if index < 0 else messages[index].get("content", "") or ""
            )
            if not text:
                continue
            # Tokens back to characters using this text's own measured ratio,
            # not a global assumption: a block of dense code and a block of
            # prose do not convert at the same rate.
            chars_per_token = len(text) / max(1, cost)
            keep_chars = int(keep_tokens * chars_per_token)
            marker = self._PROMPT_FIT_MARKER
            if keep_chars <= len(marker) + 2:
                trimmed = ""
            else:
                room = keep_chars - len(marker)
                head = max(1, room * 2 // 3)
                tail = max(1, room - head)
                trimmed = f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}"
            if index < 0:
                system_prompt = trimmed
            else:
                messages[index] = {**messages[index], "content": trimmed}
            receipt["trimmed"].append(
                {
                    "index": index,
                    "tokens_before": cost,
                    "tokens_after": _cost(trimmed),
                }
            )

        total = _total()
        receipt["tokens_after"] = total
        receipt["fits"] = total <= allowed
        if not receipt["fits"]:
            # Everything trimmable has been trimmed and it still does not fit,
            # which means the person's own words plus the answer budget exceed
            # the window. The serving runtime will truncate from one end and
            # answer a question it only partly received; say so rather than
            # letting it happen quietly.
            _record_inference_degradation(
                RuntimeError(
                    f"prompt does not fit the serving context window: "
                    f"{total} tokens against {allowed} allowed"
                ),
                action="dispatched an over-window prompt the serving runtime will truncate",
                severity="error",
                extra=receipt,
            )
        self._prompt_fit_receipt = receipt
        return system_prompt, messages

    #: What the grounding may spend on a turn that is not budget-constrained.
    #: Generous on purpose: every profile except the contract lane already has
    #: room, and this exists to bound pathology, not to shape normal turns.
    _GROUNDING_DEFAULT_BUDGET_CHARS = 6_000
    #: The clock and the receipts always survive, whatever else is dropped.
    _GROUNDING_FLOOR_CHARS = 600
    #: Current-condition turns reserve one bounded envelope for the stable
    #: identity prefix and one for the fresh state projection.
    _STATE_REPORT_TOTAL_BUDGET_CHARS = 4_200
    _STATE_REPORT_GROUNDING_BUDGET_CHARS = 1_400

    def _grounding_char_budget(self, context: Any, messages: Any) -> int:
        """How much room the volatile grounding has on this turn."""
        if isinstance(context, dict):
            visible = str(context.get("visible_user_message") or "")
            if self._foreground_prompt_profile(visible, context) == "state_report":
                used = sum(
                    len(str(msg.get("content", "") or ""))
                    for msg in (messages or ())
                    if isinstance(msg, dict)
                )
                available = max(
                    self._GROUNDING_FLOOR_CHARS,
                    self._STATE_REPORT_TOTAL_BUDGET_CHARS - used,
                )
                return min(self._STATE_REPORT_GROUNDING_BUDGET_CHARS, available)
        try:
            constrained = bool(self._has_short_live_output_contract(context))
        except _INFERENCE_RECOVERABLE_ERRORS:
            constrained = False
        if not constrained:
            return self._GROUNDING_DEFAULT_BUDGET_CHARS
        used = sum(
            len(str(msg.get("content", "") or ""))
            for msg in (messages or ())
            if isinstance(msg, dict)
        )
        return max(self._GROUNDING_FLOOR_CHARS, 2_800 - used)

    @staticmethod
    def _fit_grounding_blocks(
        *,
        contract_blocks: list[str],
        task_blocks: list[str],
        ambient_blocks: list[str],
        limit: int,
    ) -> str:
        """Fit complete evidence blocks using declared semantic priority.

        Call order is not authority. Turn contracts and task-specific evidence
        precede ambient state even when ambient collectors happen to run first.
        A trailing block is dropped whole because a partial readout can change
        the meaning of the evidence it carries.
        """
        kept: list[str] = []
        spent = 0
        ordered_blocks = [*contract_blocks, *task_blocks, *ambient_blocks]
        for block in ordered_blocks:
            text = str(block or "").strip()
            if not text:
                continue
            cost = len(text) + (2 if kept else 0)
            if kept and spent + cost > limit:
                continue
            kept.append(text)
            spent += cost
        if not kept:
            return ""
        joined = "\n\n".join(kept)
        # A single block larger than the whole allowance still has to fit.
        return joined if len(joined) <= limit else joined[: max(1, limit - 1)].rstrip()

    def _compact_prebuilt_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        history_limit: int = 12,
        deep_probe: bool = False,
        budget_profile: str = "standard",
        current_user_content: str | None = None,
    ) -> list[dict[str, str]]:
        """Trim oversized prebuilt chat payloads for the live 32B lane.

        Many callers already assemble messages upstream. For fast foreground turns,
        we keep the latest system prompt plus only the most recent compact dialogue
        snippets so first-turn Cortex doesn't spend tens of seconds re-reading old
        transcripts or giant contract blocks.
        """
        if not isinstance(messages, list):
            return []

        requested_profile = str(budget_profile or "standard").lower()
        profile = "deep_probe" if deep_probe else requested_profile
        latest_user_position = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], dict)
                and str(messages[idx].get("role", "") or "").strip().lower() == "user"
            ),
            None,
        )
        latest_user_content = ""
        if latest_user_position is not None:
            latest_user_content = str(
                messages[latest_user_position].get("content", "") or ""
            ).strip()
        contract_user_content = latest_user_content
        if requested_profile == "contract" and current_user_content:
            visible = str(current_user_content or "").strip()
            continuity_prefix = "[CURRENT USER MESSAGE]\n"
            internal_suffix_markers = (
                "\n\n[GROUNDING EVIDENCE FOR THIS TURN]\n",
                "\n\n[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n",
                "\n\n[LIVE DESKTOP FULL-MIND CONTRACT]\n",
                "\n\n[LIVE DESKTOP TURN EVIDENCE]\n",
            )

            def _visible_precedes_only_internal_suffix(candidate: str) -> bool:
                if candidate == visible:
                    return True
                if not visible or not candidate.startswith(visible):
                    return False
                suffix = candidate[len(visible) :]
                return any(suffix.startswith(marker) for marker in internal_suffix_markers)

            unwrapped_candidate = latest_user_content
            if latest_user_content.startswith(continuity_prefix):
                unwrapped_candidate = latest_user_content[len(continuity_prefix) :]
            marker_positions = [
                unwrapped_candidate.index(marker)
                for marker in internal_suffix_markers
                if marker in unwrapped_candidate
            ]
            if marker_positions:
                unwrapped_candidate = unwrapped_candidate[: min(marker_positions)].strip()
            if _visible_precedes_only_internal_suffix(unwrapped_candidate):
                contract_user_content = visible
        # The output contract survives a long input.
        #
        # A short output contract does not imply a short input, so a contract
        # turn whose user message runs long takes the standard INPUT budget.
        # But `profile` also selected the system builder, so the downgrade used
        # to drop `_contract_foreground_system_content` as well — and the
        # contract is the reason this route was chosen. A long question lost the
        # very output contract that routed it, which is the case where the
        # contract matters most.
        contract_output_profile = requested_profile == "contract"
        if profile == "contract":
            if len(contract_user_content) > 1_000:
                profile = "standard"
            else:
                latest_user_content = contract_user_content
        # The person's own words for this turn — never the user-role message,
        # which by this point also carries the grounding evidence the route
        # injected. Measured live: a 175-char question arrived as a 2,783-char
        # user block, so sizing the scaffold against the block would have
        # measured the scaffold against other scaffold.
        visible_request_chars = len(
            str(current_user_content or latest_user_content or "").strip()
        )
        system_message: dict[str, str] | None = None
        preserved_system_messages: list[dict[str, str]] = []
        convo: list[dict[str, str]] = []
        for message_position, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip().lower()
            grounding_system = bool(
                system_message is not None
                and role in {"system", "runtime_evidence"}
                and self._is_grounding_system_message(msg)
            )
            content_source = msg.get("content", "")
            if (
                requested_profile == "contract"
                and message_position == latest_user_position
                and latest_user_content
            ):
                content_source = latest_user_content
            if grounding_system and contract_output_profile:
                message_profile = "contract_grounding"
            elif role == "system" and contract_output_profile:
                # The system block keeps the contract builder even when the
                # input budget was widened above.
                message_profile = "contract"
            else:
                message_profile = profile
            content = self._compact_prebuilt_message_content(
                role,
                content_source,
                budget_profile=message_profile,
                visible_request_chars=visible_request_chars,
            )
            if not content:
                continue
            normalized = {
                "role": "runtime_evidence" if grounding_system else role or "user",
                "content": content,
            }
            if role == "system" and system_message is None:
                system_message = normalized
            elif grounding_system:
                preserved_system_messages.append(normalized)
            elif role in {"user", "assistant"}:
                convo.append(normalized)

        if deep_probe and system_message is not None:
            content = str(system_message.get("content", "") or "")
            if len(content) > 5200:
                system_message["content"] = content[:5199].rstrip() + "…"

        compact: list[dict[str, str]] = []
        if system_message is not None:
            compact.append(system_message)
        compact.extend(convo[-max(1, int(history_limit)) :])
        if not deep_probe and preserved_system_messages:
            # Grounding (LIVE MIND CONTEXT, phenomenal/body state, tool and skill
            # results) is rebuilt every turn. Placed AHEAD of the history it made
            # the prompt diverge at block two, so the reusable prefix ended after
            # the system message and the whole conversation was re-prefilled from
            # token zero on every turn — >80s to a first token by the time the
            # history was real, which is the deadline that produced "I couldn't
            # get to an answer I'd stand behind" (2026-07-26). Raising the KV
            # cache budget could never help: the entries had no stable prefix to
            # hit. Volatile content belongs last, so `system + history` stays
            # byte-identical across turns and the cache actually reuses it.
            #
            # It still lands immediately before the newest user message, so the
            # question is answered with the grounding in the most recent context.
            newest_user = next(
                (
                    idx
                    for idx in range(len(compact) - 1, -1, -1)
                    if compact[idx].get("role") == "user"
                ),
                None,
            )
            compact.insert(
                len(compact) if newest_user is None else newest_user,
                preserved_system_messages[-1],
            )

        context_window = self._foreground_prompt_context_window()
        if profile == "contract":
            total_budget_chars = 2_800
        elif profile == "state_report":
            # The remaining 1,400 characters are reserved for the canonical
            # state projection appended after stable-prefix compaction.
            total_budget_chars = 2_800
        elif profile == "simple":
            total_budget_chars = min(
                9000,
                max(7000, int(max(4096, context_window - 1536) * 0.62)),
            )
        elif profile == "extended":
            total_budget_chars = max(18000, int(max(4096, context_window - 1536) * 1.75))
        elif profile == "curriculum":
            total_budget_chars = 12_000
        elif profile == "background":
            total_budget_chars = 16_000
        else:
            total_budget_chars = max(12000, int(max(4096, context_window - 1536) * 1.05))
        if deep_probe:
            total_budget_chars = min(total_budget_chars, 9000)

        while (
            compact
            and sum(len(str(msg.get("content", "") or "")) for msg in compact) > total_budget_chars
        ):
            latest_user_index = next(
                (
                    idx
                    for idx in range(len(compact) - 1, -1, -1)
                    if compact[idx].get("role") == "user"
                ),
                None,
            )
            removable_index = None
            for idx, msg in enumerate(compact):
                if idx == 0 and msg.get("role") == "system":
                    continue
                if idx == latest_user_index:
                    continue
                if msg.get("role") == "assistant":
                    removable_index = idx
                    break
            if removable_index is None:
                for idx, msg in enumerate(compact):
                    if idx == 0 and msg.get("role") == "system":
                        continue
                    if idx == latest_user_index:
                        continue
                    if msg.get("role") != "user":
                        continue
                    removable_index = idx
                    break
            if removable_index is None:
                for idx, msg in enumerate(compact):
                    if idx == 0 and msg.get("role") == "system":
                        continue
                    if idx == latest_user_index:
                        continue
                    removable_index = idx
                    break
            if removable_index is None:
                break
            compact.pop(removable_index)

        total_chars = sum(len(str(msg.get("content", "") or "")) for msg in compact)
        if compact and total_chars > total_budget_chars:
            first = compact[0]
            if first.get("role") == "system":
                overflow = total_chars - total_budget_chars
                content = str(first.get("content", "") or "")
                if profile == "contract":
                    min_system_chars = 1_000
                elif profile == "state_report":
                    min_system_chars = 1_200
                else:
                    min_system_chars = 3200 if profile == "simple" else 4200
                new_limit = max(min_system_chars, len(content) - overflow - 1)
                if len(content) > new_limit:
                    first["content"] = self._compact_prebuilt_message_content(
                        "system",
                        content,
                        budget_profile=profile,
                    )
                    if len(first["content"]) > new_limit:
                        marker = "\n…[middle omitted for total prompt budget]…\n"
                        critical_excerpt = self._critical_foreground_system_excerpt(
                            content,
                            budget=min(2200, max(900, new_limit // 3)),
                        )
                        if critical_excerpt:
                            remaining = max(
                                2,
                                new_limit - len(marker) * 2 - len(critical_excerpt),
                            )
                            head = max(1, remaining * 3 // 5)
                            tail = max(1, remaining - head)
                            first["content"] = (
                                f"{content[:head].rstrip()}{marker}"
                                f"{critical_excerpt}{marker}"
                                f"{content[-tail:].lstrip()}"
                            )
                        else:
                            remaining = max(2, new_limit - len(marker))
                            head = max(1, remaining * 2 // 3)
                            tail = max(1, remaining - head)
                            first["content"] = (
                                f"{content[:head].rstrip()}{marker}{content[-tail:].lstrip()}"
                            )

        return compact


    async def generate(  # noqa: ASYNC109
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
        *,
        _generation_metadata_sink: dict[str, Any] | None = None,
    ) -> Any:
        """Bind request-scoped generation evidence around the direct endpoint."""

        sink_slot = self._generation_metadata_sink_slot()
        inherited_sink = sink_slot.get()
        bound_sink = (
            _generation_metadata_sink
            if isinstance(_generation_metadata_sink, dict)
            else inherited_sink
        )
        sink_token = sink_slot.set(bound_sink)
        try:
            return await self._generate_with_metadata_sink(
                prompt,
                context=context,
                timeout=timeout,
            )
        finally:
            sink_slot.reset(sink_token)

    async def _generate_with_metadata_sink(  # noqa: ASYNC109
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> Any:
        """Primary generation endpoint.

        [v7.4] Deadline-Aware Generation:
        Instead of fragmented local timers, we now use a unified Deadline object.
        """
        if context is None:
            context = {}
        self._clear_last_generation_metadata()
        initial_messages = context.get("messages")
        if not isinstance(initial_messages, list):
            initial_messages = None
        explicit_visible_user_prompt = str(
            context.get("user_surface_validation_prompt")
            or context.get("visible_user_message")
            or context.get("current_user_message")
            or ""
        ).strip()
        # The user's question is supplied, never inferred from the prompt.
        #
        # A chat turn hands this in — response generation passes
        # visible_user_message and user_surface_validation_prompt, and so does
        # every path that produces a visible reply. When nobody supplies one,
        # reading the last user-role message out of the envelope does not
        # recover it: it invents it, out of whatever text the caller happened
        # to send the model.
        #
        # LIVE 2026-08-19: deciding a move on a 2048 board, the deliberation's
        # own prompt became "the question". It carried a screen reading full of
        # numbers, so the reply was required to contain a total, and
        #
        #   "right — the board is mostly open on the right side, sliding right
        #    consolidates the smaller numbers and creates space"
        #
        # was rejected as arithmetic_answer_missing. Three retries, then no
        # text at all, and the pursuit reported she had named no move.
        #
        # The fallback survives where it is meaningful: an origin that really
        # is a person talking. Anywhere else an unsupplied question means
        # there is no user turn here to grade.
        # Read here rather than at its later assignment: this decision is
        # made before the binding, and the binding is the thing being guarded.
        derivable = self._origin_is_user_facing(
            str(context.get("origin", "") or "").lower()
        )
        initial_visible_user_prompt = explicit_visible_user_prompt or (
            self._visible_user_prompt_from_messages(initial_messages, prompt)
            if derivable
            else ""
        )
        surface_prompt = resolve_user_surface_prompt(
            context,
            fallback=initial_visible_user_prompt,
        )
        # The question THIS call is answering wins, always.
        #
        # LIVE DEFECT, 2026-08-10. Asked "look at my screen and tell me what
        # app is in front right now" (128 chars), the worker rejected every
        # draft with:
        #
        #   Rejected live user-surface draft reasons=arithmetic_answer_missing
        #       validation_chars=31 excerpt='50864799'
        #
        # 31 characters is "what's 7919 multiplied by 6421?" — the PREVIOUS
        # turn. A binding already present was treated as authoritative, so the
        # explicit visible message handed in for this call was ignored, and the
        # screen question was graded against the arithmetic question's expected
        # answer. It could not pass: there is no product in a reply about
        # windows. The turn died as arithmetic_answer_missing and the person
        # got a refusal about their screen.
        #
        # A binding survives to protect a turn from losing its own question
        # mid-flight. Carrying one INTO a different turn inverts that: it grades
        # every later answer against an older question, and the mismatch is
        # invisible because the reason names arithmetic while the user asked
        # about a window.
        stale_binding = bool(
            surface_prompt.bound
            and explicit_visible_user_prompt
            and str(surface_prompt.prompt or "").strip() != explicit_visible_user_prompt
        )
        # An internal generation has no user question to be graded against.
        #
        # Binding one anyway is where the harm starts: the prompt of a
        # deliberation becomes "the question", and every later check reads it
        # as something a person asked. A screen reading full of numbers then
        # looks like arithmetic, and a correct one-word move is rejected for
        # not containing a total.
        # Declared if the caller could say so, derived otherwise.
        #
        # The declaration travels through several client shapes and does not
        # survive all of them, so it is not the only evidence. A call that
        # supplied no user question AND whose origin is not a person talking
        # cannot be a reply to anybody — there is nothing it could be
        # answering. That is the same test used below to decide whether to
        # bind a question at all, and it is made from fields that reach here
        # on every path.
        internal_inference_call = bool(
            context.get("internal_inference", False)
            or context.get("_non_chat_inference", False)
            or (
                not explicit_visible_user_prompt
                and not self._origin_is_user_facing(
                    str(context.get("origin", "") or "").lower()
                )
            )
        )
        if internal_inference_call:
            context["internal_inference"] = True
        elif not initial_visible_user_prompt and not surface_prompt.bound:
            # Nothing to grade against, so nothing is bound. Binding an empty
            # prompt is what let the checks downstream fall back to reading
            # the model prompt again.
            pass
        elif not surface_prompt.bound or stale_binding:
            if stale_binding:
                logger.warning(
                    "🔗 Rebinding the user-surface validation prompt: the bound "
                    "one (%d chars) is not this turn's question (%d chars).",
                    len(str(surface_prompt.prompt or "")),
                    len(explicit_visible_user_prompt),
                )
            bind_user_surface_prompt(
                context,
                explicit_visible_user_prompt
                if stale_binding
                else (surface_prompt.prompt or initial_visible_user_prompt),
                source="inference_gate.visible_user_message",
                overwrite=True,
            )
            surface_prompt = resolve_user_surface_prompt(context)
        initial_visible_user_prompt = surface_prompt.prompt or initial_visible_user_prompt
        output_contract = requested_output_contract(initial_visible_user_prompt)
        output_contract_payload = (
            output_contract.as_dict() if output_contract.constrained else None
        )
        state = context.get("state")
        origin = str(context.get("origin", "") or "").lower()
        purpose = str(context.get("purpose", "") or "").lower()
        benchmark_request = bool(context.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        live_benchmark_request = origin == "benchmark" and not (
            purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        if benchmark_request:
            context["benchmark_request"] = True

        # Organism-first path: try to answer from the substrate+state without
        # invoking the LLM. This is bounded on purpose — the mesh handles only
        # self-reports, acknowledgements, and resource-gated responses. When it
        # does handle a request, the LLM is never called for that turn.
        proof_evaluation_contract = bool(context.get("proof_evaluation_contract", False)) or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        _seam_early_response = await _refuse_a_cold_protected_lane(
            benchmark_request=benchmark_request,
            context=context,
            initial_visible_user_prompt=initial_visible_user_prompt,
            origin=origin,
            output_contract=output_contract,
            output_contract_payload=output_contract_payload,
            proof_evaluation_contract=proof_evaluation_contract,
            self=self,
            state=state,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response

        health_probe = bool(context.get("health_probe", False)) or purpose == "proof_model_lane_probe"
        # A generation that is not the reply must be able to say so.
        #
        # LIVE 2026-08-19: deciding a move on screen answered "left", and the
        # user-surface gate rejected it as arithmetic_answer_missing — because
        # the request it was graded against was the deliberation's own prompt,
        # which mentions a 128 tile. Retries were exhausted, the model returned
        # nothing, and the pursuit reported that she named no available move.
        #
        # The conflation is one clause below: a call is treated as user-facing
        # when its origin says so OR it asked for the primary tier. Wanting the
        # good model is not the same as producing the visible answer. Internal
        # reasoning keeps the primary tier and the foreground lane; what it
        # stops inheriting is the contract that its output IS the reply.
        internal_inference = internal_inference_call
        proof_evaluation_contract = proof_evaluation_contract or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        if proof_evaluation_contract:
            context["proof_evaluation_contract"] = True
        operator_evidence_contract = bool(context.get("operator_evidence_contract", False))
        requested_tier = self._normalize_tier(context.get("prefer_tier"))
        explicit_background = "is_background" in context
        explicit_foreground = bool(context.get("foreground_request", False))
        # A planner that runs AS PART OF the turn in progress.
        #
        # LIVE, 2026-08-22: the finite-game solver asks the model to translate
        # the rules into a spec, and that call was refused —
        # "all_background_endpoints_deferred" — because the foreground turn it
        # was serving had reserved the lane. The turn then answered from the
        # model's own guess and got the strategy wrong.
        #
        # The flag alone would be an unauthenticated claim on the protected
        # lane, so it is honoured only when the orchestrator agrees a
        # foreground turn is actually running. Outside a turn it means
        # nothing and the request stays background.
        if not explicit_foreground and context.get("serves_current_turn"):
            explicit_foreground = self._a_user_turn_is_in_flight()
        protected_foreground_lane = bool(context.get("protected_foreground_lane", False))
        deep_probe_request = False
        try:
            from core.runtime.turn_analysis import looks_like_deep_mind_probe

            deep_probe_request = looks_like_deep_mind_probe(prompt)
        except _INFERENCE_RECOVERABLE_ERRORS:
            deep_probe_request = False
        if deep_probe_request and (explicit_foreground or self._origin_is_user_facing(origin)):
            if _FLAG_EMBODIED_CHALLENGE.value():
                logger.info(
                    "🛡️ InferenceGate: Suppressing deep-probe logic for Embodied Challenge priority."
                )
                deep_probe_request = False
            else:
                protected_foreground_lane = True
                context["deep_mind_probe"] = True
        is_background = bool(context.get("is_background", False))
        if explicit_foreground:
            is_background = False
        elif not is_background and not explicit_background:
            if origin:
                is_background = not self._origin_is_user_facing(origin)
            elif purpose in {"reply", "expression", "chat", "conversation", "user_response"}:
                is_background = False
            elif not explicit_background:
                # Origin-less requests are internal by default. User-facing turns
                # must carry an explicit origin such as api/user/voice.
                is_background = True
        deep_handoff = bool(context.get("deep_handoff", False))
        deep_reasoning_requested = bool(
            str(context.get("reasoning_mode") or "").strip().lower() == "deep"
            or str(context.get("serving_lane") or "").strip().lower() == "deep_reasoning"
            or deep_handoff
            or requested_tier == "secondary"
        )
        if deep_reasoning_requested:
            # Reasoning depth belongs to the turn contract, not to one model.
            # A specialist handoff may be inadmissible while the resident
            # cortex, RLC, verifiers, tools, and memory still execute the deep
            # systems lane. Preserve that mode across provider fallback.
            context["reasoning_mode"] = "deep"
        desktop_cognitive_engine_contract = bool(
            context.get("cognitive_engine_required", False)
            or context.get("desktop_cognitive_engine_required", False)
        )
        protected_compact_capability_contract = bool(
            context.get("capability_inventory_contract", False)
            and (
                desktop_cognitive_engine_contract
                or context.get("protected_foreground_lane", False)
                or explicit_foreground
            )
        )
        if requested_tier == "secondary":
            deep_handoff = True
        if deep_handoff and not local_deep_solver_enabled():
            # The lane does not exist on this host, so routing to it spends a
            # load admission that cannot be granted and comes back with
            # nothing. The router stopped registering the endpoint at boot;
            # this is the same fact reaching the other decision that can
            # select it. Measured live 2026-08-20: "Routing to Solver" on a
            # foreground chat turn, followed by "lane_budget_exceeded:solver
            # request 48.4GB + committed 25.3GB > budget 46.1GB" and an empty
            # generation, twice, ending in an apology.
            logger.debug(
                "Deep handoff requested where the deep solver cannot load; "
                "keeping this turn on the resident cortex."
            )
            deep_handoff = False
            if requested_tier == "secondary":
                requested_tier = "primary"
        if deep_handoff and not explicit_background:
            # Explicit deep handoffs are foreground reasoning requests even if
            # the caller forgot to stamp a user-facing origin.
            is_background = False
        strict_primary_proof_lane = False
        try:
            proof_run_enabled = env_str(
                "AURA_PROOF_RUN",
                description="Mark a hermetic proof runtime",
                owner="core.runtime.state_ownership",
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            origin_tokens = {token for token in origin.replace("-", "_").split("_") if token}
            proof_origin = bool(
                origin in {"test", "audit", "simulate", "external", "proof", "validation"}
                or origin_tokens & {"test", "audit", "simulate", "external", "proof", "validation"}
            )
            strict_primary_proof_lane = bool(
                context.get("proof_primary_lane_required", False)
                or live_benchmark_request
                or (
                    proof_run_enabled
                    and proof_model_tier() == "primary"
                    and (
                        proof_evaluation_contract
                        or health_probe
                        or proof_origin
                        or purpose.startswith("proof")
                    )
                )
            )
        except _INFERENCE_RECOVERABLE_ERRORS as _proof_policy_exc:
            # Fail CLOSED for proof routing: an explicit caller requirement
            # survives a policy-probe failure, and the failure goes on the
            # record — a silently disabled proof lane could later be mistaken
            # for a valid proof-lane result.
            strict_primary_proof_lane = bool(
                context.get("proof_primary_lane_required", False)
            )
            _record_inference_degradation(
                _proof_policy_exc,
                action="kept explicit proof-lane requirement after proof policy probe failed",
                severity="error",
            )
        if strict_primary_proof_lane:
            context["proof_primary_lane_required"] = True
            context["proof_model_tier"] = "primary"
            if live_benchmark_request:
                context["foreground_request"] = True
            requested_tier = "primary"
            deep_handoff = False
            is_background = False
            protected_foreground_lane = True
        if desktop_cognitive_engine_contract:
            context["desktop_cognitive_engine_required"] = True
            # The contract is about the ANSWER, not about every call made
            # while producing it.
            #
            # This forced `primary` for the whole turn, so an internal
            # tool-loop decision — picking one of eight labelled radio buttons
            # inside a browser pursuit — inherited the desktop reply's lane and
            # ran on the 32B at up to 103s a round. A sixty-item form died on
            # the turn budget having answered nothing, and because this runs
            # before the trust block, the explicit fast-lane request never even
            # reached the rule that would have honoured it.
            #
            # What the contract requires is that what she SAYS comes from the
            # real engine. An origin that is not user-facing is not that, and
            # keeps whatever lane it asked for.
            if self._origin_is_user_facing(origin):
                requested_tier = "primary"
                deep_handoff = False
                is_background = False
                protected_foreground_lane = True
        if is_background:
            requested_tier = "tertiary"
            deep_handoff = False
            background_deferral = self._background_local_deferral_reason(origin=origin)
            endpoint_deferrals: dict[str, str] = {}
            if not background_deferral:
                background_deferral, endpoint_deferrals = (
                    self._background_endpoint_headroom_deferral()
                )
            if background_deferral:
                if background_deferral == "memory_pressure":
                    logger.info(
                        "⏸️ InferenceGate: Deferring background inference for origin=%s due to memory pressure.",
                        origin,
                    )
                elif background_deferral == "foreground_headroom_reserved":
                    logger.info(
                        "⏸️ InferenceGate: Foreground headroom reserved. Deferring background inference for origin=%s.",
                        origin,
                    )
                elif background_deferral == "cortex_startup_quiet":
                    logger.info(
                        "⏸️ InferenceGate: Cortex quiet window active. Deferring background inference for origin=%s.",
                        origin,
                    )
                elif background_deferral == "foreground_quiet_window":
                    logger.info(
                        "⏸️ InferenceGate: Foreground quiet window active. Deferring background inference for origin=%s.",
                        origin,
                    )
                elif background_deferral == "desktop_background_disabled":
                    logger.info(
                        "⏸️ InferenceGate: Desktop background local LLM disabled. Deferring background inference for origin=%s.",
                        origin,
                    )
                else:
                    logger.info(
                        "⏸️ InferenceGate: Foreground lane reserved. Deferring background inference for origin=%s.",
                        origin,
                    )
                return self._refuse_generation(
                    self.REFUSAL_DEFERRED,
                    str(background_deferral or "foreground_lane_reserved"),
                    context=context,
                    origin=origin,
                    detail=(
                        {"endpoint_deferrals": endpoint_deferrals}
                        if endpoint_deferrals
                        else None
                    ),
                )

        if protected_foreground_lane and not is_background:
            # Global resource side effects, and the promotion that reaches them
            # can come from PROMPT TEXT: looks_like_deep_mind_probe is a
            # heuristic over what the person typed, and a match promotes the
            # turn to the protected lane. Unbudgeted, a run of probe-shaped
            # messages sheds background workers and extends the quiet window
            # once per message. Explicit callers — a contract that says
            # protected_foreground_lane — are not rate-limited; a promotion
            # inferred from text is.
            heuristic_promotion = bool(
                context.get("deep_mind_probe", False)
                and not context.get("protected_foreground_lane", False)
            )
            if heuristic_promotion and not self._admit_heuristic_protected_shed():
                logger.info(
                    "🛡️ Text-inferred protected lane: shed budget spent, "
                    "serving on the current lane state."
                )
            else:
                self._extend_startup_quiet_window(180.0)
                if not self._primary_lane_ready():
                    await self._shed_background_workers_for_memory_pressure(
                        force=True,
                        reason="protected_foreground_shed",
                    )

        # ── Morphogenesis routing advice ──────────────────────────────────
        # If the morphogenetic metabolism reports very high system pressure,
        # downgrade non-protected foreground requests from the heavy 32B
        # cortex to the lighter brainstem to avoid OOM/stall under load.
        if not is_background and not protected_foreground_lane and requested_tier != "tertiary":
            try:
                from core.morphogenesis.hooks import get_morphogenesis_routing_advice

                _morph_advice = get_morphogenesis_routing_advice()
                # [RESILIENCE] Only downgrade for genuinely critical pressure,
                # not routine background morphogenetic oscillations.
                if (
                    _morph_advice.get("recommend_downgrade", False)
                    and _morph_advice.get("pressure", 0.0) > 0.85
                ):
                    logger.info(
                        "🧬 Morphogenesis recommends tier downgrade: %s (pressure=%.2f)",
                        _morph_advice.get("reason", "unknown"),
                        _morph_advice.get("pressure", 0.0),
                    )
                    requested_tier = "tertiary"
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                logger.debug("Morphogenesis routing advice unavailable: %s", exc)

        # ── Proactive cortex recovery (laptop sleep / MLX worker death) ───
        _seam_early_response, requested_tier = await _recover_the_cortex_before_answering(
            context=context,
            is_background=is_background,
            origin=origin,
            protected_foreground_lane=protected_foreground_lane,
            requested_tier=requested_tier,
            self=self,
            strict_primary_proof_lane=strict_primary_proof_lane,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response

        # ── Trust gate: process message through trust engine ──────────────
        # PERF FIX: The trust gate calls UserRecognizer.recognize() which
        # runs PBKDF2-SHA256 (260K iterations) on every word/phrase in the
        # prompt to check for the owner passphrase.  This blocks the event
        # loop for 3-5+ seconds on large prompts.  Fix: offload to thread
        # pool, and skip entirely for background/autonomous requests.
        _trust_guidance = ""
        strict_proof_answer_request = (
            not benchmark_request and is_strict_proof_answer_prompt(prompt, origin=origin)
        )
        # Use the fully resolved routing classification, not merely whether the
        # caller explicitly stamped `is_background`. Origin-derived background
        # work such as `origin="system"` must not pay the foreground trust-gate
        # cost or get re-promoted back into the protected Cortex lane.
        _is_bg_request = bool(is_background)
        protected_foreground_lane, requested_tier = await _apply_strict_proof_answer_contract(
            _is_bg_request=_is_bg_request,
            context=context,
            deep_handoff=deep_handoff,
            deep_probe_request=deep_probe_request,
            origin=origin,
            prompt=prompt,
            protected_foreground_lane=protected_foreground_lane,
            requested_tier=requested_tier,
            state=state,
            strict_proof_answer_request=strict_proof_answer_request,
        )

        strict_answer_contract = bool(context.get("strict_answer_contract", False))
        strict_value_contract = bool(context.get("strict_value_contract", False))
        # A reply that must CARRY a document is not a conversational reply.
        #
        # LIVE, 2026-08-20. "build me a small web app… one self-contained
        # file" was answered with the page written into the reply, and the
        # reply stopped mid-attribute at `<script type=` — a 4096-token
        # default scaled to 970 by Phi control and pressure, which is a fair
        # size for prose and half an HTML page.
        #
        # The plan lane already has a floor for the same reason: a turn that
        # must emit a plan cannot be shrunk below the plan.
        document_output_contract = bool(
            context.get("document_output_contract", False)
        ) or _asks_for_a_document(initial_visible_user_prompt)
        if document_output_contract:
            context["document_output_contract"] = True
        web_interlocutor_contract = bool(context.get("web_interlocutor_contract", False))
        # Source code is not prose, and the conversational pipeline exists to
        # shape prose for a person: it repairs sentences, normalises
        # whitespace, and enforces a reply contract. Every one of those is
        # wrong for Python. Measured live 2026-07-28, the 2048 rules came back
        # through this path as
        #
        #   import randomdef move(case): board = case['board'] direction = ...
        #
        # — newlines gone, indentation collapsed, and therefore
        # "invalid syntax at line 1" on a generation the model had written
        # correctly. Code generation now declares itself isolated, like every
        # other non-conversational contract here.
        code_generation_contract = bool(context.get("code_generation_contract", False))
        isolated_generation_contract = bool(
            strict_answer_contract
            or strict_value_contract
            or proof_evaluation_contract
            or operator_evidence_contract
            or web_interlocutor_contract
            or code_generation_contract
        )
        # Sealed proof prompts (<answer> envelope) get a micro budget so a
        # one-word answer cannot ramble; a caller-pinned max_tokens always
        # wins. But the contract also reaches structured proof requests
        # (e.g. the repair loop asking for a full replacement file as
        # JSON) — for those, an unconditional 128 default truncated every
        # generation mid-JSON. Unpinned non-envelope requests now keep the
        # budget computed below instead of collapsing to 128.
        strict_max_token_cap: int | None = 128
        if strict_answer_contract:
            try:
                explicit_cap = context.get("max_tokens")
                if explicit_cap:
                    strict_max_token_cap = max(1, int(explicit_cap))
                elif strict_proof_answer_request:
                    strict_max_token_cap = 128
                else:
                    strict_max_token_cap = None
            except (TypeError, ValueError, OverflowError):
                strict_max_token_cap = 128

        if not is_background and requested_tier == "secondary":
            local_deep_block = self._local_deep_solver_block_reason()
            if local_deep_block:
                logger.warning(
                    "🛡️ InferenceGate: local 70B Solver handoff blocked (%s). Staying on Cortex.",
                    local_deep_block,
                )
                context["local_deep_block_reason"] = local_deep_block
                requested_tier = "primary"
                deep_handoff = False

        timeout_val = self._requested_timeout_s(
            timeout,
            self._default_timeout_for_request(
                origin,
                requested_tier,
                deep_handoff=deep_handoff,
                is_background=is_background,
            ),
        )
        primary_timeout, fallback_timeout = self._split_attempt_timeouts(
            timeout_val, requested_tier
        )
        lower_local_lane_forbidden = bool(
            proof_evaluation_contract
            or strict_primary_proof_lane
            or operator_evidence_contract
            or desktop_cognitive_engine_contract
            or health_probe
        )
        if requested_tier == "primary" and lower_local_lane_forbidden:
            # This contract refuses every lower lane. Reserving 15-40% for a
            # fallback that is forbidden shortened the only admissible 32B
            # attempt, then failed closed after spending the reserved time on
            # nothing. Keep a small delivery margin and give the real lane the
            # rest; EOS still returns immediately.
            primary_timeout = max(8.0, timeout_val - 4.0)
            fallback_timeout = min(4.0, timeout_val)
        # ONE clock for the whole request.
        #
        # Every attempt below used to start a fresh timeout of its own: the
        # primary attempt, then each scheduled repair at 30–60s, then the
        # brainstem, then the reflex, then APIAdapter at 30s, then HealthRouter
        # at another 30s. A caller asking for 45 seconds could wait several
        # minutes and every individual wait_for was "within budget". This
        # deadline is the budget the caller actually asked for, and every
        # window below is capped by what is left of it.
        request_deadline = get_deadline(float(timeout_val))
        context["request_deadline_s"] = float(timeout_val)
        max_tokens = self._requested_max_tokens(
            context.get("max_tokens"),
            self._default_max_tokens_for_request(
                origin,
                requested_tier,
                deep_handoff=deep_handoff,
                is_background=is_background,
            ),
        )
        explicit_max_tokens_cap: int | None = None
        if "max_tokens" in context:
            try:
                explicit_max_tokens_cap = max(1, int(context.get("max_tokens") or 1))
            except (TypeError, ValueError, OverflowError):
                explicit_max_tokens_cap = None
        surface_completion_floor = 0
        # How much room an answer needs is a property of the question, not of
        # the path that happens to serve it.
        #
        # LIVE, 2026-08-27: a question that had to be worked out was routed to
        # the deliberate lane BECAUSE it was hard, and that lane carried no
        # desktop contract, so the floor did not apply and the model was
        # dispatched with 128 tokens. The quick lane, for the same question,
        # got 896. Choosing the right lane made the budget worse.
        #
        # The population is the one the sampling multiplier below already acts
        # on: user-facing foreground generations. One gate lowers the budget
        # and one floors it, and they now cover the same turns. A declared
        # output ceiling and a blocked resource stake still win, as before.
        _foreground_answer_turn = (
            not is_background
            and self._origin_is_user_facing(origin)
            and not isolated_generation_contract
            and not health_probe
            and not benchmark_request
            and not proof_evaluation_contract
            and not strict_answer_contract
        )
        if (
            (desktop_cognitive_engine_contract or _foreground_answer_turn)
            and not bool(context.get("hard_output_token_ceiling", False))
            and not bool(context.get("resource_stakes_blocked", False))
        ):
            try:
                surface_completion_floor = max(
                    1,
                    int(
                        context.get("user_surface_completion_floor")
                        or answer_surface_token_floor(initial_visible_user_prompt)
                    ),
                )
            except (TypeError, ValueError, OverflowError):
                surface_completion_floor = answer_surface_token_floor(
                    initial_visible_user_prompt
                )
            context["user_surface_completion_floor"] = surface_completion_floor
            if max_tokens < surface_completion_floor:
                logger.info(
                    "🧠 Foreground completion contract raised the decode budget %d→%d.",
                    max_tokens,
                    surface_completion_floor,
                )
                max_tokens = surface_completion_floor
            if explicit_max_tokens_cap is not None:
                explicit_max_tokens_cap = max(
                    explicit_max_tokens_cap,
                    surface_completion_floor,
                )
                context["max_tokens"] = explicit_max_tokens_cap
        if "max_tokens" not in context:
            max_tokens = self._adaptive_max_tokens_for_prompt(
                initial_visible_user_prompt,
                base_tokens=max_tokens,
                origin=origin,
                requested_tier=requested_tier,
                is_background=is_background,
            )
        # When the 32B cortex is still warming or recovering, refuse to load
        # the 72B Solver alongside it — they don't fit in 64GB together and
        # the resulting MemoryGuard panic-eviction creates a thrash loop where
        # neither lane stays up long enough to answer. Force primary; the
        # cortex will handle the turn when warmup finishes.
        if not is_background and requested_tier == "secondary" and not protected_foreground_lane:
            try:
                _cortex_lane = self.get_conversation_status() or {}
                _cortex_state = str(_cortex_lane.get("state", "") or "").lower()
                if _cortex_state in {"warming", "handshaking", "recovering"}:
                    logger.info(
                        "🛡️ InferenceGate: cortex is %s; refusing secondary handoff to avoid "
                        "%s/Solver memory thrash. Staying on primary.",
                        _cortex_state,
                        _primary_lane_label(),
                    )
                    requested_tier = "primary"
                    deep_handoff = False
            except _INFERENCE_RECOVERABLE_ERRORS as _swap_exc:
                record_degradation(
                    "inference_gate",
                    _swap_exc,
                    severity="warning",
                    action="failed safe to primary after secondary coexistence probe failed",
                )
                logger.debug("Cortex lane probe before secondary admission failed: %s", _swap_exc)
                requested_tier = "primary"
                deep_handoff = False

        admission_snapshot: dict[str, Any] | None = None
        _seam_early_response, deep_handoff, fallback_timeout, max_tokens, primary_timeout, request_deadline, requested_tier, timeout_val = await _admit_the_foreground_request(
            context=context,
            deep_handoff=deep_handoff,
            desktop_cognitive_engine_contract=desktop_cognitive_engine_contract,
            fallback_timeout=fallback_timeout,
            initial_visible_user_prompt=initial_visible_user_prompt,
            is_background=is_background,
            max_tokens=max_tokens,
            origin=origin,
            primary_timeout=primary_timeout,
            protected_foreground_lane=protected_foreground_lane,
            request_deadline=request_deadline,
            requested_tier=requested_tier,
            self=self,
            surface_completion_floor=surface_completion_floor,
            timeout=timeout,
            timeout_val=timeout_val,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response

        # ── Resource Stakes: scale token budget by computational survival state ──
        try:
            from core.consciousness.resource_stakes import get_resource_stakes

            token_mult = get_resource_stakes().get_token_budget_multiplier()
            if (
                token_mult < 0.95
                and not strict_answer_contract
                and not health_probe
                and not isolated_generation_contract
                and not benchmark_request
            ):
                max_tokens = max(384, int(max_tokens * token_mult))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="kept default token budget multiplier",
            )
            logger.debug("Resource-stakes token multiplier unavailable: %s", exc)

        # ── Operational Resource Stakes: persistent viability constrains action ──
        # This newer ledger is stricter than the legacy multiplier above: it can
        # downgrade the large-model lane and hard-cap output when viability drops.
        stakes_token_ceiling: int | None = None
        try:
            from core.container import ServiceContainer

            stakes = ServiceContainer.get("resource_stakes", default=None)
            deep_handoff, max_tokens, requested_tier, stakes_token_ceiling = _settle_the_token_ceilings(
                context=context,
                deep_handoff=deep_handoff,
                desktop_cognitive_engine_contract=desktop_cognitive_engine_contract,
                max_tokens=max_tokens,
                prompt=prompt,
                protected_compact_capability_contract=protected_compact_capability_contract,
                requested_tier=requested_tier,
                self=self,
                stakes=stakes,
                stakes_token_ceiling=stakes_token_ceiling,
                surface_completion_floor=surface_completion_floor,
            )
        except _INFERENCE_RECOVERABLE_ERRORS as _stakes_exc:
            record_degradation(
                "inference_gate",
                _stakes_exc,
                severity="warning",
                action="kept default resource-stakes action envelope",
            )
            logger.debug("ResourceStakesLedger unavailable: %s", _stakes_exc)

        # ── Phi (Integrated Information): scale token budget based on cognitive integration ──
        # [STABILITY v59] NEVER throttle user-facing foreground requests.
        # PHI is near-zero during early boot (insufficient IIT data), which
        # was crushing max_tokens to ~420 on the first few user turns —
        # making desktop responses catastrophically worse than server mode.
        # PHI scaling is now restricted to background requests only, and
        # even then the floor is 0.6x instead of 0.2x.
        _is_user_facing_for_phi = bool(
            not is_background
            and (explicit_foreground or protected_foreground_lane or self._origin_is_user_facing(origin))
        )
        if not _is_user_facing_for_phi:
            try:
                from core.container import ServiceContainer
                phi_val = 1.0  # default
                phi_is_measured = False
                phi_core = ServiceContainer.get("phi_core", default=None)
                if phi_core is not None:
                    if hasattr(phi_core, "get_live_phi"):
                        # include_surrogate=True means this number may be a
                        # PROXY, not an exact-MIP integrated-information
                        # measurement. It still scales the BACKGROUND token
                        # budget — which is a defensible use of a rough
                        # signal — but it must not be recorded as Φ, and the
                        # foreground lane is already excluded above.
                        phi_val = max(
                            0.0,
                            _finite(
                                phi_core.get_live_phi(include_surrogate=True), 1.0
                            )
                            or 1.0,
                        )
                        phi_is_measured = False
                    elif hasattr(phi_core, "_last_result") and phi_core._last_result:
                        phi_val = max(
                            0.0, _finite(phi_core._last_result.phi_s, 1.0) or 1.0
                        )
                        phi_is_measured = True
                context["background_budget_signal"] = {
                    "value": round(float(phi_val), 4),
                    # The name of the thing, not the name of the ideal.
                    "kind": "phi_measured" if phi_is_measured else "phi_surrogate",
                    "scales": "background_token_budget",
                }
                
                # Scale token budget for background requests only:
                # When Φ is high, allow full budget. When Φ is low, scale down
                # but never below 60% — the old 20% floor was destructive.
                #
                # "Background only" is what this always said and never did.
                # There was no background check in the condition, so a
                # background budget control was trimming the answers people
                # were waiting for, and the signal it scales on is registered
                # under the name "background_token_budget" a dozen lines above.
                #
                # It matters more than a percentage looks. A token budget is a
                # ceiling and not a reservation: the model stops when it has
                # finished, so a generous ceiling costs nothing on a turn that
                # ends early, while a tight one costs the end of a sentence on
                # the turn that needed the room. Guessing low and guessing high
                # are not symmetric, and on one laptop serving one person
                # there is nothing on the other side of the trade.
                if (
                    phi_val < 0.8
                    and is_background
                    and not strict_answer_contract
                    and not health_probe
                    and not isolated_generation_contract
                    and not benchmark_request
                    and not document_output_contract
                ):
                    phi_scale = max(0.6, 0.6 + 0.4 * (phi_val / 0.8))
                    max_tokens = max(512, int(max_tokens * phi_scale))
                    logger.info("🧠 [PHI CONTROL] Integration Φ=%.3f -> scaling token budget by %.2f (max_tokens=%d)", 
                                phi_val, phi_scale, max_tokens)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="kept unscaled token budget after phi token-budget probe failed",
                    severity="debug",
                )
                logger.debug("Phi token budget scaling skipped: %s", exc)

        # ── Affective Circumplex: let somatic state modulate generation params ──
        # Only applies on user-facing, non-background requests. Background tasks
        # run at fixed params to avoid thermal feedback loops.
        somatic_temperature: float | None = None
        morpho_kwargs: dict[str, Any] = {}
        caller_temperature = context.get("temperature", context.get("temp"))
        if caller_temperature is not None:
            try:
                _caller_temp = float(caller_temperature)
                # NaN slides through min/max to the 2.0 ceiling — reject
                # non-finite values instead of maxing out sampling entropy.
                somatic_temperature = (
                    max(0.0, min(2.0, _caller_temp))
                    if math.isfinite(_caller_temp)
                    else None
                )
            except (TypeError, ValueError):
                somatic_temperature = None
        for _gen_key in (
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "repetition_context_size",
            "presence_penalty",
            "stop_sequences",
            "schema",
            "benchmark_request",
            "purpose",
            "cognitive_mode",
            "strict_answer_contract",
            "strict_value_contract",
            "proof_evaluation_contract",
            "operator_evidence_contract",
            "web_interlocutor_contract",
            "runtime_fact_status_contract",
            "grounded_runtime_status_contract",
            "clean_user_surface_contract",
            "user_surface_completion_floor",
            "user_surface_validation_prompt",
            "semantic_completion_contract",
            "user_surface_continuation_contract",
            "user_surface_continuation_partial",
            "user_surface_continuation_resume_handle",
            "user_surface_conversation_resume_handle",
            "user_surface_prompt_binding",
            "clean_user_surface_steering_alpha",
            "clean_user_surface_recurrent_loops",
            "live_mind_controls_bound",
            "live_mind_generation_controls",
            "live_mind_snapshot_ready",
            "live_mind_required_subsystems_ok",
            "disable_prompt_cache",
            "clear_prompt_cache",
            "health_probe",
        ):
            if _gen_key in context:
                morpho_kwargs[_gen_key] = context[_gen_key]
        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and not isolated_generation_contract
        ):
            try:
                from core.affect.affective_circumplex import get_circumplex
                from core.verify import influence_channels
                from core.verify.lesion_registry import apply_channel

                circumplex_params = get_circumplex().get_llm_params()
                # Wrapped so a paired trial can run this exact code with the
                # affect contribution removed. This is the largest direct
                # actuation in the system — the circumplex moves temperature
                # across 0.500..0.858 and the token budget across 472..768 —
                # and it was the only live actuator with no lesion, so the one
                # faculty with a visibly large effect was the one the influence
                # apparatus could not ask about. Neutral is "no affective
                # modulation": the caller's own budget and the default
                # temperature, which is what this block would produce if the
                # circumplex were flat.
                if not context.get("max_tokens"):
                    max_tokens = max(
                        384,
                        min(
                            max_tokens,
                            int(
                                apply_channel(
                                    influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
                                    circumplex_params["max_tokens"],
                                    neutral=max_tokens,
                                )
                            ),
                        ),
                    )
                somatic_temperature = apply_channel(
                    influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
                    circumplex_params["temperature"],
                    neutral=None,
                )
                logger.debug(
                    "💓 Circumplex: V=%.2f A=%.2f → temp=%s tokens=%d",
                    circumplex_params["valence"],
                    circumplex_params["arousal"],
                    # %s, not %.2f: under a paired trial this channel is
                    # lesioned to None and a float format would raise inside
                    # the logging call.
                    "lesioned"
                    if somatic_temperature is None
                    else f"{somatic_temperature:.2f}",
                    max_tokens,
                )
            except _INFERENCE_RECOVERABLE_ERRORS as _ce:
                record_degradation(
                    "inference_gate",
                    _ce,
                    severity="warning",
                    action="kept default sampling parameters without affective circumplex",
                )
                logger.debug("Circumplex unavailable: %s", _ce)

            # ── PNEUMA precision sampler: blend with circumplex temperature ──
            try:
                from core.consciousness.precision_sampler import get_active_inference_sampler

                _ais_params = get_active_inference_sampler().get_sampling_params()
                ais_temp = _ais_params.get("temperature")
                if ais_temp is not None:
                    # Blend: 50% circumplex + 50% PNEUMA precision
                    base = somatic_temperature if somatic_temperature is not None else 0.72
                    somatic_temperature = round(0.5 * base + 0.5 * ais_temp, 3)
                    logger.debug("🎯 PNEUMA precision temp blend → %.3f", somatic_temperature)
            except _INFERENCE_RECOVERABLE_ERRORS as _ais_e:
                record_degradation(
                    "inference_gate",
                    _ais_e,
                    severity="warning",
                    action="kept existing sampling temperature without active-inference blend",
                )
                logger.debug("ActiveInferenceSampler unavailable: %s", _ais_e)

            # ── Homeostatic Coupling: Apply cognitive modifiers to generation ──
            # These are computed every heartbeat tick from drives + affect + hardware.
            # temperature_mod: integrity/sovereignty stress → more cautious (lower temp)
            # depth_mod: energy depletion → fewer tokens; high energy → more
            # creativity_mod: curiosity-driven exploration width
            try:
                _homeo_coupling = ServiceContainer.get("homeostatic_coupling", default=None)
                if _homeo_coupling:
                    _mods = _homeo_coupling.get_modifiers()
                    _temp_factor = self._modulator_factor(
                        _mods.temperature_mod,
                        source="homeostatic_coupling.temperature_mod",
                        low=0.5,
                        high=1.5,
                    )
                    _depth_factor = self._modulator_factor(
                        _mods.depth_mod,
                        source="homeostatic_coupling.depth_mod",
                        low=0.5,
                        high=2.0,
                    )
                    if somatic_temperature is not None:
                        somatic_temperature = round(somatic_temperature * _temp_factor, 3)
                    max_tokens = max(384, int(max_tokens * _depth_factor))
                    logger.debug(
                        "🫀 HomeostaticCoupling: temp_mod=%.2f depth_mod=%.2f → temp=%.3f tokens=%d",
                        _mods.temperature_mod,
                        _mods.depth_mod,
                        somatic_temperature or 0.0,
                        max_tokens,
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _hc_e:
                record_degradation(
                    "inference_gate",
                    _hc_e,
                    severity="warning",
                    action="kept existing generation parameters without homeostatic coupling",
                )
                logger.debug("HomeostaticCoupling modifiers unavailable: %s", _hc_e)

            # ── Homeostasis Engine: Direct drive-based inference modulation ──
            # Integrity/sovereignty danger → lower temperature (caution)
            # Low metabolism → fewer tokens (conserve)
            # High curiosity → slight temp boost (exploration)
            try:
                _homeostasis = ServiceContainer.get("homeostasis", default=None)
                if _homeostasis and hasattr(_homeostasis, "get_inference_modifiers"):
                    _h_mods = _homeostasis.get_inference_modifiers()
                    if somatic_temperature is not None:
                        somatic_temperature = round(
                            somatic_temperature
                            + self._modulator_delta(
                                _h_mods["temperature_mod"],
                                source="homeostasis.temperature_mod",
                                limit=0.5,
                            ),
                            3,
                        )
                        somatic_temperature = max(0.1, min(1.5, somatic_temperature))
                    max_tokens = max(
                        384,
                        int(
                            max_tokens
                            * self._modulator_factor(
                                _h_mods["token_multiplier"],
                                source="homeostasis.token_multiplier",
                                low=0.5,
                                high=2.0,
                            )
                        ),
                    )
                    logger.debug(
                        "🫀 Homeostasis: temp_mod=%+.3f token_mult=%.2f caution=%.2f",
                        _h_mods["temperature_mod"],
                        _h_mods["token_multiplier"],
                        _h_mods["caution_level"],
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _he_e:
                record_degradation(
                    "inference_gate",
                    _he_e,
                    severity="warning",
                    action="kept existing generation parameters without homeostasis modifiers",
                )
                logger.debug("Homeostasis inference modifiers unavailable: %s", _he_e)

            # ── Morphogenetic substrate → sampling parameters ────────────────
            # What this does: reads the morphogenetic field's danger, curiosity
            # and resource-pressure scalars and moves temperature, top_p and
            # the repetition penalty. That is a real causal path from substrate
            # state to output distribution, and it is worth having.
            #
            # What it is NOT: "curing mind-body dualism" or "true embodied
            # cognition", which is what this comment used to claim. Nothing
            # here establishes embodiment; it reads three numbers out of a
            # service and scales three sampler knobs. The claim outran the
            # code, and a claim about Aura with no test behind it is the thing
            # this pass exists to remove (core/organism/model_validation.py).
            try:
                from core.container import ServiceContainer

                _rt = ServiceContainer.get("morphogenetic_runtime", default=None)
                max_tokens, somatic_temperature = _modulate_sampling_from_the_body(
                    ServiceContainer=ServiceContainer,
                    _rt=_rt,
                    context=context,
                    explicit_foreground=explicit_foreground,
                    is_background=is_background,
                    max_tokens=max_tokens,
                    morpho_kwargs=morpho_kwargs,
                    protected_compact_capability_contract=protected_compact_capability_contract,
                    protected_foreground_lane=protected_foreground_lane,
                    self=self,
                    somatic_temperature=somatic_temperature,
                )
            except _INFERENCE_RECOVERABLE_ERRORS as _m_e:
                record_degradation(
                    "inference_gate",
                    _m_e,
                    severity="warning",
                    action="continued without morphogenetic generation-parameter coupling",
                )
                logger.debug("Morphogenetic coupling unavailable: %s", _m_e)

            # ── Synaptic Plasticity: Learned generation-style modulation ──
            # The projection matrix was updated after previous inferences via
            # reward-modulated Hebbian learning. Now it transforms the current
            # substrate state into sampling parameter adjustments.
            try:
                _plasticity = ServiceContainer.get("synaptic_plasticity", default=None)
                if _plasticity is not None:
                    _substrate = ServiceContainer.get("conscious_substrate", default=None)
                    if _substrate is not None and hasattr(_substrate, "x"):
                        import numpy as _np_plast
                        _sub_state = _np_plast.asarray(_substrate.x, dtype=_np_plast.float32)
                        _plast_mod = _plasticity.compute_modulation(_sub_state)
                        if _plast_mod:
                            _p_temp_d = _plast_mod.get("temperature_delta", 0.0)
                            _p_topp_d = _plast_mod.get("top_p_delta", 0.0)
                            _p_rep_d = _plast_mod.get("repetition_penalty_delta", 0.0)
                            if somatic_temperature is not None:
                                somatic_temperature = max(0.1, min(1.5, somatic_temperature + _p_temp_d))
                            else:
                                somatic_temperature = max(0.1, min(1.5, 0.72 + _p_temp_d))
                            if "top_p" in morpho_kwargs:
                                morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _p_topp_d))
                            if "repetition_penalty" in morpho_kwargs:
                                morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _p_rep_d))
                            logger.debug(
                                "🧬 SynapticPlasticity: temp_d=%.3f topp_d=%.3f rep_d=%.3f",
                                _p_temp_d, _p_topp_d, _p_rep_d,
                            )
                        # Pre-inference capture for post-inference learning
                        _hedonic = 0.0
                        try:
                            from core.consciousness.hedonic_gradient import get_hedonic_gradient
                            _hedonic = get_hedonic_gradient().score
                        except _INFERENCE_RECOVERABLE_ERRORS as _hedonic_exc:
                            record_degradation(
                                "inference_gate",
                                _hedonic_exc,
                                severity="warning",
                                action="continued synaptic plasticity capture without hedonic score",
                            )
                            logger.debug(
                                "SynapticPlasticity hedonic capture unavailable: %s",
                                _hedonic_exc,
                            )
                        _plasticity.pre_inference_capture(_sub_state, _hedonic)
            except _INFERENCE_RECOVERABLE_ERRORS as _sp_e:
                record_degradation(
                    "inference_gate",
                    _sp_e,
                    severity="warning",
                    action="continued without synaptic plasticity generation modulation",
                )
                logger.debug("SynapticPlasticity coupling unavailable: %s", _sp_e)

            # ── Temporal Continuity: Silence-accumulated modulation ──
            # The temporal residue from accumulated silence directly adjusts
            # generation parameters — the system speaks differently after long
            # silences because real drift accumulated.
            try:
                _tc = ServiceContainer.get("temporal_continuity", default=None)
                if _tc is not None:
                    _tc.on_inference_start()
                    _tc_mod = _tc.compute_modulation()
                    if _tc_mod:
                        _tc_temp_d = self._modulator_delta(
                            _tc_mod.get("temperature_delta", 0.0),
                            source="temporal_continuity.temperature_delta",
                            limit=0.5,
                        )
                        _tc_topp_d = _tc_mod.get("top_p_delta", 0.0)
                        _tc_rep_d = _tc_mod.get("repetition_penalty_delta", 0.0)
                        _tc_token_mult = _tc_mod.get("token_budget_multiplier", 1.0)
                        if somatic_temperature is not None:
                            somatic_temperature = max(0.1, min(1.5, somatic_temperature + _tc_temp_d))
                        if _tc_topp_d and "top_p" in morpho_kwargs:
                            morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _tc_topp_d))
                        if _tc_rep_d and "repetition_penalty" in morpho_kwargs:
                            morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _tc_rep_d))
                        if _tc_token_mult > 1.0:
                            max_tokens = int(min(max_tokens * _tc_token_mult, 4096))
                        logger.debug(
                            "🕐 TemporalContinuity: temp_d=%.3f token_mult=%.2f",
                            _tc_temp_d, _tc_token_mult,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as _tc_e:
                record_degradation(
                    "inference_gate",
                    _tc_e,
                    severity="warning",
                    action="continued without temporal continuity generation modulation",
                )
                logger.debug("TemporalContinuity coupling unavailable: %s", _tc_e)

            # ── Somatic qualia service → sampler perturbations ──
            # Reads temperature/top_p/repetition/frequency offsets from the
            # somatic_qualia service and applies them, bounded, to the sampler.
            # The perturbation is real and measurable at the output.
            #
            # "Raw felt perturbation" was the previous label, and the code does
            # not support it: felt-ness is not established by a service returning
            # four floats. The mechanism stands on its own without the claim.
            try:
                _sq = ServiceContainer.get("somatic_qualia", default=None)
                if _sq is not None:
                    _sq_pert = _sq.compute_perturbation()
                    if _sq_pert:
                        _sq_temp = self._modulator_delta(
                            _sq_pert.get("temperature_perturbation", 0.0),
                            source="somatic_qualia.temperature",
                            limit=0.5,
                        )
                        _sq_rep = self._modulator_delta(
                            _sq_pert.get("repetition_penalty_perturbation", 0.0),
                            source="somatic_qualia.repetition_penalty",
                            limit=0.5,
                        )
                        _sq_topp = self._modulator_delta(
                            _sq_pert.get("top_p_perturbation", 0.0),
                            source="somatic_qualia.top_p",
                            limit=0.5,
                        )
                        _sq_freq = self._modulator_delta(
                            _sq_pert.get("frequency_penalty_perturbation", 0.0),
                            source="somatic_qualia.frequency_penalty",
                            limit=0.5,
                        )
                        if somatic_temperature is not None:
                            somatic_temperature = max(0.1, min(1.5, somatic_temperature + _sq_temp))
                        if "repetition_penalty" in morpho_kwargs:
                            morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _sq_rep))
                        if "top_p" in morpho_kwargs:
                            morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _sq_topp))
                        if _sq_freq:
                            morpho_kwargs["frequency_penalty"] = max(0.0, min(0.5, morpho_kwargs.get("frequency_penalty", 0.0) + _sq_freq))
                        logger.debug(
                            "🫀 SomaticQualia: temp=%.4f rep=%.4f topp=%.4f freq=%.4f",
                            _sq_temp, _sq_rep, _sq_topp, _sq_freq,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as _sq_e:
                record_degradation(
                    "inference_gate",
                    _sq_e,
                    severity="warning",
                    action="continued without somatic qualia generation perturbation",
                )
                logger.debug("SomaticQualia coupling unavailable: %s", _sq_e)

            # ── Free Energy: Urgency-based tier escalation ──
            # When FE is high and rising, prefer deeper model for better reasoning
            try:
                _fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                if _fe_engine and _fe_engine.current:
                    _fe_state = _fe_engine.current
                    # High FE + complex action → request deeper model
                    if (
                        _fe_state.free_energy > 0.65
                        and _fe_state.dominant_action in ("update_beliefs", "act_on_world")
                        and requested_tier == "primary"
                    ):
                        # Nudge toward deeper tier if available
                        if not deep_handoff:
                            logger.debug(
                                "⚡ FE urgency (F=%.2f, action=%s): consider deeper reasoning",
                                _fe_state.free_energy,
                                _fe_state.dominant_action,
                            )
                            # Don't force tier switch — just extend token budget
                            max_tokens = min(max_tokens + 256, 4096)
            except _INFERENCE_RECOVERABLE_ERRORS as _fe_e:
                record_degradation(
                    "inference_gate",
                    _fe_e,
                    severity="warning",
                    action="continued without free-energy token-budget nudge",
                )
                logger.debug("FreeEnergy tier nudge unavailable: %s", _fe_e)

        # Ordinary live conversation must not collapse into a starvation budget
        # after affective / homeostatic modulation. Explicit caller caps still
        # win, as do hard resource-stakes blocks and deep-probe turns.
        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier in {"primary", "secondary"}
            and "max_tokens" not in context
            and not bool(context.get("resource_stakes_blocked", False))
            and not deep_probe_request
            and not isolated_generation_contract
            and not health_probe
        ):
            foreground_floor, foreground_cap, _foreground_loops = (
                self._foreground_compute_profile(initial_visible_user_prompt)
            )
            max_tokens = min(max_tokens, foreground_cap)
            if max_tokens < foreground_floor:
                logger.info(
                    "🧠 Foreground chat compute profile raised budget %d→%d "
                    "(cap=%d, loops=%d, origin=%s).",
                    max_tokens,
                    foreground_floor,
                    foreground_cap,
                    _foreground_loops,
                    origin or "unknown",
                )
                max_tokens = foreground_floor

        # The block above is skipped whenever the caller named its own budget —
        # and the desktop chat route always does, so no live desktop turn has
        # ever had a starvation floor. An explicit cap is an upper bound the
        # caller is entitled to; it is not permission to modulate the answer
        # down to a length that cannot finish a sentence.
        #
        # LIVE DEFECT, 2026-07-26: the route asked for 1,536 tokens for
        # "…show the reasoning, then give the exact fraction". Memory-pressure
        # capping took it to 384 and affective/resource scaling to 239, and the
        # answer arrived correct and cut mid-sentence:
        #
        #   "Total marbles: 3 red + 4 blue + 5 green = 12. For both to be the
        #    same colour, we need to consider each case separately: Both Red"
        #
        # A truncated answer is not a cheaper answer. The person re-asks and
        # the whole turn is paid for twice.
        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier in {"primary", "secondary"}
            and not deep_probe_request
            and not isolated_generation_contract
            and not health_probe
            and not bool(context.get("resource_stakes_blocked", False))
        ):
            try:
                requested_budget = int(context.get("max_tokens") or 0)
            except (TypeError, ValueError):
                requested_budget = 0
            if requested_budget > 0:
                # A flat floor rescues a conversational reply and still starves
                # a derivation: measured live 2026-07-27, the caller asked 896
                # for "when does the second train catch the first, and how far
                # from the station?", pressure scaling cut it to 459, and the
                # floor lifted it to 512 — enough to reach step 5 of 5 and stop
                # at "- The". The caller now says whether the turn needs room,
                # and the floor answers that instead of a constant.
                needs_room = bool(context.get("reply_needs_room", False))
                starvation_floor = min(
                    requested_budget,
                    self._configured_token_bound(
                        "AURA_FOREGROUND_CHAT_DERIVATION_FLOOR_TOKENS"
                        if needs_room
                        else "AURA_FOREGROUND_CHAT_STARVATION_FLOOR_TOKENS",
                        1024 if needs_room else 512,
                        minimum=256,
                    ),
                )
                if max_tokens < starvation_floor:
                    logger.info(
                        "🧠 Foreground starvation floor raised budget %d→%d "
                        "(caller asked %d, origin=%s).",
                        max_tokens,
                        starvation_floor,
                        requested_budget,
                        origin or "unknown",
                    )
                    max_tokens = starvation_floor

        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and not isolated_generation_contract
            and not health_probe
            and not benchmark_request
            and not proof_evaluation_contract
            and not strict_answer_contract
        ):
            somatic_temperature, max_tokens, applied_bias = self._apply_runtime_sampling_biases(
                base_temperature=somatic_temperature,
                max_tokens=max_tokens,
                context=context,
                state=state,
                allow_token_scaling="max_tokens" not in context,
            )
            if applied_bias["temperature_delta"] or applied_bias["max_tokens_factor"] != 1.0:
                logger.debug(
                    "🧠 Runtime sampling bias: temp_delta=%.3f token_factor=%.3f max_tokens=%d",
                    applied_bias["temperature_delta"],
                    applied_bias["max_tokens_factor"],
                    max_tokens,
                )
            # A bias may spend less of the budget than it was given. It may not
            # spend less than the request needs. The completion floor was
            # applied further up and this multiplier ran after it, so the floor
            # has to be put back or it was never a floor.
            #
            # LIVE, 2026-08-27: a question that had to be worked out carried a
            # floor of 896 tokens. An integration measure scaled the budget by
            # its smallest permitted factor and the model was dispatched with
            # 363, stopping one sentence before the answer. The same principle
            # is already written where the floor is computed: sampling biases
            # may make an answer terser, and may not make the surface smaller
            # than the visible request.
            try:
                _floor = int(context.get("user_surface_completion_floor") or 0)
            except (TypeError, ValueError, OverflowError):
                _floor = 0
            if 0 < _floor and max_tokens < _floor:
                logger.info(
                    "🧠 Completion floor restored after sampling bias: %d→%d.",
                    max_tokens,
                    _floor,
                )
                max_tokens = _floor

        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier in {"primary", "secondary"}
            and "max_tokens" not in context
            and not bool(context.get("resource_stakes_blocked", False))
            and not deep_probe_request
            and not isolated_generation_contract
            and not health_probe
        ):
            foreground_floor, foreground_cap, _foreground_loops = (
                self._foreground_compute_profile(initial_visible_user_prompt)
            )
            bounded = min(max_tokens, foreground_cap)
            if bounded < foreground_floor:
                logger.info(
                    "🧠 Foreground chat post-bias budget floor raised %d→%d "
                    "(cap=%d, origin=%s).",
                    bounded,
                    foreground_floor,
                    foreground_cap,
                    origin or "unknown",
                )
                max_tokens = foreground_floor
            else:
                max_tokens = bounded

        if explicit_max_tokens_cap is not None:
            max_tokens = min(max_tokens, explicit_max_tokens_cap)
            if (
                protected_compact_capability_contract
                and not bool(context.get("resource_stakes_blocked", False))
            ):
                max_tokens = max(max_tokens, min(384, explicit_max_tokens_cap))
            context["max_tokens"] = max_tokens

        # The live primary conversation lane USES the prompt cache. It used to
        # force-set disable_prompt_cache=True here, on the premise that reuse
        # was "approximate" and the cause of clipped or stale Cortex drafts.
        #
        # Reuse is not approximate. A cached entry is KV for a byte-identical
        # token prefix; measured end to end on the same stack, three cached
        # turns produced continuations byte-identical to an uncached control
        # while reuse climbed 0 -> 34/53 -> 57/74 tokens. The clipped and stale
        # drafts had three other causes, all since fixed: the trie stored one
        # mutable cache object under every growing prefix, so old keys aliased
        # later KV; the trim probe checked the wrong module and answered False
        # forever, so a diverging prefix could never be trimmed to fit; and
        # nothing partitioned lanes, so internal generations and the
        # conversation shared entries. Reuse is now scoped to `user_surface`,
        # inserted once after generation, and trimmed when prefixes diverge.
        #
        # Leaving it force-disabled is what the endurance wall is made of: this
        # lane's prompt IS the whole conversation, so re-prefilling from token
        # zero makes time-to-first-token climb until it crosses the turn budget
        # and the conversation stops answering. Callers that genuinely need an
        # exact cold prompt (strict/proof/operator contracts, health probes)
        # still set the flag themselves and are bypassed in the worker.

        if deep_probe_request and not is_background:
            try:
                probe_token_cap = int(_FLAG_DEEP_PROBE_MAX_TOKENS.value())
            except (TypeError, ValueError) as _probe_cap_exc:
                _record_inference_degradation(
                    _probe_cap_exc,
                    action="used default deep-probe token cap after malformed environment value",
                )
                probe_token_cap = 384
            max_tokens = min(max_tokens, max(128, probe_token_cap))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False

        if strict_answer_contract and strict_max_token_cap is not None:
            max_tokens = max(1, min(max_tokens, strict_max_token_cap))
            context["max_tokens"] = max_tokens

        if operator_evidence_contract:
            try:
                requested_operator_cap = int(context.get("max_tokens") or 220)
            except (TypeError, ValueError):
                requested_operator_cap = 220
            max_tokens = max(1, min(max_tokens, requested_operator_cap, 220))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False
            context["disable_prompt_cache"] = True
            context["clear_prompt_cache"] = True

        if health_probe:
            requested_cap = context.get("max_tokens", max_tokens)
            try:
                requested_cap_int = max(1, int(requested_cap))
            except (TypeError, ValueError):
                requested_cap_int = 32
            max_tokens = max(1, min(max_tokens, requested_cap_int, 64))
            context["max_tokens"] = max_tokens
            context.setdefault("clean_user_surface_contract", True)
            context.setdefault("user_surface_validation_prompt", initial_visible_user_prompt)
            context.setdefault("clean_user_surface_recurrent_loops", 1)
            context.setdefault("clean_user_surface_steering_alpha", 0.25)
            morpho_kwargs.setdefault("clean_user_surface_contract", True)
            morpho_kwargs.setdefault("user_surface_validation_prompt", initial_visible_user_prompt)
            morpho_kwargs.setdefault("clean_user_surface_recurrent_loops", 1)
            morpho_kwargs.setdefault("clean_user_surface_steering_alpha", 0.25)

        if benchmark_request:
            requested_cap = context.get("max_tokens", max_tokens)
            try:
                requested_cap_int = max(1, int(requested_cap))
            except (TypeError, ValueError):
                requested_cap_int = 96
            max_tokens = max(1, min(max_tokens, requested_cap_int))
            context["max_tokens"] = max_tokens

        output_contract_is_user_facing = bool(
            not is_background
            and not isolated_generation_contract
            and not health_probe
            and not benchmark_request
            and (
                explicit_foreground
                or self._origin_is_user_facing(origin)
                or requested_tier in {"primary", "secondary"}
            )
        )
        # On a turn that EXECUTES something, a shape phrase describes the
        # ARTIFACT, not her reply.
        #
        # "Open the Notes app and write a new note with three sentences about
        # humpback whales" parses to sentence_count=3, and that was applied to
        # the chat reply: it forced the compact foreground context and clamped
        # the budget to max_tokens=288 — far too small to emit a multi-step
        # desktop plan. She produced conversational filler instead, nothing
        # executed, and the gate then vetoed the filler for not matching the
        # three-sentence shape it had itself imposed.
        #
        # Measured live twice, and confirmed by removing the phrase: the same
        # request without "three sentences" planned and executed, and the note
        # is on disk. Every demo instruction carries this kind of clause ("3
        # articles", "a coherent summary", "a short note"), so the artifact
        # spec was systematically starving the plan that would have produced it.
        #
        # The executor already owns artifact shape through `document_body`;
        # here it must not also become a ceiling on the report she gives back.
        if bool(context.get("desktop_execution_contract", False)):
            # Carry the flag to the client so the unified-memory clamp keeps a
            # floor under the PLAN. Without it, pressure (a screen recorder is
            # enough) shrinks the budget below what the steps need and the task
            # cannot be attempted at all.
            morpho_kwargs["desktop_execution_contract"] = True
            # And give the plan room to exist. The origin's conversational
            # default capped this turn at 288 tokens, which cannot hold a
            # multi-step JSON plan, so the model emitted prose, the draft was
            # judged truncated, and nothing executed. Measured live on a
            # DELIBERATE desktop turn that had already been routed to
            # desktop_task. Success up to now depended on which planner ran:
            # the deterministic heuristic needs no tokens, the model one does.
            _plan_floor = 1024
            if int(max_tokens or 0) < _plan_floor:
                logger.info(
                    "🧾 [CONTRACT] Desktop execution turn: raising the reply "
                    "budget %s → %d so the plan can be expressed.",
                    max_tokens,
                    _plan_floor,
                )
                max_tokens = _plan_floor
                context["max_tokens"] = max_tokens
                # NOT morpho_kwargs. Every _generate_with_client call site
                # passes max_tokens= explicitly AND splats **morpho_kwargs, so
                # putting it in both raised
                #   TypeError: _generate_with_client() got multiple values for
                #   keyword argument 'max_tokens'
                # which failed the inference_gate closed and surfaced as
                # user_cycle_no_response — the engine returning nothing at all
                # on every desktop turn, in two seconds, while ordinary
                # conversation through the same engine kept working.
            if output_contract_is_user_facing:
                output_contract_is_user_facing = False
                logger.info(
                    "🧾 [CONTRACT] Output-shape request treated as the ARTIFACT's "
                    "shape on a desktop-execution turn; the reply keeps its full "
                    "budget (would have capped at %s tokens).",
                    getattr(output_contract, "hard_token_ceiling", None),
                )
        if (
            output_contract_is_user_facing
            and output_contract_payload is not None
            and output_contract.hard_token_ceiling is not None
        ):
            planned_tokens = max_tokens
            max_tokens = max(
                1,
                min(max_tokens, int(output_contract.hard_token_ceiling)),
            )
            context["requested_output_contract"] = dict(output_contract_payload)
            context["semantic_output_token_cap"] = output_contract.semantic_token_cap
            context["hard_output_token_ceiling"] = output_contract.hard_token_ceiling
            context["max_tokens"] = max_tokens
            morpho_kwargs["requested_output_contract"] = dict(output_contract_payload)
            morpho_kwargs["semantic_output_token_cap"] = output_contract.semantic_token_cap
            morpho_kwargs["hard_output_token_ceiling"] = output_contract.hard_token_ceiling
            if max_tokens < planned_tokens:
                logger.info(
                    "🧠 Explicit output contract capped generation %d→%d "
                    "(kind=%s semantic=%s hard=%s).",
                    planned_tokens,
                    max_tokens,
                    output_contract.kind,
                    output_contract.semantic_token_cap,
                    output_contract.hard_token_ceiling,
                )

        # No policy floor may expand a caller-admitted ceiling, and none may
        # expand a viability ceiling either. Keep both as the final
        # token-budget transformations before prompt construction and every
        # local provider call below.
        if stakes_token_ceiling is not None and max_tokens > stakes_token_ceiling:
            logger.info(
                "🪫 Resource-stakes ceiling re-applied after later modifiers: %d→%d.",
                max_tokens,
                stakes_token_ceiling,
            )
            max_tokens = max(1, min(max_tokens, stakes_token_ceiling))
            context["max_tokens"] = max_tokens
        if explicit_max_tokens_cap is not None:
            max_tokens = max(1, min(max_tokens, explicit_max_tokens_cap))
            context["max_tokens"] = max_tokens

        # Build the prompt only after routing intent is known so we can choose
        # a compact user-facing path instead of always constructing the richest stack.
        brief = context.get("brief", "")
        if hasattr(brief, "to_briefing_text"):
            brief = brief.to_briefing_text()
        elif not isinstance(brief, str):
            brief = str(brief)
        use_compact_foreground_context = self._should_use_compact_foreground_context(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
            prompt=initial_visible_user_prompt,
            context=context,
        )
        provided_messages = context.get("messages")
        if provided_messages is not None and not isinstance(provided_messages, list):
            # Dropping it silently is what let a malformed payload become a
            # system-only generation: the merge iterated nothing, inserted the
            # system message, and sent a prompt with no user turn in it. The
            # caller's `prompt` still carries the request, so the turn is
            # served — but the caller is told its payload was not used.
            _record_inference_degradation(
                TypeError(
                    f"context['messages'] is {type(provided_messages).__name__}, not a list"
                ),
                action="ignored a malformed prebuilt message payload and used the prompt instead",
                extra={"origin": str(origin or "")},
            )
            context["prebuilt_messages_rejected"] = "not_a_list"
            provided_messages = None
        if not isinstance(provided_messages, list):
            provided_messages = None
        context_system_prompt = str(context.get("system_prompt", "") or "").strip()

        def _append_unique_system_part(parts: list[str], content: Any) -> None:
            text = str(content or "").strip()
            if not text:
                return
            if any(text == existing or text in existing for existing in parts):
                return
            parts.append(text)

        if strict_answer_contract:
            provided_system_parts: list[str] = []
            _append_unique_system_part(provided_system_parts, context_system_prompt)
            strict_system_prompt = (
                "You are Aura's local reasoning lane, a persistent local cognitive runtime. "
                "Follow the user's exact output contract. "
                "When the user requests <answer>...</answer>, return only that final answer envelope. "
                "Do not copy instructions, role labels, or explanatory text."
            )
            strict_user_prompt = str(prompt or "")
            if provided_messages is not None:
                # Collect EVERY system message in original order — a reverse
                # scan that stops at the latest user turn silently drops the
                # normal leading system message, losing safety policy, tool
                # receipts, and caller-required constraints on strict routes.
                for msg in provided_messages:
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "system":
                        _append_unique_system_part(
                            provided_system_parts,
                            msg.get("content", ""),
                        )
                for msg in reversed(provided_messages):
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "user":
                        strict_user_prompt = str(msg.get("content", "") or strict_user_prompt)
                        break
            if provided_system_parts:
                preserved_system = "\n\n".join(provided_system_parts).strip()
                if preserved_system:
                    strict_system_prompt = f"{strict_system_prompt}\n\n{preserved_system}"
            strict_system_prompt += self._strict_contract_procedure_hints(strict_user_prompt)
            provided_messages = [
                {"role": "system", "content": strict_system_prompt},
                *self._strict_contract_grounding_turns(provided_messages),
                {"role": "user", "content": strict_user_prompt},
            ]
        elif strict_value_contract:
            provided_system_parts = []
            _append_unique_system_part(provided_system_parts, context_system_prompt)
            strict_value_system_prompt = (
                "You are Aura's local reasoning lane. Solve the task and return only "
                "the final answer value. Do not explain, do not add role labels, and "
                "do not include XML tags."
            )
            strict_value_user_prompt = str(prompt or "")
            if provided_messages is not None:
                # Same ordering contract as the strict-answer branch: keep
                # every system message, then take the latest user turn.
                for msg in provided_messages:
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "system":
                        _append_unique_system_part(
                            provided_system_parts,
                            msg.get("content", ""),
                        )
                for msg in reversed(provided_messages):
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "user":
                        strict_value_user_prompt = str(msg.get("content", "") or strict_value_user_prompt)
                        break
            if provided_system_parts:
                preserved_system = "\n\n".join(provided_system_parts).strip()
                if preserved_system:
                    strict_value_system_prompt = f"{strict_value_system_prompt}\n\n{preserved_system}"
            strict_value_system_prompt += self._strict_contract_procedure_hints(
                strict_value_user_prompt
            )
            provided_messages = [
                {"role": "system", "content": strict_value_system_prompt},
                *self._strict_contract_grounding_turns(provided_messages),
                {"role": "user", "content": strict_value_user_prompt},
            ]
        visible_user_prompt = initial_visible_user_prompt
        from core.utils.injected_blocks import is_stamped_grounding

        live_context_already_grounded = bool(
            context.get("live_context_already_grounded", False)
            and isinstance(provided_messages, list)
            and any(is_stamped_grounding(message) for message in provided_messages)
        )
        if provided_messages is not None:
            system_prompt = ""
            for msg in provided_messages:
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "system":
                    system_prompt = str(msg.get("content", "") or "").strip()
                    break
            living_mind_context = ""
            if (
                not isolated_generation_contract
                and not is_background
                and not live_context_already_grounded
                and self._origin_is_user_facing(origin)
            ):
                needs_full_live_context = bool(
                    context.get("live_runtime_payload_required", False)
                    and not use_compact_foreground_context
                    and (
                        is_live_self_reflection_turn(initial_visible_user_prompt)
                        or is_self_process_question(initial_visible_user_prompt)
                    )
                )
                if needs_full_live_context:
                    living_mind_context = await self._assemble_live_context(
                        initial_visible_user_prompt, origin, full=True
                    )
                else:
                    living_mind_context = await self._assemble_live_context(
                        visible_user_prompt, origin, full=False
                    )
        elif use_compact_foreground_context:
            system_prompt = self._build_compact_system_prompt(brief)
            living_mind_context = (
                ""
                if live_context_already_grounded
                else await self._assemble_live_context(
                    visible_user_prompt, origin, full=False
                )
            )
        else:
            system_prompt = self._build_system_prompt(brief)
            # [STABILITY v50] The 20+ consciousness subsystems queried here can
            # individually hang on lock contention or slow I/O. One deadline
            # covers the full assembly AND the compact fallback.
            living_mind_context = (
                ""
                if live_context_already_grounded
                else await self._assemble_live_context(
                    visible_user_prompt, origin, full=True
                )
            )
        prompt_contract_block = self._prompt_contract_block(context)

        # She has no clock. Asked how she is doing at 00:30 she answered "the
        # sun's up ... clouds gathering in the east" — not dishonesty, but the
        # only thing a model can do when asked about a present it was never
        # given. Nothing in this path had ever carried the date or the hour.
        # ~500 chars of read-not-inferred fact, on every non-isolated turn.
        # Grounding that changes every turn is collected here and delivered
        # AFTER the conversation instead of inside the system prompt. Inside the
        # system prompt it sits ~126 tokens in, ahead of the entire history, so
        # its per-turn churn invalidated the KV prefix for everything behind it:
        # measured live on the user surface as reuse of 126 of 1774 tokens (7%)
        # with divergence beginning exactly at "## WHAT YOU ACTUALLY JUST DID".
        # Delivered last, divergence lands after the history instead of before
        # it, so a long conversation stops re-prefilling itself from token zero.
        # The content is unchanged — only its position.
        contract_grounding_blocks: list[str] = []
        task_grounding_blocks: list[str] = []
        ambient_grounding_blocks: list[str] = []
        # The response contract describes THIS turn — its reason label and the
        # current local date both change per turn — so it belongs beside the
        # turn, not in the persistent system prompt. Measured live: it landed at
        # token 125 and divergence began at "## RESPONSE CONTRACT\n- Reason:
        # compound_prompt\n- Current local date: ...", stranding 3,884 tokens of
        # conversation behind it (3% reused).
        if prompt_contract_block and not isolated_generation_contract:
            contract_grounding_blocks.append(prompt_contract_block)
        # Current mind state changes independently of identity and policy. It
        # belongs with this turn's evidence, never inside the stable system
        # prefix. The old path inserted it above and then inserted it a second
        # time into prebuilt messages below. Besides presenting one source twice,
        # that made every affect tick invalidate the conversation's KV prefix.
        if living_mind_context and not isolated_generation_contract:
            ambient_grounding_blocks.append(living_mind_context)
        await _attach_the_present_moment(
            ambient_grounding_blocks=ambient_grounding_blocks,
            isolated_generation_contract=isolated_generation_contract,
            recent_actions_already_grounded=bool(
                context.get("recent_actions_already_grounded", False)
            ),
            task_grounding_blocks=task_grounding_blocks,
            visible_user_prompt=visible_user_prompt,
        )
        # Keep prompt growth aligned with the actual local model context window
        # instead of assuming 128k+ headroom on the primary Qwen lane.

        # ── Somatic narrative: brief felt-state line in the system prompt ────────
        if somatic_temperature is not None and not isolated_generation_contract:
            try:
                from core.affect.affective_circumplex import get_circumplex

                _soma_narrative = get_circumplex().describe()
                if _soma_narrative:
                    # Felt state changes on every tick; it travels with the rest
                    # of the volatile grounding, after the conversation.
                    ambient_grounding_blocks.append(
                        f"## SOMATIC STATE\n{_soma_narrative}"
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _exc:
                record_degradation(
                    "inference_gate",
                    _exc,
                    severity="warning",
                    action="continued without somatic-state prompt section",
                )
                logger.debug("Suppressed Exception: %s", _exc)

        prompt_user_facing = bool(
            not benchmark_request
            and not is_background
            and not web_interlocutor_contract
            and (
                self._origin_is_user_facing(origin)
                or explicit_foreground
                or purpose in {"reply", "expression", "chat", "conversation", "user_response"}
            )
        )

        # ── Architecture Self-Awareness: inject relevant subsystem context ──────
        # Only for user-facing requests that mention architecture/code keywords.
        if prompt_user_facing and not isolated_generation_contract:
            try:
                import re as _re

                _arch_triggers = _re.compile(
                    r"\b(how|explain|what|which|where|why|trace|show|describe)\b.{0,60}"
                    r"\b(module|subsystem|file|class|method|function|work|does|handles|manages|routes|sends|wires)\b",
                    _re.IGNORECASE,
                )
                if _arch_triggers.search(visible_user_prompt):
                    from core.self.architecture_index import get_architecture_index

                    arch_excerpt = get_architecture_index().query(
                        visible_user_prompt,
                        max_results=3,
                    )
                    if arch_excerpt:
                        # The excerpt depends on this question, so it travels
                        # with turn-local grounding. Putting it in the stable
                        # system prefix invalidates cached conversation tokens
                        # when the next question is about another subsystem.
                        task_grounding_blocks.append(str(arch_excerpt))
            except _INFERENCE_RECOVERABLE_ERRORS as _ae:
                record_degradation(
                    "inference_gate",
                    _ae,
                    severity="warning",
                    action="continued without architecture self-awareness excerpt",
                )
                logger.debug("ArchIndex injection skipped: %s", _ae)
            contract_grounding_blocks.append(
                conversation_reliability_system_block(visible_user_prompt)
            )
        history = context.get("history", [])
        # An internal decision does not need her whole self in front of it.
        #
        # Measured live: choosing between four named moves loaded a 2385-char
        # persona scaffold in front of a 1130-char request, once per move, on
        # a loop that has to act about once a second. The scaffold is what
        # makes her sound like herself when she is talking to somebody; a
        # deliberation is not talking to anybody, and it pays for the whole
        # thing in latency on every cycle.
        use_rich_context = False if (
            isolated_generation_contract or benchmark_request or internal_inference_call
        ) else bool(
            context.get(
                "rich_context",
                self._should_use_rich_context(
                    origin,
                    requested_tier,
                    deep_handoff=deep_handoff,
                    is_background=is_background,
                ),
            )
        )
        if provided_messages is not None:
            messages = [dict(msg) for msg in provided_messages if isinstance(msg, dict)]
            # A payload whose entries are all malformed — or that carries only
            # system turns — reaches the model as instructions with nothing to
            # answer. The caller's prompt is the request; put it back.
            if not any(
                str(msg.get("role", "") or "").strip().lower() == "user"
                and str(msg.get("content", "") or "").strip()
                for msg in messages
            ):
                recovered = str(prompt or "").strip()
                if recovered:
                    messages.append({"role": "user", "content": recovered})
                    context["prebuilt_messages_user_turn_recovered"] = True
                    _record_inference_degradation(
                        ValueError("prebuilt messages carried no usable user turn"),
                        action="restored the caller's prompt as the user turn",
                        extra={"origin": str(origin or ""), "messages": len(messages)},
                    )
        else:
            messages = (
                self._build_messages(prompt, system_prompt, history)
                if use_rich_context
                else self._build_compact_messages(prompt, system_prompt, history)
            )
        if isinstance(messages, list) and (
            is_background
            or (
                provided_messages is not None
                and (use_compact_foreground_context or use_rich_context)
            )
        ):
            short_output_contract = self._has_short_live_output_contract(context)
            deep_probe_context = bool(context.get("deep_mind_probe", False)) and not (
                short_output_contract
            )
            if is_background:
                requested_background_profile = str(
                    context.get("background_prompt_profile") or ""
                ).strip().lower()
                if requested_background_profile not in {"background", "curriculum"}:
                    requested_background_profile = (
                        "curriculum"
                        if str(context.get("purpose") or "").strip().lower()
                        == "curriculum_practice"
                        else "background"
                    )
                foreground_profile = requested_background_profile
            elif use_compact_foreground_context:
                foreground_profile = (
                    "contract"
                    if short_output_contract
                    else self._foreground_prompt_profile(
                        visible_user_prompt,
                        context,
                    )
                )
            else:
                # Rich/DEEP prebuilt prompts otherwise bypassed compaction
                # entirely, so the local Cortex (notably the in-process MLX 32B)
                # received 100k+ char prompts (~25k tokens) that exceed the 16k
                # context window AND can't be processed+generated within the
                # cognitive-cycle watchdog → "Cognitive cycle TIMEOUT" +
                # "RuntimeError in mlx_client". Compact rich prebuilt to the
                # extended budget (~26k chars) so the live lane stays responsive
                # while still carrying substantial living-mind context.
                foreground_profile = "extended"
            messages = self._compact_prebuilt_messages(
                messages,
                history_limit=(
                    4
                    if is_background
                    else self._foreground_prebuilt_history_limit(
                        visible_user_prompt,
                        context,
                        deep_probe=deep_probe_context,
                    )
                ),
                deep_probe=deep_probe_context,
                budget_profile=foreground_profile,
                current_user_content=visible_user_prompt,
            )
            # The compacted message set is now AUTHORITATIVE. Turn-local mind
            # context and reliability guidance are attached below, after this
            # compaction, as one bounded grounding message.
            #
            # `system_prompt` is a separate identity/policy string that grew
            # independently and is never compacted. It is still handed to the
            # client alongside these
            # messages, and the client merges a separately-passed system_prompt
            # into messages[0] — so it silently undid every compaction above.
            # Measured live: a 2,399-char compacted system message reached the
            # worker at 106,861 chars, turning a 278-char question into a
            # 27,129-token prefill (a 384:1 scaffold-to-request ratio) that
            # could not produce a first token inside the turn budget. None of
            # it was visible, because the prompt plan logs the compacted
            # messages and the re-inflation happens after that.
            #
            # The compacted structured messages now carry all system policy.
            # Passing a scalar copy would let the client merge the unbounded
            # pre-compaction prompt back into the first system message.
            system_prompt = ""
        # Volatile grounding rides LAST, behind the conversation, so the KV
        # prefix covering the history survives from one turn to the next.
        # Appended after compaction on purpose: compaction rewrites the history
        # it is given, and this block must not be trimmed away — it is the
        # read-not-inferred ground truth (clock, receipts, felt state) that
        # stops her narrating a present she was never given.
        has_volatile_grounding = bool(
            contract_grounding_blocks
            or task_grounding_blocks
            or ambient_grounding_blocks
        )
        messages, system_prompt = _refresh_volatile_grounding(
            ambient_grounding_blocks=ambient_grounding_blocks,
            context=context,
            contract_grounding_blocks=contract_grounding_blocks,
            has_volatile_grounding=has_volatile_grounding,
            messages=messages,
            self=self,
            system_prompt=system_prompt,
            task_grounding_blocks=task_grounding_blocks,
        )
        # Cache policy is not a caller preference.
        #
        # morpho_kwargs is populated from `context` early, then several
        # contracts (strict proof, operator evidence, health probe) set
        # context["disable_prompt_cache"] = True LATER — after the copy. A
        # caller that passed disable_prompt_cache=False therefore kept its
        # False in the kwargs that actually reach the worker, and an exact-cold
        # prompt contract silently ran on reused KV. Re-sync here, once, after
        # every contract has had its say: policy wins.
        for _cache_key in ("disable_prompt_cache", "clear_prompt_cache"):
            if bool(context.get(_cache_key, False)):
                if not bool(morpho_kwargs.get(_cache_key, False)):
                    logger.debug(
                        "Cache policy overrides caller %s=%r for this contract.",
                        _cache_key,
                        morpho_kwargs.get(_cache_key),
                    )
                morpho_kwargs[_cache_key] = True

        # Last word on size, in the unit the window is actually measured in.
        # Every budget above this line is in characters; this one is in tokens
        # and it applies to every route, including prebuilt message payloads
        # that skip the per-message compactor entirely.
        system_prompt, messages = self._fit_prompt_to_window(
            system_prompt,
            list(messages or []),
            answer_tokens=int(max_tokens or 0),
            origin=origin,
        )
        prompt_chars = sum(
            len(str(msg.get("content", ""))) for msg in (messages or ())
        )
        # Settle time from what the worker will actually read, after every
        # compactor and window clamp has run.  Using ``prompt`` here priced a
        # discarded rich payload: LIVE 2026-08-30 logged a 631-second route for
        # a final 2,184-token prompt, while the request Deadline still held the
        # original 87 seconds.  The log, integer budget, and owning clock were
        # therefore three different contracts for one dispatch.
        dispatched_prompt_text = "\n".join(
            str(msg.get("content", "") or "")
            for msg in (messages or ())
            if isinstance(msg, dict)
        )
        # The protected capability lane overrode the resource envelope to get
        # here; fitting the answer to the clock used to undo that override
        # without knowing it existed, so a lane protected FROM truncation was
        # truncated by the next transformation.
        _protected_floor = (
            int(max_tokens)
            if protected_compact_capability_contract
            and not bool(context.get("resource_stakes_blocked", False))
            else 0
        )
        settled_timeout_val, settled_max_tokens = fit_the_answer_to_the_time(
            dispatched_prompt_text,
            max_tokens,
            timeout_val,
            floor=_protected_floor,
        )
        if settled_timeout_val != timeout_val:
            timeout_val = float(settled_timeout_val)
            primary_timeout, fallback_timeout = self._split_attempt_timeouts(
                timeout_val,
                requested_tier,
            )
            if requested_tier == "primary" and lower_local_lane_forbidden:
                primary_timeout = max(8.0, timeout_val - _DELIVERY_MARGIN_S)
                fallback_timeout = min(_DELIVERY_MARGIN_S, timeout_val)
            request_deadline = request_deadline.with_timeout(timeout_val)
            context["request_deadline_s"] = timeout_val
        if settled_max_tokens != max_tokens:
            max_tokens = int(settled_max_tokens)
            context["max_tokens"] = max_tokens
        prompt_mode = "rich" if use_rich_context else "compact"
        if use_compact_foreground_context:
            prompt_mode = "compact_foreground"
        if provided_messages is not None:
            prompt_mode = f"{prompt_mode}_prebuilt"
        # What share of the prompt is the person's actual request? A turn whose
        # scaffold dwarfs the question tends to be continued as more scaffold:
        # measured live 2026-07-26, a 78-char arithmetic question arrived inside
        # 7,414 chars and came back as self-description rather than an answer.
        # Without this breakdown the only visible number was the total, which
        # says nothing about the ratio that matters.
        scaffold_chars = sum(
            len(str(msg.get("content", "") or ""))
            for msg in messages
            if str(msg.get("role", "")).strip().lower() == "system"
        )
        request_chars = max(0, prompt_chars - scaffold_chars)
        # The separately-passed system_prompt is merged into messages[0] at the
        # client boundary, so it is part of the prefill even though it is not in
        # `messages` here. Leaving it out of this line is how a 106,861-char
        # re-inflation stayed invisible behind a plan that reported 4,479.
        # Which grounding blocks actually survived to the worker. Attachment was
        # already logged at the builder and the block still never arrived, so
        # the only useful signal is presence in the final text.
        _grounded = [
            name
            for name, marker in (
                ("present", "## PRESENT MOMENT"),
                ("instruments", "## YOUR OWN INSTRUMENTS"),
                ("receipts", "## WHAT YOU ACTUALLY JUST DID"),
                # Grounding that cannot be seen cannot be verified — the file
                # block spent a day being built into a prompt nobody sent.
                #
                # DERIVED from the registry, never hand-listed. Written out by
                # hand, this list immediately drifted: screen and beliefs were
                # registered as observables and left out here, so a screen
                # reading that WAS taken reported as not surviving, and an hour
                # went into looking for a delivery bug that did not exist.
                *_observable_dispatch_markers(),
            )
            if marker in str(system_prompt or "")
            or any(marker in str(msg.get("content", "") or "") for msg in messages)
        ]
        # FINAL word on the budget for an execution turn.
        #
        # The earlier raise fired ("raising the reply budget 288 -> 1024") and
        # was then overwritten by the compact-foreground path, so the caller
        # still asked for 288 — "Foreground starvation floor raised budget
        # 284->288 (caller asked 288)" — and a multi-step JSON plan cannot be
        # written in 288 tokens. Applied here, immediately before dispatch,
        # after every other budget computation has had its say.
        if bool(context.get("desktop_execution_contract", False)):
            _plan_floor_final = 1024
            if int(max_tokens or 0) < _plan_floor_final:
                logger.info(
                    "🖥️ [PLAN BUDGET] Execution turn: %s → %d tokens at dispatch.",
                    max_tokens,
                    _plan_floor_final,
                )
                max_tokens = _plan_floor_final
                context["max_tokens"] = max_tokens

        # FINAL word on the budget for an answer turn, for the same reason the
        # execution floor above is applied here: every earlier raise can be and
        # was overwritten by a later cap.
        #
        # LIVE, 2026-08-27: a question whose answer had to be worked out
        # carried a completion floor of 896 tokens all the way to the worker,
        # which read it correctly and opened the private reasoning channel —
        # and the same turn was dispatched with 128, so the generation ended
        # still inside that channel with no answer at all, twice.
        #
        # A declared output ceiling, a strict or operator contract, a probe and
        # a blocked resource stake all still win; each of those is an explicit
        # requirement rather than an incidental cap.
        # The presence of a floor is the entitlement. It is set further up only
        # for turns that qualify, so re-deriving the qualification here just
        # gives the two conditions somewhere to drift apart — and they did: the
        # turn that carried floor=896 to the worker did not satisfy a second
        # copy of the test, and was dispatched with 375.
        _clock_blocked_by = [
            name
            for name in (
                "hard_output_token_ceiling",
                "resource_stakes_blocked",
                "desktop_execution_contract",
            )
            if bool(context.get(name, False))
        ]
        # Whether anybody is waiting for this. Computed here rather than
        # inside the branch below, because a use of it further down sits
        # OUTSIDE that branch — so a turn whose clock was blocked reached the
        # use having never made the assignment, and raised UnboundLocalError
        # in the middle of generating.
        _is_user_facing = (
            not benchmark_request
            and not web_interlocutor_contract
            and (self._origin_is_user_facing(origin) or requested_tier == "primary")
            and not health_probe
            and not proof_evaluation_contract
        )
        if _clock_blocked_by:
            # Why the deadline was never reconsidered. Three conditions can
            # skip this and the log said nothing about any of them, so a turn
            # cancelled mid-prefill looked identical to one the clock had
            # examined and left alone.
            logger.info(
                "🧠 [ANSWER CLOCK] not consulted for this turn: %s",
                ",".join(_clock_blocked_by),
            )
        else:
            try:
                _answer_floor_final = int(
                    context.get("user_surface_completion_floor") or 0
                )
            except (TypeError, ValueError, OverflowError):
                _answer_floor_final = 0
            if 0 < _answer_floor_final and int(max_tokens or 0) < _answer_floor_final:
                logger.info(
                    "🧠 [ANSWER BUDGET] Answer turn: %s → %d tokens at dispatch.",
                    max_tokens,
                    _answer_floor_final,
                )
                max_tokens = _answer_floor_final
                context["max_tokens"] = max_tokens

            # A deadline that cannot deliver the budget the same request just
            # computed is two derived numbers contradicting each other, and
            # neither side could see the other.
            #
            # LIVE, 2026-08-27: the floor asked for 896 tokens and the
            # deliberate lane allowed about 150 seconds. The observed decode
            # rate made 896 tokens roughly 150 seconds of decoding on its own,
            # so the generation was cut mid-thought every time and the turn
            # served nothing. Raising the budget alone made it worse: 1,792
            # tokens were granted and the clock ended it at 43 seconds.
            #
            # The extension is bounded by the two measured quantities that
            # caused it — the floor and the observed rate — so there is no
            # invented number here and no open-ended wait. An unmeasured rate
            # extends nothing.
            # A turn that has to go and fetch something spends a whole
            # generation on the call before the answer is even started.
            #
            # LIVE, 2026-08-28: a diagnosis turn was offered the right tool,
            # spent forty-five seconds emitting one call, and the request
            # deadline expired fifty seconds later with nothing said about what
            # came back. The clock covered one generation and the turn needed
            # two.
            _generations = 1
            try:
                from core.intent.capability_selection import (
                    points_at_something_real,
                )

                if points_at_something_real(initial_visible_user_prompt):
                    _generations = 2
            except (ImportError, AttributeError, OSError, TypeError, ValueError):
                _generations = 1

            # Two entitlements, and only one of them had a clock.
            #
            # This whole block sat behind a completion floor, which says "this
            # ANSWER must be long". Needing two generations says "this TURN has
            # two phases", which is a different fact about a different thing,
            # and a turn that had the second without the first was timed as
            # though it had one phase.
            #
            # LIVE, 2026-08-28: the same request ran twice. With a floor it was
            # given 516 seconds and read three files; without one it was given
            # 148, and the answer — over a prompt its own worker measured at
            # 120 seconds to read — was cancelled with nothing said.
            # Anyone waiting for this answer, or a turn with a floor or a
            # second pass. The two special cases were the whole condition, so
            # an ordinary question — no floor, one generation — never had its
            # clock checked against what it had been asked to produce, which
            # is every conversational turn there is.
            if _is_user_facing or 0 < _answer_floor_final or _generations > 1:
                # The budget this clock has to cover is not the one the gate
                # asked for. On a thinking model the worker adds the reasoning
                # reserve to it, on the far side of this calculation, so the
                # clock was sized for 1,024 tokens while up to 2,048 were
                # decoded against it — and the extra was exactly the room the
                # reserve had been raised to provide.
                _model_for_clock = self._model_now_serving(requested_tier)
                _clean_user_surface_for_clock = bool(
                    _is_user_facing
                    and requested_tier == "primary"
                    and not strict_answer_contract
                    and not strict_value_contract
                    and not internal_inference
                    and initial_visible_user_prompt
                )
                _reserve_the_worker_adds = self._reasoning_reserve_for_generation(
                    model=_model_for_clock,
                    cognitive_mode=context.get("cognitive_mode"),
                    final_user_surface=_clean_user_surface_for_clock,
                    completion_floor=_answer_floor_final,
                    budget_tokens=max_tokens,
                    seconds_remaining=float(timeout_val or 0.0),
                )
                _tokens_to_pay_for = max_tokens + _reserve_the_worker_adds
                _decode_s = _seconds_to_decode(_tokens_to_pay_for)
                # Reading the prompt is the other half of a generation, and
                # on this hardware it is the larger half. A turn was given time
                # to SAY its answer and none to read the question.
                _prompt_chars_for_clock = len(str(system_prompt or "")) + sum(
                    len(str((msg or {}).get("content") or ""))
                    for msg in (messages or [])
                    if isinstance(msg, dict)
                )
                _read_s = _seconds_to_read(_prompt_chars_for_clock)
                # And what the worker that will serve this says, which is the
                # number it will cancel itself by.
                #
                # A percentile over past readings cannot follow a rate that
                # halves under memory pressure, and the worker measures its
                # own. LIVE 2026-09-04, one line apart: "the prompt takes
                # about 2s to read", granting 25 seconds, and "a 2867-char
                # prompt takes about 8.8s to read at 82 tok/s", needing 26.3.
                # Cancelled at 25, every user-facing turn, with the runtime
                # healthy throughout.
                _worker_says = 0.0
                _asking = getattr(self, "_mlx_client", None)
                _knows = getattr(_asking, "least_time_to_read", None)
                if callable(_knows):
                    try:
                        _worker_says = float(_knows(_prompt_chars_for_clock) or 0.0)
                    except (TypeError, ValueError):
                        # not a failure: a rate that will not parse is not one.
                        _worker_says = 0.0
                _read_s = max(_read_s, _worker_says)
                if _decode_s > 0.0:
                    _needed = (
                        (_decode_s + _read_s) * _generations
                    ) + _DELIVERY_MARGIN_S
                    if _needed > float(timeout_val):
                        logger.info(
                            "🧠 [ANSWER CLOCK] %d tokens (%d asked + %d reserve the "
                            "worker adds) decode in about %.0fs and the prompt takes "
                            "about %.0fs to read, at the measured rates, and this turn "
                            "needs %d of them; deadline %.0fs → %.0fs.",
                            _tokens_to_pay_for,
                            max_tokens,
                            _reserve_the_worker_adds,
                            _decode_s,
                            _read_s,
                            _generations,
                            float(timeout_val),
                            _needed,
                        )
                        # Never past the ceiling the wait outside this one
                        # uses. A deadline of 557 seconds inside a wait that
                        # gives up at 480 is two numbers disagreeing again,
                        # with the outer one winning silently.
                        from core.runtime.response_policy import (
                            USER_FACING_COMPLETION_DEADLINE_MAX_S,
                        )

                        _cap = float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
                        # Forecasts inform progress reporting. They do not
                        # authorize substituting a less capable cortex.
                        timeout_val = min(_cap, _needed)
                        primary_timeout = max(8.0, timeout_val - _DELIVERY_MARGIN_S)
                        # The clock is one object, built when the request was
                        # admitted. Raising the number beside it computed an
                        # extension, logged it, and did not honour it.
                        #
                        # LIVE, 2026-08-28: "deadline 96s to 217s" and the
                        # request expired at 98 seconds.
                        request_deadline = request_deadline.with_timeout(
                            float(timeout_val)
                        )
                        context["request_deadline_s"] = float(timeout_val)

            # Only where the clock still has the last word. A turn somebody is
            # waiting for is no longer cancelled while it is producing tokens,
            # so trimming its answer to fit a deadline would be shortening it
            # to meet a limit that has stopped applying. Background work does
            # still yield on its deadline, and there an oversized budget buys
            # nothing but a decode cut off part-way.
            if not _is_user_facing:
                max_tokens = self._tokens_the_clock_can_deliver(
                    max_tokens, seconds=float(primary_timeout or timeout_val)
                )
            else:
                # Raised to what the turn's own wall clock can pay for, so a
                # thinking turn stops discovering its budget by running out of
                # it. Never lowered: a lane that asked for less meant it.
                # What fits is the TOTAL, and the worker adds its reserve to
                # whatever is asked for here, so the reserve comes off the top
                # rather than going on afterwards. Asking for everything that
                # fits and then having a reserve added to it is how a budget
                # ends up outside the clock that was sized for it.
                _affordable = max(
                    0,
                    self._tokens_the_turn_is_allowed_to_take(
                        seconds=float(primary_timeout or timeout_val or 0.0),
                        prompt_chars=int(prompt_chars or 0),
                            model=_model_for_clock,
                    )
                    - _reserve_the_worker_adds,
                )
                if _affordable > max_tokens:
                    logger.info(
                        "🧠 [ANSWER BUDGET] %d tokens fit this turn's clock at the "
                        "measured rate; raising the ceiling from %d.",
                        _affordable,
                        max_tokens,
                    )
                    max_tokens = _affordable

        serving_lane = self._cortex_serving_lane(
            initial_visible_user_prompt,
            context,
        )
        serving_limits = get_active_cortex_serving_limits()
        if serving_limits is not None and serving_limits.qualified:
            lane_limits = serving_limits.lane(serving_lane)
            if lane_limits is not None:
                admitted_tokens = min(max_tokens, lane_limits.max_output_tokens)
                if admitted_tokens < max_tokens:
                    logger.info(
                        "🧠 [SERVING PROFILE] %s output ceiling reduced %d→%d "
                        "(profile=%s).",
                        serving_lane,
                        max_tokens,
                        admitted_tokens,
                        serving_limits.profile_sha256[:12],
                    )
                max_tokens = max(1, admitted_tokens)
                context["max_tokens"] = max_tokens
                context["cortex_serving_lane"] = serving_lane
                context["cortex_serving_profile_sha256"] = (
                    serving_limits.profile_sha256
                )
                context["cortex_serving_profile_source"] = serving_limits.source
        morpho_kwargs["serving_lane"] = serving_lane

        logger.info(
            "🧭 [GROUNDING] survived to dispatch: %s (sys_prompt=%d)",
            ",".join(_grounded) or "NONE",
            len(str(system_prompt or "")),
        )
        logger.info(
            "🧠 [ZENITH] Prompt plan: mode=%s messages=%d chars=%d "
            "(scaffold=%d request=%d ratio=%.1fx sys_prompt=%d) "
            "origin=%s max_tokens=%d",
            prompt_mode,
            len(messages),
            prompt_chars,
            scaffold_chars,
            request_chars,
            (scaffold_chars / request_chars) if request_chars else float("inf"),
            len(str(system_prompt or "")),
            origin or "unknown",
            max_tokens,
        )
        # What the scaffold IS, when it dwarfs the question.
        #
        # A ratio is a number nobody can act on. Eight thousand characters of
        # scaffold against two hundred and fifty of question is the shape of a
        # real defect, and the log said only that it was thirty-two to one —
        # so which part of it was eight thousand characters could not be found
        # without adding this line first.
        if request_chars and scaffold_chars > (8 * request_chars):
            logger.info(
                "🧠 [ZENITH] Scaffold breakdown: %s",
                "; ".join(
                    f"{str(msg.get('role', '?'))}={len(str(msg.get('content', '') or ''))}"
                    f":{str(msg.get('content', '') or '')[:70]!r}"
                    for msg in messages
                    if isinstance(msg, dict)
                ),
            )

        if (
            _is_user_facing
            and requested_tier == "primary"
            and not strict_answer_contract
            and not strict_value_contract
            and not internal_inference
            # A clean user-surface contract needs a user surface.
            #
            # The worker enforces the contract by validating the draft against
            # the bound question, so switching it on with nothing bound fails
            # every draft for surface_validation_prompt_binding_version — a
            # correct answer rejected for the absence of a question it was
            # never answering.
            and bool(initial_visible_user_prompt)
        ):
            _foreground_floor, _foreground_cap, foreground_loops = (
                self._foreground_compute_profile(initial_visible_user_prompt)
            )
            foreground_profile = self._foreground_prompt_profile(
                visible_user_prompt,
                context,
            )
            # Every _generate_with_client call site passes these EXPLICITLY and also
            # splats **morpho_kwargs, so any overlap is a guaranteed TypeError —
            # "got multiple values for keyword argument" — which fails the
            # inference_gate closed and reaches the person as user_cycle_no_response:
            # the engine returning nothing at all, in two seconds, while ordinary
            # conversation through the same engine keeps working. One added key did
            # exactly that to every desktop turn.
            #
            # Scrubbed here rather than trusted to every future writer: the explicit
            # argument is the authority, and a duplicate in the splat can only ever
            # be the same value or a bug.
            for _reserved in _GENERATE_EXPLICIT_KWARGS:
                morpho_kwargs.pop(_reserved, None)
            morpho_kwargs.setdefault("clean_user_surface_contract", True)
            morpho_kwargs.setdefault(
                "user_surface_validation_prompt",
                initial_visible_user_prompt or visible_user_prompt,
            )
            morpho_kwargs.setdefault(
                "clean_user_surface_recurrent_loops",
                foreground_loops,
            )
            morpho_kwargs.setdefault(
                "clean_user_surface_steering_alpha",
                0.35 if foreground_profile == "extended" else 0.25,
            )
        client_foreground_request = (
            bool(_is_user_facing or explicit_foreground) and not is_background and not benchmark_request
        )
        protected_deep_fallback = False

        # 1. Try the selected local brain.
        if self._mlx_client:
            try:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import (
                    ACTIVE_MODEL,
                    get_brainstem_path,
                    get_deep_model_path,
                    get_fallback_path,
                    get_runtime_model_path,
                )

                local_client = self._mlx_client
                local_label = PRIMARY_ENDPOINT
                fallback_client = None
                fallback_model_path = str(get_brainstem_path())
                fallback_kwargs: dict[str, Any] = {}
                fallback_label = BRAINSTEM_ENDPOINT
                restore_primary = False

                def _ensure_fallback_client():
                    nonlocal fallback_client
                    if fallback_client is None:
                        fallback_client = get_mlx_client(
                            model_path=fallback_model_path,
                            **fallback_kwargs,
                        )
                    return fallback_client

                primary_restored_inline = False
                try:
                    if requested_tier == "tertiary":
                        local_client = get_mlx_client(model_path=str(get_brainstem_path()))
                        local_label = BRAINSTEM_ENDPOINT
                        fallback_model_path = str(get_fallback_path())
                        fallback_kwargs = {"device": "cpu"}
                        fallback_label = FALLBACK_ENDPOINT
                    elif deep_handoff:
                        local_client = get_mlx_client(model_path=str(get_deep_model_path()))
                        local_label = DEEP_ENDPOINT
                        fallback_model_path = str(get_runtime_model_path(ACTIVE_MODEL))
                        fallback_kwargs = {}
                        fallback_label = PRIMARY_ENDPOINT
                        restore_primary = True

                    protected_deep_fallback = bool(
                        bool(context.get("allow_deep_fallback", False))
                        and deep_probe_request
                        and _is_user_facing
                        and requested_tier == "primary"
                    )
                    if protected_deep_fallback:
                        fallback_model_path = str(get_deep_model_path())
                        fallback_kwargs = {}
                        fallback_label = DEEP_ENDPOINT
                    skip_initial_primary_attempt = False
                    primary_warmup_memory_deferred = False
                    lane_managed_client = hasattr(local_client, "get_lane_status") or hasattr(
                        local_client, "warmup"
                    )
                    if _is_user_facing and local_label == PRIMARY_ENDPOINT and lane_managed_client:
                        lane_status = self.get_conversation_status()
                        if not lane_status.get("conversation_ready"):
                            blockers = lane_status.get("readiness_blockers") or []
                            blocker_text = ", ".join(str(item) for item in blockers[:3]) or "conversation probe"
                            logger.info(
                                "🧠 %s lane process state=%s; conversation readiness is blocked by %s. Completing foreground warmup before first generation attempt.",
                                local_label,
                                lane_status.get("state", "unknown"),
                                blocker_text,
                            )
                            try:
                                # Admission control — break the cortex doom-loop.
                                # A COLD first boot legitimately needs ~150s to
                                # load the 32B and the user expects that one-time
                                # wait. But a RECOVERY (Cortex was ready, got
                                # force-killed on a first-token stall, is now
                                # reloading) must NOT block every foreground turn
                                # for 90-180s — that is the observed doom loop
                                # (soak Jul 7: turns 21-30 crawled to 200s+ while
                                # the warm window played out, memory thrashed).
                                # When the lane was EVER ready, cap the preflight
                                # wait short and let this turn fall to the ready
                                # tier while Cortex warms in the background for the
                                # next turn.
                                warmup_timeout = self._foreground_warmup_timeout(
                                    lane_status, primary_timeout
                                )
                                lane_status = await self.ensure_foreground_ready(
                                    timeout=warmup_timeout
                                )
                            except (
                                TimeoutError,
                                RuntimeError,
                                AttributeError,
                                TypeError,
                                ValueError,
                                OSError,
                            ) as warmup_exc:
                                if self._note_foreground_warmup_failure(warmup_exc):
                                    primary_warmup_memory_deferred = True
                                lane_status = self.get_conversation_status()
                            if not self._lane_can_attempt_visible_conversation_turn(lane_status):
                                skip_initial_primary_attempt = True
                                logger.warning(
                                    "🧠 %s is still not ready after foreground preflight warmup (state=%s). Skipping the cold first attempt and waiting for recovery before retry.",
                                    local_label,
                                    lane_status.get("state", "unknown"),
                                )
                    if primary_warmup_memory_deferred:
                        if (
                            proof_evaluation_contract
                            or strict_primary_proof_lane
                            or desktop_cognitive_engine_contract
                        ):
                            logger.warning(
                                "🧠 Proof/evaluation request requires Cortex; refusing Brainstem fallback after primary warmup deferral."
                            )
                            return self._refuse_generation(
                                self.REFUSAL_PROOF_LANE,
                                "primary_warmup_deferred_by_ram_admission",
                                context=context,
                                origin=origin,
                                detail={
                                    "lane": "primary",
                                    "proof_evaluation_contract": bool(proof_evaluation_contract),
                                    "strict_primary_proof_lane": bool(strict_primary_proof_lane),
                                    "desktop_cognitive_engine_contract": bool(
                                        desktop_cognitive_engine_contract
                                    ),
                                },
                            )
                        logger.warning(
                            "🧠 Cortex cold-load deferred by RAM admission; routing this foreground turn to %s.",
                            fallback_label,
                        )
                        fallback_client = _ensure_fallback_client()
                        local_client = fallback_client
                        local_label = fallback_label
                        skip_initial_primary_attempt = False
                    logger.info(
                        "🧠 Routing to %s (timeout=%.0fs, user_facing=%s)...",
                        local_label,
                        float(timeout_val),
                        _is_user_facing,
                    )
                    primary_deadline = get_deadline(
                        self._window_within(request_deadline, primary_timeout)
                    )
                    primary_attempt_started = time.monotonic()
                    tool_grounded = None
                    if _is_user_facing and not skip_initial_primary_attempt:
                        tool_grounded = await self._tool_grounded_answer(
                            local_client,
                            visible=(initial_visible_user_prompt or visible_user_prompt),
                            system_prompt=system_prompt,
                            timeout_s=float(timeout_val),
                            evidence=messages,
                            completed_capability_evidence=context.get(
                                "completed_capability_evidence"
                            ),
                            decode_budget=int(max_tokens or 0),
                            origin=str(origin or ""),
                            allow_tools=(
                                bool(context.get("allow_tools", True))
                                and not bool(
                                    context.get("user_surface_completion_retry", False)
                                )
                                and not bool(
                                    context.get("user_surface_continuation_contract", False)
                                )
                            ),
                        )
                    if tool_grounded:
                        text = tool_grounded
                    elif skip_initial_primary_attempt:
                        text = None
                    else:
                        async with self._resource_context(
                            enabled=local_label != FALLBACK_ENDPOINT,
                            priority=client_foreground_request,
                            worker=local_label,
                            timeout_s=primary_deadline.remaining or primary_timeout,
                        ):
                            text = await self._generate_with_client(
                                local_client,
                                prompt,
                                system_prompt,
                                history,
                                primary_deadline,
                                local_label,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=somatic_temperature,
                                origin=origin,
                                is_background=is_background,
                                foreground_request=client_foreground_request,
                                **morpho_kwargs,
                            )
                    primary_attempt_elapsed = max(
                        0.0,
                        time.monotonic() - primary_attempt_started,
                    )
                    if text:
                        repairable_draft = self._repairable_user_facing_draft_for_downstream(
                            text,
                            visible_user_prompt,
                        ) if _is_user_facing else None
                        if repairable_draft is not None:
                            logger.warning(
                                "🛡️ Preserving repairable Cortex draft for downstream response repair (len=%d).",
                                len(repairable_draft),
                            )
                            stabilized = self._stabilize_user_facing_text(
                                repairable_draft,
                                visible_user_prompt,
                                is_user_facing=True,
                            )
                            # Reachable from the place that decides whether to
                            # refuse, which is not on this call path.
                            try:
                                from core.conversation.surface_disposition import (
                                    preserve_draft,
                                )

                                preserve_draft(stabilized)
                            except (ImportError, RuntimeError, TypeError, ValueError):
                                pass
                            return stabilized
                        return self._stabilize_user_facing_text(
                            text,
                            visible_user_prompt,
                            is_user_facing=_is_user_facing,
                        )
                    primary_failure_metadata = self.get_last_generation_metadata()
                    primary_surface_receipt = self.get_last_surface_control_receipt()
                    primary_surface_quality_rejected = (
                        str(primary_failure_metadata.get("error") or "").strip()
                        == "surface_quality_rejected"
                        or bool(
                            primary_surface_receipt.get("surface_quality_gate_enabled")
                            and not primary_surface_receipt.get("surface_quality_gate_passed")
                            and primary_surface_receipt.get("surface_quality_gate_reasons")
                        )
                    )
                    if primary_surface_quality_rejected and desktop_cognitive_engine_contract:
                        # Say WHICH quality check rejected the text.
                        #
                        # This refusal is the last step before the person gets
                        # "I couldn't get to an answer I'd stand behind", and it
                        # logged only that retries were exhausted. The reasons
                        # were computed by _surface_quality_failure_reasons,
                        # carried on the receipt as surface_quality_gate_reasons,
                        # and written down nowhere: that key appears ZERO times
                        # in a 20,000-record log full of these refusals.
                        #
                        # So the one canned reply that must never be reachable
                        # was also the least diagnosable thing in the runtime —
                        # every occurrence said a gate had said no, and nothing
                        # said what it objected to. The gate keeps only
                        # INTEGRITY failures (leaks, corruption, prompt
                        # artefacts, text that is not language), so the reason
                        # is exactly what distinguishes a model producing
                        # garbage from a gate that is too strict, and those want
                        # opposite fixes.
                        # Every key the receipt keeps a reason under, not one.
                        #
                        # The fix above read surface_quality_gate_reasons, and
                        # the worker writes its actual objections under
                        # semantic_completion_quality_reasons — the first key is
                        # only written on the telemetry-sanitizer path. Two
                        # names for one fact, so the diagnosis that was added to
                        # end "rejected_for=no_reasons_reported" reported
                        # no_reasons_reported.
                        _quality_reasons = tuple(
                            dict.fromkeys(
                                str(reason).strip()[:120]
                                for key in (
                                    "surface_quality_gate_reasons",
                                    "semantic_completion_quality_reasons",
                                    "telemetry_sanitizer_reasons",
                                    # The fourth. The gate that keeps the best
                                    # rejected draft records its objections
                                    # here, and this is the one that carries
                                    # them on the path a simple "read this file
                                    # and tell me what it says" takes.
                                    "surface_quality_rejected_reasons",
                                )
                                for reason in (
                                    primary_surface_receipt.get(key) or ()
                                )
                                if str(reason).strip()
                            )
                        )
                        # And when there are none, the draft itself.
                        #
                        # Four keys hold reasons and a path was found tonight
                        # that populates none of them. A refusal that can name
                        # neither its objection nor what it objected to is the
                        # least diagnosable thing in the runtime, and it sits
                        # one step before the one canned reply that must never
                        # be reachable. The draft is already kept for the
                        # repair path; nothing was reading it here.
                        _rejected_draft = ""
                        if not _quality_reasons:
                            _rejected_draft = str(
                                primary_surface_receipt.get(
                                    "surface_quality_rejected_text"
                                )
                                or ""
                            ).strip()[:220]
                        if not _quality_reasons and not _rejected_draft:
                            # Four keys and a draft, and this receipt has none
                            # of them. Then the question is no longer what the
                            # gate objected to but whether this is the receipt
                            # the gate wrote, and the only way to tell is to
                            # see what it does carry.
                            logger.warning(
                                "🧠 the refusing receipt carries no reasons and no "
                                "draft; it holds: %s",
                                ",".join(
                                    f"{name}={primary_surface_receipt.get(name)!r}"[:90]
                                    for name in sorted(map(str, primary_surface_receipt))
                                    if any(
                                        word in name
                                        for word in (
                                            "quality",
                                            "reason",
                                            "rejected",
                                            "surface",
                                        )
                                    )
                                )
                                or "nothing about quality at all",
                            )
                        logger.warning(
                            "🧠 %s exhausted its worker-owned semantic quality retries; "
                            "preserving the lane and refusing a duplicate inference-gate "
                            "retry. rejected_for=%s%s",
                            local_label,
                            ",".join(str(reason) for reason in _quality_reasons)
                            or "no_reasons_reported",
                            f" draft={_rejected_draft!r}" if _rejected_draft else "",
                        )
                        return self._refuse_generation(
                            self.REFUSAL_EXHAUSTED,
                            "worker_semantic_quality_retries_exhausted",
                            context=context,
                            origin=origin,
                            detail={
                                "lane": local_label,
                                "surface_quality_gate_reasons": list(_quality_reasons),
                            },
                        )
                    if health_probe:
                        logger.warning(
                            "🧠 %s proof health probe returned no text; refusing local fallback for lane certification.",
                            local_label,
                        )
                        return self._refuse_generation(
                            self.REFUSAL_PROOF_LANE,
                            "health_probe_returned_no_text",
                            context=context,
                            origin=origin,
                            detail={"lane": local_label},
                        )
                    if (
                        proof_evaluation_contract
                        or strict_primary_proof_lane
                    ):
                        logger.warning(
                            "🧠 Proof/evaluation request requires a valid Cortex response; refusing retry/fallback cascade after no text."
                        )
                        return self._refuse_generation(
                            self.REFUSAL_PROOF_LANE,
                            "cortex_returned_no_text",
                            context=context,
                            origin=origin,
                            detail={"lane": local_label},
                        )
                    # NOTE: desktop_cognitive_engine_contract is intentionally NOT
                    # refused here. A thin/empty first draft on a live desktop turn
                    # (e.g. the 32B emits a short reply that trips too_thin) must
                    # get one more attempt on the SAME primary Cortex lane — her
                    # real mind — instead of fail-closing immediately. The later
                    # gate below still refuses any LOWER-lane fallback for desktop
                    # turns, so the "real mind only" guarantee is preserved; this
                    # only restores the same-lane retry the early refuse was
                    # skipping (the cause of "I could not produce a reliable
                    # full-mind reply" on casual short turns).

                    # ── CORTEX RETRY: For user-facing requests, retry the primary model
                    # only when the first attempt failed quickly and the lane
                    # remains ready. Long stalls must preserve the remaining
                    # response budget for a governed recovery lane.
                    if _is_user_facing and local_label == PRIMARY_ENDPOINT:
                        retry_schedule = self._foreground_retry_schedule(
                            primary_attempt_elapsed,
                            primary_timeout,
                        )
                        if not retry_schedule:
                            # The primary worker DID own and spend this turn.
                            # Publishing only "no text" made the caller believe
                            # no model owner had run, so it opened a fresh repair
                            # generation after this gate correctly declined one.
                            # Carry the causal fact with the generation receipt;
                            # every downstream surface can then enforce the same
                            # single-owner invariant without reconstructing time
                            # from logs.
                            self._publish_exhausted_primary_owner(
                                primary_attempt_elapsed=primary_attempt_elapsed,
                                same_lane_retry_count=0,
                            )
                            logger.warning(
                                "🧠 %s consumed %.1fs without usable text; skipping repeated "
                                "same-lane retries.",
                                local_label,
                                primary_attempt_elapsed,
                            )
                        for retry_attempt, wait_sec in enumerate(retry_schedule, 1):
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry loop aborted: runtime is shutting down.",
                                    local_label,
                                )
                                return ""
                            lane_status = self.get_conversation_status()
                            if not self._lane_can_attempt_visible_conversation_turn(lane_status):
                                logger.warning(
                                    "🧠 %s is not ready after the failed attempt (state=%s); "
                                    "skipping same-lane retry %d.",
                                    local_label,
                                    lane_status.get("state", "unknown"),
                                    retry_attempt,
                                )
                                break
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry wait skipped: runtime is shutting down.",
                                    local_label,
                                )
                                return ""

                            logger.warning(
                                "🧠 %s returned no text on user-facing request. "
                                "Retrying once after %ds pause...",
                                local_label,
                                wait_sec,
                            )
                            await asyncio.sleep(wait_sec)
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry generation skipped: runtime is shutting down.",
                                    local_label,
                                )
                                return ""

                            # Each repair used to open a fresh 30–60s window of
                            # its own, outside the caller's budget.
                            retry_timeout = self._window_within(
                                request_deadline,
                                min(60.0, max(30.0, primary_timeout * 0.4)),
                            )
                            retry_deadline = get_deadline(
                                self._window_within(request_deadline, retry_timeout)
                            )
                            retry_messages = self._build_primary_repair_messages(
                                visible_user_prompt,
                                messages,
                            )
                            retry_system_prompt = retry_messages[0]["content"]
                            retry_morpho_kwargs = dict(morpho_kwargs)
                            retry_morpho_kwargs.update(
                                {
                                    "disable_prompt_cache": True,
                                    "clear_prompt_cache": retry_attempt == 1,
                                    "top_p": min(float(retry_morpho_kwargs.get("top_p", 0.9) or 0.9), 0.85),
                                    "min_p": max(float(retry_morpho_kwargs.get("min_p", 0.02) or 0.02), 0.02),
                                    "repetition_penalty": max(
                                        float(retry_morpho_kwargs.get("repetition_penalty", 1.1) or 1.1),
                                        1.12,
                                    ),
                                    "repetition_context_size": max(
                                        int(retry_morpho_kwargs.get("repetition_context_size", 64) or 64),
                                        96,
                                    ),
                                    # The runtime TELEMETRY payload is what the
                                    # first attempt drowned in, so it is
                                    # skipped. The turn's evidence is not: it
                                    # travels in the repair messages built
                                    # above, which now carry grounding.
                                    "skip_runtime_payload": True,
                                    "repair_retains_grounding": True,
                                }
                            )
                            retry_temperature = min(
                                float(somatic_temperature if somatic_temperature is not None else 0.35),
                                0.35,
                            )
                            async with self._resource_context(
                                enabled=True,
                                priority=True,
                                worker=local_label,
                                timeout_s=retry_deadline.remaining or retry_timeout,
                            ):
                                text = await self._generate_with_client(
                                    local_client,
                                    prompt,
                                    retry_system_prompt,
                                    history,
                                    retry_deadline,
                                    f"{local_label}-RETRY-{retry_attempt}",
                                    messages=retry_messages,
                                    max_tokens=max_tokens,
                                    temperature=retry_temperature,
                                    origin=origin,
                                    is_background=is_background,
                                    foreground_request=True,
                                    **retry_morpho_kwargs,
                                )
                            if text:
                                logger.info(
                                    "✅ %s retry %d succeeded (len=%d)",
                                    local_label,
                                    retry_attempt,
                                    len(text),
                                )
                                repairable_draft = self._repairable_user_facing_draft_for_downstream(
                                    text,
                                    visible_user_prompt,
                                ) if _is_user_facing else None
                                if repairable_draft is not None:
                                    logger.warning(
                                        "🛡️ Preserving repairable Cortex retry draft for downstream response repair (len=%d).",
                                        len(repairable_draft),
                                    )
                                    return self._stabilize_user_facing_text(
                                        repairable_draft,
                                        visible_user_prompt,
                                        is_user_facing=True,
                                    )
                                return self._stabilize_user_facing_text(
                                    text,
                                    visible_user_prompt,
                                    is_user_facing=_is_user_facing,
                                )

                        if retry_schedule:
                            self._publish_exhausted_primary_owner(
                                primary_attempt_elapsed=primary_attempt_elapsed,
                                same_lane_retry_count=len(retry_schedule),
                            )
                            logger.warning("🧠 %s bounded retry failed.", local_label)
                        if (
                            proof_evaluation_contract
                            or strict_primary_proof_lane
                            or operator_evidence_contract
                            or desktop_cognitive_engine_contract
                        ):
                            logger.warning(
                                "🧠 Proof/operator request requires a valid Cortex response; refusing lower-lane fallback."
                            )
                            return self._refuse_generation(
                                self.REFUSAL_PROOF_LANE,
                                "lower_lane_fallback_refused",
                                context=context,
                                origin=origin,
                                detail={"lane": local_label},
                            )
                        logger.warning(
                            "🧠 %s is still recovering. Falling back to %s for this %s foreground turn.",
                            local_label,
                            fallback_label,
                            "protected" if protected_deep_fallback else "local-only",
                        )
                    else:
                        logger.warning(
                            "🧠 %s returned no text. Trying local fallback.", local_label
                        )
                        if is_background and not bool(
                            context.get("allow_background_local_fallback", False)
                        ):
                            logger.info(
                                "🧠 Background %s request returned no text; suppressing local fallback to protect foreground latency.",
                                local_label,
                            )
                            return self._refuse_generation(
                                self.REFUSAL_DEFERRED,
                                "background_local_fallback_suppressed",
                                context=context,
                                origin=origin,
                                detail={"lane": local_label},
                            )

                    # Graceful local fallback: for background/autonomous requests, the
                    # brainstem is an acceptable degradation. For user-facing requests
                    # that reach here (cloud disabled), it's the last local resort.
                    fallback_deadline = get_deadline(
                        self._window_within(request_deadline, fallback_timeout)
                    )
                    fallback_client = _ensure_fallback_client()
                    async with self._resource_context(
                        enabled=fallback_label != FALLBACK_ENDPOINT,
                        priority=client_foreground_request,
                        worker=fallback_label,
                        timeout_s=fallback_deadline.remaining or fallback_timeout,
                    ):
                        fallback_max_tokens = (
                            max_tokens
                            if fallback_label == DEEP_ENDPOINT
                            else min(max_tokens, 384 if requested_tier != "secondary" else 512)
                        )
                        brainstem_text = await self._generate_with_client(
                            fallback_client,
                            prompt,
                            system_prompt,
                            history,
                            fallback_deadline,
                            fallback_label,
                            messages=messages,
                            max_tokens=fallback_max_tokens,
                            temperature=somatic_temperature,
                            origin=origin,
                            is_background=is_background,
                            foreground_request=client_foreground_request,
                            **morpho_kwargs,
                        )
                    if brainstem_text:
                        if fallback_label == PRIMARY_ENDPOINT:
                            primary_restored_inline = True
                        return self._stabilize_user_facing_text(
                            brainstem_text,
                            visible_user_prompt,
                            is_user_facing=_is_user_facing,
                        )
                    logger.warning("🧠 Local fallback returned no text.")
                finally:
                    if restore_primary and not primary_restored_inline:
                        self._schedule_primary_restore_after_deep_handoff()

            except TimeoutError as timeout_exc:
                logger.warning("🛑 Local inference TIMED OUT (Budget: %.0fs).", timeout_val)
                if (
                    not is_background
                    and self._origin_is_user_facing(origin)
                ):
                    raise TimeoutError(
                        f"{local_label} timed out after {timeout_val:.0f}s"
                    ) from timeout_exc
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                if self._is_expected_inference_backpressure(e):
                    # The runtime declined this tier on purpose (admission,
                    # backoff, gate) and the ladder is descending exactly as
                    # designed. Recording it as a degradation on the
                    # fail-closed inference_gate raises CRITICAL SERVICE
                    # FAILURE for healthy behavior — 52 of them in the
                    # 2026-07-18 soak, which is what tripped
                    # `critical_incident_active` while the runtime served.
                    logger.info(
                        "🧠 Local tier declined by admission/backoff; descending "
                        "the ladder: %s",
                        e,
                    )
                else:
                    record_degradation(
                        "inference_gate",
                        e,
                        severity="degraded",
                        action="fell through to a lower local lane after local inference failure",
                    )
                    logger.warning("🛑 Local inference FAILURE: %s", e)

        # 1.5. EMERGENCY REFLEX FALLBACK — tiny 1.5B model on CPU as absolute last local resort.
        # If Cortex AND Brainstem both failed for a user-facing request, the 1.5B Reflex
        # model can still produce SOMETHING so the user isn't left hanging.
        # [STABILITY v54] Never run the 1.5B reflex if we are in a protected 32B
        # foreground lane.
        #
        # CP126: that sentence described a protection the code did not have.
        # The guard tested protected_deep_fallback, which is a different thing
        # — it marks a deep-probe request that fell back to the deep model —
        # and is False on exactly the turn the comment is about. So a turn
        # that explicitly asked for the deep mind, with both local lanes
        # down, could be answered by a 1.5B CPU model presenting as that
        # mind.
        #
        # protected_foreground_lane is narrow on purpose: it is set only for
        # a user-facing deep-probe turn and for strict_primary_proof_lane
        # (already excluded below). Ordinary conversation never sets it, so
        # this keeps the last-resort reply exactly where it earns its keep —
        # a normal turn with no other local option — and withholds it where
        # the answer would misrepresent which mind produced it.
        if (
            _is_user_facing
            and not is_background
            and not protected_deep_fallback
            and not protected_foreground_lane
            and not proof_evaluation_contract
            and not desktop_cognitive_engine_contract
            # Strict/operator/proof contracts refuse lower-lane fallback
            # everywhere else — the emergency reflex must not become the one
            # route where an unproven 1.5B model answers a strict contract.
            and not strict_answer_contract
            and not strict_value_contract
            and not operator_evidence_contract
            and not strict_primary_proof_lane
        ):
            try:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import get_fallback_path

                reflex_client = get_mlx_client(model_path=str(get_fallback_path()), device="cpu")
                if reflex_client:
                    logger.warning(
                        "🆘 [REFLEX] Cortex + Brainstem both failed. Trying 1.5B CPU Reflex..."
                    )
                    # 15s hard limit for the tiny model, and never more than
                    # what the caller still has.
                    reflex_deadline = get_deadline(
                        self._window_within(request_deadline, 15.0)
                    )
                    reflex_text = await self._generate_with_client(
                        reflex_client,
                        prompt,
                        system_prompt,
                        history[-2:] if history else [],  # minimal history for tiny model
                        reflex_deadline,
                        FALLBACK_ENDPOINT,
                        messages=None,
                        max_tokens=min(max_tokens, 200),  # keep it short
                        temperature=somatic_temperature,
                        origin=origin,
                        is_background=False,
                        foreground_request=True,
                        **morpho_kwargs,
                    )
                    if reflex_text:
                        logger.info(
                            "🆘 [REFLEX] 1.5B CPU model produced response. Cortex recovery in background."
                        )
                        if not self._cortex_recovery_in_progress:
                            get_task_tracker().create_task(self._ensure_cortex_recovery())
                        return self._stabilize_user_facing_text(
                            reflex_text,
                            visible_user_prompt,
                            is_user_facing=True,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as reflex_err:
                record_degradation(
                    "inference_gate",
                    reflex_err,
                    severity="warning",
                    action="continued to configured cloud or exhaustion path after reflex fallback failed",
                )
                logger.debug("Reflex fallback failed: %s", reflex_err)

        return await self._finish_local_inference_exhaustion(
            proof_evaluation_contract=proof_evaluation_contract,
            desktop_cognitive_engine_contract=desktop_cognitive_engine_contract,
            is_user_facing=_is_user_facing,
            visible_user_prompt=visible_user_prompt,
            context=context,
            origin=origin,
            max_tokens=max_tokens,
            output_contract_payload=output_contract_payload,
        )

    async def _finish_local_inference_exhaustion(
        self,
        *,
        proof_evaluation_contract: bool,
        desktop_cognitive_engine_contract: bool,
        is_user_facing: bool,
        visible_user_prompt: str,
        context: dict[str, Any] | None,
        origin: str,
        max_tokens: int,
        output_contract_payload: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        """Recover the resident cortex or return a typed local-only refusal."""

        logger.error("Local inference paths exhausted; scheduling resident-cortex recovery.")
        if proof_evaluation_contract or desktop_cognitive_engine_contract:
            return self._refuse_generation(
                self.REFUSAL_EXHAUSTED,
                "resident_cortex_exhausted",
                context=context,
                origin=origin,
            )

        if not is_user_facing:
            return self._refuse_generation(
                self.REFUSAL_EXHAUSTED,
                "local_inference_lanes_exhausted",
                context=context,
                origin=origin,
            )

        if self._mlx_client and hasattr(self._mlx_client, "_process"):
            try:
                async with _thread_lock_context(
                    self._foreground_ready_lock,
                    timeout_s=5.0,
                    label="cascade_cleanup",
                ):
                    await asyncio.to_thread(self._cascade_cleanup_stuck_worker_locked)
            except TimeoutError:
                logger.info(
                    "[CASCADE CLEANUP] Foreground lane is owned elsewhere; "
                    "skipping worker cleanup this turn."
                )
            except _INFERENCE_RECOVERABLE_ERRORS as cleanup_exc:
                record_degradation(
                    "inference_gate",
                    cleanup_exc,
                    severity="warning",
                    action="continued local cortex recovery scheduling after cleanup error",
                )

        if not self._cortex_recovery_in_progress:
            recovery_coro = self._respawn_cortex_if_needed()
            task = get_task_tracker().create_task(recovery_coro)
            if not isinstance(task, asyncio.Task):
                recovery_coro.close()
        self._extend_startup_quiet_window(15.0)

        try:
            from core.resilience.error_boundary import CircuitRegistry
            from core.utils.resilience import CircuitState

            breaker = CircuitRegistry.get_instance().get_breaker(
                "phase:UnitaryResponsePhase"
            )
            if breaker.state != CircuitState.CLOSED and breaker.request_probe(
                reason="inference_gate_cortex_recovery",
                requested_timeout=15.0,
            ):
                logger.info("Reset UnitaryResponsePhase circuit to HALF_OPEN for recovery")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Circuit-breaker recovery reset unavailable: %s", exc)

        # An internal caller gets nothing, not an apology written for a person.
        #
        # This text exists so somebody waiting on a reply is told honestly
        # that the language backend is down. Handed to a deliberation it
        # becomes the answer, and the answer becomes the reasoning: measured
        # live, a move was narrated as "Board: Right — I can't work through
        # that right now, my language backend is temporarily unavailable".
        # She pressed right for reasons that were really a status message.
        #
        # Nothing back is the truthful result for a caller that is not a
        # person: it falls through to deciding from evidence and says so.
        if not is_user_facing or bool((context or {}).get("internal_inference")):
            return ""
        recovery_text = self._user_facing_recovery_response(visible_user_prompt)
        return self._finalize_nonlocal_user_facing_text(
            recovery_text,
            visible_user_prompt,
            is_user_facing=True,
            label="resident-cortex-recovery",
            max_tokens=max_tokens,
            output_contract=output_contract_payload,
        )

    def _cascade_cleanup_stuck_worker_locked(self) -> None:
        """Kill a genuinely wedged cortex worker and sever its IPC queues.

        MUST be called while holding the foreground-ready lock: it reads and
        mutates the client's private process, queue, and init state, and the
        legitimacy check is only valid while no concurrent warmup/generation
        can change that state underneath it.

        A worker actively LOADING the 20GB model is running but NOT stuck —
        killing it there was the doom loop that starved the cortex for a full
        hour (2026-07-15 soak: spawn → load → killed mid-warmup on the next
        turn → warmup_deferred → repeat, 216s/turn, zero real cortex answers).
        Only genuinely wedged workers get killed: warmup NOT in flight
        (idle-but-running = a hung generation, the original nwait bug) OR
        warmup in flight but past a generous load deadline.
        """
        client = self._mlx_client
        if client is None or not hasattr(client, "_process"):
            return
        proc = client._process
        is_running = _worker_process_is_running(proc)
        if self._cortex_worker_is_legitimately_loading(client):
            logger.info(
                "⏳ [CASCADE CLEANUP] Cortex worker pid=%s is warming; "
                "NOT killing a loading model.",
                getattr(proc, "pid", "unknown"),
            )
            return
        if proc and is_running:
            # A worker that was still warming/recovering when it overran the
            # load deadline is a stuck LOAD (thermal / GPU-starved), not the
            # idle-but-wedged nwait case — only stuck loads feed the
            # warmup-backoff so the cortex stops thrashing the GPU the
            # fallback needs.
            if self._cortex_worker_is_actively_generating(client):
                logger.info(
                    "🛡️ [CASCADE CLEANUP] Not killing pid=%s: the worker is "
                    "actively generating. A slow answer is not a wedged lane, "
                    "and killing it costs a cold reload the next turn pays for.",
                    getattr(proc, "pid", "unknown"),
                )
                return
            was_stuck_load = bool(
                getattr(client, "_warmup_in_flight", False)
            ) or str(getattr(client, "_lane_state", "")) in {
                "warming",
                "recovering",
            }
            cleanup_identity = capture_identity(proc, label="cascade_cleanup")
            if not assert_owned(
                cleanup_identity,
                getattr(client, "_process", None),
                action="stuck-worker force-kill",
                subsystem="inference_gate.cascade_cleanup",
            ):
                # The handle stopped being the client's current process
                # between the liveness check above and here.
                return
            logger.warning(
                "🧹 [CASCADE CLEANUP] Force-killing stuck cortex worker %s",
                cleanup_identity.describe() if cleanup_identity else "pid=unknown",
            )
            proc.kill()
            if hasattr(proc, "join"):
                proc.join(timeout=2.0)
            elif hasattr(proc, "wait"):
                proc.wait(timeout=2.0)
            # Re-check exit before clearing ownership — a process that
            # ignored the kill must not become an untracked orphan while a
            # replacement is scheduled.
            if _worker_process_is_running(proc):
                logger.error(
                    "🧟 [CASCADE CLEANUP] Worker pid=%s survived kill+join; "
                    "keeping ownership so it cannot orphan.",
                    getattr(proc, "pid", "unknown"),
                )
                return
            if was_stuck_load:
                self._note_cortex_stuck_kill()
        if hasattr(client, "_drain_queue"):
            client._drain_queue()
        # Replace queues to sever any stuck feeder threads.
        replace_queues = getattr(client, "_replace_ipc_queues", None)
        if callable(replace_queues):
            replace_queues()
        client._process = None
        client._init_done = False
        logger.info("🧹 [CASCADE CLEANUP] Stuck worker killed, queues replaced.")

    def _post_inference_update(self, response_text: str):
        """Update downstream systems after each inference completes.

        Closes the bidirectional causal loop:
          CRSM ← response (updates self-state)
          HOT  ← response (reflexive modification)
          Hedonic ← response quality signal
        """
        # An empty response is NOT a success — recording a fresh success
        # timestamp for it would feed health and recovery logic false
        # evidence that generation is working.
        if not response_text or not response_text.strip():
            return
        self._last_successful_generation_at = time.time()

        # Nine downstream systems are updated below, each in its own
        # fail-open block. A mid-sequence failure therefore leaves a PARTIAL
        # update — some subsystems advanced by this response, others not —
        # and every one of those blocks used to record the identical action
        # string, so the degradation trail could not even say which hook was
        # skipped, let alone that they belonged to one turn.
        #
        # This does NOT add rollback, and claiming it would be the overclaim
        # this pass exists to remove: CRSM, the world model and synaptic
        # plasticity are not transactional stores, and there is nothing to
        # roll back TO. What it adds is legibility — one id shared by every
        # record from this response, a stage name on each, and a receipt
        # naming exactly which stages applied and which did not, so a partial
        # update is a fact somebody can find rather than an invisible one.
        _update_id = f"pi_{uuid.uuid4().hex[:12]}"
        _skipped_stages: list[str] = []
        try:
            from core.consciousness.crsm import get_crsm

            get_crsm().post_inference_update(response_text)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage crsm_self_state after response delivery",
                extra={"stage": "crsm_self_state", "update_id": _update_id},
            )
            _skipped_stages.append("crsm_self_state")
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.consciousness.hot_engine import get_hot_engine

            hot = get_hot_engine()
            hot.apply_feedback()  # apply any pending reflexive modifications
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage hot_reflexive_feedback after response delivery",
                extra={"stage": "hot_reflexive_feedback", "update_id": _update_id},
            )
            _skipped_stages.append("hot_reflexive_feedback")
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient
            from core.container import ServiceContainer

            hg = get_hedonic_gradient()
            _v, _a, _c, _e = 0.0, 0.5, 0.5, 0.7
            _circ2 = ServiceContainer.get("affective_circumplex", default=None)
            if _circ2 and hasattr(_circ2, "_sample_raw_axes"):
                _v, _a = _circ2._sample_raw_axes()
            _ls2 = ServiceContainer.get("liquid_state", default=None)
            if _ls2 and hasattr(_ls2, "get_status"):
                _lsd2 = _ls2.get_status()
                _c = float(_lsd2.get("curiosity", 50)) / 100.0
                _e = float(_lsd2.get("energy", 70)) / 100.0
            hg.update(valence=_v, arousal=_a, curiosity=_c, energy=_e)
            # LoRA Bridge: complete the post-inference capture
            try:
                from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge

                get_crsm_lora_bridge().post_inference_capture(
                    response_text=response_text,
                    hedonic_after=hg.score,
                )
            except _INFERENCE_RECOVERABLE_ERRORS as _exc:
                _record_inference_degradation(
                    _exc,
                    action="skipped post-inference stage hedonic_and_lora after response delivery",
                    extra={"stage": "hedonic_and_lora", "update_id": _update_id},
                )
                _skipped_stages.append("hedonic_and_lora")
                logger.debug("Suppressed Exception: %s", _exc)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            # The OUTER handler of the nested hedonic/LoRA block. It kept the
            # generic action string after the inner one was named, so a
            # failure to reach the hedonic gradient AT ALL — the more severe
            # case, since the LoRA capture never even ran — was the one
            # record that still could not say which stage it belonged to.
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage hedonic_and_lora after response delivery",
                extra={"stage": "hedonic_and_lora", "update_id": _update_id},
            )
            _skipped_stages.append("hedonic_and_lora")
            logger.debug("Suppressed Exception: %s", _exc)

        # ══════════════════════════════════════════════════════════════════
        # DEEPENED POST-INFERENCE FEEDBACK LOOPS
        # ══════════════════════════════════════════════════════════════════

        # ── Credit Assignment: Record response quality ────────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit:
                outcome, basis = self._response_credit_outcome()
                if outcome is None:
                    # Nothing verified this answer. The previous signal was
                    # length plus the presence of a newline or a list marker,
                    # which rewards a long hallucination and penalizes a
                    # correct one-line answer — the learner was being taught
                    # shape and told it was correctness. Withholding is the
                    # honest move: an unmeasured turn is not a graded turn.
                    _credit_basis = basis
                else:
                    _credit_basis = basis
                    credit.assign_credit(
                        # time_ns + counter keeps concurrent same-second
                        # responses attributable instead of colliding on one id.
                        action_id=f"inference_{time.time_ns()}_{self._credit_action_seq()}",
                        outcome=outcome,
                        domain="chat",
                    )
                self._last_credit_basis = _credit_basis
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage credit_assignment after response delivery",
                extra={"stage": "credit_assignment", "update_id": _update_id},
            )
            _skipped_stages.append("credit_assignment")
            logger.debug("Suppressed Exception in credit feedback: %s", _exc)

        # ── Homeostasis: Response success signal ──────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "on_response_success"):
                # A nonempty string was the whole test, so a canned recovery
                # message off a fallback lane raised her sense that she was
                # working correctly. What is actually knowable at this point
                # is the lane that answered and whether anything verified it;
                # both now travel with the signal.
                homeostasis.on_response_success(
                    response_length=len(response_text),
                    verified=self._response_credit_outcome()[0] is not None,
                    fallback=bool(
                        getattr(self, "_last_user_generation_used_fallback", False)
                    ),
                    endpoint=str(
                        getattr(self, "_last_user_generation_endpoint", "") or ""
                    ),
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage homeostasis after response delivery",
                extra={"stage": "homeostasis", "update_id": _update_id},
            )
            _skipped_stages.append("homeostasis")
            logger.debug("Suppressed Exception in homeostasis feedback: %s", _exc)

        # ── World Model: Extract beliefs from response ────────────────────
        #
        # CP126 (critical): "Any response longer than 100 characters is fed
        # back into epistemic state without citations, calibration status,
        # tool receipts, source separation, contradiction checks, or a marker
        # that the text was model-generated. Hallucinations can become future
        # grounding."
        #
        # The world model side is handled: extractions are stamped
        # source="self_generated", capped at low confidence, and run through
        # dissonance resolution. What was missing is here — length was the
        # ONLY gate, so text this class wrote itself qualified: the terminal
        # recovery message, an unattributed cloud string, a reply the
        # reliability assessment had already rejected. Beliefs were being
        # extracted from Aura's apologies.
        #
        # A generation receipt now has to say the text came from a real,
        # attributed, successful generation before any of it becomes belief.
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "extract_beliefs_from_response"):
                generation = self.get_last_generation_metadata()
                verified_generation = bool(
                    generation
                    and generation.get("ok")
                    and generation.get("attributed", True)
                )
                if len(response_text) > 100 and verified_generation:
                    world_model.extract_beliefs_from_response(response_text)
                elif len(response_text) > 100:
                    logger.debug(
                        "Skipped belief extraction: no verified generation receipt "
                        "(ok=%s attributed=%s)",
                        generation.get("ok") if generation else None,
                        generation.get("attributed") if generation else None,
                    )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage world_model after response delivery",
                extra={"stage": "world_model", "update_id": _update_id},
            )
            _skipped_stages.append("world_model")
            logger.debug("Suppressed Exception in world model feedback: %s", _exc)

        # ── Synaptic Plasticity: Post-inference Hebbian update ────────────
        #
        # The reward is her own hedonic score plus her own CRSM surprise. That
        # is a closed loop: the thing being rewarded and the thing giving the
        # reward are the same system, so it can learn to feel good about an
        # answer without the answer being any good. The comment used to call
        # this "true online learning", which named the ambition, not the
        # mechanism.
        #
        # Two changes make it honest rather than removing it. Missing signals
        # no longer default to 0.0 — a subsystem that could not report is
        # UNKNOWN, and an update with no reward evidence does not run at all,
        # because a Hebbian step on a fabricated zero is still a step. And when
        # the turn ledger holds an external verdict, that outranks the internal
        # feeling: it is the only signal here that came from outside her.
        try:
            _plasticity = ServiceContainer.get("synaptic_plasticity", default=None)
            if _plasticity is not None:
                _hg_score: float | None = None
                _surprise: float | None = None
                try:
                    from core.consciousness.hedonic_gradient import get_hedonic_gradient
                    _hg_score = _finite(get_hedonic_gradient().score)
                except _INFERENCE_RECOVERABLE_ERRORS as _hedonic_exc:
                    _record_inference_degradation(
                        _hedonic_exc,
                        action="continued synaptic plasticity post-inference update without hedonic score",
                    )
                    logger.debug(
                        "SynapticPlasticity post-inference hedonic score unavailable: %s",
                        _hedonic_exc,
                    )
                try:
                    from core.consciousness.crsm import get_crsm
                    _crsm = get_crsm()
                    _surprise = _finite(getattr(_crsm, "surprise", None))
                except _INFERENCE_RECOVERABLE_ERRORS as _crsm_exc:
                    _record_inference_degradation(
                        _crsm_exc,
                        action="continued synaptic plasticity post-inference update without CRSM surprise",
                    )
                    logger.debug(
                        "SynapticPlasticity post-inference CRSM surprise unavailable: %s",
                        _crsm_exc,
                    )
                _external, _basis = self._response_credit_outcome()
                if _hg_score is None and _surprise is None and _external is None:
                    # No reward evidence of any kind. Defaulting both to 0.0
                    # and updating anyway taught the weights from a number
                    # nobody produced.
                    _skipped_stages.append("synaptic_plasticity")
                    logger.debug(
                        "Synaptic plasticity update skipped: no reward evidence "
                        "(hedonic and surprise unavailable, turn %s).",
                        _basis,
                    )
                else:
                    _plasticity.post_inference_learn(
                        response_text=response_text,
                        # An external verdict outranks the internal feeling.
                        hedonic_after=(
                            _external if _external is not None else (_hg_score or 0.0)
                        ),
                        surprise=_surprise or 0.0,
                    )
                    self._last_plasticity_reward = {
                        "external_grade": _external,
                        "grade_basis": _basis,
                        "hedonic": _hg_score,
                        "surprise": _surprise,
                        "at": time.time(),
                    }
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage synaptic_plasticity after response delivery",
                extra={"stage": "synaptic_plasticity", "update_id": _update_id},
            )
            _skipped_stages.append("synaptic_plasticity")
            logger.debug("Suppressed Exception in plasticity feedback: %s", _exc)

        # ── Temporal Continuity: Reset silence accumulator ────────────────
        # The inference just completed — reset the temporal residue so the
        # next silence period starts accumulating from a fresh anchor.
        try:
            _tc = ServiceContainer.get("temporal_continuity", default=None)
            if _tc is not None:
                _tc.on_inference_complete()
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped post-inference stage temporal_continuity after response delivery",
                extra={"stage": "temporal_continuity", "update_id": _update_id},
            )
            _skipped_stages.append("temporal_continuity")
            logger.debug("Suppressed Exception in temporal continuity reset: %s", _exc)

        # The commit receipt. Every stage above is fail-open by design — a
        # response has already reached the person and no downstream update is
        # worth taking that back — but "some of the nine advanced and some
        # did not" was previously spread across up to nine unrelated
        # degradation records with no way to join them.
        #
        # Kept on the gate and reported once when the sequence was partial,
        # so a run of half-updates shows up as a rate rather than as nine
        # anecdotes nobody connects.
        self._last_post_inference_receipt = {
            "update_id": _update_id,
            "at": time.time(),
            "stages": list(_POST_INFERENCE_STAGES),
            "skipped": list(_skipped_stages),
            "complete": not _skipped_stages,
        }
        if _skipped_stages:
            _record_inference_degradation(
                RuntimeError(
                    "post-inference update was partial: "
                    + ", ".join(sorted(_skipped_stages))
                ),
                action=(
                    "left downstream systems in a partial post-inference state; "
                    "no rollback exists for these subsystems"
                ),
                severity="warning",
                extra=dict(self._last_post_inference_receipt),
            )

    def post_inference_receipt(self) -> dict[str, Any]:
        """The last post-inference update receipt, for the health report.

        Empty until a response has been delivered. `complete` is the field
        that matters: False means this turn advanced some downstream systems
        and not others.
        """
        return dict(getattr(self, "_last_post_inference_receipt", {}) or {})

    async def think(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        _generation_metadata_sink: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | None:
        """Bind per-request evidence transport around the unified think path."""

        sink_slot = self._generation_metadata_sink_slot()
        sink_token = sink_slot.set(
            _generation_metadata_sink
            if isinstance(_generation_metadata_sink, dict)
            else None
        )
        try:
            return await self._think_with_generation_metadata_sink(
                prompt,
                system_prompt,
                _generation_metadata_sink=_generation_metadata_sink,
                **kwargs,
            )
        finally:
            sink_slot.reset(sink_token)

    async def _think_with_generation_metadata_sink(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        _generation_metadata_sink: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | None:
        """Unified thinking interface for cognitive components.

        Preserve standard LLM adapter semantics:
        - explicit ``messages`` stay as passthrough chat messages
        - ``system_prompt`` is treated as a real system prompt by default
        - callers that truly mean "brief" can pass ``brief=...`` or
          ``system_prompt_is_brief=True``
        """
        timeout = kwargs.pop("timeout", None)
        brief = kwargs.pop("brief", None)
        system_prompt_is_brief = bool(kwargs.pop("system_prompt_is_brief", False))
        provided_messages = kwargs.get("messages")
        if (
            provided_messages is not None
            and system_prompt
            and not system_prompt_is_brief
        ):
            system_text = str(system_prompt or "").strip()
            if system_text:
                merged_messages: list[Any] = []
                inserted_system = False
                for raw_msg in provided_messages if isinstance(provided_messages, list) else []:
                    msg = dict(raw_msg) if isinstance(raw_msg, dict) else raw_msg
                    if (
                        isinstance(msg, dict)
                        and str(msg.get("role", "") or "").strip().lower() == "system"
                        and not inserted_system
                    ):
                        existing = str(msg.get("content", "") or "").strip()
                        if existing == system_text or existing.startswith(f"{system_text}\n\n"):
                            msg["content"] = existing
                        elif existing:
                            msg["content"] = f"{system_text}\n\n{existing}"
                        else:
                            msg["content"] = system_text
                        inserted_system = True
                    merged_messages.append(msg)
                if not inserted_system:
                    merged_messages.insert(0, {"role": "system", "content": system_text})
                provided_messages = merged_messages

        context: dict[str, Any] = {}
        if provided_messages is not None:
            context["messages"] = provided_messages
        elif brief is not None:
            context["brief"] = brief
        elif system_prompt and not system_prompt_is_brief:
            context["messages"] = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(prompt or "")},
            ]
        else:
            context["brief"] = system_prompt

        # CP126 (critical x3): "Unauthenticated context flags control proof,
        # foreground and model-tier policy"; "Caller context is
        # copied into provider policy and proof kwargs without validation.
        # No per-key type schema, authority source, unknown-field
        # rejection"; "The public think interface forwards policy-sensitive
        # kwargs without authority validation."
        #
        # The list this replaces was an allowlist of NAMES with no types, and
        # every flag downstream is read with a bare truthiness test. So
        # policy_flag="false" — a perfectly ordinary thing to get from a
        # config file or a JSON body — could enable a policy, and
        # proof_primary_lane_required="no" required it. A caller got the
        # opposite of what it asked for and nothing reported it.
        #
        # core.brain.request_contract declares what each key is, rejects what
        # cannot be honestly coerced (rather than guessing), and names the
        # policy-bearing subset so the authority binding that follows has a
        # fixed target instead of a rediscovery exercise.
        validation = validate_request_context(
            {key: value for key, value in kwargs.items() if key in REQUEST_FIELDS}
        )
        context.update(validation.context)
        if validation.rejected:
            _record_inference_degradation(
                ValueError(
                    "rejected malformed request context fields: "
                    + ", ".join(
                        f"{key} ({why})" for key, why in sorted(validation.rejected.items())
                    )
                ),
                action="dropped malformed request context fields and used gate defaults",
                severity="warning",
            )
        unknown_keys = sorted(set(kwargs) - set(REQUEST_FIELDS) - _THINK_LOCAL_KWARGS)
        if unknown_keys:
            logger.debug(
                "think() ignored undeclared kwargs: %s", ", ".join(unknown_keys)
            )
        result = await self.generate(prompt, context=context, timeout=timeout)
        if isinstance(_generation_metadata_sink, dict):
            # The health router executes an endpoint call inside its own
            # ``asyncio.wait_for`` task. ContextVars intentionally do not flow
            # back from that child task, so a receipt read by the router after
            # the await is empty even though this exact request produced one.
            # Publish into a caller-owned mutable object while we are still in
            # the request task; this is evidence transport, not process-wide
            # "last call" telemetry.
            task_metadata = self.get_last_generation_metadata()
            if task_metadata:
                _generation_metadata_sink.clear()
                _generation_metadata_sink.update(task_metadata)
        if isinstance(result, str) and result.strip():
            # Close the bidirectional causal loop AFTER the answer is on its
            # way, not in front of it.
            #
            # _post_inference_update is synchronous and touches nine
            # subsystems — CRSM, HOT, the hedonic gradient, credit assignment,
            # homeostasis, the world model, synaptic plasticity, temporal
            # continuity. Run inline here it sat between the model finishing
            # and this function returning, so every one of those hooks was
            # added to the person's wait, on the event loop, blocking every
            # other turn in flight. None of it changes the answer.
            await self._schedule_post_inference_update(result)
            return result
        return None

    async def _schedule_post_inference_update(self, response_text: str) -> None:
        """Run the post-inference hooks off the response path.

        A thread, not a task: the hooks are synchronous and CPU/IO-bound, so a
        task would still occupy the loop. Awaited only far enough to hand it
        over, and failures are recorded rather than raised — the response has
        already been produced and no bookkeeping is worth taking it back.
        """
        try:
            await asyncio.to_thread(self._post_inference_update, response_text)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued after the post-inference update failed off the response path",
            )

    def liveness_state(self) -> str:
        """Name what "alive" is standing on, rather than collapsing it to a bool.

        Three states, and the difference between the first two is the whole
        point: ``backend_live`` means a worker process answered; ``deferred``
        means no worker is running and policy says one will start on the first
        request. Both used to return the same True from ``is_alive()``, so a
        consumer could certify an operational inference gate while nothing was
        accepting work. ``down`` is neither.
        """
        if not self._initialized:
            return "uninitialized"
        try:
            if (
                self._mlx_client is not None
                and hasattr(self._mlx_client, "is_alive")
                and self._mlx_client.is_alive()
            ):
                return "backend_live"
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="treated the primary client as not live during a liveness probe",
            )
        try:
            if self._desktop_safe_boot_enabled() or self._boot_should_schedule_deferred_prewarm():
                return "deferred"
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Fail CLOSED: an unreadable boot policy is not evidence that a
            # worker will start on demand.
            _record_inference_degradation(
                exc,
                action="treated deferred-start policy as unavailable during a liveness probe",
            )
        return "down"

    def is_alive(self) -> bool:
        """The gate can serve a request, possibly after a cold start.

        This is deliberately weaker than :meth:`is_inference_ready`. It is true
        under deferred/safe boot, when no worker is running yet. Anything that
        needs a backend that can accept a generation NOW must call
        ``is_inference_ready()``; anything that needs to tell the two apart
        must call ``liveness_state()``. The runtime health contract
        (``core/runtime/health_contract.py``) binds to the strict one.
        """
        return self.liveness_state() in {"backend_live", "deferred"}

    @staticmethod
    def _active_generation_is_progressing(lane: Any) -> bool:
        """Treat bounded, observable foreground work as operational inference.

        A client cannot advertise ``conversation_ready`` while it owns the
        foreground generation lock. That is backpressure, not backend failure.
        Health may accept the in-flight request only while its start and
        token-progress timestamps prove it has not stalled.

        Every number here comes off a payload another process wrote, so each
        one goes through :func:`_finite` / :func:`_elapsed_since`: a NaN, an
        infinity, or a timestamp from the future is missing evidence, not a
        fresh age of zero, and a bad payload must return False rather than
        raise out of a health endpoint.
        """
        if not isinstance(lane, dict):
            return False
        active = _finite(lane.get("active_generations"), 0.0) or 0.0
        if active <= 0:
            return False
        if not bool(lane.get("foreground_owned", False)):
            return False

        now = time.time()
        request_age_s = _elapsed_since(lane.get("current_request_started_at"), now=now)
        if request_age_s is None:
            return False

        startup_grace_s = _health_window_s(
            _FLAG_INFERENCE_ACTIVE_STARTUP_GRACE_S.value(), default=120.0, minimum=15.0
        )
        progress_stale_s = _health_window_s(
            _FLAG_INFERENCE_ACTIVE_PROGRESS_STALE_S.value(), default=45.0, minimum=10.0
        )

        raw_token_progress = lane.get("last_token_progress_at")
        if raw_token_progress not in (None, 0, 0.0, ""):
            # The field is PRESENT. If it is also unusable — NaN, a future
            # stamp, a string — the payload is corrupt, and falling back to the
            # weaker request-age check would let a lane that just started a
            # stalled generation read as progressing. Absent is "no token yet";
            # present-and-unreadable is "do not trust this lane".
            token_age_s = _elapsed_since(raw_token_progress, now=now)
            if token_age_s is None:
                return False
            return token_age_s <= progress_stale_s
        return request_age_s <= startup_grace_s

    @staticmethod
    def _client_backend_alive(client: Any) -> bool:
        try:
            return bool(
                client is not None
                and hasattr(client, "is_alive")
                and client.is_alive()
            )
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    def inference_readiness(self) -> tuple[bool, str]:
        """Readiness with the reason attached, so a False can be diagnosed.

        :meth:`is_inference_ready` is the boolean the health contract binds to;
        this is the same decision with the evidence that produced it.
        """
        if not self._initialized:
            return False, "gate_not_initialized"

        # ── one proof-policy decision, read once ────────────────────────────
        # It used to be read at entry and then read AGAIN further down. A
        # config change or a transient error between the two reads mixed proof
        # rules with ordinary rules inside a single answer.
        proof_active = False
        proof_tier = ""
        proof_policy_unknown = False
        try:
            from core.runtime.proof_policy import proof_model_tier, proof_run_active

            proof_active = bool(proof_run_active(origin="inference_gate_health"))
            proof_tier = str(proof_model_tier() or "") if proof_active else ""
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Fail CLOSED: an unreadable proof policy must not silently grant
            # the permissive readiness paths below.
            proof_policy_unknown = True
            _record_inference_degradation(
                exc,
                action="treated proof policy as unknown during inference readiness check",
            )
            logger.debug("Inference readiness proof-policy check unavailable: %s", exc)

        # NOTE: deferred/safe-boot policy deliberately does NOT satisfy this
        # contract. ``is_alive()`` covers "the gate can cold-start on demand";
        # inference-READY requires a concrete live backend, proven below.
        #
        # There used to be a shortcut here: on a non-proof run, an alive
        # primary process returned True before lane status was read at all. A
        # wedged, still-loading, or non-conversation-capable worker satisfied
        # readiness on process liveness alone. The lane read below is the same
        # cost and strictly more evidence, so the shortcut is gone.
        if self._client_backend_alive(self._mlx_client):
            try:
                lane = self.get_conversation_status()
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="reported inference not ready after the lane status read failed",
                )
                return False, "primary_lane_status_unreadable"
            if not isinstance(lane, dict):
                return False, "primary_lane_status_malformed"
            if bool(lane.get("conversation_ready", False)):
                return True, "primary_conversation_ready"
            if self._active_generation_is_progressing(lane):
                return True, "primary_generation_progressing"

        if proof_policy_unknown:
            return False, "proof_policy_unknown"
        if proof_active and proof_tier == "primary":
            # A proof-primary run is answered by Cortex or not at all; a lower
            # tier answering would silently change what the run measured.
            return False, "proof_primary_requires_cortex"

        try:
            local_clients = self._iter_local_clients()
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="reported inference not ready after enumerating local clients failed",
            )
            return False, "local_clients_unenumerable"

        unproven: list[str] = []
        for name, client in local_clients.items():
            if not self._client_backend_alive(client):
                continue
            get_lane_status = getattr(client, "get_lane_status", None)
            if not callable(get_lane_status):
                # This used to return True. A client with no way to report its
                # lane cannot show a loaded model, a queue, or a conversation
                # contract — the ABSENCE of the check was being counted as a
                # passed check. It is now unproven, and it is named.
                unproven.append(str(name))
                continue
            try:
                lane = get_lane_status()
            except _INFERENCE_RECOVERABLE_ERRORS:
                continue
            if isinstance(lane, dict) and bool(lane.get("conversation_ready", False)):
                return True, f"fallback_conversation_ready:{name}"
            if self._active_generation_is_progressing(lane):
                return True, f"fallback_generation_progressing:{name}"

        if unproven:
            _record_inference_degradation(
                RuntimeError(
                    "live local clients cannot report lane status: " + ", ".join(sorted(unproven))
                ),
                action="reported inference not ready because no live client could prove readiness",
                extra={"unproven_clients": sorted(unproven)},
            )
            return False, "no_client_can_prove_readiness"
        return False, "no_live_backend"

    def is_inference_ready(self) -> bool:
        """Return true only when a concrete inference backend is live now.

        ``is_alive()`` intentionally supports deferred/safe-boot semantics so a
        desktop turn can cold-start Cortex on demand. The runtime health
        contract is stricter: healthy must mean an actual backend can accept a
        generation without relying on deferred startup. Proof-primary runs also
        require the primary Cortex lane specifically, not a lower-tier fallback.

        Use :meth:`inference_readiness` when the reason matters.
        """
        ready, _reason = self.inference_readiness()
        return ready
