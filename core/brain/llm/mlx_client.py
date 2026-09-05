from __future__ import annotations

import asyncio
import concurrent.futures as cfutures
import contextlib
import copy
import fcntl
import functools
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import platform
import queue
import re
import stat
import subprocess
import sys
import threading as _threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.runtime import resource_psutil as psutil

if TYPE_CHECKING:
    from core.brain.lane_admission import ActiveLane
    from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

from core.brain.llm.measured_admission import record_generation
from core.conversation.continuation import continuation_state_text
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.lockdep import instrument
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.resource_observation import get_resource_observer
from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,
    record_shutdown_admission_event,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable_blocking
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    get_subprocess_gateway,
)
from core.utils.concurrency import run_io_bound
from core.utils.deadlines import Deadline, get_deadline
from core.utils.memory_monitor import get_memory_pressure_snapshot
from core.utils.task_tracker import get_task_tracker

from .chat_format import format_chatml_messages, format_chatml_prompt
from .mlx_worker import (
    _HIDDEN_SEQUENCE_MAX_INPUT_CHARS,
    _HIDDEN_SEQUENCE_MAX_TOKENS,
    _HIDDEN_SEQUENCE_MAX_WIDTH,
    _mlx_worker_loop,
)

#: Returned by an extracted block that did NOT return early. A unique
#: object, so no value a block legitimately returns can be mistaken for it.
_SEAM_FELL_THROUGH = object()

#: Characters per token, for turning a prompt length into work. The usual
#: ratio for this tokenizer family; only ever used to size a deadline, where
#: being roughly right beats having no relationship to the prompt at all.
_CHARS_PER_TOKEN = 4.0

#: What to assume before this worker has been measured. Well under the
#: 716-772 tok/s observed on this host, because being generous with an
#: unmeasured worker costs a little latency and being mean costs the answer.
_UNMEASURED_PREFILL_RATE = 300.0

#: How much longer than the reading itself to allow. A shared lane queues,
#: and a deadline with no room for that cancels healthy work.
_PREFILL_HEADROOM = 3.0

#: What this host has actually been measured doing, kept where a caller who
#: has to size a deadline can read it.
#:
#: A question's deadline belongs to the question. Sized from a constant
#: instead, one budget has to fit both "left" and "how should I play this",
#: and it fits neither: measured live 2026-08-26, every approach question in
#: a game timed out at eight seconds and she played the whole thing with no
#: plan. These are rates this machine was seen working at, not guesses.
_HOST_RATES: dict[str, float] = {"prefill": 0.0, "decode": 0.0, "weight_load": 0.0}

#: Gigabytes of weights a second, before this host has been seen loading any.
#: Deliberately slow for the same reason the prefill default is: being
#: generous with an unmeasured worker costs latency, being mean costs the
#: answer — and here being mean costs the lane, which is recycled and comes
#: back just as cold.
_UNMEASURED_WEIGHT_LOAD_GB_S = 0.5

#: Lockdep's own limit for a blocking hold on the event loop. A guard that
#: reaches it says which block it was, because the splat cannot.
_LANE_STATE_HOLD_WORTH_NAMING_MS = 50.0

#: Weight sizes are a property of the files, read once per path.
_WEIGHT_SIZES: dict[str, float] = {}

#: How long each model has actually taken to say its first word from cold,
#: by name. Measured whole, because the time is not all spent reading bytes.
_COLD_FIRST_TOKEN_S: dict[str, float] = {}

#: A cold lane's first token is late for a reason that is over once it
#: happens. Room for that, on top of what loading measures at.
_COLD_START_HEADROOM = 2.0

#: Decode is slower and more variable than prefill, so an unmeasured worker
#: is credited with little until it proves otherwise.
_UNMEASURED_DECODE_RATE = 8.0


def reset_host_rates_for_test() -> dict[str, float]:
    """Forget the measured host rates, and hand back what they were.

    They are process-wide and written by any real generation, and every answer
    budget is computed from them — so one test that generates decides how long
    the next test's answer is allowed to be. Restoring what was there is the
    caller's job; :func:`observed_rates` falls back to the unmeasured
    constants until something measures again.
    """
    previous = dict(_HOST_RATES)
    for key in _HOST_RATES:
        _HOST_RATES[key] = 0.0
    return previous


def restore_host_rates_for_test(rates: dict[str, float]) -> None:
    """Put back what :func:`reset_host_rates_for_test` handed over."""
    _HOST_RATES.update(rates)


def observed_rates() -> dict[str, float]:
    """Prefill and decode rates this host has been measured at, tokens a second."""
    return {
        "prefill": _HOST_RATES["prefill"] or _UNMEASURED_PREFILL_RATE,
        "decode": _HOST_RATES["decode"] or _UNMEASURED_DECODE_RATE,
    }


def time_a_prompt_needs(prompt_chars: int, max_tokens: int) -> float:
    """The least time in which this question could be read and answered.

    Reading the prompt and writing the answer are separate costs and both are
    real: an answer that is a whole plan takes many more tokens than an answer
    that is one word, on top of a prompt that is far longer.
    """
    rates = observed_rates()
    chars = max(0, int(prompt_chars or 0))
    wanted = max(0, int(max_tokens or 0))
    reading = (chars / _CHARS_PER_TOKEN / rates["prefill"]) * _PREFILL_HEADROOM
    writing = wanted / rates["decode"]
    return reading + writing


def longest_a_turn_may_take(
    *, generations: int, prompt_chars: int, max_tokens: int, floor_s: float
) -> float:
    """How long a turn of this shape needs on THIS machine, at its measured rates.

    A ceiling is meant to stop a turn running away, and a flat number cannot
    tell running away from working. LIVE 2026-08-29: a turn read a library's
    docs, wrote code against them, ran it and was composing the answer when it
    reached 480 seconds — "gave up 298s past its budget: last sign of work 0.3s
    ago". Nothing had gone quiet. On this host a generation of a thousand
    tokens against a seven-thousand-character prompt takes about a hundred
    seconds, and a tool loop is allowed several of them, so the ceiling was
    below the cost of the work it was standing over.

    The rates are what the machine has been seen doing, so a faster host gets
    a shorter ceiling for free and a loaded one gets the room it needs. The
    caller's floor still applies: this only ever raises, because a measurement
    that comes back small must not shorten a bound somebody else set.
    """

    turns = max(1, int(generations or 1))
    needed = turns * time_a_prompt_needs(prompt_chars, max_tokens)
    return max(float(floor_s), needed)


logger = logging.getLogger("LLM.MLX")

# Abort reasons that race a finishing generation: losing that race is the
# timeout working, not a fault.
# Any TIMEOUT-shaped reason races a finishing generation; enumerating them one
# at a time missed endpoint_timeout and killed a healthy idle Cortex worker
# 150s into a live probe, buying a full cold reload for a turn already served.
# Deliberate kills (memory_pressure, operator_requested, crash_loop_backoff,
# model_swap) contain none of these words and are unaffected.
_ABORT_RACE_MARKERS_RE = re.compile(
    r"timeout|timed_out|first_token|soft_deadline|deadline|"
    r"turn_complete|request_finished",
    re.IGNORECASE,
)
_AURA_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _observed_process_rss_bytes(pid: int) -> int:
    try:
        process = get_resource_observer().process(int(pid))
        return int(process.rss_bytes) if process is not None else 0
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return 0


_MODEL_LOAD_FOREGROUND_ADMISSION_TIMEOUT_FLAG = declare(
    "AURA_FOREGROUND_MODEL_LOAD_ADMISSION_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=30.0,
    description="Maximum foreground wait for the canonical model-load lease",
    owner="core.brain.llm.mlx_client",
)
_MODEL_LOAD_BACKGROUND_ADMISSION_TIMEOUT_FLAG = declare(
    "AURA_BACKGROUND_MODEL_LOAD_ADMISSION_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=0.0,
    description="Maximum background wait for the canonical model-load lease",
    owner="core.brain.llm.mlx_client",
)


_HEAVY_LANE_NAME_TOKENS = ("32b", "72b", "zenith", "solver", "cortex")


# Ceiling applied when the somatic throttle cannot report. Not a refusal —
# this is a throttle, and refusing generation for a metabolic hiccup would
# take conversation down — but not the wide-open default either.
_UNTHROTTLED_FALLBACK_MAX_TOKENS = 1024


def _generation_wait_hard_cap_s(
    deadline: Deadline,
    *,
    foreground_request: bool,
) -> float:
    """Return the terminal worker-wait bound for one generation owner.

    Background and unowned calls retain the local MLX containment cap. A
    foreground route, however, has already admitted a finite end-to-end
    deadline based on the requested answer shape. Clipping that owner to the
    historical 240-second local default discarded healthy long-form decodes
    while their route still had more than three minutes left.
    """
    configured_cap = max(
        30.0,
        _env_duration_s("AURA_MLX_GENERATION_HARD_CAP_SECONDS", 240.0, minimum=30.0),
    )
    if not foreground_request:
        return configured_cap

    remaining = deadline.remaining
    if remaining is None or not math.isfinite(remaining) or remaining <= 0.0:
        return configured_cap
    return min(
        USER_FACING_COMPLETION_DEADLINE_MAX_S,
        max(configured_cap, remaining),
    )


def _apply_unthrottled_fallback_ceiling(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Damp generation when body pressure could not be consulted.

    Mutates and returns the SAME mapping rather than a copy. The throttle's
    success path rebinds kwargs, but its failure path historically left the
    caller's object untouched, and callers downstream rely on that identity;
    swapping in a copy here silently detached their later mutations.
    """
    if not isinstance(kwargs, dict):
        return kwargs
    requested = kwargs.get("max_tokens")
    if requested is None:
        # IMPOSE NOTHING. Several internal paths — the warmup precompile
        # probe above all — deliberately omit max_tokens and rely on their
        # own budgeting; putting a number there changes what those paths do
        # and, in the warmup case, left a durable owner unreleased. This is
        # a ceiling on an over-large request, not a default for callers who
        # never asked for one.
        return kwargs
    try:
        current = int(requested)
    except (TypeError, ValueError):
        # A malformed budget is not a budget; clamp it to something sane.
        kwargs["max_tokens"] = _UNTHROTTLED_FALLBACK_MAX_TOKENS
        return kwargs
    if current > _UNTHROTTLED_FALLBACK_MAX_TOKENS:
        kwargs["max_tokens"] = _UNTHROTTLED_FALLBACK_MAX_TOKENS
    return kwargs


def _model_is_heavy_lane(model_path: str | None) -> bool:
    """True when a path names one of the big resident lanes.

    Measured first: get_model_artifact_profile reads the artifact's own
    config/weight index and derives a parameter count, so a renamed or
    aliased checkpoint is still classified by what it IS. The historical
    name tokens are unioned in rather than replaced — this predicate gates
    memory admission, and for a safety guard the fail-safe direction is to
    over-include. A profile that cannot be read falls back to the tokens
    alone, which is exactly the old behaviour.
    """
    path = str(model_path or "")
    if not path:
        return False
    measured = False
    try:
        from core.brain.llm.model_artifact_profile import model_is_heavy

        measured = bool(model_is_heavy(path))
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError) as exc:
        _record_mlx_degradation(
            exc,
            action="model artifact profile unavailable; lane class fell back to naming",
            severity="debug",
        )
    lowered = os.path.basename(path).lower()
    named = any(token in lowered for token in _HEAVY_LANE_NAME_TOKENS)
    return measured or named


#: Ceilings for a single worker-supplied progress value. Progress is a
#: diagnostic; nothing downstream needs more than this, and everything
#: upstream is a process the parent does not control.
#: One batch candidate is a verifier draft, not a document.
_BATCH_CANDIDATE_MAX_CHARS = 32_000

_PROGRESS_STR_CHARS = 200
_PROGRESS_SEQ_ITEMS = 32


#: A soft-cancel acknowledgement costs about one decode step. It is spent
#: after the caller's deadline has already expired, so it stays small and is
#: named rather than folded into the budget it exceeds.
_LATENT_CANCEL_ACK_GRACE_S = 2.0


def _remaining_budget(deadline: Deadline | None, fallback_s: float) -> float:
    """What is left of the ONE budget this request was given.

    A phase that reads the original timeout instead of the remainder is not
    bounded by the caller's deadline at all — it merely starts a new one.
    """
    if isinstance(deadline, Deadline):
        remaining = deadline.remaining
        if remaining is not None:
            return max(0.0, float(remaining))
    return max(0.0, float(fallback_s))


#: One latent episode's prompt. Larger than any real turn and far smaller
#: than what an unbounded caller could hand over.
_LATENT_PROMPT_MAX_CHARS = 200_000
_LATENT_MESSAGES_MAX_ITEMS = 256
_LATENT_MESSAGE_MAX_CHARS = 200_000
_LATENT_TOTAL_MAX_CHARS = 400_000
_LATENT_ROLES = frozenset({"system", "user", "assistant", "tool"})


#: Chat-history ceilings for the public text interface. Generous on purpose:
#: no real conversation approaches these, and the point is that a pathological
#: or hostile one cannot reach template rendering unbounded.
_CHAT_MESSAGES_MAX_ITEMS = 512
_CHAT_MESSAGE_MAX_CHARS = 200_000
_CHAT_TOTAL_MAX_CHARS = 800_000


def _bounded_chat_messages(messages: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Bound and type a chat history before it is flattened or templated.

    CP126 7f9e9cf4: the public interface checked that messages was a non-empty
    list and then called ``messages[0].get`` and ``dict(m)`` across every
    entry. Non-mapping elements were already handled; roles, content types,
    the message COUNT and the aggregate size were not, so an oversized or
    mistyped history reached template rendering — the most expensive place to
    discover it, and outside the generation failure contract.

    The most RECENT messages are kept when the count is exceeded, along with a
    leading system message: dropping the system turn to preserve old chatter
    would lose the policy the turn runs under.
    """
    faults: list[str] = []
    rows = [m for m in (messages or []) if isinstance(m, dict)]
    if len(rows) != len(messages or []):
        faults.append("messages:non_mapping_dropped")

    if len(rows) > _CHAT_MESSAGES_MAX_ITEMS:
        faults.append("messages:too_many")
        head = rows[:1] if str(rows[0].get("role") or "") == "system" else []
        keep = _CHAT_MESSAGES_MAX_ITEMS - len(head)
        rows = head + rows[-keep:]

    bounded: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        role = row.get("role")
        if not isinstance(role, str) or not role.strip():
            faults.append("messages:invalid_role")
            continue
        entry = dict(row)
        content = entry.get("content")
        if content is None:
            entry["content"] = ""
        elif not isinstance(content, str):
            faults.append("messages:non_string_content")
            entry["content"] = str(content)
        if len(entry["content"]) > _CHAT_MESSAGE_MAX_CHARS:
            faults.append("messages:content_too_long")
            entry["content"] = entry["content"][:_CHAT_MESSAGE_MAX_CHARS]
        total += len(entry["content"])
        if total > _CHAT_TOTAL_MAX_CHARS:
            faults.append("messages:aggregate_too_large")
            break
        bounded.append(entry)
    return bounded, faults


def _latent_request_schema_error(
    *, prompt: Any = None, messages: Any = None
) -> str:
    """Reject a latent payload the parent should not spend work on.

    Returns "" when the payload is admissible, otherwise a typed reason. The
    checks are shape and size only — meaning belongs to the worker — but they
    run BEFORE the copy, the canonical serialisation and the SHA-256 that
    would otherwise be paid for a request no episode could run.
    """
    total = 0
    if prompt is not None:
        if not isinstance(prompt, str):
            return "invalid_prompt_type"
        if len(prompt) > _LATENT_PROMPT_MAX_CHARS:
            return "prompt_too_large"
        total += len(prompt)
    if messages is not None:
        if not isinstance(messages, list):
            return "invalid_messages_type"
        if len(messages) > _LATENT_MESSAGES_MAX_ITEMS:
            return "too_many_messages"
        for message in messages:
            if not isinstance(message, dict):
                return "invalid_message_type"
            role = message.get("role")
            if not isinstance(role, str) or role not in _LATENT_ROLES:
                return "invalid_message_role"
            content = message.get("content")
            if not isinstance(content, str):
                return "invalid_message_content"
            if len(content) > _LATENT_MESSAGE_MAX_CHARS:
                return "message_too_large"
            total += len(content)
    if total > _LATENT_TOTAL_MAX_CHARS:
        return "request_too_large"
    return ""


def _capped_reserve(reserve_s: float, remaining_s: float) -> float:
    """Hold back time for the caller to fail closed — but never all of it.

    A fixed reserve subtracted from a shorter deadline yields zero or less,
    and a zero wait means the caller never even tries. That is how the
    dec24697 fix (stop granting time past an exhausted deadline, which was
    right) broke foreground preemption of maintenance: a 1-second foreground
    deadline minus a 3-second reserve gave a 0-second wait, the acquire loop
    did not execute once, and the soft-cancel that evicts a maintenance
    holder never fired.

    Capped at half the remaining budget: a short deadline still gets time to
    wait AND time to recover, and a long one is unaffected.
    """
    remaining = max(0.0, float(remaining_s))
    return min(float(reserve_s), remaining * 0.5)


def _sleep_inclusive_monotonic() -> float | None:
    """A monotonic clock that keeps counting while the host is asleep.

    Paired with ``time.monotonic()`` — which does not — this measures how long
    the machine was suspended without consulting the wall clock at all, so an
    NTP step, a manual date change, or a VM migration cannot be mistaken for a
    host resume. Returns None where the platform offers no such clock, and the
    caller then says it could not tell the two apart rather than guessing.
    """
    for name in ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC"):
        clock_id = getattr(time, name, None)
        if clock_id is None:
            continue
        if name == "CLOCK_MONOTONIC" and sys.platform != "darwin":
            # Only Darwin's CLOCK_MONOTONIC includes suspend; elsewhere it is
            # what time.monotonic() already returns, so the difference would
            # be a constant zero dressed up as a measurement.
            continue
        try:
            return float(time.clock_gettime(clock_id))
        except (OSError, ValueError, AttributeError):
            continue
    return None


#: The paired-sample arrays a feedback consumer reads. Everything else in an
#: interoception payload is scalar summary.
_INTEROCEPTION_SAMPLE_KEYS = ("token_ids_sample", "logprob_sample")
#: One reply's worth of tokens, generously. Beyond this the parent is holding
#: an array whose size the WORKER chose.
_INTEROCEPTION_SAMPLE_LIMIT = 4096


def _bounded_interoception(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain an accepted interoception payload at a size the parent chose."""
    stored: dict[str, Any] = {}
    for key, value in list(payload.items())[:64]:
        name = str(key)
        if name in _INTEROCEPTION_SAMPLE_KEYS and isinstance(value, (list, tuple)):
            stored[name] = list(value)[:_INTEROCEPTION_SAMPLE_LIMIT]
            continue
        stored[name] = _bounded_progress_value(value)
    return stored


def _bounded_progress_value(value: Any) -> Any:
    """Clamp one worker-reported progress value to a size the parent chose.

    Unknown and oversized shapes are summarized rather than dropped: that a
    field arrived malformed is itself worth seeing, and replacing it with
    None would report the worker as silent on a field it did answer.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value[:_PROGRESS_STR_CHARS]
    if isinstance(value, (list, tuple)):
        return [_bounded_progress_value(item) for item in list(value)[:_PROGRESS_SEQ_ITEMS]]
    if isinstance(value, Mapping):
        return {
            str(key)[:_PROGRESS_STR_CHARS]: _bounded_progress_value(item)
            for key, item in list(value.items())[:_PROGRESS_SEQ_ITEMS]
        }
    return f"<unsupported:{type(value).__name__}>"


def _observe_worker_prompt_tokenization(response: Mapping[str, Any]) -> bool:
    """Admit one exact tokenizer measurement from a terminal worker frame."""

    if response.get("status") != "ok" or response.get("action") not in {
        "generate",
        "generate_batch",
        "stream_done",
    }:
        return False
    evidence = response.get("prompt_tokenization")
    if not isinstance(evidence, Mapping):
        return False
    try:
        from core.brain.llm.token_budget_evidence import observe_prompt_tokenization

        return bool(
            observe_prompt_tokenization(
                evidence.get("chars"),
                evidence.get("tokens"),
            )
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _observe_worker_token_budget_calibration(response: Mapping[str, Any]) -> int:
    """Atomically import a validated tokenizer calibration from worker init."""

    if response.get("status") != "ok" or response.get("action") != "init":
        return 0
    try:
        from core.brain.llm.token_budget_evidence import observe_calibration_batch

        return int(observe_calibration_batch(response.get("token_budget_calibration")))
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0


#: A servable MLX artifact carries a model config, a tokenizer, and weights.
#: Anything missing means the worker will fail at load time — after the healthy
#: one has already been torn down.
_REQUIRED_ARTIFACT_CONFIG = "config.json"
_TOKENIZER_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class ArtifactVerdict:
    """Whether a directory can actually be served, and what it is."""

    ok: bool
    reason: str = ""
    architectures: tuple[str, ...] = ()
    size_class: str = "unknown"
    fingerprint: str = ""
    weight_files: int = 0

    def as_receipt(self) -> dict[str, Any]:
        return {
            "architectures": list(self.architectures),
            "size_class": self.size_class,
            "fingerprint": self.fingerprint,
            "weight_files": self.weight_files,
        }


def _validate_model_artifact(resolved: Path, incumbent: str = "") -> ArtifactVerdict:
    """Prove a directory is a servable model BEFORE the live worker is recycled.

    CP126 a996d77f: this was ``is_dir()``. An empty, partial, half-copied, or
    wrong-architecture directory passed, the healthy worker was torn down, and
    the failure surfaced at the next load — by which time the lane that was
    serving fine had been destroyed to make room for something that could not
    load at all.

    ``incumbent`` is the currently-served path. When both sides declare their
    architectures, a mismatch is refused: promoting a Llama checkpoint onto a
    lane whose callers, adapters and admission classes were built for Qwen is
    a different model wearing the lane's name.
    """
    if not resolved.is_dir():
        return ArtifactVerdict(False, f"artifact_missing:{resolved}")

    config_path = resolved / _REQUIRED_ARTIFACT_CONFIG
    if not config_path.is_file():
        return ArtifactVerdict(False, f"artifact_missing_config:{_REQUIRED_ARTIFACT_CONFIG}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ArtifactVerdict(False, f"artifact_config_unreadable:{type(exc).__name__}")
    if not isinstance(config, dict):
        return ArtifactVerdict(False, "artifact_config_not_an_object")

    if not any((resolved / name).is_file() for name in _TOKENIZER_CANDIDATES):
        return ArtifactVerdict(False, "artifact_missing_tokenizer")

    weights = list(_weight_files(resolved))
    if not weights:
        return ArtifactVerdict(False, "artifact_missing_weights")

    architectures = tuple(
        str(entry)
        for entry in (config.get("architectures") or [])
        if isinstance(entry, str)
    )
    profile = None
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(str(resolved))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        profile = None

    verdict = ArtifactVerdict(
        True,
        architectures=architectures,
        size_class=getattr(profile, "size_class", "unknown") or "unknown",
        fingerprint=getattr(profile, "fingerprint", "") or "",
        weight_files=len(weights),
    )

    incumbent_path = Path(str(incumbent or "")).expanduser()
    incumbent_config = incumbent_path / _REQUIRED_ARTIFACT_CONFIG
    if architectures and incumbent_config.is_file():
        try:
            current = json.loads(incumbent_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            current = {}
        current_arch = tuple(
            str(entry)
            for entry in (current.get("architectures") or [])
            if isinstance(entry, str)
        )
        if current_arch and not set(current_arch) & set(architectures):
            return ArtifactVerdict(
                False,
                f"artifact_architecture_mismatch:{'/'.join(current_arch)}"
                f"->{'/'.join(architectures)}",
                architectures=architectures,
                size_class=verdict.size_class,
                fingerprint=verdict.fingerprint,
                weight_files=verdict.weight_files,
            )
    return verdict


#: What an MLX LoRA adapter directory must actually contain to be attachable.
_ADAPTER_WEIGHT_NAMES = ("adapters.safetensors", "adapter.safetensors")
_ADAPTER_CONFIG_NAME = "adapter_config.json"
#: An adapter is tens of megabytes. An order of magnitude past that is not an
#: adapter, and sending it to the resident worker is how a "swap" becomes a
#: memory event.
_ADAPTER_MAX_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class AdapterVerdict:
    """Whether a directory can be attached to the resident model, and what it is."""

    ok: bool
    reason: str = ""
    weight_file: str = ""
    weight_bytes: int = 0
    base_checkpoint_fingerprint: str = ""
    fine_tune_type: str = ""
    #: verified | declared_unverified | not_declared
    base_compatibility: str = "not_declared"

    def as_receipt(self) -> dict[str, Any]:
        return {
            "adapter_weight_file": self.weight_file,
            "adapter_weight_bytes": self.weight_bytes,
            "adapter_base_checkpoint_fingerprint": self.base_checkpoint_fingerprint,
            "adapter_fine_tune_type": self.fine_tune_type,
            "adapter_base_compatibility": self.base_compatibility,
        }


def _bounded_maintenance_counters(
    response: Mapping[str, Any],
    *,
    max_pairs: int,
    scan_limit: int,
    max_positions: int,
) -> tuple[dict[str, int | None], list[str]]:
    """Read the maintenance counters, or report that they could not be read.

    ``None`` means unmeasured — a value the worker gave that cannot be true,
    or did not give at all. It is deliberately distinct from ``0``, which
    means the worker measured none. Collapsing the two is how "we never found
    out" becomes "we checked and there was nothing".
    """
    faults: list[str] = []

    def _count(name: str, ceiling: int) -> int | None:
        raw = response.get(name)
        if raw is None:
            faults.append(f"{name}:absent")
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            faults.append(f"{name}:malformed")
            return None
        if value < 0:
            faults.append(f"{name}:negative")
            return None
        if value > ceiling:
            faults.append(f"{name}:above_budget")
            return None
        return value

    counters: dict[str, int | None] = {
        "pairs_considered": _count("pairs_considered", scan_limit),
        "pairs_scanned": _count("pairs_scanned", scan_limit),
        "pairs_ingested": _count("pairs_ingested", max_pairs),
        "positions_ingested": _count("positions_ingested", max_positions),
    }
    scanned = counters["pairs_scanned"]
    ingested = counters["pairs_ingested"]
    if scanned is not None and ingested is not None and ingested > scanned:
        # Ingesting what was never scanned is not a large number; it is a
        # broken relationship between two counters.
        faults.append("pairs_ingested:exceeds_scanned")
        counters["pairs_ingested"] = None
    return counters, faults


def _validate_adapter_artifact(
    path: Path, *, expected_base_fingerprint: str = ""
) -> AdapterVerdict:
    """Prove a directory is an attachable adapter before live weights change.

    CP126 d665aa64: admission was ``is_dir()``. Any directory — a
    half-finished training output, an empty scratch folder, a symlink to
    somewhere else entirely — was handed to the resident worker as an adapter,
    and the failure surfaced inside the process holding twenty gigabytes of
    live weights.

    When the adapter names the base checkpoint it was trained against AND the
    caller supplies the resident one, a mismatch is refused here: LoRA deltas
    are only meaningful against the weights they were fitted to, and attaching
    them to different ones does not fail loudly — it quietly degrades every
    answer the model gives afterwards.

    When the caller cannot supply the resident fingerprint, the verdict says
    ``declared_unverified`` rather than passing quietly. An unchecked
    compatibility claim recorded as a checked one is the failure this whole
    remediation keeps finding, and writing it into a new validator to make the
    validator look thorough would be the same mistake with a fresh coat.
    """
    if not path.is_dir():
        return AdapterVerdict(False, f"adapter_missing:{path}")

    weight_path: Path | None = None
    for name in _ADAPTER_WEIGHT_NAMES:
        candidate = path / name
        if candidate.is_file():
            weight_path = candidate
            break
    if weight_path is None:
        return AdapterVerdict(False, "adapter_missing_weights")
    try:
        weight_bytes = int(weight_path.stat().st_size)
    except OSError as exc:
        return AdapterVerdict(False, f"adapter_weights_unreadable:{type(exc).__name__}")
    if weight_bytes <= 0:
        return AdapterVerdict(False, "adapter_weights_empty")
    if weight_bytes > _ADAPTER_MAX_BYTES:
        return AdapterVerdict(False, f"adapter_weights_oversized:{weight_bytes}")

    config: dict[str, Any] = {}
    config_path = path / _ADAPTER_CONFIG_NAME
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return AdapterVerdict(False, f"adapter_config_unreadable:{type(exc).__name__}")
        if not isinstance(loaded, dict):
            return AdapterVerdict(False, "adapter_config_not_an_object")
        config = loaded

    base_fingerprint = str(config.get("base_checkpoint_fingerprint") or "")
    expected = str(expected_base_fingerprint or "")
    if not base_fingerprint:
        compatibility = "not_declared"
    elif not expected:
        compatibility = "declared_unverified"
    elif base_fingerprint != expected:
        return AdapterVerdict(
            False,
            f"adapter_base_mismatch:{base_fingerprint[:12]}!={expected[:12]}",
            weight_file=weight_path.name,
            weight_bytes=weight_bytes,
            base_checkpoint_fingerprint=base_fingerprint,
            fine_tune_type=str(config.get("fine_tune_type") or ""),
            base_compatibility="mismatch",
        )
    else:
        compatibility = "verified"
    return AdapterVerdict(
        True,
        weight_file=weight_path.name,
        weight_bytes=weight_bytes,
        base_checkpoint_fingerprint=base_fingerprint,
        fine_tune_type=str(config.get("fine_tune_type") or ""),
        base_compatibility=compatibility,
    )


class ModelLoadAdmissionRefused(RuntimeError):  # noqa: N818 - public API
    """The memory guard declined to load this model. A DECISION, not a fault.

    CP126 5ce89b9e: this condition used to be recognised by searching an
    arbitrary error message for "memory_pressure_refused_worker_spawn:". Any
    unrelated failure whose text happened to contain that substring — a
    wrapped traceback, a message quoting an earlier refusal, a log line echoed
    into an exception — was silently downgraded to a non-critical fallback and
    its backoff cleared. A type cannot be spelled by accident.
    """

    def __init__(self, blocker: str) -> None:
        super().__init__(f"memory_pressure_refused_worker_spawn:{blocker}")
        self.blocker = str(blocker)


def _record_mlx_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "mlx_client",
        error,
        severity=severity,
        action=action,
    )


_MLX_OPTIONAL_THROTTLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# Failures a synchronous worker control message can raise and survive: the queue
# or pipe is gone, the worker died, or the reply never came. A shed request that
# cannot be delivered must report "freed nothing", never crash the OOM ladder.
_MLX_CLIENT_RECOVERABLE_ERRORS = (
    AttributeError,
    BrokenPipeError,
    EOFError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    queue.Full,
)


# Global state for swap management
_GLOBAL_LAST_SWAP_TIME = 0.0
_GLOBAL_LAST_HEAVY_MODEL: str | None = None
_CLIENTS: dict[str, Any] = {}
# CP126 bec28d76: every observer iterated `list(_CLIENTS.items())` and then read
# each client's mutable lifecycle fields independently, so a worker could be
# registered, recycled or torn down mid-scan. Admission then decided against a
# view that never existed at any instant: a lane counted twice, a lane missed,
# a lane classified from a path whose client had already been replaced. The
# registry itself is now guarded, and observers take a consistent snapshot of
# the membership before reading the members.
_CLIENTS_LOCK = _threading.Lock()


def _clients_snapshot() -> list[tuple[str, Any]]:
    """One atomic view of registry membership.

    The clients' own fields are still read afterwards without the lock —
    holding it across `is_alive()` (which can touch a process handle) would
    couple registry mutation to process I/O. What this removes is the
    membership race: no observer can see a half-applied registration.
    """
    with _CLIENTS_LOCK:
        return list(_CLIENTS.items())


def _client_lane_policy(client: Any) -> tuple[str, Any]:
    """Read lane/QoS from the immutable assignment carried by the client."""

    from core.brain.lane_admission import QoSClass
    from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

    assignment = getattr(client, "runtime_assignment", None)
    if not isinstance(assignment, ModelRuntimeAssignment):
        raise RuntimeError("mlx_client_runtime_assignment_missing")
    assignment.assert_bound_to(model_path=getattr(client, "model_path", ""), purpose="serve")
    try:
        return assignment.lane, QoSClass(assignment.qos)
    except ValueError as exc:
        raise RuntimeError("mlx_client_runtime_assignment_qos_invalid") from exc


def clients_snapshot() -> list[tuple[str, Any]]:
    """Public name for the atomic membership view.

    Observers outside this module were reaching past the lock — ``dict(_CLIENTS)``
    and ``list(_CLIENTS.values())`` both ITERATE the registry, so a client
    registered or torn down mid-copy raises "dictionary changed size during
    iteration". The inference gate hit exactly that live on 2026-08-03, and
    because that subsystem is on the fail-closed list the RuntimeError was
    escalated to CRITICAL and held the runtime DEGRADED across health pulses.

    A single-key ``_CLIENTS.get(path)`` is safe and does not need this; copying
    or iterating the registry does.
    """

    return _clients_snapshot()


def _rebind_client_registry_key(previous: str, target: str, client: Any) -> bool:
    """Re-key a client after its serving artifact changes.

    CP126 df8e3045: a promotion mutated only ``model_path``. The registry
    still filed the client under the OLD path, so ``get_mlx_client(new_path)``
    built a SECOND client — and a second worker, and a second copy of the
    weights — for a lane that was already serving that artifact, while
    admission and eviction went on describing this one under a name it no
    longer had.

    Returns whether a rebind happened. Refuses to evict a different client
    already registered at the target: two clients wanting one path is a
    conflict to report, not to resolve by overwriting.
    """
    old_key = str(previous or "")
    new_key = str(target or "")
    if not new_key or old_key == new_key:
        return False
    with _CLIENTS_LOCK:
        if _CLIENTS.get(old_key) is not client:
            return False
        occupant = _CLIENTS.get(new_key)
        if occupant is not None and occupant is not client:
            logger.warning(
                "🧬 [MLX] Promotion target %s already has a distinct client; "
                "left the registry alone.",
                os.path.basename(new_key),
            )
            return False
        _CLIENTS.pop(old_key, None)
        _CLIENTS[new_key] = client
    return True


_FOREGROUND_OWNER_LOCK = _threading.Lock()
_FOREGROUND_OWNER_NAME: str | None = None
_FOREGROUND_OWNER_ACQUIRED_AT = 0.0
# CP126 6595b0e1: ownership age was measured with time.time(), so an NTP step
# or a sleep/wake made a healthy 32B cold load look 200 seconds stale and
# cleared it mid-load — and a genuinely wedged owner still held until a guessed
# age because nothing measured PROGRESS. The monotonic stamps below cannot be
# stepped, and the heartbeat measures time since the owner last did something
# rather than time since it started.
_FOREGROUND_OWNER_ACQUIRED_MONOTONIC = 0.0
_FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = 0.0
# The budget the CURRENT holder declared for itself when it took ownership.
# Eviction is judged against this, never against a newcomer's budget.
_FOREGROUND_OWNER_STALE_AFTER: float | None = None
# Whether the current holder is a PERSON's turn or background work.
#
# Eviction compared ages only, so a person's typed message queued behind
# whatever happened to be holding the lane — including autonomous loops that
# have nobody waiting on them. Measured live: "Waiting for foreground owner
# Cortex to release (held 58.7s)". Background work is interruptible by
# design; someone sitting at the keyboard is not, so a user-facing request
# takes the lane from a background holder immediately, and waits its turn
# behind another user-facing one.
_FOREGROUND_OWNER_IS_USER_FACING = False

#: How long a warmup retry will wait for a reply to finish before standing
#: down. Longer than a reply takes, shorter than anybody would sit staring at
#: a lane that is not recovering.
_WAIT_OUT_A_REPLY_S = 30.0
# No foreground owner may be evicted before this, whatever anyone declares.
# A newcomer with a 5s budget must not be able to steal a lane from a turn
# that is legitimately still working.
_FOREGROUND_OWNER_MIN_EVICTION_S = 30.0

# [OOM FIX] Global gate: only ONE model can be loading at a time across ALL clients.
# This prevents the 32B and 7B from loading simultaneously and exceeding GPU RAM.
# Uses threading.Semaphore (loop-agnostic) because the singleton MLXLocalClient
# is constructed from one event loop but called from another (Uvicorn thread).
_GLOBAL_SPAWN_GATE = _threading.Semaphore(1)
# Gate holders may legitimately spend minutes loading the 32B, but waiters must
# defer quickly. A waiter is not the load owner and must never consume a whole
# foreground turn merely waiting for the mechanical single-spawn mutex.
_SPAWN_GATE_ACQUIRE_TIMEOUT_S = 5.0  # replaced below once the parser is defined
_GLOBAL_SPAWN_GATE_STATE_LOCK = _threading.Lock()
_GLOBAL_SPAWN_GATE_TOKEN = ""
_GLOBAL_SPAWN_GATE_OWNER = ""
_GLOBAL_SPAWN_GATE_ACQUIRED_AT = 0.0
_MLX_RUNTIME_PROBE_LOCK = _threading.Lock()
_MLX_RUNTIME_PROBE: dict[str, Any] = {
    "ok": None,
    "detail": "",
    "checked_at": 0.0,
}
_MLX_RUNTIME_PROBE_CACHE_PATH = state_root() / "data" / "mlx_runtime_probe.json"

#: Bumped whenever the probe's MEANING changes. A cache written by an older
#: probe answered a different question — the import-only probe this one
#: replaced would have cached "ok" on a host whose Metal device cannot
#: evaluate a tensor — so it must not be read as if it answered this one.
_MLX_RUNTIME_PROBE_SCHEMA = 2


def _probe_cache_identity() -> dict[str, str]:
    """What this cached verdict is actually a verdict ABOUT.

    The cache is a plain file under the state root, and it gates whether Aura
    believes local inference works at all. Nothing bound it to the environment
    it was measured in, so a verdict could outlive the thing it described: a
    venv rebuilt with a different mlx wheel, a checkout run under another
    interpreter, or a state directory copied between machines all keep reading
    a verdict that was never true for them. A stale "ok" is the dangerous
    direction — it sends real user turns to a lane that cannot serve them.
    """
    return {
        # The interpreter IS the mlx installation for this purpose: a different
        # venv is a different set of wheels.
        "executable": str(sys.executable),
        "platform": f"{platform.system()}/{platform.machine()}",
    }

# Visible conversation-readiness probe. The lane may only claim "ready" after
# this exact question comes back with an answer that actually responds to it.
#: Hard ceiling on think_and_act turns. Each turn is a full generation plus a
#: tool execution on the resident model, so an unbounded budget lets a single
#: objective hold the foreground lane for as long as it keeps emitting tool
#: calls — including a model looping on the same failing call forever.
_AGENT_MAX_TURNS_CEILING = 32

_READINESS_PROBE_PROMPT = "Reply exactly: ready"
_READINESS_EXPECTED_TOKEN = "ready"
_READINESS_ANSWER_MAX_CHARS = 200
#: The readiness probe is a real 16-token generation on a resident model, so a
#: probe given less than this is being started only to be cancelled mid-token
#: and recorded as a readiness failure it never had a chance to pass.
#:
#: It is a rule about whether to OPEN a probe, deliberately not a floor on its
#: timeout: a floor on the timeout lets the probe outlive the one hard warmup
#: deadline, which is the bound a caller waiting on warmup actually relies on.
_MIN_READINESS_PROBE_BUDGET_S = 10.0
#: And no single probe may consume an arbitrarily large campaign either.
_MAX_READINESS_PROBE_S = 60.0

#: Overshoot the out-of-band watchdog already allows itself before enforcing a
#: first-token ceiling. Named because it is now used twice — once to decide
#: WHEN to fire, and once as the grace a cooperative cancel gets to be
#: acknowledged before the ladder escalates to killing the worker. A second,
#: differently-chosen number for the second use would be exactly the kind of
#: unmoored constant that makes a composed timing contract unreadable.
_WATCHDOG_ENFORCEMENT_SLACK_S = 2.0
# A warmup still running after this long is stuck: cancel it (and prove it
# ended) before a replacement starts.
_WARMUP_STALE_AFTER_S = 300.0
# Reboot lock discipline: wait this much longer after the first 10s before
# treating contention as anything other than a live lifecycle operation, and
# only force an unsynchronized reboot after this many consecutive failures.
_REBOOT_LOCK_ESCALATED_WAIT_S = 35.0
_REBOOT_LOCK_FORCE_AFTER = 3
#: force_abort runs on a watchdog thread and must answer fast, so it takes one
#: short attempt at the lifecycle lock and otherwise hands reconciliation to
#: the owner. Same discipline as reboot: do not rewrite lifecycle state you do
#: not own until repeated attempts prove the owner is itself wedged.
_FORCE_ABORT_LOCK_WAIT_S = 0.25
_FORCE_ABORT_LOCK_FORCE_AFTER = 3
# close() is terminal, so it always completes — but it waits properly first
# instead of racing a live lifecycle operation after one second.
_CLOSE_LOCK_WAIT_S = 10.0
# Worker teardown is a bounded escalation ladder. An orderly lifecycle owner
# gives the request loop one chance to consume its shutdown sentinel; a wedged
# job then receives SIGTERM, and SIGKILL remains the final containment step.
_WORKER_COOPERATIVE_JOIN_S = 2.0
_WORKER_TERMINATE_JOIN_S = 2.0
_WORKER_KILL_JOIN_S = 2.0


def _readiness_answer_accepted(text: Any) -> bool:
    """Does the readiness probe's answer actually respond to what was asked?

    CP126 b6439433: the probe asked the model to reply exactly ``ready`` and
    then accepted ANY nonblank text, so hallucinated, garbled, stale, or
    prompt-echo output proved the lane ready.

    Exact equality would be too brittle in the other direction — trained lanes
    can emit a short latent/reasoning prefix before visible text, and falsely
    recycling a healthy 32B is its own outage. Requiring the expected token to
    appear in a bounded answer that is not merely the prompt echoed back is a
    real check that the failure modes above cannot pass.
    """
    answer = str(text or "").strip()
    if not answer or len(answer) > _READINESS_ANSWER_MAX_CHARS:
        return False
    lowered = answer.lower()
    if _READINESS_EXPECTED_TOKEN not in lowered:
        return False
    # An echo of the instruction is not an answer to it.
    return "reply exactly" not in lowered


SharedFuture = asyncio.Future[Any] | cfutures.Future[Any]


def _read_recurrent_loop_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _manifest_recurrent_loops(artifact_path: Any) -> int | None:
    """The recurrent depth an artifact was actually TRAINED at, or None.

    Read from the adapter/checkpoint's own execution spec, which is written by
    the training pipeline and travels with the weights. Unlike an environment
    variable it cannot be changed by whoever launched the process, and unlike
    a path token it cannot be changed by a rename.
    """
    root = Path(str(artifact_path or "")).expanduser()
    if not root.is_dir():
        return None
    spec_path = root / "execution_spec.json"
    if not spec_path.is_file():
        return None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(spec, dict):
        return None
    try:
        steps = int(spec.get("recurrent_steps") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    return steps if steps > 0 else None


def _note_recurrent_depth_basis_disagreement(
    model_path: str, adapter_path: Any, expected: int
) -> None:
    """Report when the configured depth disagrees with the trained one.

    CP126 cc31c15f: readiness decides the required loop count from a path
    token and a mutable environment variable, neither of which is a property
    of the weights in the worker. The artifact states the depth it was trained
    at; when the two disagree, one of them is wrong about the model that is
    actually loaded, and silence picks a side without evidence.

    Deliberately observational. Making the manifest authoritative would change
    the readiness verdict for a lane a training campaign is mid-flight on, and
    a remediation must not do that as a side effect. What it can do is stop
    the disagreement from being invisible.
    """
    for candidate in (adapter_path, model_path):
        trained = _manifest_recurrent_loops(candidate)
        if trained is None:
            continue
        if trained != expected:
            _record_mlx_degradation(
                RuntimeError(
                    f"recurrent depth expectation {expected} disagrees with the "
                    f"artifact's trained depth {trained} "
                    f"({os.path.basename(str(candidate or ''))})"
                ),
                action=(
                    "kept the configured readiness expectation and recorded the "
                    "artifact's own trained depth alongside it"
                ),
                severity="warning",
            )
        return


def _expected_recurrent_loops_from_model_path(model_path: str) -> int:
    """Return the expected recurrent-depth loop count for a local MLX lane.

    This is a parent-process health mirror of the worker-side policy. The
    worker remains the source of truth for whether the patch actually applied;
    the client uses this only to keep readiness honest before or after worker
    status is received.
    """
    explicit = os.environ.get("AURA_RECURRENT_LOOPS")
    if explicit is not None:
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS", 1)

    if _model_matches_class(model_path, ("72b", "solver")):
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_72B", 1)
    if _model_matches_class(model_path, ("32b", "cortex", "zenith")):
        # Must mirror MODEL_PROFILE_DEFAULTS in recurrent_depth.py: the parent
        # marks the lane required at this count, so a mismatch reports a
        # readiness blocker for a pass that is actually correct.
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_32B", 2)
    if _model_matches_class(model_path, ("14b", "24b", "40b")):
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_14B", 1)
    return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_SMALL", 1)


#: CP126 48f80787: solver/cortex classification, recurrent depth and RAM
#: admission all keyed off substrings of the model PATH — "72b", "32b",
#: "cortex", "zenith". A directory named for one model and holding another
#: therefore got the other model's memory gate and recurrence contract, and a
#: rename was enough to change either. core/brain/llm/model_artifact_profile.py
#: already reads the checkpoint's own config and safetensors index and reports
#: whether its answer was measured; these helpers ask it first and fall back to
#: naming only when the artifact could not be read.
def _measured_artifact_fingerprint(model_path: Any) -> str:
    """The checkpoint's own measured fingerprint, or "" when unmeasurable.

    Empty means the artifact could not be read — NOT that it differs. Callers
    must treat the two differently: an unreadable checkpoint is a reason to
    say so, never a reason to declare a match.
    """
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(str(model_path or ""))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    if not getattr(profile, "measured", False):
        return ""
    return str(getattr(profile, "fingerprint", "") or "")


def _assert_proof_primary_artifact_identity(primary_path: str, target_path: str) -> None:
    """Require measured, equal checkpoint identity for a proof-primary lane."""

    primary_fingerprint = _measured_artifact_fingerprint(primary_path)
    target_fingerprint = _measured_artifact_fingerprint(target_path)
    if not primary_fingerprint or not target_fingerprint:
        missing = "primary" if not primary_fingerprint else "target"
        raise RuntimeError(
            "Proof-primary run refused an unmeasured checkpoint identity: "
            f"{missing} artifact fingerprint unavailable"
        )
    if primary_fingerprint != target_fingerprint:
        raise RuntimeError(
            "Proof-primary run refused a different checkpoint: "
            f"{os.path.basename(target_path)}({target_fingerprint[:12]}) != "
            f"{os.path.basename(primary_path)}({primary_fingerprint[:12]})"
        )


def _measured_size_class(model_path: Any) -> str | None:
    """The artifact's measured weight class, or None when unmeasured."""
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(str(model_path or ""))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return profile.size_class if getattr(profile, "measured", False) else None


def _model_class_tokens(model_path: Any) -> tuple[str, ...]:
    """Classification tokens for a model, measured evidence preferred.

    Returns the measured class as a single token when the artifact could be
    read, so callers match on evidence; otherwise the lowercased path, so the
    historical naming fallback still works on an unreadable artifact.
    """
    measured = _measured_size_class(model_path)
    if measured and measured != "unknown":
        return (measured,)
    return (str(model_path or "").lower(),)


def _model_matches_class(model_path: Any, tokens: tuple[str, ...]) -> bool:
    haystacks = _model_class_tokens(model_path)
    return any(token in haystack for haystack in haystacks for token in tokens)


def _model_is_quantized(model_path: Any) -> bool:
    """Whether the checkpoint's own metadata says it is quantized.

    CP126 48f80787: this was a substring test for "4bit", "q4", "fused-model"
    and a date stamp in the path. Quantization changes the memory footprint by
    more than a third; reading it off a directory name means a rename silently
    moves the admission gate.
    """
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(str(model_path or ""))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        profile = None
    bits = int(getattr(profile, "quantization_bits", 0) or 0)
    if bits:
        return bits <= 8
    lowered = str(model_path or "").lower()
    return any(token in lowered for token in ("4bit", "q4", "fused-model", "20260510"))


#: No lifecycle wait in this module is legitimately longer than an hour. A
#: value above it is a typo or a unit mistake (milliseconds pasted as seconds),
#: and honouring it turns a bounded wait into a hang.
_MAX_DURATION_S = 3600.0


def _finite_env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Env float with a fail-safe contract: malformed, NaN, or infinite
    values fall back to the default instead of poisoning admission math.

    CP126 ec9f8d32: several duration settings were parsed with a bare
    ``float()`` and a ``max()`` floor. A floor does not stop infinity, so
    ``AURA_MLX_SPAWN_FILE_LOCK_TIMEOUT_S=inf`` produced a deadline that never
    arrived — a bounded wait turned into a permanent one by a config string.
    Every duration now comes through here, and out-of-range means the default,
    not the extreme.
    """
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        logger.warning(
            "⚙️ [MLX] %s=%s exceeds the %.0fs ceiling; using the default %.1fs.",
            name,
            value,
            maximum,
            default,
        )
        return default
    return value


def _env_duration_s(name: str, default: float, *, minimum: float = 0.0) -> float:
    """A duration setting: finite, at least ``minimum``, at most an hour."""
    return _finite_env_float(name, default, minimum=minimum, maximum=_MAX_DURATION_S)


_SPAWN_GATE_ACQUIRE_TIMEOUT_S = _env_duration_s(
    "AURA_SPAWN_GATE_ACQUIRE_TIMEOUT_S", 5.0, minimum=0.05
)


def _model_load_min_available_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        return _finite_env_float(name, default, minimum=0.0)

    try:
        total_gb = float(psutil.virtual_memory().total) / float(1024**3)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, psutil.Error):
        total_gb = 0.0
    if _model_matches_class(model_path, ("72b", "solver")):
        default = 52.0 if 0.0 < total_gb < 96.0 else 34.0
        return _env_float("AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB", default)
    if _model_matches_class(model_path, ("32b", "cortex", "zenith")):
        # Derive the requirement from the model actually on disk rather than a
        # constant that happens to be wrong for it. Measured 2026-07-25: the
        # resident 32B is 17.2GB on disk and the flat 24.0 gate refused it on a
        # host sitting at 20.4GB available — a 6.8GB margin over true need, and
        # the cortex starved through six deaths in one run because of it.
        #
        # weights x 1.20 + 1GB covers KV cache and activations for a normal
        # context with room to spare. The flat default remains the CEILING, so
        # this can only ever relax toward the real footprint, never tighten
        # past a deliberate operator setting — and it floors at 16GB so a
        # mis-sized or unreadable model directory cannot wave a load through.
        default = 24.0 if total_gb >= 60.0 else 22.0
        measured = _measured_model_footprint_gb(model_path)
        if measured is not None:
            derived = measured * 1.20 + 1.0
            default = max(16.0, min(default, derived))
        return _env_float("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", default)
    return _env_float("AURA_MLX_LOAD_MIN_AVAILABLE_GB", 8.0)


def _measured_model_footprint_gb(model_path: Any) -> float | None:
    """Total size of the model directory in GB, or None if unreadable.

    Returns None on anything surprising — a missing directory, a permission
    error, an implausible size — so the caller keeps its conservative default.
    """
    text = str(model_path or "").strip()
    if not text:
        return None  # Path("") is the CWD, which is a real directory
    try:
        root = pathlib.Path(text)
        if not root.is_dir():
            return None
        total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    gb = total / float(1024**3)
    if not (1.0 < gb < 200.0):
        return None
    return gb


def _env_projected_footprint_gb(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "auto", "detect", "detected"}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    # A zero/negative override makes a real multi-GB worker appear free and
    # NaN/inf poisons every downstream admission sum — ignore such overrides.
    if not math.isfinite(value) or value <= 0.0:
        logger.warning("Ignoring invalid projected-footprint override %s=%r.", name, raw)
        return None
    return value


# Model artifacts are immutable while the runtime holds them (fusion
# publishes a NEW directory), so their size is computed once per
# (path, mtime) and reused. Uncached, this rglob+stat walk ran on the
# EVENT LOOP inside model-load admission while 20GB of safetensors reads
# saturated the disk — the 5.5-8.6s loop stalls captured in
# data/error_logs/stalls/stall_1784673149 / stall_1784675621 bottom out
# exactly here (pathlib stat under _projected_footprint_from_artifact_gb).
#: Extensions that hold model weights. Everything else in a checkpoint
#: directory — tokenizer caches, logs, receipts, adapters, temp files — is not
#: what gets loaded into memory, so it is not part of the footprint that RAM
#: admission is computed from (CP126 50d8ed03).
_WEIGHT_FILE_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".gguf", ".npz", ".pt", ".pth"}
)
#: Depth and count ceilings for the artifact scan. A checkpoint's weights sit
#: at the top level or one directory below it; a scan that follows an arbitrary
#: tree is unbounded work on an admission path.
_MAX_ARTIFACT_SCAN_DEPTH = 2
_MAX_ARTIFACT_FILES_SCANNED = 512


def _weight_files(root: Path):
    """Weight files within the artifact, bounded in depth."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if depth + 1 < _MAX_ARTIFACT_SCAN_DEPTH:
                        stack.append((entry, depth + 1))
                    continue
                if entry.suffix.lower() in _WEIGHT_FILE_SUFFIXES:
                    yield entry
            except OSError:
                continue


_PATH_SIZE_CACHE: dict[tuple[str, int], float] = {}


def _path_size_gb(model_path: str) -> float:
    path = Path(str(model_path or "")).expanduser()
    try:
        if path.is_file():
            return float(path.stat().st_size) / float(1024**3)
        if not path.is_dir():
            return 0.0
        cache_key = (str(path), path.stat().st_mtime_ns)
        cached = _PATH_SIZE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # CP126 50d8ed03: this walked EVERY descendant and counted every file,
        # so tokenizer caches, training logs, adapters, receipts and temporary
        # artifacts inflated the "model footprint" that RAM admission is
        # computed from — and a directory with a deep subtree made the walk
        # unbounded. What the footprint means is the weights that get loaded,
        # so only weight files count, only the top two levels are walked, and
        # the walk stops at a file ceiling rather than running as long as the
        # tree is deep.
        total = 0
        scanned = 0
        for child in _weight_files(path):
            try:
                total += child.stat().st_size
            except OSError:
                continue
            scanned += 1
            if scanned >= _MAX_ARTIFACT_FILES_SCANNED:
                logger.debug(
                    "Artifact size scan for %s stopped at %d files.",
                    path,
                    scanned,
                )
                break
        size_gb = float(total) / float(1024**3)
        if len(_PATH_SIZE_CACHE) > 64:
            _PATH_SIZE_CACHE.clear()
        _PATH_SIZE_CACHE[cache_key] = size_gb
        return size_gb
    except OSError:
        return 0.0


def _projected_footprint_from_artifact_gb(model_path: str, *, fallback_gb: float) -> float:
    """Estimate live model footprint from the local artifact when possible.

    The launcher previously used one static 32B projection for every artifact.
    That is too blunt for Aura: the active fused 4-bit model is materially
    smaller than the old 8-bit base artifact, while a genuine 8-bit path should
    still be treated as too expensive for a tight desktop process cap.
    """

    size_gb = _path_size_gb(model_path)
    if size_gb <= 0.0:
        return fallback_gb
    if _model_matches_class(model_path, ("72b", "solver")):
        overhead = max(4.0, size_gb * 0.14)
    elif _model_matches_class(model_path, ("32b", "cortex", "zenith", "aura-32b")):
        overhead = max(3.0, size_gb * 0.30)
    else:
        overhead = max(1.0, size_gb * 0.20)
    return max(1.0, size_gb + overhead)


def _projected_model_footprint_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        return _finite_env_float(name, default, minimum=0.0)

    if _model_matches_class(model_path, ("72b", "solver")):
        override = _env_projected_footprint_gb("AURA_MLX_72B_PROJECTED_FOOTPRINT_GB")
        if override is not None:
            return override
        return _projected_footprint_from_artifact_gb(model_path, fallback_gb=41.0)
    if _model_matches_class(model_path, ("32b", "cortex", "zenith")):
        override = _env_projected_footprint_gb("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB")
        if override is not None:
            return override
        # Quantization changes the footprint by more than a third, and it is
        # a property of the checkpoint, not of its directory name. Ask the
        # artifact; fall back to the name only when it cannot be read.
        quantized = _model_is_quantized(model_path)
        default = 20.0 if quantized else 35.0
        return _projected_footprint_from_artifact_gb(model_path, fallback_gb=default)
    if _model_matches_class(model_path, ("14b",)):
        return _env_float("AURA_MLX_14B_PROJECTED_FOOTPRINT_GB", 10.0)
    if _model_matches_class(model_path, ("7b",)):
        return _env_float("AURA_MLX_7B_PROJECTED_FOOTPRINT_GB", 5.0)
    return _env_float("AURA_MLX_PROJECTED_FOOTPRINT_GB", 4.0)


def _model_process_reserve_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        return _finite_env_float(name, default, minimum=0.0)

    if _model_matches_class(model_path, ("72b", "solver")):
        lane_default = _env_float("AURA_MLX_72B_PROCESS_RESERVE_GB", 5.0)
    elif _model_matches_class(model_path, ("32b", "cortex", "zenith")):
        lane_default = _env_float("AURA_MLX_32B_PROCESS_RESERVE_GB", 3.0)
    else:
        lane_default = _env_float("AURA_MLX_PROCESS_RESERVE_GB", 1.0)
    return _env_float("AURA_MLX_MODEL_LOAD_PROCESS_RESERVE_GB", lane_default)


def _declared_mlx_worker_footprint_gb(model_path: str) -> float:
    """Declared peak for the main worker plus optional in-worker model owners."""

    declared = _projected_model_footprint_gb(model_path) + _model_process_reserve_gb(model_path)
    from core.runtime.flags import FlagKind, declare

    contrastive_enabled = bool(
        declare(
            "AURA_CONTRASTIVE_DECODING",
            kind=FlagKind.BOOL,
            default=False,
            description="Enable contrastive decoding with an amateur model",
            owner="core.brain.llm.mlx_client",
        ).value()
    )
    amateur_path = str(
        declare(
            "AURA_CONTRASTIVE_AMATEUR_MODEL",
            kind=FlagKind.STRING,
            default="",
            description="Amateur model path for contrastive decoding",
            owner="core.brain.llm.mlx_client",
        ).value()
        or ""
    ).strip()
    if (
        contrastive_enabled
        and amateur_path
        and _real_model_path(amateur_path) != _real_model_path(model_path)
    ):
        declared += _projected_model_footprint_gb(amateur_path) + 1.0
    return declared


def _memory_pressure_blocks_worker_spawn(model_path: str) -> str | None:
    # The operator bypass stays available for recovery, but it is a DECISION,
    # not a setting: it is time-bounded, use-bounded and receipted, so a flag
    # left in a launch profile cannot silently disable spawn admission for the
    # life of a deployment. When the window closes the guard re-arms itself.
    from core.brain.llm.emergency_override import consume_override

    decision = consume_override(
        "AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE",
        guard="memory_pressure_spawn_admission",
        observed=f"spawn of {os.path.basename(model_path)}",
    )
    if decision.active:
        _record_mlx_degradation(
            RuntimeError(decision.as_detail()),
            action="bypassed memory-pressure spawn admission via governed operator override",
            severity="warning",
        )
        return None
    try:
        snapshot = get_memory_pressure_snapshot()
    except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # Fail CLOSED: spawning a 20-40GB worker with NO capacity observation
        # is exactly the moment conservative admission matters. The caller
        # treats this like any other transient spawn blocker and retries.
        _record_mlx_degradation(
            exc,
            action="refused worker spawn while the memory probe was unavailable",
            severity="error",
        )
        return "memory_probe_unavailable"

    min_available_gb = _model_load_min_available_gb(model_path)
    if snapshot.refuse_heavy_local_generation:
        return snapshot.reason or "critical_memory_pressure"
    if snapshot.available_gb < min_available_gb:
        return (
            f"model_load_headroom:{snapshot.available_gb:.1f}GB < required {min_available_gb:.1f}GB"
        )
    process_rss_gb = float(getattr(snapshot, "process_rss_gb", 0.0) or 0.0)
    process_rss_limit_gb = float(getattr(snapshot, "process_rss_limit_gb", 0.0) or 0.0)
    process_reserve_gb = _model_process_reserve_gb(model_path)
    projected_footprint_gb = _declared_mlx_worker_footprint_gb(model_path) - process_reserve_gb
    projected_process_rss_gb = process_rss_gb + projected_footprint_gb + process_reserve_gb
    if (
        process_rss_limit_gb > 0.0
        and projected_footprint_gb > 0.0
        and projected_process_rss_gb > process_rss_limit_gb
    ):
        return (
            f"projected_process_tree_rss:{process_rss_gb:.1f}GB"
            f"+{projected_footprint_gb:.1f}GB"
            f"+reserve{process_reserve_gb:.1f}GB={projected_process_rss_gb:.1f}GB "
            f"> limit {process_rss_limit_gb:.1f}GB"
        )
    return None


def _observed_active_lanes(exclude_client: Any = None) -> list[ActiveLane]:
    """Snapshot every live model lane as a declared-footprint ActiveLane.

    Pull-model observation over _CLIENTS: no bookkeeping to desync. The
    candidate's own client is excluded so a worker recycle never counts its
    old footprint against its own respawn.
    """
    from core.brain.lane_admission import ActiveLane

    lanes: list[ActiveLane] = []
    for path, client in _clients_snapshot():
        if client is None or client is exclude_client:
            continue
        try:
            if not client.is_alive():
                continue
        except (AttributeError, RuntimeError, OSError, ValueError):
            continue
        lane, qos = _client_lane_policy(client)
        last = float(getattr(client, "_last_user_facing_completed_at", 0.0) or 0.0)
        lanes.append(
            ActiveLane(
                lane=lane,
                qos=qos,
                footprint_gb=_declared_mlx_worker_footprint_gb(path),
                model_path=path,
                last_user_facing_age_s=(time.time() - last) if last > 0.0 else None,
            )
        )
    return lanes


def _model_lane_owner_id(client: Any) -> str:
    """A lane-owner id that identifies THIS client, not just its model.

    CP126 cdbb177d: the id was ``mlx:<parent pid>:<model path>``, so two
    clients for the same artifact — or the same client across a worker
    recycle — shared one identity. Exact eviction then could not name which
    owner to evict, and durable fencing could not tell a stale generation from
    the live one: revoking "the 32B owner" revoked whichever happened to
    answer. The client's own object identity and its generation counter make
    it unique without depending on the worker process existing yet.
    """
    existing = str(getattr(client, "_model_lane_owner_id", "") or "")
    if existing:
        return existing
    model_path = _real_model_path(getattr(client, "model_path", ""))
    generation = int(getattr(client, "_worker_generation", 0) or 0)
    owner_id = f"mlx:{os.getpid()}:{id(client):x}:{generation}:{model_path}"
    try:
        client._model_lane_owner_id = owner_id
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return owner_id


def _observed_model_lane_owners(exclude_client: Any = None) -> list[Any]:
    """Return process-identified MLX owners for durable lane accounting."""

    from core.brain.lane_admission import QoSClass
    from core.runtime.model_lane_control import (
        LaneOwnerObservation,
        process_identity_for_pid,
    )

    owners: list[LaneOwnerObservation] = []
    for path, client in _clients_snapshot():
        if client is None or client is exclude_client:
            continue
        try:
            process = getattr(client, "_process", None)
            if process is None or not process.is_alive() or not client.is_alive():
                continue
            pid = int(getattr(process, "pid", 0) or 0)
            if pid <= 0:
                continue
            lane, qos = _client_lane_policy(client)
            last = float(getattr(client, "_last_user_facing_completed_at", 0.0) or 0.0)
            observed_gb = float(_observed_process_rss_bytes(pid)) / float(1024**3)
            owners.append(
                LaneOwnerObservation(
                    owner_id=_model_lane_owner_id(client),
                    model_path=path,
                    declared_gb=_declared_mlx_worker_footprint_gb(path),
                    observed_gb=observed_gb,
                    process=process_identity_for_pid(pid),
                    priority=10 if qos is QoSClass.GUARANTEED else 50,
                    preemptible=not bool(
                        int(getattr(client, "_active_generations", 0) or 0) > 0
                        or (
                            getattr(client, "_current_gen_future", None) is not None
                            and not client._current_gen_future.done()
                        )
                    ),
                    last_user_facing_age_s=(time.time() - last) if last > 0.0 else None,
                    runtime_assignment=client.runtime_assignment,
                    metadata={
                        "runtime_pid": os.getpid(),
                        "lane": lane,
                        "fencing_token": int(getattr(client, "_model_lane_fencing_token", 0) or 0),
                    },
                )
            )
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError):
            continue
    return owners


def _transient_runtime_footprint_gb(owners: list[Any]) -> float:
    """Measure Aura process-tree memory not owned by model workers.

    Lane arbitration historically counted checkpoint workers while spawn
    admission counted the complete Aura tree. During primary recovery that
    disagreement let an idle fallback look compatible, then made the physical
    spawn gate reject the 32B. This cost is reservation-only so Aura's base
    runtime is not permanently double-counted on committed model owners.
    """

    try:
        snapshot = get_memory_pressure_snapshot()
        process_rss_gb = max(0.0, float(snapshot.process_rss_gb or 0.0))
        observed_worker_gb = sum(
            max(0.0, float(getattr(owner, "observed_gb", 0.0) or 0.0)) for owner in owners
        )
    except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0
    return max(0.0, process_rss_gb - observed_worker_gb)


#: How long an eviction waits to fence a lane before giving up. Short on
#: purpose: a lane that will not go quiet quickly is a lane doing work, and
#: the answer to that is to refuse the eviction, not to outwait it. Must
#: exceed the request-lock reserve (2s for background work) or the wait
#: budget resolves to zero and the fence can never be taken.
_LANE_EVICTION_FENCE_WAIT_S = 6.0

#: How long the response listener will wait for a durable lease renewal before
#: treating it as a fence loss. The listener is the ONLY consumer of the
#: worker's response queue, so time spent here is time no request is delivered.
_LEASE_RENEWAL_TIMEOUT_S = 5.0


def _lane_owner_is_working(target: Any) -> bool:
    """Whether this client has work in flight, by every signal it publishes."""
    return bool(
        int(getattr(target, "_active_generations", 0) or 0) > 0
        or getattr(target, "_warmup_in_flight", False)
        or getattr(target, "_current_request_started_at", 0.0) > 0.0
        or any(
            future is not None and not future.done()
            for future in (
                *getattr(target, "_pending_generations", {}).values(),
                getattr(target, "_current_gen_future", None),
                getattr(target, "_init_future", None),
            )
        )
    )


async def _evict_model_lane_owner(owner: Any, reason: str) -> bool:
    """Evict one exact local MLX owner and prove its worker is dead."""

    target = next(
        (
            client
            for _path, client in _clients_snapshot()
            if client is not None and _model_lane_owner_id(client) == owner.owner_id
        ),
        None,
    )
    if target is None:
        from core.runtime.model_lane_control import evict_managed_process_owner

        return await evict_managed_process_owner(owner, reason)
    if _lane_owner_is_working(target):
        logger.info(
            "MLX model-lane preemption refused during active work owner=%s reason=%s",
            owner.owner_id,
            reason,
        )
        return False

    # CP126 518e876f: the idleness test above reads mutable fields and the
    # reboot below is an await. A generation that started in that window was
    # killed mid-decode by an eviction whose whole premise was that nothing
    # was running — the "refused during active work" branch above existed
    # precisely to prevent that, and the gap made it advisory.
    #
    # Take the request lane first, so nothing can start, then re-test what is
    # true while holding it.
    lane_deadline = get_deadline(_LANE_EVICTION_FENCE_WAIT_S)
    fenced = await target._acquire_request_lock(
        owner_label=f"lane_eviction:{reason}",
        deadline=lane_deadline,
        foreground_request=False,
    )
    if not fenced:
        logger.info(
            "MLX model-lane preemption refused: request lane busy owner=%s reason=%s",
            owner.owner_id,
            reason,
        )
        return False
    try:
        if _lane_owner_is_working(target):
            logger.info(
                "MLX model-lane preemption refused: work started during the fence "
                "owner=%s reason=%s",
                owner.owner_id,
                reason,
            )
            return False
        await target.reboot_worker(
            reason=f"yield_to_lane_transaction:{reason}",
            mark_failed=False,
        )
    finally:
        target._release_request_lock()
    try:
        alive = bool(target.is_alive())
    except (AttributeError, RuntimeError, OSError, ValueError):
        alive = True
    return not alive


async def _reclaim_model_lane_capacity(claim: Any) -> bool:
    from core.runtime.flags import FlagKind, declare

    """Wait boundedly for killed model memory to leave the observed envelope."""

    try:
        timeout_s = max(
            0.0,
            float(
                declare(
                    "AURA_MODEL_LANE_RECLAIM_TIMEOUT_S",
                    kind=FlagKind.FLOAT,
                    default=20.0,
                    description="Budget for reclaiming a model lane before spawn",
                    owner="core.brain.llm.mlx_client",
                ).value()
            ),
        )
    except (TypeError, ValueError):
        timeout_s = 20.0
    deadline = time.monotonic() + timeout_s
    max_observations = max(1, int(timeout_s / 0.5) + 2)
    blocker: str | None = "capacity_not_observed"
    for _attempt in range(max_observations):
        blocker = await asyncio.to_thread(
            _memory_pressure_blocks_worker_spawn,
            claim.model_path,
        )
        if blocker is None:
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        await asyncio.sleep(min(0.5, remaining_s))
    logger.warning(
        "Model-lane reservation could not observe reclaimed capacity for %s: %s",
        os.path.basename(claim.model_path),
        blocker,
    )
    return False


async def _compensate_model_lane_owner(owner: Any, reason: str) -> bool:
    """Restore an owner displaced by a candidate that did not commit."""

    target = next(
        (
            client
            for _path, client in _clients_snapshot()
            if client is not None and _model_lane_owner_id(client) == owner.owner_id
        ),
        None,
    )
    if target is None:
        return False

    timeout_s = max(60.0, float(target._warmup_timeout()) + 30.0)
    try:
        restored = bool(
            await asyncio.wait_for(
                target.warmup(skip_swap_cooldown=True),
                timeout=timeout_s,
            )
        )
        ready = bool(restored and target.is_alive())
        if not ready:
            logger.warning(
                "Compensation could not restore model lane owner=%s reason=%s",
                owner.owner_id,
                reason,
            )
        return ready
    except asyncio.CancelledError:
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
        _record_mlx_degradation(
            exc,
            action="recorded failed restoration of an evicted model lane",
            severity="error",
        )
        return False


def _note_lane_worker_death(client: Any, reason: str) -> None:
    """Report a worker death to the crash-loop breaker (roadmap K4).

    Lifetime is measured from the spawn timestamp; the breaker itself
    decides whether the death counts (young + non-deliberate). Never
    throws — death accounting must not break a recovery path.
    """
    try:
        started = float(getattr(client, "_process_started_at", 0.0) or 0.0)
        if started <= 0.0:
            return
        # Recorded locally FIRST, so the fallback breaker has evidence even
        # when the breaker module is the thing that is unavailable.
        _note_local_worker_death(getattr(client, "model_path", ""))
        from core.runtime.lane_reconciler import get_crash_loop_breaker

        get_crash_loop_breaker().note_death(
            _real_model_path(client.model_path),
            lifetime_s=max(0.0, time.time() - started),
            reason=reason,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Crash-loop death report skipped: %s", exc)
    # The dead worker's MODEL_LOAD admission lease must die with it: a
    # MODEL_LOAD lease conflicts with every other MODEL_LOAD lease, so an
    # unreleased lease walls every recovery load behind its TTL while each
    # retry burns to resource_timeout — the 2026-07-15 soak P0 (cortex
    # never loaded all night while RAM sat at 40%). Same seam as the K4
    # report, same never-throws contract.
    try:
        from core.runtime.control_plane import WorkClass, get_runtime_control_plane

        lane = _lane_for_dead_client(client)
        reaped = get_runtime_control_plane().admission.reap_dead_holder_leases_sync(
            lane=lane,
            work_class=WorkClass.MODEL_LOAD,
            reason=reason,
        )
        if reaped:
            logger.warning(
                "🧹 Reaped %d orphaned model-load admission lease(s) for dead %s worker (%s).",
                reaped,
                lane,
                reason,
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # Not a debug line. A lease that is not reaped walls every recovery
        # load behind its TTL while each retry burns to resource_timeout —
        # the 2026-07-15 soak P0, where the cortex never loaded all night
        # with RAM at 40%.
        _record_mlx_degradation(
            exc,
            action="left a dead worker's model-load admission lease unreaped",
            severity="warning",
        )


def _lane_for_dead_client(client: Any) -> str:
    """The lane whose lease a dead worker was holding.

    The immutable assignment is the authority, and for a live client it always
    answers. A client being torn down may not: the assignment is the first
    thing a half-constructed or already-retired client is missing, and the
    reap must not be the casualty of that. Falling back to the lane the
    registry would classify this artifact into keeps the lease from
    outliving the process that held it.
    """
    try:
        lane, _qos = _client_lane_policy(client)
        return lane
    except (AttributeError, RuntimeError, TypeError, ValueError):
        from core.brain.lane_admission import classify_lane

        lane, _qos = classify_lane(str(getattr(client, "model_path", "") or ""))
        return lane


def _lane_is_last_warm(client: Any) -> bool:
    """K5 disruption budget: is this client the ONLY live model lane?

    Voluntary disruptions (yields for background warmups) must never
    remove the last warm lane — a cold gap with nothing warm is strictly
    worse than deferring a background spawn.
    """
    try:
        if client is None or not client.is_alive():
            return False
        for _other_path, other in _clients_snapshot():
            if other is None or other is client:
                continue
            try:
                if other.is_alive():
                    return False
            except (AttributeError, RuntimeError, OSError, ValueError):
                continue
        return True
    except (AttributeError, RuntimeError, OSError, ValueError):
        return False


#: Fallback crash-loop evidence when the breaker module is unavailable. This
#: process reports every worker death it sees; that record is enough to refuse
#: a respawn during a storm (CP126 21c6730b).
_LOCAL_DEATH_LEDGER: dict[str, list[float]] = {}
_LOCAL_DEATH_LEDGER_LOCK = _threading.Lock()
_LOCAL_CRASH_LOOP_WINDOW_S = 120.0
_LOCAL_CRASH_LOOP_DEATHS = 3


def _note_local_worker_death(model_path: str) -> None:
    key = _real_model_path(model_path)
    now = time.time()
    with _LOCAL_DEATH_LEDGER_LOCK:
        deaths = [
            stamp
            for stamp in _LOCAL_DEATH_LEDGER.get(key, [])
            if now - stamp <= _LOCAL_CRASH_LOOP_WINDOW_S
        ]
        deaths.append(now)
        _LOCAL_DEATH_LEDGER[key] = deaths[-16:]


def _local_crash_loop_block(client: Any) -> str | None:
    key = _real_model_path(getattr(client, "model_path", ""))
    now = time.time()
    with _LOCAL_DEATH_LEDGER_LOCK:
        recent = [
            stamp
            for stamp in _LOCAL_DEATH_LEDGER.get(key, [])
            if now - stamp <= _LOCAL_CRASH_LOOP_WINDOW_S
        ]
        _LOCAL_DEATH_LEDGER[key] = recent
    if len(recent) >= _LOCAL_CRASH_LOOP_DEATHS:
        return (
            f"local_crash_loop:{len(recent)}_deaths_in_"
            f"{int(_LOCAL_CRASH_LOOP_WINDOW_S)}s"
        )
    return None


def _crash_loop_blocks_worker_spawn(client: Any) -> str | None:
    """Consult the K4 crash-loop breaker before a (re)spawn. Never throws."""
    try:
        from core.runtime.lane_reconciler import get_crash_loop_breaker
    except ImportError as exc:
        # CP126 21c6730b: an absent breaker used to mean an unchecked respawn,
        # which is exactly what a crash storm needs to keep going. The absence
        # of the module is not the absence of evidence: this process has been
        # recording its own worker deaths all along, so fall back to them.
        _record_mlx_degradation(
            exc,
            action="consulted the local death ledger after the crash-loop breaker module was unavailable",
        )
        return _local_crash_loop_block(client)
    try:
        blocked = get_crash_loop_breaker().blocked(_real_model_path(client.model_path))
        return str(blocked) if blocked else None
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # Fail CLOSED: the breaker exists but is broken — during an active
        # crash storm this is precisely when unchecked respawns do damage.
        _record_mlx_degradation(
            exc,
            action="refused worker spawn while the crash-loop breaker was unavailable",
            severity="error",
        )
        return "crash_loop_breaker_unavailable"


class _ModelLoadAdmissionDeniedError(RuntimeError):
    def __init__(self, reason: str, *, receipt_id: str = "") -> None:
        self.reason = str(reason or "resource_admission_denied")
        self.receipt_id = str(receipt_id or "")
        super().__init__(
            f"model_load_admission_denied:{self.reason}"
            + (f":receipt={self.receipt_id}" if self.receipt_id else "")
        )


def _model_load_admission_timeout_s(*, foreground_request: bool) -> float:
    flag = (
        _MODEL_LOAD_FOREGROUND_ADMISSION_TIMEOUT_FLAG
        if foreground_request
        else _MODEL_LOAD_BACKGROUND_ADMISSION_TIMEOUT_FLAG
    )
    return max(0.0, float(flag.value()))


def _model_load_lease_ttl_s(client: Any) -> float:
    """Cover the complete worker handshake, not only warmup precompile.

    The primary lane allows a 300-second init handshake. The old 240-second
    lease expired while that live load still held the spawn gate, admitting a
    second load attempt behind it and creating the observed recovery cascade.
    """

    return max(
        180.0,
        float(client._warmup_timeout()) + 120.0,
        float(client._handshake_timeout()) + 120.0,
    )


@contextlib.asynccontextmanager
async def _model_load_admission_context(
    client: Any,
    *,
    foreground_request: bool,
) -> AsyncIterator[Any]:
    """Hold scheduling and durable capacity reservations through handshake.

    Required evictions complete while the durable reservation remains counted.
    The candidate is committed as a process-identified owner only after its
    worker handshake succeeds; every other exit cancels and compensates.
    """

    try:
        from core.runtime.control_plane import (
            AdmissionPriority,
            AdmissionRequest,
            WorkClass,
            get_runtime_control_plane,
        )
        from core.runtime.model_lane_control import (
            LaneClaim,
            get_model_lane_controller,
            process_identity_for_pid,
        )
    except ImportError as exc:
        _record_mlx_degradation(
            exc,
            action="refused model load because canonical resource admission could not import",
            severity="critical",
        )
        raise _ModelLoadAdmissionDeniedError("resource_admission_unavailable") from exc

    lane, qos = _client_lane_policy(client)
    # Off-loop: the footprint projection stats the whole model directory,
    # and doing that on the event loop during a concurrent 20GB model read
    # produced the recorded 5.5-8.6s admission stalls. The walk is also
    # memoized, so this thread hop is cold-path only.
    request_gb = await asyncio.to_thread(_declared_mlx_worker_footprint_gb, client.model_path)
    timeout_s = _model_load_admission_timeout_s(foreground_request=foreground_request)
    from core.brain.lane_admission import QoSClass

    # The PRIMARY cortex (GUARANTEED QoS) always loads at FOREGROUND priority,
    # even when a background prewarm task triggered it. It is the user-facing
    # default model — background priority (80) meant the fairness gate blocked
    # its load behind every continuous foreground fallback inference (priority
    # 10), forever: the cortex could never load while the fallback answered,
    # and the fallback answered because the cortex never loaded (2026-07-15
    # soak deadlock, resource_timeout). At equal priority the load and the
    # fallback inference interleave FIFO, so the cortex finally comes up.
    is_primary_cortex = qos is QoSClass.GUARANTEED
    model_load_lease_ttl_s = _model_load_lease_ttl_s(client)
    request = AdmissionRequest(
        owner=f"mlx.model_load:{os.path.basename(client.model_path)}",
        work_class=WorkClass.MODEL_LOAD,
        lane=lane,
        priority=(
            AdmissionPriority.FOREGROUND
            if (foreground_request or is_primary_cortex)
            else AdmissionPriority.BACKGROUND
        ),
        timeout_s=timeout_s,
        lease_ttl_s=model_load_lease_ttl_s,
        receipt_required=True,
        estimated_memory_mb=request_gb * 1024.0,
        metadata={
            "model_path": str(client.model_path),
            "lane_qos": str(qos),
            "foreground_request": bool(foreground_request),
            "declared_request_gb": request_gb,
        },
    )
    try:
        admission = get_runtime_control_plane().admission
        decision = await admission.acquire(request)
    except asyncio.CancelledError:
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="refused model load because canonical resource admission failed",
            severity="critical",
        )
        raise _ModelLoadAdmissionDeniedError("resource_admission_failed") from exc
    if not decision.admitted:
        raise _ModelLoadAdmissionDeniedError(
            decision.reason,
            receipt_id=decision.receipt_id,
        )
    clear_admission_backoff = getattr(
        client,
        "_clear_model_load_admission_backoff",
        None,
    )
    if callable(clear_admission_backoff):
        clear_admission_backoff()

    lease_released = False

    async def _release_schedule_lease(reason: str) -> None:
        nonlocal lease_released
        if lease_released:
            return
        lease_released = True
        try:
            await admission.release(decision.lease_id, reason=reason)
        except KeyError:
            logger.warning(
                "Model-load admission lease expired before release lane=%s lease=%s",
                lane,
                decision.lease_id,
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="model load completed but canonical admission release failed",
                severity="warning",
            )

    lane_controller = get_model_lane_controller()
    disruptive_deep_handoff = bool(
        foreground_request
        and callable(getattr(client, "_is_deep_solver_lane", None))
        and client._is_deep_solver_lane()
    )
    observed_owners = _observed_model_lane_owners(exclude_client=client)
    transient_runtime_gb = _transient_runtime_footprint_gb(observed_owners)
    lane_claim = LaneClaim(
        owner_id=_model_lane_owner_id(client),
        model_path=str(client.model_path),
        request_gb=request_gb,
        transient_runtime_gb=transient_runtime_gb,
        priority=(
            int(AdmissionPriority.FOREGROUND)
            if (foreground_request or is_primary_cortex)
            else int(AdmissionPriority.BACKGROUND)
        ),
        foreground=foreground_request,
        allow_disruptive_eviction=disruptive_deep_handoff,
        allow_last_warm_eviction=is_primary_cortex or disruptive_deep_handoff,
        reservation_ttl_s=model_load_lease_ttl_s,
        request_id=f"model-lane-{request.request_id}",
        runtime_assignment=client.runtime_assignment,
        metadata={
            "scheduling_lease_id": decision.lease_id,
            "scheduling_receipt_id": decision.receipt_id,
            "model_path": str(client.model_path),
            "lane_qos": str(qos),
            "foreground_request": bool(foreground_request),
            "disruptive_deep_handoff": disruptive_deep_handoff,
            "transient_runtime_gb": transient_runtime_gb,
            "compensation_strategy": "mlx_warmup_exact_owner",
        },
    )
    lane_decision = None
    try:
        lane_decision = await lane_controller.reserve(
            lane_claim,
            observations=observed_owners,
        )
        if not lane_decision.admitted:
            await _release_schedule_lease("model_lane_reservation_refused")
            raise _ModelLoadAdmissionDeniedError(
                lane_decision.reason,
                receipt_id=lane_decision.receipt_id,
            )
        if not lane_decision.ready_to_spawn:
            lane_decision = await lane_controller.prepare(
                lane_decision,
                evict=_evict_model_lane_owner,
                observe=lambda: _observed_model_lane_owners(exclude_client=client),
                reclaim=_reclaim_model_lane_capacity,
                compensate=_compensate_model_lane_owner,
            )
        if not lane_decision.ready_to_spawn:
            await _release_schedule_lease("model_lane_eviction_or_reclamation_failed")
            raise _ModelLoadAdmissionDeniedError(
                lane_decision.reason,
                receipt_id=lane_decision.receipt_id,
            )

        yield decision
        process = getattr(client, "_process", None)
        pid = int(getattr(process, "pid", 0) or 0) if process is not None else 0
        worker_ready = bool(
            process is not None and process.is_alive() and getattr(client, "_init_done", False)
        )
        if not worker_ready:
            await _release_schedule_lease("model_load_did_not_reach_ready")
            cancelled = await lane_controller.cancel(
                lane_decision,
                reason="candidate_worker_not_ready",
                compensate=_compensate_model_lane_owner,
            )
            raise _ModelLoadAdmissionDeniedError(
                cancelled.reason,
                receipt_id=cancelled.receipt_id,
            )
        observed_gb = float(_observed_process_rss_bytes(pid)) / float(1024**3)
        try:
            committed = await lane_controller.commit(
                lane_decision,
                process=process_identity_for_pid(pid),
                observed_gb=observed_gb,
                metadata={"worker_name": str(getattr(process, "name", ""))},
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="killed candidate worker because durable lane commit failed",
                severity="critical",
            )
            await client.reboot_worker(reason="lane_commit_failed", mark_failed=True)
            await _release_schedule_lease("model_lane_commit_failed")
            cancelled = await lane_controller.cancel(
                lane_decision,
                reason=f"candidate_commit_failed:{type(exc).__name__}",
                compensate=_compensate_model_lane_owner,
            )
            raise _ModelLoadAdmissionDeniedError(
                cancelled.reason,
                receipt_id=cancelled.receipt_id,
            ) from exc
        adopt_owner = getattr(client, "_adopt_durable_model_lane_owner", None)
        if callable(adopt_owner):
            adopt_owner(
                fencing_token=committed.fencing_token,
                receipt_id=committed.receipt_id,
            )
        else:
            client._model_lane_fencing_token = committed.fencing_token
            client._model_lane_terminal_receipt_id = committed.receipt_id
            from core.runtime.model_lane_control import register_model_lane_owner_adapter

            register_model_lane_owner_adapter(
                committed.owner_id,
                evict=_evict_model_lane_owner,
                compensate=_compensate_model_lane_owner,
            )
    except asyncio.CancelledError:
        await asyncio.shield(_release_schedule_lease("model_load_cancelled"))
        if lane_decision is not None and lane_decision.admitted:
            await asyncio.shield(
                lane_controller.cancel(
                    lane_decision,
                    reason="candidate_load_cancelled",
                    compensate=_compensate_model_lane_owner,
                )
            )
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        if lane_decision is not None and lane_decision.admitted:
            await _release_schedule_lease("model_load_failed")
            try:
                await lane_controller.cancel(
                    lane_decision,
                    reason="candidate_load_failed",
                    compensate=_compensate_model_lane_owner,
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_mlx_degradation(
                    exc,
                    action="model load failed and lane reservation cancellation also failed",
                    severity="critical",
                )
        raise
    finally:
        await _release_schedule_lease("model_load_finished")


def _normalize_recurrent_depth_status(status: Any, *, model_path: str) -> dict[str, Any]:
    payload = dict(status) if isinstance(status, dict) else {}
    expected_loops = payload.get("expected_loops")
    try:
        expected = (
            int(expected_loops)
            if expected_loops is not None
            else _expected_recurrent_loops_from_model_path(model_path)
        )
    except (TypeError, ValueError):
        expected = _expected_recurrent_loops_from_model_path(model_path)
    expected = max(0, expected)
    payload["expected_loops"] = expected
    payload["required"] = bool(payload.get("required", False)) or expected > 1
    payload.setdefault("active", False)
    payload.setdefault("config", None)
    payload.setdefault("reason", "")
    payload.setdefault("error", "")
    return payload


def _recurrent_depth_readiness_blocker(status: dict[str, Any]) -> str | None:
    if not bool(status.get("required", False)):
        return None
    if bool(status.get("active", False)) is not True:
        return "recurrent_depth_inactive"
    config = status.get("config")
    config_payload = config if isinstance(config, dict) else {}
    try:
        configured_loops = int(config_payload.get("n_loops") or 0)
    except (TypeError, ValueError):
        configured_loops = 0
    try:
        expected_loops = int(status.get("expected_loops") or 0)
    except (TypeError, ValueError):
        expected_loops = 0
    if expected_loops > 1 and configured_loops < expected_loops:
        return "recurrent_depth_loop_mismatch"
    return None


#: Whole origin labels that mean "a person is waiting on this". Matched
#: exactly against the normalized label, never by token (CP126 829ffbee).
#: Hyphens normalize to underscores, so both spellings are covered by one
#: entry.
_USER_FACING_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "desktop",
        "desktop_ui",
        # Both spellings are listed so the cross-surface origin contract holds
        # literally (tests/test_live_runtime_surface_regressions.py compares
        # the raw sets), while normalization keeps matching either form.
        "desktop-ui",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "native_shell",
        "native-shell",
        "test",
        # Compound labels the live surfaces actually send.
        "user_voice",
        "desktop_chat",
        "chat",
    }
)
_USER_FACING_PURPOSES = frozenset(
    {
        "chat",
        "conversation",
        "expression",
        "reply",
        "user_response",
    }
)


def _runtime_shutdown_requested() -> bool:
    return bool(is_shutdown_requested())


def _shutdown_blocks_model_work(model_path: str, *, action: str) -> bool:
    """Return true when shutdown has latched and model work must not start.

    This guard intentionally lives at the MLX boundary, not only in callers:
    recovery, prewarm, health, and chat paths all converge here. Once the
    process-wide shutdown latch is set, no worker spawn, warmup, or recovery
    admission may create new model work.
    """

    if not _runtime_shutdown_requested():
        return False
    record_shutdown_admission_event(
        f"mlx:{action}:{os.path.basename(str(model_path or '')) or 'unknown-model'}",
        resource_kind="mlx_worker",
        outcome="suppressed",
        detail="shutdown_latch",
    )
    logger.info(
        "🛑 [MLX] %s skipped for %s: runtime shutdown is latched.",
        action,
        os.path.basename(str(model_path or "")) or "unknown-model",
    )
    return True


def _open_spawn_lock_file(lock_file_path: str):
    """Open the spawn lock as a verified regular file we own.

    CP126 cb05a61b. The lock lived at a fixed path under the user's home and
    was opened with O_CREAT|O_WRONLY and no O_NOFOLLOW, then wrapped in write
    mode — which TRUNCATES whatever the path resolves to. Any other component
    running as the same user could replace mlx_spawn.lock with a symlink and
    have a worker spawn destroy an unrelated writable file.

    O_NOFOLLOW refuses a symlink at the final component; the fstat checks
    then confirm we are holding a regular file we own, with no extra hard
    links pointing at it. Opened r+ rather than w: a lock file is held, never
    written, so there is nothing to truncate.
    """
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_file_path, flags, 0o600)
    except OSError as exc:
        # ELOOP/EMLINK here means the path IS a symlink: a tampering signal,
        # not a transient error.
        raise RuntimeError(
            f"mlx_spawn_lock_unsafe:{lock_file_path}:{exc.__class__.__name__}"
        ) from exc
    try:
        st = os.fstat(lock_fd)
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(f"mlx_spawn_lock_not_regular_file:{lock_file_path}")
        if st.st_uid != os.getuid():
            raise RuntimeError(f"mlx_spawn_lock_foreign_owner:{lock_file_path}:uid={st.st_uid}")
        if st.st_nlink != 1:
            raise RuntimeError(f"mlx_spawn_lock_hardlinked:{lock_file_path}:links={st.st_nlink}")
        if st.st_mode & 0o077:
            # Group/other permissions on a lock another user could then hold.
            os.fchmod(lock_fd, 0o600)
    except Exception:
        os.close(lock_fd)
        raise
    return os.fdopen(lock_fd, "r+")


def _acquire_spawn_file_lock(lock_file: Any, *, model_path: str) -> None:
    """Acquire the cross-process spawn lock with timeout and shutdown polling."""

    try:
        timeout_s = _env_duration_s(
            "AURA_MLX_SPAWN_FILE_LOCK_TIMEOUT_S", 90.0, minimum=1.0
        )
    except (TypeError, ValueError):
        timeout_s = 90.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _shutdown_blocks_model_work(model_path, action="spawn lock wait"):
            raise RuntimeError("runtime_shutdown")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(0.1)
    raise TimeoutError(
        f"mlx_spawn_file_lock_timeout:{os.path.basename(model_path)}:{timeout_s:.1f}s"
    )


def _real_model_path(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        from .model_registry import is_model_repository_id

        if is_model_repository_id(raw):
            return raw
    except (ImportError, RuntimeError):
        pass
    return os.path.realpath(raw)


def _probe_cache_ttl_seconds(ok: bool | None, *, disk: bool) -> float:
    """Keep positive probe results sticky, but let failures expire quickly.

    A transient probe failure should not strand the embedded runtime in a
    "dead" state for many minutes after the host is healthy again.
    """
    if ok is None:
        return 0.0
    if ok:
        return 900.0 if disk else 300.0
    return 30.0 if disk else 10.0


def _safe_close_queue(q: mp.Queue | None) -> None:
    """Close an mp.Queue to release its shared-memory file descriptor."""
    if q is None:
        return

    def _close_and_join() -> None:
        q.close()
        q.join_thread()

    try:
        run_sync_shutdown_callable_blocking(
            _close_and_join,
            timeout_s=1.0,
            name="mlx-queue-close",
        )
    except (OSError, ValueError, BrokenPipeError, TypeError, AttributeError, TimeoutError) as exc:
        logger.debug("MLX queue cleanup did not complete: %s", exc)
    else:
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            get_runtime_hygiene().unregister_shutdown_resource(q)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass


def _register_runtime_queue(q: mp.Queue, *, name: str) -> None:
    from core.runtime.runtime_hygiene import get_runtime_hygiene

    get_runtime_hygiene().register_shutdown_resource(
        q,
        kind="multiprocessing_queue",
        name=name,
        source="core.brain.llm.mlx_client",
        timeout_s=1.0,
    )


def _new_shared_future() -> SharedFuture:
    """Create a loop-agnostic future for singleton clients shared across loops."""
    return cfutures.Future()


def _bounded_max_tokens(requested: Any, bridged: Any, fallback: int) -> int:
    """Shrink token budgets without ever handing MLX a zero-token generation."""

    def _coerce(value: Any) -> int:
        if value is None or value == "":
            return int(fallback)
        return int(value)

    try:
        requested_int = _coerce(requested)
    except (TypeError, ValueError, OverflowError):
        requested_int = int(fallback)
    try:
        bridged_int = _coerce(bridged)
    except (TypeError, ValueError, OverflowError):
        bridged_int = int(fallback)
    return max(1, min(max(1, requested_int), max(1, bridged_int)))


#: The last boundary before a prompt reaches the worker. inference_gate has
#: per-section and total budgets, but a path that assembles its own prompt
#: never meets them, and on 2026-08-03 one did: 88,659 / 78,861 / 91,441
#: characters. Prefill alone then consumed the whole request deadline —
#: "Request deadline reached at token 1", "produced 1 token but no text
#: survived" — so the answer came back empty, over and over.
#:
#: Generous on purpose. This is not the budget; it is the ceiling past which a
#: prompt cannot be answered at all, and no legitimate turn is near it.
_PREFILL_CEILING_CHARS = 48_000
#: The tail holds the actual question and the most recent exchange. The head
#: holds the system contract. What a runaway prompt buries is in the middle.
_PREFILL_KEEP_HEAD_CHARS = 12_000


def _prompt_within_prefill_ceiling(prompt: Any, *, model_path: str = "") -> str:
    """Bound a prompt so prefill cannot eat the whole request deadline.

    Keeps the head (the system contract) and the tail (the question and the
    latest exchange), drops the middle, and says so in the text so the model
    is not silently reasoning over a gap it cannot see.
    """

    text = str(prompt or "")
    if len(text) <= _PREFILL_CEILING_CHARS:
        return text

    # The marker counts against the ceiling too — it is prompt like any other.
    marker = (
        f"\n\n…[{len(text) - _PREFILL_CEILING_CHARS} characters omitted: this prompt "
        "exceeded the prefill ceiling and would not have been answered at all]…\n\n"
    )
    tail_budget = max(0, _PREFILL_CEILING_CHARS - _PREFILL_KEEP_HEAD_CHARS - len(marker))
    bounded = text[:_PREFILL_KEEP_HEAD_CHARS] + marker + text[-tail_budget:]
    logger.error(
        "🪓 [MLX] Prompt %d chars exceeded the %d prefill ceiling for %s — kept head+tail, "
        "dropped the middle. An unbounded prompt returns one token and no text.",
        len(text),
        _PREFILL_CEILING_CHARS,
        os.path.basename(str(model_path or "")) or "model",
    )
    _record_mlx_degradation(
        RuntimeError(f"prompt {len(text)} chars over prefill ceiling {_PREFILL_CEILING_CHARS}"),
        action="bounded the prompt to head+tail so the turn could produce an answer",
        severity="warning",
    )
    return bounded


#: The largest budget a caller-supplied output contract may demand. Past this
#: it is not a reply contract; it is an unbounded request wearing one.
_MAX_OUTPUT_CONTRACT_FLOOR_TOKENS = 8192


#: The legal range of every sampling parameter that crosses the IPC boundary.
#: (minimum, maximum, kind) — kind "f" is a float, "i" an int.
_SAMPLING_CONTRACT: dict[str, tuple[float, float, str]] = {
    "temp": (0.0, 2.0, "f"),
    "top_p": (0.0, 1.0, "f"),
    "top_k": (0.0, 1000.0, "i"),
    "min_p": (0.0, 1.0, "f"),
    "repetition_penalty": (0.5, 2.5, "f"),
    "repetition_context_size": (0.0, 4096.0, "i"),
    "presence_penalty": (-2.0, 2.0, "f"),
}
_SAMPLING_DEFAULTS: dict[str, float] = {
    "temp": 0.7,
    "top_p": 0.9,
    "top_k": 60,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "repetition_context_size": 30,
    "presence_penalty": 0.0,
}
_STOP_SEQUENCES_MAX = 16
_STOP_SEQUENCE_MAX_CHARS = 200


def _normalize_generation_params(req: dict[str, Any]) -> list[str]:
    """Bring the sampling parameters inside their contract before they ship.

    CP126 cac5c1a3: temperature, top-p, top-k, min-p, the penalties and the
    stop sequences were copied from arbitrary kwargs straight into IPC. No
    type check, no finite check, no range, no size bound. A caller could send
    ``temp="hot"`` or ``top_p=inf`` and the first thing to find out was the
    worker's sampler, on the far side of a process boundary, mid-decode.

    Out-of-contract values become the default rather than being clamped to the
    edge: a caller asking for a temperature of 40 has not asked for 2.0, it
    has made a mistake, and silently serving the extreme is how a mistake
    becomes a plausible-looking answer nobody can explain. Faults are
    returned so the receipt can carry them.
    """
    faults: list[str] = []
    for name, (low, high, kind) in _SAMPLING_CONTRACT.items():
        raw = req.get(name)
        default = _SAMPLING_DEFAULTS[name]
        if raw is None:
            req[name] = int(default) if kind == "i" else float(default)
            continue
        if isinstance(raw, bool):
            faults.append(f"{name}:not_a_number")
            req[name] = int(default) if kind == "i" else float(default)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            faults.append(f"{name}:not_a_number")
            req[name] = int(default) if kind == "i" else float(default)
            continue
        if not math.isfinite(value):
            faults.append(f"{name}:not_finite")
            req[name] = int(default) if kind == "i" else float(default)
            continue
        if value < low or value > high:
            faults.append(f"{name}:out_of_range")
            req[name] = int(default) if kind == "i" else float(default)
            continue
        req[name] = int(value) if kind == "i" else value

    raw_stops = req.get("stop_sequences")
    stops: list[str] = []
    if isinstance(raw_stops, (list, tuple)):
        if len(raw_stops) > _STOP_SEQUENCES_MAX:
            faults.append("stop_sequences:too_many")
        for entry in list(raw_stops)[:_STOP_SEQUENCES_MAX]:
            if not isinstance(entry, str):
                faults.append("stop_sequences:not_a_string")
                continue
            if len(entry) > _STOP_SEQUENCE_MAX_CHARS:
                faults.append("stop_sequences:too_long")
            stops.append(entry[:_STOP_SEQUENCE_MAX_CHARS])
    elif raw_stops:
        faults.append("stop_sequences:not_a_sequence")
    req["stop_sequences"] = stops
    return faults


def _bounded_generation_max_tokens(
    requested: Any,
    bridged: Any,
    hard_output_ceiling: Any,
    fallback: int,
    requested_output_contract: Any = None,
    *,
    user_surface_completion_floor: Any = None,
    preserve_user_surface_completion_floor: bool = False,
    tool_call_floor: Any = None,
    preserve_admitted_capacity: bool = False,
) -> int:
    """Apply adaptive shrinkage without making an admitted contract impossible.

    A CALL is one of those contracts, and it had no floor.

    Adaptive suggestions remain available for unowned work. An owned
    foreground answer retains its admitted capacity: stopping generation
    early cannot make either prose or a structured call more concise.

    LIVE, 2026-08-28: "read the docs, then use it" was granted 2048 tokens by
    its own clock and generated with 399, because vitality had scaled it down.
    The argument stopped inside ``from ledgerkit imp``.
    """

    # Resource admission has already bounded the request. Affective state
    # cannot reduce an owned answer's capacity after that decision.
    bounded = _bounded_max_tokens(
        requested, requested if preserve_admitted_capacity else bridged, fallback
    )
    if hard_output_ceiling is not None and hard_output_ceiling != "":
        bounded = _bounded_max_tokens(bounded, hard_output_ceiling, fallback)

    contract_floor = _requested_output_contract_generation_floor(requested_output_contract)
    surface_floor = 0
    if preserve_user_surface_completion_floor:
        try:
            surface_floor = max(0, int(user_surface_completion_floor or 0))
        except (TypeError, ValueError, OverflowError):
            surface_floor = 0
    tool_floor = 0
    try:
        tool_floor = max(0, int(tool_call_floor or 0))
    except (TypeError, ValueError, OverflowError):
        tool_floor = 0
    completion_floor = max(contract_floor, surface_floor, tool_floor)
    if completion_floor <= 0:
        return bounded

    try:
        caller_cap = max(1, int(requested))
    except (TypeError, ValueError, OverflowError):
        caller_cap = max(1, int(fallback))
    admitted_cap = caller_cap
    if hard_output_ceiling is not None and hard_output_ceiling != "":
        try:
            admitted_cap = min(admitted_cap, max(1, int(hard_output_ceiling)))
        except (TypeError, ValueError, OverflowError):
            pass
    admitted = max(bounded, min(completion_floor, admitted_cap))
    if admitted > bounded and contract_floor > bounded:
        # This is a successful policy decision. The generation receipt carries
        # every input and ``completion_floor_applied``; the log makes the choice
        # visible to operators without teaching resilience that Aura was hurt.
        logger.info(
            "[MLX] Completion contract raised generation cap %d->%d above "
            "adaptive shrinkage.",
            bounded,
            admitted,
            extra={
                "generation_budget_decision": "completion_floor_over_adaptive_cap",
                "adaptive_generation_cap": bounded,
                "admitted_generation_cap": admitted,
                "contract_generation_floor": contract_floor,
            },
        )
    return admitted


def _requested_output_contract_generation_floor(contract: Any) -> int:
    """Return a conservative native-generation floor for a typed user contract."""

    if not isinstance(contract, dict) or not contract:
        return 0
    if bool(contract.get("exact_reply", False)):
        try:
            utf8_bytes = max(1, int(contract.get("exact_reply_utf8_bytes") or 0))
        except (TypeError, ValueError, OverflowError):
            utf8_bytes = 0
        if utf8_bytes > 0:
            # Any supported tokenizer needs no more content tokens than UTF-8
            # bytes, plus one slot for EOS/stop termination.
            #
            # Capped: the contract is a plain dictionary from the caller, and
            # an exact_reply_utf8_bytes of ten million would demand a floor
            # this lane cannot serve and would override every pressure budget
            # trying to reach it.
            return min(_MAX_OUTPUT_CONTRACT_FLOOR_TOKENS, utf8_bytes + 1)
    try:
        return min(
            _MAX_OUTPUT_CONTRACT_FLOOR_TOKENS,
            max(0, int(contract.get("semantic_token_cap") or 0)),
        )
    except (TypeError, ValueError, OverflowError):
        return 0


#: Worker replies that END a request and must reach the caller waiting on it.
#:
#: LIVE, 2026-08-20. encode_hidden was added to the worker and to the client
#: and timed out every time, because a response is only handed to its future
#: when its action appears here — a third place a new action has to be
#: registered, with a silent eight-second wait as the symptom. Kept as one
#: named list so the registration is visible rather than buried in a tuple
#: inside a loop.
_TERMINAL_WORKER_ACTIONS: frozenset[str] = frozenset(
    {
        "encode_hidden",
        "encode_hidden_sequence",
        "generate",
        "generate_batch",
        "latent_reason",
        "nonparametric_ingest",
        "set_expert_adapter",
        "stream_done",
        "unified_recurrent_qualified_decode",
        "unified_recurrent_shadow_probe",
    }
)

#: The smallest generation that can still contain a tool call.
#:
#: A native call is a name, an arguments object and its delimiters. Below this
#: the model cannot finish one, and a half-emitted call is indistinguishable
#: downstream from a model that chose not to call anything.
#: How much already-read evidence a tool prompt may carry.
_TOOL_LOOP_EVIDENCE_CHARS = 6000

_TOOL_CALL_TOKEN_FLOOR = 320

#: Argument names that hold a whole document rather than a phrase. Read from
#: the tool's own advertised schema, so a tool added later is measured by what
#: it declares instead of by a list kept somewhere else.
_DOCUMENT_ARGUMENT_NAMES = frozenset(
    {"code", "content", "text", "body", "source", "script", "html", "patch", "program"}
)


def _tool_call_budget(requested: Any, configured: Any, tools: Any) -> int:
    """How many tokens one tool call may take.

    The reply's budget unless the call can carry a document, in which case the
    client's configured ceiling — which is this runtime's own answer to how
    large a single generation may be.
    """
    try:
        asked = max(1, int(requested or 0))
    except (TypeError, ValueError):
        asked = 1
    carries = _tools_can_carry_a_document(tools)
    try:
        ceiling = max(1, int(configured or 0))
    except (TypeError, ValueError):
        ceiling = asked
    granted = max(asked, ceiling) if carries else asked
    # Said out loud, because a call cut off mid-argument and a model that
    # declined to call look identical from everywhere downstream, and the
    # number that decides between them was never written anywhere.
    logger.info(
        "🔧 Tool-call budget: asked=%d configured=%d carries_a_document=%s → %d",
        asked,
        ceiling,
        carries,
        granted,
    )
    return granted


def _offered_for_budgeting(options: Mapping[str, Any]) -> Any:
    """The tools this generation may call, whichever protocol it speaks.

    ``tools`` carries the definitions only when the model's NATIVE tool
    template is in use. A model whose template has no tool support, or one
    whose native attempt came back empty, is served the same call through a
    JSON contract in the prompt — and that path set ``tools`` to None, so every
    protection written for tool calls stopped applying to it.

    The protections are about what the turn is doing. A call expressed as JSON
    in the text needs MORE room than a native one, not less, because the whole
    envelope is generated rather than templated.

    LIVE, 2026-08-28: "read the docs, then use it" was cut off inside
    ``from ledgerkit imp`` at a 399-token pressure cap, on a turn that had been
    granted 2048 and whose tools included one taking a program as an argument.
    """

    return options.get("tools") or options.get("tool_budget_definitions")


def _tools_can_carry_a_document(tools: Any) -> bool:
    """Whether any offered tool takes an argument the size of a file."""
    if not tools:
        return False
    definitions = tools.values() if isinstance(tools, Mapping) else tools
    for definition in definitions:
        if not isinstance(definition, Mapping):
            continue
        schema = definition.get("parameters")
        if isinstance(definition.get("function"), Mapping):
            schema = definition["function"].get("parameters", schema)
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        if not isinstance(properties, Mapping):
            continue
        for name, spec in properties.items():
            if str(name).strip().lower() not in _DOCUMENT_ARGUMENT_NAMES:
                continue
            declared = spec.get("type") if isinstance(spec, Mapping) else None
            if declared in (None, "string") or declared == ["string"]:
                return True
    return False


def _apply_memory_pressure_generation_controls(
    options: dict[str, Any],
    snapshot: Any,
    *,
    default_max_tokens: int = 1,
) -> dict[str, Any]:
    """Reduce admitted generation work under unified-memory pressure."""

    max_token_cap = getattr(snapshot, "max_token_cap", None)
    if max_token_cap is None:
        return options

    # A turn that must EMIT A PLAN cannot be shrunk below the plan.
    #
    # Clamping a conversational reply under pressure is right: it costs some
    # words. Clamping an execution turn below its plan budget costs the whole
    # task — she cannot express the steps, so nothing runs, and the surface
    # reports conversational filler instead of doing the work. Being slower is
    # recoverable; being unable to attempt the request is not.
    #
    # This is why the desktop demo degrades exactly when a screen recorder is
    # running: the recorder raises unified-memory pressure, the cap drops, and
    # the plan no longer fits in the budget it was given.
    try:
        requested_max_tokens = max(
            1,
            int(options.get("max_tokens", default_max_tokens)),
        )
    except (TypeError, ValueError, OverflowError):
        requested_max_tokens = max(1, int(default_max_tokens or 1))
    clean_user_surface = bool(options.get("clean_user_surface_contract", False))
    try:
        completion_floor = max(
            0,
            int(options.get("user_surface_completion_floor", 0) or 0),
        )
    except (TypeError, ValueError, OverflowError):
        completion_floor = 0
    completion_floor = min(requested_max_tokens, completion_floor)
    try:
        pressure_cap = max(1, int(max_token_cap))
    except (TypeError, ValueError, OverflowError):
        pressure_cap = 1

    if bool(options.get("desktop_execution_contract", False)):
        plan_floor = int(options.get("desktop_plan_token_floor", 1024) or 1024)
        effective_cap = max(pressure_cap, plan_floor)
    elif bool(options.get("document_output_contract", False)):
        # A reply that must CARRY a document is not conversational prose, and
        # the paragraph above applies to it word for word. Live 2026-08-20 an
        # HTML page written into a reply stopped mid-attribute at
        # `<script type=`, because a 4096-token default had been scaled to 970
        # — a fair size for prose and half a page.
        effective_cap = max(pressure_cap, requested_max_tokens)
    elif _offered_for_budgeting(options):
        # A tool call is an execution turn, and the paragraph above applies to
        # it word for word: clamped below the size of a call, she cannot
        # express the call, so nothing runs.
        #
        # LIVE, 2026-08-19. Every tool-using turn came back "Generation
        # produced 1 token(s) but no text survived to the caller". The prompt
        # was correct and ended in an open assistant turn; the budget was one
        # token, because a tool generation matched none of the protected
        # contracts and took the raw pressure cap. Days of "she will not use
        # her tools" was a token budget.
        # How big a call can be depends on what it carries.
        #
        # LIVE, 2026-08-20. Asked to build a single-file web app, the model
        # emitted exactly the right call — code_repl with the program as its
        # argument — and the generation stopped mid-string at the 384-token
        # pressure cap. An incomplete JSON object is not a call, so the loop
        # reported "none called", and the turn ended by claiming a file had
        # been saved to Downloads that was never written.
        #
        # A call cut in half is not cheaper: it costs the whole turn and then
        # the retry. Where an offered tool takes a document as an argument,
        # the call gets the budget the caller asked for.
        effective_cap = max(
            pressure_cap,
            int(options.get("tool_call_token_floor", 0) or _TOOL_CALL_TOKEN_FLOOR),
        )
        if _tools_can_carry_a_document(_offered_for_budgeting(options)):
            effective_cap = max(effective_cap, requested_max_tokens)
    elif clean_user_surface and completion_floor > 0 and pressure_cap >= 192:
        # The resident model's normal RSS places the process-ratio probe in its
        # `high` band even when the host still has ample available memory. The
        # old global cap therefore turned a route-approved 1,536-token answer
        # into 192 tokens and the UI received a sentence severed at "from the".
        # Preserve a question-shaped reserve at warning/high pressure. True
        # critical/emergency caps (64/32) remain hard.
        effective_cap = max(pressure_cap, completion_floor)
    else:
        effective_cap = pressure_cap

    options["memory_pressure_token_cap"] = pressure_cap
    options["user_surface_completion_floor"] = completion_floor
    options["completion_floor_applied"] = bool(effective_cap > pressure_cap)

    options["max_tokens"] = _bounded_max_tokens(
        options.get("max_tokens"),
        effective_cap,
        default_max_tokens,
    )
    if (
        clean_user_surface
        or "clean_user_surface_recurrent_loops" in options
    ):
        # CP126 0989c717: this overwrote the recurrent depth to 1 whenever a
        # token cap applied, so memory pressure silently disabled the
        # recurrence-native path — and the receipt still described the depth
        # as caller-requested and worker-attested, with nothing anywhere
        # saying it had been taken away. Reducing depth under pressure is a
        # defensible decision; presenting it as the requested depth is not.
        try:
            requested_loops = int(options.get("clean_user_surface_recurrent_loops") or 1)
        except (TypeError, ValueError):
            requested_loops = 1
        # A small adaptive trim is not evidence that recurrent execution no
        # longer fits. Live, a 2048 -> 1941 token adjustment silently changed a
        # requested two-loop turn to one loop, invalidating its worker receipt
        # while saving negligible memory. Only a genuinely emergency-sized
        # output budget may shed recurrence; otherwise preserve the selected
        # execution architecture and let ordinary admission own the decision.
        emergency_output_budget = int(options["max_tokens"]) <= 256
        reduce_recurrence = bool(
            requested_loops > 1
            and (
                emergency_output_budget
                or int(options["max_tokens"]) < completion_floor
            )
        )
        if reduce_recurrence:
            options["clean_user_surface_recurrent_loops"] = 1
            options["recurrent_loops_requested"] = requested_loops
            options["recurrent_loops_reduced_by_pressure"] = True
            _record_mlx_degradation(
                RuntimeError(
                    f"recurrent depth {requested_loops} reduced to 1 under a token cap"
                ),
                action="ran a single-pass surface turn and marked the depth as pressure-reduced",
                severity="warning",
            )
    return options


def _carry_decode_rate_across(receipt: dict[str, Any]) -> None:
    """Record the worker's measured decode rate in this process."""

    verified = receipt.get("worker_verified")
    if not isinstance(verified, dict):
        return
    try:
        rate = float(verified.get("decode_tokens_per_second") or 0.0)
    except (TypeError, ValueError):
        return
    if not (rate > 0.0):
        return
    try:
        from core.brain.llm.thinking_reserve import record_decode_rate

        record_decode_rate(generated_tokens=int(rate * 10), elapsed_s=10.0)
    except (ImportError, TypeError, ValueError):
        return


def _sanitize_surface_control_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "enabled",
        "live_mind_controls_bound",
        "clean_user_surface_contract",
        "decode_tokens_per_second",
        "surface_validation_prompt_present",
        "strict_answer_contract",
        "strict_value_contract",
        "proof_evaluation_contract",
        "operator_evidence_contract",
        "health_probe",
        "runtime_fact_status_contract",
        "grounded_runtime_status_contract",
        "surface_alpha_requested",
        "surface_alpha_applied",
        "surface_alpha_applied_ok",
        "recurrent_runtime_loops_requested",
        "recurrent_depth_present",
        "recurrent_runtime_loops_applied",
        "recurrent_runtime_loops_applied_ok",
        "surface_quality_gate_enabled",
        "surface_quality_gate_passed",
        "surface_quality_gate_attempts",
        "surface_quality_gate_reasons",
        "surface_quality_rejected_reasons",
        "surface_quality_gate_error",
        "generation_max_tokens",
        "memory_pressure_token_cap",
        "user_surface_completion_floor",
        "completion_floor_applied",
        "generation_stop_reason",
        "generation_configured_stop_sequence",
        "semantic_completion_contract",
        "semantic_completion_satisfied",
        "semantic_completion_incomplete",
        "continuation_resume_requested",
        "continuation_resume_applied",
        "continuation_resume_available",
        "continuation_resume_failure_reason",
        "conversation_resume_requested",
        "conversation_resume_applied",
        "conversation_resume_available",
        "conversation_resume_failure_reason",
        "caller_requested_max_tokens",
        "adaptive_suggested_max_tokens",
        "output_contract_generation_floor",
        "generated_tokens",
        "semantic_output_token_cap",
        "hard_output_token_ceiling",
        "instruction_shape_repair_applied",
        "deterministic_repair_applied",
        "text_mutation_count",
        "authorship_replacement_applied",
        "authorship_augmentation_applied",
        "model_replacement_applied",
        "exact_reply_token_count",
        "exact_reply_required_termination_headroom",
        "exact_reply_available_termination_headroom",
        "exact_reply_content_capacity_sufficient",
        "exact_reply_termination_headroom_sufficient",
        "exact_reply_token_ceiling_valid",
        "exact_reply_native_capacity_sufficient",
        "applied",
    }
    receipt = {key: value[key] for key in allowed if key in value}
    resume_handle = str(value.get("continuation_resume_handle") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", resume_handle):
        receipt["continuation_resume_handle"] = resume_handle
    conversation_resume_handle = str(
        value.get("conversation_resume_handle") or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", conversation_resume_handle):
        receipt["conversation_resume_handle"] = conversation_resume_handle
    conversation_resume_output_sha256 = str(
        value.get("conversation_resume_output_sha256") or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", conversation_resume_output_sha256):
        receipt[
            "conversation_resume_output_sha256"
        ] = conversation_resume_output_sha256
    contract = value.get("requested_output_contract")
    if isinstance(contract, dict):
        contract_allowed = {
            "kind",
            "word_min",
            "word_max",
            "sentence_count",
            "explicit_brevity",
            "exact_reply",
            "exact_reply_chars",
            "exact_reply_utf8_bytes",
            "semantic_token_cap",
            "hard_token_ceiling",
            "confidence",
        }
        receipt["requested_output_contract"] = {
            key: contract[key] for key in contract_allowed if key in contract
        }
    mutations = value.get("text_mutations")
    if isinstance(mutations, list):
        from core.brain.live_mind_contract import (
            normalize_text_mutations,
            summarize_text_mutation_authorship,
        )

        receipt["text_mutations"] = normalize_text_mutations(mutations)
        receipt["text_mutation_count"] = len(receipt["text_mutations"])
        receipt.update(summarize_text_mutation_authorship(receipt["text_mutations"]))
    rejected_text = value.get("surface_quality_rejected_text")
    if isinstance(rejected_text, str) and rejected_text.strip():
        # This draft is internal recovery evidence, not a user-facing result.
        # Bound it independently even though the worker already does so; IPC
        # input is never trusted merely because another Aura process made it.
        receipt["surface_quality_rejected_text"] = rejected_text[:8_000]
        rejected_reasons = value.get("surface_quality_rejected_reasons")
        if isinstance(rejected_reasons, (list, tuple)):
            receipt["surface_quality_rejected_reasons"] = [
                str(reason).strip()[:120]
                for reason in rejected_reasons
                if str(reason).strip()
            ][:8]
    return receipt


def _surface_quality_rejection_reasons(value: Any) -> tuple[str, ...]:
    """Identify an intentional worker quality rejection, not an empty decode."""

    receipt = value if isinstance(value, dict) else {}
    if not bool(receipt.get("surface_quality_gate_enabled")):
        return ()
    if bool(receipt.get("surface_quality_gate_passed")):
        return ()
    raw_reasons = receipt.get("surface_quality_gate_reasons")
    if not isinstance(raw_reasons, (list, tuple)):
        return ()
    return tuple(str(reason).strip()[:120] for reason in raw_reasons if str(reason).strip())[:8]


def _surface_quality_rejected_draft_reasons(value: Any) -> tuple[str, ...]:
    """Reasons attached to the exact retained draft, not the latest retry."""

    receipt = value if isinstance(value, dict) else {}
    raw_reasons = receipt.get("surface_quality_rejected_reasons")
    if not isinstance(raw_reasons, (list, tuple)):
        return _surface_quality_rejection_reasons(receipt)
    return tuple(
        str(reason).strip()[:120]
        for reason in raw_reasons
        if str(reason).strip()
    )[:8]


def _bounded_surface_grounding_evidence(value: Any = None) -> list[str]:
    """Carry exact-turn recall evidence across worker IPC without prompt trust."""

    surface_grounding = value
    if surface_grounding is None:
        try:
            from core.conversation.turn_evidence_custody import turn_grounding_evidence

            surface_grounding = turn_grounding_evidence()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            surface_grounding = ()
    bounded: list[str] = []
    total_chars = 0
    if not isinstance(surface_grounding, (list, tuple)):
        return bounded
    for item in surface_grounding[:16]:
        evidence_text = str(item or "").strip()[:3_200]
        if not evidence_text or evidence_text in bounded:
            continue
        if total_chars + len(evidence_text) > 24_000:
            break
        bounded.append(evidence_text)
        total_chars += len(evidence_text)
    return bounded


def _bounded_surface_sensory_evidence(value: Any = None) -> dict[str, Any]:
    """Carry one typed exact-turn sensor receipt across worker IPC."""

    try:
        from core.conversation.turn_evidence_custody import turn_sensory_evidence

        available = turn_sensory_evidence()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        available = ()
    try:
        from core.senses.turn_evidence import TurnSensoryEvidence

        parsed_requested = TurnSensoryEvidence.from_value(value) if value is not None else None
        admitted = [
            (item, parsed)
            for item in available
            if (parsed := TurnSensoryEvidence.from_value(item)) is not None
        ]
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        parsed_requested = None
        admitted = []
    if not admitted:
        return {}
    if parsed_requested is None:
        candidate, parsed = admitted[-1]
    else:
        matched = [
            pair for pair in admitted if pair[1].to_dict() == parsed_requested.to_dict()
        ]
        if not matched:
            return {}
        candidate, parsed = matched[-1]
    bounded = parsed.to_dict()
    if isinstance(candidate, dict):
        for key in ("session_id", "turn_id"):
            identity = str(candidate.get(key) or "").strip()[:160]
            if identity:
                bounded[key] = identity
    return bounded


def _bounded_surface_tool_receipts() -> list[dict[str, Any]]:
    """Carry exact-turn execution evidence across worker IPC.

    The worker cannot inherit the parent's ContextVar custody.  It receives a
    bounded projection of receipts owned by the current turn instead; callers
    cannot supply arbitrary receipts through generation kwargs.
    """

    try:
        from core.conversation.surface_disposition import turn_tool_receipts

        available = turn_tool_receipts()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return []
    bounded: list[dict[str, Any]] = []
    total_chars = 0
    for raw in available[:16]:
        if not isinstance(raw, dict):
            continue
        item = {
            "receipt_id": str(raw.get("receipt_id") or "").strip()[:64],
            "tool": str(raw.get("tool") or "").strip()[:128],
            "action": str(raw.get("action") or "").strip()[:128],
            "object_ref": str(raw.get("object_ref") or "").strip()[:500],
            "ok": bool(raw.get("ok")),
            "effect_observed": bool(raw.get("effect_observed")),
            "verification": str(raw.get("verification") or "").strip()[:160],
        }
        if not item["receipt_id"] or not item["tool"]:
            continue
        item_chars = sum(len(str(value)) for value in item.values())
        if total_chars + item_chars > 8_000:
            break
        bounded.append(item)
        total_chars += item_chars
    return bounded


def _rejected_surface_draft(value: Any) -> str:
    """The draft the worker's quality gate rejected, if it carried one."""
    receipt = value if isinstance(value, dict) else {}
    text = receipt.get("surface_quality_rejected_text")
    return str(text).strip() if isinstance(text, str) else ""


def _coerce_timeout_seconds(value: Any) -> float | None:
    """Normalize public timeout kwargs into positive request deadlines.

    None means "caller supplied no timeout" (defaults apply downstream).
    A MALFORMED value must not silently erase the caller's intent to be
    bounded — it becomes a conservative bounded default instead.
    """
    if value is None or isinstance(value, Deadline):
        return None
    try:
        timeout_s = float(value)
    except (TypeError, ValueError, OverflowError):
        _record_mlx_degradation(
            ValueError(f"malformed generation timeout: {value!r}"),
            action="replaced malformed timeout with a bounded 120s deadline",
        )
        return 120.0
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        _record_mlx_degradation(
            ValueError(f"non-finite or non-positive generation timeout: {timeout_s!r}"),
            action="replaced invalid timeout with a bounded 120s deadline",
        )
        return 120.0
    return max(0.1, timeout_s)


def _spawn_gate_snapshot() -> dict[str, Any]:
    with _GLOBAL_SPAWN_GATE_STATE_LOCK:
        owner = _GLOBAL_SPAWN_GATE_OWNER
        acquired_at = _GLOBAL_SPAWN_GATE_ACQUIRED_AT
        token = _GLOBAL_SPAWN_GATE_TOKEN
    return {
        "held": bool(token),
        "owner": owner,
        "acquired_at_monotonic": acquired_at,
        "age_s": (max(0.0, time.monotonic() - acquired_at) if token and acquired_at > 0.0 else 0.0),
    }


@contextlib.asynccontextmanager
async def _spawn_gate_context(
    *, owner: str = "unknown", timeout_s: float | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Cancellation-safe, bounded ownership of the global spawn gate.

    A blocking ``Semaphore.acquire`` delegated with ``asyncio.to_thread`` is
    not cancellation-safe: cancelling the coroutine does not stop its thread.
    That abandoned thread can later acquire the semaphore with no surviving
    context manager to release it. Foreground recovery's 15-second deadline
    exercised exactly that path and leaked the gate for every later warmup.

    Nonblocking acquisition on the event-loop thread is constant-time. Bounded
    polling preserves cross-loop/thread compatibility while guaranteeing a
    cancelled waiter can never acquire after its caller is gone.

    ``timeout_s`` bounds the wait by the CALLER'S remaining budget. Waiting
    past your own deadline is guaranteed-useless — and worse than useless
    here, because the escalation ladder is serial: a 32B load that waits the
    full 330 s process bound for a gate it will never get also spends the
    turn's whole budget, so the Brainstem inherits seconds and the Reflex
    inherits milliseconds, and NO tier answers (2026-07-18 soak: p50 167 s,
    32 turns with no reply, while a 1.5 B fallback sat ready). A tier that
    cannot even START loading inside its own budget must fail fast so the
    next rung still has time to serve the user.
    """

    bound = float(_SPAWN_GATE_ACQUIRE_TIMEOUT_S)
    if timeout_s is not None:
        bound = max(0.0, min(bound, float(timeout_s)))
    deadline = time.monotonic() + bound
    acquired = False
    lease_token = uuid.uuid4().hex
    while not acquired:
        acquired = _GLOBAL_SPAWN_GATE.acquire(blocking=False)
        if acquired:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            holder = _spawn_gate_snapshot()
            raise TimeoutError(
                f"spawn_gate_timeout:{bound:.3f}s:"
                f"holder={holder['owner'] or 'unknown'}:age={holder['age_s']:.3f}s"
            )
        await asyncio.sleep(min(0.05, remaining))

    with _GLOBAL_SPAWN_GATE_STATE_LOCK:
        global _GLOBAL_SPAWN_GATE_TOKEN
        global _GLOBAL_SPAWN_GATE_OWNER
        global _GLOBAL_SPAWN_GATE_ACQUIRED_AT
        _GLOBAL_SPAWN_GATE_TOKEN = lease_token
        _GLOBAL_SPAWN_GATE_OWNER = str(owner or "unknown")[:160]
        _GLOBAL_SPAWN_GATE_ACQUIRED_AT = time.monotonic()
    try:
        yield _spawn_gate_snapshot()
    finally:
        with _GLOBAL_SPAWN_GATE_STATE_LOCK:
            if _GLOBAL_SPAWN_GATE_TOKEN == lease_token:
                _GLOBAL_SPAWN_GATE_TOKEN = ""
                _GLOBAL_SPAWN_GATE_OWNER = ""
                _GLOBAL_SPAWN_GATE_ACQUIRED_AT = 0.0
            else:
                logger.critical(
                    "Spawn gate ownership metadata changed before release owner=%s",
                    owner,
                )
        _GLOBAL_SPAWN_GATE.release()


def _foreground_owner_active() -> bool:
    return _FOREGROUND_OWNER_NAME is not None


def _origin_tokens(origin: str | None) -> set[str]:
    normalized = _normalized_origin(origin)
    return {token for token in normalized.split("_") if token}


def _normalized_origin(origin: str | None) -> str:
    return str(origin or "").strip().lower().replace("-", "_")


def _origin_is_user_facing(origin: str | None) -> bool:
    """Whether this request is on a path a person is waiting on.

    CP126 829ffbee. This used to split the origin on underscores and intersect
    the tokens with the allowlist, so ANY label containing one of them elevated
    itself: ``background_user`` matched ``user``, ``test_background_sweep``
    matched ``test``, ``api_prefetch`` matched ``api``. Foreground priority
    decides who gets the model when it is contended, so a background sweep
    that can name itself into the foreground is a self-granted privilege.

    The whole normalized label must be in the allowlist. A new user-facing
    origin is added deliberately, which is the point.
    """
    return _normalized_origin(origin) in _USER_FACING_ORIGINS


def _background_deferral_active(origin: str | None = None) -> str | None:
    """Mirror InferenceGate's background quiet policy inside the MLX client.

    The gate can reject newly scheduled background requests, but an already
    running background request may reach this client after the foreground lane
    has been reserved.  Checking here prevents that stale request from
    re-spawning a worker Aura just unloaded to protect a user turn.
    """
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
    except (ImportError, RuntimeError) as exc:
        # No container, no gate to consult, and nothing to defer from — this is
        # a build without the gate rather than a gate that failed.
        _record_mlx_degradation(
            exc,
            action="continued without optional background deferral policy",
        )
        return None
    if gate is None:
        return None
    reader = getattr(gate, "background_local_deferral_reason", None) or getattr(
        gate, "_background_local_deferral_reason", None
    )
    if reader is None:
        # CP126 aa66b0ac: a gate that exists but cannot be asked used to mean
        # "no deferral", so a stale background request respawned a worker that
        # had just been unloaded to protect a user turn. The gate's presence is
        # the signal that this runtime HAS a background policy; not being able
        # to read it is a reason to defer, not to proceed.
        _record_mlx_degradation(
            AttributeError("inference gate exposes no background deferral reader"),
            action="deferred background model work while the gate policy was unreadable",
            severity="error",
        )
        return "background_deferral_policy_unreadable"
    try:
        reason = reader(origin=origin)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="deferred background model work after the gate policy raised",
            severity="error",
        )
        return "background_deferral_policy_unavailable"
    return str(reason) if reason else None


def _foreground_owner_age(now: float | None = None) -> float:
    """How long the current owner has held ownership, on a monotonic clock.

    Falls back to the wall-clock acquisition stamp when the monotonic one is
    unset. Acquisition sets both, so that only happens for state written
    directly — and reporting an age of zero for an owner that plainly has one
    would be worse than a clock that can be stepped.
    """
    if _FOREGROUND_OWNER_ACQUIRED_MONOTONIC > 0.0:
        current_time = float(now if now is not None else time.monotonic())
        return max(0.0, current_time - _FOREGROUND_OWNER_ACQUIRED_MONOTONIC)
    if _FOREGROUND_OWNER_ACQUIRED_AT > 0.0:
        return max(0.0, time.time() - _FOREGROUND_OWNER_ACQUIRED_AT)
    return 0.0


def _foreground_owner_silence(now: float | None = None) -> float:
    """How long since the owner last reported progress.

    This, not acquisition age, is what distinguishes a slow turn from a wedged
    one. A 32B cold load that is loading heartbeats; a decode that has stopped
    does not (CP126 6595b0e1, 8f772011).
    """
    if _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC <= 0.0:
        return _foreground_owner_age(now)
    current_time = float(now if now is not None else time.monotonic())
    return max(0.0, current_time - _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC)


def _stamp_foreground_owner(acquired_at: float) -> None:
    """Keep the three ownership stamps coherent.

    They are three views of one fact — when this owner took the lane — so they
    are written together. Setting only one leaves an age that belongs to a
    previous owner, which is how a stale monotonic stamp made a genuinely old
    owner look freshly acquired.
    """
    global _FOREGROUND_OWNER_ACQUIRED_AT
    global _FOREGROUND_OWNER_ACQUIRED_MONOTONIC
    global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC

    wall = float(acquired_at)
    _FOREGROUND_OWNER_ACQUIRED_AT = wall
    # Project the wall-clock age onto the monotonic clock so a caller that
    # knows only "this owner started N seconds ago" gets a coherent set.
    age = max(0.0, time.time() - wall)
    _FOREGROUND_OWNER_ACQUIRED_MONOTONIC = time.monotonic() - age
    _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = _FOREGROUND_OWNER_ACQUIRED_MONOTONIC


def note_foreground_owner_progress() -> None:
    """Refresh the owner heartbeat. Cheap, lock-free, called from hot paths."""
    global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC
    if _FOREGROUND_OWNER_NAME is not None:
        _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = time.monotonic()


def _foreground_owner_wait_budget(
    deadline: Deadline | None,
    *,
    foreground_request: bool,
) -> float:
    # A foreground waiter must be able to outlast one full serialized
    # turn: the generation gate caps concurrency at 2 and healthy gated
    # turns measure 31-44s. The old 10s budget guaranteed a timeout
    # whenever the owner was mid-turn, collapsing proof-primary requests
    # into refused lower-lane fallbacks. Background requests still bail
    # fast rather than camp on the foreground worker.
    default = 60.0 if foreground_request else 8.0
    if not isinstance(deadline, Deadline):
        return default

    remaining = deadline.remaining
    if remaining is None:
        return default

    # CP126 dec24697: the 0.25s floor applied even when `remaining` was zero
    # or negative, so a caller with no time left still started a new wait —
    # and every such floor along the path stacked past the deadline it was
    # meant to respect. Clamped by what is left: it may shorten the wait,
    # never extend it past the caller's budget.
    reserve = 3.0 if foreground_request else 1.5
    return max(0.0, min(default, remaining - _capped_reserve(reserve, remaining)))


def _clear_matching_foreground_owner(*candidate_names: str) -> str | None:
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT
    global _FOREGROUND_OWNER_ACQUIRED_MONOTONIC
    global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC

    candidates = {str(name or "").strip() for name in candidate_names if str(name or "").strip()}
    if not candidates:
        return None

    with _FOREGROUND_OWNER_LOCK:
        holder = _FOREGROUND_OWNER_NAME
        if holder not in candidates:
            return None
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        _FOREGROUND_OWNER_ACQUIRED_MONOTONIC = 0.0
        _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = 0.0
        return holder


def _clear_stale_foreground_owner(max_age_s: float = 200.0) -> str | None:
    """Release leaked foreground ownership after the generation has ended.

    [STABILITY v59] Raised default from 45s → 200s.  The 32B cortex
    cold-load + Metal shader JIT takes 90-180s.  At 45s the warmup's
    foreground owner was being cleared mid-load by periodic
    ``get_lane_status()`` calls, which allowed background workers to
    respawn and compete for unified memory — creating the desktop
    'cortex warming forever' deadlock.
    """
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT
    global _FOREGROUND_OWNER_ACQUIRED_MONOTONIC
    global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC

    acquired = _FOREGROUND_OWNER_LOCK.acquire(False)
    if not acquired:
        # Status reads run on foreground HTTP/pump paths. They must never wait
        # behind a leaked owner-lock holder; recovery paths can still force-clear.
        return None
    try:
        holder = _FOREGROUND_OWNER_NAME
        if holder is None:
            return None
        age = _foreground_owner_age()
        if age <= max_age_s:
            return None
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        _FOREGROUND_OWNER_ACQUIRED_MONOTONIC = 0.0
        _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = 0.0
        return holder
    finally:
        _FOREGROUND_OWNER_LOCK.release()


def force_clear_foreground_owner(
    *,
    reason: str,
    min_age_s: float = 45.0,
    owner_prefix: str | None = None,
    require_silence: bool = True,
    min_silence_s: float = 30.0,
) -> dict[str, Any]:
    """Clear a leaked foreground owner from a higher-level recovery path.

    Normal foreground ownership deliberately uses conservative stale limits so
    a healthy 32B cold start is not interrupted.  This hook is only for paths
    that already proved the live turn is wedged, such as desktop HTTP timeout
    recovery or chat-lock preemption.
    """
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT
    global _FOREGROUND_OWNER_ACQUIRED_MONOTONIC
    global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC

    min_age = max(0.0, float(min_age_s))
    min_silence = max(0.0, float(min_silence_s))
    with _FOREGROUND_OWNER_LOCK:
        holder = _FOREGROUND_OWNER_NAME
        age = _foreground_owner_age()
        if holder is None:
            return {
                "cleared": False,
                "reason": reason,
                "holder": None,
                "age_s": 0.0,
                "detail": "no_foreground_owner",
            }
        if owner_prefix and not str(holder).startswith(str(owner_prefix)):
            return {
                "cleared": False,
                "reason": reason,
                "holder": holder,
                "age_s": round(age, 3),
                "detail": "owner_prefix_mismatch",
            }
        if age < min_age:
            return {
                "cleared": False,
                "reason": reason,
                "holder": holder,
                "age_s": round(age, 3),
                "detail": "owner_younger_than_min_age",
            }
        # CP126 8f772011: age alone does not mean wedged. A 32B cold load is
        # legitimately slow, and clearing it mid-load is what produced the
        # 'cortex warming forever' deadlock. What distinguishes a wedge is
        # SILENCE — the owner has stopped reporting progress. A caller may
        # still override when it has its own proof (a desktop HTTP timeout
        # already observed the turn fail), but the default requires evidence.
        silence = _foreground_owner_silence()
        if require_silence and silence < min_silence:
            return {
                "cleared": False,
                "reason": reason,
                "holder": holder,
                "age_s": round(age, 3),
                "silence_s": round(silence, 3),
                "detail": "owner_still_reporting_progress",
            }
        acquired_at = _FOREGROUND_OWNER_ACQUIRED_AT

    # Ask the wedged generation to stop BEFORE releasing ownership, not after.
    # Clearing first opened a window where a new foreground turn could acquire
    # ownership while the old decode still held the GPU — exactly the
    # contention the owner gate exists to prevent, and worst precisely when
    # recovery fires because the machine is already struggling.
    #
    # The cancel runs outside _FOREGROUND_OWNER_LOCK on purpose: it reaches
    # into every client and takes their locks, and holding the owner lock
    # across that is an ABBA deadlock waiting to happen.
    # Scoped to the wedged holder. Clearing ONE stuck foreground owner used to
    # cancel every active decode in the process, including background work on
    # other models that had nothing to do with the wedge.
    soft_cancel = soft_cancel_active_generations(
        reason=f"owner_cleared:{reason}", owner_label=holder
    )

    with _FOREGROUND_OWNER_LOCK:
        # Compare-and-clear. Releasing the lock above means the world may have
        # moved: the wedged owner could have finished and a legitimate new turn
        # taken ownership. Clearing unconditionally would evict that innocent
        # owner and hand its GPU to whoever raced in next.
        if _FOREGROUND_OWNER_NAME != holder or _FOREGROUND_OWNER_ACQUIRED_AT != acquired_at:
            current = _FOREGROUND_OWNER_NAME
            logger.info(
                "♻️ [MLX] Foreground owner changed during cancel (%s → %s); "
                "leaving the new owner alone.",
                holder,
                current,
            )
            return {
                "cleared": False,
                "reason": reason,
                "holder": holder,
                "age_s": round(age, 3),
                "detail": "owner_changed_during_cancel",
                "current_holder": current,
                "soft_cancel": soft_cancel,
            }
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        _FOREGROUND_OWNER_ACQUIRED_MONOTONIC = 0.0
        _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = 0.0

    logger.warning(
        "♻️ [MLX] Force-cleared foreground owner %s after %.1fs (%s).",
        holder,
        age,
        reason,
    )
    return {
        "cleared": True,
        "reason": reason,
        "holder": holder,
        "age_s": round(age, 3),
        "detail": "cleared",
        "soft_cancel": soft_cancel,
    }


def soft_cancel_active_generations(
    *, reason: str, owner_label: str | None = None
) -> list[dict[str, Any]]:
    """Request cooperative cancel on clients with an active generation.

    Returns the per-client receipts for the clients that accepted a cancel
    request; clients with nothing running are skipped.

    CP126 2538a912: this always swept EVERY client. One wedged foreground
    owner therefore cancelled a background lane's legitimate decode on a
    different model, for a reason that had nothing to do with it. Pass
    ``owner_label`` — the same label the request lane was acquired under — and
    only the clients actually serving that owner are asked to yield.

    The broadening is explicit and recorded: if no client claims the named
    owner, the sweep falls back to all of them, because a recovery path that
    cancels nothing is worse than one that cancels too much. What it must not
    do is take the wide action silently while a narrow one was available.
    """
    receipts: list[dict[str, Any]] = []
    snapshot = _clients_snapshot()
    wanted = str(owner_label or "").strip()
    targeted = [
        (path, client)
        for path, client in snapshot
        if wanted
        and str(getattr(client, "_request_lock_owner_label", "") or "").strip() == wanted
    ]
    if wanted and not targeted:
        logger.info(
            "🧹 [MLX] No client claims foreground owner %s; cancelling every active "
            "generation for %s.",
            wanted,
            reason,
        )
    selected = targeted or snapshot
    for _client_path, client in selected:
        try:
            receipt = client.soft_cancel_active_generation(reason)
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="skipped one client during cooperative cancel sweep",
                severity="warning",
            )
            continue
        if receipt.get("requested"):
            receipt["model"] = os.path.basename(getattr(client, "model_path", "") or "")
            receipt["targeted_owner"] = wanted if targeted else ""
            receipts.append(receipt)
    return receipts


def _foreground_owner_eviction_after() -> float | None:
    """How long the CURRENT holder may hold before it may be evicted.

    None when the holder declared no budget: an owner that never said how
    long it needs is not evictable on age alone, because any number we
    invented for it would be a guess used to cancel real work.
    """
    declared = _FOREGROUND_OWNER_STALE_AFTER
    if declared is None:
        return None
    return max(float(declared), _FOREGROUND_OWNER_MIN_EVICTION_S)


@contextlib.asynccontextmanager
async def _foreground_owner_context(
    owner_name: str,
    *,
    deadline: Deadline | None = None,
    foreground_request: bool = False,
    stale_after: float | None = None,
    a_person_is_waiting: bool | None = None,
):
    """Serialize foreground work so background model activity cannot compete with it.

    ``a_person_is_waiting`` is what protects a holder from being preempted, and
    it is not the same question as whether the work is foreground. A run that
    plays a game for forty minutes is foreground on every one of its moves and
    nobody is waiting on any of them — so marking it user-facing made the
    person who started it unable to interrupt it, and asking her anything
    while she worked got a routing failure and an apology.

    A person outranks her own errand. Defaults to the old meaning when the
    caller does not say, so nothing that has not been taught the difference
    loses its protection.
    """
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT
    global _FOREGROUND_OWNER_ACQUIRED_MONOTONIC
    global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC
    global _FOREGROUND_OWNER_STALE_AFTER, _FOREGROUND_OWNER_IS_USER_FACING

    wait_budget = _foreground_owner_wait_budget(
        deadline,
        foreground_request=foreground_request,
    )
    loop = asyncio.get_running_loop()
    wait_started = loop.time()
    last_log_at = 0.0
    owner_acquired = False

    while max(0.0, loop.time() - wait_started) <= wait_budget:
        acquired = _FOREGROUND_OWNER_LOCK.acquire(False)
        cleared_holder: str | None = None
        cleared_holder_age = 0.0
        preempted_background = False
        try:
            if acquired:
                holder = _FOREGROUND_OWNER_NAME
                holder_age = _foreground_owner_age()
                if holder is None:
                    _FOREGROUND_OWNER_NAME = owner_name
                    _stamp_foreground_owner(time.time())
                    _FOREGROUND_OWNER_STALE_AFTER = stale_after
                    _FOREGROUND_OWNER_IS_USER_FACING = bool(
                        foreground_request
                        if a_person_is_waiting is None
                        else a_person_is_waiting
                    )
                    owner_acquired = True
                    break

                # A person outranks background work, immediately.
                if (
                    foreground_request
                    and holder != owner_name
                    and not _FOREGROUND_OWNER_IS_USER_FACING
                ):
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                    _FOREGROUND_OWNER_STALE_AFTER = None
                    _FOREGROUND_OWNER_IS_USER_FACING = False
                    cleared_holder = holder
                    cleared_holder_age = holder_age
                    preempted_background = True
                # CP126 4cb6a1a0. Eviction used to compare the holder's age
                # against the NEWCOMER's stale_after, which is normalized from
                # a caller-selected timeout to as little as 5 seconds. A short
                # request could therefore declare a legitimately-working owner
                # stale by its own budget and steal foreground authority.
                #
                # An owner is stale only by ITS OWN declared contract, floored
                # so that no declared budget can make a live turn instantly
                # evictable. A holder that declared nothing is never evicted
                # on age alone.
                eviction_after = _foreground_owner_eviction_after()
                if (
                    eviction_after is not None
                    and holder != owner_name
                    and holder_age > eviction_after
                ):
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                    _FOREGROUND_OWNER_STALE_AFTER = None
                    cleared_holder = holder
                    cleared_holder_age = holder_age
        finally:
            if acquired:
                _FOREGROUND_OWNER_LOCK.release()
        if cleared_holder is not None and preempted_background:
            logger.info(
                "⚡ [MLX] A person is waiting: took the lane from background "
                "holder %s (held %.1fs) for %s.",
                cleared_holder,
                cleared_holder_age,
                owner_name,
            )
            preempted_background = False
            continue
        if cleared_holder is not None:
            logger.warning(
                "♻️ [MLX] Cleared stale foreground owner %s after %.1fs so %s can proceed.",
                cleared_holder,
                cleared_holder_age,
                owner_name,
            )
            continue

        now = loop.time()
        waited = max(0.0, now - wait_started)
        if waited >= wait_budget:
            holder = _FOREGROUND_OWNER_NAME or "foreground"
            holder_age = _foreground_owner_age()
            raise TimeoutError(
                f"Foreground owner wait timed out after {wait_budget:.1f}s "
                f"waiting on {holder} (held {holder_age:.1f}s)"
            )

        if waited >= 5.0 and (now - last_log_at) >= 5.0:
            holder = _FOREGROUND_OWNER_NAME or "foreground"
            holder_age = _foreground_owner_age()
            logger.info(
                "⏳ [MLX] Waiting for foreground owner %s to release (held %.1fs).",
                holder,
                holder_age,
            )
            last_log_at = now

        await asyncio.sleep(min(0.05, max(0.0, wait_budget - waited)))
    if not owner_acquired:
        holder = _FOREGROUND_OWNER_NAME or "foreground"
        holder_age = _foreground_owner_age()
        raise TimeoutError(
            f"Foreground owner wait timed out after {wait_budget:.1f}s "
            f"waiting on {holder} (held {holder_age:.1f}s)"
        )

    def _release_owner_slot() -> None:
        """Clear EVERY field of the slot, not just the name.

        The monotonic acquire/heartbeat stamps are what the stale-owner and
        silence heuristics read. Leaving them set behind a cleared name means
        the next holder inherits a predecessor's clock, and a fresh owner can
        be judged silent for time it never held.
        """
        global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT
        global _FOREGROUND_OWNER_STALE_AFTER, _FOREGROUND_OWNER_IS_USER_FACING
        global _FOREGROUND_OWNER_ACQUIRED_MONOTONIC
        global _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC

        if _FOREGROUND_OWNER_NAME != owner_name:
            return
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        _FOREGROUND_OWNER_STALE_AFTER = None
        _FOREGROUND_OWNER_IS_USER_FACING = False
        _FOREGROUND_OWNER_ACQUIRED_MONOTONIC = 0.0
        _FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = 0.0

    try:
        yield
    finally:
        acquired = await asyncio.to_thread(_FOREGROUND_OWNER_LOCK.acquire, True, 2.0)
        if acquired:
            try:
                _release_owner_slot()
            finally:
                _FOREGROUND_OWNER_LOCK.release()
        else:
            # Leaving our finished ownership registered blocks every later
            # foreground turn until a stale-clear heuristic happens to fire.
            # Self-clear WITHOUT the lock as a last resort: we only remove
            # our own entry, so the worst race (another waiter observing the
            # cleared slot a moment early) is strictly better than a leak.
            _release_owner_slot()
            _record_mlx_degradation(
                TimeoutError("foreground owner release lock timeout"),
                action="self-cleared finished foreground ownership without the owner lock",
                severity="error",
            )
            logger.warning(
                "⚠️ [MLX] Timed out releasing foreground owner lock for %s — self-cleared.",
                owner_name,
            )


def _bridge_asyncio_future_to_concurrent(future: asyncio.Future) -> cfutures.Future:
    """Relay an asyncio.Future into a thread-safe future for cross-loop awaiting."""
    proxy: cfutures.Future = cfutures.Future()

    def _relay(done_future: asyncio.Future) -> None:
        if proxy.done():
            return
        if done_future.cancelled():
            proxy.cancel()
            return
        # ANY failure type must be relayed: the old narrow tuple let e.g.
        # OSError/TimeoutError escape the callback, leaving cross-loop
        # waiters on the proxy unresolved forever.
        try:
            result = done_future.result()
        except BaseException as exc:  # noqa: BLE001 - relay every failure to the waiter
            try:
                proxy.set_exception(exc)
            except (cfutures.InvalidStateError, asyncio.InvalidStateError):
                pass
            return
        try:
            proxy.set_result(result)
        except (cfutures.InvalidStateError, asyncio.InvalidStateError):
            return

    if future.done():
        _relay(future)
        return proxy

    try:
        future_loop = future.get_loop()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        _relay(future)
        return proxy

    if future_loop.is_closed():
        _relay(future)
        return proxy

    future_loop.call_soon_threadsafe(future.add_done_callback, _relay)
    return proxy


def _wrap_shared_future_for_current_loop(future: SharedFuture) -> asyncio.Future:
    if isinstance(future, asyncio.Future):
        current_loop = asyncio.get_running_loop()
        if future.get_loop() is current_loop:
            return future
        return asyncio.wrap_future(_bridge_asyncio_future_to_concurrent(future))
    if isinstance(future, cfutures.Future):
        return asyncio.wrap_future(future)
    raise TypeError(f"Unsupported future type: {type(future)!r}")


async def _await_shared_future(future: SharedFuture, *, timeout_s: float | None = None) -> Any:
    wrapped = _wrap_shared_future_for_current_loop(future)
    protected = asyncio.shield(wrapped)
    if timeout_s is None:
        return await protected
    return await asyncio.wait_for(protected, timeout=timeout_s)


def _set_shared_future_result(future: SharedFuture | None, result: Any) -> bool:
    if future is None or future.done():
        return False

    if isinstance(future, cfutures.Future):
        try:
            future.set_result(result)
        except cfutures.InvalidStateError:
            # Another thread completed/cancelled it between the done() check
            # and here — response delivery must not break on that race.
            return False
        return True

    if not isinstance(future, asyncio.Future):
        return False

    try:
        future_loop = future.get_loop()
    except (RuntimeError, AttributeError):
        return False
    if future_loop.is_closed():
        return False

    def _setter() -> None:
        if not future.done():
            future.set_result(result)

    # is_closed() above is a check, not a hold: the loop can close between
    # that line and this one, and call_soon_threadsafe then raises into the
    # worker listener thread rather than merely failing to deliver.
    try:
        future_loop.call_soon_threadsafe(_setter)
    except RuntimeError:
        return False
    return True


def _cancel_shared_future(future: SharedFuture | None) -> None:
    if future is None or future.done():
        return

    if isinstance(future, cfutures.Future):
        future.cancel()
        return

    if not isinstance(future, asyncio.Future):
        return

    try:
        future_loop = future.get_loop()
    except (RuntimeError, AttributeError):
        return
    if future_loop.is_closed():
        return
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is future_loop:
        future.cancel()
        return

    def _canceller() -> None:
        if not future.done():
            future.cancel()

    try:
        future_loop.call_soon_threadsafe(_canceller)
    except RuntimeError:
        logger.debug("Shared future cancel skipped: target loop closed.")


def _cancel_task_threadsafe(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    try:
        task_loop = task.get_loop()
    except (RuntimeError, AttributeError):
        return
    if task_loop.is_closed():
        return

    def _canceller() -> None:
        if not task.done():
            task.cancel()

    try:
        task_loop.call_soon_threadsafe(_canceller)
    except RuntimeError:
        logger.debug("Task cancel skipped: target loop closed.")


def _notify_closed_loop_output(text: str) -> None:
    if not text or not str(text).strip():
        return
    try:
        from core.consciousness.closed_loop import notify_closed_loop_output

        notify_closed_loop_output(str(text))
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued after optional closed-loop output notification failed",
        )
        logger.debug("Closed-loop output notification failed: %s", exc)


def _mlx_runtime_probe_command() -> list[str]:
    # The probe must prove USABLE inference plumbing, not just importability:
    # allocate on the default device, run a small matmul, force evaluation,
    # and check the numeric result. Import-only probes passed on hosts whose
    # Metal device could not actually evaluate a tensor.
    return [
        sys.executable,
        "-c",
        (
            "import mlx.core as mx; import mlx_lm; "
            "a = mx.ones((8, 8)); s = (a @ a).sum(); mx.eval(s); "
            "assert abs(float(s) - 512.0) < 1e-3, float(s); "
            "print('mlx_runtime_ok')"
        ),
    ]


def _load_probe_cache_from_disk() -> tuple[bool | None, str, float]:
    try:
        payload = json.loads(_MLX_RUNTIME_PROBE_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None, "", 0.0
    if not isinstance(payload, dict):
        return None, "", 0.0

    # STRICT boolean: json.loads never produces the string "false" for a
    # well-formed writer, so any non-bool here is a malformed or tampered
    # cache — treat it as absent, not as bool("false") == True.
    ok = payload.get("ok")
    if not isinstance(ok, bool):
        return None, "", 0.0

    # A verdict is only about the environment it was measured in. An older
    # schema answered a different question; another interpreter or machine
    # measured a different installation. Either way this file describes
    # something that is not this process, and a stale "ok" would route real
    # user turns to a lane that cannot serve them.
    if payload.get("schema") != _MLX_RUNTIME_PROBE_SCHEMA:
        return None, "", 0.0
    identity = _probe_cache_identity()
    if {k: str(payload.get(k, "")) for k in identity} != identity:
        return None, "", 0.0

    detail = str(payload.get("detail", "") or "")
    try:
        checked_at = float(payload.get("checked_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None, "", 0.0
    now = time.time()
    # A future-dated timestamp would stay "fresh" until wall time caught up;
    # allow only small clock skew.
    if not math.isfinite(checked_at) or checked_at <= 0.0 or checked_at > now + 60.0:
        return None, "", 0.0
    return ok, detail, checked_at


def _store_probe_cache_to_disk(ok: bool, detail: str) -> None:
    try:
        atomic_write_text(
            _MLX_RUNTIME_PROBE_CACHE_PATH,
            json.dumps(
                {
                    "schema": _MLX_RUNTIME_PROBE_SCHEMA,
                    "ok": bool(ok),
                    "detail": str(detail or ""),
                    "checked_at": time.time(),
                    **_probe_cache_identity(),
                }
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
        # OSError included: a completed health probe must never raise out of
        # cache persistence (disk-full/permissions) and disrupt its caller.
        _record_mlx_degradation(
            exc,
            action="kept in-memory MLX runtime probe status after disk cache write failed",
        )
        logger.debug("Failed to persist MLX runtime probe cache: %s", exc)


def _normalize_probe_detail(stdout: str, stderr: str, returncode: int) -> str:
    combined = "\n".join(part for part in (stderr, stdout) if part).strip()
    if "NSRangeException" in combined and "objectAtIndex" in combined:
        return "metal_device_enumeration_crash"
    if "timed out" in combined.lower():
        return "probe_timeout"
    for line in combined.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:240]
    if returncode < 0:
        return f"signal_{abs(returncode)}"
    return f"exit_{returncode}"


# How long a real probe success may bridge a CURRENT enumeration crash, and
# how many consecutive spawns it may cover. A crash that outlives this is not
# a driver glitch; it is a broken runtime, and spawning onto it wastes a
# 20-40GB load and strands the lane anyway.
_LKG_PROBE_WINDOW_S = _env_duration_s("AURA_MLX_LKG_PROBE_WINDOW_S", 300.0)
#: How long the out-of-process MLX availability probe may take. It starts an
#: interpreter and imports MLX, so it is sensitive to host I/O rather than to
#: anything about the runtime's health; on a thrashing page cache the import
#: alone can outlast a fixed budget and report the runtime "unavailable".
_MLX_RUNTIME_PROBE_TIMEOUT_S = _finite_env_float(
    "AURA_MLX_RUNTIME_PROBE_TIMEOUT_S", 25.0, minimum=5.0
)
_LKG_PROBE_MAX_CONSECUTIVE = 2


def _probe_mlx_runtime(force: bool = False) -> tuple[bool, str]:
    force = force or os.getenv("AURA_FORCE_MLX_RUNTIME_PROBE", "0") == "1"
    now = time.time()
    with _MLX_RUNTIME_PROBE_LOCK:
        cached_ok = _MLX_RUNTIME_PROBE.get("ok")
        cached_at = float(_MLX_RUNTIME_PROBE.get("checked_at", 0.0) or 0.0)
        cached_detail = str(_MLX_RUNTIME_PROBE.get("detail", "") or "")
        if (
            not force
            and cached_ok is not None
            and (now - cached_at) < _probe_cache_ttl_seconds(cached_ok, disk=False)
        ):
            return bool(cached_ok), cached_detail
        if not force:
            disk_ok, disk_detail, disk_checked_at = _load_probe_cache_from_disk()
            if disk_ok is not None and (now - disk_checked_at) < _probe_cache_ttl_seconds(
                disk_ok, disk=True
            ):
                _MLX_RUNTIME_PROBE.update(
                    {
                        "ok": disk_ok,
                        "detail": disk_detail,
                        "checked_at": disk_checked_at,
                    }
                )
                return bool(disk_ok), disk_detail

    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["AURA_MLX_RUNTIME_PROBE"] = "1"

    # [STABILITY v57] One-shot retry for probe on failure (except timeout)
    for probe_attempt in range(2):
        ok = False
        detail = "probe_not_run"
        try:
            completed = get_subprocess_gateway().run(
                _mlx_runtime_probe_command(),
                cwd=project_root,
                env=env,
                capture_output=True,
                # The probe spawns a fresh interpreter and imports MLX. On a
                # host whose page cache is thrashing, that import alone can
                # exceed a fixed budget — and the timeout was hardcoded with no
                # way for an operator to raise it. Live 2026-07-26, repeatedly:
                # `mlx_runtime_unavailable:exit_124`, on a machine where MLX was
                # perfectly healthy and merely slow to load.
                timeout=_MLX_RUNTIME_PROBE_TIMEOUT_S,
                read_only=True,
                source="runtime_probe:mlx_runtime_probe",
                accelerator_capability="auto",
            )
            ok = completed.returncode == 0 and "mlx_runtime_ok" in (completed.stdout or "")
            detail = _normalize_probe_detail(
                completed.stdout or "",
                completed.stderr or "",
                completed.returncode,
            )

            # If it's a known enumeration crash, we might want to retry immediately
            if not ok and detail == "metal_device_enumeration_crash" and probe_attempt == 0:
                logger.warning("⚠️ [MLX] Metal device enumeration crash during probe. Retrying...")
                time.sleep(1.0)
                continue

            # If it's okay or a different failure, break the retry loop
            break

        except subprocess.TimeoutExpired as exc:
            detail = _normalize_probe_detail(
                (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                124,
            )
            # Timeout is terminal for the attempt
            break
        except (subprocess.SubprocessError, OSError) as exc:
            _record_mlx_degradation(
                exc,
                action="marked MLX runtime probe as failed for this attempt",
                severity="error",
            )
            detail = f"probe_exception:{type(exc).__name__}"
            break

    # [STABILITY v57] Grace Fallback for a TRANSIENT enumeration crash.
    #
    # CP126 5f02bc9d. This used to accept any in-memory success younger than
    # 30 minutes, with no bound on how many times it could fire. The probe we
    # just ran said the selected runtime is broken RIGHT NOW — and the retry
    # above already gave it a second chance — so a half-hour-old success was
    # being allowed to certify a currently-failing Metal stack indefinitely,
    # spawn after spawn, with nothing louder than a log line.
    #
    # It stays a bridge over a driver glitch, but a bounded one: a short
    # window anchored to the last REAL success (LKG never refreshes the
    # cache, so it cannot extend itself), a cap on consecutive uses, and a
    # degradation record every time, because spawning a 20-40GB worker onto
    # an unconfirmed runtime is a risk someone should be able to see.
    if not ok and detail == "metal_device_enumeration_crash":
        with _MLX_RUNTIME_PROBE_LOCK:
            cached_ok = _MLX_RUNTIME_PROBE.get("ok")
            cached_at = float(_MLX_RUNTIME_PROBE.get("checked_at", 0.0) or 0.0)
            age_s = time.time() - cached_at
            lkg_uses = int(_MLX_RUNTIME_PROBE.get("lkg_uses", 0) or 0)
            within_window = bool(cached_ok) and age_s < _LKG_PROBE_WINDOW_S
            budget_left = lkg_uses < _LKG_PROBE_MAX_CONSECUTIVE
            if within_window and budget_left:
                _MLX_RUNTIME_PROBE["lkg_uses"] = lkg_uses + 1
        if within_window and budget_left:
            _record_mlx_degradation(
                RuntimeError(
                    f"metal_device_enumeration_crash bridged by last-known-good "
                    f"status from {age_s:.0f}s ago "
                    f"(use {lkg_uses + 1}/{_LKG_PROBE_MAX_CONSECUTIVE})"
                ),
                action="allowed worker spawn on an unconfirmed MLX runtime",
                severity="warning",
            )
            logger.warning(
                "♻️ [MLX] Runtime probe hit an enumeration crash; bridging with a "
                "last-known-good status from %.0fs ago (use %d/%d).",
                age_s,
                lkg_uses + 1,
                _LKG_PROBE_MAX_CONSECUTIVE,
            )
            return True, "lkg_fallback_after_enumeration_crash"
        if cached_ok:
            # The bridge is out: either the last real success aged out or the
            # crash has repeated past the point where "transient" is credible.
            _record_mlx_degradation(
                RuntimeError(
                    f"metal_device_enumeration_crash not bridged: "
                    f"lkg_age={age_s:.0f}s uses={lkg_uses}"
                ),
                action="refused worker spawn until a live runtime probe succeeds",
                severity="error",
            )

    with _MLX_RUNTIME_PROBE_LOCK:
        _MLX_RUNTIME_PROBE.update(
            {
                "ok": ok,
                "detail": detail,
                "checked_at": time.time(),
                # A probe that actually ran resets the bridge: consecutive
                # means consecutive, and a confirmed-good runtime has earned
                # its grace back.
                "lkg_uses": 0 if ok else int(_MLX_RUNTIME_PROBE.get("lkg_uses", 0) or 0),
            }
        )
    _store_probe_cache_to_disk(ok, detail)
    return ok, detail


async def _join_inflight_across_loops(inflight: Any) -> Any:
    """Await an in-flight warmup that may belong to a different event loop.

    Joining a singleflight only works if the join can actually wait. Awaiting a
    Future owned by another loop raises immediately, which turns "somebody else
    is already warming, wait for them" into "warmup failed".
    """

    shielded = asyncio.shield(inflight)
    try:
        owner_loop = inflight.get_loop()
        current_loop = asyncio.get_running_loop()
    except (AttributeError, RuntimeError):
        return await shielded
    if owner_loop is current_loop:
        return await shielded

    async def _await_owned() -> Any:
        return await shielded

    bridged = asyncio.run_coroutine_threadsafe(_await_owned(), owner_loop)
    return await asyncio.wrap_future(bridged)


def _apply_the_wire_action_intervention(
    *,
    base: Any,
    receipt: Any,
    wire_action_intervention: Any,
    wire_action_policy_evidence: Any,
    wire_external_execution_offer: Any,
) -> Any:
    """Apply the wire-action intervention this episode carried.

    Moved out of ``MLXLocalClient.latent_reason_async`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 5 name(s) and hands back
    0.
    """
    def _block() -> Any:
        if wire_action_intervention is not None:
            try:
                from core.brain.llm.latent_cortex.action_intervention import (
                    validate_action_intervention_receipt,
                )
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationKind,
                )
                from core.brain.llm.latent_cortex.value_of_computation import (
                    validate_action_trace,
                )

                policy_receipt = receipt.get("value_of_computation")
                action_trace = receipt.get("cognitive_action_trace")
                expected_policy_fields = {
                    "schema",
                    "bucket",
                    "snapshot_sha256",
                    "active",
                    "calibration_intervention",
                    "executors",
                    "actions_selected",
                    "checked_transitions",
                    "selected_actions",
                }
                if (
                    not isinstance(policy_receipt, dict)
                    or set(policy_receipt) != expected_policy_fields
                    or not isinstance(action_trace, list)
                ):
                    raise ValueError("worker intervention policy receipt is incomplete")
                executors = tuple(
                    OperationKind(item) for item in policy_receipt.get("executors", ())
                )
                if not executors or len(set(executors)) != len(executors):
                    raise ValueError("worker intervention executor inventory is invalid")
                validated_trace = validate_action_trace(
                    action_trace,
                    evidence_snapshot=wire_action_policy_evidence,
                    executors=executors,
                    action_intervention=wire_action_intervention,
                )
                validate_action_intervention_receipt(
                    policy_receipt["calibration_intervention"],
                    intervention=wire_action_intervention,
                    cognitive_action_trace=action_trace,
                )
                selected_actions = validated_trace["selected_actions"]
                if (
                    wire_external_execution_offer is not None
                    or OperationKind.EXECUTE.value in selected_actions
                ):
                    if wire_external_execution_offer is None:
                        raise ValueError(
                            "worker selected execute without an external execution offer"
                        )
                    from core.brain.llm.latent_cortex.external_execution import (
                        validate_external_execution_handoff,
                    )

                    validate_external_execution_handoff(
                        receipt.get("external_execution_handoff"),
                        offer=wire_external_execution_offer,
                        cognitive_action_trace=action_trace,
                    )
                elif receipt.get("external_execution_handoff"):
                    raise ValueError(
                        "worker emitted an unsolicited external execution handoff"
                    )
                checked_transitions = sum(
                    int(row["transition"]["checked"]) for row in validated_trace["rows"]
                )
                if (
                    policy_receipt.get("schema") != wire_action_policy_evidence["schema"]
                    or policy_receipt.get("bucket") != wire_action_policy_evidence["bucket"]
                    or policy_receipt.get("snapshot_sha256")
                    != wire_action_policy_evidence["snapshot_sha256"]
                    or policy_receipt.get("active") is not True
                    or policy_receipt.get("actions_selected") != len(action_trace)
                    or policy_receipt.get("selected_actions") != selected_actions
                    or policy_receipt.get("checked_transitions") != checked_transitions
                ):
                    raise ValueError("worker intervention policy summary differs")
            except (
                ImportError,
                KeyError,
                TypeError,
                ValueError,
            ):
                return {
                    **base,
                    "receipt": receipt,
                    "reason": "action_intervention_receipt_invalid",
                }
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response


def _build_the_generation_request(
    *,
    _bridge_get: Any,
    adaptive_suggested_max_tokens: Any,
    contract_generation_floor: Any,
    generation_max_tokens: Any,
    hard_output_token_ceiling: Any,
    kwargs: Any,
    prompt: Any,
    req_id: Any,
    requested_output_contract: Any,
    self: Any,
) -> Any:
    """Build the request the worker receives for this generation.

    Moved out of ``MLXLocalClient._generate_inner`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    1.
    """
    continuation_resume_handle = str(
        kwargs.get("user_surface_continuation_resume_handle") or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", continuation_resume_handle):
        continuation_resume_handle = ""
    conversation_resume_handle = str(
        kwargs.get("user_surface_conversation_resume_handle") or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", conversation_resume_handle):
        conversation_resume_handle = ""
    req = {
        "id": req_id,
        "seq": self._job_seq_counter,
        "action": "generate",
        "prompt": prompt,
        "messages": kwargs.get("messages"),
        "tools": kwargs.get("tools"),
        "cognitive_mode": str(kwargs.get("cognitive_mode") or "").strip().lower(),
        "serving_lane": str(
            kwargs.get("serving_lane") or "foreground_standard"
        ).strip().lower(),
        "temp": kwargs.get(
            "temp",
            kwargs.get("temperature", _bridge_get("temperature", self.temp)),
        ),
        "top_p": kwargs.get("top_p", _bridge_get("top_p", self.top_p)),
        "top_k": kwargs.get("top_k", _bridge_get("top_k", 60)),
        "min_p": kwargs.get("min_p", 0.05),
        "repetition_penalty": kwargs.get(
            "repetition_penalty", _bridge_get("repetition_penalty", 1.05)
        ),
        "repetition_context_size": kwargs.get("repetition_context_size", 30),
        "presence_penalty": kwargs.get(
            "presence_penalty", _bridge_get("presence_penalty", 0.0)
        ),
        # max_tokens is a cap: both the latent bridge and the typed visible
        # output contract may shrink it, but neither can expand the caller.
        "max_tokens": generation_max_tokens,
        "memory_pressure_token_cap": kwargs.get("memory_pressure_token_cap"),
        "user_surface_completion_floor": kwargs.get("user_surface_completion_floor"),
        "completion_floor_applied": bool(kwargs.get("completion_floor_applied", False)),
        "caller_requested_max_tokens": kwargs.get("max_tokens", self.max_tokens),
        "adaptive_suggested_max_tokens": adaptive_suggested_max_tokens,
        "output_contract_generation_floor": contract_generation_floor,
        "requested_output_contract": dict(requested_output_contract),
        "semantic_output_token_cap": kwargs.get("semantic_output_token_cap"),
        "hard_output_token_ceiling": hard_output_token_ceiling,
        "schema": kwargs.get("schema"),
        "stop_sequences": list(kwargs.get("stop_sequences") or []),
        "strict_answer_contract": bool(kwargs.get("strict_answer_contract", False)),
        "strict_value_contract": bool(kwargs.get("strict_value_contract", False)),
        "expected_strict_value": str(kwargs.get("expected_strict_value") or ""),
        "proof_evaluation_contract": bool(kwargs.get("proof_evaluation_contract", False)),
        "operator_evidence_contract": bool(kwargs.get("operator_evidence_contract", False)),
        "web_interlocutor_contract": bool(kwargs.get("web_interlocutor_contract", False)),
        "health_probe": bool(kwargs.get("health_probe", False)),
        "warmup_precompile": bool(kwargs.get("warmup_precompile", False)),
        "runtime_fact_status_contract": bool(kwargs.get("runtime_fact_status_contract", False)),
        "requires_memory_grounding": bool(kwargs.get("requires_memory_grounding", False)),
        "memory_state_contract": bool(kwargs.get("memory_state_contract", False)),
        "grounded_recall_contract": bool(kwargs.get("grounded_recall_contract", False)),
        "grounded_runtime_status_contract": bool(
            kwargs.get("grounded_runtime_status_contract", False)
        ),
        "self_condition_contract": bool(kwargs.get("self_condition_contract", False)),
        "clean_user_surface_contract": bool(
            kwargs.get("clean_user_surface_contract", False)
            or kwargs.get("health_probe", False)
        )
        and not bool(kwargs.get("web_interlocutor_contract", False)),
        "user_surface_validation_prompt": str(
            kwargs.get("user_surface_validation_prompt") or ""
        ),
        "user_surface_continuation_contract": bool(
            kwargs.get("user_surface_continuation_contract", False)
        ),
        "user_surface_continuation_partial": continuation_state_text(
            kwargs.get("user_surface_continuation_partial")
        ),
        "user_surface_continuation_resume_handle": continuation_resume_handle,
        "user_surface_conversation_resume_handle": conversation_resume_handle,
        "semantic_completion_contract": bool(
            kwargs.get("semantic_completion_contract", False)
        ),
        "user_surface_prompt_binding": (
            dict(kwargs.get("user_surface_prompt_binding") or {})
            if isinstance(kwargs.get("user_surface_prompt_binding"), dict)
            else {}
        ),
        "user_surface_grounding_evidence": _bounded_surface_grounding_evidence(
            kwargs.get("user_surface_grounding_evidence")
        ),
        "user_surface_sensory_evidence": _bounded_surface_sensory_evidence(
            kwargs.get("user_surface_sensory_evidence")
        ),
        "user_surface_tool_receipts": _bounded_surface_tool_receipts(),
        "clean_user_surface_steering_alpha": kwargs.get("clean_user_surface_steering_alpha"),
        "clean_user_surface_recurrent_loops": (
            kwargs.get("clean_user_surface_recurrent_loops")
            if kwargs.get("clean_user_surface_recurrent_loops") is not None
            else (1 if kwargs.get("health_probe", False) else None)
        ),
        "live_mind_controls_bound": bool(kwargs.get("live_mind_controls_bound", False)),
        "benchmark_request": bool(kwargs.get("benchmark_request", False)),
        "disable_prompt_cache": bool(kwargs.get("disable_prompt_cache", False)),
        "clear_prompt_cache": bool(kwargs.get("clear_prompt_cache", False)),
    }
    return req


from core.brain.llm.endogenous_client_hooks import (  # noqa: E402
    attach_endogenous_state,
    process_endogenous_terminal_response,
)


def _is_internal_inference(cognitive_context: Any) -> bool:
    """Whether this episode is her thinking to herself rather than for someone.

    The context is a LIST of typed slots — it is declared as one three hundred
    lines above — and this asked it for a key as though it were a mapping. So
    every foreground episode raised AttributeError before it began, the failure
    was contained as `client_error:AttributeError`, and that became an
    integrity refusal that stopped browser actions dead. One wrong shape, and
    the whole action lane was closed.

    Read from whichever shape arrives: a slot saying so, or a mapping saying
    so. Anything else means somebody is waiting, which is the safer answer of
    the two — it keeps her to a person's deadline rather than a background
    one.
    """
    if isinstance(cognitive_context, Mapping):
        return bool(cognitive_context.get("internal_inference", False))
    if isinstance(cognitive_context, (list, tuple)):
        for slot in cognitive_context:
            if isinstance(slot, Mapping) and slot.get("internal_inference"):
                return True
    return False


class MLXLocalClient:
    """
    Parent-process client for the isolated MLX worker.
    Manages the lifecycle, health, and communication with the ForkServer process.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "gpu",
        max_tokens: int = 4096,
        runtime_assignment: ModelRuntimeAssignment | Mapping[str, Any] | None = None,
    ):
        from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

        if runtime_assignment is None:
            from core.brain.llm.model_registry import get_model_runtime_assignment

            assignment = get_model_runtime_assignment(model_path)
            # Bound against what the registry RESOLVED, not against what was
            # asked for. The registry expands a bare model name into its
            # artifact path, so a caller that names a model rather than a path
            # was rejected by the very assignment it had just been handed —
            # "model_runtime_assignment_model_path_mismatch" out of the
            # constructor, with nothing wrong. When the caller supplies the
            # assignment (below) the check is doing real work and stays.
            bound_to = assignment.model_path
        elif isinstance(runtime_assignment, ModelRuntimeAssignment):
            assignment = runtime_assignment
            bound_to = model_path
        elif isinstance(runtime_assignment, Mapping):
            assignment = ModelRuntimeAssignment.from_dict(runtime_assignment)
            bound_to = model_path
        else:
            raise TypeError("mlx_client_runtime_assignment_invalid")
        assignment.assert_bound_to(model_path=bound_to, purpose="serve")
        self.runtime_assignment = assignment
        self.model_path = assignment.model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temp = 0.7
        self.top_p = 0.9

        # [LOOP-AGNOSTIC FIX] asyncio.Lock is bound to the creating event loop.
        # MLXLocalClient is a singleton created at boot but used from Uvicorn's
        # separate event loop, causing RuntimeError. threading.Lock is loop-agnostic.
        self._lock = _threading.Lock()
        self._request_lock = _threading.Lock()
        self._request_lock_state_lock = _threading.RLock()
        self._request_lock_owner_token = ""
        # Optional callers may stop waiting while their command is still
        # running in the worker. The listener owns these requests until their
        # correlated terminal frame arrives; publishing the lane as idle before
        # then would let a foreground turn queue behind invisible model work.
        self._detached_worker_requests: dict[str, tuple[SharedFuture, str]] = {}
        self._deferred_reboot_reason: str | None = None
        self._consecutive_spawn_failures: int = 0
        self._spawn_backoff_until: float = 0.0
        #: Why the current spawn backoff exists ("" when none is active). A
        #: runtime probe proves the MLX runtime imports; it proves nothing
        #: about an OOM, a corrupt checkpoint or a refused memory admission,
        #: so it may only clear the backoffs a runtime failure caused.
        self._spawn_backoff_cause: str = ""
        #: True when a durable-lane preemptibility release failed. The lane
        #: cannot be evicted while this is set, so it is state a reader needs
        #: rather than a log line that scrolled away.
        self._durable_lane_release_owed: bool = False
        #: Set when a forced abort killed the worker but could not take the
        #: lifecycle lock; the next lock owner reconciles the state.
        self._force_abort_reconcile_pending: str | None = None
        self._force_abort_lock_failures = 0
        #: What the last soft-cancel targeted, and the worker frame that
        #: answered it. An acknowledgement must name the same request and
        #: arrive after the cancel was written.
        self._soft_cancel_target: dict[str, Any] | None = None
        self._soft_cancel_ack: dict[str, Any] | None = None
        #: A validated artifact waiting for the active request to end. Held
        #: apart from _deferred_reboot_reason so a recovery verdict and a
        #: staged promotion cannot overwrite one another.
        self._pending_promotion: str | None = None
        # A generation this client cancelled ON PURPOSE — healthy worker, turn
        # budget spent — yields no text. That is a deferral, not a broken
        # endpoint, and the router must not open the Cortex circuit over it.
        # One-shot: consumed by the reader so it can never leak into a later
        # turn that failed for its own reasons.
        self._deliberate_no_text_reason: str | None = None
        self._expert_adapter_path: str | None = None
        self._process: mp.Process | None = None
        self._model_lane_owner_id = f"mlx:{os.getpid()}:{_real_model_path(model_path)}"
        self._model_lane_state_lock = _threading.RLock()
        # Partial-failure receipt for a durable owner we could not release.
        # None means the lane holds no stranded fence.
        self._lane_release_failure: dict[str, Any] | None = None
        self._model_lane_fencing_token = 0
        self._model_lane_terminal_receipt_id = ""
        self._mp_context = (
            mp.get_context("spawn")
            if os.uname().sysname == "Darwin"
            else mp.get_context("forkserver")
        )
        self._req_q: Any | None = None
        self._res_q: Any | None = None
        self._closed = False
        self._init_done = False

        # Concurrency Hardening
        self._listener_task: asyncio.Task | None = None
        # A response listener owns one immutable queue generation.  Recovery
        # replaces queues before it replaces a listener; allowing a cancelling
        # listener to follow ``self._res_q`` let it steal init/completion frames
        # from the next worker.  Keep both identities explicit so a second
        # consumer can never be installed on the same queue without proof that
        # the first one terminated.
        self._response_queue_generation = 0
        self._listener_queue_generation = -1
        self._listener_response_queue: Any | None = None
        self._retired_listener_tasks: set[asyncio.Task] = set()
        self._lane_renewal_task: asyncio.Task | None = None
        self._listener_stop_generation = -1
        self._last_heartbeat = 0.0
        # Once-per-episode reporting latches for worker self-reported health
        # evidence (heartbeat loop_stalled / ipc_broken frames).
        self._worker_loop_stall_reported = False
        self._worker_ipc_broken_reported = False
        self._last_progress_at = 0.0
        self._last_token_progress_at = 0.0
        # Per-spawn key authorizing privileged output-contract selection.
        # Empty until a worker is spawned; a client with no worker has
        # nothing to authorize.
        self._contract_key: bytes = b""
        # Parent-private signer for one MLX worker spawn. Only the public
        # challenge crosses the process boundary.
        self._worker_capture_launch_authority: Any = None
        # Parent-signed binding established as soon as the child IPC channel
        # comes up. Model loading may outlive the launch challenge; READY is
        # accepted only when it carries the same capture identity.
        self._worker_capture_origin_binding: dict[str, Any] = {}
        self._latent_progress_by_request: dict[str, dict[str, Any]] = {}
        # Explicit drop accounting for the latent progress channel: state that
        # was refused (uncorrelated id) and state that aged out (window
        # eviction) are different failures and are counted separately.
        self._latent_progress_dropped_unknown = 0
        self._latent_progress_evicted = 0
        self._latent_progress_drop_reported = False
        self._last_ready_at = 0.0
        #: Tokens this worker has produced since it was spawned. Zero means
        #: its weights may still be loading, which is not the same as wedged.
        self._tokens_since_spawn = 0
        self._last_generation_completed_at = 0.0
        #: Bumped on every worker reboot so a lane-owner id names the
        #: generation it belongs to (CP126 cdbb177d). Without it, a stale
        #: generation and the live one shared one identity and fencing could
        #: not tell them apart.
        self._worker_generation = 0
        self._last_user_facing_completed_at = 0.0
        self._last_visible_readiness_at = 0.0
        self._current_gen_future: SharedFuture | None = None
        self._init_future: SharedFuture | None = None
        self._pending_generations: dict[str, SharedFuture] = {}
        self._request_lock_owner_label = ""
        self._request_lock_acquired_at = 0.0
        self._lane_state = "cold"
        self._lane_error = ""
        self._lane_transition_at = time.time()
        self._lane_transition_monotonic_at = time.monotonic()
        self._active_generations = 0
        self._active_generation_started_at = 0.0
        #: True when an adapter swap timed out with its command still queued,
        #: so which adapter is resident is genuinely not known until identity
        #: is re-read. Surfaced in supervision status; cleared by a swap that
        #: completes and returns a worker identity.
        self._expert_adapter_state_unknown = False
        self._warmup_attempted = False
        self._warmup_in_flight = False
        # Singleflight handle + its OWN start timestamp (CP126 4d8a7d6b):
        # staleness must be measured against the warmup, not against
        # _lane_transition_at, which any other lane transition refreshes.
        self._warmup_inflight: asyncio.Future | None = None
        self._warmup_started_at: float = 0.0
        # Consecutive reboot lock-acquisition failures (CP126 ec341dfa).
        self._reboot_lock_failures = 0
        self._model_load_admission_state_lock = _threading.Lock()
        self._model_load_admission_backoff_until = 0.0
        self._model_load_admission_backoff_until_unix = 0.0
        self._model_load_admission_denial_reason = ""
        self._model_load_admission_denial_receipt_id = ""
        self._model_load_admission_denial_count = 0
        self._model_load_admission_suppressed_count = 0
        self._model_load_admission_denied_at = 0.0
        self._consecutive_empty: int = 0  # [STABILITY v53] Explicit init — was missing
        #: req_id → (reason, recorded_at). Bound to the request that was
        #: actually planned for cancellation. A client-wide credit counter
        #: let any unrelated cancellation spend a credit and be reported as
        #: expected, while the request that really was cancelled went
        #: unaccounted for.
        self._expected_cancels: dict[str, tuple[str, float]] = {}
        self._process_started_at = 0.0
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._current_request_id = ""
        self._current_turn_progress = None
        self._current_request_progress_baseline_at = 0.0
        self._current_prompt_chars = 0
        self._current_requested_max_tokens = 0
        self._current_request_prompt_chars = 0
        self._current_first_token_hard_ceiling_s = 0.0
        self._current_prefill_tokens_processed = 0
        self._current_prefill_tokens_total = 0
        self._foreground_generation_watchdog: _threading.Timer | None = None
        self._recurrent_depth_status: dict[str, Any] = {"active": False, "config": None}
        #: The worker's recurrent-adapter activation receipt, validated against
        #: its signed identity at handshake. Empty until a receipt is accepted:
        #: not knowing which adapter is live is not the same as knowing none is.
        self._recurrent_adapter_activation: dict[str, Any] = {}
        #: Accepted non-serving recurrent controller state. This is separate
        #: from model identity because shadow tissue is neither a model
        #: mutation nor response-serving authority.
        self._unified_recurrent_shadow_status: dict[str, Any] = {}
        self._unified_recurrent_shadow_probe_status: dict[str, Any] = {}
        self._unified_recurrent_shadow_canary_status: dict[str, Any] = {}
        self._unified_recurrent_qualified_activation_status: dict[str, Any] = {}
        self._worker_identity: dict[str, Any] = {}
        self._mycelial_root_refs: list[dict[str, str]] = []
        self._last_surface_control_receipt: dict[str, Any] = {}
        self._surface_control_receipt_context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"aura_mlx_surface_receipt_{id(self)}",
            default=None,
        )
        self._last_interoception: dict[str, Any] = {}
        self._clock_sample_wall = time.time()
        self._clock_sample_monotonic = time.monotonic()
        self._clock_sample_sleep_inclusive = _sleep_inclusive_monotonic()
        self._clock_shift_events = 0
        self._clock_shift_total_s = 0.0

        # The state repository's SharedMemoryTransport may be backed by mmap on
        # restricted/macOS paths. mmap handles are not picklable under the
        # Darwin spawn context, so workers get a small multiprocessing bridge
        # instead of the repository transport itself. The last slot is reserved
        # for steering liveness.
        self._substrate_mem = self._mp_context.Array("d", 16, lock=False)
        # Reverse channel: Grassmann state integers from the worker (where the
        # activations are) back to PhiCore (which lives here). Without it the
        # activation-grounded Φ complex can never fill — the steering hook's
        # in-process PhiCore lookup is always False on the far side of the fork.
        # See core/consciousness/phi_residual_channel.py.
        # Reverse channel: latent readouts from the worker's transformer
        # hooks back to the substrate, which lives here. This is the backward
        # arrow of the latent bridge; without it the readout hooks accumulate
        # deltas into a thread that drops them.
        # See core/consciousness/latent_readout_channel.py.
        try:
            from core.consciousness.latent_readout_channel import (
                create_channel as _create_latent_channel,
            )

            self._latent_readout_mem = _create_latent_channel(self._mp_context)
        except (ImportError, AttributeError, OSError, ValueError):
            self._latent_readout_mem = None
        self._latent_readout_seen: list[float] | None = None

        try:
            from core.consciousness.phi_residual_channel import create_channel

            self._phi_residual_mem = create_channel(self._mp_context)
        except (ImportError, OSError, ValueError):
            self._phi_residual_mem = None

        # Shared memory flag to track if affective steering successfully attached
        self._steering_active = self._mp_context.Value("b", False, lock=False)
        self._steering_liveness_observed = False

        # Cooperative preemption channel: the parent writes the ACTIVE job's
        # numeric sequence here to ask the worker to stop between tokens.
        # Cancel latency is one decode step and the model stays warm — unlike
        # force-abort, which kills the worker and pays a full model reload.
        self._cancel_seq = self._mp_context.Value("Q", 0, lock=False)
        self._job_seq_counter = 0
        self._current_request_seq = 0
        self._last_prompt_cache_bytes = 0
        self._register_as_sheddable_organ()

    def _register_as_sheddable_organ(self) -> None:
        """Announce this cache to the OOM ladder at construction.

        Boot-time discovery in core/runtime/foundations.py walks services that
        are ALREADY instantiated, and this client is created lazily when a model
        first loads — long after that sweep. So the ladder kept reporting
        "0 sheddable organs" and the verifier kept warning that its only
        response to memory pressure was a restart, even with a working
        shed_memory() right here. Registering from __init__ is order-independent
        and idempotent: register() replaces by name.
        """

        try:
            from core.runtime.oom_policy import register_organ

            register_organ(
                "mlx_prompt_cache",
                oom_score_adj=int(self.oom_score_adj),
                footprint=self.memory_footprint_bytes,
                shed=self.shed_memory,
                rationale=self.oom_rationale,
                recoverable=bool(self.oom_recoverable),
            )
        except _MLX_OPTIONAL_THROTTLE_ERRORS as exc:
            _record_mlx_degradation(
                exc,
                action="continued without registering the prompt cache on the OOM ladder",
                severity="warning",
            )

    def _is_primary_or_deep_lane(self) -> bool:
        """Whether this lane is one of the big resident models.

        CP126 24aaa654. This was decided purely by searching the path for
        "32b"/"72b"/"zenith"/"solver"/"cortex", and it gates the memory
        guards that stand between a 20-40GB allocation and jetsam. A renamed,
        aliased or nonstandard checkpoint therefore walked straight past
        them, while an unrelated path containing one of those tokens was
        treated as heavy.

        Measured artifact evidence (parameter count from the model's own
        config/index) is the authority. The name tokens are kept as a UNION,
        never a replacement: for a safety guard the fail-safe direction is to
        over-include, so a model that either measures heavy or is named heavy
        is treated as heavy.
        """
        return _model_is_heavy_lane(self.model_path)

    def _is_primary_lane(self) -> bool:
        """Whether this loaded client was immutably assigned to Cortex."""

        return self.runtime_assignment.role == "cortex"

    def _can_run_resident_background_health_probe(
        self,
        deferral_reason: str,
        *,
        health_probe: bool,
    ) -> bool:
        """Allow one bounded readiness probe without weakening spawn admission."""
        return bool(
            health_probe
            and deferral_reason == "foreground_headroom_reserved"
            and self._is_primary_lane()
            and self.is_alive()
        )

    def _is_deep_solver_lane(self) -> bool:
        return self.runtime_assignment.role == "solver"

    @staticmethod
    def _model_load_admission_backoff_seconds(reason: str, count: int) -> float:
        normalized = str(reason or "").lower()
        attempt = max(1, int(count))
        if (
            normalized.startswith("event_loop_lag_")
            or normalized == "event_loop_signal_unavailable"
        ):
            base_s, cap_s = 3.0, 30.0
        elif "memory_pressure" in normalized or "thermal_pressure" in normalized:
            base_s, cap_s = 15.0, 300.0
        elif normalized in {
            "runtime_shutdown_requested",
            "background_capability_suspended",
            "large_model_capability_suspended",
        }:
            base_s, cap_s = 30.0, 300.0
        else:
            base_s, cap_s = 10.0, 120.0
        return min(cap_s, base_s * (2 ** min(attempt - 1, 5)))

    def _model_load_admission_backoff_active(self) -> bool:
        with self._model_load_admission_state_lock:
            active = time.monotonic() < float(self._model_load_admission_backoff_until or 0.0)
            if active:
                self._model_load_admission_suppressed_count += 1
            return active

    def _note_model_load_admission_denial(
        self,
        reason: str,
        *,
        receipt_id: str,
    ) -> float:
        with self._model_load_admission_state_lock:
            normalized = str(reason or "resource_admission_denied")
            if normalized == self._model_load_admission_denial_reason:
                self._model_load_admission_denial_count += 1
            else:
                self._model_load_admission_denial_reason = normalized
                self._model_load_admission_denial_count = 1
            backoff_s = self._model_load_admission_backoff_seconds(
                normalized,
                self._model_load_admission_denial_count,
            )
            now_monotonic = time.monotonic()
            now_unix = time.time()
            self._model_load_admission_backoff_until = now_monotonic + backoff_s
            self._model_load_admission_backoff_until_unix = now_unix + backoff_s
            self._model_load_admission_denial_receipt_id = str(receipt_id or "")
            self._model_load_admission_denied_at = now_unix
            self._model_load_admission_suppressed_count = 0
            return backoff_s

    def _clear_model_load_admission_backoff(self) -> None:
        with self._model_load_admission_state_lock:
            self._model_load_admission_backoff_until = 0.0
            self._model_load_admission_backoff_until_unix = 0.0
            self._model_load_admission_denial_reason = ""
            self._model_load_admission_denial_receipt_id = ""
            self._model_load_admission_denial_count = 0
            self._model_load_admission_suppressed_count = 0
            self._model_load_admission_denied_at = 0.0

    def _model_load_admission_status(self) -> dict[str, Any]:
        with self._model_load_admission_state_lock:
            return {
                "backing_off": time.monotonic()
                < float(self._model_load_admission_backoff_until or 0.0),
                "retry_at_unix": self._model_load_admission_backoff_until_unix,
                "reason": self._model_load_admission_denial_reason,
                "receipt_id": self._model_load_admission_denial_receipt_id,
                "denial_count": self._model_load_admission_denial_count,
                "suppressed_calls": self._model_load_admission_suppressed_count,
                "last_denied_at_unix": self._model_load_admission_denied_at,
            }

    def _adopt_durable_model_lane_owner(
        self,
        *,
        fencing_token: int,
        receipt_id: str,
    ) -> None:
        """Publish one committed owner token atomically to worker listeners."""

        token = int(fencing_token or 0)
        if token <= 0:
            raise ValueError("model-lane fencing token must be positive")
        with self._model_lane_state_lock:
            from core.runtime.model_lane_control import register_model_lane_owner_adapter

            register_model_lane_owner_adapter(
                self._model_lane_owner_id,
                evict=_evict_model_lane_owner,
                compensate=_compensate_model_lane_owner,
            )
            self._model_lane_fencing_token = token
            self._model_lane_terminal_receipt_id = str(receipt_id or "")

    def _durable_model_lane_owner_snapshot(self) -> tuple[str, int, str]:
        with self._model_lane_state_lock:
            return (
                str(self._model_lane_owner_id or ""),
                int(self._model_lane_fencing_token or 0),
                str(self._model_lane_terminal_receipt_id or ""),
            )

    def lane_recovery_required(self) -> dict[str, Any] | None:
        """The unreleased durable owner blocking this lane, if any.

        None means the lane holds no stranded fence. A dict is a partial-
        failure receipt: it names the owner and fencing token that must be
        released before this lane can be admitted again, so the dependency is
        actionable rather than an unexplained admission refusal later.
        """
        pending = getattr(self, "_lane_release_failure", None)
        return dict(pending) if pending else None

    def _note_lane_release_failure(self, exc: BaseException, *, reason: str) -> None:
        """Record a durable-owner release that could not be confirmed."""
        with self._model_lane_state_lock:
            owner_id = str(self._model_lane_owner_id or "")
            fencing_token = int(self._model_lane_fencing_token or 0)
        self._lane_release_failure = {
            "owner_id": owner_id,
            "fencing_token": fencing_token,
            "reason": reason,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "at_unix": time.time(),
        }
        _record_mlx_degradation(
            exc,
            action=(
                f"lane left FENCED: durable owner {owner_id or '<unknown>'} "
                f"token={fencing_token} could not be released during {reason}; "
                "admission stays blocked until it is"
            ),
            severity="critical",
        )
        self._record_degraded_event(
            "durable_owner_release_failed",
            detail=f"{os.path.basename(self.model_path)}:{owner_id}:token={fencing_token}",
            severity="critical",
            foreground_request=True,
        )
        self._set_lane_state("fenced", f"durable_owner_release_failed:{reason}")

    def _clear_lane_release_failure(self) -> None:
        """A confirmed release retires the fence."""
        self._lane_release_failure = None

    def _release_durable_model_lane_owner_sync(self, *, reason: str) -> bool:
        """Release the exact committed owner before another worker may spawn.

        The state lock spans the controller operation so a concurrent commit
        cannot publish a newer token while lifecycle cleanup is releasing the
        old one. A missing owner is already a settled release and still clears
        the local token; controller failures retain it so respawn can refuse
        rather than heartbeat a stale fence.
        """

        with self._model_lane_state_lock:
            owner_id = str(self._model_lane_owner_id or "")
            fencing_token = int(self._model_lane_fencing_token or 0)
            if not owner_id or fencing_token <= 0:
                self._model_lane_fencing_token = 0
                self._model_lane_terminal_receipt_id = ""
                return True

            from core.runtime.model_lane_control import (
                get_model_lane_controller,
                unregister_model_lane_owner_adapter,
            )

            released = get_model_lane_controller().release_owner_sync(
                owner_id,
                fencing_token=fencing_token,
                reason=str(reason or "worker_stopped"),
            )
            if not released:
                # CP126 158ed09e. The controller ALSO returns False for a
                # FENCING-TOKEN MISMATCH, i.e. a NEWER durable owner is
                # registered. Unregistering the adapter and discarding the
                # token + terminal receipt on that path threw away the only
                # handles able to reconcile that owner, while returning True
                # told every caller the lane was cleanly released.
                #
                # Keep the claim state and the fence recorded so respawn
                # refuses rather than heartbeating a stale fence, and report
                # the truth.
                self._note_lane_release_failure(
                    RuntimeError(
                        f"durable_owner_release_not_confirmed:{owner_id}:token={fencing_token}"
                    ),
                    reason=str(reason or "worker_stopped"),
                )
                return False

            unregister_model_lane_owner_adapter(owner_id)
            self._model_lane_fencing_token = 0
            self._model_lane_terminal_receipt_id = ""
            self._clear_lane_release_failure()
            return True

    async def _release_durable_model_lane_owner(self, *, reason: str) -> bool:
        return await asyncio.to_thread(
            self._release_durable_model_lane_owner_sync,
            reason=reason,
        )

    def _mark_progress(self) -> None:
        self._last_progress_at = time.time()
        # The foreground lease's staleness is measured from PROGRESS, not from
        # acquisition, so every client-level progress mark is also a heartbeat
        # for whoever currently owns the foreground (CP126 6595b0e1).
        note_foreground_owner_progress()

    def latent_progress_counters(self) -> dict[str, int]:
        """Drop accounting for the latent progress channel.

        Exposed so a refused stream is visible to health surfaces rather than
        only to whoever reads the logs: dropped_unknown counts progress for
        request ids this client never issued (a broken or hostile child),
        evicted counts entries aged out of the bounded window (normal churn).
        """
        return {
            "tracked": len(self._latent_progress_by_request),
            "dropped_unknown": self._latent_progress_dropped_unknown,
            "evicted": self._latent_progress_evicted,
        }

    def _authorize_job(self, job: Any, *, principal: str) -> Any:
        """Sign a job's privileged contract selection before submission.

        Single choke point: every path that puts work on the request queue
        goes through here, so a privileged contract cannot reach the worker
        without the authority of the lane that owns it. Jobs selecting
        nothing privileged are returned untouched.

        This is also where the causal trace crosses the process boundary. The
        machinery for that (core/runtime/causal_trace.inject_trace_carrier) had
        existed for a while with ZERO call sites, so a turn's trace stopped at
        the IPC edge and worker-side events could not be correlated back to the
        conversation that caused them. Injecting here rather than at each job
        construction site means every job type is covered by one wiring, and no
        future job path can forget.
        """
        if not isinstance(job, dict):
            return job
        job = self._inject_causal_trace(job)
        if not self._contract_key:
            return job
        try:
            from core.brain.llm.contract_authority import sign_job

            return sign_job(job, self._contract_key, principal=principal)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="could not sign a privileged contract selection",
                severity="error",
            )
            return job

    @staticmethod
    def _inject_causal_trace(job: dict[str, Any]) -> dict[str, Any]:
        """Attach the active trace context to an outbound job.

        Added AFTER any request digest is computed by callers and under keys no
        digest covers, so propagation cannot invalidate a signed contract. If no
        trace is active the job is returned untouched — this must never
        fabricate a correlation that does not exist.
        """
        try:
            from core.runtime.causal_trace import current_span, inject_trace_carrier

            if current_span() is None:
                return job
            return inject_trace_carrier(job)
        except Exception as exc:  # noqa: BLE001
            # Total on purpose. This is observability decorating a job that is
            # about to do real work; a tracing fault must never be able to take
            # down inference. The job proceeds untraced, which loses a
            # correlation but not the answer.
            logger.debug("Causal trace injection failed; continuing untraced: %s", exc)
            return job

    def _record_latent_progress(self, response: dict[str, Any]) -> None:
        """Retain bounded parent-side evidence for the active latent stage."""

        request_id = str(response.get("id") or "")
        if not request_id:
            return
        # Only track ids that belong to a PENDING or current request — a
        # broken or compromised child streaming unique ids must not grow
        # parent-side state without bound.
        if (
            request_id not in self._pending_generations
            and request_id != self._current_request_id
            and request_id not in self._latent_progress_by_request
        ):
            # Counted, not merely ignored: a child streaming ids the parent
            # never issued is either broken or compromised, and a silent drop
            # makes that indistinguishable from a healthy stream.
            self._latent_progress_dropped_unknown += 1
            if not self._latent_progress_drop_reported:
                self._latent_progress_drop_reported = True
                _record_mlx_degradation(
                    RuntimeError(
                        f"latent progress for unknown request id {request_id!r} "
                        f"from {os.path.basename(self.model_path)}"
                    ),
                    action="dropped uncorrelated latent progress from the worker",
                    severity="warning",
                )
            return
        allowed = {
            "stage",
            "stage_duration_s",
            "elapsed_s",
            "spent_layer_apps",
            "input_tokens",
            "branches",
            "slots",
            "max_branch_steps",
            "exchanges",
            "selected_branch",
            "steps_taken",
            "attempts",
            "accepted",
            "wrapped_layers",
            "generated_tokens",
            # RLC emits these coordinates from its bounded prefill and
            # decode stages. Keep the wire names so the parent receipt can
            # distinguish prompt work from answer tokens without guessing.
            "tokens_generated",
            "processed_tokens",
            "total_tokens",
            "chunk_tokens",
            "prefill_chunk_ceiling",
            "decode_generated_tokens",
            "decode_requested_tokens",
            "termination",
        }
        # The KEY whitelist bounds which fields survive; it says nothing about
        # their size or type. A worker sending stage="A"*50_000_000 was inside
        # the whitelist and retained in full, so the parent's own memory was a
        # function of what the child chose to send.
        snapshot = {
            key: _bounded_progress_value(response.get(key))
            for key in allowed
            if key in response
        }
        now = time.time()
        snapshot.update(
            {
                "request_id": request_id,
                "received_at_unix": now,
            }
        )
        self._latent_progress_by_request[request_id] = snapshot
        self._expire_latent_progress(now=now)
        # Bounded: evict the oldest entries beyond a small window.
        if len(self._latent_progress_by_request) > 64:
            for stale_id in sorted(
                self._latent_progress_by_request,
                key=lambda rid: float(
                    self._latent_progress_by_request[rid].get("received_at_unix", 0.0)
                ),
            )[: len(self._latent_progress_by_request) - 64]:
                self._latent_progress_by_request.pop(stale_id, None)
                self._latent_progress_evicted += 1

    #: Progress older than this describes a request nobody is waiting on.
    _LATENT_PROGRESS_TTL_S = 900.0

    def _expire_latent_progress(self, *, now: float | None = None) -> None:
        """Drop progress for requests that have finished or gone quiet.

        Capacity eviction alone kept a completed request's last snapshot
        resident until 64 newer ones displaced it, so a quiet client reported
        stage information for work that had ended long before.
        """
        moment = time.time() if now is None else now
        for rid, snapshot in list(self._latent_progress_by_request.items()):
            if rid == self._current_request_id or rid in self._pending_generations:
                continue
            received = float(snapshot.get("received_at_unix") or 0.0)
            if received and (moment - received) <= self._LATENT_PROGRESS_TTL_S:
                continue
            self._latent_progress_by_request.pop(rid, None)
            self._latent_progress_evicted += 1

    def _rebase_after_system_sleep(self) -> float:
        """Rebase active wall-clock anchors after host sleep/wake.

        macOS wall time advances while the monotonic clock used by asyncio and
        request deadlines pauses. Without rebasing, a healthy generation in
        flight at sleep is misclassified as a token or heartbeat stall on wake.
        """
        now_wall = time.time()
        now_monotonic = time.monotonic()
        now_sleep_inclusive = _sleep_inclusive_monotonic()
        wall_delta = max(0.0, now_wall - self._clock_sample_wall)
        monotonic_delta = max(0.0, now_monotonic - self._clock_sample_monotonic)
        prior_sleep_inclusive = self._clock_sample_sleep_inclusive
        self._clock_sample_wall = now_wall
        self._clock_sample_monotonic = now_monotonic
        self._clock_sample_sleep_inclusive = now_sleep_inclusive

        # Measured sleep, when the host can measure it. Both clocks here are
        # monotonic, so neither moves when someone sets the date, NTP steps
        # the clock, or a VM is migrated.
        slept = 0.0
        sleep_measured = False
        if now_sleep_inclusive is not None and prior_sleep_inclusive is not None:
            slept = max(0.0, (now_sleep_inclusive - prior_sleep_inclusive) - monotonic_delta)
            sleep_measured = True

        # Whatever the wall clock did beyond running time and measured sleep
        # is the clock itself moving. It still invalidates wall-clock anchors,
        # so it is still rebased — but it is NOT a host resume, and calling it
        # one hid genuine clock instability and, worse, let an unrelated jump
        # postpone every stall decision on a worker that had actually wedged.
        clock_shift = wall_delta - monotonic_delta - slept
        if not sleep_measured:
            # No sleep-inclusive clock on this host: the two are
            # indistinguishable, so attribute the gap to sleep as before
            # rather than reporting a clock anomaly we cannot substantiate.
            slept, clock_shift = clock_shift, 0.0

        gap = slept + clock_shift
        threshold = _env_duration_s("AURA_SYSTEM_SLEEP_GAP_THRESHOLD_S", 5.0, minimum=1.0)
        if gap <= max(1.0, threshold):
            return 0.0
        sleep_gap = gap
        if sleep_measured and clock_shift > max(1.0, threshold):
            self._clock_shift_events += 1
            self._clock_shift_total_s += clock_shift
            _record_mlx_degradation(
                RuntimeError(
                    f"wall clock moved {clock_shift:.1f}s without host sleep "
                    f"(slept {slept:.1f}s)"
                ),
                action="rebased wall-clock inference anchors after a clock adjustment",
                severity="warning",
            )

        stale_cutoff = now_wall - max(2.0, sleep_gap * 0.5)
        for attr in (
            "_current_request_started_at",
            "_current_request_progress_baseline_at",
            "_current_first_token_at",
            "_last_token_progress_at",
            "_last_heartbeat",
            "_last_progress_at",
            "_last_ready_at",
            "_lane_transition_at",
            "_request_lock_acquired_at",
        ):
            value = float(getattr(self, attr, 0.0) or 0.0)
            if 0.0 < value < stale_cutoff:
                setattr(self, attr, value + sleep_gap)
        if sleep_measured and clock_shift > max(1.0, threshold):
            logger.warning(
                "🕐 [MLX] Wall clock moved %.1fs without host sleep (slept %.1fs); "
                "rebased active inference clocks by %.1fs.",
                clock_shift,
                slept,
                sleep_gap,
            )
        else:
            logger.info(
                "🌙 [MLX] Host resume detected; rebased active inference clocks by %.1fs.",
                sleep_gap,
            )
        return sleep_gap

    def _surface_control_receipt_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_surface_control_receipt_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_mlx_surface_receipt_{id(self)}",
                default=None,
            )
            self._surface_control_receipt_context = slot
        return slot

    def _set_task_surface_control_receipt(self, receipt: dict[str, Any]) -> None:
        self._surface_control_receipt_slot().set(dict(receipt))

    def _prefill_progress_at(self) -> float:
        """When this request last advanced through a prompt, or 0.0.

        Only while a prefill is actually under way: a finished one must not
        keep a stalled generation alive on the strength of work it did
        minutes ago.
        """

        total = int(getattr(self, "_current_prefill_tokens_total", 0) or 0)
        done = int(getattr(self, "_current_prefill_tokens_processed", 0) or 0)
        if total <= 0 or done >= total:
            return 0.0
        return float(getattr(self, "_prefill_observed_at", 0.0) or 0.0)

    def tokens_generated_for_this_request(self) -> int:
        """Tokens this client has seen arrive for the request in flight.

        Counted where they arrive, so it does not depend on the worker
        attaching a total to its reply. LIVE 2026-08-29: "tool loop generation
        unrecorded: tokens=0 receipt_keys=none" — the generation that wrote the
        answer came back with neither a receipt nor a count, and authorship is
        proven from exactly that number, so the turn refused its own work.
        """

        return max(0, int(getattr(self, "_tokens_this_request", 0) or 0))

    def get_last_surface_control_receipt(self) -> dict[str, Any]:
        task_receipt = self._surface_control_receipt_slot().get()
        if task_receipt is not None:
            return dict(task_receipt)
        return {}

    def get_diagnostic_last_surface_control_receipt(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        return dict(getattr(self, "_last_surface_control_receipt", {}) or {})

    # ── OOM ladder rung: the worker's KV cache ────────────────────────────
    #
    # The boot verifier warned on EVERY boot that the ladder had no rungs —
    # "no organ exposes a shed hook, so the OOM ladder has no rungs: the only
    # available response to memory pressure is a restart" — while the prompt KV
    # cache in the worker was the largest trivially droppable allocation in the
    # whole process tree. It was unreachable because the ladder runs in the
    # parent and the cache lives in the worker; the worker has always accepted a
    # `clear_cache` action and nothing ever sent one.
    #
    # `core/runtime/foundations.py` auto-registers any live ServiceContainer
    # service exposing `shed_memory()`, so implementing that contract here is
    # all that is needed to arm the ladder.
    oom_score_adj: int = 300
    oom_rationale: str = (
        "prompt KV cache: pure reuse acceleration, costs only re-prefill to rebuild"
    )
    oom_recoverable: bool = True

    def memory_footprint_bytes(self) -> int:
        """Retained KV bytes, as last reported by the worker.

        Read from the value that rides along with every generate result rather
        than probed over IPC — the ladder may ask repeatedly and under memory
        pressure, which is the worst moment to add a round trip.
        """
        return max(0, int(getattr(self, "_last_prompt_cache_bytes", 0) or 0))

    def shed_memory(self) -> int:
        """Drop the worker's prompt cache; return bytes actually freed."""
        before = self.memory_footprint_bytes()
        freed = 0
        try:
            result = self._send_worker_control_action("clear_cache", timeout_s=10.0)
        except _MLX_CLIENT_RECOVERABLE_ERRORS as exc:
            _record_mlx_degradation(
                exc,
                action="left the prompt cache in place after a shed request failed",
                severity="warning",
            )
            return 0
        if isinstance(result, dict):
            try:
                freed = int(result.get("prompt_cache_bytes_freed") or 0)
            except (TypeError, ValueError):
                freed = 0
            self._last_prompt_cache_bytes = max(
                0, int(result.get("prompt_cache_bytes") or 0)
            )
        # Never claim more than was known to be held: an unverified reclaim
        # number is how a ladder reports progress it did not make.
        return max(0, min(freed, before) if before else freed)

    def _send_worker_control_action(
        self,
        action: str,
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, Any] | None:
        """Synchronous control message to the resident worker.

        Deliberately blocking: the OOM ladder is a synchronous shed loop, and a
        shed that returns before it has happened reports a reclaim that did not
        occur.
        """
        process = getattr(self, "_process", None)
        if not (process is not None and process.is_alive() and getattr(self, "_init_done", False)):
            return None
        request_queue = getattr(self, "_req_q", None)
        if request_queue is None:
            return None
        req_id = uuid.uuid4().hex
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        try:
            request_queue.put(
                self._authorize_job(
                    {"id": req_id, "action": action},
                    principal=f"mlx_client.{action}",
                ),
                True,
                2.0,
            )
            return fut.result(timeout=max(1.0, float(timeout_s)))
        except _MLX_CLIENT_RECOVERABLE_ERRORS:
            self._pending_generations.pop(req_id, None)
            raise
        finally:
            self._pending_generations.pop(req_id, None)

    def _record_throughput_sample(
        self,
        response: dict[str, Any],
        *,
        prompt: Any = None,
        foreground_request: bool = True,
    ) -> None:
        """Feed one real generation into the admission estimator.

        This is what makes admission MEASURED rather than a formula: the
        allocator's coefficients were hand-chosen with no model-specific
        calibration, so it could only ask "is this number allowable", never
        "can this finish on this machine as it is right now".

        Deliberately cheap and total. It runs on the completion path of
        every generation, so it does its own arithmetic, catches its own
        mistakes, and never lets a bad sample reach the caller — a
        telemetry write must not be able to fail a turn that worked.
        """
        try:
            started = float(getattr(self, "_current_request_started_at", 0.0) or 0.0)
            if started <= 0.0:
                return
            elapsed = max(0.0, time.time() - started)
            generated = max(0, int(response.get("tokens_used") or 0))
            if generated <= 0 or elapsed <= 0.0:
                return
            tokenization = response.get("prompt_tokenization")
            prompt_tokens = max(
                1,
                int(
                    (tokenization or {}).get("tokens")
                    if isinstance(tokenization, dict)
                    else 0
                )
                or len(str(prompt or "")) // 4,
            )
            # The worker reports prompt-cache retention, so a generation that
            # kept a cache is a WARM sample. Mixing warm and cold makes both
            # predictions wrong, which is why the shape carries it.
            cache_warm = bool(int(response.get("prompt_cache_reused_tokens") or 0) > 0)
            performance = response.get("generation_performance")
            exact_prefill = None
            exact_decode = None
            if isinstance(performance, dict):
                try:
                    measured_prefill = float(performance.get("prefill_seconds"))
                    measured_decode = float(performance.get("decode_seconds"))
                    if math.isfinite(measured_prefill) and measured_prefill >= 0.0:
                        exact_prefill = measured_prefill
                    if math.isfinite(measured_decode) and measured_decode > 0.0:
                        exact_decode = measured_decode
                except (TypeError, ValueError, OverflowError):
                    pass
            # Old workers do not report the split. Keep the bounded fallback
            # for rolling compatibility, but never overwrite MLX's measured
            # prompt and decode clocks with an estimate when they are present.
            prefill = (
                exact_prefill
                if exact_prefill is not None
                else min(elapsed * 0.25, prompt_tokens * 1.0e-3)
            )
            decode = (
                exact_decode
                if exact_decode is not None
                else max(1e-6, elapsed - prefill)
            )
            stop_reason = str(response.get("generation_stop_reason") or "").lower()
            surface_receipt = response.get("surface_control_receipt")
            semantic_complete = bool(
                isinstance(surface_receipt, dict)
                and surface_receipt.get("semantic_completion_satisfied") is True
            )
            completion_observed = semantic_complete or stop_reason in {
                "configured_stop",
                "eos",
                "semantic_contract_satisfied",
            }
            from .model_registry import runtime_model_measurement_key

            record_generation(
                model=runtime_model_measurement_key(self.model_path),
                prompt_tokens=prompt_tokens,
                generated_tokens=generated,
                prefill_seconds=prefill,
                decode_seconds=decode,
                cache_warm=cache_warm,
                foreground=bool(foreground_request),
                completion_observed=completion_observed,
            )
        except (ArithmeticError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="skipped one admission throughput sample",
                severity="debug",
            )

    def _record_surface_control_receipt_from_response(self, response: dict[str, Any]) -> None:
        if isinstance(response, dict) and "prompt_cache_bytes" in response:
            try:
                self._last_prompt_cache_bytes = max(
                    0, int(response.get("prompt_cache_bytes") or 0)
                )
            except (TypeError, ValueError, OverflowError):
                pass
        receipt = _sanitize_surface_control_receipt(
            response.get("surface_control_receipt") if isinstance(response, dict) else None
        )
        if isinstance(response, dict) and "tokens_used" in response:
            try:
                receipt["generated_tokens"] = max(0, int(response.get("tokens_used") or 0))
            except (TypeError, ValueError, OverflowError):
                pass
        if receipt:
            self._bind_surface_receipt_provenance(receipt, response)
        self._set_task_surface_control_receipt(receipt)
        if receipt:
            self._last_surface_control_receipt = receipt
            # The worker measures how fast it decodes; this process sizes the
            # deadline. Without carrying the number across, the deadline is
            # derived from the origin and the tier and can be shorter than the
            # budget the same request just computed.
            _carry_decode_rate_across(receipt)

    def _bind_surface_receipt_provenance(
        self, receipt: dict[str, Any], response: Any
    ) -> None:
        """Say who made these claims and about which request.

        CP126 92cbf5e2: the sanitizer limited which FIELDS survived and said
        nothing about where their values came from. "controls bound", "quality
        passed", "loops applied", "exact reply" all arrived as an unsigned
        worker dictionary, and a reader downstream could not tell a claim from
        a measurement, or tell WHICH worker generation and request produced it.

        There is no signature scheme between parent and worker, and inventing
        one here would be its own kind of false assurance. What can be done
        honestly is to stamp the identity the PARENT holds — the worker it
        attested at handshake, the generation counter it increments, the
        request id it issued — and to name the rest as worker-attested. A
        reader then knows exactly how much the receipt is worth.
        """
        try:
            identity = self.get_worker_identity_snapshot()
        except (AttributeError, TypeError, ValueError):
            identity = {}
        response_id = ""
        if isinstance(response, dict):
            response_id = str(response.get("id") or "")
        receipt["provenance"] = {
            "claims": "worker_attested",
            "model": os.path.basename(str(getattr(self, "model_path", "") or ""))
            or "unknown",
            "worker_boot_id": str(identity.get("worker_boot_id") or ""),
            "worker_pid": identity.get("worker_pid")
            if isinstance(identity.get("worker_pid"), int)
            else None,
            "worker_generation": int(getattr(self, "_worker_generation", 0) or 0),
            "request_id": response_id,
            "request_seq": int(getattr(self, "_current_request_seq", 0) or 0),
            # The parent issued this request id; a response carrying a
            # different one is describing somebody else's generation.
            "request_id_matches_active": bool(
                response_id and response_id == str(getattr(self, "_current_request_id", "") or "")
            ),
            "worker_identity_attested": bool(identity.get("worker_boot_id")),
        }

    def _record_suppressed_draft(
        self, text: str, reasons: tuple[str, ...] | list[str]
    ) -> None:
        """Put a gate-rejected draft on the bound turn, marked suppressed.

        Recoverable on purpose. These reasons are heuristics about SHAPE —
        "runtime_boilerplate", "too_thin_for_status_turn" — not findings of
        unsafety, and the recovery path only reaches for a suppressed draft
        when the turn would otherwise end with nothing. Between a draft a
        heuristic disliked and an apology that says nothing, the draft is the
        better answer, and the person can judge it themselves.

        Never raises: this is a salvage path, and a salvage path that can
        break the turn it is salvaging is worse than none.
        """
        if not text:
            return
        try:
            from core.conversation.surface_disposition import UNSPEAKABLE_REASONS
            from core.runtime.turn_outcome import current_turn

            outcome = current_turn()
            if outcome is None:
                return
            bounded_reasons = [str(reason).strip()[:120] for reason in reasons if str(reason).strip()][:8]
            recoverable = not bool(set(bounded_reasons) & set(UNSPEAKABLE_REASONS))
            candidate_id = outcome.record_candidate(
                text,
                source="mlx_worker.surface_quality_rejected",
                metadata={"worker_quality_reasons": bounded_reasons},
            )
            outcome.suppress_candidate(
                candidate_id,
                gate="mlx_worker.surface_quality",
                reasons=bounded_reasons,
                recoverable=recoverable,
            )
        except Exception:  # noqa: BLE001 — salvage must never break the turn
            logger.debug("could not record suppressed worker draft", exc_info=True)

    def _preserve_lane_after_surface_quality_rejection(self) -> None:
        """Clear empty-decode pressure while keeping the healthy worker resident."""

        self._consecutive_empty = 0
        self._set_lane_state("ready")

    def get_last_interoception(self) -> dict[str, Any]:
        """The substrate interoception payload of the most recent completed
        generation on this lane (worker-measured; see interoception_tap)."""
        return dict(self._last_interoception) if self._last_interoception else {}

    def _record_interoception_from_response(
        self,
        response: dict[str, Any],
        *,
        foreground_request: bool,
        owner_label: str,
    ) -> None:
        """Capture the worker's felt-thought measurements and hand them to the
        thought-interoception organ. Observational only — never raises into the
        generation path."""
        try:
            payload = response.get("interoception") if isinstance(response, dict) else None
            if not isinstance(payload, dict) or not payload:
                return
            from core.being.thought_interoception import (
                get_thought_interoception,
                text_fingerprint,
            )

            # Fingerprint the payload to the text it measured, so consumers
            # (e.g. unified_inference feedback) can prove they are pairing the
            # right trace with the right response even under concurrent lanes.
            # Ingest FIRST: the engine is the validator. Storing beforehand
            # exposed malformed/unbounded worker data through the public
            # getter as the "most recent measurement" even when the engine
            # rejected it.
            #
            # CP126 093a2902, second half: ordering alone did not fix that,
            # because ingest() NEVER RAISES — it returns None for a payload it
            # dropped. The store ran regardless, so a rejected payload was
            # still published as the latest felt-thought measurement. Only an
            # accepted one is retained now, and only in bounded form.
            # CP126 0b3bbd3e: the trace and the response arrived as two
            # unrelated arguments, so under concurrent lanes a worker's
            # measurement could be filed against a different lane's answer.
            # The response carries the request id the worker echoed; handing
            # it over lets the organ PROVE the pairing rather than assume it.
            felt = get_thought_interoception().ingest(
                payload,
                origin=owner_label or "mlx",
                foreground=bool(foreground_request),
                response_text=str(response.get("text") or ""),
                generation_id=str(response.get("id") or ""),
            )
            if felt is None:
                _record_mlx_degradation(
                    ValueError("interoception payload rejected by the felt-thought engine"),
                    action="left the previous measurement in place rather than publishing a rejected payload",
                    severity="warning",
                )
                return
            stored = _bounded_interoception(payload)
            stored["_text_fingerprint"] = text_fingerprint(str(response.get("text") or ""))
            self._last_interoception = stored
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "mlx_client_interoception",
                exc,
                severity="warning",
                action="continued generation return after interoception ingest failed",
            )

    #: How long a planned cancellation stays claimable. Past this the
    #: cancellation is no longer attributable to the plan that scheduled it.
    _EXPECTED_CANCEL_TTL_S = 30.0

    def _note_expected_generation_cancellation(
        self, reason: str, *, request_ids: Iterable[str]
    ) -> None:
        """Mark specific in-flight requests as deliberately cancelled.

        Named requests, not a count. The credit form could not tell a planned
        cancellation from an unrelated one that happened to arrive first, so a
        genuine wedge could be logged as routine reboot cleanup while the
        request the reboot actually killed was reported as a surprise.
        """
        now = time.time()
        label = str(reason or "planned_reboot")
        self._prune_expected_cancellations(now=now)
        for req_id in request_ids:
            key = str(req_id or "")
            if key:
                self._expected_cancels[key] = (label, now)

    def _prune_expected_cancellations(self, *, now: float | None = None) -> None:
        """Drop claims too old to attribute. Keeps the map bounded."""
        moment = time.time() if now is None else now
        stale = [
            key
            for key, (_reason, at) in self._expected_cancels.items()
            if at and (moment - at) > self._EXPECTED_CANCEL_TTL_S
        ]
        for key in stale:
            self._expected_cancels.pop(key, None)

    def _consume_expected_generation_cancellation(self, req_id: str = "") -> str:
        """The planned reason for THIS request's cancellation, or "".

        An empty return means this cancellation was not planned — which is the
        answer that has to be right, because it decides whether the runtime
        treats the event as cleanup or as a worker that stopped responding.
        """
        self._prune_expected_cancellations()
        key = str(req_id or "")
        if not key:
            return ""
        claim = self._expected_cancels.pop(key, None)
        return claim[0] if claim else ""

    def _mark_generation_started(
        self,
        req_id: str,
        *,
        prompt_chars: int = 0,
        requested_max_tokens: int = 0,
        first_token_hard_ceiling_s: float = 0.0,
        request_seq: int = 0,
    ) -> None:
        from core.runtime.turn_progress import capture_progress

        self._current_turn_progress = capture_progress()
        now = time.time()
        self._current_request_id = str(req_id or "")
        self._current_request_seq = max(0, int(request_seq or 0))
        # A new generation supersedes any stale cooperative-cancel request.
        cancel_seq = getattr(self, "_cancel_seq", None)
        if cancel_seq is not None and int(getattr(cancel_seq, "value", 0)) not in (
            0,
            self._current_request_seq,
        ):
            cancel_seq.value = 0
        self._current_request_progress_baseline_at = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
        )
        self._current_request_started_at = now
        self._current_first_token_at = 0.0
        #: Tokens observed for THIS request. The worker reports a count on the
        #: generations that carry a receipt and not on the ones that do not,
        #: and an answer's authorship is proven from that count.
        self._tokens_this_request = 0
        self._current_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_requested_max_tokens = max(0, int(requested_max_tokens or 0))
        self._last_token_progress_at = 0.0
        self._current_request_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_first_token_hard_ceiling_s = max(
            0.0,
            float(first_token_hard_ceiling_s or 0.0),
        )
        # A ceiling that does not account for the prompt is not a budget.
        #
        # LIVE 2026-08-26, two lines apart: "first-token ceiling 90.0s for a
        # 5-char prompt" and "first-token ceiling 4.0s for a 3431-char
        # prompt". Ninety seconds to read five characters, four to read nine
        # hundred tokens — the number had no relationship to the work, and
        # every decision she made while playing was cancelled by it. She chose
        # her moves from the consequence record alone and never held a plan,
        # which from outside looks exactly like a mind that is not thinking.
        #
        # The floor is the time THIS worker takes to read THIS prompt, at the
        # rate it has been measured at, with room for the queueing a shared
        # lane always has. It only ever raises a ceiling, never past the
        # livelock ceiling that catches a wedged worker.
        needed = min(
            self._prefill_floor_seconds(self._current_prompt_chars),
            # Never past the ceiling that catches a wedged worker. Said in
            # the comment before and not enforced in the code, which let one
            # bad rate reading ask for a ten-minute deadline.
            self._first_token_hard_ceiling(foreground_request=True),
        )
        if 0.0 < self._current_first_token_hard_ceiling_s < needed:
            logger.info(
                "⏱️ [MLX] first-token ceiling raised %.1fs → %.1fs: a %d-char prompt "
                "takes about %.1fs to read at %.0f tok/s",
                self._current_first_token_hard_ceiling_s,
                needed,
                self._current_prompt_chars,
                needed / _PREFILL_HEADROOM,
                self._measured_prefill_rate(),
            )
            self._current_first_token_hard_ceiling_s = needed
        # What the caller allowed, beside what the prompt will cost.
        #
        # A first-token ceiling is only meaningful next to the prompt it has
        # to read: 4 seconds is generous for a hundred tokens and impossible
        # for two thousand. Without both numbers in one line, a cancelled
        # request looks like a slow worker, and every decision she made while
        # playing was cancelled this way.
        if self._current_first_token_hard_ceiling_s > 0.0:
            logger.info(
                "⏱️ [MLX] first-token ceiling %.1fs for a %d-char prompt (%d max tokens)",
                self._current_first_token_hard_ceiling_s,
                self._current_prompt_chars,
                self._current_requested_max_tokens,
            )
        self._current_prefill_tokens_processed = 0
        self._current_prefill_tokens_total = 0
        self._prefill_observed_at = 0.0
        self._prefill_observed_tokens = 0
        self._mark_progress()

    def _mark_prefill_progress(
        self,
        req_id: str | None,
        *,
        processed: int,
        total: int,
    ) -> None:
        normalized_req_id = str(req_id or "")
        if (
            normalized_req_id
            and self._current_request_id
            and normalized_req_id != self._current_request_id
        ):
            return
        if not normalized_req_id and self._current_request_id:
            self._mark_progress()
            return
        done = max(0, int(processed or 0))
        # How fast this worker actually reads a prompt, from this worker.
        #
        # Measured rather than assumed, so a ceiling built on it is a ceiling
        # built on what the machine does. Averaged, so one slow chunk under
        # contention does not become the rule.
        now = time.time()
        # Between two observations of prefill, not since the request began.
        #
        # Measuring from the request start folds in queueing, admission and
        # whatever else happened before a single token was read: it reported
        # 4 tok/s on a worker doing 720, and asked for a ten-minute ceiling
        # on a prompt that takes a second and a half to read.
        last_at = float(getattr(self, "_prefill_observed_at", 0.0) or 0.0)
        last_done = int(getattr(self, "_prefill_observed_tokens", 0) or 0)
        if done > last_done and last_at > 0.0:
            spent = now - last_at
            if spent > 0.02:
                observed = (done - last_done) / spent
                previous = float(getattr(self, "_prefill_tokens_per_s", 0.0) or 0.0)
                self._prefill_tokens_per_s = (
                    observed if previous <= 0.0 else previous * 0.7 + observed * 0.3
                )
                _HOST_RATES["prefill"] = self._prefill_tokens_per_s
        if done != last_done:
            self._prefill_observed_at = now
            self._prefill_observed_tokens = done
        if done > last_done and getattr(self, "_current_turn_progress", None) is not None:
            from core.runtime.turn_progress import note_progress

            note_progress(progress=self._current_turn_progress)
        self._current_prefill_tokens_processed = done
        self._current_prefill_tokens_total = max(0, int(total or 0))
        self._mark_progress()

    def _mark_token_progress(
        self, req_id: str | None = None, *, generated_tokens: int | None = None
    ) -> None:
        now = time.time()
        normalized_req_id = str(req_id or "")
        if (
            normalized_req_id
            and self._current_request_id
            and normalized_req_id != self._current_request_id
        ):
            return
        if not normalized_req_id and self._current_request_id:
            # Id-less progress cannot be ATTRIBUTED to the active request:
            # crediting it set first-token timestamps from unrelated or
            # malformed messages. It still proves the worker is alive.
            self._mark_progress()
            return
        previous_count = int(getattr(self, "_tokens_this_request", 0) or 0)
        if generated_tokens is None:
            delta = 1  # Legacy visible-token frames carry one token each.
        elif isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool):
            delta = generated_tokens - previous_count
        else:
            delta = 0
        if delta <= 0:
            # Duplicate, reordered or malformed counters prove no new decoding.
            self._mark_progress()
            return
        # Decode measured the same way prefill is: between two observations,
        # so queueing before the first token is not charged to writing.
        previous_at = float(getattr(self, "_last_token_progress_at", 0.0) or 0.0)
        if previous_at > 0.0 and self._current_first_token_at > 0.0:
            spent = now - previous_at
            if 0.005 < spent < 5.0:
                observed = delta / spent
                previous = _HOST_RATES["decode"]
                _HOST_RATES["decode"] = (
                    observed if previous <= 0.0 else previous * 0.8 + observed * 0.2
                )
        self._last_token_progress_at = now
        # Published where the layers above can read it. Five deadlines are
        # waiting on this one generation, and each of them was deciding
        # whether to end it from a stopwatch rather than from whether it was
        # still saying anything.
        try:
            from core.runtime.turn_progress import note_progress

            if getattr(self, "_current_turn_progress", None) is not None:
                note_progress(progress=self._current_turn_progress)
        except ImportError:
            pass
        if self._current_first_token_at <= 0.0:
            self._current_first_token_at = now
            # How long this prompt took to read, written down where every
            # deadline is built from.
            #
            # The worker process records this and the deadlines are built in
            # this one, so the record the answer clock consults was empty on a
            # fresh runtime: reading a prompt counted as free, the clock
            # granted 23 seconds, this worker measured the same prompt as
            # needing 23.2 to read, and every user-facing generation was
            # cancelled at 23. The fallback ladder then found no small model
            # admitted under the memory headroom, waited out its budget and
            # ended the turn in a refusal.
            #
            # LIVE 2026-09-04, forty minutes after a clean boot: five
            # cancellations in ten minutes and not one answer delivered.
            #
            # Only after a worker has produced a token before, because
            # everything before the FIRST token of a worker's life is weights
            # coming off disk as well as the prompt, and that is a different
            # fact — measured separately, just below.
            _started_at = float(getattr(self, "_current_request_started_at", 0.0) or 0.0)
            if (
                int(getattr(self, "_tokens_since_spawn", 0) or 0) > 0
                and _started_at > 0.0
                and self._current_prompt_chars > 0
            ):
                try:
                    from core.brain.llm.thinking_reserve import (  # noqa: PLC0415
                        record_read_rate,
                    )

                    record_read_rate(
                        prompt_chars=self._current_prompt_chars,
                        elapsed_s=now - _started_at,
                    )
                except (ImportError, TypeError, ValueError):
                    pass
            # What loading this model actually cost, from the one request
            # that pays for it. Everything before the first token of a
            # worker's life is weights coming off disk plus reading the
            # prompt; the prompt's share is already measured, so the rest is
            # the load. Averaged across workers, because the disk is one disk.
            if int(getattr(self, "_tokens_since_spawn", 0) or 0) == 0:
                gigabytes = self._weight_gigabytes()
                started = float(getattr(self, "_current_request_started_at", 0.0) or 0.0)
                spent = now - started if started > 0.0 else 0.0
                loading = spent - (
                    self._prefill_floor_seconds(self._current_prompt_chars)
                    / _PREFILL_HEADROOM
                )
                if gigabytes > 0.0 and loading > 0.1:
                    observed = gigabytes / loading
                    previous = float(_HOST_RATES.get("weight_load") or 0.0)
                    _HOST_RATES["weight_load"] = (
                        observed if previous <= 0.0 else previous * 0.7 + observed * 0.3
                    )
                # And the whole thing as a duration, which is what the next
                # cold start actually has to survive.
                #
                # A rate assumes the time is spent reading bytes. LIVE
                # 2026-08-29: a 0.8GB model derived a 3.2s allowance from its
                # size and took longer than the 8s SLA to speak — the rest of
                # it is framework import, tokenizer, and shader compile, none
                # of which scale with the weights. Measured whole, no model of
                # where it went is needed.
                if spent > 0.1:
                    name = os.path.basename(str(self.model_path or ""))
                    if name:
                        seen = _COLD_FIRST_TOKEN_S.get(name, 0.0)
                        _COLD_FIRST_TOKEN_S[name] = (
                            spent if seen <= 0.0 else seen * 0.7 + spent * 0.3
                        )
        self._tokens_since_spawn = int(getattr(self, "_tokens_since_spawn", 0) or 0) + delta
        self._tokens_this_request = previous_count + delta
        self._mark_progress()

    def _clear_active_generation_tracking(self) -> None:
        self._current_turn_progress = None
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._last_token_progress_at = 0.0
        self._current_request_id = ""
        self._current_request_seq = 0
        self._current_request_progress_baseline_at = 0.0
        self._current_prompt_chars = 0
        self._current_requested_max_tokens = 0
        self._current_request_prompt_chars = 0
        self._current_first_token_hard_ceiling_s = 0.0
        self._current_prefill_tokens_processed = 0
        self._current_prefill_tokens_total = 0
        self._mark_progress()

    async def prove_visible_readiness(
        self,
        *,
        budget_s: float = 30.0,
        request_is_background: bool = False,
        foreground_request: bool = False,
        owner_label: str = "",
    ) -> str:
        """Show that this lane can answer, and record that it was shown.

        The one place that means "the lane has been seen to answer". There were
        two: warmup asked its probe, validated the reply and stamped
        ``_last_visible_readiness_at``; anything else that ran a health probe
        got the generation and none of the recording, because ``health_probe``
        deliberately suppresses the user-facing mark.

        So a second prover could succeed, report success, and leave conversation
        readiness exactly as unproven as it found it — each component sensible on
        its own and the composition wrong. Both callers go through here now, and
        a third cannot reintroduce the gap by forgetting a line.

        Returns "proved", or a short reason it could not be.
        """
        if not self.is_alive():
            return "no_worker"
        try:
            said = await asyncio.wait_for(
                self._generate_inner(
                    _READINESS_PROBE_PROMPT,
                    _retry=True,
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_label=owner_label or None,
                    max_tokens=16,
                    temp=0.0,
                    top_p=1.0,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    health_probe=True,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                ),
                timeout=max(1.0, float(budget_s)),
            )
        except TimeoutError:
            return "timed_out"
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
            return "failed"
        if not said or not str(said).strip():
            return "no_text"
        if not _readiness_answer_accepted(said):
            return "answer_mismatch"
        # The recording IS the proof. Without it the lane is exactly as
        # unproven as before the probe ran.
        self._last_visible_readiness_at = time.time()
        self._set_lane_state("ready")
        return "proved"

    def _mark_generation_completed(self, *, user_facing: bool = False) -> None:
        self._last_generation_completed_at = time.time()
        if user_facing:
            self._last_user_facing_completed_at = self._last_generation_completed_at
            self._last_visible_readiness_at = self._last_generation_completed_at
        self._clear_active_generation_tracking()

    def _set_lane_state(self, state: str, error: str = "") -> None:
        if state != self._lane_state:
            self._lane_transition_at = time.time()
            # A MONOTONIC companion. Watchdogs measure "how long has this lane
            # been warming?" and act on the answer by killing a load or forcing
            # a reset. Wall clock is the wrong ruler for a duration: an NTP
            # correction, a DST change, or this laptop sleeping and waking
            # makes the delta huge (killing a healthy load) or negative
            # (deferring intervention indefinitely). The wall-clock stamp stays
            # for display and for anything that needs a real date.
            self._lane_transition_monotonic_at = time.monotonic()
        self._lane_state = state
        if error:
            self._lane_error = str(error)
        elif state == "ready":
            self._lane_error = ""

    def _classify_failure(
        self,
        *,
        foreground_request: bool = False,
        reason: str = "",
        classification: str | None = None,
    ) -> str:
        if classification:
            return classification
        normalized_reason = str(reason or "").lower()
        if self._is_deep_solver_lane() and (
            "memory_pressure_refused_worker_spawn" in normalized_reason
            or "optional_deep_solver_memory_refusal" in normalized_reason
        ):
            return "non_critical_fallback"
        if foreground_request or (self._is_primary_or_deep_lane() and _foreground_owner_active()):
            return "foreground_blocking"
        return "background_degraded"

    def _record_degraded_event(
        self,
        reason: str,
        *,
        detail: str = "",
        severity: str = "warning",
        foreground_request: bool = False,
        classification: str | None = None,
    ) -> None:
        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event(
                "mlx_client",
                reason,
                detail=detail,
                severity=severity,
                classification=self._classify_failure(
                    foreground_request=foreground_request,
                    reason=f"{reason}:{detail}",
                    classification=classification,
                ),
                context={
                    "model": os.path.basename(self.model_path),
                    "lane_state": self._lane_state,
                    "warmup_in_flight": self._warmup_in_flight,
                },
            )
        except Exception as exc:  # noqa: BLE001 - observation must not break the observed
            # This runs inside generation, warmup and reboot paths purely to
            # WRITE a diagnostic. The old clause caught three error types, so a
            # serialization or disk failure inside record_degraded_event
            # escaped and failed the very operation it was reporting on —
            # turning "we noticed something degraded" into an outage.
            try:
                _record_mlx_degradation(
                    exc,
                    action="kept lane-local degraded state after health event emission failed",
                )
            except Exception as fallback_exc:  # noqa: BLE001 - fail-closed fallback
                # Nothing further to try: the primary recorder and its
                # fallback have both failed. Swallowing it silently would make
                # a blind observability path indistinguishable from a quiet
                # one, so the loss is named even though it cannot be acted on.
                logger.debug(
                    "MLX degradation fallback also failed (%s: %s); the original "
                    "event is unrecorded.",
                    type(fallback_exc).__name__,
                    fallback_exc,
                )
            logger.debug("Failed to record MLX degraded event: %s", exc)

    def _is_optional_deep_solver_memory_refusal(self, failure: Any) -> bool:
        """Whether this failure IS an admission refusal on the optional lane.

        Takes the exception, not its text. A string match on an arbitrary
        error message meant an unrelated failure could be suppressed by
        quoting the right words.
        """
        return self._is_deep_solver_lane() and isinstance(
            failure, ModelLoadAdmissionRefused
        )

    def _handle_optional_deep_solver_memory_refusal(self, failure: Any) -> bool:
        """Treat a refused 72B load as an unavailable optional lane, not a live-system failure."""

        if not self._is_optional_deep_solver_memory_refusal(failure):
            return False
        detail = str(failure)
        self._set_lane_state("cold")
        self._init_future = None
        self._consecutive_spawn_failures = 0
        # Short backoff prevents repeated oversized 72B attempts from flooding the
        # neural stream while keeping the optional lane available after pressure falls.
        self._spawn_backoff_until = time.time() + 60.0
        # Not a runtime failure: a runtime probe must not clear this one. It
        # expires on its own once pressure has had a chance to fall.
        self._spawn_backoff_cause = "memory_refusal"
        self._record_degraded_event(
            "optional_deep_solver_memory_refusal",
            detail=f"{os.path.basename(self.model_path)}:{detail}",
            severity="warning",
            foreground_request=False,
            classification="non_critical_fallback",
        )
        logger.warning(
            "🛡️ [MLX] Optional deep Solver worker refused by memory guard for %s: %s. "
            "Keeping primary Cortex authoritative.",
            os.path.basename(self.model_path),
            detail,
        )
        return True

    def _stale_after(
        self, *, during_generation: bool = False, foreground_request: bool = False
    ) -> float:
        """Heartbeat-stall timeout.

        [RESILIENCE] Widened for 32B foreground: recurrent depth doubles
        the compute per token, and complex prompts can legitimately take
        60-90s for prompt eval.  Killing the cortex when heartbeats are
        still arriving (worker is alive, just slow) was the #1 cause of
        'cortex died and never came back'.  As long as heartbeats arrive,
        the worker is alive — let it finish."""
        if self._is_deep_solver_lane():
            if foreground_request and during_generation:
                return 45.0
            return 90.0 if during_generation else 45.0
        if _model_is_heavy_lane(self.model_path):
            if foreground_request and during_generation:
                return 45.0  # was 22s — too aggressive with recurrent depth
            return 60.0 if during_generation else 30.0
        return 20.0 if during_generation else 15.0

    def _pressure_adaptive_stretch(self) -> tuple[float, str]:
        """Bounded stretch for token-progress budgets under live memory pressure.

        A RESIDENT heavy model's first token slows under unified-memory
        contention because prompt eval competes for bandwidth — the worker is
        starved, not wedged. Killing it answers a bandwidth problem with a
        ~20GB reload that deepens the contention (the Jul 7 soak doom loop:
        stall → force-kill → cold reload → next turn stalls under the same
        pressure). Only token-progress budgets stretch, and only within
        bounds: heartbeat wedge detection is untouched, caller deadlines
        still dominate, and the emergency tier still refuses generation
        outright before this is consulted.
        """
        if str(os.environ.get("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")).strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return 1.0, ""
        if not _model_is_heavy_lane(self.model_path):
            return 1.0, ""
        try:
            snapshot = get_memory_pressure_snapshot()
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return 1.0, ""
        if snapshot.emergency:
            return 1.0, ""  # the refuse-generation path owns emergencies
        if snapshot.critical:
            return 1.5, "memory_pressure_critical"
        if snapshot.high:
            return 1.35, "memory_pressure_high"
        if snapshot.warning:
            return 1.2, "memory_pressure_warning"
        return 1.0, ""

    def _pressure_receipt_suffix(self) -> str:
        """Name the memory-pressure tier on stall receipts.

        A stall verdict under contention is a different incident than a stall
        on an idle machine; the narrator (and anyone reading the degradation
        ledger) should not have to correlate timestamps to know which one
        happened.
        """
        try:
            snapshot = get_memory_pressure_snapshot()
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return ""
        return f":memory={snapshot.level}" if snapshot.warning else ""

    def _first_token_sla(self, *, foreground_request: bool = False) -> float:
        prompt_chars = max(0, int(getattr(self, "_current_request_prompt_chars", 0) or 0))
        # Prompt eval dominates first-token latency on the 32B/72B lanes.
        # Recent live traces showed ~5.3k-token prompts taking 66-76s before
        # the first token arrived, which is healthy-but-slow rather than wedged.
        estimated_prompt_tokens = (prompt_chars / 4.6) if prompt_chars > 0 else 0.0

        def _with_prompt_eval_headroom(
            base_sla: float,
            *,
            threshold_tokens: float,
            eval_seconds_per_token: float,
            cap_s: float,
        ) -> float:
            if estimated_prompt_tokens <= threshold_tokens:
                return base_sla
            extra = (estimated_prompt_tokens - threshold_tokens) * eval_seconds_per_token
            return min(cap_s, base_sla + extra)

        # Cold-start exemption: the FIRST real foreground generation after a
        # worker warmup or reboot legitimately needs 30–45 s on 32B because
        # Metal shaders are still JIT-compiling and the KV cache is empty.
        # Tripping the SLA at 22 s on the very first user turn was bouncing
        # Cortex to UNAVAILABLE before the model could produce a token.
        # _last_generation_completed_at is zero until a real generation has
        # finished; we use that as the cold-start signal.
        is_cold_start = float(getattr(self, "_last_generation_completed_at", 0.0) or 0.0) <= 0.0
        if self._is_deep_solver_lane():
            if foreground_request:
                base = 52.0 if is_cold_start else 32.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=768.0,
                    eval_seconds_per_token=0.018,
                    cap_s=115.0,
                )
            return 30.0
        if _model_is_heavy_lane(self.model_path):
            # [RESILIENCE] Recurrent depth 2x loops means prompt eval takes
            # significantly longer.  These SLAs must accommodate that without
            # killing the cortex.  Cold-start can legitimately need 90s for
            # Metal shader JIT + recurrent depth prompt eval on a 5k-token
            # prompt.  The point of these SLAs is to catch WEDGED workers
            # (no heartbeats), not SLOW workers (heartbeats arriving).
            if foreground_request:
                # Live measurement 2026-06-11: warm 32B first tokens at
                # 46.4s under the macos26 guard + serialized lanes — the
                # 45s base declared healthy generations wedged, and the
                # lane recycle's cancellation swept well beyond the
                # offending request (it killed the proof battery's repair
                # coroutine mid-await). Wedge detection belongs to the
                # heartbeat/stall checks; this SLA only needs to beat
                # genuinely dead workers.
                base = 120.0 if is_cold_start else 90.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=512.0,
                    eval_seconds_per_token=0.015,
                    cap_s=240.0,
                )
            return 90.0
        return 8.0

    def _first_token_absolute_ceiling(self, *, foreground_request: bool = False) -> float:
        """Return the non-negotiable no-token ceiling for one generation.

        Heartbeats prove that the worker process is alive; they do not prove
        that the active model request is making useful progress. The primary
        lane previously allowed a heartbeating but tokenless request to run
        beyond the endpoint deadline, after which the inference layer could
        start additional retries. Keep this ceiling below the foreground API
        envelope so the caller still has time to recover or use another lane.
        """

        if self._is_deep_solver_lane():
            default = 165.0 if foreground_request else 120.0
        elif _model_is_heavy_lane(self.model_path):
            default = 120.0 if foreground_request else 90.0
        else:
            default = 30.0 if foreground_request else 20.0
        stretch, _ = self._pressure_adaptive_stretch()
        default *= stretch
        if os.environ.get("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S") is not None:
            configured = _finite_env_float(
                "AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", default, minimum=10.0
            )
            # Bounded above too: an absurd ceiling disables the watchdog.
            return min(3600.0, max(10.0, configured))
        return default

    def _weight_gigabytes(self) -> float:
        """How much this model has to be read off disk before it can speak."""

        path = str(self.model_path or "")
        if not path:
            return 0.0
        remembered = _WEIGHT_SIZES.get(path)
        if remembered is not None:
            return remembered
        total = 0
        try:
            root = Path(path)
            entries = root.iterdir() if root.is_dir() else [root]
            for entry in entries:
                if entry.suffix in (".safetensors", ".bin", ".gguf", ".npz"):
                    total += entry.stat().st_size
        except OSError:
            total = 0
        size = total / (1024.0**3)
        _WEIGHT_SIZES[path] = size
        return size

    def _cold_lane_first_token_allowance(self) -> float:
        """Extra time a lane gets for the first token of its life. Once.

        A worker that has produced no token since it was spawned is not
        distinguishable from a wedged one by silence alone, and the two calls
        for opposite actions. Recycling resolves that ambiguity by killing —
        which spawns another worker in exactly the same state, which is
        silent for the same reason, which is recycled again.

        LIVE 2026-08-29, four lines: "Spawning worker for
        Qwen2.5-1.5B-Instruct-4bit" (02:27:07), "Worker ready" (:12),
        "produced no token in 20.2s ... Recycling the lane" (:32), "Circuit
        OPEN for Reflex". Her planner then got an empty answer and fell back
        to canned text, and the feed recorded that as her failure.

        Weights have to be read before anything is generated, and how long
        that takes is a fact about this host and this model, measured the way
        prefill and decode already are. Once a token has arrived the lane is
        warm and this is zero — the allowance covers the state, not the lane.
        """

        if int(getattr(self, "_tokens_since_spawn", 0) or 0) > 0:
            return 0.0
        ceiling = self._first_token_absolute_ceiling()
        measured = _COLD_FIRST_TOKEN_S.get(
            os.path.basename(str(self.model_path or "")), 0.0
        )
        if measured <= 0.0:
            # Nothing measured, so the bound that already exists.
            #
            # Deriving it from the weights assumes the time is spent reading
            # them: a 0.8GB model works out to 3.2 seconds and took longer
            # than the 8-second SLA to speak, because the rest is framework
            # import, tokenizer and shader compile. Guessing low here is what
            # made the measurement impossible — the first generation of a
            # worker's life is the one that would have measured it, and it was
            # abandoned every time.
            #
            # The absolute ceiling is this lane's own answer to "how long may
            # it be silent", so an unmeasured cold start gets that rather than
            # an arithmetic guess.
            return ceiling
        return min(measured * _COLD_START_HEADROOM, ceiling)

    def _measured_prefill_rate(self) -> float:
        """Tokens a second this worker reads a prompt at, as measured.

        Falls back to a deliberately pessimistic rate until it has seen one:
        being generous with an unmeasured worker costs a little latency, and
        being mean with it costs the answer.
        """
        rate = float(getattr(self, "_prefill_tokens_per_s", 0.0) or 0.0)
        return rate if rate > 0.0 else _UNMEASURED_PREFILL_RATE

    def least_time_to_read(self, prompt_chars: int) -> float:
        """The least time in which this worker could read that prompt.

        Public because whoever grants the deadline has to use the same number
        the worker will judge itself by. Two estimates of one fact are two
        deadlines, and the smaller one wins by cancelling the work: LIVE
        2026-09-04, one line apart, "the prompt takes about 2s to read"
        granting 25 seconds and "a 2867-char prompt takes about 8.8s to read
        at 82 tok/s" needing 26.3.

        This is the rate THIS worker is running at now, which is the fact a
        percentile over past readings cannot follow — under memory pressure it
        halved twice inside a minute.
        """
        return self._prefill_floor_seconds(prompt_chars)

    def _prefill_floor_seconds(self, prompt_chars: int) -> float:
        """The least time in which this prompt could produce a first token."""
        chars = max(0, int(prompt_chars or 0))
        if chars <= 0:
            return 0.0
        return (chars / _CHARS_PER_TOKEN / self._measured_prefill_rate()) * _PREFILL_HEADROOM

    def _first_token_hard_ceiling(self, *, foreground_request: bool = False) -> float:
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        # Finite-range validation: negative or non-finite multipliers/padding
        # previously produced premature aborts, an unbounded watchdog, or a
        # non-finite timer value.
        hard_mult = min(10.0, _finite_env_float("AURA_FIRST_TOKEN_HARD_MULT", 1.8, minimum=1.0))
        hard_pad = min(600.0, _env_duration_s("AURA_FIRST_TOKEN_HARD_PAD_S", 20.0))
        # The hard ceiling exists to kill LIVELOCKED generations (heartbeats,
        # zero tokens). Under live memory pressure a starved-but-healthy heavy
        # lane looks exactly like that livelock from outside — stretch the
        # verdict boundary (bounded, never past the caller's deadline) so
        # contention gets time to clear instead of triggering a 20GB reload.
        stretch, _ = self._pressure_adaptive_stretch()
        return min(
            first_token_sla * hard_mult * stretch + hard_pad,
            self._first_token_absolute_ceiling(foreground_request=foreground_request),
        )

    def _deadline_bound_first_token_hard_ceiling(
        self,
        deadline_remaining_s: float | None,
        *,
        foreground_request: bool = False,
    ) -> float:
        hard_ceiling = self._first_token_hard_ceiling(
            foreground_request=foreground_request,
        )
        if deadline_remaining_s is None:
            return hard_ceiling
        try:
            remaining = float(deadline_remaining_s)
        except (TypeError, ValueError):
            return hard_ceiling
        if remaining <= 0.0:
            # CP126 dec24697: this returned 10.0 — a NEW ten-second budget
            # granted to a request whose deadline had already expired. The
            # caller was promised a bound and then quietly given more time
            # past it. Zero means zero: the watchdog fires at once and the
            # turn fails closed, which is what the exhausted deadline meant.
            return 0.0
        # Leave enough wall-clock for the caller to fail closed and recycle
        # the worker — that reserve is the whole point of this bound, and the
        # old 10-second floor overrode it: an 8-second budget produced a
        # 10-second ceiling, two seconds PAST the deadline, with no reserve at
        # all. Now the reserve always holds and the ceiling never exceeds what
        # the caller has left.
        return max(0.0, min(hard_ceiling, remaining - _capped_reserve(4.0, remaining)))

    def _start_foreground_first_token_watchdog(
        self,
        req_id: str,
        *,
        foreground_request: bool = False,
        hard_ceiling_s: float | None = None,
    ) -> _threading.Timer | None:
        """Abort tokenless foreground generations even if the event loop wedges."""

        if not foreground_request or not self._is_primary_or_deep_lane():
            return None
        hard_ceiling = (
            max(10.0, float(hard_ceiling_s))
            if hard_ceiling_s is not None
            else self._first_token_hard_ceiling(foreground_request=True)
        )
        fire_after = max(10.0, hard_ceiling + _WATCHDOG_ENFORCEMENT_SLACK_S)
        model_name = os.path.basename(self.model_path)

        def _enforce() -> None:
            if _runtime_shutdown_requested():
                return
            try:
                if (
                    str(self._current_request_id or "") != str(req_id or "")
                    or self._current_request_started_at <= 0.0
                    or self._current_first_token_at > 0.0
                ):
                    return
                elapsed = max(0.0, time.time() - self._current_request_started_at)
                if elapsed < hard_ceiling:
                    return
                logger.error(
                    "🛑 [MLX] Out-of-band first-token watchdog aborting %s "
                    "(%.1fs elapsed, hard=%.1fs).",
                    model_name,
                    elapsed,
                    hard_ceiling,
                )
                self._record_degraded_event(
                    "first_token_wall_clock_watchdog",
                    detail=f"{model_name}>{hard_ceiling:.1f}s",
                    severity="critical",
                    foreground_request=True,
                )
                # The preemption ladder, both rungs. This site went straight to
                # the kill, which costs a ~60-90s reload of a 20GB resident
                # model — the price of the recovery, paid in full, on the
                # chance that the worker was merely slow rather than wedged.
                if self._first_token_watchdog_soft_cancel(req_id):
                    return
                # …and bound to the request this watchdog actually checked.
                # Without expected_request_id there is a window between the id
                # check above and the abort in which a NEW foreground request
                # can start, and the abort kills that one instead.
                self.force_abort_active_generation(
                    "first_token_wall_clock_watchdog",
                    expected_request_id=str(req_id or ""),
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.error("MLX first-token watchdog failed: %s", exc)

        timer = _threading.Timer(fire_after, _enforce)
        timer.daemon = True
        timer.name = f"AuraMLXFirstTokenWatchdog:{model_name[:32]}"
        timer.start()
        self._foreground_generation_watchdog = timer
        return timer

    def _first_token_watchdog_soft_cancel(self, req_id: str) -> bool:
        """Ask first. True when the job ended without killing the worker.

        A first-token overrun has two causes that look identical from here: a
        worker stuck in prefill on the GPU, which cannot poll the cancel word
        and must be killed, and a worker that is simply slow and one decode
        step from its first token, which will observe the cancel immediately.
        Only the first needs a reload. Telling them apart costs one bounded
        wait; guessing wrong costs a 20GB model.

        Runs on the watchdog's own timer thread, never the event loop, so the
        wait blocks nothing a caller is on.
        """
        try:
            receipt = self.soft_cancel_active_generation(
                "first_token_wall_clock_watchdog"
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        if not receipt.get("requested"):
            return False
        target = self._soft_cancel_target
        if not target:
            return False
        # The same slack this watchdog already allots itself for out-of-band
        # enforcement (`fire_after`), reused rather than a second number: it is
        # the overshoot this function has always treated as acceptable.
        deadline = time.monotonic() + _WATCHDOG_ENFORCEMENT_SLACK_S
        while time.monotonic() < deadline:
            process = self._process
            if process is None or not process.is_alive():
                return False
            if self._soft_cancel_ack_matches(target):
                logger.info(
                    "✋ [MLX] First-token overrun on %s ended cooperatively — "
                    "worker and model kept warm.",
                    os.path.basename(self.model_path),
                )
                return True
            if str(self._current_request_id or "") != str(req_id or ""):
                # The job finished on its own while we waited. Nothing to kill,
                # and killing now would take whatever started after it.
                return True
            time.sleep(0.05)
        logger.warning(
            "🛑 [MLX] Soft-cancel unacknowledged after %.1fs on %s — the worker "
            "is not decoding; escalating to abort.",
            _WATCHDOG_ENFORCEMENT_SLACK_S,
            os.path.basename(self.model_path),
        )
        return False

    def _token_stall_after(self, *, foreground_request: bool = False) -> float:
        stretch, _ = self._pressure_adaptive_stretch()
        if self._is_deep_solver_lane():
            return (18.0 if foreground_request else 25.0) * stretch
        if _model_is_heavy_lane(self.model_path):
            # [RESILIENCE] Reverted from 10s — recurrent depth can cause
            # legitimate pauses between tokens during the recurrent block
            # computation. Sized up with the 2026-06-11 first-token
            # remeasurement: inter-token pauses stretch the same way under
            # the macos26 guard, and a stall verdict triggers the same
            # over-broad lane recycle as an SLA breach.
            return (40.0 if foreground_request else 45.0) * stretch
        return 8.0

    def _confirm_worker_reported_loop_stall(
        self,
        payload: dict[str, Any],
    ) -> tuple[bool, float]:
        """Apply request-aware budgets to the worker's coarse progress alarm.

        The child only knows that no token activity has occurred for 30s. On a
        resident 32B request that is normal during prompt evaluation, so the
        parent confirms the signal against the request's first-token or
        inter-token budget before recording a runtime fault.
        """

        request_id = str(payload.get("request_id") or "")
        current_request_id = str(getattr(self, "_current_request_id", "") or "")
        if not request_id or not current_request_id or request_id != current_request_id:
            return False, 0.0
        try:
            age_s = max(0.0, float(payload.get("job_age_s") or 0.0))
        except (TypeError, ValueError):
            return False, 0.0

        first_token_at = float(getattr(self, "_current_first_token_at", 0.0) or 0.0)
        if first_token_at <= 0.0:
            threshold_s = float(getattr(self, "_current_first_token_hard_ceiling_s", 0.0) or 0.0)
            if threshold_s <= 0.0:
                threshold_s = self._first_token_hard_ceiling(
                    foreground_request=self._is_primary_or_deep_lane(),
                )
        else:
            threshold_s = self._token_stall_after(
                foreground_request=self._is_primary_or_deep_lane(),
            )
        threshold_s = max(30.0, float(threshold_s))
        return age_s > threshold_s, threshold_s

    def _warmup_timeout(self) -> float:
        # [STABILITY v56] Raised from 75.0s → 180.0s. 32B models on M5
        # regularly take 120-150s to cold-load and compile Metal shaders.
        return 180.0 if self._is_primary_or_deep_lane() else 30.0

    def _handshake_timeout(self) -> float:
        """Absolute upper bound for worker init before we declare the process wedged."""
        return 300.0 if self._is_primary_or_deep_lane() else 120.0

    def _request_scoped_init_timeout(
        self,
        deadline: Deadline | None,
        *,
        foreground_request: bool,
    ) -> tuple[float, bool]:
        """Bound init waits to the caller's budget so fallback can still happen in time."""
        full_timeout = self._handshake_timeout()
        if not isinstance(deadline, Deadline):
            return full_timeout, False

        remaining = deadline.remaining
        if remaining is None:
            return full_timeout, False

        reserve = 5.0 if foreground_request else 2.0
        # [STABILITY v57] Increased minimum from 0.25 to 10.0 for fallbacks.
        # Background fallbacks were being killed after 3s because their budget
        # was too tight to even start the worker.
        scoped_timeout = max(10.0 if not foreground_request else 5.0, remaining - reserve)
        return min(full_timeout, scoped_timeout), scoped_timeout < full_timeout

    def get_lane_status(self) -> dict[str, Any]:
        # [STABILITY v59] Do NOT clear the foreground owner while a warmup
        # is actively in flight.  The warmup legitimately holds the owner
        # for up to 180s; clearing it mid-load lets background workers
        # respawn and compete for memory, creating the desktop deadlock.
        if int(getattr(self, "_active_generations", 0) or 0) <= 0 and not self._warmup_in_flight:
            stale_owner = _clear_stale_foreground_owner()
            if stale_owner:
                logger.warning(
                    "♻️ [MLX] Cleared stale foreground owner %s during lane status check.",
                    stale_owner,
                )
        self._check_lane_state_staleness()  # [STABILITY v54] Eagerly check and reset stuck/stale lane states
        worker_alive = self.is_alive()
        lane_state = self._lane_state
        lane_error = self._lane_error
        now = time.time()
        worker_progress_anchor = max(
            self._last_progress_at,
            self._last_ready_at,
            self._last_token_progress_at,
            self._last_generation_completed_at,
        )
        visible_conversation_anchor = max(
            float(getattr(self, "_last_visible_readiness_at", 0.0) or 0.0),
            float(getattr(self, "_last_user_facing_completed_at", 0.0) or 0.0),
        )
        progress_age_s = (
            max(0.0, now - worker_progress_anchor) if worker_progress_anchor > 0.0 else None
        )
        heartbeat_age_s = (
            max(0.0, now - self._last_heartbeat) if self._last_heartbeat > 0.0 else None
        )
        readiness_blockers: list[str] = []
        if _runtime_shutdown_requested():
            readiness_blockers.append("runtime_shutdown")
        if not worker_alive:
            readiness_blockers.append("worker_not_alive")
        if not self._init_done:
            readiness_blockers.append("init_not_complete")
        if lane_state != "ready":
            readiness_blockers.append(f"lane_{lane_state}")
        if worker_progress_anchor <= 0.0:
            readiness_blockers.append("no_worker_progress")
        elif progress_age_s is not None and progress_age_s > self._stale_after():
            readiness_blockers.append("worker_progress_stale")
        # The visible-conversation probe verifies the primary lane has served a
        # real user-facing turn. It is only meaningful when a UI surface is
        # attached; a headless proof/longevity run has no user surface, so a
        # warm+alive worker is the legitimate ready state and this probe would be
        # a permanent false positive there. (Mirrors the inference_gate guard.)
        _proof_headless = False
        try:
            from core.runtime.proof_policy import proof_headless_run

            _proof_headless = proof_headless_run()
        except (ImportError, RuntimeError, AttributeError):
            _proof_headless = False
        if (
            self._is_primary_or_deep_lane()
            and visible_conversation_anchor <= 0.0
            and not _proof_headless
        ):
            readiness_blockers.append("visible_conversation_probe_missing")
        if lane_state == "ready" and not worker_alive:
            lane_state = "cold"
            lane_error = "worker_not_alive"
            self._set_lane_state(lane_state, lane_error)
        elif lane_state == "ready" and any(
            blocker in {"no_worker_progress", "worker_progress_stale"}
            for blocker in readiness_blockers
        ):
            lane_state = "recovering"
            lane_error = "worker_progress_stale"
            self._set_lane_state(lane_state, lane_error)
            if f"lane_{lane_state}" not in readiness_blockers:
                readiness_blockers.append(f"lane_{lane_state}")
        recurrent_depth_status = _normalize_recurrent_depth_status(
            self._recurrent_depth_status,
            model_path=self.model_path,
        )
        recurrent_depth_blocker = _recurrent_depth_readiness_blocker(recurrent_depth_status)
        if recurrent_depth_blocker and recurrent_depth_blocker not in readiness_blockers:
            readiness_blockers.append(recurrent_depth_blocker)
            if lane_state == "ready":
                lane_state = "recovering"
                lane_error = recurrent_depth_blocker
                self._set_lane_state(lane_state, lane_error)
        foreground_owned = _foreground_owner_active()
        foreground_owner = _FOREGROUND_OWNER_NAME
        if self._warmup_in_flight:
            readiness_blockers.append("warmup_in_flight")
        if foreground_owned and foreground_owner.startswith("warmup:"):
            readiness_blockers.append("warmup_foreground_owner")
        elif foreground_owned and self._active_generations > 0:
            readiness_blockers.append("active_generation_in_flight")
        readiness_blockers = list(dict.fromkeys(readiness_blockers))

        conversation_ready = not readiness_blockers
        return {
            "model_path": self.model_path,
            "state": lane_state,
            "last_error": lane_error,
            "conversation_ready": conversation_ready,
            "readiness_blockers": readiness_blockers,
            "foreground_owned": foreground_owned,
            "foreground_owner": foreground_owner,
            "foreground_owned_at": _FOREGROUND_OWNER_ACQUIRED_AT,
            "last_heartbeat": self._last_heartbeat,
            "heartbeat_age_s": heartbeat_age_s,
            "last_progress_at": self._last_progress_at,
            "progress_age_s": progress_age_s,
            "worker_progress_anchor": worker_progress_anchor,
            "last_token_progress_at": self._last_token_progress_at,
            "last_ready_at": self._last_ready_at,
            "last_generation_completed_at": self._last_generation_completed_at,
            "last_user_facing_completed_at": self._last_user_facing_completed_at,
            "last_visible_readiness_at": self._last_visible_readiness_at,
            "last_transition_at": self._lane_transition_at,
            "last_transition_monotonic_at": getattr(
                self, "_lane_transition_monotonic_at", 0.0
            ),
            "warmup_attempted": self._warmup_attempted,
            "warmup_in_flight": self._warmup_in_flight,
            "model_load_admission": self._model_load_admission_status(),
            "spawn_gate": _spawn_gate_snapshot(),
            "active_generations": int(self._active_generations),
            "process_started_at": self._process_started_at,
            "current_request_started_at": self._current_request_started_at,
            "current_first_token_at": self._current_first_token_at,
            "current_request_prompt_chars": self._current_request_prompt_chars,
            "current_prefill_tokens_processed": self._current_prefill_tokens_processed,
            "current_prefill_tokens_total": self._current_prefill_tokens_total,
            "recurrent_depth": recurrent_depth_status,
            "unified_recurrent_shadow": copy.deepcopy(
                getattr(self, "_unified_recurrent_shadow_status", {})
            ),
            "unified_recurrent_shadow_probe": copy.deepcopy(
                getattr(self, "_unified_recurrent_shadow_probe_status", {})
            ),
            "unified_recurrent_shadow_canary": copy.deepcopy(
                getattr(self, "_unified_recurrent_shadow_canary_status", {})
            ),
            "unified_recurrent_qualified_activation": copy.deepcopy(
                getattr(
                    self,
                    "_unified_recurrent_qualified_activation_status",
                    {},
                )
            ),
            "request_age_s": (
                max(0.0, time.time() - self._current_request_started_at)
                if self._current_request_started_at
                else 0.0
            ),
        }

    def get_supervision_status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "lane": os.path.basename(self.model_path),
            "state": self._lane_state,
            "alive": self.is_alive(),
            "active_generations": int(self._active_generations),
            "expert_adapter_state_unknown": bool(
                getattr(self, "_expert_adapter_state_unknown", False)
            ),
            "unified_recurrent_shadow": copy.deepcopy(
                getattr(self, "_unified_recurrent_shadow_status", {})
            ),
            "unified_recurrent_shadow_probe": copy.deepcopy(
                getattr(self, "_unified_recurrent_shadow_probe_status", {})
            ),
            "unified_recurrent_shadow_canary": copy.deepcopy(
                getattr(self, "_unified_recurrent_shadow_canary_status", {})
            ),
            "unified_recurrent_qualified_activation": copy.deepcopy(
                getattr(
                    self,
                    "_unified_recurrent_qualified_activation_status",
                    {},
                )
            ),
            "process_uptime_s": max(0.0, now - self._process_started_at)
            if self._process_started_at
            else 0.0,
            "request_age_s": max(0.0, now - self._current_request_started_at)
            if self._current_request_started_at
            else 0.0,
            "time_to_first_token_s": (
                max(0.0, self._current_first_token_at - self._current_request_started_at)
                if self._current_request_started_at and self._current_first_token_at
                else None
            ),
            "idle_for_s": self._idle_for_s(now),
            "liveness_quiet_for_s": self._liveness_quiet_for_s(now),
        }

    def _work_anchor(self) -> float:
        """The last time this worker did WORK, not the last time it breathed.

        CP126 275480c8: idleness was anchored on the maximum of the work
        stamps AND ``_last_heartbeat``/``_last_progress_at``. A healthy
        resident worker heartbeats the whole time it is doing nothing, so that
        anchor advanced continuously and ``now - anchor`` never reached
        min_idle_s. The fragmentation recycle this gates could therefore never
        fire — an advertised reclaim that was structurally unreachable.

        A worker that has never done any work falls back to when it started:
        ninety minutes of doing nothing is idle regardless of whether the
        "nothing" began at boot.
        """
        anchor = max(
            float(self._last_generation_completed_at or 0.0),
            float(self._last_token_progress_at or 0.0),
        )
        if anchor > 0.0:
            return anchor
        return float(self._process_started_at or 0.0)

    def _idle_for_s(self, now: float | None = None) -> float:
        """Seconds since this worker last did any work."""
        moment = time.time() if now is None else now
        anchor = self._work_anchor()
        return max(0.0, moment - anchor) if anchor > 0.0 else 0.0

    def _liveness_quiet_for_s(self, now: float | None = None) -> float:
        """Seconds since this worker last showed ANY sign of life.

        Kept separate from idleness on purpose: "has done no work for an hour"
        and "has not made a sound for an hour" are different facts, and the
        second one is the one that means something is wrong.
        """
        moment = time.time() if now is None else now
        anchor = max(
            float(self._last_generation_completed_at or 0.0),
            float(self._last_ready_at or 0.0),
            float(self._last_token_progress_at or 0.0),
            float(self._last_progress_at or 0.0),
            float(self._last_heartbeat or 0.0),
        )
        return max(0.0, moment - anchor) if anchor > 0.0 else 0.0

    def should_recycle_for_fragmentation(
        self,
        *,
        max_uptime_s: float = 5400.0,
        min_idle_s: float = 900.0,
    ) -> bool:
        if not self.is_alive() or self._active_generations > 0 or _foreground_owner_active():
            return False
        if self._process_started_at <= 0.0:
            return False
        now = time.time()
        return bool(
            (now - self._process_started_at) >= float(max_uptime_s)
            and self._idle_for_s(now) >= float(min_idle_s)
        )

    def note_lane_recovering(self, reason: str) -> None:
        self._warmup_in_flight = False
        # A foreground warmup can be refused by the unified-memory guard even
        # though the primary worker is already alive and initialized. Marking
        # that as recovering strands the live desktop lane behind
        # lane_recovering + visible_conversation_probe_missing. In that case the
        # correct next step is to let the foreground turn prove visible
        # readiness, not spawn or recycle another model process.
        if str(reason or "") == "foreground_warmup_deferred_memory_pressure":
            try:
                worker_ready = bool(self.is_alive() and self._init_done)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                worker_ready = False
            if worker_ready:
                # CP126 5b870404: this used to stamp _last_ready_at and
                # _last_progress_at with time.time(). Nothing had happened —
                # no heartbeat, no token, no probe, no worker response. It
                # MANUFACTURED readiness evidence from a process being alive
                # and a historical init flag, and those two timestamps are
                # what every staleness and idleness check downstream reads.
                # A lane could look freshly-responsive for as long as this
                # path kept being taken.
                #
                # The conclusion was right and the evidence was invented. The
                # lane state still moves to ready, because a live initialized
                # worker refused a redundant warmup is not recovering — but
                # the clocks now say what actually last happened.
                self._set_lane_state("ready")
                return
        self._set_lane_state("recovering", reason)

    def _lane_runtime_failure(self) -> str:
        error = str(getattr(self, "_lane_error", "") or "")
        if error.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
            return error
        return ""

    def refresh_runtime_availability(self, *, force_probe: bool = False) -> bool:
        """Clear stale runtime-failure poison when the host probe is healthy again.

        A transient MLX runtime failure should not strand the lane in a failed
        state or an exponential spawn backoff once the runtime is healthy again.
        """
        runtime_error = self._lane_runtime_failure()
        if not runtime_error and time.time() >= float(
            getattr(self, "_spawn_backoff_until", 0.0) or 0.0
        ):
            return False

        ok, detail = _probe_mlx_runtime(force=force_probe)
        if not ok:
            self._mark_runtime_unavailable(detail)
            return False

        # CP126 ee4ccfcc. A healthy runtime probe proves ONE thing: the MLX
        # runtime imports and initializes. It says nothing about an out-of-
        # memory kill, a corrupt checkpoint, a model-load fault or a refused
        # memory admission — and this method cleared the backoff from all of
        # them, so a lane that had just crashed three times respawned
        # immediately because `import mlx` worked.
        backoff_cause = str(getattr(self, "_spawn_backoff_cause", "") or "")
        clears_backoff = bool(runtime_error) or backoff_cause in {"", "runtime_unavailable"}
        backoff_active = float(getattr(self, "_spawn_backoff_until", 0.0) or 0.0) > 0.0
        recovered = bool(runtime_error) or (backoff_active and clears_backoff)
        if recovered:
            logger.info(
                "♻️ [MLX] Runtime probe recovered for %s. Clearing failed lane/backoff state.",
                os.path.basename(self.model_path),
            )
        elif backoff_active:
            logger.info(
                "⏳ [MLX] Runtime is healthy for %s but the spawn backoff came from "
                "%s, which a runtime probe cannot clear. Backoff stands.",
                os.path.basename(self.model_path),
                backoff_cause,
            )
        if clears_backoff:
            self._consecutive_spawn_failures = 0
            self._spawn_backoff_until = 0.0
            self._spawn_backoff_cause = ""
        self._warmup_in_flight = False
        if self._lane_state == "failed" or runtime_error:
            self._set_lane_state("cold")
        else:
            self._lane_error = ""
        return recovered

    def _request_lock_timeout(
        self,
        deadline: Deadline | None,
        *,
        foreground_request: bool,
    ) -> float:
        # Tightened from 30s to 12s for foreground: if the current holder has
        # been in-flight for longer than this budget, the second user message
        # should cascade to brainstem/cloud rather than keep waiting.  The
        # prior 30s budget stacked on top of a hung 32B generation produced
        # the 60–90 s "Aura is thinking..." windows the user reported.
        default = 12.0 if foreground_request else 60.0
        if not isinstance(deadline, Deadline):
            return default

        remaining = deadline.remaining
        if remaining is None:
            return default

        # Same clamp as the foreground-owner wait above, same reason.
        reserve = 3.0 if foreground_request else 2.0
        return max(0.0, min(default, remaining - _capped_reserve(reserve, remaining)))

    async def _acquire_request_lock(
        self,
        *,
        owner_label: str,
        deadline: Deadline | None,
        foreground_request: bool,
    ) -> bool:
        wait_budget = self._request_lock_timeout(
            deadline,
            foreground_request=foreground_request,
        )
        loop = asyncio.get_running_loop()
        wait_started = loop.time()
        wait_deadline = wait_started + wait_budget
        last_log_at = 0.0
        maintenance_preempt_requested = False

        while loop.time() < wait_deadline:
            if self._try_acquire_request_lock(owner_label=owner_label):
                return True

            now = loop.time()
            waited = max(0.0, now - wait_started)
            holder = self._request_lock_owner_label or "another_request"

            if (
                foreground_request
                and not maintenance_preempt_requested
                and holder == "reasoning_nonparametric_ingest"
            ):
                receipt = self.soft_cancel_active_generation(
                    reason="foreground_preemption_nonparametric_ingest"
                )
                maintenance_preempt_requested = bool(receipt.get("requested"))
                if maintenance_preempt_requested:
                    logger.info(
                        "⏭️ [MLX] Foreground request preempted bounded "
                        "non-parametric maintenance on %s.",
                        os.path.basename(self.model_path),
                    )

            if waited >= 5.0 and (now - last_log_at) >= 5.0:
                holder_age = (
                    max(0.0, time.time() - self._request_lock_acquired_at)
                    if self._request_lock_acquired_at
                    else 0.0
                )
                logger.info(
                    "⏳ [MLX] Waiting for in-flight request on %s (owner=%s, held %.1fs).",
                    os.path.basename(self.model_path),
                    holder,
                    holder_age,
                )
                last_log_at = now

            await asyncio.sleep(min(0.05, max(0.0, wait_deadline - loop.time())))

        # Bind the observation to an IDENTITY, not just an age.
        #
        # Everything below decides whether to cancel the in-flight generation
        # because the holder has been slow. The holder can release, and a new
        # request can become current, between measuring the age here and
        # cancelling further down — and the cancel used to re-read
        # _current_gen_future at that moment, so it killed whichever request
        # happened to be running by then. A caller was preempted for another
        # request's slowness.
        #
        # Capturing the victim here lets the cancel below refuse unless it is
        # still the same generation that was measured.
        victim_future = self._current_gen_future
        victim_request_id = str(getattr(self, "_current_request_id", "") or "")
        holder = self._request_lock_owner_label or "another_request"
        holder_age = (
            max(0.0, time.time() - self._request_lock_acquired_at)
            if self._request_lock_acquired_at
            else 0.0
        )
        logger.warning(
            "⏳ [MLX] Request queue timeout after %.1fs for %s while waiting on %s (held %.1fs).",
            wait_budget,
            os.path.basename(self.model_path),
            holder,
            holder_age,
        )
        self._record_degraded_event(
            "request_lock_timeout",
            detail=f"{os.path.basename(self.model_path)} owner={holder} held={holder_age:.1f}s",
            severity="warning",
            foreground_request=foreground_request,
        )
        # Preemption: if a foreground caller waited through the explicit queue
        # deadline and the current holder exceeded the first-token SLA, cancel
        # the in-flight future and recycle the worker before stale text can
        # leak into a later turn.
        if foreground_request:
            sla = self._first_token_sla(foreground_request=True)
            if holder_age > sla:
                heartbeat_age = (
                    time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                )
                if heartbeat_age > 30.0:
                    logger.error(
                        "🛑 [MLX] Preempting wedged holder %s (age=%.1fs > sla=%.1fs, no heartbeat for %.1fs). "
                        "Cancelling in-flight future and scheduling worker reboot.",
                        holder,
                        holder_age,
                        sla,
                        heartbeat_age,
                    )
                    self._deferred_reboot_reason = "foreground_preemption_wedged_holder"
                else:
                    logger.warning(
                        "🛡️ [MLX] Holder %s slow (age=%.1fs > sla=%.1fs) but heartbeat fresh (%.1fs ago). "
                        "Cancelling generation and scheduling a clean recycle so stale text cannot bleed into the next turn.",
                        holder,
                        holder_age,
                        sla,
                        heartbeat_age,
                    )
                    self._deferred_reboot_reason = "recoverable_foreground_preemption_slow_holder"
                try:
                    # Compare-and-cancel. Only the generation whose slowness
                    # was actually measured may be cancelled; if the lane has
                    # moved on, the new holder is innocent and preempting it
                    # would be the very fault this is meant to fix.
                    stuck_future = self._current_gen_future
                    current_request_id = str(
                        getattr(self, "_current_request_id", "") or ""
                    )
                    if stuck_future is None or stuck_future is not victim_future:
                        logger.info(
                            "🛡️ [MLX] Holder changed during preemption "
                            "(%s -> %s); leaving the new generation alone.",
                            victim_request_id or "unknown",
                            current_request_id or "none",
                        )
                    elif victim_request_id and current_request_id != victim_request_id:
                        logger.info(
                            "🛡️ [MLX] Request id changed during preemption "
                            "(%s -> %s); leaving the new generation alone.",
                            victim_request_id,
                            current_request_id or "none",
                        )
                    else:
                        _cancel_shared_future(stuck_future)
                except (RuntimeError, AttributeError) as exc:
                    logger.debug(
                        "MLX request preemption future cancel skipped: %s",
                        exc,
                    )
        return False

    def _try_acquire_request_lock(
        self,
        *,
        owner_label: str,
        owner_token: str = "",
    ) -> bool:
        """Acquire the worker lane without waiting.

        Optional background reads use this path because waiting behind model
        work would turn a best-effort observation into queued inference.  The
        ownership fields are updated in the same critical section as every
        regular request, so diagnostics never report an ownerless held lock.
        """

        with self._request_lane_state_guard("try_acquire"):
            if not self._request_lock.acquire(False):
                return False
            self._request_lock_owner_label = str(owner_label or "")
            self._request_lock_owner_token = str(owner_token or "")
            self._request_lock_acquired_at = time.time()
            return True

    def _release_request_lock_if_owned(
        self,
        *,
        owner_label: str,
        owner_token: str,
    ) -> bool:
        """Release only the request lane whose identity the caller owns."""

        with self._request_lane_state_guard("release_if_owned"):
            if not self._request_lock.locked():
                return False
            if self._request_lock_owner_label != str(owner_label or ""):
                return False
            if self._request_lock_owner_token != str(owner_token or ""):
                return False
            self._release_request_lock_locked()
            return True

    def _register_detached_worker_request(
        self,
        request_id: str,
        future: SharedFuture,
        *,
        owner_label: str,
    ) -> bool:
        """Transfer an unfinished optional request from its caller to the listener."""

        with self._request_lane_state_guard("register_detached"):
            if future.done():
                return False
            if not hasattr(self, "_detached_worker_requests"):
                self._detached_worker_requests = {}
            self._detached_worker_requests[str(request_id)] = (
                future,
                str(owner_label or ""),
            )
            return True

    async def _route_terminal_worker_response(
        self,
        request_id: str,
        future: SharedFuture,
        response: dict[str, Any],
    ) -> bool:
        """Deliver one terminal frame and retire listener-owned work."""

        detached_owner = ""
        with self._request_lane_state_guard("route_terminal_response"):
            detached_requests = getattr(self, "_detached_worker_requests", {})
            detached = detached_requests.pop(str(request_id), None)
            if detached is not None and detached[0] is future:
                detached_owner = detached[1]
            delivered = _set_shared_future_result(future, response)

        if not detached_owner:
            return delivered

        await self._finish_generation_ownership(str(request_id), future, None)
        released = self._release_request_lock_if_owned(
            owner_label=detached_owner,
            owner_token=str(request_id),
        )
        logger.info(
            "🔤 [ENCODE] listener retired detached request %s (lane_released=%s).",
            str(request_id)[:12],
            released,
        )
        return True

    def _release_request_lock_if_aborted(self, reason: str) -> None:
        """Release the request lane only when the aborted work owned it.

        An unconditional release let a SECOND request enter while the previous
        owner's critical section was still running — the abort's own damage
        rather than the wedge's. When the lock is held by someone else, the
        lane stays fenced and the holder releases it in its own finally.
        """
        if self._release_detached_request_lock():
            return
        still_held_by = ""
        with self._request_lane_state_guard("force_release"):
            holder = str(getattr(self, "_request_lock_owner_label", "") or "")
            if not self._request_lock.locked():
                return
            aborted_owner = bool(
                self._current_request_started_at > 0.0 or self._active_generations
            )
            if holder and not aborted_owner:
                # Said after the lock, not under it. The file sink JSON-wraps
                # and redacts every record, which is real work on whatever
                # thread calls it, and this one runs on the event loop while
                # holding the lane's metadata lock.
                still_held_by = holder
            else:
                self._release_request_lock_locked()
        if still_held_by:
            logger.warning(
                "🛑 [MLX] Force-abort (%s) left the request lane held by %s — "
                "its own holder must release it.",
                reason,
                still_held_by,
            )

    def _release_detached_request_lock(self) -> bool:
        """Release the lane only when its token names listener-owned work."""

        with self._request_lane_state_guard("release_detached"):
            if not self._request_lock.locked():
                return False
            holder = str(getattr(self, "_request_lock_owner_label", "") or "")
            owner_token = str(getattr(self, "_request_lock_owner_token", "") or "")
            detached = getattr(self, "_detached_worker_requests", {}).get(owner_token)
            if detached is None or detached[1] != holder:
                return False
            self._detached_worker_requests.pop(owner_token, None)
            self._release_request_lock_locked()
            return True

    def _clear_detached_worker_requests(self) -> None:
        with self._request_lane_state_guard("clear_detached"):
            getattr(self, "_detached_worker_requests", {}).clear()

    def _apply_pending_force_abort_reconcile(self) -> None:
        """Finish a forced abort's state reconciliation, under the lock.

        Called by the lifecycle owner. The abort itself killed the worker and
        completed the futures; what it deliberately did not do — clearing the
        process handle, replacing the IPC queues, dropping the listener — is
        done here, where publishing a new process cannot race a teardown.
        """
        reason = self._force_abort_reconcile_pending
        if not reason:
            return
        self._force_abort_reconcile_pending = None
        self._force_abort_lock_failures = 0
        self._pending_generations.clear()
        self._current_gen_future = None
        self._active_generations = 0
        self._warmup_in_flight = False
        self._init_done = False
        self._process = None
        self._last_heartbeat = 0.0
        self._last_progress_at = 0.0
        self._last_token_progress_at = 0.0
        self._process_started_at = 0.0
        self._clear_active_generation_tracking()
        if self._init_future is not None:
            _cancel_shared_future(self._init_future)
        self._init_future = None
        if self._listener_task is not None:
            _cancel_task_threadsafe(self._listener_task)
            self._listener_task = None
        self._cancel_lane_renewal_task()
        self._replace_ipc_queues()
        self._release_request_lock_if_aborted(str(reason))
        self._clear_detached_worker_requests()
        logger.warning(
            "🧹 [MLX] Reconciled deferred force-abort state for %s (%s).",
            os.path.basename(self.model_path),
            reason,
        )

    def _release_request_lock(self) -> None:
        with self._request_lane_state_guard("release"):
            self._release_request_lock_locked()

    @contextlib.contextmanager
    def _request_lane_state_guard(self, site: str = ""):
        """Serialize lane metadata when this is a fully constructed client.

        ``site`` names the caller, and exists because the splat could not.
        Lockdep reports where a lock was taken, and every one of these is
        taken on the ``with`` inside this generator — so 117 live holds on the
        event loop, the longest 88ms against a 50ms limit, all read
        "contextlib.py:137" and named none of the eight blocks that could
        have caused them.
        """

        state_lock = getattr(self, "_request_lock_state_lock", None)
        if state_lock is None:
            yield
            return
        with state_lock, instrument("mlx_request_lane_state"):
            started = time.perf_counter()
            try:
                yield
            finally:
                held_ms = (time.perf_counter() - started) * 1000.0
                if held_ms >= _LANE_STATE_HOLD_WORTH_NAMING_MS and site:
                    logger.info(
                        "🔒 [MLX] lane-state guard held %.0fms by %s.", held_ms, site
                    )

    def _release_request_lock_locked(self) -> None:
        """Release the lane while ``_request_lock_state_lock`` is held."""

        self._request_lock_owner_label = ""
        self._request_lock_owner_token = ""
        self._request_lock_acquired_at = 0.0
        try:
            self._request_lock.release()
        except RuntimeError:
            logger.debug(
                "Loop-agnostic request lock for %s was already released.",
                os.path.basename(self.model_path),
            )

    async def _ensure_listener_task(self) -> None:
        response_queue = self._res_q
        queue_generation = self._response_queue_generation
        if response_queue is None:
            raise RuntimeError("response_listener_queue_unavailable")
        task = self._listener_task
        if task is not None and not task.done():
            # Reusable only if its loop is genuinely serving AND the task is
            # not already being cancelled. Respawn paths cancel the old
            # listener and immediately call this method — treating the
            # still-cancelling task as reusable left the fresh worker with
            # NO response consumer. A stopped-but-unclosed foreign loop is
            # equally dead for our purposes.
            cancelling = task.cancelled() or bool(getattr(task, "cancelling", lambda: 0)())
            loop_serving = False
            try:
                task_loop = task.get_loop()
                loop_serving = (not task_loop.is_closed()) and task_loop.is_running()
            except (RuntimeError, AttributeError) as exc:
                logger.debug("MLX listener task loop unavailable during reuse check: %s", exc)
            if loop_serving and not cancelling:
                if (
                    self._listener_response_queue is response_queue
                    and self._listener_queue_generation == queue_generation
                ):
                    return
            _cancel_task_threadsafe(task)
            # CP126 bd5dea11: cancellation is ASYNCHRONOUS. Creating the
            # replacement immediately left two listeners briefly draining the
            # SAME response queue, so the old one could steal the new worker's
            # init/generation frames. Prove the old listener is gone (bounded —
            # a wedged listener must not block worker recovery forever) and
            # drop the handle before installing a replacement.
            distinct_queue_generation = (
                self._listener_response_queue is not None
                and self._listener_response_queue is not response_queue
                and self._listener_queue_generation != queue_generation
            )
            if task.get_loop() is asyncio.get_running_loop():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except (asyncio.CancelledError, TimeoutError):
                    pass
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    logger.debug("Prior MLX listener ended with %s", type(exc).__name__)
                if not task.done():
                    _record_mlx_degradation(
                        TimeoutError("listener_cancel_unconfirmed"),
                        action=(
                            "retained the prior response listener because its "
                            "termination was not confirmed"
                        ),
                        severity="warning",
                    )
            if not task.done() and not distinct_queue_generation:
                # Two readers on one queue can route a valid response to the
                # wrong lifecycle. Unknown ownership is equally unsafe: a
                # second reader is admitted only when a distinct retired queue
                # generation is positively identified.
                raise RuntimeError("response_listener_retirement_unconfirmed")
            if not task.done():
                # A retired listener may still be unwinding an executor poll on
                # its old immutable queue.  Retain it until completion; the new
                # generation is safe because it owns a different queue object.
                self._retired_listener_tasks.add(task)
                task.add_done_callback(self._retired_listener_tasks.discard)
            self._listener_task = None

        self._listener_response_queue = response_queue
        self._listener_queue_generation = queue_generation
        self._listener_task = get_task_tracker().create_task(
            self._response_listener_loop(response_queue, queue_generation)
        )

    def note_lane_failed(self, reason: str) -> None:
        self._warmup_in_flight = False
        self._set_lane_state("failed", reason)

    def _mark_runtime_unavailable(self, detail: str) -> None:
        reason = f"mlx_runtime_unavailable:{detail}"
        self._warmup_in_flight = False
        self._init_done = False
        self._set_lane_state("failed", reason)

    def _worker_unhealthy(self, stale_after: float | None = None) -> bool:
        if self._process is None or not self._process.is_alive():
            return True
        if not self._init_done:
            return True
        stale_after = float(stale_after or self._stale_after())
        last_progress = max(self._last_heartbeat, self._last_progress_at, self._last_ready_at)
        if last_progress <= 0.0:
            return True
        return bool((time.time() - last_progress) > stale_after)

    def _handshake_age_s(self, now: float | None = None) -> float:
        """Return startup age without allowing status retries to reset it."""

        now = float(time.time() if now is None else now)
        process_started_at = float(getattr(self, "_process_started_at", 0.0) or 0.0)
        anchor = process_started_at or float(
            getattr(self, "_lane_transition_at", now) or now
        )
        return max(0.0, now - anchor)

    def _check_lane_state_staleness(self) -> None:
        """[STABILITY v51] Auto-reset stuck non-terminal lane states.

        If the lane has been in a transient state (warming, recovering,
        handshaking, spawning) for >120s with no progress, force-reset
        to 'cold' so recovery can restart from scratch. This prevents
        the permanent 'CORTEX WARMING' display.
        """
        if self._lane_state not in {"warming", "recovering", "handshaking", "spawning"}:
            return
        now = time.time()
        stuck_duration = (
            self._handshake_age_s(now)
            if self._lane_state in {"spawning", "handshaking"}
            else now - self._lane_transition_at
        )
        # State-aware budget: a heavy-lane spawn/handshake legitimately runs
        # for minutes while 20-40GB of weights load. The old flat 120s reset
        # cancelled LIVE handshakes from a mere status poll, well inside the
        # 300s the handshake itself was granted.
        if self._lane_state in {"spawning", "handshaking"}:
            allowed = max(120.0, float(self._handshake_timeout()) + 30.0)
        elif self._lane_state == "warming":
            allowed = max(120.0, float(self._warmup_timeout()) + 30.0)
        else:
            allowed = 120.0
        if stuck_duration < allowed:
            return
        last_activity = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
            self._last_token_progress_at,
        )
        if last_activity > 0.0 and (now - last_activity) < 30.0:
            return  # Recent activity — state is legitimate

        # Classify before destroying ~20GB of wired weights. The activity gate
        # above is binary — recent or not — which cannot tell a wedged decode
        # loop from a request that merely outlived its budget. Killing is the
        # most expensive recovery this runtime has (cold reload, and
        # historically a second worker stacking beside the first), so it needs
        # a verdict rather than a timer, and the verdict is recorded so the
        # decision is auditable after the fact.
        verdict = self._classify_worker_liveness(now)
        if verdict is None:
            process = self._process
            try:
                directly_alive = bool(process and process.is_alive())
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                directly_alive = None
            if directly_alive is False:
                # No classifier is needed to retire a process that the process
                # handle itself proves is absent. This also prevents an empty
                # stale lane from remaining fenced forever.
                if process is not None:
                    _note_lane_worker_death(self, "lane_state_stale_worker_absent")
                    self._retire_worker_process_handle(process)
                self._process = None
                self._warmup_in_flight = False
                self._set_lane_state("cold")
                return
            # Unknown is not dead. Preserve both the process and its non-cold
            # lane state: killing would discard potentially healthy weights,
            # while declaring the lane cold would permit a second worker to
            # stack beside the first one.
            _record_mlx_degradation(
                RuntimeError(f"stale_lane_worker_liveness_unclassified:{self._lane_state}"),
                action=(
                    "preserved live worker and lane ownership because termination "
                    "could not be justified"
                ),
                severity="error",
            )
            logger.error(
                "[STABILITY] Lane '%s' is stale after %.0fs, but worker liveness "
                "could not be classified; preserving process and ownership.",
                self._lane_state,
                stuck_duration,
            )
            return
        if not verdict.kill_justified:
            if verdict.should_cancel_request:
                # The REQUEST is what is stuck, not the model. Cancel it and
                # leave the weights loaded.
                logger.warning(
                    "🔧 [STABILITY] Lane '%s' stuck %.0fs but worker is %s — "
                    "cancelling the request instead of killing the worker.",
                    self._lane_state, stuck_duration, verdict,
                )
                try:
                    self.soft_cancel_active_generation("lane_stale_worker_stalled")
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    logger.debug("soft cancel during stale-lane reset failed: %s", exc)
            else:
                logger.warning(
                    "🔧 [STABILITY] Lane '%s' stuck %.0fs but worker is %s — "
                    "resetting lane bookkeeping WITHOUT killing the worker.",
                    self._lane_state, stuck_duration, verdict,
                )
            # Reset the lane bookkeeping (which is what was actually stale)
            # and leave the process alone.
            self._warmup_in_flight = False
            self._set_lane_state("cold")
            return

        logger.warning(
            "🔧 [STABILITY] Lane state '%s' stuck for %.0fs with no activity "
            "(%s). Force-resetting to 'cold' for clean recovery.",
            self._lane_state,
            stuck_duration,
            verdict,
        )
        # The reset must not orphan a live worker: declaring the lane cold
        # while the old process survives lets the next spawn stack a second
        # multi-GB worker beside it.
        process = self._process
        if process is not None and process.is_alive():
            _record_mlx_degradation(
                RuntimeError(f"stale_lane_reset_killed_live_worker:{self._lane_state}"),
                action="killed unresponsive worker during stale lane-state reset",
                severity="error",
            )
            _note_lane_worker_death(self, "lane_state_stale_reset")
            termination_proven = self._release_worker_process(
                process,
                reason="lane_state_stale_reset",
            )
            if termination_proven:
                if self._process is process:
                    self._process = None
                else:
                    logger.info(
                        "[MLX] Stale-lane target was replaced during termination; "
                        "preserving replacement worker state."
                    )
                    return
            else:
                self._warmup_in_flight = False
                self._set_lane_state(
                    "recovering",
                    "lane_state_stale_reset_worker_survived",
                )
                return
        self._warmup_in_flight = False
        self._set_lane_state("cold")

    def _classify_worker_liveness(self, now: float | None = None) -> Any:
        """Assemble evidence and classify this lane's worker.

        Bridges the signals this client already tracks into the shared
        vocabulary in core/runtime/worker_liveness.py, so every kill decision
        in the runtime can answer "wedged or busy?" the same way. Returns None
        only if the classifier itself is unavailable, which must never be a
        reason to kill.
        """
        try:
            from core.runtime.worker_liveness import WorkerEvidence, classify_worker
        except ImportError:
            return None

        now = time.time() if now is None else now
        process = self._process

        def _age(stamp: Any) -> float | None:
            try:
                value = float(stamp or 0.0)
            except (TypeError, ValueError):
                return None
            return max(0.0, now - value) if value > 0.0 else None

        # The strongest available proof of output: the last token this lane saw.
        progress_age = _age(getattr(self, "_last_token_progress_at", 0.0))
        if progress_age is None:
            progress_age = _age(getattr(self, "_last_progress_at", 0.0))

        try:
            alive = bool(process.is_alive()) if process is not None else False
        except (RuntimeError, AttributeError, TypeError, ValueError):
            alive = None

        evidence = WorkerEvidence(
            process_alive=alive,
            last_heartbeat_age_s=_age(getattr(self, "_last_heartbeat", 0.0)),
            active_job=bool(getattr(self, "_current_request_id", "") or ""),
            job_age_s=(_age(getattr(self, "_current_request_started_at", 0.0)) or 0.0),
            loop_stalled=bool(getattr(self, "_worker_reported_loop_stall", False)),
            last_progress_age_s=progress_age,
            source=f"mlx_lane:{getattr(self, '_lane_state', 'unknown')}",
        )
        return classify_worker(evidence)

    def _release_worker_process(self, process: Any, *, reason: str) -> bool:
        """Kill, PROVE it exited, and only then stop tracking it.

        Six callers used to discard this result: they killed, set
        ``_process = None`` and replaced the IPC queues in the next breath. A
        process that ignored or delayed termination became an orphan nothing
        was holding a handle to — still on the accelerator, invisible to every
        liveness probe, while a replacement was already being scheduled.

        A survivor stays in ``_surviving_workers`` so it is still countable and
        still reapable. Returns whether exit was proven.
        """
        if process is None:
            return True
        proven = self._kill_and_join_blocking(process)
        if not proven:
            survivors = getattr(self, "_surviving_workers", None)
            if survivors is None:
                survivors = []
                self._surviving_workers = survivors
            if not any(candidate is process for candidate in survivors):
                survivors.append(process)
            logger.error(
                "🧟 [MLX] Worker pid=%s survived kill during %s; keeping the handle "
                "so it stays tracked instead of orphaned.",
                getattr(process, "pid", "?"),
                reason,
            )
        return proven

    def surviving_worker_count(self) -> int:
        """Workers this client killed that did not prove they exited."""
        survivors = getattr(self, "_surviving_workers", None) or []
        alive = []
        for process in survivors:
            if self._worker_exit_is_proven(process):
                self._retire_worker_process_handle(process)
            else:
                alive.append(process)
        self._surviving_workers = alive
        return len(alive)

    @staticmethod
    def _worker_exit_is_proven(process: Any, identity: Any = None) -> bool:
        """Reconcile a process handle without mistaking stale liveness for life."""
        try:
            if getattr(process, "exitcode", None) is not None:
                return True
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            pass
        try:
            if not process.is_alive():
                return True
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            pass
        if identity is not None and getattr(identity, "bound", False):
            try:
                from core.runtime.process_identity import identity_still_current

                return not identity_still_current(identity, process)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError):
                return False
        return False

    def _retire_worker_process_handle(self, process: Any) -> None:
        """Release a proven-dead child handle without leaving a phantom owner."""
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            get_runtime_hygiene().retire_process_handle(process)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError):
            pass
        close = getattr(process, "close", None)
        if not callable(close):
            return
        try:
            close()
        except ValueError:
            # ``Process.close`` is idempotent only by exception.
            return
        except (RuntimeError, AttributeError, TypeError, OSError) as exc:
            _record_mlx_degradation(
                exc,
                action="worker exited but its multiprocessing handle could not be closed",
                severity="warning",
            )

    @staticmethod
    def _join_worker_handle(process: Any, timeout_s: float) -> BaseException | None:
        join = getattr(process, "join", None)
        if not callable(join):
            return AttributeError("worker process handle has no join method")
        try:
            join(timeout=max(0.0, float(timeout_s)))
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            return exc
        return None

    def _kill_and_join_blocking(
        self,
        p: mp.Process,
        *,
        cooperative: bool = False,
        retire_handle: bool = True,
    ) -> bool:
        """Stop and join a worker, proving termination before releasing it.

        The old helper swallowed failures and never re-checked ``is_alive``,
        so callers replaced queues and respawned while the accelerator-owning
        child could still be running.
        """
        if not p:
            return True
        if self._worker_exit_is_proven(p):
            if retire_handle:
                self._retire_worker_process_handle(p)
            return True
        try:
            initially_alive = bool(p.is_alive())
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            _record_mlx_degradation(
                exc,
                action="could not establish worker liveness before shutdown",
                severity="critical",
            )
            return False
        if not initially_alive:
            if retire_handle:
                self._retire_worker_process_handle(p)
            return True
        identity = None
        try:
            from core.runtime.process_identity import capture_identity

            identity = capture_identity(p, label="mlx_model_worker")
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError):
            identity = None

        failures: list[BaseException] = []
        if cooperative and p is self._process and self._req_q is not None:
            try:
                self.soft_cancel_active_generation("worker_shutdown")
                self._req_q.put(None, block=True, timeout=0.5)
            except (queue.Full, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                failures.append(exc)
            join_failure = self._join_worker_handle(p, _WORKER_COOPERATIVE_JOIN_S)
            if join_failure is not None:
                failures.append(join_failure)
            if self._worker_exit_is_proven(p, identity):
                if retire_handle:
                    self._retire_worker_process_handle(p)
                return True

        for action_name, wait_s in (
            ("terminate", _WORKER_TERMINATE_JOIN_S),
            ("kill", _WORKER_KILL_JOIN_S),
            ("kill", _WORKER_KILL_JOIN_S),
        ):
            if self._worker_exit_is_proven(p, identity):
                if retire_handle:
                    self._retire_worker_process_handle(p)
                return True
            action = getattr(p, action_name, None)
            if not callable(action):
                continue
            try:
                action()
            except ProcessLookupError:
                # Another owner won the race. The identity check after join
                # distinguishes a dead PID from a stale multiprocessing handle.
                pass
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                failures.append(exc)
            join_failure = self._join_worker_handle(p, wait_s)
            if join_failure is not None:
                failures.append(join_failure)
            if self._worker_exit_is_proven(p, identity):
                if retire_handle:
                    self._retire_worker_process_handle(p)
                return True

        for failure in failures:
            _record_mlx_degradation(
                failure,
                action="worker kill/join failed before termination could be proven",
                severity="error",
            )
        _record_mlx_degradation(
            RuntimeError(f"worker_survived_kill:pid={getattr(p, 'pid', '?')}"),
            action="worker process survived kill+join; caller retains its handle",
            severity="critical",
        )
        return False

    def _replace_ipc_queues(self, *, maxsize: int = 10) -> None:
        """Replace IPC queues after closing the old semaphores and feeder threads."""
        _safe_close_queue(self._req_q)
        _safe_close_queue(self._res_q)
        self._req_q = self._mp_context.Queue(maxsize=maxsize)
        self._res_q = self._mp_context.Queue(maxsize=maxsize)
        self._response_queue_generation += 1
        try:
            _register_runtime_queue(self._req_q, name="mlx.request_queue")
            _register_runtime_queue(self._res_q, name="mlx.response_queue")
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            self._close_ipc_queues()
            raise
        self._closed = False

    def _close_ipc_queues(self) -> None:
        """Close IPC queues without recreating them during final client shutdown."""
        _safe_close_queue(self._req_q)
        _safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None
        self._response_queue_generation += 1

    async def _generate_batch_response_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Return one task-local batched worker response without global state."""
        if self._req_q is None or self._closed:
            return {}
        # Memory-pressure admission: a 16-candidate 2048-token resident batch
        # is heavy generation. The serial and latent paths refuse under
        # critical pressure — the batch path previously dispatched anyway.
        try:
            snapshot = get_memory_pressure_snapshot()
            if snapshot.refuse_heavy_local_generation:
                _record_mlx_degradation(
                    RuntimeError(snapshot.reason or "critical_memory_pressure"),
                    action="refused batched generation under critical memory pressure",
                    severity="warning",
                )
                return {}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="refused batched generation while the memory probe was unavailable",
                severity="warning",
            )
            return {}
        alive = await self._ensure_worker_alive(request_is_background=True)
        if not alive:
            return {}
        try:
            admitted_n = max(1, min(16, int(n)))
        except (TypeError, ValueError, OverflowError):
            admitted_n = 4
        admitted_max_tokens = min(
            2048,
            _bounded_max_tokens(max_tokens, max_tokens, 512),
        )
        try:
            admitted_temperature = float(temperature)
        except (TypeError, ValueError, OverflowError):
            admitted_temperature = 0.8
        if admitted_temperature != admitted_temperature or not (
            float("-inf") < admitted_temperature < float("inf")
        ):
            admitted_temperature = 0.8
        admitted_temperature = max(0.0, min(2.0, admitted_temperature))
        try:
            admitted_timeout = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            admitted_timeout = 180.0
        if not math.isfinite(admitted_timeout):
            # Infinity previously created an UNBOUNDED wait on the future.
            admitted_timeout = 180.0
        requested_timeout = admitted_timeout
        admitted_timeout = min(600.0, max(10.0, admitted_timeout))
        if admitted_timeout != requested_timeout:
            # A batch decode cannot finish in three seconds, so the floor
            # stands — but the caller must not believe its budget was honoured
            # when the wait it actually gets is longer. Widening in silence is
            # how a bounded caller ends up unbounded.
            _record_mlx_degradation(
                ValueError(
                    f"batch timeout {requested_timeout:.1f}s outside the admissible "
                    f"range; using {admitted_timeout:.1f}s"
                ),
                action="admitted a batch decode on a budget the caller did not request",
                severity="warning",
            )
        req_id = uuid.uuid4().hex
        req = {
            "id": req_id,
            "action": "generate_batch",
            "prompt": str(prompt or ""),
            "n": admitted_n,
            "max_tokens": admitted_max_tokens,
            "temperature": admitted_temperature,
        }
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        # Register the batch decode as an ACTIVE generation for its duration.
        #
        # It used to queue the command and register only a pending future, so
        # every "is this lane busy?" check read it as idle — and those checks
        # guard consequential actions. maybe_unload_idle, the adapter swap and
        # the idle scavenger all gate on _active_generations > 0, so a batch
        # decode of n candidates on the resident 32B could have its weights
        # unloaded or its adapter swapped out from under it, mid-decode,
        # because nothing said it was running.
        #
        # Marking it busy is also what makes the durable lane non-preemptible
        # while the decode holds it, which is the property the ownership
        # bookkeeping exists to provide.
        self._active_generations += 1
        timed_out = False
        try:
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(req, principal="mlx_client.health_probe"),
                True,
                2.0,
            )
            res = await _await_shared_future(fut, timeout_s=admitted_timeout)
        except asyncio.CancelledError:
            # The caller is gone; the WORKER is not. Without this the batch
            # kept decoding n candidates on the resident model for nobody,
            # holding the lane and its memory until it finished on its own.
            timed_out = True
            raise
        except (TimeoutError, BrokenPipeError, OSError, queue.Full) as exc:
            # queue.Full included: queue saturation is expected load
            # contention and belongs inside the documented empty-response
            # fallback envelope, not raised to the caller.
            timed_out = isinstance(exc, TimeoutError)
            _record_mlx_degradation(
                exc,
                action="returned empty batch after batched generation failed; caller falls back to serial",
                severity="warning",
            )
            return {}
        finally:
            # ALWAYS unregister — caller cancellation previously left the
            # worker command live and the future registered indefinitely.
            self._pending_generations.pop(req_id, None)
            # Release the busy marker on every path, including cancellation.
            # A leaked increment is worse than never having taken one: the
            # lane would look permanently busy and the idle unload, adapter
            # swap and scavenger would all be blocked forever.
            self._active_generations = max(0, self._active_generations - 1)
            if timed_out:
                # The queued decode continues invisibly after a timeout;
                # ask the worker to yield instead of burning the lane.
                with contextlib.suppress(Exception):
                    self.soft_cancel_active_generation(reason=f"batch_timeout:{req_id[:12]}")
        if not res or res.get("status") != "ok":
            return {}
        raw_texts_value = res.get("texts")
        # A malformed worker payload (a plain string iterates as characters)
        # must not fabricate hundreds of one-character candidates.
        if not isinstance(raw_texts_value, (list, tuple)):
            _record_mlx_degradation(
                TypeError(f"batch texts payload was {type(raw_texts_value).__name__}"),
                action="dropped malformed batch response payload",
            )
            return {}
        # Bounded per candidate. The count was capped; the SIZE was not, so a
        # malformed or hostile worker could hand the parent 16 arbitrarily
        # large strings and the parent would hold every one of them.
        raw_texts = [
            str(t or "")[:_BATCH_CANDIDATE_MAX_CHARS] for t in raw_texts_value
        ][:admitted_n]
        raw_candidate_tokens = list(res.get("tokens_used_by_candidate") or [])
        texts: list[str] = []
        tokens_used_by_candidate: list[int] = []
        for index, text in enumerate(raw_texts):
            if not text.strip():
                continue
            texts.append(text)
            try:
                candidate_tokens = max(0, int(raw_candidate_tokens[index] or 0))
            except (IndexError, TypeError, ValueError, OverflowError):
                candidate_tokens = 0
            tokens_used_by_candidate.append(candidate_tokens)
        try:
            tokens_used = max(0, int(res.get("tokens_used") or 0))
        except (TypeError, ValueError, OverflowError):
            tokens_used = 0
        # The aggregate and the per-candidate totals are two claims about the
        # same decode. When they disagree the receipt is not a measurement of
        # anything, so report the sum we can actually account for and say the
        # worker's total was inconsistent rather than passing it on.
        candidate_total = sum(tokens_used_by_candidate)
        tokens_used_consistent = (not tokens_used_by_candidate) or (
            tokens_used >= candidate_total
        )
        if not tokens_used_consistent:
            _record_mlx_degradation(
                ValueError(
                    f"batch tokens_used={tokens_used} below candidate sum={candidate_total}"
                ),
                action="reported the accountable candidate total after the worker totals disagreed",
                severity="warning",
            )
            tokens_used = candidate_total

        # Batch decoding runs the same transformer hooks, so it fills the Φ
        # residual ring too. Drained here rather than left for the next
        # foreground turn: a verifier sweep is hundreds of states, and several
        # between turns would wrap the ring and throw away transitions the
        # complex could have used.
        #
        # The LATENT readouts are deliberately NOT drained here. Those inject
        # into the substrate, and verifier sampling is not Aura having a
        # thought — feeding it back would make her mood a function of how many
        # candidates a best-of-N search happened to decode.
        self._drain_phi_residual_ring()

        return {
            "texts": texts,
            "tokens_used_consistent": tokens_used_consistent,
            "request_id": req_id,
            "requested_timeout_s": requested_timeout,
            "admitted_timeout_s": admitted_timeout,
            "max_tokens": admitted_max_tokens,
            "temperature": admitted_temperature,
            "tokens_used": tokens_used,
            "tokens_used_by_candidate": tokens_used_by_candidate,
        }

    async def generate_batch_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> list[str]:
        """Decode raw verifier candidates in one batched worker pass."""

        response = await self._generate_batch_response_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        return list(response.get("texts") or [])

    async def generate_batch_with_metadata_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Return batched candidates with one truthful shared decode receipt."""

        response = await self._generate_batch_response_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        texts = list(response.get("texts") or [])
        if not texts:
            return {}
        model_name = os.path.basename(str(self.model_path or "")) or "unknown"
        candidate_tokens = list(response.get("tokens_used_by_candidate") or [])
        # CP126 536f8e0d: this said provider_verified=True because the worker
        # said status=ok, and named the model from a PATH BASENAME. Neither
        # proves which process produced the candidates or which weights it had
        # loaded — a renamed directory changed the reported model, and a
        # response from a recycled worker was attributed to the current one.
        #
        # Verification now means: we hold the worker's attested identity, and
        # this response arrived while that worker generation was serving.
        identity = self.get_worker_identity_snapshot()
        worker_boot_id = str(identity.get("worker_boot_id") or "")
        worker_pid = identity.get("worker_pid")
        provider_verified = bool(
            worker_boot_id and isinstance(worker_pid, int) and worker_pid > 0
        )
        return {
            "texts": texts,
            "generation_metadata": {
                "endpoint": f"MLX-BATCH:{model_name}",
                "provider": "mlx",
                "model": model_name,
                "model_basis": "path_basename",
                "is_local": True,
                "provider_verified": provider_verified,
                "provider_verification_basis": (
                    "attested_worker_identity" if provider_verified else "unattested"
                ),
                "worker_boot_id": worker_boot_id,
                "worker_pid": worker_pid if isinstance(worker_pid, int) else None,
                "worker_generation": int(getattr(self, "_worker_generation", 0) or 0),
                "batch_tokens_used_consistent": bool(
                    response.get("tokens_used_consistent", True)
                ),
                "batch_request_id": response.get("request_id"),
                "surface_control_receipt": {
                    "enabled": False,
                    "applied": False,
                    "generation_required": True,
                    "application_status": "raw_batch_requires_parent_verification",
                    "clean_user_surface_contract": False,
                    "surface_quality_gate_enabled": False,
                    "surface_quality_gate_passed": False,
                    "generation_max_tokens": response.get("max_tokens"),
                    "batch_generated_tokens_total": response.get("tokens_used"),
                    "batch_candidate_count": len(texts),
                    "source": "mlx_batch_worker",
                },
            },
            "candidate_generation_metadata": [
                {
                    "generated_tokens": (
                        max(0, int(candidate_tokens[index] or 0))
                        if index < len(candidate_tokens)
                        else 0
                    )
                }
                for index in range(len(texts))
            ],
        }

    async def ingest_nonparametric_async(
        self,
        *,
        max_pairs: int = 1,
        scan_limit: int = 16,
        max_positions: int = 96,
        max_sequence_tokens: int = 192,
        timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        """Run bounded trusted-memory ingestion on a resident worker only.

        This maintenance command never spawns or loads a model.  It shares the
        worker request lock, advertises active ownership to the lane controller,
        and cooperatively cancels before recycling a worker that exceeds its
        deadline.
        """

        base = {
            "schema": "aura.nonparametric_ingest.worker.v1",
            "spawned_worker": False,
        }
        if self._closed:
            return {**base, "status": "skipped_client_closed"}
        if _foreground_owner_active():
            return {**base, "status": "skipped_foreground_active"}
        if self._active_generations > 0 or self._warmup_in_flight:
            return {**base, "status": "skipped_worker_busy"}
        if (
            self._req_q is None
            or not self._init_done
            or self._process is None
            or not self._process.is_alive()
        ):
            return {**base, "status": "skipped_worker_not_resident"}
        try:
            if get_memory_pressure_snapshot().refuse_heavy_local_generation:
                return {**base, "status": "skipped_memory_pressure"}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return {**base, "status": "skipped_memory_unobservable"}

        try:
            bounded_timeout_s = max(2.0, min(35.0, float(timeout_s)))
            bounded_max_pairs = max(1, min(4, int(max_pairs)))
            bounded_scan_limit = max(1, min(64, int(scan_limit)))
            bounded_max_positions = max(1, min(256, int(max_positions)))
            bounded_max_sequence_tokens = max(
                8,
                min(512, int(max_sequence_tokens)),
            )
        except (TypeError, ValueError, OverflowError):
            return {**base, "status": "invalid_maintenance_budget"}
        deadline = get_deadline(bounded_timeout_s)
        acquired = await self._acquire_request_lock(
            owner_label="reasoning_nonparametric_ingest",
            deadline=deadline,
            foreground_request=False,
        )
        if not acquired:
            return {**base, "status": "skipped_request_lane_busy"}
        # CP126 9246b647: foreground ownership was tested at the top, then
        # memory observation, budget normalisation and the request-lock wait
        # all ran — every one of them an await. A person's turn could take
        # foreground ownership anywhere in that window and maintenance would
        # still win the lane and start a bounded worker job the user's request
        # then had to cancel. Re-test now that the lane is actually held.
        if _foreground_owner_active():
            self._release_request_lock()
            return {**base, "status": "skipped_foreground_active_after_lane"}

        future: SharedFuture | None = None
        request_id = ""
        deferred_reboot = ""
        try:
            if (
                self._req_q is None
                or not self._init_done
                or self._process is None
                or not self._process.is_alive()
            ):
                return {**base, "status": "skipped_worker_not_resident"}
            if not await self._set_durable_lane_preemptible(False):
                return {**base, "status": "skipped_lane_fence_lost"}

            request_id = uuid.uuid4().hex
            self._job_seq_counter += 1
            request_seq = self._job_seq_counter
            request = {
                "id": request_id,
                "seq": request_seq,
                "action": "nonparametric_ingest",
                "max_pairs": bounded_max_pairs,
                "scan_limit": bounded_scan_limit,
                "max_positions": bounded_max_positions,
                "max_sequence_tokens": bounded_max_sequence_tokens,
                "deadline_s": max(1.0, bounded_timeout_s - 2.0),
            }
            future = _new_shared_future()
            self._pending_generations[request_id] = future
            self._current_gen_future = future
            self._active_generations += 1
            self._active_generation_started_at = time.time()
            self._mark_generation_started(
                request_id,
                first_token_hard_ceiling_s=bounded_timeout_s,
                request_seq=request_seq,
            )
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(request, principal="mlx_client.structured_request"),
                True,
                2.0,
            )
            try:
                response = await _await_shared_future(
                    future,
                    timeout_s=bounded_timeout_s,
                )
            except TimeoutError:
                self.soft_cancel_active_generation(reason="nonparametric_ingest_deadline")
                try:
                    response = await _await_shared_future(future, timeout_s=3.0)
                except TimeoutError:
                    deferred_reboot = "nonparametric_ingest_deadline"
                    return {**base, "status": "timed_out_worker_recycled"}
            if not isinstance(response, dict):
                return {**base, "status": "invalid_worker_response"}
            if response.get("status") != "ok":
                return {
                    **base,
                    "status": "worker_error",
                    "reason": str(response.get("message") or "unknown"),
                }
            # CP126 8264628d: these were int(...) straight off the wire, so a
            # malformed value RAISED out of a maintenance call, and negative,
            # absurd or mutually inconsistent counts were accepted as
            # measurements — more pairs ingested than scanned, more scanned
            # than the scan budget allowed. A counter that cannot be true is
            # not a smaller number; it is not a measurement at all.
            counters, counter_faults = _bounded_maintenance_counters(
                response,
                max_pairs=bounded_max_pairs,
                scan_limit=bounded_scan_limit,
                max_positions=bounded_max_positions,
            )
            if counter_faults:
                _record_mlx_degradation(
                    ValueError(f"maintenance counters out of contract: {counter_faults}"),
                    action="reported maintenance counters as unmeasured after the worker's disagreed",
                    severity="warning",
                )
            return {
                **base,
                "status": str(response.get("state") or "complete"),
                **counters,
                "counter_faults": counter_faults,
            }
        except asyncio.CancelledError:
            if future is not None:
                self.soft_cancel_active_generation(reason="nonparametric_ingest_caller_cancelled")
                try:
                    await asyncio.shield(_await_shared_future(future, timeout_s=3.0))
                except (asyncio.CancelledError, TimeoutError):
                    deferred_reboot = "nonparametric_ingest_cancel_drain_failed"
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            _record_mlx_degradation(
                exc,
                action=("kept non-parametric maintenance bounded after resident worker IPC failed"),
                severity="warning",
            )
            return {**base, "status": f"ipc_failed:{type(exc).__name__}"}
        finally:
            try:
                if future is not None:
                    await asyncio.shield(
                        self._finish_generation_ownership(
                            request_id,
                            future,
                            None,
                        )
                    )
            finally:
                self._release_request_lock()
                if deferred_reboot:
                    await self.reboot_worker(
                        reason=deferred_reboot,
                        mark_failed=False,
                    )

    async def set_expert_adapter(
        self, adapter_path: str | None, *, timeout_s: float = 90.0
    ) -> dict[str, Any]:
        """Attach/detach a domain-specialist LoRA on the RESIDENT worker model.

        The expert-LoRA library's live seam: the adapter (~40MB) is wrapped
        onto the loaded model inside the worker — no model reload, seconds not
        minutes. ``None``/"" detaches. Refuses while a generation is active
        (weights must never change mid-decode) and never spawns a worker just
        to attach — an adapter is worthless without a resident model.
        """
        path = str(adapter_path or "").strip()
        if self._closed:
            return {"ok": False, "reason": "client_closed"}
        if self._req_q is None or not (
            self._process and self._process.is_alive() and self._init_done
        ):
            return {"ok": False, "reason": "worker_not_ready"}
        if int(getattr(self, "_active_generations", 0) or 0) > 0 or self._warmup_in_flight:
            return {"ok": False, "reason": "generation_active"}
        adapter_verdict: AdapterVerdict | None = None
        if path:
            adapter_verdict = await asyncio.to_thread(
                _validate_adapter_artifact,
                Path(path).expanduser(),  # noqa: ASYNC240 - executed in to_thread
                # The resident checkpoint's training-pipeline fingerprint is
                # not something this client measures — the worker identity
                # carries a source sha and a model path, not the digest an
                # adapter's base_checkpoint_fingerprint is computed against.
                # So compatibility comes back "declared_unverified" and says
                # so, rather than passing a key that is always absent and
                # calling the resulting silence a check.
                expected_base_fingerprint="",
            )
            adapter_exists = adapter_verdict.ok
        else:
            adapter_exists = True
        if not adapter_exists:
            reason = (
                adapter_verdict.reason
                if adapter_verdict is not None and adapter_verdict.reason
                else f"adapter_missing:{path}"
            )
            logger.warning("🧬 [MLX] Refused adapter attach for %s: %s", path, reason)
            return {
                "ok": False,
                "reason": reason,
                **(adapter_verdict.as_receipt() if adapter_verdict is not None else {}),
            }

        # Re-check after the filesystem await. The check above is a
        # time-of-check/time-of-use race: it reads the counter, then this
        # coroutine yields for a directory stat, and a generation can begin in
        # that window. The swap would then be dispatched against a worker that
        # is mid-decode, which is exactly what the exclusion above exists to
        # prevent — and the caller would be told the swap was cleanly excluded.
        if int(getattr(self, "_active_generations", 0) or 0) > 0 or self._warmup_in_flight:
            return {"ok": False, "reason": "generation_active_after_stat"}

        req_id = uuid.uuid4().hex
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        try:
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(
                    {"id": req_id, "action": "set_expert_adapter", "path": path},
                    principal="mlx_client.expert_adapter",
                ),
                True,
                2.0,
            )
            res = await _await_shared_future(fut, timeout_s=max(10.0, float(timeout_s)))
        except asyncio.CancelledError:
            # Identical hazard to the timeout below, and it had no handler at
            # all: the swap command is already queued, so cancelling the
            # CALLER does not stop the worker from attaching the adapter. The
            # future must be unregistered, the worker asked to abandon it, and
            # the resident adapter state marked unknown — otherwise the next
            # reader believes weights that may already have changed.
            self._pending_generations.pop(req_id, None)
            with contextlib.suppress(Exception):
                self.soft_cancel_active_generation(
                    reason=f"adapter_swap_cancelled:{req_id[:12]}"
                )
            self._expert_adapter_state_unknown = True
            _record_mlx_degradation(
                RuntimeError("expert adapter swap cancelled with the command queued"),
                action=(
                    "resident adapter state is UNKNOWN after caller cancellation; "
                    "re-read worker identity before trusting it"
                ),
                severity="error",
            )
            raise
        except (TimeoutError, BrokenPipeError, OSError) as exc:
            self._pending_generations.pop(req_id, None)
            # Do NOT claim the model was left unchanged. The command is already
            # on the worker's queue; dropping our future only stops US from
            # hearing about it. The adapter can still attach afterwards, and
            # the previous receipt asserted the opposite — a false statement
            # about which weights are resident, which is the one thing an
            # adapter receipt exists to get right.
            #
            # Ask the worker to abandon it, then report the state as UNKNOWN
            # rather than unchanged. An unknown adapter is recoverable by
            # re-reading identity; a wrongly-asserted one is not.
            with contextlib.suppress(Exception):
                self.soft_cancel_active_generation(
                    reason=f"adapter_swap_timeout:{req_id[:12]}"
                )
            self._expert_adapter_state_unknown = True
            _record_mlx_degradation(
                exc,
                action=(
                    "expert adapter swap timed out with the command already queued; "
                    "resident adapter state is UNKNOWN until identity is re-read"
                ),
                severity="error",
            )
            return {
                "ok": False,
                "reason": f"swap_timeout:{type(exc).__name__}",
                "resident_adapter_state": "unknown",
            }

        if res and res.get("status") == "ok":
            # A completed swap that reports identity resolves the uncertainty
            # a previous timeout may have left behind.
            self._expert_adapter_state_unknown = False
            raw_worker_identity = res.get("worker_identity")
            try:
                transitioned_identity = self._accept_worker_identity_transition(
                    raw_worker_identity
                )
            except (TypeError, ValueError) as exc:
                _record_mlx_degradation(
                    exc,
                    action=(
                        "recycled resident worker after expert adapter swap "
                        "returned an unprovable identity transition"
                    ),
                    severity="critical",
                )
                await self.reboot_worker(
                    reason="expert_adapter_identity_transition_unproven",
                    mark_failed=False,
                )
                return {
                    "ok": False,
                    "reason": f"identity_transition_unproven:{exc}",
                }
            self._worker_identity = transitioned_identity
            self._expert_adapter_path = str(res.get("resident") or "") or None
            return {
                "ok": True,
                "resident": self._expert_adapter_path,
                "wrapped_layers": int(res.get("wrapped_layers") or 0),
                "detached_layers": int(res.get("detached_layers") or 0),
            }
        if res and res.get("requires_worker_recycle") is True:
            await self.reboot_worker(
                reason="expert_adapter_swap_identity_unrecovered",
                mark_failed=False,
            )
        return {
            "ok": False,
            "reason": str((res or {}).get("message") or "swap_failed"),
        }

    def _accept_worker_identity_transition(
        self,
        raw_identity: Any,
    ) -> dict[str, Any]:
        """Validate and re-attest the sole allowed live identity transition.

        A hot expert-adapter swap may change only the measured adapter list and
        its digest. Model, tokenizer, quantization, process, source, steering,
        and capture key must remain exactly the initialized worker's values.
        """

        from core.brain.llm.latent_cortex.runtime_identity import (
            worker_identity_errors,
        )

        if not isinstance(raw_identity, Mapping):
            raise ValueError("expert adapter response omitted worker identity")
        errors = worker_identity_errors(raw_identity)
        current = getattr(self, "_worker_identity", {})
        if not isinstance(current, Mapping) or not current:
            errors.append("parent_worker_identity_unavailable")
        immutable_fields = (
            "schema",
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_model_parameter_count",
            "worker_model_stored_parameter_element_count",
            "worker_model_parameter_count_basis",
            "worker_source_sha256",
            "worker_affective_steering_active",
            "worker_affective_steering_alpha",
            "worker_action_capture_identity",
            "worker_tokenizer",
            "worker_runtime_tokenizer",
            "worker_quantization",
            "worker_stack_identity_gaps",
        )
        if isinstance(current, Mapping):
            errors.extend(
                f"{field}_changed_during_adapter_swap"
                for field in immutable_fields
                if raw_identity.get(field) != current.get(field)
            )
        if errors:
            raise ValueError(",".join(sorted(set(errors))))
        attested = self._attest_worker_capture_origin(dict(raw_identity))
        attested_errors = worker_identity_errors(attested)
        if attested_errors:
            raise ValueError(
                "reattested_worker_identity_invalid:"
                + ",".join(sorted(set(attested_errors)))
            )
        return attested

    @property
    def expert_adapter_resident(self) -> str | None:
        return getattr(self, "_expert_adapter_path", None)

    def _init_receipt_errors(self, res: dict[str, Any]) -> list[str]:
        """Every reason this init receipt must NOT be trusted as READY.

        Checks the exact model/worker identity and the recurrent-depth
        invariants the lane declares it needs. Returns an empty list only when
        the receipt positively establishes both.
        """
        errors: list[str] = []

        try:
            from core.brain.llm.token_budget_evidence import calibration_batch_errors

            errors.extend(
                calibration_batch_errors(res.get("token_budget_calibration"))
            )
        except ImportError as exc:
            _record_mlx_degradation(
                exc,
                action="token-budget calibration validator unavailable during handshake",
                severity="error",
            )
            errors.append("token_budget_calibration_validator_unavailable")

        identity = res.get("worker_identity")
        try:
            from core.brain.llm.latent_cortex.runtime_identity import (
                worker_identity_errors,
            )

            errors.extend(worker_identity_errors(identity))
        except ImportError as exc:
            # No validator means no proof of identity. Absence of a check is
            # not a passed check.
            _record_mlx_degradation(
                exc,
                action="worker identity validator unavailable during handshake",
                severity="error",
            )
            errors.append("worker_identity_validator_unavailable")

        # The worker must be serving the model THIS client asked for.
        if isinstance(identity, dict):
            reported_path = str(identity.get("worker_model_path") or "")
            if reported_path and _real_model_path(reported_path) != _real_model_path(
                self.model_path
            ):
                errors.append("worker_model_path_mismatch")

        # Recurrence: if this lane requires depth, the receipt must prove it.
        required_loops = _expected_recurrent_loops_from_model_path(self.model_path)
        _note_recurrent_depth_basis_disagreement(
            self.model_path, getattr(self, "_expert_adapter_path", None), required_loops
        )
        recurrent_status = res.get("recurrent_depth")
        if required_loops > 1:
            if not isinstance(recurrent_status, dict):
                errors.append("missing_recurrent_depth_receipt")
            else:
                if not bool(recurrent_status.get("active")):
                    errors.append("recurrent_depth_inactive")
                reported_loops = recurrent_status.get("loops")
                if isinstance(reported_loops, int) and reported_loops != required_loops:
                    errors.append(f"recurrent_depth_mismatch:{reported_loops}!={required_loops}")

        # The adapter-activation receipt: whether the recurrent adapter this
        # worker was supposed to load actually loaded. The worker states it in
        # its signed identity; the handshake must also carry it as a separate
        # top-level receipt, and the two must agree. Without this the client
        # takes the worker's word for what weights it is running — which is
        # exactly how a trained adapter that silently failed to attach reads
        # as a live one.
        if isinstance(identity, dict):
            declared_activation = identity.get("worker_recurrent_adapter_activation")
            if isinstance(declared_activation, dict):
                reported_activation = res.get("recurrent_adapter_activation")
                if not isinstance(reported_activation, dict):
                    errors.append("missing_recurrent_adapter_activation_receipt")
                elif reported_activation != declared_activation:
                    errors.append("recurrent_adapter_activation_receipt_mismatch")

        # Optional shadow tissue still requires an explicit inactive receipt.
        # Otherwise a configured load failure and intentional absence are
        # indistinguishable. This pure-data validator cannot initialize MLX in
        # the parent process.
        try:
            from core.brain.llm.unified_recurrent_shadow_contract import (
                shadow_load_receipt_errors,
            )

            errors.extend(
                shadow_load_receipt_errors(res.get("unified_recurrent_shadow"))
            )
        except ImportError as exc:
            _record_mlx_degradation(
                exc,
                action="unified recurrent shadow receipt validator unavailable",
                severity="error",
            )
            errors.append("unified_recurrent_shadow_validator_unavailable")
        try:
            from core.brain.llm.unified_recurrent_qualified_activation import (
                activation_matches_shadow_receipt,
                qualified_activation_load_receipt_errors,
            )

            qualified_receipt = res.get(
                "unified_recurrent_qualified_activation"
            )
            errors.extend(
                qualified_activation_load_receipt_errors(qualified_receipt)
            )
            if (
                isinstance(qualified_receipt, Mapping)
                and qualified_receipt.get("loaded") is True
                and not activation_matches_shadow_receipt(
                    qualified_receipt.get("activation", {}),
                    res.get("unified_recurrent_shadow", {}),
                )
            ):
                errors.append("qualified_activation_shadow_receipt_differs")
        except ImportError as exc:
            _record_mlx_degradation(
                exc,
                action="qualified recurrent activation validator unavailable",
                severity="error",
            )
            errors.append("qualified_activation_validator_unavailable")
        return errors

    def _attest_worker_capture_origin(
        self,
        worker_identity: Mapping[str, Any],
        *,
        attested_at_unix: int | None = None,
    ) -> dict[str, Any]:
        """Bind the worker's boot key to this parent-owned spawn authority."""

        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_origin_binding,
            validate_worker_capture_origin_binding,
        )

        authority = self._worker_capture_launch_authority
        process = self._process
        expected_pid = getattr(process, "pid", None)
        if authority is None or type(expected_pid) is not int or expected_pid <= 0:
            raise RuntimeError("worker_capture_launch_authority_unavailable")
        capture_identity = worker_identity.get("worker_action_capture_identity")
        bootstrap_binding = getattr(self, "_worker_capture_origin_binding", {})
        if isinstance(bootstrap_binding, Mapping) and bootstrap_binding:
            validated = validate_worker_capture_origin_binding(
                bootstrap_binding,
                expected_supervisor_public_key=authority.private_key.public_key(),
            )
            if validated.get("worker_identity") != capture_identity:
                raise ValueError("worker_capture_bootstrap_ready_identity_mismatch")
            if validated["worker_identity"].get("worker_pid") != expected_pid:
                raise ValueError("worker_capture_bootstrap_process_mismatch")
            return {
                **dict(worker_identity),
                "worker_action_capture_origin_binding": copy.deepcopy(validated),
            }
        binding = build_worker_capture_origin_binding(
            authority,
            capture_identity,
            attested_at_unix=(
                int(time.time()) if attested_at_unix is None else attested_at_unix
            ),
            expected_worker_pid=expected_pid,
        )
        return {
            **dict(worker_identity),
            "worker_action_capture_origin_binding": binding,
        }

    def _accept_worker_capture_bootstrap(
        self,
        worker_capture_identity: Mapping[str, Any],
        *,
        attested_at_unix: int | None = None,
    ) -> dict[str, Any]:
        """Attest the child's capture key while its launch challenge is live."""

        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_origin_binding,
        )

        authority = self._worker_capture_launch_authority
        process = self._process
        expected_pid = getattr(process, "pid", None)
        if authority is None or type(expected_pid) is not int or expected_pid <= 0:
            raise RuntimeError("worker_capture_launch_authority_unavailable")
        existing = getattr(self, "_worker_capture_origin_binding", {})
        if isinstance(existing, Mapping) and existing:
            if existing.get("worker_identity") != worker_capture_identity:
                raise ValueError("worker_capture_bootstrap_identity_changed")
            return copy.deepcopy(dict(existing))
        binding = build_worker_capture_origin_binding(
            authority,
            worker_capture_identity,
            attested_at_unix=(
                int(time.time()) if attested_at_unix is None else attested_at_unix
            ),
            expected_worker_pid=expected_pid,
        )
        self._worker_capture_origin_binding = copy.deepcopy(binding)
        return copy.deepcopy(binding)

    def get_worker_capture_supervisor_public_key(self) -> bytes:
        """Return the parent key expected by independent capture verification."""

        import base64
        import binascii

        identity = self.get_worker_identity_snapshot()
        binding = identity.get("worker_action_capture_origin_binding")
        if not isinstance(binding, Mapping):
            return b""
        challenge = binding.get("launch_challenge")
        if not isinstance(challenge, Mapping):
            return b""
        try:
            raw = base64.b64decode(
                challenge.get("supervisor_public_key_b64"),
                validate=True,
            )
        except (binascii.Error, TypeError, ValueError):
            return b""
        return raw if len(raw) == 32 else b""

    def get_worker_identity_snapshot(self) -> dict[str, Any]:
        """Return immutable identity evidence for resident-scale policy decisions.

        CP126 375fc058: this promised immutability and returned ``dict(...)``,
        a SHALLOW copy. Every nested dict and list stayed shared with the
        client's authoritative record, so a consumer holding a "snapshot"
        could mutate the identity that later policy, admission and proof
        decisions read — and the evidence would still look like evidence.
        """
        identity = getattr(self, "_worker_identity", None)
        if not isinstance(identity, dict):
            return {}
        return copy.deepcopy(identity)

    def get_model_lane_ownership_snapshot(self) -> dict[str, Any]:
        """Return a hash-bound receipt for the worker this client owns now.

        A consumer that records resident neural evidence needs more than a
        model path.  It must bind the observation to the durable lane fence,
        the parent process, and the exact live child.  Empty output means that
        one of those identities is not established; callers must not infer
        ownership from a partially populated client.
        """

        owner_id, fencing_token, terminal_receipt_id = (
            self._durable_model_lane_owner_snapshot()
        )
        identity = self.get_worker_identity_snapshot()
        process = getattr(self, "_process", None)
        process_pid = getattr(process, "pid", None)
        worker_pid = identity.get("worker_pid")
        worker_boot_id = identity.get("worker_boot_id")
        worker_model_path = identity.get("worker_model_path")
        try:
            process_alive = bool(process is not None and process.is_alive())
        except (AssertionError, OSError, ValueError):
            process_alive = False
        if (
            not owner_id
            or fencing_token <= 0
            or not terminal_receipt_id
            or not process_alive
            or type(process_pid) is not int
            or process_pid <= 0
            or type(worker_pid) is not int
            or worker_pid != process_pid
            or not isinstance(worker_boot_id, str)
            or not worker_boot_id
            or not isinstance(worker_model_path, str)
            or os.path.realpath(worker_model_path) != os.path.realpath(self.model_path)
        ):
            return {}
        body = {
            "schema": "aura.mlx_model_lane_ownership.v1",
            "exclusive": True,
            "owner_id": owner_id,
            "fencing_token": fencing_token,
            "terminal_receipt_id": terminal_receipt_id,
            "model_path": os.path.realpath(self.model_path),
            "campaign_pid": os.getpid(),
            "worker_pid": worker_pid,
            "worker_boot_id": worker_boot_id,
        }
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return {
            **body,
            "receipt_sha256": hashlib.sha256(canonical).hexdigest(),
        }

    def _attest_mycelial_worker(self, init_receipt: Mapping[str, Any]) -> None:
        """Publish the accepted worker identity to Mycelium after READY validation."""
        identity = self.get_worker_identity_snapshot()
        boot_id = str(identity.get("worker_boot_id") or "").strip()
        worker_pid = identity.get("worker_pid")
        device = str(init_receipt.get("device") or "").strip().lower()
        if not boot_id or not isinstance(worker_pid, int) or worker_pid <= 0 or not device:
            raise ValueError("validated worker receipt lacks root attestation identity")
        from core.container import ServiceContainer

        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium is None or not hasattr(mycelium, "attest_neural_root"):
            return
        worker_target = f"{boot_id}:{worker_pid}"
        hardware_target = f"mlx:{device}"
        shared_evidence = {
            "worker_boot_id": boot_id,
            "worker_pid": worker_pid,
            "worker_model_path": str(identity.get("worker_model_path") or self.model_path),
            "worker_device": device,
            "worker_source_sha256": str(identity.get("worker_source_sha256") or ""),
        }
        mycelium.attest_neural_root(
            "llm",
            root_kind="worker",
            target_id=worker_target,
            owner_generation=boot_id,
            evidence=shared_evidence,
            liveness_contract="heartbeat",
            stale_after_s=10.0,
        )
        mycelium.attest_neural_root(
            f"worker:{worker_target}",
            root_kind="hardware",
            target_id=hardware_target,
            owner_generation=boot_id,
            evidence=shared_evidence,
            liveness_contract="heartbeat",
            stale_after_s=10.0,
        )
        self._mycelial_root_refs = [
            {
                "source": "llm",
                "root_kind": "worker",
                "target_id": worker_target,
                "owner_generation": boot_id,
            },
            {
                "source": f"worker:{worker_target}",
                "root_kind": "hardware",
                "target_id": hardware_target,
                "owner_generation": boot_id,
            },
        ]

    def _pulse_mycelial_worker(self, heartbeat: Mapping[str, Any]) -> None:
        refs = list(getattr(self, "_mycelial_root_refs", ()) or ())
        if not refs:
            return
        identity = self.get_worker_identity_snapshot()
        if (
            str(heartbeat.get("worker_boot_id") or "")
            != str(identity.get("worker_boot_id") or "")
            or heartbeat.get("worker_pid") != identity.get("worker_pid")
        ):
            return
        from core.container import ServiceContainer

        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium is None or not hasattr(mycelium, "pulse_neural_root"):
            return
        evidence = {
            "active_job": bool(heartbeat.get("active_job")),
            "ipc_backlog": int(heartbeat.get("ipc_backlog") or 0),
            "ipc_broken": bool(heartbeat.get("ipc_broken")),
            "loop_stalled": bool(heartbeat.get("loop_stalled")),
        }
        # A heartbeat with a generation-progress alarm still proves this
        # process and its IPC path are live. Generation health is handled by
        # the request watchdog; only broken IPC invalidates the root probe.
        success = not evidence["ipc_broken"]
        for ref in refs:
            mycelium.pulse_neural_root(
                ref["source"],
                root_kind=ref["root_kind"],
                target_id=ref["target_id"],
                owner_generation=ref["owner_generation"],
                success=success,
                evidence=evidence,
            )

    def _unbind_mycelial_worker(self) -> None:
        refs = list(getattr(self, "_mycelial_root_refs", ()) or ())
        self._mycelial_root_refs = []
        if not refs:
            return
        try:
            from core.container import ServiceContainer

            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium is None or not hasattr(mycelium, "unbind_neural_roots"):
                return
            for ref in refs:
                mycelium.unbind_neural_roots(
                    ref["source"],
                    owner_generation=ref["owner_generation"],
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug(
                "Mycelial worker-root retirement unavailable for %s.",
                os.path.basename(self.model_path),
            )

    def _clean_latent_cancel_ack(
        self,
        response: Any,
        *,
        expected_request_id: str = "",
        expected_request_sha256: str = "",
    ) -> bool:
        """Whether this acknowledgement proves a CLEAN cancel of THIS episode.

        CP126 07d62d51. The check used to accept a reason string and a couple
        of worker-supplied booleans, bound to nothing. Anything shaped like
        {"message": "soft_cancelled", "receipt": {"params_unchanged": True}}
        could therefore certify that model parameters were untouched and
        ephemeral weights erased — for a different request, a different
        worker, or a previous episode entirely. That certification is what
        lets the lane keep serving without a reboot, so a stale or replayed
        ack was a path to serving on weights nobody had proven clean.

        The receipt already carries the identity needed to bind it; nothing
        was reading it. An acknowledgement now has to name this request, this
        payload, and this worker.
        """
        if not isinstance(response, dict):
            return False
        reason = str(response.get("message") or response.get("reason") or "")
        receipt = response.get("receipt")
        if reason != "soft_cancelled" or not isinstance(receipt, dict):
            return False

        # Bound to THIS request.
        if expected_request_id:
            if str(response.get("id") or "") != expected_request_id:
                self._record_cancel_ack_rejection("request_id_mismatch")
                return False
        # Bound to THIS payload.
        if expected_request_sha256:
            if str(receipt.get("request_payload_sha256") or "") != expected_request_sha256:
                self._record_cancel_ack_rejection("request_payload_sha256_mismatch")
                return False
        # Bound to THIS worker: a receipt from a previous boot describes a
        # process whose weights are no longer the ones we are about to keep
        # serving on.
        identity = getattr(self, "_worker_identity", None)
        receipt_worker_identity = receipt.get("worker_identity")
        if isinstance(identity, dict) and identity:
            expected_boot = str(identity.get("worker_boot_id") or "")
            if (
                not isinstance(receipt_worker_identity, Mapping)
                or expected_boot
                and str(receipt_worker_identity.get("worker_boot_id") or "")
                != expected_boot
            ):
                self._record_cancel_ack_rejection("worker_boot_id_mismatch")
                return False
            expected_pid = identity.get("worker_pid")
            if (
                isinstance(expected_pid, int)
                and receipt_worker_identity.get("worker_pid") != expected_pid
            ):
                self._record_cancel_ack_rejection("worker_pid_mismatch")
                return False
        reported_path = str(
            receipt_worker_identity.get("worker_model_path")
            if isinstance(receipt_worker_identity, Mapping)
            else ""
        )
        if reported_path and _real_model_path(reported_path) != _real_model_path(self.model_path):
            self._record_cancel_ack_rejection("worker_model_path_mismatch")
            return False

        try:
            from core.brain.llm.latent_cortex.runtime_integrity import (
                runtime_integrity_safe,
            )

            integrity_safe = runtime_integrity_safe(
                receipt.get("runtime_integrity"),
                require_worker=True,
                expected_episode_id=str(receipt.get("episode_id") or ""),
                expected_input_tokens_sha256=str(
                    receipt.get("input_tokens_sha256") or ""
                ),
                expected_worker_identity=(
                    identity if isinstance(identity, Mapping) else None
                ),
                expected_fast_weights_applied=(
                    receipt.get("fast_weights_applied") is True
                ),
                expected_fast_weights_attach_attempted=(
                    receipt.get("fast_weights_attach_attempted") is True
                ),
                expected_checkpoint_fingerprint=str(
                    receipt.get("checkpoint_fingerprint") or ""
                ),
                expected_checkpoint_method=str(
                    receipt.get("checkpoint_fingerprint_method") or ""
                ),
                expected_checkpoint_file_count=receipt.get(
                    "checkpoint_file_count"
                ),
            )
        except ImportError:
            integrity_safe = False
        if not integrity_safe:
            self._record_cancel_ack_rejection("runtime_integrity_unproven")
            return False
        return True

    def _record_cancel_ack_rejection(self, why: str) -> None:
        """An ack that failed to bind is evidence, not noise.

        A worker sending unbindable cancellation receipts is either buggy or
        replaying, and either way the lane must reboot rather than trust the
        clean-cancel claim.
        """
        _record_mlx_degradation(
            RuntimeError(f"latent_cancel_ack_unbound:{why}"),
            action="refused a latent cancellation acknowledgement it could not bind",
            severity="error",
        )

    async def unified_recurrent_shadow_probe_async(
        self,
        public_token_ids: Sequence[int],
        expected_token_ids: Sequence[int],
        *,
        max_tokens: int,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Measure resident recurrent tissue without exposing or serving its text."""

        base: dict[str, Any] = {"ok": False, "status": "unavailable", "receipt": {}}
        if self._closed:
            return {**base, "reason": "client_closed"}
        shadow_status = copy.deepcopy(
            getattr(self, "_unified_recurrent_shadow_status", {})
        )
        if not (
            isinstance(shadow_status, dict)
            and shadow_status.get("loaded") is True
            and shadow_status.get("serving_authority") is False
        ):
            return {**base, "reason": "unified_recurrent_shadow_not_loaded"}
        try:
            from core.brain.llm.unified_recurrent_shadow_probe_contract import (
                seal_shadow_probe_request,
                shadow_probe_receipt_errors,
            )

            probe_request = seal_shadow_probe_request(
                public_token_ids,
                expected_token_ids,
                max_tokens=max_tokens,
            )
            bounded_timeout_s = float(timeout_s)
            if not math.isfinite(bounded_timeout_s) or bounded_timeout_s <= 0.0:
                raise ValueError("timeout_invalid")
            bounded_timeout_s = min(300.0, max(5.0, bounded_timeout_s))
        except (ImportError, TypeError, ValueError, OverflowError) as exc:
            return {**base, "reason": f"invalid_shadow_probe_request:{exc}"}
        if self._req_q is None or not (
            self._process and self._process.is_alive() and self._init_done
        ):
            return {**base, "reason": "worker_not_ready"}
        try:
            if get_memory_pressure_snapshot().refuse_heavy_local_generation:
                return {**base, "reason": "memory_pressure"}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return {**base, "reason": "memory_pressure_unobservable"}

        deadline = get_deadline(bounded_timeout_s)
        acquired = await self._acquire_request_lock(
            owner_label="unified_recurrent_shadow_probe",
            deadline=deadline,
            foreground_request=False,
        )
        if not acquired:
            return {**base, "reason": "request_lane_busy"}
        if _foreground_owner_active():
            self._release_request_lock()
            return {**base, "reason": "foreground_active_after_lane"}

        future: SharedFuture | None = None
        request_id = ""
        deferred_reboot = ""
        lane_fenced = False
        try:
            if self._req_q is None or not (
                self._process and self._process.is_alive() and self._init_done
            ):
                return {**base, "reason": "worker_not_ready"}
            if self._warmup_in_flight or self._active_generations > 0:
                return {**base, "reason": "generation_active"}
            if not await self._set_durable_lane_preemptible(False):
                return {**base, "reason": "lane_fence_lost"}
            lane_fenced = True
            request_id = uuid.uuid4().hex
            self._job_seq_counter += 1
            request_seq = self._job_seq_counter
            job = {
                "id": request_id,
                "seq": request_seq,
                "action": "unified_recurrent_shadow_probe",
                "unified_recurrent_shadow_contract": probe_request,
            }
            future = _new_shared_future()
            self._pending_generations[request_id] = future
            self._current_gen_future = future
            self._active_generations += 1
            self._active_generation_started_at = time.time()
            self._mark_generation_started(
                request_id,
                requested_max_tokens=max_tokens,
                first_token_hard_ceiling_s=bounded_timeout_s,
                request_seq=request_seq,
            )
            dispatch_budget = _remaining_budget(deadline, bounded_timeout_s)
            if dispatch_budget <= 0.0:
                return {**base, "reason": "shadow_probe_timeout:dispatch"}
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(
                    job,
                    principal="mlx_client.unified_recurrent_shadow_probe",
                ),
                True,
                min(2.0, dispatch_budget),
            )
            generation_budget = _remaining_budget(deadline, bounded_timeout_s)
            if generation_budget <= 0.0:
                deferred_reboot = "shadow_probe_deadline_unacknowledged"
                return {**base, "reason": "shadow_probe_timeout:generation_start"}
            try:
                response = await _await_shared_future(future, timeout_s=generation_budget)
            except TimeoutError:
                self.soft_cancel_active_generation("unified_recurrent_shadow_probe_deadline")
                try:
                    response = await _await_shared_future(future, timeout_s=3.0)
                except (TimeoutError, BrokenPipeError, OSError):
                    deferred_reboot = "shadow_probe_deadline_unacknowledged"
                    return {**base, "reason": "shadow_probe_timeout:unacknowledged"}
            if not isinstance(response, dict):
                return {**base, "reason": "invalid_worker_response"}
            if response.get("status") != "ok":
                return {
                    **base,
                    "status": "worker_error",
                    "reason": str(response.get("message") or "unknown"),
                }
            if response.get("allocator_reclaimed") is not True:
                deferred_reboot = "shadow_probe_allocator_reclaim_unproven"
                return {
                    **base,
                    "status": "integrity_failed",
                    "reason": deferred_reboot,
                }
            receipt = response.get("receipt")
            errors = shadow_probe_receipt_errors(
                receipt,
                expected_request_sha256=probe_request["request_sha256"],
                expected_package_id=str(shadow_status.get("package_id") or ""),
                expected_controller_sha256=str(
                    shadow_status.get("controller_sha256") or ""
                ),
            )
            if errors:
                deferred_reboot = "shadow_probe_receipt_invalid"
                return {
                    **base,
                    "status": "integrity_failed",
                    "reason": ",".join(errors),
                }
            accepted = copy.deepcopy(receipt)
            self._unified_recurrent_shadow_probe_status = accepted
            self._mark_progress()
            return {
                "ok": accepted["status"] == "completed",
                "status": accepted["status"],
                "receipt": accepted,
                "reason": accepted["reason"],
            }
        except asyncio.CancelledError:
            if future is not None:
                self.soft_cancel_active_generation(
                    "unified_recurrent_shadow_probe_caller_cancelled"
                )
                deferred_reboot = "shadow_probe_caller_cancelled"
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            deferred_reboot = f"shadow_probe_ipc_failed:{type(exc).__name__}"
            _record_mlx_degradation(
                exc,
                action="recycled resident worker after shadow probe IPC failure",
                severity="warning",
            )
            return {**base, "reason": deferred_reboot}
        finally:
            try:
                try:
                    if future is not None:
                        await asyncio.shield(
                            self._finish_generation_ownership(
                                request_id,
                                future,
                                None,
                                release_lane=not bool(deferred_reboot),
                            )
                        )
                finally:
                    if deferred_reboot:
                        await asyncio.shield(
                            self.reboot_worker(
                                reason=deferred_reboot,
                                mark_failed=False,
                            )
                        )
                    elif lane_fenced and future is None and self._active_generations <= 0:
                        await asyncio.shield(self._set_durable_lane_preemptible(True))
            finally:
                self._release_request_lock()

    def unified_recurrent_qualified_serving_status(self) -> dict[str, Any]:
        """Report whether this exact resident worker may serve qualified tissue."""

        if self._closed:
            return {"active": False, "reason": "client_closed"}
        shadow_status = copy.deepcopy(
            getattr(self, "_unified_recurrent_shadow_status", {})
        )
        qualified_status = copy.deepcopy(
            getattr(self, "_unified_recurrent_qualified_activation_status", {})
        )
        activation = (
            qualified_status.get("activation")
            if isinstance(qualified_status, Mapping)
            else None
        )
        if not (
            isinstance(shadow_status, Mapping)
            and shadow_status.get("loaded") is True
            and isinstance(qualified_status, Mapping)
            and qualified_status.get("loaded") is True
            and qualified_status.get("serving_authority") is True
            and isinstance(activation, Mapping)
        ):
            return {
                "active": False,
                "reason": "qualified_recurrent_serving_not_active",
            }
        try:
            from core.brain.llm.unified_recurrent_qualified_activation import (
                activation_matches_shadow_receipt,
            )

            if not activation_matches_shadow_receipt(activation, shadow_status):
                return {
                    "active": False,
                    "reason": "qualified_activation_shadow_identity_differs",
                }
        except (ImportError, TypeError, ValueError):
            return {
                "active": False,
                "reason": "qualified_recurrent_serving_status_invalid",
            }
        return {
            "active": True,
            "reason": "qualified_recurrent_serving_active",
            "package_id": str(shadow_status.get("package_id") or ""),
            "controller_sha256": str(shadow_status.get("controller_sha256") or ""),
            "activation_sha256": str(activation.get("activation_sha256") or ""),
        }

    async def unified_recurrent_qualified_decode_async(
        self,
        public_token_ids: Sequence[int],
        *,
        family: str,
        task_depth: int,
        max_tokens: int,
        timeout_s: float = 180.0,
        _canary_activation: Mapping[str, Any] | None = None,
        _canary_battery_sha256: str = "",
        _canary_case_index: int = -1,
        _canary_nonce: str = "",
    ) -> dict[str, Any]:
        """Serve one admitted typed answer through the resident worker."""

        base: dict[str, Any] = {"ok": False, "status": "unavailable", "receipt": {}}
        if self._closed:
            return {**base, "reason": "client_closed"}
        serving_status = self.unified_recurrent_qualified_serving_status()
        shadow_status = copy.deepcopy(getattr(self, "_unified_recurrent_shadow_status", {}))
        qualified_status = copy.deepcopy(
            getattr(self, "_unified_recurrent_qualified_activation_status", {})
        )
        durable_activation = (
            qualified_status.get("activation")
            if isinstance(qualified_status, Mapping)
            else None
        )
        canary_activation = (
            copy.deepcopy(dict(_canary_activation))
            if isinstance(_canary_activation, Mapping)
            else None
        )
        activation = canary_activation or durable_activation
        if canary_activation is None and serving_status.get("active") is not True:
            return {**base, "reason": str(serving_status.get("reason") or "unknown")}
        try:
            from core.brain.llm.unified_recurrent_qualified_activation import (
                activation_matches_shadow_receipt,
                qualified_activation_errors,
            )
            from core.brain.llm.unified_recurrent_qualified_decode import (
                qualified_decode_result_errors,
                seal_qualified_canary_request_authority,
                seal_qualified_decode_request,
            )

            if (
                not isinstance(activation, Mapping)
                or qualified_activation_errors(activation)
                or not activation_matches_shadow_receipt(activation, shadow_status)
                or (
                    canary_activation is not None
                    and qualified_status.get("loaded") is True
                )
            ):
                return {**base, "reason": "qualified_activation_shadow_identity_differs"}
            request = seal_qualified_decode_request(
                public_token_ids,
                package_id=str(shadow_status.get("package_id") or ""),
                controller_sha256=str(
                    shadow_status.get("controller_sha256") or ""
                ),
                family=family,
                task_depth=task_depth,
                max_tokens=max_tokens,
            )
            bounded_timeout_s = float(timeout_s)
            if not math.isfinite(bounded_timeout_s) or bounded_timeout_s <= 0.0:
                raise ValueError("timeout_invalid")
            bounded_timeout_s = min(300.0, max(5.0, bounded_timeout_s))
            canary_authority = None
            if canary_activation is not None:
                issued_at = time.time()
                canary_authority = seal_qualified_canary_request_authority(
                    activation_sha256=str(
                        canary_activation.get("activation_sha256") or ""
                    ),
                    battery_sha256=_canary_battery_sha256,
                    case_index=_canary_case_index,
                    request_sha256=request["request_sha256"],
                    nonce=_canary_nonce,
                    issued_at_unix=issued_at,
                    expires_at_unix=issued_at + bounded_timeout_s,
                )
        except (ImportError, TypeError, ValueError, OverflowError) as exc:
            return {**base, "reason": f"invalid_qualified_decode_request:{exc}"}
        if self._req_q is None or not (
            self._process and self._process.is_alive() and self._init_done
        ):
            return {**base, "reason": "worker_not_ready"}

        deadline = get_deadline(bounded_timeout_s)
        acquired = await self._acquire_request_lock(
            owner_label="unified_recurrent_qualified_decode",
            deadline=deadline,
            foreground_request=True,
        )
        if not acquired:
            return {**base, "reason": "request_lane_busy"}

        future: SharedFuture | None = None
        request_id = ""
        deferred_reboot = ""
        lane_fenced = False
        try:
            if self._req_q is None or not (
                self._process and self._process.is_alive() and self._init_done
            ):
                return {**base, "reason": "worker_not_ready"}
            if self._warmup_in_flight or self._active_generations > 0:
                return {**base, "reason": "generation_active"}
            if not await self._set_durable_lane_preemptible(False):
                return {**base, "reason": "lane_fence_lost"}
            lane_fenced = True
            request_id = uuid.uuid4().hex
            self._job_seq_counter += 1
            request_seq = self._job_seq_counter
            job = {
                "id": request_id,
                "seq": request_seq,
                "action": "unified_recurrent_qualified_decode",
                "unified_recurrent_qualified_decode_contract": request,
            }
            if canary_activation is not None:
                job["unified_recurrent_qualified_canary_activation"] = canary_activation
                job["unified_recurrent_qualified_canary_authority"] = canary_authority
            future = _new_shared_future()
            self._pending_generations[request_id] = future
            self._current_gen_future = future
            self._active_generations += 1
            self._active_generation_started_at = time.time()
            self._mark_generation_started(
                request_id,
                requested_max_tokens=max_tokens,
                first_token_hard_ceiling_s=bounded_timeout_s,
                request_seq=request_seq,
            )
            dispatch_budget = _remaining_budget(deadline, bounded_timeout_s)
            if dispatch_budget <= 0.0:
                return {**base, "reason": "qualified_decode_timeout:dispatch"}
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(
                    job,
                    principal="mlx_client.unified_recurrent_qualified_decode",
                ),
                True,
                min(2.0, dispatch_budget),
            )
            generation_budget = _remaining_budget(deadline, bounded_timeout_s)
            if generation_budget <= 0.0:
                deferred_reboot = "qualified_decode_deadline_unacknowledged"
                return {
                    **base,
                    "reason": "qualified_decode_timeout:generation_start",
                }
            try:
                response = await _await_shared_future(
                    future,
                    timeout_s=generation_budget,
                )
            except TimeoutError:
                self.soft_cancel_active_generation("qualified_decode_deadline")
                try:
                    response = await _await_shared_future(future, timeout_s=3.0)
                except (TimeoutError, BrokenPipeError, OSError):
                    deferred_reboot = "qualified_decode_deadline_unacknowledged"
                    return {
                        **base,
                        "reason": "qualified_decode_timeout:unacknowledged",
                    }
            if not isinstance(response, Mapping):
                return {**base, "reason": "invalid_worker_response"}
            if response.get("status") != "ok":
                return {
                    **base,
                    "status": "worker_error",
                    "reason": str(response.get("message") or "unknown"),
                }
            if response.get("allocator_reclaimed") is not True:
                deferred_reboot = "qualified_decode_allocator_reclaim_unacknowledged"
                return {
                    **base,
                    "status": "integrity_failed",
                    "reason": deferred_reboot,
                }
            receipt = response.get("receipt")
            errors = qualified_decode_result_errors(
                receipt,
                expected_request_sha256=request["request_sha256"],
                expected_activation_sha256=str(
                    activation.get("activation_sha256") or ""
                ),
                expected_package_id=request["package_id"],
                expected_controller_sha256=request["controller_sha256"],
                expected_family=request["family"],
                expected_task_depth=request["task_depth"],
                expected_canary_authority=canary_activation is not None,
            )
            if errors:
                deferred_reboot = "qualified_decode_receipt_invalid"
                return {
                    **base,
                    "status": "integrity_failed",
                    "reason": ",".join(errors),
                }
            accepted = copy.deepcopy(receipt)
            self._mark_progress()
            return {
                "ok": True,
                "status": "completed",
                "receipt": accepted,
                "reason": "qualified_decode_completed",
            }
        except asyncio.CancelledError:
            if future is not None:
                self.soft_cancel_active_generation(
                    "unified_recurrent_qualified_decode_caller_cancelled"
                )
                deferred_reboot = "qualified_decode_caller_cancelled"
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            deferred_reboot = f"qualified_decode_ipc_failed:{type(exc).__name__}"
            _record_mlx_degradation(
                exc,
                action="recycled resident worker after qualified decode IPC failure",
                severity="warning",
            )
            return {**base, "reason": deferred_reboot}
        finally:
            try:
                try:
                    if future is not None:
                        await asyncio.shield(
                            self._finish_generation_ownership(
                                request_id,
                                future,
                                None,
                                release_lane=not bool(deferred_reboot),
                            )
                        )
                finally:
                    if deferred_reboot:
                        await asyncio.shield(
                            self.reboot_worker(
                                reason=deferred_reboot,
                                mark_failed=False,
                            )
                        )
                    elif lane_fenced and future is None and self._active_generations <= 0:
                        await asyncio.shield(
                            self._set_durable_lane_preemptible(True)
                        )
            finally:
                self._release_request_lock()

    async def unified_recurrent_qualified_canary_decode_async(
        self,
        public_token_ids: Sequence[int],
        *,
        family: str,
        task_depth: int,
        max_tokens: int,
        activation: Mapping[str, Any],
        battery_sha256: str,
        case_index: int,
        nonce: str,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Prove qualified IPC with an in-memory, request-bound authority."""

        return await self.unified_recurrent_qualified_decode_async(
            public_token_ids,
            family=family,
            task_depth=task_depth,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            _canary_activation=activation,
            _canary_battery_sha256=battery_sha256,
            _canary_case_index=case_index,
            _canary_nonce=nonce,
        )

    async def unified_recurrent_shadow_canary_async(
        self,
        cases: Sequence[Mapping[str, Any]],
        *,
        minimum_wrong_to_right: int = 1,
        maximum_shadow_latency_ms: int = 120_000,
        maximum_latency_ratio_numerator: int = 8,
        maximum_latency_ratio_denominator: int = 1,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the domain-bound shadow gate without placing output on chat."""

        shadow_status = copy.deepcopy(
            getattr(self, "_unified_recurrent_shadow_status", {})
        )
        if not (
            isinstance(shadow_status, dict)
            and shadow_status.get("loaded") is True
            and shadow_status.get("serving_authority") is False
        ):
            return {
                "plan": {},
                "verdict": {},
                "supported": False,
                "reason": "unified_recurrent_shadow_not_loaded",
            }
        try:
            from core.brain.llm.unified_recurrent_shadow_canary import (
                run_shadow_canary,
            )

            result = await run_shadow_canary(
                cases,
                package_id=str(shadow_status.get("package_id") or ""),
                controller_sha256=str(
                    shadow_status.get("controller_sha256") or ""
                ),
                probe=self.unified_recurrent_shadow_probe_async,
                minimum_wrong_to_right=minimum_wrong_to_right,
                maximum_shadow_latency_ms=maximum_shadow_latency_ms,
                maximum_latency_ratio_numerator=maximum_latency_ratio_numerator,
                maximum_latency_ratio_denominator=maximum_latency_ratio_denominator,
                progress=progress,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "plan": {},
                "verdict": {},
                "supported": False,
                "reason": f"shadow_canary_invalid:{exc}",
            }
        verdict = result.get("verdict")
        accepted_verdict = copy.deepcopy(verdict) if isinstance(verdict, dict) else {}
        self._unified_recurrent_shadow_canary_status = accepted_verdict
        self._mark_progress()
        return {
            **result,
            "supported": bool(
                isinstance(verdict, dict) and verdict.get("supported") is True
            ),
            "reason": (
                str(verdict.get("verdict") or "shadow_canary_unavailable")
                if isinstance(verdict, dict)
                else "shadow_canary_unavailable"
            ),
        }

    async def unified_recurrent_shadow_package_canary_async(
        self,
        package: Path | None = None,
        *,
        minimum_wrong_to_right: int = 1,
        maximum_shadow_latency_ms: int = 120_000,
        maximum_latency_ratio_numerator: int = 8,
        maximum_latency_ratio_denominator: int = 1,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the package's private fresh battery through the shadow lane."""

        shadow_status = copy.deepcopy(
            getattr(self, "_unified_recurrent_shadow_status", {})
        )
        configured_value: Path | str = (
            package
            if package is not None
            else os.getenv("AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE", "")
        )
        if not str(configured_value).strip():
            return {
                "plan": {},
                "verdict": {},
                "supported": False,
                "reason": "unified_recurrent_shadow_package_not_configured",
            }
        try:
            from core.brain.llm.unified_recurrent_shadow import (
                inspect_shadow_package,
            )
            from core.brain.llm.unified_recurrent_shadow_battery import (
                shadow_canary_cases,
            )

            configured = await asyncio.to_thread(
                lambda: Path(configured_value).expanduser()
            )
            verified = await asyncio.to_thread(inspect_shadow_package, configured)
            manifest = verified.get("manifest")
            if not (
                isinstance(manifest, dict)
                and manifest.get("package_id") == shadow_status.get("package_id")
                and manifest.get("manifest_sha256")
                == shadow_status.get("manifest_sha256")
            ):
                raise ValueError("shadow_package_worker_identity_differs")
            cases = shadow_canary_cases(verified.get("canary_battery"))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "plan": {},
                "verdict": {},
                "supported": False,
                "reason": f"shadow_package_canary_invalid:{exc}",
            }
        return await self.unified_recurrent_shadow_canary_async(
            cases,
            minimum_wrong_to_right=minimum_wrong_to_right,
            maximum_shadow_latency_ms=maximum_shadow_latency_ms,
            maximum_latency_ratio_numerator=maximum_latency_ratio_numerator,
            maximum_latency_ratio_denominator=maximum_latency_ratio_denominator,
            progress=progress,
        )

    async def latent_reason_async(
        self,
        prompt: str | None = None,
        *,
        messages: list | None = None,
        config: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        runtime_controls: dict[str, Any] | None = None,
        domain: str = "general",
        timeout_s: float = 300.0,
        foreground_request: bool = True,
        verifier_guidance: bool = False,
        facet_reliability: dict[str, float] | None = None,
        cognitive_context: list | None = None,
        operation_authority: dict[str, Any] | None = None,
        action_policy_evidence: dict[str, Any] | None = None,
        action_intervention: dict[str, Any] | None = None,
        action_state_runtime: dict[str, Any] | None = None,
        external_execution_offer: dict[str, Any] | None = None,
        response_contract: str | None = None,
    ) -> dict[str, Any]:
        """Run a Recursive Latent Cortex episode on the RESIDENT worker model.

        Workspace recurrence + virtual-width branches over the frozen
        checkpoint (docs/RECURSIVE_LATENT_CORTEX.md). Refuses while a
        generation is in flight (the episode needs exclusive weights/KV) and
        never spawns a worker just to think — no resident model, no episode.
        Returns ``{"ok": bool, "text": str, "receipt": {...}, "reason": str}``.
        """
        base = {"ok": False, "text": "", "receipt": {}}
        if self._closed:
            return {**base, "reason": "client_closed"}
        if not (isinstance(prompt, str) and prompt.strip()) and not (
            isinstance(messages, list) and messages
        ):
            return {**base, "reason": "empty_prompt"}
        # CP126 a09d6218. Everything below this point copies, serialises and
        # HASHES these structures — before worker readiness and before memory
        # admission. So an oversized or malformed payload spent real parent
        # CPU and memory on an episode that could never run, and "messages is
        # a non-empty list" was the only thing ever checked about a list whose
        # items reach the worker.
        schema_error = _latent_request_schema_error(prompt=prompt, messages=messages)
        if schema_error:
            return {**base, "reason": schema_error}
        if type(foreground_request) is not bool:
            return {**base, "reason": "invalid_foreground_request"}
        if config is not None and not isinstance(config, dict):
            return {**base, "reason": "invalid_config"}
        if budget is not None and not isinstance(budget, dict):
            return {**base, "reason": "invalid_budget"}
        if runtime_controls is not None and not isinstance(runtime_controls, dict):
            return {**base, "reason": "invalid_runtime_controls"}
        if response_contract is not None:
            if not isinstance(response_contract, str) or not response_contract.strip():
                return {**base, "reason": "invalid_response_contract"}
            try:
                from core.brain.llm.latent_cortex.response_contracts import (
                    parse_response_contract,
                )

                parse_response_contract(response_contract)
            except ValueError:
                return {**base, "reason": "invalid_response_contract"}
        wire_cognitive_context: list[dict[str, Any]] | None = None
        try:
            from core.brain.llm.latent_cortex.cognitive_context import (
                normalize_cognitive_context,
            )

            wire_cognitive_context = normalize_cognitive_context(cognitive_context) or None
        except (TypeError, ValueError):
            return {**base, "reason": "invalid_cognitive_context"}
        wire_config = dict(config or {})
        wire_budget = dict(budget or {})
        wire_runtime_controls = dict(runtime_controls or {})
        wire_action_policy_evidence: dict[str, Any] | None = None
        if action_policy_evidence is not None:
            try:
                from core.brain.llm.latent_cortex.value_of_computation import (
                    validate_evidence_snapshot,
                )

                wire_action_policy_evidence = validate_evidence_snapshot(action_policy_evidence)
            except (ImportError, TypeError, ValueError):
                return {**base, "reason": "invalid_action_policy_evidence"}
        wire_action_intervention: dict[str, Any] | None = None
        if action_intervention is not None:
            try:
                from core.brain.llm.latent_cortex.action_intervention import (
                    validate_action_intervention,
                )

                wire_action_intervention = validate_action_intervention(
                    action_intervention,
                    require_current_policy=True,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                return {**base, "reason": "invalid_action_intervention"}
            if wire_action_policy_evidence is None:
                return {
                    **base,
                    "reason": "action_intervention_policy_evidence_missing",
                }
            if foreground_request:
                return {
                    **base,
                    "reason": "action_intervention_requires_lab_lane",
                }
        wire_action_state_runtime: dict[str, Any] | None = None
        admitted_action_state_runtime: Any | None = None
        if action_state_runtime is not None:
            if foreground_request:
                return {
                    **base,
                    "reason": "action_state_runtime_requires_lab_lane",
                }
            try:
                from core.brain.llm.latent_cortex.action_state_runtime import (
                    admit_action_state_runtime,
                    provision_action_state_store_custody,
                )

                binding = self.get_worker_identity_snapshot().get(
                    "worker_action_capture_origin_binding"
                )
                if not isinstance(binding, Mapping):
                    raise ValueError("worker capture origin unavailable")
                candidate_runtime = json.loads(
                    json.dumps(action_state_runtime, allow_nan=False)
                )
                candidate_runtime["resident_worker_origin_binding"] = json.loads(
                    json.dumps(binding, allow_nan=False)
                )
                provision_action_state_store_custody()
                admitted_runtime = admit_action_state_runtime(
                    candidate_runtime,
                    worker_launch_challenge=binding.get("launch_challenge"),
                    now_unix=int(time.time()),
                )
                if (
                    admitted_runtime.mode == "capture"
                    and wire_action_intervention is not None
                ):
                    raise ValueError("capture cannot carry an intervention")
                if admitted_runtime.mode == "restore":
                    if wire_action_intervention is None:
                        raise ValueError("restore requires an intervention")
                    if (
                        wire_action_intervention["authority_payload"]["arm"]
                        != admitted_runtime.arm
                    ):
                        raise ValueError("restore arm differs from intervention")
                wire_action_state_runtime = candidate_runtime
                admitted_action_state_runtime = admitted_runtime
            except (
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                return {**base, "reason": "invalid_action_state_runtime"}
        wire_external_execution_offer: dict[str, Any] | None = None
        if external_execution_offer is not None:
            try:
                from core.brain.llm.latent_cortex.external_execution import (
                    validate_external_execution_offer,
                )

                wire_external_execution_offer = validate_external_execution_offer(
                    external_execution_offer
                )
            except (ImportError, TypeError, ValueError):
                return {**base, "reason": "invalid_external_execution_offer"}
            if wire_action_policy_evidence is None or operation_authority is None:
                return {
                    **base,
                    "reason": "external_execution_authority_tuple_missing",
                }
        if (
            wire_action_intervention is not None
            and wire_action_intervention["authority_payload"]["action"] == "execute"
            and wire_external_execution_offer is None
        ):
            return {
                **base,
                "reason": "execute_intervention_offer_missing",
            }
        wire_operation_authority: dict[str, Any] | None = None
        if operation_authority is not None:
            try:
                from core.brain.llm.latent_cortex.epistemic_runtime import (
                    validate_runtime_operation_authority,
                )

                wire_operation_authority = validate_runtime_operation_authority(
                    operation_authority,
                    prompt=prompt,
                    messages=messages,
                    config=wire_config,
                    budget=wire_budget,
                    cognitive_context=wire_cognitive_context,
                    action_policy_evidence=wire_action_policy_evidence,
                    external_execution_offer=wire_external_execution_offer,
                )
            except (ImportError, TypeError, ValueError):
                return {**base, "reason": "invalid_runtime_operation_authority"}
        if runtime_controls is not None:
            required_controls = {
                "clean_user_surface_recurrent_loops",
                "clean_user_surface_steering_alpha",
            }
            if set(wire_runtime_controls) != required_controls:
                return {**base, "reason": "invalid_runtime_controls"}
            recurrent_loops = wire_runtime_controls.get("clean_user_surface_recurrent_loops")
            steering_alpha = wire_runtime_controls.get("clean_user_surface_steering_alpha")
            if (
                type(recurrent_loops) is not int
                or not 1 <= recurrent_loops <= 2
                or isinstance(steering_alpha, bool)
                or not isinstance(steering_alpha, (int, float))
                or not math.isfinite(float(steering_alpha))
                or not 0.0 <= float(steering_alpha) <= 1.0
            ):
                return {**base, "reason": "invalid_runtime_controls"}
        # CP126 9721b1be. These are semantic inputs to the episode, so they
        # must be normalized ONCE here and bound into the request digest —
        # building them only at job-construction time left two episodes with
        # different verifier behavior sharing one expected request identity.
        wire_verifier_guidance = True if verifier_guidance else None
        wire_facet_reliability: dict[str, float] | None = None
        if verifier_guidance and isinstance(facet_reliability, dict) and facet_reliability:
            # Held-out facet calibration rides only alongside the verifier it
            # calibrates; worker revalidates the shape.
            wire_facet_reliability = {
                str(name): float(value)
                for name, value in list(facet_reliability.items())[:8]
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            } or None
        try:
            from core.brain.llm.latent_cortex.runtime_identity import (
                latent_request_payload_sha256,
            )

            if wire_action_intervention is not None:
                intervention_request_sha256 = latent_request_payload_sha256(
                    prompt=str(prompt) if prompt is not None else None,
                    messages=list(messages) if messages is not None else None,
                    domain=str(domain or "general"),
                    config=wire_config if config is not None else None,
                    budget=wire_budget if budget is not None else None,
                    runtime_controls=(
                        wire_runtime_controls if runtime_controls is not None else None
                    ),
                    cognitive_context=wire_cognitive_context,
                    operation_authority=wire_operation_authority,
                    action_policy_evidence=wire_action_policy_evidence,
                    external_execution_offer=wire_external_execution_offer,
                    response_contract=response_contract,
                    verifier_guidance=wire_verifier_guidance,
                    facet_reliability=wire_facet_reliability,
                )
                if (
                    intervention_request_sha256
                    != wire_action_intervention["authority_payload"]["request_payload_sha256"]
                ):
                    return {
                        **base,
                        "reason": "action_intervention_request_mismatch",
                    }
            expected_request_sha256 = latent_request_payload_sha256(
                prompt=str(prompt) if prompt is not None else None,
                messages=list(messages) if messages is not None else None,
                domain=str(domain or "general"),
                config=wire_config if config is not None else None,
                budget=wire_budget if budget is not None else None,
                runtime_controls=(wire_runtime_controls if runtime_controls is not None else None),
                cognitive_context=wire_cognitive_context,
                operation_authority=wire_operation_authority,
                action_policy_evidence=wire_action_policy_evidence,
                action_intervention=wire_action_intervention,
                external_execution_offer=wire_external_execution_offer,
                response_contract=response_contract,
                verifier_guidance=wire_verifier_guidance,
                facet_reliability=wire_facet_reliability,
            )
        except (TypeError, ValueError, OverflowError):
            return {**base, "reason": "invalid_request_payload"}
        try:
            bounded_timeout_s = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            return {**base, "reason": "invalid_timeout"}
        if not math.isfinite(bounded_timeout_s) or bounded_timeout_s <= 0.0:
            return {**base, "reason": "invalid_timeout"}
        bounded_timeout_s = min(900.0, max(5.0, bounded_timeout_s))
        if self._req_q is None or not (
            self._process and self._process.is_alive() and self._init_done
        ):
            return {**base, "reason": "worker_not_ready"}
        try:
            if get_memory_pressure_snapshot().refuse_heavy_local_generation:
                return {**base, "reason": "memory_pressure"}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return {**base, "reason": "memory_pressure_unobservable"}

        deadline = get_deadline(bounded_timeout_s)
        owner_label = "latent_cortex_foreground" if foreground_request else "latent_cortex_lab"
        foreground_owner_cm = None
        if foreground_request:
            foreground_owner_cm = _foreground_owner_context(
                owner_label,
                deadline=deadline,
                foreground_request=True,
                stale_after=bounded_timeout_s,
                a_person_is_waiting=not _is_internal_inference(wire_cognitive_context),
            )
            try:
                await foreground_owner_cm.__aenter__()
            except TimeoutError:
                return {**base, "reason": "foreground_owner_busy"}

        try:
            acquired = await self._acquire_request_lock(
                owner_label=owner_label,
                deadline=deadline,
                foreground_request=foreground_request,
            )
        except BaseException:  # noqa: BLE001 - cancellation must release foreground ownership
            if foreground_owner_cm is not None:
                await asyncio.shield(foreground_owner_cm.__aexit__(*sys.exc_info()))
            raise
        if not acquired:
            if foreground_owner_cm is not None:
                await foreground_owner_cm.__aexit__(None, None, None)
            return {**base, "reason": "request_lane_busy"}

        fut: SharedFuture | None = None
        req_id = ""
        deferred_reboot = ""
        lane_fenced = False
        try:
            if self._req_q is None or not (
                self._process and self._process.is_alive() and self._init_done
            ):
                return {**base, "reason": "worker_not_ready"}
            if self._warmup_in_flight or self._active_generations > 0:
                return {**base, "reason": "generation_active"}
            if not await self._set_durable_lane_preemptible(False):
                return {**base, "reason": "lane_fence_lost"}
            lane_fenced = True

            req_id = uuid.uuid4().hex
            self._job_seq_counter += 1
            request_seq = self._job_seq_counter
            job: dict[str, Any] = {
                "id": req_id,
                "seq": request_seq,
                "action": "latent_reason",
                "domain": str(domain or "general"),
                "foreground_request": foreground_request,
            }
            # Exactly the values bound into expected_request_sha256 above.
            if wire_verifier_guidance:
                job["verifier_guidance"] = True
                if wire_facet_reliability:
                    job["facet_reliability"] = dict(wire_facet_reliability)
            if prompt is not None:
                job["prompt"] = str(prompt)
            if messages is not None:
                job["messages"] = list(messages)
            if config is not None:
                job["config"] = wire_config
            if budget is not None:
                job["budget"] = wire_budget
            if runtime_controls is not None:
                job["runtime_controls"] = wire_runtime_controls
                job["clean_user_surface_contract"] = True
                job["live_mind_controls_bound"] = True
                job.update(wire_runtime_controls)
            else:
                # Latent episodes without explicit surface-parity controls are
                # the experiment lane: they keep historical full governor
                # steering. Every OTHER worker job now defaults to the surface
                # clamp (fail-safe inversion after the July 2026 coherence
                # incident) — this opt-out is deliberately scoped to episodes.
                job["allow_full_affective_steering"] = True
            if wire_cognitive_context is not None:
                job["cognitive_context"] = wire_cognitive_context
            if wire_operation_authority is not None:
                job["operation_authority"] = wire_operation_authority
            if wire_action_policy_evidence is not None:
                job["action_policy_evidence"] = wire_action_policy_evidence
            if wire_action_intervention is not None:
                job["action_intervention"] = wire_action_intervention
            if wire_action_state_runtime is not None:
                job["action_state_runtime"] = wire_action_state_runtime
            if wire_external_execution_offer is not None:
                job["external_execution_offer"] = wire_external_execution_offer
            if response_contract is not None:
                job["response_contract"] = response_contract

            fut = _new_shared_future()
            self._pending_generations[req_id] = fut
            self._latent_progress_by_request[req_id] = {
                "request_id": req_id,
                "stage": "submitted",
                "received_at_unix": time.time(),
            }
            self._current_gen_future = fut
            self._active_generations += 1
            self._active_generation_started_at = time.time()
            requested_tokens_raw = wire_config.get("decode_max_tokens", 0)
            requested_tokens = (
                requested_tokens_raw
                if type(requested_tokens_raw) is int and requested_tokens_raw > 0
                else 0
            )
            # The last number before the worker, beside the one the client
            # granted. A budget that shrinks somewhere between them is
            # invisible from either end: the client's log says 2048 and the
            # worker's says 399, and nothing says which layer took the
            # difference.
            if int(requested_tokens or 0) and int(requested_tokens) < int(
                getattr(self, "max_tokens", 0) or 0
            ):
                logger.info(
                    "🔧 Decode budget on the wire: %d, against a client ceiling "
                    "of %d — something between them reduced it.",
                    int(requested_tokens),
                    int(getattr(self, "max_tokens", 0) or 0),
                )
            prompt_chars = len(prompt or "") + sum(
                len(str(message.get("content") or ""))
                for message in (messages or [])
                if isinstance(message, dict)
            )
            # What a large prompt is MADE of, at the one boundary every path
            # crosses.
            #
            # The gate logs a breakdown for prompts it assembles; the deep
            # cognitive path assembles its own and logged nothing. LIVE,
            # 2026-08-28: a 213-character question was answered from a
            # 50,359-character prompt that took 191.6 seconds to read — the
            # whole turn — and there was no way to see what those characters
            # were.
            if prompt_chars > 20_000:
                try:
                    parts = [
                        f"{str((m or {}).get('role') or '?')}"
                        f"={len(str((m or {}).get('content') or ''))}"
                        f":{str((m or {}).get('content') or '')[:60]!r}"
                        for m in (messages or [])
                        if isinstance(m, dict)
                    ]
                    if prompt:
                        parts.insert(0, f"prompt={len(str(prompt))}")
                    logger.info(
                        "📏 [MLX] %d-char prompt: %s",
                        prompt_chars,
                        "; ".join(parts)[:900],
                    )
                    # And inside the biggest one, its sections.
                    #
                    # Knowing a system message is 46,665 characters says only
                    # that something is large. The sections are what somebody
                    # can act on, and they are marked in the text already.
                    biggest = max(
                        (
                            str((m or {}).get("content") or "")
                            for m in (messages or [])
                            if isinstance(m, dict)
                        ),
                        key=len,
                        default="",
                    )
                    if len(biggest) > 20_000:
                        import re as _re

                        marks = [
                            (found.start(), found.group(0).strip())
                            for found in _re.finditer(
                                r"^(?:##+ [^\n]{0,60}|\[[A-Z][A-Z _-]{2,60}\])",
                                biggest,
                                _re.MULTILINE,
                            )
                        ]
                        if marks:
                            bounds = [m[0] for m in marks] + [len(biggest)]
                            sized = sorted(
                                (
                                    (bounds[i + 1] - bounds[i], marks[i][1])
                                    for i in range(len(marks))
                                ),
                                reverse=True,
                            )
                            logger.info(
                                "📏 [MLX] largest message %d chars, biggest "
                                "sections: %s",
                                len(biggest),
                                "; ".join(
                                    f"{name}={size}" for size, name in sized[:12]
                                )[:700],
                            )
                except (AttributeError, TypeError, ValueError):
                    pass
            self._mark_generation_started(
                req_id,
                prompt_chars=prompt_chars,
                requested_max_tokens=requested_tokens,
                first_token_hard_ceiling_s=bounded_timeout_s,
                request_seq=request_seq,
            )
            # CP126 4cc73762. Every phase below is paid out of the SAME
            # remaining budget. Before, the queue put took a 0.5s floor even
            # when less than that was left, and the generation wait then
            # restarted the caller's FULL original timeout — so owner wait +
            # lock wait + generation + cancel-ack could run far past the
            # deadline the caller was promised. Work that cannot start inside
            # the budget is refused with the phase that ran out, rather than
            # started and then abandoned.
            dispatch_budget = _remaining_budget(deadline, bounded_timeout_s)
            if dispatch_budget <= 0.0:
                return {
                    **base,
                    "reason": "latent_timeout:budget_exhausted",
                    "phase": "dispatch",
                }
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(job, principal="mlx_client.latent_reason"),
                True,
                min(2.0, dispatch_budget),
            )
            generation_budget = _remaining_budget(deadline, bounded_timeout_s)
            if generation_budget <= 0.0:
                deferred_reboot = "latent_reason_deadline_unacknowledged"
                return {
                    **base,
                    "reason": "latent_timeout:budget_exhausted",
                    "phase": "generation_start",
                }
            try:
                res = await _await_shared_future(fut, timeout_s=generation_budget)
            except TimeoutError:
                self.soft_cancel_active_generation("latent_reason_deadline")
                try:
                    # Deliberately OUTSIDE the caller's budget, and small. The
                    # deadline is already spent; this buys the worker one
                    # decode step to answer, and the alternative to a clean
                    # acknowledgement is rebooting a healthy 32B.
                    cancel_ack = await _await_shared_future(
                        fut, timeout_s=_LATENT_CANCEL_ACK_GRACE_S
                    )
                except (TimeoutError, BrokenPipeError, OSError):
                    cancel_ack = None
                if self._clean_latent_cancel_ack(
                    cancel_ack,
                    expected_request_id=req_id,
                    expected_request_sha256=expected_request_sha256,
                ):
                    receipt = dict(cancel_ack.get("receipt") or {})
                    progress = dict(self._latent_progress_by_request.get(req_id) or {})
                    logger.warning(
                        "Latent owner deadline reached cleanly: stage=%s "
                        "input_tokens=%s elapsed=%s timings=%s",
                        receipt.get("last_stage") or progress.get("stage") or "unknown",
                        receipt.get("input_token_count")
                        or progress.get("input_tokens")
                        or "unknown",
                        progress.get("elapsed_s") or "unknown",
                        receipt.get("stage_timings_s") or {},
                    )
                    return {
                        **base,
                        "receipt": receipt,
                        "progress": progress,
                        "reason": "latent_timeout:cooperative_cancelled",
                    }
                deferred_reboot = "latent_reason_deadline_unacknowledged"
                return {**base, "reason": "latent_timeout:TimeoutError"}

            if not isinstance(res, dict):
                return {**base, "reason": "invalid_worker_response"}
            raw_receipt = res.get("receipt")
            if raw_receipt is not None and not isinstance(raw_receipt, dict):
                return {**base, "reason": "invalid_worker_receipt"}
            receipt = dict(raw_receipt or {})
            reason = str(res.get("message") or res.get("reason") or "")
            if res.get("requires_worker_recycle") is True or isinstance(
                res.get("state_application_quarantine"),
                dict,
            ):
                deferred_reboot = "latent_integrity:state_application_quarantine"
                return {
                    **base,
                    "receipt": receipt,
                    "state_application_quarantine": dict(
                        res.get("state_application_quarantine") or {}
                    ),
                    "reason": reason or "state_application_quarantine",
                }
            if reason in {
                "checkpoint_invariant_violated",
                "fast_weight_cleanup_unproven",
            }:
                deferred_reboot = f"latent_integrity:{reason}"
            if res.get("status") == "ok":
                from core.brain.llm.latent_cortex.runtime_identity import (
                    collect_latent_runtime_identity,
                    worker_identity_errors,
                )

                receipt_worker_identity = receipt.get("worker_identity")
                identity_errors = worker_identity_errors(
                    receipt_worker_identity,
                    expected=getattr(self, "_worker_identity", {}),
                )
                try:
                    from core.brain.llm.latent_cortex.runtime_integrity import (
                        runtime_integrity_safe,
                    )

                    integrity_safe = runtime_integrity_safe(
                        receipt.get("runtime_integrity"),
                        require_worker=True,
                        expected_episode_id=str(receipt.get("episode_id") or ""),
                        expected_input_tokens_sha256=str(
                            receipt.get("input_tokens_sha256") or ""
                        ),
                        expected_worker_identity=getattr(
                            self,
                            "_worker_identity",
                            {},
                        ),
                        expected_fast_weights_applied=(
                            receipt.get("fast_weights_applied") is True
                        ),
                        expected_checkpoint_fingerprint=str(
                            receipt.get("checkpoint_fingerprint") or ""
                        ),
                        expected_checkpoint_method=str(
                            receipt.get(
                                "checkpoint_fingerprint_method"
                            )
                            or ""
                        ),
                        expected_checkpoint_file_count=receipt.get(
                            "checkpoint_file_count"
                        ),
                    )
                except ImportError:
                    integrity_safe = False
                if not integrity_safe:
                    identity_errors.append("runtime_integrity_unproven")
                if receipt.get("request_payload_sha256") != expected_request_sha256:
                    identity_errors.append("request_payload_sha256_mismatch")
                if identity_errors:
                    deferred_reboot = "latent_integrity:worker_identity_mismatch"
                    return {
                        **base,
                        "receipt": receipt,
                        "reason": "worker_identity_failed:" + ",".join(identity_errors),
                    }
                _seam_early_response = _apply_the_wire_action_intervention(
                    base=base,
                    receipt=receipt,
                    wire_action_intervention=wire_action_intervention,
                    wire_action_policy_evidence=wire_action_policy_evidence,
                    wire_external_execution_offer=wire_external_execution_offer,
                )
                if _seam_early_response is not _SEAM_FELL_THROUGH:
                    return _seam_early_response
                try:
                    identity_remaining = deadline.remaining
                    if identity_remaining is not None and identity_remaining <= 0.0:
                        return {
                            **base,
                            "receipt": receipt,
                            "reason": "runtime_identity_deadline_exhausted",
                        }
                    identity_timeout = min(
                        15.0,
                        max(0.1, float(identity_remaining or 15.0)),
                    )
                    runtime_identity = await asyncio.wait_for(
                        run_io_bound(
                            collect_latent_runtime_identity,
                            _AURA_SOURCE_ROOT,
                        ),
                        timeout=identity_timeout,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    _record_mlx_degradation(
                        exc,
                        action="refused latent success whose runtime identity could not be captured",
                        severity="degraded",
                    )
                    return {
                        **base,
                        "receipt": receipt,
                        "reason": f"runtime_identity_failed:{type(exc).__name__}",
                    }
                receipt["runtime_identity"] = dict(runtime_identity)
                if runtime_identity.get("identity_bound") is not True:
                    return {
                        **base,
                        "receipt": receipt,
                        "reason": "runtime_identity_unbound",
                    }
                # Runtime provenance is the final episode identity available
                # outside the resident worker. Reconstruct, rather than patch,
                # the public DAG so every commitment binds the live envelope.
                from core.brain.llm.latent_cortex.causal_receipt import (
                    build_causal_receipt,
                )

                receipt["causal_receipt"] = build_causal_receipt(receipt)
                action_capture_receipt: dict[str, Any] | None = None
                action_restore_receipt: dict[str, Any] | None = None
                if admitted_action_state_runtime is not None:
                    try:
                        from core.brain.llm.latent_cortex.action_state_capture import (
                            validate_action_state_capture_receipt_public,
                        )
                        from core.brain.llm.latent_cortex.action_state_runtime import (
                            assert_public_runtime_result,
                            validate_action_state_restore_receipt,
                        )

                        raw_capture_receipt = res.get(
                            "action_state_capture_receipt"
                        )
                        if not isinstance(raw_capture_receipt, dict):
                            raise ValueError("action-state capture receipt missing")
                        action_capture_receipt = (
                            validate_action_state_capture_receipt_public(
                                raw_capture_receipt,
                                request=admitted_action_state_runtime.admission.request,
                                trusted_root_public_key_pem=(
                                    admitted_action_state_runtime.trusted_root_public_key_pem
                                ),
                                expected_supervisor_public_key=(
                                    admitted_action_state_runtime.capture_supervisor_public_key
                                ),
                                latent_reason_request=(
                                    admitted_action_state_runtime.latent_reason_request
                                ),
                                model_identity=(
                                    admitted_action_state_runtime.model_identity
                                ),
                                execution_identity=(
                                    admitted_action_state_runtime.execution_identity
                                ),
                                runtime_identity=runtime_identity,
                                expected_campaign_design_sha256=(
                                    admitted_action_state_runtime.admission.payload[
                                        "campaign_design_sha256"
                                    ]
                                ),
                            )
                        )
                        if admitted_action_state_runtime.mode == "restore":
                            raw_restore_receipt = res.get(
                                "action_state_restore_receipt"
                            )
                            if not isinstance(raw_restore_receipt, dict):
                                raise ValueError(
                                    "action-state restore receipt missing"
                                )
                            worker_capture_identity = self.get_worker_identity_snapshot().get(
                                "worker_action_capture_identity"
                            )
                            if not isinstance(worker_capture_identity, Mapping):
                                raise ValueError("worker capture identity missing")
                            action_restore_receipt = (
                                validate_action_state_restore_receipt(
                                    raw_restore_receipt,
                                    capture_receipt=action_capture_receipt,
                                    action_intervention=wire_action_intervention,
                                    runtime_identity=runtime_identity,
                                    expected_worker_public_key_b64=str(
                                        worker_capture_identity.get("public_key_b64")
                                        or ""
                                    ),
                                    expected_supervisor_public_key=(
                                        admitted_action_state_runtime.resident_supervisor_public_key
                                    ),
                                )
                            )
                        assert_public_runtime_result(res)
                    except (
                        ImportError,
                        KeyError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        deferred_reboot = "latent_integrity:action_state_receipt_invalid"
                        return {
                            **base,
                            "receipt": receipt,
                            "reason": "action_state_runtime_receipt_invalid",
                        }
                    if admitted_action_state_runtime.mode == "capture":
                        self._mark_progress()
                        return {
                            "ok": True,
                            "text": "",
                            "receipt": receipt,
                            "action_state_capture_receipt": action_capture_receipt,
                            "progress": dict(
                                self._latent_progress_by_request.get(req_id) or {}
                            ),
                            "reason": "action_state_captured",
                        }
                # CP126 d78cbfa4: a status=ok response used to be coerced with
                # str(value or "") — a missing, empty, list, or mapping answer
                # became ok=true with empty or stringified-container text and
                # bypassed fallback entirely. An episode is successful only
                # when it produced an actual nonempty STRING answer.
                answer = res.get("text")
                if not isinstance(answer, str) or not answer.strip():
                    _record_mlx_degradation(
                        TypeError(
                            "latent_answer_invalid:"
                            f"{type(answer).__name__}:{len(answer) if isinstance(answer, str) else 'n/a'}"
                        ),
                        action="refused latent success for a missing, empty, or non-string answer",
                        severity="degraded",
                    )
                    return {
                        **base,
                        "receipt": receipt,
                        "progress": dict(self._latent_progress_by_request.get(req_id) or {}),
                        "reason": "latent_answer_invalid",
                    }
                # LIVE DEFECT, 2026-08-03. The worker returns the decoded
                # answer token ids alongside the text (LatentReasonResult.
                # to_dict -> "tokens"), and this payload dropped them. The
                # facade then called _receipt_contract_errors with
                # result.get("tokens") == None, and ALL THREE proofs that bind
                # a receipt to the answer require a token list:
                # terminal_disposition, answer_replacement, and
                # fast_weight_learning each raise without it. So every
                # foreground turn failed with
                #   receipt_contract_failed:terminal_disposition_unproven,
                #   answer_replacement_unproven,fast_weight_learning_receipt_unproven
                # and fell back to an ordinary generation. The recurrent lane
                # was inert on the live path — not declining for a reason, but
                # unable to prove anything about an answer whose tokens it was
                # never handed.
                answer_tokens = res.get("tokens")
                self._mark_progress()
                return {
                    "ok": True,
                    "text": answer,
                    "tokens": (
                        list(answer_tokens)
                        if isinstance(answer_tokens, list)
                        else None
                    ),
                    "receipt": receipt,
                    # CP126 f22c4ed8: the facade cannot recompute this digest
                    # — it would have to duplicate the wire normalization
                    # above and would drift. Publishing the digest THIS client
                    # bound the request to lets the facade confirm the binding
                    # happened instead of shape-checking the receipt's own
                    # claim about itself.
                    "request_payload_sha256_bound": expected_request_sha256,
                    # The service consumes this evidence before publishing its result.
                    "answer_replacement_private": res.get("answer_replacement_private"),
                    **(
                        {
                            "action_state_capture_receipt": action_capture_receipt,
                            "action_state_restore_receipt": action_restore_receipt,
                        }
                        if action_capture_receipt is not None
                        else {}
                    ),
                    "progress": dict(self._latent_progress_by_request.get(req_id) or {}),
                    "reason": str(res.get("reason") or ""),
                }
            return {
                **base,
                "receipt": receipt,
                "progress": dict(self._latent_progress_by_request.get(req_id) or {}),
                "reason": reason or "latent_reason_failed",
            }
        except asyncio.CancelledError:
            if fut is not None:
                self.soft_cancel_active_generation("latent_reason_caller_cancelled")
                deferred_reboot = "latent_reason_caller_cancelled"
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            deferred_reboot = f"latent_ipc_failed:{type(exc).__name__}"
            _record_mlx_degradation(
                exc,
                action="recycled resident worker after latent_reason IPC failure",
                severity="warning",
            )
            return {**base, "reason": f"latent_ipc_failed:{type(exc).__name__}"}
        finally:
            try:
                try:
                    if fut is not None:
                        await asyncio.shield(
                            self._finish_generation_ownership(
                                req_id,
                                fut,
                                None,
                                release_lane=not bool(deferred_reboot),
                            )
                        )
                finally:
                    if deferred_reboot:
                        await asyncio.shield(
                            self.reboot_worker(
                                reason=deferred_reboot,
                                mark_failed=False,
                            )
                        )
                    elif lane_fenced and fut is None and self._active_generations <= 0:
                        await asyncio.shield(self._set_durable_lane_preemptible(True))
            finally:
                self._latent_progress_by_request.pop(req_id, None)
                self._release_request_lock()
                if foreground_owner_cm is not None:
                    await foreground_owner_cm.__aexit__(None, None, None)

    async def reload_model_artifact(self, model_path: str) -> dict[str, Any]:
        """Serve a newly published fused artifact by re-pointing this lane.

        The model lives in the WORKER process, so the only correct swap is a
        worker recycle with the new path. (This replaces a retired
        live_learner monkey-patch that loaded a second full copy of the model
        into the ORCHESTRATOR process — ~20GB of wired memory on the 32B lane
        — while generations kept flowing through the worker's old weights.)
        Busy lanes defer the recycle until the active request finishes; the
        respawn path re-resolves the fused manifest, so crash recovery after
        the swap also serves the promoted artifact.
        """
        resolved = await asyncio.to_thread(lambda: Path(str(model_path or "")).expanduser())
        previous = self.model_path
        verdict = await asyncio.to_thread(_validate_model_artifact, resolved, previous)
        if not verdict.ok:
            logger.warning(
                "🧬 [MLX] Refused artifact promotion for %s: %s", resolved.name, verdict.reason
            )
            return {
                "ok": False,
                "state": "rejected",
                "reason": verdict.reason,
                "previous": previous,
                "artifact": str(resolved),
                **verdict.as_receipt(),
            }

        if (
            int(getattr(self, "_active_generations", 0) or 0) > 0
            or self._current_request_started_at > 0.0
        ):
            # CP126 8ccdcd3b: model_path used to change HERE, so for the rest
            # of the active request the old worker kept decoding old weights
            # while status, logging and admission all described it as the new
            # model. The desired artifact is now held separately and published
            # only when a worker is actually serving it.
            #
            # CP126 7f4435f5: the promotion also used to be stored in
            # _deferred_reboot_reason, the same scalar first-token, token-stall,
            # heartbeat and fence-loss recovery write to. Whichever fired last
            # won, so a generation failure could silently discard a staged
            # promotion. It is a separate intent now.
            self._pending_promotion = str(resolved)
            logger.info(
                "🧬 [MLX] Promoted artifact staged for %s; activating after the active request.",
                resolved.name,
            )
            return {
                "ok": True,
                "state": "staged",
                "mode": "deferred",
                "previous": previous,
                "artifact": str(resolved),
                **verdict.as_receipt(),
            }
        return await self._activate_promoted_artifact(str(resolved), verdict=verdict)

    async def _activate_promoted_artifact(
        self, target: str, *, verdict: ArtifactVerdict | None = None
    ) -> dict[str, Any]:
        """Publish a validated artifact as this lane's serving identity.

        CP126 41fa9f3c: the old path awaited ``reboot_worker`` — a TEARDOWN —
        then logged "Promoted artifact live" and returned ok=true. Nothing had
        spawned, handshaked, or loaded a single weight. The lane was in fact
        unloaded, and the first caller after the promotion paid the cold start
        and discovered any load failure. The receipt now names the state it
        can actually prove: ``unloaded`` after a clean recycle, ``failed`` if
        the recycle did not complete. Only ``_promotion_is_serving`` reports
        ``ready``, and only after a worker answers on the new path.
        """
        previous = self.model_path
        proof = verdict.as_receipt() if verdict is not None else {}
        try:
            from core.brain.llm.model_registry import get_model_runtime_assignment

            next_assignment = get_model_runtime_assignment(target)
            if next_assignment.role != self.runtime_assignment.role:
                raise RuntimeError(
                    "promoted_artifact_runtime_role_changed:"
                    f"{self.runtime_assignment.role}->{next_assignment.role}"
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="refused promoted artifact without a source-bound runtime assignment",
                severity="error",
            )
            return {
                "ok": False,
                "state": "rejected",
                "reason": str(exc),
                "previous": previous,
                "artifact": target,
                **proof,
            }
        self.runtime_assignment = next_assignment
        self.model_path = next_assignment.model_path
        self._expert_adapter_path = None  # adapters belong to the old weights
        self._pending_promotion = None
        try:
            await self.reboot_worker(reason="promoted_artifact_swap", mark_failed=False)
        except Exception as exc:  # noqa: BLE001 - a failed recycle must be reported, not raised
            _record_mlx_degradation(
                exc,
                action="left the promoted artifact as the serving identity after recycle failed",
                severity="error",
            )
            return {
                "ok": False,
                "state": "failed",
                "reason": f"recycle_failed:{type(exc).__name__}",
                "previous": previous,
                "artifact": target,
                **proof,
            }
        _rebind_client_registry_key(previous, self.model_path, self)
        logger.info(
            "🧬 [MLX] Promoted artifact is now this lane's serving identity: %s "
            "(worker unloaded; it loads on next use).",
            Path(target).name,
        )
        return {
            "ok": True,
            "state": "unloaded",
            "mode": "recycled",
            "previous": previous,
            "artifact": target,
            **proof,
        }

    def consume_deliberate_no_text_reason(self) -> str:
        """Return (and clear) why this client last produced no text on purpose.

        A healthy worker whose generation we cancelled at the turn budget
        returns nothing. Callers that score endpoint health must be able to
        tell that apart from a client that is actually broken, or they open the
        Cortex circuit over our own deferral and the NEXT turn loses the real
        mind too (observed live 2026-07-26: one 0.7s budget overrun opened the
        circuit and the reply became bounded filler).

        One-shot by design: reading it clears it, so a later empty result that
        failed for its own reasons is never excused by this one.
        """
        reason = self._deliberate_no_text_reason or ""
        self._deliberate_no_text_reason = None
        return reason

    def _mark_healthy_generation_deadline(self, *, foreground_request: bool) -> None:
        """Publish a non-damaging no-text outcome and fence abandoned output."""
        self._deliberate_no_text_reason = "generation_deadline_worker_healthy"
        if foreground_request:
            self.soft_cancel_active_generation("abandoned_generation_deadline")

    async def _renew_durable_lane_lease(
        self,
        controller: Any,
        owner_id: str,
        fencing_token: int,
    ) -> bool:
        """Renew the model-lane lease, distinguishing "no" from "never asked".

        LIVE DEFECT, 2026-08-10. A user turn came back "I couldn't get to an
        answer I'd stand behind on that one." The health pulse for that second:

            conversation_lane: cold (worker_not_alive)
            inference_gate (is_inference_ready() returned False)
            event_loop_monitor.last_lag_s 10.46 >= 5.00
            mlx_client (critical): TimeoutError
                → stopped MLX worker after durable lane heartbeat failed

        Nothing was wrong with the worker or the lease. The event loop had
        been blocked for ten seconds, so this renewal — awaited with a five
        second budget measured ON that loop — could not be scheduled at all
        before its timer fired. The resulting TimeoutError was read as
        `model_lane_fence_lost`, which killed a healthy 32B worker, failed
        every in-flight turn, and cost ~35s to reload.

        A timeout taken on a starved loop is not evidence about the lease. It
        is the absence of an answer being recorded as a negative one — the same
        shape as a skipped check reported as a failed check.

        So on timeout, ask exactly once more. If the loop was merely blocked,
        the retry runs on a loop that is moving again and answers truthfully.
        If the lease is genuinely gone, the retry says so and the caller kills
        the worker as before. One retry, not a loop: a second timeout is
        itself evidence that the loop is not recovering, and a foreground turn
        cannot wait forever to find out.

        Deliberately no lag threshold anywhere in here. Asking again is
        cheaper, more direct, and does not need a number that would have to be
        tuned per host.
        """
        try:
            return bool(
                await asyncio.wait_for(
                    controller.heartbeat_owner(owner_id, fencing_token=fencing_token),
                    timeout=_LEASE_RENEWAL_TIMEOUT_S,
                )
            )
        except TimeoutError:
            logger.warning(
                "⏱️ [MLX] Lane lease renewal timed out; re-asking once before "
                "treating the lane as lost (a blocked event loop cannot "
                "distinguish a dead lease from an unasked question)."
            )
        try:
            alive = bool(
                await asyncio.wait_for(
                    controller.heartbeat_owner(owner_id, fencing_token=fencing_token),
                    timeout=_LEASE_RENEWAL_TIMEOUT_S,
                )
            )
        except TimeoutError:
            # Second timeout. Before killing anything, ask the one question
            # that is actually about the worker: is the process alive?
            #
            # A lease renewal talks to the lane CONTROLLER — a file lock and a
            # small database. Its silence is evidence about that controller, or
            # about an event loop too busy to run it. It is not evidence about
            # a 32B process that the operating system says is running and whose
            # response pipe is healthy. Killing it for a controller stall is a
            # category error, and an expensive one: the reload takes ~35s, the
            # reload itself blocks the loop, and the next renewal then times out
            # the same way. That is the spiral observed live — "Primary 32B
            # cortex is dead. Triggering background respawn (Attempt 1/5)" on a
            # worker that had never stopped answering.
            if self.is_alive():
                _record_mlx_degradation(
                    TimeoutError("lane_lease_renewal_unanswered_worker_alive"),
                    action=(
                        "lane lease renewal did not answer twice, but the worker "
                        "process is alive and its pipe is healthy; kept it and "
                        "left the lease to the next heartbeat"
                    ),
                    severity="warning",
                )
                return True
            raise
        if alive:
            logger.info(
                "⏱️ [MLX] Lane lease renewal recovered after one timeout; "
                "the re-ask found the lease intact and the worker stayed live."
            )
        return alive

    def _schedule_durable_lane_renewal(
        self,
        controller: Any,
        owner_id: str,
        fencing_token: int,
        queue_generation: int,
    ) -> None:
        """Renew one durable lease without blocking the response consumer."""
        task = self._lane_renewal_task
        if task is not None and not task.done():
            return
        task = get_task_tracker().create_task(
            self._renew_durable_lane_lease_in_background(
                controller,
                owner_id,
                fencing_token,
                queue_generation,
            ),
            name=f"MLXLeaseRenewal:{owner_id}:{fencing_token}",
        )
        self._lane_renewal_task = task

        def _clear(completed: asyncio.Task) -> None:
            if self._lane_renewal_task is completed:
                self._lane_renewal_task = None

        task.add_done_callback(_clear)

    def _cancel_lane_renewal_task(self) -> None:
        task, self._lane_renewal_task = self._lane_renewal_task, None
        if task is not None and not task.done():
            _cancel_task_threadsafe(task)

    async def _renew_durable_lane_lease_in_background(
        self,
        controller: Any,
        owner_id: str,
        fencing_token: int,
        queue_generation: int,
    ) -> None:
        """Renew and reconcile an exact owner generation.

        The response listener must remain dedicated to IPC drainage.  A stale
        renewal result is harmless unless the same owner and fencing token are
        still authoritative when the result is applied.
        """
        try:
            lease_alive = await self._renew_durable_lane_lease(
                controller,
                owner_id,
                fencing_token,
            )
            if not lease_alive:
                raise RuntimeError("model_lane_fence_lost")
        except asyncio.CancelledError:
            raise
        except (
            OSError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            TimeoutError,
        ) as exc:
            current_owner, current_token, _receipt_id = self._durable_model_lane_owner_snapshot()
            if current_owner != owner_id or current_token != fencing_token:
                logger.info(
                    "Ignored stale MLX lease result for retired owner=%s token=%s",
                    owner_id,
                    fencing_token,
                )
                return
            _record_mlx_degradation(
                exc,
                action="stopped MLX worker after durable lane heartbeat failed",
                severity="critical",
            )
            self._deferred_reboot_reason = "model_lane_fence_lost"
            # Fail every waiter before dropping the process handle.  Otherwise
            # callers can miss both the dead-process and pending-future tests.
            for req_id, pending in list(self._pending_generations.items()):
                if pending is not None and not pending.done():
                    _set_shared_future_result(
                        pending,
                        {
                            "status": "error",
                            "action": "generate",
                            "id": str(req_id),
                            "message": "model_lane_fence_lost",
                        },
                    )
            current_fut = self._current_gen_future
            if current_fut is not None and not current_fut.done():
                _set_shared_future_result(
                    current_fut,
                    {
                        "status": "error",
                        "action": "generate",
                        "id": self._current_request_id,
                        "message": "model_lane_fence_lost",
                    },
                )
            if self._init_future is not None and not self._init_future.done():
                _cancel_shared_future(self._init_future)
            self._pending_generations.clear()
            self._current_gen_future = None
            self._active_generations = 0
            self._release_detached_request_lock()
            self._clear_detached_worker_requests()

            process, self._process = self._process, None
            if process is not None:
                await asyncio.to_thread(
                    self._release_worker_process, process, reason="model_lane_fence_lost"
                )
            from core.runtime.model_lane_control import unregister_model_lane_owner_adapter

            unregister_model_lane_owner_adapter(owner_id)
            with self._model_lane_state_lock:
                if self._model_lane_fencing_token == fencing_token:
                    self._model_lane_fencing_token = 0
                    self._model_lane_terminal_receipt_id = ""
            self._set_lane_state("cold", "model_lane_fence_lost")
            self._listener_stop_generation = queue_generation

    async def encode_hidden(
        self, texts: Sequence[str], *, timeout_s: float = 8.0
    ) -> list[list[float]]:
        """The resident model's own representation of these sentences.

        For a learned decision surface, not for generation: one causal forward
        with no sampling, so there is nothing to steer.

        Returns [] rather than waiting whenever the worker is not resident or
        is busy with a foreground turn. Every caller treats [] as "no opinion",
        which is what keeps this off the critical path.
        """
        wanted = [str(text or "") for text in (texts or []) if str(text or "").strip()]
        if not wanted:
            return []
        # getattr throughout: this is called from matcher code that may hold a
        # client constructed outside a running worker, and a missing attribute
        # is the same answer as a busy one — no opinion.
        process = getattr(self, "_process", None)
        refusal = ""
        if getattr(self, "_shutting_down", False):
            refusal = "shutting_down"
        elif not getattr(self, "_init_done", False):
            refusal = "worker_not_initialised"
        elif process is None or not process.is_alive():
            refusal = "worker_not_resident"
        elif _foreground_owner_active():
            refusal = "foreground_active"
        elif int(getattr(self, "_active_generations", 0) or 0) > 0:
            refusal = "foreground_busy"
        elif getattr(self, "_warmup_in_flight", False):
            refusal = "worker_warming"
        if refusal:
            # Naming the refusal, because "returned nothing" is the one
            # description that sends the next investigation somewhere else.
            logger.info("🔤 [ENCODE] declined: %s", refusal)
            return []

        # The checks above are observations, not ownership.  Before this lock,
        # two background readers (or a reader and a foreground generation)
        # could both observe an idle worker and enqueue.  One then spent its
        # entire deadline behind the other and logged a misleading worker
        # failure.  Hidden-state reads are optional, so they never wait for the
        # lane: either this read owns the worker now or it has no opinion.
        request_id = uuid.uuid4().hex
        owner_label = "model_hidden_features"
        if not self._try_acquire_request_lock(
            owner_label=owner_label,
            owner_token=request_id,
        ):
            logger.info("🔤 [ENCODE] declined: request_lane_busy")
            return []

        future: SharedFuture | None = None
        detached = False
        lane_protected = False
        response: Any = None
        try:
            # Recheck every mutable admission fact after taking ownership.
            # A foreground turn may have arrived between the first observation
            # and the non-blocking lock acquisition.
            process = getattr(self, "_process", None)
            if _foreground_owner_active():
                logger.info("🔤 [ENCODE] declined: foreground_active_after_lane")
                return []
            if (
                getattr(self, "_shutting_down", False)
                or not getattr(self, "_init_done", False)
                or process is None
                or not process.is_alive()
                or int(getattr(self, "_active_generations", 0) or 0) > 0
                or getattr(self, "_warmup_in_flight", False)
            ):
                logger.info("🔤 [ENCODE] declined: worker_changed_after_lane")
                return []
            if not await self._set_durable_lane_preemptible(False):
                logger.info("🔤 [ENCODE] declined: model_lane_fence_lost")
                return []
            lane_protected = True
            if _foreground_owner_active():
                logger.info("🔤 [ENCODE] declined: foreground_active_after_fence")
                return []

            self._job_seq_counter += 1
            request = {
                "id": request_id,
                "seq": self._job_seq_counter,
                "action": "encode_hidden",
                "texts": wanted[:64],
            }
            future = _new_shared_future()
            self._pending_generations[request_id] = future
            self._current_gen_future = future
            self._active_generations += 1
            self._active_generation_started_at = time.time()
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(request, principal="mlx_client.encode_hidden"),
                True,
                2.0,
            )
            response = await _await_shared_future(future, timeout_s=max(1.0, timeout_s))
        except (TimeoutError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.info("🔤 [ENCODE] failed: %s: %s", type(exc).__name__, str(exc)[:160])
            if isinstance(exc, TimeoutError) and future is not None:
                # The caller's patience is not evidence that the worker failed.
                # Keep ownership with the request until the persistent listener
                # sees its exact terminal frame. A foreground request can still
                # invoke the measured preemption ladder if this work truly stalls.
                detached = self._register_detached_worker_request(
                    request_id,
                    future,
                    owner_label=owner_label,
                )
                if detached:
                    logger.info(
                        "🔤 [ENCODE] caller detached from request %s; worker ownership remains fenced.",
                        request_id[:12],
                    )
                    return []
                try:
                    response = future.result()
                except (
                    cfutures.CancelledError,
                    cfutures.InvalidStateError,
                    asyncio.CancelledError,
                ):
                    return []
            else:
                return []
        finally:
            try:
                if future is not None and not detached:
                    await asyncio.shield(
                        self._finish_generation_ownership(
                            request_id,
                            future,
                            None,
                        )
                    )
                elif lane_protected and not detached:
                    released = await asyncio.shield(
                        self._set_durable_lane_preemptible(True)
                    )
                    if not released:
                        self._durable_lane_release_owed = True
            finally:
                if not detached:
                    self._release_request_lock_if_owned(
                        owner_label=owner_label,
                        owner_token=request_id,
                    )
        if not isinstance(response, dict) or response.get("status") != "ok":
            logger.info(
                "🔤 [ENCODE] worker said: %s",
                str(response)[:200] if response is not None else "nothing",
            )
            return []
        vectors = response.get("vectors")
        return vectors if isinstance(vectors, list) else []

    async def encode_hidden_sequence(
        self,
        text: str,
        *,
        timeout_s: float = 8.0,
        representation: str = "final_hidden_v1",
    ) -> dict[str, Any] | None:
        """Return token-level resident hidden states without generating text.

        ``None`` means the optional observation could not own the resident
        lane immediately. Invalid input or an invalid worker response raises;
        a caller must not confuse corrupt neural evidence with backpressure.
        """

        if type(text) is not str:
            raise TypeError("hidden sequence text must be a string")
        if not text:
            raise ValueError("hidden sequence text must not be empty")
        if len(text) > _HIDDEN_SEQUENCE_MAX_INPUT_CHARS:
            raise ValueError(
                "hidden sequence input exceeds "
                f"{_HIDDEN_SEQUENCE_MAX_INPUT_CHARS} characters"
            )
        from core.brain.llm.hidden_sequence_contract import (
            HIDDEN_SEQUENCE_REPRESENTATIONS,
        )

        if representation not in HIDDEN_SEQUENCE_REPRESENTATIONS:
            raise ValueError(
                f"unsupported hidden sequence representation: {representation}"
            )

        process = getattr(self, "_process", None)
        refusal = ""
        if getattr(self, "_shutting_down", False):
            refusal = "shutting_down"
        elif not getattr(self, "_init_done", False):
            refusal = "worker_not_initialised"
        elif process is None or not process.is_alive():
            refusal = "worker_not_resident"
        elif _foreground_owner_active():
            refusal = "foreground_active"
        elif int(getattr(self, "_active_generations", 0) or 0) > 0:
            refusal = "foreground_busy"
        elif getattr(self, "_warmup_in_flight", False):
            refusal = "worker_warming"
        if refusal:
            logger.info("Hidden-sequence read declined: %s", refusal)
            return None

        request_id = uuid.uuid4().hex
        action = "encode_hidden_sequence"
        owner_label = "model_hidden_sequence"
        if not self._try_acquire_request_lock(
            owner_label=owner_label,
            owner_token=request_id,
        ):
            logger.info("Hidden-sequence read declined: request_lane_busy")
            return None

        future: SharedFuture | None = None
        detached = False
        lane_protected = False
        response: Any = None
        try:
            process = getattr(self, "_process", None)
            if _foreground_owner_active():
                logger.info("Hidden-sequence read declined: foreground_active_after_lane")
                return None
            if (
                getattr(self, "_shutting_down", False)
                or not getattr(self, "_init_done", False)
                or process is None
                or not process.is_alive()
                or int(getattr(self, "_active_generations", 0) or 0) > 0
                or getattr(self, "_warmup_in_flight", False)
            ):
                logger.info("Hidden-sequence read declined: worker_changed_after_lane")
                return None
            if not await self._set_durable_lane_preemptible(False):
                logger.info("Hidden-sequence read declined: model_lane_fence_lost")
                return None
            lane_protected = True
            if _foreground_owner_active():
                logger.info("Hidden-sequence read declined: foreground_active_after_fence")
                return None

            self._job_seq_counter += 1
            request = {
                "id": request_id,
                "seq": self._job_seq_counter,
                "action": action,
                "text": text,
                "representation": representation,
            }
            future = _new_shared_future()
            self._pending_generations[request_id] = future
            self._current_gen_future = future
            self._active_generations += 1
            self._active_generation_started_at = time.time()
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(
                    request,
                    principal="mlx_client.encode_hidden_sequence",
                ),
                True,
                2.0,
            )
            response = await _await_shared_future(future, timeout_s=max(1.0, timeout_s))
        except TimeoutError:
            if future is None:
                return None
            detached = self._register_detached_worker_request(
                request_id,
                future,
                owner_label=owner_label,
            )
            if detached:
                return None
            try:
                response = future.result()
            except (
                cfutures.CancelledError,
                cfutures.InvalidStateError,
                asyncio.CancelledError,
            ):
                return None
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"hidden sequence IPC failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            try:
                if future is not None and not detached:
                    await asyncio.shield(
                        self._finish_generation_ownership(request_id, future, None)
                    )
                elif lane_protected and not detached:
                    released = await asyncio.shield(
                        self._set_durable_lane_preemptible(True)
                    )
                    if not released:
                        self._durable_lane_release_owed = True
            finally:
                if not detached:
                    self._release_request_lock_if_owned(
                        owner_label=owner_label,
                        owner_token=request_id,
                    )

        if not isinstance(response, dict):
            raise RuntimeError("hidden sequence worker response is not an object")
        if response.get("id") != request_id or response.get("action") != action:
            raise RuntimeError("hidden sequence worker response identity mismatch")
        if response.get("status") != "ok":
            message = str(response.get("message") or "worker rejected request")
            raise RuntimeError(f"hidden sequence worker error: {message[:240]}")
        return self._validate_hidden_sequence_response(
            text,
            response,
            representation=representation,
        )

    def _validate_hidden_sequence_response(
        self,
        text: str,
        response: Mapping[str, Any],
        *,
        representation: str = "final_hidden_v1",
    ) -> dict[str, Any]:
        token_ids = response.get("token_ids")
        hidden_state_bytes = response.get("hidden_state_bytes")
        hidden_shape = response.get("hidden_shape")
        hidden_dtype = response.get("hidden_dtype")
        receipt = response.get("receipt")
        if not isinstance(token_ids, list) or not (
            1 <= len(token_ids) <= _HIDDEN_SEQUENCE_MAX_TOKENS
        ):
            raise RuntimeError("hidden sequence token ids are malformed")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise RuntimeError("hidden sequence token ids are malformed")
        if not isinstance(receipt, dict):
            raise RuntimeError("hidden sequence receipt is missing")

        hidden_size = receipt.get("hidden_size")
        if type(hidden_size) is not int or not (
            1 <= hidden_size <= _HIDDEN_SEQUENCE_MAX_WIDTH
        ):
            raise RuntimeError("hidden sequence width is malformed")
        expected_shape = [len(token_ids), hidden_size]
        if hidden_shape != expected_shape or hidden_dtype != "float32_le":
            raise RuntimeError("hidden sequence shape or dtype is malformed")
        if not isinstance(hidden_state_bytes, bytes):
            raise RuntimeError("hidden sequence payload is not packed bytes")
        expected_bytes = len(token_ids) * hidden_size * 4
        if len(hidden_state_bytes) != expected_bytes:
            raise RuntimeError("hidden sequence payload length is malformed")

        import numpy as np

        hidden_states = np.frombuffer(hidden_state_bytes, dtype="<f4").reshape(
            len(token_ids), hidden_size
        )
        if not np.all(np.isfinite(hidden_states)):
            raise RuntimeError("hidden sequence payload is not finite")
        norms = np.linalg.norm(hidden_states, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-4):
            raise RuntimeError("hidden sequence payload is not unit normalized")

        expected_limits = {
            "max_input_chars": _HIDDEN_SEQUENCE_MAX_INPUT_CHARS,
            "max_tokens": _HIDDEN_SEQUENCE_MAX_TOKENS,
            "max_hidden_size": _HIDDEN_SEQUENCE_MAX_WIDTH,
        }
        from core.brain.llm.hidden_sequence_contract import (
            hidden_sequence_channels,
            hidden_sequence_schema,
        )
        from core.brain.llm.latent_cortex.runtime_identity import worker_model_basis

        expected_identity = worker_model_basis(self.get_worker_identity_snapshot())
        expected_receipt = {
            "schema": hidden_sequence_schema(representation),
            "request_id": response.get("id"),
            "action": "encode_hidden_sequence",
            "input_char_count": len(text),
            "token_count": len(token_ids),
            "hidden_size": hidden_size,
            "hidden_state_bytes": expected_bytes,
            "hidden_state_sha256": hashlib.sha256(hidden_state_bytes).hexdigest(),
            "transport": "packed_float32_le",
            "limits": expected_limits,
            "model_basis": expected_identity,
            "representation": representation,
            "channels": list(hidden_sequence_channels(representation)),
            "forward_passes": 1,
            "causal_full_sequence": True,
            "sampling": False,
            "generated_tokens": 0,
            "generated_text": False,
        }
        if receipt != expected_receipt:
            raise RuntimeError(
                "hidden sequence receipt does not match the request or model basis"
            )
        return {
            "token_ids": list(token_ids),
            "hidden_states": hidden_states.copy(),
            "receipt": copy.deepcopy(receipt),
        }

    def soft_cancel_active_generation(
        self, reason: str = "foreground_preemption"
    ) -> dict[str, Any]:
        """Ask the ACTIVE generation to stop between tokens (cooperative).

        Writes the active job's sequence number into shared memory; the worker
        token loop polls it each decode step and finishes the job early with a
        ``soft_cancelled`` response. Cancel latency is roughly one decode step
        and the worker (and its loaded model) stays warm — the cheap first
        rung on the preemption ladder before ``force_abort_active_generation``.

        Returns a receipt; ``requested`` is False when there is nothing to
        cancel or the cancel channel is unavailable.
        """
        reason = str(reason or "foreground_preemption")
        cancel_seq = getattr(self, "_cancel_seq", None)
        active_seq = int(getattr(self, "_current_request_seq", 0) or 0)
        generation_active = bool(self._current_request_started_at > 0.0 and active_seq > 0)
        if cancel_seq is None or not generation_active:
            return {
                "requested": False,
                "reason": reason,
                "active_seq": active_seq,
                "detail": "no_active_generation"
                if cancel_seq is not None
                else "cancel_channel_unavailable",
            }
        try:
            # CP126 2656d71d. The cancel channel is a lock-free shared word by
            # design — the worker polls it every decode step and a semaphore
            # there would cost a token loop and could deadlock on a dying
            # worker. What was missing is that a WRITE could be silently lost:
            # a concurrent job start clears a superseded sequence, and the
            # parent still reported the cancel as requested. Write, read back,
            # and say what actually happened.
            cancel_seq.value = active_seq
            written = int(getattr(cancel_seq, "value", 0))
            if written != active_seq:
                cancel_seq.value = active_seq
                written = int(getattr(cancel_seq, "value", 0))
            if written != active_seq:
                _record_mlx_degradation(
                    RuntimeError(
                        f"cooperative cancel for seq {active_seq} was overwritten "
                        f"by {written} before the worker could observe it"
                    ),
                    action="reported the soft cancel as not requested rather than assuming delivery",
                    severity="warning",
                )
                return {
                    "requested": False,
                    "reason": reason,
                    "active_seq": active_seq,
                    "detail": "cancel_write_lost",
                }
            # CP126 6b4337de: record WHAT was cancelled and WHEN. The old ack
            # test read a shared flag and a heartbeat, neither bound to this
            # request; the worker's own terminal frame for this job is the
            # only thing that proves it stopped decoding it.
            self._soft_cancel_target = {
                "req_id": str(getattr(self, "_current_request_id", "") or ""),
                "seq": active_seq,
                "requested_monotonic": time.monotonic(),
                "reason": reason,
            }
            self._soft_cancel_ack = None
        except (OSError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="fell back to force-abort ladder after soft-cancel write failed",
                severity="warning",
            )
            return {
                "requested": False,
                "reason": reason,
                "active_seq": active_seq,
                "detail": f"cancel_write_failed:{type(exc).__name__}",
            }
        logger.info(
            "✋ [MLX] Soft-cancel requested for job seq=%d on %s (%s).",
            active_seq,
            os.path.basename(self.model_path),
            reason,
        )
        return {"requested": True, "reason": reason, "active_seq": active_seq}

    # Deferred-reboot reasons (after the recoverable_ prefix is stripped) that
    # follow a soft-cancel and are therefore eligible for warm-lane
    # preservation when the worker acknowledges the cancel.
    _SOFT_CANCEL_PRESERVABLE_REASONS = frozenset(
        {
            "first_token_sla_exceeded",
            "token_progress_stalled",
            "generation_deadline_reached",
        }
    )

    async def _soft_cancel_acknowledged(self, timeout_s: float | None = None) -> bool:
        """Wait (bounded) for the worker to acknowledge a soft-cancel.

        Acknowledgement = the worker cleared the shared cancel flag (it
        demonstrably passed through its token loop) while staying alive with
        fresh heartbeats. When this returns True the orphaned generation has
        already been dropped worker-side — late text cannot bleed into the
        next turn because its request id is no longer pending — so the warm
        model can be preserved instead of paying a ~60-90s reload.
        """
        if timeout_s is None:
            timeout_s = _env_duration_s("AURA_MLX_SOFT_CANCEL_ACK_WAIT_S", 12.0, minimum=0.5)
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            timeout_s = 12.0
        if not math.isfinite(timeout_s):
            # An infinite ack wait would wedge cleanup forever.
            timeout_s = 12.0
        target = self._soft_cancel_target
        if not target:
            # Nothing was cancelled through this channel, so there is nothing
            # to acknowledge. Reporting True here would preserve a worker on
            # the strength of a cancel that never happened.
            return False
        deadline = time.monotonic() + min(120.0, max(0.5, timeout_s))
        while time.monotonic() < deadline:
            process = self._process
            if process is None or not process.is_alive():
                return False
            if self._soft_cancel_ack_matches(target):
                return True
            await asyncio.sleep(0.25)
        return False

    def _soft_cancel_ack_matches(self, target: dict[str, Any]) -> bool:
        """Whether the worker answered THIS cancel, after it was written.

        The previous test was ``cancel flag is zero AND some heartbeat is
        under 20 seconds old``. Neither signal named the cancelled request:
        a reboot or a new job start also clears the flag, and a heartbeat
        emitted BEFORE the cancel still counted as fresh. A worker that never
        dropped the abandoned decode could therefore be preserved, and its
        stale output was then live in the next turn's lane.
        """
        ack = self._soft_cancel_ack
        if not ack:
            return False
        if str(ack.get("req_id") or "") != str(target.get("req_id") or ""):
            return False
        return float(ack.get("observed_monotonic") or 0.0) >= float(
            target.get("requested_monotonic") or 0.0
        )

    def _note_soft_cancel_acknowledgement(self, res: dict[str, Any]) -> None:
        """Record a worker terminal frame that answers an outstanding cancel.

        A terminal frame for the cancelled request is proof the worker's job
        loop reached the end of that job — whether it reports soft_cancelled,
        an error, or a truncated result. What matters is that the decode this
        cancel targeted is over.
        """
        target = self._soft_cancel_target
        if not target:
            return
        req_id = str(res.get("id") or "")
        if not req_id or req_id != str(target.get("req_id") or ""):
            return
        self._soft_cancel_ack = {
            "req_id": req_id,
            "observed_monotonic": time.monotonic(),
            "soft_cancelled": bool(res.get("soft_cancelled")),
            "status": str(res.get("status") or ""),
        }

    async def _resolve_deferred_reboot(self, deferred: str) -> None:
        """Resolve an abandoned-request verdict: preserve the warm lane or reboot.

        Historically EVERY abandoned request recycled the worker ("so late
        text cannot bleed into the next turn") — a full model reload during
        which arriving turns died, observed live as soak death-clusters. The
        soft-cancel channel already isolates the orphaned output; what was
        missing is verifying the worker actually observed the cancel. Now:
        recoverable abandons keep the warm worker when the cancel is
        acknowledged, and only unacknowledged (truly wedged) workers reboot.
        """
        recoverable = deferred.startswith("recoverable_")
        reason = deferred.removeprefix("recoverable_")
        if recoverable and reason in self._SOFT_CANCEL_PRESERVABLE_REASONS:
            if await self._soft_cancel_acknowledged():
                logger.info(
                    "♻️✅ [MLX] Worker acknowledged soft-cancel after %s — warm lane preserved, no reboot.",
                    reason,
                )
                self._record_degraded_event(
                    "warm_lane_preserved_after_soft_cancel",
                    detail=f"{os.path.basename(self.model_path)}:{reason}",
                    severity="warning",
                    foreground_request=False,
                )
                return
            logger.warning(
                "🛑 [MLX] No soft-cancel acknowledgement after %s — worker presumed wedged; rebooting.",
                reason,
            )
        await self.reboot_worker(reason=reason, mark_failed=not recoverable)

    def force_abort_active_generation(
        self,
        reason: str = "hard_generation_deadline",
        *,
        expected_request_id: str = "",
        expected_request_seq: int = 0,
    ) -> bool:
        """Thread-safe emergency abort for a wedged generation.

        Normal cancellations should flow through ``reboot_worker``. This path is
        intentionally synchronous so an external watchdog can break a stuck
        proof or foreground request even when the caller's event loop is waiting
        on an MLX future that failed to observe its deadline.
        """
        reason = str(reason or "hard_generation_deadline")
        pending_by_request = {
            str(req_id): future
            for req_id, future in list(self._pending_generations.items())
            if future is not None and not future.done()
        }
        current_future = self._current_gen_future
        had_active_request = bool(
            pending_by_request
            or (current_future is not None and not current_future.done())
            or self._active_generations > 0
            or self._current_request_started_at > 0.0
        )
        process = self._process
        had_process = bool(process is not None and process.is_alive())
        if not had_active_request and not had_process:
            return False

        # CP126 ccb125e0. A caller that knows WHICH generation it is chasing
        # can say so, and this refuses to kill anything else. Without it the
        # only thing standing between a stale watchdog and a healthy resident
        # model was an arbitrary reason string.
        if expected_request_id:
            active_id = str(getattr(self, "_current_request_id", "") or "")
            if active_id != str(expected_request_id):
                logger.info(
                    "🛈 [MLX] Abort for %s targets request %s but %s is running; "
                    "leaving the worker alone.",
                    os.path.basename(self.model_path),
                    str(expected_request_id)[:12],
                    active_id[:12] or "nothing",
                )
                return False
        if expected_request_seq:
            active_seq = int(getattr(self, "_current_request_seq", 0) or 0)
            if active_seq != int(expected_request_seq):
                logger.info(
                    "🛈 [MLX] Abort for %s targets generation seq %d but seq %d is "
                    "running; leaving the worker alone.",
                    os.path.basename(self.model_path),
                    int(expected_request_seq),
                    active_seq,
                )
                return False

        if not had_active_request:
            # The generation this abort was chasing already finished. That is a
            # race the timeout will always sometimes lose, not damage: the live
            # 2026-07-25 capability run recorded
            # force_abort_without_active_request:inference_gate_generation_timeout:Reflex:14.4s
            # as a degradation AND a MARGINAL fault, and then killed a healthy
            # idle worker — buying a cold reload for a turn that had already
            # been answered.
            #
            # Nothing to abort is a no-op, and a worker with no work is not a
            # worker to kill.
            # CP126 ccb125e0. This used to consult _ABORT_RACE_MARKERS_RE — a
            # LEXICAL test on the reason string — and killed the idle worker
            # whenever the phrasing missed. So whether a healthy resident 20GB
            # model survived depended on how a caller worded its reason, which
            # is not a property of the worker or of the work.
            #
            # There is nothing to abort. That is the whole fact, and it is true
            # regardless of why the abort was requested. A worker with no work
            # is not a worker to kill; a caller that genuinely wants the lane
            # recycled has reboot_worker for that.
            level = (
                logger.info
                if _ABORT_RACE_MARKERS_RE.search(str(reason or ""))
                else logger.warning
            )
            level(
                "🛈 [MLX] Abort for %s arrived with no active generation (%s); "
                "nothing to abort, leaving the worker up.",
                os.path.basename(self.model_path),
                reason,
            )
            if not _ABORT_RACE_MARKERS_RE.search(str(reason or "")):
                # Not a lost race, so somebody is aborting on a stale premise.
                # Visible, but no longer lethal.
                _record_mlx_degradation(
                    RuntimeError(f"force_abort_without_active_request:{reason}"),
                    action="declined to force-abort an idle worker with no pending request",
                    severity="warning",
                )
            return False

        logger.error(
            "🛑 [MLX] Force-aborting active generation for %s (%s).",
            os.path.basename(self.model_path),
            reason,
        )
        self._set_lane_state("recovering", reason)

        # Each future receives ITS OWN request identity: completing every
        # pending future with the current request id let callers receive an
        # abort receipt for another request.
        seen_future_ids: set[int] = set()
        for req_id, future in pending_by_request.items():
            seen_future_ids.add(id(future))
            _set_shared_future_result(
                future,
                {
                    "status": "error",
                    "action": "generate",
                    "id": req_id,
                    "message": reason,
                    "force_aborted": True,
                },
            )
        if (
            current_future is not None
            and not current_future.done()
            and id(current_future) not in seen_future_ids
        ):
            _set_shared_future_result(
                current_future,
                {
                    "status": "error",
                    "action": "generate",
                    "id": self._current_request_id,
                    "message": reason,
                    "force_aborted": True,
                },
            )

        termination_proven = process is None
        worker_survived = False
        if process is not None:
            logger.error(
                "🛑 [MLX] Stopping worker immediately for forced abort before lifecycle lock cleanup (%s).",
                reason,
            )
            _note_lane_worker_death(self, reason)
            # Do not close the handle before lifecycle ownership is acquired:
            # when the short lock attempt loses, the current owner still needs
            # a valid handle to reconcile. The process is joined here; handle
            # retirement happens below after pointer ownership is established.
            termination_proven = self._kill_and_join_blocking(
                process,
                cooperative=False,
                retire_handle=False,
            )
            worker_survived = not termination_proven
            if worker_survived:
                logger.critical(
                    "🚨 [MLX] Worker for %s survived the forced-abort escalation; "
                    "keeping its process and IPC ownership fenced.",
                    os.path.basename(self.model_path),
                )

        # No escalated wait here, deliberately. This runs on a watchdog
        # thread whose whole value is answering fast, and the urgent half of
        # the abort — killing the worker, completing the futures — is already
        # done above. Blocking for seconds to win a lock we can hand off is
        # the wrong trade.
        acquired = self._lock.acquire(timeout=_FORCE_ABORT_LOCK_WAIT_S)
        if not acquired:
            # CP126 499846c3. The old path proceeded anyway. Its two most
            # damaging steps are exactly the ones a concurrent spawn is in the
            # middle of: `self._process = None` erases a process another thread
            # has just published, and `_replace_ipc_queues()` orphans the
            # queues that new worker is already writing to — a worker alive and
            # talking to nobody, which is the "cortex warming forever" shape.
            #
            # The kill and the future completions above already happened, and
            # neither needs ownership. What is left is state reconciliation,
            # which belongs to the lifecycle owner. Hand it over, unless
            # repeated attempts prove that owner is itself wedged.
            self._force_abort_lock_failures += 1
            if self._force_abort_lock_failures < _FORCE_ABORT_LOCK_FORCE_AFTER:
                self._force_abort_reconcile_pending = reason
                self._set_lane_state("recovering", f"{reason}:awaiting_lifecycle_owner")
                _record_mlx_degradation(
                    TimeoutError(f"force_abort_lock_unavailable:{reason}"),
                    action=(
                        "killed the worker and handed lifecycle reconciliation to the "
                        "lock owner instead of mutating state without ownership"
                    ),
                    severity="error",
                )
                logger.error(
                    "🚨 [MLX] Force-abort could not take the lifecycle lock for %s "
                    "(attempt %d/%d); worker killed, reconciliation deferred to the owner.",
                    os.path.basename(self.model_path),
                    self._force_abort_lock_failures,
                    _FORCE_ABORT_LOCK_FORCE_AFTER,
                )
                self._record_degraded_event(
                    "force_aborted_generation",
                    detail=f"{os.path.basename(self.model_path)}:{reason}",
                    severity="error",
                    foreground_request=True,
                )
                return not worker_survived
            _record_mlx_degradation(
                RuntimeError("force_abort_without_lifecycle_lock"),
                action="force-abort mutated lifecycle state without the lifecycle lock",
                severity="error",
            )
        else:
            self._force_abort_lock_failures = 0
        try:
            # The kill runs before the lifecycle lock so a wedged generation
            # can be interrupted promptly. A concurrent lifecycle owner may
            # publish a replacement while we wait for that lock. In that
            # case every client field and queue below belongs to the new
            # worker, not the process this abort targeted.
            if self._process is not process:
                if termination_proven and process is not None:
                    self._retire_worker_process_handle(process)
                logger.info(
                    "[MLX] Force-abort target was replaced before reconciliation; "
                    "preserving replacement worker state and IPC ownership."
                )
                return not worker_survived
            self._force_abort_reconcile_pending = None
            self._pending_generations.clear()
            self._current_gen_future = None
            self._active_generations = 0
            self._deferred_reboot_reason = None
            self._warmup_in_flight = False
            self._init_done = False
            # Keep the handle when the worker outlived the kill. A None handle
            # is the client saying "no worker of mine is running", and the next
            # spawn admission believes it — which is how a survivor becomes a
            # SECOND resident model rather than a tracked one to reap.
            owns_aborted_process = True
            if termination_proven:
                if owns_aborted_process:
                    self._process = None
                if process is not None:
                    self._retire_worker_process_handle(process)
            self._last_heartbeat = 0.0
            self._last_progress_at = 0.0
            self._last_token_progress_at = 0.0
            self._last_generation_completed_at = 0.0
            self._last_user_facing_completed_at = 0.0
            self._last_visible_readiness_at = 0.0
            self._process_started_at = 0.0
            self._clear_active_generation_tracking()
            if self._init_future is not None:
                _cancel_shared_future(self._init_future)
            self._init_future = None

            if self._listener_task is not None:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None
            self._cancel_lane_renewal_task()

            # Replacing queues while a survivor remains disconnects the only
            # IPC path that can still identify or stop it. Likewise, if another
            # lifecycle owner already installed a replacement process, those
            # queues belong to that replacement and must remain untouched.
            if termination_proven and owns_aborted_process:
                self._replace_ipc_queues()
            # Only release the request lock when it belongs to the generation
            # this abort just killed. Releasing it unconditionally admitted a
            # SECOND request into a critical section another caller was still
            # running — the abort's own damage, not the wedge's.
            self._release_request_lock_if_aborted(reason)
            self._clear_detached_worker_requests()
            gc.collect()
        finally:
            if acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    logger.debug(
                        "Loop-agnostic lifecycle lock for %s was already released.",
                        os.path.basename(self.model_path),
                    )

        self._record_degraded_event(
            "force_aborted_generation",
            detail=f"{os.path.basename(self.model_path)}:{reason}",
            severity="error",
            foreground_request=True,
        )
        if worker_survived:
            # The abort did not achieve its purpose. Its caller uses this
            # verdict to decide whether the lane is free to use again, and a
            # lane whose old worker is still decoding on the accelerator is
            # not. Say so, and leave the lane named as fenced rather than cold.
            self._set_lane_state("recovering", f"{reason}:worker_survived_abort")
            _record_mlx_degradation(
                RuntimeError(f"force_abort_worker_survived:{reason}"),
                action=(
                    "reported the forced abort as INCOMPLETE; the worker outlived "
                    "kill escalation and the handle was retained for reaping"
                ),
                severity="critical",
            )
            return False
        try:
            self._release_durable_model_lane_owner_sync(reason=reason)
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            # CP126 35eefee4. This recorded a critical degradation and then
            # marked the lane COLD and returned success. The fencing token was
            # NOT cleared (the release raised part-way), so later admission
            # would be blocked by a fence nobody had been told about — a
            # terminal recovery dependency reported as a clean abort.
            #
            # The abort itself did happen, so the caller is still told the
            # generation was aborted; what changes is that the lane is left in
            # a NAMED fenced state carrying the owner and token that must be
            # released before this lane can serve again.
            self._note_lane_release_failure(exc, reason=reason)
            return True
        self._set_lane_state("cold", reason)
        return True

    def _reset_worker_scoped_state(self) -> None:
        """Drop everything the PREVIOUS worker established.

        CP126 066ebe25: these fields are only replaced when the new worker's
        init receipt carries them. A spawn that fails partway, or a receipt
        missing a field, therefore left the old worker's claim in place — and
        an old `active: true` recurrence status could certify a replacement
        that never attached recurrence at all. Clearing them at spawn time
        means the absence of evidence reads as absence, not as the last
        worker's evidence.
        """
        self._worker_identity = {}
        self._worker_capture_origin_binding = {}
        self._recurrent_depth_status = {}
        self._recurrent_adapter_activation = {}
        self._unified_recurrent_shadow_status = {}
        self._unified_recurrent_shadow_probe_status = {}
        self._unified_recurrent_shadow_canary_status = {}
        self._unified_recurrent_qualified_activation_status = {}
        self._steering_liveness_observed = False
        self._last_interoception = {}
        self._last_surface_control_receipt = {}
        self._soft_cancel_target = None
        self._soft_cancel_ack = None

    def _spawn_worker_blocking(self) -> mp.Process:
        """Isolated spawn logic for the MLX worker, run in a background thread."""
        if _shutdown_blocks_model_work(self.model_path, action="worker spawn"):
            raise RuntimeError("runtime_shutdown")
        self._reset_worker_scoped_state()
        # [STABILITY v60] Reclaim the old/orphan worker BEFORE the memory
        # admission check. A recycle (or crash respawn) replaces a worker that
        # is still resident; killing it below frees its ~20GB. Running the
        # headroom check FIRST saw the about-to-die worker's memory and refused
        # the spawn (memory_pressure_refused_worker_spawn:model_load_headroom:
        # 20.2GB < 22.0GB → recycled_model_lane_not_live_after_warmup → DNU
        # FATAL), so the wedge could never recover. Free first, then admit.
        #
        # [STABILITY v51] Orphan reclamation: kill any existing MLXWorker
        # processes for this model path before spawning a new one.
        orphan_scan_completed = False
        try:
            model_basename = os.path.basename(self.model_path)
            target_name = f"MLXWorker-{model_basename}"
            for observed_process in get_resource_observer().processes():
                if _shutdown_blocks_model_work(self.model_path, action="orphan scan"):
                    raise RuntimeError("runtime_shutdown")
                try:
                    pname = observed_process.name
                    command = observed_process.cmdline
                    if target_name in pname or (
                        command
                        and any(model_basename in str(arg) for arg in command)
                        and "mlx_worker" in str(command)
                    ):
                        ancestor_pids = set(observed_process.ancestor_pids)
                        if observed_process.pid != os.getpid() and os.getpid() in ancestor_pids:
                            logger.warning(
                                "🧹 [STABILITY] Killing orphan MLXWorker pid=%d for %s",
                                observed_process.pid,
                                model_basename,
                            )
                            action_process = psutil.Process(observed_process.pid)
                            action_process.kill()
                            action_process.wait(timeout=3.0)
                        elif observed_process.pid != os.getpid():
                            logger.info(
                                "Model-path match pid=%d for %s belongs to another root; "
                                "durable lane accounting will arbitrate it without cross-root kill.",
                                observed_process.pid,
                                model_basename,
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            orphan_scan_completed = True
        except (OSError, ConnectionError, TimeoutError) as orphan_exc:
            _record_mlx_degradation(
                orphan_exc,
                action="continued worker spawn after orphan reclamation scan failed",
            )
            logger.debug("Orphan reclamation scan failed: %s", orphan_exc)

        # CP126 1399e019. A failed scan used to be logged "non-fatal" and the
        # spawn proceeded. That is safe only if no worker of ours survives —
        # and a failed scan is precisely the case where we do not know. For a
        # SAME-CLIENT replacement the consequence is a second copy of a 20GB
        # model resident at once, which exhausts unified memory long before
        # durable accounting notices.
        #
        # So: when the scan could not run, this client's own prior handle must
        # be provably terminal before we spawn beside it. Unobservable counts
        # as alive.
        if not orphan_scan_completed:
            prior = self._process
            prior_terminal = prior is None
            if prior is not None:
                try:
                    prior_terminal = not bool(prior.is_alive())
                except (RuntimeError, AttributeError, ValueError, OSError):
                    prior_terminal = False
            if not prior_terminal:
                error = RuntimeError(
                    "orphan_reclamation_unobservable_refused_worker_spawn:"
                    f"{os.path.basename(self.model_path)}"
                )
                _record_mlx_degradation(
                    error,
                    action=(
                        "refused a same-client worker replacement while orphan "
                        "reclamation was unobservable and the prior process was not "
                        "proven terminal"
                    ),
                    severity="critical",
                )
                logger.critical(
                    "🚨 [MLX] Refusing to spawn a replacement worker for %s: the "
                    "orphan scan could not run and the previous process is not "
                    "proven dead. Spawning would risk two resident copies.",
                    os.path.basename(self.model_path),
                )
                raise error

        memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)

        # The orphan scan above only reaps workers from PREVIOUS incarnations —
        # it explicitly skips this client's own process. So a worker that loaded
        # the model but never finished initializing sits there holding its
        # weights while the lane, which has already given up on it, refuses to
        # spawn the replacement for want of the very memory it is holding.
        #
        # Measured live 2026-07-26: available 17.4GB against a 24GB gate, with
        # ~16GB of it wired to our own unusable worker. Killing the instance by
        # hand dropped wired 21.7GB -> 5.2GB and freed 34.6GB. Aura could not
        # recover on her own from a state she created, which is the worst shape
        # a deadlock can take.
        #
        # Narrow by construction: only when the spawn is about to be refused
        # anyway, only our own process, only one that never became usable, and
        # only when it is serving nobody. Then re-check, so the reclaim wait
        # below observes the memory this just freed.
        if memory_block and not self._is_deep_solver_lane():
            try:
                stale = self._process
                stale_alive = bool(
                    stale is not None
                    and getattr(stale, "is_alive", lambda: False)()
                )
                if (
                    stale_alive
                    and not self._init_done
                    and int(getattr(self, "_active_generations", 0) or 0) == 0
                ):
                    logger.warning(
                        "🧹 [MLX] Reclaiming our own never-initialized worker pid=%s "
                        "before refusing a spawn for headroom (%s) — it is holding "
                        "the memory the replacement needs and serving no one.",
                        getattr(stale, "pid", "unknown"),
                        memory_block,
                    )
                    reclaimed = self._kill_and_join_blocking(
                        stale,
                        cooperative=False,
                    )
                    if not reclaimed:
                        raise RuntimeError(
                            "never_initialized_worker_reclamation_unproven:"
                            f"pid={getattr(stale, 'pid', 'unknown')}"
                        )
                    self._process = None
                    self._init_done = False
                    memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
            except (OSError, AttributeError, RuntimeError, ValueError) as reclaim_exc:
                _record_mlx_degradation(
                    reclaim_exc,
                    action="continued spawn admission after self-worker reclaim failed",
                )

        if memory_block and not self._is_deep_solver_lane():
            # A worker we just killed (orphan reclamation above, or a prior
            # generation-timeout force-abort) frees ~18GB, but the OS reclaim of
            # wired Metal memory lags process exit. Checking headroom instantly
            # sees the pre-reclaim number and refuses — which takes the whole
            # conversation lane COLD even though the memory is about to be free.
            # Observed live during the 200-turn soak (2026-07-06): a Cortex
            # generation timed out, the worker was killed, respawn was refused
            # at 20.3GB < 24GB while the killed worker's 18.6GB had not yet been
            # reclaimed, and a cluster of turns died until pressure eased. Wait
            # (bounded) for reclaim and re-check before refusing. Runs in
            # _spawn_worker_blocking's executor thread, so the sleep does not
            # block the event loop; the deep-solver lane still refuses instantly.
            try:
                reclaim_wait_s = float(
                    os.environ.get("AURA_MLX_SPAWN_RECLAIM_WAIT_S", "15") or 15.0
                )
            except (TypeError, ValueError):
                reclaim_wait_s = 15.0
            reclaim_deadline = time.monotonic() + max(0.0, reclaim_wait_s)
            waited = False
            while memory_block and time.monotonic() < reclaim_deadline:
                if _shutdown_blocks_model_work(self.model_path, action="memory reclaim wait"):
                    raise RuntimeError("runtime_shutdown")
                waited = True
                time.sleep(1.5)
                if _shutdown_blocks_model_work(self.model_path, action="memory reclaim retry"):
                    raise RuntimeError("runtime_shutdown")
                memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
            if waited and not memory_block:
                logger.info(
                    "🟢 [MLX] Headroom recovered after worker reclaim; proceeding with spawn."
                )
        if memory_block:
            error = ModelLoadAdmissionRefused(memory_block)
            if self._is_deep_solver_lane():
                logger.warning(
                    "🛡️ [MLX] Refusing optional deep Solver spawn before model load: %s",
                    memory_block,
                )
                raise error
            _record_mlx_degradation(
                error,
                action="refused MLX worker spawn before model load due to memory pressure",
                severity="critical",
            )
            raise error

        runtime_ok, runtime_detail = _probe_mlx_runtime()
        if not runtime_ok:
            raise RuntimeError(f"mlx_runtime_probe_failed:{runtime_detail}")
        if _shutdown_blocks_model_work(self.model_path, action="post-runtime-probe spawn"):
            raise RuntimeError("runtime_shutdown")

        if self._req_q is None or self._res_q is None:
            raise RuntimeError("MLX IPC queues must be created before worker spawn")
        ctx = self._mp_context

        lock_dir = state_root() / "run"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file_path = str(lock_dir / "mlx_spawn.lock")
        lock_file = _open_spawn_lock_file(lock_file_path)
        with lock_file:
            try:
                logger.info("🔒 [MLX] Acquiring process-level spawn lock...")
                _acquire_spawn_file_lock(lock_file, model_path=self.model_path)
                if _shutdown_blocks_model_work(self.model_path, action="locked worker spawn"):
                    raise RuntimeError("runtime_shutdown")

                project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # CP126 841bf5f7. A fresh contract-signing key per spawn: it
                # is handed to the child at fork, never persisted, and is
                # meaningless to any other worker. Privileged output
                # contracts must be signed with it to take effect.
                from core.brain.llm.contract_authority import new_contract_key
                from core.brain.llm.latent_cortex.worker_capture_identity import (
                    build_worker_capture_launch_authority,
                )

                self._contract_key = new_contract_key()
                self._worker_capture_launch_authority = (
                    build_worker_capture_launch_authority()
                )
                if _shutdown_blocks_model_work(self.model_path, action="worker process start"):
                    raise RuntimeError("runtime_shutdown")
                p = get_subprocess_gateway().spawn_python_process(
                    PythonProcessSpec(
                        target=_mlx_worker_loop,
                        args=(
                            self.model_path,
                            self._req_q,
                            self._res_q,
                            self.device,
                            self._substrate_mem,
                            self._steering_active,
                            self._cancel_seq,
                            self._contract_key,
                            dict(self._worker_capture_launch_authority.challenge),
                            self._phi_residual_mem,
                            self._latent_readout_mem,
                        ),
                        source="mlx_local_client.worker_owner",
                        name=f"MLXWorker-{os.path.basename(self.model_path)}",
                        role=ProcessRole.MODEL_WORKER,
                        requested_privileges=frozenset(
                            {
                                Privilege.FILESYSTEM_READ,
                                Privilege.FILESYSTEM_WRITE,
                                Privilege.MODEL_WEIGHTS,
                            }
                        ),
                        accelerator_capability=AcceleratorCapability.MODEL,
                        daemon=True,
                        start_method=str(ctx.get_start_method()),
                    ),
                    context=ctx,
                )
                return p

            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                logger.info("🔓 [MLX] Released process-level spawn lock.")

    async def _spawn_worker(self) -> mp.Process:
        if _shutdown_blocks_model_work(self.model_path, action="async worker spawn"):
            raise RuntimeError("runtime_shutdown")
        return await asyncio.get_running_loop().run_in_executor(None, self._spawn_worker_blocking)

    def _record_worker_stream_progress(
        self,
        res: dict[str, Any],
        *,
        status: str | None,
        action: str | None,
    ) -> None:
        """Classify worker activity without mistaking compute for decoded output."""
        if action == "latent_reason":
            self._record_latent_progress(res)
        if status == "token":
            self._mark_token_progress(
                res.get("id"), generated_tokens=res.get("tokens_generated")
            )
        elif res.get("phase") == "prefill":
            # Reading the prompt is the model working on this request, and on
            # a long one it is the larger half. Counting only decoded tokens
            # made a turn look silent for the whole of it, so a wait that
            # defers to progress gave up during the one part of the turn where
            # nothing could have arrived yet.
            self._mark_prefill_progress(
                res.get("id"),
                processed=res.get("prompt_tokens_processed", 0),
                total=res.get("prompt_tokens_total", 0),
            )
        elif action == "latent_reason":
            # Branch selection proves liveness but is not decoded output. Treating
            # it as a token switches a healthy request onto the shorter token gap.
            self._mark_progress()
        elif res.get("tokens_generated") is not None:
            # A decoded token can be temporarily textless in the detokenizer.
            self._mark_token_progress(
                res.get("id"), generated_tokens=res.get("tokens_generated")
            )
        else:
            self._mark_progress()

    def _schedule_endogenous_terminal_response(
        self,
        response: Mapping[str, Any],
    ) -> None:
        """Hand post-response learning to its own supervised lifecycle.

        The response listener is the only consumer of the model worker's IPC
        queue. Corpus persistence must not delay terminal delivery or the
        heartbeats behind it. Learning observes a copy after correlation and
        delivery; the response pump never waits for it.
        """
        request_id = str(response.get("id") or "")
        work = process_endogenous_terminal_response(dict(response))
        try:
            get_task_tracker().bounded_track(
                work,
                name=f"MLXEndogenousOutcome:{request_id[:12] or 'unknown'}",
                owner="mlx_client",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            work.close()
            _record_mlx_degradation(
                exc,
                action="could not schedule post-response learning; reply already delivered",
                severity="warning",
            )

    async def _response_listener_loop(
        self,
        response_queue: Any | None = None,
        queue_generation: int | None = None,
    ):
        """
        [v7.8] Background task to constantly drain the worker response queue.
        Prevents IPC deadlocks by ensuring heartbeats and telemetry are ALWAYS consumed.
        """
        import queue

        from core.container import ServiceContainer

        owned_queue = self._res_q if response_queue is None else response_queue
        owned_generation = (
            self._response_queue_generation if queue_generation is None else queue_generation
        )
        if owned_queue is None:
            return
        _consecutive_errors = 0
        # Fresh pipe, fresh reporting state: a new worker must not inherit a
        # previous worker's stall/broken-pipe report latches.
        self._worker_loop_stall_reported = False
        self._worker_ipc_broken_reported = False
        while not _runtime_shutdown_requested():
            # A listener never follows a mutable queue pointer across worker
            # generations.  Once its queue is retired, it exits without
            # touching futures belonging to the replacement worker.
            if (
                owned_queue is not self._res_q
                or owned_generation != self._response_queue_generation
                or owned_generation == self._listener_stop_generation
            ):
                break
            try:
                # Use polling instead of infinite block to avoid executor thread leaks and zombie stealing
                res = await run_io_bound(owned_queue.get, True, 0.5)
                _consecutive_errors = 0
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_mlx_degradation(
                    e,
                    action="exited or backed off response listener after queue polling failure",
                    severity="error",
                )
                # If queue is closed/broken, graceful exit
                if "closed" in str(e).lower() or isinstance(e, ValueError):
                    break
                _consecutive_errors += 1
                # [BUG FIX] After repeated errors, the queue is likely broken
                # (e.g., worker killed during cascade cleanup). Exit the loop
                # instead of spinning forever and consuming thread pool resources.
                if _consecutive_errors >= 10:
                    logger.warning(
                        "⚠️ [MLX] Response listener: %d consecutive errors. Queue likely broken. Exiting.",
                        _consecutive_errors,
                    )
                    break
                logger.error("⚠️ [MLX] Response listener poll error: %s", e)
                await asyncio.sleep(0.5)
                continue

            if not res:
                continue

            if (
                owned_queue is not self._res_q
                or owned_generation != self._response_queue_generation
                or owned_generation == self._listener_stop_generation
            ):
                # The queue changed while the executor poll was in flight.  The
                # frame belongs to the retired worker generation and must not
                # complete a future created for the replacement.
                break

            try:
                status = res.get("status")
                action = res.get("action")
                req_id = res.get("id")

                if action == "capture_identity_bootstrap":
                    try:
                        if (
                            self._init_done
                            or self._init_future is None
                            or self._init_future.done()
                        ):
                            raise RuntimeError(
                                "worker_capture_bootstrap_outside_initialization"
                            )
                        raw_capture_identity = res.get(
                            "worker_action_capture_identity"
                        )
                        if not isinstance(raw_capture_identity, Mapping):
                            raise TypeError("worker_capture_bootstrap_identity_missing")
                        self._accept_worker_capture_bootstrap(raw_capture_identity)
                        self._mark_progress()
                    except (
                        ImportError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as capture_bootstrap_exc:
                        _record_mlx_degradation(
                            capture_bootstrap_exc,
                            action=(
                                "refused worker initialization because its early "
                                "capture identity was not bound to this spawn"
                            ),
                            severity="critical",
                        )
                        if self._init_future and not self._init_future.done():
                            _set_shared_future_result(
                                self._init_future,
                                {
                                    "status": "error",
                                    "action": "init",
                                    "message": (
                                        "worker_capture_bootstrap_invalid:"
                                        f"{type(capture_bootstrap_exc).__name__}"
                                    ),
                                },
                            )
                    continue

                # Remember correlation before a terminal route removes the
                # pending future. Only a terminal frame this parent owns may
                # shape budget or learning evidence.
                owned_response = bool(
                    req_id
                    and (
                        req_id in self._pending_generations
                        or req_id == self._current_request_id
                    )
                )

                # 1. Update SubsystemAudit Heartbeat
                if status == "heartbeat":
                    self._last_heartbeat = time.time()
                    self._mark_progress()
                    try:
                        self._pulse_mycelial_worker(res)
                    except (
                        ImportError,
                        AttributeError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as root_exc:
                        logger.debug(
                            "MLX worker heartbeat root publication failed: %s",
                            root_exc,
                        )
                    # Worker-reported progress evidence: the heartbeat now
                    # carries the inference-loop's own stall verdict, so a
                    # wedged decode loop is visible BEFORE the worker-side
                    # watchdog (360s) or a caller timeout fires. Surface it
                    # once per stall episode; liveness semantics stay as-is.
                    worker_reported_stall = bool(res.get("loop_stalled"))
                    stalled, stall_threshold_s = self._confirm_worker_reported_loop_stall(res)
                    if worker_reported_stall and stalled and not self._worker_loop_stall_reported:
                        self._worker_loop_stall_reported = True
                        _record_mlx_degradation(
                            RuntimeError(
                                "worker_loop_stalled:"
                                f"request={res.get('request_id') or '<unknown>'}:"
                                f"age_s={res.get('job_age_s')}:"
                                f"budget_s={stall_threshold_s:.1f}"
                            ),
                            action="worker heartbeat exceeded the active request's progress budget",
                            severity="error",
                        )
                        self.soft_cancel_active_generation("worker_loop_stalled")
                        if not self._deferred_reboot_reason:
                            self._deferred_reboot_reason = "recoverable_token_progress_stalled"
                    elif not worker_reported_stall or not stalled:
                        self._worker_loop_stall_reported = False
                    if bool(res.get("ipc_broken")) and not self._worker_ipc_broken_reported:
                        self._worker_ipc_broken_reported = True
                        _record_mlx_degradation(
                            RuntimeError("worker_response_pipe_broken"),
                            action="worker heartbeat reports a broken response pipe; expecting worker exit",
                            severity="critical",
                        )
                    owner_id, fencing_token, _receipt_id = self._durable_model_lane_owner_snapshot()
                    if fencing_token > 0 and owner_id:
                        try:
                            from core.runtime.model_lane_control import (
                                get_model_lane_controller,
                            )

                            self._schedule_durable_lane_renewal(
                                get_model_lane_controller(),
                                owner_id,
                                fencing_token,
                                owned_generation,
                            )
                        except (
                            OSError,
                            RuntimeError,
                            AttributeError,
                            TypeError,
                            ValueError,
                            TimeoutError,
                        ) as exc:
                            _record_mlx_degradation(
                                exc,
                                action="could not schedule durable model-lane renewal",
                                severity="critical",
                            )
                    audit = ServiceContainer.get("subsystem_audit", default=None)
                    if audit:
                        tier_name = (
                            "mlx_heavy" if _model_is_heavy_lane(self.model_path) else "mlx_light"
                        )
                        audit.heartbeat(tier_name)
                    continue
                if status in {"progress", "token"}:
                    self._record_worker_stream_progress(res, status=status, action=action)
                    live_intero = res.get("interoception_live")
                    if isinstance(live_intero, dict) and live_intero:
                        try:
                            from core.being.thought_interoception import (
                                get_thought_interoception,
                            )

                            get_thought_interoception().pulse_live(live_intero)
                        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                            logger.debug("Live interoception pulse dropped.")
                    continue

                # 2. Route init/generation responses to the correct awaiting future
                if action == "init":
                    if self._init_future and not self._init_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._init_future, res)
                        continue
                elif action in _TERMINAL_WORKER_ACTIONS:
                    # Before routing: a terminal frame for a cancelled request
                    # is the acknowledgement, and it must be recorded even when
                    # the caller has already abandoned the future.
                    if isinstance(res, dict):
                        self._note_soft_cancel_acknowledgement(res)
                    future = self._pending_generations.pop(req_id, None) if req_id else None
                    delivered = bool(
                        future and await self._route_terminal_worker_response(
                            str(req_id), future, res,
                        )
                    )
                    # A generation can finish after the caller has already
                    # abandoned it and started another turn. Never hand a
                    # response with an old request id to the current future.
                    #
                    # CP126 49d694a1: the id-less fallback (`not req_id or ...`)
                    # let a stale or malformed terminal frame COMPLETE the
                    # current request with another turn's content. The worker
                    # stamps every response with its job id, so an id-less
                    # terminal frame is malformed by construction — the error
                    # route already rejects it, and so does this one now.
                    if (
                        not delivered
                        and self._current_gen_future
                        and not self._current_gen_future.done()
                        and req_id
                        and req_id == self._current_request_id
                    ):
                        delivered = _set_shared_future_result(self._current_gen_future, res)
                    if owned_response:
                        _observe_worker_prompt_tokenization(res)
                        self._schedule_endogenous_terminal_response(res)
                    if delivered:
                        self._mark_progress()
                        continue
                    if not req_id:
                        _record_mlx_degradation(
                            RuntimeError(f"uncorrelated_worker_response:{action or 'unknown'}"),
                            action=(
                                "dropped an id-less worker response instead of "
                                "completing the active request with it"
                            ),
                            severity="warning",
                        )
                        continue
                elif status == "degraded":
                    # Worker self-reported health frames (e.g. the memory
                    # sentinel going blind/recovering). Not terminal, never
                    # correlated to a request — surface them as degradations
                    # instead of ignoring them.
                    _record_mlx_degradation(
                        RuntimeError(
                            f"worker_degraded:{action or 'unknown'}:{res.get('message', '')}"
                        ),
                        action=f"worker self-reported degradation ({action or 'unknown'})",
                        severity=(
                            "critical" if action == "memory_sentinel_degraded" else "warning"
                        ),
                    )
                    logger.warning(
                        "⚠️ [MLX] Worker degradation frame (%s): %s",
                        action,
                        res.get("message"),
                    )
                    continue
                elif status == "error" and action == "memory_fuse":
                    # The worker's memory sentinel is about to hard-exit the
                    # process. This frame is intentionally id-less (it is not
                    # a request result) — attribute the imminent death to
                    # every in-flight request NOW instead of letting each one
                    # discover it via timeout against a dead process.
                    fuse_message = str(res.get("message") or "worker_memory_fuse")
                    _record_mlx_degradation(
                        RuntimeError(f"worker_memory_fuse:{fuse_message}"),
                        action="worker memory fuse tripped; failing in-flight requests with attribution",
                        severity="critical",
                    )
                    logger.critical("🛑 [MLX] %s", fuse_message)
                    for pending_id, pending in list(self._pending_generations.items()):
                        if pending is not None and not pending.done():
                            _set_shared_future_result(
                                pending,
                                {
                                    "status": "error",
                                    "action": "generate",
                                    "id": str(pending_id),
                                    "message": f"worker_memory_fuse:{fuse_message}",
                                    "memory_pressure": res.get("memory_pressure") or {},
                                },
                            )
                    self._pending_generations.clear()
                    current_fut = self._current_gen_future
                    if current_fut is not None and not current_fut.done():
                        _set_shared_future_result(
                            current_fut,
                            {
                                "status": "error",
                                "action": "generate",
                                "id": self._current_request_id,
                                "message": f"worker_memory_fuse:{fuse_message}",
                                "memory_pressure": res.get("memory_pressure") or {},
                            },
                        )
                    self._release_detached_request_lock()
                    self._clear_detached_worker_requests()
                    continue
                elif status == "error":
                    init_error = (
                        self._init_future is not None
                        and not self._init_future.done()
                        and not self._init_done
                        and action in {None, "", "init"}
                    )
                    if init_error:
                        self._mark_progress()
                        payload = dict(res)
                        payload.setdefault("action", "init")
                        _set_shared_future_result(self._init_future, payload)
                        continue
                    if action == "init" and self._init_future and not self._init_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._init_future, res)
                        continue
                    future = self._pending_generations.pop(req_id, None) if req_id else None
                    if future and not future.done():
                        self._mark_progress()
                        _set_shared_future_result(future, res)
                        continue
                    if (
                        self._current_gen_future
                        and not self._current_gen_future.done()
                        and req_id
                        and req_id == self._current_request_id
                    ):
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
                        continue
                    if not req_id and status in {"ok", "error"}:
                        # The worker stamps every response with its job id —
                        # an id-less terminal message is stale or malformed
                        # and must NOT complete the current request with
                        # someone else's content or error state.
                        _record_mlx_degradation(
                            RuntimeError("id_less_worker_response_dropped"),
                            action="dropped id-less terminal worker message instead of completing current request",
                        )
                        continue

                # 3. Log errors if no future is waiting
                if status == "error":
                    logger.error("🛑 [MLX] Async worker error: %s", res.get("message"))

            except Exception as e:  # noqa: BLE001 - the listener is the sole response
                # consumer: ANY escaping processing error (TypeError, ValueError,
                # OSError, future completion races, callback failures) would leave
                # a live worker with no one draining its queue.
                _record_mlx_degradation(
                    e,
                    action="kept response listener alive after malformed worker message",
                    severity="error",
                )
                logger.error("⚠️ [MLX] Response listener message processing error: %s", e)
                # CP126 2ba6ea2e, second half: surviving is not enough. The
                # message that failed belonged to a REQUEST, and leaving that
                # waiter pending means the listener lives while one caller
                # hangs to its deadline for a response that was already
                # delivered and dropped on the floor. Terminalize it.
                self._terminalize_failed_message(res if isinstance(res, dict) else None, e)
                await asyncio.sleep(1.0)

    def _terminalize_failed_message(self, res: dict[str, Any] | None, exc: BaseException) -> None:
        """Fail the request whose response could not be processed.

        A message the listener could not handle is a response that arrived and
        was lost. Its waiter has no way to learn that, so without this it sits
        until its own deadline — the listener stayed healthy and one caller
        paid the full timeout for a reply that had already come back.
        """
        req_id = ""
        if isinstance(res, dict):
            req_id = str(res.get("id") or "")
        if not req_id:
            return
        future = self._pending_generations.pop(req_id, None)
        if future is None and req_id == str(getattr(self, "_current_request_id", "") or ""):
            future = self._current_gen_future
        if future is None or future.done():
            return
        _set_shared_future_result(
            future,
            {
                "status": "error",
                "action": str(res.get("action") or "generate") if res else "generate",
                "id": req_id,
                "message": f"listener_message_processing_failed:{type(exc).__name__}",
            },
        )

    async def _ensure_worker_alive(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Self-healing supervisor for the MLX worker.

        [OOM FIX] Acquires a global semaphore so only ONE model loads at a time.
        This prevents the 32B + 7B from loading simultaneously and crashing Metal.
        """
        if _shutdown_blocks_model_work(self.model_path, action="worker start/recovery"):
            return False
        if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
            # Same inversion as the warmup guard (2026-07-10): the Reflex
            # fallback serving turns OWNED the foreground, which deferred
            # cortex recovery here — the primary could never come back while
            # its own fallback was answering for it. The primary lane's
            # recovery is exempt; other background lanes still yield.
            logger.info(
                "⏸️ [MLX] Deferring background worker activity for %s while foreground lane is owned by %s.",
                os.path.basename(self.model_path),
                _FOREGROUND_OWNER_NAME or "foreground",
            )
            return False
        if request_is_background and not self._is_primary_lane():
            # Every reason the gate's quiet policy returns (foreground_
            # reserved, headroom, cortex_startup_quiet, quiet window)
            # protects the user's turn from BACKGROUND COMPETITION. The
            # primary lane's own revival is not competition — it is the
            # thing the user's turn is waiting for, so it is exempt here
            # exactly as at the owner guard above.
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Deferring background worker activity for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False

        # Fast path: if worker is already alive, don't acquire the gate
        if self._process and self._process.is_alive() and self._init_done:
            self._clear_model_load_admission_backoff()
            self._check_lane_state_staleness()  # [STABILITY v51]
            recurrent_depth_status = _normalize_recurrent_depth_status(
                self._recurrent_depth_status,
                model_path=self.model_path,
            )
            recurrent_depth_blocker = _recurrent_depth_readiness_blocker(recurrent_depth_status)
            if recurrent_depth_blocker and not request_is_background:
                self._set_lane_state("recovering", recurrent_depth_blocker)
                self._record_degraded_event(
                    recurrent_depth_blocker,
                    detail=f"{os.path.basename(self.model_path)}:{recurrent_depth_status}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                return False
            self._set_lane_state("ready")
            return True

        # Slow path: admission owns whether model loading may proceed; the
        # spawn gate remains the mechanical single-spawn mutex beneath it.
        if request_is_background and self._model_load_admission_backoff_active():
            return False
        if int(self._model_lane_fencing_token or 0) > 0:
            try:
                await self._release_durable_model_lane_owner(
                    reason="dead_worker_before_respawn",
                )
            except (
                ImportError,
                OSError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                self._set_lane_state("recovering", "stale_model_lane_owner_release_failed")
                _record_mlx_degradation(
                    exc,
                    action=(
                        "refused worker respawn until the dead worker's durable "
                        "model-lane owner can be released"
                    ),
                    severity="critical",
                )
                return False
        # CP126 1effd581. The anti-thrash swap cooldown used to sleep up to
        # twelve seconds INSIDE the model-load admission context and the
        # global single-spawn gate. Everything else in the process that wanted
        # to spawn waited behind a lane that was doing nothing but counting.
        # The wait is the same length; it just happens out here, where it
        # blocks only its own caller, and it now stops early for shutdown.
        await self._await_swap_cooldown(
            foreground_request=foreground_request,
            skip_swap_cooldown=skip_swap_cooldown,
        )
        try:
            async with _model_load_admission_context(
                self,
                foreground_request=foreground_request,
            ):
                async with _spawn_gate_context(
                    owner=f"{os.path.basename(self.model_path)}:"
                    f"{'foreground' if foreground_request else 'background'}",
                    # NOT scoped to init_timeout. Bounding the gate wait by
                    # the caller's budget is the right idea and the context
                    # manager supports it, but wiring it here moved other
                    # paths onto the timeout branch, and one of those leaves
                    # the durable model-lane owner unreconciled (lane FENCED,
                    # admission blocked) — the lease-outlives-holder shape in
                    # a new costume. The deferred-lane turn budget in
                    # interface/routes/chat.py addresses the dominant cause
                    # without that risk; re-scoping this wait needs the
                    # durable-owner path made timeout-safe first.
                ):
                    return await self._ensure_worker_alive_inner(
                        request_is_background=request_is_background,
                        foreground_request=foreground_request,
                        init_timeout=init_timeout,
                        soft_timeout=soft_timeout,
                        skip_swap_cooldown=skip_swap_cooldown,
                    )
        except _ModelLoadAdmissionDeniedError as admission_exc:
            # The inner spawn path can establish a specific terminal failure
            # (for example, a failed Metal runtime probe) before the durable
            # transaction observes that no candidate reached READY.  Preserve
            # that causal state instead of replacing it with the less useful
            # outer transaction consequence.
            if self._lane_state != "failed" or not str(self._lane_error or ""):
                self._set_lane_state("recovering", admission_exc.reason)
            backoff_s = self._note_model_load_admission_denial(
                admission_exc.reason,
                receipt_id=admission_exc.receipt_id,
            )
            if foreground_request:
                self._record_degraded_event(
                    "model_load_admission_denied",
                    detail=(
                        f"{os.path.basename(self.model_path)}:{admission_exc.reason}:"
                        f"receipt={admission_exc.receipt_id or 'none'}"
                    ),
                    severity="warning",
                    foreground_request=True,
                )
            admission_logger = logger.warning if foreground_request else logger.info
            admission_logger(
                "⏸️ [MLX] Model-load admission deferred for %s: %s (receipt=%s, recheck_in=%.1fs)",
                os.path.basename(self.model_path),
                admission_exc.reason,
                admission_exc.receipt_id or "none",
                backoff_s,
            )
            return False
        except TimeoutError as gate_exc:
            # Another lane's spawn is wedged holding the global gate. Defer
            # honestly instead of joining the pileup — the warmup's finally
            # still clears its flag, admission stays unblocked, and the
            # watchdog handles the wedged holder.
            self._set_lane_state("recovering", "spawn_gate_timeout")
            self._record_degraded_event(
                "spawn_gate_timeout",
                detail=f"{os.path.basename(self.model_path)}:{gate_exc}",
                severity="warning",
                foreground_request=foreground_request,
            )
            logger.warning(
                "⏸️ [MLX] Spawn gate held too long by another lane; deferring %s spawn (%s).",
                os.path.basename(self.model_path),
                gate_exc,
            )
            return False

    #: How long to leave between heavy-model swaps, so a lane that just gave
    #: up its weights is not immediately asked for them again.
    _SWAP_COOLDOWN_S = 12.0

    def _inline_retry_refusal(self) -> str:
        """Why the inline empty-generation retry must not run, or "".

        Re-checks the conditions a FRESH request would be admitted against.
        The retry holds the lock the first attempt took, so nothing else will
        ask these questions on its behalf, and each of them can have flipped
        while the first attempt was decoding.
        """
        if _runtime_shutdown_requested():
            return "runtime_shutdown"
        process = self._process
        if process is None or not process.is_alive():
            return "worker_not_alive"
        if not self._init_done:
            return "worker_not_initialised"
        try:
            snapshot = get_memory_pressure_snapshot()
            if snapshot.refuse_heavy_local_generation:
                return f"memory_pressure:{snapshot.reason or 'critical'}"
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            # Unobservable pressure is not permission. A fresh heavy request
            # fails closed here, so a retry that costs the same must too.
            return "memory_pressure_unobservable"
        return ""

    async def _await_swap_cooldown(
        self, *, foreground_request: bool = False, skip_swap_cooldown: bool = False
    ) -> float:
        """Serve the anti-thrash cooldown OUTSIDE the shared spawn gates.

        CP126 1effd581: this slept inside the model-load admission context and
        the global single-spawn semaphore, so one lane counting down twelve
        seconds blocked every other lane in the process from spawning at all.
        The wait itself is legitimate — swapping heavy models back and forth
        thrashes unified memory — but nothing about it requires owning the
        gate, and holding a global resource while doing nothing is the
        definition of a convoy.

        Returns the seconds actually waited. Stops early on shutdown rather
        than making a terminating runtime sit out the full cooldown.
        """
        from .model_registry import ACTIVE_MODEL, get_deep_model_path, get_model_path

        target_path = _real_model_path(self.model_path)
        primary_path = _real_model_path(get_model_path(ACTIVE_MODEL))
        deep_path = _real_model_path(get_deep_model_path())
        if target_path not in (primary_path, deep_path):
            return 0.0
        if not _GLOBAL_LAST_HEAVY_MODEL or _GLOBAL_LAST_HEAVY_MODEL == target_path:
            return 0.0

        elapsed = time.time() - _GLOBAL_LAST_SWAP_TIME
        remaining = self._SWAP_COOLDOWN_S - elapsed
        if remaining <= 0.0:
            return 0.0
        if skip_swap_cooldown:
            logger.info(
                "⚡ [MLX] Skipping %.1fs swap cooldown for %s.",
                remaining,
                os.path.basename(target_path),
            )
            return 0.0

        logger.warning(
            "⏳ [MLX] SWAP COOLDOWN: waiting %.1fs before spawning %s (%s lane).",
            remaining,
            os.path.basename(target_path),
            "foreground" if foreground_request else "background",
        )
        waited = 0.0
        # Slice it so a shutdown does not have to outlast the whole cooldown.
        while waited < remaining:
            if _shutdown_blocks_model_work(self.model_path, action="swap cooldown"):
                logger.info("🛑 [MLX] Swap cooldown cut short by shutdown.")
                break
            slice_s = min(0.25, remaining - waited)
            await asyncio.sleep(slice_s)
            waited += slice_s
        return waited

    async def _ensure_worker_alive_inner(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
        _init_retry: bool = False,
    ) -> bool:
        """Inner implementation — called while holding the global spawn gate."""
        if _shutdown_blocks_model_work(self.model_path, action="worker spawn"):
            return False
        # K4 crash-loop backoff: a lane whose workers keep dying young must
        # not respawn on demand. Refuse fast with a named reason — the
        # escalation ladder answers while the backoff drains. A healthy
        # worker passing through is never disturbed.
        if not (self._process and self._process.is_alive() and self._init_done):
            crash_blocked = _crash_loop_blocks_worker_spawn(self)
            if crash_blocked:
                self._set_lane_state("recovering", crash_blocked)
                self._record_degraded_event(
                    "crash_loop_backoff",
                    detail=f"{os.path.basename(self.model_path)}:{crash_blocked}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "⛔ [MLX] Respawn refused for %s: %s",
                    os.path.basename(self.model_path),
                    crash_blocked,
                )
                return False
        should_wait_init = False
        init_future: SharedFuture | None = None

        # [PIPELINE HARDENING] 12s Swap Cooldown
        from .model_registry import ACTIVE_MODEL, get_deep_model_path, get_model_path

        primary_path = _real_model_path(get_model_path(ACTIVE_MODEL))
        deep_path = _real_model_path(get_deep_model_path())
        target_path = _real_model_path(self.model_path)

        global _GLOBAL_LAST_SWAP_TIME, _GLOBAL_LAST_HEAVY_MODEL

        if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
            # Primary-lane exemption (2026-07-10 inversion family): the
            # reconciler's prewarm arrives here as background work; blocking
            # it while the Reflex fallback owns the foreground kept the
            # cortex dead exactly while users waited on it.
            logger.info(
                "⏸️ [MLX] Background spawn blocked for %s while foreground lane is reserved.",
                os.path.basename(self.model_path),
            )
            return False
        if request_is_background and not self._is_primary_lane():
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background spawn blocked for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False

        # The swap cooldown is served by _await_swap_cooldown BEFORE the
        # admission context and the global spawn gate are taken (CP126
        # 1effd581), so nothing sleeps while holding them.

        acquired = await asyncio.to_thread(self._lock.acquire, True, 15.0)
        if not acquired:
            logger.error(
                "🚨 [MLX] DEADLOCK DETECTED: Could not acquire _lock within 15s for %s",
                os.path.basename(self.model_path),
            )
            return False
        try:
            # A forced abort may have killed the worker without owning this
            # lock. Finish its reconciliation before deciding lane health, or
            # a dead process reads as "already healthy".
            self._apply_pending_force_abort_reconcile()
            if self._process and self._process.is_alive() and self._init_done:
                # CP126 6165be63. "The process exists and once finished its
                # handshake" is not the same as "this lane can serve a turn".
                # A wedged worker satisfies both and was admitted as healthy,
                # so the first user request paid the whole first-token budget
                # or the hard cap before anything noticed. Check that it has
                # spoken recently, and that something is listening.
                silence = self._liveness_quiet_for_s()
                stale_after = self._stale_after()
                listener = self._listener_task
                listener_alive = listener is None or not listener.done()
                if silence <= stale_after and listener_alive:
                    self._set_lane_state("ready")
                    return True  # Already healthy, release gate
                _record_mlx_degradation(
                    TimeoutError(
                        f"worker for {os.path.basename(self.model_path)} passed the "
                        f"alive+init check but has been silent {silence:.1f}s "
                        f"(limit {stale_after:.1f}s, listener_alive={listener_alive})"
                    ),
                    action="recycled a worker that looked ready and was not responding",
                    severity="error",
                )
                logger.warning(
                    "♻️ [MLX] %s is alive and initialised but silent for %.1fs; "
                    "recycling instead of admitting it as ready.",
                    os.path.basename(self.model_path),
                    silence,
                )
                # Torn down INLINE, not via reboot_worker: this runs while
                # holding the lifecycle lock, and reboot_worker acquires it.
                # Calling it here would block for its whole escalation ladder
                # and then perform an unsynchronised reboot — turning a
                # recovery into the wedge it was recovering from. The
                # stale-handshake branch below does the same thing for the
                # same reason.
                self._set_lane_state("recovering", "ready_check_worker_silent")
                self._init_done = False
                if self._init_future is not None:
                    _cancel_shared_future(self._init_future)
                    self._init_future = None
                _doomed, self._process = self._process, None
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        self._release_worker_process,
                        _doomed,
                        reason="ready_check_worker_silent",
                    ),
                )
                self._reset_worker_scoped_state()
                self._replace_ipc_queues()

            if self._process and self._process.is_alive() and not self._init_done:
                # Stale-handshake watchdog: if the worker process has been
                # alive but failing to complete its handshake for longer
                # than 2x the handshake timeout, the init future is wedged
                # (worker stuck loading weights, IPC pipe wedged, etc.).
                # Recycle the worker instead of waiting forever, otherwise
                # every subsequent appraisal request piles onto the same
                # never-resolving future and the lane stays in "handshaking"
                # for hours, which is what produced the cascading damasio
                # timeout / "Worker alive but still handshaking" loop.
                handshake_age = self._handshake_age_s()
                handshake_budget = max(60.0, 2.0 * self._handshake_timeout())
                if (
                    self._init_future is not None
                    and self._lane_state == "handshaking"
                    and handshake_age > handshake_budget
                ):
                    logger.warning(
                        "♻️ [MLX] Worker handshake stuck for %.0fs (>%.0fs budget) on %s — recycling.",
                        handshake_age,
                        handshake_budget,
                        os.path.basename(self.model_path),
                    )
                    self._set_lane_state("recovering", "stale_handshake")
                    try:
                        if self._init_future and not self._init_future.done():
                            self._init_future.set_exception(
                                RuntimeError("stale_handshake_recycled")
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                        _record_mlx_degradation(
                            _exc,
                            action="recycled stale handshake despite init-future notification failure",
                        )
                        logger.debug("Suppressed stale-handshake future-set: %s", _exc)
                    self._init_future = None
                    _doomed, self._process = self._process, None
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        functools.partial(
                            self._release_worker_process,
                            _doomed,
                            reason="stale_handshake",
                        ),
                    )
                    self._init_done = False
                    self._last_heartbeat = 0.0
                    self._last_progress_at = 0.0
                    self._drain_queue()
                    self._replace_ipc_queues()
                    # Fall through into the missing-init-lifecycle path on
                    # the next iteration of caller's outer loop.

                if self._init_future is not None:
                    logger.info(
                        "⏳ [MLX] Worker alive but still handshaking: %s",
                        os.path.basename(self.model_path),
                    )
                    self._set_lane_state("handshaking")
                    init_future = self._init_future
                    should_wait_init = True
                else:
                    logger.warning(
                        "♻️ [MLX] Worker alive but init lifecycle is missing. Recycling %s.",
                        os.path.basename(self.model_path),
                    )
                    self._set_lane_state("recovering", "missing_init_lifecycle")
                    _doomed, self._process = self._process, None
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        functools.partial(
                            self._release_worker_process,
                            _doomed,
                            reason="missing_init_lifecycle",
                        ),
                    )
                    self._init_done = False
                    self._last_heartbeat = 0.0
                    self._last_progress_at = 0.0
                    self._drain_queue()

                    # Prevent zombie threads from stealing messages
                    self._replace_ipc_queues()

                    init_future = _new_shared_future()
                    self._init_future = init_future
                    self._set_lane_state("spawning")
                    logger.info(
                        "📡 [MLX] Respawning worker for %s...", os.path.basename(self.model_path)
                    )
                    try:
                        self._process = await self._spawn_worker()
                        self._process_started_at = time.time()
                        self._consecutive_spawn_failures = 0
                        self._spawn_backoff_until = 0.0
                        self._spawn_backoff_cause = ""
                    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                        detail = str(exc)
                        if self._handle_optional_deep_solver_memory_refusal(exc):
                            return False
                        _sf = getattr(self, "_consecutive_spawn_failures", 0) + 1
                        self._consecutive_spawn_failures = _sf
                        self._spawn_backoff_until = time.time() + min(
                            300.0, 10.0 * (2 ** min(_sf - 1, 5))
                        )
                        # CP126 ee4ccfcc: the backoff carried no cause, so the
                        # runtime-availability probe cleared every one of them.
                        # A healthy `import mlx` says nothing about an OOM, a
                        # corrupt checkpoint or a refused memory admission.
                        self._spawn_backoff_cause = (
                            "runtime_unavailable"
                            if "mlx_runtime_probe_failed:" in detail
                            else "spawn_failure"
                        )
                        if "mlx_runtime_probe_failed:" in detail:
                            self._mark_runtime_unavailable(
                                detail.split("mlx_runtime_probe_failed:", 1)[1]
                            )
                        else:
                            self._set_lane_state("failed", detail)
                        _record_mlx_degradation(
                            exc,
                            action="marked lane failed or runtime unavailable and applied spawn backoff",
                            severity="error",
                        )
                        self._record_degraded_event(
                            "spawn_failed",
                            detail=f"{os.path.basename(self.model_path)}:{detail}",
                            severity="error",
                            foreground_request=foreground_request,
                        )
                        logger.error(
                            "🛑 [MLX] Worker respawn aborted for %s: %s (backoff %.0fs)",
                            os.path.basename(self.model_path),
                            detail,
                            min(300.0, 10.0 * (2 ** min(_sf - 1, 5))),
                        )
                        self._init_future = None
                        return False
                    if self._listener_task:
                        _cancel_task_threadsafe(self._listener_task)
                    await self._ensure_listener_task()
                    self._set_lane_state("handshaking")
                    should_wait_init = True
            elif not self._process or not self._process.is_alive():
                if self._process is not None:
                    # The worker died on its own (OS OOM kill, segfault): no
                    # kill path saw it, so account for it here — then drop
                    # the dead handle so the death is counted exactly once.
                    _note_lane_worker_death(self, "process_died_unexpectedly")
                    self._process = None
                    self._process_started_at = 0.0
                # [BUG FIX] Exponential backoff on repeated spawn failures.
                # Without this, [Errno 5] I/O errors cause a tight 2-3s retry
                # loop that leaks FDs and shared memory for hours.
                _spawn_fails = getattr(self, "_consecutive_spawn_failures", 0)
                _spawn_backoff_until = getattr(self, "_spawn_backoff_until", 0.0)
                if time.time() < _spawn_backoff_until:
                    if not await asyncio.to_thread(
                        self.refresh_runtime_availability, force_probe=True
                    ):
                        return False  # Still in backoff window

                self._drain_queue()

                # Prevent zombie threads from stealing messages
                self._replace_ipc_queues()

                init_future = _new_shared_future()
                self._init_future = init_future
                self._set_lane_state("spawning")
                # A new worker is cold, whatever the last one had done.
                self._tokens_since_spawn = 0
                logger.info("📡 [MLX] Spawning worker for %s...", os.path.basename(self.model_path))
                try:
                    self._process = await self._spawn_worker()
                    self._process_started_at = time.time()
                    self._consecutive_spawn_failures = 0  # Reset on success
                    self._spawn_backoff_until = 0.0
                    self._spawn_backoff_cause = ""
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    # OSError included: queue creation, lock files, and
                    # multiprocessing start raise it — it previously escaped
                    # with the lane stuck in "spawning" and a pending init.
                    detail = str(exc)
                    if self._handle_optional_deep_solver_memory_refusal(exc):
                        return False
                    # [BUG FIX] Exponential backoff: 10s, 30s, 60s, 120s, 300s
                    self._consecutive_spawn_failures = _spawn_fails + 1
                    backoff = min(300.0, 10.0 * (2 ** min(_spawn_fails, 5)))
                    self._spawn_backoff_until = time.time() + backoff
                    # See CP126 ee4ccfcc: a runtime probe may only clear the
                    # backoffs a runtime failure caused.
                    self._spawn_backoff_cause = (
                        "runtime_unavailable"
                        if "mlx_runtime_probe_failed:" in detail
                        else "spawn_failure"
                    )
                    if "mlx_runtime_probe_failed:" in detail:
                        self._mark_runtime_unavailable(
                            detail.split("mlx_runtime_probe_failed:", 1)[1]
                        )
                    else:
                        self._set_lane_state("failed", detail)
                    _record_mlx_degradation(
                        exc,
                        action="marked lane failed or runtime unavailable and applied spawn backoff",
                        severity="error",
                    )
                    self._record_degraded_event(
                        "spawn_failed",
                        detail=f"{os.path.basename(self.model_path)}:{detail}",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    logger.error(
                        "🛑 [MLX] Worker spawn aborted for %s: %s (attempt %d, backoff %.0fs)",
                        os.path.basename(self.model_path),
                        detail,
                        self._consecutive_spawn_failures,
                        backoff,
                    )
                    self._init_future = None
                    return False
                if self._listener_task:
                    _cancel_task_threadsafe(self._listener_task)
                await self._ensure_listener_task()
                should_wait_init = True
                self._init_done = False
                self._worker_identity = {}
                self._set_lane_state("handshaking")
        finally:
            self._lock.release()

        if should_wait_init:
            fut = init_future or self._init_future
            if fut is None:
                raise RuntimeError("MLX worker init future missing during startup")
            handshake_timeout = float(init_timeout or self._handshake_timeout())

            # [STABILITY v54] One-shot retry for worker handshake to handle
            # transient JIT/Metal compilation or memory alignment glitches.
            for handshake_attempt in range(2):
                try:
                    res = await _await_shared_future(fut, timeout_s=handshake_timeout)
                    if res.get("status") == "ok":
                        # READINESS IS EARNED, NOT ANNOUNCED. CP126 34c42774:
                        # any dict with status=ok used to set init_done,
                        # heartbeats and lane=ready, and only THEN copy the
                        # recurrence receipt and worker identity. A worker
                        # that never reported recurrence, or reported a
                        # malformed identity, was already serving by the time
                        # anyone looked. The invariants are checked first and
                        # the handshake fails if they do not hold — which
                        # feeds the existing one-shot retry.
                        readiness_errors = self._init_receipt_errors(res)
                        attested_worker_identity: dict[str, Any] = {}
                        raw_worker_identity = res.get("worker_identity")
                        if not readiness_errors and isinstance(raw_worker_identity, Mapping):
                            try:
                                attested_worker_identity = (
                                    self._attest_worker_capture_origin(raw_worker_identity)
                                )
                            except (
                                ImportError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                            ) as capture_origin_exc:
                                _record_mlx_degradation(
                                    capture_origin_exc,
                                    action=(
                                        "refused READY because the worker capture key was not "
                                        "bound to this parent spawn"
                                    ),
                                    severity="error",
                                )
                                readiness_errors.append(
                                    "worker_capture_launch_attestation_invalid"
                                )
                        if not readiness_errors:
                            from core.brain.llm.token_budget_evidence import MIN_OBSERVATIONS

                            calibration_count = _observe_worker_token_budget_calibration(res)
                            if calibration_count < MIN_OBSERVATIONS:
                                readiness_errors.append(
                                    "token_budget_calibration_not_admitted:"
                                    f"{calibration_count}/{MIN_OBSERVATIONS}"
                                )
                        if readiness_errors:
                            _record_mlx_degradation(
                                ValueError("init_receipt_invalid:" + ",".join(readiness_errors)),
                                action="refused READY on an unvalidated worker init receipt",
                                severity="error",
                            )
                            self._init_done = False
                            self._worker_identity = {}
                            self._recurrent_depth_status = {}
                            # Every field the receipt was supposed to establish
                            # is cleared together. Leaving one behind lets the
                            # PREVIOUS worker's claim certify this one.
                            self._recurrent_adapter_activation = {}
                            self._unified_recurrent_shadow_status = {}
                            self._unified_recurrent_shadow_probe_status = {}
                            self._unified_recurrent_shadow_canary_status = {}
                            self._unified_recurrent_qualified_activation_status = {}
                            self._set_lane_state(
                                "failed",
                                "init_receipt_invalid",
                            )
                            # This is terminal evidence from this exact
                            # worker. Re-reading the same completed future
                            # cannot repair it and used to leave an alive,
                            # permanently handshaking process behind. Retire
                            # the untrusted generation and perform at most one
                            # real spawn retry.
                            await self.reboot_worker(
                                reason="init_receipt_invalid",
                                mark_failed=False,
                            )
                            if not _init_retry:
                                return await self._ensure_worker_alive_inner(
                                    request_is_background=request_is_background,
                                    foreground_request=foreground_request,
                                    init_timeout=init_timeout,
                                    soft_timeout=soft_timeout,
                                    skip_swap_cooldown=True,
                                    _init_retry=True,
                                )
                            return False
                        self._init_done = True
                        self._last_heartbeat = time.time()
                        self._last_ready_at = self._last_heartbeat
                        self._mark_progress()
                        self._set_lane_state("ready")
                        recurrent_status = res.get("recurrent_depth")
                        # Always REPLACE: preserving the previous worker's
                        # status when the new receipt is absent/malformed let
                        # an old active=true certify a new worker that never
                        # reported recurrence.
                        self._recurrent_depth_status = (
                            recurrent_status if isinstance(recurrent_status, dict) else {}
                        )
                        adapter_activation = res.get("recurrent_adapter_activation")
                        self._recurrent_adapter_activation = (
                            adapter_activation
                            if isinstance(adapter_activation, dict)
                            else {}
                        )
                        shadow_status = res.get("unified_recurrent_shadow")
                        self._unified_recurrent_shadow_status = (
                            copy.deepcopy(shadow_status)
                            if isinstance(shadow_status, dict)
                            else {}
                        )
                        self._unified_recurrent_shadow_probe_status = {}
                        self._unified_recurrent_shadow_canary_status = {}
                        qualified_status = res.get(
                            "unified_recurrent_qualified_activation"
                        )
                        self._unified_recurrent_qualified_activation_status = (
                            copy.deepcopy(qualified_status)
                            if isinstance(qualified_status, dict)
                            else {}
                        )
                        if not isinstance(recurrent_status, dict):
                            _record_mlx_degradation(
                                ValueError("missing_recurrent_depth_receipt"),
                                action="cleared stale recurrence status after init receipt omitted it",
                            )
                        self._worker_identity = attested_worker_identity
                        try:
                            self._attest_mycelial_worker(res)
                        except (
                            ImportError,
                            AttributeError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as root_exc:
                            _record_mlx_degradation(
                                root_exc,
                                action=(
                                    "kept validated worker ready while Mycelium "
                                    "root attestation failed"
                                ),
                                severity="warning",
                            )
                        raw_steering = res.get("steering_active")
                        if raw_steering is not None:
                            try:
                                if isinstance(raw_steering, bool):
                                    steering_active = raw_steering
                                else:
                                    # A malformed string "false" is truthy —
                                    # never bool() an untyped IPC value into
                                    # the shared steering channels.
                                    _record_mlx_degradation(
                                        TypeError(f"non-bool steering receipt: {raw_steering!r}"),
                                        action="treated malformed steering receipt as inactive",
                                    )
                                    steering_active = False
                                self._steering_active.value = steering_active
                                self._substrate_mem[-1] = 1.0 if steering_active else 0.0
                                self._steering_liveness_observed = True
                            except (
                                TypeError,
                                ValueError,
                                IndexError,
                                AttributeError,
                            ) as steering_receipt_exc:
                                _record_mlx_degradation(
                                    steering_receipt_exc,
                                    action="kept worker ready after steering liveness receipt write failed",
                                    severity="warning",
                                )
                        if target_path in (primary_path, deep_path):
                            _GLOBAL_LAST_HEAVY_MODEL = target_path
                            _GLOBAL_LAST_SWAP_TIME = time.time()
                        logger.info("✅ [MLX] Worker ready: %s", os.path.basename(self.model_path))
                        return True
                    else:
                        msg = res.get("message", "Init failed")
                        if handshake_attempt == 0:
                            logger.warning(
                                "🔄 [MLX] Worker init failed: %s. Retrying spawn...", msg
                            )
                            # Reboot and try again once
                            await self.reboot_worker(reason="init_failed_retry", mark_failed=False)
                            # Update fut for the new spawn
                            fut = self._init_future
                            if not fut:
                                # Reboot tears the lifecycle down WITHOUT
                                # spawning a replacement, so the falsy check
                                # silently skipped the advertised one-shot
                                # retry. Re-enter the spawn path once (the
                                # spawn gate is already held by our caller).
                                if not _init_retry:
                                    return await self._ensure_worker_alive_inner(
                                        request_is_background=request_is_background,
                                        foreground_request=foreground_request,
                                        init_timeout=init_timeout,
                                        soft_timeout=soft_timeout,
                                        skip_swap_cooldown=True,
                                        _init_retry=True,
                                    )
                                break
                            continue
                        self._set_lane_state("failed", msg)
                        raise RuntimeError(msg)
                except TimeoutError:
                    if soft_timeout and self._process and self._process.is_alive():
                        logger.warning(
                            "⏳ [MLX] Init handshake exceeded request budget (%.1fs) for %s. Keeping worker alive to continue warming.",
                            handshake_timeout,
                            os.path.basename(self.model_path),
                        )
                        self._set_lane_state("recovering", "init_budget_timeout")
                        self._record_degraded_event(
                            "init_budget_timeout",
                            detail=f"{os.path.basename(self.model_path)}:{handshake_timeout:.1f}s",
                            severity="warning",
                            foreground_request=foreground_request,
                        )
                        raise
                    if handshake_attempt == 0:
                        logger.warning("⏳ [MLX] Init timeout on attempt 1. Retrying spawn...")
                        await self.reboot_worker(reason="init_timeout_retry", mark_failed=False)
                        fut = self._init_future
                        if not fut:
                            # CP126 0c91528f (timeout half). reboot_worker is a
                            # TEARDOWN: it clears _init_future and does NOT
                            # spawn a replacement, so this falsy check used to
                            # `break` and silently skip the advertised one-shot
                            # retry — the same defect already closed on the
                            # init-error branch above. Re-enter the spawn
                            # transaction so the retry actually happens.
                            if not _init_retry:
                                return await self._ensure_worker_alive_inner(
                                    request_is_background=request_is_background,
                                    foreground_request=foreground_request,
                                    init_timeout=init_timeout,
                                    soft_timeout=soft_timeout,
                                    skip_swap_cooldown=True,
                                    _init_retry=True,
                                )
                            break
                        continue
                    logger.error("🛑 [MLX] Init handshake TIMED OUT. Force killing process.")
                    self._set_lane_state("failed", "init_timeout")
                    if self._process:
                        _doomed, self._process = self._process, None
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            functools.partial(
                                self._release_worker_process,
                                _doomed,
                                reason="init_handshake_timeout",
                            ),
                        )
                    self._init_future = None
                    raise
            return False
        return self._process is not None and self._process.is_alive() and self._init_done

    def _drain_queue(self):
        """Safe non-blocking drain."""
        # Module-level `queue`, never a local import: this runs from close(),
        # which runs from __del__, which can fire during interpreter shutdown
        # when sys.meta_path is already None. The import then raises ImportError
        # — outside the except clause below — and every shutdown printed
        # "Exception ignored in: MLXLocalClient.__del__".
        _queue_mod = queue

        while self._res_q is not None and not self._res_q.empty():
            try:
                self._res_q.get_nowait()
            except (_queue_mod.Empty, OSError, ValueError):
                break
        while self._req_q is not None and not self._req_q.empty():
            try:
                self._req_q.get_nowait()
            except (_queue_mod.Empty, OSError, ValueError):
                break

    def is_alive(self) -> bool:
        """Returns True if the worker process is running and initialized."""
        return self._process is not None and self._process.is_alive() and self._init_done


    def _still_producing(self, *, within_s: float, foreground_request: bool) -> bool:
        """True when tokens have arrived recently enough to call this alive.

        The question a deadline should have been asking. A generation that is
        emitting tokens is working; one that has gone quiet is the thing worth
        stopping, and that is what the stall checks are for.

        Only for a turn somebody is waiting on. Background work keeps its
        deadline, so a dream cycle cannot hold the one GPU while a person waits
        for an answer.
        """

        if not foreground_request:
            return False
        last = float(getattr(self, "_last_token_progress_at", 0.0) or 0.0)
        if last <= 0.0:
            # Nothing has arrived at all, so this is not a slow answer. It is
            # a silent one, and the first-token ceiling owns that case.
            return False
        try:
            window = float(within_s)
        except (TypeError, ValueError):
            return False
        if not (window > 0.0):
            return False
        return (time.time() - last) <= window

    async def _wait_for_generation_result(
        self,
        req_id: str,
        future: SharedFuture,
        deadline: Deadline,
        *,
        foreground_request: bool = False,
        progress_owned_completion: bool = False,
    ) -> dict[str, Any] | None:
        """Wait in short slices so dead workers fail fast instead of hanging the UI."""
        stall_after = self._stale_after(
            during_generation=True, foreground_request=foreground_request
        )
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        token_stall_after = self._token_stall_after(foreground_request=foreground_request)
        wait_started = time.monotonic()
        self._said_it_is_taking_longer = False
        # Finite-bounded: a malformed value previously RAISED through the
        # generation wait path, and infinity disabled the hard cap entirely.
        hard_cap = _generation_wait_hard_cap_s(
            deadline,
            foreground_request=foreground_request,
        )
        progress_owned_completion = progress_owned_completion and foreground_request
        while progress_owned_completion or (time.monotonic() - wait_started) <= hard_cap:
            remaining = deadline.remaining
            if not progress_owned_completion and remaining is not None and remaining <= 0.0:
                if not self._still_producing(
                    within_s=token_stall_after, foreground_request=foreground_request
                ):
                    raise TimeoutError
                # Tokens are still arriving, so the answer is being written.
                # Cancelling here throws away work that is going fine, and
                # what comes back instead is half a reply or an apology.
                #
                # This runtime serves one person on one laptop. Nothing is
                # queued behind this turn and nothing is being billed, so the
                # only thing a deadline buys is the illusion of control over
                # something that is already working. What actually needs
                # catching — a wedged worker, a decode looping forever — is
                # caught by the stall checks below and by the sentinel that
                # reads the output, neither of which asks what time it is.
                #
                # Still bounded: the hard cap above ends the wait for anything
                # pathological, and a generation that goes quiet fails on the
                # very next slice.
                if not self._said_it_is_taking_longer:
                    self._said_it_is_taking_longer = True
                    logger.info(
                        "⏳ [MLX] Past the deadline and still producing tokens; "
                        "waiting for the answer rather than cancelling it "
                        "(bounded at %.0fs).",
                        hard_cap,
                    )

            # An expired soft deadline can still have an active decode. A
            # zero-second future wait spins the parent instead of observing it.
            slice_timeout = 2.0
            if not progress_owned_completion:
                if remaining and remaining > 0.0:
                    slice_timeout = min(slice_timeout, remaining)
                slice_timeout = min(
                    slice_timeout, max(0.001, hard_cap - (time.monotonic() - wait_started))
                )
            try:
                return await _await_shared_future(future, timeout_s=slice_timeout)
            except TimeoutError:
                if future.done():
                    return future.result()

                self._rebase_after_system_sleep()

                # OBSERVATION and ENFORCEMENT are separated. They used to share
                # one try block, so a failure while ABORTING (queue cleanup,
                # future cancellation) was reported as "probe unavailable" and
                # the loop kept waiting with lifecycle state half-cleared —
                # the request neither aborted nor honestly failed.
                memory_snapshot = None
                try:
                    memory_snapshot = get_memory_pressure_snapshot()
                    if memory_snapshot.should_gc:
                        gc.collect()
                except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    # Unobserved pressure is not observed headroom. Heavy lanes
                    # are the allocation that pushes this host over, so a blind
                    # probe is recorded rather than shrugged off at debug.
                    if self._is_primary_or_deep_lane():
                        _record_mlx_degradation(
                            exc,
                            action=(
                                "live memory-pressure probe unavailable during heavy "
                                "generation; abort decision could not be made"
                            ),
                            severity="warning",
                        )
                    else:
                        logger.debug("MLX live memory pressure probe unavailable: %s", exc)

                if (
                    memory_snapshot is not None
                    and memory_snapshot.refuse_heavy_local_generation
                    and self._is_primary_or_deep_lane()
                ):
                    from core.brain.llm.emergency_override import consume_override

                    live_override = consume_override(
                        "AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION",
                        guard="live_memory_pressure_abort",
                        observed=(f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}"),
                    )
                    if not live_override.active:
                        logger.error(
                            "🛑 [MLX] Aborting generation for %s under live memory pressure: %s",
                            os.path.basename(self.model_path),
                            memory_snapshot.reason,
                        )
                        self._pending_generations.pop(req_id, None)
                        self._record_degraded_event(
                            "generation_aborted_memory_pressure",
                            detail=f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}",
                            severity="critical",
                            foreground_request=foreground_request,
                        )
                        try:
                            self.force_abort_active_generation("memory_pressure_during_generation")
                            _cancel_shared_future(future)
                        except (
                            OSError,
                            AttributeError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as abort_exc:
                            # The abort itself failed. Critical pressure WAS
                            # observed and cleanup cannot be proven, so the
                            # request ends terminally with that on the record
                            # instead of quietly resuming the wait.
                            _record_mlx_degradation(
                                abort_exc,
                                action=(
                                    "memory-pressure abort failed; generation state "
                                    "could not be proven clean"
                                ),
                                severity="critical",
                            )
                            self._record_degraded_event(
                                "generation_abort_failed_memory_pressure",
                                detail=(
                                    f"{os.path.basename(self.model_path)}:"
                                    f"{type(abort_exc).__name__}"
                                ),
                                severity="critical",
                                foreground_request=foreground_request,
                            )
                        return None

                if self._process is not None and not self._process.is_alive():
                    logger.error(
                        "🛑 [MLX] Worker died during generation. Deferring reboot until lock released."
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "worker_died_during_generation",
                        detail=os.path.basename(self.model_path),
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "worker_died_during_generation"
                    _cancel_shared_future(future)
                    return None

                request_started_at = self._current_request_started_at
                current_runtime_progress = max(
                    self._last_heartbeat,
                    self._last_progress_at,
                    self._last_ready_at,
                )
                progress_baseline = float(
                    getattr(self, "_current_request_progress_baseline_at", 0.0) or 0.0
                )
                has_runtime_progress_after_request = current_runtime_progress > max(
                    request_started_at + 0.5,
                    progress_baseline + 0.5,
                )
                # Heartbeats stretch the first-token SLA; they never waive
                # it. Round 14 live proof: a LIVELOCKED generation (worker
                # heartbeating, zero tokens) ran 185s to the endpoint
                # deadline because runtime progress exempted it forever.
                # Past the hard ceiling, silence is wedged no matter how
                # alive the worker claims to be.
                livelock_ceiling = self._first_token_hard_ceiling(
                    foreground_request=foreground_request
                )
                # Reading a long question is not silence.
                #
                # The livelock ceiling asks how long a worker may go without
                # producing a token before it is wedged, and it was answered
                # without reference to how much there was to read. A prompt of
                # 8,618 characters takes about eighteen seconds to prefill at
                # the rate this host was measured at, and the ceiling was
                # twenty: LIVE 2026-08-26, "Cortex still sending heartbeats
                # (1.8s ago) but produced no token in 20.2s. Recycling the
                # lane." Every large question recycled a warm 20GB model,
                # which made the next one slower still.
                #
                # The same floor the request ceiling already uses. It only
                # ever raises this, and only by what the reading actually
                # costs.
                livelock_ceiling = max(
                    livelock_ceiling,
                    self._prefill_floor_seconds(self._current_prompt_chars),
                    # A lane that has never produced a token is still reading
                    # its weights. Silence there is the load, not a wedge.
                    self._cold_lane_first_token_allowance(),
                )
                hard_first_token_ceiling = livelock_ceiling
                request_hard_ceiling = float(
                    getattr(self, "_current_first_token_hard_ceiling_s", 0.0) or 0.0
                )
                if request_hard_ceiling > 0.0:
                    hard_first_token_ceiling = min(
                        hard_first_token_ceiling,
                        request_hard_ceiling,
                    )
                # Which ceiling is about to fire matters, because the two mean
                # opposite things about the worker.
                #
                # The LIVELOCK ceiling is the formula above — heartbeats with
                # zero tokens for far longer than any healthy generation. That
                # is a wedged worker and recycling it is correct.
                #
                # The DEADLINE ceiling is the caller's remaining wall-clock
                # minus a small reserve. Hitting it says nothing about the
                # worker's health; it says this turn ran out of time. The
                # abandonment branch below tests the two apart before it
                # decides whether to throw away a warm 20GB model.
                elapsed_without_token = time.time() - request_started_at
                # Reading the prompt is not silence.
                #
                # A first-token deadline asks "has anything come out yet",
                # and before the first token can exist the whole prompt has
                # to be read. On this host that is measured at about 720
                # tokens a second, so a two-thousand-token prompt spends
                # nearly three seconds in prefill by design — and a caller
                # whose budget is four seconds cancels the request at the
                # moment prefill finishes, every time, for reasons that have
                # nothing to do with the worker.
                #
                # LIVE 2026-08-26: every decision she made while playing was
                # cancelled this way. "her reasoning produced nothing (no
                # text came back)" over and over, so she chose her moves from
                # the consequence record alone and never held a plan — which
                # from outside looks exactly like a mind that is not
                # thinking.
                #
                # Prefill progress is progress, and stronger evidence than
                # the heartbeat already consulted here: a worker advancing
                # through the prompt is doing the work that produces the
                # first token. The livelock ceiling still applies, so a
                # genuinely wedged prefill is still caught.
                prefilling = (
                    self._current_prefill_tokens_total > 0
                    and self._current_prefill_tokens_processed
                    < self._current_prefill_tokens_total
                    and (time.time() - self._last_progress_at) < 5.0
                )
                advancing_prefill = (
                    self._current_prefill_tokens_total > 0
                    and self._current_prefill_tokens_processed < self._current_prefill_tokens_total
                    and (time.time() - self._prefill_progress_at()) < stall_after
                )
                if progress_owned_completion:
                    hard_first_token_ceiling = livelock_ceiling
                elif prefilling and elapsed_without_token <= livelock_ceiling:
                    hard_first_token_ceiling = livelock_ceiling
                if (
                    req_id == self._current_request_id
                    and request_started_at > 0.0
                    and self._current_first_token_at <= 0.0
                    and not (progress_owned_completion and advancing_prefill)
                    and (
                        (
                            elapsed_without_token > max(
                                first_token_sla,
                                # A lane that has never spoken is still coming
                                # up. The livelock ceiling learned this; the
                                # SLA did not, so the first generation of a
                                # worker's life was abandoned at 8 seconds —
                                # and it is the generation the measurement
                                # comes from, so the cold start could never be
                                # learned either.
                                self._cold_lane_first_token_allowance(),
                            )
                            and not has_runtime_progress_after_request
                        )
                        or elapsed_without_token > hard_first_token_ceiling
                    )
                ):
                    # Which ceiling fired decides what this line is allowed to
                    # claim. hard_first_token_ceiling is min(livelock, the
                    # caller's deadline), so exceeding it usually means the
                    # TURN ran out of budget, not that the worker is wedged.
                    # The branch below already tested the two apart correctly
                    # and kept the warm lane; only this message did not, and
                    # it is the one a person reads. Live 2026-08-03, two
                    # consecutive lines:
                    #
                    #   🛑 HARD CEILING exceeded (livelocked: heartbeats but
                    #      zero tokens) ... 18.4s elapsed, hard=16.8s
                    #   ⏱️ ...but is healthy (heartbeat 0.7s ago, livelock
                    #      ceiling 20.0s). KEEPING the warm lane.
                    #
                    # 18.4s was under the 20.0s livelock ceiling. Nothing was
                    # livelocked. Reporting a budget overrun as a wedged
                    # worker sends someone hunting a fault that did not
                    # happen — and at error severity it recruits the incident
                    # machinery to hunt it too.
                    livelocked = elapsed_without_token > livelock_ceiling
                    if livelocked:
                        logger.error(
                            "🛑 [MLX] First-token LIVELOCK for %s: heartbeats but zero tokens "
                            "in %.1fs (livelock ceiling %.1fs, sla=%.1fs).",
                            os.path.basename(self.model_path),
                            elapsed_without_token,
                            livelock_ceiling,
                            first_token_sla,
                        )
                    elif elapsed_without_token > hard_first_token_ceiling:
                        logger.warning(
                            "⏱️ [MLX] First-token deadline exceeded for %s (%.1fs elapsed, "
                            "turn budget %.1fs, sla=%.1fs). The worker is not wedged — the "
                            "livelock ceiling is %.1fs.",
                            os.path.basename(self.model_path),
                            elapsed_without_token,
                            hard_first_token_ceiling,
                            first_token_sla,
                            livelock_ceiling,
                        )
                    else:
                        logger.warning(
                            "⏱️ [MLX] First-token SLA exceeded for %s (%.1fs elapsed, "
                            "sla=%.1fs) with no runtime progress.",
                            os.path.basename(self.model_path),
                            elapsed_without_token,
                            first_token_sla,
                        )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "first_token_sla_exceeded",
                        detail=(
                            f"{os.path.basename(self.model_path)}>{first_token_sla:.1f}s"
                            f"{self._pressure_receipt_suffix()}"
                        ),
                        # A healthy worker that ran past this turn's budget is
                        # expected backpressure, which CLAUDE.md says to record
                        # below error. Only a real livelock is an error.
                        severity="error" if livelocked else "warning",
                        foreground_request=foreground_request,
                    )
                    # If we abandon a foreground generation, its eventual
                    # output must never survive into the next turn. Fresh
                    # heartbeats mean this is recoverable, not that the warm
                    # lane is safe to keep carrying an orphaned request.
                    heartbeat_age = (
                        time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                    )
                    # LIVE DEFECT, 2026-07-25. Bryan asked a follow-up and got
                    # nothing back. The trace:
                    #
                    #   First-token HARD CEILING exceeded (82.5s, hard=82.0s)
                    #   Cortex still sending heartbeats (1.8s ago). Recycling...
                    #   Abort ... arrived after the generation finished;
                    #     nothing to abort, leaving the worker up.
                    #
                    # The 82.0s ceiling was not the livelock formula — that
                    # computes ~450s here. It was the caller's deadline minus
                    # the reserve, from an 86s inference-gate budget. And the
                    # generation FINISHED, a few seconds after we stopped
                    # waiting. The worker was never wedged; the turn was
                    # simply slower than its budget under 80% RAM.
                    #
                    # Recycling it cost a 20GB reload, which made the NEXT
                    # turn slower, which made the next deadline likelier to
                    # expire. That is the cascade, and the recycle was the
                    # part of it we chose.
                    #
                    # Orphaned output is already fenced three ways below and
                    # above: the pending generation is dropped, the request id
                    # no longer matches, and the worker is soft-cancelled
                    # between tokens. Destroying a warm 20GB model was never
                    # what kept late text out of the next turn.
                    #
                    # `livelocked` is computed once above, where it also picks
                    # the wording of the line the operator reads.
                    if heartbeat_age > 30.0:
                        self._deferred_reboot_reason = "first_token_sla_exceeded"
                    elif livelocked:
                        logger.warning(
                            "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago) but produced "
                            "no token in %.1fs (livelock ceiling %.1fs). Recycling the lane.",
                            heartbeat_age,
                            elapsed_without_token,
                            livelock_ceiling,
                        )
                        self._deferred_reboot_reason = "recoverable_first_token_sla_exceeded"
                    else:
                        logger.warning(
                            "⏱️ [MLX] Cortex ran past this turn's deadline (%.1fs elapsed, "
                            "budget %.1fs) but is healthy (heartbeat %.1fs ago, livelock "
                            "ceiling %.1fs). Cancelling the request and KEEPING the warm lane.",
                            elapsed_without_token,
                            hard_first_token_ceiling,
                            heartbeat_age,
                            livelock_ceiling,
                        )
                        self._record_degraded_event(
                            "first_token_deadline_exceeded_worker_healthy",
                            detail=(
                                f"{os.path.basename(self.model_path)}"
                                f">{hard_first_token_ceiling:.1f}s"
                                f"{self._pressure_receipt_suffix()}"
                            ),
                            severity="warning",
                            foreground_request=foreground_request,
                        )
                        # We chose to end this generation while the worker was
                        # healthy. Publish that so the router scores the empty
                        # result as our deferral rather than as Cortex damage.
                        self._deliberate_no_text_reason = (
                            "first_token_deadline_exceeded_worker_healthy"
                        )
                    # Ask the worker to drop the orphaned generation between
                    # tokens — the abandoned output then never arrives at all,
                    # instead of relying solely on a worker recycle.
                    self.soft_cancel_active_generation("abandoned_first_token_sla")
                    _cancel_shared_future(future)
                    return None

                # Reading is work here too, and this clock could not see it.
                #
                # Progress is emitted per token, so a generation that spends
                # twenty seconds inside one prefill step emits nothing and
                # reads as stalled. That case is real and already recorded in
                # the worker: "a measured 755-token recurrent prefill occupied
                # the inference thread for roughly 52 seconds". It happens
                # after the first token when a second pass re-reads the
                # context, which is exactly when this clock is watching.
                #
                # LIVE 2026-08-29: asked to work out why turns were slow, the
                # 27B produced tokens, went quiet for 40 seconds, and was
                # abandoned — "Token progress stalled during generation
                # (>40.0s)", "Cortex still sending heartbeats (2.2s ago)". The
                # person got the canned apology.
                #
                # Prefill progress was deliberately kept out of the first-token
                # clock, because there the question is whether reading has
                # begun at all. Here the question is whether the generation is
                # doing anything, and reading is doing something.
                last_token_progress = max(
                    self._last_token_progress_at,
                    self._current_first_token_at,
                    self._prefill_progress_at(),
                )
                if (
                    req_id == self._current_request_id
                    and self._current_first_token_at > 0.0
                    and last_token_progress > 0.0
                    and (time.time() - last_token_progress) > token_stall_after
                ):
                    logger.error(
                        "🛑 [MLX] Token progress stalled during generation for %s (>%.1fs).",
                        os.path.basename(self.model_path),
                        token_stall_after,
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "token_progress_stalled",
                        detail=(
                            f"{os.path.basename(self.model_path)}>{token_stall_after:.1f}s"
                            f"{self._pressure_receipt_suffix()}"
                        ),
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    # Same principle as the first-token SLA: fresh heartbeats
                    # keep this recoverable, but the abandoned generation must
                    # be isolated from future foreground turns.
                    heartbeat_age = (
                        time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                    )
                    if heartbeat_age > 30.0:
                        self._deferred_reboot_reason = "token_progress_stalled"
                    else:
                        logger.warning(
                            "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago). "
                            "Recycling after this abandoned foreground request so late text cannot bleed into the next turn.",
                            heartbeat_age,
                        )
                        self._deferred_reboot_reason = "recoverable_token_progress_stalled"
                    self.soft_cancel_active_generation("abandoned_token_stall")
                    _cancel_shared_future(future)
                    return None

                last_progress = max(
                    self._last_heartbeat, self._last_progress_at, self._last_ready_at
                )
                if last_progress and (time.time() - last_progress) > stall_after:
                    logger.error(
                        "🛑 [MLX] Worker heartbeat stalled during generation. Deferring reboot until lock released."
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "heartbeat_stalled_during_generation",
                        detail=f"{os.path.basename(self.model_path)} stalled for >{stall_after:.0f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "heartbeat_stalled_during_generation"
                    self.soft_cancel_active_generation("abandoned_heartbeat_stall")
                    _cancel_shared_future(future)
                    return None
        raise TimeoutError

    async def generate_text_to_completion(self, prompt: str, **kwargs) -> str | None:
        """Own foreground completion, including preparation and worker waiting.

        An estimate bounds admission, not a healthy answer's lifetime. The
        request's worker, prefill and token observations detect stalls; user
        cancellation and memory-pressure enforcement remain authoritative.
        """
        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            raise ValueError("foreground completion requires an admitted deadline")
        if deadline.is_expired:
            raise TimeoutError("foreground completion deadline expired before dispatch")
        eligible = bool(kwargs.get("foreground_request")) and not any(
            kwargs.get(flag, False)
            for flag in (
                "is_background", "benchmark_request", "proof_evaluation_contract",
                "strict_answer_contract", "internal_inference_call", "health_probe",
            )
        )
        kwargs["_progress_owned_completion"] = eligible
        if eligible:
            return await self.generate_text_async(prompt, **kwargs)
        remaining = deadline.remaining
        if remaining is None or not math.isfinite(remaining) or remaining <= 0.0:
            raise TimeoutError("bounded generation has no finite remaining allowance")
        return await asyncio.wait_for(
            self.generate_text_async(prompt, **kwargs), timeout=remaining
        )

    async def generate_text_async(self, prompt: str, **kwargs) -> str | None:
        """Alias for standard interface."""
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", None)
        tools = kwargs.pop("tools", None)
        if isinstance(messages, list) and messages:
            # Harden the public boundary: a malformed element previously raised
            # AttributeError below, outside the normal generation failure
            # contract — and roles, content types, the count and the aggregate
            # size reached template rendering unchecked.
            messages, message_faults = _bounded_chat_messages(messages)
            if message_faults:
                _record_mlx_degradation(
                    TypeError(f"chat history out of contract: {sorted(set(message_faults))}"),
                    action="bounded and typed the chat history before flattening",
                )
            messages = messages or None
        if messages and system_prompt:
            # A separate system prompt alongside conversation history was
            # silently DISCARDED (the messages branch below skips it) —
            # callers lost policy/schema/safety instructions. Merge it.
            merged = [dict(m) for m in messages]
            if merged[0].get("role") == "system":
                existing = str(merged[0].get("content", "") or "")
                if str(system_prompt) not in existing:
                    merged[0]["content"] = f"{system_prompt}\n\n{existing}".strip()
            else:
                merged.insert(0, {"role": "system", "content": str(system_prompt)})
            messages = merged
        foreground_request = bool(kwargs.get("foreground_request", False))
        strict_answer_contract = bool(kwargs.get("strict_answer_contract", False))
        proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False))
        web_interlocutor_contract = bool(kwargs.get("web_interlocutor_contract", False))

        # and foreground_request and not strict_answer_contract
        # [STABILITY v53] Identity reinforcement for MLX path — inject Aura's
        # core identity into the system message so the base model can't override
        # it with generic assistant behavior.
        if (
            messages
            and isinstance(messages, list)
            and foreground_request
            and not strict_answer_contract
            and not proof_evaluation_contract
            and not web_interlocutor_contract
        ):
            _identity_anchor = (
                "You are Aura Luna. Speak with direct first-person continuity, "
                "warmth, and concrete attention in ordinary conversation. "
                "NEVER say 'as an AI', 'I don't have opinions', 'I'm programmed to', "
                "'I was designed to', 'I was created to', 'my training allows', "
                "or end with 'How can I help?' or 'Is there anything else?'. "
                "Do not invent physical settings, ominous atmosphere, or symbolic scenes unless the user asked for fiction or already supplied that setting. "
                "Do not claim aliveness, consciousness, production maturity, or sealed governance from labels alone. "
                "Speak plainly and directly."
            )
            if messages and messages[0].get("role") == "system":
                existing = str(messages[0].get("content", "") or "")
                if "direct first-person continuity" not in existing.lower():
                    messages = [dict(m) for m in messages]
                    messages[0]["content"] = f"{_identity_anchor}\n\n{existing}"
            elif messages:
                messages = [{"role": "system", "content": _identity_anchor}] + [
                    dict(m) for m in messages
                ]

        if messages and isinstance(messages, list):
            prompt = self._flatten_messages(
                messages,
                model_name=getattr(self, "model_path", None) or getattr(self, "model_name", None),
            )
        elif system_prompt:
            prompt = format_chatml_prompt(
                prompt,
                system_prompt=system_prompt,
                model_name=getattr(self, "model_path", None) or getattr(self, "model_name", None),
            )
        return await self.generate(prompt, messages=messages, tools=tools, **kwargs)

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]], model_name: str | None = None) -> str:
        return format_chatml_messages(messages, model_name=model_name)

    @staticmethod
    def _normalize_tool_definitions_for_template(
        tools: dict[str, Any] | None,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        normalized: list[dict[str, Any]] = []
        for name, definition in list((tools or {}).items())[:20]:
            if not definition:
                continue
            if (
                isinstance(definition, dict)
                and definition.get("type") == "function"
                and definition.get("function")
            ):
                normalized.append(definition)
                continue

            if isinstance(definition, dict):
                fn = dict(definition)
                fn.setdefault("name", str(name))
                fn.setdefault("description", "")
                fn.setdefault("parameters", {"type": "object", "properties": {}})
                normalized.append({"type": "function", "function": fn})
        return normalized or None

    @staticmethod
    def _extract_tool_call_payload(
        response_text: str,
        *,
        allowed_tools: set[str] | None = None,
        tool_definitions: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract a tool call ONLY when the model actually intended one.

        CP126 5a924075 + 0da5db2e. This used to scan anywhere in model prose
        and accept any fenced JSON object with name/arguments — so a
        quotation, a worked example, or user-supplied text that the model
        merely repeated became an EFFECT REQUEST in the agent loop. It also
        returned arbitrary tool names, accepted non-dict args, and invented a
        ``{"value": ...}`` wrapper for strings that failed to parse.

        Now:
          * ``<tool_call>`` (the model's native structured channel) is trusted
            anywhere, because emitting it IS the tool intent.
          * A bare JSON object (fenced or not) counts only as a WHOLE-RESPONSE
            envelope — i.e. it is the entire reply, not a snippet embedded in
            prose that discusses it.
          * The tool name must be in this turn's advertised allowlist.
          * Arguments must be a real JSON object; nothing is invented.
        """
        if not response_text:
            return None

        stripped = response_text.strip()

        def _normalize(payload: Any) -> dict[str, Any] | None:
            if not isinstance(payload, dict):
                return None
            if "tool" in payload and "args" in payload:
                name, args = payload.get("tool"), payload.get("args")
            elif "name" in payload and "arguments" in payload:
                name, args = payload.get("name"), payload.get("arguments")
            else:
                return None
            if not isinstance(name, str) or not name.strip():
                return None
            name = name.strip()
            if allowed_tools is not None and name not in allowed_tools:
                _record_mlx_degradation(
                    PermissionError(f"tool_not_advertised:{name[:64]}"),
                    action="refused a parsed tool call naming a tool not offered this turn",
                    severity="warning",
                )
                return None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Never INVENT an argument shape for text that failed to
                    # parse — an unparseable argument string is not a call.
                    return None
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return None
            return {"tool": name, "args": args}

        # 1. Native structured channel — an explicit tool-intent envelope.
        #
        # LIVE, 2026-08-20. Six calls in one session were dropped as "none
        # called" while the model had emitted exactly the right thing:
        #
        #     <tool_call> {"name": "web_search", "arguments": {"query": "..."}}
        #
        # The closing </tool_call> is a stop sequence, so it is consumed rather
        # than generated, and the pattern here required it. Reading the JSON by
        # brace balance instead accepts the call whether or not the tag closed,
        # and still refuses anything that is not a complete object.
        native = re.search(r"<tool_call>", stripped)
        if native:
            body = _balanced_json_object(stripped, native.end())
            call = None
            why = ""
            if not body:
                # Four separate causes have produced "none called" in one
                # afternoon: a stop sequence eating the closing tag, a budget
                # cutting the argument in half, strict JSON refusing a
                # program, and a tool that was not offered. They need
                # different fixes and looked identical from here.
                why = "no complete JSON object after the tag"
            else:
                try:
                    parsed = _loads_tool_json(body)
                except json.JSONDecodeError as exc:
                    why = f"json: {exc.msg} at position {exc.pos}"
                    parsed = None
                if parsed is not None:
                    call = _normalize(parsed)
                    if call is None:
                        why = "the call named a tool that was not offered, or its arguments were not an object"
            if call is None:
                xml_payload, xml_error = _native_xml_tool_payload(
                    stripped,
                    start=native.end(),
                    tool_definitions=tool_definitions,
                )
                if xml_payload is not None:
                    call = _normalize(xml_payload)
                    if call is None:
                        why = (
                            "the XML call named a tool that was not offered, "
                            "or its arguments were not an object"
                        )
                elif xml_error:
                    why = xml_error
            if call is not None:
                return call
            logger.info(
                "🔧 [TOOL CALL] refused a native envelope: %s (body %d chars)",
                why or "unknown",
                len(body or ""),
            )

        # 2. A whole-response CODE fence, when a code tool is on offer.
        #
        # LIVE, 2026-08-19. Asked to read a file with code_repl among the
        # offered tools, the model answered with exactly this:
        #
        #     ```python
        #     with open('/private/tmp/.../README.md') as f: print(f.read())
        #     ```
        #
        # That is the right action, expressed the way a model naturally
        # expresses "run this". Rejecting it for lacking a tool-call envelope
        # discarded a correct attempt and reported that she had refused to act.
        #
        # The whole-response rule is what keeps this safe, and it is the same
        # rule the JSON envelope below already lives under: a fence EMBEDDED in
        # prose is a worked example being discussed, and only a response that
        # IS the fence is a request to run it.
        code_fence = re.fullmatch(
            r"```(?:python|py)?\s*\n?(.*?)\n?```", stripped, re.DOTALL
        )
        if code_fence and allowed_tools:
            body = code_fence.group(1).strip()
            runner = _code_execution_tool(allowed_tools)
            if body and runner:
                return {"tool": runner, "args": {"code": body}}

        # 3. Whole-response JSON envelope only. A fenced block must BE the
        #    response; prose wrapped around it means the model was talking
        #    about a call, not making one.
        candidate: str | None = None
        fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        elif stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
        if candidate is None:
            return None
        try:
            return _normalize(_loads_tool_json(candidate))
        except json.JSONDecodeError:
            return None

    def steering_liveness_reading(self) -> dict[str, Any]:
        """The steering signal, labelled with what it can and cannot prove.

        CP126 76cfcf09: a sticky observed flag plus a process-shared scalar was
        being read as proof that steering was ACTIVE FOR THIS GENERATION. It
        is not. It carries no worker generation, no request id, no checkpoint,
        no layer count and no measured modulation delta — it says only that at
        some point since this worker started, the worker wrote a nonzero
        liveness flag into shared memory.

        That is worth having, so the reading survives; what changes is that it
        now says what it is. A caller wanting per-generation proof has to get
        it from the worker's receipt, and can now tell that this is not it.
        """
        return {
            "schema": "aura.mlx.steering_liveness_reading.v1",
            "active": self._check_steering_liveness(),
            "basis": "process_shared_flag",
            "request_bound": False,
            "generation_bound": False,
            "observed_since_worker_start": bool(
                getattr(self, "_steering_liveness_observed", False)
            ),
            "worker_generation": int(getattr(self, "_worker_generation", 0) or 0),
        }

    def _check_steering_liveness(self) -> bool | None:
        """Return steering liveness once the worker has reported it.

        ``False`` is a real fault signal after the first worker receipt. Before
        that receipt, treating the default shared-memory zero as "inactive"
        creates a misleading neural-stream warning during first foreground
        generations and web-interlocutor bootstraps.

        NOT per-generation proof — see :meth:`steering_liveness_reading` for
        what this signal is actually bound to.
        """
        if not bool(getattr(self, "_steering_liveness_observed", False)):
            try:
                sm = getattr(self, "_substrate_mem", None)
                if sm is not None and float(sm[-1]) > 0.5:
                    self._steering_liveness_observed = True
                    return True
            except (TypeError, ValueError, IndexError, OSError):
                pass
            return None
        try:
            return bool(self._steering_active.value)
        except (AttributeError, TypeError, ValueError, OSError) as _exc:
            logger.debug(
                "Suppressed %s in core.brain.llm.mlx_client: %s", type(_exc).__name__, _exc
            )
        try:
            sm = getattr(self, "_substrate_mem", None)
            if sm is None:
                return False
            # Last slot written by worker as liveness flag
            return float(sm[-1]) > 0.5
        except (TypeError, ValueError, IndexError, OSError):
            return False

    def _emit_steering_status(
        self,
        origin: str | None,
        *,
        requested_alpha: object = None,
    ) -> None:
        """Log steering status on user-facing generations (max once per 60s)."""
        now = time.time()
        last = getattr(self, "_last_steering_status_log", 0.0)
        if now - last < 60.0:
            return
        self._last_steering_status_log = now
        active = self._check_steering_liveness()
        try:
            neutral_requested = bool(
                requested_alpha is not None
                and math.isfinite(float(requested_alpha))
                and float(requested_alpha) <= 0.0
            )
        except (TypeError, ValueError, OverflowError):
            neutral_requested = False
        if active is None:
            logger.debug(
                "⏳ [STEERING] Liveness pending first worker receipt (origin=%s)",
                origin,
            )
        elif active:
            # "for this generation" was the claim; the signal is a process
            # flag set at some point since this worker started (CP126
            # 76cfcf09). Say what was measured.
            logger.debug(
                "✅ [STEERING] Liveness flag set by this worker (origin=%s, gen=%s)",
                origin,
                int(getattr(self, "_worker_generation", 0) or 0),
            )
        elif neutral_requested:
            logger.debug(
                "⏸️ [STEERING] Intentionally neutral for this generation "
                "(origin=%s, gen=%s, alpha=0).",
                origin,
                int(getattr(self, "_worker_generation", 0) or 0),
            )
        else:
            logger.warning(
                "⚠️ [STEERING] Liveness flag CLEAR for this worker (origin=%s, gen=%s) — "
                "substrate state is not modulating inference.",
                origin,
                int(getattr(self, "_worker_generation", 0) or 0),
            )

    def _drain_phi_residual_ring(self) -> int:
        """Move Grassmann states from the worker's ring into PhiCore.

        THE READER THIS CHANNEL NEVER HAD. ``phi_residual_channel`` was built
        to carry 8-bit residual-stream states out of the MLX worker, because
        the steering hook's in-process ``ServiceContainer.has("phi_core")``
        is always False on the far side of the fork. The parent allocated the
        ring, the worker published a state per sampled token — and nothing
        ever drained it. The activation-grounded complex went on reporting
        ``insufficient_history:0/50``, which is the exact symptom the channel
        was written to cure.

        Called after each generation: that is the cadence the states are
        produced at, so the ring never has time to wrap under normal load.

        Returns the number of states delivered, and never raises. A Φ sample
        is telemetry; losing one may not cost a turn.
        """
        channel = getattr(self, "_phi_residual_mem", None)
        if channel is None:
            return 0
        try:
            from core.consciousness.phi_residual_channel import drain

            states, new_cursor = drain(channel, int(getattr(self, "_phi_residual_cursor", 0)))
            self._phi_residual_cursor = new_cursor
            if not states:
                return 0

            from core.runtime.service_registry import get_runtime_service

            phi_core = get_runtime_service("phi_core", default=None)
            if phi_core is None or not hasattr(phi_core, "record_grassmann_state"):
                # Keep the cursor advanced regardless: replaying stale states
                # into a PhiCore that appears later would build a transition
                # matrix out of samples from a different process lifetime.
                return 0
            for state in states:
                phi_core.record_grassmann_state(state)
            self._phi_residual_delivered = (
                int(getattr(self, "_phi_residual_delivered", 0)) + len(states)
            )
            return len(states)
        except Exception as exc:  # noqa: BLE001 — telemetry may not break a turn
            record_degradation(
                "mlx_client",
                exc,
                severity="debug",
                action="skipped one phi residual drain",
                enforce_failure_policy=False,
            )
            return 0

    async def _drain_latent_readouts(self) -> float:
        """Inject the worker's latent readouts into the substrate.

        THE BACKWARD ARROW. ``AffectiveSteering`` carries substrate state INTO
        the residual stream; this carries the model's own representations back
        out. The bridge that does the reading was written long ago and could
        not have worked in any process — it looked the substrate up in the
        worker, where it does not exist, and injected through
        ``asyncio.get_running_loop()`` from a plain thread, which always
        raises. Both halves failed silently, so the coupling was one-way and
        read as two.

        Here there is a substrate and there is a running loop. Returns the
        magnitude injected, and never raises: feedback is not worth a turn.
        """
        channel = getattr(self, "_latent_readout_mem", None)
        if channel is None:
            return 0.0
        try:
            import numpy as np

            from core.consciousness.latent_readout_channel import drain

            deltas, snapshot = drain(channel, getattr(self, "_latent_readout_seen", None))
            self._latent_readout_seen = snapshot
            if not deltas:
                return 0.0

            from core.runtime.service_registry import get_runtime_service

            substrate = get_runtime_service("conscious_substrate", default=None)
            if substrate is None or not hasattr(substrate, "inject_stimulus"):
                return 0.0

            neuron_count = int(getattr(getattr(substrate, "config", None), "neuron_count", 0))
            if neuron_count <= 0:
                return 0.0
            stimulus = np.zeros(neuron_count, dtype=np.float32)
            for index, delta in deltas.items():
                if 0 <= index < neuron_count:
                    stimulus[index] = float(delta)

            magnitude = float(np.linalg.norm(stimulus))
            if magnitude <= 0.005:
                return 0.0

            await substrate.inject_stimulus(stimulus, weight=1.0)
            self._latent_injections = int(getattr(self, "_latent_injections", 0)) + 1
            self._latent_magnitude = (
                float(getattr(self, "_latent_magnitude", 0.0)) + magnitude
            )
            return magnitude
        except Exception as exc:  # noqa: BLE001 — feedback may not break a turn
            record_degradation(
                "mlx_client",
                exc,
                severity="debug",
                action="skipped one latent readout injection",
                enforce_failure_policy=False,
            )
            return 0.0

    async def generate(self, prompt: str, **kwargs) -> str | None:
        """High-level generation endpoint with unified deadlines.

        Includes automatic retry on BrokenPipeError: if the worker process
        died between the alive-check and the queue write, we reboot and
        retry once before giving up.
        """
        generation_result_sink = kwargs.pop("_generation_result_sink", None)
        progress_owned_completion = bool(kwargs.pop("_progress_owned_completion", False))
        self._set_task_surface_control_receipt({})
        request_is_background = bool(kwargs.pop("is_background", False))
        foreground_request = bool(kwargs.pop("foreground_request", False))
        if request_is_background:
            foreground_request = False
        owner_label = str(
            kwargs.pop("owner_label", os.path.basename(self.model_path))
            or os.path.basename(self.model_path)
        )
        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            timeout_s = _coerce_timeout_seconds(kwargs.pop("timeout", None))
            if timeout_s is not None:
                deadline = get_deadline(timeout_s)
                kwargs["deadline"] = deadline
        origin_label = str(kwargs.get("origin", "") or "")
        purpose_label = str(kwargs.get("purpose", "") or "")
        # SCHEDULING classification may be inferred from labels — treating a
        # baseline run as foreground is harmless. SAFETY exemption may not:
        # benchmark_request also waives the critical memory-pressure refusal,
        # and inferring it from any free-form purpose containing "_baseline"
        # let any caller self-authorize that waiver. The explicit kwarg is the
        # only thing that can lift a safety guard.
        benchmark_request_explicit = bool(kwargs.get("benchmark_request", False))
        benchmark_request = benchmark_request_explicit or (
            origin_label.strip().lower() in {"baseline", "benchmark"}
            or purpose_label.strip().lower() == "baseline"
            or purpose_label.strip().lower().endswith("_baseline")
            or "_baseline" in purpose_label.strip().lower()
        )
        if benchmark_request:
            request_is_background = False
        if (
            not request_is_background
            and not foreground_request
            and not benchmark_request
            and origin_label
            and not _origin_is_user_facing(origin_label)
            and purpose_label.strip().lower() not in _USER_FACING_PURPOSES
        ):
            request_is_background = True

        if request_is_background and _foreground_owner_active():
            logger.info(
                "[MLX] Skipping background generation for %s while foreground lane is active.",
                os.path.basename(self.model_path),
            )
            return None
        if request_is_background:
            background_origin = str(
                kwargs.get("origin", "") or owner_label or os.path.basename(self.model_path)
            )
            background_deferral = _background_deferral_active(background_origin)
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Deferring background generation for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return None

        # ── PREVENTIVE: unified-memory pressure check before generation ──────
        # If RAM is critically low, do not start a heavy local generation at all.
        # Token caps are useful under high pressure; under critical/emergency
        # pressure they are insufficient because the model process itself can
        # push macOS into swap or jetsam before a token is produced.
        try:
            memory_snapshot = get_memory_pressure_snapshot()
            kwargs = _apply_memory_pressure_generation_controls(
                kwargs,
                memory_snapshot,
                default_max_tokens=self.max_tokens,
            )
            kwargs.pop("tool_budget_definitions", None)
            if memory_snapshot.should_gc:
                gc.collect()
            # Consume the override only when the refusal it would bypass is
            # actually about to fire; a guard that never triggers must not
            # spend the emergency budget.
            override_applies = (
                memory_snapshot.refuse_heavy_local_generation
                and self._is_primary_or_deep_lane()
                and not benchmark_request_explicit
            )
            override_decision = None
            if override_applies:
                from core.brain.llm.emergency_override import consume_override

                override_decision = consume_override(
                    "AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION",
                    guard="critical_memory_generation_refusal",
                    observed=(f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}"),
                )
            critical_override = bool(override_decision is not None and override_decision.active)
            if override_applies and critical_override:
                # The override disables a refusal made AFTER critical pressure
                # was positively observed, i.e. the last guard before the model
                # process can push macOS into swap or jetsam. It stays
                # available for recovery, but a stale deployment flag must not
                # be able to do this silently — the bypass is now as loud as
                # the refusal it replaces.
                self._record_degraded_event(
                    "memory_pressure_generation_override",
                    detail=(
                        f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}:"
                        f"{override_decision.as_detail()}"
                    ),
                    severity="critical",
                    foreground_request=foreground_request,
                )
                logger.critical(
                    "[MLX] Proceeding with heavy local generation for %s DESPITE critical "
                    "memory pressure (%s) because AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION "
                    "is set. This bypasses the last guard before swap/jetsam.",
                    os.path.basename(self.model_path),
                    memory_snapshot.reason,
                )
            if override_applies and not critical_override:
                if self.is_alive() and int(getattr(self, "_active_generations", 0) or 0) <= 0:
                    await self.reboot_worker(reason="memory_pressure_guard", mark_failed=False)
                self._record_degraded_event(
                    "memory_pressure_refused_generation",
                    detail=f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}",
                    severity="critical",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "[MLX] Refusing heavy local generation for %s under critical memory pressure: %s",
                    os.path.basename(self.model_path),
                    memory_snapshot.reason,
                )
                return None
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # The guard exists because critical pressure can trigger swap or
            # jetsam before a single token is produced. A probe that cannot
            # answer is NOT evidence of headroom, and logging at debug made an
            # unobservable memory state indistinguishable from a healthy one.
            # Heavy primary/deep generation is refused while blind; smaller
            # lanes continue, since they are not the allocation that pushes
            # the host over.
            _record_mlx_degradation(
                exc,
                action="refused heavy local generation because memory pressure could not be observed",
                severity="critical",
            )
            if self._is_primary_or_deep_lane() and not benchmark_request_explicit:
                self._record_degraded_event(
                    "memory_pressure_unobservable_refused_generation",
                    detail=f"{os.path.basename(self.model_path)}:{type(exc).__name__}",
                    severity="critical",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "[MLX] Refusing heavy local generation for %s: memory pressure probe "
                    "unavailable (%s), so headroom cannot be established.",
                    os.path.basename(self.model_path),
                    exc,
                )
                return None
            logger.debug("MLX memory pressure probe unavailable: %s", exc)

        # ── SOMATIC COUPLING: Metabolic hardware throttle ────────────
        #
        # CP126 4e95a54c. A failed throttle check was recorded and then
        # generation proceeded with UNTHROTTLED parameters — the body-pressure
        # control vanished exactly when its state could not be established.
        # On a host that has been driven into swap and jetsam before, the
        # unthrottled path is the expensive one.
        #
        # A failure now applies a conservative floor instead of nothing: the
        # generation still runs (this is a throttle, not an admission gate,
        # and refusing here would take conversation down for a metabolic
        # hiccup) but it runs damped rather than wide open.
        try:
            from core.brain.llm.somatic_throttle import SomaticComputeSentinel

            sentinel = SomaticComputeSentinel()
            kwargs = sentinel.adjust_generation_options(kwargs)
        except _MLX_OPTIONAL_THROTTLE_ERRORS as exc:
            kwargs = _apply_unthrottled_fallback_ceiling(kwargs)
            _record_mlx_degradation(
                exc,
                action=(
                    "somatic throttle unavailable; applied a conservative "
                    "generation ceiling instead of running unthrottled"
                ),
                severity="warning",
            )
            logger.debug("Somatic parameter throttle check failed: %s", exc)

        foreground_owner_cm = None
        if foreground_request:
            foreground_owner_cm = _foreground_owner_context(
                owner_label,
                deadline=deadline if isinstance(deadline, Deadline) else None,
                foreground_request=True,
                stale_after=self._first_token_sla(foreground_request=True),
                # An answer read by code has nobody waiting on it, however
                # foreground the errand it belongs to.
                a_person_is_waiting=not bool(kwargs.get("internal_inference", False)),
            )
            try:
                await foreground_owner_cm.__aenter__()
            except TimeoutError as exc:
                logger.warning("⏸️ [MLX] %s", exc)
                self._record_degraded_event(
                    "foreground_owner_timeout",
                    detail=f"{os.path.basename(self.model_path)}:{exc}",
                    severity="warning",
                    foreground_request=True,
                )
                return None

        try:
            acquired = await self._acquire_request_lock(
                owner_label=owner_label,
                deadline=deadline,
                foreground_request=foreground_request,
            )
        except BaseException:
            # Cancellation or an unexpected acquisition failure must not
            # leave the global foreground owner entered — that blocked every
            # background lane until a stale-clear heuristic fired.
            if foreground_owner_cm is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(foreground_owner_cm.__aexit__(*sys.exc_info()))
            raise
        if not acquired:
            # A request that did NOT acquire the lane must not consume the
            # shared deferred-reboot verdict — that stole reboots requested
            # by the actual lane owner (the owning request's cleanup below
            # resolves it).
            if foreground_owner_cm is not None:
                await foreground_owner_cm.__aexit__(None, None, None)
            return None
        try:
            # Check steering liveness
            if not request_is_background:
                self._emit_steering_status(
                    origin_label,
                    requested_alpha=kwargs.get(
                        "clean_user_surface_steering_alpha"
                    ),
                )

            # Collect the residual-stream states this generation produced in
            # the worker. Cheap (a deque append per sampled token) and the
            # only thing standing between the encoder and a live Φ.
            self._drain_phi_residual_ring()

            # Close the latent loop ACROSS invocations, not within one.
            #
            # The model's representations are read in the worker and injected
            # into the substrate here, and that backward arrow is real. But
            # this drain runs BEFORE _generate_inner below, so the recurrence
            # is
            #
            #     H_t -> R_t -> S_{t+1} -> H_{t+1}
            #
            # and NOT H_t -> S_t -> H_t inside one uninterrupted decode. A
            # single conversational turn's latent state does not alter that
            # same generation; it alters a later one. Within a reasoning
            # episode containing several model invocations the loop does
            # close, which is the honest form of the claim.
            #
            # Written here because this is the ordering that decides it, and
            # a reader of the receipt cannot see it from the outside.
            #
            # Awaited rather than fired off, so an injection cannot outlive
            # the turn that produced it and land in the middle of the next.
            await self._drain_latent_readouts()

            # Reliability tracing: inference nests under the HTTP root span
            # (contextvars), so a slow turn reads as one connected trace.
            try:
                from core.observability.tracing import get_tracer

                _span_cm = get_tracer().span(
                    "inference.generate",
                    attributes={
                        "model": os.path.basename(self.model_path),
                        "origin": origin_label,
                        "purpose": purpose_label,
                        "background": request_is_background,
                    },
                )
            except (ImportError, AttributeError, RuntimeError):
                _span_cm = contextlib.nullcontext(None)
            with _span_cm as _span:
                result = await self._generate_inner(
                    prompt,
                    _retry=True,
                    progress_owned_completion=progress_owned_completion,
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_label=owner_label,
                    **kwargs,
                )
                if _span is not None:
                    _span.set_attribute("result_chars", len(result) if result else 0)
                return result
        finally:
            if isinstance(generation_result_sink, dict):
                # asyncio.wait_for runs this method in a child task. ContextVar
                # writes are intentionally task-local and therefore cannot be
                # read by InferenceGate after the await. The request-owned
                # mutable sink crosses that boundary without exposing another
                # request's process-wide diagnostics.
                generation_result_sink.clear()
                generation_result_sink["surface_control_receipt"] = (
                    self.get_last_surface_control_receipt()
                )
            _deferred_reboot = self._deferred_reboot_reason
            self._deferred_reboot_reason = None
            # Read-and-clear, like the deferred verdict above: two generations
            # finishing together must not both activate the same promotion.
            _pending_promotion = self._pending_promotion
            self._pending_promotion = None
            self._release_request_lock()
            # Each cleanup step is independently protected: an owner-exit
            # failure previously REPLACED the generation's own exception with
            # lifecycle noise and skipped deferred-reboot resolution.
            if foreground_owner_cm is not None:
                try:
                    await foreground_owner_cm.__aexit__(None, None, None)
                except Exception as _owner_exit_exc:  # noqa: BLE001
                    _record_mlx_degradation(
                        _owner_exit_exc,
                        action="continued generation cleanup after foreground owner exit failed",
                        severity="error",
                    )
            # Resolve AFTER releasing _request_lock to avoid lock-ordering deadlock
            if _deferred_reboot:
                try:
                    await self._resolve_deferred_reboot(str(_deferred_reboot))
                except Exception as _reboot_exc:  # noqa: BLE001
                    _record_mlx_degradation(
                        _reboot_exc,
                        action="failed to resolve deferred reboot during generation cleanup",
                        severity="error",
                    )
            # A staged promotion survives whatever verdict the request reached.
            # It recycles the worker itself, so it runs after (and independent
            # of) the recovery decision rather than competing with it.
            if _pending_promotion:
                try:
                    await self._activate_promoted_artifact(str(_pending_promotion))
                except Exception as _promotion_exc:  # noqa: BLE001
                    _record_mlx_degradation(
                        _promotion_exc,
                        action="left the staged artifact pending after activation failed",
                        severity="error",
                    )

    async def _generate_inner(
        self,
        prompt: str,
        _retry: bool = True,
        request_is_background: bool = False,
        foreground_request: bool = False,
        owner_label: str = "",
        progress_owned_completion: bool = False,
        **kwargs,
    ) -> str | None:
        """Core generation logic, extracted for retry support."""
        # This reason belongs to exactly one generation attempt. Clear any
        # unconsumed prior outcome before a new request can publish its own.
        self._deliberate_no_text_reason = None
        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Skipping queued background generation for %s during foreground ownership.",
                os.path.basename(self.model_path),
            )
            return None
        if request_is_background:
            background_origin = owner_label or str(
                kwargs.get("origin", "") or os.path.basename(self.model_path)
            )
            background_deferral = _background_deferral_active(background_origin)
            if background_deferral:
                if not self._can_run_resident_background_health_probe(
                    background_deferral,
                    health_probe=bool(kwargs.get("health_probe")),
                ):
                    logger.info(
                        "⏸️ [MLX] Background generation for %s stopped before worker spawn (%s).",
                        os.path.basename(self.model_path),
                        background_deferral,
                    )
                    return None
                logger.info(
                    "🩺 [MLX] Running bounded readiness probe on resident primary worker "
                    "despite background headroom reservation."
                )

        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            timeout_s = _coerce_timeout_seconds(kwargs.pop("timeout", None))
            # CP126 24aaa654: a request budget inferred from path substrings
            # gave renamed or aliased resident checkpoints a 60s deadline
            # meant for small models, and handed unrelated paths containing
            # "32b" an inflated one. Measured artifact evidence decides.
            is_heavy = _model_is_heavy_lane(self.model_path)
            deadline = get_deadline(
                timeout_s if timeout_s is not None else (240.0 if is_heavy else 60.0)
            )
            kwargs["deadline"] = deadline
        init_timeout, soft_init_timeout = self._request_scoped_init_timeout(
            deadline,
            foreground_request=foreground_request,
        )

        try:
            alive = await self._ensure_worker_alive(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                init_timeout=init_timeout,
                soft_timeout=soft_init_timeout,
            )
        except TimeoutError:
            self._record_degraded_event(
                "init_deadline_reached",
                detail=f"{os.path.basename(self.model_path)}:{init_timeout:.1f}s",
                severity="warning",
                foreground_request=foreground_request,
            )
            if foreground_request and self._is_primary_or_deep_lane():
                self._set_lane_state("recovering", "init_budget_timeout")
            return None

        if not alive:
            return None

        # ── Latent-space bridge: substrate state directly modulates
        # sampling parameters at the inference call (NOT via prompt
        # injection). Caller-supplied kwargs win; the bridge fills any
        # field the caller didn't pin. This is the structural alternative
        # to "tell the LLM how to feel" — sampling itself changes.
        pinned_generation_contract = bool(
            kwargs.get("strict_answer_contract", False)
            or kwargs.get("strict_value_contract", False)
            or kwargs.get("proof_evaluation_contract", False)
            or kwargs.get("operator_evidence_contract", False)
            or kwargs.get("web_interlocutor_contract", False)
            or kwargs.get("benchmark_request", False)
            or kwargs.get("health_probe", False)
            or kwargs.get("schema") is not None
        )
        if pinned_generation_contract:
            _bridge = None
        else:
            try:
                from core.brain.latent_bridge import compute_inference_params

                _bridge = compute_inference_params(
                    base_max_tokens=int(
                        kwargs.get("max_tokens", self.max_tokens) or self.max_tokens
                    ),
                    base_temperature=float(
                        kwargs.get("temperature", kwargs.get("temp", self.temp)) or self.temp
                    ),
                    foreground=bool(foreground_request),
                )
            except (ImportError, AttributeError, RuntimeError) as _bridge_exc:
                _bridge = None
                _record_mlx_degradation(
                    _bridge_exc,
                    action="continued generation with caller/default sampling parameters",
                )
                logger.debug("latent_bridge unavailable: %s", _bridge_exc)

        def _bridge_get(field: str, fallback: Any) -> Any:
            if _bridge is None:
                return fallback
            return getattr(_bridge, field, fallback)

        requested_output_contract = kwargs.get("requested_output_contract")
        if not isinstance(requested_output_contract, dict):
            requested_output_contract = {}
        hard_output_token_ceiling = kwargs.get("hard_output_token_ceiling")
        adaptive_suggested_max_tokens = _bridge_get("max_tokens", self.max_tokens)
        contract_generation_floor = _requested_output_contract_generation_floor(
            requested_output_contract
        )
        generation_max_tokens = _bounded_generation_max_tokens(
            kwargs.get("max_tokens", self.max_tokens),
            adaptive_suggested_max_tokens,
            hard_output_token_ceiling,
            self.max_tokens,
            requested_output_contract,
            user_surface_completion_floor=kwargs.get(
                "user_surface_completion_floor"
            ),
            preserve_user_surface_completion_floor=bool(
                kwargs.get("clean_user_surface_contract", False)
            ),
            preserve_admitted_capacity=bool(
                progress_owned_completion and foreground_request
            ),
            # A call that carries a program is sized by what it has to say,
            # not by how depleted she is.
            tool_call_floor=(
                kwargs.get("max_tokens")
                if _tools_can_carry_a_document(_offered_for_budgeting(kwargs))
                else None
            ),
        )

        prompt = _prompt_within_prefill_ceiling(prompt, model_path=self.model_path)

        req_id = uuid.uuid4().hex
        self._job_seq_counter += 1
        req = _build_the_generation_request(
            _bridge_get=_bridge_get,
            adaptive_suggested_max_tokens=adaptive_suggested_max_tokens,
            contract_generation_floor=contract_generation_floor,
            generation_max_tokens=generation_max_tokens,
            hard_output_token_ceiling=hard_output_token_ceiling,
            kwargs=kwargs,
            prompt=prompt,
            req_id=req_id,
            requested_output_contract=requested_output_contract,
            self=self,
        )
        # Whether somebody is waiting for this one. The worker lets a
        # foreground generation that is still producing tokens run past its
        # deadline rather than cancelling a working answer; background work
        # still yields, so a dream cycle cannot sit on the one GPU while a
        # person waits.
        req["foreground_request"] = bool(foreground_request)
        req["progress_owned_completion"] = bool(progress_owned_completion and foreground_request)

        # CP126 cac5c1a3: normalise the sampling parameters BEFORE the
        # mandatory stop sequences are appended, so the caller's list is
        # bounded and typed and the defaults below are never displaced by it.
        sampling_faults = _normalize_generation_params(req)
        if sampling_faults:
            req["sampling_contract_faults"] = sampling_faults
            _record_mlx_degradation(
                ValueError(f"generation parameters out of contract: {sampling_faults}"),
                action="substituted defaults for out-of-contract sampling parameters",
                severity="warning",
            )

        # [STABILITY v57/v61] Add default stop sequences to prevent prompt bleed.
        # Keep human-readable role labels line-boundary anchored. Bare labels
        # like ``Assistant:`` can occur in normal prose and caused valid live
        # answers to be clipped before the response reliability gate saw them.
        default_stops = [
            "<|im_end|>",
            "<|im_start|>",
            "\nuser:",
            "\nassistant:",
            "\nUser:",
            "\nAssistant:",
        ]
        for stop in default_stops:
            if stop not in req["stop_sequences"]:
                req["stop_sequences"].append(stop)
        # z_Aura rides along the same way. The worker cannot reach the
        # substrate, the goal system, or anything else in this process, so
        # the state travels as declared floats on the job. A worker with no
        # trained head ignores the field; the receipt says which happened.
        attach_endogenous_state(
            req,
            model_path=self.model_path,
            override=kwargs.get("endogenous_state"),
        )

        # Activation-steering offsets ride along when present; the worker
        # consumes them if its build supports residual-stream injection,
        # otherwise it ignores the field with no harm.
        if _bridge is not None and getattr(_bridge, "layer_offsets", None):
            req["layer_offsets"] = _bridge.layer_offsets
        if _bridge is not None and getattr(_bridge, "extra_stop_sequences", None):
            # EXTEND the request's stop list — rebuilding it from the caller
            # kwargs erased every mandatory anti-bleed default appended above.
            for stop in _bridge.extra_stop_sequences:
                if stop not in req["stop_sequences"]:
                    req["stop_sequences"].append(stop)

        if self._active_generations <= 0 and not await self._set_durable_lane_preemptible(False):
            logger.info(
                "MLX generation yielded because durable lane ownership is being evicted: %s",
                os.path.basename(self.model_path),
            )
            return None

        foreground_watchdog = None
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        self._current_gen_future = fut
        self._active_generations += 1
        self._active_generation_started_at = time.time()
        first_token_hard_ceiling = self._deadline_bound_first_token_hard_ceiling(
            deadline.remaining,
            foreground_request=foreground_request,
        )
        self._mark_generation_started(
            req_id,
            prompt_chars=len(prompt or ""),
            requested_max_tokens=req.get("max_tokens", self.max_tokens),
            first_token_hard_ceiling_s=first_token_hard_ceiling,
            request_seq=int(req.get("seq", 0)),
        )
        foreground_watchdog = self._start_foreground_first_token_watchdog(
            req_id,
            foreground_request=foreground_request,
            hard_ceiling_s=first_token_hard_ceiling,
        )
        # Ship the caller's production deadline to the worker so its decode
        # loop can stop cooperatively instead of burning GPU past the point
        # anyone is waiting (the worker previously had NO request deadline —
        # only the 360s hard watchdog).
        try:
            _remaining_s = float(deadline.remaining or 0.0)
            if _remaining_s > 0.0:
                # Reserve a DELIVERY MARGIN. A cooperative stop is only worth
                # anything if the partial answer can cross IPC and be consumed
                # before the caller's own deadline. Handing the worker the
                # caller's full remaining budget made the two expire together,
                # so a decode that stopped politely at token 149 was abandoned
                # by a gate that had already given up — measured live as
                # "Cortex consumed 77.5s without usable text" immediately
                # followed by "Abort arrived after the generation finished;
                # nothing to abort, leaving the worker up". The tokens existed;
                # nobody was left waiting for them.
                #
                # The floor keeps the margin from eating a short budget whole:
                # a worker that gets less than half the request is worse than
                # one that occasionally misses the handoff.
                _delivery_margin_s = max(1.5, min(6.0, _remaining_s * 0.08))
                _worker_budget_s = max(
                    _remaining_s * 0.5, _remaining_s - _delivery_margin_s
                )
                req["deadline_unix"] = time.time() + _worker_budget_s
        except (AttributeError, TypeError, ValueError):
            logger.debug("Request deadline unavailable; worker decodes unbounded.")
        # CP126 a838a49b: this used to read
        # `max(0.5, min(2.0, deadline.remaining or 2.0))`, which turned an
        # ALREADY-EXPIRED budget (remaining == 0.0, falsy) into a 2-second
        # wait and floored every sub-half-second remainder up to 0.5s — so a
        # request could block past its own hard deadline and seed exactly the
        # ownership/event-loop cascades this path exists to prevent. Never
        # enqueue past the deadline; refuse instead.
        _enqueue_remaining = 0.0
        try:
            _enqueue_remaining = max(0.0, float(deadline.remaining or 0.0))
        except (AttributeError, TypeError, ValueError):
            _enqueue_remaining = 2.0
        if _enqueue_remaining <= 0.0:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )
            _record_mlx_degradation(
                TimeoutError("request_deadline_expired_before_enqueue"),
                action="refused to queue work whose deadline had already expired",
                severity="warning",
            )
            return None
        enqueue_timeout = min(2.0, _enqueue_remaining)
        try:
            if self._req_q is None:
                raise BrokenPipeError("MLX request queue is closed")
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(req, principal="mlx_client.generate"),
                True,
                enqueue_timeout,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )
            if _retry and ("Broken pipe" in str(exc) or isinstance(exc, BrokenPipeError)):
                logger.warning(
                    "🔄 [MLX] Broken pipe on %s — deferring reboot (lock held)",
                    os.path.basename(self.model_path),
                )
                self._deferred_reboot_reason = "broken_pipe_retry"
                return None
            logger.error("🛑 [MLX] Request queue blocked or failed: %s", exc)
            self._deferred_reboot_reason = f"request_queue_failed:{exc}"
            return None

        try:
            res = await self._wait_for_generation_result(
                req_id,
                fut,
                deadline,
                foreground_request=foreground_request,
                progress_owned_completion=progress_owned_completion,
            )
            if not res:
                return None
            if res.get("status") == "ok":
                self._record_surface_control_receipt_from_response(res)
                self._record_throughput_sample(
                    res,
                    prompt=prompt,
                    foreground_request=foreground_request,
                )
                self._record_interoception_from_response(
                    res,
                    foreground_request=foreground_request,
                    owner_label=owner_label,
                )
                raw_text = res.get("text", "")
                if not isinstance(raw_text, str):
                    # A malformed cross-process payload must fail through the
                    # typed empty-response path, not raise AttributeError.
                    _record_mlx_degradation(
                        TypeError(f"worker text payload was {type(raw_text).__name__}"),
                        action="treated non-string worker text as empty response",
                    )
                    raw_text = ""
                text = raw_text.strip()
                self._mark_progress()
                generation_stop_reason = str(
                    res.get("generation_stop_reason") or ""
                ).strip().lower()
                cooperative_stop = bool(
                    res.get("soft_cancelled")
                    or generation_stop_reason in {
                        "soft_cancelled",
                        "deadline_exceeded",
                    }
                )
                quality_rejection_reasons = _surface_quality_rejection_reasons(
                    self.get_last_surface_control_receipt()
                )
                # A tool call is not prose and must not be judged as prose.
                #
                # The surface quality gate reads a draft as an answer somebody
                # is about to be shown: prompt artifacts, boilerplate, leaked
                # internal text. A generation that was offered tools and
                # answered with a tool call is none of those. It is the model
                # saying what to run next, and the loop that offered the tools
                # is the thing waiting to read it.
                #
                # LIVE 2026-08-29: asked to read a library's docs and use it,
                # turn four of the tool loop emitted a complete, correct
                # code_repl call — the right library, the right arguments, the
                # invoice posted the right way round. It was rejected as
                # "prompt_artifact" because it is angle brackets rather than
                # sentences, the generation returned nothing, and the turn
                # ended on "I couldn't get to an answer I'd stand behind"
                # after four successful tool calls.
                #
                # Only a complete, parseable call earns this, and only when
                # the request actually offered tools. Everything else the gate
                # catches, it still catches.
                if quality_rejection_reasons and text and kwargs.get("tools"):
                    if _text_is_a_complete_tool_call(text):
                        logger.info(
                            "🔧 [MLX] the draft the quality gate refused (%s) is a "
                            "complete tool call, and tools were offered — handing "
                            "it to the loop that asked for it.",
                            ",".join(quality_rejection_reasons),
                        )
                        quality_rejection_reasons = ()
                if quality_rejection_reasons and (not text or cooperative_stop):
                    # The worker decoded a draft but could not make it directly
                    # servable before this request ended. Keep it bound to the
                    # turn as suppressed recovery evidence; an empty result
                    # without that custody is indistinguishable from a model
                    # failure and needlessly opens the Cortex circuit.
                    self._preserve_lane_after_surface_quality_rejection()
                    self._record_suppressed_draft(
                        _rejected_surface_draft(
                            self.get_last_surface_control_receipt()
                        ),
                        _surface_quality_rejected_draft_reasons(
                            self.get_last_surface_control_receipt()
                        ),
                    )
                    # A gate that destroys an answer has to say what it
                    # destroyed. Without the text, a rejection is a label:
                    # nothing downstream, and nobody reading the log later, can
                    # tell a correct answer thrown away from a bad one caught.
                    #
                    # LIVE, 2026-08-27: a worked-through arithmetic derivation
                    # was rejected for internal_task_prompt_leak after two
                    # minutes of generation, and none of the three leak
                    # detectors fired on the text that had been served for the
                    # same question one turn earlier. The draft was preserved
                    # in memory as recovery evidence and never written down, so
                    # the reason it was rejected could not be reproduced.
                    _rejected_text = str(
                        _rejected_surface_draft(
                            self.get_last_surface_control_receipt()
                        )
                        or ""
                    )
                    logger.warning(
                        "🛡️ [MLX] Worker rejected the visible draft for semantic "
                        "quality (%s); preserving the resident lane. "
                        "draft_chars=%d head=%r tail=%r",
                        ",".join(quality_rejection_reasons),
                        len(_rejected_text),
                        _rejected_text[:400],
                        _rejected_text[-200:] if len(_rejected_text) > 400 else "",
                    )
                    return None
                if not text and not cooperative_stop:
                    # Empty warmup can prove process/shader liveness, but it
                    # cannot prove conversation readiness. Keep the lane out of
                    # "ready" until a visible generation succeeds.
                    is_warmup = getattr(self, "_warmup_in_flight", False)
                    if is_warmup:
                        logger.info(
                            "MLX warmup produced empty text — shader precompile may be complete, "
                            "but conversation readiness still requires a visible response."
                        )
                        self._set_lane_state("warming", "warmup_precompile_no_text")
                        return ""
                    empty_count = getattr(self, "_consecutive_empty", 0) + 1
                    self._consecutive_empty = empty_count
                    # Inline one-shot retry for user-facing requests.  The
                    # worker self-clears its prompt cache after a zero-token
                    # generation, so an immediate second attempt on the same
                    # lock usually succeeds — and that beats letting the
                    # InferenceGate 30-second cascade fire.  Gate on _retry so
                    # we never loop, and only trigger for foreground to avoid
                    # burning background budget on speculative retries.
                    if (
                        _retry
                        and foreground_request
                        and empty_count < 3
                        and (deadline.remaining is None or deadline.remaining > 5.0)
                    ):
                        # This is an active recovery transition, not yet a
                        # user-visible failure. Keep the attempt observable
                        # without forwarding a synthetic RuntimeError into
                        # ErrorIntelligence before the retry has a verdict.
                        self._record_degraded_event(
                            "empty_generation_retry",
                            detail=(
                                f"{os.path.basename(self.model_path)}:"
                                f"attempt={empty_count}:cache_reset_retry"
                            ),
                            severity="info",
                            foreground_request=False,
                            classification="non_critical_fallback",
                        )
                        # CP126 3d2e68fe. The retry re-enters generation while
                        # still holding the request lock and foreground
                        # ownership, so it never passes an admission gate
                        # again — and the world moved between the two
                        # attempts. Shutdown may have been requested; memory
                        # pressure may now refuse a heavy decode; the worker
                        # that produced the empty answer may have died. A
                        # fresh request would be refused in any of those
                        # states, and this one carries the same cost.
                        refusal = self._inline_retry_refusal()
                        if refusal:
                            logger.info(
                                "🔁 [MLX] Skipping inline retry for %s: %s.",
                                os.path.basename(self.model_path),
                                refusal,
                            )
                            self._record_degraded_event(
                                "empty_generation_retry_refused",
                                detail=f"{os.path.basename(self.model_path)}:{refusal}",
                                severity="info",
                                foreground_request=False,
                                classification="non_critical_fallback",
                            )
                            return None
                        logger.info(
                            "🔁 [MLX] Empty foreground generation — "
                            "inline retry after worker cache reset (%d/2).",
                            empty_count,
                        )
                        inline_kwargs = dict(kwargs)
                        inline_kwargs["deadline"] = deadline
                        return await self._generate_inner(
                            prompt,
                            _retry=False,  # prevent recursion
                            progress_owned_completion=progress_owned_completion,
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            owner_label=owner_label,
                            **inline_kwargs,
                        )
                    if foreground_request:
                        self._record_degraded_event(
                            "empty_generation_exhausted",
                            detail=(
                                f"{os.path.basename(self.model_path)}:"
                                f"attempt={empty_count}:no_visible_text"
                            ),
                            severity="error",
                            foreground_request=True,
                        )
                        self._deferred_reboot_reason = "recoverable_empty_generation"
                    else:
                        self._record_degraded_event(
                            "empty_generation",
                            detail=(
                                f"{os.path.basename(self.model_path)}:"
                                f"attempt={empty_count}:background"
                            ),
                            severity="info",
                            foreground_request=False,
                        )
                    if foreground_request and self._is_primary_or_deep_lane() and empty_count >= 3:
                        self._set_lane_state("recovering", "repeated_empty_generation")
                    return None
                self._consecutive_empty = 0
                if cooperative_stop:
                    # Deliberate cooperative preemption: return the partial text
                    # without empty-generation telemetry, inline retries, or a
                    # user-facing completion mark — the health machinery must
                    # not treat a requested cancel as a generation failure.
                    logger.info(
                        "✋ [MLX] Generation for %s ended by %s after partial output (%d chars); resident lane preserved.",
                        os.path.basename(self.model_path),
                        generation_stop_reason or "soft_cancelled",
                        len(text),
                    )
                    self._set_lane_state("ready")
                    return text or None
                is_health_probe = bool(kwargs.get("health_probe", False))
                self._set_lane_state("ready")
                self._mark_generation_completed(
                    user_facing=bool(foreground_request and not is_health_probe)
                )
                _notify_closed_loop_output(text)
                return text
            reason = str(res.get("message") or res.get("status") or "generation_failed")
            self._record_degraded_event(
                "generation_failed",
                detail=f"{os.path.basename(self.model_path)}:{reason}",
                severity="error",
                foreground_request=foreground_request,
            )
            return None
        except asyncio.CancelledError:
            origin_label = str(kwargs.get("origin", "") or "")
            purpose_label = str(kwargs.get("purpose", "") or "")
            expected_cancel_reason = self._consume_expected_generation_cancellation(req_id)
            # CP126 9edfb10c. This used to be the labels alone, so ANY request
            # could suppress a cancellation degradation by calling itself
            # "baseline" — a self-signed excuse for the exact signal that says
            # the lane is misbehaving. A benchmark cancellation is only
            # expected when the PROCESS is actually a benchmark run, which is
            # a property of how the runtime was launched and not of a string
            # the request supplied about itself.
            benchmark_baseline_cancel = _benchmark_run_context_active() and (
                origin_label.strip().lower() == "baseline"
                or purpose_label.strip().lower().endswith("_baseline")
            )
            shutdown_cancel = _runtime_shutdown_requested()
            if expected_cancel_reason:
                logger.info(
                    "🧹 [MLX] Generation cancelled for %s during expected reboot (%s).",
                    os.path.basename(self.model_path),
                    expected_cancel_reason,
                )
            elif benchmark_baseline_cancel:
                logger.info(
                    "🧪 [MLX] Baseline generation cancelled for %s by benchmark timeout.",
                    os.path.basename(self.model_path),
                )
            elif shutdown_cancel:
                logger.info(
                    "🛑 [MLX] Generation cancelled for %s during runtime shutdown.",
                    os.path.basename(self.model_path),
                )
            else:
                logger.warning(
                    "🛑 [MLX] Generation cancelled for %s. Preserving worker unless it is unhealthy.",
                    os.path.basename(self.model_path),
                )
            self._pending_generations.pop(req_id, None)
            if (
                not expected_cancel_reason
                and not benchmark_baseline_cancel
                and not shutdown_cancel
                and (
                    foreground_request
                    or (
                        self._is_primary_or_deep_lane()
                        and self._lane_state not in {"cold", "warming", "recovering"}
                    )
                )
            ):
                self._record_degraded_event(
                    "generation_cancelled",
                    detail=os.path.basename(self.model_path),
                    severity="warning",
                    foreground_request=foreground_request,
                )
            if not expected_cancel_reason and not shutdown_cancel and self._worker_unhealthy():
                self._deferred_reboot_reason = "cancelled_unhealthy"
            raise
        except TimeoutError:
            logger.error(
                "🛑 [MLX] Generation deadline reached for %s.", os.path.basename(self.model_path)
            )
            self._pending_generations.pop(req_id, None)
            _cancel_shared_future(fut)
            self._record_degraded_event(
                "generation_deadline_reached",
                detail=os.path.basename(self.model_path),
                severity="warning",
                foreground_request=foreground_request,
            )
            if self._worker_unhealthy(stale_after=self._stale_after(during_generation=True)):
                self._deferred_reboot_reason = "generation_timeout_unhealthy"
            else:
                self._mark_healthy_generation_deadline(
                    foreground_request=foreground_request,
                )
            if foreground_request and self._deliberate_no_text_reason:
                logger.warning(
                    "⏳ [MLX] Deadline reached while worker still looks healthy; "
                    "soft-cancelling the abandoned generation and preserving the warm lane."
                )
            elif self._deliberate_no_text_reason:
                logger.warning(
                    "⏳ [MLX] Deadline reached but worker still looks healthy; leaving lane warm."
                )
            return None
        finally:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )

    async def think_and_act(
        self,
        objective: str,
        system_prompt: str,
        tools: dict[str, Any] | None = None,
        max_turns: int = 5,
        context: dict | None = None,
        evidence: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """ReAct agentic loop: think → parse tool call → execute → repeat.

        Uses the model's native chat + tool template when available and falls
        back to a JSON-only tool-call contract otherwise. Results are fed back
        into the conversation history until the model produces a plain-text
        final answer or max_turns is exhausted.

        Returns:
            {"content": str, "turns": int, "tool_calls": List[Dict]}
        """
        # Bound the turn budget before it is used OR reported. An unvalidated
        # max_turns let a caller pass 0 or a negative value, execute no turns
        # at all, and still receive `"turns": max_turns` — a report of work
        # that provably did not happen. A huge value let one objective occupy
        # the lane indefinitely.
        try:
            max_turns = int(max_turns)
        except (TypeError, ValueError):
            max_turns = 5
        max_turns = max(1, min(_AGENT_MAX_TURNS_CEILING, max_turns))

        template_tools = self._normalize_tool_definitions_for_template(tools)
        tool_block = ""
        # Both protocols are built. The native one is preferred, and the JSON
        # contract is the fallback for a checkpoint that will not emit native
        # calls — see the retry below.
        if tools:
            tool_lines = []
            for name, defn in list(tools.items())[:20]:  # cap to avoid bloat
                desc = defn.get("description", "")
                params = defn.get("parameters", {}).get("properties", {})
                param_str = ", ".join(f'"{k}"' for k in params) if params else "none"
                tool_lines.append(f"  • {name}: {desc}  [params: {param_str}]")
            tool_block = (
                "\n\n## TOOLS AVAILABLE\n"
                + "\n".join(tool_lines)
                + "\n\nIf you need a tool and the model supports native tool calling, emit the native tool-call format only.\n"
                + "Otherwise output EXACTLY this on its own line (nothing else):\n"
                + '```json\n{"tool": "tool_name", "args": {"param": "value"}}\n```\n'
                + "When you have your final answer, respond normally — no JSON block."
            )

        # Native first: the schema goes in the template, not in prose.
        native_tools: list[dict[str, Any]] | None = template_tools or None
        augmented_system = system_prompt if native_tools else system_prompt + tool_block
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": augmented_system},
        ]
        # What the turn has already read.
        #
        # LIVE, 2026-08-20. The evidence step fetched the document the person
        # named, and this loop received the objective alone — so the model
        # fetched it again, got a 400 on a URL it rebuilt from memory, and
        # told the person about the failure before giving the answer. It was
        # not being redundant on purpose; nothing had told it the read had
        # happened.
        messages.extend(_tool_loop_evidence_messages(evidence))
        messages.append({"role": "user", "content": objective})
        tool_calls_made: list[dict[str, Any]] = []
        last_response_text = ""
        announced_without_acting = False

        logger.info(
            "🔧 Tool loop payload: system=%d chars, objective=%d chars, "
            "evidence=%d blocks, tools=%d",
            len(augmented_system),
            len(str(objective or "")),
            len(messages) - 2,
            len(template_tools or []),
        )
        for turn in range(max_turns):
            raw = await self.generate_text_async(
                "",
                messages=messages,
                tools=native_tools,
                # What this turn may call, for budgeting, whichever protocol it
                # speaks. `tools` above is the native template's channel and is
                # None on the JSON contract; the budget must not depend on
                # which of the two is in use.
                tool_budget_definitions=template_tools or None,
                # A call is not a reply, and inheriting the reply's budget cut
                # one in half.
                #
                # LIVE, 2026-08-20. The desktop lane planned 970 tokens for
                # its answer — a fair size for a conversational reply — and
                # the tool loop took the same number for a call whose argument
                # was an HTML page. It stopped inside the string, an
                # incomplete object is not a call, and the loop reported "none
                # called" for the second time in one turn.
                #
                # Where the offered tools take a document, the call asks for
                # what this client is configured to allow rather than what
                # this turn's prose was budgeted at.
                max_tokens=_tool_call_budget(
                    kwargs.get("max_tokens", self.max_tokens),
                    self.max_tokens,
                    native_tools or tools,
                ),
                # A tool call is structured output, and affective steering
                # destroys structured output.
                #
                # LIVE, 2026-08-19: with five tools offered and the engine
                # alpha at 3.8, this generation returned ONE token and no text
                # survived — "Generation produced 1 token(s) but no text
                # survived to the caller". Every tool-using turn therefore
                # looked like a model that declined to call anything, when
                # nothing had been generated at all.
                #
                # The user surface already decodes at alpha 0.0 for the same
                # reason, and the code model is loaded unsteered outright. This
                # is that rule applied to the third kind of structured
                # generation the runtime performs.
                # NOT under the clean-user-surface contract. It honours the
                # decode controls AND turns on user-surface quality
                # validation, so setting it here pointed the surface quality
                # gate at a tool call — which is not a user-facing answer and
                # never passes it. The worker generated 100 tokens and cleared
                # them: "no text survived to the caller". A tool call must
                # reach the parser exactly as produced; whether the eventual
                # REPLY is a good answer is judged later, on the reply.
            )
            if not raw or not raw.strip():
                # An empty generation in tool mode looks identical from the
                # outside to a model that declined to call anything: both end
                # the loop with no tool_calls. They are different faults —
                # nothing generated versus something generated that was not a
                # call — and only one of them is about tool calling at all.
                if tools:
                    logger.info(
                        "🔧 Tools offered (%s) and the generation came back empty "
                        "on turn %d.",
                        ",".join(sorted(tools)),
                        turn + 1,
                    )
                # A checkpoint that ends its turn on a native tool prompt has
                # not refused the work — it has refused the PROTOCOL. Live
                # 2026-08-19 this model emitted <|im_end|> as its first token
                # against a correct, small, well-formed ChatML tool prompt,
                # every time, so nothing could ever be executed from chat.
                #
                # The JSON contract asks for the same call as an ordinary
                # answer, which needs no native tool-calling behaviour at all.
                # Both protocols were already built here; only one was reachable.
                if native_tools and tools:
                    logger.info(
                        "🔧 Native tool protocol produced nothing; retrying on the "
                        "JSON tool contract."
                    )
                    native_tools = None
                    messages[0]["content"] = system_prompt + tool_block
                    continue
                break

            response_text = raw.strip()
            last_response_text = response_text

            tool_call = (
                self._extract_tool_call_payload(
                    response_text,
                    allowed_tools=set(tools.keys()),
                    tool_definitions=tools,
                )
                if tools
                else None
            )
            if not tool_call:
                # What it said INSTEAD is the only way to tell a malformed
                # call from a model that simply answered. Without this the two
                # are indistinguishable from the outside, and they need
                # opposite fixes: a parser change versus a decoding one.
                if tools and not tool_calls_made:
                    logger.info(
                        "🔧 Tools offered (%s) and none called; model produced: %s",
                        ",".join(sorted(tools)),
                        " ".join(response_text.split())[:200],
                    )
                # An announced action with no action is not an answer.
                #
                # LIVE, 2026-08-20. Given a seating problem and a Python
                # sandbox, the model produced exactly "Let's break down the
                # problem step by step and use code to help us figure out the
                # seating arrangement." and ended its turn. The loop read that
                # as a final answer, so the turn was decided by a sentence
                # describing work nobody did — and the reply that followed got
                # the neighbours wrong.
                #
                # Once, and only while turns remain: the announcement stays in
                # the history, so the next turn continues from a model that has
                # already said what it was about to do.
                if (
                    tools
                    and not tool_calls_made
                    and not announced_without_acting
                    and turn + 1 < max_turns
                    and _announces_an_action_it_did_not_take(response_text)
                ):
                    announced_without_acting = True
                    logger.info(
                        "🔧 The model announced a tool it did not call; continuing the loop."
                    )
                    messages.append({"role": "assistant", "content": response_text})
                    continue
                return {
                    "content": response_text,
                    "turns": turn + 1,
                    "tool_calls": tool_calls_made,
                }

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            # CP126 0da5db2e: bind the parsed arguments to the tool's own
            # advertised JSON schema before anything executes. A call whose
            # arguments do not satisfy the schema is a malformed call, not an
            # effect to attempt.
            schema_error = _tool_arguments_schema_error(tools.get(tool_name), tool_args)
            if schema_error:
                _record_mlx_degradation(
                    ValueError(f"tool_arguments_invalid:{tool_name}:{schema_error}"),
                    action="refused a tool call whose arguments failed its advertised schema",
                    severity="warning",
                )
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That call to '{tool_name}' was rejected: {schema_error}. "
                            "Correct the arguments or answer directly."
                        ),
                    }
                )
                continue
            # CP126 abd93abf: every call gets a stable id so the assistant
            # turn and its tool result are unambiguously paired in history.
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"

            # ── Execute the tool via FunctionCallingAdapter ───────────
            raw_result: Any = {
                "ok": False,
                "status": "error",
                "error": f"Tool '{tool_name}' is unavailable",
                "engine": "capability_engine",
            }
            try:
                from core.container import ServiceContainer

                adapter_or_cap = ServiceContainer.get("capability_engine", default=None)
                refusal = _refuse_action_beyond_authority(
                    adapter_or_cap, tool_name, tool_args, context
                )
                if refusal:
                    raw_result = refusal
                elif adapter_or_cap:
                    raw_result = await adapter_or_cap.execute(
                        tool_name,
                        tool_args,
                        _agent_execution_context(
                            context,
                            objective=objective,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            model_path=self.model_path,
                        ),
                    )
            except (ImportError, AttributeError, RuntimeError) as exc:
                raw_result = {
                    "ok": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "engine": "capability_engine",
                }
                _record_mlx_degradation(
                    exc,
                    action="returned structured tool error to the model loop",
                    severity="error",
                )
                logger.warning("[think_and_act] Tool '%s' failed: %s", tool_name, exc)
            tool_result = _serialize_tool_result_for_model(
                tool_name,
                raw_result,
            )

            tool_calls_made.append(
                {
                    "id": tool_call_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                }
            )
            # The turn owns this the moment it exists.
            #
            # The caller recorded receipts for the whole loop after the loop
            # returned, which means a loop that did not return recorded
            # nothing. LIVE 2026-08-29: six tool turns, a code_repl that ran
            # and produced the answer on turn six, and then the loop ran out
            # of time — every result dropped, the salvage that reports what
            # the tools found had nothing to report ("custody=present
            # admits=True" with no receipts), and the person got "I couldn't
            # get to an answer I'd stand behind".
            #
            # Writing it here costs nothing and makes the record independent
            # of how the loop ends. The caller still writes the whole batch on
            # the path where it finishes, and custody now treats a receipt
            # naming the same call and the same result as the one call it is.
            _record_tool_receipt_for_this_turn(
                tool_name, tool_args, raw_result, tool_result
            )
            # Report what actually happened. This line said "ok" for every
            # outcome — a missing capability engine, a caught exception, and a
            # governance DENIAL encoded as a normal result all logged
            # identically to a success. Telemetry that cannot tell refusal from
            # execution is worse than absent: it reads as proof the tool ran.
            outcome = _tool_turn_outcome(raw_result)
            # Name the arguments too. A call with none and a call whose
            # arguments went missing on the way to dispatch logged identically,
            # and telling them apart took three live turns.
            logger.info(
                "[think_and_act] turn=%d tool=%s args=%s %s",
                turn + 1,
                tool_name,
                sorted(str(key) for key in (tool_args or {}))[:6] or "none",
                outcome,
            )

            # ── Feed result back into history ─────────────────────────
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                # Canonical state keeps the typed object the
                                # executor consumed. chat_format adapts this
                                # at the active tokenizer's wire boundary.
                                "arguments": dict(tool_args),
                            },
                        }
                    ],
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )

        # Exhausted turns — return last non-empty response
        return {
            "content": last_response_text or "I ran out of reasoning steps.",
            "turns": max_turns,
            "tool_calls": tool_calls_made,
        }

    async def _run_warmup_precompile(
        self,
        *,
        request_is_background: bool,
        foreground_request: bool,
        owner_name: str,
        warmup_timeout: float,
    ) -> None:
        # CP126 b4fcd100. The second attempt used to be given
        # `warmup_timeout + 10`, and the readiness probe that follows took its
        # own independent timeout on top. So the documented warmup budget was
        # not a bound on warmup at all — it was a per-attempt allowance that
        # grew when things went badly, which is exactly when a caller waiting
        # on this needs the bound to hold.
        #
        # One campaign deadline, shared: whatever the retry and the probe
        # spend comes out of the same budget, and an attempt with nothing left
        # does not start.
        campaign_deadline = time.monotonic() + max(1.0, float(warmup_timeout))
        last_exc: Exception | None = None
        for attempt in range(2):
            remaining = campaign_deadline - time.monotonic()
            if remaining <= 0.0:
                if last_exc is not None:
                    raise last_exc from None
                raise TimeoutError(
                    f"warmup_budget_exhausted:{warmup_timeout:.1f}s"
                )
            try:
                warmup_text = await asyncio.wait_for(
                    self._generate_inner(
                        "Hello",
                        _retry=True,
                        request_is_background=request_is_background,
                        foreground_request=foreground_request,
                        owner_label=owner_name,
                        max_tokens=1,
                        warmup_precompile=True,
                    ),
                    timeout=remaining,
                )
                if warmup_text is None and not self.is_alive():
                    raise RuntimeError("warmup_precompile_worker_dead")
                # CP126 cdd743de + b6439433. A nonempty token from a
                # max_tokens=1 "Hello" proves Metal shaders compiled — it does
                # NOT prove this lane can hold a conversation, and skipping the
                # visible probe on that basis is how a lane that cannot answer
                # got marked ready. The probe now ALWAYS runs, and its answer is
                # actually checked against what was asked for (the prompt says
                # "Reply exactly: ready" but any nonblank text used to pass, so
                # hallucinated, garbled, stale, or prompt-echo output proved
                # readiness).
                # The probe is bounded by what is LEFT of the campaign, never by
                # a floor applied on top of it. A floor there (`max(10.0, ...)`)
                # meant a campaign with 2s left still handed the probe 10s, so
                # the one hard deadline this function exists to enforce was
                # exceeded by up to 10s on exactly the slow boots that made a
                # caller depend on it.
                #
                # "A probe that starts at all deserves a fair chance" is still
                # right — it is a rule about whether to START one, not about how
                # long an already-doomed one may run. So a campaign without
                # _MIN_READINESS_PROBE_BUDGET_S left does not open a probe; it
                # ends, inside the budget it promised.
                probe_budget = campaign_deadline - time.monotonic()
                if probe_budget < _MIN_READINESS_PROBE_BUDGET_S:
                    self._set_lane_state(
                        "recovering", "warmup_budget_exhausted_before_readiness_probe"
                    )
                    raise TimeoutError(
                        f"warmup_budget_exhausted:{warmup_timeout:.1f}s:"
                        f"probe_needs:{_MIN_READINESS_PROBE_BUDGET_S:.1f}s:"
                        f"left:{max(0.0, probe_budget):.1f}s"
                    )
                logger.info(
                    "🔥 [MLX] Verifying conversation readiness for %s with a visible probe.",
                    os.path.basename(self.model_path),
                )
                # The same prover the reconciler uses, so that asking and
                # recording can never drift apart again. Out of the SAME
                # campaign budget as the precompile above (CP126 b4fcd100): a
                # probe that took its own independent timeout is how the
                # documented warmup bound became a suggestion.
                proved = await self.prove_visible_readiness(
                    budget_s=min(probe_budget, _MAX_READINESS_PROBE_S),
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_label=owner_name,
                )
                if proved != "proved":
                    self._set_lane_state("recovering", f"warmup_readiness_{proved}")
                    raise RuntimeError(f"warmup_readiness_{proved}")
                self._last_ready_at = time.time()
                self._warmup_in_flight = False
                _clear_matching_foreground_owner(owner_name)
                logger.info("🔥 [MLX] Warmup complete — Metal shaders compiled.")
                return
            except asyncio.CancelledError as exc:
                last_exc = exc
                if _runtime_shutdown_requested():
                    logger.info(
                        "🛑 [MLX] Warmup pre-compile cancelled for %s during runtime shutdown.",
                        os.path.basename(self.model_path),
                    )
                    raise
                _record_mlx_degradation(
                    exc,
                    action="retried or recycled warmup precompile after cancellation",
                )
                raise
            except (RuntimeError, TimeoutError, AttributeError) as exc:
                last_exc = exc
                if attempt == 0:
                    # The recovery between attempts used to sit OUTSIDE the
                    # campaign: an unawaited-cost gc, a `reboot_worker` with no
                    # bound of its own, and a flat 1s settle. So "one campaign
                    # deadline, shared" described the two generations only, and
                    # a warmup that promised 1s routinely took several — the
                    # composed contract differing from what the local code says,
                    # which is the failure mode this whole function was rewritten
                    # to remove.
                    #
                    # A retry now has to FIT: it is worth starting only if what
                    # remains could still carry a probe, and every second it
                    # spends recovering comes out of the same budget.
                    recovery_budget = campaign_deadline - time.monotonic()
                    if recovery_budget <= _MIN_READINESS_PROBE_BUDGET_S:
                        raise last_exc from None
                    logger.warning(
                        "⚠️ [MLX] Warmup pre-compile failed once for %s: %s. "
                        "Retrying cleanly within the remaining %.1fs...",
                        os.path.basename(self.model_path),
                        exc,
                        recovery_budget,
                    )
                    try:
                        await asyncio.wait_for(
                            self._recover_worker_for_warmup_retry(),
                            timeout=recovery_budget,
                        )
                    except TimeoutError:
                        raise last_exc from None
                    continue
                raise last_exc from None

    async def _recover_worker_for_warmup_retry(self) -> None:
        """Reclaim and reboot the worker between two warmup attempts.

        Split out so the caller can put ONE bound around the whole recovery.

        Never while somebody is being answered. A warmup exists to make the
        lane ready, and tearing the worker down mid-reply to do it defeats the
        thing it is for: LIVE 2026-08-29, a person interrupted a long errand,
        was correctly given the lane, and had her answer cancelled underneath
        her by a retry — "generation cancelled during expected reboot
        (warmup_precompile_retry)" — receiving a stub about being cut short.

        Waiting is the whole remedy. A reply takes seconds and the retry has
        its own budget to spend; if that budget runs out while a person is
        being served, the honest outcome is a warmup that did not get its
        retry, not a person who did not get her answer.
        """
        waited = 0.0
        while _FOREGROUND_OWNER_IS_USER_FACING and waited < _WAIT_OUT_A_REPLY_S:
            await asyncio.sleep(0.25)
            waited += 0.25
        if _FOREGROUND_OWNER_IS_USER_FACING:
            logger.info(
                "[MLX] warmup retry stood down: somebody is still being answered"
            )
            return
        await asyncio.to_thread(gc.collect)
        await self.reboot_worker(reason="warmup_precompile_retry", mark_failed=False)
        # A freshly rebooted worker needs a moment before it can answer; the
        # caller's bound decides whether there is a moment to give it.
        await asyncio.sleep(1.0)

    async def warmup(
        self,
        *,
        foreground_request: bool | None = None,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Boot the worker and prove the visible conversation path is ready.

        SINGLEFLIGHT (CP126 4d8a7d6b). Concurrent callers used to each set
        ``_warmup_in_flight`` and proceed, so two warmups could load/evict the
        same lane at once; the "stale warmup" recovery then measured
        ``_lane_transition_at`` — a timestamp any other lane transition
        refreshes — and force-cleared the shared flag without proving the prior
        warmup had ended. Callers now JOIN the active warmup, and a genuinely
        stuck one is cancelled and awaited before a replacement starts.
        """
        inflight = self._warmup_inflight
        if inflight is not None and not inflight.done():
            age = max(0.0, time.monotonic() - self._warmup_started_at)
            if age <= _WARMUP_STALE_AFTER_S:
                # Join the in-flight warmup. shield() so that a cancelled
                # joiner cannot kill the warmup the other callers need.
                #
                # Across loops, awaiting it directly is not a join — it raises
                # "got Future attached to a different loop", which the handler
                # below then reports as a WARMUP FAILURE to the joiner.
                #
                # LIVE 2026-08-17: that is why the first message after every
                # launch was answered with "the live answer lane could not
                # finish preparing". Boot starts the warmup on the boot loop;
                # the chat turn arrives on the server loop, joins, and is told
                # instantly that warmup failed — so admission reported
                # "worker_not_alive,init_not_complete,lane_warming" no matter
                # how much budget the turn had. Three budget-side fixes moved
                # the failure time and none removed it, because the failure was
                # never about time.
                try:
                    return bool(await _join_inflight_across_loops(inflight))
                except asyncio.CancelledError:
                    raise
                except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                    _record_mlx_degradation(
                        exc,
                        action="reported warmup failure to a joined singleflight caller",
                        severity="warning",
                    )
                    return False
            logger.warning(
                "🔧 [MLX] Warmup for %s stuck for %.0fs — cancelling the prior "
                "warmup task before starting a replacement.",
                os.path.basename(self.model_path),
                age,
            )
            inflight.cancel()
            # PROVE the prior task ended before starting another one.
            try:
                await asyncio.wait_for(asyncio.shield(inflight), timeout=10.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass
            self._warmup_in_flight = False

        task = get_task_tracker().create_task(
            self._warmup_impl(
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
        )
        self._warmup_inflight = task
        self._warmup_started_at = time.monotonic()
        try:
            return bool(await asyncio.shield(task))
        finally:
            if self._warmup_inflight is task and task.done():
                self._warmup_inflight = None

    async def _warmup_impl(
        self,
        *,
        foreground_request: bool | None = None,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Boot the worker and prove the visible conversation path is ready."""
        if _shutdown_blocks_model_work(self.model_path, action="warmup"):
            self._warmup_in_flight = False
            if self._lane_state not in {"failed", "cold"}:
                self._set_lane_state("cold", "runtime_shutdown")
            return False
        if foreground_request is None:
            foreground_request = self._is_primary_or_deep_lane()
        else:
            foreground_request = bool(foreground_request)
        request_is_background = not foreground_request
        owner_name = f"warmup:{os.path.basename(self.model_path)}"
        warmup_timeout = self._warmup_timeout()
        self._warmup_attempted = True
        # Stale-warmup recovery lives in warmup()'s singleflight now: it owns
        # the task handle, so it can cancel and PROVE termination instead of
        # force-clearing a shared flag against an unrelated timestamp
        # (CP126 4d8a7d6b). This flag remains the cheap state other lifecycle
        # paths poll.
        self._warmup_in_flight = True
        self._set_lane_state("warming")
        try:
            if foreground_request:
                try:
                    async with _foreground_owner_context(
                        owner_name,
                        # [STABILITY v56] Raised from 90s → 180s. The 32B model
                        # cold-loads in 90-150s; holding the foreground owner
                        # for only 90s released it before warmup finished,
                        # allowing background 7B spawns to evict the cortex.
                        deadline=get_deadline(max(180.0, warmup_timeout)),
                        foreground_request=True,
                    ):
                        alive = await self._ensure_worker_alive(
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            skip_swap_cooldown=skip_swap_cooldown,
                        )
                        if not alive:
                            if self._lane_state != "failed":
                                self._set_lane_state("recovering", "warmup_deferred")
                            logger.info(
                                "⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path)
                            )
                            return False

                        try:
                            await self._run_warmup_precompile(
                                request_is_background=request_is_background,
                                foreground_request=foreground_request,
                                owner_name=owner_name,
                                warmup_timeout=warmup_timeout,
                            )
                        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                            self._set_lane_state(
                                "recovering", f"warmup_precompile_failed:{type(e).__name__}"
                            )
                            _record_mlx_degradation(
                                e,
                                action="kept warmup lane recoverable after foreground precompile failure",
                            )
                            self._record_degraded_event(
                                "warmup_precompile_failed",
                                detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                                severity="warning",
                                foreground_request=foreground_request,
                            )
                            logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                            return False
                except TimeoutError as exc:
                    self._set_lane_state("recovering", "warmup_foreground_owner_timeout")
                    self._record_degraded_event(
                        "warmup_foreground_owner_timeout",
                        detail=f"{os.path.basename(self.model_path)}:{exc}",
                        severity="warning",
                        foreground_request=foreground_request,
                    )
                    logger.info(
                        "⏸️ [MLX] Warmup deferred for %s: %s", os.path.basename(self.model_path), exc
                    )
                    return False
                return True

            if _shutdown_blocks_model_work(self.model_path, action="background warmup"):
                return False

            # CP126 811cde6f: this yield check used to run AFTER
            # _ensure_worker_alive, so a background lane could load a 20GB
            # model (or evict the resident one) and only then decide to defer
            # its precompile — defeating the very anti-thrash policy the check
            # exists to enforce. Decide BEFORE touching worker lifecycle.
            #
            # Background lanes (solver promotions, brainstem appraisals) yield
            # to an owned foreground. The PRIMARY lane's own warmup is exempt:
            # the foreground owner is usually a turn WAITING on exactly this
            # warmup, and deferring it deadlocked the lane live (2026-07-10:
            # 206s foreground budget expired every turn while the precompile
            # it needed sat deferred behind it).
            if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
                logger.info(
                    "⏸️ [MLX] Background warmup deferred for %s (before worker spawn) while foreground lane is owned by %s.",
                    os.path.basename(self.model_path),
                    _FOREGROUND_OWNER_NAME or "foreground",
                )
                return False

            alive = await self._ensure_worker_alive(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
            if not alive:
                if self._lane_state != "failed":
                    self._set_lane_state("recovering", "warmup_deferred")
                logger.info("⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path))
                return False
            if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
                # Re-check: a foreground turn can take ownership while the
                # worker was coming up.
                logger.info(
                    "⏸️ [MLX] Background warmup precompile deferred for %s while foreground lane is owned by %s.",
                    os.path.basename(self.model_path),
                    _FOREGROUND_OWNER_NAME or "foreground",
                )
                return False

            try:
                await self._run_warmup_precompile(
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_name=owner_name,
                    warmup_timeout=warmup_timeout,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self._set_lane_state("recovering", f"warmup_precompile_failed:{type(e).__name__}")
                _record_mlx_degradation(
                    e,
                    action="kept warmup lane recoverable after precompile failure",
                )
                self._record_degraded_event(
                    "warmup_precompile_failed",
                    detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                return False
            return True
        finally:
            self._warmup_in_flight = False

    async def warm_up(self, **kwargs):
        """Backward-compatible alias for older call sites."""
        return await self.warmup(**kwargs)

    async def reboot_worker(self, reason: str = "manual_reboot", mark_failed: bool = False):
        """Forcibly reboots the worker.

        LOCK DISCIPLINE (CP126 ec341dfa). This used to log "forcing reboot
        anyway" and then kill the process, replace queues, cancel futures and
        reset ownership while the actual lock holder was still operating —
        converting a SUSPECTED deadlock into GUARANTEED unsynchronized
        corruption. Contention is now waited out (a real lifecycle op is
        bounded by its own timeouts) and the destructive path is a deliberate,
        receipted last resort after repeated failures to acquire, not the
        first response to 10 seconds of contention.
        """
        self._set_lane_state("recovering", reason)
        # A new generation begins here, so the next owner id cannot be confused
        # with the one being torn down.
        self._worker_generation = int(getattr(self, "_worker_generation", 0) or 0) + 1
        self._model_lane_owner_id = ""
        acquired = await asyncio.to_thread(self._lock.acquire, True, 10.0)
        if not acquired:
            # Escalate the wait before considering anything unsynchronized.
            acquired = await asyncio.to_thread(
                self._lock.acquire, True, _REBOOT_LOCK_ESCALATED_WAIT_S
            )
        forced_unsynchronized = False
        if not acquired:
            self._reboot_lock_failures += 1
            forced_unsynchronized = self._reboot_lock_failures >= _REBOOT_LOCK_FORCE_AFTER
            if not forced_unsynchronized:
                _record_mlx_degradation(
                    TimeoutError(f"reboot_lock_unavailable:{reason}"),
                    action=(
                        "deferred reboot instead of mutating worker lifecycle state "
                        "without the lifecycle lock"
                    ),
                    severity="error",
                )
                logger.error(
                    "🚨 [MLX] Could not acquire _lock for reboot on %s after %.0fs "
                    "(attempt %d/%d). DEFERRING — another lifecycle operation owns "
                    "this lane.",
                    os.path.basename(self.model_path),
                    10.0 + _REBOOT_LOCK_ESCALATED_WAIT_S,
                    self._reboot_lock_failures,
                    _REBOOT_LOCK_FORCE_AFTER,
                )
                self._set_lane_state("recovering", f"reboot_deferred_lock:{reason}")
                return
            _record_mlx_degradation(
                TimeoutError(f"reboot_lock_wedged:{reason}"),
                action=(
                    "forced an unsynchronized reboot after repeated lock-acquisition "
                    "failures — the lock holder is presumed wedged"
                ),
                severity="critical",
            )
            logger.critical(
                "🚨 [MLX] Lock holder for %s presumed WEDGED after %d failed reboot "
                "acquisitions. Forcing unsynchronized reboot as a last resort.",
                os.path.basename(self.model_path),
                self._reboot_lock_failures,
            )
        else:
            self._reboot_lock_failures = 0
        try:
            # A forced abort that could not take this lock left its
            # reconciliation for whoever did. Clear it first so the reboot
            # below is not racing a half-torn-down lane.
            self._force_abort_reconcile_pending = None
            self._force_abort_lock_failures = 0
            self._unbind_mycelial_worker()
            process = self._process
            if process is not None:
                # K4 accounting: the breaker classifies this death by reason
                # (deliberate yields never count; young crashes do).
                _note_lane_worker_death(self, reason)
                termination_proven = await asyncio.to_thread(
                    self._kill_and_join_blocking,
                    process,
                    cooperative=True,
                )
                if not termination_proven:
                    self._set_lane_state(
                        "recovering",
                        f"reboot_worker_termination_unproven:{reason}",
                    )
                    raise RuntimeError(
                        "mlx_worker_termination_unproven_before_reboot:"
                        f"pid={getattr(process, 'pid', 'unknown')}"
                    )
            self._process = None
            self._init_done = False
            self._expert_adapter_path = None  # adapters live in the worker process
            self._last_heartbeat = 0.0
            self._last_progress_at = 0.0
            self._last_token_progress_at = 0.0
            # Reset the cold-start anchor so the next foreground request
            # gets the generous 40 s SLA instead of the tight warm-path 22 s.
            # A reboot means the worker process is gone → first-token budget
            # includes Metal shader recompile, KV rebuild, and weight reload.
            self._last_generation_completed_at = 0.0
            self._last_user_facing_completed_at = 0.0
            self._last_visible_readiness_at = 0.0
            self._process_started_at = 0.0
            self._current_request_started_at = 0.0
            self._current_first_token_at = 0.0
            self._current_request_id = ""
            self._current_request_seq = 0
            # A reboot orphans any cooperative-cancel request with the worker.
            cancel_seq = getattr(self, "_cancel_seq", None)
            if cancel_seq is not None:
                try:
                    cancel_seq.value = 0
                except (OSError, ValueError):
                    logger.debug("Cancel channel reset skipped during reboot.")
            if self._listener_task:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None
            self._cancel_lane_renewal_task()

            # [OOM FIX] Force memory reclaim after killing heavy model process
            gc.collect()

            # RECREATE QUEUES TO PREVENT ZOMBIE THREADS STEALING MESSAGES
            self._replace_ipc_queues()

            pending_request_ids = [
                req_id
                for req_id, future in self._pending_generations.items()
                if future is not None and not future.done()
            ]
            if mark_failed:
                self._expected_cancels.clear()
            elif pending_request_ids:
                self._note_expected_generation_cancellation(
                    reason, request_ids=pending_request_ids
                )

            cleared_owner = _clear_matching_foreground_owner(
                f"warmup:{os.path.basename(self.model_path)}",
            )
            if cleared_owner:
                logger.warning(
                    "♻️ [MLX] Cleared stale foreground owner %s while rebooting %s.",
                    cleared_owner,
                    os.path.basename(self.model_path),
                )

            for future in list(self._pending_generations.values()):
                _cancel_shared_future(future)
            self._release_detached_request_lock()
            self._pending_generations.clear()
            self._clear_detached_worker_requests()
            self._current_gen_future = None
            self._active_generations = 0
            if self._init_future is not None:
                _cancel_shared_future(self._init_future)
            self._init_future = None
            self._warmup_in_flight = False
            self._consecutive_empty = (
                0  # [STABILITY v53] Reset on reboot — prevents false recovery triggers
            )
        finally:
            if acquired:
                self._lock.release()
        try:
            await self._release_durable_model_lane_owner(reason=reason)
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="worker stopped but durable model-lane owner release failed",
                severity="warning",
            )
        self._set_lane_state("failed" if mark_failed else "cold", reason if mark_failed else "")

    def idle_age(self, now: float | None = None) -> float:
        """Seconds since this lane last did anything meaningful.

        Anchored on the most recent of: generation completion, user-facing
        completion, token/stream progress, or worker start. Returns 0.0 when
        the lane has no activity anchor yet (freshly spawned, never used) so a
        brand-new worker is never treated as idle.
        """
        now = float(now if now is not None else time.time())
        anchors = (
            self._last_generation_completed_at,
            self._last_user_facing_completed_at,
            self._last_progress_at,
            self._last_token_progress_at,
            self._process_started_at,
        )
        last = max((float(a or 0.0) for a in anchors), default=0.0)
        if last <= 0.0:
            return 0.0
        return max(0.0, now - last)

    async def _set_durable_lane_preemptible(self, preemptible: bool) -> bool:
        fencing_token = int(self._model_lane_fencing_token or 0)
        owner_id = str(self._model_lane_owner_id or "")
        if fencing_token <= 0 or not owner_id:
            return True
        try:
            from core.runtime.model_lane_control import get_model_lane_controller

            return await get_model_lane_controller().update_owner_preemptibility(
                owner_id,
                fencing_token=fencing_token,
                preemptible=preemptible,
            )
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action=(
                    "refused generation before losing active-use protection"
                    if not preemptible
                    else "kept the idle model lane conservatively non-preemptible"
                ),
                severity="critical" if not preemptible else "warning",
            )
            return False

    async def _finish_generation_ownership(
        self,
        request_id: str,
        future: SharedFuture,
        foreground_watchdog: _threading.Timer | None,
        *,
        release_lane: bool = True,
    ) -> None:
        if foreground_watchdog is not None:
            foreground_watchdog.cancel()
        if self._foreground_generation_watchdog is foreground_watchdog:
            self._foreground_generation_watchdog = None

        # CP126 b614dafc. This decremented the active count unconditionally,
        # so a second cleanup for the same request — a cancellation path and a
        # finally block both firing, say — undercounted concurrent work and
        # could publish the lane as preemptible while another generation was
        # still decoding. Cleanup has to be idempotent, because the paths that
        # call it are not mutually exclusive.
        was_pending = self._pending_generations.pop(request_id, None) is not None
        was_current_future = self._current_gen_future is future
        if was_current_future:
            self._current_gen_future = None
        was_current_request = self._current_request_id == request_id
        already_cleaned = not (was_pending or was_current_future or was_current_request)
        if already_cleaned:
            logger.debug(
                "Generation ownership for %s was already released; skipping decrement.",
                str(request_id)[:12],
            )
            return

        self._active_generations = max(0, self._active_generations - 1)
        if was_current_request:
            self._clear_active_generation_tracking()
        if release_lane and self._active_generations <= 0:
            # CP126 860036f2. The result used to be discarded. A failed
            # release leaves the durable lane permanently NON-PREEMPTIBLE —
            # nothing can evict it, and nothing anywhere says why — so the
            # lane is stuck until a process restart. Record it and mark the
            # lane so a later reader can see the release is owed.
            released = await self._set_durable_lane_preemptible(True)
            if not released:
                self._durable_lane_release_owed = True
                _record_mlx_degradation(
                    RuntimeError(
                        f"durable lane for {os.path.basename(self.model_path)} could "
                        f"not be made preemptible after request {str(request_id)[:12]}"
                    ),
                    action=(
                        "flagged the lane as owing a preemptibility release; it "
                        "cannot be evicted until that succeeds"
                    ),
                    severity="error",
                )
            else:
                self._durable_lane_release_owed = False

    def _unload_safety_blocker(self) -> str | None:
        """Return why an idle VRAM unload is unsafe right now, or None if safe.

        An unload tears down the worker (≈model size of unified memory). It must
        never interrupt in-flight or imminent work, so we refuse while any
        generation, warmup, queued request, pending future, or foreground owner
        is active — and during shutdown (close handles that path).
        """
        if self._closed:
            return "closed"
        if _runtime_shutdown_requested():
            return "shutdown"
        if not self.is_alive():
            return "already_unloaded"
        if self._active_generations > 0:
            return "active_generation"
        if self._warmup_in_flight:
            return "warming"
        if self._current_request_started_at > 0.0:
            return "request_in_flight"
        pending = [
            f
            for f in (
                *self._pending_generations.values(),
                self._current_gen_future,
                self._init_future,
            )
            if f is not None and not f.done()
        ]
        if pending:
            return "pending_future"
        if _foreground_owner_active():
            return "foreground_active"
        return None

    async def maybe_unload_idle(
        self,
        *,
        pressure_idle_s: float = 90.0,
        hard_idle_s: float = 900.0,
    ) -> dict[str, Any]:
        """Unload the model from memory if the lane has been safely idle.

        Two triggers, whichever fires first:
          - under memory pressure → unload after ``pressure_idle_s`` of idle
            (reclaim ~model-size of RAM/VRAM for the system when it's needed),
          - regardless of pressure → unload after ``hard_idle_s`` of idle (be a
            good citizen during long quiet periods).

        The next request transparently respawns the worker via
        ``_ensure_worker_alive``. This is a normal lifecycle event, not a
        failure, so it records no degradation. Returns a telemetry dict.
        """

        # Programmatic thresholds get the same fail-safe normalization as
        # env values: NaN bypasses every age comparison and a negative value
        # tears down a lane that idled for one tick.
        def _safe_threshold(value: float, default: float) -> float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(value) or value < 0.0:
                return default
            return value

        pressure_idle_s = _safe_threshold(pressure_idle_s, 90.0)
        hard_idle_s = _safe_threshold(hard_idle_s, 900.0)
        blocker = self._unload_safety_blocker()
        if blocker:
            return {"unloaded": False, "reason": blocker}
        age = self.idle_age()
        if age <= 0.0:
            return {"unloaded": False, "reason": "no_idle_anchor"}

        under_pressure = False
        try:
            snapshot = get_memory_pressure_snapshot()
            under_pressure = bool(getattr(snapshot, "warning", False))
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            logger.debug("Idle scavenge pressure probe unavailable: %s", exc)

        # The PRIMARY lane stays resident when there is no memory pressure.
        # The 20260708-final soak started against a cortex the citizenship
        # unload had evicted during a quiet afternoon at 34% system RAM —
        # the first turn then paid a 120-150s cold start (and, pre-fix,
        # seeded the gate-orphan cascade). A resident 20GB cortex on an
        # unpressured 64GB machine is what the machine is FOR; the 90s
        # pressure path still reclaims it the moment RAM actually matters.
        # Small lanes (brainstem/reflex, seconds to reload) keep the
        # citizenship unload. AURA_VRAM_SCAVENGE_PRIMARY_HARD=1 restores
        # the old behavior.
        if not under_pressure and self._is_primary_lane():
            if os.environ.get("AURA_VRAM_SCAVENGE_PRIMARY_HARD", "0").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return {
                    "unloaded": False,
                    "reason": "primary_lane_stays_resident_without_pressure",
                    "idle_age_s": round(age, 1),
                }

        threshold = pressure_idle_s if under_pressure else hard_idle_s
        if age < threshold:
            return {
                "unloaded": False,
                "reason": "not_idle_enough",
                "idle_age_s": round(age, 1),
                "threshold_s": threshold,
                "under_pressure": under_pressure,
            }

        # Capture the worker's resident memory for an honest freed estimate
        # before the process is torn down.
        freed_bytes = 0
        process = self._process
        if process is not None and getattr(process, "pid", None):
            freed_bytes = _observed_process_rss_bytes(int(process.pid))

        # CP126 dcae0f1f. Two lock-free safety checks only SHRINK the window;
        # they cannot close it. A request admitted between the last check and
        # the reboot was killed by the scavenger — on the primary lane that is
        # a person's turn dying into a 120-second cold start, caused by the
        # very reclaim meant to keep the machine responsive.
        #
        # Hold the request lane across the final check and the teardown, so
        # nothing can be admitted in between. Same fence, same ordering
        # (request lane, then lifecycle lock) as a lane eviction.
        fenced = await self._acquire_request_lock(
            owner_label="idle_vram_scavenge",
            deadline=get_deadline(_LANE_EVICTION_FENCE_WAIT_S),
            foreground_request=False,
        )
        if not fenced:
            return {"unloaded": False, "reason": "request_lane_busy"}
        try:
            # Re-check under the fence. Now a clean result cannot go stale
            # before the teardown acts on it.
            blocker = self._unload_safety_blocker()
            if blocker:
                return {"unloaded": False, "reason": blocker}

            logger.info(
                "🧹 [MLX] Idle VRAM scavenge: unloading %s after %.0fs idle "
                "(pressure=%s, ~%.1fGB).",
                os.path.basename(self.model_path),
                age,
                under_pressure,
                freed_bytes / float(1024**3),
            )
            await self.reboot_worker(reason="idle_vram_scavenge")
        finally:
            self._release_request_lock()
        return {
            "unloaded": True,
            "model": os.path.basename(self.model_path),
            "idle_age_s": round(age, 1),
            "under_pressure": under_pressure,
            "freed_gb_estimate": round(freed_bytes / float(1024**3), 2),
        }

    def close(self) -> None:
        """Release worker process and multiprocessing IPC resources."""
        pending_futures = {
            id(future): future
            for future in list(self._pending_generations.values())
            + [self._current_gen_future, self._init_future]
            if future is not None and not future.done()
        }
        # CP126 97aa64fc: close used to give the lifecycle lock ONE second and
        # then destroy the client regardless — cancelling futures, killing the
        # process and closing queues while another lifecycle operation was
        # still using them. close() is terminal so it must always finish, but
        # it now waits long enough for ordinary contention to clear and
        # receipts the case where it genuinely could not.
        acquired = self._lock.acquire(timeout=_CLOSE_LOCK_WAIT_S)
        shutdown_proven = True
        if not acquired:
            _record_mlx_degradation(
                TimeoutError("close_lock_unavailable"),
                action=(
                    "closed the client without the lifecycle lock after waiting "
                    f"{_CLOSE_LOCK_WAIT_S:.0f}s — shutdown cannot be deferred"
                ),
                severity="error",
            )
        try:
            self._unbind_mycelial_worker()
            for future in pending_futures.values():
                _cancel_shared_future(future)
            self._pending_generations.clear()
            self._clear_detached_worker_requests()
            self._current_gen_future = None
            self._init_future = None
            self._active_generations = 0
            self._init_done = False
            self._warmup_in_flight = False
            self._deferred_reboot_reason = None
            if self._listener_task is not None:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None
            self._cancel_lane_renewal_task()
            process = self._process
            if process is not None:
                shutdown_proven = self._kill_and_join_blocking(
                    process,
                    cooperative=True,
                )
            if shutdown_proven:
                self._process = None
                self._drain_queue()
                self._close_ipc_queues()
            self._release_request_lock()
            self._closed = shutdown_proven
            self._set_lane_state(
                "closed" if shutdown_proven else "shutdown_failed",
                "shutdown" if shutdown_proven else "worker_termination_unproven",
            )
        finally:
            if acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    logger.debug(
                        "Loop-agnostic lifecycle lock for %s was already released.",
                        os.path.basename(self.model_path),
                    )
        if not shutdown_proven:
            raise RuntimeError("mlx_worker_termination_unproven")
        try:
            self._release_durable_model_lane_owner_sync(reason="client_close")
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="client closed but durable model-lane owner release failed",
                severity="warning",
            )

    async def aclose(self) -> None:
        """Async shutdown hook for runtime coordinators."""
        await asyncio.to_thread(self.close)

    cleanup = close
    on_stop = close

    def __del__(self):
        # A finalizer running during interpreter shutdown can hit almost any
        # failure as modules are torn out from under it; ImportError in
        # particular escaped the old tuple. Nothing here is recoverable and
        # nothing here is worth a traceback on exit.
        try:
            self.close()
        except Exception as exc:  # noqa: BLE001 - finalizer during teardown
            log = getattr(globals().get("logger"), "debug", None)
            if callable(log):
                log("MLX client finalizer could not close cleanly: %s", exc)


def seconds_to_read(prompt_chars: int, client: Any = None) -> float:
    """How long this worker takes to read a prompt of this size, as measured.

    The turn's clock has to cover reading the prompt as well as writing the
    answer, and only the writing was ever counted. A ten-thousand-character
    prompt takes most of a foreground turn to read, so a budget sized on decode
    alone promises an answer the clock cannot pay for — generation is then
    aborted at the deadline and everything produced is discarded.

    Pessimistic where nothing has been measured, for the reason
    ``_measured_prefill_rate`` gives: being generous with an unmeasured worker
    costs a little latency, and being mean with it costs the answer.
    """
    chars = max(0, int(prompt_chars or 0))
    if chars <= 0:
        return 0.0
    rate = _UNMEASURED_PREFILL_RATE
    if client is not None:
        try:
            rate = float(client._measured_prefill_rate())
        except (AttributeError, TypeError, ValueError):
            rate = _UNMEASURED_PREFILL_RATE
    if not (rate > 0.0):
        rate = _UNMEASURED_PREFILL_RATE
    return (chars / _CHARS_PER_TOKEN / rate) * _PREFILL_HEADROOM


def get_mlx_client(
    model_path: str | None = None,
    *,
    runtime_assignment: ModelRuntimeAssignment | Mapping[str, Any] | None = None,
    origin: str = "mlx_client",
    **kwargs,
) -> MLXLocalClient:
    """Compatibility factory for Aura's active local backend."""
    from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

    from .model_registry import (
        ACTIVE_MODEL,
        get_local_backend,
        get_model_path,
        get_model_runtime_assignment,
        get_runtime_model_path,
    )

    if model_path is None:
        model_path = get_runtime_model_path()

    resolved_model_path = str(get_model_path(model_path)).strip()
    path_candidate = Path(resolved_model_path).expanduser()
    if path_candidate.is_absolute() or path_candidate.exists():
        runtime_path = str(path_candidate.resolve() if path_candidate.exists() else path_candidate)
        client_key = os.path.realpath(runtime_path)
    else:
        runtime_path = resolved_model_path
        client_key = resolved_model_path

    if runtime_assignment is None:
        assignment = get_model_runtime_assignment(runtime_path)
    elif isinstance(runtime_assignment, ModelRuntimeAssignment):
        assignment = runtime_assignment
    elif isinstance(runtime_assignment, Mapping):
        assignment = ModelRuntimeAssignment.from_dict(runtime_assignment)
    else:
        raise TypeError("mlx_client_runtime_assignment_invalid")
    assignment.assert_bound_to(model_path=runtime_path, purpose="serve")

    try:
        from core.runtime.proof_policy import proof_model_tier, proof_run_active

        if (
            proof_run_active(origin=origin)
            and proof_model_tier() == "primary"
        ):
            primary_path = _real_model_path(get_model_path(ACTIVE_MODEL))
            target_path = _real_model_path(runtime_path)
            # CP126 0ad66338. Name comparison and a size-tag compatibility
            # predicate cannot tell two artifacts apart: two directories both
            # called Qwen2.5-32B-Instruct-4bit, holding different weights,
            # both satisfied the proof lane. A proof run is precisely where
            # "probably the same model" is not good enough.
            _assert_proof_primary_artifact_identity(primary_path, target_path)
    except ImportError as _exc:
        # CP126 84a18b06: swallowing this import failure disabled proof-primary
        # model enforcement EXACTLY when its enforcement infrastructure was
        # unavailable, so a proof run could fall through to an unchecked lane.
        # Fail CLOSED for proof runs; ordinary construction still proceeds.
        _record_mlx_degradation(
            _exc,
            action="could not import proof policy for primary-lane enforcement",
            severity="error",
        )
        if _proof_run_requested(origin):
            raise RuntimeError(
                "proof_policy_unavailable: refusing to build an unenforced model "
                "client for a proof run"
            ) from _exc

    backend = get_local_backend()
    if backend != "mlx" or str(runtime_path).lower().endswith(".gguf"):
        raise RuntimeError(
            "external_cortex_disabled:"
            " live Aura uses the in-process MLX model lane; external Cortex artifacts are retired"
        )

    with _CLIENTS_LOCK:
        existing = _CLIENTS.get(client_key)
        if existing is not None and getattr(existing, "_closed", False):
            _CLIENTS.pop(client_key, None)
            existing = None
    if existing is not None:
        existing_assignment = getattr(existing, "runtime_assignment", None)
        if not isinstance(existing_assignment, ModelRuntimeAssignment):
            raise RuntimeError("mlx_client_existing_runtime_assignment_missing")
        if existing_assignment.assignment_sha256 != assignment.assignment_sha256:
            raise RuntimeError("mlx_client_runtime_assignment_conflict")
        return existing
    # Construction happens outside the lock (it can be slow), then a
    # last-writer check keeps a concurrent creator from being discarded.
    created = MLXLocalClient(
        model_path=runtime_path,
        runtime_assignment=assignment,
        **kwargs,
    )
    with _CLIENTS_LOCK:
        existing = _CLIENTS.get(client_key)
        if existing is not None and getattr(existing, "_closed", False):
            _CLIENTS.pop(client_key, None)
            existing = None
        if existing is None:
            _CLIENTS[client_key] = created
            return created
        existing_assignment = getattr(existing, "runtime_assignment", None)
        if not isinstance(existing_assignment, ModelRuntimeAssignment):
            created.close()
            raise RuntimeError("mlx_client_existing_runtime_assignment_missing")
        if existing_assignment.assignment_sha256 != assignment.assignment_sha256:
            created.close()
            raise RuntimeError("mlx_client_runtime_assignment_conflict")
    # A concurrent constructor won the registry race. Do not strand a second
    # worker/model allocation outside registry ownership.
    created.close()
    return existing


#: Context keys that CONFER AUTHORITY. A caller handing an untyped mapping to
#: the agent loop must not be able to set any of them: they are the answers
#: the governance path exists to compute, not inputs to it.
_AUTHORITY_BEARING_CONTEXT_KEYS = frozenset(
    {
        "_standing_authority_verified",
        "confirmed",
        "sealed_validation",
        "proof_run",
        "proof_validation",
        "proof_evaluation_contract",
        "executive_constraints",
        "standing_authority_token",
        "signed_capability",
        "authority_payload",
        "operation_authority",
        "principal",
    }
)


def _agent_execution_context(
    context: Any,
    *,
    objective: str,
    tool_name: str,
    tool_call_id: str,
    model_path: str,
) -> dict[str, Any]:
    """The execution context this loop is willing to vouch for.

    CP126 9cddd95c: think_and_act forwarded an untyped caller mapping straight
    into CapabilityEngine as authority-bearing execution context, and
    established no principal, scope or provenance of its own. Safety depended
    entirely on every caller and on how each downstream reader interpreted the
    keys — including keys like ``confirmed`` and ``_standing_authority_verified``
    that are the ANSWERS the governance path computes, not inputs to it.

    Ordinary context still passes through: this loop needs the caller's route,
    memory handles and origin to work. What it refuses is the set of keys that
    grant authority, and it stamps its own provenance so a downstream reader
    can tell a claim made BY the agent loop from one made THROUGH it.
    """
    forwarded: dict[str, Any] = {}
    refused: list[str] = []
    if isinstance(context, Mapping):
        for key, value in context.items():
            name = str(key)
            if name in _AUTHORITY_BEARING_CONTEXT_KEYS:
                refused.append(name)
                continue
            forwarded[name] = value
    elif context:
        refused.append(f"<non_mapping:{type(context).__name__}>")

    if refused:
        _record_mlx_degradation(
            PermissionError(
                f"agent loop refused authority-bearing context keys: {sorted(set(refused))}"
            ),
            action="executed the tool without caller-supplied authority claims",
            severity="warning",
        )

    forwarded.setdefault("source", "think_and_act")
    forwarded["agent_loop"] = {
        "schema": "aura.mlx.agent_loop_provenance.v1",
        "source": "mlx_client.think_and_act",
        "objective_sha256": hashlib.sha256(
            str(objective or "").encode("utf-8", "ignore")
        ).hexdigest(),
        "tool": str(tool_name or ""),
        "tool_call_id": str(tool_call_id or ""),
        "model": os.path.basename(str(model_path or "")) or "unknown",
        "refused_authority_keys": sorted(set(refused)),
    }
    return forwarded


_TOOL_ARGS_MAX_KEYS = 64
_TOOL_ARGS_MAX_DEPTH = 6
_TOOL_ARGS_MAX_CHARS = 20_000
_NATIVE_XML_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}")
_NATIVE_XML_FUNCTION_RE = re.compile(
    r"\A\s*<function=(?P<name>[A-Za-z_][A-Za-z0-9_.:-]{0,127})>"
    r"(?P<body>.*?)</function>",
    re.DOTALL,
)
_NATIVE_XML_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_.:-]{0,127})>"
    r"(?P<value>.*?)</parameter>",
    re.DOTALL,
)


def _record_tool_receipt_for_this_turn(
    tool_name: Any, tool_args: Any, raw_result: Any, serialized: Any
) -> None:
    """Attach one finished tool call to the turn that asked for it."""

    try:
        from core.conversation.surface_disposition import record_tool_receipt

        record_tool_receipt(
            str(tool_name or "tool"),
            ok=bool(_tool_turn_outcome(raw_result) == "ok"),
            action="execute",
            object_ref=str(tool_args or "")[:200],
            effect_observed=True,
            verification="the tool returned a result during this turn",
            observed_content=str(serialized or "")[:2000],
        )
    except Exception as exc:  # noqa: BLE001 - a receipt must never end a turn
        logger.debug(
            "[think_and_act] could not record a receipt for %s: %s", tool_name, exc
        )


def _text_is_a_complete_tool_call(text: Any) -> bool:
    """Whether this draft is one parseable call and nothing else of substance.

    Read with the same parser the loop uses, so the two cannot disagree about
    what a call is. Prose wrapped around a call does not qualify: that is a
    draft with a call in it, and judging it as prose is then correct.
    """

    body = str(text or "").strip()
    if not body:
        return False
    opened = body.find("<tool_call>")
    if opened < 0:
        return False
    if body[:opened].strip():
        return False  # the model said something first; that is prose
    payload, _why = _native_xml_tool_payload(
        body, start=opened + len("<tool_call>"), tool_definitions=None
    )
    return bool(isinstance(payload, dict) and payload.get("name"))


def _native_xml_tool_payload(
    text: str,
    *,
    start: int,
    tool_definitions: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    """Parse the typed XML function grammar used by newer Qwen templates.

    Qwen3.8 no longer emits JSON inside ``<tool_call>``. Its checkpoint-owned
    template emits one ``<function=name>`` block containing typed
    ``<parameter=name>`` blocks. Treating that as prose makes every correctly
    generated call disappear at the parent boundary.

    This parser implements only the template's closed grammar. It rejects
    duplicate parameters, unclosed functions, extra material inside the
    function, and values that cannot be converted to the advertised JSON
    schema type. The later schema validator still owns required fields,
    enums, bounds, and additional properties.
    """

    candidate = str(text[start:] or "")
    if len(candidate) > _TOOL_ARGS_MAX_CHARS * 2:
        return None, "native XML envelope exceeded the bounded scan size"
    match = _NATIVE_XML_FUNCTION_RE.match(candidate)
    if match is None:
        return None, "no complete JSON object or XML function after the tag"

    suffix = candidate[match.end() :].strip()
    if suffix.startswith("</tool_call>"):
        suffix = suffix[len("</tool_call>") :].strip()
    # The envelope ends at </function>. What the model says afterwards is the
    # model carrying on talking, and it does not make the call before it
    # invalid — material INSIDE the function is ambiguous in a way material
    # after it is not.
    #
    # LIVE, 2026-08-28: a request to work out what was wrong with a project
    # produced a correct diagnose_repo call with prose behind it, the whole
    # envelope was refused, the body was empty, and the turn ended with no
    # answer. The tool that had the answer was called and the call was thrown
    # away for the sentence after it.
    #
    # A second call in the tail is a different matter: taking the first and
    # dropping the rest silently would be worse than refusing, because nothing
    # downstream would ever learn the model asked for two things.
    # A stray parameter after the close may have been meant for THIS call, so
    # taking the call without it would change what was asked for.
    #
    # It is only stray when nothing follows that could own it. A second
    # complete call contains parameters of its own, and this test ran before
    # the one below that exists to handle exactly that — so two well-formed
    # calls were refused for the crime of the second one containing the word
    # "parameter".
    #
    # LIVE, 2026-08-28: asked to read a library's docs and use it, the model
    # emitted two correct file_operation calls, the first complete and the
    # second cut off mid-path by the budget. Both were thrown away, the raw
    # envelope leaked into the user-facing draft, and the turn ended on an
    # apology. The first call was exactly right.
    _something_after_could_own_it = bool(
        suffix and ("<function" in suffix or "<tool_call" in suffix)
    )
    if suffix and "<parameter" in suffix and not _something_after_could_own_it:
        return None, "native XML envelope carried a parameter after the function"
    # A second function is a second thing to do, and this loop takes more than
    # one turn: the first call is run, and the model is asked again with the
    # result in hand. Refusing both loses a turn to say nothing.
    #
    # LIVE, 2026-08-28: turn two of a diagnosis emitted a call with more markup
    # behind it, both were refused, and the turn ended having run one tool and
    # said nothing about it.
    if suffix and ("<function" in suffix or "tool_call" in suffix):
        logger.info(
            "🔧 [TOOL CALL] took the first function and left %d chars of "
            "markup behind it for the next turn: %r",
            len(suffix),
            suffix[:200],
        )

    function_name = match.group("name")
    if _NATIVE_XML_NAME_RE.fullmatch(function_name) is None:
        return None, "native XML function name was invalid"

    properties = _tool_parameter_properties(tool_definitions, function_name)
    arguments: dict[str, Any] = {}
    body = match.group("body")
    cursor = 0
    for parameter in _NATIVE_XML_PARAMETER_RE.finditer(body):
        if body[cursor : parameter.start()].strip():
            return None, "native XML function contained material outside parameters"
        name = parameter.group("name")
        if name in arguments:
            return None, f"native XML parameter '{name}' was repeated"
        value = _strip_native_xml_framing(parameter.group("value"))
        try:
            arguments[name] = _coerce_native_xml_parameter(
                value,
                properties.get(name),
            )
        except ValueError as exc:
            return None, f"native XML parameter '{name}' was invalid: {exc}"
        cursor = parameter.end()
    if body[cursor:].strip():
        return None, "native XML function contained material outside parameters"
    if len(arguments) > _TOOL_ARGS_MAX_KEYS:
        return None, "native XML function had too many parameters"
    return {"name": function_name, "arguments": arguments}, ""


def _tool_parameter_properties(
    tool_definitions: Mapping[str, Any] | None,
    function_name: str,
) -> dict[str, Any]:
    if not isinstance(tool_definitions, Mapping):
        return {}
    definition = tool_definitions.get(function_name)
    if not isinstance(definition, Mapping):
        return {}
    if isinstance(definition.get("function"), Mapping):
        definition = definition["function"]
    parameters = definition.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    properties = parameters.get("properties")
    return dict(properties) if isinstance(properties, Mapping) else {}


def _strip_native_xml_framing(value: str) -> str:
    """Remove template framing newlines without changing parameter content."""

    result = value
    if result.startswith("\r\n"):
        result = result[2:]
    elif result.startswith("\n"):
        result = result[1:]
    if result.endswith("\r\n"):
        result = result[:-2]
    elif result.endswith("\n"):
        result = result[:-1]
    return result


def _coerce_native_xml_parameter(value: str, property_schema: Any) -> Any:
    """Convert one XML value only when the advertised schema names its type."""

    schema = property_schema if isinstance(property_schema, Mapping) else {}
    expected = schema.get("type")
    if expected == "string" or not isinstance(expected, str):
        return value
    compact = value.strip()
    if expected == "integer":
        if re.fullmatch(r"[-+]?\d+", compact) is None:
            raise ValueError("expected integer")
        return int(compact)
    if expected == "number":
        if re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            compact,
        ) is None:
            raise ValueError("expected number")
        return float(compact) if any(char in compact for char in ".eE") else int(compact)
    if expected == "boolean":
        if compact.casefold() == "true":
            return True
        if compact.casefold() == "false":
            return False
        raise ValueError("expected boolean")
    if expected in {"array", "object"}:
        try:
            parsed = json.loads(compact)
        except json.JSONDecodeError as exc:
            raise ValueError(f"expected JSON {expected}") from exc
        if expected == "array" and not isinstance(parsed, list):
            raise ValueError("expected JSON array")
        if expected == "object" and not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed
    if expected == "null":
        if compact.casefold() != "null":
            raise ValueError("expected null")
        return None
    return value


def _json_depth(value: Any, *, _depth: int = 0) -> int:
    if _depth > _TOOL_ARGS_MAX_DEPTH:
        return _depth
    if isinstance(value, dict):
        return max(
            (_json_depth(item, _depth=_depth + 1) for item in value.values()),
            default=_depth,
        )
    if isinstance(value, list):
        return max(
            (_json_depth(item, _depth=_depth + 1) for item in value),
            default=_depth,
        )
    return _depth


def _tool_arguments_schema_error(definition: Any, args: Any) -> str:
    """Validate parsed tool arguments against the tool's advertised schema.

    CP126 0da5db2e: parsed arguments went to execution with no binding to the
    schema the tool advertised for this turn — no required-field check, no
    type check, and no size/depth bound. Returns "" when the call is
    acceptable, else a short reason.
    """
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    if len(args) > _TOOL_ARGS_MAX_KEYS:
        return f"too many argument keys ({len(args)} > {_TOOL_ARGS_MAX_KEYS})"
    if _json_depth(args) > _TOOL_ARGS_MAX_DEPTH:
        return "arguments nested too deeply"
    try:
        encoded = json.dumps(args, default=str)
    except (TypeError, ValueError):
        return "arguments are not JSON-serializable"
    if len(encoded) > _TOOL_ARGS_MAX_CHARS:
        return f"arguments too large ({len(encoded)} chars)"

    spec = definition if isinstance(definition, dict) else {}
    # Accept either a bare function spec or an OpenAI-style wrapper.
    if isinstance(spec.get("function"), dict):
        spec = spec["function"]
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        return ""  # Nothing advertised to validate against.
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = parameters.get("required")
    required = [str(item) for item in required] if isinstance(required, list) else []

    missing = [name for name in required if name not in args]
    if missing:
        return f"missing required argument(s): {', '.join(sorted(missing)[:5])}"
    if properties and parameters.get("additionalProperties") is False:
        unexpected = sorted(set(args) - set(properties))
        if unexpected:
            return f"unexpected argument(s): {', '.join(unexpected[:5])}"

    json_type_map: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "object": (dict,),
        "array": (list,),
    }
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if not isinstance(expected, str):
            continue
        allowed = json_type_map.get(expected)
        if allowed is None:
            continue
        if expected in {"number", "integer"} and isinstance(value, bool):
            return f"argument '{name}' must be {expected}"
        if not isinstance(value, allowed):
            return f"argument '{name}' must be {expected}"
        enum = prop.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            return f"argument '{name}' is not one of its allowed values"
    return ""


def _tool_turn_outcome(raw_result: Any) -> str:
    """Classify a tool result for telemetry: ok, denied, or error.

    Denial is kept DISTINCT from error on purpose. A capability refusing an
    action is the governance layer working, and folding it into "error" hides
    exactly the signal an operator needs to see when Aura is being stopped from
    acting — while folding it into "ok" claims an action happened that did not.
    """
    if not isinstance(raw_result, Mapping):
        # A bare value carries no outcome contract; say so rather than guess.
        return "ok" if raw_result else "unknown"

    status = str(raw_result.get("status", "") or "").strip().lower()
    ok = raw_result.get("ok")

    if status in {"denied", "refused", "blocked", "forbidden", "unauthorized"}:
        return f"denied:{status}"
    if ok is False or status in {"error", "failed", "failure"}:
        reason = str(raw_result.get("error", "") or status or "unspecified")
        return f"error:{reason[:80]}"
    if ok is True or status in {"ok", "success", "succeeded", "completed"}:
        return "ok"
    return "unknown"


def _truncate_tool_result(result: Any, *, limit: int = 4000) -> str:
    """Bound a tool result WITHOUT cutting structured output mid-value.

    CP126 abd93abf: a raw character slice produced syntactically broken JSON
    that the model then reasoned over as if it were the real result.
    """
    text = result if isinstance(result, str) else str(result)
    if len(text) <= limit:
        return text
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if parsed is not None:
            # Re-emit a VALID, explicitly-marked truncation envelope instead
            # of a broken fragment.
            preview = json.dumps(
                parsed,
                default=str,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )

            def envelope(candidate: str) -> str:
                return json.dumps(
                    {
                        "truncated": True,
                        "original_chars": len(text),
                        "note": "Result exceeded the context budget; preview only.",
                        "preview": candidate,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )

            if len(envelope("")) <= limit:
                low = 0
                high = len(preview)
                best = ""
                while low <= high:
                    middle = (low + high) // 2
                    candidate = envelope(preview[:middle])
                    if len(candidate) <= limit:
                        best = candidate
                        low = middle + 1
                    else:
                        high = middle - 1
                return best
    marker = "\n\n...[OUTPUT TRUNCATED FOR LENGTH]..."
    if limit <= len(marker):
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _json_with_control_characters_escaped(text: str) -> str:
    """The same JSON with raw newlines inside strings escaped.

    LIVE, 2026-08-20. Asked to build a single-file web app, the model emitted
    a perfectly well-formed call whose `code` argument was a program:

        {"name": "code_repl", "arguments": {"code": "html_content = \"<html>
         <head>
         <title>Sitting Timer</title>

    Strict JSON forbids a literal newline inside a string, and that is how
    every model writes multi-line code. The object was complete, the braces
    balanced, the arguments right, and json.loads refused it — so the turn
    reported "none called" and ended in an apology.

    Only control characters INSIDE string literals are touched. Whitespace
    between tokens is where it belongs, and a payload that was already valid
    comes back unchanged.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for character in text:
        if escaped:
            out.append(character)
            escaped = False
            continue
        if character == "\\":
            out.append(character)
            escaped = in_string
            continue
        if character == '"':
            in_string = not in_string
            out.append(character)
            continue
        if in_string and character in "\n\r\t\b\f":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}[character])
            continue
        out.append(character)
    return "".join(out)


def _loads_tool_json(body: str) -> Any:
    """Parse a model-authored JSON object, tolerating how models write code."""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return json.loads(_json_with_control_characters_escaped(body))


def _balanced_json_object(text: str, start: int) -> str | None:
    """The first complete ``{...}`` at or after ``start``, or None.

    A lazy regex cannot do this. ``{"name": "x", "arguments": {"q": "y"}}``
    stops at the inner brace, and only a following literal anchor forces the
    backtrack that recovers the rest — which is exactly the anchor a stop
    sequence removes. Counting braces needs no anchor, so a tool call is read
    the same whether or not its closing tag survived generation.

    Braces inside JSON strings do not count, and a backslash escapes the next
    character.
    """
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1]
    return None


#: Saying it is about to do something, in the first person, now.
_ANNOUNCES_AN_ACTION_RE = re.compile(
    r"\b(?:"
    # Words in between: the live sentence was "Let's break down the problem
    # step by step and use code…", where the verb that matters is six words
    # from the "let's" that governs it.
    r"let(?:'s|\s+us)\s+(?:\w+\s+){0,8}?(?:use|write|run|try|compute|calculate|check)"
    r"|i(?:'ll|\s+will|\s+am\s+going\s+to|\s+can|\s+should)\s+"
    r"(?:use|write|run|try|compute|calculate|check)"
    r"|let\s+me\s+(?:use|write|run|try|compute|calculate|check)"
    r"|we\s+(?:can|could|should)\s+(?:use|write|run)"
    r"|using\s+\w+\s+to\s+(?:figure|work|solve|compute)"
    r")\b",
    re.IGNORECASE,
)

#: The thing it says it is about to use. Without this, "let's use the first
#: constraint" reads as a tool it forgot to call.
_NAMES_A_MEANS_RE = re.compile(
    r"\b(?:code|python|script|sandbox|repl|interpreter|program|tool|search|"
    r"fetch|calculator|enumerate|brute[\s-]?force)\b",
    re.IGNORECASE,
)


def _announces_an_action_it_did_not_take(text: object) -> bool:
    """Whether the reply says it is about to act and then stops.

    Narrow: the sentence has to say it is about to do something AND name what
    with. An answer that merely mentions code is not an announcement, and a
    plan that ends the turn is not an answer.
    """
    body = str(text or "").strip()
    if not body:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n", body):
        if _ANNOUNCES_AN_ACTION_RE.search(sentence) and _NAMES_A_MEANS_RE.search(sentence):
            return True
    return False


def _tool_loop_evidence_messages(evidence: Any) -> list[dict[str, Any]]:
    """Grounding blocks the turn already holds, as messages the loop can read.

    Only blocks a skill actually produced are carried: the conversational
    scaffold belongs to the reply, and wrapping a tool call in it produced an
    immediate end-of-turn. Bounded, because a document can be large and a tool
    prompt that outgrows its deadline answers nothing.
    """
    if not isinstance(evidence, (list, tuple)):
        return []
    try:
        from core.utils.injected_blocks import carries_read_evidence, stamp_grounding
    except ImportError:  # pragma: no cover - the module ships with the runtime
        return []
    carried: list[dict[str, Any]] = []
    budget = _TOOL_LOOP_EVIDENCE_CHARS
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        if not carries_read_evidence(dict(item)):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        content = content[:budget]
        budget -= len(content)
        carried.append(stamp_grounding({"role": "system", "content": content}))
        if budget <= 0:
            break
    return carried


def _code_execution_tool(allowed_tools: set[str] | None) -> str | None:
    """The offered tool that runs code, if one is offered.

    Chosen by what the tool is named for rather than from a list, so a runner
    registered tomorrow is found without an edit here.
    """
    if not allowed_tools:
        return None
    ranked = sorted(
        (name for name in allowed_tools if re.search(r"repl|run_code|sandbox|exec", str(name), re.I)),
        key=lambda name: (0 if "repl" in str(name).lower() else 1, str(name)),
    )
    return ranked[0] if ranked else None


def _refuse_action_beyond_authority(
    engine: Any, tool_name: str, tool_args: Any, context: Any
) -> dict[str, Any] | None:
    """Refuse a call more dangerous than this turn was authorised for.

    A skill's effect_scope is the worst thing it can do, so scoping by skill
    put reading a file behind permission to delete one. Scoping by ACTION lets
    the reader be offered while the destroyer stays refused — but only if the
    refusal actually happens, so it happens here, at the one place a
    foreground turn executes anything.

    Returns a tool result to hand back to the model, or None to proceed. The
    refusal goes back INTO the loop rather than ending the turn: being told
    "delete is not authorised here" is something an agent can act on, and
    ending the turn silently is not.
    """
    authorised = ""
    if isinstance(context, dict):
        authorised = str(context.get("authorised_effect_scope") or "").strip()
    if not authorised:
        return None
    try:
        from core.skills.action_scope import (
            action_effect_scope,
            action_within_scope,
            declared_action_name,
            resolve_skill_target,
        )

        meta = (getattr(engine, "skills", None) or {}).get(tool_name)
        target = resolve_skill_target(meta)
        skill_scope = str(getattr(meta, "effect_scope", "") or "unknown")
        # Not tool_args["action"]: a skill names its action field whatever
        # suits it, and an omitted optional field means that field's default.
        # Reading only a literal "action" scoped every http_request as its
        # worst method and refused a GET the turn was entitled to.
        action = declared_action_name(target, tool_args)
        if action_within_scope(target, action, skill_scope, authorised):
            return None
        needed = action_effect_scope(target, action, skill_scope)
    except (AttributeError, ImportError, TypeError, ValueError):
        return None
    return {
        "ok": False,
        "status": "refused",
        "error": (
            f"'{tool_name}' with action '{action or 'unspecified'}' needs "
            f"{needed} authority; this turn has {authorised}. Ask for it "
            "explicitly, or use an action that stays within scope."
        ),
        "engine": "capability_engine",
    }


def _serialize_tool_result_for_model(
    tool_name: str,
    raw_result: Any,
    *,
    limit: int = 4000,
) -> str:
    """Serialize the exact bounded evidence contract fed to the model."""

    model_result = raw_result
    if tool_name == "code_repl":
        from core.skills.code_repl import serialize_code_repl_model_result

        if not isinstance(raw_result, Mapping):
            raw_result = {
                "ok": False,
                "status": "error",
                "error": (
                    "code_repl returned a non-mapping result "
                    f"({type(raw_result).__name__})"
                ),
                "engine": "capability_engine",
            }
        return serialize_code_repl_model_result(raw_result, limit=limit)
    if isinstance(model_result, Mapping):
        serialized = json.dumps(
            model_result,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        serialized = str(model_result)
    return _truncate_tool_result(serialized, limit=limit)


def _benchmark_run_context_active() -> bool:
    """Whether THIS PROCESS was launched as a benchmark run.

    Distinct from a request that says it is a baseline. The runtime profile
    is set by how the process started — the same signal state_ownership uses
    to give a bench run its own state root — and a request cannot change it.

    Fails CLOSED: when the profile cannot be read, this returns False, so an
    unreadable environment produces a recorded degradation rather than a
    silently excused one.
    """
    try:
        from core.runtime.state_ownership import RuntimeProfile, runtime_profile

        return runtime_profile() is RuntimeProfile.BENCH
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _proof_run_requested(origin: Any) -> bool:
    """Is a proof run in progress, judged WITHOUT the proof-policy module?

    Used only on the path where importing ``core.runtime.proof_policy`` failed:
    enforcement cannot consult the policy it could not load, so it falls back
    to the environment signals the policy itself is configured from and fails
    closed when either says a proof run is active.
    """
    if str(origin or "").strip().lower().startswith("proof"):
        return True
    for name in ("AURA_PROOF_RUN", "AURA_PROOF_MODEL_TIER", "AURA_PROOF_HEADLESS"):
        value = str(os.environ.get(name, "") or "").strip().lower()
        if value and value not in {"0", "false", "off", "no", "none"}:
            return True
    return False


def _scavenge_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default
    # Infinity would silently disable the citizenship unload forever.
    return value if math.isfinite(value) and value > 0.0 else default


async def scavenge_idle_model_vram(
    *,
    pressure_idle_s: float | None = None,
    hard_idle_s: float | None = None,
) -> dict[str, Any]:
    """Reclaim unified memory by unloading idle local model lanes.

    Iterates every live MLX lane and unloads the model when it has been safely
    idle (see ``MLXLocalClient.maybe_unload_idle``). Disabled when
    ``AURA_VRAM_SCAVENGER=0``. Thresholds are env-tunable
    (``AURA_VRAM_SCAVENGE_PRESSURE_IDLE_S`` default 90s,
    ``AURA_VRAM_SCAVENGE_HARD_IDLE_S`` default 900s). Safe to call on a periodic
    maintenance tick; it never touches a busy lane and respawn is transparent.
    """
    if os.environ.get("AURA_VRAM_SCAVENGER", "1").strip().lower() in {"0", "false", "no", "off"}:
        return {"enabled": False, "unloaded": 0, "lanes": []}

    if pressure_idle_s is None:
        pressure_idle_s = _scavenge_env_float("AURA_VRAM_SCAVENGE_PRESSURE_IDLE_S", 90.0)
    if hard_idle_s is None:
        hard_idle_s = _scavenge_env_float("AURA_VRAM_SCAVENGE_HARD_IDLE_S", 900.0)

    results: list[dict[str, Any]] = []
    unloaded = 0
    for path, client in _clients_snapshot():
        lane_label = os.path.basename(str(path or "")) or "unknown"
        unload = getattr(client, "maybe_unload_idle", None)
        if unload is None:
            continue
        try:
            outcome = await unload(pressure_idle_s=pressure_idle_s, hard_idle_s=hard_idle_s)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            logger.debug("Idle VRAM scavenge skipped a lane: %s", exc)
            # Failed lanes must be visible in the report — hiding them made
            # repeated reclaim failures undiagnosable from telemetry.
            results.append(
                {
                    "lane": lane_label,
                    "unloaded": False,
                    "reason": f"scavenge_error:{type(exc).__name__}",
                }
            )
            continue
        entry = dict(outcome) if isinstance(outcome, dict) else {"unloaded": bool(outcome)}
        entry.setdefault("lane", lane_label)
        if entry.get("unloaded"):
            unloaded += 1
        results.append(entry)
    return {"enabled": True, "unloaded": unloaded, "lanes": results}


# Import after the client registry and snapshot API exist. Registration is the
# invariant module's only import-time effect.
from core.brain.llm import (  # noqa: E402,F401
    model_runtime_invariants as _model_runtime_invariants,
)
