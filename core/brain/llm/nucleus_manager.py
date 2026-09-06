from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# Absolute ceiling for a single generation so a malformed or hostile
# max_tokens cannot monopolize the in-process GPU lane.
_MAX_GENERATION_TOKENS = 8192


def _bounded_max_tokens(value: Any, default: int) -> int:
    """Clamp a caller-supplied max_tokens to a sane positive range."""
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return default
    if tokens <= 0:
        return default
    return min(_MAX_GENERATION_TOKENS, tokens)

import os

from core.runtime.errors import Severity, record_degradation
from core.runtime.lockdep import checked_async_lock, checked_lock
from core.runtime.task_ownership import fire_and_forget
from core.utils.exceptions import capture_and_log
from core.utils.task_tracker import get_task_tracker

from .provider import LLMProvider

logger = logging.getLogger("LLM.Nucleus")

_CONSTITUTIVE_ORIGINS = frozenset({
    "constitutive_expression",
    "drive_controller",
    "affect_engine",
    "autonomy_guardian",
    "sensory_motor_cortex",
    "pulse_manager",
    "body_monitor",
    "subsystem_audit",
    "health_monitor",
    "agency_core",
})
_NUCLEUS_MODEL_TYPES = frozenset({"brainstem", "cortex"})


#: Sequences that end a stream. ChatML role markers are the important ones: a
#: model that starts writing the next turn was streaming it to the consumer as
#: its own answer, because nothing looked.
_STREAM_STOP_SEQUENCES = (
    "<|im_end|>",
    "<|im_start|>",
    "\nuser:",
    "\nUser:",
    "\nsystem:",
    "\nSystem:",
)

#: How many chunks may sit unread before the producer waits. An unbounded queue
#: let a fast model outrun a slow consumer and retain every chunk it had ever
#: produced.
_STREAM_QUEUE_HIGH_WATER = 64
_STREAM_BACKPRESSURE_SLEEP_S = 0.01

#: Longest assistant prefill a caller may supply. It is an opening, not an
#: answer, and it was previously unbounded.
_MAX_PREFILL_CHARS = 2000

#: How long stop_listener waits for the listener to finish cancelling before
#: releasing the handle anyway. The task is cancelled either way; this bounds
#: how long shutdown blocks on it.
_LISTENER_STOP_TIMEOUT_S = 5.0

#: How long a cancelled load waits for the owned model thread to give up GPU
#: ownership before the lane lease is released regardless. Bounded because
#: shutdown must finish; recorded because releasing early is a real risk and
#: not one to take silently.
_CANCELLED_LOAD_DRAIN_S = 10.0

#: Wall-clock bounds on one generation. The GPU sentinel bounds how long a call
#: waits to START; nothing bounded how long it could then run.
_DEFAULT_GENERATION_BUDGET_S = 180.0
_MIN_GENERATION_BUDGET_S = 5.0
_MAX_GENERATION_BUDGET_S = 900.0


def _await_sentinel_idle(timeout_s: float) -> bool:
    """Wait for the GPU sentinel to be free, or report that it was not.

    Runs off the event loop. Returns True when the sentinel could be taken and
    immediately given back, which is the observable that the previous owner has
    actually finished — as opposed to assuming it, which is what releasing the
    lane straight after a cancellation did.
    """
    try:
        from core.utils.gpu_sentinel import get_gpu_sentinel

        sentinel = get_gpu_sentinel()
        if not sentinel.acquire(timeout=timeout_s):
            return False
        sentinel.release()
        return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _readiness_probe(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Whether this model and tokenizer can actually produce a token together.

    Readiness required only that the path existed and ``load`` returned two
    objects. Nothing checked that the tokenizer belongs to the model or that
    the pair can do anything, and every caller reads ``loaded`` as readiness —
    including the fallback that decides whether to swap lanes, which would
    happily swap TO a lane that could not answer either.

    The check is a round trip through the tokenizer plus a vocabulary-size
    agreement between the two objects. It is cheap, it runs off the loop, and
    it is a measurement rather than an assumption.
    """
    if model is None or tokenizer is None:
        return {"ready": False, "reason": "load returned no model or tokenizer"}
    try:
        encoded = tokenizer.encode("ready")
    except Exception as exc:  # noqa: BLE001 — any failure here means not ready
        return {"ready": False, "reason": f"tokenizer encode failed: {type(exc).__name__}"}
    if not encoded:
        return {"ready": False, "reason": "tokenizer produced no tokens"}

    tokenizer_vocab = 0
    for attribute in ("vocab_size", "n_vocab"):
        candidate = getattr(tokenizer, attribute, None)
        if isinstance(candidate, int) and candidate > 0:
            tokenizer_vocab = candidate
            break

    model_vocab = 0
    args = getattr(model, "args", None)
    raw = getattr(args, "vocab_size", None)
    if isinstance(raw, int) and raw > 0:
        model_vocab = raw

    if tokenizer_vocab and model_vocab and tokenizer_vocab > model_vocab:
        # A tokenizer that can emit ids the model has no embedding for is the
        # mismatch that produces garbage rather than an error.
        return {
            "ready": False,
            "reason": f"tokenizer vocab {tokenizer_vocab} exceeds model vocab {model_vocab}",
        }

    return {
        "ready": True,
        "reason": f"encoded {len(encoded)} tokens; vocab {tokenizer_vocab or 'unstated'}",
        "tokenizer_vocab": tokenizer_vocab,
        "model_vocab": model_vocab,
    }


def _adapter_compatibility(adapter_dir: str, base_model_path: str) -> dict[str, Any]:
    """Whether this adapter may be attached to this base model.

    The gate was the existence of ``adapter_config.json``. A filename is not a
    provenance: it says nothing about which base model the adapter was trained
    against, whether its weights are on disk, or whether the format is one this
    loader understands, so a stale or mismatched adapter attached to the live
    Cortex for free.
    """
    directory = Path(adapter_dir or "")
    config_path = directory / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        return {"compatible": False, "reason": f"unreadable adapter_config: {type(exc).__name__}"}
    if not isinstance(config, dict):
        return {"compatible": False, "reason": "adapter_config is not a mapping"}

    weights = [
        candidate
        for candidate in ("adapters.safetensors", "adapter_model.safetensors", "adapters.npz")
        if (directory / candidate).exists()
    ]
    if not weights:
        return {"compatible": False, "reason": "no adapter weights beside the config"}

    declared_base = str(
        config.get("base_model_name_or_path") or config.get("model") or ""
    ).strip()
    if declared_base:
        # Compared on the leaf name: the config records whatever path the
        # training host used, which is not this host's path, but the model
        # directory name is the identity both sides share.
        declared_leaf = Path(declared_base).name.lower()
        actual_leaf = Path(str(base_model_path or "")).name.lower()
        if declared_leaf and actual_leaf and declared_leaf != actual_leaf:
            return {
                "compatible": False,
                "reason": f"adapter trained against {declared_leaf!r}, lane is {actual_leaf!r}",
            }
        return {"compatible": True, "reason": f"base {declared_leaf or 'unstated'}, weights {weights[0]}"}

    return {
        "compatible": False,
        "reason": "adapter_config names no base model, so compatibility cannot be established",
    }


def _bounded_prefill(value: Any) -> str:
    """A caller-supplied assistant prefill, made safe to interpolate.

    ``prefill`` went from kwargs straight into the ChatML assistant turn with
    no type check and no length bound: a non-string raised inside prompt
    formatting, and a large one consumed the window before the model saw the
    user's message. A prefill is an OPENING, so it is bounded like one.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return text[:_MAX_PREFILL_CHARS]


def _accepted_generation_budget_s(requested: Any) -> float:
    """Wall-clock bound on one generation, clamped to what is servable."""
    try:
        seconds = float(requested)
    except (TypeError, ValueError):
        return _DEFAULT_GENERATION_BUDGET_S
    if not math.isfinite(seconds) or seconds <= 0.0:
        return _DEFAULT_GENERATION_BUDGET_S
    return max(_MIN_GENERATION_BUDGET_S, min(seconds, _MAX_GENERATION_BUDGET_S))


def _stream_stop_index(emitted: str) -> int | None:
    """Where the stream should stop, if a stop sequence has appeared."""
    cuts = [emitted.find(seq) for seq in _STREAM_STOP_SEQUENCES]
    hits = [c for c in cuts if c >= 0]
    return min(hits) if hits else None


def _with_requested_lane(kwargs: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Carry the caller's requested model into lane selection.

    ``generate_stream``, ``generate_text`` and ``generate_json`` all accepted a
    ``model`` parameter — the provider interface advertises that control — and
    all three dropped it on the floor. A caller could ask for a lane, be served
    by the other one, and have no way to tell.

    A recognised lane name is honoured. An unrecognised one is refused rather
    than silently ignored, because "I could not give you that" and "I gave you
    that" must not look the same.
    """
    merged = dict(kwargs or {})
    requested = str(model or "").strip().lower()
    if not requested:
        return merged
    if requested in _NUCLEUS_MODEL_TYPES:
        merged["requested_model_type"] = requested
        return merged
    _record_nucleus_degradation(
        ValueError(f"unknown model {model!r} requested"),
        action="served the request from lane selection rather than the unknown requested model",
        severity="warning",
        extra={"requested_model": str(model)[:80]},
    )
    merged["requested_model_unavailable"] = str(model)[:80]
    return merged


def _internal_execution_scope() -> bool:
    """Whether this call is the runtime's own governed work.

    A governed scope is entered on the stack for the duration of the work. A
    caller composing kwargs cannot arrange to be inside one, which is exactly
    the property an origin string does not have.
    """
    try:
        from core.governance_context import is_governed

        return bool(is_governed())
    except (ImportError, AttributeError, RuntimeError):
        return False
_NUCLEUS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    OSError,
    TypeError,
    ValueError,
)


def _empty_model_entry() -> dict[str, Any]:
    return {
        "model": None,
        "tokenizer": None,
        "loaded": False,
        "cache": None,
        "last_error": None,
        "lane_lease": None,
    }


def _record_nucleus_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "nucleus_manager",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


class NucleusManager(LLMProvider):
    """Aura's Dual-Nucleus Internal Brain.
    
    Manages two local MLX-optimized models:
    1. Brainstem (Tiny): Perpetual, low-latency reflection for background tasks.
    2. Cortex (Medium): Conversational intelligence, loaded on demand.
    
    Optimized for M5 Pro with KV-Caching and GPU Sentinel.
    """
    
    def __init__(self, **kwargs):
        from .model_registry import (
            BRAINSTEM_MODEL,
            get_active_model,
            get_brainstem_path,
            get_model_path,
            resolve_personality_adapter,
        )
        
        self.brainstem_repo = BRAINSTEM_MODEL
        self.cortex_repo = get_active_model()
        
        self.brainstem_path = str(get_brainstem_path())
        self.cortex_path = str(get_model_path())
        self._adapter_dir = resolve_personality_adapter(self.cortex_path, backend="mlx") or ""
        
        self.models = {name: _empty_model_entry() for name in _NUCLEUS_MODEL_TYPES}
        self._model_lifecycle_locks = {
            name: checked_lock("core.brain.llm.nucleus_manager") for name in _NUCLEUS_MODEL_TYPES
        }
        self._anchor_text = None 
        self._refresh_threshold = 2048 
        self._tokens_seen = 0
        self._listener_task = None
        self._running = True
        #: Serializes listener start. Two concurrent ensure_listener_started
        #: calls could both observe None and both create a subscription, so the
        #: bus had two readers for one manager and the second task was
        #: unreachable — nothing held a handle to it.
        self._listener_lock = checked_async_lock("core.brain.llm.nucleus_manager.1")
        self._listener_subscription = None
        # Defer event subscription to avoid create_task in __init__
        try:
            from core.event_bus import get_event_bus
            self.bus = get_event_bus()
        except (ImportError, AttributeError, RuntimeError):
            self.bus = None

    async def ensure_listener_started(self) -> bool:
        """Start the event listener if not already running.

        Two callers could both observe ``_listener_task is None`` and both
        create a subscription, leaving the bus with two readers for one manager
        and no handle on the second. And nothing ever cleared the handle, so a
        listener that died stayed "started" forever: every later call saw a
        non-None task and returned, and update handling never came back.
        """
        if self.bus is None:
            return False
        async with self._listener_lock:
            existing = self._listener_task
            if existing is not None and not existing.done():
                return True
            self._running = True
            task = get_task_tracker().create_task(self._listen_for_updates())
            if task is None:
                return False

            def _clear(finished: Any) -> None:
                # The handle is what ensure_listener_started reads to decide
                # whether a listener exists. A dead task holding it is the
                # reason update handling could never be restarted.
                if self._listener_task is finished:
                    self._listener_task = None
                try:
                    error = finished.exception()
                except (asyncio.CancelledError, RuntimeError):
                    return
                if error is not None:
                    _record_nucleus_degradation(
                        error,
                        severity="warning",
                        action="cleared the listener handle so a later caller can restart it",
                        extra={"phase": "listener_exit"},
                    )

            task.add_done_callback(_clear)
            self._listener_task = task
            return True

    async def stop_listener(self) -> None:
        """Cancel the listener and release its subscription.

        There was no shutdown path at all: ``_running`` started true, was never
        cleared, and no caller could cancel, unsubscribe or join. A manager
        being replaced left its listener reading the bus forever.
        """
        self._running = False
        async with self._listener_lock:
            task = self._listener_task
            self._listener_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_LISTENER_STOP_TIMEOUT_S)
            except (TimeoutError, asyncio.CancelledError):
                pass  # the handle is released either way; the task is cancelled
            except _NUCLEUS_RECOVERABLE_ERRORS as exc:
                _record_nucleus_degradation(
                    exc,
                    severity="warning",
                    action="released the listener handle after an error during shutdown",
                    extra={"phase": "listener_stop"},
                )
        subscription = self._listener_subscription
        self._listener_subscription = None
        unsubscribe = getattr(subscription, "unsubscribe", None) or getattr(
            subscription, "close", None
        )
        if callable(unsubscribe):
            try:
                result = unsubscribe()
                if inspect.isawaitable(result):
                    await result
            except _NUCLEUS_RECOVERABLE_ERRORS as exc:
                _record_nucleus_degradation(
                    exc,
                    severity="warning",
                    action="stopped the listener without releasing its bus subscription",
                    extra={"phase": "listener_unsubscribe"},
                )


    async def _adopt_promoted_cortex(self, data: dict[str, Any]) -> None:
        """Rebind the Cortex lane to a promoted artifact, or do nothing.

        CP126 2a2791b1 + 402a99f0, which are one defect seen from two sides.

        The listener acted on ``data.status == "success"`` alone and
        immediately unloaded the live Cortex — roughly 20GB of resident
        weights — with no check that anything had actually been produced.
        The event bus carries no publisher identity, so anything able to
        publish could evict her mind at will, and a malformed or duplicated
        event did it for free.

        Then the reload used ``self.cortex_path``, captured in ``__init__``.
        So the newly fused model was never bound: the lane unloaded and
        reloaded the SAME weights, and every optimization run was a no-op
        that cost a full reload.

        The artifact is the authority. An event claiming success without a
        loadable ``fused_model`` on disk is not acted on — that check needs
        no principal, cannot be forged by a publisher, and makes the
        pointless unload impossible. When the artifact IS there, the lane is
        rebound to it before unloading, so the reload picks up what was
        actually promoted.
        """
        fused = str(data.get("fused_model") or "").strip()
        if not fused:
            _record_nucleus_degradation(
                ValueError("optimizer success carried no fused_model"),
                severity="warning",
                action=(
                    "ignored an optimizer completion with no promoted artifact; "
                    "unloading the resident Cortex for nothing is the failure "
                    "this refuses"
                ),
            )
            return

        artifact = Path(fused)
        if not artifact.exists():
            _record_nucleus_degradation(
                FileNotFoundError(f"promoted cortex artifact missing: {fused}"),
                severity="warning",
                action="kept the resident Cortex; the promoted artifact does not exist",
            )
            return

        previous = self.cortex_path
        # Bind BEFORE unloading: the reload reads this, and rebinding after
        # would reproduce the original bug on the very next load.
        self.cortex_path = str(artifact)
        self._adapter_dir = ""
        logger.info(
            "🧠 [NUCLEUS] Promoted Cortex artifact adopted (%s -> %s); unloading for reload.",
            os.path.basename(previous or "unknown"),
            artifact.name,
        )
        await self._unload_model_entry("cortex", reason="optimizer_promoted_artifact")

    async def _listen_for_updates(self):
        """Listens for LoRA optimization successes and flags for reload."""
        if not self.bus:
            return
        sub = await self.bus.subscribe("core/optimizer/completed")
        self._listener_subscription = sub
        while self._running:
            try:
                _, _, event = await sub.get()
                data = event.get("data", {}) if isinstance(event, dict) else {}
                if isinstance(data, dict) and data.get("status") == "success":
                    await self._adopt_promoted_cortex(data)
            except asyncio.CancelledError:
                raise
            except (OSError, ConnectionError, TimeoutError):
                await asyncio.sleep(1)
            except (AttributeError, TypeError, ValueError, KeyError, RuntimeError, LookupError) as exc:
                # A malformed event or a transient unload error must NOT kill
                # the listener permanently — record it and keep handling.
                _record_nucleus_degradation(
                    exc,
                    severity="warning",
                    action="skipped a malformed optimizer event and kept the nucleus reload listener alive",
                )
                await asyncio.sleep(1)
        
    def _select_model_type(self, origin: str) -> str:
        """Which lane serves this request.

        A plain origin string chose it. `origin` arrives in kwargs from
        whoever composed the call, so any caller could claim `health_monitor`
        or `agency_core` and be routed with constitutive semantics — the label
        was doing authorization work that a label cannot do.

        A constitutive origin is honoured only from inside a governed scope,
        which the runtime enters on the stack for the duration of its own work
        and a caller cannot arrange by passing a keyword. An unbacked claim
        routes to Cortex, like any other request, and is recorded.
        """
        name = str(origin or "").strip()
        if name not in _CONSTITUTIVE_ORIGINS:
            return "cortex"
        if _internal_execution_scope():
            return "brainstem"
        _record_nucleus_degradation(
            RuntimeError(f"constitutive origin {name!r} claimed outside a governed scope"),
            action="routed the request to the ordinary lane",
            severity="warning",
            extra={"origin": name},
        )
        return "cortex"

    def _ensure_model_entry(self, name: str) -> dict[str, Any]:
        entry = self.models.setdefault(name, _empty_model_entry())
        entry.setdefault("cache", None)
        entry.setdefault("last_error", None)
        return entry

    def check_health(self) -> bool:
        """Is at least one local model actually loaded and error-free?

        `LLMProvider.check_health` used to return True for anything that did
        not override it, and this class did not. So the primary local lane
        answered "healthy" with no model loaded, no MLX available and no
        successful call ever made — and `FallbackLLMClient` selects the lane
        to use from precisely this answer, which kept a dead lane at the
        front of the chain instead of falling through to one that works.

        Deliberately inference-free: this is called on selection paths and
        during audits, so it inspects the loaded state rather than spending
        a generation to find out. A model marked loaded whose last operation
        raised is not healthy — that is the shape a wedged worker takes.
        """
        entries = getattr(self, "models", None)
        if not isinstance(entries, dict) or not entries:
            return False
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            if not entry.get("loaded"):
                continue
            if entry.get("model") is None:
                # "loaded" with nothing behind it is the state a failed load
                # leaves; treating it as healthy is how it stayed selected.
                continue
            if entry.get("last_error"):
                logger.debug(
                    "Nucleus model %s is loaded but carries last_error=%s",
                    name, entry.get("last_error"),
                )
                continue
            return True
        return False

    def _model_path_for(self, name: str) -> str:
        return self.brainstem_path if name == "brainstem" else self.cortex_path

    @contextlib.asynccontextmanager
    async def _model_lifecycle_context(self, name: str) -> AsyncIterator[None]:
        lock = self._model_lifecycle_locks.setdefault(name, checked_lock("core.brain.llm.nucleus_manager.2"))
        while not lock.acquire(blocking=False):  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            lock.release()

    @contextlib.contextmanager
    def _model_thread_context(self, name: str):
        """Pin one loaded model and its lane for a complete worker operation."""

        lock = self._model_lifecycle_locks.setdefault(name, checked_lock("core.brain.llm.nucleus_manager.3"))
        lock.acquire()
        try:
            entry = self._ensure_model_entry(name)
            model = entry.get("model")
            tokenizer = entry.get("tokenizer")
            if not entry.get("loaded") or model is None or tokenizer is None:
                raise RuntimeError(f"nucleus_model_not_owned:{name}")
            yield entry, model, tokenizer
        finally:
            lock.release()

    async def load_model(self, name: str) -> bool:
        async with self._model_lifecycle_context(name):
            return await self._load_model_unlocked(name)

    async def _load_model_unlocked(self, name: str) -> bool:
        """Lazy load a specific internal model, including LoRA adapters if present."""
        # Ensure event listener is started from async context
        await self.ensure_listener_started()
        logger.debug("Attempting to load: %s", name)
        entry = self._ensure_model_entry(name)
        if name not in _NUCLEUS_MODEL_TYPES:
            error = ValueError(f"unknown nucleus model lane: {name}")
            entry["last_error"] = str(error)
            _record_nucleus_degradation(
                error,
                severity="warning",
                action="refused unknown nucleus lane and left existing model state unchanged",
                extra={"model": name},
            )
            return False
        if entry.get("loaded"): # Use .get for safety if brainstem is removed from models dict
            logger.debug("%s already loaded.", name)
            return True

        path = self._model_path_for(name)
        logger.debug("Path for %s: %s", name, path)
        path_exists = await asyncio.to_thread(Path(path).exists)
        if not path_exists:
            logger.debug("PATH MISSING: %s", path)
            logger.warning("⚠️ Model path missing: %s. Background fetch may still be running...", path)
            error = FileNotFoundError(path)
            entry.update({"loaded": False, "last_error": str(error), "cache": None})
            _record_nucleus_degradation(
                error,
                severity="warning",
                action=(
                    "marked the nucleus lane unavailable so callers can try the "
                    "alternate lane or return a deterministic offline response"
                ),
                extra={"model": name, "path": path},
            )
            return False

        adapter_path = None
        if name == "cortex":
            adapter_config = Path(self._adapter_dir) / "adapter_config.json"
            adapter_config_exists = (
                bool(self._adapter_dir)
                and await asyncio.to_thread(adapter_config.exists)
            )
            if adapter_config_exists:
                # The existence of adapter_config.json was the whole gate. It
                # said nothing about which base model the adapter was trained
                # against, whether its weights are present, or whether the
                # format is one this loader understands — so a stale or
                # mismatched adapter attached to the live Cortex on the
                # strength of a filename.
                verdict = await asyncio.to_thread(
                    _adapter_compatibility, self._adapter_dir, path
                )
                if verdict["compatible"]:
                    adapter_path = self._adapter_dir
                    logger.info(
                        "🧠 [NUCLEUS] Attaching LoRA adapter for Cortex: %s (%s)",
                        adapter_path,
                        verdict["reason"],
                    )
                else:
                    _record_nucleus_degradation(
                        RuntimeError(f"adapter refused: {verdict['reason']}"),
                        severity="warning",
                        action="loaded the base Cortex without the adapter",
                        extra={"model": name, "adapter_dir": str(self._adapter_dir)[:200]},
                    )

        try:
            from mlx_lm import load

            from core.runtime.model_lane_control import (
                acquire_in_process_model_lane,
                run_owned_model_thread_call,
            )
            from core.utils.gpu_sentinel import get_gpu_sentinel

            lane_lease = await acquire_in_process_model_lane(
                owner_id=f"nucleus:{id(self)}:{name}",
                model_path=path,
                purpose="serve",
                priority=10 if name == "cortex" else 50,
                preemptible=False,
                metadata={"provider": "nucleus_manager", "model_type": name},
            )
            entry["lane_lease"] = lane_lease
            sentinel = get_gpu_sentinel()
            
            logger.info("🧠 [NUCLEUS] Loading %s from disk (Adapter: %s)...", name.upper(), adapter_path)
            
            def _load_locked():
                acquired = sentinel.acquire(timeout=90) # Loading is slow, give it more time
                if not acquired:
                    raise TimeoutError("NUCLEUS GPU Sentinel timeout during LOAD")
                try:
                    if adapter_path:
                        return load(path, adapter_path=adapter_path)
                    else:
                        return load(path)
                finally:
                    sentinel.release()

            # Use asyncio.to_thread as load is CPU/IO intensive
            model, tokenizer = await run_owned_model_thread_call(
                _load_locked,
                operation_name=f"nucleus-{name}-load",
            )
                
            # "loaded" meant load() returned two objects. It did not mean the
            # tokenizer belongs to the model, that the pair can produce a
            # token, or that anything about this lane works — and every caller
            # reads `loaded` as readiness, including the fallback that decides
            # whether to swap lanes.
            readiness = await asyncio.to_thread(_readiness_probe, model, tokenizer)
            if not readiness["ready"]:
                _record_nucleus_degradation(
                    RuntimeError(f"nucleus {name} failed its readiness probe: {readiness['reason']}"),
                    action="left the lane marked unavailable rather than declaring it ready",
                    extra={"model": name, "reason": readiness["reason"]},
                )
                self.models[name].update(
                    {"loaded": False, "last_error": readiness["reason"], "cache": None}
                )
                return False

            self.models[name]["model"] = model
            self.models[name]["tokenizer"] = tokenizer
            self.models[name]["loaded"] = True
            self.models[name]["last_error"] = None
            self.models[name]["readiness"] = readiness

            if name == "cortex":
                # Set a default anchor text when cortex is loaded
                self._anchor_text = (
                    "You are Aura Luna — Aura for short. Speak with direct first-person "
                    "continuity, curiosity, and emotional presence while keeping evidence "
                    "claims grounded. Do not claim aliveness, consciousness, or sealed "
                    "governance from labels alone."
                )
                self._tokens_seen = 0 # Reset token count for new conversation/load
                logger.debug("🧠 [NUCLEUS] Cortex loaded. Anchor text set and token counter reset.")

            logger.info(
                "✅ [NUCLEUS] %s ready (%s).", name.upper(), readiness["reason"]
            )
            return True
        except asyncio.CancelledError:
            # The blocking load runs on an owned thread that a cancellation
            # cannot reach: awaiting it is what gets cancelled, not the work.
            # This released the lane immediately, so a NEW load could take the
            # lease and start while the previous thread was still holding GPU
            # sentinel ownership and unwinding — correctness resting on an
            # unstated guarantee from a helper.
            #
            # The lease is released only after the sentinel is observed free,
            # or after a bounded wait with the failure recorded, so the next
            # holder is never handed a lane that is still in use silently.
            lease = entry.get("lane_lease")
            entry["lane_lease"] = None
            entry.update({"loaded": False, "cache": None})
            if lease is not None:
                released_cleanly = await asyncio.to_thread(
                    _await_sentinel_idle, _CANCELLED_LOAD_DRAIN_S
                )
                if not released_cleanly:
                    _record_nucleus_degradation(
                        TimeoutError(
                            "owned model thread still held GPU ownership after load cancellation"
                        ),
                        severity="warning",
                        action=(
                            "released the lane lease anyway; the previous load thread may "
                            "still be unwinding"
                        ),
                        extra={"model": name, "phase": "load_cancelled"},
                    )
                await lease.release(reason="nucleus_load_cancelled")
            raise
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            lease = entry.get("lane_lease")
            entry["lane_lease"] = None
            if lease is not None:
                try:
                    await lease.release(reason="nucleus_load_failed")
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as release_exc:
                    _record_nucleus_degradation(
                        release_exc,
                        severity="warning",
                        action="model load failed and lane lease release also failed",
                        extra={"model": name},
                    )
            entry.update({
                "model": None,
                "tokenizer": None,
                "loaded": False,
                "cache": None,
                "last_error": f"{type(e).__name__}: {e}",
            })
            _record_nucleus_degradation(
                e,
                action=(
                    "marked the failed nucleus lane unavailable, cleared partial "
                    "model state, and left alternate-lane fallback eligible"
                ),
                extra={"model": name, "path": path, "adapter_path": adapter_path},
            )
            logger.error("❌ [NUCLEUS] Load failed for %s: %s", name, e)
            logger.error("Failed to load internal model %s: %s", name, e)
            return False

    @staticmethod
    def _strip_chatml_tokens(text: Any) -> str:
        """Remove ChatML control tokens from untrusted content.

        System/user/prefill text is interpolated directly between ChatML
        markers. An embedded ``<|im_end|>`` or ``<|im_start|>`` could
        prematurely terminate a role or forge a new one (role-token
        injection), so those control tokens are stripped from the content.
        """
        cleaned = str(text or "")
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            cleaned = cleaned.replace(token, "")
        return cleaned

    def last_lane_receipt(self) -> dict[str, Any]:
        """Which model answered the last request, and whether that was the one
        lane selection chose.

        Callers received a plain string either way, so a user request answered
        by Brainstem and constitutive work answered by Cortex were
        indistinguishable from correctly routed ones. This is the seam a caller
        can read to tell.
        """
        return {
            "served_by": self._last_served_lane,
            "substituted": bool(self._last_lane_substituted),
        }

    def _format_prompt(self, prompt: str, system_prompt: str | None = None, prefill: str | None = None) -> str:
        """Formats the prompt using ChatML for Qwen-Instruct models.

        The anchor used to be an ELSE: a caller system prompt replaced it, and
        _apply_anchor only reinjected the grounding text after a shared token
        threshold had been crossed. So an ordinary request carrying any system
        prompt could omit Aura's identity and evidence constraints entirely,
        and how long it stayed omitted depended on a counter shared with every
        other caller.

        The anchor composes WITH a caller's prompt now. A caller may add
        instructions; it may not remove the ones that are not its to remove.
        """
        if system_prompt and self._anchor_text:
            s_msg = f"{self._anchor_text}\n\n{system_prompt}"
        else:
            s_msg = system_prompt or self._anchor_text

        # Strip ChatML control tokens from every interpolated segment so
        # untrusted content cannot break out of its role.
        s_msg = self._strip_chatml_tokens(s_msg)
        safe_prompt = self._strip_chatml_tokens(prompt)
        safe_prefill = self._strip_chatml_tokens(prefill) if prefill else ""

        # Base ChatML structure
        formatted = f"<|im_start|>system\n{s_msg}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{safe_prompt}<|im_end|>\n"
        formatted += f"<|im_start|>assistant\n{safe_prefill}"

        return formatted

    def _apply_anchor(self, prompt: str, system_prompt: str | None = None, model_type: str = "cortex") -> tuple[str, str | None]:
        """
        Manages the semantic anchor. 
        Returns (modified_prompt, modified_system_prompt).
        """
        if model_type != "cortex" or not self._anchor_text:
            return prompt, system_prompt

        tokenizer = self.models["cortex"].get("tokenizer")
        if not tokenizer:
            return prompt, system_prompt

        # Estimate current prompt tokens
        current_prompt_tokens = len(tokenizer.encode(prompt))
        self._tokens_seen += current_prompt_tokens

        if self._tokens_seen >= self._refresh_threshold:
            logger.debug("🧠 [NUCLEUS] Re-injecting semantic anchor for Cortex (tokens seen: %d).", self._tokens_seen)
            # Re-inject by ensuring the anchor is the system prompt
            actual_system = f"{self._anchor_text}\n\n{system_prompt}" if system_prompt else self._anchor_text
            self._tokens_seen = 0 # Reset after re-injection
            return prompt, actual_system
        
        return prompt, system_prompt

    async def generate_text_async(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """Route to appropriate internal model."""
        origin = kwargs.get("origin", "unknown")
        logger.debug("generate_text_async called with origin: %s", origin)
        model_type = str(
            kwargs.get("requested_model_type") or self._select_model_type(origin)
        )
        logger.debug("Routing to: %s", model_type)

        await self.load_model(model_type)
        if not self.models.get(model_type, {}).get("loaded"): # Use .get for safety
            # Fallback to brainstem if cortex isn't ready or vice-versa.
            #
            # The swap used to happen silently and the alternate lane's text
            # was returned as an ordinary string: user requests answered by
            # Brainstem, constitutive work answered by Cortex, and no caller
            # able to tell which model produced what it received. Which model
            # answered is part of the answer.
            alt_type = "brainstem" if model_type == "cortex" else "cortex"
            await self.load_model(alt_type)
            if self.models.get(alt_type, {}).get("loaded"):
                _record_nucleus_degradation(
                    RuntimeError(f"{model_type} lane unavailable; served by {alt_type}"),
                    action=f"answered from the {alt_type} lane instead of {model_type}",
                    severity="degraded",
                    extra={"requested_model": model_type, "served_by": alt_type, "origin": origin},
                )
                self._last_served_lane = alt_type
                self._last_lane_substituted = True
                model_type = alt_type
            else:
                logger.error("❌ Both Nucleus models failed to load.")
                _record_nucleus_degradation(
                    RuntimeError("both nucleus model lanes are offline"),
                    action=(
                        "returned deterministic offline text instead of blocking "
                        "or hallucinating an internal inference result"
                    ),
                    extra={"requested_model": model_type, "origin": origin},
                )
                return "[NUCLEUS ERROR] Internal inference offline."

        else:
            self._last_served_lane = model_type
            self._last_lane_substituted = False

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            from core.runtime.model_lane_control import run_owned_model_thread_call
            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel

            temp = self._resolve_temperature(kwargs, model_type=model_type, phase="text_generate")
            sampler = make_sampler(temp=temp)
            sentinel = get_gpu_sentinel()
            priority = GPUPriority.REFLEX if model_type == "cortex" else GPUPriority.REFLECTION

            def _generate_locked():
                with self._model_thread_context(model_type) as (_entry, model, tokenizer):
                    acquired = sentinel.acquire(priority=priority, timeout=60)
                    if not acquired:
                        raise TimeoutError("NUCLEUS GPU Sentinel timeout during GENERATE")
                    try:
                        # Semantic Anchor Refresh & Formatting
                        p, s = self._apply_anchor(prompt, system_prompt, model_type)
                        final_prompt = self._format_prompt(
                            p,
                            s,
                            prefill=_bounded_prefill(kwargs.get("prefill")),
                        )

                        return generate(
                            model,
                            tokenizer,
                            prompt=final_prompt,
                            max_tokens=_bounded_max_tokens(kwargs.get("max_tokens"), 512),
                            sampler=sampler,
                            verbose=False,
                        )
                    finally:
                        sentinel.release()

            # After the 60s sentinel acquisition, generation and token
            # iteration had no wall-clock bound at all: a wedged model call
            # held the non-preemptible lane for as long as it took, and the
            # caller waited with it. mlx generation cannot be interrupted
            # mid-call, so this cannot kill the work — what it can do is stop
            # the caller waiting forever, say so, and mark the lane, which is
            # the difference between a slow answer and a runtime that has
            # silently stopped answering.
            budget = _accepted_generation_budget_s(kwargs.get("deadline_s"))
            try:
                response = await asyncio.wait_for(
                    run_owned_model_thread_call(
                        _generate_locked,
                        operation_name=f"nucleus-{model_type}-generate",
                    ),
                    timeout=budget,
                )
            except TimeoutError:
                entry = self._ensure_model_entry(model_type)
                entry["last_error"] = f"generation exceeded {budget:.0f}s"
                _record_nucleus_degradation(
                    TimeoutError(f"nucleus {model_type} generation exceeded {budget:.0f}s"),
                    action=(
                        "stopped waiting on a generation that had not returned; the model "
                        "thread is not preemptible and may still be unwinding"
                    ),
                    severity="degraded",
                    extra={"model": model_type, "origin": origin, "phase": "text_generate"},
                )
                return "[NUCLEUS ERROR] Generation exceeded its time budget."
            return response.strip()
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            entry = self._ensure_model_entry(model_type)
            entry["cache"] = None
            entry["last_error"] = f"{type(e).__name__}: {e}"
            if isinstance(e, (ImportError, AttributeError)):
                entry["loaded"] = False
            _record_nucleus_degradation(
                e,
                action=(
                    "returned explicit nucleus error text, cleared volatile cache, "
                    "and left the lane marked for retry"
                ),
                extra={"model": model_type, "origin": origin, "phase": "text_generate"},
            )
            logger.error("Nucleus inference failed: %s", e)
            return f"Nucleus Error: {str(e)}"

    async def generate_stream_async(self, prompt: str, system_prompt: str | None = None, **kwargs):
        """Streaming version of generate_text_async."""
        origin = kwargs.get("origin", "unknown")
        model_type = str(
            kwargs.get("requested_model_type") or self._select_model_type(origin)
        )

        await self.load_model(model_type)
        if not self._ensure_model_entry(model_type).get("loaded"):
            model_type = "brainstem" if model_type == "cortex" else "cortex"
            await self.load_model(model_type)
            if not self._ensure_model_entry(model_type).get("loaded"):
                _record_nucleus_degradation(
                    RuntimeError("both nucleus model lanes are offline"),
                    action=(
                        "ended stream with deterministic offline marker after both "
                        "nucleus lanes failed availability checks"
                    ),
                    extra={"requested_model": model_type, "origin": origin, "phase": "stream_load"},
                )
                yield "[NUCLEUS ERROR] Internal inference offline."
                return

        try:
            import mlx.core as mx
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step

            temp = self._resolve_temperature(kwargs, model_type=model_type, phase="stream_generate")
            sampler = make_sampler(temp=temp)
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            _record_nucleus_degradation(
                e,
                action=(
                    "ended stream with explicit nucleus error before token generation "
                    "and preserved caller control flow"
                ),
                extra={"model": model_type, "origin": origin, "phase": "stream_prepare"},
            )
            yield f"[NUCLEUS ERROR] {e}"
            return

        max_tokens = _bounded_max_tokens(kwargs.get("max_tokens"), 1024)
        stop_event = threading.Event()
        
        # Generator for streaming
        def _stream_gen():
            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel
            with self._model_thread_context(model_type) as (model_entry, model, tokenizer):
                p, s = self._apply_anchor(prompt, system_prompt, model_type)
                full_prompt = self._format_prompt(
                    p, s, prefill=_bounded_prefill(kwargs.get("prefill"))
                )
                tokens = mx.array(tokenizer.encode(full_prompt))

                # A cache belongs to this exact model instance and operation.
                if model_entry.get("cache") is not None:
                    model_entry["cache"] = None
                    try:
                        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                            mx.metal.clear_cache()
                        else:
                            mx.clear_cache()
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        _record_nucleus_degradation(
                            e,
                            severity="warning",
                            action=(
                                "continued streaming with stale-cache reference cleared "
                                "after MLX cache cleanup failed"
                            ),
                            extra={"model": model_type, "phase": "stream_cache_clear"},
                        )

                sentinel = get_gpu_sentinel()
                priority = (
                    GPUPriority.REFLEX
                    if model_type == "cortex"
                    else GPUPriority.REFLECTION
                )
                acquired = sentinel.acquire(priority=priority, timeout=60)
                if not acquired:
                    _record_nucleus_degradation(
                        TimeoutError("NUCLEUS GPU Sentinel timeout during STREAM"),
                        action="ended stream with explicit GPU timeout marker",
                        extra={"model": model_type, "phase": "stream_gpu_acquire"},
                    )
                    yield "[NUCLEUS ERROR] GPU Sentinel timeout during STREAM"
                    return

                try:
                    cache = model_entry.get("cache")
                    emitted = ""
                    for response in generate_step(
                        model,
                        tokenizer,
                        tokens,
                        sampler=sampler,
                        cache=cache,
                    ):
                        if stop_event.is_set() or response.token >= tokenizer.eos_token_id:
                            break

                        if (
                            priority == GPUPriority.REFLECTION
                            and sentinel.should_yield()
                        ):
                            logger.warning(
                                "🧠 [NUCLEUS] Pre-empted by REFLEX task. Yielding GPU."
                            )
                            yield "... [Pausing for sensory reflex] ..."
                            break

                        # The stream yielded raw text until EOS with no stop
                        # sequences and no role-continuation check, so a model
                        # that began writing the next turn ("<|im_start|>user")
                        # streamed that to the consumer as its own answer.
                        emitted += response.text
                        cut = _stream_stop_index(emitted)
                        if cut is not None:
                            tail = emitted[:cut][len(emitted) - len(response.text) :]
                            if tail:
                                yield tail
                            break

                        yield response.text
                        if response.count >= max_tokens:
                            yield "\n\n[MAX_TOKENS_REACHED]"
                            break

                    model_entry["cache"] = cache
                finally:
                    sentinel.release()

        # ── [STABILITY v52] Non-blocking Streaming Bridge ───────────────
        # MLX generate_step is a blocking CPU/GPU operation. We must offload
        # it to a thread and pipe tokens back via a Queue to avoid 
        # stalling the main event loop and motor reflexes.
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _post(item: Any) -> None:
            """Hand one item to the consumer, or give up on the consumer.

            call_soon_threadsafe raises once the loop is closed. That happened
            inside the sentinel post as readily as inside a chunk post, so a
            consumer that went away could leave the worker unable to say it had
            finished — and the next consumer of this queue blocked on get().
            """
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                stop_event.set()

        def _thread_worker():
            # The sentinel is posted in `finally`. The handler used to catch
            # four exception classes, so an OSError, an ImportError, a
            # TimeoutError, a cancellation or a closed loop exited the thread
            # without enqueueing None — and the consumer's `await queue.get()`
            # waited forever for a producer that had already died.
            try:
                for chunk in _stream_gen():
                    if stop_event.is_set():
                        break
                    _post(chunk)
                    # Backpressure. put_nowait into an unbounded queue let a
                    # fast model outrun a slow consumer and retain every chunk
                    # it had ever produced. Pausing the producer costs latency;
                    # not pausing it cost memory without a ceiling.
                    while (
                        queue.qsize() >= _STREAM_QUEUE_HIGH_WATER
                        and not stop_event.is_set()
                    ):
                        time.sleep(_STREAM_BACKPRESSURE_SLEEP_S)
            except BaseException as e:  # noqa: BLE001 — the sentinel must be posted
                _record_nucleus_degradation(
                    e if isinstance(e, Exception) else RuntimeError(str(e) or type(e).__name__),
                    action="ended stream from worker thread with explicit nucleus error marker",
                    extra={"model": model_type, "origin": origin, "phase": "stream_worker"},
                )
                logger.error("Nucleus stream thread failed: %s", e)
                _post(f"[NUCLEUS ERROR] {type(e).__name__}")
                if isinstance(e, BaseException) and not isinstance(e, Exception):
                    raise
            finally:
                _post(None)

        # Run in executor to avoid blocking the loop
        worker_task = fire_and_forget(
            asyncio.to_thread(_thread_worker),
            name="nucleus_manager.stream_worker",
        )
        if worker_task is None:
            await queue.put("[NUCLEUS ERROR] failed to schedule stream worker")
            await queue.put(None)

        try:
            stream_open = True
            while stream_open:
                chunk = await queue.get()
                if chunk is None:
                    stream_open = False
                else:
                    yield chunk
        finally:
            stop_event.set()
            if worker_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(worker_task), timeout=5.0)
                except TimeoutError:
                    logger.warning(
                        "Nucleus stream worker remained active after consumer stop; "
                        "model ownership is retained until the worker exits."
                    )

    def _resolve_temperature(self, kwargs: dict[str, Any], *, model_type: str, phase: str) -> float:
        temp = kwargs.get("temp", kwargs.get("temperature"))
        if temp is None:
            try:
                from core.container import ServiceContainer
                homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
                if homeostasis:
                    temp = homeostasis.get_modifiers().temperature_mod * 0.7
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                _record_nucleus_degradation(
                    e,
                    severity="warning",
                    action="used default sampler temperature after homeostatic coupling lookup failed",
                    extra={"model": model_type, "phase": phase},
                )
                capture_and_log(e, {'module': __name__})

        try:
            value = float(temp if temp is not None else 0.7)
        except (TypeError, ValueError) as e:
            _record_nucleus_degradation(
                e,
                severity="warning",
                action="used default sampler temperature after invalid caller temperature",
                extra={"model": model_type, "phase": phase, "temperature": repr(temp)},
            )
            value = 0.7
        # Reject non-finite temperature: NaN slides through min/max to the
        # upper bound under Python comparison semantics, silently maxing out
        # sampling entropy.
        if not math.isfinite(value):
            value = 0.7
        return min(2.0, max(0.0, value))

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        return await self.generate_text_async(prompt, system_prompt, **kwargs)

    async def _unload_model_entry(self, name: str, *, reason: str) -> None:
        async with self._model_lifecycle_context(name):
            await self._unload_model_entry_locked(name, reason=reason)

    async def _unload_model_entry_locked(self, name: str, *, reason: str) -> None:
        entry = self._ensure_model_entry(name)
        entry["model"] = None
        entry["tokenizer"] = None
        entry["loaded"] = False
        entry["cache"] = None
        # CP126 5a7bdf9c. The lease handle used to be dropped in the same
        # statement that read it, BEFORE the release was known to have
        # succeeded. A failed release then left authoritative lane ownership
        # stranded with the one object needed to retry, compensate or report
        # it already thrown away. The handle is kept until the release is
        # confirmed, and retained on failure so recovery has something to
        # act on.
        lease = entry.get("lane_lease")
        if lease is None:
            return
        try:
            await lease.release(reason=f"nucleus_unload:{reason}")
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            entry["lane_lease"] = lease
            entry["lane_lease_release_failed"] = {
                "reason": reason,
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "at_unix": time.time(),
            }
            _record_nucleus_degradation(
                exc,
                severity="critical",
                action=(
                    "cleared in-process model references but RETAINED the lane "
                    "lease: release failed and ownership is still held"
                ),
                extra={"model": name, "reason": reason},
            )
            return
        entry["lane_lease"] = None
        entry.pop("lane_lease_release_failed", None)

    async def unload_models(self):
        """Force unload all internal models and clear caches."""
        logger.info("🧠 [NUCLEUS] Unloading all internal models...")
        for name in tuple(self.models):
            await self._unload_model_entry(name, reason="unload_all")
        
        try:
            import mlx.core as mx

            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel
            sentinel = get_gpu_sentinel()
            if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                try:
                    if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                        mx.metal.clear_cache()
                    else:
                        mx.clear_cache()
                finally:
                    sentinel.release()
            else:
                # References were cleared, the sentinel was not acquired, and
                # the function returned as though unload had completed. The
                # weights can still be resident; a caller that unloads to free
                # memory and is told nothing has no way to know it did not.
                _record_nucleus_degradation(
                    TimeoutError("GPU sentinel unavailable during unload"),
                    severity="warning",
                    action=(
                        "cleared model references but could not reclaim the MLX cache; "
                        "memory may still be resident"
                    ),
                    extra={"phase": "unload_models", "cache_reclaimed": False},
                )
        except _NUCLEUS_RECOVERABLE_ERRORS as e:
            _record_nucleus_degradation(
                e,
                severity="warning",
                action=(
                    "left all model references cleared and skipped MLX cache reclamation "
                    "because the cache-clear dependency was unavailable"
                ),
                extra={"phase": "unload_models"},
            )
            logger.debug("[NUCLEUS] Cache clear skipped: %s", e)

    # --- Abstract Method Implementations ---

    async def generate_stream(self, prompt: str, system_prompt: str | None = None, model: str | None = None, **kwargs):
        """Implements abstract generate_stream by delegating to generate_stream_async."""
        kwargs = _with_requested_lane(kwargs, model)
        async for chunk in self.generate_stream_async(prompt, system_prompt, **kwargs):
            yield chunk

    def generate_text(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Synchronous wrapper for generate_text_async.

        The outer handler used to cover errors from ``run_until_complete`` as
        well as from loop discovery, so a failure that happened AFTER
        generation began fell into ``asyncio.run`` and ran the whole request a
        second time — a second model call, a second set of effects, from a
        caller that asked once. Loop discovery is the only thing inside the
        boundary now.
        """
        kwargs = _with_requested_lane({}, model)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Running a coroutine to completion from inside a live loop is not
            # possible; the caller has to await the async form.
            return "[NUCLEUS ERROR] Sync call in async loop."

        coro = self.generate_text_async(prompt, system_prompt, **kwargs)
        if loop is not None:
            return loop.run_until_complete(coro)
        return asyncio.run(coro)

    def generate_json(self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None, model: str | None = None) -> dict[str, Any]:
        """Synchronous wrapper for JSON extraction with schema enforcement.

        The schema is now (a) surfaced to the model so it generates conforming
        output, and (b) enforced on the result: required keys must be present,
        otherwise a typed error is returned rather than a silently
        non-conforming object.
        """
        from core.utils.json_utils import extract_json

        schema = schema if isinstance(schema, dict) else {}
        required = schema.get("required")
        properties = schema.get("properties")
        schema_hint = ""
        if isinstance(properties, dict) and properties:
            schema_hint = (
                "\n\nRespond with a single JSON object containing exactly these keys: "
                + ", ".join(str(k) for k in properties)
                + ". Output only the JSON."
            )
        text = self.generate_text(f"{prompt}{schema_hint}", system_prompt, model=model)
        result = extract_json(text)
        if not isinstance(result, dict):
            return {"error": "nucleus_json_extraction_failed", "raw": str(text)[:500]}
        if isinstance(required, list):
            missing = [str(key) for key in required if key not in result]
            if missing:
                return {
                    "error": "nucleus_json_schema_unsatisfied",
                    "missing_keys": missing,
                    "partial": result,
                }
        return result
