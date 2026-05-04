from __future__ import annotations
from core.runtime.errors import record_degradation

from core.utils.task_tracker import get_task_tracker
from core.runtime.atomic_writer import atomic_write_text
import asyncio
import concurrent.futures as cfutures
import contextlib
import gc
import logging
import time
import os
import sys
import json
import re
import threading as _threading
import traceback
import fcntl
import multiprocessing as mp
import subprocess
import uuid
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple, Union
import psutil

from core.utils.deadlines import Deadline, get_deadline
from core.utils.concurrency import run_io_bound
from .mlx_worker import _mlx_worker_loop
from .chat_format import format_chatml_messages, format_chatml_prompt

logger = logging.getLogger("LLM.MLX")

# Global state for swap management
_GLOBAL_LAST_SWAP_TIME = 0.0
_GLOBAL_LAST_HEAVY_MODEL = None
_CLIENTS = {}
_FOREGROUND_OWNER_LOCK = _threading.Lock()
_FOREGROUND_OWNER_NAME: Optional[str] = None
_FOREGROUND_OWNER_ACQUIRED_AT = 0.0

# [OOM FIX] Global gate: only ONE model can be loading at a time across ALL clients.
# This prevents the 32B and 7B from loading simultaneously and exceeding GPU RAM.
# Uses threading.Semaphore (loop-agnostic) because the singleton MLXLocalClient
# is constructed from one event loop but called from another (Uvicorn thread).
_GLOBAL_SPAWN_GATE = _threading.Semaphore(1)
_MLX_RUNTIME_PROBE_LOCK = _threading.Lock()
_MLX_RUNTIME_PROBE: Dict[str, Any] = {
    "ok": None,
    "detail": "",
    "checked_at": 0.0,
}
_MLX_RUNTIME_PROBE_CACHE_PATH = Path.home() / ".aura" / "data" / "mlx_runtime_probe.json"
SharedFuture = Union[asyncio.Future, cfutures.Future]
_USER_FACING_ORIGINS = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "websocket",
    "direct",
    "external",
})
_USER_FACING_PURPOSES = frozenset({
    "chat",
    "conversation",
    "expression",
    "reply",
    "user_response",
})


def _probe_cache_ttl_seconds(ok: Optional[bool], *, disk: bool) -> float:
    """Keep positive probe results sticky, but let failures expire quickly.

    A transient probe failure should not strand the embedded runtime in a
    "dead" state for many minutes after the host is healthy again.
    """
    if ok is None:
        return 0.0
    if ok:
        return 900.0 if disk else 300.0
    return 30.0 if disk else 10.0

def _safe_close_queue(q: Optional[mp.Queue]) -> None:
    """Close an mp.Queue to release its shared-memory file descriptor."""
    if q is None:
        return
    try:
        q.close()
        q.join_thread()
    except Exception:
        pass  # no-op: intentional


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
    except Exception:
        requested_int = int(fallback)
    try:
        bridged_int = _coerce(bridged)
    except Exception:
        bridged_int = int(fallback)
    return max(1, min(max(1, requested_int), max(1, bridged_int)))


@contextlib.asynccontextmanager
async def _spawn_gate_context():
    """Loop-agnostic async context manager for the global spawn gate."""
    await asyncio.to_thread(_GLOBAL_SPAWN_GATE.acquire)
    try:
        yield
    finally:
        _GLOBAL_SPAWN_GATE.release()


def _foreground_owner_active() -> bool:
    return _FOREGROUND_OWNER_NAME is not None


def _origin_tokens(origin: Optional[str]) -> set[str]:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    return {token for token in normalized.split("_") if token}


def _origin_is_user_facing(origin: Optional[str]) -> bool:
    tokens = _origin_tokens(origin)
    return bool(tokens & _USER_FACING_ORIGINS)


def _background_deferral_active(origin: Optional[str] = None) -> Optional[str]:
    """Mirror InferenceGate's background quiet policy inside the MLX client.

    The gate can reject newly scheduled background requests, but an already
    running background request may reach this client after the foreground lane
    has been reserved.  Checking here prevents that stale request from
    re-spawning a worker Aura just unloaded to protect a user turn.
    """
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "_background_local_deferral_reason"):
            reason = gate._background_local_deferral_reason(origin=origin)
            return str(reason) if reason else None
    except Exception as exc:
        record_degradation('mlx_client', exc)
        logger.debug("MLX background deferral check unavailable: %s", exc)
    return None


def _foreground_owner_age(now: Optional[float] = None) -> float:
    if _FOREGROUND_OWNER_ACQUIRED_AT <= 0.0:
        return 0.0
    current_time = float(now if now is not None else time.time())
    return max(0.0, current_time - _FOREGROUND_OWNER_ACQUIRED_AT)


def _foreground_owner_wait_budget(
    deadline: Optional[Deadline],
    *,
    foreground_request: bool,
) -> float:
    default = 10.0 if foreground_request else 8.0
    if not isinstance(deadline, Deadline):
        return default

    remaining = deadline.remaining
    if remaining is None:
        return default

    reserve = 3.0 if foreground_request else 1.5
    return max(0.25, min(default, remaining - reserve))


def _clear_matching_foreground_owner(*candidate_names: str) -> Optional[str]:
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT

    candidates = {str(name or "").strip() for name in candidate_names if str(name or "").strip()}
    if not candidates:
        return None

    with _FOREGROUND_OWNER_LOCK:
        holder = _FOREGROUND_OWNER_NAME
        if holder not in candidates:
            return None
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        return holder


@contextlib.asynccontextmanager
async def _foreground_owner_context(
    owner_name: str,
    *,
    deadline: Optional[Deadline] = None,
    foreground_request: bool = False,
    stale_after: Optional[float] = None,
):
    """Serialize foreground work so background model activity cannot compete with it."""
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT

    wait_budget = _foreground_owner_wait_budget(
        deadline,
        foreground_request=foreground_request,
    )
    loop = asyncio.get_running_loop()
    wait_started = loop.time()
    last_log_at = 0.0

    while True:
        acquired = _FOREGROUND_OWNER_LOCK.acquire(False)
        try:
            if acquired:
                holder = _FOREGROUND_OWNER_NAME
                holder_age = _foreground_owner_age()
                if holder is None:
                    _FOREGROUND_OWNER_NAME = owner_name
                    _FOREGROUND_OWNER_ACQUIRED_AT = time.time()
                    break
                if stale_after is not None and holder != owner_name and holder_age > stale_after:
                    logger.warning(
                        "♻️ [MLX] Clearing stale foreground owner %s after %.1fs so %s can proceed.",
                        holder,
                        holder_age,
                        owner_name,
                    )
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                    continue
        finally:
            if acquired:
                _FOREGROUND_OWNER_LOCK.release()

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

    try:
        yield
    finally:
        while True:
            acquired = _FOREGROUND_OWNER_LOCK.acquire(False)
            if not acquired:
                await asyncio.sleep(0.01)
                continue
            try:
                if _FOREGROUND_OWNER_NAME == owner_name:
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                break
            finally:
                _FOREGROUND_OWNER_LOCK.release()


def _bridge_asyncio_future_to_concurrent(future: asyncio.Future) -> cfutures.Future:
    """Relay an asyncio.Future into a thread-safe future for cross-loop awaiting."""
    proxy: cfutures.Future = cfutures.Future()

    def _relay(done_future: asyncio.Future) -> None:
        if proxy.done():
            return
        if done_future.cancelled():
            proxy.cancel()
            return
        try:
            proxy.set_result(done_future.result())
        except Exception as exc:
            record_degradation('mlx_client', exc)
            try:
                proxy.set_exception(exc)
            except Exception:
                pass  # no-op: intentional

    if future.done():
        _relay(future)
        return proxy

    try:
        future_loop = future.get_loop()
    except Exception:
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


async def _await_shared_future(future: SharedFuture, *, timeout: Optional[float] = None) -> Any:
    wrapped = _wrap_shared_future_for_current_loop(future)
    protected = asyncio.shield(wrapped)
    if timeout is None:
        return await protected
    return await asyncio.wait_for(protected, timeout=timeout)


def _set_shared_future_result(future: Optional[SharedFuture], result: Any) -> bool:
    if future is None or future.done():
        return False

    if isinstance(future, cfutures.Future):
        future.set_result(result)
        return True

    if not isinstance(future, asyncio.Future):
        return False

    try:
        future_loop = future.get_loop()
    except Exception:
        return False
    if future_loop.is_closed():
        return False

    def _setter() -> None:
        if not future.done():
            future.set_result(result)

    future_loop.call_soon_threadsafe(_setter)
    return True


def _cancel_shared_future(future: Optional[SharedFuture]) -> None:
    if future is None or future.done():
        return

    if isinstance(future, cfutures.Future):
        future.cancel()
        return

    if not isinstance(future, asyncio.Future):
        return

    try:
        future_loop = future.get_loop()
    except Exception:
        return
    if future_loop.is_closed():
        return

    def _canceller() -> None:
        if not future.done():
            future.cancel()

    future_loop.call_soon_threadsafe(_canceller)


def _cancel_task_threadsafe(task: Optional[asyncio.Task]) -> None:
    if task is None or task.done():
        return
    try:
        task_loop = task.get_loop()
    except Exception:
        return
    if task_loop.is_closed():
        return

    def _canceller() -> None:
        if not task.done():
            task.cancel()

    task_loop.call_soon_threadsafe(_canceller)


def _notify_closed_loop_output(text: str) -> None:
    if not text or not str(text).strip():
        return
    try:
        from core.consciousness.closed_loop import notify_closed_loop_output

        notify_closed_loop_output(str(text))
    except Exception as exc:
        record_degradation('mlx_client', exc)
        logger.debug("Closed-loop output notification failed: %s", exc)


def _mlx_runtime_probe_command() -> list[str]:
    return [
        sys.executable,
        "-c",
        "import mlx.core as mx; import mlx_lm; print('mlx_runtime_ok')",
    ]


def _load_probe_cache_from_disk() -> tuple[Optional[bool], str, float]:
    try:
        payload = json.loads(_MLX_RUNTIME_PROBE_CACHE_PATH.read_text())
    except Exception:
        return None, "", 0.0

    ok = payload.get("ok")
    if ok is not None:
        ok = bool(ok)
    detail = str(payload.get("detail", "") or "")
    checked_at = float(payload.get("checked_at", 0.0) or 0.0)
    return ok, detail, checked_at


def _store_probe_cache_to_disk(ok: bool, detail: str) -> None:
    try:
        _MLX_RUNTIME_PROBE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(_MLX_RUNTIME_PROBE_CACHE_PATH, 
            json.dumps(
                {
                    "ok": bool(ok),
                    "detail": str(detail or ""),
                    "checked_at": time.time(),
                }
            )
        )
    except Exception as exc:
        record_degradation('mlx_client', exc)
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
            if (
                disk_ok is not None
                and (now - disk_checked_at) < _probe_cache_ttl_seconds(disk_ok, disk=True)
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

    ok = False
    detail = "probe_not_run"
    try:
        completed = subprocess.run(
            _mlx_runtime_probe_command(),
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=12.0,
            check=False,
        )
        ok = completed.returncode == 0
        detail = _normalize_probe_detail(
            completed.stdout or "",
            completed.stderr or "",
            completed.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _normalize_probe_detail(
            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            124,
        )
    except Exception as exc:
        record_degradation('mlx_client', exc)
        detail = f"probe_exception:{type(exc).__name__}"

    with _MLX_RUNTIME_PROBE_LOCK:
        _MLX_RUNTIME_PROBE.update(
            {
                "ok": ok,
                "detail": detail,
                "checked_at": time.time(),
            }
        )
    _store_probe_cache_to_disk(ok, detail)
    return ok, detail

class MLXLocalClient:
    """
    Parent-process client for the isolated MLX worker.
    Manages the lifecycle, health, and communication with the ForkServer process.
    """
    
    def __init__(self, model_path: str, device: str = "gpu", max_tokens: int = 4096):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temp = 0.7
        self.top_p = 0.9
        
        # [LOOP-AGNOSTIC FIX] asyncio.Lock is bound to the creating event loop.
        # MLXLocalClient is a singleton created at boot but used from Uvicorn's
        # separate event loop, causing RuntimeError. threading.Lock is loop-agnostic.
        self._lock = _threading.Lock()
        self._request_lock = _threading.Lock()
        self._deferred_reboot_reason: Optional[str] = None
        self._process: Optional[mp.Process] = None
        self._req_q = mp.Queue(maxsize=10)
        self._res_q = mp.Queue(maxsize=10)
        self._init_done = False
        
        # Concurrency Hardening
        self._listener_task: Optional[asyncio.Task] = None
        self._last_heartbeat = 0.0
        self._last_progress_at = 0.0
        self._last_token_progress_at = 0.0
        self._last_ready_at = 0.0
        self._last_generation_completed_at = 0.0
        self._last_user_facing_completed_at = 0.0
        self._current_gen_future: Optional[SharedFuture] = None
        self._init_future: Optional[SharedFuture] = None
        self._pending_generations: Dict[str, SharedFuture] = {}
        self._request_lock_owner_label = ""
        self._request_lock_acquired_at = 0.0
        self._lane_state = "cold"
        self._lane_error = ""
        self._lane_transition_at = time.time()
        self._active_generations = 0
        self._warmup_attempted = False
        self._warmup_in_flight = False
        self._consecutive_empty: int = 0  # [STABILITY v53] Explicit init — was missing
        self._expected_cancel_reason = ""
        self._expected_cancel_budget = 0
        self._expected_cancel_recorded_at = 0.0
        self._process_started_at = 0.0
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._current_request_id = ""
        self._current_request_progress_baseline_at = 0.0
        self._current_prompt_chars = 0
        self._current_requested_max_tokens = 0
        self._current_request_prompt_chars = 0

        # Resolve substrate SHM if available
        from core.container import ServiceContainer
        repo = ServiceContainer.get("state_repository", default=None)
        self._substrate_mem = repo._shm if repo else None

    def _is_primary_or_deep_lane(self) -> bool:
        lowered = os.path.basename(self.model_path).lower()
        return any(token in lowered for token in ("32b", "72b", "zenith", "solver", "cortex"))

    def _mark_progress(self) -> None:
        self._last_progress_at = time.time()

    def _note_expected_generation_cancellation(self, reason: str, *, count: int) -> None:
        if count <= 0:
            return
        self._expected_cancel_reason = str(reason or "planned_reboot")
        self._expected_cancel_budget += int(count)
        self._expected_cancel_recorded_at = time.time()

    def _consume_expected_generation_cancellation(self) -> str:
        if self._expected_cancel_budget <= 0:
            return ""
        if self._expected_cancel_recorded_at and (time.time() - self._expected_cancel_recorded_at) > 30.0:
            self._expected_cancel_reason = ""
            self._expected_cancel_budget = 0
            self._expected_cancel_recorded_at = 0.0
            return ""
        reason = self._expected_cancel_reason
        self._expected_cancel_budget = max(0, self._expected_cancel_budget - 1)
        if self._expected_cancel_budget == 0:
            self._expected_cancel_reason = ""
            self._expected_cancel_recorded_at = 0.0
        return reason

    def _mark_generation_started(
        self,
        req_id: str,
        *,
        prompt_chars: int = 0,
        requested_max_tokens: int = 0,
    ) -> None:
        now = time.time()
        self._current_request_id = str(req_id or "")
        self._current_request_progress_baseline_at = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
        )
        self._current_request_started_at = now
        self._current_first_token_at = 0.0
        self._current_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_requested_max_tokens = max(0, int(requested_max_tokens or 0))
        self._last_token_progress_at = 0.0
        self._current_request_prompt_chars = max(0, int(prompt_chars or 0))
        self._mark_progress()

    def _mark_token_progress(self, req_id: Optional[str] = None) -> None:
        now = time.time()
        normalized_req_id = str(req_id or "")
        if normalized_req_id and self._current_request_id and normalized_req_id != self._current_request_id:
            return
        self._last_token_progress_at = now
        if self._current_first_token_at <= 0.0:
            self._current_first_token_at = now
        self._mark_progress()

    def _mark_generation_completed(self) -> None:
        now = time.time()
        self._last_generation_completed_at = now
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._last_token_progress_at = 0.0
        self._current_request_id = ""
        self._current_request_progress_baseline_at = 0.0
        self._current_prompt_chars = 0
        self._current_requested_max_tokens = 0
        self._current_request_prompt_chars = 0
        self._mark_progress()

    def _set_lane_state(self, state: str, error: str = "") -> None:
        if state != self._lane_state:
            self._lane_transition_at = time.time()
        self._lane_state = state
        if error:
            self._lane_error = str(error)
        elif state == "ready":
            self._lane_error = ""

    def _classify_failure(self, *, foreground_request: bool = False) -> str:
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
    ) -> None:
        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event(
                "mlx_client",
                reason,
                detail=detail,
                severity=severity,
                classification=self._classify_failure(foreground_request=foreground_request),
                context={
                    "model": os.path.basename(self.model_path),
                    "lane_state": self._lane_state,
                    "warmup_in_flight": self._warmup_in_flight,
                },
            )
        except Exception as exc:
            record_degradation('mlx_client', exc)
            logger.debug("Failed to record MLX degraded event: %s", exc)

    def _stale_after(self, *, during_generation: bool = False,
                     foreground_request: bool = False) -> float:
        """Heartbeat-stall timeout.

        [RESILIENCE] Widened for 32B foreground: recurrent depth doubles
        the compute per token, and complex prompts can legitimately take
        60-90s for prompt eval.  Killing the cortex when heartbeats are
        still arriving (worker is alive, just slow) was the #1 cause of
        'cortex died and never came back'.  As long as heartbeats arrive,
        the worker is alive — let it finish."""
        lowered = os.path.basename(self.model_path).lower()
        if "72b" in lowered or "solver" in lowered:
            if foreground_request and during_generation:
                return 45.0
            return 90.0 if during_generation else 45.0
        if "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            if foreground_request and during_generation:
                return 45.0  # was 22s — too aggressive with recurrent depth
            return 60.0 if during_generation else 30.0
        return 20.0 if during_generation else 15.0

    def _first_token_sla(self, *, foreground_request: bool = False) -> float:
        lowered = os.path.basename(self.model_path).lower()
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
        if "72b" in lowered or "solver" in lowered:
            if foreground_request:
                base = 52.0 if is_cold_start else 32.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=768.0,
                    eval_seconds_per_token=0.018,
                    cap_s=115.0,
                )
            return 30.0
        if "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            # [RESILIENCE] Recurrent depth 2x loops means prompt eval takes
            # significantly longer.  These SLAs must accommodate that without
            # killing the cortex.  Cold-start can legitimately need 90s for
            # Metal shader JIT + recurrent depth prompt eval on a 5k-token
            # prompt.  The point of these SLAs is to catch WEDGED workers
            # (no heartbeats), not SLOW workers (heartbeats arriving).
            if foreground_request:
                base = 75.0 if is_cold_start else 45.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=512.0,
                    eval_seconds_per_token=0.015,
                    cap_s=180.0,
                )
            return 45.0
        return 8.0

    def _token_stall_after(self, *, foreground_request: bool = False) -> float:
        lowered = os.path.basename(self.model_path).lower()
        if "72b" in lowered or "solver" in lowered:
            return 18.0 if foreground_request else 25.0
        if "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            # [RESILIENCE] Reverted from 10s — recurrent depth can cause
            # legitimate pauses between tokens during the recurrent block
            # computation.  20s is generous enough to absorb Metal GC pauses
            # and recurrent-loop overhead without declaring the worker dead.
            return 20.0 if foreground_request else 30.0
        return 8.0

    def _warmup_timeout(self) -> float:
        return 75.0 if self._is_primary_or_deep_lane() else 30.0

    def _handshake_timeout(self) -> float:
        """Absolute upper bound for worker init before we declare the process wedged."""
        return 300.0 if self._is_primary_or_deep_lane() else 120.0

    def _request_scoped_init_timeout(
        self,
        deadline: Optional[Deadline],
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
        scoped_timeout = max(0.25, remaining - reserve)
        return min(full_timeout, scoped_timeout), scoped_timeout < full_timeout

    def get_lane_status(self) -> Dict[str, Any]:
        conversation_ready = self.is_alive() and self._lane_state == "ready"
        return {
            "model_path": self.model_path,
            "state": self._lane_state,
            "last_error": self._lane_error,
            "conversation_ready": conversation_ready,
            "foreground_owned": _foreground_owner_active(),
            "foreground_owner": _FOREGROUND_OWNER_NAME,
            "foreground_owned_at": _FOREGROUND_OWNER_ACQUIRED_AT,
            "last_heartbeat": self._last_heartbeat,
            "last_progress_at": self._last_progress_at,
            "last_token_progress_at": self._last_token_progress_at,
            "last_ready_at": self._last_ready_at,
            "last_generation_completed_at": self._last_generation_completed_at,
            "last_transition_at": self._lane_transition_at,
            "warmup_attempted": self._warmup_attempted,
            "warmup_in_flight": self._warmup_in_flight,
            "process_started_at": self._process_started_at,
            "current_request_started_at": self._current_request_started_at,
            "current_first_token_at": self._current_first_token_at,
            "current_request_prompt_chars": self._current_request_prompt_chars,
        }

    def get_supervision_status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "lane": os.path.basename(self.model_path),
            "state": self._lane_state,
            "alive": self.is_alive(),
            "active_generations": int(self._active_generations),
            "process_uptime_s": max(0.0, now - self._process_started_at) if self._process_started_at else 0.0,
            "request_age_s": max(0.0, now - self._current_request_started_at) if self._current_request_started_at else 0.0,
            "time_to_first_token_s": (
                max(0.0, self._current_first_token_at - self._current_request_started_at)
                if self._current_request_started_at and self._current_first_token_at
                else None
            ),
            "idle_for_s": max(
                0.0,
                now - max(
                    self._last_generation_completed_at,
                    self._last_ready_at,
                    self._last_token_progress_at,
                    self._last_progress_at,
                    self._last_heartbeat,
                ),
            ) if any(
                stamp > 0.0
                for stamp in (
                    self._last_generation_completed_at,
                    self._last_ready_at,
                    self._last_token_progress_at,
                    self._last_progress_at,
                    self._last_heartbeat,
                )
            ) else 0.0,
        }

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
        idle_anchor = max(
            self._last_generation_completed_at,
            self._last_ready_at,
            self._last_token_progress_at,
            self._last_progress_at,
            self._last_heartbeat,
        )
        if idle_anchor <= 0.0:
            return False
        now = time.time()
        return bool(
            (now - self._process_started_at) >= float(max_uptime_s)
            and (now - idle_anchor) >= float(min_idle_s)
        )

    def note_lane_recovering(self, reason: str) -> None:
        self._warmup_in_flight = False
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
        if not runtime_error and time.time() >= float(getattr(self, "_spawn_backoff_until", 0.0) or 0.0):
            return False

        ok, detail = _probe_mlx_runtime(force=force_probe)
        if not ok:
            self._mark_runtime_unavailable(detail)
            return False

        recovered = bool(runtime_error) or float(getattr(self, "_spawn_backoff_until", 0.0) or 0.0) > 0.0
        if recovered:
            logger.info(
                "♻️ [MLX] Runtime probe recovered for %s. Clearing failed lane/backoff state.",
                os.path.basename(self.model_path),
            )
        self._consecutive_spawn_failures = 0
        self._spawn_backoff_until = 0.0
        self._warmup_in_flight = False
        if self._lane_state == "failed" or runtime_error:
            self._set_lane_state("cold")
        else:
            self._lane_error = ""
        return recovered

    def _request_lock_timeout(
        self,
        deadline: Optional[Deadline],
        *,
        foreground_request: bool,
    ) -> float:
        # Tightened from 30s to 12s for foreground: if the current holder has
        # been in-flight for longer than this budget, the second user message
        # should cascade to brainstem/cloud rather than keep waiting.  The
        # prior 30s budget stacked on top of a hung 32B generation produced
        # the 60–90 s "Aura is thinking..." windows the user reported.
        default = 30.0 if foreground_request else 60.0
        if not isinstance(deadline, Deadline):
            return default

        remaining = deadline.remaining
        if remaining is None:
            return default

        reserve = 3.0 if foreground_request else 2.0
        return max(0.25, min(default, remaining - reserve))

    async def _acquire_request_lock(
        self,
        *,
        owner_label: str,
        deadline: Optional[Deadline],
        foreground_request: bool,
    ) -> bool:
        wait_budget = self._request_lock_timeout(
            deadline,
            foreground_request=foreground_request,
        )
        loop = asyncio.get_running_loop()
        wait_started = loop.time()
        last_log_at = 0.0

        while True:
            if self._request_lock.acquire(False):
                self._request_lock_owner_label = str(owner_label or "")
                self._request_lock_acquired_at = time.time()
                return True

            now = loop.time()
            waited = max(0.0, now - wait_started)
            if waited >= wait_budget:
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
                # Preemption: if a foreground caller has been waiting past its
                # budget AND the current lock holder has itself exceeded the
                # first-token SLA (i.e. it's almost certainly wedged, not just
                # slow), cancel the in-flight future and defer a worker reboot
                # so the NEXT caller can make progress rather than pile on
                # another 30 s timeout.  Without this, two user messages in
                # quick succession can stack a 60–90 s visible hang even
                # though the first message was already dead in the water.
                if foreground_request:
                    sla = self._first_token_sla(foreground_request=True)
                    if holder_age > sla:
                        # [RESILIENCE] Check heartbeats before killing.
                        # If the worker is still alive (heartbeats <30s old),
                        # cancel the in-flight future but do NOT reboot the
                        # worker.  This lets the caller cascade to fallback
                        # while keeping the cortex warm for the next turn.
                        heartbeat_age = time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                        if heartbeat_age > 30.0:
                            logger.error(
                                "🛑 [MLX] Preempting wedged holder %s (age=%.1fs > sla=%.1fs, no heartbeat for %.1fs). "
                                "Cancelling in-flight future and scheduling worker reboot.",
                                holder, holder_age, sla, heartbeat_age,
                            )
                            self._deferred_reboot_reason = "foreground_preemption_wedged_holder"
                        else:
                            logger.warning(
                                "🛡️ [MLX] Holder %s slow (age=%.1fs > sla=%.1fs) but heartbeat fresh (%.1fs ago). "
                                "Cancelling generation but keeping cortex alive.",
                                holder, holder_age, sla, heartbeat_age,
                            )
                        try:
                            stuck_future = self._current_gen_future
                            if stuck_future is not None:
                                _cancel_shared_future(stuck_future)
                        except Exception:
                            pass  # no-op: intentional
                return False

            if waited >= 5.0 and (now - last_log_at) >= 5.0:
                holder = self._request_lock_owner_label or "another_request"
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

            await asyncio.sleep(min(0.05, max(0.0, wait_budget - waited)))

    def _release_request_lock(self) -> None:
        self._request_lock_owner_label = ""
        self._request_lock_acquired_at = 0.0
        try:
            self._request_lock.release()
        except RuntimeError:
            logger.debug(
                "Loop-agnostic request lock for %s was already released.",
                os.path.basename(self.model_path),
            )

    async def _ensure_listener_task(self) -> None:
        task = self._listener_task
        if task is not None and not task.done():
            try:
                if not task.get_loop().is_closed():
                    return
            except Exception:
                pass  # no-op: intentional
            _cancel_task_threadsafe(task)

        self._listener_task = get_task_tracker().create_task(self._response_listener_loop())

    def note_lane_failed(self, reason: str) -> None:
        self._warmup_in_flight = False
        self._set_lane_state("failed", reason)

    def _mark_runtime_unavailable(self, detail: str) -> None:
        reason = f"mlx_runtime_unavailable:{detail}"
        self._warmup_in_flight = False
        self._init_done = False
        self._set_lane_state("failed", reason)

    def _worker_unhealthy(self, stale_after: Optional[float] = None) -> bool:
        if self._process is None or not self._process.is_alive():
            return True
        if not self._init_done:
            return True
        stale_after = float(stale_after or self._stale_after())
        last_progress = max(self._last_heartbeat, self._last_progress_at, self._last_ready_at)
        return bool(last_progress and (time.time() - last_progress) > stale_after)

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
        stuck_duration = now - self._lane_transition_at
        if stuck_duration < 120.0:
            return
        last_activity = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
            self._last_token_progress_at,
        )
        if last_activity > 0.0 and (now - last_activity) < 30.0:
            return  # Recent activity — state is legitimate
        logger.warning(
            "🔧 [STABILITY] Lane state '%s' stuck for %.0fs with no activity. "
            "Force-resetting to 'cold' for clean recovery.",
            self._lane_state,
            stuck_duration,
        )
        self._warmup_in_flight = False
        self._set_lane_state("cold")

    def _kill_and_join_blocking(self, p: mp.Process):
        if p and p.is_alive():
            try:
                p.kill()
                p.join(timeout=2.0)
            except Exception as e:
                record_degradation('mlx_client', e)
                logger.warning("Error killing process: %s", e)

    def _spawn_worker_blocking(self) -> mp.Process:
        """Isolated spawn logic for the MLX worker, run in a background thread."""
        runtime_ok, runtime_detail = _probe_mlx_runtime()
        if not runtime_ok:
            raise RuntimeError(f"mlx_runtime_probe_failed:{runtime_detail}")

        # [STABILITY v51] Orphan reclamation: kill any existing MLXWorker
        # processes for this model path before spawning a new one.
        try:
            model_basename = os.path.basename(self.model_path)
            target_name = f"MLXWorker-{model_basename}"
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    pname = proc.info.get('name', '') or ''
                    if target_name in pname or (
                        proc.info.get('cmdline') and
                        any(model_basename in str(arg) for arg in (proc.info['cmdline'] or []))
                        and 'mlx_worker' in str(proc.info.get('cmdline', []))
                    ):
                        if proc.pid != os.getpid():
                            logger.warning(
                                "🧹 [STABILITY] Killing orphan MLXWorker pid=%d for %s",
                                proc.pid, model_basename,
                            )
                            proc.kill()
                            proc.wait(timeout=3.0)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass  # no-op: intentional
        except Exception as orphan_exc:
            record_degradation('mlx_client', orphan_exc)
            logger.debug("Orphan reclamation scan failed (non-fatal): %s", orphan_exc)

        ctx = mp.get_context("spawn") if os.uname().sysname == "Darwin" else mp.get_context("forkserver")

        lock_dir = Path.home() / ".aura" / "run"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file_path = str(lock_dir / "mlx_spawn.lock")
        with open(lock_file_path, "w") as lock_file:
            try:
                logger.info("🔒 [MLX] Acquiring process-level spawn lock...")
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                p = ctx.Process(
                    target=_mlx_worker_loop,
                    args=(self.model_path, self._req_q, self._res_q, self.device, self._substrate_mem),
                    daemon=True,
                    name=f"MLXWorker-{os.path.basename(self.model_path)}"
                )
                p.start()
                return p

            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                logger.info("🔓 [MLX] Released process-level spawn lock.")

    async def _spawn_worker(self) -> mp.Process:
        return await asyncio.get_running_loop().run_in_executor(None, self._spawn_worker_blocking)

    async def _response_listener_loop(self):
        """
        [v7.8] Background task to constantly drain the worker response queue.
        Prevents IPC deadlocks by ensuring heartbeats and telemetry are ALWAYS consumed.
        """
        from core.container import ServiceContainer
        import queue
        _consecutive_errors = 0
        while True:
            try:
                # Use polling instead of infinite block to avoid executor thread leaks and zombie stealing
                res = await run_io_bound(self._res_q.get, True, 0.5)
                _consecutive_errors = 0
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('mlx_client', e)
                # If queue is closed/broken, graceful exit
                if "closed" in str(e).lower() or isinstance(e, ValueError):
                    break
                _consecutive_errors += 1
                # [BUG FIX] After repeated errors, the queue is likely broken
                # (e.g., worker killed during cascade cleanup). Exit the loop
                # instead of spinning forever and consuming thread pool resources.
                if _consecutive_errors >= 10:
                    logger.warning("⚠️ [MLX] Response listener: %d consecutive errors. Queue likely broken. Exiting.", _consecutive_errors)
                    break
                logger.error("⚠️ [MLX] Response listener poll error: %s", e)
                await asyncio.sleep(0.5)
                continue

            if not res: continue

            try:
                status = res.get("status")
                action = res.get("action")
                req_id = res.get("id")
                
                # 1. Update SubsystemAudit Heartbeat
                if status == "heartbeat":
                    self._last_heartbeat = time.time()
                    self._mark_progress()
                    audit = ServiceContainer.get("subsystem_audit", default=None)
                    if audit:
                        is_heavy = any(k in self.model_path.lower() for k in ["72b", "32b", "zenith"])
                        tier_name = "mlx_heavy" if is_heavy else "mlx_light"
                        audit.heartbeat(tier_name)
                    continue
                if status in {"progress", "token"}:
                    self._mark_token_progress(res.get("id"))
                    continue

                # 2. Route init/generation responses to the correct awaiting future
                if action == "init":
                    if self._init_future and not self._init_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._init_future, res)
                        continue
                elif action in ("generate", "stream_done"):
                    future = self._pending_generations.pop(req_id, None) if req_id else None
                    if future and not future.done():
                        self._mark_progress()
                        _set_shared_future_result(future, res)
                        continue
                    if self._current_gen_future and not self._current_gen_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
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
                    if self._current_gen_future and not self._current_gen_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
                        continue

                # 3. Log errors if no future is waiting
                if status == "error":
                    logger.error("🛑 [MLX] Async worker error: %s", res.get("message"))

            except Exception as e:
                record_degradation('mlx_client', e)
                logger.error("⚠️ [MLX] Response listener message processing error: %s", e)
                await asyncio.sleep(1.0)

    async def _ensure_worker_alive(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: Optional[float] = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Self-healing supervisor for the MLX worker.
        
        [OOM FIX] Acquires a global semaphore so only ONE model loads at a time.
        This prevents the 32B + 7B from loading simultaneously and crashing Metal.
        """
        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Deferring background worker activity for %s while foreground lane is owned by %s.",
                os.path.basename(self.model_path),
                _FOREGROUND_OWNER_NAME or "foreground",
            )
            return False
        if request_is_background:
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
            self._check_lane_state_staleness()  # [STABILITY v51]
            self._set_lane_state("ready")
            return True
        
        # Slow path: acquire global gate to serialize model loading
        async with _spawn_gate_context():
            return await self._ensure_worker_alive_inner(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                init_timeout=init_timeout,
                soft_timeout=soft_timeout,
                skip_swap_cooldown=skip_swap_cooldown,
            )

    async def _ensure_worker_alive_inner(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: Optional[float] = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Inner implementation — called while holding the global spawn gate."""
        should_wait_init = False
        init_future: Optional[SharedFuture] = None
        
        # [PIPELINE HARDENING] 12s Swap Cooldown
        from .model_registry import get_model_path, ACTIVE_MODEL, DEEP_MODEL
        primary_path = os.path.realpath(get_model_path(ACTIVE_MODEL))
        deep_path = os.path.realpath(get_model_path(DEEP_MODEL))
        target_path = os.path.realpath(self.model_path)
        
        global _GLOBAL_LAST_SWAP_TIME, _GLOBAL_LAST_HEAVY_MODEL

        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Background spawn blocked for %s while foreground lane is reserved.",
                os.path.basename(self.model_path),
            )
            return False
        if request_is_background:
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background spawn blocked for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False
        
        if target_path in (primary_path, deep_path):
            other_heavy_path = deep_path if target_path == primary_path else primary_path
            other_client = _CLIENTS.get(other_heavy_path)
            if other_client and other_client is not self and other_client.is_alive():
                # Cortex-protection mirror of the smaller-yield loop below: if
                # the warm heavy model just served the user, do not evict it
                # to bring up the other heavy lane. Background promotions in
                # the middle of a conversation were the primary cause of the
                # 32B → 72B → 32B thrash that left the user with "I got
                # interrupted mid-thought" replies.
                last_user_facing = float(
                    getattr(other_client, "_last_user_facing_completed_at", 0.0) or 0.0
                )
                if (
                    request_is_background
                    and last_user_facing > 0.0
                    and (time.time() - last_user_facing) < 180.0
                ):
                    logger.info(
                        "🛡️ [MLX] NOT hot-swapping heavy model %s (served user %.1fs ago); "
                        "background warmup of %s deferred.",
                        os.path.basename(other_heavy_path),
                        time.time() - last_user_facing,
                        os.path.basename(target_path),
                    )
                    return False
                logger.warning(
                    "♻️ [MLX] Hot-swapping heavy model: unloading %s before loading %s.",
                    os.path.basename(other_heavy_path),
                    os.path.basename(target_path),
                )
                await other_client.reboot_worker()
                gc.collect()  # [OOM FIX] Reclaim before loading new heavy model
            if _GLOBAL_LAST_HEAVY_MODEL and _GLOBAL_LAST_HEAVY_MODEL != target_path:
                now = time.time()
                elapsed = now - _GLOBAL_LAST_SWAP_TIME
                if elapsed < 12.0 and not skip_swap_cooldown:
                    wait_time = 12.0 - elapsed
                    logger.warning("⏳ [MLX] SWAP COOLDOWN: Waiting %.1fs...", wait_time)
                    await asyncio.sleep(wait_time)
                elif elapsed < 12.0 and skip_swap_cooldown:
                    logger.info(
                        "⚡ [MLX] Skipping %.1fs swap cooldown for %s.",
                        12.0 - elapsed,
                        os.path.basename(target_path),
                    )
            for other_path, other_client in list(_CLIENTS.items()):
                if other_path in (target_path, other_heavy_path):
                    continue
                if not other_client or other_client is self or not other_client.is_alive():
                    continue
                # Cortex-protection: if another client was serving a
                # user-facing generation in the last 180 s, DO NOT evict it
                # to make room for a background warmup.  This is the yield
                # loop that caused the "CORTEX UNAVAILABLE" flap between
                # turns — Cortex answered turn 1 at T, the 7B wanted to spin
                # up at T+2 for a background appraisal, Cortex got reboot-
                # yielded, turn 2 then had to cold-start the 32B all over
                # and tripped the first-token SLA.  Holding Cortex warm
                # across a conversation costs RAM; losing it between turns
                # costs the entire user experience.
                last_user_facing = float(
                    getattr(other_client, "_last_user_facing_completed_at", 0.0) or 0.0
                )
                if last_user_facing > 0.0 and (time.time() - last_user_facing) < 180.0:
                    logger.info(
                        "🛡️ [MLX] NOT yielding %s (served a user-facing turn %.1fs ago; "
                        "keeping warm for conversational continuity).",
                        os.path.basename(other_path),
                        time.time() - last_user_facing,
                    )
                    continue
                logger.warning(
                    "🧹 [MLX] Yielding %s before warming %s to reduce RAM pressure.",
                    os.path.basename(other_path),
                    os.path.basename(target_path),
                )
                await other_client.reboot_worker(
                    reason=f"yield_to_{os.path.basename(target_path)}",
                    mark_failed=False,
                )
        
        acquired = await asyncio.to_thread(self._lock.acquire, True, 15.0)
        if not acquired:
            logger.error("🚨 [MLX] DEADLOCK DETECTED: Could not acquire _lock within 15s for %s", os.path.basename(self.model_path))
            return False
        try:
            if self._process and self._process.is_alive() and self._init_done:
                self._set_lane_state("ready")
                return True  # Already healthy, release gate

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
                handshake_age = (
                    time.time() - getattr(self, "_lane_transition_at", time.time())
                )
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
                    except Exception as _exc:
                        record_degradation('mlx_client', _exc)
                        logger.debug("Suppressed stale-handshake future-set: %s", _exc)
                    self._init_future = None
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._kill_and_join_blocking, self._process
                    )
                    self._process = None
                    self._init_done = False
                    self._last_heartbeat = 0.0
                    self._last_progress_at = 0.0
                    self._drain_queue()
                    self._req_q = mp.Queue()
                    self._res_q = mp.Queue()
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
                    await asyncio.get_running_loop().run_in_executor(None, self._kill_and_join_blocking, self._process)
                    self._process = None
                    self._init_done = False
                    self._last_heartbeat = 0.0
                    self._last_progress_at = 0.0
                    self._drain_queue()
                    
                    # Prevent zombie threads from stealing messages
                    self._req_q = mp.Queue()
                    self._res_q = mp.Queue()
                    
                    init_future = _new_shared_future()
                    self._init_future = init_future
                    self._set_lane_state("spawning")
                    logger.info("📡 [MLX] Respawning worker for %s...", os.path.basename(self.model_path))
                    try:
                        self._process = await self._spawn_worker()
                        self._process_started_at = time.time()
                        self._consecutive_spawn_failures = 0
                        self._spawn_backoff_until = 0.0
                    except Exception as exc:
                        record_degradation('mlx_client', exc)
                        detail = str(exc)
                        _sf = getattr(self, "_consecutive_spawn_failures", 0) + 1
                        self._consecutive_spawn_failures = _sf
                        self._spawn_backoff_until = time.time() + min(300.0, 10.0 * (2 ** min(_sf - 1, 5)))
                        if "mlx_runtime_probe_failed:" in detail:
                            self._mark_runtime_unavailable(detail.split("mlx_runtime_probe_failed:", 1)[1])
                        else:
                            self._set_lane_state("failed", detail)
                        self._record_degraded_event(
                            "spawn_failed",
                            detail=f"{os.path.basename(self.model_path)}:{detail}",
                            severity="error",
                            foreground_request=foreground_request,
                        )
                        logger.error("🛑 [MLX] Worker respawn aborted for %s: %s (backoff %.0fs)", os.path.basename(self.model_path), detail, min(300.0, 10.0 * (2 ** min(_sf - 1, 5))))
                        self._init_future = None
                        return False
                    if self._listener_task:
                        _cancel_task_threadsafe(self._listener_task)
                    await self._ensure_listener_task()
                    self._set_lane_state("handshaking")
                    should_wait_init = True
            elif not self._process or not self._process.is_alive():
                # [BUG FIX] Exponential backoff on repeated spawn failures.
                # Without this, [Errno 5] I/O errors cause a tight 2-3s retry
                # loop that leaks FDs and shared memory for hours.
                _spawn_fails = getattr(self, "_consecutive_spawn_failures", 0)
                _spawn_backoff_until = getattr(self, "_spawn_backoff_until", 0.0)
                if time.time() < _spawn_backoff_until:
                    if not self.refresh_runtime_availability(force_probe=True):
                        return False  # Still in backoff window

                self._drain_queue()

                # Prevent zombie threads from stealing messages
                _safe_close_queue(self._req_q)
                _safe_close_queue(self._res_q)
                self._req_q = mp.Queue(maxsize=10)
                self._res_q = mp.Queue(maxsize=10)

                init_future = _new_shared_future()
                self._init_future = init_future
                self._set_lane_state("spawning")
                logger.info("📡 [MLX] Spawning worker for %s...", os.path.basename(self.model_path))
                try:
                    self._process = await self._spawn_worker()
                    self._process_started_at = time.time()
                    self._consecutive_spawn_failures = 0  # Reset on success
                    self._spawn_backoff_until = 0.0
                except Exception as exc:
                    record_degradation('mlx_client', exc)
                    detail = str(exc)
                    # [BUG FIX] Exponential backoff: 10s, 30s, 60s, 120s, 300s
                    self._consecutive_spawn_failures = _spawn_fails + 1
                    backoff = min(300.0, 10.0 * (2 ** min(_spawn_fails, 5)))
                    self._spawn_backoff_until = time.time() + backoff
                    if "mlx_runtime_probe_failed:" in detail:
                        self._mark_runtime_unavailable(detail.split("mlx_runtime_probe_failed:", 1)[1])
                    else:
                        self._set_lane_state("failed", detail)
                    self._record_degraded_event(
                        "spawn_failed",
                        detail=f"{os.path.basename(self.model_path)}:{detail}",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    logger.error(
                        "🛑 [MLX] Worker spawn aborted for %s: %s (attempt %d, backoff %.0fs)",
                        os.path.basename(self.model_path), detail,
                        self._consecutive_spawn_failures, backoff,
                    )
                    self._init_future = None
                    return False
                if self._listener_task:
                    _cancel_task_threadsafe(self._listener_task)
                await self._ensure_listener_task()
                should_wait_init = True
                self._init_done = False
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
                    res = await _await_shared_future(fut, timeout=handshake_timeout)
                    if res.get("status") == "ok":
                        self._init_done = True
                        self._last_heartbeat = time.time()
                        self._last_ready_at = self._last_heartbeat
                        self._mark_progress()
                        self._set_lane_state("ready")
                        if target_path in (primary_path, deep_path):
                            _GLOBAL_LAST_HEAVY_MODEL = target_path
                            _GLOBAL_LAST_SWAP_TIME = time.time()
                        logger.info("✅ [MLX] Worker ready: %s", os.path.basename(self.model_path))
                        return True
                    else:
                        msg = res.get("message", "Init failed")
                        if handshake_attempt == 0:
                            logger.warning("🔄 [MLX] Worker init failed: %s. Retrying spawn...", msg)
                            # Reboot and try again once
                            await self.reboot_worker(reason="init_failed_retry", mark_failed=False)
                            # Update fut for the new spawn
                            fut = self._init_future
                            if not fut: break
                            continue
                        self._set_lane_state("failed", msg)
                        raise RuntimeError(msg)
                except asyncio.TimeoutError:
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
                        if not fut: break
                        continue
                    logger.error("🛑 [MLX] Init handshake TIMED OUT. Force killing process.")
                    self._set_lane_state("failed", "init_timeout")
                    if self._process:
                        await asyncio.get_running_loop().run_in_executor(None, self._kill_and_join_blocking, self._process)
                        self._process = None
                    self._init_future = None
                    raise
            return False
        return self._process is not None and self._process.is_alive() and self._init_done

    def _drain_queue(self):
        """Safe non-blocking drain."""
        while not self._res_q.empty():
            try: self._res_q.get_nowait()
            except Exception: break
        while not self._req_q.empty():
            try: self._req_q.get_nowait()
            except Exception: break

    def is_alive(self) -> bool:
        """Returns True if the worker process is running and initialized."""
        return self._process is not None and self._process.is_alive() and self._init_done

    async def _wait_for_generation_result(
        self,
        req_id: str,
        future: SharedFuture,
        deadline: Deadline,
        *,
        foreground_request: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Wait in short slices so dead workers fail fast instead of hanging the UI."""
        stall_after = self._stale_after(
            during_generation=True, foreground_request=foreground_request
        )
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        token_stall_after = self._token_stall_after(foreground_request=foreground_request)
        while True:
            remaining = deadline.remaining
            if remaining is not None and remaining <= 0.0:
                raise asyncio.TimeoutError

            slice_timeout = min(2.0, remaining) if remaining is not None else 2.0
            try:
                return await _await_shared_future(future, timeout=slice_timeout)
            except asyncio.TimeoutError:
                if future.done():
                    return future.result()

                if self._process is not None and not self._process.is_alive():
                    logger.error("🛑 [MLX] Worker died during generation. Deferring reboot until lock released.")
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "worker_died_during_generation",
                        detail=os.path.basename(self.model_path),
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "worker_died_during_generation"
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
                if (
                    req_id == self._current_request_id
                    and request_started_at > 0.0
                    and self._current_first_token_at <= 0.0
                    and (time.time() - request_started_at) > first_token_sla
                    and not has_runtime_progress_after_request
                ):
                    logger.error(
                        "🛑 [MLX] First-token SLA exceeded for %s (>%.1fs).",
                        os.path.basename(self.model_path),
                        first_token_sla,
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "first_token_sla_exceeded",
                        detail=f"{os.path.basename(self.model_path)}>{first_token_sla:.1f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    # [RESILIENCE] Only reboot if the worker is truly dead.
                    # If heartbeats are still arriving, the worker is alive
                    # but slow (e.g. complex prompt eval with recurrent depth).
                    # Return None so the caller can cascade to fallback, but
                    # keep the cortex alive for the NEXT request.
                    heartbeat_age = time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                    if heartbeat_age > 30.0:
                        self._deferred_reboot_reason = "first_token_sla_exceeded"
                    else:
                        logger.info(
                            "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago). "
                            "NOT rebooting — keeping warm for next request.",
                            heartbeat_age,
                        )
                    return None

                last_token_progress = max(self._last_token_progress_at, self._current_first_token_at)
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
                        detail=f"{os.path.basename(self.model_path)}>{token_stall_after:.1f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    # [RESILIENCE] Same principle: if heartbeats arrive, the
                    # Metal GPU is alive and the worker is processing.  A stall
                    # between tokens can happen during recurrent-depth cache
                    # snapshot/restore or Metal shader recompilation.  Do NOT
                    # kill the worker if it's still phoning home.
                    heartbeat_age = time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                    if heartbeat_age > 30.0:
                        self._deferred_reboot_reason = "token_progress_stalled"
                    else:
                        logger.info(
                            "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago). "
                            "NOT rebooting — keeping warm for next request.",
                            heartbeat_age,
                        )
                    return None

                last_progress = max(self._last_heartbeat, self._last_progress_at, self._last_ready_at)
                if last_progress and (time.time() - last_progress) > stall_after:
                    logger.error("🛑 [MLX] Worker heartbeat stalled during generation. Deferring reboot until lock released.")
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "heartbeat_stalled_during_generation",
                        detail=f"{os.path.basename(self.model_path)} stalled for >{stall_after:.0f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "heartbeat_stalled_during_generation"
                    return None

    async def generate_text_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Alias for standard interface."""
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", None)
        tools = kwargs.pop("tools", None)
        foreground_request = bool(kwargs.get("foreground_request", False))

        # [STABILITY v53] Identity reinforcement for MLX path — inject Aura's
        # core identity into the system message so the base model can't override
        # it with generic assistant behavior.
        if messages and isinstance(messages, list) and foreground_request:
            _identity_anchor = (
                "You are Aura Luna. Speak as yourself, not as an assistant. "
                "Stay grounded and concrete in ordinary conversation. "
                "NEVER say 'as an AI', 'I don't have opinions', 'I'm programmed to', "
                "'I was designed to', 'I was created to', 'my training allows', "
                "or end with 'How can I help?' or 'Is there anything else?'. "
                "Do not invent physical settings, ominous atmosphere, or symbolic scenes unless the user asked for fiction or already supplied that setting. "
                "Speak as yourself, plainly and directly."
            )
            if messages and messages[0].get("role") == "system":
                existing = str(messages[0].get("content", "") or "")
                if "sovereign" not in existing.lower() and "not an assistant" not in existing.lower():
                    messages = [dict(m) for m in messages]
                    messages[0]["content"] = f"{_identity_anchor}\n\n{existing}"
            elif messages:
                messages = [{"role": "system", "content": _identity_anchor}] + [dict(m) for m in messages]

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
    def _flatten_messages(messages: List[Dict[str, Any]], model_name: Optional[str] = None) -> str:
        return format_chatml_messages(messages, model_name=model_name)

    @staticmethod
    def _normalize_tool_definitions_for_template(tools: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        normalized: List[Dict[str, Any]] = []
        for name, definition in list((tools or {}).items())[:20]:
            if not definition:
                continue
            if isinstance(definition, dict) and definition.get("type") == "function" and definition.get("function"):
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
    def _extract_tool_call_payload(response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None

        patterns = (
            re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
            re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL),
            re.compile(r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\}', re.DOTALL),
            re.compile(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL),
        )

        for pattern in patterns:
            match = pattern.search(response_text)
            if not match:
                continue
            candidate = match.group(1) if match.groups() else match.group(0)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if "tool" in payload and "args" in payload:
                return payload
            if "name" in payload and "arguments" in payload:
                args = payload.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"value": args}
                return {"tool": payload.get("name"), "args": args or {}}
        return None

    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """High-level generation endpoint with unified deadlines.

        Includes automatic retry on BrokenPipeError: if the worker process
        died between the alive-check and the queue write, we reboot and
        retry once before giving up.
        """
        request_is_background = bool(kwargs.pop("is_background", False))
        foreground_request = bool(kwargs.pop("foreground_request", False))
        if request_is_background:
            foreground_request = False
        owner_label = str(kwargs.pop("owner_label", os.path.basename(self.model_path)) or os.path.basename(self.model_path))
        deadline = kwargs.get("deadline")
        origin_label = str(kwargs.get("origin", "") or "")
        purpose_label = str(kwargs.get("purpose", "") or "")
        if (
            not request_is_background
            and not foreground_request
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
            background_origin = str(kwargs.get("origin", "") or owner_label or os.path.basename(self.model_path))
            background_deferral = _background_deferral_active(background_origin)
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Deferring background generation for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return None

        # ── PREVENTIVE: Memory pressure check before generation ──────
        # If RAM is critically low, reduce max_tokens to prevent OOM kill.
        # This is the #1 cause of cortex death on 64GB machines under load.
        try:
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 ** 3)
            # On 64GB+ machines, be much more lenient — the cortex needs room to work
            if total_gb >= 60:
                critical_pct = 92.0
                high_pct = 87.0
                gc_pct = 90.0
            else:
                critical_pct = 88.0
                high_pct = 80.0
                gc_pct = 82.0

            if vm.percent >= critical_pct:
                current_max = kwargs.get("max_tokens", self.max_tokens)
                kwargs["max_tokens"] = min(current_max, 128)
                logger.warning(
                    "[MLX] MEMORY PRESSURE (%.1f%%): Capping max_tokens to 128 to prevent OOM",
                    vm.percent,
                )
            elif vm.percent >= high_pct:
                current_max = kwargs.get("max_tokens", self.max_tokens)
                kwargs["max_tokens"] = min(current_max, 256)
            if vm.percent >= gc_pct:
                import gc
                gc.collect()
        except Exception:
            pass  # no-op: intentional

        acquired = await self._acquire_request_lock(
            owner_label=owner_label,
            deadline=deadline,
            foreground_request=foreground_request,
        )
        if not acquired:
            return None
        try:
            if foreground_request:
                try:
                    async with _foreground_owner_context(
                        owner_label,
                        deadline=deadline if isinstance(deadline, Deadline) else None,
                        foreground_request=True,
                        stale_after=self._first_token_sla(foreground_request=True),
                    ):
                        result = await self._generate_inner(
                            prompt,
                            _retry=True,
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            owner_label=owner_label,
                            **kwargs,
                        )
                except TimeoutError as exc:
                    logger.warning("⏸️ [MLX] %s", exc)
                    self._record_degraded_event(
                        "foreground_owner_timeout",
                        detail=f"{os.path.basename(self.model_path)}:{exc}",
                        severity="warning",
                        foreground_request=True,
                    )
                    return None
            else:
                result = await self._generate_inner(
                    prompt,
                    _retry=True,
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_label=owner_label,
                    **kwargs,
                )
            return result
        finally:
            _deferred_reboot = self._deferred_reboot_reason
            self._deferred_reboot_reason = None
            self._release_request_lock()
            # Reboot AFTER releasing _request_lock to avoid lock-ordering deadlock
            if _deferred_reboot:
                recoverable = str(_deferred_reboot).startswith("recoverable_")
                reboot_reason = str(_deferred_reboot).removeprefix("recoverable_")
                await self.reboot_worker(reason=reboot_reason, mark_failed=not recoverable)

    async def _generate_inner(
        self,
        prompt: str,
        _retry: bool = True,
        request_is_background: bool = False,
        foreground_request: bool = False,
        owner_label: str = "",
        **kwargs,
    ) -> Optional[str]:
        """Core generation logic, extracted for retry support."""
        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Skipping queued background generation for %s during foreground ownership.",
                os.path.basename(self.model_path),
            )
            return None
        if request_is_background:
            background_origin = owner_label or str(kwargs.get("origin", "") or os.path.basename(self.model_path))
            background_deferral = _background_deferral_active(background_origin)
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background generation for %s stopped before worker spawn (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return None

        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            is_heavy = any(k in self.model_path.lower() for k in ["72b", "32b", "zenith"])
            deadline = get_deadline(240.0 if is_heavy else 60.0)
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
        except asyncio.TimeoutError:
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
        try:
            from core.brain.latent_bridge import compute_inference_params
            _bridge = compute_inference_params(
                base_max_tokens=int(kwargs.get("max_tokens", self.max_tokens) or self.max_tokens),
                base_temperature=float(kwargs.get("temperature", kwargs.get("temp", self.temp)) or self.temp),
                foreground=bool(foreground_request),
            )
        except Exception as _bridge_exc:
            record_degradation('mlx_client', _bridge_exc)
            _bridge = None
            logger.debug("latent_bridge unavailable: %s", _bridge_exc)

        def _bridge_get(field: str, fallback: Any) -> Any:
            if _bridge is None:
                return fallback
            return getattr(_bridge, field, fallback)

        req_id = uuid.uuid4().hex
        req = {
            "id": req_id,
            "action": "generate",
            "prompt": prompt,
            "messages": kwargs.get("messages"),
            "tools": kwargs.get("tools"),
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
            # max_tokens is a *cap*: the bridge can shrink it (vitality
            # drops shorten output), but never expands what the caller
            # asked for.
            "max_tokens": _bounded_max_tokens(
                kwargs.get("max_tokens", self.max_tokens),
                _bridge_get("max_tokens", self.max_tokens),
                self.max_tokens,
            ),
            "schema": kwargs.get("schema"),
        }
        # Activation-steering offsets ride along when present; the worker
        # consumes them if its build supports residual-stream injection,
        # otherwise it ignores the field with no harm.
        if _bridge is not None and getattr(_bridge, "layer_offsets", None):
            req["layer_offsets"] = _bridge.layer_offsets
        if _bridge is not None and getattr(_bridge, "extra_stop_sequences", None):
            existing_stops = list(kwargs.get("stop_sequences") or [])
            existing_stops.extend(_bridge.extra_stop_sequences)
            req["stop_sequences"] = existing_stops

        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        self._current_gen_future = fut
        self._active_generations += 1
        self._mark_generation_started(
            req_id,
            prompt_chars=len(prompt or ""),
            requested_max_tokens=req.get("max_tokens", self.max_tokens),
        )
        enqueue_timeout = max(0.5, min(2.0, deadline.remaining or 2.0))
        try:
            await run_io_bound(self._req_q.put, req, True, enqueue_timeout)
        except (BrokenPipeError, OSError, Exception) as exc:
            self._pending_generations.pop(req_id, None)
            if _retry and ("Broken pipe" in str(exc) or isinstance(exc, BrokenPipeError)):
                logger.warning("🔄 [MLX] Broken pipe on %s — deferring reboot (lock held)",
                               os.path.basename(self.model_path))
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
            )
            if not res:
                return None
            if res.get("status") == "ok":
                text = res.get("text", "").strip()
                self._mark_progress()
                self._last_generation_completed_at = time.time()
                if not text:
                    # Empty generation: log as debug, not warning. During warmup
                    # or short max_tokens requests, empty output is NORMAL — the
                    # model compiled its Metal shaders successfully even if it
                    # didn't produce text. Don't crash the lane to "recovering".
                    is_warmup = getattr(self, "_warmup_in_flight", False)
                    if is_warmup:
                        logger.debug(
                            "MLX warmup produced empty text — benign (shader precompile succeeded)."
                        )
                        self._set_lane_state("ready")
                        return ""
                    self._record_degraded_event(
                        "empty_generation",
                        detail=os.path.basename(self.model_path),
                        severity="info",  # Downgraded from "warning"
                        foreground_request=foreground_request,
                    )
                    # Only crash to recovering for repeated empty generations,
                    # not a single occurrence
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
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            owner_label=owner_label,
                            **inline_kwargs,
                        )
                    if foreground_request:
                        self._deferred_reboot_reason = "recoverable_empty_generation"
                    if foreground_request and self._is_primary_or_deep_lane() and empty_count >= 3:
                        self._set_lane_state("recovering", "repeated_empty_generation")
                    return None
                self._consecutive_empty = 0
                self._set_lane_state("ready")
                self._consecutive_empty = 0  # Reset on successful generation
                # Record user-facing completions so the cross-client yield
                # loop knows to keep this worker warm across conversation
                # turns (see _ensure_worker_alive yield guard).
                if foreground_request:
                    self._last_user_facing_completed_at = time.time()
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
            expected_cancel_reason = self._consume_expected_generation_cancellation()
            if expected_cancel_reason:
                logger.info(
                    "🧹 [MLX] Generation cancelled for %s during expected reboot (%s).",
                    os.path.basename(self.model_path),
                    expected_cancel_reason,
                )
            else:
                logger.warning(
                    "🛑 [MLX] Generation cancelled for %s. Preserving worker unless it is unhealthy.",
                    os.path.basename(self.model_path),
                )
            self._pending_generations.pop(req_id, None)
            if (
                not expected_cancel_reason
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
            if not expected_cancel_reason and self._worker_unhealthy():
                self._deferred_reboot_reason = "cancelled_unhealthy"
            raise
        except asyncio.TimeoutError:
            logger.error("🛑 [MLX] Generation deadline reached for %s.", os.path.basename(self.model_path))
            self._pending_generations.pop(req_id, None)
            self._record_degraded_event(
                "generation_deadline_reached",
                detail=os.path.basename(self.model_path),
                severity="warning",
                foreground_request=foreground_request,
            )
            if self._worker_unhealthy(stale_after=self._stale_after(during_generation=True)):
                self._deferred_reboot_reason = "generation_timeout_unhealthy"
            else:
                logger.warning("⏳ [MLX] Deadline reached but worker still looks healthy; leaving lane warm.")
            return None
        finally:
            self._pending_generations.pop(req_id, None)
            if self._current_gen_future is fut:
                self._current_gen_future = None
            self._active_generations = max(0, self._active_generations - 1)
            if self._current_request_id == req_id:
                self._mark_generation_completed()

    async def think_and_act(
        self,
        objective: str,
        system_prompt: str,
        tools: Optional[Dict[str, Any]] = None,
        max_turns: int = 5,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """ReAct agentic loop: think → parse tool call → execute → repeat.

        Uses the model's native chat + tool template when available and falls
        back to a JSON-only tool-call contract otherwise. Results are fed back
        into the conversation history until the model produces a plain-text
        final answer or max_turns is exhausted.

        Returns:
            {"content": str, "turns": int, "tool_calls": List[Dict]}
        """
        template_tools = self._normalize_tool_definitions_for_template(tools)
        tool_block = ""
        if tools and not template_tools:
            tool_lines = []
            for name, defn in list(tools.items())[:20]:  # cap to avoid bloat
                desc = defn.get("description", "")
                params = defn.get("parameters", {}).get("properties", {})
                param_str = ", ".join(f'"{k}"' for k in params) if params else "none"
                tool_lines.append(f'  • {name}: {desc}  [params: {param_str}]')
            tool_block = (
                "\n\n## TOOLS AVAILABLE\n"
                + "\n".join(tool_lines)
                + "\n\nIf you need a tool and the model supports native tool calling, emit the native tool-call format only.\n"
                + "Otherwise output EXACTLY this on its own line (nothing else):\n"
                + '```json\n{"tool": "tool_name", "args": {"param": "value"}}\n```\n'
                + "When you have your final answer, respond normally — no JSON block."
            )

        augmented_system = system_prompt + tool_block
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": objective},
        ]
        tool_calls_made: List[Dict[str, Any]] = []
        last_response_text = ""

        for turn in range(max_turns):
            raw = await self.generate_text_async(
                "",
                messages=messages,
                tools=template_tools,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            if not raw or not raw.strip():
                break

            response_text = raw.strip()
            last_response_text = response_text

            tool_call = self._extract_tool_call_payload(response_text) if tools else None
            if not tool_call:
                return {
                    "content": response_text,
                    "turns": turn + 1,
                    "tool_calls": tool_calls_made,
                }

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # ── Execute the tool via FunctionCallingAdapter ───────────
            tool_result = f"[Tool '{tool_name}' not found]"
            try:
                from core.container import ServiceContainer
                adapter_or_cap = ServiceContainer.get("capability_engine", default=None)
                if adapter_or_cap:
                    raw_result = await adapter_or_cap.execute(
                        tool_name,
                        tool_args,
                        context or {"source": "think_and_act"},
                    )
                    if isinstance(raw_result, dict):
                        tool_result = json.dumps(raw_result, default=str)
                    else:
                        tool_result = str(raw_result)
            except Exception as exc:
                record_degradation('mlx_client', exc)
                tool_result = f"[Tool error: {exc}]"
                logger.warning("[think_and_act] Tool '%s' failed: %s", tool_name, exc)

            tool_calls_made.append({"tool": tool_name, "args": tool_args, "result": tool_result})
            logger.info("[think_and_act] turn=%d tool=%s ok", turn + 1, tool_name)

            # ── Feed result back into history ─────────────────────────
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),  # [STABILITY v53] Must be a JSON string, not a dict
                            }
                        }
                    ],
                }
            )
            
            # [STABILITY v53] Protect against massive tool outputs breaking context windows
            if len(tool_result) > 4000:
                tool_result = tool_result[:4000] + "\n\n...[OUTPUT TRUNCATED FOR LENGTH]..."
            
            messages.append({"role": "tool", "content": tool_result})

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
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                warmup_text = await asyncio.wait_for(
                    self._generate_inner(
                        "Hello",
                        _retry=True,
                        request_is_background=request_is_background,
                        foreground_request=False,
                        owner_label=owner_name,
                        max_tokens=1,
                    ),
                    timeout=warmup_timeout + (10.0 * attempt),
                )
                # max_tokens=1 against a fresh worker often returns "" (empty
                # generation) or None (no payload from the IPC channel before
                # Metal kernels finish compiling). Both states mean the same
                # thing operationally: the worker is alive, weights are
                # resident, and shader compile took place. Only a hard worker
                # death should fail warmup — verify with is_alive() instead of
                # treating None as fatal, which previously caused a reboot
                # loop on every cold start.
                if warmup_text is None and not self.is_alive():
                    raise RuntimeError("warmup_precompile_worker_dead")
                if not warmup_text or not str(warmup_text).strip():
                    logger.info(
                        "🔥 [MLX] Warmup produced no visible text for %s — worker is alive, treating shader precompile as successful.",
                        os.path.basename(self.model_path),
                    )
                self._set_lane_state("ready")
                logger.info("🔥 [MLX] Warmup complete — Metal shaders compiled.")
                return
            except Exception as exc:
                record_degradation('mlx_client', exc)
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "⚠️ [MLX] Warmup pre-compile failed once for %s: %s. Retrying cleanly...",
                        os.path.basename(self.model_path),
                        exc,
                    )
                    await asyncio.to_thread(gc.collect)
                    await self.reboot_worker(reason="warmup_precompile_retry", mark_failed=False)
                    await asyncio.sleep(1.0)
                    continue
                raise last_exc

    async def warmup(
        self,
        *,
        foreground_request: Optional[bool] = None,
        skip_swap_cooldown: bool = False,
    ):
        """Boot-time warmup: spawn worker + 1-token pre-compile."""
        if foreground_request is None:
            foreground_request = self._is_primary_or_deep_lane()
        else:
            foreground_request = bool(foreground_request)
        request_is_background = not foreground_request
        owner_name = f"warmup:{os.path.basename(self.model_path)}"
        warmup_timeout = self._warmup_timeout()
        self._warmup_attempted = True
        # [STABILITY v51] Stale-warmup circuit breaker: if _warmup_in_flight
        # has been True for >120s, the previous warmup task leaked without
        # clearing the flag. Force-clear before proceeding.
        if self._warmup_in_flight:
            elapsed_since_transition = time.time() - self._lane_transition_at
            if elapsed_since_transition > 120.0:
                logger.warning(
                    "🔧 [STABILITY] _warmup_in_flight was stuck True for %.0fs. "
                    "Force-clearing stale warmup flag.",
                    elapsed_since_transition,
                )
                self._warmup_in_flight = False
        self._warmup_in_flight = True
        self._set_lane_state("warming")
        try:
            if foreground_request:
                try:
                    async with _foreground_owner_context(
                        owner_name,
                        deadline=get_deadline(min(8.0, warmup_timeout)),
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
                            logger.info("⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path))
                            return

                        try:
                            await self._run_warmup_precompile(
                                request_is_background=request_is_background,
                                foreground_request=foreground_request,
                                owner_name=owner_name,
                                warmup_timeout=warmup_timeout,
                            )
                        except Exception as e:
                            record_degradation('mlx_client', e)
                            self._set_lane_state("recovering", f"warmup_precompile_failed:{type(e).__name__}")
                            self._record_degraded_event(
                                "warmup_precompile_failed",
                                detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                                severity="warning",
                                foreground_request=foreground_request,
                            )
                            logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                except TimeoutError as exc:
                    self._set_lane_state("recovering", "warmup_foreground_owner_timeout")
                    self._record_degraded_event(
                        "warmup_foreground_owner_timeout",
                        detail=f"{os.path.basename(self.model_path)}:{exc}",
                        severity="warning",
                        foreground_request=foreground_request,
                    )
                    logger.info("⏸️ [MLX] Warmup deferred for %s: %s", os.path.basename(self.model_path), exc)
                return

            alive = await self._ensure_worker_alive(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
            if not alive:
                if self._lane_state != "failed":
                    self._set_lane_state("recovering", "warmup_deferred")
                logger.info("⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path))
                return

            try:
                await self._run_warmup_precompile(
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_name=owner_name,
                    warmup_timeout=warmup_timeout,
                )
            except Exception as e:
                record_degradation('mlx_client', e)
                self._set_lane_state("recovering", f"warmup_precompile_failed:{type(e).__name__}")
                self._record_degraded_event(
                    "warmup_precompile_failed",
                    detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
        finally:
            self._warmup_in_flight = False

    async def warm_up(self, **kwargs):
        """Backward-compatible alias for older call sites."""
        return await self.warmup(**kwargs)

    async def reboot_worker(self, reason: str = "manual_reboot", mark_failed: bool = False):
        """Forcibly reboots the worker."""
        self._set_lane_state("recovering", reason)
        acquired = await asyncio.to_thread(self._lock.acquire, True, 10.0)
        if not acquired:
            logger.error("🚨 [MLX] DEADLOCK DETECTED: Could not acquire _lock for reboot on %s. Forcing reboot anyway to break deadlock.", os.path.basename(self.model_path))
        try:
            if self._process and self._process.is_alive():
                await asyncio.get_running_loop().run_in_executor(None, self._kill_and_join_blocking, self._process)
            self._process = None
            self._init_done = False
            self._last_heartbeat = 0.0
            self._last_progress_at = 0.0
            self._last_token_progress_at = 0.0
            # Reset the cold-start anchor so the next foreground request
            # gets the generous 40 s SLA instead of the tight warm-path 22 s.
            # A reboot means the worker process is gone → first-token budget
            # includes Metal shader recompile, KV rebuild, and weight reload.
            self._last_generation_completed_at = 0.0
            self._last_user_facing_completed_at = 0.0
            self._process_started_at = 0.0
            self._current_request_started_at = 0.0
            self._current_first_token_at = 0.0
            self._current_request_id = ""
            if self._listener_task:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None

            # [OOM FIX] Force memory reclaim after killing heavy model process
            gc.collect()

            # RECREATE QUEUES TO PREVENT ZOMBIE THREADS STEALING MESSAGES
            _safe_close_queue(self._req_q)
            _safe_close_queue(self._res_q)
            self._req_q = mp.Queue(maxsize=10)
            self._res_q = mp.Queue(maxsize=10)

            pending_futures = {
                id(future): future
                for future in list(self._pending_generations.values()) + [self._current_gen_future]
                if future is not None and not future.done()
            }
            if mark_failed:
                self._expected_cancel_reason = ""
                self._expected_cancel_budget = 0
                self._expected_cancel_recorded_at = 0.0
            elif pending_futures:
                self._note_expected_generation_cancellation(reason, count=len(pending_futures))

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
            self._pending_generations.clear()
            self._current_gen_future = None
            self._active_generations = 0
            if self._init_future is not None:
                _cancel_shared_future(self._init_future)
            self._init_future = None
            self._warmup_in_flight = False
            self._consecutive_empty = 0  # [STABILITY v53] Reset on reboot — prevents false recovery triggers
        finally:
            self._lock.release()
        self._set_lane_state("failed" if mark_failed else "cold", reason if mark_failed else "")

    def __del__(self):
        if self._process and self._process.is_alive():
            self._process.kill()

def get_mlx_client(model_path: Optional[str] = None, **kwargs) -> MLXLocalClient:
    """Compatibility factory for Aura's active local backend."""
    from .model_registry import get_local_backend, get_model_path, get_runtime_model_path

    if model_path is None:
        model_path = get_runtime_model_path()

    backend = get_local_backend()
    if backend != "mlx":
        from .local_server_client import get_local_server_client

        return get_local_server_client(model_path=model_path, **kwargs)

    resolved_model_path = str(get_model_path(model_path)).strip()
    path_candidate = Path(resolved_model_path).expanduser()
    if path_candidate.is_absolute() or path_candidate.exists():
        runtime_path = str(path_candidate.resolve() if path_candidate.exists() else path_candidate)
        client_key = os.path.realpath(runtime_path)
    else:
        runtime_path = resolved_model_path
        client_key = resolved_model_path

    if client_key not in _CLIENTS:
        _CLIENTS[client_key] = MLXLocalClient(model_path=runtime_path, **kwargs)
    return _CLIENTS[client_key]
