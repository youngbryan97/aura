from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import threading as _threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, Optional

import httpx
import psutil

from core.utils.deadlines import Deadline, get_deadline

from .chat_format import format_chatml_messages, format_chatml_prompt
from .model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
    find_llama_server_bin,
    get_endpoint_name_for_model,
    get_local_backend,
    resolve_personality_adapter,
)

logger = logging.getLogger("LLM.LocalRuntime")

_DEFAULT_PORTS = {
    PRIMARY_ENDPOINT: int(os.getenv("AURA_CORTEX_PORT", "11435")),
    DEEP_ENDPOINT: int(os.getenv("AURA_SOLVER_PORT", "11436")),
    BRAINSTEM_ENDPOINT: int(os.getenv("AURA_BRAINSTEM_PORT", "11437")),
    FALLBACK_ENDPOINT: int(os.getenv("AURA_REFLEX_PORT", "11438")),
}
_RUNTIME_SLOT_LOCK = _threading.Lock()


def _parallel_lane_runtime_allowed() -> bool:
    """Allow Cortex + Brainstem co-residency on roomy machines unless overridden.

    Only lightweight lanes (Brainstem, Reflex) may co-reside with one heavy
    lane (Cortex OR Solver).  Two heavy lanes must never co-reside — that is
    enforced separately in ``_yield_runtime_slot``.
    """
    setting = str(os.getenv("AURA_LOCAL_RUNTIME_SINGLETON", "auto")).strip().lower()
    if setting in {"1", "true", "yes", "on"}:
        return False
    if setting in {"0", "false", "no", "off"}:
        return True

    try:
        vm = psutil.virtual_memory()
        total_gb = vm.total / float(1024 ** 3)
        available_gb = vm.available / float(1024 ** 3)
        min_total_gb = float(os.getenv("AURA_PARALLEL_LANE_MIN_TOTAL_GB", "48"))
        max_pressure = float(os.getenv("AURA_PARALLEL_LANE_MAX_PRESSURE_PCT", "80"))
        min_available_gb = float(os.getenv("AURA_PARALLEL_LANE_MIN_AVAILABLE_GB", "12"))
        return bool(
            total_gb >= min_total_gb
            and vm.percent < max_pressure
            and available_gb >= min_available_gb
        )
    except Exception:
        return False


def _normalize_model_identity(value: str) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".gguf"):
        text = text[:-5]
    return text


@contextlib.asynccontextmanager
async def _thread_lock_context(
    lock: Any,
    *,
    timeout: Optional[float] = None,
    label: str = "lock",
):
    """Acquire a thread lock without parking executor threads behind cancellations."""
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + max(0.0, float(timeout))
    acquired = False
    while True:
        acquired = bool(lock.acquire(False))
        if acquired:
            break
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                break
            await asyncio.sleep(min(0.05, remaining))
        else:
            await asyncio.sleep(0.05)
    if not acquired:
        raise TimeoutError(f"{label}_timeout")
    try:
        yield
    finally:
        try:
            lock.release()
        except RuntimeError:
            logger.debug("Loop-agnostic lock %s was already released.", label)


def _readable_lane_name(model_path: str) -> str:
    model_name = Path(model_path).name
    if not model_name.lower().endswith(".gguf"):
        model_name = Path(model_path).stem
    return get_endpoint_name_for_model(model_name)


def _lane_port(model_path: str) -> int:
    return _DEFAULT_PORTS.get(_readable_lane_name(model_path), 11435)


def _lane_context_size(model_path: str) -> int:
    lane = _readable_lane_name(model_path)
    if lane == DEEP_ENDPOINT:
        return int(os.getenv("AURA_SOLVER_CTX", "8192"))
    if lane == PRIMARY_ENDPOINT:
        # 72B Q4: 8K context balances quality and VRAM; 16K for 32B if swapped back
        return int(os.getenv("AURA_CORTEX_CTX", "8192"))
    if lane == BRAINSTEM_ENDPOINT:
        return int(os.getenv("AURA_BRAINSTEM_CTX", "8192"))
    return int(os.getenv("AURA_REFLEX_CTX", "4096"))


def _lane_gpu_layers(model_path: str) -> str:
    lane = _readable_lane_name(model_path)
    if lane == FALLBACK_ENDPOINT:
        return os.getenv("AURA_REFLEX_GPU_LAYERS", "0")
    return os.getenv("AURA_LOCAL_GPU_LAYERS", "99")


class LocalServerClient:
    """Managed local OpenAI-compatible server client.

    This preserves the shape Aura expects from the old MLX local client while
    moving inference to a separate local runtime boundary.
    """

    def __init__(self, model_path: str, device: str = "gpu", max_tokens: int = 512):
        self.model_path = str(model_path)
        self.device = device
        self.max_tokens = max_tokens
        self.temp = 0.7
        self.top_p = 0.9
        self.model = None
        self.tokenizer = None

        self._process: Optional[subprocess.Popen] = None
        self._log_handle = None
        # These clients are created during boot but reused from the API server's
        # event loop, so asyncio locks will eventually explode with
        # "bound to a different event loop".
        self._spawn_lock = _threading.Lock()
        self._request_lock = _threading.Lock()
        self._http: Optional[httpx.AsyncClient] = None
        self._init_future: Optional[asyncio.Task] = None
        self._lane_state = "cold"
        self._lane_error = ""
        self._lane_transition_at = time.time()
        self._last_ready_at = 0.0
        self._last_progress_at = 0.0
        self._last_generation_completed_at = 0.0
        self._warmup_attempted = False
        self._warmup_in_flight = False
        self._consecutive_empty = 0
        self._process_started_at = 0.0
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._adapter_path = None
        self._detected_runtime_models: list[str] = []
        self._runtime_identity_ok = False

        self._backend = get_local_backend()
        self._port = _lane_port(self.model_path)
        self._lane_name = _readable_lane_name(self.model_path)
        self._runtime_url = self._resolve_runtime_url()
        self._runtime_model = self._resolve_runtime_model_name()
        self._external_only = self._backend in {"openai_compat", "ollama"}

    def _resolve_runtime_url(self) -> str:
        lane = self._lane_name.upper()
        explicit = os.getenv(f"AURA_{lane}_URL")
        if explicit:
            return explicit.rstrip("/")

        if self._backend == "ollama":
            return os.getenv("AURA_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        if self._backend == "openai_compat":
            return os.getenv("AURA_LOCAL_LLM_URL", "http://127.0.0.1:11435").rstrip("/")
        return f"http://127.0.0.1:{self._port}"

    @staticmethod
    def _sanitize_message(message: Dict[str, Any]) -> Dict[str, Any]:
        role = str(message.get("role", "user") or "user")
        content = message.get("content", "")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False, default=str)
            except Exception:
                content = str(content)

        payload: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        for optional_key in ("name", "tool_call_id"):
            value = message.get(optional_key)
            if value is not None:
                payload[optional_key] = str(value)
        return payload

    @classmethod
    def _coerce_response_text(cls, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            parts = [cls._coerce_response_text(item) for item in payload]
            return "".join(part for part in parts if part)
        if isinstance(payload, dict):
            for key in (
                "content",
                "text",
                "reasoning_content",
                "reasoning",
                "output_text",
                "generated_text",
                "value",
            ):
                value = payload.get(key)
                text = cls._coerce_response_text(value)
                if text.strip():
                    return text
            return ""
        return str(payload)

    @classmethod
    def _extract_response_text(cls, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            candidates = [
                message,
                choice.get("text"),
                choice.get("delta"),
                choice.get("content"),
                data.get("response"),
                data.get("output_text"),
                data.get("generated_text"),
            ]
            for candidate in candidates:
                text = cls._coerce_response_text(candidate).strip()
                if text:
                    return text
        return ""

    def _resolve_runtime_model_name(self) -> str:
        lane = self._lane_name.upper()
        explicit = os.getenv(f"AURA_{lane}_MODEL_ID")
        if explicit:
            return explicit
        if self._backend == "ollama":
            return os.getenv("AURA_LOCAL_OLLAMA_MODEL", Path(self.model_path).stem)
        return Path(self.model_path).stem

    @staticmethod
    def _estimate_message_tokens(message: Dict[str, Any]) -> int:
        content = str((message or {}).get("content", "") or "")
        return max(8, (len(content) // 4) + 8)

    def _fit_messages_to_context(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        max_tokens: int,
    ) -> list[Dict[str, Any]]:
        fitted = [dict(msg) for msg in list(messages or []) if isinstance(msg, dict)]
        if not fitted:
            return fitted

        context_limit = int(_lane_context_size(self.model_path) or 4096)
        token_budget = max(256, context_limit - max(int(max_tokens or 0), 128) - 256)
        total_tokens = sum(self._estimate_message_tokens(msg) for msg in fitted)
        if total_tokens <= token_budget:
            return fitted

        system_msg = fitted[0] if fitted and fitted[0].get("role") == "system" else None
        body = fitted[1:] if system_msg else fitted[:]

        while body and (sum(self._estimate_message_tokens(msg) for msg in ([system_msg] if system_msg else []) + body) > token_budget):
            body.pop(0)

        if system_msg:
            remaining_budget = token_budget - sum(self._estimate_message_tokens(msg) for msg in body)
            allowed_chars = max(256, (remaining_budget - 8) * 4)
            content = str(system_msg.get("content", "") or "")
            if len(content) > allowed_chars:
                system_msg = {**system_msg, "content": content[:allowed_chars]}
            fitted = [system_msg, *body]
        else:
            fitted = body

        logger.debug(
            "✂️ [%s] Trimmed prompt payload from ~%d to ~%d tokens for local runtime stability.",
            self._lane_name,
            total_tokens,
            sum(self._estimate_message_tokens(msg) for msg in fitted),
        )
        return fitted

    def _set_lane_state(self, state: str, error: str = "") -> None:
        if state != self._lane_state:
            self._lane_transition_at = time.time()
        self._lane_state = state
        if error:
            self._lane_error = str(error)
        elif state == "ready":
            self._lane_error = ""

    def _mark_progress(self) -> None:
        self._last_progress_at = time.time()

    def _mark_generation_started(self) -> None:
        now = time.time()
        self._current_request_started_at = now
        self._current_first_token_at = 0.0
        self._mark_progress()

    def _mark_generation_completed(self) -> None:
        now = time.time()
        self._last_generation_completed_at = now
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._mark_progress()

    def _is_primary_or_deep_lane(self) -> bool:
        return self._lane_name in {PRIMARY_ENDPOINT, DEEP_ENDPOINT}

    def _warmup_timeout(self) -> float:
        return 180.0 if self._is_primary_or_deep_lane() else 120.0

    def _startup_timeout(self) -> float:
        return 360.0 if self._is_primary_or_deep_lane() else 180.0

    def _lock_timeout(
        self,
        *,
        deadline: Optional[Deadline],
        default: float,
        minimum: float,
    ) -> float:
        if isinstance(deadline, Deadline) and deadline.remaining is not None:
            remaining = float(deadline.remaining)
            # [STABILITY v53] Respect the deadline even if it's below minimum.
            # Previously, min floor could exceed the request's actual deadline,
            # causing locks to hold past the caller's timeout budget.
            if remaining <= 0.5:
                return max(0.5, remaining)
            return max(minimum, min(remaining, default))
        return max(minimum, default)

    def _is_runtime_resident(self) -> bool:
        proc_alive = self._process is not None and self._process.poll() is None
        return proc_alive or self._lane_state in {"spawning", "handshaking", "warming", "ready", "recovering"}

    def _request_scoped_init_timeout(
        self,
        deadline: Optional[Deadline],
        *,
        foreground_request: bool,
    ) -> tuple[float, bool]:
        full_timeout = self._startup_timeout()
        if not isinstance(deadline, Deadline):
            return full_timeout, False

        remaining = deadline.remaining
        if remaining is None:
            return full_timeout, False

        reserve = 5.0 if foreground_request else 2.0
        scoped_timeout = max(0.25, remaining - reserve)
        return min(full_timeout, scoped_timeout), scoped_timeout < full_timeout

    def _classify_failure(self, *, foreground_request: bool = False) -> str:
        if foreground_request:
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
                "local_runtime_client",
                reason,
                detail=detail,
                severity=severity,
                classification=self._classify_failure(foreground_request=foreground_request),
                context={
                    "model": Path(self.model_path).name,
                    "lane": self._lane_name,
                    "backend": self._backend,
                    "lane_state": self._lane_state,
                    "warmup_in_flight": self._warmup_in_flight,
                },
            )
        except Exception as exc:
            record_degradation('local_server_client', exc)
            logger.debug("Failed to record degraded event for %s: %s", self._lane_name, exc)

    def get_lane_status(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "expected_model": self._runtime_model,
            "state": self._lane_state,
            "last_error": self._lane_error,
            "conversation_ready": self._lane_state == "ready" and self.is_alive(),
            "last_ready_at": self._last_ready_at,
            "last_progress_at": self._last_progress_at,
            "last_generation_completed_at": self._last_generation_completed_at,
            "last_transition_at": self._lane_transition_at,
            "warmup_attempted": self._warmup_attempted,
            "warmup_in_flight": self._warmup_in_flight,
            "detected_models": list(self._detected_runtime_models),
            "runtime_identity_ok": self._runtime_identity_ok,
            "process_started_at": self._process_started_at,
            "current_request_started_at": self._current_request_started_at,
            "current_first_token_at": self._current_first_token_at,
        }

    def get_supervision_status(self) -> Dict[str, Any]:
        now = time.time()
        idle_anchor = max(self._last_generation_completed_at, self._last_ready_at, self._last_progress_at)
        return {
            "lane": self._lane_name,
            "state": self._lane_state,
            "alive": self.is_alive(),
            "process_uptime_s": max(0.0, now - self._process_started_at) if self._process_started_at else 0.0,
            "request_age_s": max(0.0, now - self._current_request_started_at) if self._current_request_started_at else 0.0,
            "time_to_first_token_s": (
                max(0.0, self._current_first_token_at - self._current_request_started_at)
                if self._current_request_started_at and self._current_first_token_at
                else None
            ),
            "idle_for_s": max(0.0, now - idle_anchor) if idle_anchor > 0.0 else 0.0,
        }

    def should_recycle_for_fragmentation(
        self,
        *,
        max_uptime_s: float = 5400.0,
        min_idle_s: float = 900.0,
    ) -> bool:
        if not self.is_alive() or self._current_request_started_at > 0.0:
            return False
        if self._process_started_at <= 0.0:
            return False
        idle_anchor = max(self._last_generation_completed_at, self._last_ready_at, self._last_progress_at)
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

    def note_lane_failed(self, reason: str) -> None:
        self._warmup_in_flight = False
        self._set_lane_state("failed", reason)

    def is_alive(self) -> bool:
        """Check if the inference server is actually responding.

        The old check trusted _lane_state, which could get stuck at 'failed'
        even when the server had recovered. Now we do a real HTTP health check
        as a fallback when the state says dead but the process is running.
        """
        if self._external_only:
            if self._lane_state == "ready":
                return True
            # Server might have recovered — do a real health check
            return self._http_health_check()

        process_running = self._process is not None and self._process.poll() is None
        if process_running and self._lane_state == "ready":
            return True

        # Managed llama.cpp lanes can remain alive across desktop restarts, which
        # means Aura may reconnect to a healthy reserved port without owning the
        # subprocess handle that originally launched it.
        known_identity_mismatch = (not self._runtime_identity_ok) and bool(self._detected_runtime_models)
        if not known_identity_mismatch and self._http_health_check():
            if self._lane_state != "ready":
                logger.info(
                    "[%s] is_alive: runtime healthy with lane_state='%s'. Repairing to 'ready'.",
                    self._lane_name,
                    self._lane_state,
                )
                self._set_lane_state("ready")
            return True
        return False

    def _http_health_check(self) -> bool:
        """Quick synchronous HTTP health check to the inference server."""
        try:
            import urllib.request
            url = f"{self._runtime_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = resp.read().decode()
                    return '"ok"' in data or '"status":"ok"' in data.replace(" ", "")
        except Exception:
            pass
        return False

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=None)
        return self._http

    def _resolve_llama_server_bin(self) -> Optional[str]:
        return find_llama_server_bin()

    def _log_path(self) -> Path:
        candidate_dirs = []
        try:
            from core.config import config

            candidate_dirs.append(Path(config.paths.home_dir) / "logs")
        except Exception as _exc:
            record_degradation('local_server_client', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        candidate_dirs.append(Path(__file__).resolve().parents[3] / ".aura_runtime" / "logs")
        candidate_dirs.append(Path.home() / ".aura" / "logs")

        for log_dir in candidate_dirs:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                probe = log_dir / ".write_probe"
                atomic_write_text(probe, "ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
                return log_dir / f"local-runtime-{self._lane_name.lower()}.log"
            except Exception:
                continue
        raise PermissionError("No writable local runtime log directory is available.")

    def _spawn_server_blocking(self) -> subprocess.Popen:
        binary = self._resolve_llama_server_bin()
        if not binary:
            raise RuntimeError("local_runtime_unavailable:llama-server-missing")
        if not Path(self.model_path).exists():
            raise RuntimeError(f"local_runtime_unavailable:model_missing:{self.model_path}")

        if self._log_handle is None or self._log_handle.closed:
            self._log_handle = self._log_path().open("a", encoding="utf-8")

        prompt_cache_enabled = str(os.getenv("AURA_LOCAL_PROMPT_CACHE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        cache_ram_mib = str(os.getenv("AURA_LOCAL_CACHE_RAM_MIB", "256") or "256").strip()
        parallel_slots = str(os.getenv("AURA_LOCAL_PARALLEL_SLOTS", "1") or "1").strip()
        cmd = [
            binary,
            "-m",
            self.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--ctx-size",
            str(_lane_context_size(self.model_path)),
            "--jinja",
            "-ngl",
            _lane_gpu_layers(self.model_path),
            # Performance: Flash Attention + quantized KV cache + larger batch
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "-b", "2048",
            "-ub", "512",
            "--parallel", parallel_slots,
            "--cache-ram", cache_ram_mib,
        ]
        cmd.append("--cache-prompt" if prompt_cache_enabled else "--no-cache-prompt")

        # Only attach the GGUF LoRA when it matches this runtime model/lane.
        lora_path = resolve_personality_adapter(self._runtime_model or self.model_path, backend="gguf")
        if lora_path and os.path.isfile(lora_path):
            cmd.extend(["--lora", lora_path])
            logger.info("🧠 [%s] Loading personality LoRA adapter: %s", self._lane_name, lora_path)

        logger.info("📡 [%s] Spawning local runtime: %s", self._lane_name, " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        self._process_started_at = time.time()
        return process

    def _runtime_identity_matches(self, model_id: str) -> bool:
        candidate = _normalize_model_identity(model_id)
        expected = _normalize_model_identity(self._runtime_model)
        if not candidate:
            return False
        if expected and candidate == expected:
            return True
        return get_endpoint_name_for_model(candidate) == self._lane_name

    @staticmethod
    def _extract_runtime_model_ids(payload: Dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for item in list(payload.get("data") or []):
            if isinstance(item, dict):
                for key in ("id", "model", "name"):
                    value = item.get(key)
                    if value:
                        ids.append(str(value))
        for item in list(payload.get("models") or []):
            if isinstance(item, dict):
                for key in ("model", "name", "id"):
                    value = item.get(key)
                    if value:
                        ids.append(str(value))
        deduped: list[str] = []
        seen = set()
        for value in ids:
            normalized = _normalize_model_identity(value)
            if normalized and normalized not in seen:
                deduped.append(value)
                seen.add(normalized)
        return deduped

    def _reclaim_runtime_port_blocking(self) -> bool:
        reclaimed = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr.port != self._port:
                        continue
                    logger.warning(
                        "🧹 [%s] Reclaiming reserved lane port %s from PID %s (%s).",
                        self._lane_name,
                        self._port,
                        proc.pid,
                        proc.name(),
                    )
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    reclaimed = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return reclaimed

    async def _server_healthy(self) -> tuple[bool, bool]:
        client = await self._client()
        try:
            response = await client.get(f"{self._runtime_url}/health", timeout=5.0)
            if response.status_code != 200:
                return False, False
        except Exception:
            return False, False

        if self._external_only:
            self._runtime_identity_ok = True
            self._detected_runtime_models = []
            return True, False

        try:
            response = await client.get(f"{self._runtime_url}/v1/models", timeout=5.0)
            if response.status_code != 200:
                self._runtime_identity_ok = False
                self._detected_runtime_models = []
                return False, False
            payload = response.json()
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            self._runtime_identity_ok = False
            self._detected_runtime_models = []
            return False, False

        model_ids = self._extract_runtime_model_ids(payload)
        self._detected_runtime_models = model_ids[:4]
        if not model_ids:
            self._runtime_identity_ok = True
            return True, False

        self._runtime_identity_ok = any(self._runtime_identity_matches(model_id) for model_id in model_ids)
        if self._runtime_identity_ok:
            return True, False

        logger.warning(
            "⚠️ [%s] Runtime identity mismatch on port %s. Expected %s, observed %s.",
            self._lane_name,
            self._port,
            self._runtime_model,
            ", ".join(model_ids[:3]) or "unknown",
        )
        self._lane_error = f"runtime_model_mismatch:{','.join(model_ids[:2]) or 'unknown'}"
        return False, True

    async def _ensure_runtime_ready(
        self,
        *,
        deadline: Optional[Deadline] = None,
        foreground_request: bool = False,
    ) -> bool:
        init_timeout, _soft_timeout = self._request_scoped_init_timeout(
            deadline,
            foreground_request=foreground_request,
        )
        slot_timeout = self._lock_timeout(
            deadline=deadline,
            default=max(init_timeout, 45.0),
            minimum=15.0,
        )
        spawn_timeout = self._lock_timeout(
            deadline=deadline,
            default=min(max(init_timeout, 20.0), 90.0),
            minimum=10.0,
        )

        try:
            async with _thread_lock_context(
                _RUNTIME_SLOT_LOCK,
                timeout=slot_timeout,
                label=f"{self._lane_name}_runtime_slot_lock",
            ):
                async with _thread_lock_context(
                    self._spawn_lock,
                    timeout=spawn_timeout,
                    label=f"{self._lane_name}_spawn_lock",
                ):
                    server_healthy, identity_mismatch = await self._server_healthy()
                    if server_healthy:
                        self._set_lane_state("ready")
                        self._last_ready_at = time.time()
                        if self._process_started_at <= 0.0:
                            self._process_started_at = time.time()
                        return True

                    if identity_mismatch and not self._external_only:
                        await asyncio.to_thread(self._reclaim_runtime_port_blocking)
                        server_healthy, _identity_mismatch = await self._server_healthy()
                        if server_healthy:
                            self._set_lane_state("ready")
                            self._last_ready_at = time.time()
                            if self._process_started_at <= 0.0:
                                self._process_started_at = time.time()
                            return True

                    if not self._external_only:
                        if not await self._yield_runtime_slot(foreground_request=foreground_request):
                            return False

                    self._set_lane_state("spawning")
                    if self._external_only:
                        started_at = time.monotonic()
                        while (time.monotonic() - started_at) < init_timeout:
                            server_healthy, _identity_mismatch = await self._server_healthy()
                            if server_healthy:
                                self._set_lane_state("ready")
                                self._last_ready_at = time.time()
                                if self._process_started_at <= 0.0:
                                    self._process_started_at = time.time()
                                return True
                            await asyncio.sleep(0.5)
                        self.note_lane_failed("local_runtime_unavailable:server_unreachable")
                        return False

                    if self._process is None or self._process.poll() is not None:
                        self._process = await asyncio.to_thread(self._spawn_server_blocking)

                    started_at = time.monotonic()
                    while (time.monotonic() - started_at) < init_timeout:
                        if self._process and self._process.poll() is not None:
                            self.note_lane_failed(
                                f"local_runtime_unavailable:exit_{self._process.returncode}"
                            )
                            return False
                        server_healthy, _identity_mismatch = await self._server_healthy()
                        if server_healthy:
                            self._set_lane_state("handshaking")
                            self._last_ready_at = time.time()
                            if self._process_started_at <= 0.0:
                                self._process_started_at = time.time()
                            return True
                        await asyncio.sleep(0.5)

                    self.note_lane_recovering("runtime_start_timeout")
                    return False
        except TimeoutError as exc:
            self.note_lane_recovering(str(exc))
            return False

    async def _yield_runtime_slot(self, *, foreground_request: bool) -> bool:
        conflicting_clients = []
        allow_parallel_lightweight = _parallel_lane_runtime_allowed()

        # ── RAM-pressure forced eviction ──────────────────────────────
        # If memory pressure is critical, force-evict ALL non-self heavy
        # lanes regardless of the parallel lightweight setting.  This
        # prevents the 72B + 32B + 7B triple-residency explosion.
        ram_pressure_critical = False
        try:
            vm = psutil.virtual_memory()
            ram_pressure_critical = vm.percent >= 85 or (vm.available / float(1024 ** 3)) < 8
        except Exception:
            pass

        for client in _SERVER_CLIENTS.values():
            if client is self or client._external_only:
                continue
            if not client._is_runtime_resident():
                continue
            if self._is_primary_or_deep_lane() and client._is_primary_or_deep_lane():
                conflicting_clients.append(client)
                continue
            if ram_pressure_critical and client._is_primary_or_deep_lane():
                # Under RAM pressure, even lightweight lanes requesting
                # startup should evict any heavy non-self lane.
                conflicting_clients.append(client)
                continue
            if not allow_parallel_lightweight:
                conflicting_clients.append(client)

        if not conflicting_clients:
            return True

        heavy_resident = any(client._is_primary_or_deep_lane() for client in conflicting_clients)
        if not foreground_request and heavy_resident:
            logger.info(
                "⏸️ [%s] Deferring background runtime startup while a heavyweight lane is resident.",
                self._lane_name,
            )
            self._set_lane_state("cold")
            self._lane_error = "background_deferred:foreground_reserved"
            return False

        for client in conflicting_clients:
            logger.info(
                "🧹 [%s] Releasing runtime slot held by %s before startup.%s",
                self._lane_name,
                client._lane_name,
                " (RAM pressure forced)" if ram_pressure_critical else "",
            )
            try:
                async with _thread_lock_context(
                    client._request_lock,
                    timeout=20.0,
                    label=f"{client._lane_name}_request_lock",
                ):
                    await client.reboot_worker(reason=f"yield_to:{self._lane_name}", mark_failed=False)
            except TimeoutError:
                logger.warning(
                    "⏸️ [%s] Timed out waiting for %s request lock while yielding the runtime slot.",
                    self._lane_name,
                    client._lane_name,
                )
                client.note_lane_recovering(f"yield_lock_timeout:{self._lane_name}")
                return False
        return True

    async def _restart_server(self):
        """Kill and restart the llama-server process to recover from compute errors."""
        logger.warning("[%s] Restarting server due to compute error...", self._lane_name)
        try:
            if self._process and self._process.poll() is None:
                self._process.kill()
                await asyncio.to_thread(self._process.wait, 5.0)
                logger.info("[%s] Old server process killed.", self._lane_name)
        except Exception as e:
            record_degradation('local_server_client', e)
            logger.debug("[%s] Kill failed: %s", self._lane_name, e)

        self._process = None
        self._lane_state = "cold"
        self._warmup_attempted = False

        # Re-warmup
        try:
            await self.warmup()
            logger.info("[%s] Server restarted successfully.", self._lane_name)
        except Exception as e:
            record_degradation('local_server_client', e)
            logger.error("[%s] Server restart failed: %s", self._lane_name, e)

    async def warmup(self, *, foreground_request: Optional[bool] = None):
        self._warmup_attempted = True
        self._warmup_in_flight = True
        self._set_lane_state("warming")
        request_is_foreground = (
            self._is_primary_or_deep_lane()
            if foreground_request is None
            else bool(foreground_request)
        )
        try:
            ready = await self._ensure_runtime_ready(
                deadline=get_deadline(self._warmup_timeout()),
                foreground_request=request_is_foreground,
            )
            if not ready:
                return

            # Compile the chat path before the first user-facing turn.
            text = await self.generate_text_async(
                prompt="Warm up Aura's local response lane.",
                messages=[
                    {"role": "system", "content": "You are Aura. Reply with a single short token: ready"},
                    {"role": "user", "content": "ready?"},
                ],
                max_tokens=8,
                temp=0.0,
                foreground_request=request_is_foreground,
                owner_label=f"warmup:{self._lane_name}",
            )
            if text is not None:
                self._set_lane_state("ready")
                self._last_ready_at = time.time()
                logger.info("✅ [%s] Local runtime warmup complete.", self._lane_name)
            else:
                self.note_lane_recovering("warmup_no_text")
        finally:
            self._warmup_in_flight = False

    async def warm_up(self, **kwargs):
        return await self.warmup(**kwargs)

    async def reboot_worker(self, reason: str = "manual_reboot", mark_failed: bool = False):
        self._set_lane_state("recovering", reason)
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as _exc:
                record_degradation('local_server_client', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
            self._http = None

        proc = self._process
        self._process = None
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                await asyncio.to_thread(proc.wait, 5.0)
            except Exception as _exc:
                record_degradation('local_server_client', _exc)
                logger.debug("Suppressed Exception: %s", _exc)

        # ── Orphan Reclamation ─────────────────────────────────────────
        # After a desktop restart, the old llama-server may survive on
        # our reserved port with no subprocess handle to kill.  Force-
        # reclaim the port so the memory is actually freed.
        try:
            reclaimed = await asyncio.to_thread(self._reclaim_runtime_port_blocking)
            if reclaimed:
                logger.info(
                    "🧹 [%s] reboot_worker: reclaimed orphaned process on port %s.",
                    self._lane_name,
                    self._port,
                )
        except Exception as _exc:
            record_degradation('local_server_client', _exc)
            logger.debug("reboot_worker: port reclamation failed: %s", _exc)

        if self._log_handle is not None and not self._log_handle.closed:
            try:
                self._log_handle.flush()
            except Exception as _exc:
                record_degradation('local_server_client', _exc)
                logger.debug("Suppressed Exception: %s", _exc)

        self._warmup_in_flight = False
        self._consecutive_empty = 0
        self._process_started_at = 0.0
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._last_generation_completed_at = 0.0
        self._set_lane_state("failed" if mark_failed else "cold", reason if mark_failed else "")

    async def generate_text_async(self, prompt: str, **kwargs) -> Optional[str]:
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", None)
        deadline = kwargs.get("deadline")
        foreground_request = bool(kwargs.get("foreground_request", False))
        owner_label = str(kwargs.get("owner_label", self._lane_name) or self._lane_name)
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens) or self.max_tokens)
        temperature = float(kwargs.get("temp", kwargs.get("temperature", self.temp)) or self.temp)
        schema = kwargs.get("schema")

        # ── Latent-space bridge: substrate-driven sampling for the
        # external/cloud lane too. Caller-supplied values still win;
        # the bridge fills any field the caller didn't pin and caps
        # max_tokens downward when vitality is low. The cloud lane
        # therefore exhibits the same affect-driven sampling shape
        # as the local MLX lane — one substrate, one sampling.
        try:
            from core.brain.latent_bridge import compute_inference_params as _bridge_params
            _bridge = _bridge_params(
                base_max_tokens=max_tokens,
                base_temperature=temperature,
                foreground=foreground_request,
            )
            max_tokens = min(max_tokens, _bridge.max_tokens)
            if "temp" not in kwargs and "temperature" not in kwargs:
                temperature = _bridge.temperature
            kwargs.setdefault("top_p", _bridge.top_p)
            kwargs.setdefault("top_k", _bridge.top_k)
            kwargs.setdefault("repetition_penalty", _bridge.repetition_penalty)
            kwargs.setdefault("presence_penalty", _bridge.presence_penalty)
            if _bridge.extra_stop_sequences:
                stops = list(kwargs.get("stop_sequences") or [])
                stops.extend(_bridge.extra_stop_sequences)
                kwargs["stop_sequences"] = stops
        except Exception as _bexc:
            record_degradation('local_server_client', _bexc)
            pass

        if messages and isinstance(messages, list):
            payload_messages = [
                self._sanitize_message(msg)
                for msg in messages
                if isinstance(msg, dict)
            ]
        else:
            if system_prompt:
                prompt = format_chatml_prompt(
                    prompt,
                    system_prompt=system_prompt,
                    model_name=self._runtime_model or self.model_path,
                )
            payload_messages = [{"role": "user", "content": prompt}]

        # [STABILITY v53] Identity reinforcement at the message level.
        # Ensure the system message contains Aura's core identity anchor so the
        # base model's RLHF training can't override it. This is the deepest level
        # we can inject identity before the model sees the conversation.
        if payload_messages and foreground_request:
            _identity_anchor = (
                "You are Aura Luna. Speak as yourself, not as an assistant. "
                "Stay grounded and concrete in ordinary conversation. "
                "NEVER say 'as an AI', 'I don't have opinions', 'I'm programmed to', "
                "'I was designed to', 'I was created to', 'my training allows', "
                "or end with 'How can I help?' or 'Is there anything else?'. "
                "Do not invent physical settings, ominous atmosphere, or symbolic scenes unless the user asked for fiction or already supplied that setting. "
                "Speak as yourself, plainly and directly."
            )
            if payload_messages[0].get("role") == "system":
                existing = str(payload_messages[0].get("content", "") or "")
                if "sovereign" not in existing.lower() and "not an assistant" not in existing.lower():
                    payload_messages[0]["content"] = f"{_identity_anchor}\n\n{existing}"
            else:
                payload_messages.insert(0, {"role": "system", "content": _identity_anchor})

        payload_messages = self._fit_messages_to_context(payload_messages, max_tokens=max_tokens)

        if not await self._ensure_runtime_ready(
            deadline=deadline if isinstance(deadline, Deadline) else None,
            foreground_request=foreground_request,
        ):
            return None

        request_timeout = self._lock_timeout(
            deadline=deadline if isinstance(deadline, Deadline) else None,
            default=20.0 if foreground_request else 10.0,
            minimum=5.0,
        )
        try:
            async with _thread_lock_context(
                self._request_lock,
                timeout=request_timeout,
                label=f"{self._lane_name}_request_lock",
            ):
                if foreground_request:
                    from .mlx_client import _foreground_owner_context

                    async with _foreground_owner_context(
                        owner_label,
                        deadline=deadline if isinstance(deadline, Deadline) else None,
                        foreground_request=foreground_request,
                    ):
                        return await self._chat_completion(
                            payload_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            schema=schema,
                            deadline=deadline,
                            foreground_request=foreground_request,
                        )
                return await self._chat_completion(
                    payload_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schema,
                    deadline=deadline,
                    foreground_request=foreground_request,
                )
        except TimeoutError as exc:
            self.note_lane_recovering(str(exc))
            return None

    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        return await self.generate_text_async(prompt, **kwargs)

    async def _chat_completion(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        schema: Optional[Dict[str, Any]],
        deadline: Optional[Deadline],
        foreground_request: bool,
    ) -> Optional[str]:
        client = await self._client()
        self._mark_generation_started()
        try:
            payload: Dict[str, Any] = {
                "model": self._runtime_model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "min_p": 0.05,                # Filter low-probability tokens for quality
                "repetition_penalty": 1.1,    # Reduce stale/looping responses
                "top_p": self.top_p,
            }
            if schema:
                payload["response_format"] = {"type": "json_object"}

            timeout = None
            if isinstance(deadline, Deadline) and deadline.remaining is not None:
                timeout = max(1.0, deadline.remaining)
            else:
                timeout = 180.0 if self._is_primary_or_deep_lane() else 120.0

            try:
                response = await client.post(
                    f"{self._runtime_url}/v1/chat/completions",
                    json=payload,
                    timeout=timeout,
                )
            except Exception as exc:
                record_degradation('local_server_client', exc)
                self._record_degraded_event(
                    "request_failed",
                    detail=f"{self._lane_name}:{type(exc).__name__}",
                    severity="error",
                    foreground_request=foreground_request,
                )
                self.note_lane_recovering(f"request_failed:{type(exc).__name__}")
                return None

            if response.status_code != 200:
                detail = f"http_{response.status_code}"
                response_text = (response.text or "")[:240]
                if response.status_code == 400 and "exceeds the available context size" in response_text.lower():
                    self._record_degraded_event(
                        "context_overflow",
                        detail=f"{self._lane_name}:{response_text[:160]}",
                        severity="warning",
                        foreground_request=foreground_request,
                    )
                    if foreground_request:
                        self.note_lane_recovering("context_overflow")
                    return None

                # 500 "Compute error" is a fatal server issue — needs restart
                if response.status_code == 500 and "compute error" in response_text.lower():
                    logger.error("[%s] COMPUTE ERROR from server. Triggering restart.", self._lane_name)
                    self._record_degraded_event(
                        "compute_error",
                        detail=f"{self._lane_name}:server_compute_error",
                        severity="critical",
                        foreground_request=foreground_request,
                    )
                    # [STABILITY v53] Mark lane as recovering BEFORE firing restart
                    # so new requests don't hit the dying server. Previously the lane
                    # stayed "ready" while restart happened in background.
                    self.note_lane_recovering("compute_error_restart")
                    try:
                        import asyncio
                        task = asyncio.get_event_loop().create_task(
                            self._restart_server(),
                            name=f"restart_server:{self._lane_name}",
                        )
                        # Log if restart task crashes
                        task.add_done_callback(
                            lambda t: logger.error("Server restart failed: %s", t.exception())
                            if not t.cancelled() and t.exception() else None
                        )
                    except Exception as restart_err:
                        record_degradation('local_server_client', restart_err)
                        logger.error("[%s] Failed to schedule server restart: %s", self._lane_name, restart_err)
                    return None

                self._record_degraded_event(
                    "request_failed",
                    detail=f"{self._lane_name}:{detail}",
                    severity="error",
                    foreground_request=foreground_request,
                )
                # Don't mark lane as dead for transient errors — only for persistent failures
                if not foreground_request:
                    self.note_lane_recovering(detail)
                return None

            try:
                data = response.json()
            except Exception as exc:
                record_degradation('local_server_client', exc)
                self.note_lane_recovering(f"invalid_json:{type(exc).__name__}")
                return None

            text = ""
            try:
                text = self._extract_response_text(data)
            except Exception:
                text = ""

            if not text.strip():
                self._consecutive_empty += 1
                is_warmup = bool(self._warmup_in_flight)
                if is_warmup:
                    logger.debug(
                        "[%s] Empty warmup generation treated as benign runtime precompile.",
                        self._lane_name,
                    )
                    self._set_lane_state("ready")
                    self._last_ready_at = time.time()
                    return ""

                self._record_degraded_event(
                    "empty_generation",
                    detail=f"{self._lane_name}:response_keys={','.join(sorted(data.keys())[:6])}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                if foreground_request and self._is_primary_or_deep_lane():
                    if self._consecutive_empty >= 3:
                        self.note_lane_recovering("repeated_empty_generation")
                    else:
                        logger.warning(
                            "[%s] Empty generation received but runtime stayed healthy; preserving lane for retry (streak=%d).",
                            self._lane_name,
                            self._consecutive_empty,
                        )
                        self._set_lane_state("ready")
                        self._last_ready_at = time.time()
                return None

            from .mlx_client import _notify_closed_loop_output

            self._current_first_token_at = time.time()
            self._mark_progress()
            self._last_generation_completed_at = time.time()
            self._last_ready_at = time.time()
            self._set_lane_state("ready")
            self._consecutive_empty = 0
            _notify_closed_loop_output(text)
            return text.strip()
        finally:
            self._mark_generation_completed()
        
        
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        text = await self.generate_text_async(
            prompt,
            system_prompt=system_prompt,
            model=model,
            **kwargs,
        )
        if not text:
            return
        for token in text.split():
            yield token + " "
            await asyncio.sleep(0)


_SERVER_CLIENTS: Dict[str, LocalServerClient] = {}


def get_local_server_client(model_path: Optional[str] = None, **kwargs) -> LocalServerClient:
    if model_path is None:
        raise ValueError("model_path is required for local server clients")
    abs_path = os.path.realpath(model_path)
    if abs_path not in _SERVER_CLIENTS:
        _SERVER_CLIENTS[abs_path] = LocalServerClient(model_path=abs_path, **kwargs)
    return _SERVER_CLIENTS[abs_path]
