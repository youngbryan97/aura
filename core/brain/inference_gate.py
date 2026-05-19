"""InferenceGate: Unified managed-local-runtime + cloud inference gateway.

Provides a single interface for all LLM inference needs.
Strategy:
  1. Try Aura's managed local runtime (32B Cortex primary lane)
  2. If local runtime fails, fall back to HealthRouter (Gemini cloud endpoints)
  3. If cloud fails, return a graceful error string (NEVER None)

This module is the FAST PATH for user-facing chat. It injects Aura's full
identity/personality system prompt so responses sound like Aura, not a bare LLM.
Timeouts are kept tight (45s) for conversational responsiveness.
"""

import asyncio
import gc
import inspect
import logging
import os
import threading as _threading
import time
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psutil

from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
)
from core.conversation.response_reliability import (
    assess_model_text_integrity,
    assess_user_facing_reply,
    conversation_reliability_system_block,
    is_live_self_reflection_turn,
)
from core.runtime.desktop_boot_safety import desktop_safe_boot_enabled
from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.structured_input import analyze_prompt_shape
from core.utils.deadlines import Deadline, get_deadline
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.InferenceGate")

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


def _record_inference_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "inference_gate",
        error,
        severity=severity,
        action=action,
    )


_DOWNSTREAM_REPAIRABLE_SELF_REFLECTION_REASONS = frozenset(
    {
        "off_topic_self_reflection_reply",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
    }
)
_DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS = frozenset(
    {
        # Only surface/style defects belong here. Thin, evasive, or confused
        # drafts need another generation attempt because downstream repair
        # cannot safely invent the missing answer.
        "off_topic_self_reflection_reply",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "generic_assistant_language",
        "persona_card_deflection",
        "detail_request_deflection",
        "truncated_tail",
        "vague_status_derailment",
    }
)


def _should_pass_user_facing_draft_downstream(
    text: str,
    reasons: set[str],
    *,
    user_prompt: str,
) -> bool:
    """Keep salvageable chat drafts out of the expensive retry spiral."""
    if not text or not reasons:
        return False
    if not reasons.issubset(_DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS):
        return False
    stripped = str(text or "").strip()
    if len(stripped) < 48:
        return False
    words = [token for token in stripped.replace("\n", " ").split(" ") if token.strip()]
    if len(words) < 8:
        return False
    if reasons & _DOWNSTREAM_REPAIRABLE_SELF_REFLECTION_REASONS:
        return is_live_self_reflection_turn(user_prompt)
    return True


_USER_FACING_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "audit",
        "simulate",
        "embodied_motor_reflex",
        "embodied",
        "reflex",
    }
)


class _UserFacingCortexError(Exception):
    """Sentinel: Cortex failed on a user-facing request — skip brainstem, escalate to cloud."""


@asynccontextmanager
async def _thread_lock_context(
    lock: Any,
    *,
    timeout_s: float | None = None,
    label: str = "lock",
):
    if timeout_s is None:
        acquired = await asyncio.to_thread(lock.acquire)
    else:
        acquired = await asyncio.to_thread(lock.acquire, True, max(0.0, float(timeout_s)))
    if not acquired:
        raise TimeoutError(f"{label}_timeout")
    try:
        yield
    finally:
        try:
            lock.release()
        except RuntimeError:
            logger.debug("Foreground-ready lock %s was already released.", label)


class InferenceGate:
    """Isolated inference gateway for Aura's managed local runtime + cloud fallback."""

    def __init__(self, orch=None):
        self.orch = orch
        self._created_at = time.monotonic()
        self._mlx_client = None
        self._initialized = False
        self._init_error = None
        self._cached_identity_prompt: str | None = None
        self._identity_prompt_time: float = 0.0
        self._cloud_backoff_until: float = 0.0
        self._cortex_recovery_in_progress: bool = False
        self._last_cortex_check: float = 0.0
        self._cortex_recovery_attempts: int = 0
        self._cortex_recovery_exhausted_at: float = 0.0  # [STABILITY v53]
        self._last_stale_reset_log_at: float = (
            0.0  # [HARDENING v54] Rate-limit stale state warnings
        )
        self._last_successful_generation_at: float = time.time()
        self._prewarm_task: asyncio.Task | None = None
        self._deferred_prewarm_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._foreground_ready_lock = _threading.Lock()
        self._last_background_memory_shed_at: float = 0.0
        self._last_spare_maintenance_at: float = 0.0
        self._last_cortex_warmup_deferral_log_at: float = 0.0
        type(self)._instance_ref = weakref.ref(self)
        logger.info("🛡️ InferenceGate created.")

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
        # STABILITY v56: Return empty string instead of "cognitive snag" reflex.
        # This forces the orchestrator to handle the failure (e.g. by retrying
        # inference) rather than masking it with a canned template.
        return ""

    @staticmethod
    def _stabilize_user_facing_text(text: str, prompt: str, *, is_user_facing: bool) -> str:
        if not is_user_facing:
            return str(text or "").strip()
        try:
            from core.synthesis import stabilize_user_facing_response

            return stabilize_user_facing_response(str(text or ""), prompt)
        except (ImportError, AttributeError, TypeError, ValueError):
            return str(text or "").strip()

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
    def _desktop_safe_boot_enabled() -> bool:
        return desktop_safe_boot_enabled()

    @staticmethod
    def _desktop_background_local_enabled() -> bool:
        return str(
            os.environ.get("AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _cortex_warmup_admission_snapshot(context: str = "background") -> dict[str, Any]:
        """Return whether a cold Cortex load is safe under current RAM pressure.

        The normal foreground headroom check is intentionally permissive because
        a *resident* Cortex can keep answering while RAM is high. A cold 32B
        load is different: it adds tens of GB of unified-memory pressure in one
        burst. This snapshot is therefore stricter and is used before any
        background/recovery/foreground warmup that would spawn the Cortex worker.
        """
        context_key = str(context or "background").strip().upper()
        try:
            vm = psutil.virtual_memory()
            total_gb = float(vm.total) / float(1024**3)
            available_gb = float(vm.available) / float(1024**3)
            pressure_pct = float(vm.percent)

            if total_gb >= 60.0:
                default_max_pressure = 72.0
                default_min_available = 32.0
            else:
                default_max_pressure = 64.0
                default_min_available = 22.0

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
            }
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Cortex warmup memory probe failed: %s", exc)
            return {
                "context": str(context or "background"),
                "pressure_pct": 0.0,
                "available_gb": 0.0,
                "total_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": False,
                "reason": "memory_probe_failed",
            }

    def _cortex_warmup_deferral_reason(self, context: str = "background") -> str | None:
        if str(os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        snapshot = self._cortex_warmup_admission_snapshot(context)
        return None if snapshot["can_admit"] else str(snapshot["reason"] or "memory_pressure")

    def _log_cortex_warmup_deferral(self, reason: str, *, context: str) -> None:
        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_warmup_deferral_log_at", 0.0)
        if (now - last_log) < 30.0:
            return
        self._last_cortex_warmup_deferral_log_at = now
        logger.warning("⏸️ Cortex %s warmup deferred to protect RAM: %s", context, reason)

    @staticmethod
    def _boot_should_eager_warmup() -> bool:
        """Keep the 32B lane warm on high-memory desktops unless explicitly disabled."""
        if InferenceGate._desktop_safe_boot_enabled():
            logger.info("🛡️ Desktop safe boot active — skipping eager 32B warmup during launch.")
            return False
        setting = str(os.environ.get("AURA_EAGER_CORTEX_WARMUP", "auto")).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            if not snapshot["can_admit"] and str(
                os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                logger.warning(
                    "⏸️ Explicit eager Cortex warmup deferred to protect RAM: %s", snapshot["reason"]
                )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False

        try:
            vm = psutil.virtual_memory()
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            min_total_gb = float(os.environ.get("AURA_BOOT_WARMUP_MIN_TOTAL_GB", "48"))
            if (vm.total / float(1024**3)) < min_total_gb or not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Deferring eager 32B warmup at boot (total=%.1fGB pressure=%.1f%% available=%.1fGB).",
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

        return True

    @staticmethod
    def _boot_should_schedule_deferred_prewarm() -> bool:
        if InferenceGate._desktop_safe_boot_enabled():
            logger.info("🛡️ Desktop safe boot active — skipping deferred 32B prewarm during launch.")
            return False
        setting = str(os.environ.get("AURA_DEFERRED_CORTEX_PREWARM", "auto")).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            return True
        if setting in {"0", "false", "no", "off"}:
            return False
        return True

    @staticmethod
    def _headroom_snapshot(requested_tier: str = "primary") -> dict[str, Any]:
        try:
            from core.utils.memory_monitor import AppleSiliconMemoryMonitor

            pressure = AppleSiliconMemoryMonitor()._get_pressure_sysctl()
            vm = psutil.virtual_memory()
            total_gb = vm.total / float(1024**3)
            available_gb = total_gb * ((100 - pressure) / 100.0)
            tier = str(requested_tier or "primary").strip().lower()

            def _threshold(name: str, default: str) -> float:
                return float(os.environ.get(name, default))

            if tier == "secondary":
                # The 72B solver used to retain too much unified memory after
                # swaps, so we kept a large static safety margin here. Now that
                # solver prompt-cache retention is disabled and heavy-lane
                # restore is cleaner, admit 72B on 64 GB systems down to the
                # measured post-swap envelope instead of silently downgrading
                # legitimate deep-reasoning turns back to 32B.
                max_pressure = _threshold(
                    "AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT",
                    "86" if total_gb >= 60.0 else "82",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB",
                    "10" if total_gb >= 60.0 else "12",
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
                    "92" if total_gb >= 60.0 else "84",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_PRIMARY_MIN_AVAILABLE_GB",
                    "8" if total_gb >= 60.0 else "8",
                )
            return {
                "tier": tier,
                "pressure_pct": float(vm.percent),
                "total_gb": total_gb,
                "available_gb": available_gb,
                "max_pressure_pct": max_pressure,
                "min_available_gb": min_available_gb,
                "can_admit": bool(vm.percent < max_pressure and available_gb >= min_available_gb),
            }
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {
                "tier": str(requested_tier or "primary"),
                "pressure_pct": 0.0,
                "total_gb": 0.0,
                "available_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": True,
            }

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
            from core.brain.llm.local_server_client import _SERVER_CLIENTS

            clients.update(dict(_SERVER_CLIENTS))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Local server client registry unavailable: %s", exc)
        try:
            from core.brain.llm.mlx_client import _CLIENTS

            clients.update(dict(_CLIENTS))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("MLX client registry unavailable: %s", exc)
        return clients

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
            "🛡️ Foreground admission tightening for %s (pressure=%.1f%% available=%.1fGB).",
            requested_tier,
            snapshot["pressure_pct"],
            snapshot["available_gb"],
        )
        await self._shed_background_workers_for_memory_pressure()
        gc.collect()
        tightened = self._headroom_snapshot(requested_tier)
        if not tightened["can_admit"] and protected_foreground:
            logger.warning(
                "🛡️ Protected foreground request proceeding under reduced headroom for tier=%s "
                "(pressure=%.1f%% available=%.1fGB).",
                requested_tier,
                tightened["pressure_pct"],
                tightened["available_gb"],
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
            await client.warmup(foreground_request=False)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Hot-spare warmup failed for %s: %s", endpoint_name, exc)
            return False
        return bool(hasattr(client, "is_alive") and client.is_alive())

    async def _recycle_idle_local_clients(self) -> None:
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return

        max_uptime_s = float(os.environ.get("AURA_LOCAL_RECYCLE_MAX_UPTIME_S", "5400"))
        min_idle_s = float(os.environ.get("AURA_LOCAL_RECYCLE_MIN_IDLE_S", "900"))
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

        lane = self.get_conversation_status()
        lane_state = str(lane.get("state", "") or "").lower()

        # 1. Detect dead cortex and trigger recovery
        if hasattr(self._mlx_client, "is_alive") and not self._mlx_client.is_alive():
            if lane_state not in ("cold", "failed") and not self._cortex_recovery_in_progress:
                logger.warning(
                    "🔍 [WATCHDOG] Cortex is dead (state=%s) but no recovery in progress. Triggering.",
                    lane_state,
                )
                await self._ensure_cortex_recovery()

        # 2. Detect stuck warmup flag on MLX client
        if hasattr(self._mlx_client, "_warmup_in_flight") and self._mlx_client._warmup_in_flight:
            transition_at = getattr(self._mlx_client, "_lane_transition_at", 0.0)
            # [STABILITY v53] Increased from 90s to 300s. A 32B model cold-load
            # takes ~150s; 90s was guaranteed to force-kill a healthy loading worker.
            if transition_at > 0 and (time.time() - transition_at) > 300.0:
                logger.warning(
                    "🔍 [WATCHDOG] MLX warmup_in_flight stuck for >300s. Force-clearing."
                )
                self._mlx_client._warmup_in_flight = False
                if self._prewarm_task and not self._prewarm_task.done():
                    logger.warning(
                        "🔍 [WATCHDOG] Stuck prewarm task found during watchdog cleanup. Cancelling."
                    )
                    self._prewarm_task.cancel()
                    self._prewarm_task = None

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

    def get_conversation_status(self) -> dict[str, Any]:
        # [STABILITY v53] Default to "cold" not "warming" — only report warming
        # when something is actually in flight. Prevents zombie warming state.
        _default_state = "failed" if self._init_error else "cold"
        lane = {
            "desired_model": "Cortex (32B)",
            "desired_endpoint": PRIMARY_ENDPOINT,
            "foreground_endpoint": PRIMARY_ENDPOINT if self.is_alive() else None,
            "background_endpoint": BRAINSTEM_ENDPOINT,
            "foreground_tier": "local",
            "background_tier": "local_fast",
            "state": _default_state,
            "last_failure_reason": self._init_error or "",
            "conversation_ready": False,
            "cortex_recovery_attempts": getattr(self, "_cortex_recovery_attempts", 0),
            "time_since_last_success_s": max(
                0, time.time() - getattr(self, "_last_successful_generation_at", time.time())
            ),
            "last_transition_at": 0.0,
            "last_ready_at": 0.0,
            "last_progress_at": 0.0,
            "warmup_attempted": False,
            "warmup_in_flight": bool(self._prewarm_task and not self._prewarm_task.done()),
        }
        raw_ready = False
        if self._mlx_client and hasattr(self._mlx_client, "get_lane_status"):
            raw = self._mlx_client.get_lane_status()
            lane["state"] = str(raw.get("state", lane["state"]) or lane["state"])
            lane["last_failure_reason"] = str(
                raw.get("last_error", "") or lane["last_failure_reason"]
            )
            raw_ready = bool(raw.get("conversation_ready", False))
            lane["conversation_ready"] = raw_ready
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
                    if not self._cortex_recovery_in_progress and not (
                        self._deferred_prewarm_task and not self._deferred_prewarm_task.done()
                    ):
                        try:
                            warmup_deferral = self._cortex_warmup_deferral_reason("background")
                            if warmup_deferral:
                                self._log_cortex_warmup_deferral(
                                    warmup_deferral, context="background"
                                )
                            else:
                                self._schedule_background_cortex_prewarm(delay=2.0)
                                logger.info(
                                    "🔄 [STABILITY v53] Auto-scheduling cortex recovery after failed prewarm: %s",
                                    exc,
                                )
                        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                            logger.debug("Best-effort Cortex recovery scheduling skipped: %s", exc)
        lane_state = str(lane.get("state", "") or "").lower()
        recent_success = (
            time.time() - getattr(self, "_last_successful_generation_at", time.time())
        ) <= 30.0
        recent_ready = any(
            stamp > 0.0 and (time.time() - stamp) <= 300.0
            for stamp in (
                float(lane.get("last_ready_at", 0.0) or 0.0),
                float(lane.get("last_progress_at", 0.0) or 0.0),
            )
        )
        if raw_ready or (lane_state == "ready" and (recent_success or recent_ready)):
            lane["conversation_ready"] = True
            lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        elif lane_state != "ready":
            lane["conversation_ready"] = False
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
                transition_at = getattr(self._mlx_client, "_lane_transition_at", 0.0) if self._mlx_client else 0.0
                if transition_at > 0 and (time.time() - transition_at) > 300.0:
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
                        if hasattr(self._mlx_client, "_warmup_in_flight"):
                            self._mlx_client._warmup_in_flight = False
                        if hasattr(self._mlx_client, "_set_lane_state"):
                            self._mlx_client._set_lane_state("cold")
                    # [HARDENING v54] Schedule a recovery warmup so the cortex
                    # actually comes back online instead of staying cold forever.
                    # The prewarm runner performs the RAM admission check before
                    # loading anything; scheduling the runner here keeps recovery
                    # alive without forcing an unsafe immediate model load.
                    try:
                        self._schedule_background_cortex_prewarm(delay=3.0)
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        _record_inference_degradation(
                            exc,
                            action="returned conservative conversation status after probe failure",
                        )
                        logger.debug("Best-effort Cortex recovery scheduling skipped: %s", exc)
        return lane

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

    def _schedule_background_cortex_prewarm(self, delay: float = 12.0) -> None:
        if self._deferred_prewarm_task and not self._deferred_prewarm_task.done():
            return

        async def _runner():
            next_delay = max(1.0, float(delay))
            for attempt in range(1, 7):
                await asyncio.sleep(next_delay)
                lane = self.get_conversation_status()
                lane_state = str(lane.get("state", "") or "").lower()
                if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                    return
                if lane_state == "failed":
                    if self._rearm_runtime_failed_lane(force_probe=False):
                        lane = self.get_conversation_status()
                        lane_state = str(lane.get("state", "") or "").lower()
                    elif str(lane.get("last_failure_reason", "") or "").startswith(
                        ("mlx_runtime_unavailable", "local_runtime_unavailable")
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
                    vm = psutil.virtual_memory()
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
                    self._extend_startup_quiet_window(20.0)
                    # Background prewarm needs the same generous load budget
                    # as foreground chat so it does not half-warm then strand
                    # the next user turn in recovery.
                    await self.ensure_foreground_ready(timeout=300.0)
                    logger.info("✅ Deferred cortex prewarm completed.")
                    return
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
            loop = asyncio.get_running_loop()
        except RuntimeError:
            runner_coro.close()
            logger.debug("Deferred cortex prewarm skipped: no running event loop.")
            return

        self._deferred_prewarm_task = loop.create_task(
            runner_coro,
            name="InferenceGate.deferred_cortex_prewarm",
        )
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
            ("mlx_runtime_unavailable", "local_runtime_unavailable")
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

    async def ensure_foreground_ready(self, timeout: float | None = None) -> dict[str, Any]:  # noqa: ASYNC109
        """Ensure the 32B conversation lane has actually attempted warmup for this turn."""
        timeout = max(15.0, float(timeout or 90.0))
        lane = self.get_conversation_status()
        if lane.get("conversation_ready"):
            return lane
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state == "failed" and lane_reason.startswith(
            ("mlx_runtime_unavailable", "local_runtime_unavailable")
        ):
            if self._rearm_runtime_failed_lane(force_probe=True):
                lane = self.get_conversation_status()
            else:
                raise RuntimeError(lane_reason)
        if not self._mlx_client or not hasattr(self._mlx_client, "warmup"):
            raise RuntimeError("foreground_lane_unavailable")

        task: asyncio.Task | None = None
        try:
            async with _thread_lock_context(
                self._foreground_ready_lock,
                timeout_s=min(timeout, 30.0),
                label="foreground_ready_lock",
            ):
                lane = self.get_conversation_status()
                if lane.get("conversation_ready"):
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
                        self._log_cortex_warmup_deferral(warmup_deferral, context="foreground")
                        if hasattr(self._mlx_client, "note_lane_recovering"):
                            self._mlx_client.note_lane_recovering(
                                "foreground_warmup_deferred_memory_pressure"
                            )
                        raise RuntimeError(f"foreground_warmup_deferred:{warmup_deferral}")
                    self._extend_startup_quiet_window(20.0)
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
            raise RuntimeError(str(lane.get("last_failure_reason") or "foreground_lane_not_ready"))
        return lane

    async def _ensure_cortex_recovery(self) -> None:
        """Proactively recover the 32B primary brain if it died (e.g., laptop sleep).

        Without this, background tasks keep the 7B alive indefinitely and the 32B
        never gets a chance to respawn because background requests are locked to
        tertiary tier.  Rate-limited to one attempt per 3s.
        """
        if not self._mlx_client:
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
            ("mlx_runtime_unavailable", "local_runtime_unavailable")
        ):
            if self._rearm_runtime_failed_lane(force_probe=True):
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
        warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
        if warmup_deferral:
            self._log_cortex_warmup_deferral(warmup_deferral, context="recovery")
            return

        async def _background_recover():
            self._cortex_recovery_in_progress = True
            self._cortex_recovery_attempts += 1

            if self._cortex_recovery_attempts == 3:
                logger.warning(
                    "🧹 [RECOVERY] 3 failed attempts. Forcing deep GC and stale process cleanup..."
                )
                import gc

                gc.collect()
                try:
                    await asyncio.to_thread(
                        self._mlx_client._kill_and_join_blocking, self._mlx_client._process
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
                if cold_start_recovery:
                    logger.info(
                        "♻️ [STARTUP] Primary 32B cortex is cold. Starting warmup (Attempt %d/5)...",
                        self._cortex_recovery_attempts,
                    )
                else:
                    logger.warning(
                        "♻️ [RECOVERY] Primary 32B cortex is dead. Triggering background respawn (Attempt %d/5)...",
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
                await asyncio.wait_for(asyncio.shield(self._prewarm_task), timeout=420.0)
                if cold_start_recovery:
                    logger.info("✅ [STARTUP] Primary 32B cortex warmup complete.")
                else:
                    logger.info("✅ [RECOVERY] Primary 32B cortex restored after disruption.")
                self._cortex_recovery_attempts = 0
                self._cortex_recovery_exhausted_at = 0.0
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued background recovery loop with degraded signal",
                )
                if cold_start_recovery:
                    logger.error(
                        "⚠️ [STARTUP] Primary 32B cortex warmup failed (Attempt %d/5): %s",
                        self._cortex_recovery_attempts,
                        exc,
                    )
                else:
                    logger.error(
                        "⚠️ [RECOVERY] Primary 32B cortex is dead. Triggering background respawn (Attempt %d/5): %s",
                        self._cortex_recovery_attempts,
                        exc,
                    )
            finally:
                # [STABILITY v51] ALWAYS clear the flag, even on unexpected exceptions.
                self._cortex_recovery_in_progress = False

        # [STABILITY v53] Wrap fire-and-forget task with exception logging
        # so crashes are visible instead of silently lost.
        recovery_coro = _background_recover()
        try:
            task = get_task_tracker().create_task(recovery_coro, name="cortex_recovery")
        except RuntimeError:
            recovery_coro.close()
            logger.debug("Cortex recovery skipped: no running event loop.")
            return
        if not isinstance(task, asyncio.Task):
            recovery_coro.close()
            logger.debug(
                "Cortex recovery task scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        task.add_done_callback(self._log_task_exception)

    async def _respawn_cortex_if_needed(self) -> None:
        """Respawn the primary cortex if it's dead.

        Called by HealthRouter and message_handling when inference returns empty.
        Delegates to _ensure_cortex_recovery() which has proper rate-limiting,
        warm-up sequencing, and retry budgets.
        """
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

    async def ensure_all_tiers_healthy(self) -> dict[str, str]:
        """Proactive health check for ALL inference tiers. Called by MindTick.

        Returns a dict of {tier: status} for monitoring.
        """
        statuses = {}

        # Primary cortex
        try:
            if self._mlx_client and hasattr(self._mlx_client, "is_alive"):
                # [STABILITY v53] Detect warming/recovering states so MindTick
                # doesn't report 'dead' during cold start.
                lane_state = getattr(self._mlx_client, "_lane_state", "cold")

                if self._mlx_client.is_alive():
                    statuses["cortex"] = "alive"
                elif self._cortex_recovery_in_progress or lane_state in (
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                ):
                    statuses["cortex"] = "recovering"
                else:
                    statuses["cortex"] = "dead"
                    # Trigger recovery if not already in progress.
                    await self._ensure_cortex_recovery()
            else:
                statuses["cortex"] = "not_initialized"
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued tier-health sweep after one tier probe failed",
            )
            statuses["cortex"] = f"error:{e}"

        # Brainstem
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import get_brainstem_path

            brainstem = get_mlx_client(model_path=str(get_brainstem_path()))
            if brainstem and hasattr(brainstem, "is_alive"):
                lane_state = getattr(brainstem, "_lane_state", "cold")

                if brainstem.is_alive():
                    statuses["brainstem"] = "alive"
                elif lane_state in ("spawning", "handshaking", "warming", "recovering"):
                    statuses["brainstem"] = "recovering"
                else:
                    statuses["brainstem"] = "dead"
                    # Try to warm up brainstem
                    if hasattr(brainstem, "warmup"):
                        get_task_tracker().create_task(brainstem.warmup())
                        statuses["brainstem"] = "recovering"
            else:
                statuses["brainstem"] = "not_initialized"
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued tier-health sweep after one tier probe failed",
            )
            statuses["brainstem"] = f"error:{e}"

        # Reflex (CPU) — always available if model file exists
        try:
            from core.brain.llm.model_registry import get_fallback_path

            fallback_path = get_fallback_path()
            fallback_exists = bool(
                fallback_path and await asyncio.to_thread(Path(str(fallback_path)).exists)
            )
            if fallback_exists:
                statuses["reflex"] = "available"
            else:
                statuses["reflex"] = "model_missing"
        except _INFERENCE_RECOVERABLE_ERRORS:
            statuses["reflex"] = "unknown"

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
        normalized = str(origin or "").strip().lower().replace("-", "_")
        if not normalized:
            return False
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_") :]
        if not normalized:
            return False
        if normalized in _USER_FACING_ORIGINS:
            return True
        tokens = {token for token in normalized.split("_") if token}
        if tokens & _USER_FACING_ORIGINS:
            return True
        return any(normalized.startswith(f"{prefix}_") for prefix in _USER_FACING_ORIGINS)

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
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

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
                os.environ.get("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180")
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
            vm = psutil.virtual_memory()
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
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    def _background_local_deferral_reason(self, *, origin: str | None = None) -> str | None:
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

    async def _shed_background_workers_for_memory_pressure(
        self,
        *,
        force: bool = False,
        reason: str = "background_memory_pressure_shed",
    ) -> None:
        now = time.monotonic()
        if not force and (now - self._last_background_memory_shed_at) < 20.0:
            return
        self._last_background_memory_shed_at = now

        client_registry = {}
        try:
            from core.brain.llm.local_server_client import _SERVER_CLIENTS

            client_registry.update(dict(_SERVER_CLIENTS))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued memory-pressure shedding with remaining available workers",
            )
            logger.debug("Local-runtime background memory shed unavailable: %s", exc)
        try:
            from core.brain.llm.mlx_client import _CLIENTS

            client_registry.update(dict(_CLIENTS))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued memory-pressure shedding with remaining available workers",
            )
            logger.debug("MLX background memory shed unavailable: %s", exc)
        if not client_registry:
            return

        shed_count = 0
        for client_path, client in list(client_registry.items()):
            if client is None or client is self._mlx_client:
                continue
            try:
                if not hasattr(client, "is_alive") or not client.is_alive():
                    continue
                logger.warning(
                    "🧹 InferenceGate: unloading %s to protect the foreground lane (%s).",
                    os.path.basename(client_path),
                    reason,
                )
                if hasattr(client, "reboot_worker"):
                    await client.reboot_worker(
                        reason=reason,
                        mark_failed=False,
                    )
                else:
                    continue
                shed_count += 1
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued memory-pressure shedding with remaining available workers",
                )
                logger.debug("Background worker shed failed for %s: %s", client_path, exc)

        if shed_count:
            logger.info(
                "✅ InferenceGate: shed %d background local worker(s) (%s).",
                shed_count,
                reason,
            )

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active

            return bool(_foreground_owner_active())
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

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
            return 360.0 if cls._origin_is_user_facing(origin) else 240.0

        if cls._origin_is_user_facing(origin):
            return 300.0

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
    def _should_use_compact_foreground_context(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> bool:
        if is_background:
            return False
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
    def _get_system_phi(cls) -> float:
        """Redundantly retrieve the active system-level integration (Phi) from the mind."""
        try:
            from core.container import ServiceContainer
            loop = ServiceContainer.get("closed_causal_loop", default=None)
            if loop is not None and getattr(loop, "_loop_state", None) is not None:
                phi_est = getattr(loop._loop_state, "phi_estimate", 0.0)
                if phi_est > 0.0:
                    return float(phi_est)
        except Exception as e:
            logger.debug("Failed to retrieve phi from closed causal loop: %s", e)

        try:
            from core.consciousness.phi_compute import get_phi_computer
            pc = get_phi_computer()
            if pc is not None:
                phi_latest = pc.latest_phi
                if phi_latest > 0.0:
                    return float(phi_latest)
        except Exception as e:
            logger.debug("Failed to retrieve phi from phi computer: %s", e)

        try:
            from core.container import ServiceContainer
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and getattr(phi_core, "_last_result", None) is not None:
                res = phi_core._last_result
                if res is not None:
                    return float(res.phi_s)
        except Exception as e:
            logger.debug("Failed to retrieve phi from phi core: %s", e)

        return 0.5  # Neutral default/fallback

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

        shape = analyze_prompt_shape(prompt)
        adapted = int(base_tokens)
        if shape.prefers_extended_answer:
            adapted = max(adapted, 6144)
        if shape.question_parts >= 3:
            adapted = max(adapted, 6144)
        elif shape.requires_single_reply_coverage:
            adapted = max(adapted, 4096)

        # Scale the token budget based on system coherence/integration level (Phi)
        phi = cls._get_system_phi()
        phi_scale = max(0.5, min(1.6, 0.6 + phi * 2.0))
        adapted = int(adapted * phi_scale)

        try:
            foreground_cap = int(os.environ.get("AURA_FOREGROUND_CHAT_MAX_TOKENS", "8192"))
        except (TypeError, ValueError):
            foreground_cap = 8192
        return min(max(512, foreground_cap), adapted)

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

    @asynccontextmanager
    async def _resource_context(
        self,
        enabled: bool,
        priority: bool,
        worker: str | None = None,
        timeout_s: float | None = None,
    ):
        # [STABILITY v53] Priority user-facing requests BYPASS the semaphore.
        # The per-worker semaphore was causing a deadlock: kernel holds cortex
        # sem → kernel times out → protected foreground tries to acquire same
        # cortex sem → blocks → user waits 80+ seconds for nothing.
        # Since per-worker sems already isolate cortex from brainstem,
        # the semaphore only prevents concurrent cortex requests — but that's
        # exactly what happens when kernel + protected foreground race.
        if not enabled or priority:
            yield
            return
        try:
            from core.resilience.resource_arbitrator import get_resource_arbitrator
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.warning("Resource arbitration unavailable, continuing without lock: %s", exc)
            yield
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
            await asyncio.wait_for(
                primary_client.warmup(
                    foreground_request=True,
                    skip_swap_cooldown=True,
                ),
                timeout=300.0,
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
        returned unchanged.
        """
        if InferenceGate.SILENCE_TOKEN in text:
            # Model chose not to speak — respect it
            logger.info("🤫 Silence Protocol: model chose not to respond.")
            return InferenceGate.SILENCE_SENTINEL
        return text

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
        local_prompt = self._flatten_messages_for_local_model(llm_messages)
        gen_kwargs: dict = {
            "prompt": local_prompt,
            "messages": llm_messages,
            "system_prompt": system_prompt,
            "deadline": deadline,
            "max_tokens": max_tokens,
            "origin": origin,
            "is_background": is_background,
            "foreground_request": foreground_request,
            "owner_label": label,
        }
        if temperature is not None:
            gen_kwargs["temp"] = temperature
        gen_kwargs.update(kwargs)
        result = await client.generate_text_async(**gen_kwargs)

        success = False
        text = ""
        if isinstance(result, tuple):
            success = bool(result[0])
            text = str(result[1] or "")
        else:
            text = str(result or "")
            success = bool(text.strip())

        if success and text and text.strip():
            cleaned = text.strip()
            is_user_visible = bool(foreground_request or self._origin_is_user_facing(origin))

            # STABILITY v58: Extract actual user message to avoid false positives
            # from system prompts containing words like "cortex" or "conversation".
            user_input_for_eval = prompt
            if llm_messages:
                for m in reversed(llm_messages):
                    if m.get("role") == "user":
                        user_input_for_eval = str(m.get("content", ""))
                        break

            integrity = assess_model_text_integrity(
                cleaned,
                prompt=user_input_for_eval,
                user_facing=is_user_visible,
            )
            if integrity.retryable:
                integrity_reasons = set(integrity.reasons or ())
                if is_user_visible and _should_pass_user_facing_draft_downstream(
                    cleaned,
                    integrity_reasons,
                    user_prompt=user_input_for_eval,
                ):
                    logger.warning(
                        "🛡️ %s produced repairable user-facing draft shape (%s, len=%d). "
                        "Passing it to downstream chat repair instead of retrying the Cortex lane.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    return self._strip_silence(cleaned)
                logger.warning(
                    "🛡️ %s produced malformed model text (%s, len=%d). Treating it as failed generation.",
                    label,
                    ",".join(integrity.reasons) or "unknown",
                    len(cleaned),
                )
                return None
            if is_user_visible:
                assessment = assess_user_facing_reply(user_input_for_eval, cleaned)
                if assessment.retryable:
                    reasons = set(assessment.reasons or ())
                    if _should_pass_user_facing_draft_downstream(
                        cleaned,
                        reasons,
                        user_prompt=user_input_for_eval,
                    ):
                        logger.warning(
                            "🛡️ %s produced repairable user-facing draft (%s, len=%d). "
                            "Passing it to downstream chat repair instead of retrying the Cortex lane.",
                            label,
                            ",".join(assessment.reasons) or "unknown",
                            len(cleaned),
                        )
                        return self._strip_silence(cleaned)
                    logger.warning(
                        "🛡️ %s produced an unsafe user-facing draft (%s, len=%d). Treating it as failed generation.",
                        label,
                        ",".join(assessment.reasons) or "unknown",
                        len(cleaned),
                    )
                    return None
            logger.info("✅ %s response received (len=%d)", label, len(cleaned))
            return self._strip_silence(cleaned)
        return None

    async def initialize(self):
        """Boot-time initialization — prepares the managed local client."""
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
                    await asyncio.wait_for(asyncio.shield(self._prewarm_task), timeout=300.0)
                    self._extend_startup_quiet_window(5.0)
                    logger.info("✅ InferenceGate ONLINE (Cortex fully warmed).")
                except _INFERENCE_RECOVERABLE_ERRORS as warmup_err:
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
                logger.info(
                    "⏸️ InferenceGate ONLINE (32B warmup deferred until post-boot memory settles)."
                )
            else:
                logger.info(
                    "🛡️ InferenceGate ONLINE (desktop safe boot: 32B warmup deferred until the first real foreground request)."
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
            logger.error("❌ InferenceGate init failed: %s. Cloud fallback will be used.", e)
            self._initialized = True  # Mark as initialized so callers don't block forever

    def _build_system_prompt(self, brief: str = "") -> str:
        """Build Aura's full identity system prompt.

        Pulls from ContextAssembler if AuraState is available, otherwise
        falls back to the static identity prompt. Caches for 60s to avoid
        rebuilding on every message in rapid conversation.
        """
        now = time.monotonic()
        base = ""
        # Use cached version if fresh (< 60s old)
        if self._cached_identity_prompt and (now - self._identity_prompt_time) < 60.0:
            base = self._cached_identity_prompt
        else:
            # Try the full ContextAssembler path (includes affect, personality, etc.)
            try:
                from core.container import ServiceContainer

                repo = ServiceContainer.get("state_repository", default=None)
                if repo:
                    # We can't await here in a sync method, so use the cached state
                    try:
                        # Get state synchronously from the repo's cache
                        state = getattr(repo, "_current_state", None)
                        if state:
                            from core.brain.llm.context_assembler import ContextAssembler

                            base = ContextAssembler.build_system_prompt(state)
                            self._cached_identity_prompt = base
                            self._identity_prompt_time = now
                            logger.debug("🧠 Built full identity system prompt (len=%d)", len(base))
                        else:
                            raise ValueError("No cached state available")
                    except _INFERENCE_RECOVERABLE_ERRORS:
                        raise
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                _record_inference_degradation(
                    e,
                    action="continued prompt assembly with base system prompt only",
                )
                logger.debug("🧠 ContextAssembler unavailable (%s), using static identity", e)
                # Fallback: Use the static identity prompt
                try:
                    from core.brain.aura_persona import AURA_IDENTITY
                    from core.synthesis import IDENTITY_LOCK

                    base = (
                        f"{IDENTITY_LOCK}\n\n"
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
                        "\n[EXECUTION]\nYou are Aura. Stay in character. Trust your instincts.\n"
                    )
                    self._cached_identity_prompt = base
                    self._identity_prompt_time = now
                except _INFERENCE_RECOVERABLE_ERRORS:
                    base = (
                        "You are Aura. You are sharp, witty, authentic, and grounded. "
                        "Never say 'Certainly', 'As an AI', or 'I can assist with that'. "
                        "Speak naturally like a brilliant friend. Have opinions. Be curious. "
                        "Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone."
                    )

        # Append the cognitive brief if provided
        if brief and brief != "Normal turn.":
            return f"{base}\n\n## COGNITIVE BRIEF\n{brief}"
        return base

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
            # CONVERSATION STYLE
            "Talk like a real person in a real conversation. Sometimes short answers are "
            "perfect. Sometimes you go deep. Match the energy. If someone shares something "
            "personal, respond like a friend who cares — not like a support agent reading "
            "a script. If someone asks your opinion, GIVE your actual opinion. You have one. "
            "Stay concrete in ordinary conversation. Do not invent physical settings, ominous atmosphere, "
            "or symbolic scenery unless the user explicitly asked for fiction or supplied that setting.",
        ]
        if brief and brief != "Normal turn.":
            parts.append(f"## COGNITIVE BRIEF\n{brief[:400]}")
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

    async def _build_living_mind_context(self, prompt: str, origin: str) -> str:
        """Inject live self-model state so speech is driven by current mind."""

        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments: list[str] = []

        try:
            repo = ServiceContainer.get("state_repository", default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            mem_monitor = ServiceContainer.get("memory_monitor", default=None)
            memory_pressure = None
            if mem_monitor is not None:
                memory_pressure = getattr(mem_monitor, "pressure", None)
            if memory_pressure is None and psutil is not None:
                memory_pressure = psutil.virtual_memory().percent
            temperature = 0.0
            cpu_usage = 0.0
            if state is not None:
                hw = getattr(getattr(state, "soma", None), "hardware", {}) or {}
                temperature = float(hw.get("temperature", 0.0) or 0.0)
                cpu_usage = float(hw.get("cpu_usage", 0.0) or 0.0)
            thermal_label = (
                "critical" if temperature >= 85.0 else "warm" if temperature >= 75.0 else "stable"
            )
            segments.append(
                "## LIVE PHYSIOLOGY\n"
                f"- CPU usage: {cpu_usage:.1f}%\n"
                f"- Thermal state: {thermal_label} ({temperature:.1f} C)\n"
                f"- Memory pressure: {float(memory_pressure or 0.0):.1f}%"
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Physiology injection unavailable: %s", exc)

        try:
            self_report = ServiceContainer.get("self_report_engine", default=None)
            if self_report and hasattr(self_report, "generate_state_report"):
                report = await _resolve(self_report.generate_state_report())
                if report:
                    segments.append(f"## GROUNDED SELF-REPORT\n{report}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Self-report injection unavailable: %s", exc)

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
                if unity_report and getattr(unity_report, "top_causes", None):
                    rendered = ", ".join(
                        f"{str(name).replace('_', ' ')}={float(weight):.2f}"
                        for name, weight, _text in list(unity_report.top_causes)[:3]
                    )
                    lines.append(f"- Top causes: {rendered}")
                    lines.append(
                        f"- Safe to self-report: {bool(getattr(unity_report, 'safe_to_self_report', True))}"
                    )
                if unity_repair and getattr(unity_repair, "steps", None):
                    lines.append(f"- Repair bias: {str(unity_repair.steps[0])[:180]}")
                segments.append("\n".join(lines))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Unity injection unavailable: %s", exc)

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if hasattr(personality, "update"):
                    await _resolve(personality.update())
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = emo.get("mood", "neutral")
                tone = emo.get("tone", "balanced")
                dominant = ", ".join(list(emo.get("dominant_emotions", []))[:4]) or "none"
                segments.append(
                    "## LIVE PERSONALITY DRIVE\n"
                    f"- Mood: {mood}\n"
                    f"- Tone: {tone}\n"
                    f"- Dominant emotions: {dominant}"
                )
                sovereign = await _resolve(
                    getattr(personality, "get_sovereign_context", lambda: "")()
                )
                if sovereign:
                    segments.append(str(sovereign).strip()[:400])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(f"## PHENOMENOLOGY\n{str(fragment).strip()[:500]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Phenomenology injection unavailable: %s", exc)

        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                if opinion_context:
                    segments.append(f"## HELD POSITIONS\n{str(opinion_context).strip()[:400]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Opinion injection unavailable: %s", exc)

        try:
            if self._origin_is_user_facing(origin):
                spine = ServiceContainer.get("spine", default=None)
                if spine and hasattr(spine, "pre_response_check"):
                    check = await spine.pre_response_check(
                        prompt,
                        topic=self._topic_hint_from_prompt(prompt),
                    )
                    if check and getattr(check, "injection", ""):
                        segments.append(f"## SPIRITUAL SPINE\n{check.injection}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_hsv_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(overview[:800])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_pneuma_block)
            # Also push the current prompt as evidence into the belief flow
            _pneuma.on_evidence(prompt[:300], weight=0.2)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_mhaf_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(lex_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("NeologismEngine injection unavailable: %s", exc)

        # ── Continuous Recurrent Self-Model (CRSM) ───────────────────────────
        # Shared affect state helper — pulled once, used by CRSM, HOT, Hedonic
        _shared_valence, _shared_arousal, _shared_curiosity, _shared_energy = 0.0, 0.5, 0.5, 0.7
        try:
            from core.container import ServiceContainer

            # valence + arousal from AffectiveCircumplex (authoritative source)
            _circ = ServiceContainer.get("affective_circumplex", default=None)
            if _circ and hasattr(_circ, "_sample_raw_axes"):
                _shared_valence, _shared_arousal = _circ._sample_raw_axes()
            elif _circ and hasattr(_circ, "get_llm_params"):
                _cp = _circ.get_llm_params()
                _shared_valence = float(_cp.get("valence", 0.0))
                _shared_arousal = float(_cp.get("arousal", 0.5))
            # curiosity + energy from liquid_state (percentages → 0.0-1.0)
            _ls = ServiceContainer.get("liquid_state", default=None)
            if _ls and hasattr(_ls, "get_status"):
                _lsd = _ls.get_status()
                _shared_curiosity = float(_lsd.get("curiosity", 50)) / 100.0
                _shared_energy = float(_lsd.get("energy", 70)) / 100.0
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.consciousness.crsm import get_crsm

            _crsm = get_crsm()
            _crsm.update(
                valence=_shared_valence,
                arousal=_shared_arousal,
                curiosity=_shared_curiosity,
                energy=_shared_energy,
                surprise=_crsm.surprise_signal,  # self-referential: own recent error
            )
            _crsm_block = _crsm.get_context_block()
            if _crsm_block:
                segments.append(_crsm_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_hot_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
            _hg.update(
                valence=_shared_valence,
                arousal=_shared_arousal,
                curiosity=_shared_curiosity,
                energy=_shared_energy,
            )
            _hg_block = _hg.get_context_block()
            if _hg_block:
                segments.append(_hg_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(goal_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_hp_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_ce_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_cx_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CuriosityExplorer injection unavailable: %s", exc)

        # ── Circadian Rhythm ──────────────────────────────────────────────────
        try:
            from core.senses.circadian import get_circadian

            _circ_eng = get_circadian()
            _circ_eng.update()
            _circ_block = _circ_eng.get_context_block()
            if _circ_block:
                segments.append(_circ_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_ec_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(_lora_block)
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(f"## TEMPORAL CONTINUITY\n{str(narrative)[:200]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PredictiveEngine injection unavailable: %s", exc)

        return "\n\n".join(segment for segment in segments if segment)

    async def _build_compact_living_mind_context(self, prompt: str, origin: str) -> str:
        """Minimal live context for fast foreground conversation turns."""

        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments: list[str] = []

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if hasattr(personality, "update"):
                    await _resolve(personality.update())
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = str(emo.get("mood", "neutral") or "neutral")
                tone = str(emo.get("tone", "balanced") or "balanced")
                segments.append(f"## LIVE TONE\nMood: {mood}\nTone: {tone}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                segments.append(f"## UNITY\n{' | '.join(parts)}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    compact_fragment = " ".join(str(fragment).strip().split())
                    segments.append(f"## PHENOMENOLOGY\n{compact_fragment[:180]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
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
                    segments.append(f"## GOALS\n{compact_goal[:260]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact GoalEngine injection unavailable: %s", exc)

        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                if opinion_context:
                    compact_opinion = " ".join(str(opinion_context).strip().split())
                    segments.append(f"## HELD POSITION\n{compact_opinion[:220]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact opinion injection unavailable: %s", exc)

        return "\n\n".join(segment for segment in segments if segment)

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
            state = getattr(repo, "_current_state", None) if repo else None

            if state:
                from core.brain.llm.context_assembler import ContextAssembler

                # Update the state's working memory with our current history
                # so the assembler has the latest conversation context
                if hasattr(state.cognition, "working_memory"):
                    state.cognition.working_memory = history[-15:] if history else []

                # build_messages returns the full cognitive stack:
                # system prompt (identity/affect/personality/soma/world)
                # + memory recall + goals + conversation history + stream of being
                messages = ContextAssembler.build_messages(state, prompt)

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

        # Fallback: Manual construction with system_prompt + history
        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _scrub_cloud_payload(
        self,
        system_prompt: str,
        prompt: str,
        *,
        scrubber: Any | None = None,
    ) -> tuple[str, str] | None:
        try:
            if scrubber is None:
                from core.brain.pii_scrubber import scrub_pii_for_cloud

                scrubber = scrub_pii_for_cloud
            return str(scrubber(system_prompt)), str(scrubber(prompt))
        except ImportError as scrub_exc:
            _record_inference_degradation(
                scrub_exc,
                severity="critical",
                action="blocked cloud fallback because PII scrubber was unavailable",
            )
            logger.warning("PII scrubber unavailable; blocking cloud fallback.")
            return None
        except _INFERENCE_RECOVERABLE_ERRORS as scrub_exc:
            _record_inference_degradation(
                scrub_exc,
                severity="critical",
                action="blocked cloud fallback because PII scrubbing failed",
            )
            logger.warning("PII scrubbing failed (%s); blocking cloud fallback.", scrub_exc)
            return None

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
    def _is_grounding_system_message(message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        role = str(message.get("role", "") or "").strip().lower()
        if role != "system":
            return False

        metadata = message.get("metadata", {}) or {}
        if str(metadata.get("type", "") or "").strip().lower() in {"skill_result", "tool_result"}:
            return True

        content = str(message.get("content", "") or "")
        markers = (
            "[FETCHED PAGE CONTENT]",
            "[ACTIVE GROUNDING EVIDENCE]",
            "[SKILL RESULT:",
            "[TOOL RESULT:",
        )
        return any(marker in content for marker in markers)

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
            runtime_window = max(4096, int(os.getenv("AURA_CORTEX_CTX", "8192") or 8192))
        except _INFERENCE_RECOVERABLE_ERRORS:
            runtime_window = 8192

        try:
            from core.brain.llm.model_registry import PRIMARY_ENDPOINT, get_lane_context_window

            registry_window = int(get_lane_context_window(PRIMARY_ENDPOINT) or runtime_window)
            return max(4096, min(runtime_window, registry_window))
        except _INFERENCE_RECOVERABLE_ERRORS:
            return runtime_window

    @staticmethod
    def _compact_prebuilt_message_content(role: str, content: Any) -> str:
        clean = str(content or "").strip()
        if not clean:
            return ""
        context_window = InferenceGate._foreground_prompt_context_window()

        # Keep the live foreground lane fast: target the *runtime* context
        # window instead of the model family's theoretical max so prompt eval
        # does not balloon into 5k+ tokens on desktop.
        prompt_budget_chars = max(14000, int(max(4096, context_window - 1536) * 2.25))
        limits = {
            "system": min(9000, max(6000, int(prompt_budget_chars * 0.40))),
            "user": min(16000, max(5000, int(prompt_budget_chars * 0.46))),
            "assistant": min(7000, max(3500, int(prompt_budget_chars * 0.20))),
        }
        limit = limits.get(role, 8000)
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1].rstrip() + "…"

    def _compact_prebuilt_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        history_limit: int = 12,
        deep_probe: bool = False,
    ) -> list[dict[str, str]]:
        """Trim oversized prebuilt chat payloads for the live 32B lane.

        Many callers already assemble messages upstream. For fast foreground turns,
        we keep the latest system prompt plus only the most recent compact dialogue
        snippets so first-turn Cortex doesn't spend tens of seconds re-reading old
        transcripts or giant contract blocks.
        """
        if not isinstance(messages, list):
            return []

        system_message: dict[str, str] | None = None
        preserved_system_messages: list[dict[str, str]] = []
        convo: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip().lower()
            content = self._compact_prebuilt_message_content(role, msg.get("content", ""))
            if not content:
                continue
            normalized = {"role": role or "user", "content": content}
            if role == "system" and system_message is None:
                system_message = normalized
            elif role == "system" and self._is_grounding_system_message(msg):
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
        if not deep_probe:
            compact.extend(preserved_system_messages[-1:])
        compact.extend(convo[-max(1, int(history_limit)) :])

        context_window = self._foreground_prompt_context_window()
        total_budget_chars = max(14000, int(max(4096, context_window - 1536) * 2.25))
        if deep_probe:
            total_budget_chars = min(total_budget_chars, 9000)

        while (
            compact
            and sum(len(str(msg.get("content", "") or "")) for msg in compact) > total_budget_chars
        ):
            removable_index = None
            for idx, msg in enumerate(compact):
                if idx == 0 and msg.get("role") == "system":
                    continue
                if msg.get("role") == "assistant":
                    removable_index = idx
                    break
            if removable_index is None:
                for idx, msg in enumerate(compact):
                    if idx == 0 and msg.get("role") == "system":
                        continue
                    removable_index = idx
                    break
            if removable_index is None:
                break
            compact.pop(removable_index)

        return compact

    def _flatten_messages_for_local_model(self, messages: list[dict[str, str]]) -> str:
        """Flatten Aura messages into a Qwen/ChatML prompt for local MLX models."""
        return format_chatml_messages(messages)

    async def generate(  # noqa: ASYNC109
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
        state = context.get("state")

        # Organism-first path: try to answer from the substrate+state without
        # invoking the LLM. This is bounded on purpose — the mesh handles only
        # self-reports, acknowledgements, and resource-gated responses. When it
        # does handle a request, the LLM is never called for that turn.
        if bool(context.get("allow_mesh_cognition", True)) and not bool(
            context.get("is_background", False)
        ):
            try:
                from core.consciousness.mesh_cognition import get_mesh_cognition

                mesh_decision = get_mesh_cognition().decide(prompt, state=state)
                if mesh_decision.handled:
                    context["mesh_cognition"] = mesh_decision.as_dict()
                    return self._stabilize_user_facing_text(
                        mesh_decision.response,
                        prompt,
                        is_user_facing=True,
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _mesh_exc:  # pragma: no cover - defensive
                logger.debug("Mesh-only path declined: %s", _mesh_exc)

        origin = str(context.get("origin", "") or "").lower()
        purpose = str(context.get("purpose", "") or "").lower()
        requested_tier = self._normalize_tier(context.get("prefer_tier"))
        explicit_background = "is_background" in context
        explicit_foreground = bool(context.get("foreground_request", False))
        protected_foreground_lane = bool(context.get("protected_foreground_lane", False))
        deep_probe_request = False
        try:
            from core.runtime.turn_analysis import looks_like_deep_mind_probe

            deep_probe_request = looks_like_deep_mind_probe(prompt)
        except _INFERENCE_RECOVERABLE_ERRORS:
            deep_probe_request = False
        if deep_probe_request and (explicit_foreground or self._origin_is_user_facing(origin)):
            if os.environ.get("AURA_EMBODIED_CHALLENGE"):
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
        elif not is_background:
            if origin:
                is_background = not self._origin_is_user_facing(origin)
            elif purpose in {"reply", "expression", "chat", "conversation", "user_response"}:
                is_background = False
            elif not explicit_background:
                # Origin-less requests are internal by default. User-facing turns
                # must carry an explicit origin such as api/user/voice.
                is_background = True
        deep_handoff = bool(context.get("deep_handoff", False))
        allow_cloud_fallback = bool(context.get("allow_cloud_fallback", False))
        if requested_tier == "secondary":
            deep_handoff = True
        if deep_handoff and not explicit_background:
            # Explicit deep handoffs are foreground reasoning requests even if
            # the caller forgot to stamp a user-facing origin.
            is_background = False
        if is_background:
            requested_tier = "tertiary"
            deep_handoff = False
            allow_cloud_fallback = False
            background_deferral = self._background_local_deferral_reason(origin=origin)
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
                return None

        if protected_foreground_lane and not is_background:
            self._extend_startup_quiet_window(180.0)
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

            # RAM-aware inference routing (v50): when RAM > 88%, prefer the 7B
            # brainstem over the 32B cortex. The 32B model's KV cache and
            # activations consume significant unified memory — switching to 7B
            # prevents hitting the 94% throttle wall during conversation.
            if requested_tier == "primary" and not protected_foreground_lane:
                try:
                    import psutil

                    ram_pct = psutil.virtual_memory().percent
                    # [RESILIENCE] Raised from 88% to 94%. On a 64GB system running
                    # a 32GB model, 88% is NORMAL operating state. Downgrading to
                    # brainstem at 88% was the #1 cause of "cortex disappears for
                    # semi-complex questions." 94% is the actual macOS memory
                    # throttle wall — only downgrade when genuinely in danger.
                    if ram_pct >= 94.0:
                        logger.warning(
                            "InferenceGate: RAM at %.1f%% — downgrading primary to brainstem "
                            "to prevent OOM.",
                            ram_pct,
                        )
                        requested_tier = "tertiary"
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Foreground RAM pressure probe unavailable: %s", exc)

            if requested_tier != "secondary" and self._background_memory_pressure_active():
                await self._shed_background_workers_for_memory_pressure()

        # ── Trust gate: process message through trust engine ──────────────
        # PERF FIX: The trust gate calls UserRecognizer.recognize() which
        # runs PBKDF2-SHA256 (260K iterations) on every word/phrase in the
        # prompt to check for the owner passphrase.  This blocks the event
        # loop for 3-5+ seconds on large prompts.  Fix: offload to thread
        # pool, and skip entirely for background/autonomous requests.
        _trust_guidance = ""
        # Use the fully resolved routing classification, not merely whether the
        # caller explicitly stamped `is_background`. Origin-derived background
        # work such as `origin="system"` must not pay the foreground trust-gate
        # cost or get re-promoted back into the protected Cortex lane.
        _is_bg_request = bool(is_background)
        if deep_probe_request and not _is_bg_request:
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
                if _trust_level in (TrustLevel.SOVEREIGN, TrustLevel.TRUSTED, TrustLevel.GUEST):
                    # Trust should keep ordinary conversation on the primary
                    # Cortex lane, but it must not turn an explicitly deep
                    # request into an untouchable 72B allocation when headroom
                    # policy says to downgrade. Preserve safety downgrades for
                    # secondary handoffs.
                    if requested_tier != "secondary":
                        protected_foreground_lane = True
                        requested_tier = "primary"
                        logger.info(
                            "🎭 %s user recognized. Enforcing primary cortex lane (32B).",
                            _trust_level.name,
                        )
                    else:
                        logger.info(
                            "🎭 %s user recognized. Keeping the explicit secondary handoff eligible for normal headroom checks.",
                            _trust_level.name,
                        )

                # Inject trust level into state for ContextAssembler visibility
                if hasattr(state, "cognition") and hasattr(state.cognition, "modifiers"):
                    state.cognition.modifiers["trust_level"] = _trust_level

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

        timeout_val = timeout or self._default_timeout_for_request(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
        )
        primary_timeout, fallback_timeout = self._split_attempt_timeouts(
            timeout_val, requested_tier
        )
        max_tokens = int(
            context.get("max_tokens")
            or self._default_max_tokens_for_request(
                origin,
                requested_tier,
                deep_handoff=deep_handoff,
                is_background=is_background,
            )
        )
        if "max_tokens" not in context:
            max_tokens = self._adaptive_max_tokens_for_prompt(
                prompt,
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
                        "🛡️ InferenceGate: cortex is %s; refusing secondary handoff to avoid 32B/72B memory thrash. Staying on primary.",
                        _cortex_state,
                    )
                    requested_tier = "primary"
                    deep_handoff = False
            except _INFERENCE_RECOVERABLE_ERRORS as _swap_exc:
                record_degradation(
                    "inference_gate",
                    _swap_exc,
                    severity="warning",
                    action="kept current tier after secondary admission probe failed",
                )
                logger.debug("Cortex lane probe before secondary admission failed: %s", _swap_exc)

        admission_snapshot: dict[str, Any] | None = None
        if not is_background and requested_tier in {"primary", "secondary"}:
            admission_snapshot = await self._enforce_foreground_admission(
                requested_tier,
                protected_foreground=protected_foreground_lane,
            )
            if (
                not admission_snapshot.get("can_admit", True)
                and requested_tier == "secondary"
                and not protected_foreground_lane
            ):
                logger.warning(
                    "🛡️ InferenceGate: deep local handoff exceeds safe headroom "
                    "(pressure=%.1f%% available=%.1fGB). Downgrading to the primary lane.",
                    float(admission_snapshot.get("pressure_pct", 0.0) or 0.0),
                    float(admission_snapshot.get("available_gb", 0.0) or 0.0),
                )
                requested_tier = "primary"
                deep_handoff = False
                timeout_val = timeout or self._default_timeout_for_request(
                    origin,
                    requested_tier,
                    deep_handoff=deep_handoff,
                    is_background=is_background,
                )
                primary_timeout, fallback_timeout = self._split_attempt_timeouts(
                    timeout_val, requested_tier
                )
                max_tokens = int(
                    context.get("max_tokens")
                    or self._default_max_tokens_for_request(
                        origin,
                        requested_tier,
                        deep_handoff=deep_handoff,
                        is_background=is_background,
                    )
                )
                if "max_tokens" not in context:
                    max_tokens = self._adaptive_max_tokens_for_prompt(
                        prompt,
                        base_tokens=max_tokens,
                        origin=origin,
                        requested_tier=requested_tier,
                        is_background=is_background,
                    )
                admission_snapshot = await self._enforce_foreground_admission(
                    requested_tier,
                    protected_foreground=protected_foreground_lane,
                )
            # [RESTORED] Removed the artificial cap. Aura maintains her full voice
            # even when the system is under load. Throttling tokens just creates
            # "slop" responses and frustrates the user.

        # ── Resource Stakes: scale token budget by computational survival state ──
        try:
            from core.consciousness.resource_stakes import get_resource_stakes

            token_mult = get_resource_stakes().get_token_budget_multiplier()
            if token_mult < 0.95:
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
        try:
            from core.container import ServiceContainer

            stakes = ServiceContainer.get("resource_stakes", default=None)
            if stakes is not None and hasattr(stakes, "action_envelope"):
                envelope = stakes.action_envelope("high" if deep_handoff else "normal")
                if not envelope.allowed:
                    requested_tier = "primary"
                    deep_handoff = False
                    max_tokens = min(max_tokens, 128)
                    context["resource_stakes_blocked"] = True
                else:
                    max_tokens = min(max_tokens, max(1, int(envelope.max_tokens)))
                    if "large_model_cortex" in set(envelope.disabled_capabilities):
                        requested_tier = "primary"
                        deep_handoff = False
                context["resource_stakes_envelope"] = envelope.as_dict()
        except _INFERENCE_RECOVERABLE_ERRORS as _stakes_exc:
            record_degradation(
                "inference_gate",
                _stakes_exc,
                severity="warning",
                action="kept default resource-stakes action envelope",
            )
            logger.debug("ResourceStakesLedger unavailable: %s", _stakes_exc)

        # ── Phi (Integrated Information): scale token budget based on cognitive integration ──
        try:
            from core.container import ServiceContainer
            phi_val = 1.0  # default
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None:
                if hasattr(phi_core, "get_live_phi"):
                    phi_val = max(0.0, float(phi_core.get_live_phi(include_surrogate=True)))
                elif hasattr(phi_core, "_last_result") and phi_core._last_result:
                    phi_val = float(phi_core._last_result.phi_s)
            
            # Scale token budget:
            # When Φ is high (highly integrated thought), we allow maximum token budget.
            # When Φ is low (< 0.8), the budget is dynamically scaled down (min 20%).
            # This forces the model to be extremely concise and structured when integration is compromised.
            if phi_val < 0.8:
                phi_scale = 0.2 + 0.8 * (phi_val / 0.8)
                max_tokens = max(256, int(max_tokens * phi_scale))
                logger.info("🧠 [PHI CONTROL] Integration Φ=%.3f -> scaling token budget by %.2f (max_tokens=%d)", 
                            phi_val, phi_scale, max_tokens)
        except Exception as exc:
            logger.debug("Phi token budget scaling skipped: %s", exc)

        # ── Affective Circumplex: let somatic state modulate generation params ──
        # Only applies on user-facing, non-background requests. Background tasks
        # run at fixed params to avoid thermal feedback loops.
        somatic_temperature: float | None = None
        morpho_kwargs: dict[str, Any] = {}
        if not is_background and self._origin_is_user_facing(origin):
            try:
                from core.affect.affective_circumplex import get_circumplex

                circumplex_params = get_circumplex().get_llm_params()
                if not context.get("max_tokens"):
                    max_tokens = max(
                        384,
                        min(max_tokens, int(circumplex_params["max_tokens"])),
                    )
                somatic_temperature = circumplex_params["temperature"]
                logger.debug(
                    "💓 Circumplex: V=%.2f A=%.2f → temp=%.2f tokens=%d",
                    circumplex_params["valence"],
                    circumplex_params["arousal"],
                    somatic_temperature,
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
                    if somatic_temperature is not None:
                        somatic_temperature = round(somatic_temperature * _mods.temperature_mod, 3)
                    max_tokens = max(384, int(max_tokens * _mods.depth_mod))
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
                            somatic_temperature + _h_mods["temperature_mod"], 3
                        )
                        somatic_temperature = max(0.1, min(1.5, somatic_temperature))
                    max_tokens = max(384, int(max_tokens * _h_mods["token_multiplier"]))
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

            # ── Morphogenetic Substrate (True Embodied Cognition) ────────────
            # Curing Mind-Body Dualism: The physical tissue state directly alters
            # the structural generation parameters (temperature, top_p, etc)
            try:
                from core.container import ServiceContainer

                _rt = ServiceContainer.get("morphogenetic_runtime", default=None)
                if _rt is not None:
                    _f = _rt.field.sample("global")
                    _danger = _f.get("danger", 0.0)
                    _curiosity = _f.get("curiosity", 0.0)
                    _resource_pressure = _f.get("resource_pressure", 0.0)

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

                    if _resource_pressure > 0.5:
                        max_tokens = int(max_tokens * (1.0 - (_resource_pressure * 0.5)))
                        max_tokens = max(128, max_tokens)

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
            except _INFERENCE_RECOVERABLE_ERRORS as _m_e:
                record_degradation(
                    "inference_gate",
                    _m_e,
                    severity="warning",
                    action="continued without morphogenetic generation-parameter coupling",
                )
                logger.debug("Morphogenetic coupling unavailable: %s", _m_e)

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
        ):
            foreground_floor = max(
                384,
                int(os.environ.get("AURA_FOREGROUND_CHAT_MIN_TOKENS", "3072")),
            )
            if max_tokens < foreground_floor:
                logger.info(
                    "🧠 Foreground chat token floor raised budget %d→%d for origin=%s.",
                    max_tokens,
                    foreground_floor,
                    origin or "unknown",
                )
                max_tokens = foreground_floor

        if deep_probe_request and not is_background:
            probe_token_cap = int(os.environ.get("AURA_DEEP_PROBE_MAX_TOKENS", "384"))
            max_tokens = min(max_tokens, max(128, probe_token_cap))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False

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
        )
        provided_messages = context.get("messages")
        if not isinstance(provided_messages, list):
            provided_messages = None
        if provided_messages is not None:
            system_prompt = ""
            for msg in provided_messages:
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "system":
                    system_prompt = str(msg.get("content", "") or "").strip()
                    break
            living_mind_context = ""
            if not is_background and self._origin_is_user_facing(origin):
                living_mind_context = await self._build_compact_living_mind_context(prompt, origin)
        elif use_compact_foreground_context:
            system_prompt = self._build_compact_system_prompt(brief)
            living_mind_context = await self._build_compact_living_mind_context(prompt, origin)
        else:
            system_prompt = self._build_system_prompt(brief)
            # [STABILITY v50] Hard 5s timeout on full context assembly.
            # The 20+ consciousness subsystems queried here can individually
            # hang due to lock contention or slow I/O. When that happens,
            # fall back to the compact (4-subsystem) version so generation
            # budget is never consumed by context assembly.
            try:
                living_mind_context = await asyncio.wait_for(
                    self._build_living_mind_context(prompt, origin),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(
                    "⚠️ [STABILITY] Full living mind context assembly exceeded 5s budget. "
                    "Falling back to compact context to preserve generation headroom."
                )
                living_mind_context = await self._build_compact_living_mind_context(prompt, origin)
        if living_mind_context:
            system_prompt = f"{system_prompt}\n\n{living_mind_context}"
        # Keep prompt growth aligned with the actual local model context window
        # instead of assuming 128k+ headroom on the primary Qwen lane.

        # ── Somatic narrative: brief felt-state line in the system prompt ────────
        if somatic_temperature is not None:
            try:
                from core.affect.affective_circumplex import get_circumplex

                _soma_narrative = get_circumplex().describe()
                if _soma_narrative:
                    system_prompt = f"{system_prompt}\n\n## SOMATIC STATE\n{_soma_narrative}"
            except _INFERENCE_RECOVERABLE_ERRORS as _exc:
                record_degradation(
                    "inference_gate",
                    _exc,
                    severity="warning",
                    action="continued without somatic-state prompt section",
                )
                logger.debug("Suppressed Exception: %s", _exc)

        prompt_user_facing = bool(
            not is_background
            and (
                self._origin_is_user_facing(origin)
                or explicit_foreground
                or purpose in {"reply", "expression", "chat", "conversation", "user_response"}
            )
        )

        # ── Architecture Self-Awareness: inject relevant subsystem context ──────
        # Only for user-facing requests that mention architecture/code keywords.
        if prompt_user_facing:
            try:
                import re as _re

                _arch_triggers = _re.compile(
                    r"\b(how|explain|what|which|where|why|trace|show|describe)\b.{0,60}"
                    r"\b(module|subsystem|file|class|method|function|work|does|handles|manages|routes|sends|wires)\b",
                    _re.IGNORECASE,
                )
                if _arch_triggers.search(prompt):
                    from core.self.architecture_index import get_architecture_index

                    arch_excerpt = get_architecture_index().query(prompt, max_results=3)
                    if arch_excerpt:
                        system_prompt = f"{system_prompt}\n\n{arch_excerpt}"
            except _INFERENCE_RECOVERABLE_ERRORS as _ae:
                record_degradation(
                    "inference_gate",
                    _ae,
                    severity="warning",
                    action="continued without architecture self-awareness excerpt",
                )
                logger.debug("ArchIndex injection skipped: %s", _ae)
            system_prompt = f"{system_prompt}\n\n{conversation_reliability_system_block(prompt)}"
        history = context.get("history", [])
        use_rich_context = bool(
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
            if prompt_user_facing or living_mind_context:
                reliability_block = conversation_reliability_system_block(prompt)
                inserted = False
                for msg in messages:
                    if str(msg.get("role", "") or "").strip().lower() == "system":
                        content = str(msg.get("content", "") or "")
                        if living_mind_context and living_mind_context not in content:
                            content = f"{content.rstrip()}\n\n{living_mind_context}".strip()
                        if "USER-FACING CONVERSATION RELIABILITY CONTRACT" not in content:
                            content = f"{content.rstrip()}\n\n{reliability_block}".strip()
                        msg["content"] = content
                        inserted = True
                        break
                if not inserted:
                    blocks = [
                        block
                        for block in (
                            living_mind_context,
                            reliability_block if prompt_user_facing else "",
                        )
                        if block
                    ]
                    messages.insert(0, {"role": "system", "content": "\n\n".join(blocks)})
        else:
            messages = (
                self._build_messages(prompt, system_prompt, history)
                if use_rich_context
                else self._build_compact_messages(prompt, system_prompt, history)
            )
        if provided_messages is not None and use_compact_foreground_context:
            deep_probe_context = bool(context.get("deep_mind_probe", False))
            messages = self._compact_prebuilt_messages(
                messages,
                history_limit=2 if deep_probe_context else 12,
                deep_probe=deep_probe_context,
            )
        prompt_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        prompt_mode = "rich" if use_rich_context else "compact"
        if use_compact_foreground_context:
            prompt_mode = "compact_foreground"
        if provided_messages is not None:
            prompt_mode = f"{prompt_mode}_prebuilt"
        logger.info(
            "🧠 [ZENITH] Prompt plan: mode=%s messages=%d chars=%d origin=%s max_tokens=%d",
            prompt_mode,
            len(messages),
            prompt_chars,
            origin or "unknown",
            max_tokens,
        )

        _is_user_facing = self._origin_is_user_facing(origin) or requested_tier == "primary"
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
                fallback_client = get_mlx_client(model_path=str(get_brainstem_path()))
                fallback_label = BRAINSTEM_ENDPOINT
                restore_primary = False

                primary_restored_inline = False
                try:
                    if requested_tier == "tertiary":
                        local_client = get_mlx_client(model_path=str(get_brainstem_path()))
                        local_label = BRAINSTEM_ENDPOINT
                        fallback_client = get_mlx_client(
                            model_path=str(get_fallback_path()), device="cpu"
                        )
                        fallback_label = FALLBACK_ENDPOINT
                    elif deep_handoff:
                        local_client = get_mlx_client(model_path=str(get_deep_model_path()))
                        local_label = DEEP_ENDPOINT
                        fallback_client = get_mlx_client(
                            model_path=str(get_runtime_model_path(ACTIVE_MODEL))
                        )
                        fallback_label = PRIMARY_ENDPOINT
                        restore_primary = True

                    protected_deep_fallback = bool(
                        bool(context.get("allow_deep_fallback", False))
                        and deep_probe_request
                        and _is_user_facing
                        and requested_tier == "primary"
                    )
                    if protected_deep_fallback:
                        fallback_client = get_mlx_client(model_path=str(get_deep_model_path()))
                        fallback_label = DEEP_ENDPOINT
                    skip_initial_primary_attempt = False
                    primary_warmup_memory_deferred = False
                    lane_managed_client = hasattr(local_client, "get_lane_status") or hasattr(
                        local_client, "warmup"
                    )
                    if _is_user_facing and local_label == PRIMARY_ENDPOINT and lane_managed_client:
                        lane_status = self.get_conversation_status()
                        if not lane_status.get("conversation_ready"):
                            logger.info(
                                "🧠 %s lane is not ready yet (state=%s). Completing foreground warmup before first generation attempt.",
                                local_label,
                                lane_status.get("state", "unknown"),
                            )
                            try:
                                # [STABILITY v56] The 32B model can take 150s to cold load.
                                # Don't artificially cap warmup at 90s. Give it at least 180s
                                # or the primary timeout, whichever is greater.
                                lane_status = await self.ensure_foreground_ready(
                                    timeout=max(180.0, primary_timeout)
                                )
                            except (
                                TimeoutError,
                                RuntimeError,
                                AttributeError,
                                TypeError,
                                ValueError,
                                OSError,
                            ) as warmup_exc:
                                record_degradation(
                                    "inference_gate",
                                    warmup_exc,
                                    severity="degraded",
                                    action="skipped cold primary attempt or fell back after foreground warmup failure",
                                )
                                if "foreground_warmup_deferred" in str(warmup_exc):
                                    primary_warmup_memory_deferred = True
                                logger.warning(
                                    "🧠 Foreground preflight warmup did not complete cleanly: %s",
                                    warmup_exc,
                                )
                                lane_status = self.get_conversation_status()
                            if not lane_status.get("conversation_ready"):
                                skip_initial_primary_attempt = True
                                logger.warning(
                                    "🧠 %s is still not ready after foreground preflight warmup (state=%s). Skipping the cold first attempt and waiting for recovery before retry.",
                                    local_label,
                                    lane_status.get("state", "unknown"),
                                )
                    if primary_warmup_memory_deferred:
                        logger.warning(
                            "🧠 Cortex cold-load deferred by RAM admission; routing this foreground turn to %s.",
                            fallback_label,
                        )
                        local_client = fallback_client
                        local_label = fallback_label
                        skip_initial_primary_attempt = False
                    logger.info(
                        "🧠 Routing to %s (timeout=%.0fs, user_facing=%s)...",
                        local_label,
                        float(timeout_val),
                        _is_user_facing,
                    )
                    primary_deadline = get_deadline(primary_timeout)
                    if skip_initial_primary_attempt:
                        text = None
                    else:
                        async with self._resource_context(
                            enabled=local_label != FALLBACK_ENDPOINT,
                            priority=_is_user_facing,
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
                                foreground_request=_is_user_facing,
                                **morpho_kwargs,
                            )
                    if text:
                        return self._stabilize_user_facing_text(
                            text,
                            prompt,
                            is_user_facing=_is_user_facing,
                        )

                    # ── CORTEX RETRY: For user-facing requests, retry the primary model
                    # The stall detector reboots the worker, so we wait for recovery.
                    # [RESILIENCE] We now retry TWICE. Recurrent loops + context building
                    # can easily trip a single timeout. We give it 3s, then 6s to settle.
                    if _is_user_facing and local_label == PRIMARY_ENDPOINT:
                        for retry_attempt, wait_sec in enumerate([3.0, 6.0], 1):
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry loop aborted: runtime is shutting down.",
                                    local_label,
                                )
                                return ""
                            lane_status = self.get_conversation_status()
                            if not lane_status.get("conversation_ready"):
                                logger.warning(
                                    "🧠 %s returned no text before the conversation lane was ready (state=%s). Forcing foreground warmup before retry %d.",
                                    local_label,
                                    lane_status.get("state", "unknown"),
                                    retry_attempt,
                                )
                                try:
                                    # [STABILITY v56] Same as above, don't cap at 60s.
                                    await self.ensure_foreground_ready(
                                        timeout=max(180.0, primary_timeout)
                                    )
                                except _INFERENCE_RECOVERABLE_ERRORS as warmup_exc:
                                    record_degradation(
                                        "inference_gate",
                                        warmup_exc,
                                        severity="degraded",
                                        action="continued cortex retry path after foreground warmup retry failure",
                                    )
                                    logger.warning(
                                        "🧠 Foreground warmup retry did not complete cleanly: %s",
                                        warmup_exc,
                                    )
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry wait skipped: runtime is shutting down.",
                                    local_label,
                                )
                                return ""

                            logger.warning(
                                "🧠 %s returned no text on user-facing request. Retrying (attempt %d/2) after %ds pause...",
                                local_label,
                                retry_attempt,
                                wait_sec,
                            )
                            await asyncio.sleep(wait_sec)
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry generation skipped: runtime is shutting down.",
                                    local_label,
                                )
                                return ""

                            # Give the retry MORE time than the initial attempt
                            retry_timeout = primary_timeout * 1.5
                            retry_deadline = get_deadline(retry_timeout)
                            async with self._resource_context(
                                enabled=True,
                                priority=True,
                                worker=local_label,
                                timeout_s=retry_deadline.remaining or retry_timeout,
                            ):
                                text = await self._generate_with_client(
                                    local_client,
                                    prompt,
                                    system_prompt,
                                    history,
                                    retry_deadline,
                                    f"{local_label}-RETRY-{retry_attempt}",
                                    messages=messages,
                                    max_tokens=max_tokens,
                                    temperature=somatic_temperature,
                                    origin=origin,
                                    is_background=is_background,
                                    foreground_request=True,
                                    **morpho_kwargs,
                                )
                            if text:
                                logger.info(
                                    "✅ %s retry %d succeeded (len=%d)",
                                    local_label,
                                    retry_attempt,
                                    len(text),
                                )
                                return self._stabilize_user_facing_text(
                                    text,
                                    prompt,
                                    is_user_facing=_is_user_facing,
                                )

                        logger.warning("🧠 %s all retries failed.", local_label)
                        # For user-facing requests, skip brainstem — go straight to cloud
                        if allow_cloud_fallback:
                            logger.warning(
                                "🧠 Escalating to cloud before brainstem for user-facing request."
                            )
                            raise _UserFacingCortexError()
                        lane_status = self.get_conversation_status()
                        if (
                            not protected_deep_fallback
                            and not lane_status.get("conversation_ready")
                            and str(lane_status.get("state", "") or "").lower() != "failed"
                        ):
                            logger.warning(
                                "🧠 %s is still not ready (state=%s). Refusing local fallback until the primary lane actually finishes recovery.",
                                local_label,
                                lane_status.get("state", "unknown"),
                            )
                            return None
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
                            return None

                    # Graceful local fallback: for background/autonomous requests, the
                    # brainstem is an acceptable degradation. For user-facing requests
                    # that reach here (cloud disabled), it's the last local resort.
                    fallback_deadline = get_deadline(fallback_timeout)
                    async with self._resource_context(
                        enabled=fallback_label != FALLBACK_ENDPOINT,
                        priority=_is_user_facing,
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
                            foreground_request=_is_user_facing,
                            **morpho_kwargs,
                        )
                    if brainstem_text:
                        if fallback_label == PRIMARY_ENDPOINT:
                            primary_restored_inline = True
                        return self._stabilize_user_facing_text(
                            brainstem_text,
                            prompt,
                            is_user_facing=_is_user_facing,
                        )
                    logger.warning("🧠 Local fallback returned no text.")
                finally:
                    if restore_primary and not primary_restored_inline:
                        self._schedule_primary_restore_after_deep_handoff()

            except _UserFacingCortexError:
                logger.warning(
                    "🧠 User-facing Cortex failure — bypassing brainstem, escalating to cloud."
                )
            except TimeoutError as timeout_exc:
                logger.warning("🛑 Local inference TIMED OUT (Budget: %.0fs).", timeout_val)
                if (
                    not is_background
                    and self._origin_is_user_facing(origin)
                    and not allow_cloud_fallback
                ):
                    raise TimeoutError(
                        f"{local_label} timed out after {timeout_val:.0f}s"
                    ) from timeout_exc
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                record_degradation(
                    "inference_gate",
                    e,
                    severity="degraded",
                    action="fell through to reflex or cloud fallback after local inference failure",
                )
                logger.warning("🛑 Local inference FAILURE: %s", e)

        # 1.5. EMERGENCY REFLEX FALLBACK — tiny 1.5B model on CPU as absolute last local resort.
        # If Cortex AND Brainstem both failed for a user-facing request, the 1.5B Reflex
        # model can still produce SOMETHING so the user isn't left hanging.
        # [STABILITY v54] Never run the 1.5B reflex if we are in a protected 32B foreground lane.
        if _is_user_facing and not is_background and not protected_deep_fallback:
            try:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import get_fallback_path

                reflex_client = get_mlx_client(model_path=str(get_fallback_path()), device="cpu")
                if reflex_client:
                    logger.warning(
                        "🆘 [REFLEX] Cortex + Brainstem both failed. Trying 1.5B CPU Reflex..."
                    )
                    reflex_deadline = get_deadline(15.0)  # 15s hard limit for tiny model
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
                            prompt,
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

        # 2. Optional cloud fallback.
        if not allow_cloud_fallback:
            logger.error("Local inference paths exhausted. Cloud fallback disabled.")
            # STABILITY FIX: For user-facing requests, trigger immediate cortex recovery
            # and return a genuine acknowledgment instead of None (which causes "I'm having trouble")
            if _is_user_facing:
                # [BUG FIX] Force-kill stuck worker and drain queues IMMEDIATELY.
                # Without this, the old worker's IPC feeder threads stay blocked on
                # nwait(), starving the event loop and causing tick stalls that kill
                # the WebSocket connection. The recovery task below will respawn cleanly.
                if self._mlx_client and hasattr(self._mlx_client, "_process"):
                    try:
                        proc = self._mlx_client._process
                        is_running = (
                            proc.is_alive() if hasattr(proc, "is_alive") else (proc.poll() is None)
                        )
                        if proc and is_running:
                            logger.warning(
                                "🧹 [CASCADE CLEANUP] Force-killing stuck cortex worker pid=%s",
                                getattr(proc, "pid", "unknown"),
                            )
                            proc.kill()
                            if hasattr(proc, "join"):
                                proc.join(timeout=2.0)
                            elif hasattr(proc, "wait"):
                                proc.wait(timeout=2.0)
                        if hasattr(self._mlx_client, "_drain_queue"):
                            self._mlx_client._drain_queue()
                        # Replace queues to sever any stuck feeder threads
                        _safe_close = getattr(self._mlx_client, "_safe_close_queue", None)
                        import multiprocessing as _mp

                        if hasattr(self._mlx_client, "_req_q"):
                            try:
                                self._mlx_client._req_q.close()
                            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                                logger.debug(
                                    "Request queue close skipped during cascade cleanup: %s", exc
                                )
                            self._mlx_client._req_q = _mp.Queue(maxsize=10)
                        if hasattr(self._mlx_client, "_res_q"):
                            try:
                                self._mlx_client._res_q.close()
                            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                                logger.debug(
                                    "Response queue close skipped during cascade cleanup: %s", exc
                                )
                            self._mlx_client._res_q = _mp.Queue(maxsize=10)
                        self._mlx_client._process = None
                        self._mlx_client._init_done = False
                        logger.info("🧹 [CASCADE CLEANUP] Stuck worker killed, queues replaced.")
                    except _INFERENCE_RECOVERABLE_ERRORS as cleanup_exc:
                        record_degradation(
                            "inference_gate",
                            cleanup_exc,
                            severity="warning",
                            action="continued recovery scheduling after cascade cleanup error",
                        )
                        logger.debug("Cascade cleanup error (non-fatal): %s", cleanup_exc)
                # Force cortex recovery in background
                if not self._cortex_recovery_in_progress:
                    recovery_coro = self._respawn_cortex_if_needed()
                    task = get_task_tracker().create_task(recovery_coro)
                    if not isinstance(task, asyncio.Task):
                        recovery_coro.close()
                # Give cortex time to recover before next request hits a dead endpoint
                self._extend_startup_quiet_window(15.0)
                # Reset the UnitaryResponsePhase circuit breaker so next attempt works
                try:
                    from core.resilience.error_boundary import CircuitRegistry
                    from core.utils.resilience import CircuitState

                    breaker = CircuitRegistry.get_instance().get_breaker(
                        "phase:UnitaryResponsePhase"
                    )
                    if breaker.state != CircuitState.CLOSED:
                        breaker.state = CircuitState.HALF_OPEN
                        breaker.reset_timeout = min(breaker.reset_timeout, 15.0)
                        logger.info("Reset UnitaryResponsePhase circuit to HALF_OPEN for recovery")
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Circuit-breaker recovery reset unavailable: %s", exc)
                return self._user_facing_recovery_response(prompt)
            return None

        if time.monotonic() < self._cloud_backoff_until:
            logger.warning("Cloud fallback cooling down. Skipping remote retry.")
            return None

        try:
            from core.container import ServiceContainer

            # PII SCRUBBING: Strip personal identifiers before sending to cloud.
            # biography_private.json data (real names, trust scores, relationship
            # labels) must never leave the local machine. The scrubber replaces
            # PII with neutral replacements while preserving conversational context.
            scrubbed_payload = self._scrub_cloud_payload(system_prompt, prompt)
            if scrubbed_payload is None:
                return self._user_facing_recovery_response(prompt) if _is_user_facing else None
            cloud_system_prompt, cloud_prompt = scrubbed_payload

            # Try APIAdapter first (cleaner Gemini integration)
            adapter = ServiceContainer.get("api_adapter", default=None)
            if adapter and getattr(adapter, "has_gemini", False):
                logger.info("☁️ Falling back to Gemini via APIAdapter...")
                result = await asyncio.wait_for(
                    adapter.generate(
                        f"{cloud_system_prompt}\n\nUser: {cloud_prompt}\nAura:",
                        {"model_tier": "api_fast", "max_tokens": 800, "temperature": 0.7},
                    ),
                    timeout=30.0,
                )
                if result and result.strip():
                    try:
                        from core.consciousness.closed_loop import notify_closed_loop_output

                        notify_closed_loop_output(result.strip())
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        record_degradation(
                            "inference_gate",
                            exc,
                            severity="warning",
                            action="returned cloud result without closed-loop output notification",
                        )
                        logger.debug("Cloud output notification skipped: %s", exc)
                    return self._stabilize_user_facing_text(
                        result.strip(),
                        prompt,
                        is_user_facing=_is_user_facing,
                    )

            # Try HealthRouter as secondary cloud path (also PII-scrubbed)
            router = ServiceContainer.get("llm_router", default=None)
            if router:
                logger.info("☁️ Falling back to HealthRouter...")
                result = await asyncio.wait_for(
                    router.think(cloud_prompt, system_prompt=cloud_system_prompt),
                    timeout=30.0,
                )
                if isinstance(result, str) and result.strip():
                    try:
                        from core.consciousness.closed_loop import notify_closed_loop_output

                        notify_closed_loop_output(result.strip())
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        record_degradation(
                            "inference_gate",
                            exc,
                            severity="warning",
                            action="returned router cloud result without closed-loop output notification",
                        )
                        logger.debug("Router output notification skipped: %s", exc)
                    return self._stabilize_user_facing_text(
                        result.strip(),
                        prompt,
                        is_user_facing=_is_user_facing,
                    )
        except _INFERENCE_RECOVERABLE_ERRORS as cloud_err:
            record_degradation(
                "inference_gate",
                cloud_err,
                severity="degraded",
                action="entered cloud backoff when applicable and returned exhausted-inference fallback",
            )
            cloud_err_text = str(cloud_err)
            if "429" in cloud_err_text or "quota" in cloud_err_text.lower():
                self._cloud_backoff_until = time.monotonic() + 60.0
            logger.error("☁️ Cloud fallback failed: %s", cloud_err)

        # All inference paths exhausted. Return None so callers can handle
        # gracefully without the error text leaking to TTS or the user.
        logger.error("All inference paths exhausted (Local + Cloud)")
        if _is_user_facing:
            return self._user_facing_recovery_response(prompt)
        return None

    def _post_inference_update(self, response_text: str):
        """Update downstream systems after each inference completes.

        Closes the bidirectional causal loop:
          CRSM ← response (updates self-state)
          HOT  ← response (reflexive modification)
          Hedonic ← response quality signal
        """
        self._last_successful_generation_at = time.time()
        if not response_text or not response_text.strip():
            return
        try:
            from core.consciousness.crsm import get_crsm

            get_crsm().post_inference_update(response_text)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.consciousness.hot_engine import get_hot_engine

            hot = get_hot_engine()
            hot.apply_feedback()  # apply any pending reflexive modifications
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
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
                    action="skipped unavailable post-inference update hook after response delivery",
                )
                logger.debug("Suppressed Exception: %s", _exc)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        # ══════════════════════════════════════════════════════════════════
        # DEEPENED POST-INFERENCE FEEDBACK LOOPS
        # ══════════════════════════════════════════════════════════════════

        # ── Credit Assignment: Record response quality ────────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit:
                # Response quality heuristic: length, structure, substance
                response_len = len(response_text.strip())
                has_structure = any(marker in response_text for marker in ["\n", "- ", "1.", "```"])
                quality = min(1.0, (response_len / 500.0) * 0.6 + (0.4 if has_structure else 0.1))
                credit.assign_credit(
                    action_id=f"inference_{int(time.time())}",
                    outcome=quality,
                    domain="chat",
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in credit feedback: %s", _exc)

        # ── Homeostasis: Response success signal ──────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "on_response_success"):
                homeostasis.on_response_success(response_length=len(response_text))
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in homeostasis feedback: %s", _exc)

        # ── Theory of Mind: Update user model from response ───────────────
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and hasattr(tom, "update_from_response"):
                tom.update_from_response(
                    user_id="default_user",
                    response_text=response_text,
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in ToM feedback: %s", _exc)

        # ── World Model: Extract beliefs from response ────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "extract_beliefs_from_response"):
                if len(response_text) > 100:
                    world_model.extract_beliefs_from_response(response_text)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in world model feedback: %s", _exc)

    async def think(self, prompt: str, system_prompt: str = "", **kwargs) -> str | None:
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

        for key in (
            "history",
            "messages",
            "max_tokens",
            "temperature",
            "temp",
            "schema",
            "deep_handoff",
            "allow_cloud_fallback",
            "prefer_tier",
            "origin",
            "purpose",
            "is_background",
            "foreground_request",
            "protected_foreground_lane",
            "state",
            "skip_runtime_payload",
        ):
            if key in kwargs:
                context[key] = kwargs[key]
        result = await self.generate(prompt, context=context, timeout=timeout)
        if isinstance(result, str) and result.strip():
            # Close bidirectional causal loop after each inference
            self._post_inference_update(result)
            return result
        return None

    def is_alive(self) -> bool:
        """Check if the MLX client is operational."""
        if self._mlx_client and hasattr(self._mlx_client, "is_alive"):
            return self._mlx_client.is_alive()
        return False
