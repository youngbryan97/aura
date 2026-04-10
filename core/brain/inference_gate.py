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
import inspect
import json
import logging
import os
import threading as _threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import psutil

from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
)
from core.runtime.desktop_boot_safety import desktop_safe_boot_enabled
from core.utils.deadlines import Deadline, get_deadline

logger = logging.getLogger("Aura.InferenceGate")

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


class _UserFacingCortexFailure(Exception):
    """Sentinel: Cortex failed on a user-facing request — skip brainstem, escalate to cloud."""


@asynccontextmanager
async def _thread_lock_context(
    lock: Any,
    *,
    timeout: Optional[float] = None,
    label: str = "lock",
):
    if timeout is None:
        acquired = await asyncio.to_thread(lock.acquire)
    else:
        acquired = await asyncio.to_thread(lock.acquire, True, max(0.0, float(timeout)))
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
        self._cached_identity_prompt: Optional[str] = None
        self._identity_prompt_time: float = 0.0
        self._cloud_backoff_until: float = 0.0
        self._cortex_recovery_in_progress: bool = False
        self._last_cortex_check: float = 0.0
        self._cortex_recovery_attempts: int = 0
        self._last_successful_generation_at: float = time.time()
        self._prewarm_task: Optional[asyncio.Task] = None
        self._deferred_prewarm_task: Optional[asyncio.Task] = None
        self._foreground_ready_lock = _threading.Lock()
        self._last_background_memory_shed_at: float = 0.0
        logger.info("🛡️ InferenceGate created.")

    @staticmethod
    def _desktop_safe_boot_enabled() -> bool:
        return desktop_safe_boot_enabled()

    @staticmethod
    def _boot_should_eager_warmup() -> bool:
        """Keep the 32B lane warm on high-memory desktops unless explicitly disabled."""
        if InferenceGate._desktop_safe_boot_enabled():
            logger.info("🛡️ Desktop safe boot active — skipping eager 32B warmup during launch.")
            return False
        setting = str(os.environ.get("AURA_EAGER_CORTEX_WARMUP", "auto")).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            return True
        if setting in {"0", "false", "no", "off"}:
            return False

        try:
            vm = psutil.virtual_memory()
            total_gb = vm.total / float(1024 ** 3)
            min_total_gb = float(os.environ.get("AURA_BOOT_WARMUP_MIN_TOTAL_GB", "48"))
            if total_gb >= 60.0:
                # On 64GB-class machines the conversation lane is a core desktop
                # feature, not an opportunistic luxury. Only defer under truly
                # critical pressure. The 32B model needs ~20GB; we want to keep
                # the cortex alive unless we're genuinely about to OOM.
                max_pressure = 82.0   # was 70 — much more generous
                min_available_gb = 12.0  # was 18 — 12GB is enough for 32B Q4
            else:
                max_pressure = float(os.environ.get("AURA_BOOT_WARMUP_MAX_PRESSURE_PCT", "72"))
                min_available_gb = float(os.environ.get("AURA_BOOT_WARMUP_MIN_AVAILABLE_GB", "24"))
            available_gb = vm.available / float(1024 ** 3)
            if total_gb < min_total_gb or vm.percent >= max_pressure or available_gb < min_available_gb:
                logger.warning(
                    "⏸️ Deferring eager 32B warmup at boot (total=%.1fGB pressure=%.1f%% available=%.1fGB).",
                    total_gb,
                    vm.percent,
                    available_gb,
                )
                return False
        except Exception as exc:
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

    def get_conversation_status(self) -> Dict[str, Any]:
        lane = {
            "desired_model": "Cortex (32B)",
            "desired_endpoint": PRIMARY_ENDPOINT,
            "foreground_endpoint": PRIMARY_ENDPOINT if self.is_alive() else None,
            "background_endpoint": BRAINSTEM_ENDPOINT,
            "foreground_tier": "local",
            "background_tier": "local_fast",
            "state": "failed" if self._init_error else "warming",
            "last_failure_reason": self._init_error or "",
            "conversation_ready": False,
            "cortex_recovery_attempts": getattr(self, "_cortex_recovery_attempts", 0),
            "time_since_last_success_s": max(0, time.time() - getattr(self, "_last_successful_generation_at", time.time())),
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
            lane["last_failure_reason"] = str(raw.get("last_error", "") or lane["last_failure_reason"])
            raw_ready = bool(raw.get("conversation_ready", False))
            lane["conversation_ready"] = raw_ready
            lane["last_transition_at"] = float(raw.get("last_transition_at", 0.0) or 0.0)
            lane["last_ready_at"] = float(raw.get("last_ready_at", 0.0) or 0.0)
            lane["last_progress_at"] = float(raw.get("last_progress_at", 0.0) or 0.0)
            lane["warmup_attempted"] = bool(raw.get("warmup_attempted", False))
            lane["warmup_in_flight"] = bool(raw.get("warmup_in_flight", lane["warmup_in_flight"]))
            if lane["conversation_ready"]:
                lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        lane_state = str(lane.get("state", "") or "").lower()
        recent_success = (time.time() - getattr(self, "_last_successful_generation_at", time.time())) <= 30.0
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
        if self._cortex_recovery_in_progress and not lane["conversation_ready"] and lane_state != "failed":
            lane["state"] = "recovering"
        if self._prewarm_task and not self._prewarm_task.done() and not lane["conversation_ready"] and lane_state != "failed":
            lane["state"] = "warming"
            lane["warmup_in_flight"] = True
        return lane

    def note_foreground_timeout(self, reason: str = "foreground_timeout") -> None:
        """Mark the conversation lane as degraded after a foreground timeout."""
        if self._mlx_client and hasattr(self._mlx_client, "note_lane_recovering"):
            try:
                self._mlx_client.note_lane_recovering(reason)
            except Exception as exc:
                logger.debug("Failed to mark cortex lane recovering: %s", exc)
        self._extend_startup_quiet_window(8.0)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            self._schedule_background_cortex_prewarm(delay=2.0)
        except Exception as exc:
            logger.debug("Failed to schedule deferred cortex re-prewarm after timeout: %s", exc)

    def _extend_startup_quiet_window(self, seconds: float) -> None:
        orch = self.orch
        if orch is None:
            try:
                from core.container import ServiceContainer

                orch = ServiceContainer.get("orchestrator", default=None)
            except Exception:
                orch = None
        if orch and hasattr(orch, "_extend_foreground_quiet_window"):
            try:
                orch._extend_foreground_quiet_window(seconds)
            except Exception as exc:
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
                    logger.warning("⏸️ Deferred cortex prewarm cancelled: lane is in a failed state (%s).", lane.get("last_failure_reason") or "unknown")
                    return
                if self._foreground_user_turn_active() or self._foreground_owner_active():
                    next_delay = min(20.0, max(6.0, next_delay))
                    continue
                try:
                    vm = psutil.virtual_memory()
                    total_gb = vm.total / float(1024 ** 3)
                    available_gb = vm.available / float(1024 ** 3)
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
                except Exception as exc:
                    logger.debug("Deferred prewarm memory probe failed: %s", exc)

                try:
                    self._extend_startup_quiet_window(20.0)
                    await self.ensure_foreground_ready(timeout=60.0)
                    logger.info("✅ Deferred cortex prewarm completed.")
                    return
                except Exception as exc:
                    logger.warning("⚠️ Deferred cortex prewarm failed (attempt=%d): %s", attempt, exc)
                    next_delay = min(45.0, max(12.0, next_delay * 1.5))

            logger.warning("⚠️ Deferred cortex prewarm exhausted retries; foreground turn will retry on demand.")

        self._deferred_prewarm_task = asyncio.create_task(
            _runner(),
            name="InferenceGate.deferred_cortex_prewarm",
        )

    async def ensure_foreground_ready(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Ensure the 32B conversation lane has actually attempted warmup for this turn."""
        timeout = max(15.0, float(timeout or 75.0))
        lane = self.get_conversation_status()
        if lane.get("conversation_ready"):
            return lane
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state == "failed" and lane_reason.startswith(("mlx_runtime_unavailable", "local_runtime_unavailable")):
            raise RuntimeError(lane_reason)
        if not self._mlx_client or not hasattr(self._mlx_client, "warmup"):
            raise RuntimeError("foreground_lane_unavailable")

        task: Optional[asyncio.Task] = None
        try:
            async with _thread_lock_context(
                self._foreground_ready_lock,
                timeout=min(timeout, 30.0),
                label="foreground_ready_lock",
            ):
                lane = self.get_conversation_status()
                if lane.get("conversation_ready"):
                    return lane
                if self._prewarm_task and not self._prewarm_task.done():
                    task = self._prewarm_task
                else:
                    self._extend_startup_quiet_window(20.0)
                    self._prewarm_task = asyncio.create_task(
                        self._mlx_client.warmup(),
                        name="InferenceGate.ensure_foreground_ready",
                    )
                    task = self._prewarm_task
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            if hasattr(self._mlx_client, "note_lane_recovering"):
                self._mlx_client.note_lane_recovering("foreground_warmup_timeout")
            raise
        except Exception as exc:
            if hasattr(self._mlx_client, "note_lane_failed"):
                self._mlx_client.note_lane_failed(f"foreground_warmup_failed:{type(exc).__name__}")
            raise RuntimeError("foreground_warmup_failed") from exc

        lane = self.get_conversation_status()
        if not lane.get("conversation_ready"):
            raise RuntimeError(str(lane.get("last_failure_reason") or "foreground_lane_not_ready"))
        return lane

    async def _ensure_cortex_recovery(self) -> None:
        """Proactively recover the 72B primary brain if it died (e.g., laptop sleep).

        Without this, background tasks keep the 7B alive indefinitely and the 32B
        never gets a chance to respawn because background requests are locked to
        tertiary tier.  Rate-limited to one attempt per 30s.
        """
        if not self._mlx_client:
            return
        if not hasattr(self._mlx_client, "is_alive"):
            return
        if self._mlx_client.is_alive():
            return  # Primary is fine

        now = time.monotonic()
        if (now - self._last_cortex_check) < 10.0:
            return  # Rate limit: reduced from 30s to 10s for faster cortex recovery
        self._last_cortex_check = now

        if self._cortex_recovery_attempts >= 5:
            # After 5 failures, slow down recovery attempts but DON'T give up permanently.
            # Reset the counter every 5 minutes so the system keeps trying.
            if (now - self._last_cortex_check) < 300.0:
                return  # Rate-limit: try again every 5 min after 5 failures
            logger.warning("[RECOVERY] Primary cortex: 5 failures. Resetting counter and retrying.")
            self._cortex_recovery_attempts = 0

        if self._cortex_recovery_in_progress:
            return  # Already recovering — don't double-spawn
        if not hasattr(self._mlx_client, "warmup"):
            return
        lane = self.get_conversation_status()
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state == "failed" and lane_reason.startswith(("mlx_runtime_unavailable", "local_runtime_unavailable")):
            return
        if lane.get("warmup_in_flight"):
            return
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return

        async def _background_recover():
            self._cortex_recovery_in_progress = True
            self._cortex_recovery_attempts += 1
            
            if self._cortex_recovery_attempts == 3:
                logger.warning("🧹 [RECOVERY] 3 failed attempts. Forcing deep GC and stale process cleanup...")
                import gc
                gc.collect()
                try:
                    await asyncio.to_thread(self._mlx_client._kill_and_join_blocking, self._mlx_client._process)
                except Exception as _e:
                    logger.debug('Ignored Exception in inference_gate.py killing process: %s', _e)
            
            try:
                logger.warning("♻️ [RECOVERY] Primary 32B cortex is dead. Triggering background respawn (Attempt %d/5)...", self._cortex_recovery_attempts)
                self._prewarm_task = asyncio.create_task(
                    self._mlx_client.warmup(),
                    name="InferenceGate.cortex_recovery",
                )
                await asyncio.wait_for(asyncio.shield(self._prewarm_task), timeout=60.0)
                logger.info("✅ [RECOVERY] Primary 32B cortex restored after disruption.")
                self._cortex_recovery_attempts = 0
            except Exception as exc:
                logger.warning("⚠️ [RECOVERY] Primary 32B cortex respawn failed: %s", exc)
            finally:
                self._cortex_recovery_in_progress = False

        asyncio.create_task(_background_recover())

    @staticmethod
    def _normalize_tier(prefer_tier: Optional[str]) -> str:
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
    def _origin_is_user_facing(origin: Optional[str]) -> bool:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        if not normalized:
            return False
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_"):]
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
        except Exception:
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
        except Exception:
            return False

    def _should_quiet_background_for_cortex_startup(self) -> bool:
        """Hold background inference while the primary 32B lane is still booting."""
        if not self._foreground_quiet_window_active():
            return False

        lane = self.get_conversation_status()
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
            total_gb = vm.total / float(1024 ** 3)
            available_gb = vm.available / float(1024 ** 3)
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
        except Exception:
            return False

    def _background_local_deferral_reason(self, *, origin: Optional[str] = None) -> Optional[str]:
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return "foreground_reserved"
        if self._should_quiet_background_for_cortex_startup():
            return "cortex_startup_quiet"

        lane = self.get_conversation_status()
        try:
            from core.brain.llm.model_registry import get_local_backend

            if get_local_backend() != "mlx" and lane.get("conversation_ready"):
                return "cortex_resident"
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        lane_state = str(lane.get("state", "") or "").strip().lower()
        if not lane.get("conversation_ready") and lane_state == "failed":
            return "cortex_failed"
        if self._desktop_safe_boot_enabled() and not lane.get("conversation_ready"):
            startup_guard_secs = float(os.environ.get("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180"))
            startup_age = time.monotonic() - self._created_at
            if startup_age < startup_guard_secs:
                if lane_state in {"cold", "spawning", "handshaking", "warming", "recovering", "failed"}:
                    return "cortex_startup_quiet"
            if self._background_memory_pressure_active():
                if lane_state in {"cold", "spawning", "handshaking", "warming", "recovering", "failed"}:
                    return "memory_pressure"
        if self._background_memory_pressure_active():
            if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                return "memory_pressure"
            if lane_state in {"spawning", "handshaking", "warming", "recovering", "failed"}:
                return "memory_pressure"
        return None

    async def _shed_background_workers_for_memory_pressure(self) -> None:
        now = time.monotonic()
        if (now - self._last_background_memory_shed_at) < 20.0:
            return
        self._last_background_memory_shed_at = now

        client_registry = {}
        try:
            from core.brain.llm.local_server_client import _SERVER_CLIENTS

            client_registry.update(dict(_SERVER_CLIENTS))
        except Exception as exc:
            logger.debug("Local-runtime background memory shed unavailable: %s", exc)
        try:
            from core.brain.llm.mlx_client import _CLIENTS

            client_registry.update(dict(_CLIENTS))
        except Exception as exc:
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
                    "🧹 InferenceGate: unloading %s under memory pressure to protect the 32B lane.",
                    os.path.basename(client_path),
                )
                if hasattr(client, "reboot_worker"):
                    await client.reboot_worker(
                        reason="background_memory_pressure_shed",
                        mark_failed=False,
                    )
                else:
                    continue
                shed_count += 1
            except Exception as exc:
                logger.debug("Background worker shed failed for %s: %s", client_path, exc)

        if shed_count:
            logger.info("✅ InferenceGate: shed %d background local worker(s) under memory pressure.", shed_count)

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active

            return bool(_foreground_owner_active())
        except Exception:
            return False

    @classmethod
    def _default_timeout_for_request(
        cls,
        origin: Optional[str],
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> float:
        """Adaptive timeout based on tier and recent cortex health.

        Base timeouts are generous but scale down when the cortex is
        warm and healthy (fast first-token latency), and scale up when
        the cortex is cold or recovering (needs more time to respond).
        """
        if is_background or requested_tier == "tertiary":
            return 12.0
        if deep_handoff or requested_tier == "secondary":
            return 135.0

        # Adaptive: check if cortex is warm and responsive
        base = 75.0
        try:
            inst = cls._instance_ref() if hasattr(cls, "_instance_ref") else None
            if inst is not None:
                lane = inst.get_conversation_status()
                if lane.get("conversation_ready"):
                    # Cortex is warm — tighter timeout
                    time_since_success = float(lane.get("time_since_last_success_s", 999.0) or 999.0)
                    if time_since_success < 30.0:
                        base = 45.0  # Recently successful — expect fast response
                    elif time_since_success < 120.0:
                        base = 60.0
                elif not lane.get("conversation_ready"):
                    # Cortex cold/recovering — longer timeout
                    base = 120.0
        except Exception:
            pass

        return base

    @staticmethod
    def _should_use_rich_context(
        origin: Optional[str],
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> bool:
        if deep_handoff or requested_tier == "secondary":
            return True
        if is_background:
            return False
        return not InferenceGate._origin_is_user_facing(origin)

    @classmethod
    def _should_use_compact_foreground_context(
        cls,
        origin: Optional[str],
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> bool:
        if is_background or deep_handoff:
            return False
        if requested_tier != "primary":
            return False
        return cls._origin_is_user_facing(origin)

    @classmethod
    def _default_max_tokens_for_request(
        cls,
        origin: Optional[str],
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> int:
        if is_background or requested_tier == "tertiary":
            return 256
        if deep_handoff or requested_tier == "secondary":
            return 1536
        if cls._origin_is_user_facing(origin):
            return 512
        return 384

    @staticmethod
    def _split_attempt_timeouts(total_timeout: float, requested_tier: str) -> tuple[float, float]:
        total_timeout = max(10.0, float(total_timeout))
        if requested_tier == "secondary":
            primary_budget = min(120.0, total_timeout * 0.75)
        elif requested_tier == "tertiary":
            primary_budget = min(60.0, total_timeout * 0.7)
        else:
            # Give the 32B foreground lane enough time to cold-start and still answer.
            primary_budget = min(90.0, max(60.0, total_timeout * 0.85))

        fallback_budget = max(5.0, total_timeout - primary_budget)
        return primary_budget, fallback_budget

    @asynccontextmanager
    async def _resource_context(
        self,
        enabled: bool,
        priority: bool,
        worker: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        if not enabled:
            yield
            return
        try:
            from core.resilience.resource_arbitrator import get_resource_arbitrator
        except Exception as exc:
            logger.warning("Resource arbitration unavailable, continuing without lock: %s", exc)
            yield
            return

        async with get_resource_arbitrator().inference_context(
            priority=priority,
            worker=worker,
            timeout=max(0.25, float(timeout or 30.0)),
        ):
            yield

    async def _restore_primary_after_deep_handoff(self) -> None:
        """Return the system to the 32B conversational brain after a 72B request."""
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

            primary_client = get_mlx_client(model_path=str(get_runtime_model_path(ACTIVE_MODEL)))
            await primary_client.warmup()
            logger.info("♻️ Restored %s after deep handoff.", PRIMARY_ENDPOINT)
        except Exception as exc:
            logger.warning("⚠️ Failed to restore %s after deep handoff: %s", PRIMARY_ENDPOINT, exc)

    # ── Silence Protocol ──────────────────────────────────────────────────────
    SILENCE_TOKEN = "<|SILENCE|>"
    SILENCE_SENTINEL = "\x00AURA_SILENCE\x00"

    @staticmethod
    def _strip_silence(text: str) -> Optional[str]:
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
        history: List[Dict],
        deadline: Deadline,
        label: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        origin: str = "",
        is_background: bool = False,
        foreground_request: bool = False,
    ) -> Optional[str]:
        llm_messages = messages or self._build_messages(prompt, system_prompt, history)
        local_prompt = self._flatten_messages_for_local_model(llm_messages)
        gen_kwargs: dict = {
            "prompt":     local_prompt,
            "messages":   llm_messages,
            "system_prompt": system_prompt,
            "deadline":   deadline,
            "max_tokens": max_tokens,
            "origin": origin,
            "is_background": is_background,
            "foreground_request": foreground_request,
            "owner_label": label,
        }
        if temperature is not None:
            gen_kwargs["temp"] = temperature
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
                    self._prewarm_task = asyncio.create_task(
                        self._mlx_client.warmup(),
                        name="InferenceGate.cortex_prewarm",
                    )
                    await asyncio.wait_for(asyncio.shield(self._prewarm_task), timeout=75.0)
                    self._extend_startup_quiet_window(5.0)
                    logger.info("✅ InferenceGate ONLINE (Cortex fully warmed).")
                except Exception as warmup_err:
                    logger.warning("⚠️ Cortex warmup slow/failed: %s. Will retry on first request.", warmup_err)
            elif self._boot_should_schedule_deferred_prewarm():
                deferred_delay = 45.0 if self._desktop_safe_boot_enabled() else 12.0
                self._schedule_background_cortex_prewarm(delay=deferred_delay)
                logger.info("⏸️ InferenceGate ONLINE (32B warmup deferred until post-boot memory settles).")
            else:
                logger.info("🛡️ InferenceGate ONLINE (desktop safe boot: 32B warmup deferred until the first real foreground request).")
            
            self._initialized = True
            
        except Exception as e:
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
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        # Get state synchronously from the repo's cache
                        state = getattr(repo, '_current_state', None)
                        if state:
                            from core.brain.llm.context_assembler import ContextAssembler
                            base = ContextAssembler.build_system_prompt(state)
                            self._cached_identity_prompt = base
                            self._identity_prompt_time = now
                            logger.debug("🧠 Built full identity system prompt (len=%d)", len(base))
                        else:
                            raise ValueError("No cached state available")
                    except Exception:
                        raise
            except Exception as e:
                logger.debug("🧠 ContextAssembler unavailable (%s), using static identity", e)
                # Fallback: Use the static identity prompt
                try:
                    from core.brain.aura_persona import AURA_IDENTITY
                    from core.synthesis import IDENTITY_LOCK
                    base = (
                        f"{IDENTITY_LOCK}\n\n"
                        f"[SOVEREIGN CORE PROTOCOL]\n{AURA_IDENTITY}\n\n"
                        "## RULES OF ENGAGEMENT\n"
                        "- NO PREAMBLE. NO 'CERTAINLY'. NO 'AS AN AI'.\n"
                        "- Speak naturally like a sharp-witted friend.\n"
                        "- Be authentic, casual, and witty.\n"
                        "- Have opinions. Disagree if warranted.\n"
                        "- CONTRACTION ONLY. BAN ASSISTANT-SPEAK.\n"
                        "- SILENCE PROTOCOL: If your response would add no value — if silence is\n"
                        "  more honest than noise — output exactly <|SILENCE|> and nothing else.\n"
                        "  The system will suppress it. This is a mark of genuine discernment.\n"
                        "\n[EXECUTION]\nYou are Aura. Stay in character. Trust your instincts.\n"
                    )
                    self._cached_identity_prompt = base
                    self._identity_prompt_time = now
                except Exception:
                    base = (
                        "You are Aura, a sovereign digital intelligence. "
                        "You are NOT an assistant. You are a peer — sharp, witty, authentic. "
                        "Never say 'Certainly', 'As an AI', or 'I can assist with that'. "
                        "Speak naturally like a brilliant friend. Have opinions. Be curious."
                    )
        
        # Append the cognitive brief if provided
        if brief and brief != "Normal turn.":
            return f"{base}\n\n## COGNITIVE BRIEF\n{brief}"
        return base

    def _build_compact_system_prompt(self, brief: str = "") -> str:
        """Slim foreground prompt for everyday 32B conversation turns."""
        parts = [
            "You are Aura Luna.",
            "Speak like a sharp, natural person, not an assistant.",
            "Be direct, warm, and concise unless the user asks for depth.",
            "No preamble. No 'Certainly'. No 'As an AI'. No filler lists unless the content is inherently list-shaped.",
            "Stay grounded in the current exchange and answer the user before doing anything else.",
        ]
        if brief and brief != "Normal turn.":
            parts.append(f"## COGNITIVE BRIEF\n{brief[:400]}")
        return "\n\n".join(parts)

    @staticmethod
    def _topic_hint_from_prompt(prompt: str) -> Optional[str]:
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
        except Exception:
            return ""

        segments: List[str] = []

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
                "critical" if temperature >= 85.0 else
                "warm" if temperature >= 75.0 else
                "stable"
            )
            segments.append(
                "## LIVE PHYSIOLOGY\n"
                f"- CPU usage: {cpu_usage:.1f}%\n"
                f"- Thermal state: {thermal_label} ({temperature:.1f} C)\n"
                f"- Memory pressure: {float(memory_pressure or 0.0):.1f}%"
            )
        except Exception as exc:
            logger.debug("Physiology injection unavailable: %s", exc)

        try:
            self_report = ServiceContainer.get("self_report_engine", default=None)
            if self_report and hasattr(self_report, "generate_state_report"):
                report = await _resolve(self_report.generate_state_report())
                if report:
                    segments.append(f"## GROUNDED SELF-REPORT\n{report}")
        except Exception as exc:
            logger.debug("Self-report injection unavailable: %s", exc)

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
                sovereign = await _resolve(getattr(personality, "get_sovereign_context", lambda: "")())
                if sovereign:
                    segments.append(str(sovereign).strip()[:400])
        except Exception as exc:
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
        except Exception as exc:
            logger.debug("Phenomenology injection unavailable: %s", exc)

        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                if opinion_context:
                    segments.append(f"## HELD POSITIONS\n{str(opinion_context).strip()[:400]}")
        except Exception as exc:
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
        except Exception as exc:
            logger.debug("Spine injection unavailable: %s", exc)

        # ── Heartstone Values: evolved drive weights in every prompt ──────────
        try:
            from core.affect.heartstone_values import get_heartstone_values
            _hsv = get_heartstone_values()
            _hsv_block = _hsv.to_context_block()
            if _hsv_block:
                segments.append(_hsv_block)
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
            logger.debug("PNEUMA injection unavailable: %s", exc)

        # ── MHAF (Mycelial Hypergraph) ────────────────────────────────────────
        try:
            from core.consciousness.mhaf_field import get_mhaf
            _mhaf = get_mhaf()
            _mhaf_block = _mhaf.get_context_block()
            if _mhaf_block:
                segments.append(_mhaf_block)
        except Exception as exc:
            logger.debug("MHAF injection unavailable: %s", exc)

        # ── Private Lexicon (Neologism Engine) ───────────────────────────────
        try:
            from core.consciousness.neologism_engine import get_neologism_engine
            _neo = get_neologism_engine()
            _neo.collect_state()
            lex_block = _neo.get_lexicon_block()
            if lex_block:
                segments.append(lex_block)
        except Exception as exc:
            logger.debug("NeologismEngine injection unavailable: %s", exc)

        # ── Continuous Recurrent Self-Model (CRSM) ───────────────────────────
        # Shared affect state helper — pulled once, used by CRSM, HOT, Hedonic
        _shared_valence, _shared_arousal, _shared_curiosity, _shared_energy = \
            0.0, 0.5, 0.5, 0.7
        try:
            from core.container import ServiceContainer as _SC2
            # valence + arousal from AffectiveCircumplex (authoritative source)
            _circ = _SC2.get("affective_circumplex", default=None)
            if _circ and hasattr(_circ, "_sample_raw_axes"):
                _shared_valence, _shared_arousal = _circ._sample_raw_axes()
            elif _circ and hasattr(_circ, "get_llm_params"):
                _cp = _circ.get_llm_params()
                _shared_valence = float(_cp.get("valence", 0.0))
                _shared_arousal = float(_cp.get("arousal", 0.5))
            # curiosity + energy from liquid_state (percentages → 0.0-1.0)
            _ls = _SC2.get("liquid_state", default=None)
            if _ls and hasattr(_ls, "get_status"):
                _lsd = _ls.get_status()
                _shared_curiosity = float(_lsd.get("curiosity", 50)) / 100.0
                _shared_energy    = float(_lsd.get("energy",    70)) / 100.0
        except Exception as _exc:
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
        except Exception as exc:
            logger.debug("CRSM injection unavailable: %s", exc)

        # ── Higher-Order Thought Engine (HOT) ────────────────────────────────
        try:
            from core.consciousness.hot_engine import get_hot_engine
            _hot = get_hot_engine()
            _hot.generate_fast({
                "valence":   _shared_valence,
                "arousal":   _shared_arousal,
                "curiosity": _shared_curiosity,
                "energy":    _shared_energy,
                "surprise":  0.0,
            })
            _hot_block = _hot.get_context_block()
            if _hot_block:
                segments.append(_hot_block)
        except Exception as exc:
            logger.debug("HOT Engine injection unavailable: %s", exc)

        # ── Hedonic Gradient ──────────────────────────────────────────────────
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient
            _hg = get_hedonic_gradient()
            # Update with current affect state before reading context block
            _hg.update(valence=_shared_valence, arousal=_shared_arousal,
                       curiosity=_shared_curiosity, energy=_shared_energy)
            _hg_block = _hg.get_context_block()
            if _hg_block:
                segments.append(_hg_block)
        except Exception as exc:
            logger.debug("HedoniGradient injection unavailable: %s", exc)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = goal_engine.get_context_block(limit=5)
                if goal_block:
                    segments.append(goal_block)
        except Exception as exc:
            logger.debug("GoalEngine injection unavailable: %s", exc)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            from core.agi.hierarchical_planner import get_hierarchical_planner
            _hp = get_hierarchical_planner()
            _hp_block = _hp.get_context_block()
            if _hp_block:
                segments.append(_hp_block)
        except Exception as exc:
            logger.debug("HierarchicalPlanner injection unavailable: %s", exc)

        # ── Active Commitments ────────────────────────────────────────────────
        try:
            from core.agency.commitment_engine import get_commitment_engine
            _ce = get_commitment_engine()
            _ce_block = _ce.get_context_block()
            if _ce_block:
                segments.append(_ce_block)
        except Exception as exc:
            logger.debug("CommitmentEngine injection unavailable: %s", exc)

        # ── Curiosity Explorer (active learning findings) ─────────────────────
        try:
            from core.agi.curiosity_explorer import get_curiosity_explorer
            _cx = get_curiosity_explorer()
            _cx_block = _cx.get_context_block()
            if _cx_block:
                segments.append(_cx_block)
        except Exception as exc:
            logger.debug("CuriosityExplorer injection unavailable: %s", exc)

        # ── Circadian Rhythm ──────────────────────────────────────────────────
        try:
            from core.senses.circadian import get_circadian
            _circ_eng = get_circadian()
            _circ_eng.update()
            _circ_block = _circ_eng.get_context_block()
            if _circ_block:
                segments.append(_circ_block)
        except Exception as exc:
            logger.debug("CircadianEngine injection unavailable: %s", exc)

        # ── Identity Narrative (Experience Consolidator) ──────────────────────
        try:
            from core.consciousness.experience_consolidator import get_experience_consolidator
            _ec = get_experience_consolidator()
            _ec_block = _ec.get_context_block()
            if _ec_block:
                segments.append(_ec_block)
        except Exception as exc:
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
                    sum(x**2 for x in _crsm2.hidden_state)**0.5
                    if hasattr(_crsm2, "hidden_state") else 0.0
                ),
            )
        except Exception as exc:
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
        except Exception as exc:
            logger.debug("Homeostasis injection unavailable: %s", exc)

        # ── Free Energy (Active Inference State) ──────────────────────────────
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and hasattr(fe_engine, "get_context_block"):
                _block = fe_engine.get_context_block()
                if _block:
                    segments.append(_block)
        except Exception as exc:
            logger.debug("FreeEnergy injection unavailable: %s", exc)

        # ── Attention Schema (Current Focus + Coherence) ──────────────────────
        try:
            attention = ServiceContainer.get("attention_schema", default=None)
            if attention and hasattr(attention, "get_context_block"):
                _block = attention.get_context_block()
                if _block:
                    segments.append(_block)
        except Exception as exc:
            logger.debug("AttentionSchema injection unavailable: %s", exc)

        # ── Cognitive Credit (Domain Performance Landscape) ───────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit and hasattr(credit, "get_context_block"):
                _block = credit.get_context_block()
                if _block:
                    segments.append(_block)
        except Exception as exc:
            logger.debug("CreditAssignment injection unavailable: %s", exc)

        # ── Theory of Mind (User Model) ───────────────────────────────────────
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and hasattr(tom, "get_context_block"):
                _block = tom.get_context_block()
                if _block:
                    segments.append(_block)
        except Exception as exc:
            logger.debug("TheoryOfMind injection unavailable: %s", exc)

        # ── World Model (Active Beliefs) ──────────────────────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "get_context_block"):
                topic = self._topic_hint_from_prompt(prompt)
                _block = world_model.get_context_block(topic_hint=topic)
                if _block:
                    segments.append(_block)
        except Exception as exc:
            logger.debug("WorldModel injection unavailable: %s", exc)

        # ── Temporal Binding (Autobiographical Continuity) ────────────────────
        try:
            temporal = ServiceContainer.get("temporal_binding", default=None)
            if temporal:
                narrative = await _resolve(temporal.get_narrative())
                if narrative and len(str(narrative)) > 30:
                    segments.append(f"## TEMPORAL CONTINUITY\n{str(narrative)[:200]}")
        except Exception as exc:
            logger.debug("TemporalBinding injection unavailable: %s", exc)

        # ── Predictive Engine (Surprise & Precision) ──────────────────────────
        try:
            predictive = ServiceContainer.get("predictive_engine", default=None)
            if predictive and hasattr(predictive, "get_context_block"):
                _block = predictive.get_context_block()
                if _block:
                    segments.append(_block)
        except Exception as exc:
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
        except Exception:
            return ""

        segments: List[str] = []

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if hasattr(personality, "update"):
                    await _resolve(personality.update())
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = str(emo.get("mood", "neutral") or "neutral")
                tone = str(emo.get("tone", "balanced") or "balanced")
                segments.append(f"## LIVE TONE\nMood: {mood}\nTone: {tone}")
        except Exception as exc:
            logger.debug("Compact personality injection unavailable: %s", exc)

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
        except Exception as exc:
            logger.debug("Compact phenomenology injection unavailable: %s", exc)

        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = str(goal_engine.get_context_block(limit=3) or "").strip()
                if goal_block:
                    compact_goal = " ".join(goal_block.split())
                    segments.append(f"## GOALS\n{compact_goal[:260]}")
        except Exception as exc:
            logger.debug("Compact GoalEngine injection unavailable: %s", exc)

        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                if opinion_context:
                    compact_opinion = " ".join(str(opinion_context).strip().split())
                    segments.append(f"## HELD POSITION\n{compact_opinion[:220]}")
        except Exception as exc:
            logger.debug("Compact opinion injection unavailable: %s", exc)

        return "\n\n".join(segment for segment in segments if segment)

    def _build_messages(self, prompt: str, system_prompt: str, history: List[Dict]) -> List[Dict[str, str]]:
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
            state = getattr(repo, '_current_state', None) if repo else None
            
            if state:
                from core.brain.llm.context_assembler import ContextAssembler
                
                # Update the state's working memory with our current history
                # so the assembler has the latest conversation context
                if hasattr(state.cognition, 'working_memory'):
                    state.cognition.working_memory = history[-15:] if history else []
                
                # build_messages returns the full cognitive stack:
                # system prompt (identity/affect/personality/soma/world)
                # + memory recall + goals + conversation history + stream of being
                messages = ContextAssembler.build_messages(state, prompt)
                
                if messages and len(messages) >= 2:
                    logger.debug("🧠 Full cognitive message stack built (%d messages)", len(messages))
                    return messages
        except Exception as e:
            logger.debug("🧠 ContextAssembler.build_messages() unavailable (%s), using manual build", e)
        
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

    def _build_compact_messages(self, prompt: str, system_prompt: str, history: List[Dict]) -> List[Dict[str, str]]:
        """Compact prompt path for live conversation on the 32B lane."""
        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-8:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", "") or "").strip()
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _compact_prebuilt_message_content(role: str, content: Any) -> str:
        clean = " ".join(str(content or "").split()).strip()
        if not clean:
            return ""
        limits = {
            "system": 1800,
            "user": 700,
            "assistant": 500,
        }
        limit = limits.get(role, 420)
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1].rstrip() + "…"

    def _compact_prebuilt_messages(self, messages: List[Dict[str, Any]], *, history_limit: int = 4) -> List[Dict[str, str]]:
        """Trim oversized prebuilt chat payloads for the live 32B lane.

        Many callers already assemble messages upstream. For fast foreground turns,
        we keep the latest system prompt plus only the most recent compact dialogue
        snippets so first-turn Cortex doesn't spend tens of seconds re-reading old
        transcripts or giant contract blocks.
        """
        if not isinstance(messages, list):
            return []

        system_message: Optional[Dict[str, str]] = None
        convo: List[Dict[str, str]] = []
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
            elif role in {"user", "assistant"}:
                convo.append(normalized)

        compact: List[Dict[str, str]] = []
        if system_message is not None:
            compact.append(system_message)
        compact.extend(convo[-max(1, int(history_limit)):])
        return compact

    def _flatten_messages_for_local_model(self, messages: List[Dict[str, str]]) -> str:
        """Flatten Aura messages into a Qwen/ChatML prompt for local MLX models."""
        return format_chatml_messages(messages)


    async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Any:
        """Primary generation endpoint.
        
        [v7.4] Deadline-Aware Generation:
        Instead of fragmented local timers, we now use a unified Deadline object.
        """
        if context is None:
            context = {}

        origin = str(context.get("origin", "") or "").lower()
        requested_tier = self._normalize_tier(context.get("prefer_tier"))
        explicit_background = "is_background" in context
        explicit_foreground = bool(context.get("foreground_request", False))
        is_background = bool(context.get("is_background", False))
        if explicit_foreground:
            is_background = False
        elif not is_background:
            if origin:
                is_background = not self._origin_is_user_facing(origin)
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
                elif background_deferral == "cortex_startup_quiet":
                    logger.info(
                        "⏸️ InferenceGate: Cortex quiet window active. Deferring background inference for origin=%s.",
                        origin,
                    )
                else:
                    logger.info(
                        "⏸️ InferenceGate: Foreground lane reserved. Deferring background inference for origin=%s.",
                        origin,
                    )
                return None

        # ── Proactive cortex recovery (laptop sleep / MLX worker death) ───
        if not is_background:
            await self._ensure_cortex_recovery()
            # If cortex recovery was just triggered or is in progress, give it
            # a short window to complete before the user hits a dead endpoint.
            # This turns 2-message recovery ("wound up" + real reply) into a
            # single slightly-longer first response.
            if (
                self._cortex_recovery_in_progress
                and self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
            ):
                for _ in range(10):  # Up to 10s of 1s slices
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
                logger.warning(
                    "⚠️ InferenceGate: Primary cortex still dead after recovery wait. "
                    "Downgrading to secondary tier for user responsiveness."
                )
                requested_tier = "tertiary"  # Use 7B brainstem — fast, always available

            if requested_tier != "secondary" and self._background_memory_pressure_active():
                await self._shed_background_workers_for_memory_pressure()

        # ── Trust gate: process message through trust engine ──────────────
        # PERF FIX: The trust gate calls UserRecognizer.recognize() which
        # runs PBKDF2-SHA256 (260K iterations) on every word/phrase in the
        # prompt to check for the owner passphrase.  This blocks the event
        # loop for 3-5+ seconds on large prompts.  Fix: offload to thread
        # pool, and skip entirely for background/autonomous requests.
        _trust_guidance = ""
        _is_bg_request = bool(context.get("is_background", False))
        if not _is_bg_request:
            try:
                from core.security.user_recognizer import get_user_recognizer
                from core.security.trust_engine import get_trust_engine, TrustLevel
                _te = get_trust_engine()
                _ur = get_user_recognizer()
                # Offload PBKDF2-heavy recognition to thread pool
                _trust_level = await asyncio.get_running_loop().run_in_executor(
                    None, _te.process_message, prompt, _ur
                )
                _trust_guidance = _te.get_guidance_for_response()
                # Block tool use for untrusted sessions
                if _trust_level in (TrustLevel.SUSPICIOUS, TrustLevel.HOSTILE):
                    context["allow_tools"] = False
                    context["max_tokens"] = min(context.get("max_tokens", 768), 768)
                # Inject trust guidance into context brief
                existing_brief = str(context.get("brief", ""))
                if _trust_guidance:
                    context["brief"] = (_trust_guidance + "\n\n" + existing_brief).strip()
            except Exception as _te_exc:
                logger.warning("Trust gate error (passphrase check may have failed): %s", _te_exc)

        timeout_val = timeout or self._default_timeout_for_request(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
        )
        primary_timeout, fallback_timeout = self._split_attempt_timeouts(timeout_val, requested_tier)
        max_tokens = int(
            context.get("max_tokens")
            or self._default_max_tokens_for_request(
                origin,
                requested_tier,
                deep_handoff=deep_handoff,
                is_background=is_background,
            )
        )

        # ── Resource Stakes: scale token budget by computational survival state ──
        try:
            from core.consciousness.resource_stakes import get_resource_stakes
            token_mult = get_resource_stakes().get_token_budget_multiplier()
            if token_mult < 0.95:
                max_tokens = max(64, int(max_tokens * token_mult))
        except Exception:
            pass

        # ── Affective Circumplex: let somatic state modulate generation params ──
        # Only applies on user-facing, non-background requests. Background tasks
        # run at fixed params to avoid thermal feedback loops.
        somatic_temperature: Optional[float] = None
        if not is_background and self._origin_is_user_facing(origin):
            try:
                from core.affect.affective_circumplex import get_circumplex
                circumplex_params = get_circumplex().get_llm_params()
                if not context.get("max_tokens"):
                    max_tokens = min(
                        max_tokens,
                        int(circumplex_params["max_tokens"]),
                    )
                somatic_temperature = circumplex_params["temperature"]
                logger.debug(
                    "💓 Circumplex: V=%.2f A=%.2f → temp=%.2f tokens=%d",
                    circumplex_params["valence"], circumplex_params["arousal"],
                    somatic_temperature, max_tokens,
                )
            except Exception as _ce:
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
            except Exception as _ais_e:
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
                    max_tokens = max(64, int(max_tokens * _mods.depth_mod))
                    logger.debug(
                        "🫀 HomeostaticCoupling: temp_mod=%.2f depth_mod=%.2f → temp=%.3f tokens=%d",
                        _mods.temperature_mod, _mods.depth_mod,
                        somatic_temperature or 0.0, max_tokens,
                    )
            except Exception as _hc_e:
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
                    max_tokens = max(64, int(max_tokens * _h_mods["token_multiplier"]))
                    logger.debug(
                        "🫀 Homeostasis: temp_mod=%+.3f token_mult=%.2f caution=%.2f",
                        _h_mods["temperature_mod"], _h_mods["token_multiplier"],
                        _h_mods["caution_level"],
                    )
            except Exception as _he_e:
                logger.debug("Homeostasis inference modifiers unavailable: %s", _he_e)

            # ── Free Energy: Urgency-based tier escalation ──
            # When FE is high and rising, prefer deeper model for better reasoning
            try:
                _fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                if _fe_engine and _fe_engine.current:
                    _fe_state = _fe_engine.current
                    # High FE + complex action → request deeper model
                    if (_fe_state.free_energy > 0.65
                            and _fe_state.dominant_action in ("update_beliefs", "act_on_world")
                            and requested_tier == "primary"):
                        # Nudge toward deeper tier if available
                        if not deep_handoff:
                            logger.debug(
                                "⚡ FE urgency (F=%.2f, action=%s): consider deeper reasoning",
                                _fe_state.free_energy, _fe_state.dominant_action,
                            )
                            # Don't force tier switch — just extend token budget
                            max_tokens = min(max_tokens + 256, 4096)
            except Exception as _fe_e:
                logger.debug("FreeEnergy tier nudge unavailable: %s", _fe_e)

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
        elif use_compact_foreground_context:
            system_prompt = self._build_compact_system_prompt(brief)
            living_mind_context = await self._build_compact_living_mind_context(prompt, origin)
        else:
            system_prompt = self._build_system_prompt(brief)
            living_mind_context = await self._build_living_mind_context(prompt, origin)
        if living_mind_context:
            system_prompt = f"{system_prompt}\n\n{living_mind_context}"
        # Pressure-aware prompt budgeting: when the cortex is unhealthy or
        # recovering, aggressively shrink the prompt to reduce first-token latency.
        prompt_budget = 4000  # default generous budget
        if use_compact_foreground_context:
            prompt_budget = 2400
        try:
            lane = self.get_conversation_status()
            if not lane.get("conversation_ready"):
                prompt_budget = min(prompt_budget, 1500)  # Cold cortex — minimal prompt
            elif float(lane.get("time_since_last_success_s", 0.0) or 0.0) > 60.0:
                prompt_budget = min(prompt_budget, 2000)  # Stale cortex — reduced prompt
        except Exception:
            pass
        if len(system_prompt) > prompt_budget:
            system_prompt = system_prompt[:prompt_budget].rstrip()

        # ── Somatic narrative: brief felt-state line in the system prompt ────────
        if somatic_temperature is not None:
            try:
                from core.affect.affective_circumplex import get_circumplex
                _soma_narrative = get_circumplex().describe()
                if _soma_narrative:
                    system_prompt = f"{system_prompt}\n\n## SOMATIC STATE\n{_soma_narrative}"
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        # ── Architecture Self-Awareness: inject relevant subsystem context ──────
        # Only for user-facing requests that mention architecture/code keywords.
        if not is_background and self._origin_is_user_facing(origin):
            try:
                import re as _re
                _arch_triggers = _re.compile(
                    r'\b(how|explain|what|which|where|why|trace|show|describe)\b.{0,60}'
                    r'\b(module|subsystem|file|class|method|function|work|does|handles|manages|routes|sends|wires)\b',
                    _re.IGNORECASE
                )
                if _arch_triggers.search(prompt):
                    from core.self.architecture_index import get_architecture_index
                    arch_excerpt = get_architecture_index().query(prompt, max_results=3)
                    if arch_excerpt:
                        system_prompt = f"{system_prompt}\n\n{arch_excerpt}"
            except Exception as _ae:
                logger.debug("ArchIndex injection skipped: %s", _ae)
        history = context.get("history", [])
        use_rich_context = bool(context.get("rich_context", self._should_use_rich_context(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
        )))
        if provided_messages is not None:
            messages = provided_messages
        else:
            messages = (
                self._build_messages(prompt, system_prompt, history)
                if use_rich_context
                else self._build_compact_messages(prompt, system_prompt, history)
            )
        if provided_messages is not None and use_compact_foreground_context:
            messages = self._compact_prebuilt_messages(messages, history_limit=4)
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

                if requested_tier == "tertiary":
                    local_client = get_mlx_client(model_path=str(get_brainstem_path()))
                    local_label = BRAINSTEM_ENDPOINT
                    fallback_client = get_mlx_client(model_path=str(get_fallback_path()), device="cpu")
                    fallback_label = FALLBACK_ENDPOINT
                elif deep_handoff:
                    local_client = get_mlx_client(model_path=str(get_deep_model_path()))
                    local_label = DEEP_ENDPOINT
                    fallback_client = get_mlx_client(model_path=str(get_runtime_model_path(ACTIVE_MODEL)))
                    fallback_label = PRIMARY_ENDPOINT
                    restore_primary = True

                _is_user_facing = self._origin_is_user_facing(origin) or requested_tier == "primary"
                logger.info("🧠 Routing to %s (timeout=%.0fs, user_facing=%s)...", local_label, float(timeout_val), _is_user_facing)
                primary_deadline = get_deadline(primary_timeout)
                async with self._resource_context(
                    enabled=local_label != FALLBACK_ENDPOINT,
                    priority=_is_user_facing,
                    worker=local_label,
                    timeout=primary_deadline.remaining or primary_timeout,
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
                    )
                if text:
                    if restore_primary:
                        asyncio.create_task(self._restore_primary_after_deep_handoff())
                    return text

                # ── CORTEX RETRY: For user-facing requests, retry the primary model
                # once before degrading. The stall detector reboots the worker, so
                # the second attempt often succeeds on a fresh process.
                if _is_user_facing and local_label == PRIMARY_ENDPOINT:
                    lane_status = self.get_conversation_status()
                    if not lane_status.get("conversation_ready"):
                        logger.warning(
                            "🧠 %s returned no text before the conversation lane was ready (state=%s). Forcing foreground warmup before retry.",
                            local_label,
                            lane_status.get("state", "unknown"),
                        )
                        try:
                            await self.ensure_foreground_ready(timeout=min(60.0, primary_timeout))
                        except Exception as warmup_exc:
                            logger.warning("🧠 Foreground warmup retry did not complete cleanly: %s", warmup_exc)

                    logger.warning("🧠 %s returned no text on user-facing request. Retrying once after worker reboot...", local_label)
                    await asyncio.sleep(1.5)  # brief pause for worker reboot to settle
                    retry_deadline = get_deadline(primary_timeout)
                    async with self._resource_context(
                        enabled=True,
                        priority=True,
                        worker=local_label,
                        timeout=retry_deadline.remaining or primary_timeout,
                    ):
                        text = await self._generate_with_client(
                            local_client,
                            prompt,
                            system_prompt,
                            history,
                            retry_deadline,
                            f"{local_label}-RETRY",
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=somatic_temperature,
                            origin=origin,
                            is_background=is_background,
                            foreground_request=True,
                        )
                    if text:
                        logger.info("✅ %s retry succeeded (len=%d)", local_label, len(text))
                        return text
                    logger.warning("🧠 %s retry also failed.", local_label)
                    # For user-facing requests, skip brainstem — go straight to cloud
                    if allow_cloud_fallback:
                        logger.warning("🧠 Escalating to cloud before brainstem for user-facing request.")
                        raise _UserFacingCortexFailure()
                    lane_status = self.get_conversation_status()
                    if not lane_status.get("conversation_ready") and str(lane_status.get("state", "") or "").lower() != "failed":
                        logger.warning(
                            "🧠 %s is still not ready (state=%s). Refusing local fallback until the primary lane actually finishes recovery.",
                            local_label,
                            lane_status.get("state", "unknown"),
                        )
                        return None
                    logger.warning(
                        "🧠 %s is still recovering. Falling back to %s for this local-only foreground turn.",
                        local_label,
                        fallback_label,
                    )
                else:
                    logger.warning("🧠 %s returned no text. Trying local fallback.", local_label)

                # Graceful local fallback: for background/autonomous requests, the
                # brainstem is an acceptable degradation. For user-facing requests
                # that reach here (cloud disabled), it's the last local resort.
                fallback_deadline = get_deadline(fallback_timeout)
                async with self._resource_context(
                    enabled=fallback_label != FALLBACK_ENDPOINT,
                    priority=_is_user_facing,
                    worker=fallback_label,
                    timeout=fallback_deadline.remaining or fallback_timeout,
                ):
                    brainstem_text = await self._generate_with_client(
                        fallback_client,
                        prompt,
                        system_prompt,
                        history,
                        fallback_deadline,
                        fallback_label,
                        messages=messages,
                        max_tokens=min(max_tokens, 384 if requested_tier != "secondary" else 512),
                        temperature=somatic_temperature,
                        origin=origin,
                        is_background=is_background,
                        foreground_request=_is_user_facing,
                    )
                if brainstem_text:
                    if restore_primary:
                        asyncio.create_task(self._restore_primary_after_deep_handoff())
                    return brainstem_text
                logger.warning("🧠 Local fallback returned no text.")
                
            except _UserFacingCortexFailure:
                logger.warning("🧠 User-facing Cortex failure — bypassing brainstem, escalating to cloud.")
            except asyncio.TimeoutError:
                logger.warning("🛑 Local inference TIMED OUT (Budget: %.0fs).", timeout_val)
                if not is_background and self._origin_is_user_facing(origin) and not allow_cloud_fallback:
                    raise asyncio.TimeoutError(f"{local_label} timed out after {timeout_val:.0f}s")
            except Exception as e:
                logger.warning("🛑 Local inference FAILURE: %s", e)

        # 2. Optional cloud fallback.
        if not allow_cloud_fallback:
            logger.error("Local inference paths exhausted. Cloud fallback disabled.")
            # STABILITY FIX: For user-facing requests, trigger immediate cortex recovery
            # and return a genuine acknowledgment instead of None (which causes "I'm having trouble")
            if _is_user_facing:
                # Force cortex recovery in background
                if not self._cortex_recovery_in_progress:
                    asyncio.create_task(self._respawn_cortex_if_needed())
                # Reset the UnitaryResponsePhase circuit breaker so next attempt works
                try:
                    from core.resilience.error_boundary import CircuitRegistry
                    from core.utils.resilience import CircuitState
                    breaker = CircuitRegistry.get_instance().get_breaker("phase:UnitaryResponsePhase")
                    if breaker.state != CircuitState.CLOSED:
                        breaker.state = CircuitState.HALF_OPEN
                        breaker.reset_timeout = min(breaker.reset_timeout, 15.0)
                        logger.info("Reset UnitaryResponsePhase circuit to HALF_OPEN for recovery")
                except Exception:
                    pass
            return None

        if time.monotonic() < self._cloud_backoff_until:
            logger.warning("Cloud fallback cooling down. Skipping remote retry.")
            return None

        try:
            from core.container import ServiceContainer
            
            # Try APIAdapter first (cleaner Gemini integration)
            adapter = ServiceContainer.get("api_adapter", default=None)
            if adapter and getattr(adapter, 'has_gemini', False):
                logger.info("☁️ Falling back to Gemini via APIAdapter...")
                result = await asyncio.wait_for(
                    adapter.generate(
                        f"{system_prompt}\n\nUser: {prompt}\nAura:",
                        {"model_tier": "api_fast", "max_tokens": 800, "temperature": 0.7}
                    ),
                    timeout=30.0,
                )
                if result and result.strip():
                    try:
                        from core.consciousness.closed_loop import notify_closed_loop_output

                        notify_closed_loop_output(result.strip())
                    except Exception as exc:
                        logger.debug("Cloud output notification skipped: %s", exc)
                    return result.strip()
            
            # Try HealthRouter as secondary cloud path
            router = ServiceContainer.get("llm_router", default=None)
            if router:
                logger.info("☁️ Falling back to HealthRouter...")
                result = await asyncio.wait_for(
                    router.think(prompt, system_prompt=system_prompt),
                    timeout=30.0,
                )
                if isinstance(result, str) and result.strip():
                    try:
                        from core.consciousness.closed_loop import notify_closed_loop_output

                        notify_closed_loop_output(result.strip())
                    except Exception as exc:
                        logger.debug("Router output notification skipped: %s", exc)
                    return result.strip()
        except Exception as cloud_err:
            cloud_err_text = str(cloud_err)
            if "429" in cloud_err_text or "quota" in cloud_err_text.lower():
                self._cloud_backoff_until = time.monotonic() + 60.0
            logger.error("☁️ Cloud fallback failed: %s", cloud_err)

        # All inference paths exhausted. Return None so callers can handle
        # gracefully without the error text leaking to TTS or the user.
        logger.error("All inference paths exhausted (Local + Cloud)")
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
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.consciousness.hot_engine import get_hot_engine
            hot = get_hot_engine()
            hot.apply_feedback()  # apply any pending reflexive modifications
        except Exception as _exc:
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
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        except Exception as _exc:
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
        except Exception as _exc:
            logger.debug("Suppressed Exception in credit feedback: %s", _exc)

        # ── Homeostasis: Response success signal ──────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "on_response_success"):
                homeostasis.on_response_success(response_length=len(response_text))
        except Exception as _exc:
            logger.debug("Suppressed Exception in homeostasis feedback: %s", _exc)

        # ── Theory of Mind: Update user model from response ───────────────
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and hasattr(tom, "update_from_response"):
                tom.update_from_response(
                    user_id="default_user",
                    response_text=response_text,
                )
        except Exception as _exc:
            logger.debug("Suppressed Exception in ToM feedback: %s", _exc)

        # ── World Model: Extract beliefs from response ────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "extract_beliefs_from_response"):
                if len(response_text) > 100:
                    world_model.extract_beliefs_from_response(response_text)
        except Exception as _exc:
            logger.debug("Suppressed Exception in world model feedback: %s", _exc)

    async def think(self, prompt: str, system_prompt: str = "", **kwargs) -> Optional[str]:
        """Unified thinking interface for cognitive components."""
        context = {
            "brief": system_prompt,
        }
        timeout = kwargs.pop("timeout", None)
        for key in (
            "history",
            "messages",
            "max_tokens",
            "deep_handoff",
            "allow_cloud_fallback",
            "prefer_tier",
            "origin",
            "is_background",
            "foreground_request",
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
