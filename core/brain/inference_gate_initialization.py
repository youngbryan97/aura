"""How the gate comes up: once, under its lock, with a receipt of what came up.

Lifted whole out of `inference_gate`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any


class _GateInitializationMixin:
    """Lifted whole out of InferenceGate; see inference_gate.py."""

    async def initialize(self) -> None:
        """Boot-time initialization — prepares the managed local client.

        Singleflight: concurrent initialize calls must not race client
        replacement or spawn duplicate prewarm/maintenance tasks.
        """
        from .inference_gate import (
            LockRank,
            checked_async_lock,
            logger,
        )

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
        from .inference_gate import (
            copy,
        )

        return copy.deepcopy(getattr(self, "_initialization_receipt", {}))

    async def _initialize_locked(self) -> None:
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _primary_lane_label,
            _record_inference_degradation,
            get_task_tracker,
            logger,
        )

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
                        logger.info("InferenceGate ONLINE (Cortex fully warmed).")
                    else:
                        receipt["reason"] = str(incomplete_reason or "warmup_incomplete")
                        logger.warning(
                            "InferenceGate ONLINE with Cortex warmup incomplete "
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
                        "Cortex warmup slow/failed: %s. Will retry on first request.", warmup_err
                    )
            elif self._boot_should_schedule_deferred_prewarm():
                deferred_delay = 45.0 if self._desktop_safe_boot_enabled() else 12.0
                self._schedule_background_cortex_prewarm(delay=deferred_delay)
                receipt["mode"] = "deferred_prewarm"
                receipt["reason"] = "warmup_deferred_until_post_boot"
                logger.info(
                    "InferenceGate ONLINE (%s warmup deferred until post-boot memory settles).",
                    _primary_lane_label(),
                )
            else:
                receipt["mode"] = "ram_admitted"
                receipt["reason"] = "warmup_requires_ram_admission"
                logger.info(
                    "InferenceGate ONLINE (desktop resource guard: %s warmup is RAM-admitted).",
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
                "InferenceGate init failed: %s. Gate remains unhealthy until explicit recovery succeeds.",
                e,
            )

