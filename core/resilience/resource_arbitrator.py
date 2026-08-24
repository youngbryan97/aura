"""Compatibility facade for canonical runtime resource admission.

Inference and evolution callers historically used ``ResourceArbitrator``.
The actual policy owner is now ``RuntimeControlPlane.admission``; this facade
preserves the public context-manager API and the cross-process evolution lock.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from core.runtime.control_plane import (
    AdmissionPriority,
    AdmissionRequest,
    ResourceAdmissionController,
    WorkClass,
    get_runtime_control_plane,
)
from core.runtime.errors import record_degradation
from core.runtime.model_runtime_assignment import ModelRuntimeAssignment
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ResourceArbitrator")


class ResourceArbitrator:
    """Route legacy inference/evolution tokens through canonical admission."""

    def __init__(
        self,
        *,
        admission: ResourceAdmissionController | None = None,
        lock_path: str | Path | None = None,
    ) -> None:
        self._admission = admission or get_runtime_control_plane().admission
        self._lock_path = str(lock_path or (state_root() / "run" / "vram.lock"))
        self._state_lock = threading.RLock()
        self._inference_leases: dict[str, list[str]] = {}
        self._evolution_lease_id = ""
        self._evolution_request_id = ""
        self._mp_fd: int | None = None

    @property
    def admission(self) -> ResourceAdmissionController:
        """Expose the canonical owner for identity and health verification."""

        return self._admission

    @staticmethod
    def _worker_lane(worker: str | ModelRuntimeAssignment | None) -> str:
        if isinstance(worker, ModelRuntimeAssignment):
            return worker.lane
        raw = str(worker or "default").strip() or "default"
        # These are protocol labels emitted by InferenceGate, not model
        # locators. Only exact labels normalize here: paths and suggestive
        # names cannot acquire serving authority after a model has loaded.
        exact_labels = {
            "cortex": "cortex",
            "mlx-cortex": "cortex",
            "solver": "solver",
            "mlx-solver": "solver",
            "brainstem": "brainstem",
            "mlx-brainstem": "brainstem",
            "reflex": "reflex",
            "mlx-reflex": "reflex",
        }
        normalized = raw.casefold().replace("_", "-")
        if normalized in exact_labels:
            return exact_labels[normalized]
        return raw

    def _get_mp_lock(self) -> int | None:
        try:
            target = Path(self._lock_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            return os.open(str(target), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            record_degradation(
                "resource_arbitrator",
                exc,
                severity="warning",
                action="evolution admission failed because cross-process VRAM lock was unavailable",
                receipt_required=False,
            )
            logger.error("Failed to open VRAM lock file: %s", exc)
            return None

    async def acquire_inference(  # noqa: ASYNC109 - compatibility API names this budget timeout
        self,
        priority: bool = False,
        timeout: float | None = None,  # noqa: ASYNC109 - compatibility API
        worker: str | ModelRuntimeAssignment | None = None,
    ) -> bool:
        lane = self._worker_lane(worker)
        effective_timeout = (
            max(0.0, float(timeout))
            if timeout is not None
            else (90.0 if priority else 30.0)
        )
        request = AdmissionRequest(
            owner=f"resource_arbitrator.inference:{lane}",
            work_class=WorkClass.INFERENCE,
            lane=lane,
            priority=(
                AdmissionPriority.FOREGROUND
                if priority
                else AdmissionPriority.INTERACTIVE
            ),
            timeout_s=effective_timeout,
            lease_ttl_s=max(30.0, effective_timeout + 30.0),
            metadata={"worker": lane, "legacy_facade": True},
        )
        decision = await self._admission.acquire(request)
        if not decision.admitted:
            logger.warning(
                "Inference admission denied worker=%s priority=%s outcome=%s reason=%s receipt=%s",
                lane,
                priority,
                decision.outcome.value,
                decision.reason,
                decision.receipt_id or "none",
            )
            return False
        with self._state_lock:
            self._inference_leases.setdefault(lane, []).append(decision.lease_id)
        logger.debug(
            "Inference lease acquired worker=%s priority=%s lease=%s",
            lane,
            priority,
            decision.lease_id,
        )
        return True

    async def release_inference(
        self,
        worker: str | ModelRuntimeAssignment | None = None,
    ) -> None:
        lane = self._worker_lane(worker)
        with self._state_lock:
            leases = self._inference_leases.get(lane, [])
            lease_id = leases.pop() if leases else ""
            if not leases:
                self._inference_leases.pop(lane, None)
        if not lease_id:
            logger.warning("Attempted to release unowned inference lease worker=%s", lane)
            return
        try:
            await self._admission.release(lease_id, reason="inference_finished")
        except KeyError:
            logger.warning("Inference lease expired before release worker=%s lease=%s", lane, lease_id)

    async def acquire_evolution(  # noqa: ASYNC109 - compatibility API
        self,
        timeout: float = 300.0,  # noqa: ASYNC109 - compatibility API
    ) -> bool:
        effective_timeout = max(0.0, float(timeout))
        request = AdmissionRequest(
            owner="resource_arbitrator.evolution",
            work_class=WorkClass.EVOLUTION,
            lane="unified_memory",
            priority=AdmissionPriority.BACKGROUND,
            timeout_s=effective_timeout,
            lease_ttl_s=max(60.0, effective_timeout + 60.0),
            preemptible=False,
            receipt_required=True,
            metadata={"legacy_facade": True, "cross_process_lock": self._lock_path},
        )
        decision = await self._admission.acquire(request)
        if not decision.admitted:
            logger.warning(
                "Evolution admission denied outcome=%s reason=%s receipt=%s",
                decision.outcome.value,
                decision.reason,
                decision.receipt_id or "none",
            )
            return False

        fd = self._get_mp_lock()
        if fd is None:
            await self._admission.release(
                decision.lease_id,
                reason="cross_process_lock_unavailable",
            )
            return False

        deadline = time.monotonic() + effective_timeout
        attempts = max(1, int(effective_timeout / 0.1) + 2)
        acquired = False
        try:
            for _attempt in range(attempts):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        except (OSError, RuntimeError, ValueError) as exc:
            record_degradation(
                "resource_arbitrator",
                exc,
                severity="warning",
                action="evolution cross-process lock acquisition failed after process-local admission",
                receipt_required=False,
            )

        if not acquired:
            os.close(fd)
            await self._admission.release(
                decision.lease_id,
                reason="cross_process_lock_timeout",
            )
            logger.warning("Evolution cross-process lock timed out after %.2fs", effective_timeout)
            return False

        with self._state_lock:
            duplicate_evolution = bool(self._evolution_lease_id)
        if duplicate_evolution:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            await self._admission.release(
                decision.lease_id,
                reason="duplicate_evolution_lease",
            )
            return False
        with self._state_lock:
            self._mp_fd = fd
            self._evolution_lease_id = decision.lease_id
            self._evolution_request_id = request.request_id
        logger.info("Evolution lease acquired receipt=%s", decision.receipt_id or "none")
        return True

    async def release_evolution(self) -> None:
        with self._state_lock:
            fd = self._mp_fd
            lease_id = self._evolution_lease_id
            self._mp_fd = None
            self._evolution_lease_id = ""
            self._evolution_request_id = ""
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        if lease_id:
            try:
                await self._admission.release(lease_id, reason="evolution_finished")
            except KeyError:
                logger.warning("Evolution lease expired before release lease=%s", lease_id)

    def is_inference_busy(self) -> bool:
        return self._admission.active_lease_count(WorkClass.INFERENCE) > 0

    def is_alive(self) -> bool:
        return self._admission.is_alive()

    def is_ready(self) -> bool:
        return self.is_alive()

    def get_status(self) -> dict[str, object]:
        with self._state_lock:
            lanes = {
                lane: list(leases)
                for lane, leases in sorted(self._inference_leases.items())
            }
            evolution_lease = self._evolution_lease_id
        return {
            "inference_active": self.is_inference_busy(),
            "inference_lanes": lanes,
            "evolution_active": bool(evolution_lease),
            "evolution_lease_id": evolution_lease,
            "admission": self._admission.status(),
        }

    @contextlib.asynccontextmanager
    async def inference_context(  # noqa: ASYNC109 - compatibility API
        self,
        priority: bool = False,
        worker: str | ModelRuntimeAssignment | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - compatibility API
    ) -> AsyncGenerator[None, None]:
        acquired = await self.acquire_inference(
            priority=priority,
            timeout=timeout,
            worker=worker,
        )
        if not acquired:
            raise TimeoutError(f"inference_token_timeout:{worker or 'default'}")
        try:
            yield
        finally:
            await self.release_inference(worker=worker)

    @contextlib.asynccontextmanager
    async def evolution_context(  # noqa: ASYNC109 - compatibility API
        self,
        timeout: float = 300.0,  # noqa: ASYNC109 - compatibility API
    ) -> AsyncGenerator[bool, None]:
        acquired = await self.acquire_evolution(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release_evolution()


_ARBITRATOR: ResourceArbitrator | None = None
_ARBITRATOR_LOCK = threading.Lock()


def get_resource_arbitrator() -> ResourceArbitrator:
    global _ARBITRATOR
    if _ARBITRATOR is None:
        with _ARBITRATOR_LOCK:
            if _ARBITRATOR is None:
                _ARBITRATOR = ResourceArbitrator()
    return _ARBITRATOR


def reset_resource_arbitrator() -> None:
    global _ARBITRATOR
    with _ARBITRATOR_LOCK:
        _ARBITRATOR = None


__all__ = [
    "ResourceArbitrator",
    "get_resource_arbitrator",
    "reset_resource_arbitrator",
]
