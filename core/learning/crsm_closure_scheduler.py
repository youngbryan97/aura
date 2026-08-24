"""Autonomous, governed closer for Aura's CRSM-to-LoRA learning loop.

Captured experience is not learned merely because a training command returned
zero. Closure means that an exact, resource-admitted pipeline ran under one
semantic-weight authority receipt and the CRSM monitor subsequently observed a
current consumed marker. This scheduler owns that full transaction.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.resource_observation import get_resource_observer
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.CRSMClosureScheduler")

SERVICE_NAME = "crsm_closure"
AUTHORITY_SOURCE = "system_maintenance:crsm_closure"
PLASTIC_TARGET = "crsm_lora_adapter"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_PIPELINE_STAGES = 2
_ALLOWED_STAGE_COMMANDS: dict[str, tuple[str, ...]] = {
    "prepare_dataset": ("python", "training/build_dataset_v3.py"),
    "crsm_delta_train_fuse_publish": (
        "python",
        "training/train_and_fuse.py",
        "--crsm-delta",
        "--tag",
        "crsm-closeout",
    ),
}
_RECOVERABLE = (
    ImportError,
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    subprocess.SubprocessError,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


class CRSMClosureScheduler:
    """Periodic, idle-gated owner of one CRSM learning transaction."""

    def __init__(self, orchestrator: Any = None) -> None:
        self._orchestrator = orchestrator
        self._task: asyncio.Task[Any] | None = None
        self._active = False
        self._running_cycle = False
        self._state_lock = asyncio.Lock()
        self._state_loaded = False
        self._last_attempt_at = 0.0
        self._last_status = "never"
        self._last_phase = ""
        self._last_detail: dict[str, Any] = {}
        self._loop_cache: dict[str, Any] = {}
        self._resource_requirements_loaded = False
        self._resource_model_path_cache = ""
        self.check_interval_s = float(_env_int("AURA_CRSM_AUTOCLOSE_CHECK_INTERVAL_S", 900))
        self.cooldown_s = float(_env_int("AURA_CRSM_AUTOCLOSE_COOLDOWN_S", 6 * 3600))
        self.retry_s = float(_env_int("AURA_CRSM_AUTOCLOSE_RETRY_S", 900))
        self.min_free_gb = float(_env_int("AURA_CRSM_AUTOCLOSE_MIN_FREE_GB", 40))
        self._required_free_gb_cache = self.min_free_gb
        self._model_request_gb_cache = self.min_free_gb
        # A large-Cortex CRSM delta plus fuse can run for well over an hour; a
        # shorter timeout wastes the whole pass. Budget three
        # hours and let iteration limits shorten work rather than the clock.
        self.train_timeout_s = float(
            _env_int("AURA_CRSM_AUTOCLOSE_TIMEOUT_S", 3 * 3600)
        )
        self._state_path = (
            Path(os.getenv("AURA_STATE_DIR", str(state_root() / "run")))
            / "crsm_closure_state.json"
        )

    @staticmethod
    def enabled() -> bool:
        """Autonomous closure is normal operation; the env flag is a kill switch."""
        return _env_flag("AURA_CRSM_AUTOCLOSE", True)

    async def start(self) -> None:
        if not self.enabled():
            logger.info("CRSM-to-LoRA autonomous closure disabled by AURA_CRSM_AUTOCLOSE=0")
            return
        if self._task is not None and not self._task.done():
            return
        await self._ensure_state_loaded()
        await self._ensure_resource_requirements()
        from core.utils.task_tracker import get_task_tracker

        self._active = True
        self._task = get_task_tracker().create_task(
            self._run(), name="crsm_closure_scheduler", owner=SERVICE_NAME
        )
        logger.info(
            "CRSM-to-LoRA closure scheduler online (check %.0fs, cooldown %.0fs, floor %.0fGB)",
            self.check_interval_s,
            self.cooldown_s,
            self.min_free_gb,
        )

    async def stop(self) -> None:
        self._active = False
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown is bounded upstream
            pass

    async def _run(self) -> None:
        await asyncio.sleep(min(self.check_interval_s, 600.0))
        while self._active:
            try:
                await self._maybe_close()
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="skipped one closure check; retained captures for the next check",
                )
            await asyncio.sleep(self.check_interval_s)

    # -- durable scheduler state -----------------------------------------
    def _read_state_file(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        return dict(json.loads(self._state_path.read_text(encoding="utf-8")))

    async def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        async with self._state_lock:
            if self._state_loaded:
                return
            try:
                state = await asyncio.to_thread(self._read_state_file)
            except (OSError, TypeError, ValueError) as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="ignored corrupt closure scheduler state and started from a clean cache",
                )
                state = {}
            self._last_attempt_at = float(state.get("last_attempt_at", 0.0) or 0.0)
            self._last_status = str(state.get("last_status", "never") or "never")
            self._last_phase = str(state.get("last_phase", "") or "")
            self._last_detail = dict(state.get("last_detail") or {})
            self._state_loaded = True

    def _set_cached_status(
        self,
        status: str,
        *,
        phase: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._last_status = str(status)
        self._last_phase = str(phase)
        self._last_detail = dict(detail or {})

    async def _persist_status(
        self,
        status: str,
        *,
        phase: str = "",
        detail: dict[str, Any] | None = None,
        starts_cooldown: bool = False,
    ) -> None:
        await self._ensure_state_loaded()
        async with self._state_lock:
            if starts_cooldown:
                self._last_attempt_at = time.time()
            self._set_cached_status(status, phase=phase, detail=detail)
            payload = {
                "schema_version": 1,
                "last_attempt_at": self._last_attempt_at,
                "last_status": self._last_status,
                "last_phase": self._last_phase,
                "last_detail": self._last_detail,
                "updated_at": time.time(),
            }
            try:
                from core.governance_context import local_internal_governed_scope

                with local_internal_governed_scope(
                    "crsm_closure_scheduler.state",
                    domain="file_write",
                    constraints={"artifact": "scheduler_state", "effect": "durable_cooldown"},
                ):
                    await async_atomic_write_text(
                        self._state_path,
                        json.dumps(payload, sort_keys=True),
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="continued with in-memory closure state after durable state write failed",
                )

    # -- observation and admission --------------------------------------
    async def _read_loop_state(self) -> tuple[Any, dict[str, Any]]:
        from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

        monitor = get_crsm_loop_monitor()
        state = dict(await asyncio.to_thread(monitor.loop_state))
        self._loop_cache = state
        return monitor, state

    async def _next_action(self, monitor: Any, state: dict[str, Any]) -> dict[str, Any]:
        cached = state.get("next_action")
        if isinstance(cached, dict):
            return dict(cached)
        return dict(await asyncio.to_thread(monitor.next_action))

    def _idle_allows(self) -> bool:
        from core.runtime.background_policy import (
            MAINTENANCE_BACKGROUND_POLICY,
            background_activity_allowed,
        )

        return bool(
            background_activity_allowed(
                self._orchestrator,
                profile=MAINTENANCE_BACKGROUND_POLICY,
            )
        )

    def _base_model_path(self) -> Path:
        configured = os.getenv("AURA_LORA_BASE_MODEL")
        if not configured:
            from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

            configured = get_runtime_model_path(ACTIVE_MODEL)
        candidate = Path(configured).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("crsm_training_model_unavailable") from exc
        if not resolved.is_dir():
            raise RuntimeError("crsm_training_model_not_directory")
        return resolved

    def _compute_resource_requirements(
        self,
        model_path: Path | None = None,
    ) -> tuple[float, float]:
        try:
            from core.runtime.model_lane_control import estimate_model_job_footprint_gb

            model_path = model_path or self._base_model_path()
            train_peak = estimate_model_job_footprint_gb(
                str(model_path),
                purpose="train",
            )
            fused_peak = estimate_model_job_footprint_gb(
                str(model_path),
                purpose="fuse",
            )
            request_gb = max(float(train_peak), float(fused_peak))
            return max(self.min_free_gb, request_gb + 2.0), request_gb
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return self.min_free_gb, self.min_free_gb

    async def _ensure_resource_requirements(self) -> None:
        resolved_model = self._base_model_path()
        model_path = str(resolved_model)
        if (
            self._resource_requirements_loaded
            and self._resource_model_path_cache == model_path
        ):
            return
        required, request = await asyncio.to_thread(
            self._compute_resource_requirements,
            resolved_model,
        )
        self._required_free_gb_cache = float(required)
        self._model_request_gb_cache = float(request)
        self._resource_model_path_cache = model_path
        self._resource_requirements_loaded = True

    def _ram_admits(self) -> tuple[bool, str]:
        try:
            memory = get_resource_observer().memory(include_process_tree=False)
            if not memory.available:
                detail = memory.error or "unavailable"
                return False, f"ram_probe_unavailable:{detail}"
            free_gb = memory.available_bytes / (1024**3)
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            return False, f"ram_probe_unavailable:{type(exc).__name__}"
        required_gb = self._required_free_gb_cache
        if free_gb < required_gb:
            return False, f"insufficient_free_ram:{free_gb:.1f}GB<{required_gb:.1f}GB"
        return True, f"free_ram:{free_gb:.1f}GB>={required_gb:.1f}GB"

    @staticmethod
    def _validated_stage_command(action: dict[str, Any]) -> tuple[str, list[str]]:
        if action.get("required") is not True:
            raise ValueError("monitor action is not marked required")
        phase = str(action.get("phase") or "").strip()
        expected = _ALLOWED_STAGE_COMMANDS.get(phase)
        supplied = tuple(str(part) for part in (action.get("command") or ()))
        if expected is None:
            raise ValueError(f"unsupported CRSM closure phase:{phase or 'missing'}")
        if supplied != expected:
            raise ValueError(f"CRSM closure command substitution rejected for phase:{phase}")
        script = (_REPO_ROOT / expected[1]).resolve()
        try:
            script.relative_to(_REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"CRSM closure script escaped repository:{script}") from exc
        if not script.is_file():
            raise ValueError(f"CRSM closure script missing:{script}")
        return phase, [sys.executable, str(script), *expected[2:]]

    @staticmethod
    def _authority_context(reason: str, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "reason": str(reason),
            "loop_state": str(state.get("state") or "unknown"),
            "unconsumed": int(state.get("unconsumed", 0) or 0),
            "target_module": PLASTIC_TARGET,
            "effect_scope": "model_weight_mutation",
            "bounded_pipeline": True,
            "max_pipeline_stages": _MAX_PIPELINE_STAGES,
            "allowed_phases": sorted(_ALLOWED_STAGE_COMMANDS),
            "rollback_strategy": "retain_previous_fused_model_and_atomically_publish_manifest",
        }

    async def _authorize_closure(self, reason: str, state: dict[str, Any]) -> tuple[Any, Any]:
        from core.executive.authority_gateway import get_authority_gateway

        gateway = get_authority_gateway()
        decision = await gateway.authorize_semantic_weight_update(
            AUTHORITY_SOURCE,
            "close_crsm_lora_learning_loop",
            target_module=PLASTIC_TARGET,
            context=self._authority_context(reason, state),
            priority=0.7,
        )
        return gateway, decision

    # -- triggers --------------------------------------------------------
    async def _maybe_close(self) -> None:
        if self._running_cycle or not self.enabled():
            return
        await self._ensure_state_loaded()
        retry_statuses = {
            "cancelled",
            "blocked",
            "execution_failed",
            "incomplete",
            "prepare_failed",
            "train_failed",
        }
        cooldown = self.retry_s if self._last_status in retry_statuses else self.cooldown_s
        if time.time() - self._last_attempt_at < cooldown:
            return
        _monitor, state = await self._read_loop_state()
        if state.get("state") != "open" or not self._idle_allows():
            return
        await self._execute_closure(reason="scheduled_idle")

    async def run_closure_now(self, *, reason: str = "manual") -> dict[str, Any]:
        """Run the governed transaction now, retaining RAM and single-flight gates."""
        if self._running_cycle:
            return {"status": "blocked", "reasons": ["closure_already_running"]}
        if not self.enabled():
            return {"status": "blocked", "reasons": ["disabled_by_env"]}
        return await self._execute_closure(reason=reason)

    async def _execute_closure(
        self,
        *,
        reason: str,
    ) -> dict[str, Any]:
        self._running_cycle = True
        gateway: Any = None
        decision: Any = None
        outcome: dict[str, Any] = {"status": "execution_failed", "reasons": ["not_started"]}
        try:
            await self._ensure_state_loaded()
            monitor, state = await self._read_loop_state()
            if state.get("state") != "open":
                outcome = {"status": "noop", "reasons": ["loop_not_open"], "loop": state}
                return outcome

            await self._ensure_resource_requirements()

            ram_ok, ram_reason = self._ram_admits()
            if not ram_ok:
                self._set_cached_status("deferred", detail={"reason": ram_reason})
                logger.info("CRSM closure deferred (%s): %s", reason, ram_reason)
                outcome = {"status": "deferred", "reasons": [ram_reason], "loop": state}
                return outcome

            action = await self._next_action(monitor, state)
            try:
                phase, command = self._validated_stage_command(action)
            except ValueError as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="blocked an unrecognized or substituted CRSM closure command",
                )
                await self._persist_status(
                    "blocked",
                    detail={"reason": str(exc)},
                    starts_cooldown=True,
                )
                outcome = {
                    "status": "blocked",
                    "reasons": [f"invalid_monitor_command:{exc}"],
                    "loop": state,
                }
                return outcome

            gateway, decision = await self._authorize_closure(reason, state)
            if not bool(getattr(decision, "approved", False)):
                authority_reason = str(getattr(decision, "reason", "authority_denied"))
                self._set_cached_status("will_declined", phase=phase, detail={"reason": authority_reason})
                logger.info("Authority declined CRSM closure (%s): %s", reason, authority_reason)
                outcome = {
                    "status": "will_declined",
                    "reasons": [authority_reason],
                    "loop": state,
                }
                return outcome

            outcome = await self._run_authorized_pipeline(
                monitor=monitor,
                initial_state=state,
                initial_phase=phase,
                initial_command=command,
                decision=decision,
                reason=reason,
            )
            return outcome
        except asyncio.CancelledError:
            outcome = {"status": "cancelled", "reasons": ["scheduler_cancelled"]}
            if self._last_attempt_at > 0.0:
                await self._persist_status(
                    "cancelled",
                    phase=self._last_phase,
                    detail={"reason": "scheduler_cancelled"},
                )
            raise
        except _RECOVERABLE as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="left the CRSM loop open after closure execution failed",
            )
            outcome = {
                "status": "execution_failed",
                "reasons": [f"{type(exc).__name__}:{exc}"],
                "loop": dict(self._loop_cache),
            }
            if self._last_attempt_at > 0.0:
                await self._persist_status(
                    "execution_failed",
                    phase=self._last_phase,
                    detail={"error": f"{type(exc).__name__}:{exc}"},
                )
            return outcome
        finally:
            if gateway is not None and bool(getattr(decision, "approved", False)):
                success = outcome.get("status") == "closed"
                reasons = ";".join(str(item) for item in (outcome.get("reasons") or []))
                closure = gateway.finalize_tool_execution(
                    executive_intent_id=getattr(decision, "executive_intent_id", None),
                    capability_token_id=getattr(decision, "capability_token_id", None),
                    standing_authority_token=getattr(decision, "standing_authority_token", None),
                    success=success,
                    result=outcome,
                    error="" if success else reasons or str(outcome.get("status")),
                )
                outcome["authority_closure"] = closure
            self._running_cycle = False

    async def _run_authorized_pipeline(
        self,
        *,
        monitor: Any,
        initial_state: dict[str, Any],
        initial_phase: str,
        initial_command: list[str],
        decision: Any,
        reason: str,
    ) -> dict[str, Any]:
        phase = initial_phase
        command = initial_command
        state = dict(initial_state)
        seen_phases: set[str] = set()
        stages: list[dict[str, Any]] = []
        attempt_recorded = False

        for _index in range(_MAX_PIPELINE_STAGES):
            if phase in seen_phases:
                return await self._incomplete_result(
                    state,
                    stages,
                    phase=phase,
                    reason=f"repeated_pipeline_phase:{phase}",
                )
            seen_phases.add(phase)

            ram_ok, ram_reason = self._ram_admits()
            if not ram_ok:
                if stages:
                    return await self._incomplete_result(
                        state,
                        stages,
                        phase=phase,
                        reason=ram_reason,
                    )
                self._set_cached_status("deferred", phase=phase, detail={"reason": ram_reason})
                return {"status": "deferred", "reasons": [ram_reason], "loop": state}

            if not attempt_recorded:
                await self._persist_status(
                    "running",
                    phase=phase,
                    detail={"reason": reason},
                    starts_cooldown=True,
                )
                attempt_recorded = True

            try:
                from core.governance_context import governed_scope

                async with governed_scope(decision):
                    result = await self._run_training(
                        command,
                        phase=phase,
                        decision=decision,
                    )
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                result = {
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}:{exc}",
                }
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action=f"left the CRSM loop open after {phase} failed to execute",
                )

            stage_receipt = {
                "phase": phase,
                "returncode": result.get("returncode"),
                "stderr_tail": str(result.get("stderr", ""))[-500:],
            }
            stages.append(stage_receipt)
            if result.get("returncode") != 0:
                failed_status = (
                    "train_failed"
                    if phase == "crsm_delta_train_fuse_publish"
                    else "prepare_failed"
                )
                await self._persist_status(
                    failed_status,
                    phase=phase,
                    detail=stage_receipt,
                )
                return {
                    "status": failed_status,
                    "returncode": result.get("returncode"),
                    "stderr_tail": stage_receipt["stderr_tail"],
                    "stages": stages,
                    "loop": dict(self._loop_cache or state),
                }

            monitor, state = await self._read_loop_state()
            if state.get("state") == "closed":
                closure_errors = self._authority_closure_errors(state, decision)
                if closure_errors:
                    return await self._incomplete_result(
                        state,
                        stages,
                        phase=phase,
                        reason="authority_closure_mismatch:" + ",".join(closure_errors),
                    )
                await self._persist_status(
                    "closed",
                    phase=phase,
                    detail={"reason": str(state.get("reason") or "verified_closed")},
                )
                logger.info("CRSM-to-LoRA loop verified closed via %s", reason)
                return {
                    "status": "closed",
                    "loop": state,
                    "reason": reason,
                    "stages": stages,
                    "will_receipt_id": getattr(decision, "will_receipt_id", None),
                }

            action = await self._next_action(monitor, state)
            try:
                next_phase, next_command = self._validated_stage_command(action)
            except ValueError as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="stopped CRSM closure after a stage produced an invalid next action",
                )
                return await self._incomplete_result(
                    state,
                    stages,
                    phase=phase,
                    reason=f"invalid_next_action:{exc}",
                )
            if next_phase in seen_phases:
                return await self._incomplete_result(
                    state,
                    stages,
                    phase=next_phase,
                    reason=f"successful_stage_did_not_advance:{next_phase}",
                )
            phase, command = next_phase, next_command

        return await self._incomplete_result(
            state,
            stages,
            phase=phase,
            reason="pipeline_stage_budget_exhausted",
        )

    def _authority_closure_errors(
        self,
        state: dict[str, Any],
        decision: Any,
    ) -> list[str]:
        expected_receipt = str(getattr(decision, "will_receipt_id", "") or "")
        expected_intent = str(getattr(decision, "executive_intent_id", "") or "")
        marker = dict(state.get("consumption_marker") or {})
        active_governance = dict(state.get("active_model_governance") or {})
        crsm_state = dict((state.get("training_state") or {}).get("crsm_delta") or {})
        training_governance = dict(crsm_state.get("governance") or {})
        errors: list[str] = []
        if not bool(state.get("marker_matches_dataset")):
            errors.append("marker_dataset_identity")
        if float(marker.get("consumed_at", 0.0) or 0.0) < self._last_attempt_at:
            errors.append("marker_predates_attempt")
        if str(marker.get("governance_receipt_id") or "") != expected_receipt:
            errors.append("marker_receipt")
        if str(marker.get("authority_intent_id") or "") != expected_intent:
            errors.append("marker_intent")
        if str(marker.get("model_path") or "") != str(state.get("active_model") or ""):
            errors.append("marker_active_model")
        if active_governance.get("will_receipt_id") != expected_receipt:
            errors.append("active_manifest_receipt")
        if active_governance.get("executive_intent_id") != expected_intent:
            errors.append("active_manifest_intent")
        if crsm_state.get("status") != "fused_published_marker_ready":
            errors.append("training_state_status")
        if training_governance.get("will_receipt_id") != expected_receipt:
            errors.append("training_state_receipt")
        if training_governance.get("executive_intent_id") != expected_intent:
            errors.append("training_state_intent")
        return errors

    async def _incomplete_result(
        self,
        state: dict[str, Any],
        stages: list[dict[str, Any]],
        *,
        phase: str,
        reason: str,
    ) -> dict[str, Any]:
        error = RuntimeError(f"CRSM closure incomplete:{reason}")
        record_degradation(
            SERVICE_NAME,
            error,
            action="kept the CRSM loop open because post-execution proof was incomplete",
        )
        await self._persist_status(
            "incomplete",
            phase=phase,
            detail={"reason": reason, "observed_state": str(state.get("state") or "unknown")},
        )
        return {
            "status": "incomplete",
            "reasons": [reason],
            "stages": stages,
            "loop": state,
        }

    def _model_lane_claim(self) -> Any:
        from core.runtime.model_lane_control import LaneClaim
        from core.runtime.model_runtime_assignment import (
            issue_unqualified_model_runtime_assignment,
        )

        model_path = str(self._base_model_path())
        request_gb = self._model_request_gb_cache
        request_id = f"crsm-closeout-{uuid.uuid4()}"
        return LaneClaim(
            owner_id=f"crsm_closure:{os.getpid()}:{request_id}",
            model_path=model_path,
            request_gb=request_gb,
            purpose="compound",
            priority=80,
            preemptible=True,
            foreground=False,
            reservation_ttl_s=self.train_timeout_s + 60.0,
            owner_lease_ttl_s=self.train_timeout_s + 60.0,
            request_id=request_id,
            runtime_assignment=issue_unqualified_model_runtime_assignment(
                model_path=model_path,
                purpose="compound",
                authority_source="crsm_closure_scheduler",
            ),
            metadata={
                "source": AUTHORITY_SOURCE,
                "pipeline": "crsm_delta_train_fuse_publish",
                "bounded": True,
                "allow_inherited_model_children": True,
                "allowed_inherited_model_purposes": ["train", "fuse", "benchmark"],
                "allowed_inherited_model_roots": [
                    str((_REPO_ROOT / "training" / "fused-model").resolve())
                ],
            },
        )

    async def _run_training(
        self,
        command: list[str],
        *,
        phase: str,
        decision: Any,
    ) -> dict[str, Any]:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        receipt_id = str(getattr(decision, "will_receipt_id", "") or "")
        intent_id = str(getattr(decision, "executive_intent_id", "") or "")
        if not receipt_id or not intent_id:
            raise RuntimeError("semantic weight authority is missing receipt or executive intent")

        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"
        if phase == "crsm_delta_train_fuse_publish":
            child_env.update(
                {
                    "AURA_GOVERNANCE_MODE": "delegated_subprocess",
                    "AURA_REQUIRE_GOVERNANCE": "0",
                    "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID": receipt_id,
                    "AURA_DELEGATED_GOVERNANCE_DOMAIN": "semantic_weight_update",
                    "AURA_DELEGATED_GOVERNANCE_SOURCE": AUTHORITY_SOURCE,
                    "AURA_DELEGATED_AUTHORITY_INTENT_ID": intent_id,
                    "AURA_DELEGATED_GOVERNANCE_PARENT_PID": str(os.getpid()),
                    # The live parent is expected on this scheduler-owned path;
                    # model-lane admission will evict or refuse conflicting
                    # model owners before the worker is allowed to start.
                    "AURA_TRAINING_ALLOW_LIVE_AURA": "1",
                }
            )
        else:
            # Dataset preparation is still authorized by the active parent
            # scope, but it does not need portable authority secrets.
            for key in (
                "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID",
                "AURA_DELEGATED_GOVERNANCE_DOMAIN",
                "AURA_DELEGATED_GOVERNANCE_SOURCE",
                "AURA_DELEGATED_AUTHORITY_INTENT_ID",
                "AURA_DELEGATED_GOVERNANCE_PARENT_PID",
            ):
                child_env.pop(key, None)
            child_env["AURA_GOVERNANCE_MODE"] = "delegated_subprocess_child"
            child_env["AURA_REQUIRE_GOVERNANCE"] = "0"
        lane_claim = (
            self._model_lane_claim()
            if phase == "crsm_delta_train_fuse_publish"
            else None
        )
        result = await get_subprocess_gateway().run_async(
            command,
            cwd=_REPO_ROOT,
            env=child_env,
            # Training can emit hours of progress. Inheriting the runtime log
            # streams avoids retaining an unbounded PIPE buffer in Aura's heap.
            capture_output=False,
            timeout=self.train_timeout_s,
            offline_tooling=False,
            source=f"{AUTHORITY_SOURCE}:{phase}",
            model_lane_claim=lane_claim,
            accelerator_capability="model",
        )
        return {
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[-1000:],
            "stderr": (result.stderr or "")[-1000:],
        }

    def get_status(self) -> dict[str, Any]:
        """Return only cached state; health polling must never perform disk I/O."""
        return {
            "enabled": self.enabled(),
            "running_cycle": self._running_cycle,
            "check_interval_s": self.check_interval_s,
            "cooldown_s": self.cooldown_s,
            "retry_s": self.retry_s,
            "min_free_gb": self.min_free_gb,
            "required_free_gb": self._required_free_gb_cache,
            "state_loaded": self._state_loaded,
            "last_attempt_at": self._last_attempt_at,
            "last_status": self._last_status,
            "last_phase": self._last_phase,
            "last_detail": dict(self._last_detail),
            "loop": dict(self._loop_cache),
        }


_scheduler: CRSMClosureScheduler | None = None


def get_crsm_closure_scheduler(orchestrator: Any = None) -> CRSMClosureScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = CRSMClosureScheduler(orchestrator)
    return _scheduler


def reset_crsm_closure_scheduler_for_test() -> None:
    global _scheduler
    _scheduler = None


__all__ = [
    "AUTHORITY_SOURCE",
    "CRSMClosureScheduler",
    "PLASTIC_TARGET",
    "SERVICE_NAME",
    "get_crsm_closure_scheduler",
    "reset_crsm_closure_scheduler_for_test",
]
