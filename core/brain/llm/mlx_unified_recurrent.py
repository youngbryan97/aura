"""The unified recurrent lane: the shadow probe, the qualified decode, and the canaries that keep them honest.

A recurrent decode qualified for serving is a different thing from one that has
only been shadowed, and the difference is evidence rather than a flag. The
probes run the loop without serving from it, the canaries re-run a decode whose
seal has already been checked, and the qualified path refuses to serve anything
that has not been through both.
"""
from __future__ import annotations

import asyncio
import copy
import math
import os
import queue
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.utils.concurrency import run_io_bound
from core.utils.deadlines import get_deadline
from core.utils.memory_monitor import get_memory_pressure_snapshot


class _RunsTheUnifiedRecurrentLane:
    """Lifted whole from MLXLocalClient; see mlx_client.py."""

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
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .mlx_client import (
            SharedFuture,
            _await_shared_future,
            _bounded_maintenance_counters,
            _foreground_owner_active,
            _new_shared_future,
            _record_mlx_degradation,
        )


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

    async def unified_recurrent_shadow_probe_async(
        self,
        public_token_ids: Sequence[int],
        expected_token_ids: Sequence[int],
        *,
        max_tokens: int,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Measure resident recurrent tissue without exposing or serving its text."""
        from .mlx_client import (
            SharedFuture,
            _await_shared_future,
            _foreground_owner_active,
            _new_shared_future,
            _record_mlx_degradation,
            _remaining_budget,
        )


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
        from .mlx_client import (
            SharedFuture,
            _await_shared_future,
            _new_shared_future,
            _record_mlx_degradation,
            _remaining_budget,
        )


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
