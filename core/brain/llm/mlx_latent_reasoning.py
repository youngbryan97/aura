"""Thinking without writing it down, and reading the hidden state back.

A latent turn asks the worker to run the loop and hand back what it was
holding rather than what it would have said. Everything that needs is here:
the request itself, the two encode paths, the readouts drained off the worker
while it runs, the progress counters a caller watches to know the lane is
alive, and the cancel acknowledgement that has to arrive before the lane can
be reused.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger("LLM.MLX")

if TYPE_CHECKING:  # only the annotations need it, and the module it lives in
    from .mlx_client import SharedFuture  # imports this one to build the class

import asyncio
import concurrent.futures as cfutures
import copy
import hashlib
import json
import math
import os
import queue
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.concurrency import run_io_bound
from core.utils.deadlines import get_deadline
from core.utils.memory_monitor import get_memory_pressure_snapshot

from .mlx_worker import (
    _HIDDEN_SEQUENCE_MAX_INPUT_CHARS,
    _HIDDEN_SEQUENCE_MAX_TOKENS,
    _HIDDEN_SEQUENCE_MAX_WIDTH,
)


class _ReasonsInLatentSpace:
    """Lifted whole from MLXLocalClient; see mlx_client.py."""

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

    def _record_latent_progress(self, response: dict[str, Any]) -> None:
        """Retain bounded parent-side evidence for the active latent stage."""
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .mlx_client import (
            _bounded_progress_value,
            _record_mlx_degradation,
        )


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
        previous = self._latent_progress_by_request.get(request_id, {})
        # Stage-only messages must not erase the last measured coordinates.
        for coordinate in ("processed_tokens", "decode_generated_tokens"):
            prior = previous.get(coordinate)
            value = snapshot.get(coordinate)
            if type(prior) is int and (type(value) is not int or value < prior):
                snapshot[coordinate] = prior
        self._latent_progress_by_request[request_id] = snapshot
        owner_id, publish = getattr(self, "_latent_delivery_progress", (None, None))
        if owner_id == request_id:
            for coordinate, total_key, phase in (
                ("processed_tokens", "total_tokens", "prefill"),
                ("decode_generated_tokens", "decode_requested_tokens", "decode"),
            ):
                completed = snapshot.get(coordinate)
                prior = previous.get(coordinate, 0)
                total = snapshot.get(total_key, 0)
                if (
                    type(completed) is int and completed > 0
                    and (type(prior) is not int or completed > prior)
                ):
                    turn_progress = getattr(self, "_latent_turn_progress", None)
                    if turn_progress is not None:
                        from core.runtime.turn_progress import note_progress

                        note_progress(progress=turn_progress)
                    if publish is not None:
                        publish(
                            phase=phase, completed=completed,
                            total=total if type(total) is int and total >= 0 else 0,
                        )
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
        from .mlx_client import (
            _real_model_path,
        )

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
        from .mlx_client import (
            _record_mlx_degradation,
        )

        _record_mlx_degradation(
            RuntimeError(f"latent_cancel_ack_unbound:{why}"),
            action="refused a latent cancellation acknowledgement it could not bind",
            severity="error",
        )

    async def _cancel_latent_request_cleanly(
        self, fut: SharedFuture, *, req_id: str, expected_request_sha256: str, reason: str,
    ) -> dict[str, Any] | None:
        """Keep request ownership until its recurrent cleanup receipt is checked."""
        from .mlx_client import (
            _LATENT_CANCEL_ACK_GRACE_S,
            _await_shared_future,
        )

        if self._current_request_id != req_id:
            return None
        self.soft_cancel_active_generation(reason)
        try:
            cancel_ack = await _await_shared_future(fut, timeout_s=_LATENT_CANCEL_ACK_GRACE_S)
        except (TimeoutError, BrokenPipeError, OSError):
            return None
        if self._clean_latent_cancel_ack(
            cancel_ack,
            expected_request_id=req_id,
            expected_request_sha256=expected_request_sha256,
        ):
            return cancel_ack
        return None

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
        from .mlx_client import (
            _AURA_SOURCE_ROOT,
            _SEAM_FELL_THROUGH,
            _apply_the_wire_action_intervention,
            _await_shared_future,
            _foreground_owner_context,
            _is_internal_inference,
            _latent_request_schema_error,
            _new_shared_future,
            _record_mlx_degradation,
            _remaining_budget,
        )

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
            from core.runtime.chat_delivery_progress import capture_generation_progress

            self._latent_delivery_progress = (req_id, capture_generation_progress())
            from core.runtime.turn_progress import capture_progress

            self._latent_turn_progress = capture_progress()
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
                cancel_ack = await self._cancel_latent_request_cleanly(
                    fut, req_id=req_id,
                    expected_request_sha256=expected_request_sha256,
                    reason="latent_reason_deadline",
                )
                if cancel_ack is not None:
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
                cancel_ack = await asyncio.shield(self._cancel_latent_request_cleanly(
                    fut, req_id=req_id,
                    expected_request_sha256=expected_request_sha256,
                    reason="latent_reason_caller_cancelled",
                ))
                if cancel_ack is None:
                    deferred_reboot = "latent_reason_caller_cancelled"
                else:
                    self._set_lane_state("ready")
                    logger.info("Latent request %s stopped with verified cleanup; resident lane preserved.", req_id)
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
                if getattr(self, "_latent_delivery_progress", (None, None))[0] == req_id:
                    self._latent_delivery_progress = (None, None)
                    self._latent_turn_progress = None
                self._release_request_lock()
                if foreground_owner_cm is not None:
                    await foreground_owner_cm.__aexit__(None, None, None)

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
        from .mlx_client import (
            _await_shared_future,
            _foreground_owner_active,
            _new_shared_future,
        )

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
        from .mlx_client import (
            _await_shared_future,
            _foreground_owner_active,
            _new_shared_future,
        )


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
