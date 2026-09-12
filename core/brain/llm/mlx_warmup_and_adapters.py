"""Getting the worker ready, and changing what it is running.

Warm-up is the part nobody sees and everybody feels: a precompile pass, a
retry that recovers the worker rather than giving up on it, and a deferral
that steps aside while a person is waiting. Beside it, the two ways the lane
changes underneath a running client — a batch request, and a swap of the
expert adapter or the artifact itself, each refused unless the thing being
swapped in has been validated first.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger("LLM.MLX")

if TYPE_CHECKING:  # annotation only; the module it lives in imports this one
    from .mlx_client import ArtifactVerdict

import asyncio
import contextlib
import gc
import math
import os
import queue
import time
import uuid
from pathlib import Path
from typing import Any

from core.utils.concurrency import run_io_bound
from core.utils.deadlines import get_deadline
from core.utils.memory_monitor import get_memory_pressure_snapshot
from core.utils.task_tracker import get_task_tracker


class _WarmsUpAndSwapsAdapters:
    """Lifted whole from MLXLocalClient; see mlx_client.py."""

    async def _generate_batch_response_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Return one task-local batched worker response without global state."""
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .mlx_client import (
            _BATCH_CANDIDATE_MAX_CHARS,
            _await_shared_future,
            _bounded_max_tokens,
            _new_shared_future,
            _record_mlx_degradation,
        )

        if self._req_q is None or self._closed:
            return {}
        # Memory-pressure admission: a 16-candidate 2048-token resident batch
        # is heavy generation. The serial and latent paths refuse under
        # critical pressure — the batch path previously dispatched anyway.
        try:
            snapshot = get_memory_pressure_snapshot()
            if snapshot.refuse_heavy_local_generation:
                _record_mlx_degradation(
                    RuntimeError(snapshot.reason or "critical_memory_pressure"),
                    action="refused batched generation under critical memory pressure",
                    severity="warning",
                )
                return {}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="refused batched generation while the memory probe was unavailable",
                severity="warning",
            )
            return {}
        alive = await self._ensure_worker_alive(request_is_background=True)
        if not alive:
            return {}
        try:
            admitted_n = max(1, min(16, int(n)))
        except (TypeError, ValueError, OverflowError):
            admitted_n = 4
        admitted_max_tokens = min(
            2048,
            _bounded_max_tokens(max_tokens, max_tokens, 512),
        )
        try:
            admitted_temperature = float(temperature)
        except (TypeError, ValueError, OverflowError):
            admitted_temperature = 0.8
        if admitted_temperature != admitted_temperature or not (
            float("-inf") < admitted_temperature < float("inf")
        ):
            admitted_temperature = 0.8
        admitted_temperature = max(0.0, min(2.0, admitted_temperature))
        try:
            admitted_timeout = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            admitted_timeout = 180.0
        if not math.isfinite(admitted_timeout):
            # Infinity previously created an UNBOUNDED wait on the future.
            admitted_timeout = 180.0
        requested_timeout = admitted_timeout
        admitted_timeout = min(600.0, max(10.0, admitted_timeout))
        if admitted_timeout != requested_timeout:
            # A batch decode cannot finish in three seconds, so the floor
            # stands — but the caller must not believe its budget was honoured
            # when the wait it actually gets is longer. Widening in silence is
            # how a bounded caller ends up unbounded.
            _record_mlx_degradation(
                ValueError(
                    f"batch timeout {requested_timeout:.1f}s outside the admissible "
                    f"range; using {admitted_timeout:.1f}s"
                ),
                action="admitted a batch decode on a budget the caller did not request",
                severity="warning",
            )
        req_id = uuid.uuid4().hex
        req = {
            "id": req_id,
            "action": "generate_batch",
            "prompt": str(prompt or ""),
            "n": admitted_n,
            "max_tokens": admitted_max_tokens,
            "temperature": admitted_temperature,
        }
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        # Register the batch decode as an ACTIVE generation for its duration.
        #
        # It used to queue the command and register only a pending future, so
        # every "is this lane busy?" check read it as idle — and those checks
        # guard consequential actions. maybe_unload_idle, the adapter swap and
        # the idle scavenger all gate on _active_generations > 0, so a batch
        # decode of n candidates on the resident 32B could have its weights
        # unloaded or its adapter swapped out from under it, mid-decode,
        # because nothing said it was running.
        #
        # Marking it busy is also what makes the durable lane non-preemptible
        # while the decode holds it, which is the property the ownership
        # bookkeeping exists to provide.
        self._active_generations += 1
        timed_out = False
        try:
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(req, principal="mlx_client.health_probe"),
                True,
                2.0,
            )
            res = await _await_shared_future(fut, timeout_s=admitted_timeout)
        except asyncio.CancelledError:
            # The caller is gone; the WORKER is not. Without this the batch
            # kept decoding n candidates on the resident model for nobody,
            # holding the lane and its memory until it finished on its own.
            timed_out = True
            raise
        except (TimeoutError, BrokenPipeError, OSError, queue.Full) as exc:
            # queue.Full included: queue saturation is expected load
            # contention and belongs inside the documented empty-response
            # fallback envelope, not raised to the caller.
            timed_out = isinstance(exc, TimeoutError)
            _record_mlx_degradation(
                exc,
                action="returned empty batch after batched generation failed; caller falls back to serial",
                severity="warning",
            )
            return {}
        finally:
            # ALWAYS unregister — caller cancellation previously left the
            # worker command live and the future registered indefinitely.
            self._pending_generations.pop(req_id, None)
            # Release the busy marker on every path, including cancellation.
            # A leaked increment is worse than never having taken one: the
            # lane would look permanently busy and the idle unload, adapter
            # swap and scavenger would all be blocked forever.
            self._active_generations = max(0, self._active_generations - 1)
            if timed_out:
                # The queued decode continues invisibly after a timeout;
                # ask the worker to yield instead of burning the lane.
                with contextlib.suppress(Exception):
                    self.soft_cancel_active_generation(reason=f"batch_timeout:{req_id[:12]}")
        if not res or res.get("status") != "ok":
            return {}
        raw_texts_value = res.get("texts")
        # A malformed worker payload (a plain string iterates as characters)
        # must not fabricate hundreds of one-character candidates.
        if not isinstance(raw_texts_value, (list, tuple)):
            _record_mlx_degradation(
                TypeError(f"batch texts payload was {type(raw_texts_value).__name__}"),
                action="dropped malformed batch response payload",
            )
            return {}
        # Bounded per candidate. The count was capped; the SIZE was not, so a
        # malformed or hostile worker could hand the parent 16 arbitrarily
        # large strings and the parent would hold every one of them.
        raw_texts = [
            str(t or "")[:_BATCH_CANDIDATE_MAX_CHARS] for t in raw_texts_value
        ][:admitted_n]
        raw_candidate_tokens = list(res.get("tokens_used_by_candidate") or [])
        texts: list[str] = []
        tokens_used_by_candidate: list[int] = []
        for index, text in enumerate(raw_texts):
            if not text.strip():
                continue
            texts.append(text)
            try:
                candidate_tokens = max(0, int(raw_candidate_tokens[index] or 0))
            except (IndexError, TypeError, ValueError, OverflowError):
                candidate_tokens = 0
            tokens_used_by_candidate.append(candidate_tokens)
        try:
            tokens_used = max(0, int(res.get("tokens_used") or 0))
        except (TypeError, ValueError, OverflowError):
            tokens_used = 0
        # The aggregate and the per-candidate totals are two claims about the
        # same decode. When they disagree the receipt is not a measurement of
        # anything, so report the sum we can actually account for and say the
        # worker's total was inconsistent rather than passing it on.
        candidate_total = sum(tokens_used_by_candidate)
        tokens_used_consistent = (not tokens_used_by_candidate) or (
            tokens_used >= candidate_total
        )
        if not tokens_used_consistent:
            _record_mlx_degradation(
                ValueError(
                    f"batch tokens_used={tokens_used} below candidate sum={candidate_total}"
                ),
                action="reported the accountable candidate total after the worker totals disagreed",
                severity="warning",
            )
            tokens_used = candidate_total

        # Batch decoding runs the same transformer hooks, so it fills the Φ
        # residual ring too. Drained here rather than left for the next
        # foreground turn: a verifier sweep is hundreds of states, and several
        # between turns would wrap the ring and throw away transitions the
        # complex could have used.
        #
        # The LATENT readouts are deliberately NOT drained here. Those inject
        # into the substrate, and verifier sampling is not Aura having a
        # thought — feeding it back would make her mood a function of how many
        # candidates a best-of-N search happened to decode.
        self._drain_phi_residual_ring()

        return {
            "texts": texts,
            "tokens_used_consistent": tokens_used_consistent,
            "request_id": req_id,
            "requested_timeout_s": requested_timeout,
            "admitted_timeout_s": admitted_timeout,
            "max_tokens": admitted_max_tokens,
            "temperature": admitted_temperature,
            "tokens_used": tokens_used,
            "tokens_used_by_candidate": tokens_used_by_candidate,
        }

    async def generate_batch_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> list[str]:
        """Decode raw verifier candidates in one batched worker pass."""

        response = await self._generate_batch_response_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        return list(response.get("texts") or [])

    async def generate_batch_with_metadata_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Return batched candidates with one truthful shared decode receipt."""

        response = await self._generate_batch_response_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        texts = list(response.get("texts") or [])
        if not texts:
            return {}
        model_name = os.path.basename(str(self.model_path or "")) or "unknown"
        candidate_tokens = list(response.get("tokens_used_by_candidate") or [])
        # CP126 536f8e0d: this said provider_verified=True because the worker
        # said status=ok, and named the model from a PATH BASENAME. Neither
        # proves which process produced the candidates or which weights it had
        # loaded — a renamed directory changed the reported model, and a
        # response from a recycled worker was attributed to the current one.
        #
        # Verification now means: we hold the worker's attested identity, and
        # this response arrived while that worker generation was serving.
        identity = self.get_worker_identity_snapshot()
        worker_boot_id = str(identity.get("worker_boot_id") or "")
        worker_pid = identity.get("worker_pid")
        provider_verified = bool(
            worker_boot_id and isinstance(worker_pid, int) and worker_pid > 0
        )
        return {
            "texts": texts,
            "generation_metadata": {
                "endpoint": f"MLX-BATCH:{model_name}",
                "provider": "mlx",
                "model": model_name,
                "model_basis": "path_basename",
                "is_local": True,
                "provider_verified": provider_verified,
                "provider_verification_basis": (
                    "attested_worker_identity" if provider_verified else "unattested"
                ),
                "worker_boot_id": worker_boot_id,
                "worker_pid": worker_pid if isinstance(worker_pid, int) else None,
                "worker_generation": int(getattr(self, "_worker_generation", 0) or 0),
                "batch_tokens_used_consistent": bool(
                    response.get("tokens_used_consistent", True)
                ),
                "batch_request_id": response.get("request_id"),
                "surface_control_receipt": {
                    "enabled": False,
                    "applied": False,
                    "generation_required": True,
                    "application_status": "raw_batch_requires_parent_verification",
                    "clean_user_surface_contract": False,
                    "surface_quality_gate_enabled": False,
                    "surface_quality_gate_passed": False,
                    "generation_max_tokens": response.get("max_tokens"),
                    "batch_generated_tokens_total": response.get("tokens_used"),
                    "batch_candidate_count": len(texts),
                    "source": "mlx_batch_worker",
                },
            },
            "candidate_generation_metadata": [
                {
                    "generated_tokens": (
                        max(0, int(candidate_tokens[index] or 0))
                        if index < len(candidate_tokens)
                        else 0
                    )
                }
                for index in range(len(texts))
            ],
        }

    async def set_expert_adapter(
        self, adapter_path: str | None, *, timeout_s: float = 90.0
    ) -> dict[str, Any]:
        """Attach/detach a domain-specialist LoRA on the RESIDENT worker model.

        The expert-LoRA library's live seam: the adapter (~40MB) is wrapped
        onto the loaded model inside the worker — no model reload, seconds not
        minutes. ``None``/"" detaches. Refuses while a generation is active
        (weights must never change mid-decode) and never spawns a worker just
        to attach — an adapter is worthless without a resident model.
        """
        from .mlx_client import (
            AdapterVerdict,
            _await_shared_future,
            _new_shared_future,
            _record_mlx_degradation,
            _validate_adapter_artifact,
        )

        path = str(adapter_path or "").strip()
        if self._closed:
            return {"ok": False, "reason": "client_closed"}
        if self._req_q is None or not (
            self._process and self._process.is_alive() and self._init_done
        ):
            return {"ok": False, "reason": "worker_not_ready"}
        if int(getattr(self, "_active_generations", 0) or 0) > 0 or self._warmup_in_flight:
            return {"ok": False, "reason": "generation_active"}
        adapter_verdict: AdapterVerdict | None = None
        if path:
            adapter_verdict = await asyncio.to_thread(
                _validate_adapter_artifact,
                Path(path).expanduser(),  # noqa: ASYNC240 - executed in to_thread
                # The resident checkpoint's training-pipeline fingerprint is
                # not something this client measures — the worker identity
                # carries a source sha and a model path, not the digest an
                # adapter's base_checkpoint_fingerprint is computed against.
                # So compatibility comes back "declared_unverified" and says
                # so, rather than passing a key that is always absent and
                # calling the resulting silence a check.
                expected_base_fingerprint="",
            )
            adapter_exists = adapter_verdict.ok
        else:
            adapter_exists = True
        if not adapter_exists:
            reason = (
                adapter_verdict.reason
                if adapter_verdict is not None and adapter_verdict.reason
                else f"adapter_missing:{path}"
            )
            logger.warning("🧬 [MLX] Refused adapter attach for %s: %s", path, reason)
            return {
                "ok": False,
                "reason": reason,
                **(adapter_verdict.as_receipt() if adapter_verdict is not None else {}),
            }

        # Re-check after the filesystem await. The check above is a
        # time-of-check/time-of-use race: it reads the counter, then this
        # coroutine yields for a directory stat, and a generation can begin in
        # that window. The swap would then be dispatched against a worker that
        # is mid-decode, which is exactly what the exclusion above exists to
        # prevent — and the caller would be told the swap was cleanly excluded.
        if int(getattr(self, "_active_generations", 0) or 0) > 0 or self._warmup_in_flight:
            return {"ok": False, "reason": "generation_active_after_stat"}

        req_id = uuid.uuid4().hex
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        try:
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(
                    {"id": req_id, "action": "set_expert_adapter", "path": path},
                    principal="mlx_client.expert_adapter",
                ),
                True,
                2.0,
            )
            res = await _await_shared_future(fut, timeout_s=max(10.0, float(timeout_s)))
        except asyncio.CancelledError:
            # Identical hazard to the timeout below, and it had no handler at
            # all: the swap command is already queued, so cancelling the
            # CALLER does not stop the worker from attaching the adapter. The
            # future must be unregistered, the worker asked to abandon it, and
            # the resident adapter state marked unknown — otherwise the next
            # reader believes weights that may already have changed.
            self._pending_generations.pop(req_id, None)
            with contextlib.suppress(Exception):
                self.soft_cancel_active_generation(
                    reason=f"adapter_swap_cancelled:{req_id[:12]}"
                )
            self._expert_adapter_state_unknown = True
            _record_mlx_degradation(
                RuntimeError("expert adapter swap cancelled with the command queued"),
                action=(
                    "resident adapter state is UNKNOWN after caller cancellation; "
                    "re-read worker identity before trusting it"
                ),
                severity="error",
            )
            raise
        except (TimeoutError, BrokenPipeError, OSError) as exc:
            self._pending_generations.pop(req_id, None)
            # Do NOT claim the model was left unchanged. The command is already
            # on the worker's queue; dropping our future only stops US from
            # hearing about it. The adapter can still attach afterwards, and
            # the previous receipt asserted the opposite — a false statement
            # about which weights are resident, which is the one thing an
            # adapter receipt exists to get right.
            #
            # Ask the worker to abandon it, then report the state as UNKNOWN
            # rather than unchanged. An unknown adapter is recoverable by
            # re-reading identity; a wrongly-asserted one is not.
            with contextlib.suppress(Exception):
                self.soft_cancel_active_generation(
                    reason=f"adapter_swap_timeout:{req_id[:12]}"
                )
            self._expert_adapter_state_unknown = True
            _record_mlx_degradation(
                exc,
                action=(
                    "expert adapter swap timed out with the command already queued; "
                    "resident adapter state is UNKNOWN until identity is re-read"
                ),
                severity="error",
            )
            return {
                "ok": False,
                "reason": f"swap_timeout:{type(exc).__name__}",
                "resident_adapter_state": "unknown",
            }

        if res and res.get("status") == "ok":
            # A completed swap that reports identity resolves the uncertainty
            # a previous timeout may have left behind.
            self._expert_adapter_state_unknown = False
            raw_worker_identity = res.get("worker_identity")
            try:
                transitioned_identity = self._accept_worker_identity_transition(
                    raw_worker_identity
                )
            except (TypeError, ValueError) as exc:
                _record_mlx_degradation(
                    exc,
                    action=(
                        "recycled resident worker after expert adapter swap "
                        "returned an unprovable identity transition"
                    ),
                    severity="critical",
                )
                await self.reboot_worker(
                    reason="expert_adapter_identity_transition_unproven",
                    mark_failed=False,
                )
                return {
                    "ok": False,
                    "reason": f"identity_transition_unproven:{exc}",
                }
            self._worker_identity = transitioned_identity
            self._expert_adapter_path = str(res.get("resident") or "") or None
            return {
                "ok": True,
                "resident": self._expert_adapter_path,
                "wrapped_layers": int(res.get("wrapped_layers") or 0),
                "detached_layers": int(res.get("detached_layers") or 0),
            }
        if res and res.get("requires_worker_recycle") is True:
            await self.reboot_worker(
                reason="expert_adapter_swap_identity_unrecovered",
                mark_failed=False,
            )
        return {
            "ok": False,
            "reason": str((res or {}).get("message") or "swap_failed"),
        }

    @property
    def expert_adapter_resident(self) -> str | None:
        return getattr(self, "_expert_adapter_path", None)

    async def reload_model_artifact(self, model_path: str) -> dict[str, Any]:
        """Serve a newly published fused artifact by re-pointing this lane.

        The model lives in the WORKER process, so the only correct swap is a
        worker recycle with the new path. (This replaces a retired
        live_learner monkey-patch that loaded a second full copy of the model
        into the ORCHESTRATOR process — ~20GB of wired memory on the cortex lane
        — while generations kept flowing through the worker's old weights.)
        Busy lanes defer the recycle until the active request finishes; the
        respawn path re-resolves the fused manifest, so crash recovery after
        the swap also serves the promoted artifact.
        """
        from .mlx_client import (
            _validate_model_artifact,
        )

        resolved = await asyncio.to_thread(lambda: Path(str(model_path or "")).expanduser())
        previous = self.model_path
        verdict = await asyncio.to_thread(_validate_model_artifact, resolved, previous)
        if not verdict.ok:
            logger.warning(
                "🧬 [MLX] Refused artifact promotion for %s: %s", resolved.name, verdict.reason
            )
            return {
                "ok": False,
                "state": "rejected",
                "reason": verdict.reason,
                "previous": previous,
                "artifact": str(resolved),
                **verdict.as_receipt(),
            }

        if (
            int(getattr(self, "_active_generations", 0) or 0) > 0
            or self._current_request_started_at > 0.0
        ):
            # CP126 8ccdcd3b: model_path used to change HERE, so for the rest
            # of the active request the old worker kept decoding old weights
            # while status, logging and admission all described it as the new
            # model. The desired artifact is now held separately and published
            # only when a worker is actually serving it.
            #
            # CP126 7f4435f5: the promotion also used to be stored in
            # _deferred_reboot_reason, the same scalar first-token, token-stall,
            # heartbeat and fence-loss recovery write to. Whichever fired last
            # won, so a generation failure could silently discard a staged
            # promotion. It is a separate intent now.
            self._pending_promotion = str(resolved)
            logger.info(
                "🧬 [MLX] Promoted artifact staged for %s; activating after the active request.",
                resolved.name,
            )
            return {
                "ok": True,
                "state": "staged",
                "mode": "deferred",
                "previous": previous,
                "artifact": str(resolved),
                **verdict.as_receipt(),
            }
        return await self._activate_promoted_artifact(str(resolved), verdict=verdict)

    async def _activate_promoted_artifact(
        self, target: str, *, verdict: ArtifactVerdict | None = None
    ) -> dict[str, Any]:
        """Publish a validated artifact as this lane's serving identity.

        CP126 41fa9f3c: the old path awaited ``reboot_worker`` — a TEARDOWN —
        then logged "Promoted artifact live" and returned ok=true. Nothing had
        spawned, handshaked, or loaded a single weight. The lane was in fact
        unloaded, and the first caller after the promotion paid the cold start
        and discovered any load failure. The receipt now names the state it
        can actually prove: ``unloaded`` after a clean recycle, ``failed`` if
        the recycle did not complete. Only ``_promotion_is_serving`` reports
        ``ready``, and only after a worker answers on the new path.
        """
        from .mlx_client import (
            _rebind_client_registry_key,
            _record_mlx_degradation,
        )

        previous = self.model_path
        proof = verdict.as_receipt() if verdict is not None else {}
        try:
            from core.brain.llm.model_registry import get_model_runtime_assignment

            next_assignment = get_model_runtime_assignment(target)
            if next_assignment.role != self.runtime_assignment.role:
                raise RuntimeError(
                    "promoted_artifact_runtime_role_changed:"
                    f"{self.runtime_assignment.role}->{next_assignment.role}"
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="refused promoted artifact without a source-bound runtime assignment",
                severity="error",
            )
            return {
                "ok": False,
                "state": "rejected",
                "reason": str(exc),
                "previous": previous,
                "artifact": target,
                **proof,
            }
        self.runtime_assignment = next_assignment
        self.model_path = next_assignment.model_path
        self._expert_adapter_path = None  # adapters belong to the old weights
        self._pending_promotion = None
        try:
            await self.reboot_worker(reason="promoted_artifact_swap", mark_failed=False)
        except Exception as exc:  # noqa: BLE001 - a failed recycle must be reported, not raised
            _record_mlx_degradation(
                exc,
                action="left the promoted artifact as the serving identity after recycle failed",
                severity="error",
            )
            return {
                "ok": False,
                "state": "failed",
                "reason": f"recycle_failed:{type(exc).__name__}",
                "previous": previous,
                "artifact": target,
                **proof,
            }
        _rebind_client_registry_key(previous, self.model_path, self)
        logger.info(
            "🧬 [MLX] Promoted artifact is now this lane's serving identity: %s "
            "(worker unloaded; it loads on next use).",
            Path(target).name,
        )
        return {
            "ok": True,
            "state": "unloaded",
            "mode": "recycled",
            "previous": previous,
            "artifact": target,
            **proof,
        }

    async def _run_warmup_precompile(
        self,
        *,
        request_is_background: bool,
        foreground_request: bool,
        owner_name: str,
        warmup_timeout: float,
    ) -> None:
        # CP126 b4fcd100. The second attempt used to be given
        # `warmup_timeout + 10`, and the readiness probe that follows took its
        # own independent timeout on top. So the documented warmup budget was
        # not a bound on warmup at all — it was a per-attempt allowance that
        # grew when things went badly, which is exactly when a caller waiting
        # on this needs the bound to hold.
        #
        # One campaign deadline, shared: whatever the retry and the probe
        # spend comes out of the same budget, and an attempt with nothing left
        # does not start.
        from .mlx_client import (
            _MAX_READINESS_PROBE_S,
            _MIN_READINESS_PROBE_BUDGET_S,
            _background_deferral_active,
            _clear_matching_foreground_owner,
            _record_mlx_degradation,
            _runtime_shutdown_requested,
            _WarmupDeferredError,
        )

        campaign_deadline = time.monotonic() + max(1.0, float(warmup_timeout))
        last_exc: Exception | None = None
        for attempt in range(2):
            remaining = campaign_deadline - time.monotonic()
            if remaining <= 0.0:
                if last_exc is not None:
                    raise last_exc from None
                raise TimeoutError(
                    f"warmup_budget_exhausted:{warmup_timeout:.1f}s"
                )
            try:
                warmup_text = await asyncio.wait_for(
                    self._generate_inner(
                        "Hello",
                        _retry=True,
                        request_is_background=request_is_background,
                        foreground_request=foreground_request,
                        owner_label=owner_name,
                        max_tokens=1,
                        warmup_precompile=True,
                    ),
                    timeout=remaining,
                )
                if warmup_text is None and not self.is_alive():
                    # A worker that was never started is not a worker that
                    # died.
                    #
                    # `_generate_inner` returns None and logs "stopped before
                    # worker spawn" when a background deferral is in force —
                    # the runtime deciding, on purpose, not to spawn a 27B
                    # while the cortex is starting. The warm-up then found no
                    # worker alive and called it dead: a degradation at
                    # warning, a MARGINAL fault record, and a resilience hit
                    # of frustration 0.13 and depletion 0.05, every boot, for
                    # the runtime doing exactly what it meant to do.
                    #
                    # LIVE, 2026-09-09, three lines apart: `Background
                    # generation for Aura-Qwen3.8-27B stopped before worker
                    # spawn (cortex_startup_quiet)` then `FAULT
                    # RUNTIME-MLX_CLIENT [MARGINAL] ...
                    # warmup_precompile_worker_dead`.
                    deferral = _background_deferral_active(
                        owner_name or os.path.basename(self.model_path)
                    )
                    if deferral:
                        raise _WarmupDeferredError(str(deferral))
                    raise RuntimeError("warmup_precompile_worker_dead")
                # CP126 cdd743de + b6439433. A nonempty token from a
                # max_tokens=1 "Hello" proves Metal shaders compiled — it does
                # NOT prove this lane can hold a conversation, and skipping the
                # visible probe on that basis is how a lane that cannot answer
                # got marked ready. The probe now ALWAYS runs, and its answer is
                # actually checked against what was asked for (the prompt says
                # "Reply exactly: ready" but any nonblank text used to pass, so
                # hallucinated, garbled, stale, or prompt-echo output proved
                # readiness).
                # The probe is bounded by what is LEFT of the campaign, never by
                # a floor applied on top of it. A floor there (`max(10.0, ...)`)
                # meant a campaign with 2s left still handed the probe 10s, so
                # the one hard deadline this function exists to enforce was
                # exceeded by up to 10s on exactly the slow boots that made a
                # caller depend on it.
                #
                # "A probe that starts at all deserves a fair chance" is still
                # right — it is a rule about whether to START one, not about how
                # long an already-doomed one may run. So a campaign without
                # _MIN_READINESS_PROBE_BUDGET_S left does not open a probe; it
                # ends, inside the budget it promised.
                probe_budget = campaign_deadline - time.monotonic()
                if probe_budget < _MIN_READINESS_PROBE_BUDGET_S:
                    self._set_lane_state(
                        "recovering", "warmup_budget_exhausted_before_readiness_probe"
                    )
                    raise TimeoutError(
                        f"warmup_budget_exhausted:{warmup_timeout:.1f}s:"
                        f"probe_needs:{_MIN_READINESS_PROBE_BUDGET_S:.1f}s:"
                        f"left:{max(0.0, probe_budget):.1f}s"
                    )
                logger.info(
                    "🔥 [MLX] Verifying conversation readiness for %s with a visible probe.",
                    os.path.basename(self.model_path),
                )
                # The same prover the reconciler uses, so that asking and
                # recording can never drift apart again. Out of the SAME
                # campaign budget as the precompile above (CP126 b4fcd100): a
                # probe that took its own independent timeout is how the
                # documented warmup bound became a suggestion.
                proved = await self.prove_visible_readiness(
                    budget_s=min(probe_budget, _MAX_READINESS_PROBE_S),
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_label=owner_name,
                )
                if proved != "proved":
                    self._set_lane_state("recovering", f"warmup_readiness_{proved}")
                    raise RuntimeError(f"warmup_readiness_{proved}")
                self._last_ready_at = time.time()
                self._warmup_in_flight = False
                _clear_matching_foreground_owner(owner_name)
                logger.info("🔥 [MLX] Warmup complete — Metal shaders compiled.")
                return
            except asyncio.CancelledError as exc:
                last_exc = exc
                if _runtime_shutdown_requested():
                    logger.info(
                        "🛑 [MLX] Warmup pre-compile cancelled for %s during runtime shutdown.",
                        os.path.basename(self.model_path),
                    )
                    raise
                _record_mlx_degradation(
                    exc,
                    action="retried or recycled warmup precompile after cancellation",
                )
                raise
            except (RuntimeError, TimeoutError, AttributeError) as exc:
                last_exc = exc
                if attempt == 0:
                    # The recovery between attempts used to sit OUTSIDE the
                    # campaign: an unawaited-cost gc, a `reboot_worker` with no
                    # bound of its own, and a flat 1s settle. So "one campaign
                    # deadline, shared" described the two generations only, and
                    # a warmup that promised 1s routinely took several — the
                    # composed contract differing from what the local code says,
                    # which is the failure mode this whole function was rewritten
                    # to remove.
                    #
                    # A retry now has to FIT: it is worth starting only if what
                    # remains could still carry a probe, and every second it
                    # spends recovering comes out of the same budget.
                    recovery_budget = campaign_deadline - time.monotonic()
                    if recovery_budget <= _MIN_READINESS_PROBE_BUDGET_S:
                        raise last_exc from None
                    logger.warning(
                        "⚠️ [MLX] Warmup pre-compile failed once for %s: %s. "
                        "Retrying cleanly within the remaining %.1fs...",
                        os.path.basename(self.model_path),
                        exc,
                        recovery_budget,
                    )
                    try:
                        await asyncio.wait_for(
                            self._recover_worker_for_warmup_retry(),
                            timeout=recovery_budget,
                        )
                    except TimeoutError:
                        raise last_exc from None
                    continue
                raise last_exc from None

    async def _recover_worker_for_warmup_retry(self) -> None:
        """Reclaim and reboot the worker between two warmup attempts.

        Split out so the caller can put ONE bound around the whole recovery.

        Never while somebody is being answered. A warmup exists to make the
        lane ready, and tearing the worker down mid-reply to do it defeats the
        thing it is for: LIVE 2026-08-29, a person interrupted a long errand,
        was correctly given the lane, and had her answer cancelled underneath
        her by a retry — "generation cancelled during expected reboot
        (warmup_precompile_retry)" — receiving a stub about being cut short.

        Waiting is the whole remedy. A reply takes seconds and the retry has
        its own budget to spend; if that budget runs out while a person is
        being served, the honest outcome is a warmup that did not get its
        retry, not a person who did not get her answer.
        """
        from .mlx_client import (
            _FOREGROUND_OWNER_IS_USER_FACING,
            _WAIT_OUT_A_REPLY_S,
        )

        waited = 0.0
        while _FOREGROUND_OWNER_IS_USER_FACING and waited < _WAIT_OUT_A_REPLY_S:
            await asyncio.sleep(0.25)
            waited += 0.25
        if _FOREGROUND_OWNER_IS_USER_FACING:
            logger.info(
                "[MLX] warmup retry stood down: somebody is still being answered"
            )
            return
        await asyncio.to_thread(gc.collect)
        await self.reboot_worker(reason="warmup_precompile_retry", mark_failed=False)
        # A freshly rebooted worker needs a moment before it can answer; the
        # caller's bound decides whether there is a moment to give it.
        await asyncio.sleep(1.0)

    async def warmup(
        self,
        *,
        foreground_request: bool | None = None,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Boot the worker and prove the visible conversation path is ready.

        SINGLEFLIGHT (CP126 4d8a7d6b). Concurrent callers used to each set
        ``_warmup_in_flight`` and proceed, so two warmups could load/evict the
        same lane at once; the "stale warmup" recovery then measured
        ``_lane_transition_at`` — a timestamp any other lane transition
        refreshes — and force-cleared the shared flag without proving the prior
        warmup had ended. Callers now JOIN the active warmup, and a genuinely
        stuck one is cancelled and awaited before a replacement starts.
        """
        from .mlx_client import (
            _WARMUP_STALE_AFTER_S,
            _join_inflight_across_loops,
            _record_mlx_degradation,
        )

        inflight = self._warmup_inflight
        if inflight is not None and not inflight.done():
            age = max(0.0, time.monotonic() - self._warmup_started_at)
            if age <= _WARMUP_STALE_AFTER_S:
                # Join the in-flight warmup. shield() so that a cancelled
                # joiner cannot kill the warmup the other callers need.
                #
                # Across loops, awaiting it directly is not a join — it raises
                # "got Future attached to a different loop", which the handler
                # below then reports as a WARMUP FAILURE to the joiner.
                #
                # LIVE 2026-08-17: that is why the first message after every
                # launch was answered with "the live answer lane could not
                # finish preparing". Boot starts the warmup on the boot loop;
                # the chat turn arrives on the server loop, joins, and is told
                # instantly that warmup failed — so admission reported
                # "worker_not_alive,init_not_complete,lane_warming" no matter
                # how much budget the turn had. Three budget-side fixes moved
                # the failure time and none removed it, because the failure was
                # never about time.
                try:
                    return bool(await _join_inflight_across_loops(inflight))
                except asyncio.CancelledError:
                    raise
                except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                    _record_mlx_degradation(
                        exc,
                        action="reported warmup failure to a joined singleflight caller",
                        severity="warning",
                    )
                    return False
            logger.warning(
                "🔧 [MLX] Warmup for %s stuck for %.0fs — cancelling the prior "
                "warmup task before starting a replacement.",
                os.path.basename(self.model_path),
                age,
            )
            inflight.cancel()
            # PROVE the prior task ended before starting another one.
            try:
                await asyncio.wait_for(asyncio.shield(inflight), timeout=10.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass
            self._warmup_in_flight = False

        task = get_task_tracker().create_task(
            self._warmup_impl(
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
        )
        self._warmup_inflight = task
        self._warmup_started_at = time.monotonic()
        try:
            return bool(await asyncio.shield(task))
        finally:
            if self._warmup_inflight is task and task.done():
                self._warmup_inflight = None

    async def _warmup_impl(
        self,
        *,
        foreground_request: bool | None = None,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Boot the worker and prove the visible conversation path is ready."""
        from .mlx_client import (
            _FOREGROUND_OWNER_NAME,
            _foreground_owner_active,
            _foreground_owner_context,
            _record_mlx_degradation,
            _shutdown_blocks_model_work,
            _WarmupDeferredError,
        )

        if _shutdown_blocks_model_work(self.model_path, action="warmup"):
            self._warmup_in_flight = False
            if self._lane_state not in {"failed", "cold"}:
                self._set_lane_state("cold", "runtime_shutdown")
            return False
        if foreground_request is None:
            foreground_request = self._is_primary_or_deep_lane()
        else:
            foreground_request = bool(foreground_request)
        request_is_background = not foreground_request
        owner_name = f"warmup:{os.path.basename(self.model_path)}"
        warmup_timeout = self._warmup_timeout()
        self._warmup_attempted = True
        # Stale-warmup recovery lives in warmup()'s singleflight now: it owns
        # the task handle, so it can cancel and PROVE termination instead of
        # force-clearing a shared flag against an unrelated timestamp
        # (CP126 4d8a7d6b). This flag remains the cheap state other lifecycle
        # paths poll.
        self._warmup_in_flight = True
        self._set_lane_state("warming")
        try:
            if foreground_request:
                try:
                    async with _foreground_owner_context(
                        owner_name,
                        # [STABILITY v56] Raised from 90s → 180s. The 32B model
                        # cold-loads in 90-150s; holding the foreground owner
                        # for only 90s released it before warmup finished,
                        # allowing background 7B spawns to evict the cortex.
                        deadline=get_deadline(max(180.0, warmup_timeout)),
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
                            logger.info(
                                "⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path)
                            )
                            return False

                        try:
                            await self._run_warmup_precompile(
                                request_is_background=request_is_background,
                                foreground_request=foreground_request,
                                owner_name=owner_name,
                                warmup_timeout=warmup_timeout,
                            )
                        except _WarmupDeferredError as deferred:
                            # Nothing failed and nothing is recorded against
                            # her: the runtime declined to spawn, and the
                            # warm-up says so and stands down.
                            logger.info(
                                "⏸️ [MLX] Warmup deferred for %s: the runtime is not "
                                "spawning workers right now (%s).",
                                os.path.basename(self.model_path),
                                deferred,
                            )
                            return False
                        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                            self._set_lane_state(
                                "recovering", f"warmup_precompile_failed:{type(e).__name__}"
                            )
                            _record_mlx_degradation(
                                e,
                                action="kept warmup lane recoverable after foreground precompile failure",
                            )
                            self._record_degraded_event(
                                "warmup_precompile_failed",
                                detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                                severity="warning",
                                foreground_request=foreground_request,
                            )
                            logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                            return False
                except TimeoutError as exc:
                    self._set_lane_state("recovering", "warmup_foreground_owner_timeout")
                    self._record_degraded_event(
                        "warmup_foreground_owner_timeout",
                        detail=f"{os.path.basename(self.model_path)}:{exc}",
                        severity="warning",
                        foreground_request=foreground_request,
                    )
                    logger.info(
                        "⏸️ [MLX] Warmup deferred for %s: %s", os.path.basename(self.model_path), exc
                    )
                    return False
                return True

            if _shutdown_blocks_model_work(self.model_path, action="background warmup"):
                return False

            # CP126 811cde6f: this yield check used to run AFTER
            # _ensure_worker_alive, so a background lane could load a 20GB
            # model (or evict the resident one) and only then decide to defer
            # its precompile — defeating the very anti-thrash policy the check
            # exists to enforce. Decide BEFORE touching worker lifecycle.
            #
            # Background lanes (solver promotions, brainstem appraisals) yield
            # to an owned foreground. The PRIMARY lane's own warmup is exempt:
            # the foreground owner is usually a turn WAITING on exactly this
            # warmup, and deferring it deadlocked the lane live (2026-07-10:
            # 206s foreground budget expired every turn while the precompile
            # it needed sat deferred behind it).
            if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
                logger.info(
                    "⏸️ [MLX] Background warmup deferred for %s (before worker spawn) while foreground lane is owned by %s.",
                    os.path.basename(self.model_path),
                    _FOREGROUND_OWNER_NAME or "foreground",
                )
                return False

            alive = await self._ensure_worker_alive(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
            if not alive:
                if self._lane_state != "failed":
                    self._set_lane_state("recovering", "warmup_deferred")
                logger.info("⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path))
                return False
            if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
                # Re-check: a foreground turn can take ownership while the
                # worker was coming up.
                logger.info(
                    "⏸️ [MLX] Background warmup precompile deferred for %s while foreground lane is owned by %s.",
                    os.path.basename(self.model_path),
                    _FOREGROUND_OWNER_NAME or "foreground",
                )
                return False

            try:
                await self._run_warmup_precompile(
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_name=owner_name,
                    warmup_timeout=warmup_timeout,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self._set_lane_state("recovering", f"warmup_precompile_failed:{type(e).__name__}")
                _record_mlx_degradation(
                    e,
                    action="kept warmup lane recoverable after precompile failure",
                )
                self._record_degraded_event(
                    "warmup_precompile_failed",
                    detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                return False
            return True
        finally:
            self._warmup_in_flight = False

    async def warm_up(self, **kwargs):
        """Backward-compatible alias for older call sites."""
        return await self.warmup(**kwargs)
