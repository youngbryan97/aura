"""Bringing the worker up, and putting it back when it dies.

Lifted whole out of MLXLocalClient, which is the largest module in this
tree and the one every live turn goes through. Spawning, the readiness
handshake, the durable lane lease and the reboot are one subject: they all
answer "is there a worker, and if not, why not, and what does it cost to
get one".

Every name these methods take from `mlx_client` is imported at CALL time,
including the ones `mlx_client` itself imported from somewhere else. The
module these came from imports this one to build the class, so a
module-level import would be a cycle — and more than that, a test patches
`mlx_client.get_resource_observer`, not the module it came from. Importing
from the origin here would leave those patches landing on a name nothing in
this file reads. `runtime_hygiene_patches` takes the same approach for the
same reason.
"""
from __future__ import annotations

import asyncio
import copy
import fcntl
import functools
import gc
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.runtime.descriptor_owner import OwnsDescriptors
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
)

from .mlx_worker import _mlx_worker_loop
from .worker_progress import create_channel as create_worker_progress_channel


class _KeepsTheWorkerAlive(OwnsDescriptors):
    """The worker lifecycle half of MLXLocalClient; see mlx_client.py.

    A client owns a worker process and the pipes, locks and shared memory it
    talks to it through, so a copy is the client itself and the subject fork
    never rewinds one. It did: a restore reset the request lock's owner record
    while a request still held the lock, the request could not release a lock
    no longer recorded as its own, and every call after it waited on "another
    request, held 0.0s" until it timed out.
    """

    async def _renew_durable_lane_lease(
        self,
        controller: Any,
        owner_id: str,
        fencing_token: int,
    ) -> bool:
        """Renew the model-lane lease, distinguishing "no" from "never asked".

        LIVE DEFECT, 2026-08-10. A user turn came back "I couldn't get to an
        answer I'd stand behind on that one." The health pulse for that second:

            conversation_lane: cold (worker_not_alive)
            inference_gate (is_inference_ready() returned False)
            event_loop_monitor.last_lag_s 10.46 >= 5.00
            mlx_client (critical): TimeoutError
                → stopped MLX worker after durable lane heartbeat failed

        Nothing was wrong with the worker or the lease. The event loop had
        been blocked for ten seconds, so this renewal — awaited with a five
        second budget measured ON that loop — could not be scheduled at all
        before its timer fired. The resulting TimeoutError was read as
        `model_lane_fence_lost`, which killed a healthy 32B worker, failed
        every in-flight turn, and cost ~35s to reload.

        A timeout taken on a starved loop is not evidence about the lease. It
        is the absence of an answer being recorded as a negative one — the same
        shape as a skipped check reported as a failed check.

        So on timeout, ask exactly once more. If the loop was merely blocked,
        the retry runs on a loop that is moving again and answers truthfully.
        If the lease is genuinely gone, the retry says so and the caller kills
        the worker as before. One retry, not a loop: a second timeout is
        itself evidence that the loop is not recovering, and a foreground turn
        cannot wait forever to find out.

        Deliberately no lag threshold anywhere in here. Asking again is
        cheaper, more direct, and does not need a number that would have to be
        tuned per host.
        """
        from .mlx_client import (
            _LEASE_RENEWAL_TIMEOUT_S,
            _record_mlx_degradation,
            logger,
        )

        try:
            return bool(
                await asyncio.wait_for(
                    controller.heartbeat_owner(owner_id, fencing_token=fencing_token),
                    timeout=_LEASE_RENEWAL_TIMEOUT_S,
                )
            )
        except TimeoutError:
            logger.warning(
                "⏱️ [MLX] Lane lease renewal timed out; re-asking once before "
                "treating the lane as lost (a blocked event loop cannot "
                "distinguish a dead lease from an unasked question)."
            )
        try:
            alive = bool(
                await asyncio.wait_for(
                    controller.heartbeat_owner(owner_id, fencing_token=fencing_token),
                    timeout=_LEASE_RENEWAL_TIMEOUT_S,
                )
            )
        except TimeoutError:
            # Second timeout. Before killing anything, ask the one question
            # that is actually about the worker: is the process alive?
            #
            # A lease renewal talks to the lane CONTROLLER — a file lock and a
            # small database. Its silence is evidence about that controller, or
            # about an event loop too busy to run it. It is not evidence about
            # a 32B process that the operating system says is running and whose
            # response pipe is healthy. Killing it for a controller stall is a
            # category error, and an expensive one: the reload takes ~35s, the
            # reload itself blocks the loop, and the next renewal then times out
            # the same way. That is the spiral observed live — "Primary 32B
            # cortex is dead. Triggering background respawn (Attempt 1/5)" on a
            # worker that had never stopped answering.
            if self.is_alive():
                _record_mlx_degradation(
                    TimeoutError("lane_lease_renewal_unanswered_worker_alive"),
                    action=(
                        "lane lease renewal did not answer twice, but the worker "
                        "process is alive and its pipe is healthy; kept it and "
                        "left the lease to the next heartbeat"
                    ),
                    severity="warning",
                )
                return True
            raise
        if alive:
            logger.info(
                "⏱️ [MLX] Lane lease renewal recovered after one timeout; "
                "the re-ask found the lease intact and the worker stayed live."
            )
        return alive

    async def _renew_durable_lane_lease_in_background(
        self,
        controller: Any,
        owner_id: str,
        fencing_token: int,
        queue_generation: int,
    ) -> None:
        """Renew and reconcile an exact owner generation.

        The response listener must remain dedicated to IPC drainage.  A stale
        renewal result is harmless unless the same owner and fencing token are
        still authoritative when the result is applied.
        """
        from .mlx_client import (
            _cancel_shared_future,
            _record_mlx_degradation,
            _set_shared_future_result,
            logger,
        )

        try:
            lease_alive = await self._renew_durable_lane_lease(
                controller,
                owner_id,
                fencing_token,
            )
            if not lease_alive:
                raise RuntimeError("model_lane_fence_lost")
        except asyncio.CancelledError:
            raise
        except (
            OSError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            TimeoutError,
        ) as exc:
            current_owner, current_token, _receipt_id = self._durable_model_lane_owner_snapshot()
            if current_owner != owner_id or current_token != fencing_token:
                logger.info(
                    "Ignored stale MLX lease result for retired owner=%s token=%s",
                    owner_id,
                    fencing_token,
                )
                return
            _record_mlx_degradation(
                exc,
                action="stopped MLX worker after durable lane heartbeat failed",
                severity="critical",
            )
            self._deferred_reboot_reason = "model_lane_fence_lost"
            # Fail every waiter before dropping the process handle.  Otherwise
            # callers can miss both the dead-process and pending-future tests.
            for req_id, pending in list(self._pending_generations.items()):
                if pending is not None and not pending.done():
                    _set_shared_future_result(
                        pending,
                        {
                            "status": "error",
                            "action": "generate",
                            "id": str(req_id),
                            "message": "model_lane_fence_lost",
                        },
                    )
            current_fut = self._current_gen_future
            if current_fut is not None and not current_fut.done():
                _set_shared_future_result(
                    current_fut,
                    {
                        "status": "error",
                        "action": "generate",
                        "id": self._current_request_id,
                        "message": "model_lane_fence_lost",
                    },
                )
            if self._init_future is not None and not self._init_future.done():
                _cancel_shared_future(self._init_future)
            self._pending_generations.clear()
            self._current_gen_future = None
            self._active_generations = 0
            self._release_detached_request_lock()
            self._clear_detached_worker_requests()

            process, self._process = self._process, None
            if process is not None:
                await asyncio.to_thread(
                    self._release_worker_process, process, reason="model_lane_fence_lost"
                )
            from core.runtime.model_lane_control import unregister_model_lane_owner_adapter

            unregister_model_lane_owner_adapter(owner_id)
            with self._model_lane_state_lock:
                if self._model_lane_fencing_token == fencing_token:
                    self._model_lane_fencing_token = 0
                    self._model_lane_terminal_receipt_id = ""
            self._set_lane_state("cold", "model_lane_fence_lost")
            self._listener_stop_generation = queue_generation




    def _spawn_worker_blocking(self) -> mp.Process:
        """Isolated spawn logic for the MLX worker, run in a background thread."""
        from .mlx_client import (
            ModelLoadAdmissionRefused,
            _acquire_spawn_file_lock,
            _memory_pressure_blocks_worker_spawn,
            _open_spawn_lock_file,
            _probe_mlx_runtime,
            _record_mlx_degradation,
            _shutdown_blocks_model_work,
            get_resource_observer,
            get_subprocess_gateway,
            logger,
            psutil,
            state_root,
        )

        if _shutdown_blocks_model_work(self.model_path, action="worker spawn"):
            raise RuntimeError("runtime_shutdown")
        self._reset_worker_scoped_state()
        # [STABILITY v60] Reclaim the old/orphan worker BEFORE the memory
        # admission check. A recycle (or crash respawn) replaces a worker that
        # is still resident; killing it below frees its ~20GB. Running the
        # headroom check FIRST saw the about-to-die worker's memory and refused
        # the spawn (memory_pressure_refused_worker_spawn:model_load_headroom:
        # 20.2GB < 22.0GB → recycled_model_lane_not_live_after_warmup → DNU
        # FATAL), so the wedge could never recover. Free first, then admit.
        #
        # [STABILITY v51] Orphan reclamation: kill any existing MLXWorker
        # processes for this model path before spawning a new one.
        orphan_scan_completed = False
        try:
            model_basename = os.path.basename(self.model_path)
            target_name = f"MLXWorker-{model_basename}"
            for observed_process in get_resource_observer().processes():
                if _shutdown_blocks_model_work(self.model_path, action="orphan scan"):
                    raise RuntimeError("runtime_shutdown")
                try:
                    pname = observed_process.name
                    command = observed_process.cmdline
                    if target_name in pname or (
                        command
                        and any(model_basename in str(arg) for arg in command)
                        and "mlx_worker" in str(command)
                    ):
                        ancestor_pids = set(observed_process.ancestor_pids)
                        if observed_process.pid != os.getpid() and os.getpid() in ancestor_pids:
                            logger.warning(
                                "🧹 [STABILITY] Killing orphan MLXWorker pid=%d for %s",
                                observed_process.pid,
                                model_basename,
                            )
                            action_process = psutil.Process(observed_process.pid)
                            action_process.kill()
                            action_process.wait(timeout=3.0)
                        elif observed_process.pid != os.getpid():
                            logger.info(
                                "Model-path match pid=%d for %s belongs to another root; "
                                "durable lane accounting will arbitrate it without cross-root kill.",
                                observed_process.pid,
                                model_basename,
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            orphan_scan_completed = True
        except (OSError, ConnectionError, TimeoutError) as orphan_exc:
            _record_mlx_degradation(
                orphan_exc,
                action="continued worker spawn after orphan reclamation scan failed",
            )
            logger.debug("Orphan reclamation scan failed: %s", orphan_exc)

        # CP126 1399e019. A failed scan used to be logged "non-fatal" and the
        # spawn proceeded. That is safe only if no worker of ours survives —
        # and a failed scan is precisely the case where we do not know. For a
        # SAME-CLIENT replacement the consequence is a second copy of a 20GB
        # model resident at once, which exhausts unified memory long before
        # durable accounting notices.
        #
        # So: when the scan could not run, this client's own prior handle must
        # be provably terminal before we spawn beside it. Unobservable counts
        # as alive.
        if not orphan_scan_completed:
            prior = self._process
            prior_terminal = prior is None
            if prior is not None:
                try:
                    prior_terminal = not bool(prior.is_alive())
                except (RuntimeError, AttributeError, ValueError, OSError) as exc:
                    logger.debug("Prior worker liveness unreadable, not treating it as terminal: %s", exc)
                    prior_terminal = False
            if not prior_terminal:
                error = RuntimeError(
                    "orphan_reclamation_unobservable_refused_worker_spawn:"
                    f"{os.path.basename(self.model_path)}"
                )
                _record_mlx_degradation(
                    error,
                    action=(
                        "refused a same-client worker replacement while orphan "
                        "reclamation was unobservable and the prior process was not "
                        "proven terminal"
                    ),
                    severity="critical",
                )
                logger.critical(
                    "🚨 [MLX] Refusing to spawn a replacement worker for %s: the "
                    "orphan scan could not run and the previous process is not "
                    "proven dead. Spawning would risk two resident copies.",
                    os.path.basename(self.model_path),
                )
                raise error

        memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)

        # The orphan scan above only reaps workers from PREVIOUS incarnations —
        # it explicitly skips this client's own process. So a worker that loaded
        # the model but never finished initializing sits there holding its
        # weights while the lane, which has already given up on it, refuses to
        # spawn the replacement for want of the very memory it is holding.
        #
        # Measured live 2026-07-26: available 17.4GB against a 24GB gate, with
        # ~16GB of it wired to our own unusable worker. Killing the instance by
        # hand dropped wired 21.7GB -> 5.2GB and freed 34.6GB. Aura could not
        # recover on her own from a state she created, which is the worst shape
        # a deadlock can take.
        #
        # Narrow by construction: only when the spawn is about to be refused
        # anyway, only our own process, only one that never became usable, and
        # only when it is serving nobody. Then re-check, so the reclaim wait
        # below observes the memory this just freed.
        if memory_block and not self._is_deep_solver_lane():
            try:
                stale = self._process
                stale_alive = bool(
                    stale is not None
                    and getattr(stale, "is_alive", lambda: False)()
                )
                if (
                    stale_alive
                    and not self._init_done
                    and int(getattr(self, "_active_generations", 0) or 0) == 0
                ):
                    logger.warning(
                        "🧹 [MLX] Reclaiming our own never-initialized worker pid=%s "
                        "before refusing a spawn for headroom (%s) — it is holding "
                        "the memory the replacement needs and serving no one.",
                        getattr(stale, "pid", "unknown"),
                        memory_block,
                    )
                    reclaimed = self._kill_and_join_blocking(
                        stale,
                        cooperative=False,
                    )
                    if not reclaimed:
                        raise RuntimeError(
                            "never_initialized_worker_reclamation_unproven:"
                            f"pid={getattr(stale, 'pid', 'unknown')}"
                        )
                    self._process = None
                    self._init_done = False
                    memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
            except (OSError, AttributeError, RuntimeError, ValueError) as reclaim_exc:
                _record_mlx_degradation(
                    reclaim_exc,
                    action="continued spawn admission after self-worker reclaim failed",
                )

        if memory_block and not self._is_deep_solver_lane():
            # A worker we just killed (orphan reclamation above, or a prior
            # generation-timeout force-abort) frees ~18GB, but the OS reclaim of
            # wired Metal memory lags process exit. Checking headroom instantly
            # sees the pre-reclaim number and refuses — which takes the whole
            # conversation lane COLD even though the memory is about to be free.
            # Observed live during the 200-turn soak (2026-07-06): a Cortex
            # generation timed out, the worker was killed, respawn was refused
            # at 20.3GB < 24GB while the killed worker's 18.6GB had not yet been
            # reclaimed, and a cluster of turns died until pressure eased. Wait
            # (bounded) for reclaim and re-check before refusing. Runs in
            # _spawn_worker_blocking's executor thread, so the sleep does not
            # block the event loop; the deep-solver lane still refuses instantly.
            try:
                reclaim_wait_s = float(
                    os.environ.get("AURA_MLX_SPAWN_RECLAIM_WAIT_S", "15") or 15.0
                )
            except (TypeError, ValueError):
                reclaim_wait_s = 15.0
            reclaim_deadline = time.monotonic() + max(0.0, reclaim_wait_s)
            waited = False
            while memory_block and time.monotonic() < reclaim_deadline:
                if _shutdown_blocks_model_work(self.model_path, action="memory reclaim wait"):
                    raise RuntimeError("runtime_shutdown")
                waited = True
                time.sleep(1.5)
                if _shutdown_blocks_model_work(self.model_path, action="memory reclaim retry"):
                    raise RuntimeError("runtime_shutdown")
                memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
            if waited and not memory_block:
                logger.info(
                    "🟢 [MLX] Headroom recovered after worker reclaim; proceeding with spawn."
                )
        if memory_block:
            error = ModelLoadAdmissionRefused(memory_block)
            if self._is_deep_solver_lane():
                logger.warning(
                    "🛡️ [MLX] Refusing optional deep Solver spawn before model load: %s",
                    memory_block,
                )
                raise error
            _record_mlx_degradation(
                error,
                action="refused MLX worker spawn before model load due to memory pressure",
                severity="critical",
            )
            raise error

        runtime_ok, runtime_detail = _probe_mlx_runtime()
        if not runtime_ok:
            raise RuntimeError(f"mlx_runtime_probe_failed:{runtime_detail}")
        if _shutdown_blocks_model_work(self.model_path, action="post-runtime-probe spawn"):
            raise RuntimeError("runtime_shutdown")

        if self._req_q is None or self._res_q is None:
            raise RuntimeError("MLX IPC queues must be created before worker spawn")
        ctx = self._mp_context

        lock_dir = state_root() / "run"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file_path = str(lock_dir / "mlx_spawn.lock")
        lock_file = _open_spawn_lock_file(lock_file_path)
        with lock_file:
            try:
                logger.info("🔒 [MLX] Acquiring process-level spawn lock...")
                _acquire_spawn_file_lock(lock_file, model_path=self.model_path)
                if _shutdown_blocks_model_work(self.model_path, action="locked worker spawn"):
                    raise RuntimeError("runtime_shutdown")

                project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # CP126 841bf5f7. A fresh contract-signing key per spawn: it
                # is handed to the child at fork, never persisted, and is
                # meaningless to any other worker. Privileged output
                # contracts must be signed with it to take effect.
                from core.brain.llm.contract_authority import new_contract_key
                from core.brain.llm.latent_cortex.worker_capture_identity import (
                    build_worker_capture_launch_authority,
                )

                self._contract_key = new_contract_key()
                self._worker_capture_launch_authority = (
                    build_worker_capture_launch_authority()
                )
                self._worker_progress_channel = create_worker_progress_channel(ctx)
                if _shutdown_blocks_model_work(self.model_path, action="worker process start"):
                    raise RuntimeError("runtime_shutdown")
                p = get_subprocess_gateway().spawn_python_process(
                    PythonProcessSpec(
                        target=_mlx_worker_loop,
                        args=(
                            self.model_path,
                            self._req_q,
                            self._res_q,
                            self.device,
                            self._substrate_mem,
                            self._steering_active,
                            self._cancel_seq,
                            self._contract_key,
                            dict(self._worker_capture_launch_authority.challenge),
                            self._phi_residual_mem,
                            self._latent_readout_mem,
                            self._worker_progress_channel,
                        ),
                        source="mlx_local_client.worker_owner",
                        name=f"MLXWorker-{os.path.basename(self.model_path)}",
                        role=ProcessRole.MODEL_WORKER,
                        requested_privileges=frozenset(
                            {
                                Privilege.FILESYSTEM_READ,
                                Privilege.FILESYSTEM_WRITE,
                                Privilege.MODEL_WEIGHTS,
                            }
                        ),
                        accelerator_capability=AcceleratorCapability.MODEL,
                        daemon=True,
                        start_method=str(ctx.get_start_method()),
                    ),
                    context=ctx,
                )
                return p

            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                logger.info("🔓 [MLX] Released process-level spawn lock.")

    async def _spawn_worker(self) -> mp.Process:
        from .mlx_client import (
            _shutdown_blocks_model_work,
        )

        if _shutdown_blocks_model_work(self.model_path, action="async worker spawn"):
            raise RuntimeError("runtime_shutdown")
        return await asyncio.get_running_loop().run_in_executor(None, self._spawn_worker_blocking)

    async def _ensure_worker_alive(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Self-healing supervisor for the MLX worker.

        [OOM FIX] Acquires a global semaphore so only ONE model loads at a time.
        This prevents the 32B + 7B from loading simultaneously and crashing Metal.
        """
        from .mlx_client import (
            _FOREGROUND_OWNER_NAME,
            _background_deferral_active,
            _foreground_owner_active,
            _model_load_admission_context,
            _ModelLoadAdmissionDeniedError,
            _normalize_recurrent_depth_status,
            _record_mlx_degradation,
            _recurrent_depth_readiness_blocker,
            _shutdown_blocks_model_work,
            _spawn_gate_context,
            logger,
        )

        if _shutdown_blocks_model_work(self.model_path, action="worker start/recovery"):
            return False
        if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
            # Same inversion as the warmup guard (2026-07-10): the Reflex
            # fallback serving turns OWNED the foreground, which deferred
            # cortex recovery here — the primary could never come back while
            # its own fallback was answering for it. The primary lane's
            # recovery is exempt; other background lanes still yield.
            logger.info(
                "⏸️ [MLX] Deferring background worker activity for %s while foreground lane is owned by %s.",
                os.path.basename(self.model_path),
                _FOREGROUND_OWNER_NAME or "foreground",
            )
            return False
        if request_is_background and not self._is_primary_lane():
            # Every reason the gate's quiet policy returns (foreground_
            # reserved, headroom, cortex_startup_quiet, quiet window)
            # protects the user's turn from BACKGROUND COMPETITION. The
            # primary lane's own revival is not competition — it is the
            # thing the user's turn is waiting for, so it is exempt here
            # exactly as at the owner guard above.
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
            self._clear_model_load_admission_backoff()
            self._check_lane_state_staleness()  # [STABILITY v51]
            recurrent_depth_status = _normalize_recurrent_depth_status(
                self._recurrent_depth_status,
                model_path=self.model_path,
            )
            recurrent_depth_blocker = _recurrent_depth_readiness_blocker(recurrent_depth_status)
            if recurrent_depth_blocker and not request_is_background:
                self._set_lane_state("recovering", recurrent_depth_blocker)
                self._record_degraded_event(
                    recurrent_depth_blocker,
                    detail=f"{os.path.basename(self.model_path)}:{recurrent_depth_status}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                return False
            self._set_lane_state("ready")
            return True

        # Slow path: admission owns whether model loading may proceed; the
        # spawn gate remains the mechanical single-spawn mutex beneath it.
        if request_is_background and self._model_load_admission_backoff_active():
            return False
        if int(self._model_lane_fencing_token or 0) > 0:
            try:
                await self._release_durable_model_lane_owner(
                    reason="dead_worker_before_respawn",
                )
            except (
                ImportError,
                OSError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                self._set_lane_state("recovering", "stale_model_lane_owner_release_failed")
                _record_mlx_degradation(
                    exc,
                    action=(
                        "refused worker respawn until the dead worker's durable "
                        "model-lane owner can be released"
                    ),
                    severity="critical",
                )
                return False
        # CP126 1effd581. The anti-thrash swap cooldown used to sleep up to
        # twelve seconds INSIDE the model-load admission context and the
        # global single-spawn gate. Everything else in the process that wanted
        # to spawn waited behind a lane that was doing nothing but counting.
        # The wait is the same length; it just happens out here, where it
        # blocks only its own caller, and it now stops early for shutdown.
        await self._await_swap_cooldown(
            foreground_request=foreground_request,
            skip_swap_cooldown=skip_swap_cooldown,
        )
        try:
            async with _model_load_admission_context(
                self,
                foreground_request=foreground_request,
            ):
                async with _spawn_gate_context(
                    owner=f"{os.path.basename(self.model_path)}:"
                    f"{'foreground' if foreground_request else 'background'}",
                    # NOT scoped to init_timeout. Bounding the gate wait by
                    # the caller's budget is the right idea and the context
                    # manager supports it, but wiring it here moved other
                    # paths onto the timeout branch, and one of those leaves
                    # the durable model-lane owner unreconciled (lane FENCED,
                    # admission blocked) — the lease-outlives-holder shape in
                    # a new costume. The deferred-lane turn budget in
                    # interface/routes/chat.py addresses the dominant cause
                    # without that risk; re-scoping this wait needs the
                    # durable-owner path made timeout-safe first.
                ):
                    return await self._ensure_worker_alive_inner(
                        request_is_background=request_is_background,
                        foreground_request=foreground_request,
                        init_timeout=init_timeout,
                        soft_timeout=soft_timeout,
                        skip_swap_cooldown=skip_swap_cooldown,
                    )
        except _ModelLoadAdmissionDeniedError as admission_exc:
            # The inner spawn path can establish a specific terminal failure
            # (for example, a failed Metal runtime probe) before the durable
            # transaction observes that no candidate reached READY.  Preserve
            # that causal state instead of replacing it with the less useful
            # outer transaction consequence.
            if self._lane_state != "failed" or not str(self._lane_error or ""):
                self._set_lane_state("recovering", admission_exc.reason)
            backoff_s = self._note_model_load_admission_denial(
                admission_exc.reason,
                receipt_id=admission_exc.receipt_id,
            )
            if foreground_request:
                self._record_degraded_event(
                    "model_load_admission_denied",
                    detail=(
                        f"{os.path.basename(self.model_path)}:{admission_exc.reason}:"
                        f"receipt={admission_exc.receipt_id or 'none'}"
                    ),
                    severity="warning",
                    foreground_request=True,
                )
            admission_logger = logger.warning if foreground_request else logger.info
            admission_logger(
                "⏸️ [MLX] Model-load admission deferred for %s: %s (receipt=%s, recheck_in=%.1fs)",
                os.path.basename(self.model_path),
                admission_exc.reason,
                admission_exc.receipt_id or "none",
                backoff_s,
            )
            return False
        except TimeoutError as gate_exc:
            # Another lane's spawn is wedged holding the global gate. Defer
            # honestly instead of joining the pileup — the warmup's finally
            # still clears its flag, admission stays unblocked, and the
            # watchdog handles the wedged holder.
            self._set_lane_state("recovering", "spawn_gate_timeout")
            self._record_degraded_event(
                "spawn_gate_timeout",
                detail=f"{os.path.basename(self.model_path)}:{gate_exc}",
                severity="warning",
                foreground_request=foreground_request,
            )
            logger.warning(
                "⏸️ [MLX] Spawn gate held too long by another lane; deferring %s spawn (%s).",
                os.path.basename(self.model_path),
                gate_exc,
            )
            return False

    async def _ensure_worker_alive_inner_part_1(
        self,
        listener_alive: bool,
        silence: Any,
        stale_after: Any,
    ) -> Any:
        from .mlx_client import (
            _cancel_shared_future,
            _record_mlx_degradation,
            logger,
        )

        _record_mlx_degradation(
            TimeoutError(
                f"worker for {os.path.basename(self.model_path)} passed the "
                f"alive+init check but has been silent {silence:.1f}s "
                f"(limit {stale_after:.1f}s, listener_alive={listener_alive})"
            ),
            action="recycled a worker that looked ready and was not responding",
            severity="error",
        )
        logger.warning(
            "♻️ [MLX] %s is alive and initialised but silent for %.1fs; "
            "recycling instead of admitting it as ready.",
            os.path.basename(self.model_path),
            silence,
        )
        # Torn down INLINE, not via reboot_worker: this runs while
        # holding the lifecycle lock, and reboot_worker acquires it.
        # Calling it here would block for its whole escalation ladder
        # and then perform an unsynchronised reboot — turning a
        # recovery into the wedge it was recovering from. The
        # stale-handshake branch below does the same thing for the
        # same reason.
        self._set_lane_state("recovering", "ready_check_worker_silent")
        self._init_done = False
        if self._init_future is not None:
            _cancel_shared_future(self._init_future)
            self._init_future = None
        _doomed, self._process = self._process, None
        await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self._release_worker_process,
                _doomed,
                reason="ready_check_worker_silent",
            ),
        )
        self._reset_worker_scoped_state()
        self._replace_ipc_queues()
        return _doomed

    async def _ensure_worker_alive_inner_part_2(
        self,
        handshake_age: Any,
        handshake_budget: Any,
    ) -> Any:
        from .mlx_client import (
            _record_mlx_degradation,
            logger,
        )

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
        except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
            _record_mlx_degradation(
                _exc,
                action="recycled stale handshake despite init-future notification failure",
            )
            logger.debug("Suppressed stale-handshake future-set: %s", _exc)
        self._init_future = None
        _doomed, self._process = self._process, None
        await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self._release_worker_process,
                _doomed,
                reason="stale_handshake",
            ),
        )
        self._init_done = False
        self._last_heartbeat = 0.0
        self._last_progress_at = 0.0
        self._drain_queue()
        self._replace_ipc_queues()
        return _doomed

    async def _ensure_worker_alive_inner_part_3(self) -> Any:
        from .mlx_client import (
            _new_shared_future,
            logger,
        )

        logger.warning(
            "♻️ [MLX] Worker alive but init lifecycle is missing. Recycling %s.",
            os.path.basename(self.model_path),
        )
        self._set_lane_state("recovering", "missing_init_lifecycle")
        _doomed, self._process = self._process, None
        await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self._release_worker_process,
                _doomed,
                reason="missing_init_lifecycle",
            ),
        )
        self._init_done = False
        self._last_heartbeat = 0.0
        self._last_progress_at = 0.0
        self._drain_queue()

        # Prevent zombie threads from stealing messages
        self._replace_ipc_queues()

        init_future = _new_shared_future()
        self._init_future = init_future
        self._set_lane_state("spawning")
        logger.info(
            "📡 [MLX] Respawning worker for %s...", os.path.basename(self.model_path)
        )
        return init_future

    def _ensure_worker_alive_inner__sf(
        self,
        detail: str,
        exc: Any,
        foreground_request: bool,
    ) -> None:
        from .mlx_client import (
            _record_mlx_degradation,
            logger,
        )

        _sf = getattr(self, "_consecutive_spawn_failures", 0) + 1
        self._consecutive_spawn_failures = _sf
        self._spawn_backoff_until = time.time() + min(
            300.0, 10.0 * (2 ** min(_sf - 1, 5))
        )
        # CP126 ee4ccfcc: the backoff carried no cause, so the
        # runtime-availability probe cleared every one of them.
        # A healthy `import mlx` says nothing about an OOM, a
        # corrupt checkpoint or a refused memory admission.
        self._spawn_backoff_cause = (
            "runtime_unavailable"
            if "mlx_runtime_probe_failed:" in detail
            else "spawn_failure"
        )
        if "mlx_runtime_probe_failed:" in detail:
            self._mark_runtime_unavailable(
                detail.split("mlx_runtime_probe_failed:", 1)[1]
            )
        else:
            self._set_lane_state("failed", detail)
        _record_mlx_degradation(
            exc,
            action="marked lane failed or runtime unavailable and applied spawn backoff",
            severity="error",
        )
        self._record_degraded_event(
            "spawn_failed",
            detail=f"{os.path.basename(self.model_path)}:{detail}",
            severity="error",
            foreground_request=foreground_request,
        )
        logger.error(
            "🛑 [MLX] Worker respawn aborted for %s: %s (backoff %.0fs)",
            os.path.basename(self.model_path),
            detail,
            min(300.0, 10.0 * (2 ** min(_sf - 1, 5))),
        )
        self._init_future = None

    def _ensure_worker_alive_inner_bug_fix_exponential(
        self,
        _spawn_fails: Any,
        detail: str,
        exc: Any,
        foreground_request: bool,
    ) -> None:
        from .mlx_client import (
            _record_mlx_degradation,
            logger,
        )

        # [BUG FIX] Exponential backoff: 10s, 30s, 60s, 120s, 300s
        self._consecutive_spawn_failures = _spawn_fails + 1
        backoff = min(300.0, 10.0 * (2 ** min(_spawn_fails, 5)))
        self._spawn_backoff_until = time.time() + backoff
        # See CP126 ee4ccfcc: a runtime probe may only clear the
        # backoffs a runtime failure caused.
        self._spawn_backoff_cause = (
            "runtime_unavailable"
            if "mlx_runtime_probe_failed:" in detail
            else "spawn_failure"
        )
        if "mlx_runtime_probe_failed:" in detail:
            self._mark_runtime_unavailable(
                detail.split("mlx_runtime_probe_failed:", 1)[1]
            )
        else:
            self._set_lane_state("failed", detail)
        _record_mlx_degradation(
            exc,
            action="marked lane failed or runtime unavailable and applied spawn backoff",
            severity="error",
        )
        self._record_degraded_event(
            "spawn_failed",
            detail=f"{os.path.basename(self.model_path)}:{detail}",
            severity="error",
            foreground_request=foreground_request,
        )
        logger.error(
            "🛑 [MLX] Worker spawn aborted for %s: %s (attempt %d, backoff %.0fs)",
            os.path.basename(self.model_path),
            detail,
            self._consecutive_spawn_failures,
            backoff,
        )
        self._init_future = None

    def _ensure_worker_alive_inner_readiness_earned_announced(self, res: Any) -> tuple[Any, Any]:
        from .mlx_client import (
            _observe_worker_token_budget_calibration,
            _record_mlx_degradation,
        )

        # READINESS IS EARNED, NOT ANNOUNCED. CP126 34c42774:
        # any dict with status=ok used to set init_done,
        # heartbeats and lane=ready, and only THEN copy the
        # recurrence receipt and worker identity. A worker
        # that never reported recurrence, or reported a
        # malformed identity, was already serving by the time
        # anyone looked. The invariants are checked first and
        # the handshake fails if they do not hold — which
        # feeds the existing one-shot retry.
        readiness_errors = self._init_receipt_errors(res)
        attested_worker_identity: dict[str, Any] = {}
        raw_worker_identity = res.get("worker_identity")
        if not readiness_errors and isinstance(raw_worker_identity, Mapping):
            try:
                attested_worker_identity = (
                    self._attest_worker_capture_origin(raw_worker_identity)
                )
            except (
                ImportError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as capture_origin_exc:
                _record_mlx_degradation(
                    capture_origin_exc,
                    action=(
                        "refused READY because the worker capture key was not "
                        "bound to this parent spawn"
                    ),
                    severity="error",
                )
                readiness_errors.append(
                    "worker_capture_launch_attestation_invalid"
                )
        if not readiness_errors:
            from core.brain.llm.token_budget_evidence import MIN_OBSERVATIONS

            calibration_count = _observe_worker_token_budget_calibration(res)
            if calibration_count < MIN_OBSERVATIONS:
                readiness_errors.append(
                    "token_budget_calibration_not_admitted:"
                    f"{calibration_count}/{MIN_OBSERVATIONS}"
                )
        return attested_worker_identity, readiness_errors

    async def _ensure_worker_alive_inner_part_7(self, readiness_errors: Any) -> None:
        from .mlx_client import (
            _record_mlx_degradation,
        )

        _record_mlx_degradation(
            ValueError("init_receipt_invalid:" + ",".join(readiness_errors)),
            action="refused READY on an unvalidated worker init receipt",
            severity="error",
        )
        self._init_done = False
        self._worker_identity = {}
        self._recurrent_depth_status = {}
        # Every field the receipt was supposed to establish
        # is cleared together. Leaving one behind lets the
        # PREVIOUS worker's claim certify this one.
        self._recurrent_adapter_activation = {}
        self._unified_recurrent_shadow_status = {}
        self._unified_recurrent_shadow_probe_status = {}
        self._unified_recurrent_shadow_canary_status = {}
        self._unified_recurrent_qualified_activation_status = {}
        self._set_lane_state(
            "failed",
            "init_receipt_invalid",
        )
        # This is terminal evidence from this exact
        # worker. Re-reading the same completed future
        # cannot repair it and used to leave an alive,
        # permanently handshaking process behind. Retire
        # the untrusted generation and perform at most one
        # real spawn retry.
        await self.reboot_worker(
            reason="init_receipt_invalid",
            mark_failed=False,
        )

    def _ensure_worker_alive_inner_part_8(self, attested_worker_identity: Any, res: Any) -> Any:
        from .mlx_client import (
            _record_mlx_degradation,
        )

        self._init_done = True
        self._last_heartbeat = time.time()
        self._last_ready_at = self._last_heartbeat
        self._mark_progress()
        self._set_lane_state("ready")
        recurrent_status = res.get("recurrent_depth")
        # Always REPLACE: preserving the previous worker's
        # status when the new receipt is absent/malformed let
        # an old active=true certify a new worker that never
        # reported recurrence.
        self._recurrent_depth_status = (
            recurrent_status if isinstance(recurrent_status, dict) else {}
        )
        adapter_activation = res.get("recurrent_adapter_activation")
        self._recurrent_adapter_activation = (
            adapter_activation
            if isinstance(adapter_activation, dict)
            else {}
        )
        shadow_status = res.get("unified_recurrent_shadow")
        self._unified_recurrent_shadow_status = (
            copy.deepcopy(shadow_status)
            if isinstance(shadow_status, dict)
            else {}
        )
        self._unified_recurrent_shadow_probe_status = {}
        self._unified_recurrent_shadow_canary_status = {}
        qualified_status = res.get(
            "unified_recurrent_qualified_activation"
        )
        self._unified_recurrent_qualified_activation_status = (
            copy.deepcopy(qualified_status)
            if isinstance(qualified_status, dict)
            else {}
        )
        if not isinstance(recurrent_status, dict):
            _record_mlx_degradation(
                ValueError("missing_recurrent_depth_receipt"),
                action="cleared stale recurrence status after init receipt omitted it",
            )
        self._worker_identity = attested_worker_identity
        try:
            self._attest_mycelial_worker(res)
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as root_exc:
            _record_mlx_degradation(
                root_exc,
                action=(
                    "kept validated worker ready while Mycelium "
                    "root attestation failed"
                ),
                severity="warning",
            )
        self._steering_disposition = str(
            res.get("steering_disposition") or ""
        )
        raw_steering = res.get("steering_active")
        return raw_steering

    async def _ensure_worker_alive_inner(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
        _init_retry: bool = False,
    ) -> bool:
        """Inner implementation — called while holding the global spawn gate."""
        from .mlx_client import (
            SharedFuture,
            _await_shared_future,
            _background_deferral_active,
            _cancel_task_threadsafe,
            _crash_loop_blocks_worker_spawn,
            _foreground_owner_active,
            _new_shared_future,
            _note_lane_worker_death,
            _real_model_path,
            _record_heavy_model_swap,
            _record_mlx_degradation,
            _shutdown_blocks_model_work,
            logger,
        )

        if _shutdown_blocks_model_work(self.model_path, action="worker spawn"):
            return False
        # K4 crash-loop backoff: a lane whose workers keep dying young must
        # not respawn on demand. Refuse fast with a named reason — the
        # escalation ladder answers while the backoff drains. A healthy
        # worker passing through is never disturbed.
        if not (self._process and self._process.is_alive() and self._init_done):
            crash_blocked = _crash_loop_blocks_worker_spawn(self)
            if crash_blocked:
                self._set_lane_state("recovering", crash_blocked)
                self._record_degraded_event(
                    "crash_loop_backoff",
                    detail=f"{os.path.basename(self.model_path)}:{crash_blocked}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "⛔ [MLX] Respawn refused for %s: %s",
                    os.path.basename(self.model_path),
                    crash_blocked,
                )
                return False
        should_wait_init = False
        init_future: SharedFuture | None = None

        # [PIPELINE HARDENING] 12s Swap Cooldown
        from .model_registry import ACTIVE_MODEL, get_deep_model_path, get_model_path

        primary_path = _real_model_path(get_model_path(ACTIVE_MODEL))
        deep_path = _real_model_path(get_deep_model_path())
        target_path = _real_model_path(self.model_path)


        if request_is_background and _foreground_owner_active() and not self._is_primary_lane():
            # Primary-lane exemption (2026-07-10 inversion family): the
            # reconciler's prewarm arrives here as background work; blocking
            # it while the Reflex fallback owns the foreground kept the
            # cortex dead exactly while users waited on it.
            logger.info(
                "⏸️ [MLX] Background spawn blocked for %s while foreground lane is reserved.",
                os.path.basename(self.model_path),
            )
            return False
        if request_is_background and not self._is_primary_lane():
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background spawn blocked for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False

        # The swap cooldown is served by _await_swap_cooldown BEFORE the
        # admission context and the global spawn gate are taken (CP126
        # 1effd581), so nothing sleeps while holding them.

        acquired = await asyncio.to_thread(self._lock.acquire, True, 15.0)
        if not acquired:
            logger.error(
                "🚨 [MLX] DEADLOCK DETECTED: Could not acquire _lock within 15s for %s",
                os.path.basename(self.model_path),
            )
            return False
        try:
            # A forced abort may have killed the worker without owning this
            # lock. Finish its reconciliation before deciding lane health, or
            # a dead process reads as "already healthy".
            self._apply_pending_force_abort_reconcile()
            if self._process and self._process.is_alive() and self._init_done:
                # CP126 6165be63. "The process exists and once finished its
                # handshake" is not the same as "this lane can serve a turn".
                # A wedged worker satisfies both and was admitted as healthy,
                # so the first user request paid the whole first-token budget
                # or the hard cap before anything noticed. Check that it has
                # spoken recently, and that something is listening.
                silence = self._liveness_quiet_for_s()
                stale_after = self._stale_after()
                listener = self._listener_task
                listener_alive = listener is None or not listener.done()
                if silence <= stale_after and listener_alive:
                    self._set_lane_state("ready")
                    return True  # Already healthy, release gate
                _doomed = await self._ensure_worker_alive_inner_part_1(listener_alive, silence, stale_after)

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
                handshake_age = self._handshake_age_s()
                handshake_budget = max(60.0, 2.0 * self._handshake_timeout())
                if (
                    self._init_future is not None
                    and self._lane_state == "handshaking"
                    and handshake_age > handshake_budget
                ):
                    _doomed = await self._ensure_worker_alive_inner_part_2(handshake_age, handshake_budget)
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
                    init_future = await self._ensure_worker_alive_inner_part_3()
                    try:
                        self._process = await self._spawn_worker()
                        self._process_started_at = time.time()
                        self._consecutive_spawn_failures = 0
                        self._spawn_backoff_until = 0.0
                        self._spawn_backoff_cause = ""
                    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                        detail = str(exc)
                        if self._handle_optional_deep_solver_memory_refusal(exc):
                            return False
                        self._ensure_worker_alive_inner__sf(detail, exc, foreground_request)
                        return False
                    if self._listener_task:
                        _cancel_task_threadsafe(self._listener_task)
                    await self._ensure_listener_task()
                    self._set_lane_state("handshaking")
                    should_wait_init = True
            elif not self._process or not self._process.is_alive():
                if self._process is not None:
                    # The worker died on its own (OS OOM kill, segfault): no
                    # kill path saw it, so account for it here — then drop
                    # the dead handle so the death is counted exactly once.
                    _note_lane_worker_death(self, "process_died_unexpectedly")
                    self._process = None
                    self._process_started_at = 0.0
                # [BUG FIX] Exponential backoff on repeated spawn failures.
                # Without this, [Errno 5] I/O errors cause a tight 2-3s retry
                # loop that leaks FDs and shared memory for hours.
                _spawn_fails = getattr(self, "_consecutive_spawn_failures", 0)
                _spawn_backoff_until = getattr(self, "_spawn_backoff_until", 0.0)
                if time.time() < _spawn_backoff_until:
                    if not await asyncio.to_thread(
                        self.refresh_runtime_availability, force_probe=True
                    ):
                        return False  # Still in backoff window

                self._drain_queue()

                # Prevent zombie threads from stealing messages
                self._replace_ipc_queues()

                init_future = _new_shared_future()
                self._init_future = init_future
                self._set_lane_state("spawning")
                # A new worker is cold, whatever the last one had done.
                self._tokens_since_spawn = 0
                logger.info("📡 [MLX] Spawning worker for %s...", os.path.basename(self.model_path))
                try:
                    self._process = await self._spawn_worker()
                    self._process_started_at = time.time()
                    self._consecutive_spawn_failures = 0  # Reset on success
                    self._spawn_backoff_until = 0.0
                    self._spawn_backoff_cause = ""
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    # OSError included: queue creation, lock files, and
                    # multiprocessing start raise it — it previously escaped
                    # with the lane stuck in "spawning" and a pending init.
                    detail = str(exc)
                    if self._handle_optional_deep_solver_memory_refusal(exc):
                        return False
                    self._ensure_worker_alive_inner_bug_fix_exponential(_spawn_fails, detail, exc, foreground_request)
                    return False
                if self._listener_task:
                    _cancel_task_threadsafe(self._listener_task)
                await self._ensure_listener_task()
                should_wait_init = True
                self._init_done = False
                self._worker_identity = {}
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
                    if soft_timeout:
                        # The caller's own budget: a wall clock by nature,
                        # and it keeps the worker alive when it runs out.
                        res = await _await_shared_future(fut, timeout_s=handshake_timeout)
                    else:
                        res = await self._await_init_while_the_worker_loads(
                            fut, stall_s=handshake_timeout
                        )
                    if res.get("status") == "ok":
                        attested_worker_identity, readiness_errors = self._ensure_worker_alive_inner_readiness_earned_announced(res)
                        if readiness_errors:
                            await self._ensure_worker_alive_inner_part_7(readiness_errors)
                            if not _init_retry:
                                return await self._ensure_worker_alive_inner(
                                    request_is_background=request_is_background,
                                    foreground_request=foreground_request,
                                    init_timeout=init_timeout,
                                    soft_timeout=soft_timeout,
                                    skip_swap_cooldown=True,
                                    _init_retry=True,
                                )
                            return False
                        raw_steering = self._ensure_worker_alive_inner_part_8(attested_worker_identity, res)
                        if raw_steering is not None:
                            try:
                                if isinstance(raw_steering, bool):
                                    steering_active = raw_steering
                                else:
                                    # A malformed string "false" is truthy —
                                    # never bool() an untyped IPC value into
                                    # the shared steering channels.
                                    _record_mlx_degradation(
                                        TypeError(f"non-bool steering receipt: {raw_steering!r}"),
                                        action="treated malformed steering receipt as inactive",
                                    )
                                    steering_active = False
                                self._steering_active.value = steering_active
                                self._substrate_mem[-1] = 1.0 if steering_active else 0.0
                                self._steering_liveness_observed = True
                            except (
                                TypeError,
                                ValueError,
                                IndexError,
                                AttributeError,
                            ) as steering_receipt_exc:
                                _record_mlx_degradation(
                                    steering_receipt_exc,
                                    action="kept worker ready after steering liveness receipt write failed",
                                    severity="warning",
                                )
                        if target_path in (primary_path, deep_path):
                            _record_heavy_model_swap(target_path)
                        logger.info("✅ [MLX] Worker ready: %s", os.path.basename(self.model_path))
                        return True
                    else:
                        msg = res.get("message", "Init failed")
                        if handshake_attempt == 0:
                            logger.warning(
                                "🔄 [MLX] Worker init failed: %s. Retrying spawn...", msg
                            )
                            # Reboot and try again once
                            await self.reboot_worker(reason="init_failed_retry", mark_failed=False)
                            # Update fut for the new spawn
                            fut = self._init_future
                            if not fut:
                                # Reboot tears the lifecycle down WITHOUT
                                # spawning a replacement, so the falsy check
                                # silently skipped the advertised one-shot
                                # retry. Re-enter the spawn path once (the
                                # spawn gate is already held by our caller).
                                if not _init_retry:
                                    return await self._ensure_worker_alive_inner(
                                        request_is_background=request_is_background,
                                        foreground_request=foreground_request,
                                        init_timeout=init_timeout,
                                        soft_timeout=soft_timeout,
                                        skip_swap_cooldown=True,
                                        _init_retry=True,
                                    )
                                break
                            continue
                        self._set_lane_state("failed", msg)
                        raise RuntimeError(msg)
                except TimeoutError:
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
                        if not fut:
                            # CP126 0c91528f (timeout half). reboot_worker is a
                            # TEARDOWN: it clears _init_future and does NOT
                            # spawn a replacement, so this falsy check used to
                            # `break` and silently skip the advertised one-shot
                            # retry — the same defect already closed on the
                            # init-error branch above. Re-enter the spawn
                            # transaction so the retry actually happens.
                            if not _init_retry:
                                return await self._ensure_worker_alive_inner(
                                    request_is_background=request_is_background,
                                    foreground_request=foreground_request,
                                    init_timeout=init_timeout,
                                    soft_timeout=soft_timeout,
                                    skip_swap_cooldown=True,
                                    _init_retry=True,
                                )
                            break
                        continue
                    logger.error("🛑 [MLX] Init handshake TIMED OUT. Force killing process.")
                    self._set_lane_state("failed", "init_timeout")
                    if self._process:
                        _doomed, self._process = self._process, None
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            functools.partial(
                                self._release_worker_process,
                                _doomed,
                                reason="init_handshake_timeout",
                            ),
                        )
                    self._init_future = None
                    raise
            return False
        return self._process is not None and self._process.is_alive() and self._init_done

    async def reboot_worker(self, reason: str='manual_reboot', mark_failed: bool=False) -> None:
        """Forcibly reboots the worker.

        LOCK DISCIPLINE (CP126 ec341dfa). This used to log "forcing reboot
        anyway" and then kill the process, replace queues, cancel futures and
        reset ownership while the actual lock holder was still operating —
        converting a SUSPECTED deadlock into GUARANTEED unsynchronized
        corruption. Contention is now waited out (a real lifecycle op is
        bounded by its own timeouts) and the destructive path is a deliberate,
        receipted last resort after repeated failures to acquire, not the
        first response to 10 seconds of contention.
        """
        from .mlx_client import (
            _REBOOT_LOCK_ESCALATED_WAIT_S,
            _REBOOT_LOCK_FORCE_AFTER,
            _cancel_shared_future,
            _cancel_task_threadsafe,
            _clear_matching_foreground_owner,
            _note_lane_worker_death,
            _record_mlx_degradation,
            logger,
        )

        self._set_lane_state("recovering", reason)
        # A new generation begins here, so the next owner id cannot be confused
        # with the one being torn down.
        self._worker_generation = int(getattr(self, "_worker_generation", 0) or 0) + 1
        self._model_lane_owner_id = ""
        acquired = await asyncio.to_thread(self._lock.acquire, True, 10.0)
        if not acquired:
            # Escalate the wait before considering anything unsynchronized.
            acquired = await asyncio.to_thread(
                self._lock.acquire, True, _REBOOT_LOCK_ESCALATED_WAIT_S
            )
        forced_unsynchronized = False
        if not acquired:
            self._reboot_lock_failures += 1
            forced_unsynchronized = self._reboot_lock_failures >= _REBOOT_LOCK_FORCE_AFTER
            if not forced_unsynchronized:
                _record_mlx_degradation(
                    TimeoutError(f"reboot_lock_unavailable:{reason}"),
                    action=(
                        "deferred reboot instead of mutating worker lifecycle state "
                        "without the lifecycle lock"
                    ),
                    severity="error",
                )
                logger.error(
                    "🚨 [MLX] Could not acquire _lock for reboot on %s after %.0fs "
                    "(attempt %d/%d). DEFERRING — another lifecycle operation owns "
                    "this lane.",
                    os.path.basename(self.model_path),
                    10.0 + _REBOOT_LOCK_ESCALATED_WAIT_S,
                    self._reboot_lock_failures,
                    _REBOOT_LOCK_FORCE_AFTER,
                )
                self._set_lane_state("recovering", f"reboot_deferred_lock:{reason}")
                return
            _record_mlx_degradation(
                TimeoutError(f"reboot_lock_wedged:{reason}"),
                action=(
                    "forced an unsynchronized reboot after repeated lock-acquisition "
                    "failures — the lock holder is presumed wedged"
                ),
                severity="critical",
            )
            logger.critical(
                "🚨 [MLX] Lock holder for %s presumed WEDGED after %d failed reboot "
                "acquisitions. Forcing unsynchronized reboot as a last resort.",
                os.path.basename(self.model_path),
                self._reboot_lock_failures,
            )
        else:
            self._reboot_lock_failures = 0
        try:
            # A forced abort that could not take this lock left its
            # reconciliation for whoever did. Clear it first so the reboot
            # below is not racing a half-torn-down lane.
            self._force_abort_reconcile_pending = None
            self._force_abort_lock_failures = 0
            self._unbind_mycelial_worker()
            process = self._process
            if process is not None:
                # K4 accounting: the breaker classifies this death by reason
                # (deliberate yields never count; young crashes do).
                _note_lane_worker_death(self, reason)
                termination_proven = await asyncio.to_thread(
                    self._kill_and_join_blocking,
                    process,
                    cooperative=True,
                )
                if not termination_proven:
                    self._set_lane_state(
                        "recovering",
                        f"reboot_worker_termination_unproven:{reason}",
                    )
                    raise RuntimeError(
                        "mlx_worker_termination_unproven_before_reboot:"
                        f"pid={getattr(process, 'pid', 'unknown')}"
                    )
            self._process = None
            self._init_done = False
            self._expert_adapter_path = None  # adapters live in the worker process
            self._last_heartbeat = 0.0
            self._last_progress_at = 0.0
            self._last_token_progress_at = 0.0
            self._last_worker_job_activity_at = 0.0
            self._worker_progress_channel = None
            # Reset the cold-start anchor so the next foreground request
            # gets the generous 40 s SLA instead of the tight warm-path 22 s.
            # A reboot means the worker process is gone → first-token budget
            # includes Metal shader recompile, KV rebuild, and weight reload.
            self._last_generation_completed_at = 0.0
            self._last_user_facing_completed_at = 0.0
            self._last_visible_readiness_at = 0.0
            self._process_started_at = 0.0
            self._current_request_started_at = 0.0
            self._current_first_token_at = 0.0
            self._current_request_id = ""
            self._current_request_seq = 0
            # A reboot orphans any cooperative-cancel request with the worker.
            cancel_seq = getattr(self, "_cancel_seq", None)
            if cancel_seq is not None:
                try:
                    cancel_seq.value = 0
                except (OSError, ValueError):
                    logger.debug("Cancel channel reset skipped during reboot.")
            if self._listener_task:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None
            self._cancel_lane_renewal_task()

            # [OOM FIX] Force memory reclaim after killing heavy model process
            gc.collect()

            # RECREATE QUEUES TO PREVENT ZOMBIE THREADS STEALING MESSAGES
            self._replace_ipc_queues()

            pending_request_ids = [
                req_id
                for req_id, future in self._pending_generations.items()
                if future is not None and not future.done()
            ]
            if mark_failed:
                self._expected_cancels.clear()
            elif pending_request_ids:
                self._note_expected_generation_cancellation(
                    reason, request_ids=pending_request_ids
                )

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
            self._release_detached_request_lock()
            self._pending_generations.clear()
            self._clear_detached_worker_requests()
            self._current_gen_future = None
            self._active_generations = 0
            if self._init_future is not None:
                _cancel_shared_future(self._init_future)
            self._init_future = None
            self._warmup_in_flight = False
            self._consecutive_empty = (
                0  # [STABILITY v53] Reset on reboot — prevents false recovery triggers
            )
        finally:
            if acquired:
                self._lock.release()
        try:
            await self._release_durable_model_lane_owner(reason=reason)
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="worker stopped but durable model-lane owner release failed",
                severity="warning",
            )
        self._set_lane_state("failed" if mark_failed else "cold", reason if mark_failed else "")

