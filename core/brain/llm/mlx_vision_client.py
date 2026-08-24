from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.model_lane_control import (
    LaneClaim,
    LaneOwnerObservation,
    ModelLaneControlError,
    compensate_registered_model_owner,
    estimate_model_job_footprint_gb,
    prepare_model_lane_claim,
    process_identity_for_pid,
    register_model_lane_owner_adapter,
    unregister_model_lane_owner_adapter,
)
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.resource_observation import get_resource_observer
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.shutdown_execution import run_sync_shutdown_callable_blocking
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    get_subprocess_gateway,
)

from .mlx_vision_worker import _mlx_vision_worker_loop

logger = logging.getLogger("MLXVisionClient")
# Qwen3-VL-4B replaced Qwen2-VL-2B on 12 Aug 2026. This is the model that
# answers "read my screen", and Qwen2-VL-2B was two generations behind on
# exactly the axis that matters for it: DocVQA 95.3 vs the 2B-era scores,
# clearing every Gemma 3 size including the 27B. mlx-vlm 0.4.4 already ships
# the qwen3_vl backend, so this needed no new dependency.
#
# Weights are pinned to the local HF cache — see docs on offline operation in
# core/memory/embedding_model.py's sibling note. Nothing here reaches a
# hosted endpoint; MLX loads safetensors off this disk.
DEFAULT_VISION_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


class MLXVisionClient:
    """
    Manages an isolated MLX vision model worker for multimodal inference.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VISION_MODEL,
        *,
        lane_controller: Any | None = None,
    ) -> None:
        self.model_path = str(model_path or DEFAULT_VISION_MODEL)
        self._process: Any | None = None
        self._req_q: Any | None = None
        self._res_q: Any | None = None
        self._lock = threading.Lock()
        self._start_guard = threading.Lock()
        self._pending_lock = threading.RLock()
        self._pending_requests: dict[str, dict[str, Any] | None] = {}
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._init_done = False
        self._init_error = ""
        self._last_heartbeat_at = 0.0
        self._heartbeat_count = 0
        self._last_stop_reason = "not_started"
        self._lane_controller: Any | None = None
        self._lane_controller_override = lane_controller
        self._lane_decision: Any | None = None
        self._lane_owner_id = f"mlx-vision:{os.getpid()}:{uuid.uuid4()}"

    @staticmethod
    def _safe_close_queue(queue_obj: Any) -> None:
        if queue_obj is None:
            return

        def _close_and_join() -> None:
            close = getattr(queue_obj, "close", None)
            if callable(close):
                close()
            join_thread = getattr(queue_obj, "join_thread", None)
            if callable(join_thread):
                join_thread()

        try:
            run_sync_shutdown_callable_blocking(
                _close_and_join,
                timeout_s=1.0,
                name="mlx-vision-queue-close",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            logger.debug("Suppressed vision queue close error: %s", exc)
        else:
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().unregister_shutdown_resource(queue_obj)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                pass

    def _close_queues(self) -> None:
        self._safe_close_queue(self._req_q)
        self._safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None

    def _replace_queues(self, ctx: Any) -> None:
        self._close_queues()
        factory = ctx.Queue if hasattr(ctx, "Queue") else mp.Queue
        self._req_q = factory(maxsize=10)
        self._res_q = factory(maxsize=10)
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            hygiene = get_runtime_hygiene()
            hygiene.register_shutdown_resource(
                self._req_q,
                kind="multiprocessing_queue",
                name="mlx_vision.request_queue",
                source="core.brain.llm.mlx_vision_client",
                timeout_s=1.0,
            )
            hygiene.register_shutdown_resource(
                self._res_q,
                kind="multiprocessing_queue",
                name="mlx_vision.response_queue",
                source="core.brain.llm.mlx_vision_client",
                timeout_s=1.0,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            self._close_queues()
            raise

    def _spawn_worker_blocking(self) -> bool:
        if is_shutdown_requested():
            logger.info("Vision worker start refused: runtime shutdown requested")
            return False
        with self._lock:
            if is_shutdown_requested():
                return False
            if self._process is not None and self._process.is_alive():
                with self._pending_lock:
                    existing_ready = self._init_done and not self._init_error
                if existing_ready:
                    return True
                # A live process without a valid init acknowledgement is not
                # reusable. The start guard excludes a concurrent legitimate
                # initialization, so this is residue from a failed lifecycle.
                self.stop(reason="vision_worker_stale_uninitialized")

            logger.info("Starting MLX Vision Worker for %s", self.model_path)
            # Ensure spawn method for MLX Metal compatibility
            ctx: Any = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
            self._stop_event.clear()
            with self._pending_lock:
                self._pending_requests.clear()
                self._init_done = False
                self._init_error = ""
                self._last_heartbeat_at = 0.0
                self._heartbeat_count = 0
                self._last_stop_reason = ""
            self._replace_queues(ctx)

            try:
                process = get_subprocess_gateway().spawn_python_process(
                    PythonProcessSpec(
                        target=_mlx_vision_worker_loop,
                        args=(self.model_path, self._req_q, self._res_q),
                        source="mlx_vision_client.worker_owner",
                        name="MLX-Vision-Worker",
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
                        start_method="spawn",
                    ),
                    context=ctx,
                )
            except RuntimeError:
                if is_shutdown_requested():
                    self._process = None
                    self._close_queues()
                    return False
                raise
            self._process = process

            self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
            self._listener_thread.start()

            # Wait for init
            start_time = time.time()
            exit_seen_at = 0.0
            while time.time() - start_time < 30.0:
                if is_shutdown_requested():
                    self.stop()
                    return False
                with self._pending_lock:
                    init_done = self._init_done
                    init_error = self._init_error
                if init_error:
                    logger.error("Vision worker initialization failed: %s", init_error)
                    self.stop(reason="vision_worker_init_failed")
                    return False
                if self._process is None or not self._process.is_alive():
                    # Give the listener a bounded chance to consume the
                    # worker's specific init-error receipt after process exit.
                    # Without this grace, the generic exit observation races
                    # and overwrites the actionable model-load cause.
                    if not exit_seen_at:
                        exit_seen_at = time.monotonic()
                    if time.monotonic() - exit_seen_at < 0.5:
                        time.sleep(0.05)
                        continue
                    with self._pending_lock:
                        self._init_error = self._init_error or "worker_exited_before_init"
                    self.stop(reason="vision_worker_exited_before_init")
                    return False
                if init_done:
                    return True
                time.sleep(0.1)

            logger.error("Vision worker failed to initialize within 30s")
            with self._pending_lock:
                self._init_error = "initialization_timeout_30s"
            self.stop(reason="vision_worker_init_timeout")
            return False

    async def _evict_for_lane(
        self,
        _owner: LaneOwnerObservation,
        reason: str,
    ) -> bool:
        await self.stop_async(reason=f"lane_eviction:{reason}")
        return self._process is None

    async def _compensate_lane(
        self,
        _owner: LaneOwnerObservation,
        reason: str,
    ) -> bool:
        logger.warning("Restoring preempted vision lane after %s", reason)
        return await self.start_async()

    @contextlib.asynccontextmanager
    async def _start_context(self) -> AsyncIterator[None]:
        # Shared across event loops; polling a nonblocking thread lock avoids
        # an orphaned to_thread acquire after caller cancellation.
        while not self._start_guard.acquire(blocking=False):  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            self._start_guard.release()

    async def start_async(self) -> bool:
        if is_shutdown_requested():
            return False
        lane_controller = None
        lane_decision = None
        async with self._start_context():
            try:
                if self.readiness_status()["ready"]:
                    return True
                from core.brain.llm.model_registry import get_model_runtime_assignment

                claim = LaneClaim(
                    owner_id=self._lane_owner_id,
                    model_path=self.model_path,
                    request_gb=estimate_model_job_footprint_gb(
                        self.model_path,
                        purpose="serve",
                    ),
                    purpose="serve",
                    priority=60,
                    preemptible=True,
                    foreground=True,
                    request_id=f"vision-model-{uuid.uuid4()}",
                    runtime_assignment=get_model_runtime_assignment(self.model_path),
                    metadata={"owner": "MLXVisionClient", "modality": "vision"},
                )
                lane_controller, lane_decision = await prepare_model_lane_claim(
                    claim,
                    controller=self._lane_controller_override,
                )
                from core.utils.task_tracker import get_task_tracker

                spawn_task = get_task_tracker().create_task(
                    asyncio.to_thread(self._spawn_worker_blocking),
                    name=f"VisionModelSpawn:{Path(self.model_path).name}",
                )
                try:
                    ready = bool(await asyncio.shield(spawn_task))
                except asyncio.CancelledError:
                    try:
                        ready = bool(await asyncio.shield(spawn_task))
                    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                        ready = False
                    if ready or self._process is not None:
                        await asyncio.to_thread(self.stop)
                    await asyncio.shield(
                        lane_controller.cancel(
                            lane_decision,
                            reason="vision_model_start_cancelled",
                            compensate=compensate_registered_model_owner,
                        )
                    )
                    raise
                process = self._process
                pid = int(getattr(process, "pid", 0) or 0) if process is not None else 0
                if not ready or process is None or not process.is_alive() or pid <= 0:
                    cancelled = await lane_controller.cancel(
                        lane_decision,
                        reason="vision_model_worker_not_ready",
                        compensate=compensate_registered_model_owner,
                    )
                    logger.error(
                        "Vision worker admission cancelled: %s receipt=%s",
                        cancelled.reason,
                        cancelled.receipt_id,
                    )
                    return False
                try:
                    observed_process = get_resource_observer().process(pid)
                    observed_gb = (
                        float(observed_process.rss_bytes) / float(1024**3)
                        if observed_process is not None
                        else 0.0
                    )
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    observed_gb = 0.0
                try:
                    committed = await lane_controller.commit(
                        lane_decision,
                        process=process_identity_for_pid(pid),
                        observed_gb=observed_gb,
                        metadata={
                            "worker_name": str(getattr(process, "name", "")),
                            "modality": "vision",
                        },
                    )
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    await asyncio.to_thread(self.stop)
                    await lane_controller.cancel(
                        lane_decision,
                        reason=f"vision_model_commit_failed:{type(exc).__name__}",
                        compensate=compensate_registered_model_owner,
                    )
                    raise ModelLaneControlError("vision_model_lane_commit_failed") from exc
                self._lane_controller = lane_controller
                self._lane_decision = committed
                register_model_lane_owner_adapter(
                    committed.owner_id,
                    evict=self._evict_for_lane,
                    compensate=self._compensate_lane,
                )
                return True
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                committed_before_cleanup = self._lane_decision is not None
                if self._process is not None:
                    await asyncio.to_thread(
                        self.stop,
                        reason="vision_model_start_failed",
                    )
                if (
                    not committed_before_cleanup
                    and lane_controller is not None
                    and lane_decision is not None
                ):
                    try:
                        await lane_controller.cancel(
                            lane_decision,
                            reason="vision_model_start_failed",
                            compensate=compensate_registered_model_owner,
                        )
                    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                        logger.exception("Vision model reservation cancellation failed")
                raise

    def start(self) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.start_async())
        raise RuntimeError("MLXVisionClient.start cannot block an event loop; use start_async")

    def _handle_worker_message(self, msg: dict[str, Any]) -> bool:
        """Consume worker lifecycle messages before request routing.

        The worker deliberately reports initialization failure as
        ``action=init,status=error``. Treating action alone as success made a
        missing or broken model look ready for a short race window.
        """
        status = str(msg.get("status") or "").strip().lower()
        action = str(msg.get("action") or "").strip().lower()

        if status == "heartbeat":
            with self._pending_lock:
                self._last_heartbeat_at = time.time()
                self._heartbeat_count += 1
            controller = self._lane_controller
            decision = self._lane_decision
            if controller is not None and decision is not None:
                alive = controller.heartbeat_owner_sync(
                    decision.owner_id,
                    fencing_token=decision.fencing_token,
                )
                if not alive:
                    logger.error(
                        "Vision worker lost model-lane fence owner=%s token=%s",
                        decision.owner_id,
                        decision.fencing_token,
                    )
                    with self._pending_lock:
                        self._init_error = "model_lane_fence_lost"
                    self._stop_event.set()
                    process = self._process
                    if process is not None and process.is_alive():
                        process.terminate()
            return True

        if action == "init":
            with self._pending_lock:
                if status in {"ok", "success"}:
                    self._init_done = True
                    self._init_error = ""
                else:
                    self._init_done = False
                    self._init_error = str(
                        msg.get("message")
                        or f"invalid_init_status:{status or 'missing'}"
                    )[:500]
            return True
        return False

    def readiness_status(self) -> dict[str, Any]:
        """Return a bounded, non-starting view of the vision worker contract."""
        process = self._process
        try:
            process_alive = bool(process is not None and process.is_alive())
        except (AssertionError, OSError, RuntimeError, ValueError):
            process_alive = False
        listener = self._listener_thread
        listener_alive = bool(listener is not None and listener.is_alive())
        with self._pending_lock:
            init_done = bool(self._init_done)
            init_error = str(self._init_error or "")
            heartbeat_at = float(self._last_heartbeat_at or 0.0)
            heartbeat_count = int(self._heartbeat_count)
            pending = len(self._pending_requests)
            stop_reason = str(self._last_stop_reason or "")
        queues_ready = self._req_q is not None and self._res_q is not None
        worker_ready = bool(
            process_alive and listener_alive and queues_ready and init_done and not init_error
        )
        lane_committed = self._lane_decision is not None
        ready = bool(worker_ready and lane_committed)
        if ready:
            reason = "ready"
        elif init_error:
            reason = f"init_failed:{init_error}"
        elif not process_alive:
            reason = stop_reason or "not_started"
        elif not init_done:
            reason = "initializing"
        elif not listener_alive:
            reason = "listener_not_alive"
        elif not queues_ready:
            reason = "ipc_unavailable"
        else:
            reason = "model_lane_not_committed"
        return {
            "schema": "aura.mlx_vision.readiness.v1",
            "ready": ready,
            "worker_ready": worker_ready,
            "reason": reason,
            "model": self.model_path,
            "pid": int(getattr(process, "pid", 0) or 0) if process_alive else 0,
            "process_alive": process_alive,
            "listener_alive": listener_alive,
            "ipc_ready": queues_ready,
            "init_done": init_done,
            "init_error": init_error,
            "lane_committed": lane_committed,
            "heartbeat_count": heartbeat_count,
            "heartbeat_age_s": (
                round(max(0.0, time.time() - heartbeat_at), 3)
                if heartbeat_at > 0.0
                else None
            ),
            "pending_requests": pending,
        }

    def _listener_loop(self) -> None:
        while not self._stop_event.is_set() and not is_shutdown_requested():
            try:
                if self._res_q is None:
                    break
                msg: dict[str, Any] = self._res_q.get(timeout=1.0)
                if not isinstance(msg, dict):
                    logger.error("Vision worker returned non-mapping lifecycle message")
                    continue
                if self._handle_worker_message(msg):
                    continue

                req_id = msg.get("id")
                if req_id:
                    with self._pending_lock:
                        if req_id in self._pending_requests:
                            self._pending_requests[req_id] = msg
            except queue.Empty:
                process = self._process
                if process is not None and not process.is_alive():
                    with self._pending_lock:
                        if not self._init_done and not self._init_error:
                            self._init_error = "worker_exited_before_init"
                    return
                continue
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.error("Vision listener error: %s", e)

    def _see_started_blocking(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 512,
        temp: float = 0.0,
        timeout_s: float = 120.0,
    ) -> str:
        """
        Send a base64 image (with optional data:image prefix) and prompt to the vision model.
        Blocks until completion.
        """
        if self._req_q is None:
            raise RuntimeError("Vision worker queue unavailable after start")

        req_id = str(uuid.uuid4())
        with self._pending_lock:
            self._pending_requests[req_id] = None
        try:
            self._req_q.put(
                {
                    "id": req_id,
                    "action": "see",
                    "prompt": prompt,
                    "image_base64": image_base64,
                    "max_tokens": max_tokens,
                    "temp": temp,
                },
                timeout=2.0,
            )
        except queue.Full as exc:
            with self._pending_lock:
                self._pending_requests.pop(req_id, None)
            raise RuntimeError("Vision worker request queue is full") from exc

        # Wait for response (structurally bounded by the deadline).
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        response = None
        while time.monotonic() < deadline:
            with self._pending_lock:
                response = self._pending_requests.get(req_id)
            if response is not None:
                break
            if is_shutdown_requested():
                with self._pending_lock:
                    self._pending_requests.pop(req_id, None)
                self.stop()
                raise RuntimeError("runtime_shutdown")
            time.sleep(0.1)
            if self._process and not self._process.is_alive():
                with self._pending_lock:
                    self._pending_requests.pop(req_id, None)
                self.stop()
                raise RuntimeError("Vision worker crashed during inference")
        if response is None:
            with self._pending_lock:
                self._pending_requests.pop(req_id, None)
            self.stop()
            raise TimeoutError(f"Vision worker inference timed out after {float(timeout_s):.1f}s")

        with self._pending_lock:
            resp = self._pending_requests.pop(req_id)
        if resp is None:  # pragma: no cover - loop exits only after response assignment
            raise RuntimeError("Vision worker returned no response payload")
        # CP126 7031837e. Anything not carrying status=error was treated as
        # success and stringified — so a missing status, a null response, a
        # mapping, or any malformed IPC payload became confident vision
        # "output". A cross-process reply must satisfy a schema before it is
        # believed, because the failure mode is fabricated perception.
        if not isinstance(resp, dict):
            raise RuntimeError(
                f"Vision worker returned a non-mapping response: {type(resp).__name__}"
            )
        status = resp.get("status")
        if status == "error":
            raise RuntimeError(f"Vision model error: {resp.get('message')}")
        if status not in ("ok", "success", None):
            raise RuntimeError(f"Vision worker returned unknown status: {status!r}")
        payload = resp.get("response")
        if payload is None:
            raise RuntimeError("Vision worker response carried no 'response' field")
        if not isinstance(payload, str):
            raise RuntimeError(
                f"Vision worker response must be text, got {type(payload).__name__}"
            )
        return payload

    async def see_async(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 512,
        temp: float = 0.0,
        timeout_s: float = 120.0,
    ) -> str:
        if not await self.start_async():
            raise RuntimeError("Vision worker unavailable")
        readiness = self.readiness_status()
        if not readiness["ready"]:
            raise RuntimeError(f"Vision worker not ready: {readiness['reason']}")
        return await asyncio.to_thread(
            self._see_started_blocking,
            prompt,
            image_base64,
            max_tokens,
            temp,
            timeout_s,
        )

    def see(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 512,
        temp: float = 0.0,
        timeout_s: float = 120.0,
    ) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.see_async(
                    prompt,
                    image_base64,
                    max_tokens=max_tokens,
                    temp=temp,
                    timeout_s=timeout_s,
                )
            )
        raise RuntimeError("MLXVisionClient.see cannot block an event loop; use see_async")

    async def describe(
        self,
        image_path: str,
        *,
        prompt: str = "Describe this image precisely, including visible text and relevant details.",
        max_bytes: int = 25 * 1024 * 1024,
        timeout_s: float = 120.0,
    ) -> str:
        def _read_image() -> bytes:
            path = Path(image_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size <= 0 or size > int(max_bytes):
                raise ValueError(f"vision_image_size_out_of_bounds:{size}")
            return path.read_bytes()

        image_bytes = await asyncio.to_thread(_read_image)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return await self.see_async(
            prompt,
            encoded,
            timeout_s=timeout_s,
        )

    def stop(self, *, reason: str = "vision_worker_stopped") -> None:
        lane_controller, self._lane_controller = self._lane_controller, None
        lane_decision, self._lane_decision = self._lane_decision, None
        if lane_decision is not None:
            unregister_model_lane_owner_adapter(lane_decision.owner_id)
        self._stop_event.set()
        if self._req_q:
            try:
                self._req_q.put(None)
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                logger.debug(
                    "Suppressed %s in core.brain.llm.mlx_vision_client: %s",
                    type(_exc).__name__,
                    _exc,
                )
        process, self._process = self._process, None
        # A multiprocessing.Process that was created but never started has a
        # ``_popen`` of None, and join() *asserts* rather than returning. So
        # any failure during spawn turned every subsequent stop() into an
        # AssertionError, which then buried the original cause — the operator
        # sees "can only join a started process" and never learns why the
        # worker did not come up.
        #
        # ``hasattr`` before the value check, not ``getattr(..., None)``: a
        # test double has no ``_popen`` at all and is perfectly joinable, and
        # collapsing "absent" into "never started" skipped the very teardown
        # those doubles exist to observe.
        never_started = (
            process is not None
            and hasattr(process, "_popen")
            and process._popen is None
        )
        if never_started:
            close = getattr(process, "close", None)
            if callable(close):
                with contextlib.suppress(ValueError, OSError):
                    close()
            process = None
        if process:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        self._listener_thread = None
        with self._pending_lock:
            self._pending_requests.clear()
            self._init_done = False
            self._last_stop_reason = str(reason or "vision_worker_stopped")
        self._close_queues()
        if lane_controller is not None and lane_decision is not None:
            try:
                lane_controller.release_owner_sync(
                    lane_decision.owner_id,
                    fencing_token=lane_decision.fencing_token,
                    reason=reason,
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                logger.exception("Vision model-lane owner release failed")
        logger.info("Vision worker stopped.")

    async def stop_async(self, *, reason: str = "vision_worker_stopped") -> None:
        await asyncio.to_thread(self.stop, reason=reason)
        logger.debug("Vision worker async stop completed: %s", reason)

    close = stop
    cleanup = stop
    on_stop = stop

    def __del__(self) -> None:
        try:
            self.stop()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as _exc:
            logger.debug(
                "Suppressed %s in core.brain.llm.mlx_vision_client: %s", type(_exc).__name__, _exc
            )


# ── one worker, shared ────────────────────────────────────────────────────
#
# Constructing this class spawns a subprocess that loads a 1.2 GB model. Each
# call site that builds its own therefore holds its own copy — on a host whose
# resident 32B already wires ~20 GB, two or three of those is the difference
# between a working machine and a swapping one. The class stays constructible
# (tests and the adapters that pass an explicit model_path rely on it); this
# is the accessor for everything that just wants "the vision worker".

_SHARED_CLIENT: MLXVisionClient | None = None
# Checked rather than raw: lockdep only sees the locks it wraps, and this one
# is held across a worker stop() that can take seconds. REGISTRY because it
# guards a process-wide singleton and is taken before anything under it.
_SHARED_CLIENT_LOCK = checked_lock("mlx_vision.shared_client", rank=LockRank.REGISTRY)


def get_vision_client(model_path: str = DEFAULT_VISION_MODEL) -> MLXVisionClient:
    """The process-wide vision worker, started lazily on first sight."""
    global _SHARED_CLIENT
    with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is None or _SHARED_CLIENT.model_path != str(model_path):
            if _SHARED_CLIENT is not None:
                # A different model was asked for. Stop the old worker rather
                # than leaking it — two resident vision models is exactly the
                # cost this accessor exists to avoid.
                try:
                    _SHARED_CLIENT.stop(reason="vision_model_changed")
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                    logger.exception("Replacing vision worker: stop of the old one failed")
            _SHARED_CLIENT = MLXVisionClient(model_path)
        return _SHARED_CLIENT


def reset_vision_client_for_test() -> None:
    global _SHARED_CLIENT
    with _SHARED_CLIENT_LOCK:
        _SHARED_CLIENT = None
