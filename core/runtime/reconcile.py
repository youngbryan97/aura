"""core/runtime/reconcile.py — level-triggered reconciliation.

Clean-room adoption of Kubernetes controller-runtime: the work queue, the
rate limiter, the reconcile contract, owner references, and finalizers.

The idea worth stealing is not "a loop that fixes things". It is
**level-triggered, not edge-triggered**.

An edge-triggered handler responds to events: "a lane died, restart it".
It is correct exactly as long as no event is ever missed, delivered twice,
delivered out of order, or delivered while the process was restarting —
which is to say, it is correct in testing and wrong in production. Aura
has paid for this: a false-death event respawned a second 32B without the
first being reaped, and the duplicate-runtime cascade followed.

A level-triggered reconciler never trusts the event. It reads the current
observed state, compares it to the desired state, and takes whatever
single step moves one toward the other. The event is *only a hint that now
is a good time to look*. Missing an event costs latency, never
correctness; a duplicate event is a no-op; a restart loses nothing,
because the reconciler re-derives everything from observation. That is why
Kubernetes controllers survive their own crashes, and it is exactly the
property Aura's organ management needs.

Supporting machinery, all with the semantics that make the above work:

* **Deduplicating work queue.** Ten events for one key while it is being
  processed collapse to exactly one more reconcile — never ten.
* **Per-key exponential backoff.** A key that keeps failing backs off on
  its own without starving anything else. ``forget()`` on success is what
  makes the backoff recover rather than ratchet.
* **``Result.requeue_after``** lets a reconciler say "I made progress but
  I am not done; look again in 30s" without a sleep that pins a worker.
* **Generation / observed_generation** distinguishes "the desired state
  changed" from "I already handled this desired state" — the difference
  between converged and merely quiet.
* **Finalizers** guarantee cleanup runs before an object is really gone,
  even if the process died between the delete request and the cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Reconcile")

#: Rate-limiter shape. The kernel of the idea is a very small base delay
#: (so a transient blip retries almost immediately) and a large ceiling
#: (so a genuinely broken key does not spin forever).
BASE_DELAY_S = 0.1
MAX_DELAY_S = 300.0

#: Default cadence for the periodic resync. Level-triggered controllers
#: re-reconcile everything on a slow timer precisely because they do NOT
#: trust that they saw every event.
DEFAULT_RESYNC_S = 600.0

#: How long an idle consumer waits before waking to re-check whether it
#: should still be running. Every blocking wait in this module is bounded
#: by it: a consumer that can only be woken by a producer cannot notice
#: shutdown, and a lost wakeup becomes a permanent stall rather than one
#: extra poll. This is the pattern that would have caught the mind_tick
#: wedge hours earlier.
IDLE_POLL_S = 1.0


@dataclass(frozen=True)
class Request:
    """What to reconcile. Just a key — never the object itself.

    Passing the object would reintroduce edge-triggering through the back
    door: the reconciler would act on a snapshot from whenever the event
    fired instead of reading current state.
    """

    key: str
    reason: str = ""


@dataclass(frozen=True)
class Result:
    """What the reconciler wants to happen next."""

    requeue: bool = False
    requeue_after_s: float = 0.0

    @classmethod
    def done(cls) -> Result:
        """Converged. Nothing to do until something changes."""
        return cls()

    @classmethod
    def again(cls, after_s: float = 0.0) -> Result:
        """Progress made, not finished. Look again after the delay."""
        return cls(requeue=True, requeue_after_s=max(0.0, after_s))


@dataclass
class ObjectMeta:
    """The bookkeeping every reconciled object carries.

    ``generation`` increments when the *desired* state changes.
    ``observed_generation`` records the generation a reconciler has fully
    handled. When they differ, the object is not converged, no matter how
    healthy it looks.
    """

    name: str
    generation: int = 1
    observed_generation: int = 0
    finalizers: tuple[str, ...] = ()
    owner: str | None = None
    deletion_requested_at: float | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.observed_generation >= self.generation and not self.deleting

    @property
    def deleting(self) -> bool:
        return self.deletion_requested_at is not None

    def bump(self) -> int:
        self.generation += 1
        return self.generation

    def observe(self) -> None:
        self.observed_generation = self.generation

    def add_finalizer(self, name: str) -> None:
        if name not in self.finalizers:
            self.finalizers = (*self.finalizers, name)

    def remove_finalizer(self, name: str) -> None:
        self.finalizers = tuple(f for f in self.finalizers if f != name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generation": self.generation,
            "observed_generation": self.observed_generation,
            "converged": self.converged,
            "finalizers": list(self.finalizers),
            "owner": self.owner,
            "deleting": self.deleting,
            "labels": dict(self.labels),
        }


class RateLimitingQueue:
    """Deduplicating work queue with per-key exponential backoff.

    Three states per key, exactly as controller-runtime has them:
    *queued* (waiting for a worker), *processing* (a worker holds it), and
    *dirty* (something asked for it again while it was processing, so it
    gets exactly one more pass when the current one finishes).
    """

    def __init__(self, name: str = "queue") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._queue: list[Request] = []
        self._queued: set[str] = set()
        self._processing: set[str] = set()
        self._dirty: dict[str, Request] = {}
        self._failures: dict[str, int] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._wakeup = asyncio.Event()
        self._shutdown = False
        self.adds = 0
        self.deduped = 0
        self.retries = 0

    # ── producing ─────────────────────────────────────────────────────
    def add(self, request: Request | str) -> bool:
        """Enqueue. Returns False when deduplicated against a pending item."""
        req = request if isinstance(request, Request) else Request(key=request)
        with self._lock:
            if self._shutdown:
                return False
            self.adds += 1
            if req.key in self._processing:
                # Coalesce: one more pass after the current one, not N.
                self._dirty[req.key] = req
                self.deduped += 1
                return False
            if req.key in self._queued:
                self.deduped += 1
                return False
            self._queued.add(req.key)
            self._queue.append(req)
        self._signal()
        return True

    def add_after(self, request: Request | str, delay_s: float) -> None:
        """Enqueue once, after a delay, collapsing repeated scheduling."""
        req = request if isinstance(request, Request) else Request(key=request)
        if delay_s <= 0:
            self.add(req)
            return
        try:
            loop = asyncio.get_running_loop()
        # not a failure: off a loop there is nothing to delay with, so the
        # request goes in now rather than not at all.
        except RuntimeError:
            self.add(req)
            return
        with self._lock:
            existing = self._timers.pop(req.key, None)
        if existing is not None:
            existing.cancel()

        def fire() -> None:
            with self._lock:
                self._timers.pop(req.key, None)
            self.add(req)

        handle = loop.call_later(delay_s, fire)
        with self._lock:
            self._timers[req.key] = handle

    def add_rate_limited(self, request: Request | str) -> float:
        """Requeue with this key's current backoff. Returns the delay used."""
        req = request if isinstance(request, Request) else Request(key=request)
        with self._lock:
            failures = self._failures.get(req.key, 0) + 1
            self._failures[req.key] = failures
            self.retries += 1
        delay = min(MAX_DELAY_S, BASE_DELAY_S * (2 ** (failures - 1)))
        self.add_after(req, delay)
        return delay

    def forget(self, key: str) -> None:
        """Reset a key's backoff. Called on success — this is what makes
        the limiter recover instead of ratcheting up forever."""
        with self._lock:
            self._failures.pop(key, None)

    def failures(self, key: str) -> int:
        with self._lock:
            return self._failures.get(key, 0)

    # ── consuming ─────────────────────────────────────────────────────
    def _signal(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        # not a failure: nothing is waiting to be woken when no loop runs.
        except RuntimeError:
            return
        loop.call_soon_threadsafe(self._wakeup.set)

    async def get(self) -> Request | None:
        """Wait for the next item. Returns None once shut down and drained."""
        drained = False
        while not drained:
            with self._lock:
                if self._queue:
                    req = self._queue.pop(0)
                    self._queued.discard(req.key)
                    self._processing.add(req.key)
                    return req
                if self._shutdown:
                    drained = True
                    continue
                self._wakeup.clear()
            # Timed rather than indefinite: a consumer that can only be
            # woken by a producer cannot notice that shutdown was
            # requested while it was asleep, and a lost wakeup becomes a
            # permanent stall instead of one extra poll interval.
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=IDLE_POLL_S)
            except TimeoutError:
                continue
        return None

    def done(self, request: Request) -> None:
        """Release a key. A dirty key gets exactly one more pass."""
        with self._lock:
            self._processing.discard(request.key)
            pending = self._dirty.pop(request.key, None)
            if pending is not None and pending.key not in self._queued:
                self._queued.add(pending.key)
                self._queue.append(pending)
        if pending is not None:
            self._signal()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            for handle in self._timers.values():
                handle.cancel()
            self._timers.clear()
        self._signal()

    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "depth": len(self._queue),
                "processing": len(self._processing),
                "dirty": len(self._dirty),
                "adds": self.adds,
                "deduped": self.deduped,
                "retries": self.retries,
                "backing_off": {k: v for k, v in self._failures.items() if v},
            }


ReconcileFn = Callable[[Request], "Result | Awaitable[Result]"]


class Controller:
    """One reconciler, N workers, one queue, one resync timer."""

    def __init__(
        self,
        name: str,
        reconcile: ReconcileFn,
        *,
        workers: int = 1,
        resync_s: float = DEFAULT_RESYNC_S,
        list_keys: Callable[[], list[str]] | None = None,
    ) -> None:
        self.name = name
        self._reconcile = reconcile
        self._workers = max(1, workers)
        self._resync_s = resync_s
        self._list_keys = list_keys
        self.queue = RateLimitingQueue(name=f"{name}.queue")
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self.reconciles = 0
        self.failures = 0
        self.last_error = ""
        self._durations: list[float] = []

    # ── event sources ─────────────────────────────────────────────────
    def enqueue(self, key: str, reason: str = "") -> None:
        """A hint that now is a good time to look at ``key``.

        Deliberately no payload: see the module docstring on why passing
        the observed object would reintroduce edge-triggering.
        """
        self.queue.add(Request(key=key, reason=reason))

    def watch_bus(self, topic: str, key_of: Callable[[Any], str | None]) -> asyncio.Task:
        """Subscribe to an event-bus topic as a hint source."""

        async def listen() -> None:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            queue = await bus.subscribe(topic)
            try:
                while self._running:
                    # Bounded so `self._running` is re-checked on a known
                    # cadence: a listener blocked forever on a quiet topic
                    # outlives the controller it belongs to.
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=IDLE_POLL_S)
                    except TimeoutError:
                        continue
                    try:
                        payload = event[1] if isinstance(event, tuple) else event
                        key = key_of(payload)
                    except Exception:  # noqa: BLE001 — a bad hint is still just a hint
                        logger.debug("%s: key extraction failed for %s", self.name, topic, exc_info=True)
                        continue
                    if key:
                        self.enqueue(key, reason=f"event:{topic}")
            except asyncio.CancelledError:
                raise
            finally:
                with contextlib.suppress(Exception):
                    await bus.unsubscribe(topic, queue)

        task = get_task_tracker().create_task(
            listen(),
            name=f"{self.name}.watch.{topic}",
        )
        self._tasks.append(task)
        return task

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._workers):
            self._tasks.append(
                get_task_tracker().create_task(
                    self._worker(index),
                    name=f"{self.name}.worker{index}",
                )
            )
        if self._resync_s > 0 and self._list_keys is not None:
            self._tasks.append(
                get_task_tracker().create_task(
                    self._resync_loop(),
                    name=f"{self.name}.resync",
                )
            )
        logger.info(
            "🔁 controller %s started (%d worker(s), resync %.0fs)",
            self.name,
            self._workers,
            self._resync_s,
        )

    async def stop(self) -> None:
        self._running = False
        self.queue.shutdown()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _worker(self, index: int) -> None:
        while self._running:
            try:
                request = await asyncio.wait_for(self.queue.get(), timeout=IDLE_POLL_S)
            except TimeoutError:
                continue
            if request is None:
                return
            started = time.perf_counter()
            try:
                result = self._reconcile(request)
                if asyncio.iscoroutine(result):
                    result = await result
                if not isinstance(result, Result):
                    result = Result.done()
                self.queue.forget(request.key)
                if result.requeue or result.requeue_after_s > 0:
                    self.queue.add_after(request, result.requeue_after_s)
                self.reconciles += 1
            except asyncio.CancelledError:
                self.queue.done(request)
                raise
            except Exception as exc:  # noqa: BLE001 — a failing key must not kill the worker
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                delay = self.queue.add_rate_limited(request)
                logger.warning(
                    "🔁 %s reconcile(%s) failed (%s); retrying in %.1fs (attempt %d)",
                    self.name,
                    request.key,
                    self.last_error,
                    delay,
                    self.queue.failures(request.key),
                )
                from core.runtime.errors import record_degradation

                record_degradation(
                    f"controller.{self.name}",
                    exc,
                    severity="warning",
                    action=f"requeued {request.key} with backoff {delay:.1f}s",
                    enforce_failure_policy=False,
                )
            finally:
                elapsed = time.perf_counter() - started
                self._durations.append(elapsed)
                if len(self._durations) > 256:
                    del self._durations[:-256]
                # Aura.Reconcile.DurationMs named this file as its owner and
                # had no writer. The retained list above is the thing the
                # histogram exists to replace — it holds 256 samples, derives
                # p50/p95 from them, and loses every observation before that,
                # so "is reconcile slower than last week" was unanswerable.
                # Both are kept: the list still feeds report().
                try:
                    from core.observability.histograms import record as record_histogram

                    record_histogram("Aura.Reconcile.DurationMs", max(0.0, elapsed) * 1000.0)
                except (ImportError, RuntimeError, ValueError, TypeError) as exc:
                    # The comment above is about this histogram having had no
                    # writer. A silent except is the same outcome by a
                    # different route.
                    logger.debug(
                        "Aura.Reconcile.DurationMs not recorded (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                self.queue.done(request)

    async def _resync_loop(self) -> None:
        """Periodic full resync — the belt to the event stream's braces."""
        while self._running:
            try:
                await asyncio.sleep(self._resync_s)
            except asyncio.CancelledError:
                # A cancel aimed at THIS loop has to keep going up; one that
                # only interrupted the sleep ends the loop quietly.
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
                return
            if not self._running or self._list_keys is None:
                return
            try:
                for key in self._list_keys():
                    self.enqueue(key, reason="resync")
            except Exception:  # noqa: BLE001
                logger.debug("%s resync listing failed", self.name, exc_info=True)

    def report(self) -> dict[str, Any]:
        durations = sorted(self._durations)
        p50 = durations[len(durations) // 2] if durations else 0.0
        p95 = durations[int(len(durations) * 0.95)] if durations else 0.0
        return {
            "name": self.name,
            "running": self._running,
            "workers": self._workers,
            "resync_s": self._resync_s,
            "reconciles": self.reconciles,
            "failures": self.failures,
            "last_error": self.last_error,
            "p50_s": round(p50, 4),
            "p95_s": round(p95, 4),
            "queue": self.queue.report(),
        }


class ControllerManager:
    """Owns every controller so one call starts and stops them all."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controllers: dict[str, Controller] = {}

    def register(self, controller: Controller) -> Controller:
        with self._lock:
            existing = self._controllers.get(controller.name)
            if existing is not None:
                return existing
            self._controllers[controller.name] = controller
            return controller

    def get(self, name: str) -> Controller | None:
        with self._lock:
            return self._controllers.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._controllers)

    async def start_all(self) -> list[str]:
        with self._lock:
            controllers = list(self._controllers.values())
        started: list[str] = []
        for controller in controllers:
            try:
                await controller.start()
                started.append(controller.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("controller %s failed to start: %s", controller.name, exc)
        return started

    async def stop_all(self) -> None:
        with self._lock:
            controllers = list(self._controllers.values())
        for controller in controllers:
            with contextlib.suppress(Exception):
                await controller.stop()

    def report(self) -> dict[str, Any]:
        with self._lock:
            controllers = list(self._controllers.values())
        return {
            "count": len(controllers),
            "controllers": [c.report() for c in controllers],
            "total_queue_depth": sum(c.queue.depth() for c in controllers),
            "unconverged": [c.name for c in controllers if c.queue.depth() > 0],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._controllers.clear()


_MANAGER = ControllerManager()


def get_controller_manager() -> ControllerManager:
    return _MANAGER


def controller(
    name: str,
    *,
    workers: int = 1,
    resync_s: float = DEFAULT_RESYNC_S,
    list_keys: Callable[[], list[str]] | None = None,
) -> Callable[[ReconcileFn], Controller]:
    """Declare a controller next to the reconcile function it runs::

        @controller("model_lane", resync_s=60.0, list_keys=lane_names)
        async def reconcile_lane(request: Request) -> Result:
            desired = desired_lane(request.key)
            observed = observe_lane(request.key)
            if observed == desired:
                return Result.done()
            step_toward(desired)
            return Result.again(after_s=2.0)
    """

    def decorate(fn: ReconcileFn) -> Controller:
        return _MANAGER.register(
            Controller(name, fn, workers=workers, resync_s=resync_s, list_keys=list_keys)
        )

    return decorate


async def run_finalizers(
    meta: ObjectMeta,
    handlers: dict[str, Callable[[], Any]],
) -> bool:
    """Run pending finalizers; returns True when the object may really go.

    A finalizer that fails keeps the object alive and the deletion
    pending: cleanup that silently failed is
    indistinguishable from cleanup that never ran.
    """
    if not meta.deleting:
        return False
    for name in list(meta.finalizers):
        handler = handlers.get(name)
        if handler is None:
            logger.warning(
                "finalizer %r on %s has no handler; deletion stays pending",
                name,
                meta.name,
            )
            return False
        try:
            outcome = handler()
            if asyncio.iscoroutine(outcome):
                await outcome
        except Exception as exc:  # noqa: BLE001
            logger.warning("finalizer %r on %s failed: %s", name, meta.name, exc)
            return False
        meta.remove_finalizer(name)
    return not meta.finalizers


def reconcile_report() -> dict[str, Any]:
    return _MANAGER.report()


def reset_reconcile_for_test() -> None:
    _MANAGER.reset_for_test()


__all__ = [
    "BASE_DELAY_S",
    "DEFAULT_RESYNC_S",
    "MAX_DELAY_S",
    "Controller",
    "ControllerManager",
    "ObjectMeta",
    "RateLimitingQueue",
    "ReconcileFn",
    "Request",
    "Result",
    "controller",
    "get_controller_manager",
    "reconcile_report",
    "reset_reconcile_for_test",
    "run_finalizers",
]
