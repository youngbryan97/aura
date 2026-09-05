"""Idle reasoning pre-computation — spend the organism's idle life on hard problems.

Aura runs continuously (the cognitive heartbeat ticks with no user present). That
idle time is free compute. When a hard turn finishes verifier-dirty under the
foreground latency budget, we enqueue it; during idle the amplifier re-attempts it
with a generous budget and — because amplify_turn writes verifier-clean answers to
the solved-cache — the *next* time the user asks, the answer is already there.

This is the compute-amortization loophole: we don't beat the cost-of-compute law,
we move the cost off the critical path into time that was otherwise idle.

Bounded by construction (NO-UNBOUNDED): a capped queue, a small per-tick item
limit, and a per-item timeout. Only runs when the caller says the system is idle.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from core.brain.reasoning_solved_cache import (
    DEFAULT_CACHEABLE_TASK_TYPES,
    _problem_key,
    get_reasoning_solved_cache,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningPrecompute")

_DEFAULT_MAX_QUEUE = 256
SolveFn = Callable[[str, str], Awaitable[Any]]


def _flag_on(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {"0", "false", "off", "no"}


class PrecomputeQueue:
    """Bounded, deduplicated queue of hard problems to pre-solve during idle."""

    def __init__(self, *, max_queue: int = _DEFAULT_MAX_QUEUE) -> None:
        self._max_queue = max(8, int(max_queue))
        self._lock = threading.RLock()
        # key -> (objective, task_type); OrderedDict gives FIFO + cheap dedup.
        self._queue: "OrderedDict[str, tuple[str, str]]" = OrderedDict()
        self._stats = {"enqueued": 0, "deduped": 0, "already_cached": 0, "solved": 0, "failed": 0}

    def enqueue(self, objective: str, task_type: str) -> bool:
        tt = str(task_type or "").strip().lower()
        obj = str(objective or "").strip()
        if not obj or tt not in DEFAULT_CACHEABLE_TASK_TYPES:
            return False  # only worth pre-solving source-independent, cacheable work
        key = _problem_key(obj, tt)
        # Skip what we already know.
        try:
            if get_reasoning_solved_cache().get(obj, tt) is not None:
                self._stats["already_cached"] += 1
                return False
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reasoning_precompute_cache_check", exc)
        with self._lock:
            if key in self._queue:
                self._stats["deduped"] += 1
                return False
            self._queue[key] = (obj, tt)
            self._stats["enqueued"] += 1
            while len(self._queue) > self._max_queue:
                self._queue.popitem(last=False)  # drop oldest
        return True

    def _pop(self) -> tuple[str, str] | None:
        with self._lock:
            if not self._queue:
                return None
            _key, item = self._queue.popitem(last=False)
            return item

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._stats, "pending": len(self._queue)}

    async def tick(
        self,
        solve_fn: SolveFn | None = None,
        *,
        max_items: int = 1,
        per_item_timeout: float = 60.0,
    ) -> int:
        """Drain up to ``max_items`` from the queue, solving each. Returns count solved.

        ``solve_fn(objective, task_type)`` must run the amplifier (whose verifier-clean
        results land in the solved-cache as a side effect). Defaults to a live MLX
        solver built lazily — only constructed if actually called.
        """
        if not _flag_on("AURA_REASONING_PRECOMPUTE"):
            return 0
        fn = solve_fn or self._default_solve_fn()
        if fn is None:
            return 0
        solved = 0
        for _ in range(max(1, int(max_items))):
            item = self._pop()
            if item is None:
                break
            objective, task_type = item
            # It may have been solved (foreground) since enqueue.
            try:
                if get_reasoning_solved_cache().get(objective, task_type) is not None:
                    self._stats["already_cached"] += 1
                    continue
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass
            try:
                result = await asyncio.wait_for(fn(objective, task_type), timeout=per_item_timeout)
                if bool(getattr(result, "verified", False)):
                    self._stats["solved"] += 1
                    solved += 1
                    logger.info("🧠 [Precompute] idle-solved a queued problem (%s).", task_type)
                else:
                    self._stats["failed"] += 1
            except (asyncio.TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                self._stats["failed"] += 1
                record_degradation("reasoning_precompute_solve", exc)
        return solved

    def _default_solve_fn(self) -> SolveFn | None:
        """Lazy live solver over the resident MLX cortex (reuses, no second copy)."""

        async def _solve(objective: str, task_type: str) -> Any:
            from core.brain.inference_gate import InferenceGate
            from core.brain.reasoning_amplifier_v2 import amplify_turn

            gate = (
                InferenceGate.get_instance()
                if hasattr(InferenceGate, "get_instance")
                else InferenceGate()
            )

            async def _gen(prompt: str, temperature: float) -> str:
                out = await gate.generate(
                    prompt,
                    context={"prefer_tier": "primary", "is_background": True, "temperature": temperature},
                )
                if isinstance(out, dict):
                    out = out.get("content") or out.get("response") or ""
                return str(out or "")

            # Idle ⇒ generous budget; mark context so the dirty branch doesn't re-enqueue.
            return await amplify_turn(
                objective,
                _gen,
                task_type=task_type,
                time_budget_s=90.0,
                extra_context={"skip_precompute_enqueue": True},
            )

        return _solve


_queue_singleton: PrecomputeQueue | None = None
_singleton_lock = threading.Lock()


def get_precompute_queue() -> PrecomputeQueue:
    global _queue_singleton
    if _queue_singleton is None:
        with _singleton_lock:
            if _queue_singleton is None:
                _queue_singleton = PrecomputeQueue()
    return _queue_singleton


def reset_precompute_queue() -> None:
    global _queue_singleton
    with _singleton_lock:
        _queue_singleton = None


async def idle_precompute_tick(
    solve_fn: SolveFn | None = None, *, max_items: int = 1, per_item_timeout: float = 60.0
) -> int:
    """Convenience hook for the idle/autonomy loop to call on each idle tick."""
    return await get_precompute_queue().tick(
        solve_fn, max_items=max_items, per_item_timeout=per_item_timeout
    )
