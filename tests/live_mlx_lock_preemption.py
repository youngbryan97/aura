"""live_mlx_lock_preemption.py — verify the lock preemption fix end-to-end.

This test exercises the REAL code paths inside core/brain/llm/mlx_client.py
without needing a GPU or the full orchestrator.  It instantiates an
MLXLocalClient, manually acquires the request lock on a background "owner"
(simulating a wedged in-flight generation), then runs _acquire_request_lock
from a foreground caller and asserts:

    1. Foreground waits at most the new 12 s request_lock_timeout (not 30 s).
    2. When the holder has been wedged past the first-token SLA, the
       preemption path sets _deferred_reboot_reason and cancels the stuck
       future.
    3. When the holder is still within the SLA window, NO preemption fires
       (so healthy slow requests aren't killed prematurely).

Run:
    ~/.aura/live-source/.venv/bin/python3.12 tests/live_mlx_lock_preemption.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_client():
    from core.brain.llm.mlx_client import MLXLocalClient
    # Use a path containing "32b" so the foreground budgets match the Cortex
    # thresholds we just tightened (first_token_sla=22s, lock_timeout=12s).
    return MLXLocalClient(model_path="/tmp/fake-qwen-32b-instruct")


async def scenario_a_preemption_fires() -> tuple[bool, str]:
    """Holder has been stuck for 25s — past the 22s foreground SLA.
    Foreground caller should time out on the 12s lock budget AND
    trigger preemption (cancel future + defer reboot)."""
    client = _make_client()

    # Simulate a wedged in-flight generation: acquire lock directly, set an
    # old acquired_at timestamp, plant a stuck SharedFuture.
    from core.brain.llm.mlx_client import _new_shared_future
    client._request_lock.acquire()
    client._request_lock_owner_label = "Cortex"
    client._request_lock_acquired_at = time.time() - 25.0  # 25s ago
    stuck = _new_shared_future()
    client._current_gen_future = stuck

    t0 = time.perf_counter()
    acquired = await client._acquire_request_lock(
        owner_label="second_user_message",
        deadline=None,
        foreground_request=True,
    )
    elapsed = time.perf_counter() - t0

    try:
        if acquired:
            return False, "unexpected: should have timed out on wedged holder"
        if elapsed > 13.0:
            return False, f"waited too long: {elapsed:.2f}s (budget=12s)"
        if client._deferred_reboot_reason != "foreground_preemption_wedged_holder":
            return False, f"preemption did not fire; reason={client._deferred_reboot_reason}"
        # Stuck future must have been cancelled
        if not stuck.done():
            return False, "stuck future was not cancelled by preemption"
        return True, f"timed out in {elapsed:.2f}s, preempted + future cancelled"
    finally:
        # release for cleanup
        try:
            client._request_lock.release()
        except Exception:
            pass


async def scenario_b_no_premature_preemption() -> tuple[bool, str]:
    """Holder has only been running 5s — well within the 22s SLA.
    Foreground should still time out on the 12s wait budget, but must NOT
    preempt a healthy slow request."""
    client = _make_client()
    from core.brain.llm.mlx_client import _new_shared_future
    client._request_lock.acquire()
    client._request_lock_owner_label = "Cortex"
    client._request_lock_acquired_at = time.time() - 5.0  # only 5s
    healthy_future = _new_shared_future()
    client._current_gen_future = healthy_future

    t0 = time.perf_counter()
    acquired = await client._acquire_request_lock(
        owner_label="second_user_message",
        deadline=None,
        foreground_request=True,
    )
    elapsed = time.perf_counter() - t0

    try:
        if acquired:
            return False, "unexpected: lock should still be held"
        if client._deferred_reboot_reason:
            return False, (
                f"preemption fired prematurely on healthy request "
                f"(age=5s, sla=22s); reason={client._deferred_reboot_reason}"
            )
        if healthy_future.done():
            return False, "healthy future was cancelled unexpectedly"
        return True, f"timed out in {elapsed:.2f}s without premature preemption"
    finally:
        try:
            client._request_lock.release()
        except Exception:
            pass


async def scenario_c_lock_timeout_tightened() -> tuple[bool, str]:
    """Verify the foreground lock timeout is now 12s (not 30s)."""
    client = _make_client()
    budget = client._request_lock_timeout(deadline=None, foreground_request=True)
    bg_budget = client._request_lock_timeout(deadline=None, foreground_request=False)
    if budget > 15.0:
        return False, f"foreground budget too loose: {budget}s (expected ≤15)"
    if bg_budget > 15.0:
        return False, f"background budget regressed: {bg_budget}s"
    return True, f"foreground={budget}s, background={bg_budget}s"


async def scenario_d_empty_retry_path_compiles() -> tuple[bool, str]:
    """Smoke test: the new inline empty-retry code path parses and the
    foreground threshold methods now accept foreground_request kwarg."""
    client = _make_client()
    fto = client._first_token_sla(foreground_request=True)
    bg_fto = client._first_token_sla(foreground_request=False)
    ts = client._token_stall_after(foreground_request=True)
    stale = client._stale_after(during_generation=True, foreground_request=True)
    if fto >= 30.0:
        return False, f"foreground first_token_sla not tightened: {fto}s"
    if ts >= 24.0:
        return False, f"foreground token_stall_after not tightened: {ts}s"
    if stale >= 40.0:
        return False, f"foreground stale_after not tightened: {stale}s"
    return True, (
        f"foreground first_token_sla={fto}s (bg={bg_fto}s), "
        f"token_stall_after={ts}s, stale_after={stale}s"
    )


async def scenario_e_worker_cache_clear_on_empty() -> tuple[bool, str]:
    """Verify the worker source now clears prompt_cache_lru on zero-token
    generation (read the source, we can't spawn a real MLX subprocess here)."""
    path = PROJECT_ROOT / "core" / "brain" / "llm" / "mlx_worker.py"
    src = path.read_text()
    needle = "prompt_cache_lru.clear()"
    # Should appear inside the zero-token warning block AND inside
    # the clear_cache action — both must be present.
    if src.count(needle) < 2:
        return False, f"expected ≥2 prompt_cache_lru.clear() calls, got {src.count(needle)}"
    # Empty-token self-heal should sit in a block that also hits
    # "yielded ZERO tokens" warning.
    zero_block_start = src.find("yielded ZERO tokens")
    if zero_block_start < 0:
        return False, "zero-token warning block missing"
    # Check the clear() appears within a reasonable window after that warning
    tail = src[zero_block_start:zero_block_start + 2000]
    if needle not in tail:
        return False, "prompt_cache_lru.clear() not in zero-token self-heal block"
    return True, "worker self-heals prompt cache on zero-token + clear_cache action"


async def main() -> int:
    scenarios = [
        ("A. foreground preempts wedged holder past SLA", scenario_a_preemption_fires),
        ("B. foreground does NOT preempt healthy slow holder", scenario_b_no_premature_preemption),
        ("C. foreground lock timeout tightened (30s→12s)", scenario_c_lock_timeout_tightened),
        ("D. foreground stall thresholds tightened", scenario_d_empty_retry_path_compiles),
        ("E. worker clears prompt_cache_lru on zero-token", scenario_e_worker_cache_clear_on_empty),
    ]

    print("🔬 MLX Lock Preemption Live Verification\n")
    failed = 0
    for name, fn in scenarios:
        t0 = time.perf_counter()
        try:
            ok, detail = await fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        elapsed = (time.perf_counter() - t0) * 1000
        status = "✓" if ok else "✗"
        print(f"  [{status}] {name} ({elapsed:.0f}ms) — {detail}")
        if not ok:
            failed += 1

    print()
    print(f"RESULT: {len(scenarios) - failed}/{len(scenarios)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
