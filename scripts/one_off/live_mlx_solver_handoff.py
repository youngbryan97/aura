#!/usr/bin/env python3
"""Manual live harness for the MLX 72B solver handoff path.

This script proves three things independently:
1. the 32B primary lane can warm and answer
2. the 72B solver can warm and answer directly
3. the InferenceGate deep-handoff path can swap to 72B and recover 32B
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


def _dump(label: str, payload: object) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(json.dumps(payload, indent=2, default=str), flush=True)


def _memory_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "ram_percent": round(float(vm.percent), 1),
        "available_gb": round(vm.available / float(1024 ** 3), 2),
        "used_gb": round(vm.used / float(1024 ** 3), 2),
        "total_gb": round(vm.total / float(1024 ** 3), 2),
    }


def _worker_snapshot(client: Any) -> dict[str, Any]:
    lane = client.get_lane_status() if hasattr(client, "get_lane_status") else {}
    proc = getattr(client, "_process", None)
    pid = getattr(proc, "pid", None)
    rss_gb = None
    alive = bool(proc and proc.is_alive())
    if pid and alive:
        try:
            rss_gb = round(psutil.Process(pid).memory_info().rss / float(1024 ** 3), 2)
        except Exception:
            rss_gb = None
    return {
        "pid": pid,
        "alive": alive,
        "state": lane.get("state"),
        "conversation_ready": bool(lane.get("conversation_ready")),
        "last_error": lane.get("last_error"),
        "warmup_attempted": bool(lane.get("warmup_attempted")),
        "warmup_in_flight": bool(lane.get("warmup_in_flight")),
        "worker_rss_gb": rss_gb,
    }


def _coerce_result(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple):
        success = bool(result[0])
        text = str(result[1] or "")
        meta = result[2] if len(result) > 2 else None
        return {"success": success, "text": text, "meta": meta}
    text = str(result or "")
    return {"success": bool(text.strip()), "text": text, "meta": None}


async def _generate_direct(
    client: Any,
    *,
    label: str,
    prompt: str,
    max_tokens: int = 120,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    from core.utils.deadlines import get_deadline

    started = time.perf_counter()
    result = await client.generate_text_async(
        prompt,
        foreground_request=True,
        owner_label=f"live_solver:{label}",
        deadline=get_deadline(timeout_s),
        max_tokens=max_tokens,
        temp=0.2,
        top_p=0.9,
    )
    parsed = _coerce_result(result)
    parsed["elapsed_s"] = round(time.perf_counter() - started, 2)
    parsed["worker"] = _worker_snapshot(client)
    parsed["memory"] = _memory_snapshot()
    return parsed


async def _wait_for_primary_ready(
    gate: Any,
    *,
    timeout_s: float = 90.0,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_state = None
    while time.monotonic() < deadline:
        lane = gate.get_conversation_status()
        state = (
            lane.get("state"),
            lane.get("conversation_ready"),
            lane.get("last_failure_reason"),
        )
        if state != last_state:
            _dump(
                "primary_restore_poll",
                {
                    "lane": lane,
                    "memory": _memory_snapshot(),
                },
            )
            last_state = state
        if lane.get("conversation_ready"):
            return lane
        await asyncio.sleep(poll_s)
    raise TimeoutError(f"primary lane did not recover within {timeout_s:.0f}s")


async def _warmup(client: Any, *, label: str, skip_swap_cooldown: bool = False) -> None:
    started = time.perf_counter()
    await client.warmup(
        foreground_request=True,
        skip_swap_cooldown=skip_swap_cooldown,
    )
    _dump(
        f"{label}_warmup",
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "worker": _worker_snapshot(client),
            "memory": _memory_snapshot(),
        },
    )


async def main() -> int:
    from core.brain.inference_gate import InferenceGate
    from core.brain.llm.mlx_client import get_mlx_client
    from core.brain.llm.model_registry import (
        ACTIVE_MODEL,
        DEEP_MODEL,
        get_deep_model_path,
        get_local_backend,
        get_runtime_model_path,
    )

    primary_path = str(get_runtime_model_path(ACTIVE_MODEL))
    deep_path = str(get_deep_model_path())
    primary_client = get_mlx_client(model_path=primary_path)
    deep_client = get_mlx_client(model_path=deep_path)

    _dump(
        "runtime",
        {
            "backend": get_local_backend(),
            "active_model": ACTIVE_MODEL,
            "deep_model": DEEP_MODEL,
            "primary_path": primary_path,
            "deep_path": deep_path,
            "memory": _memory_snapshot(),
        },
    )

    await _warmup(primary_client, label="primary")
    primary_direct = await _generate_direct(
        primary_client,
        label="primary_direct",
        prompt="Reply with exactly PRIMARY_OK and nothing else.",
        max_tokens=24,
        timeout_s=120.0,
    )
    _dump("primary_direct", primary_direct)

    await _warmup(deep_client, label="deep")
    deep_direct = await _generate_direct(
        deep_client,
        label="deep_direct",
        prompt=(
            "Reply with exactly SOLVER_OK and nothing else. "
            "Do not add punctuation or explanation."
        ),
        max_tokens=24,
        timeout_s=240.0,
    )
    _dump("deep_direct", deep_direct)

    await _warmup(primary_client, label="primary_restore_manual", skip_swap_cooldown=True)

    gate = InferenceGate()
    gate._mlx_client = primary_client

    gate_prompt = (
        "Operationally diagnose a Python asyncio deadlock where one task holds a request lock, "
        "another foreground task times out waiting for it, and the stalled worker never releases "
        "the lock. Give exactly 3 root causes and 3 concrete fixes."
    )
    gate_context = {
        "origin": "user",
        "purpose": "reply",
        "prefer_tier": "secondary",
        "deep_handoff": True,
        "allow_cloud_fallback": False,
        "allow_mesh_cognition": False,
        "history": [],
        "max_tokens": 256,
    }

    _dump(
        "gate_before_deep_handoff",
        {
            "lane": gate.get_conversation_status(),
            "primary_worker": _worker_snapshot(primary_client),
            "deep_worker": _worker_snapshot(deep_client),
            "memory": _memory_snapshot(),
        },
    )

    started = time.perf_counter()
    gate_response = await gate.generate(gate_prompt, context=gate_context, timeout=240.0)
    gate_elapsed = round(time.perf_counter() - started, 2)
    _dump(
        "gate_deep_handoff_response",
        {
            "elapsed_s": gate_elapsed,
            "response_len": len(str(gate_response or "")),
            "response_preview": str(gate_response or "")[:800],
            "lane": gate.get_conversation_status(),
            "primary_worker": _worker_snapshot(primary_client),
            "deep_worker": _worker_snapshot(deep_client),
            "memory": _memory_snapshot(),
        },
    )

    restored_lane = await _wait_for_primary_ready(gate, timeout_s=90.0)
    _dump(
        "gate_primary_restored",
        {
            "lane": restored_lane,
            "primary_worker": _worker_snapshot(primary_client),
            "deep_worker": _worker_snapshot(deep_client),
            "memory": _memory_snapshot(),
        },
    )

    primary_after_handoff = await _generate_direct(
        primary_client,
        label="primary_after_handoff",
        prompt="Reply with exactly PRIMARY_BACK and nothing else.",
        max_tokens=24,
        timeout_s=120.0,
    )
    _dump("primary_after_handoff", primary_after_handoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
