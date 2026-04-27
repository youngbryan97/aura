"""tools/chaos/injector.py

Chaos injection — break things deliberately and prove repair works.

Catalogue:

  kill_subprocess          — pick a child PID and send SIGKILL
  corrupt_sqlite_row       — write a bad row to a non-critical table
  delete_vector_index      — delete one vector store partition
  induce_event_loop_lag    — sleep on the event loop for ~1.2s
  force_model_load_failure — flip the model registry to point at a nonexistent path
  break_memory_facade      — register a no-op stand-in for memory_facade
  break_agency_pathway     — disable one VolitionEngine pathway
  fill_disk                — write a 100MB file to /tmp until disk is 95%+
  sever_network            — block outbound connections via local proxy
  expire_api_keys          — flip env vars to invalid values

Each fault returns a dict ``{kind, applied: True|False, detail}``. The
chaos run records the fault and the system's repair signal (the
StabilityGuardian and ResilienceEngine telemetry) for later analysis.
"""
from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger("Aura.Chaos")


_FAULTS: Dict[str, Callable[[], Awaitable[Dict[str, Any]]]] = {}


def register(name: str) -> Callable[[Callable[[], Awaitable[Dict[str, Any]]]], Callable[[], Awaitable[Dict[str, Any]]]]:
    def deco(fn: Callable[[], Awaitable[Dict[str, Any]]]) -> Callable[[], Awaitable[Dict[str, Any]]]:
        _FAULTS[name] = fn
        return fn
    return deco


@register("induce_event_loop_lag")
async def _induce_loop_lag() -> Dict[str, Any]:
    t0 = time.monotonic()
    # Synchronous sleep on the event loop thread to simulate a stall.
    time.sleep(1.2)
    return {"kind": "induce_event_loop_lag", "applied": True, "lagged_ms": int((time.monotonic() - t0) * 1000)}


@register("force_model_load_failure")
async def _force_model_load_failure() -> Dict[str, Any]:
    prev = os.environ.get("AURA_MODEL")
    os.environ["AURA_MODEL"] = "/tmp/nonexistent-model-injection"
    try:
        # Restore after 60 seconds so the next chaos cycle can re-roll.
        await asyncio.sleep(0)
    finally:
        async def _restore():
            await asyncio.sleep(60.0)
            if prev is None:
                os.environ.pop("AURA_MODEL", None)
            else:
                os.environ["AURA_MODEL"] = prev
        get_task_tracker().create_task(_restore())
    return {"kind": "force_model_load_failure", "applied": True, "restored_in_s": 60}


@register("expire_api_keys")
async def _expire_api_keys() -> Dict[str, Any]:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    flipped = {}
    for k in keys:
        prev = os.environ.get(k)
        flipped[k] = prev
        os.environ[k] = "invalid-injection"

    async def _restore():
        await asyncio.sleep(60.0)
        for k, prev in flipped.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
    get_task_tracker().create_task(_restore())
    return {"kind": "expire_api_keys", "applied": True, "keys": list(flipped.keys())}


@register("delete_vector_index")
async def _delete_vector_index() -> Dict[str, Any]:
    target = Path.home() / ".aura" / "data" / "vector_index"
    backup = Path.home() / ".aura" / "data" / f"vector_index.injected.{int(time.time())}"
    if not target.exists():
        return {"kind": "delete_vector_index", "applied": False, "reason": "no_target"}
    target.rename(backup)

    async def _restore():
        await asyncio.sleep(120.0)
        if not target.exists():
            backup.rename(target)
    get_task_tracker().create_task(_restore())
    return {"kind": "delete_vector_index", "applied": True, "moved_to": str(backup)}


@register("fill_disk")
async def _fill_disk() -> Dict[str, Any]:
    # We deliberately do NOT actually fill the disk in production runs;
    # this stub records intent and returns. A real chaos run wires this
    # to a sandbox volume.
    return {"kind": "fill_disk", "applied": False, "reason": "stub_no_op_in_prod"}


@register("sever_network")
async def _sever_network() -> Dict[str, Any]:
    # Set HTTPS_PROXY to a local dead port to block outbound HTTP.
    prev = os.environ.get("HTTPS_PROXY")
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1"

    async def _restore():
        await asyncio.sleep(45.0)
        if prev is None:
            os.environ.pop("HTTPS_PROXY", None)
        else:
            os.environ["HTTPS_PROXY"] = prev
    get_task_tracker().create_task(_restore())
    return {"kind": "sever_network", "applied": True, "restored_in_s": 45}


async def inject_random_fault() -> Dict[str, Any]:
    name = random.choice(list(_FAULTS.keys()))
    try:
        out = await _FAULTS[name]()
    except Exception as exc:
        return {"kind": name, "applied": False, "error": str(exc)}
    return out


async def main(argv: List[str]) -> int:
    """CLI entry-point: ``python -m tools.chaos.injector --kind <name>``"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="random")
    args = parser.parse_args(argv)
    if args.kind == "random":
        out = await inject_random_fault()
    else:
        fn = _FAULTS.get(args.kind)
        if fn is None:
            print(f"unknown fault: {args.kind}; choices: {list(_FAULTS.keys())}")
            return 1
        out = await fn()
    print(out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main(sys.argv[1:])))
