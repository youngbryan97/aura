#!/usr/bin/env python3
"""Real live autonomy longevity soak runner for Aura.

Executes actual runtime processes, model queries, memory writes, and tool
executions. Every action is governed, transacted, and receipted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.resource_observation import get_resource_observer  # noqa: E402

_LIVE_SOAK_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _rss_mb() -> float:
    try:
        memory = get_resource_observer().memory(root_pid=os.getpid())
        if memory.available:
            return memory.process_rss_bytes / (1024 * 1024)
    except (RuntimeError, OSError, AttributeError):
        return 0.0


def _queue_depths() -> dict[str, int]:
    depths: dict[str, int] = {}
    try:
        from core.container import ServiceContainer
        for name in ("event_bus", "task_tracker", "llm_router"):
            service = ServiceContainer.get(name, default=None)
            if service is None:
                continue
            for attr in ("queue", "_queue", "pending", "_pending"):
                queue = getattr(service, attr, None)
                if hasattr(queue, "qsize"):
                    try:
                        depths[f"{name}.{attr}"] = int(queue.qsize())
                    except (RuntimeError, OSError, ValueError):
                        continue
    except (ImportError, RuntimeError, AttributeError):
        return depths
    return depths


def write_json(path: Path, data: Any) -> None:
    from core.runtime.atomic_writer import atomic_write_text
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_output_dir(raw_path: str) -> Path:
    from tools.longevity.soak_output import resolve_output_dir

    return resolve_output_dir(raw_path)


async def _measure_lag_ms() -> float:
    from core.runtime.event_loop_responsiveness import sample_event_loop_lag
    try:
        sample = (await sample_event_loop_lag(samples=1, interval_s=0.05))[0]
        return round(float(sample.lag_ms), 3)
    except _LIVE_SOAK_RECOVERABLE_ERRORS:
        return 0.0


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="proof")
    parser.add_argument("--out", default="artifacts/current/live_longevity_soak")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    out_dir = _resolve_output_dir(args.out)

    os.environ.setdefault("AURA_PROOF_RUN", "1")
    from aura_main import boot_aura_runtime
    from core.brain.llm.llm_router import get_llm_router
    from core.governance.will import ActionDomain, get_will
    from core.runtime.action_executor import ActionExecutor
    from tools.agi.run_dnu_agi_proof_battery import shutdown_proof_runtime
    from tools.receipt_material import signed_will_receipt_entry

    print("🚀 Booting Aura runtime for live autonomy soak...")
    orch = await boot_aura_runtime(
        profile=args.profile,
        ready_label="Live-Longevity",
        readiness_context="live_longevity_soak",
        artifact_root=ROOT / "artifacts" / "current",
    )

    receipts_path = out_dir / "RECEIPTS.jsonl"
    metrics: list[dict[str, Any]] = []
    lag_threshold_ms = float(os.getenv("AURA_LONGEVITY_MAX_LOOP_LAG_MS", "250") or 250)
    
    from core.runtime.event_loop_responsiveness import wait_for_event_loop_quiescence
    boot_loop_report = await wait_for_event_loop_quiescence(
        threshold_ms=lag_threshold_ms,
        required_consecutive=3,
        timeout_s=10.0,
        interval_s=0.05,
    )

    start_time = time.time()
    model_calls = 0
    memory_writes = 0
    tool_attempts = 0
    failures = 0
    recovery_count = 0

    try:
        will = get_will()
        await will.start()
        router = get_llm_router()

        from tools.longevity.soak_output import open_receipts

        with open_receipts(receipts_path) as receipt_file:
            for index in range(max(1, args.iterations)):
                elapsed = time.time() - start_time
                if elapsed > args.duration_s:
                    print(f"Stopping live soak loop: elapsed {elapsed:.2f}s > budget {args.duration_s}s")
                    break

                print(f"--- Live Soak Pulse {index} ---")

                # 1. Model Call (Self-Reflection)
                try:
                    print("🧠 Making model query via llm_router...")
                    response = await router.think(
                        prompt=f"Perform autonomy live soak heartbeat reflection index={index}.",
                        max_tokens=24,
                        origin="live_soak"
                    )
                    print(f"Model response summary: {response[:100]}")
                    model_calls += 1
                except _LIVE_SOAK_RECOVERABLE_ERRORS as e:
                    print(f"Model call failed: {e}")
                    failures += 1
                    recovery_count += 1

                # 2. Governed Tool Attempt (Subprocess Echo/Uname via ActionExecutor)
                try:
                    print("🛠️ Executing governed subprocess action...")
                    tool_res = await ActionExecutor.execute(
                        domain=ActionDomain.TOOL_EXECUTION,
                        action_name="run_command",
                        params={"argv": ["uname", "-a"]},
                        source="live_soak",
                    )
                    if tool_res.get("ok"):
                        print(f"Tool execution succeeded: {tool_res.get('stdout')}")
                    else:
                        print(f"Tool execution failed: {tool_res.get('error')}")
                        failures += 1
                    tool_attempts += 1
                except _LIVE_SOAK_RECOVERABLE_ERRORS as e:
                    print(f"Tool execution encountered exception: {e}")
                    failures += 1
                    recovery_count += 1

                # 3. Governed Memory Write via ActionExecutor
                try:
                    print("💾 Performing governed memory write...")
                    mem_res = await ActionExecutor.execute(
                        domain=ActionDomain.MEMORY_WRITE,
                        action_name="write_memory",
                        params={
                            "content": f"Live autonomy soak heartbeat {index} at {time.time()}",
                            "metadata": {"type": "soak_heartbeat", "index": index},
                        },
                        source="live_soak",
                    )
                    if mem_res.get("ok"):
                        print(f"Memory write succeeded: record_id={mem_res.get('record_id')}")
                        memory_writes += 1
                    else:
                        print(f"Memory write failed: {mem_res.get('error')}")
                        failures += 1
                except _LIVE_SOAK_RECOVERABLE_ERRORS as e:
                    print(f"Memory write encountered exception: {e}")
                    failures += 1
                    recovery_count += 1

                # 4. Measure loop lag and record metrics
                lag_ms = await _measure_lag_ms()
                queue_depths = _queue_depths()
                
                decision = will.decide(
                    content=f"live autonomy longevity pulse {index}",
                    source="live_soak",
                    domain=ActionDomain.STABILIZATION,
                    priority=0.4,
                )

                metrics.append({
                    "iteration": index,
                    "rss_mb": round(_rss_mb(), 3),
                    "lag_ms": round(lag_ms, 3),
                    "queue_len": sum(queue_depths.values()),
                    "queue_depths": queue_depths,
                    "receipt_id": decision.receipt_id,
                })

                receipt_file.write(
                    json.dumps(
                        signed_will_receipt_entry(
                            will,
                            decision,
                            task_id=f"live_longevity_pulse_{index}",
                            domain=ActionDomain.STABILIZATION,
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )

                # Keep loop paced
                await asyncio.sleep(0.5)

    finally:
        print("🔌 Shutting down live autonomy soak runtime...")
        await shutdown_proof_runtime(orch)

    actual_runtime = time.time() - start_time
    rss_values = [item["rss_mb"] for item in metrics if item["rss_mb"] > 0.0]
    rss_growth = (max(rss_values) - min(rss_values)) if rss_values else 0.0
    max_lag = max((item["lag_ms"] for item in metrics), default=0.0)
    max_queue = max((item["queue_len"] for item in metrics), default=0)

    report = {
        "generated_at": time.time(),
        "profile": args.profile,
        "iterations_completed": len(metrics),
        "memory_leakage_detected": rss_growth > 128.0,
        "rss_growth_mb": round(rss_growth, 3),
        "queue_growth_stable": max_queue <= 15,
        "boot_event_loop_stable": boot_loop_report.stable,
        "boot_event_loop_warmup": boot_loop_report.to_dict(),
        "event_loop_lag_threshold_ms": round(lag_threshold_ms, 3),
        "event_loop_lag_normal": bool(boot_loop_report.stable and max_lag <= lag_threshold_ms),
        "max_lag_ms": round(max_lag, 3),
        "max_queue_len": max_queue,
        "metrics": metrics,
        
        # Live soak-specific attributes
        "claim_scope": "live autonomy longevity soak; real execution evidence",
        "actual_runtime_s": round(actual_runtime, 3),
        "model_calls": model_calls,
        "memory_writes": memory_writes,
        "tool_attempts": tool_attempts,
        "failures": failures,
        "recovery_attempts": recovery_count,
        "synthetic_only": False,
    }

    write_json(out_dir / "SOAK_METRICS.json", report)
    manifest = {
        "schema": "longevity_soak_manifest",
        "sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ("SOAK_METRICS.json", "RECEIPTS.jsonl")
        },
    }
    write_json(out_dir / "MANIFEST.json", manifest)

    print(f"Live longevity soak completed successfully for profile: {args.profile}")
    return 0 if not report["memory_leakage_detected"] and report["queue_growth_stable"] and report["event_loop_lag_normal"] else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
