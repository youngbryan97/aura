#!/usr/bin/env python3
"""Proof-profile longevity soak runner for Aura.

This is a short, bounded certification soak, not evidence of indefinite
autonomy. It boots the canonical runtime, executes repeated governance/health
pulses, samples real process metrics, and validates queue/lag stability.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.resource_observation import get_resource_observer  # noqa: E402


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
                        pass
    except (ImportError, RuntimeError, AttributeError):
        pass
    return depths


def write_json(path: Path, data: Any) -> None:
    from core.runtime.atomic_writer import atomic_write_text

    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_output_dir(raw_path: str) -> Path:
    from tools.longevity.soak_output import resolve_output_dir

    return resolve_output_dir(raw_path)


async def _measure_lag_ms() -> float:
    from core.runtime.event_loop_responsiveness import sample_event_loop_lag

    sample = (await sample_event_loop_lag(samples=1, interval_s=0.05))[0]
    return round(float(sample.lag_ms), 3)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="proof")
    parser.add_argument("--out", default="artifacts/current/longevity_soak")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--duration-s", type=float, default=0.0,
        help="real-time soak duration in seconds; >0 enables endurance mode (pulse every --tick-s)",
    )
    parser.add_argument(
        "--tick-s", type=float, default=30.0,
        help="seconds between pulses in endurance mode",
    )
    parser.add_argument(
        "--trace-malloc", action="store_true",
        default=bool(os.environ.get("AURA_SOAK_TRACEMALLOC")),
        help="capture tracemalloc top-growth snapshots to localize leaks",
    )
    args = parser.parse_args(argv)

    out_dir = _resolve_output_dir(args.out)

    os.environ.setdefault("AURA_PROOF_RUN", "1")
    from aura_main import boot_aura_runtime
    from core.will import ActionDomain, get_will
    from tools.agi.run_dnu_agi_proof_battery import shutdown_proof_runtime
    from tools.receipt_material import signed_will_receipt_entry

    orch = await boot_aura_runtime(
        profile=args.profile,
        ready_label="Proof-Longevity",
        readiness_context="longevity_soak",
        artifact_root=ROOT / "artifacts" / "current",
    )

    receipts_path = out_dir / "RECEIPTS.jsonl"
    metrics: list[dict[str, Any]] = []
    lag_threshold_ms = float(os.getenv("AURA_LONGEVITY_MAX_LOOP_LAG_MS", "250") or 250)
    from core.runtime.event_loop_responsiveness import wait_for_event_loop_quiescence

    boot_loop_report = await wait_for_event_loop_quiescence(
        threshold_ms=lag_threshold_ms,
        required_consecutive=int(os.getenv("AURA_LONGEVITY_STABLE_LOOP_SAMPLES", "3") or 3),
        timeout_s=float(os.getenv("AURA_LONGEVITY_LOOP_STABILIZE_TIMEOUT_S", "20") or 20),
        interval_s=0.05,
    )
    leak_baseline = None
    leak_top: list[dict[str, Any]] = []
    try:
        will = get_will()
        await will.start()
        if args.trace_malloc:
            tracemalloc.start(25)
            leak_baseline = tracemalloc.take_snapshot()
            print("🔬 [tracemalloc] leak baseline captured after boot+warmup", flush=True)
        duration_s = max(0.0, float(args.duration_s or 0.0))
        tick_s = max(1.0, float(args.tick_s or 30.0))
        soak_started = time.monotonic()
        from tools.longevity.soak_output import open_receipts

        with open_receipts(receipts_path) as receipt_file:
            index = 0
            while (
                (time.monotonic() - soak_started) < duration_s
                if duration_s > 0.0
                else index < max(1, args.iterations)
            ):
                lag_ms = await _measure_lag_ms()
                decision = will.decide(
                    content=f"longevity proof pulse {index}",
                    source="longevity_soak",
                    domain=ActionDomain.STABILIZATION,
                    priority=0.4,
                )
                queue_depths = _queue_depths()
                metrics.append(
                    {
                        "iteration": index,
                        "elapsed_s": round(time.monotonic() - soak_started, 2),
                        "rss_mb": round(_rss_mb(), 3),
                        "lag_ms": round(lag_ms, 3),
                        "queue_len": sum(queue_depths.values()),
                        "queue_depths": queue_depths,
                        "receipt_id": decision.receipt_id,
                    }
                )
                receipt_file.write(
                    json.dumps(
                        signed_will_receipt_entry(
                            will,
                            decision,
                            task_id=f"longevity_pulse_{index}",
                            domain=ActionDomain.STABILIZATION,
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
                index += 1
                if leak_baseline is not None and index % 20 == 0:
                    # NOTE: do NOT gc.collect() here — a full collect on the live
                    # object graph holds the GIL for seconds and would itself trip
                    # the event-loop-lag monitor, corrupting the very metric this
                    # soak measures. Reclaimability is probed once at shutdown.
                    snap = tracemalloc.take_snapshot()
                    top = snap.compare_to(leak_baseline, "lineno")[:12]
                    elapsed = time.monotonic() - soak_started
                    print(
                        f"🔬 [tracemalloc] top growth @ iter {index} "
                        f"(elapsed {elapsed:.0f}s, rss {_rss_mb():.0f}MB):",
                        flush=True,
                    )
                    for stat in top:
                        # Full call chain (last 3 frames) localizes the accumulator.
                        frames = [f.strip() for f in stat.traceback.format()[-3:]]
                        print(
                            f"     {stat.size_diff / 1024 / 1024:+7.1f}MB "
                            f"{stat.count_diff:+8d}  {' <- '.join(reversed(frames))}",
                            flush=True,
                        )
                # Endurance mode: pace the pulses across the real-time window.
                if duration_s > 0.0:
                    remaining = duration_s - (time.monotonic() - soak_started)
                    if remaining <= 0.0:
                        break
                    await asyncio.sleep(min(tick_s, remaining))
    finally:
        if leak_baseline is not None:
            try:
                gc.collect()
                snap = tracemalloc.take_snapshot()
                for stat in snap.compare_to(leak_baseline, "traceback")[:25]:
                    leak_top.append(
                        {
                            "size_diff_mb": round(stat.size_diff / 1024 / 1024, 3),
                            "count_diff": stat.count_diff,
                            "where": stat.traceback.format()[-1].strip(),
                            # Call chain (innermost-last frames) localizes the
                            # accumulator, not just the leaf allocation site.
                            "call_chain": [f.strip() for f in stat.traceback.format()[-4:]],
                        }
                    )
            except (RuntimeError, ValueError, OSError):
                pass
            finally:
                tracemalloc.stop()
        await shutdown_proof_runtime(orch)

    rss_values = [item["rss_mb"] for item in metrics if item["rss_mb"] > 0.0]
    rss_growth = (max(rss_values) - min(rss_values)) if rss_values else 0.0
    max_lag = max((item["lag_ms"] for item in metrics), default=0.0)
    max_queue = max((item["queue_len"] for item in metrics), default=0)

    elapsed_s = round(time.monotonic() - soak_started, 1)
    endurance = duration_s > 0.0
    report = {
        "generated_at": time.time(),
        "profile": args.profile,
        "soak_mode": "endurance" if endurance else "proof_iterations",
        "requested_duration_s": round(duration_s, 1),
        "elapsed_s": elapsed_s,
        "tick_s": round(tick_s, 3) if endurance else None,
        "iterations_completed": len(metrics),
        "memory_leakage_detected": rss_growth > 128.0,
        "rss_growth_mb": round(rss_growth, 3),
        "queue_growth_stable": max_queue <= 10,
        "boot_event_loop_stable": boot_loop_report.stable,
        "boot_event_loop_warmup": boot_loop_report.to_dict(),
        "event_loop_lag_threshold_ms": round(lag_threshold_ms, 3),
        "event_loop_lag_normal": bool(boot_loop_report.stable and max_lag <= lag_threshold_ms),
        "max_lag_ms": round(max_lag, 3),
        "max_queue_len": max_queue,
        "leak_top_growth": leak_top,
        "metrics": metrics,
        "claim_scope": (
            f"real-time endurance soak ({elapsed_s:.0f}s / requested {duration_s:.0f}s): "
            "boot + memory/lag/queue stability over the window"
            if endurance
            else "short proof-profile soak; not indefinite-autonomy evidence"
        ),
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

    print(f"Longevity soak suite completed successfully for profile: {args.profile}")
    return 0 if not report["memory_leakage_detected"] and report["queue_growth_stable"] and report["event_loop_lag_normal"] else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
